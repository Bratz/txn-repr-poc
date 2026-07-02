"""
Persisted India multi-rail model - save / load / predict without retraining.

The deployable model is the FROZEN v1 encoder + the intake probes (no LLM, CPU). This mirrors
predict.py's encoder-reconstruction convention: the party-summary table rides inside the
encoder state_dict, so the assembler is rebuilt with party_store=None and the real weights
arrive via load_state_dict; the quantizer grids and column vocabs are persisted alongside.

Bundle (a directory):
  encoder.pt     torch bundle - enc_cfg, schema buckets, vocabs, quantizer, encoder_state
  probes.joblib  the sklearn intake probes (rail / status / eta / per-exception) + the
                 in-flight 'inflight_booked' head (fit on message-prefix POOLS - see
                 fit_inflight_head; predict_stream is its only valid consumer)
  meta.json      human-readable summary

  python run_india.py --save model_india           # train + persist
  python serve_india.py --model-dir model_india --input data/india_rails_payments.parquet
  python serve_india.py --model-dir model_india --input data/test_input_messages_100.csv
                                                   # message rows -> streaming booked-proba
"""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

from encoder.tabular_encoder import EncoderConfig, TabularEncoder
from encoders.column_assembler import ColumnAssembler, ColumnVocabs
from encoders.quantizer import AdaptiveQuantizer
from run_seq import embed_all_rows

_ENC = "encoder.pt"
_PROBES = "probes.joblib"


# --------------------------------------------------------------------------- #
# In-flight (message-prefix) scoring - ONE pooling code path for fit and serve
# --------------------------------------------------------------------------- #

def _visible_sorted(msg):
    """Engine perspective: keep only messages our bank sees, in lifecycle order."""
    if "msg_direction" in msg.columns:
        msg = msg[msg["msg_direction"].notna()]
    elif "visible" in msg.columns:
        msg = msg[msg["visible"] == 1]
    return msg.sort_values(["end_to_end_id", "seq"]).reset_index(drop=True)


def _embed_messages(encoder, vocabs, msg, device):
    """Embed message rows WITHOUT the intake drift warning - sparse UNK columns are the
    designed shape of a lifecycle message, not vocab drift."""
    return embed_all_rows(encoder, vocabs.encode(msg), len(msg), device).cpu().numpy()


def _prefix_features(pool, last_row):
    """In-flight feature vector = mean-pooled embeddings + one-hot(last msg_type, last tx_sts).

    The explicit last-message signal exists because mean-pooling dilutes an outcome message to
    1/k of the pool: without it the score failed to snap after pacs.002/camt.054 arrived. With
    it, 'camt.054 seen' / 'pacs.002 RJCT seen' are directly readable by the head.
    """
    from data.iso_lifecycle import MSG_TYPES, TX_STS
    mt = np.zeros(len(MSG_TYPES), dtype=np.float32)
    m = last_row.get("msg_type")
    if m in MSG_TYPES:
        mt[MSG_TYPES.index(m)] = 1.0
    st = np.zeros(len(TX_STS), dtype=np.float32)
    s = last_row.get("tx_sts")
    s = "" if (s is None or (isinstance(s, float) and np.isnan(s))) else str(s)
    if s in TX_STS:
        st[TX_STS.index(s)] = 1.0
    return np.concatenate([pool, mt, st])


def fit_inflight_head(encoder, vocabs, msg, device, max_uetrs=4000, log=print):
    """Fit the streaming 'will it be booked?' head on message-prefix pools.

    Trains on ALL visible prefixes (k=1..n) per UETR, so the served probability is valid at
    any lifecycle state - including snapping toward 1/0 once an outcome message (pacs.002 /
    camt.054) is in the prefix. Pooling here is the SAME operation predict_stream applies
    (mean of the frozen per-message embeddings seen so far) - no train/serve skew.

    ponytail: capped at max_uetrs payments (one logistic head does not need 100k prefixes and
    full-corpus embedding is minutes of CPU); raise the cap if the head ever looks data-bound.
    """
    from sklearn.linear_model import LogisticRegression
    msg = _visible_sorted(msg)
    keep = msg["end_to_end_id"].drop_duplicates().head(max_uetrs)
    msg = msg[msg["end_to_end_id"].isin(set(keep))].reset_index(drop=True)
    e = _embed_messages(encoder, vocabs, msg, device)
    booked = msg.groupby("end_to_end_id")["msg_type"].agg(lambda s: int("camt.054" in set(s)))
    X, y = [], []
    for u, g in msg.groupby("end_to_end_id", sort=False):
        idx = g.index.to_numpy()
        csum = np.cumsum(e[idx], axis=0)
        for k in range(1, len(idx) + 1):
            X.append(_prefix_features(csum[k - 1] / k, g.iloc[k - 1]))
            y.append(int(booked[u]))
    X, y = np.asarray(X), np.asarray(y)
    head = LogisticRegression(max_iter=1000, class_weight="balanced").fit(X, y)
    log(f"[inflight] booked head fit on {len(y):,} prefixes from {len(keep):,} payments "
        f"(prevalence {y.mean():.2f})")
    return head


# --------------------------------------------------------------------------- #
# Save
# --------------------------------------------------------------------------- #

def save_india_model(save_dir, *, enc_cfg, vocabs, quantizer, encoder, schema, probes):
    """Persist the frozen encoder bundle + intake probes."""
    import joblib
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    torch.save({
        "enc_cfg": asdict(enc_cfg),
        "schema_buckets": schema["buckets"],
        "twin": schema["twin"],
        "numerical_col": vocabs.numerical_col,
        "ccy_col": vocabs.ccy_col,
        "vocabs": {
            "high_card": vocabs.high_card,
            "high_card_freq": {c: v.tolist() for c, v in vocabs.high_card_freq.items()},
            "core": vocabs.core,
        },
        "quantizer": quantizer.to_dict(),
        "encoder_state": encoder.state_dict(),
    }, save_dir / _ENC)
    joblib.dump(probes, save_dir / _PROBES)
    (save_dir / "meta.json").write_text(json.dumps({
        "rails": list(probes["rail"].classes_),
        "statuses": list(probes["status"].classes_),
        "exceptions": list(probes["exc"].keys()),
        "tasks": list(probes.get("tasks", {})),
        "inflight": "inflight_booked" in probes,
        "hidden": enc_cfg.hidden,
    }, indent=2))
    return save_dir


# --------------------------------------------------------------------------- #
# Load + predict
# --------------------------------------------------------------------------- #

class IndiaScorer:
    """Loaded India model: predict rail / status / ETA / exception risks for payment rows.

    Two grains, two entry points - do not cross them:
      * predict(df)         INTAKE grain: complete payment rows. Its probes were fit on
                            complete-row embeddings; message-prefix pools are OOD for them.
      * predict_stream(msg) IN-FLIGHT grain: engine-visible message rows per UETR. Uses the
                            'inflight_booked' head, fit on the same prefix-pool operation
                            (fit_inflight_head) - the only valid consumer of pooled prefixes.
    """

    def __init__(self, encoder, vocabs, probes, device):
        self.encoder = encoder
        self.vocabs = vocabs
        self.probes = probes
        self.device = device

    @torch.no_grad()
    def _embed(self, df):
        # warn on serve-time vocab drift (e.g. an id column trained int but served float):
        # a high UNK rate means the representation has quietly degraded to OOV embeddings.
        hot = {c: round(r, 3) for c, r in self.vocabs.unk_rate(df).items() if r > 0.5}
        if hot:
            import warnings
            warnings.warn(f"high UNK rate on {hot} — possible dtype/vocab drift", stacklevel=2)
        full = self.vocabs.encode(df)
        return embed_all_rows(self.encoder, full, len(df), self.device).cpu().numpy()

    def predict(self, df, top_exceptions: int = 3):
        """Return a DataFrame: predicted rail (+ confidence), status, ETA, and the top-k
        exception risk scores per payment (uncalibrated balanced-probe rankings)."""
        import pandas as pd
        from data.rails import IDENTIFIER_TYPES, eligible_rails
        e = self._embed(df)
        rail_p = self.probes["rail"].predict_proba(e).copy()
        rail_cls = list(self.probes["rail"].classes_)

        # Hybrid routing: the learned probe gives PREFERENCE, but a payment can only go on
        # an ELIGIBLE rail (cap/min/cross-border are hard rules the classifier doesn't know).
        # Mask out ineligible rails per row so predictions are always valid.
        amts = df["IntrBkSttlmAmt"].to_numpy(float)
        idents = (df["identifier_type"].astype(str).to_numpy()
                  if "identifier_type" in df.columns else np.array([None] * len(df)))
        xb = ((df["Dbtr_Ctry"].to_numpy() != df["Cdtr_Ctry"].to_numpy())
              if {"Dbtr_Ctry", "Cdtr_Ctry"} <= set(df.columns) else np.zeros(len(df), bool))
        for i in range(len(df)):
            ident = idents[i] if idents[i] in IDENTIFIER_TYPES else None
            elig = eligible_rails(amts[i], ident, bool(xb[i]))
            if not elig:
                # contradictory instrument (e.g. VPA over the UPI cap, or BIC domestic):
                # fall back to eligibility by amount + cross-border only.
                elig = eligible_rails(amts[i], None, bool(xb[i]))
            keep = np.array([c in elig for c in rail_cls])
            if elig and keep.any():
                rail_p[i, ~keep] = 0.0

        rail_p = rail_p / rail_p.sum(1, keepdims=True).clip(min=1e-9)
        out = pd.DataFrame(index=df.index)
        if "payment_id" in df.columns:
            out["payment_id"] = df["payment_id"].to_numpy()
        out["rail_pred"] = np.array(rail_cls)[rail_p.argmax(1)]
        out["rail_conf"] = rail_p.max(1).round(3)
        out["status_pred"] = self.probes["status"].predict(e)
        out["eta_min_pred"] = np.clip(self.probes["eta"].predict(e), 0, None).round(1)
        for tname, m in self.probes.get("tasks", {}).items():   # §5 heads (risk/geo/expense)
            out[f"{tname}_pred"] = m.predict(e)
        risks = {n: m.predict_proba(e)[:, 1] for n, m in self.probes["exc"].items()}
        names = list(risks)
        R = np.vstack([risks[n] for n in names]).T if names else np.zeros((len(df), 0))
        topk = []
        for i in range(len(df)):
            order = np.argsort(-R[i])[:top_exceptions]
            topk.append(", ".join(f"{names[j]} {R[i, j]:.2f}" for j in order))
        out["top_exception_risks"] = topk
        return out.reset_index(drop=True)

    def predict_stream(self, msg_df):
        """Score message prefixes per UETR: P(booked | messages seen so far).

        Input: engine-visible message rows (build_messages output / the stream CSV) - any
        subset of a payment's lifecycle counts as 'seen so far'. Returns one row per UETR:
        n_msgs, last_msg_type, booked_proba. Requires a bundle with the 'inflight_booked'
        head (run_india --save on this revision or later).
        """
        import pandas as pd
        head = self.probes.get("inflight_booked")
        if head is None:
            raise SystemExit("this bundle has no in-flight head - retrain with "
                             "`run_india.py --save` to add probes['inflight_booked']")
        msg = _visible_sorted(msg_df)
        if not len(msg):
            return pd.DataFrame(columns=["end_to_end_id", "n_msgs", "last_msg_type",
                                         "booked_proba"])
        e = _embed_messages(self.encoder, self.vocabs, msg, self.device)
        rows = []
        for u, g in msg.groupby("end_to_end_id", sort=False):
            idx = g.index.to_numpy()
            x = _prefix_features(e[idx].mean(0), g.iloc[-1])  # same op fit_inflight_head used
            if head.n_features_in_ != x.shape[0]:
                raise SystemExit("in-flight head predates the current feature format - "
                                 "refit with `run_india.py --save`")
            rows.append({"end_to_end_id": u, "n_msgs": int(len(idx)),
                         "last_msg_type": g["msg_type"].iloc[-1],
                         "booked_proba": round(float(head.predict_proba(x[None])[0, 1]), 4)})
        return pd.DataFrame(rows)


def load_india_model(save_dir, device=None) -> IndiaScorer:
    import joblib
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    save_dir = Path(save_dir)
    if not (save_dir / _ENC).exists() or not (save_dir / _PROBES).exists():
        raise SystemExit(f"no India model at {save_dir} (run `run_india.py --save {save_dir}` first)")
    b = torch.load(save_dir / _ENC, map_location="cpu", weights_only=False)
    enc_cfg = EncoderConfig(**b["enc_cfg"])
    vocabs = ColumnVocabs(
        high_card=b["vocabs"]["high_card"],
        high_card_freq={c: np.asarray(v) for c, v in b["vocabs"]["high_card_freq"].items()},
        core=b["vocabs"]["core"],
        numerical_col=b["numerical_col"], ccy_col=b["ccy_col"],
    )
    quantizer = AdaptiveQuantizer.from_dict(b["quantizer"])
    schema = {"buckets": b["schema_buckets"]}
    assembler = ColumnAssembler(schema, vocabs, quantizer, party_store=None,
                                embedding_dim=enc_cfg.hidden, high_card_embedder="partitioned")
    encoder = TabularEncoder(assembler, enc_cfg)
    encoder.load_state_dict(b["encoder_state"])
    encoder.freeze()
    encoder.to(device)
    probes = joblib.load(Path(save_dir) / _PROBES)
    return IndiaScorer(encoder, vocabs, probes, device)


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #

def main():
    import argparse

    import pandas as pd

    ap = argparse.ArgumentParser(description="Predict with a saved India multi-rail model")
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--input", required=True, help="parquet/csv of projected payment rows")
    ap.add_argument("--out", default="india_predictions.csv")
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    scorer = load_india_model(args.model_dir)
    p = Path(args.input)
    if not p.exists():
        raise SystemExit(f"input not found: {p}")
    if p.suffix == ".xml":
        from data.iso20022_pacs008 import parse_pacs008_frame
        df = parse_pacs008_frame(p)                          # raw pacs.008 -> projected rows
    elif p.suffix == ".parquet":
        df = pd.read_parquet(p)
    else:
        df = pd.read_csv(p)
    if args.limit:
        df = df.head(args.limit)
    if "msg_type" in df.columns:                         # message stream -> in-flight scoring
        res = scorer.predict_stream(df)
        res.to_csv(args.out, index=False)
        print(f"scored {len(res):,} in-flight payments (streams) -> {args.out}")
        print(f"booked_proba mean {res['booked_proba'].mean():.3f}")
    else:
        res = scorer.predict(df)
        res.to_csv(args.out, index=False)
        print(f"predicted {len(res):,} payments -> {args.out}")
        print(f"rail pred dist: {res['rail_pred'].value_counts().to_dict()}")


if __name__ == "__main__":
    main()
