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


def _prefix_times(g):
    """(t_last, gap_since_previous) in minutes for a prefix's rows; zeros if absent."""
    if "t_offset_min" not in g.columns:
        return 0.0, 0.0
    t = np.nan_to_num(g["t_offset_min"].to_numpy(dtype=float))
    return float(t[-1]), float(t[-1] - t[-2]) if len(t) > 1 else 0.0


def _prefix_features(pool, g):
    """In-flight feature vector for a prefix (its message rows, lifecycle order):
    mean-pooled embeddings + one-hot(last msg_type, last tx_sts) + elapsed-time signals.

    The last-message one-hots exist because mean-pooling dilutes an outcome message to 1/k of
    the pool (the score failed to snap on camt.054 without them). The time tail - log-minutes
    since pain.001 and log-gap since the previous message - makes a slow/stale lifecycle
    readable (a pacs.008 followed by hours of silence scores differently from a fresh one).
    """
    from data.iso_lifecycle import MSG_TYPES, TX_STS
    last = g.iloc[-1]
    mt = np.zeros(len(MSG_TYPES), dtype=np.float32)
    m = last.get("msg_type")
    if m in MSG_TYPES:
        mt[MSG_TYPES.index(m)] = 1.0
    st = np.zeros(len(TX_STS), dtype=np.float32)
    s = last.get("tx_sts")
    s = "" if (s is None or (isinstance(s, float) and np.isnan(s))) else str(s)
    if s in TX_STS:
        st[TX_STS.index(s)] = 1.0
    t_last, gap = _prefix_times(g)
    tail = np.array([np.log1p(max(t_last, 0.0)), np.log1p(max(gap, 0.0))], dtype=np.float32)
    return np.concatenate([pool, mt, st, tail])


def fit_inflight_heads(encoder, vocabs, msg, pay, device, max_uetrs=4000, log=print):
    """Fit the in-flight lifecycle heads on message-prefix pools -> dict of sklearn heads.

    Trains on ALL visible prefixes (k=1..n) per UETR, so every head is valid at any lifecycle
    state (and snaps once outcome messages arrive). The feature op is the SAME one
    predict_stream applies - no train/serve skew. Heads:
      booked         P(camt.054 eventually)            LogisticRegression (binary)
      outcome        terminal status distribution      LogisticRegression (multiclass)
      reason         reject/hold reason code           LogisticRegression (multiclass)
      eta_remaining  minutes to lifecycle end          Ridge (log1p space)
      cancel/return  recall / return likelihood        LogisticRegression, None if degenerate

    ponytail: capped at max_uetrs payments (linear heads don't need 100k prefixes and
    full-corpus embedding is minutes of CPU); raise the cap if a head ever looks data-bound.
    """
    from sklearn.linear_model import LogisticRegression, Ridge
    msg = _visible_sorted(msg)
    keep = msg["end_to_end_id"].drop_duplicates().head(max_uetrs)
    msg = msg[msg["end_to_end_id"].isin(set(keep))].reset_index(drop=True)
    e = _embed_messages(encoder, vocabs, msg, device)
    booked = msg.groupby("end_to_end_id")["msg_type"].agg(lambda s: int("camt.054" in set(s)))
    lab = pay.set_index("payment_id")
    X, y = [], {k: [] for k in ("booked", "outcome", "reason", "eta", "cancel", "return")}
    for u, g in msg.groupby("end_to_end_id", sort=False):
        row = lab.loc[int(g["payment_id"].iloc[0])]
        total = float(row.get("time_to_settle_min", 0.0) or 0.0)
        idx = g.index.to_numpy()
        csum = np.cumsum(e[idx], axis=0)
        for k in range(1, len(idx) + 1):
            gk = g.iloc[:k]
            X.append(_prefix_features(csum[k - 1] / k, gk))
            t_last, _ = _prefix_times(gk)
            y["booked"].append(int(booked[u]))
            y["outcome"].append(str(row.get("terminal_status", "STP")))
            y["reason"].append(str(row.get("reject_reason", "none")))
            y["eta"].append(np.log1p(max(total - t_last, 0.0)))
            y["cancel"].append(int(row.get("cancel_requested", 0)))
            y["return"].append(int(row.get("returned", 0)))
    X = np.asarray(X)

    def _clf(target):
        arr = np.asarray(y[target])
        if len(set(arr.tolist())) < 2:          # degenerate at this scale -> no head
            return None
        return LogisticRegression(max_iter=1000, class_weight="balanced").fit(X, arr)

    heads = {"booked": _clf("booked"), "outcome": _clf("outcome"), "reason": _clf("reason"),
             "eta_remaining": Ridge().fit(X, np.asarray(y["eta"])),
             "cancel": _clf("cancel"), "return": _clf("return")}
    log(f"[inflight] lifecycle heads fit on {len(X):,} prefixes from {len(keep):,} payments "
        f"(booked prevalence {np.mean(y['booked']):.2f}; "
        f"degenerate: {[k for k, h in heads.items() if h is None]})")
    return heads


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
        "inflight": "inflight" in probes,
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
        """Score message prefixes per UETR -> the predicted-lifecycle object.

        Input: engine-visible message rows (build_messages output / the stream CSV) - any
        subset of a payment's lifecycle counts as 'seen so far'. Per UETR: booked_proba,
        settlement_outcome distribution, eta_remaining_min, reject_reason_if_failed, and
        recall/return likelihoods (None where the head was degenerate at training scale).
        Requires a bundle with probes['inflight'] (run_india --save on this revision+).
        """
        import pandas as pd
        heads = self.probes.get("inflight")
        if not heads:
            raise SystemExit("this bundle has no in-flight lifecycle heads - retrain with "
                             "`run_india.py --save` to add probes['inflight']")
        msg = _visible_sorted(msg_df)
        if not len(msg):
            return pd.DataFrame(columns=["end_to_end_id", "n_msgs", "last_msg_type",
                                         "booked_proba"])
        e = _embed_messages(self.encoder, self.vocabs, msg, self.device)
        rows = []
        for u, g in msg.groupby("end_to_end_id", sort=False):
            idx = g.index.to_numpy()
            x = _prefix_features(e[idx].mean(0), g)[None]   # same op fit_inflight_heads used
            if heads["booked"].n_features_in_ != x.shape[1]:
                raise SystemExit("in-flight heads predate the current feature format - "
                                 "refit with `run_india.py --save`")
            out = {"end_to_end_id": u, "n_msgs": int(len(idx)),
                   "last_msg_type": g["msg_type"].iloc[-1],
                   "booked_proba": round(float(heads["booked"].predict_proba(x)[0, 1]), 4)}
            oc = heads.get("outcome")
            out["settlement_outcome"] = ({c: round(float(p), 4) for c, p in
                                          zip(oc.classes_, oc.predict_proba(x)[0])}
                                         if oc is not None else None)
            out["eta_remaining_min"] = round(float(np.clip(
                np.expm1(heads["eta_remaining"].predict(x)[0]), 0, None)), 1)
            rs = heads.get("reason")
            if rs is not None:                              # top NON-'none' reason code
                p = rs.predict_proba(x)[0]
                cand = [(c, float(v)) for c, v in zip(rs.classes_, p) if c != "none"]
                code, score = max(cand, key=lambda t: t[1]) if cand else (None, 0.0)
                out["reject_reason_if_failed"] = ({"code": code, "score": round(score, 4)}
                                                  if code else None)
            else:
                out["reject_reason_if_failed"] = None
            for name, key in (("cancel_proba", "cancel"), ("return_proba", "return")):
                h = heads.get(key)
                out[name] = (round(float(h.predict_proba(x)[0, 1]), 4)
                             if h is not None else None)
            rows.append(out)
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
