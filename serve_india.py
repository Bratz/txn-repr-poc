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
_HIST = "hist.pt"


# --------------------------------------------------------------------------- #
# Velocity (per-account burst) - the v2 time-aware history encoder, served
# --------------------------------------------------------------------------- #

def fit_next_heads(encoder, vocabs, hist, pay, device, horizon_days=35, log=print):
    """Fit the next-event heads (occurrence / gap / amount) on windowed history prefixes,
    reusing the bundle's history encoder (one temporal encoder serves velocity AND
    forecasting). Occurrence gets an isotonic calibrator from a held-out actor split.
    Returns the heads dict for probes['next'], or None if the data has no usable windows.
    """
    from sklearn.linear_model import LogisticRegression, Ridge
    from data.next_event import windowed_examples
    from run_seq import encode_histories
    seqs, lab = windowed_examples(pay, horizon_days=horizon_days)
    if not len(lab):
        log("[next] no windowed examples - heads skipped")
        return None
    e_all = torch.as_tensor(_embed_messages(encoder, vocabs, pay, device)).to(device)
    H = encode_histories(hist, e_all, seqs, device).cpu().numpy()
    actors = sorted(set(lab["actor"]))
    rng = np.random.default_rng(0); rng.shuffle(actors)
    ev_a = set(actors[: max(1, len(actors) // 5)])
    ev = lab["actor"].isin(ev_a).to_numpy(); tr = ~ev
    y = lab["has_next"].to_numpy()
    occ = (LogisticRegression(max_iter=1000, class_weight="balanced").fit(H[tr], y[tr])
           if len(set(y[tr].tolist())) > 1 else None)
    cal = (_fit_iso(occ.predict_proba(H[ev])[:, 1], y[ev])
           if occ is not None and len(set(y[ev].tolist())) > 1 else None)
    unc = lab["gap_days"].notna().to_numpy()
    gap = Ridge().fit(H[tr & unc], np.log1p(lab["gap_days"].to_numpy()[tr & unc]))
    amt = Ridge().fit(H[tr & unc], np.log1p(lab["next_amount"].to_numpy()[tr & unc]))
    log(f"[next] heads fit on {int(tr.sum()):,} windows "
        f"(occurrence prevalence {y.mean():.2f}; horizon {horizon_days}d)")
    return {"occurrence": occ, "occurrence_cal": cal, "gap": gap, "amount": amt,
            "horizon_days": horizon_days}


def fit_exception_calibration(probes, e_ev, pay_ev, log=print):
    """Isotonic calibrators for the per-exception intake probes, fit on held-out rows.
    Stored as probes['exc_cal']; predict() applies them when present."""
    cals = {}
    for code, m in probes.get("exc", {}).items():
        col = f"exc_{code}"
        if col not in pay_ev.columns:
            continue
        y = pay_ev[col].to_numpy()
        if y.sum() >= 5 and y.mean() < 1.0:
            iso = _fit_iso(m.predict_proba(e_ev)[:, 1], y)
            if iso is not None:
                cals[code] = iso
    probes["exc_cal"] = cals
    log(f"[calibration] exception calibrators fit for {len(cals)} codes on "
        f"{len(pay_ev):,} held-out rows")
    return probes


def fit_velocity(encoder, vocabs, pay, device, hist_epochs=2, burst_k=2,
                 burst_min_events=4, log=print):
    """Pretrain a small history encoder over per-account payment sequences (frozen per-row
    embeddings + inter-arrival/calendar encoding) and fit the burst head on h_USR.

    Returns (hcfg_dict, recon_fields, state_dict, head) for persistence; head is None if the
    burst label is degenerate at this scale. Serving is STATELESS: the engine sends an
    account's recent payment rows to predict_velocity - no store in the scorer.
    """
    from sklearn.linear_model import LogisticRegression
    from dataclasses import asdict
    from data.sequence_assembly import assemble_sequences, velocity_labels
    from encoder.history_encoder import HistoryConfig, HistoryEncoder
    from encoder.history_encoder import pretrain as hist_pretrain
    from run_seq import encode_histories

    e_all = torch.as_tensor(_embed_messages(encoder, vocabs, pay, device)).to(device)
    D = int(e_all.shape[1])
    seqs = assemble_sequences(pay, actor_col="DbtrAcct_Id", max_len=64, min_len=2)
    if not seqs:
        log("[velocity] no multi-payment accounts - head skipped")
        return None
    recon_fields = {"Ccy": vocabs.core_size("Ccy"),
                    "identifier_type": vocabs.core_size("identifier_type")}
    full = vocabs.encode(pay)
    targets_all = {n: full["core"][n] for n in recon_fields}
    hcfg = HistoryConfig(hidden=D, layers=2, heads=4, ff_mult=2, epochs=hist_epochs)
    hist = HistoryEncoder(recon_fields, hcfg).to(device)
    hist_pretrain(hist, e_all, targets_all, seqs, hcfg, batch_size=64, log=log)
    hist.freeze()
    h = encode_histories(hist, e_all, seqs, device).cpu().numpy()
    # ponytail: burst window relaxed vs run_seq's defaults (k=3/min 6) - India accounts
    # average ~5 payments, so the stricter window is all-zeros here. Still timing-only.
    yv = velocity_labels(seqs, k=burst_k, min_events=burst_min_events)
    # head on 80% of accounts, isotonic calibration on the held-out 20%
    perm = np.random.default_rng(0).permutation(len(seqs)); cut = int(len(seqs) * 0.8)
    head, cal = None, None
    if len(set(yv.tolist())) > 1 and len(set(yv[perm[:cut]].tolist())) > 1:
        head = LogisticRegression(max_iter=1000, class_weight="balanced").fit(
            h[perm[:cut]], yv[perm[:cut]])
        cal = _fit_iso(head.predict_proba(h[perm[cut:]])[:, 1], yv[perm[cut:]])
    log(f"[velocity] hist encoder + head fit on {len(seqs):,} account sequences "
        f"(burst prevalence {yv.mean():.3f}{'' if head else '; head degenerate -> None'})")
    return {"hcfg": asdict(hcfg), "recon_fields": recon_fields,
            "state": hist.state_dict(), "head": head, "cal": cal}


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


# --------------------------------------------------------------------------- #
# Calibration - isotonic, fit on held-out data, applied wherever a proba is emitted
# --------------------------------------------------------------------------- #

def _fit_iso(p_raw, y):
    """Isotonic calibrator raw-score -> P(y=1); None if the label is degenerate."""
    from sklearn.isotonic import IsotonicRegression
    y = np.asarray(y)
    if len(set(y.tolist())) < 2:
        return None
    return IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip").fit(
        np.asarray(p_raw, dtype=float), y)


def _cal1(iso, p):
    """Apply a calibrator to a scalar proba (identity if None)."""
    return float(p) if iso is None else float(iso.predict(np.atleast_1d(float(p)))[0])


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


def fit_inflight_heads(encoder, vocabs, msg, pay, device, max_uetrs=4000, msg_eval=None,
                       log=print):
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

    def _prefix_xy(frame):
        frame = _visible_sorted(frame)
        keep = frame["end_to_end_id"].drop_duplicates().head(max_uetrs)
        frame = frame[frame["end_to_end_id"].isin(set(keep))].reset_index(drop=True)
        e = _embed_messages(encoder, vocabs, frame, device)
        booked = frame.groupby("end_to_end_id")["msg_type"].agg(
            lambda s: int("camt.054" in set(s)))
        lab = pay.set_index("payment_id")
        X, y = [], {k: [] for k in ("booked", "outcome", "reason", "eta", "cancel", "return")}
        for u, g in frame.groupby("end_to_end_id", sort=False):
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
        return np.asarray(X), y, len(keep)

    X, y, n_uetrs = _prefix_xy(msg)

    def _clf(target):
        arr = np.asarray(y[target])
        if len(set(arr.tolist())) < 2:          # degenerate at this scale -> no head
            return None
        return LogisticRegression(max_iter=1000, class_weight="balanced").fit(X, arr)

    heads = {"booked": _clf("booked"), "outcome": _clf("outcome"), "reason": _clf("reason"),
             "eta_remaining": Ridge().fit(X, np.asarray(y["eta"])),
             "cancel": _clf("cancel"), "return": _clf("return")}
    log(f"[inflight] lifecycle heads fit on {len(X):,} prefixes from {n_uetrs:,} payments "
        f"(booked prevalence {np.mean(y['booked']):.2f}; "
        f"degenerate: {[k for k, h in heads.items() if h is None]})")

    if msg_eval is not None and len(msg_eval):
        X2, y2, _ = _prefix_xy(msg_eval)
        cal = {}
        for k in ("booked", "cancel", "return"):
            h = heads.get(k)
            if h is not None:
                cal[k] = _fit_iso(h.predict_proba(X2)[:, 1], np.asarray(y2[k]))
        oc = heads.get("outcome")
        if oc is not None:
            p = oc.predict_proba(X2)
            yo = np.asarray(y2["outcome"])
            cal["outcome"] = {c: _fit_iso(p[:, i], (yo == c).astype(int))
                              for i, c in enumerate(oc.classes_)}
        heads["_cal"] = cal
        log(f"[inflight] isotonic calibration fit on {len(X2):,} held-out prefixes")
    return heads


# --------------------------------------------------------------------------- #
# Save
# --------------------------------------------------------------------------- #

def save_india_model(save_dir, *, enc_cfg, vocabs, quantizer, encoder, schema, probes,
                     velocity=None):
    """Persist the frozen encoder bundle + intake probes (+ the velocity hist encoder)."""
    import joblib
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    if velocity is not None:
        torch.save({k: velocity[k] for k in ("hcfg", "recon_fields", "state")},
                   save_dir / _HIST)
        probes = {**probes, "velocity": velocity["head"],
                  "velocity_cal": velocity.get("cal")}
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
        "velocity": velocity is not None and velocity["head"] is not None,
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

    def __init__(self, encoder, vocabs, probes, device, hist=None, velocity_head=None):
        self.encoder = encoder
        self.vocabs = vocabs
        self.probes = probes
        self.device = device
        self.hist = hist                      # frozen v2 history encoder (velocity), optional
        self.velocity_head = velocity_head

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
        exc_cal = self.probes.get("exc_cal", {})
        risks = {}
        for n, m in self.probes["exc"].items():
            p = m.predict_proba(e)[:, 1]
            iso = exc_cal.get(n)
            risks[n] = iso.predict(p) if iso is not None else p
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
            cal = heads.get("_cal", {})
            out = {"end_to_end_id": u, "n_msgs": int(len(idx)),
                   "last_msg_type": g["msg_type"].iloc[-1],
                   "booked_proba": round(_cal1(cal.get("booked"),
                                               heads["booked"].predict_proba(x)[0, 1]), 4)}
            oc = heads.get("outcome")
            if oc is not None:
                p = oc.predict_proba(x)[0]
                occal = cal.get("outcome") or {}
                p = np.array([_cal1(occal.get(c), pi) for c, pi in zip(oc.classes_, p)])
                p = p / max(p.sum(), 1e-9)                   # calibrated per-class, renormed
                out["settlement_outcome"] = {c: round(float(pi), 4)
                                             for c, pi in zip(oc.classes_, p)}
            else:
                out["settlement_outcome"] = None
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
                out[name] = (round(_cal1(cal.get(key), h.predict_proba(x)[0, 1]), 4)
                             if h is not None else None)
            rows.append(out)
        return pd.DataFrame(rows)

    def predict_velocity(self, txns_df):
        """Per-account burst score from the account's recent payment rows (>=2, any order -
        sequenced by IntrBkSttlmDt). STATELESS: the engine owns the history store and sends
        the window. Returns actor, n_txns, burst_proba (time-aware v2 encoder) and burst_rule
        (the transparent last-k-gaps rule) for comparison.
        """
        import pandas as pd
        from data.sequence_assembly import assemble_sequences, velocity_labels
        from run_seq import encode_histories
        if self.hist is None or self.velocity_head is None:
            raise SystemExit("this bundle has no velocity head - refit with fit_velocity "
                             "(run_india.py --save on this revision or later)")
        df = txns_df.reset_index(drop=True)
        e_all = torch.as_tensor(self._embed(df)).to(self.device)
        seqs = assemble_sequences(df, actor_col="DbtrAcct_Id", max_len=64, min_len=2)
        if not seqs:
            return pd.DataFrame(columns=["actor", "n_txns", "burst_proba", "burst_rule"])
        h = encode_histories(self.hist, e_all, seqs, self.device).cpu().numpy()
        proba = self.velocity_head.predict_proba(h)[:, 1]
        iso = self.probes.get("velocity_cal")
        if iso is not None:
            proba = iso.predict(proba)
        rule = velocity_labels(seqs, k=2, min_events=4)      # same window the head was fit on
        return pd.DataFrame({"actor": [s["actor"] for s in seqs],
                             "n_txns": [len(s["pos"]) for s in seqs],
                             "burst_proba": np.round(proba, 4),
                             "burst_rule": rule.astype(int)})

    def predict_next(self, txns_df, min_history=3):
        """Next-event forecast per account from its recent payment rows (engine-supplied,
        stateless - same pattern as predict_velocity). Per actor with >= min_history rows:
        next_event_proba (calibrated, within the trained horizon), expected_gap_days,
        expected_amount, and top_payee_hint (most-frequent counterparty - the transparent
        baseline the deferred retrieval head must beat).
        """
        import pandas as pd
        from run_seq import encode_histories
        heads = self.probes.get("next")
        if self.hist is None or not heads or heads.get("occurrence") is None:
            raise SystemExit("this bundle has no next-event heads - refit with "
                             "fit_next_heads (run_india.py --save on this revision+)")
        df = txns_df.reset_index(drop=True)
        dates = pd.to_datetime(df["IntrBkSttlmDt"])
        e_all = torch.as_tensor(self._embed(df)).to(self.device)
        from data.next_event import _seq
        rows, seqs = [], []
        for actor, idx in df.groupby("DbtrAcct_Id").groups.items():
            pos = np.asarray(idx, dtype=np.int64)
            pos = pos[np.argsort(dates.values[pos])]
            if len(pos) < min_history:
                continue
            seqs.append(_seq(actor, pos[-64:], dates.values[pos[-64:]]))
            vals, counts = np.unique(df["CdtrAcct_Id"].astype(str).to_numpy()[pos],
                                     return_counts=True)
            rows.append({"actor": actor, "n_history": int(len(pos)),
                         "top_payee_hint": str(vals[counts.argmax()])})
        if not rows:
            return pd.DataFrame(columns=["actor", "n_history", "next_event_proba",
                                         "expected_gap_days", "expected_amount",
                                         "top_payee_hint"])
        H = encode_histories(self.hist, e_all, seqs, self.device).cpu().numpy()
        p = heads["occurrence"].predict_proba(H)[:, 1]
        if heads.get("occurrence_cal") is not None:
            p = heads["occurrence_cal"].predict(p)
        out = pd.DataFrame(rows)
        out["next_event_proba"] = np.round(p, 4)
        out["expected_gap_days"] = np.round(np.expm1(heads["gap"].predict(H)).clip(0), 1)
        out["expected_amount"] = np.round(np.expm1(heads["amount"].predict(H)).clip(0), 2)
        out["horizon_days"] = heads["horizon_days"]
        return out[["actor", "n_history", "next_event_proba", "expected_gap_days",
                    "expected_amount", "top_payee_hint", "horizon_days"]]

    def liquidity_forecast(self, df, bucket_edges_min=(1, 15, 60, 240, 1440)):
        """Treasury view: expected OUTFLOW by rail x settlement-time bucket, aggregated from
        the per-payment ETA predictions. Pure arithmetic over existing heads - no new model.

        Uses the ETA point estimate per payment (simplification: not a distribution) and
        outward payments only when a `direction` column is present (outflow = money leaving).
        Returns {"buckets": [...labels...], "by_rail": {rail: [amount per bucket]},
                 "total": [amount per bucket], "n_payments": int, "total_amount": float}.
        """
        sub = df[df["direction"] == "outward"] if "direction" in df.columns else df
        sub = sub.reset_index(drop=True)
        if not len(sub):
            return {"buckets": [], "by_rail": {}, "total": [], "n_payments": 0,
                    "total_amount": 0.0}
        res = self.predict(sub)
        eta = res["eta_min_pred"].to_numpy(dtype=float)
        amt = sub["IntrBkSttlmAmt"].to_numpy(dtype=float)
        rails = res["rail_pred"].to_numpy()
        edges = list(bucket_edges_min)
        labels = ([f"<{edges[0]}m"]
                  + [f"{a}-{b}m" for a, b in zip(edges[:-1], edges[1:])]
                  + [f">{edges[-1]}m"])
        idx = np.searchsorted(edges, eta, side="right")
        by_rail = {}
        for r in sorted(set(rails.tolist())):
            m = rails == r
            by_rail[r] = [round(float(amt[m & (idx == b)].sum()), 2)
                          for b in range(len(labels))]
        total = [round(float(amt[idx == b].sum()), 2) for b in range(len(labels))]
        return {"buckets": labels, "by_rail": by_rail, "total": total,
                "n_payments": int(len(sub)), "total_amount": round(float(amt.sum()), 2)}

    def explain(self, df, top_k=5):
        """Column-occlusion drivers per payment (FAITHFUL attribution, no LLM): occlude one
        feature column at a time (categoricals -> 'Unknown', amount -> 0.0), re-embed, and
        report the drop in the predicted rail/risk probability and the ETA shift. ~n_cols
        extra forward passes per payment - cap the batch accordingly.
        """
        import pandas as pd
        df = df.reset_index(drop=True)
        cols = (list(self.vocabs.high_card) + [self.vocabs.numerical_col]
                + list(self.vocabs.core))
        meta = [c for c in ("Dbtr_Nm", "Cdtr_Nm", "UltmtDbtr_Nm", "UltmtCdtr_Nm",
                            "Dbtr_Ctry", "Cdtr_Ctry", "Dbtr_Industry", "Cdtr_Industry",
                            "Dbtr_SubIndustry", "Cdtr_SubIndustry") if c in df.columns]
        cols = [c for c in cols if c in df.columns] + meta
        frames = [df] + [df.assign(**{c: 0.0 if c == self.vocabs.numerical_col else "Unknown"})
                         for c in cols]
        big = pd.concat(frames, ignore_index=True)
        e = self._embed(big)
        n = len(df)
        base, occ = e[:n], e[n:].reshape(len(cols), n, -1)
        rail, risk = self.probes["rail"], self.probes.get("tasks", {}).get("risk")
        p_rail = rail.predict_proba(base)
        cls = p_rail.argmax(1)
        p_risk = risk.predict_proba(base) if risk is not None else None
        rcls = p_risk.argmax(1) if p_risk is not None else None
        eta = self.probes["eta"].predict(base)
        out = []
        for i in range(n):
            drivers = []
            for j, c in enumerate(cols):
                d_rail = float(p_rail[i, cls[i]] - rail.predict_proba(occ[j, i:i+1])[0, cls[i]])
                d_risk = (float(p_risk[i, rcls[i]]
                                - risk.predict_proba(occ[j, i:i+1])[0, rcls[i]])
                          if risk is not None else 0.0)
                d_eta = float(eta[i] - self.probes["eta"].predict(occ[j, i:i+1])[0])
                drivers.append({"field": c, "rail_impact": round(d_rail, 4),
                                "risk_impact": round(d_risk, 4),
                                "eta_impact_min": round(d_eta, 1)})
            drivers.sort(key=lambda d: -(abs(d["rail_impact"]) + abs(d["risk_impact"])))
            out.append(drivers[:top_k])
        return out


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
    hist = None
    if (save_dir / _HIST).exists():
        from encoder.history_encoder import HistoryConfig, HistoryEncoder
        hb = torch.load(save_dir / _HIST, map_location="cpu", weights_only=False)
        hist = HistoryEncoder(hb["recon_fields"], HistoryConfig(**hb["hcfg"]))
        hist.load_state_dict(hb["state"])
        hist.freeze()
        hist.to(device)
    return IndiaScorer(encoder, vocabs, probes, device,
                       hist=hist, velocity_head=probes.get("velocity"))


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
