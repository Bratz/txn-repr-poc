"""
Payment-level digital twin - end to end (docs/PAYMENT_TWIN.md).

Reuses the backbone: the frozen v1 encoder gives f(payment); the v2 Layer-3b history
encoder (architecture only) gives the in-flight representation over workflow steps.

  INTAKE (v1)   payment f(x) -> frozen-rep probes:
                  - exception likelihood (multi-label, PR-AUC per type)
                  - terminal status (STP/REPAIRED/MANUAL_REVIEW/REJECTED)
                  - time-to-settle (ETA, MAE vs a mean baseline)
  IN-FLIGHT (v2) prefix of (step, outcome, time) events -> history encoder -> next-step
                  exception prediction (which exception the next step throws, or none)

All scored on HELD-OUT payments. Beyond arXiv:2410.07851; never touches run_gpu.py.
  --smoke : tiny configs + CPU + row caps, validates the whole twin chain.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from run_gpu import _index_batch, _to_device
from run_seq import embed_all_rows

ROOT = Path(__file__).resolve().parent


def load_twin(args):
    import pandas as pd
    sp = Path(args.schema)
    schema = json.loads(sp.read_text()) if sp.exists() else None
    if schema is None:
        raise SystemExit("run data/synth_workflow.py first (need the twin schema)")

    def _read(p):
        p = Path(p)
        return pd.read_parquet(p) if p.suffix == ".parquet" else pd.read_csv(p)
    pay = _read(args.payments)
    evt = _read(args.events)
    if args.limit:
        keep = set(pay["payment_id"].to_numpy()[: args.limit])
        pay = pay[pay["payment_id"].isin(keep)].reset_index(drop=True)
        evt = evt[evt["payment_id"].isin(keep)].reset_index(drop=True)
    return pay, evt, schema


# --------------------------------------------------------------------------- #
# INTAKE (v1) - frozen-rep probes
# --------------------------------------------------------------------------- #

def intake_eval(e, pay, twin, tr, ev):
    from sklearn.linear_model import LogisticRegression, Ridge
    from sklearn.metrics import (accuracy_score, average_precision_score,
                                 f1_score, mean_absolute_error)

    def probe(y):
        if len(set(y[tr])) < 2 or len(set(y[ev])) < 2:
            return None
        clf = LogisticRegression(max_iter=1000, class_weight="balanced").fit(e[tr], y[tr])
        return float(average_precision_score(y[ev], clf.predict_proba(e[ev])[:, 1]))

    exc = {c.replace("exc_", ""): probe(pay[c].to_numpy()) for c in twin["exc_columns"]}

    ys = pay[twin["status_column"]].to_numpy()
    clf = LogisticRegression(max_iter=1000, class_weight="balanced").fit(e[tr], ys[tr])
    pred = clf.predict(e[ev])
    from collections import Counter
    maj = Counter(ys[tr]).most_common(1)[0][0]

    yr = pay[twin["eta_column"]].to_numpy(float)
    reg = Ridge().fit(e[tr], yr[tr])
    return {
        "exception_pr_auc": exc,
        "status_accuracy": float(accuracy_score(ys[ev], pred)),
        "status_macro_f1": float(f1_score(ys[ev], pred, average="macro")),
        "status_majority_baseline": float((ys[ev] == maj).mean()),
        "eta_mae_min": float(mean_absolute_error(yr[ev], reg.predict(e[ev]))),
        "eta_baseline_mae_min": float(mean_absolute_error(yr[ev], np.full(len(ev), yr[tr].mean()))),
    }


def demo(e, pay, twin, tr, ev, n, seed=0):
    """Print the twin's intake forecast for a few held-out payments, plus an
    unsupervised anomaly score (IsolationForest on f(x)) and the actual outcome."""
    from sklearn.ensemble import IsolationForest
    from sklearn.linear_model import LogisticRegression, Ridge

    exc_clf = {c: LogisticRegression(max_iter=1000, class_weight="balanced").fit(e[tr], pay[c].to_numpy()[tr])
               for c in twin["exc_columns"] if len(set(pay[c].to_numpy()[tr])) > 1}
    ys = pay[twin["status_column"]].to_numpy()
    st = LogisticRegression(max_iter=1000, class_weight="balanced").fit(e[tr], ys[tr])
    eta = Ridge().fit(e[tr], pay[twin["eta_column"]].to_numpy(float)[tr])
    iso = IsolationForest(n_estimators=120, random_state=0).fit(e[tr])
    anom = -iso.score_samples(e[ev])                     # higher = more anomalous
    thr = float(np.quantile(anom, 0.9))

    sel = np.random.default_rng(seed).choice(len(ev), size=min(n, len(ev)), replace=False)
    print("\n=== per-payment twin demo (held-out payments) ===")
    for j in sel:
        i = int(ev[j])
        r = pay.iloc[i]
        top = sorted(((c.replace("exc_", ""), float(exc_clf[c].predict_proba(e[i:i+1])[0, 1]))
                      for c in exc_clf), key=lambda t: -t[1])[:3]
        sp = st.predict_proba(e[i:i+1])[0]
        actual = [c.replace("exc_", "") for c in twin["exc_columns"] if r[c] == 1] or ["none"]
        print(f"\npayment {int(r[twin['id_column']])}  {r['direction']}  "
              f"{r['IntrBkSttlmAmt']:.0f} {r['Ccy']}  {r['Dbtr_Ctry']}->{r['Cdtr_Ctry']}")
        print("  predicted exceptions: " + ", ".join(f"{k} {p:.2f}" for k, p in top))
        print(f"  predicted status: {st.classes_[sp.argmax()]} ({sp.max():.2f})   "
              f"predicted ETA: {eta.predict(e[i:i+1])[0]:.0f} min")
        print(f"  anomaly score: {anom[j]:.3f}" + ("   <-- ANOMALY (top 10%)" if anom[j] >= thr else ""))
        print(f"  ACTUAL: status {r[twin['status_column']]} | exceptions {actual} | "
              f"ETA {r[twin['eta_column']]:.0f} min")


# --------------------------------------------------------------------------- #
# IN-FLIGHT (v2) - step encoder + history-encoder backbone -> next exception
# --------------------------------------------------------------------------- #

class StepEmbedder(nn.Module):
    """Embed a workflow step (step id + this-step exception + payment direction) -> D."""

    def __init__(self, n_steps, n_exc, n_dir, d):
        super().__init__()
        self.step = nn.Embedding(n_steps, d)
        self.exc = nn.Embedding(n_exc, d)
        self.direction = nn.Embedding(n_dir, d)

    def forward(self, step, exc, direction):                # (B,L),(B,L),(B,)
        return self.step(step) + self.exc(exc) + self.direction(direction).unsqueeze(1)


class Inflight(nn.Module):
    """StepEmbedder + the v2 history-encoder backbone + a next-exception head."""

    def __init__(self, n_steps, n_exc, n_dir, n_next, hcfg):
        super().__init__()
        from encoder.history_encoder import HistoryEncoder
        self.steps = StepEmbedder(n_steps, n_exc, n_dir, hcfg.hidden)
        self.hist = HistoryEncoder(recon_fields={}, config=hcfg)   # architecture only
        self.head = nn.Linear(hcfg.hidden, n_next)

    def forward(self, b):
        e_seq = self.steps(b["step"], b["exc"], b["direction"])
        h_usr, _ = self.hist(e_seq, b, static=None, event_mask=None)
        return self.head(h_usr)


def build_step_examples(evt, pay, step_vocab, exc_vocab, dir_vocab, next_vocab, max_len):
    """For each payment, every prefix of its primary step-events -> predict the next
    step's exception (or 'none'). Returns a list of example dicts."""
    dir_by_id = dict(zip(pay["payment_id"], pay["direction"]))
    ex = []
    evt = evt[evt["outcome"] != "repaired"]                 # one row per step
    for pid, sub in evt.groupby("payment_id"):
        sub = sub.sort_values("seq")
        steps = sub["step"].map(step_vocab).to_numpy()
        excs = sub["excode"].map(exc_vocab).to_numpy()
        t = sub["t_min"].to_numpy(dtype=np.float32)
        nxt = sub["excode"].map(next_vocab).to_numpy()      # target uses same code space
        d = dir_vocab[dir_by_id[pid]]
        for k in range(1, len(steps)):                      # prefix 0..k-1 -> next = k
            s = slice(max(0, k - max_len), k)
            ex.append({"step": steps[s], "exc": excs[s], "t": t[s],
                       "direction": d, "target": int(nxt[k])})
    return ex


def collate_steps(batch, device):
    B = len(batch)
    L = max(len(e["step"]) for e in batch)
    step = np.zeros((B, L), np.int64); exc = np.zeros((B, L), np.int64)
    dt = np.zeros((B, L), np.float32); pad = np.ones((B, L), bool)
    direction = np.zeros(B, np.int64); target = np.zeros(B, np.int64)
    for i, e in enumerate(batch):
        n = len(e["step"]); step[i, :n] = e["step"]; exc[i, :n] = e["exc"]
        d = np.diff(e["t"], prepend=e["t"][:1]); dt[i, :n] = np.clip(d, 0, None)
        pad[i, :n] = False; direction[i] = e["direction"]; target[i] = e["target"]
    z = torch.zeros((B, L), dtype=torch.long, device=device)
    return {"step": torch.tensor(step, device=device), "exc": torch.tensor(exc, device=device),
            "dt": torch.tensor(dt, device=device), "dow": z, "dom": z, "month": z,
            "pad_mask": torch.tensor(pad, device=device),
            "direction": torch.tensor(direction, device=device),
            "target": torch.tensor(target, device=device)}


def run_inflight(evt, pay, tr_ids, ev_ids, schema, device, epochs, smoke, log=print):
    from sklearn.metrics import average_precision_score, f1_score
    from encoder.history_encoder import HistoryConfig

    twin = schema["twin"]
    step_vocab = {s: i for i, s in enumerate(sorted({s for w in twin["workflow"].values() for s in w}))}
    codes = ["none"] + twin["exception_codes"]
    exc_vocab = {c: i for i, c in enumerate(codes)}
    next_vocab = exc_vocab
    dir_vocab = {d: i for i, d in enumerate(twin["directions"])}
    D = 64 if smoke else 128
    hcfg = HistoryConfig(hidden=D, layers=2 if smoke else 4, heads=2 if smoke else 8,
                         ff_mult=2 if smoke else 4)

    MAX_PREFIX = 16                                  # workflows are <= 7 steps; ample
    tr_ex = build_step_examples(evt[evt.payment_id.isin(tr_ids)], pay, step_vocab,
                                exc_vocab, dir_vocab, next_vocab, MAX_PREFIX)
    ev_ex = build_step_examples(evt[evt.payment_id.isin(ev_ids)], pay, step_vocab,
                                exc_vocab, dir_vocab, next_vocab, MAX_PREFIX)
    log(f"[in-flight] prefixes: train {len(tr_ex):,} / held-out {len(ev_ex):,}")
    if not tr_ex or not ev_ex:
        return {"note": "not enough step prefixes"}

    model = Inflight(len(step_vocab), len(exc_vocab), len(dir_vocab), len(next_vocab), hcfg).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    bs = 128
    model.train()
    for ep in range(epochs):
        perm = np.random.permutation(len(tr_ex))
        tot = 0.0
        for s in range(0, len(perm), bs):
            b = collate_steps([tr_ex[i] for i in perm[s:s + bs]], device)
            logits = model(b)
            loss = nn.functional.cross_entropy(logits, b["target"])
            opt.zero_grad(); loss.backward(); opt.step(); tot += float(loss)
        log(f"  in-flight epoch {ep+1}/{epochs}  loss {tot/max(1,len(perm)//bs):.4f}")

    model.eval()
    probs, tgts = [], []
    with torch.no_grad():
        for s in range(0, len(ev_ex), bs):
            b = collate_steps(ev_ex[s:s + bs], device)
            probs.append(torch.softmax(model(b), dim=1).cpu().numpy())
            tgts.append(b["target"].cpu().numpy())
    P = np.concatenate(probs); y = np.concatenate(tgts)
    pred = P.argmax(1)
    # binary: does the next step throw ANY exception? (class 0 == 'none')
    any_exc = (y != 0).astype(int)
    p_exc = 1.0 - P[:, 0]
    return {
        "n_eval_prefixes": int(len(y)),
        "next_exception_macro_f1": float(f1_score(y, pred, average="macro")),
        "next_any_exception_pr_auc": (float(average_precision_score(any_exc, p_exc))
                                      if len(set(any_exc)) > 1 else None),
        "next_any_exception_prevalence": float(any_exc.mean()),
    }


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #

def main():
    from encoder.tabular_encoder import EncoderConfig, build_pretraining_stack
    from encoder.tabular_encoder import pretrain as enc_pretrain

    ap = argparse.ArgumentParser(description="payment-level digital twin")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--payments", default=str(ROOT / "data" / "pacs008_twin_payments.parquet"))
    ap.add_argument("--events", default=str(ROOT / "data" / "pacs008_twin_events.parquet"))
    ap.add_argument("--schema", default=str(ROOT / "data" / "column_schema_twin.json"))
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--inflight-epochs", type=int, default=4)
    ap.add_argument("--demo", type=int, default=0,
                    help="print the twin's forecast + anomaly score for N held-out payments")
    ap.add_argument("--out", default=str(ROOT / "results_twin.json"))
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device={device}  mode={'smoke' if args.smoke else 'full'}")
    pay, evt, schema = load_twin(args)
    twin = schema["twin"]
    print(f"payments {len(pay):,} | events {len(evt):,} | "
          f"status {pay['terminal_status'].value_counts().to_dict()}")

    # frozen v1 encoder -> f(payment)
    enc_cfg = (EncoderConfig(hidden=64, layers=2, heads=2, ff_mult=2, epochs=1)
               if args.smoke else EncoderConfig())
    torch.manual_seed(0)
    encoder, assembler, vocabs = build_pretraining_stack(pay, schema, enc_cfg, party_epochs=1)
    encoder.to(device)
    print("[A] pretrain v1 encoder ...")
    enc_pretrain(encoder, _to_device(vocabs.encode(pay), device), enc_cfg,
                 batch_size=128 if args.smoke else 256)
    encoder.freeze()
    e_pay = embed_all_rows(encoder, vocabs.encode(pay), len(pay), device).cpu().numpy()

    # held-out split on payment ids
    rng = np.random.default_rng(0)
    idx = rng.permutation(len(pay))
    cut = int(len(idx) * 0.8)
    tr, ev = idx[:cut], idx[cut:]
    tr_ids = set(pay["payment_id"].to_numpy()[tr]); ev_ids = set(pay["payment_id"].to_numpy()[ev])

    intake = intake_eval(e_pay, pay, twin, tr, ev)
    print("[intake] exception PR-AUC:",
          {k: (round(v, 3) if v is not None else None) for k, v in intake["exception_pr_auc"].items()})
    print(f"[intake] status acc {intake['status_accuracy']:.3f} (majority {intake['status_majority_baseline']:.3f}) "
          f"macroF1 {intake['status_macro_f1']:.3f} | "
          f"ETA MAE {intake['eta_mae_min']:.0f} vs baseline {intake['eta_baseline_mae_min']:.0f} min")

    if args.demo:
        demo(e_pay, pay, twin, tr, ev, args.demo)

    inflight = run_inflight(evt, pay, tr_ids, ev_ids, schema, device, args.inflight_epochs, args.smoke)
    if "note" not in inflight:
        print(f"[in-flight] next-exception macroF1 {inflight['next_exception_macro_f1']:.3f} | "
              f"next-any PR-AUC {inflight['next_any_exception_pr_auc']} "
              f"(prev {inflight['next_any_exception_prevalence']:.2f})")

    results = {"mode": "smoke" if args.smoke else "full", "device": device,
               "n_payments": int(len(pay)), "intake": intake, "in_flight": inflight}
    Path(args.out).write_text(json.dumps(results, indent=2, default=float))
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
