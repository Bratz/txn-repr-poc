"""
India multi-rail twin - end to end on the backbone (docs/INDIA_RAILS.md).

Runs entirely on CPU (no frozen-LLM path). The frozen v1 encoder gives f(payment); the v2
Layer-3b history encoder (architecture only) gives the in-flight representation over steps.

  INTAKE (v1)   payment f(x) -> frozen-rep probes:
                  - rail routing (RTGS/NEFT/IMPS/UPI/SWIFT) vs a TREE baseline on the raw
                    visible features (the recurring honesty check: trees are strong on
                    visible-feature tasks)
                  - exception likelihood incl. sla_breach / limit_exceeded (PR-AUC per type)
                  - terminal status, time-to-settle (ETA, MAE vs a mean baseline)
  IN-FLIGHT (v2) prefix of (step, outcome, time) events, conditioned on the RAIL -> history
                  encoder -> next-step exception prediction

All scored on HELD-OUT payments. Beyond arXiv:2410.07851; never touches run_gpu.py.
  --smoke : tiny configs + CPU + row caps, validates the whole chain.

  python data/synth_india_rails.py --payments 60000   # generate first
  python run_india.py --smoke                          # CPU smoke
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from run_seq import frozen_embeddings
from run_twin import Inflight, collate_steps          # generic; reused as-is

ROOT = Path(__file__).resolve().parent


def load_india(args):
    schema = json.loads(Path(args.schema).read_text())

    def _read(p):
        p = Path(p)
        return pd.read_parquet(p) if p.suffix == ".parquet" else pd.read_csv(p)
    pay, evt = _read(args.payments), _read(args.events)
    if args.limit:
        keep = set(pay["payment_id"].to_numpy()[: args.limit])
        pay = pay[pay["payment_id"].isin(keep)].reset_index(drop=True)
        evt = evt[evt["payment_id"].isin(keep)].reset_index(drop=True)
    return pay, evt, schema


# --------------------------------------------------------------------------- #
# INTAKE - frozen-rep probes (+ tree baseline for routing)
# --------------------------------------------------------------------------- #

def _visible_features(pay):
    """Raw features a tree could legitimately use at intake (no rail-derived leakage)."""
    return np.hstack([
        np.log1p(pay["IntrBkSttlmAmt"].to_numpy(float)).reshape(-1, 1),
        pd.factorize(pay["identifier_type"])[0].reshape(-1, 1),
        (pay["Dbtr_Ctry"].to_numpy() != pay["Cdtr_Ctry"].to_numpy()).astype(int).reshape(-1, 1),
        pd.factorize(pay["Ccy"])[0].reshape(-1, 1),
        pd.factorize(pay["Dbtr_Industry"])[0].reshape(-1, 1),
        pd.factorize(pay["Cdtr_Industry"])[0].reshape(-1, 1),
    ]).astype(float)


def train_probes(e, pay, schema, rows):
    """Fit the deployable intake probes (encoder-based) on `rows` -> a picklable dict.

    The tree baseline is a training-time diagnostic only (its raw-feature factorize codes
    aren't stable across datasets), so it is NOT part of the saved model.
    """
    from sklearn.linear_model import LogisticRegression, Ridge
    twin = schema["twin"]
    probes = {
        "rail": LogisticRegression(max_iter=2000, class_weight="balanced").fit(
            e[rows], pay[twin["rail_column"]].to_numpy()[rows]),
        "status": LogisticRegression(max_iter=1000, class_weight="balanced").fit(
            e[rows], pay[twin["status_column"]].to_numpy()[rows]),
        "eta": Ridge().fit(e[rows], pay[twin["eta_column"]].to_numpy(float)[rows]),
        "exc": {}, "exc_columns": list(twin["exc_columns"]),
    }
    for c in twin["exc_columns"]:
        y = pay[c].to_numpy()
        if len(set(y[rows])) > 1:
            probes["exc"][c.replace("exc_", "")] = LogisticRegression(
                max_iter=1000, class_weight="balanced").fit(e[rows], y[rows])

    # §5 single-record task heads on the SAME frozen backbone (risk / geography / expense).
    # Read from the schema tasks manifest; train only those present + non-degenerate.
    probes["tasks"] = {}
    for t in schema.get("tasks", []):
        col = t["label_column"]
        if t.get("records") == "multi" or col not in pay.columns:
            continue                                       # recurrence is multi-record: skip
        if col == twin["rail_column"]:
            continue                                       # rail handled by probes["rail"]
        y = pay[col].to_numpy()
        if len(set(y[rows])) > 1:
            probes["tasks"][t["name"]] = LogisticRegression(
                max_iter=2000, class_weight="balanced").fit(e[rows], y[rows])
    return probes


def intake_eval(e, pay, schema, tr, ev):
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.linear_model import LogisticRegression, Ridge
    from sklearn.metrics import (accuracy_score, average_precision_score,
                                 f1_score, mean_absolute_error)
    twin = schema["twin"]
    out = {}

    # --- rail routing (multiclass): frozen-rep probe vs tree on raw visible features ---
    # NB: the SWIFT class is trivially separable (foreign Ccy/country leak it) and the
    # over-cap/below-min injected rows carry the *attempted* (ineligible) rail as label, so
    # the headline 5-class accuracy is inflated. We additionally report PER-CLASS metrics, a
    # DOMESTIC-only accuracy (the genuinely non-trivial RTGS/NEFT/IMPS slice), and a
    # mis-routed-excluded ("clean") accuracy so the inflation is visible, not hidden.
    from sklearn.metrics import precision_recall_fscore_support
    from data.rails import DOMESTIC_RAILS
    yr = pay[twin["rail_column"]].to_numpy()
    probe = LogisticRegression(max_iter=2000, class_weight="balanced").fit(e[tr], yr[tr])
    ppred = probe.predict(e[ev])
    Xt = _visible_features(pay)
    tree = HistGradientBoostingClassifier(max_iter=200).fit(Xt[tr], yr[tr])
    tpred = tree.predict(Xt[ev])
    maj = Counter(yr[tr]).most_common(1)[0][0]

    labels = sorted(set(yr[tr]))
    prec, rec, f1c, sup = precision_recall_fscore_support(
        yr[ev], ppred, labels=labels, average=None, zero_division=0)
    per_class = {lab: {"precision": float(prec[i]), "recall": float(rec[i]),
                       "f1": float(f1c[i]), "support": int(sup[i])}
                 for i, lab in enumerate(labels)}
    dom = np.isin(yr[ev], list(DOMESTIC_RAILS))
    clean = (~pay["is_mis_routed"].to_numpy()[ev].astype(bool)
             if "is_mis_routed" in pay.columns else np.ones(len(ev), bool))
    out["rail_routing"] = {
        "probe_accuracy": float(accuracy_score(yr[ev], ppred)),
        "probe_macro_f1": float(f1_score(yr[ev], ppred, average="macro")),
        "tree_accuracy": float(accuracy_score(yr[ev], tpred)),
        "tree_macro_f1": float(f1_score(yr[ev], tpred, average="macro")),
        "majority_baseline": float((yr[ev] == maj).mean()),
        "domestic_probe_accuracy": (float((ppred[dom] == yr[ev][dom]).mean())
                                    if dom.any() else None),
        "clean_probe_accuracy": (float((ppred[clean] == yr[ev][clean]).mean())
                                 if clean.any() else None),
        "per_class": per_class,
    }

    # --- exceptions (binary PR-AUC per type, incl. sla_breach / limit_exceeded) ---
    def probe_bin(y):
        if len(set(y[tr])) < 2 or len(set(y[ev])) < 2:
            return None
        c = LogisticRegression(max_iter=1000, class_weight="balanced").fit(e[tr], y[tr])
        return float(average_precision_score(y[ev], c.predict_proba(e[ev])[:, 1]))
    out["exception_pr_auc"] = {c.replace("exc_", ""): probe_bin(pay[c].to_numpy())
                               for c in twin["exc_columns"]}

    # --- terminal status ---
    ys = pay[twin["status_column"]].to_numpy()
    c = LogisticRegression(max_iter=1000, class_weight="balanced").fit(e[tr], ys[tr])
    spred = c.predict(e[ev])
    smaj = Counter(ys[tr]).most_common(1)[0][0]
    out["status"] = {"accuracy": float(accuracy_score(ys[ev], spred)),
                     "macro_f1": float(f1_score(ys[ev], spred, average="macro")),
                     "majority_baseline": float((ys[ev] == smaj).mean())}

    # --- ETA (minutes) ---
    yt = pay[twin["eta_column"]].to_numpy(float)
    reg = Ridge().fit(e[tr], yt[tr])
    out["eta"] = {"mae_min": float(mean_absolute_error(yt[ev], reg.predict(e[ev]))),
                  "baseline_mae_min": float(mean_absolute_error(
                      yt[ev], np.full(len(ev), yt[tr].mean())))}
    return out


# --------------------------------------------------------------------------- #
# DEMO - per-payment forecast vs ground truth (held-out)
# --------------------------------------------------------------------------- #

def demo(e, pay, schema, tr, ev, log=print):
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.linear_model import LogisticRegression, Ridge
    from data.rails import eligible_rails
    twin = schema["twin"]
    yr = pay[twin["rail_column"]].to_numpy()

    # train the intake models once on the train split
    rail = LogisticRegression(max_iter=2000, class_weight="balanced").fit(e[tr], yr[tr])
    Xt = _visible_features(pay)
    tree = HistGradientBoostingClassifier(max_iter=200).fit(Xt[tr], yr[tr])
    status = LogisticRegression(max_iter=1000, class_weight="balanced").fit(
        e[tr], pay[twin["status_column"]].to_numpy()[tr])
    eta = Ridge().fit(e[tr], pay[twin["eta_column"]].to_numpy(float)[tr])
    exc_models = {}
    for c in twin["exc_columns"]:
        y = pay[c].to_numpy()
        if len(set(y[tr])) > 1:
            exc_models[c.replace("exc_", "")] = LogisticRegression(
                max_iter=1000, class_weight="balanced").fit(e[tr], y[tr])

    # one held-out payment per rail
    rails_ev = yr[ev]
    picks = [ev[np.where(rails_ev == r)[0][0]] for r in twin["rails"]
             if (rails_ev == r).any()]

    log("\n=== per-payment forecast on held-out payments (predicted | ACTUAL) ===")
    log("(exception scores are uncalibrated balanced-probe risk rankings, not probabilities)")
    for i in picks:
        row = pay.iloc[i]
        xi = e[i:i + 1]
        pr = rail.predict(xi)[0]; conf = float(rail.predict_proba(xi)[0].max())
        ptree = tree.predict(Xt[i:i + 1])[0]
        pst = status.predict(xi)[0]
        peta = max(0.0, float(eta.predict(xi)[0]))          # settle time can't be negative
        top = sorted(((n, float(m.predict_proba(xi)[0, 1])) for n, m in exc_models.items()),
                     key=lambda kv: -kv[1])[:3]
        actual_exc = [c.replace("exc_", "") for c in twin["exc_columns"] if row[c] == 1] or ["none"]
        xb = "x-border" if row["Dbtr_Ctry"] != row["Cdtr_Ctry"] else "domestic"
        log(f"\npayment {int(row['payment_id'])}  {row['IntrBkSttlmAmt']:,.0f} {row['Ccy']}  "
            f"{xb}  via {row['identifier_type']}")
        log(f"  predicted rail: {pr} ({conf:.2f})  [tree: {ptree}]   "
            f"status: {pst}   ETA: {peta:.0f} min")
        log(f"  top exception risks: " + ", ".join(f"{n} {p:.2f}" for n, p in top))
        log(f"  ACTUAL: rail {row['rail']} | status {row['terminal_status']} | "
            f"exceptions {actual_exc} | ETA {row['time_to_settle_min']:.0f} min")

    # deterministic routing-rule sanity table (data/rails.eligible_rails)
    log("\n=== routing eligibility by amount (INR) / identifier ===")
    for amt in (500, 50_000, 150_000, 300_000, 800_000):
        log(f"  Rs {amt:>9,}  domestic -> {eligible_rails(amt)}   "
            f"x-border -> {eligible_rails(amt, xborder=True)}")


# --------------------------------------------------------------------------- #
# IN-FLIGHT - rail-conditioned next-step exception
# --------------------------------------------------------------------------- #

def build_rail_examples(evt, pay, step_vocab, exc_vocab, rail_vocab, next_vocab, max_len):
    """Every prefix of a payment's primary step-events -> the next step's exception (or
    'none'), conditioned on the rail (which determines the workflow)."""
    rail_by_id = dict(zip(pay["payment_id"], pay["rail"]))
    ex = []
    evt = evt[evt["outcome"] != "repaired"]                 # one row per step
    for pid, sub in evt.groupby("payment_id"):
        sub = sub.sort_values("seq")
        steps = sub["step"].map(step_vocab).to_numpy()
        excs = sub["excode"].map(exc_vocab).to_numpy()
        t = sub["t_min"].to_numpy(dtype=np.float32)
        nxt = sub["excode"].map(next_vocab).to_numpy()
        r = rail_vocab[rail_by_id[pid]]
        for k in range(1, len(steps)):
            s = slice(max(0, k - max_len), k)
            ex.append({"step": steps[s], "exc": excs[s], "t": t[s],
                       "direction": r, "target": int(nxt[k])})   # "direction" slot = rail
    return ex


def run_inflight(evt, pay, tr_ids, ev_ids, schema, device, epochs, smoke, log=print):
    from sklearn.metrics import average_precision_score, f1_score
    from encoder.history_encoder import HistoryConfig

    twin = schema["twin"]
    step_vocab = {s: i for i, s in
                  enumerate(sorted({s for w in twin["workflow"].values() for s in w}))}
    codes = ["none"] + twin["exception_codes"]
    exc_vocab = {c: i for i, c in enumerate(codes)}
    rail_vocab = {r: i for i, r in enumerate(twin["rails"])}
    D = 64 if smoke else 128
    hcfg = HistoryConfig(hidden=D, layers=2 if smoke else 4, heads=2 if smoke else 8,
                         ff_mult=2 if smoke else 4)
    MAX_PREFIX = 16

    tr = build_rail_examples(evt[evt.payment_id.isin(tr_ids)], pay, step_vocab,
                             exc_vocab, rail_vocab, exc_vocab, MAX_PREFIX)
    ev = build_rail_examples(evt[evt.payment_id.isin(ev_ids)], pay, step_vocab,
                             exc_vocab, rail_vocab, exc_vocab, MAX_PREFIX)
    log(f"[in-flight] prefixes: train {len(tr):,} / held-out {len(ev):,}")
    if not tr or not ev:
        return {"note": "not enough step prefixes"}

    model = Inflight(len(step_vocab), len(exc_vocab), len(rail_vocab), len(exc_vocab), hcfg).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    bs = 128
    model.train()
    for ep in range(epochs):
        perm = np.random.permutation(len(tr))
        tot = 0.0
        for s in range(0, len(perm), bs):
            b = collate_steps([tr[i] for i in perm[s:s + bs]], device)
            loss = torch.nn.functional.cross_entropy(model(b), b["target"])
            opt.zero_grad(); loss.backward(); opt.step(); tot += float(loss.detach())
        log(f"  in-flight epoch {ep+1}/{epochs}  loss {tot/max(1,len(perm)//bs):.4f}")

    model.eval()
    probs, tgts = [], []
    with torch.no_grad():
        for s in range(0, len(ev), bs):
            b = collate_steps(ev[s:s + bs], device)
            probs.append(torch.softmax(model(b), dim=1).cpu().numpy())
            tgts.append(b["target"].cpu().numpy())
    P = np.concatenate(probs); y = np.concatenate(tgts)
    any_exc = (y != 0).astype(int)
    return {
        "n_eval_prefixes": int(len(y)),
        "next_exception_macro_f1": float(f1_score(y, P.argmax(1), average="macro")),
        "next_any_exception_pr_auc": (float(average_precision_score(any_exc, 1.0 - P[:, 0]))
                                      if len(set(any_exc)) > 1 else None),
        "next_any_exception_prevalence": float(any_exc.mean()),
    }


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #

def main():
    ap = argparse.ArgumentParser(description="India multi-rail twin (CPU-friendly)")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--payments", default=str(ROOT / "data" / "india_rails_payments.parquet"))
    ap.add_argument("--events", default=str(ROOT / "data" / "india_rails_events.parquet"))
    ap.add_argument("--schema", default=str(ROOT / "data" / "column_schema_india.json"))
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--demo", action="store_true", help="print per-payment forecasts vs actuals")
    ap.add_argument("--save", default=None, metavar="DIR",
                    help="persist the frozen encoder + intake probes to DIR for serve_india.py")
    ap.add_argument("--inflight-epochs", type=int, default=4)
    ap.add_argument("--out", default=str(ROOT / "results_india.json"))
    args = ap.parse_args()

    np.random.seed(0)                               # reproducible in-flight training
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device={device}  mode={'smoke' if args.smoke else 'full'}")
    pay, evt, schema = load_india(args)
    twin = schema["twin"]
    print(f"payments {len(pay):,} | events {len(evt):,} | "
          f"rails {pay['rail'].value_counts().to_dict()}")

    # frozen v1 encoder -> f(payment)
    encoder, vocabs, e_pay, enc_cfg = frozen_embeddings(pay, schema, args.smoke, device)

    # held-out split on payment ids
    rng = np.random.default_rng(0)
    idx = rng.permutation(len(pay))
    cut = int(len(idx) * 0.8)
    tr, ev = idx[:cut], idx[cut:]
    tr_ids = set(pay["payment_id"].to_numpy()[tr]); ev_ids = set(pay["payment_id"].to_numpy()[ev])

    intake = intake_eval(e_pay, pay, schema, tr, ev)
    r = intake["rail_routing"]
    print(f"[rail-routing] probe acc {r['probe_accuracy']:.3f} (5-class, SWIFT trivially "
          f"separable) | domestic-only {r['domestic_probe_accuracy']} | "
          f"clean {r['clean_probe_accuracy']} | tree acc {r['tree_accuracy']:.3f} | "
          f"majority {r['majority_baseline']:.3f}")
    print("[intake] exception PR-AUC:",
          {k: (round(v, 3) if v is not None else None) for k, v in intake["exception_pr_auc"].items()})
    s = intake["status"]
    print(f"[intake] status acc {s['accuracy']:.3f} (majority {s['majority_baseline']:.3f}) "
          f"macroF1 {s['macro_f1']:.3f} | ETA MAE {intake['eta']['mae_min']:.1f} "
          f"vs baseline {intake['eta']['baseline_mae_min']:.1f} min")

    if args.demo:
        demo(e_pay, pay, schema, tr, ev)

    if args.save:
        from encoders.quantizer import AdaptiveQuantizer
        from serve_india import save_india_model
        quantizer = AdaptiveQuantizer().fit(pay[vocabs.numerical_col].to_numpy(),
                                             pay[vocabs.ccy_col].to_numpy())
        probes = train_probes(e_pay, pay, schema, tr)        # deployable probes (train split)
        path = save_india_model(args.save, enc_cfg=enc_cfg, vocabs=vocabs, quantizer=quantizer,
                                encoder=encoder, schema=schema, probes=probes)
        print(f"[save] model -> {path}")

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
