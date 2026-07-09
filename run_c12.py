"""
C12 - life-long milestones (PRAGMA's timed profile items) at the intake grain.

Per payment, prior-only milestone features (tenure, first cross-border, first
recall, prior count - data/lifelong.py) are fused with the frozen f(x):
probe on [f(x)] vs probe on [f(x) + lifelong], rail / risk / return heads,
same payment split. Measured on the documented weak spot: the YOUNG-account
slice (few prior events) vs the full slice.

Honesty note (pre-stated): the default generator plants NO tenure->label
correlation, so the expected full-slice result is a null - the point is the
measured feature path (and the young-slice cold-start question). Under
--momentum > 0 (C11's knob) first_recall becomes genuinely predictive of
returns; both readings are reported.

  python run_c12.py --smoke
  python run_c12.py                # 4000x20k, full encoder (CPU ~1h)
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent


def _acc(clf_x, y, tr, ev, young_ev):
    from sklearn.linear_model import LogisticRegression
    clf = LogisticRegression(max_iter=2000, class_weight="balanced").fit(clf_x[tr], y[tr])
    pred = clf.predict(clf_x[ev])
    full = float((pred == y[ev]).mean())
    young = float((clf.predict(clf_x[young_ev]) == y[young_ev]).mean()) if len(young_ev) else float("nan")
    return full, young


def _pr(clf_x, y, tr, ev):
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import average_precision_score
    if len(set(y[tr].tolist())) < 2:
        return float("nan")
    clf = LogisticRegression(max_iter=2000, class_weight="balanced").fit(clf_x[tr], y[tr])
    return float(average_precision_score(y[ev], clf.predict_proba(clf_x[ev])[:, 1]))


def main():
    from data.lifelong import LIFELONG_COLS, lifelong_features
    from data.synth_india_rails import IndiaConfig, build_dataset, build_schema
    from run_seq import frozen_embeddings

    ap = argparse.ArgumentParser(description="C12 - life-long milestone fusion at intake")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--accounts", type=int, default=4000)
    ap.add_argument("--payments", type=int, default=20000)
    ap.add_argument("--momentum", type=float, default=0.0,
                    help="recall_momentum for the fleet (C11 knob; 0 = default null)")
    ap.add_argument("--young", type=int, default=2, help="young slice: n_prior <= this")
    ap.add_argument("--out", default=str(ROOT / "results_c12.json"))
    args = ap.parse_args()

    torch.manual_seed(0); np.random.seed(0)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if args.smoke:
        args.accounts, args.payments = 150, 1200
    pay, _, accs = build_dataset(IndiaConfig(
        num_accounts=args.accounts, num_payments=args.payments, seed=23,
        recall_momentum=args.momentum))
    schema = build_schema(pay, accs)
    print(f"device={device}  payments {len(pay):,}  momentum {args.momentum}")

    encoder, vocabs, e_pay, _ = frozen_embeddings(pay, schema, args.smoke, device)
    ll = lifelong_features(pay).to_numpy()
    fused = np.hstack([e_pay, ll])
    n_prior = np.expm1(ll[:, LIFELONG_COLS.index("ll_n_prior")])

    rng = np.random.default_rng(0)
    idx = rng.permutation(len(pay))
    cut = int(len(idx) * 0.8)
    tr, ev = idx[:cut], idx[cut:]
    young_ev = ev[n_prior[ev] <= args.young]
    print(f"eval {len(ev):,} of which young (n_prior<={args.young}) {len(young_ev):,}")

    res = {"mode": "smoke" if args.smoke else "full", "momentum": args.momentum,
           "n_eval": int(len(ev)), "n_young": int(len(young_ev)), "tasks": {}}
    print("\nC12 (base f(x) vs fused +lifelong):")
    for task, col in (("rail", "rail"), ("risk", "risk_label")):
        y = pay[col].astype(str).to_numpy()
        bf, by = _acc(e_pay, y, tr, ev, young_ev)
        ff, fy = _acc(fused, y, tr, ev, young_ev)
        res["tasks"][task] = {"base_full": bf, "fused_full": ff,
                              "base_young": by, "fused_young": fy}
        print(f"  {task:6} full {bf:.3f} -> {ff:.3f} ({(ff-bf)*100:+.1f}pp) | "
              f"young {by:.3f} -> {fy:.3f} ({(fy-by)*100:+.1f}pp)")
    y = pay["returned"].to_numpy()
    b, f = _pr(e_pay, y, tr, ev), _pr(fused, y, tr, ev)
    res["tasks"]["return_pr_auc"] = {"base": b, "fused": f}
    print(f"  return PR-AUC {b:.3f} -> {f:.3f} ({(f-b)*100:+.1f}pp)"
          f"  (prevalence {y.mean():.3%})")

    deltas_young = [res["tasks"][t]["fused_young"] - res["tasks"][t]["base_young"]
                    for t in ("rail", "risk")]
    deltas_full = [res["tasks"][t]["fused_full"] - res["tasks"][t]["base_full"]
                   for t in ("rail", "risk")]
    res["C12_pass"] = bool(np.nanmean(deltas_young) >= 0.01
                           and np.nanmean(deltas_full) >= -0.005)
    print(f"  -> C12 {'PASS' if res['C12_pass'] else 'FAIL/NULL'} "
          f"(young lift >= +1pp without full-slice damage)")
    Path(args.out).write_text(json.dumps(res, indent=2))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
