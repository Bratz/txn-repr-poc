"""
Multi-source fusion measurement: does the eFRM channel source add signal the ISO view
cannot carry?

Three probes on the same held-out outward payments, label = atoFlag (account-takeover drain):
  ISO only      linear probe on the frozen payment embedding f(x)
  channel only  linear probe on the eFRM context features
  FUSED         probe on [f(x) (+) channel features]

By construction the ATO pattern (new device + credential change + payee-add + burst login)
lives only in the channel source, so the honest expectation is ISO ~ prevalence and
fusion ~ channel - the measurement demonstrates that the second source is NECESSARY for this
label, not that fusion is magic. On real data the interesting question is the interaction
terms; the fusion path is where they would show.

  python run_efrm.py --smoke
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from data.efrm_source import build_channel_events, ctx_matrix
from run_seq import frozen_embeddings

ROOT = Path(__file__).resolve().parent


def _pr(Xtr, ytr, Xev, yev):
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import average_precision_score
    if len(set(ytr.tolist())) < 2 or len(set(yev.tolist())) < 2:
        return float("nan")
    clf = LogisticRegression(max_iter=1000, class_weight="balanced").fit(Xtr, ytr)
    return float(average_precision_score(yev, clf.predict_proba(Xev)[:, 1]))


def main():
    ap = argparse.ArgumentParser(description="eFRM channel source - multi-source fusion eval")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--payments", default=str(ROOT / "data" / "india_rails_payments.parquet"))
    ap.add_argument("--schema", default=str(ROOT / "data" / "column_schema_india.json"))
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--out", default=str(ROOT / "results_efrm.json"))
    args = ap.parse_args()

    np.random.seed(0)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    pay = pd.read_parquet(args.payments)
    if args.limit:
        pay = pay.head(args.limit).reset_index(drop=True)
    schema = json.loads(Path(args.schema).read_text())

    ctx, events = build_channel_events(pay)
    print(f"payments {len(pay):,} -> outward with channel context {len(ctx):,} "
          f"(ATO prevalence {ctx['atoFlag'].mean():.3f}; events {len(events):,})")

    encoder, vocabs, e_pay, _ = frozen_embeddings(pay, schema, args.smoke, device)
    pos = pay.reset_index().set_index("payment_id").loc[ctx["payment_id"], "index"].to_numpy()
    E = e_pay[pos]                                   # ISO embedding per context row
    C = ctx_matrix(ctx)                              # channel features per context row
    y = ctx["atoFlag"].to_numpy()

    idx = np.random.permutation(len(y)); cut = int(len(y) * 0.8)
    tr, ev = idx[:cut], idx[cut:]
    res = {
        "n": int(len(y)), "prevalence": float(y.mean()),
        "iso_only_pr_auc": _pr(E[tr], y[tr], E[ev], y[ev]),
        "channel_only_pr_auc": _pr(C[tr], y[tr], C[ev], y[ev]),
        "fused_pr_auc": _pr(np.hstack([E, C])[tr], y[tr], np.hstack([E, C])[ev], y[ev]),
    }
    print(f"ATO detection (held-out, prevalence {y[ev].mean():.3f}):")
    print(f"  ISO view only      PR-AUC {res['iso_only_pr_auc']:.3f}   <- blind by construction")
    print(f"  channel view only  PR-AUC {res['channel_only_pr_auc']:.3f}")
    print(f"  FUSED              PR-AUC {res['fused_pr_auc']:.3f}")
    Path(args.out).write_text(json.dumps(res, indent=2))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
