"""
Progressive-enrichment evaluation: can the TFM impute missing fields and predict in-flight
status from a PARTIAL (outbound, mid-enrichment) payment?

Reuses the encoder's masked-column reconstruction (= imputation) and a probe on f(x):
for each enrichment stage we mask the not-yet-available fields (data/enrichment_stages) and
report (a) per-field imputation top-1, and (b) in-flight status accuracy. Both should improve
as the payment enriches; inbound payments are the 'complete' row.

ponytail: no new training — the random-masked encoder already learns to reconstruct any
hidden subset (a stage is one such subset). If early-stage imputation is weak, THEN add
stage-aware masking to the pretrain loop (deferred until the numbers say it's needed).

  python run_impute.py --smoke
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from data.enrichment_stages import STAGE_ADDS, missing_mask, stage_names
from run_gpu import _to_device
from run_seq import frozen_embeddings

ROOT = Path(__file__).resolve().parent


def main():
    ap = argparse.ArgumentParser(description="progressive-enrichment imputation + in-flight status")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--payments", default=str(ROOT / "data" / "india_rails_payments.parquet"))
    ap.add_argument("--schema", default=str(ROOT / "data" / "column_schema_india.json"))
    ap.add_argument("--label", default="rail", help="in-flight status label to predict")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--out", default=str(ROOT / "results_impute.json"))
    args = ap.parse_args()

    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score

    np.random.seed(0)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    p = Path(args.payments)
    pay = pd.read_parquet(p) if p.suffix == ".parquet" else pd.read_csv(p)
    if args.limit:
        pay = pay.head(args.limit).reset_index(drop=True)
    schema = json.loads(Path(args.schema).read_text())
    print(f"device={device}  mode={'smoke' if args.smoke else 'full'}  payments {len(pay):,}")

    encoder, vocabs, _, _ = frozen_embeddings(pay, schema, args.smoke, device)
    full = _to_device(vocabs.encode(pay), device)
    recon_names = [r[0] for r in encoder.recon]
    B = len(pay)
    idx = np.random.permutation(B); cut = int(B * 0.8); tr, ev = idx[:cut], idx[cut:]
    y = pay[args.label].to_numpy()

    imp, status = {}, {}
    for si, (sname, _) in enumerate(STAGE_ADDS):
        mask_row = missing_mask(recon_names, si)
        miss = [recon_names[j] for j, m in enumerate(mask_row) if m]
        cm = (torch.tensor(mask_row, device=device).unsqueeze(0).expand(B, -1)
              if any(mask_row) else None)
        if cm is not None:
            for f, a in encoder.reconstruction_accuracy(full, cm).items():
                imp.setdefault(f, {})[sname] = a["top1"]
        with torch.no_grad():
            emb = (encoder.forward(full, cm)[0] if cm is not None
                   else encoder.encode(full)).cpu().numpy()
        clf = LogisticRegression(max_iter=1000, class_weight="balanced").fit(emb[tr], y[tr])
        status[sname] = float(accuracy_score(y[ev], clf.predict(emb[ev])))
        print(f"[{sname:9s}] missing {len(miss)}: {miss}  ->  status({args.label}) acc {status[sname]:.3f}")

    print(f"\nimputation top-1  (field x stage; '-' = field already available):")
    cols = stage_names()
    print(f"  {'field':18s} " + " ".join(f"{c[:9]:>9s}" for c in cols))
    for f in recon_names:
        cells = [f"{imp[f][c]:.2f}" if f in imp and c in imp[f] else "-" for c in cols]
        print(f"  {f:18s} " + " ".join(f"{c:>9s}" for c in cells))

    Path(args.out).write_text(json.dumps({"status_by_stage": status,
                                          "imputation_top1": imp}, indent=2))
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
