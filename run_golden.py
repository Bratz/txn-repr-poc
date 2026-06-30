"""
Golden cross-task bench: score ONE curated set of payments across ALL tasks, on the same
frozen backbone, and exercise the live pacs.008 path on the very same cases.

Tasks covered (predicted vs the rule-true golden label):
  §5 single-record : risk, geography, expense
  India rail       : rail routing, terminal status, ETA, exception risks
  (in-flight next-step exception is a sequence task, evaluated in run_india/run_twin, not on
   these 14 cases; recurrence is multi-record / group-level -> also out of this per-payment bench)

It also writes the golden set to pacs.008 XML, re-parses it through Layer-1, and re-scores -
confirming the message-level path agrees with direct scoring on the message-native fields.

  python run_golden.py --smoke      # CPU, fast
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd
import torch

from data.golden_cases import build_golden
from data.iso20022_pacs008 import parse_pacs008_frame, write_pacs008
from data.synth_india_rails import IndiaConfig, build_dataset, build_schema
from run_india import train_probes
from run_seq import embed_all_rows
from serve_india import IndiaScorer

# task -> (golden label column, predicted column from the scorer)
SINGLE_TASKS = [
    ("risk", "risk_label", "risk_pred"),
    ("geography", "geo_label", "geography_pred"),
    ("expense", "expense_label", "expense_pred"),
    ("rail", "rail", "rail_pred"),
    ("status", "terminal_status", "status_pred"),
]


def _train_backbone(args, device):
    from encoder.tabular_encoder import EncoderConfig, build_pretraining_stack
    from encoder.tabular_encoder import pretrain as enc_pretrain
    cfg = IndiaConfig(num_accounts=600 if args.smoke else 1500,
                      num_payments=args.train_payments, seed=23)
    pay, evt, accs = build_dataset(cfg)
    schema = build_schema(pay, accs)
    enc_cfg = (EncoderConfig(hidden=64, layers=2, heads=2, ff_mult=2, epochs=1)
               if args.smoke else EncoderConfig())
    torch.manual_seed(0)
    encoder, _, vocabs = build_pretraining_stack(pay, schema, enc_cfg, party_epochs=1)
    encoder.to(device)
    enc_pretrain(encoder, vocabs.encode(pay), enc_cfg, batch_size=128 if args.smoke else 256)
    encoder.freeze()
    e = embed_all_rows(encoder, vocabs.encode(pay), len(pay), device).cpu().numpy()
    probes = train_probes(e, pay, schema, np.arange(len(pay)))
    return encoder, vocabs, probes, schema


def _agree(res, gold):
    print("\n=== golden cross-task bench (predicted vs rule-true label) ===")
    disp = pd.DataFrame({"case": gold["case"]})
    for name, lab, pred in SINGLE_TASKS:
        disp[name] = [f"{a}->{b}{'' if a == b else '  X'}"
                      for a, b in zip(gold[lab], res[pred])]
    disp["ETA exp/pred"] = [f"{a:.0f}/{b:.0f}"
                            for a, b in zip(gold["time_to_settle_min"], res["eta_min_pred"])]
    print(disp.to_string(index=False))
    print("\nper-task agreement (predicted == rule-true label):")
    for name, lab, pred in SINGLE_TASKS:
        acc = float((gold[lab].to_numpy() == res[pred].to_numpy()).mean())
        print(f"  {name:10s} {acc:.2f}  ({int(acc*len(gold))}/{len(gold)})")
    mae = float(np.abs(gold["time_to_settle_min"].to_numpy() - res["eta_min_pred"].to_numpy()).mean())
    print(f"  {'ETA':10s} MAE {mae:.1f} min")


def main():
    ap = argparse.ArgumentParser(description="Golden cross-task bench (CPU-friendly)")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--train-payments", type=int, default=12000)
    args = ap.parse_args()
    np.random.seed(0)                               # reproducible in-flight training
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device={device}  mode={'smoke' if args.smoke else 'full'}")

    encoder, vocabs, probes, schema = _train_backbone(args, device)
    scorer = IndiaScorer(encoder, vocabs, probes, device)

    gpay, gevt = build_golden()
    print(f"golden: {len(gpay)} cases | {len(gevt)} events | "
          f"tasks: {list(probes.get('tasks', {})) + ['rail', 'status', 'eta', 'exceptions']}")
    res = scorer.predict(gpay)
    _agree(res, gpay)

    # exception risk vs actual, per case
    print("\nexception forecast (actual -> top predicted risk):")
    exc_cols = [c for c in gpay.columns if c.startswith("exc_")]
    for i, row in gpay.iterrows():
        actual = [c.replace("exc_", "") for c in exc_cols if row[c] == 1] or ["none"]
        print(f"  {row['case']:20s} actual {str(actual):28s} pred {res.iloc[i]['top_exception_risks']}")

    # --- live pacs.008 round-trip on the SAME cases ---
    xml = write_pacs008(gpay.to_dict(orient="records"))
    rt = scorer.predict(parse_pacs008_frame(xml))
    rail_match = float((res["rail_pred"].to_numpy() == rt["rail_pred"].to_numpy()).mean())
    print(f"\n[pacs.008 round-trip] golden -> XML -> Layer-1 -> score: "
          f"rail predictions match direct {rail_match:.2f} "
          f"(message-native fields preserved; industry/identifier re-derived)")


if __name__ == "__main__":
    main()
