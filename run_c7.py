"""
C7 - paper-native sequences: can Eq. 5 multi-record interleaving, plus one derived
gap-band column, match the v2 history encoder on the C3/C4 regime task?

The source paper's decoder already contains a sequence interface: M records of one
account interleaved as [R1] Phi(f(x_1)) ... [RM] Phi(f(x_M)) with distinct record
markers (their recurrence task). What it lacks vs our v2 history encoder is explicit
TIME - so each record gains `gap_band`, the bucketed days since the account's
previous event, added table-natively to the CORE columns (the paper's own
quantize-the-number philosophy, applied as an input column). No new machinery.

Records are sampled EVENLY over the account's history (take="spread"): the regime
label is a mid-history change, and a first-R or last-R window would see only one
regime. R defaults to 8 = the generator's minimum events per account, so no actor
is dropped.

Pre-registered (docs/V2_DIRECTION.md): C7 passes if held-out PR-AUC is within 5 pp
of the v2 history-encoder incumbent at matched scale (full-scale incumbent 0.919,
results_seq_full.json). If it passes, the v2 history encoder is deletable and the
whole system is the paper's stack. Same actor split as run_seq (seed 0, 20% eval).

  python run_c7.py --smoke          # MockLLM + tiny encoder, plumbing check
  python run_c7.py                  # full: frozen Phi-1.5 (GPU - runbook run G)
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parent

GAP_BANDS = ("first", "0d", "1-2d", "3-7d", "8-14d", "15-30d", ">30d")
_GAP_EDGES = (0, 2, 7, 14, 30)          # upper-inclusive day edges after "0d"


def add_gap_band(df, actor_col="DbtrAcct_Id", date_col="IntrBkSttlmDt"):
    """Derived core column: bucketed days since the same actor's previous event.
    Table-native time - the encoder embeds it like any categorical column."""
    df = df.reset_index(drop=True).copy()
    dates = pd.to_datetime(df[date_col])
    band = np.full(len(df), "first", dtype=object)
    for _, idx in df.groupby(actor_col).groups.items():
        pos = np.asarray(idx, dtype=np.int64)
        pos = pos[np.argsort(dates.values[pos])]
        gaps = np.diff(dates.values[pos]).astype("timedelta64[D]").astype(int)
        lab = np.array(GAP_BANDS)[np.searchsorted(_GAP_EDGES, gaps, side="left") + 1]
        band[pos[1:]] = lab
    df["gap_band"] = band
    return df


def main():
    from data.sequence_assembly import split_by_actor
    from encoder.tabular_encoder import EncoderConfig, build_pretraining_stack
    from encoder.tabular_encoder import pretrain as enc_pretrain
    from run_gpu import (_index_batch, _multi_records, _recurrence_groups, _to_device,
                         build_task_specs, load_data_and_schema, train_multitask)

    ap = argparse.ArgumentParser(description="C7 - paper-native Eq.5 sequences vs the "
                                             "v2 history encoder (regime task)")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--data", default=str(ROOT / "data" / "pacs008_seq.parquet"))
    ap.add_argument("--schema", default=str(ROOT / "data" / "column_schema_seq.json"))
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--records", type=int, default=8,
                    help="R records interleaved per account (evenly spread)")
    ap.add_argument("--epochs", type=int, default=1, help="paper pins 1 decoder epoch")
    ap.add_argument("--llm", default="microsoft/phi-1_5")
    ap.add_argument("--out", default=str(ROOT / "results_c7.json"))
    args = ap.parse_args()

    torch.manual_seed(0); np.random.seed(0)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device={device}  mode={'smoke' if args.smoke else 'full'}  R={args.records}")
    df, schema = load_data_and_schema(args)
    df = add_gap_band(df)
    schema = dict(schema)
    schema["buckets"] = {**schema["buckets"],
                         "core": list(schema["buckets"]["core"]) + ["gap_band"]}
    et = schema["entity_task"]
    print(f"rows {len(df):,}  actors {df[et['actor']].nunique():,}  "
          f"gap bands {df['gap_band'].value_counts().to_dict()}")

    # frozen encoder over the gap-band-augmented table (matched to run_seq's config)
    enc_cfg = (EncoderConfig(hidden=64, layers=2, heads=2, ff_mult=2, epochs=1)
               if args.smoke else EncoderConfig())
    encoder, _, vocabs = build_pretraining_stack(df, schema, enc_cfg, party_epochs=1)
    encoder.to(device)
    print("[A] pretrain v1 encoder (with gap_band core column) ...")
    enc_pretrain(encoder, _to_device(vocabs.encode(df), device), enc_cfg,
                 batch_size=128 if args.smoke else 256)
    encoder.freeze()

    # ONE multi-record task: the paper's Eq. 5, records spread over the history
    task = {"name": "regime", "label_column": et["label_column"],
            "label_values": list(et["label_values"]), "records": "multi",
            "group_column": et["actor"], "take": "spread"}
    if args.smoke:
        from decoder.multimodal_decoder import MockLLM
        llm = MockLLM(vocab_size=64, hidden=enc_cfg.hidden, num_layers=2, num_heads=4)
    else:
        from decoder.multimodal_decoder import HFCausalLM
        llm = HFCausalLM(args.llm)
    llm = llm.to(device)
    specs = build_task_specs({"tasks": [task]}, llm, args.smoke, device)
    from decoder.multimodal_decoder import DecoderConfig, MultimodalDecoder
    dec = MultimodalDecoder(encoder, llm, DecoderConfig(
        n_tasks=1, max_records=args.records, phi_mode="prompt")).to(device)

    # same actor partition as run_seq: split_by_actor(seed=0) over per-actor stubs
    stubs = [{"actor": a} for a in df[et["actor"]].unique()]
    tr_stubs, ev_stubs = split_by_actor(stubs, frac_eval=0.2, seed=0)
    ev_actors = {s["actor"] for s in ev_stubs}
    tdf = df[~df[et["actor"]].isin(ev_actors)].reset_index(drop=True)
    edf = df[df[et["actor"]].isin(ev_actors)].reset_index(drop=True)

    full_tr = _to_device(vocabs.encode(tdf), device)
    train_multitask(dec, specs, tdf, full_tr, R=args.records, epochs=args.epochs,
                    batch_size=16 if args.smoke else 32, device=device, log=print,
                    label="C7 paper-native")

    # held-out actors: PR-AUC of p(Shift) from the answer-token distribution
    spec = specs[0]
    groups, labels = _recurrence_groups(edf, spec, args.records)
    y = np.array([1 if s == et["positive_class"] else 0 for s in labels])
    full_ev = _to_device(vocabs.encode(edf), device)
    probs = []
    with torch.no_grad():
        for s in range(0, len(groups), 64):
            chunk = groups[s:s + 64]
            recs = _multi_records(full_ev, chunk, device)
            tids = torch.full((len(chunk),), spec["task_id"], dtype=torch.long,
                              device=device)
            instr = spec["instr"].unsqueeze(0).expand(len(chunk), -1)
            p = dec.predict_proba(recs, tids, instr, spec["answers"])
            probs.append(p.cpu().numpy())
    probs = np.vstack(probs)
    pos_j = list(spec["label_values"]).index(et["positive_class"])
    from sklearn.metrics import average_precision_score
    pr = float(average_precision_score(y, probs[:, pos_j]))

    res = {"mode": "smoke" if args.smoke else "full", "device": device,
           "R": args.records, "n_eval_actors": int(len(y)),
           "prevalence": float(y.mean()), "c7_paper_native_pr_auc": pr,
           "incumbent_hist_encoder_pr_auc": 0.919,
           "incumbent_note": "full-scale run_seq h_USR probe (results_seq_full.json); "
                             "compare at MATCHED scale - smoke incumbents differ",
           "threshold": "pass if within 5 pp of the incumbent at matched scale"}
    print(f"\nC7 (held-out actors n={len(y)}, prevalence {y.mean():.3f}):")
    print(f"  paper-native Eq.5 + gap_band  PR-AUC {pr:.3f}")
    print(f"  v2 history encoder incumbent  PR-AUC 0.919 (full scale)")
    print(f"  -> C7 verdict at matched scale only; this run: "
          f"{'smoke plumbing' if args.smoke else 'full'}")
    Path(args.out).write_text(json.dumps(res, indent=2))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
