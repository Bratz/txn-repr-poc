"""
C9 - richer adapter family on the SAME frozen encoder: does LoRA close C2's gap?

Three arms on the C2 risk task (rule-generated label, positive = High), same
stratified split:

  A. probe     linear probe on frozen f(x)                     (the baseline arm)
  B. lora      LoRA adapters inside the frozen encoder + linear head, trained
               end-to-end; base weights NEVER move (encoder/lora.py freezes them
               structurally; B=0 init makes step 0 identical to arm A's encoder)
  C. catboost  CatBoost on raw factorized features             (the adversary)

Pre-registered (docs/V2_DIRECTION.md, C9): pass = arm B closes >= half of the
probe-to-CatBoost gap, with trainable-parameter counts reported honestly. A miss
localises C2's deficit at representation bandwidth, not adapter capacity.

  python run_c9.py --smoke        # 500-row reference sample, tiny encoder
  python run_c9.py --limit 20000  # CPU-feasible full-path run
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn

ROOT = Path(__file__).resolve().parent


def raw_feature_matrix(df, schema):
    """C2-style adversary features: every schema column, factorized; amount as float."""
    b = schema["buckets"]
    cols = (list(b["high_card_categorical"]) + list(b["core"]) + list(b["meta_party"]))
    X = [pd.factorize(df[c].astype(str))[0] for c in cols if c in df.columns]
    X.append(df[b["numerical"][0]].to_numpy(dtype=float))
    return np.column_stack(X)


def train_lora_arm(encoder, full, y, tr, ev, device, rank, alpha, epochs, batch_size,
                   lr=1e-3, patience=2, seed=0, log=print):
    """Arm B: inject LoRA, train adapters + a linear head, early-stop on val PR-AUC.
    Returns (pr_auc_eval, trainable_params)."""
    from sklearn.metrics import average_precision_score
    from sklearn.model_selection import train_test_split

    from encoder.lora import inject_lora, lora_parameters, mark_only_lora_trainable
    from run_gpu import _index_batch, _to_device

    n_adapted = inject_lora(encoder, rank=rank, alpha=alpha)
    mark_only_lora_trainable(encoder)
    head = nn.Linear(encoder.D, 1).to(device)
    lora_n = sum(p.numel() for p in lora_parameters(encoder))
    head_n = sum(p.numel() for p in head.parameters())
    log(f"[C9-B] {n_adapted} layers adapted; trainable = {lora_n:,} LoRA + {head_n:,} head")

    tr_idx, va_idx = train_test_split(tr, test_size=0.2, stratify=y[tr], random_state=seed)
    pos_w = torch.tensor([(y[tr_idx] == 0).sum() / max((y[tr_idx] == 1).sum(), 1)],
                         dtype=torch.float32, device=device)
    opt = torch.optim.AdamW(list(lora_parameters(encoder)) + list(head.parameters()),
                            lr=lr, weight_decay=0.01)
    params = list(lora_parameters(encoder)) + list(head.parameters())

    def _scores(idx):
        encoder.eval(); head.eval()
        out = []
        with torch.no_grad():
            for s in range(0, len(idx), 512):
                b = _to_device(_index_batch(full, torch.as_tensor(idx[s:s + 512])), device)
                out.append(head(encoder.encode(b)).squeeze(-1).cpu())
        return torch.cat(out).numpy()

    best, best_state, bad = -1.0, None, 0
    rng = np.random.default_rng(seed)
    for ep in range(epochs):
        encoder.train(); head.train()
        order = rng.permutation(len(tr_idx))
        for s in range(0, len(order), batch_size):
            idx = tr_idx[order[s:s + batch_size]]
            b = _to_device(_index_batch(full, torch.as_tensor(idx)), device)
            logits = head(encoder.encode(b)).squeeze(-1)
            loss = nn.functional.binary_cross_entropy_with_logits(
                logits, torch.as_tensor(y[idx], dtype=torch.float32, device=device),
                pos_weight=pos_w)
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()
        val = float(average_precision_score(y[va_idx], _scores(va_idx)))
        log(f"[C9-B] epoch {ep + 1}/{epochs}  val PR-AUC {val:.3f}")
        if val > best + 1e-4:
            best, bad = val, 0
            best_state = ([p.detach().clone() for p in lora_parameters(encoder)],
                          {k: v.detach().clone() for k, v in head.state_dict().items()})
        else:
            bad += 1
            if bad >= patience:
                break
    if best_state is not None:                       # restore the best-val epoch
        for p, saved in zip(lora_parameters(encoder), best_state[0]):
            p.data.copy_(saved)
        head.load_state_dict(best_state[1])
    return float(average_precision_score(y[ev], _scores(ev))), lora_n + head_n


def main():
    from run_gpu import load_data_and_schema
    from run_seq import catboost_pr, frozen_embeddings, probe_pr

    ap = argparse.ArgumentParser(description="C9 - LoRA on the frozen encoder vs "
                                             "probe vs CatBoost (C2 risk task)")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--data", default=str(ROOT / "data" / "pacs008_synth.parquet"))
    ap.add_argument("--schema", default=str(ROOT / "data" / "column_schema.json"))
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--rank", type=int, default=8)
    ap.add_argument("--alpha", type=float, default=8.0)
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--out", default=str(ROOT / "results_c9.json"))
    args = ap.parse_args()

    torch.manual_seed(0); np.random.seed(0)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    df, schema = load_data_and_schema(args)
    df = df.reset_index(drop=True)
    y = (df[schema["label_column"]].astype(str) == "High").to_numpy().astype(int)
    print(f"device={device}  rows {len(df):,}  positive High {y.mean():.3%}")

    from sklearn.model_selection import train_test_split
    idx = np.arange(len(df))
    tr, ev = train_test_split(idx, test_size=0.2, stratify=y, random_state=7)

    encoder, vocabs, e_pay, enc_cfg = frozen_embeddings(df, schema, args.smoke, device)
    full = vocabs.encode(df)

    a_pr = probe_pr(e_pay[tr], y[tr], e_pay[ev], y[ev])
    X = raw_feature_matrix(df, schema)
    c_pr = catboost_pr(X[tr], y[tr], X[ev], y[ev], iters=100 if args.smoke else 300)
    b_pr, b_params = train_lora_arm(
        encoder, full, y, tr, ev, device, rank=args.rank, alpha=args.alpha,
        epochs=2 if args.smoke else args.epochs,
        batch_size=64 if args.smoke else 128)

    gap = c_pr - a_pr
    closure = (b_pr - a_pr) / gap if gap > 1e-6 else float("nan")
    res = {"mode": "smoke" if args.smoke else "full", "device": device,
           "n_rows": int(len(df)), "prevalence": float(y.mean()),
           "probe_pr_auc": a_pr, "lora_pr_auc": b_pr, "catboost_pr_auc": c_pr,
           "lora_trainable_params": int(b_params), "rank": args.rank,
           "gap_closure": None if np.isnan(closure) else float(closure),
           "C9_pass": bool(gap > 1e-6 and closure >= 0.5)}
    print(f"\nC9 (positive=High, prevalence {y.mean():.3%}):")
    print(f"  A probe on frozen f(x)   PR-AUC {a_pr:.3f}")
    print(f"  B LoRA (r={args.rank}, {b_params:,} trainable, base frozen) "
          f"PR-AUC {b_pr:.3f}")
    print(f"  C CatBoost on raw        PR-AUC {c_pr:.3f}")
    print(f"  -> gap closure {closure:.1%} | C9 "
          f"{'PASS' if res['C9_pass'] else 'FAIL'} (threshold: >= 50%)"
          if not np.isnan(closure) else "  -> no probe-to-CatBoost gap to close")
    Path(args.out).write_text(json.dumps(res, indent=2))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
