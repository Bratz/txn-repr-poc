"""
C10 - read-out bandwidth: is C2's residual deficit the single-token contract?

C9 (converged, 20k CPU) showed LoRA doubles the single-token probe (0.152 ->
0.301) but leaves ~60% of the CatBoost gap open. TGPT's diagnosis says the
bottleneck is the READ-OUT: the whole row compressed to one 512-d vector
before any head sees it. C10 widens only the read-out - same frozen base, no
adapters, same task and splits:

  single   probe on f(x) = the row token            (C9's arm A, the contract)
  mean     probe on the mean of all COLUMN tokens   (order-free wide read-out)
  concat   probe on all tokens flattened (T+1)*D    (full bandwidth)

Pre-registered (docs/V2_DIRECTION.md, C10): pass = the best wide variant
closes >= half of the gap C9's LoRA arm left open (LoRA -> CatBoost), head
parameter counts reported. LoRA/CatBoost references load from
results_c9_converged.json (override with --lora-ref / recomputed CatBoost).

  python run_c10.py --smoke
  python run_c10.py --limit 20000     # the CPU verdict run (matches C9's slice)
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent


@torch.no_grad()
def embed_all_tokens(encoder, full, n, device, batch_size=256):
    """All transformer token outputs per row -> (N, T+1, D); [:,0] is f(x)."""
    from run_gpu import _index_batch, _to_device
    out = []
    for s in range(0, n, batch_size):
        idx = torch.arange(s, min(s + batch_size, n))
        _, toks = encoder.forward(_to_device(_index_batch(full, idx), device))
        out.append(toks.cpu())
    return torch.cat(out)


def main():
    from run_c9 import raw_feature_matrix
    from run_gpu import load_data_and_schema
    from run_seq import catboost_pr, frozen_embeddings, probe_pr

    ap = argparse.ArgumentParser(description="C10 - wide read-outs vs the "
                                             "single-token contract (C2 risk task)")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--data", default=str(ROOT / "data" / "pacs008_synth.parquet"))
    ap.add_argument("--schema", default=str(ROOT / "data" / "column_schema.json"))
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--lora-ref", type=float, default=None,
                    help="C9 LoRA PR-AUC reference (default: results_c9_converged.json)")
    ap.add_argument("--out", default=str(ROOT / "results_c10.json"))
    args = ap.parse_args()

    torch.manual_seed(0); np.random.seed(0)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    df, schema = load_data_and_schema(args)
    df = df.reset_index(drop=True)
    y = (df[schema["label_column"]].astype(str) == "High").to_numpy().astype(int)
    print(f"device={device}  rows {len(df):,}  positive High {y.mean():.3%}")

    from sklearn.model_selection import train_test_split
    idx = np.arange(len(df))
    tr, ev = train_test_split(idx, test_size=0.2, stratify=y, random_state=7)  # = C9

    encoder, vocabs, e_pay, enc_cfg = frozen_embeddings(df, schema, args.smoke, device)
    full = vocabs.encode(df)
    toks = embed_all_tokens(encoder, full, len(df), device)          # (N, T+1, D)
    N, T1, D = toks.shape
    print(f"token outputs {toks.shape}: 1 row token + {T1 - 1} column tokens")
    assert np.allclose(toks[:, 0].numpy(), e_pay, atol=1e-4)         # [:,0] IS f(x)

    variants = {
        "single": e_pay,                                             # the contract
        "mean": toks[:, 1:].mean(1).numpy(),                         # wide, order-free
        "concat": toks.reshape(N, -1).numpy().astype(np.float32),    # full bandwidth
    }
    res = {"mode": "smoke" if args.smoke else "full", "device": device,
           "n_rows": int(N), "prevalence": float(y.mean()),
           "n_tokens": int(T1), "variants": {}}
    for name, X in variants.items():
        pr = probe_pr(X[tr], y[tr], X[ev], y[ev])
        res["variants"][name] = {"pr_auc": pr, "head_params": int(X.shape[1] + 1)}
        print(f"  {name:7} probe PR-AUC {pr:.3f}  ({X.shape[1] + 1:,} head params)")

    cb = catboost_pr(raw_feature_matrix(df, schema)[tr], y[tr],
                     raw_feature_matrix(df, schema)[ev], y[ev],
                     iters=100 if args.smoke else 300)
    lora_ref = args.lora_ref
    if lora_ref is None and (ROOT / "results_c9_converged.json").exists():
        lora_ref = json.loads((ROOT / "results_c9_converged.json").read_text())["lora_pr_auc"]
    res["catboost_pr_auc"] = cb
    res["lora_ref"] = lora_ref

    best_name = max(res["variants"], key=lambda k: res["variants"][k]["pr_auc"])
    best = res["variants"][best_name]["pr_auc"]
    print(f"  catboost (adversary)     PR-AUC {cb:.3f}")
    if lora_ref is not None and cb - lora_ref > 1e-6:
        closure = (best - lora_ref) / (cb - lora_ref)
        res["best_variant"] = best_name
        res["closure_of_lora_gap"] = float(closure)
        res["C10_pass"] = bool(closure >= 0.5)
        print(f"\nC10: best wide read-out = {best_name} {best:.3f} | LoRA ref "
              f"{lora_ref:.3f} | CatBoost {cb:.3f}")
        print(f"  -> closes {closure:.1%} of the gap LoRA left open | "
              f"C10 {'PASS' if res['C10_pass'] else 'FAIL'} (threshold >= 50%)")
    else:
        res["C10_pass"] = None
        print("\nC10: no LoRA reference (run run_c9.py first or pass --lora-ref) - "
              "variants reported without a verdict")
    Path(args.out).write_text(json.dumps(res, indent=2))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
