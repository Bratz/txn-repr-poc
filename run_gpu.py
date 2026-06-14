"""
End-to-end GPU run: resolve BOTH falsifiable claims in one pass.

  generate/load data
    -> C1: pretrain partitioned + classical encoders (§3.4), report recon-accuracy
            gap + high-card param ratio  (configs/default.yaml C1 thresholds)
    -> freeze the partitioned encoder f
    -> C2: instruction-tune the decoder's {Φ, ψ, φ} (§4) on templated risk prompts
            with f and the LLM FROZEN; eval vs the CatBoost baseline (+ optional
            full fine-tune) on the SAME split  (C2 thresholds)
    -> write results.json

Two modes, identical control flow:
  --smoke : MockLLM + tiny config + row cap → validates the whole chain on CPU.
  (default): EncoderConfig() 25M / Phi-1.5 frozen LLM → the real run (GPU).

# NB (φ on real HF): the per-layer prefix is injected via past_key_values. Some
# transformers versions ignore past_key_values when use_cache=False during a
# training forward, in which case φ would not receive gradient. This run prints a
# hard grad-check after the first decoder step; if φ.grad is absent/zero on a real
# LLM, switch φ to peft.PrefixTuning (true per-layer prefix) — the Adapter Φ, task
# embedding ψ, and sentinel are unaffected. The MockLLM path validates the logic.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent


# --------------------------------------------------------------------------- #
# Data + config
# --------------------------------------------------------------------------- #

def load_data_and_schema(args):
    import pandas as pd

    schema_path = Path(args.schema)
    if not schema_path.exists():
        schema_path = ROOT / "data" / "column_schema.example.json"
    schema = json.loads(schema_path.read_text())

    path = Path(args.data)
    df = None
    if path.exists():
        try:
            df = pd.read_parquet(path) if path.suffix == ".parquet" else pd.read_csv(path)
        except Exception as e:
            print(f"(could not read {path.name}: {e}; using reference sample)")
    if df is None:
        df = pd.read_csv(ROOT / "data" / "pacs008_sample_500.csv")
        print("NOTE: using committed reference sample (run synth_pacs008.py for full data)")
    if args.limit and args.limit < len(df):
        print(f"NOTE: capping rows {len(df):,} -> {args.limit:,} (--limit)")
        df = df.head(args.limit)
    return df, schema


def split(df, eval_rows, label_col, seed=7):
    """Stratified train/eval split so the rare positive appears in eval."""
    from sklearn.model_selection import train_test_split
    n_eval = min(eval_rows, max(1, len(df) // 5))
    labels = df[label_col].astype(str).to_numpy()
    strat = labels if len(set(labels)) > 1 and n_eval >= len(set(labels)) else None
    tr, te = train_test_split(np.arange(len(df)), test_size=n_eval,
                              random_state=seed, stratify=strat)
    return df.iloc[tr], df.iloc[te]


def _sanitize(obj):
    """Replace NaN/Inf with None so results.json is valid JSON."""
    if isinstance(obj, dict):
        return {k: _sanitize(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_sanitize(v) for v in obj]
    if isinstance(obj, float) and not np.isfinite(obj):
        return None
    return obj


# --------------------------------------------------------------------------- #
# C1 — encoder param-efficiency + reconstruction accuracy
# --------------------------------------------------------------------------- #

def run_c1(df, schema, enc_cfg, train_df, eval_df, thresholds, device, log):
    from encoder.tabular_encoder import build_pretraining_stack, pretrain

    out, frozen = {}, {}
    for variant in ("partitioned", "classical"):
        log(f"[C1/{variant}] build + pretrain encoder ...")
        enc, asm, vocabs = build_pretraining_stack(df, schema, enc_cfg,
                                                   high_card_embedder=variant)
        enc.to(device)
        pretrain(enc, _to_device(vocabs.encode(train_df), device), enc_cfg,
                 batch_size=enc_cfg_batch(enc_cfg), log=log)
        enc.freeze()
        acc = enc.reconstruction_accuracy(_to_device(vocabs.encode(eval_df), device))
        hc = sum(e.num_embedding_parameters() for e in asm.hc_emb.values())
        out[variant] = {"recon_accuracy": acc, "hc_params": hc}
        if variant == "partitioned":
            frozen = {"encoder": enc, "assembler": asm, "vocabs": vocabs}

    hc_cols = schema["buckets"]["high_card_categorical"]
    gap = sum(out["classical"]["recon_accuracy"][c]["top1"]
              - out["partitioned"]["recon_accuracy"][c]["top1"] for c in hc_cols) / len(hc_cols)
    ratio = out["partitioned"]["hc_params"] / out["classical"]["hc_params"]
    verdict = {
        "high_card_param_ratio": ratio,
        "mean_top1_recon_gap_pp": gap * 100,
        "param_efficiency_pass": ratio <= thresholds["param_ratio"],
        "accuracy_giveup_pass": gap * 100 <= thresholds["recon_accuracy_gap"],
    }
    return {"per_variant": out, "verdict": verdict}, frozen


# --------------------------------------------------------------------------- #
# C2 — decoder instruction tuning vs baselines
# --------------------------------------------------------------------------- #

def build_llm(args, device):
    """Return (llm, instruction_ids_1d, answer_token_ids, hidden)."""
    if args.smoke:
        from decoder.multimodal_decoder import MockLLM
        llm = MockLLM(vocab_size=64, hidden=args.smoke_hidden, num_layers=2, num_heads=4)
        instr = torch.randint(0, 64, (4,))
        answers = [0, 1, 2]                         # one token per risk class
        return llm.to(device), instr.to(device), answers
    from decoder.multimodal_decoder import HFCausalLM
    llm = HFCausalLM(args.llm).to(device)
    tok = llm.tokenizer
    # single-token, distinct answers: letters mapped to the 3 risk classes.
    prompt = ("Classify the transaction's risk. Answer with a single letter: "
              "A for Low, B for Medium, C for High. Answer:")
    instr = torch.tensor(tok(prompt, add_special_tokens=False)["input_ids"], device=device)
    answers = [tok(f" {ltr}", add_special_tokens=False)["input_ids"][0] for ltr in "ABC"]
    if len(set(answers)) != 3:
        raise RuntimeError(f"answer tokens not distinct: {answers}; pick other letters")
    return llm, instr, answers


def _train_decoder(dec, train_batch, tgt_train, instr, n, epochs, batch_size,
                   device, log, label, grad_check=False):
    """Instruction-tuning loop shared by the adapter and the full-tune comparator."""
    opt = torch.optim.Adam([p for p in dec.parameters() if p.requires_grad], lr=1e-4)
    log(f"[C2] {label}: train {dec.trainable_parameters():,} params, {epochs} epoch(s) ...")
    dec.train()
    checked = not grad_check
    for ep in range(epochs):
        perm = torch.randperm(n)
        tot, nb = 0.0, 0
        for s in range(0, n, batch_size):
            sl = perm[s:s + batch_size]
            b = _index_batch(train_batch, sl)
            B = len(sl)
            loss = dec(b, torch.zeros(B, dtype=torch.long, device=device),
                       instr.unsqueeze(0).expand(B, -1), tgt_train[sl])
            opt.zero_grad(); loss.backward()
            if not checked:                          # φ grad-check (see module header)
                g = dec.phi_param().grad
                if g is None or float(g.abs().sum()) == 0.0:
                    log(f"WARNING: phi ({dec.phi_mode}) received no gradient. Use "
                        "--phi-mode prompt (robust soft prompt) or peft.PrefixTuning.")
                else:
                    log(f"  phi grad-check OK (mode={dec.phi_mode}, "
                        f"|grad|={float(g.abs().sum()):.3e})")
                checked = True
            opt.step(); tot += loss.item(); nb += 1
        log(f"  {label} epoch {ep+1}/{epochs}  loss {tot/nb:.4f}")
    return dec


def run_c2(schema, frozen, llm_bundle, train_df, eval_df, dec_cfg, thresholds,
           device, decoder_epochs, batch_size, log,
           full_tune_llm=None, full_tune_epochs=1):
    from dataclasses import replace

    from decoder.multimodal_decoder import MultimodalDecoder
    from eval.baselines import catboost_fit_predict
    from eval.metrics import c2_table

    enc, vocabs = frozen["encoder"], frozen["vocabs"]
    llm, instr, answer_tokens = llm_bundle
    label_values = schema["label_values"]
    label_col = schema["label_column"]

    def targets(frame):
        idx = frame[label_col].map({v: i for i, v in enumerate(label_values)}).to_numpy()
        return torch.tensor([[answer_tokens[i]] for i in idx], device=device)

    train_batch = _to_device(vocabs.encode(train_df), device)
    n = len(train_df)
    tgt_train = targets(train_df)
    y_eval = eval_df[label_col].astype(str).to_numpy()

    # --- adapter: frozen encoder + frozen LLM + trainable {Φ, ψ, φ} ---
    dec = MultimodalDecoder(enc, llm, dec_cfg).to(device)
    dec.assert_frozen()
    _train_decoder(dec, train_batch, tgt_train, instr, n, decoder_epochs,
                   batch_size, device, log, "adapter", grad_check=True)
    adapter_proba = _predict_chunked(dec, vocabs, eval_df, instr, answer_tokens,
                                     device, batch_size)
    results = {"adapter": (y_eval, adapter_proba)}
    trainable = {"adapter": dec.trainable_parameters()}

    # --- optional C2 full fine-tune comparator: frozen encoder + UNFROZEN LLM ---
    if full_tune_llm is not None:
        ft = MultimodalDecoder(enc, full_tune_llm, replace(dec_cfg, train_llm=True)).to(device)
        ft.assert_frozen()                           # encoder frozen; LLM trainable
        log(f"[C2] full fine-tune comparator: LLM UNFROZEN, "
            f"{ft.trainable_parameters():,} trainable params (heavy) ...")
        _train_decoder(ft, train_batch, tgt_train, instr, n, full_tune_epochs,
                       batch_size, device, log, "full_tune")
        results["full_tune"] = (y_eval, _predict_chunked(
            ft, vocabs, eval_df, instr, answer_tokens, device, batch_size))
        trainable["full_tune"] = ft.trainable_parameters()
    else:
        # analytic reference when the full-tune run is skipped
        trainable["full_tune"] = _LLM_FULL.get(getattr(llm, "name", "mock"), 1_300_000_000)

    # --- CatBoost baseline on the SAME split ---
    log("[C2] CatBoost baseline on the same split ...")
    cb_iters = 50 if (len(train_df) < 2000) else 300
    y_cb, cb_proba, _, _ = catboost_fit_predict(train_df, eval_df, schema,
                                                iterations=cb_iters, log=log)
    results["catboost"] = (y_cb, cb_proba)

    tbl = c2_table(results, label_values, "High", thresholds["fixed_fpr"],
                   trainable_params=trainable, thresholds=thresholds)
    tbl["trainable_params"] = trainable
    return tbl, dec, instr, answer_tokens


_LLM_FULL = {"phi-1_5": 1_300_000_000, "microsoft/phi-1_5": 1_300_000_000}


# --------------------------------------------------------------------------- #
# small tensor helpers
# --------------------------------------------------------------------------- #

def _to_device(batch, device):
    return {
        "high_card": {c: t.to(device) for c, t in batch["high_card"].items()},
        "core": {c: t.to(device) for c, t in batch["core"].items()},
        "amount": batch["amount"], "ccy": batch["ccy"],
    }


def _index_batch(batch, idx):
    idx_cpu = idx.cpu().numpy()
    return {
        "high_card": {c: t[idx] for c, t in batch["high_card"].items()},
        "core": {c: t[idx] for c, t in batch["core"].items()},
        "amount": batch["amount"][idx_cpu], "ccy": batch["ccy"][idx_cpu],
    }


@torch.no_grad()
def _predict_chunked(dec, vocabs, eval_df, instr, answer_tokens, device, batch_size):
    full = _to_device(vocabs.encode(eval_df), device)
    n = len(eval_df)
    parts = []
    for s in range(0, n, batch_size):
        idx = torch.arange(s, min(s + batch_size, n))
        b = _index_batch(full, idx)
        B = len(idx)
        p = dec.predict_proba(b, torch.zeros(B, dtype=torch.long, device=device),
                              instr.unsqueeze(0).expand(B, -1), answer_tokens)
        parts.append(p.cpu().numpy())
    return np.concatenate(parts, axis=0)


def enc_cfg_batch(cfg):
    return 64 if cfg.hidden <= 64 else 256


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #

def main():
    import yaml

    ap = argparse.ArgumentParser(description="End-to-end C1 + C2 run")
    ap.add_argument("--smoke", action="store_true", help="MockLLM + tiny config (CPU)")
    ap.add_argument("--data", default=str(ROOT / "data" / "pacs008_synth.parquet"))
    ap.add_argument("--schema", default=str(ROOT / "data" / "column_schema.json"))
    ap.add_argument("--config", default=str(ROOT / "configs" / "default.yaml"))
    ap.add_argument("--llm", default="microsoft/phi-1_5")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--eval-rows", type=int, default=4096)
    ap.add_argument("--decoder-epochs", type=int, default=1)   # paper §5.2: 1 epoch
    ap.add_argument("--full-tune", action="store_true",
                    help="also train the C2 full fine-tune comparator (UNFROZEN LLM; "
                         "heavy) to resolve the pr_auc_gap_vs_fulltune threshold")
    ap.add_argument("--full-tune-epochs", type=int, default=1)
    ap.add_argument("--phi-mode", choices=["prompt", "prefix"], default="prompt",
                    help="prompt = robust soft prompt (default); prefix = per-layer "
                         "(more faithful; needs peft on real HF)")
    ap.add_argument("--smoke-hidden", type=int, default=32)
    ap.add_argument("--out", default=str(ROOT / "results.json"))
    ap.add_argument("--save-dir", default=None,
                    help="persist the trained model here for predict.py")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device={device}  mode={'smoke' if args.smoke else 'full'}")

    cfg = yaml.safe_load(Path(args.config).read_text())
    claims = cfg["claims"]
    c1_thr = {"param_ratio": claims["C1_partitioning_param_efficiency"]["metrics"]["param_ratio"]["threshold"],
              "recon_accuracy_gap": claims["C1_partitioning_param_efficiency"]["metrics"]["recon_accuracy_gap"]["threshold"]}
    c2m = claims["C2_adapter_vs_baselines"]["metrics"]
    c2_thr = {"pr_auc_gain_vs_catboost": c2m["pr_auc_gain_vs_catboost"]["threshold"],
              "trainable_param_ratio": c2m["trainable_param_ratio"]["threshold"],
              "pr_auc_gap_vs_fulltune": c2m["pr_auc_gap_vs_fulltune"]["threshold"],
              "fixed_fpr": cfg["eval"]["fixed_fpr"]}

    from encoder.tabular_encoder import EncoderConfig
    from decoder.multimodal_decoder import DecoderConfig

    if args.smoke:
        enc_cfg = EncoderConfig(hidden=args.smoke_hidden, layers=2, heads=2,
                                ff_mult=2, dropout=0.0, epochs=1)
        dec_cfg = DecoderConfig(n_tasks=1, adapter_heads=4, prefix_len=4,
                                phi_mode=args.phi_mode)
    else:
        enc_cfg = EncoderConfig()                  # pinned 25M / 3 epochs
        dec_cfg = DecoderConfig(n_tasks=1, phi_mode=args.phi_mode)

    df, schema = load_data_and_schema(args)
    train_df, eval_df = split(df, args.eval_rows, schema["label_column"])
    torch.manual_seed(0)

    c1, frozen = run_c1(df, schema, enc_cfg, train_df, eval_df, c1_thr, device, print)
    llm_bundle = build_llm(args, device)
    if not args.smoke:
        llm_bundle[0].name = args.llm
    # fresh LLM instance for the full-tune comparator (its weights get trained,
    # so it must not share with the adapter's frozen LLM). Same prompt/tokens.
    full_tune_llm = build_llm(args, device)[0] if args.full_tune else None
    if full_tune_llm is not None and not args.smoke:
        full_tune_llm.name = args.llm

    c2, dec, instr, answer_tokens = run_c2(
        schema, frozen, llm_bundle, train_df, eval_df, dec_cfg, c2_thr,
        device, args.decoder_epochs, enc_cfg_batch(enc_cfg), print,
        full_tune_llm=full_tune_llm, full_tune_epochs=args.full_tune_epochs)

    if args.save_dir:
        from predict import save_model
        save_model(
            args.save_dir, enc_cfg=enc_cfg, dec_cfg=dec_cfg, vocabs=frozen["vocabs"],
            quantizer=frozen["assembler"].amt_emb.quantizer, encoder=frozen["encoder"],
            decoder=dec, llm_name=("mock" if args.smoke else args.llm),
            label_values=schema["label_values"], instruction_ids=instr,
            answer_token_ids=answer_tokens, schema=schema)
        print(f"saved model -> {args.save_dir}")

    results = {"mode": "smoke" if args.smoke else "full", "device": device,
               "n_rows": int(len(df)), "C1": c1["verdict"], "C2": c2["verdict"],
               "C2_trainable_params": c2["trainable_params"],
               "C2_per_model": {k: {kk: vv for kk, vv in v.items() if isinstance(vv, (int, float, str))}
                                for k, v in c2["per_model"].items()}}
    results = _sanitize(results)
    Path(args.out).write_text(json.dumps(results, indent=2, default=float, allow_nan=False))
    print("\n=== RESULTS ===")
    print(json.dumps(results, indent=2, default=float, allow_nan=False))
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
