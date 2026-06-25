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

_LETTERS = "ABCDEFGH"
_LLM_FULL = {"phi-1_5": 1_300_000_000, "microsoft/phi-1_5": 1_300_000_000}


def build_llm(args, device):
    """Return the (frozen) LLM only — task prompts are built separately."""
    if args.smoke:
        from decoder.multimodal_decoder import MockLLM
        llm = MockLLM(vocab_size=64, hidden=args.smoke_hidden, num_layers=2, num_heads=4)
        return llm.to(device)
    from decoder.multimodal_decoder import HFCausalLM
    return HFCausalLM(args.llm).to(device)


def build_task_specs(schema, llm, smoke, device):
    """Augment each schema task with task_id, instruction ids, and answer tokens.

    Tasks are read from the schema manifest (§0.4 — never hard-coded). Each label
    maps to one distinct single LLM-vocab token (a letter) so predict_proba scores
    a well-formed label distribution.
    """
    specs = []
    for i, t in enumerate(schema["tasks"]):
        lv = t["label_values"]
        if len(lv) > len(_LETTERS):
            raise ValueError(f"task {t['name']} has too many labels for letter answers")
        if smoke:
            instr = torch.randint(0, llm.vocab_size, (4,), device=device)
            answers = list(range(len(lv)))           # distinct token ids 0..L-1
        else:
            tok = llm.tokenizer
            letters = _LETTERS[:len(lv)]
            opts = ", ".join(f"{ltr} for {v}" for ltr, v in zip(letters, lv))
            unit = "transactions" if t.get("records") == "multi" else "transaction"
            prompt = (f"Task: {t['name']}. Classify the {unit}. "
                      f"Answer with a single letter: {opts}. Answer:")
            instr = torch.tensor(tok(prompt, add_special_tokens=False)["input_ids"],
                                 device=device)
            answers = [tok(f" {ltr}", add_special_tokens=False)["input_ids"][0]
                       for ltr in letters]
            if len(set(answers)) != len(lv):
                raise RuntimeError(f"answer tokens not distinct for {t['name']}: {answers}")
        specs.append({**t, "task_id": i, "instr": instr, "answers": answers,
                      "label_index": {v: j for j, v in enumerate(lv)}})
    return specs


# -- per-task training-example construction --------------------------------- #

def _single_examples(tdf, task):
    """Single-record task: every row is an example. Returns (positions, targets)."""
    y = tdf[task["label_column"]].astype(str).map(task["label_index"]).to_numpy()
    tgt = np.array([task["answers"][int(j)] for j in y], dtype=np.int64)
    return np.arange(len(tdf)), tgt


def _recurrence_groups(tdf, task, R):
    """Multi-record task: each (debtor,creditor) group with ≥R txns → R records
    (first R by settlement date). Returns (groups, label_strings) where each group
    is an array of R row-positions into `tdf`."""
    gcol, lcol = task["group_column"], task["label_column"]
    groups, labels = [], []
    for _, sub in tdf.groupby(gcol):
        sub = sub.sort_values("IntrBkSttlmDt")
        pos = sub.index.to_numpy()
        if len(pos) < R:
            continue
        groups.append(pos[:R])
        labels.append(str(sub[lcol].iloc[0]))        # constant within a group
    return groups, labels


def _multi_records(full, groups, device):
    """Assemble R slot-batches from a list of same-length groups (Eq. 5 records)."""
    R = len(groups[0])
    return [_index_batch(full, torch.tensor([g[j] for g in groups], device=device))
            for j in range(R)]


def train_multitask(dec, specs, tdf, full, R, epochs, batch_size, device, log,
                    label="adapter", grad_check=False):
    """Joint instruction tuning across all tasks; batches interleaved per epoch."""
    opt = torch.optim.Adam([p for p in dec.parameters() if p.requires_grad], lr=1e-4)
    log(f"[C2] {label}: train {dec.trainable_parameters():,} params over "
        f"{len(specs)} tasks, {epochs} epoch(s) ...")

    prepared = []
    for task in specs:
        if task.get("records") == "multi":
            groups, labs = _recurrence_groups(tdf, task, R)
            tgt = np.array([task["answers"][task["label_index"][s]] for s in labs],
                           dtype=np.int64)
            prepared.append(("multi", task, groups, tgt))
            log(f"  task {task['name']}: {len(groups):,} multi-record examples (R={R})")
        else:
            pos, tgt = _single_examples(tdf, task)
            prepared.append(("single", task, pos, tgt))
            log(f"  task {task['name']}: {len(pos):,} single-record examples")

    dec.train()
    checked = not grad_check
    for ep in range(epochs):
        jobs = []                                    # (kind, task, data, tgt, idx-slice)
        for kind, task, data, tgt in prepared:
            n = len(data)
            perm = torch.randperm(n)
            for s in range(0, n, batch_size):
                jobs.append((kind, task, data, tgt, perm[s:s + batch_size]))
        for j in torch.randperm(len(jobs)).tolist():  # interleave tasks
            kind, task, data, tgt, sl = jobs[j]
            sl_np = sl.numpy()
            B = len(sl_np)
            tids = torch.full((B,), task["task_id"], dtype=torch.long, device=device)
            instr = task["instr"].unsqueeze(0).expand(B, -1)
            tgt_b = torch.tensor([[tgt[k]] for k in sl_np], device=device)
            if kind == "single":
                records = _index_batch(full, sl.to(device))
            else:
                records = _multi_records(full, [data[k] for k in sl_np], device)
            loss = dec(records, tids, instr, tgt_b)
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
            opt.step()
    return dec


@torch.no_grad()
def predict_task(dec, task, edf, full, R, device, batch_size):
    """Adapter probabilities + true labels for one task on the eval frame."""
    if task.get("records") == "multi":
        groups, labels = _recurrence_groups(edf, task, R)
        y_true = np.array(labels)
        parts = []
        for s in range(0, len(groups), batch_size):
            gsel = groups[s:s + batch_size]
            B = len(gsel)
            tids = torch.full((B,), task["task_id"], dtype=torch.long, device=device)
            p = dec.predict_proba(_multi_records(full, gsel, device), tids,
                                  task["instr"].unsqueeze(0).expand(B, -1), task["answers"])
            parts.append(p.cpu().numpy())
        return y_true, (np.concatenate(parts, axis=0) if parts
                        else np.zeros((0, len(task["label_values"]))))
    y_true = edf[task["label_column"]].astype(str).to_numpy()
    n = len(edf)
    parts = []
    for s in range(0, n, batch_size):
        idx = torch.arange(s, min(s + batch_size, n))
        B = len(idx)
        tids = torch.full((B,), task["task_id"], dtype=torch.long, device=device)
        p = dec.predict_proba(_index_batch(full, idx.to(device)), tids,
                              task["instr"].unsqueeze(0).expand(B, -1), task["answers"])
        parts.append(p.cpu().numpy())
    return y_true, np.concatenate(parts, axis=0)


def run_c2(schema, frozen, llm, specs, train_df, eval_df, dec_cfg, thresholds,
           device, decoder_epochs, batch_size, R, log,
           full_tune_llm=None, full_tune_epochs=1):
    from dataclasses import replace

    from decoder.multimodal_decoder import MultimodalDecoder
    from eval.baselines import catboost_fit_predict
    from eval.metrics import c2_table, evaluate_task

    enc, vocabs = frozen["encoder"], frozen["vocabs"]
    tdf = train_df.reset_index(drop=True)
    edf = eval_df.reset_index(drop=True)
    full_tr = _to_device(vocabs.encode(tdf), device)
    full_ev = _to_device(vocabs.encode(edf), device)

    # --- adapter: frozen encoder + frozen LLM + trainable {Φ, ψ, φ}, all tasks ---
    dec = MultimodalDecoder(enc, llm, dec_cfg).to(device)
    dec.assert_frozen()
    train_multitask(dec, specs, tdf, full_tr, R, decoder_epochs, batch_size,
                    device, log, "adapter", grad_check=True)
    trainable = {"adapter": dec.trainable_parameters()}

    # --- per-task eval: adapter (all tasks) + CatBoost (single-record tasks) ---
    cb_iters = 50 if (len(tdf) < 2000) else 300
    per_task = {}
    risk_results = {}                                # for the headline C2 (risk) table
    def _safe_eval(task, y_true, proba):
        # Guard tiny/degenerate eval folds (e.g. the smoke split) so the chain
        # still completes; the full run has ample examples per task.
        if len(y_true) < 2 or len(set(y_true.tolist())) < 2:
            return {"metric": task.get("metric"), "note": "insufficient eval examples",
                    "n": int(len(y_true))}
        return evaluate_task(task, y_true, proba, thresholds["fixed_fpr"])

    for task in specs:
        y_true, ad_proba = predict_task(dec, task, edf, full_ev, R, device, batch_size)
        entry = {"adapter": _safe_eval(task, y_true, ad_proba)}
        # PAPER: §5.3. "The Recurrence task, which involves passing multiple
        # transactions, is not suitable for non-sequential classifiers and is
        # therefore excluded from the CatBoost evaluation for a fair comparison."
        # So CatBoost scores only the single-record tasks; recurrence is LLM-only.
        if task.get("records") != "multi":
            tschema = dict(schema, label_column=task["label_column"],
                           label_values=task["label_values"])
            y_cb, cb_proba, _, _ = catboost_fit_predict(tdf, edf, tschema,
                                                        iterations=cb_iters, log=log)
            entry["catboost"] = _safe_eval(task, y_cb, cb_proba)
            if task["name"] == "risk":
                risk_results = {"adapter": (y_true, ad_proba), "catboost": (y_cb, cb_proba)}
        per_task[task["name"]] = entry
        log(f"  [{task['name']}] adapter: {_fmt_metric(entry['adapter'])}")

    # --- optional C2 full fine-tune comparator on the RISK task (headline claim) ---
    risk_spec = next(t for t in specs if t["name"] == "risk")
    if full_tune_llm is not None:
        ft = MultimodalDecoder(enc, full_tune_llm, replace(dec_cfg, train_llm=True)).to(device)
        ft.assert_frozen()                           # encoder frozen; LLM trainable
        log(f"[C2] full fine-tune comparator (risk): LLM UNFROZEN, "
            f"{ft.trainable_parameters():,} trainable params (heavy) ...")
        train_multitask(ft, [risk_spec], tdf, full_tr, R, full_tune_epochs,
                        batch_size, device, log, "full_tune")
        y_ft, ft_proba = predict_task(ft, risk_spec, edf, full_ev, R, device, batch_size)
        risk_results["full_tune"] = (y_ft, ft_proba)
        trainable["full_tune"] = ft.trainable_parameters()
    else:
        trainable["full_tune"] = _LLM_FULL.get(getattr(llm, "name", "mock"), 1_300_000_000)

    tbl = c2_table(risk_results, risk_spec["label_values"], "High",
                   thresholds["fixed_fpr"], trainable_params=trainable,
                   thresholds=thresholds)
    tbl["trainable_params"] = trainable
    tbl["per_task"] = per_task
    return tbl, dec, specs


def _fmt_metric(m):
    if "note" in m:
        return m["note"]
    if m.get("metric") == "multiclass":
        return f"acc={m['accuracy']:.3f} macroF1={m['macro_f1']:.3f}"
    return f"PR-AUC={m['pr_auc']:.3f}"


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
    ap.add_argument("--recur-records", type=int, default=3,
                    help="R: records per multi-record (recurrence) example, Eq. 5")
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

    df, schema = load_data_and_schema(args)
    n_tasks = len(schema["tasks"])
    R = args.recur_records
    if args.smoke:
        enc_cfg = EncoderConfig(hidden=args.smoke_hidden, layers=2, heads=2,
                                ff_mult=2, dropout=0.0, epochs=1)
        dec_cfg = DecoderConfig(n_tasks=n_tasks, max_records=R, adapter_heads=4,
                                prefix_len=4, phi_mode=args.phi_mode)
    else:
        enc_cfg = EncoderConfig()                  # pinned 25M / 3 epochs
        dec_cfg = DecoderConfig(n_tasks=n_tasks, max_records=R, phi_mode=args.phi_mode)

    train_df, eval_df = split(df, args.eval_rows, schema["label_column"])
    torch.manual_seed(0)

    c1, frozen = run_c1(df, schema, enc_cfg, train_df, eval_df, c1_thr, device, print)

    llm = build_llm(args, device)
    if not args.smoke:
        llm.name = args.llm
    specs = build_task_specs(schema, llm, args.smoke, device)
    print(f"[C2] tasks: {[t['name'] for t in specs]}  (n_tasks={n_tasks}, R={R})")
    # fresh LLM instance for the full-tune comparator (its weights get trained,
    # so it must not share with the adapter's frozen LLM).
    full_tune_llm = build_llm(args, device) if args.full_tune else None
    if full_tune_llm is not None and not args.smoke:
        full_tune_llm.name = args.llm

    c2, dec, specs = run_c2(
        schema, frozen, llm, specs, train_df, eval_df, dec_cfg, c2_thr,
        device, args.decoder_epochs, enc_cfg_batch(enc_cfg), R, print,
        full_tune_llm=full_tune_llm, full_tune_epochs=args.full_tune_epochs)

    if args.save_dir:
        # Persist the full §5 task suite; predict.py scores any task by name
        # (risk is the default). Legacy fields below mirror the risk task.
        from predict import save_model
        risk = next(t for t in specs if t["name"] == "risk")
        save_model(
            args.save_dir, enc_cfg=enc_cfg, dec_cfg=dec_cfg, vocabs=frozen["vocabs"],
            quantizer=frozen["assembler"].amt_emb.quantizer, encoder=frozen["encoder"],
            decoder=dec, llm_name=("mock" if args.smoke else args.llm),
            label_values=risk["label_values"], instruction_ids=risk["instr"],
            answer_token_ids=risk["answers"], schema=schema, tasks=specs)
        print(f"saved model ({len(specs)} tasks) -> {args.save_dir}")

    results = {"mode": "smoke" if args.smoke else "full", "device": device,
               "n_rows": int(len(df)), "tasks": [t["name"] for t in specs],
               "C1": c1["verdict"], "C2_risk": c2["verdict"],
               "C2_trainable_params": c2["trainable_params"],
               "C2_per_task": c2["per_task"]}
    results = _sanitize(results)
    Path(args.out).write_text(json.dumps(results, indent=2, default=float, allow_nan=False))
    print("\n=== RESULTS ===")
    print(json.dumps(results, indent=2, default=float, allow_nan=False))
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
