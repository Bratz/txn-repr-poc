"""
v2 / sequence run - Layer 3b end to end, with the C3/C4/C5 claims measured
(docs/V2_DIRECTION.md). Beyond arXiv:2410.07851; never touches v1's run_gpu.py.

  Stage A : build + freeze the v1 per-transaction encoder  -> e_t per row
  Stage B : assemble per-actor ordered histories (held-out BY ACTOR),
            pretrain the Layer-3b history encoder on frozen e_t (masking + CoLES)
  Claims  : on the entity regime-change label, all scored on UNSEEN accounts:
            C3 (temporal lift)  sequence h_USR  vs  order-blind pooled mean(e_t)
            C4 (held-out gen.)  sequence h_USR  vs  CatBoost on per-account aggregates
            C5 (LLM necessity)  Option A (linear probe on h_USR)
                                vs  Option B (h_USR -> frozen LLM + adapters)

Run on the §7 behavioural data (data/synth_sequences.py), which puts the signal in the
TIMING with aggregates matched across classes - so the pooled and CatBoost baselines are
honestly weak. On v1 data (no regime label) it falls back to an account-risk demo.

  --smoke : tiny config + MockLLM + CPU + row cap, validates the whole v2 chain.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from run_gpu import _index_batch, _to_device, load_data_and_schema

ROOT = Path(__file__).resolve().parent


def embed_all_rows(encoder, full, n, device, batch_size=512):
    out = []
    for s in range(0, n, batch_size):
        idx = torch.arange(s, min(s + batch_size, n))
        out.append(encoder.encode(_to_device(_index_batch(full, idx), device)).cpu())
    return torch.cat(out, dim=0)


def encode_histories(hist, e_all, seqs, device, static_all=None, batch_size=128):
    """Frozen entity representations h_USR for a list of sequences -> (N, D) torch."""
    from data.sequence_assembly import collate
    reps = []
    for s in range(0, len(seqs), batch_size):
        b = {k: v.to(device) for k, v in collate(seqs[s:s + batch_size]).items()}
        static = static_all[b["pos"][:, 0]] if static_all is not None else None
        reps.append(hist.encode(e_all[b["pos"]], b, static))
    return torch.cat(reps, dim=0) if reps else torch.zeros(0, hist.D, device=device)


def pooled_features(e_all, seqs) -> np.ndarray:
    """C3 baseline: order-blind mean of the account's per-transaction embeddings."""
    return np.stack([e_all[s["pos"]].mean(0).cpu().numpy() for s in seqs], axis=0)


def agg_features(df, seqs) -> np.ndarray:
    """C4 baseline features: per-account summary stats (incl. inter-arrival stats)."""
    amt = df["IntrBkSttlmAmt"].to_numpy(dtype=float)
    cdtr = df["CdtrAcct_Id"].to_numpy()
    feats = []
    for s in seqs:
        a = amt[s["pos"]]
        iat = s["dt"][1:] if len(s["dt"]) > 1 else np.array([0.0])
        feats.append([
            len(s["pos"]), a.mean(), a.std(), float(np.median(a)),
            iat.mean(), iat.std(), iat.min(), iat.max(), float(s["dt"].sum()),
            len(set(cdtr[s["pos"]].tolist())),
        ])
    return np.asarray(feats, dtype=float)


def labels_for(seqs, actor_label: dict) -> np.ndarray:
    return np.array([int(actor_label[s["actor"]]) for s in seqs])


def pr_auc(y, p) -> float:
    from sklearn.metrics import average_precision_score
    return float(average_precision_score(y, p))


def probe_pr(X_tr, y_tr, X_ev, y_ev) -> float:
    from sklearn.linear_model import LogisticRegression
    clf = LogisticRegression(max_iter=1000, class_weight="balanced").fit(X_tr, y_tr)
    return pr_auc(y_ev, clf.predict_proba(X_ev)[:, 1])


def catboost_pr(X_tr, y_tr, X_ev, y_ev, iters=200) -> float:
    from catboost import CatBoostClassifier
    clf = CatBoostClassifier(iterations=iters, depth=6, learning_rate=0.1,
                             loss_function="Logloss", auto_class_weights="Balanced",
                             random_seed=7, verbose=False, allow_writing_files=False)
    clf.fit(X_tr, y_tr)
    return pr_auc(y_ev, clf.predict_proba(X_ev)[:, 1])


def option_b_pr(encoder, llm, h_tr, y_tr, h_ev, y_ev, device, smoke,
                epochs=2, batch_size=64, log=print) -> tuple[float, int]:
    """C5: feed frozen h_USR into the frozen LLM + trainable adapters; eval PR-AUC."""
    from decoder.multimodal_decoder import DecoderConfig, MultimodalDecoder
    if smoke:
        instr = torch.randint(0, llm.vocab_size, (4,), device=device)
        answers = [0, 1]
    else:
        tok = llm.tokenizer
        instr = torch.tensor(tok("Has this account's behaviour shifted? Answer A=no, "
                                 "B=yes. Answer:", add_special_tokens=False)["input_ids"],
                             device=device)
        answers = [tok(f" {c}", add_special_tokens=False)["input_ids"][0] for c in "AB"]
    dec = MultimodalDecoder(encoder, llm,
                            DecoderConfig(n_tasks=1, max_records=1, phi_mode="prompt")).to(device)
    dec.assert_frozen()
    tgt = torch.tensor([[answers[int(y)]] for y in y_tr], device=device)
    opt = torch.optim.Adam([p for p in dec.parameters() if p.requires_grad], lr=1e-4)
    n = len(y_tr)
    dec.train()
    for ep in range(epochs):
        perm = torch.randperm(n)
        for s in range(0, n, batch_size):
            idx = perm[s:s + batch_size]
            B = len(idx)
            loss = dec(h_tr[idx], torch.zeros(B, dtype=torch.long, device=device),
                       instr.unsqueeze(0).expand(B, -1), tgt[idx])
            opt.zero_grad(); loss.backward(); opt.step()
    with torch.no_grad():
        p = dec.predict_proba(h_ev, torch.zeros(len(y_ev), dtype=torch.long, device=device),
                              instr.unsqueeze(0).expand(len(y_ev), -1), answers)[:, 1]
    return pr_auc(y_ev, p.cpu().numpy()), dec.trainable_parameters()


def main():
    from encoder.history_encoder import HistoryConfig, HistoryEncoder
    from encoder.history_encoder import pretrain as hist_pretrain
    from encoder.tabular_encoder import EncoderConfig, build_pretraining_stack
    from encoder.tabular_encoder import pretrain as enc_pretrain
    from data.sequence_assembly import assemble_sequences, split_by_actor

    ap = argparse.ArgumentParser(description="v2 Layer-3b sequence run + C3/C4/C5")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--data", default=str(ROOT / "data" / "pacs008_seq.parquet"))
    ap.add_argument("--schema", default=str(ROOT / "data" / "column_schema_seq.json"))
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--actor", default="DbtrAcct_Id")
    ap.add_argument("--max-len", type=int, default=256)
    ap.add_argument("--no-static", action="store_true",
                    help="ablate the party-store [USR] injection (learned token only)")
    ap.add_argument("--hist-epochs", type=int, default=3)
    ap.add_argument("--out", default=str(ROOT / "results_seq.json"))
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device={device}  mode={'smoke' if args.smoke else 'full'}")
    df, schema = load_data_and_schema(args)
    df = df.reset_index(drop=True)

    # --- Stage A: v1 encoder -> frozen e_t ---------------------------------- #
    enc_cfg = (EncoderConfig(hidden=64, layers=2, heads=2, ff_mult=2, epochs=1)
               if args.smoke else EncoderConfig())
    torch.manual_seed(0)
    encoder, assembler, vocabs = build_pretraining_stack(df, schema, enc_cfg, party_epochs=1)
    encoder.to(device)
    print("[A] pretrain v1 encoder ...")
    enc_pretrain(encoder, _to_device(vocabs.encode(df), device), enc_cfg,
                 batch_size=128 if args.smoke else 256)
    encoder.freeze()
    D = enc_cfg.hidden

    full = vocabs.encode(df)
    e_all = embed_all_rows(encoder, full, len(df), device).to(device)
    quant = assembler.amt_emb.quantizer
    targets_all = {
        "amount_level": torch.as_tensor(quant.transform(full["amount"], full["ccy"]),
                                        dtype=torch.long),
        "Ccy": full["core"]["Ccy"], "SttlmMtd": full["core"]["SttlmMtd"],
    }
    recon_fields = {"amount_level": quant.num_levels, "Ccy": vocabs.core_size("Ccy"),
                    "SttlmMtd": vocabs.core_size("SttlmMtd")}

    static_all = None
    role = next((p for p, sp in assembler.party_roles.items() if sp["key"] == args.actor), None)
    if role is not None and not args.no_static:
        with torch.no_grad():
            static_all = assembler.party_emb[role](
                full["high_card"][args.actor].to(device)).detach()
        print(f"[B] [USR] = party-store profile (role '{role}')")

    # --- Stage B: sequences + history encoder ------------------------------- #
    seqs = assemble_sequences(df, actor_col=args.actor, max_len=args.max_len, min_len=2)
    train_seqs, eval_seqs = split_by_actor(seqs, frac_eval=0.2, seed=0)
    print(f"[B] sequences {len(seqs):,}  train {len(train_seqs):,} / held-out {len(eval_seqs):,}")
    if not train_seqs or not eval_seqs:
        raise SystemExit("not enough multi-event histories; raise --limit or --accounts")
    hcfg = HistoryConfig(hidden=D, layers=2 if args.smoke else 4,
                         heads=2 if args.smoke else 8, ff_mult=2 if args.smoke else 4,
                         epochs=args.hist_epochs)
    hist = HistoryEncoder(recon_fields, hcfg).to(device)
    print(f"[B] history encoder: {hist.num_trainable_parameters():,} trainable params")
    hist_pretrain(hist, e_all, targets_all, train_seqs, hcfg,
                  batch_size=64 if args.smoke else 128, static_all=static_all)
    hist.freeze()

    # --- entity label (regime if present, else account-risk fallback) ------- #
    et = schema.get("entity_task")
    if et and et["label_column"] in df.columns:
        lc, pos = et["label_column"], et["positive_class"]
        actor_label = (df.groupby(args.actor)[lc].first() == pos).astype(int).to_dict()
        task_name = et["name"]
    else:
        actor_label = (df.assign(_h=(df[schema["label_column"]].astype(str) == "High"))
                       .groupby(args.actor)["_h"].max().astype(int).to_dict())
        task_name = "account_ever_high_risk (fallback - not temporal)"
    y_tr, y_ev = labels_for(train_seqs, actor_label), labels_for(eval_seqs, actor_label)

    results = {"mode": "smoke" if args.smoke else "full", "device": device,
               "task": task_name, "n_sequences": len(seqs),
               "held_out": {"n": len(y_ev), "prevalence": float(y_ev.mean()) if len(y_ev) else None}}

    if len(set(y_tr.tolist())) < 2 or len(set(y_ev.tolist())) < 2:
        results["note"] = "degenerate label at this scale; raise --limit / --accounts"
        print(f"[claims] {results['note']}")
    else:
        # representations on the SAME held-out actors
        h_tr = encode_histories(hist, e_all, train_seqs, device, static_all)
        h_ev = encode_histories(hist, e_all, eval_seqs, device, static_all)
        seq_pr = probe_pr(h_tr.cpu().numpy(), y_tr, h_ev.cpu().numpy(), y_ev)        # Option A
        pool_pr = probe_pr(pooled_features(e_all, train_seqs), y_tr,
                           pooled_features(e_all, eval_seqs), y_ev)                  # C3 base
        cb_pr = catboost_pr(agg_features(df, train_seqs), y_tr,
                            agg_features(df, eval_seqs), y_ev,
                            iters=50 if args.smoke else 200)                          # C4 base
        from decoder.multimodal_decoder import MockLLM
        llm = (MockLLM(vocab_size=64, hidden=64, num_layers=2, num_heads=4).to(device)
               if args.smoke else None)
        if llm is None:
            from decoder.multimodal_decoder import HFCausalLM
            llm = HFCausalLM("microsoft/phi-1_5").to(device)
        b_pr, b_params = option_b_pr(encoder, llm, h_tr.detach(), y_tr, h_ev.detach(), y_ev,
                                     device, args.smoke, epochs=args.hist_epochs)       # C5

        results["claims"] = {
            "seq_pr_auc": seq_pr, "pooled_pr_auc": pool_pr, "catboost_pr_auc": cb_pr,
            "optionB_llm_pr_auc": b_pr, "optionB_trainable_params": b_params,
            "C3_temporal_lift_pp": (seq_pr - pool_pr) * 100,
            "C3_pass": (seq_pr - pool_pr) * 100 >= 10.0,
            "C4_heldout_gain_pp": (seq_pr - cb_pr) * 100,
            "C4_pass": (seq_pr - cb_pr) * 100 >= 5.0,
            "C5_llm_gap_pp": (b_pr - seq_pr) * 100,
            "C5_drop_llm": (b_pr - seq_pr) <= 0.02,   # dropping the LLM costs <= 2pp
        }
        c = results["claims"]
        print(f"\n=== entity task: {task_name}  (held-out prevalence {y_ev.mean():.2f}) ===")
        print(f"  sequence h_USR  PR-AUC {seq_pr:.3f}   (Option A)")
        print(f"  pooled mean(e)  PR-AUC {pool_pr:.3f}   -> C3 temporal lift "
              f"{c['C3_temporal_lift_pp']:+.1f} pp  {'PASS' if c['C3_pass'] else 'FAIL'}")
        print(f"  CatBoost aggr.  PR-AUC {cb_pr:.3f}   -> C4 held-out gain "
              f"{c['C4_heldout_gain_pp']:+.1f} pp  {'PASS' if c['C4_pass'] else 'FAIL'}")
        print(f"  LLM on h_USR    PR-AUC {b_pr:.3f}   -> C5 gap {c['C5_llm_gap_pp']:+.1f} pp  "
              f"{'DROP LLM' if c['C5_drop_llm'] else 'keep LLM'}")

    Path(args.out).write_text(json.dumps(results, indent=2, default=float))
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
