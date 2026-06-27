"""
Frozen-Phi payment twin - the twin's INTAKE classification heads replaced entirely by
a frozen LLM + adapters (instruction tuning), per the user's "extend the frozen-Phi
approach" direction. Grounded in the same frozen-pretrained-LM pattern shown to work for
time series (GPT4TS / One-Fits-All, Time-LLM) and transactions (arXiv:2410.07851).

For each classification target the frozen encoder's f(payment) is fed - via the decoder's
precomputed-feature path - into a frozen LLM with small trainable adapters {Phi, psi, phi},
and the answer is read off the LLM's next token:
  * terminal status  (4-class: STP / REPAIRED / MANUAL_REVIEW / REJECTED)
  * each exception   (binary: will this exception fire?)

  --smoke : MockLLM + CPU. WARNING: MockLLM is a random toy - these numbers validate the
            plumbing, they are NOT a fair test of frozen Phi. Use the GPU command in the
            module footer (real microsoft/phi-1_5) for the real numbers.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from run_gpu import _to_device
from run_seq import embed_all_rows
from run_twin import load_twin

ROOT = Path(__file__).resolve().parent
_LETTERS = "ABCDEFGH"


def _task_tokens(llm, n_classes, smoke, device, label):
    if smoke:
        return (torch.randint(0, llm.vocab_size, (4,), device=device), list(range(n_classes)))
    tok = llm.tokenizer
    letters = _LETTERS[:n_classes]
    prompt = f"Classify the {label}. Answer with a single letter: " + ", ".join(letters) + ". Answer:"
    instr = torch.tensor(tok(prompt, add_special_tokens=False)["input_ids"], device=device)
    answers = [tok(f" {c}", add_special_tokens=False)["input_ids"][0] for c in letters]
    return instr, answers


def llm_head(encoder, llm, feat_tr, y_tr, feat_ev, y_ev, n_classes, smoke, device,
             epochs=3, bs=128, label=""):
    """Train a frozen-LLM + adapter head on feat -> class; return eval metrics."""
    from decoder.multimodal_decoder import DecoderConfig, MultimodalDecoder
    from sklearn.metrics import accuracy_score, average_precision_score, f1_score

    instr, answers = _task_tokens(llm, n_classes, smoke, device, label)
    dec = MultimodalDecoder(encoder, llm,
                            DecoderConfig(n_tasks=1, max_records=1, phi_mode="prompt")).to(device)
    dec.assert_frozen()                                  # encoder + LLM frozen; only {Phi,psi,phi} train
    tgt = torch.tensor([[answers[int(y)]] for y in y_tr], device=device)
    opt = torch.optim.Adam([p for p in dec.parameters() if p.requires_grad], lr=1e-4)
    n = len(y_tr)
    dec.train()
    for _ in range(epochs):
        perm = torch.randperm(n)
        for s in range(0, n, bs):
            idx = perm[s:s + bs]
            B = len(idx)
            loss = dec(feat_tr[idx], torch.zeros(B, dtype=torch.long, device=device),
                       instr.unsqueeze(0).expand(B, -1), tgt[idx])
            opt.zero_grad(); loss.backward(); opt.step()
    dec.eval()
    parts = []
    with torch.no_grad():
        for s in range(0, len(y_ev), bs):
            f = feat_ev[s:s + bs]
            B = f.shape[0]
            parts.append(dec.predict_proba(f, torch.zeros(B, dtype=torch.long, device=device),
                                           instr.unsqueeze(0).expand(B, -1), answers).cpu().numpy())
    P = np.concatenate(parts, axis=0)
    if n_classes == 2:
        return {"pr_auc": float(average_precision_score(y_ev, P[:, 1])),
                "trainable_params": dec.trainable_parameters()}
    pred = P.argmax(1)
    return {"accuracy": float(accuracy_score(y_ev, pred)),
            "macro_f1": float(f1_score(y_ev, pred, average="macro")),
            "trainable_params": dec.trainable_parameters()}


def _full_examples(evt, pay, step_vocab, exc_vocab, dir_vocab):
    """One full step-sequence per payment (for the self-supervised history pretrain)."""
    dir_by_id = dict(zip(pay["payment_id"], pay["direction"]))
    evt = evt[evt["outcome"] != "repaired"]
    ex = []
    for pid, sub in evt.groupby("payment_id"):
        sub = sub.sort_values("seq")
        steps = sub["step"].map(step_vocab).to_numpy()
        if len(steps) < 1:
            continue
        ex.append({"step": steps, "exc": sub["excode"].map(exc_vocab).to_numpy(),
                   "t": sub["t_min"].to_numpy(np.float32),
                   "direction": dir_vocab[dir_by_id[pid]], "target": 0})
    return ex


def inflight_phi(evt, pay, tr_ids, ev_ids, encoder, llm, schema, device, smoke, epochs, log=print):
    """In-flight next-exception via frozen Phi (Time-LLM style): a self-supervised history
    encoder makes a prefix representation h; the frozen LLM + adapters read h to predict
    the next step's exception. h is a FROZEN feature, consistent with the intake path."""
    from encoder.history_encoder import HistoryConfig, HistoryEncoder
    from run_twin import StepEmbedder, build_step_examples, collate_steps

    twin = schema["twin"]
    step_vocab = {s: i for i, s in enumerate(sorted({s for w in twin["workflow"].values() for s in w}))}
    codes = ["none"] + twin["exception_codes"]
    exc_vocab = {c: i for i, c in enumerate(codes)}
    dir_vocab = {d: i for i, d in enumerate(twin["directions"])}
    D = encoder.D                                        # h must match the decoder adapter's d_enc
    hcfg = HistoryConfig(hidden=D, layers=2 if smoke else 4, heads=2 if smoke else 8,
                         ff_mult=2 if smoke else 4)
    steps_mod = StepEmbedder(len(step_vocab), len(exc_vocab), len(dir_vocab), D).to(device)
    hist = HistoryEncoder(recon_fields={"step": len(step_vocab), "excode": len(exc_vocab)},
                          config=hcfg).to(device)

    # self-supervised pretrain of step-encoder + history-encoder (masked-event + CoLES)
    full = _full_examples(evt[evt.payment_id.isin(tr_ids)], pay, step_vocab, exc_vocab, dir_vocab)
    opt = torch.optim.Adam(list(steps_mod.parameters()) + list(hist.parameters()), lr=1e-3)
    bs = 128
    for ep in range(epochs):
        perm = np.random.permutation(len(full)); tot = 0.0
        for s in range(0, len(perm), bs):
            b = collate_steps([full[i] for i in perm[s:s + bs]], device)
            e_seq = steps_mod(b["step"], b["exc"], b["direction"])
            loss, _ = hist.composite_loss(e_seq, b, {"step": b["step"], "excode": b["exc"]})
            opt.zero_grad(); loss.backward(); opt.step(); tot += float(loss)
        log(f"  hist-pretrain epoch {ep+1}/{epochs}  loss {tot/max(1,len(perm)//bs):.4f}")
    steps_mod.eval(); hist.freeze()

    def reps(ids):
        ex = build_step_examples(evt[evt.payment_id.isin(ids)], pay, step_vocab, exc_vocab,
                                 dir_vocab, exc_vocab, 16)
        H, Y = [], []
        for s in range(0, len(ex), bs):
            b = collate_steps(ex[s:s + bs], device)
            with torch.no_grad():
                h, _ = hist(steps_mod(b["step"], b["exc"], b["direction"]), b, event_mask=None)
            H.append(h); Y.append(b["target"].cpu().numpy())
        return torch.cat(H, 0), np.concatenate(Y)

    h_tr, y_tr = reps(tr_ids)
    h_ev, y_ev = reps(ev_ids)
    log(f"[in-flight] frozen h: train {len(y_tr):,} / held-out {len(y_ev):,} prefixes")
    return llm_head(encoder, llm, h_tr, y_tr, h_ev, y_ev, len(codes), smoke, device,
                    epochs=epochs, label="next exception")


def main():
    from encoder.tabular_encoder import EncoderConfig, build_pretraining_stack
    from encoder.tabular_encoder import pretrain as enc_pretrain

    ap = argparse.ArgumentParser(description="frozen-Phi payment twin (intake classification)")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--payments", default=str(ROOT / "data" / "pacs008_twin_payments.parquet"))
    ap.add_argument("--events", default=str(ROOT / "data" / "pacs008_twin_events.parquet"))
    ap.add_argument("--schema", default=str(ROOT / "data" / "column_schema_twin.json"))
    ap.add_argument("--llm", default="microsoft/phi-1_5")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--max-exc", type=int, default=None, help="cap #exception heads (speed)")
    ap.add_argument("--out", default=str(ROOT / "results_twin_llm.json"))
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device={device}  mode={'smoke' if args.smoke else 'full'}")
    if args.smoke:
        print("WARNING: smoke uses MockLLM (random toy) - plumbing check, NOT a fair Phi test.")
    pay, evt, schema = load_twin(args)
    twin = schema["twin"]

    enc_cfg = (EncoderConfig(hidden=64, layers=2, heads=2, ff_mult=2, epochs=1)
               if args.smoke else EncoderConfig())
    torch.manual_seed(0)
    encoder, _asm, vocabs = build_pretraining_stack(pay, schema, enc_cfg, party_epochs=1)
    encoder.to(device)
    print("[A] pretrain v1 encoder ...")
    enc_pretrain(encoder, _to_device(vocabs.encode(pay), device), enc_cfg,
                 batch_size=128 if args.smoke else 256)
    encoder.freeze()
    e_pay = embed_all_rows(encoder, vocabs.encode(pay), len(pay), device).to(device)

    rng = np.random.default_rng(0)
    idx = rng.permutation(len(pay)); cut = int(len(idx) * 0.8)
    tr, ev = idx[:cut], idx[cut:]

    if args.smoke:
        from decoder.multimodal_decoder import MockLLM
        llm = MockLLM(vocab_size=64, hidden=64, num_layers=2, num_heads=4).to(device)
    else:
        from decoder.multimodal_decoder import HFCausalLM
        llm = HFCausalLM(args.llm).to(device)

    out = {"mode": "smoke" if args.smoke else "full", "device": device, "intake": {}}

    # terminal status (multiclass) through frozen Phi
    sv = {s: i for i, s in enumerate(twin["terminal_status"])}
    ys = pay[twin["status_column"]].map(sv).to_numpy()
    out["intake"]["status"] = llm_head(encoder, llm, e_pay[tr], ys[tr], e_pay[ev], ys[ev],
                                       len(sv), args.smoke, device, args.epochs, label="terminal status")
    print(f"[Phi-intake] status: {out['intake']['status']}")

    # each exception (binary) through frozen Phi
    exc_cols = twin["exc_columns"][: args.max_exc] if args.max_exc else twin["exc_columns"]
    out["intake"]["exceptions"] = {}
    for c in exc_cols:
        y = pay[c].to_numpy()
        if len(set(y[tr])) < 2 or len(set(y[ev])) < 2:
            continue
        ep = 2 if args.smoke else args.epochs
        r = llm_head(encoder, llm, e_pay[tr], y[tr], e_pay[ev], y[ev], 2, args.smoke,
                     device, ep, label=c.replace("exc_", ""))
        out["intake"]["exceptions"][c.replace("exc_", "")] = r["pr_auc"]
        print(f"[Phi-intake] {c.replace('exc_','')}: PR-AUC {r['pr_auc']:.3f}")

    # in-flight next-exception through frozen Phi (self-supervised history rep -> Phi)
    tr_ids = set(pay["payment_id"].to_numpy()[tr])
    ev_ids = set(pay["payment_id"].to_numpy()[ev])
    out["in_flight"] = {"next_exception": inflight_phi(
        evt, pay, tr_ids, ev_ids, encoder, llm, schema, device, args.smoke, args.epochs)}
    print(f"[Phi-in-flight] next-exception: {out['in_flight']['next_exception']}")

    Path(args.out).write_text(json.dumps(out, indent=2, default=float))
    print(f"\nwrote {args.out}")


# GPU (real frozen Phi-1.5):
#   pip install transformers huggingface_hub
#   python data/synth_workflow.py --accounts 20000 --payments 200000 \
#     --out-prefix data/pacs008_twin --schema-out data/column_schema_twin.json
#   python run_twin_llm.py            # frozen microsoft/phi-1_5 + adapters
if __name__ == "__main__":
    main()
