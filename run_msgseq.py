"""
v2 sequence over the ISO 20022 message lifecycle (Track M P2): predict the gpi tracker
SETTLEMENT OUTCOME (ACSC settled / ACSP+G002 held / RJCT rejected) of a payment from its
PRE-OUTCOME message prefix (pain.001, pacs.008, pacs.009 cover), with the v2 history encoder
vs an order-blind pooled baseline.

Non-circular: the pacs.002 / camt.054 / pacs.004 messages that STATE the outcome are excluded
from the prefix; we predict the pacs.002 tracker status from what precedes it.

Honest expectation (measured, not assumed): a single payment's lifecycle prefix is short
(2-3 messages) and near-fixed-order (pain.001 -> pacs.008 -> pacs.009), so the SEQUENCE carries
little beyond the bag-of-messages -> the temporal lift over pooling should be ~0. Where the
history encoder genuinely wins is LONG per-ENTITY histories with temporal correlation (see
run_seq velocity/C3). Reproducing that for tracker outcomes needs the generator to correlate an
account's successive payments (a future data change) - documented, not faked.

  python run_msgseq.py --smoke

ponytail: the sequence encoder here is expected to tie the pooled baseline; we build it to
MEASURE that honestly, and the code path is the upgrade hook for the entity-level version.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from data.iso_lifecycle import MSG_TYPES
from run_gpu import _to_device
from run_seq import embed_all_rows, encode_histories, frozen_embeddings

ROOT = Path(__file__).resolve().parent

PREFIX_TYPES = ("pain.001", "pacs.008", "pacs.009")     # pre-outcome content only
TRACKER_CLASS = {"ACSC": 0, "ACSP": 1, "RJCT": 2}       # settled / held / rejected
TRACKER_NAMES = ["ACSC (settled)", "ACSP/G002 (held)", "RJCT (rejected)"]


def message_prefix_sequences(msg, keep=PREFIX_TYPES, min_len=2):
    """Per-UETR ordered sequence of the pre-outcome messages -> sequence dicts (collate-ready).
    dt is the minute-gap from the previous message; calendar features are zero (a single
    payment's messages are minutes apart, so intra-payment calendar carries no signal)."""
    msg = msg.reset_index(drop=True)
    keep = set(keep)
    seq_col, type_col, off_col = msg["seq"].values, msg["msg_type"].values, msg["t_offset_min"].values
    seqs = []
    for e2e, idx in msg.groupby("end_to_end_id").groups.items():
        pos = np.asarray(idx, dtype=np.int64)
        pos = pos[np.argsort(seq_col[pos])]                        # lifecycle order
        pos = np.array([p for p in pos if type_col[p] in keep], dtype=np.int64)
        if len(pos) < min_len:
            continue
        t = off_col[pos].astype(np.float32)
        dt = np.diff(t, prepend=t[:1]).astype(np.float32)          # gap from prev (first = 0)
        z = np.zeros(len(pos), np.int64)
        seqs.append({"actor": e2e, "pos": pos, "dt": dt,
                     "dow": z, "dom": z.copy(), "month": z.copy()})
    return seqs


def tracker_labels(msg, seqs):
    """Map each sequence's UETR to its pacs.002 tracker outcome (3-class). Sequences whose
    payment never produced a pacs.002 (never reached clearing) are dropped."""
    p2 = msg[msg["msg_type"] == "pacs.002"]
    sts = dict(zip(p2["end_to_end_id"], p2["tx_sts"]))
    kept, y = [], []
    for s in seqs:
        code = TRACKER_CLASS.get(sts.get(s["actor"]))
        if code is not None:
            kept.append(s); y.append(code)
    return kept, np.asarray(y, dtype=np.int64)


def _split(seqs, y, frac_eval=0.2, seed=0):
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(seqs)); cut = int(len(seqs) * (1 - frac_eval))
    tr, ev = perm[:cut], perm[cut:]
    return ([seqs[i] for i in tr], y[tr]), ([seqs[i] for i in ev], y[ev])


def _pooled(e_all, seqs):
    return np.stack([e_all[s["pos"]].mean(0).cpu().numpy() for s in seqs])


def _probe(Xtr, ytr, Xev, yev):
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score, f1_score
    clf = LogisticRegression(max_iter=1000, class_weight="balanced").fit(Xtr, ytr)
    pred = clf.predict(Xev)
    return float(accuracy_score(yev, pred)), float(f1_score(yev, pred, average="macro"))


def main():
    from encoder.history_encoder import HistoryConfig, HistoryEncoder
    from encoder.history_encoder import pretrain as hist_pretrain

    ap = argparse.ArgumentParser(description="v2 message-lifecycle sequence -> gpi tracker outcome")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--payments", default=str(ROOT / "data" / "india_rails_payments.parquet"))
    ap.add_argument("--messages", default=str(ROOT / "data" / "india_rails_messages.parquet"))
    ap.add_argument("--schema", default=str(ROOT / "data" / "column_schema_india.json"))
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--hist-epochs", type=int, default=3)
    ap.add_argument("--out", default=str(ROOT / "results_msgseq.json"))
    args = ap.parse_args()

    np.random.seed(0)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    pay = pd.read_parquet(args.payments)
    msg = pd.read_parquet(args.messages)
    if args.limit:
        pay = pay.head(args.limit).reset_index(drop=True)
        msg = msg[msg["payment_id"].isin(pay["payment_id"])].reset_index(drop=True)
    schema = json.loads(Path(args.schema).read_text())
    print(f"device={device}  mode={'smoke' if args.smoke else 'full'}  payments {len(pay):,}"
          f"  messages {len(msg):,}")

    # frozen v1 encoder -> per-message embeddings
    encoder, vocabs, _, enc_cfg = frozen_embeddings(pay, schema, args.smoke, device)
    full_msg = vocabs.encode(msg)
    e_msg = embed_all_rows(encoder, full_msg, len(msg), device).to(device)
    D = enc_cfg.hidden

    # sequences + tracker labels
    seqs = message_prefix_sequences(msg)
    seqs, y = tracker_labels(msg, seqs)
    print(f"prefix sequences {len(seqs):,}  (mean len {np.mean([len(s['pos']) for s in seqs]):.1f})"
          f"  class balance {np.bincount(y, minlength=3).tolist()} = {TRACKER_NAMES}")
    if len(seqs) < 50 or len(set(y.tolist())) < 2:
        raise SystemExit("not enough labelled sequences; raise --limit")
    (tr_seqs, y_tr), (ev_seqs, y_ev) = _split(seqs, y)

    # history-encoder reconstruction targets: Ccy + msg_type (discrete, per message row)
    recon_fields = {"Ccy": vocabs.core_size("Ccy"), "msg_type": len(MSG_TYPES)}
    mt_index = {m: i for i, m in enumerate(MSG_TYPES)}
    targets_all = {
        "Ccy": full_msg["core"]["Ccy"],
        "msg_type": torch.tensor([mt_index[t] for t in msg["msg_type"]], dtype=torch.long),
    }
    hcfg = HistoryConfig(hidden=D, layers=2 if args.smoke else 4,
                         heads=2 if args.smoke else 8, ff_mult=2 if args.smoke else 4,
                         epochs=args.hist_epochs)
    hist = HistoryEncoder(recon_fields, hcfg).to(device)
    print(f"history encoder: {hist.num_trainable_parameters():,} trainable params")
    hist_pretrain(hist, e_msg, targets_all, tr_seqs, hcfg, batch_size=64 if args.smoke else 128)
    hist.freeze()

    # sequence rep (h_USR) vs order-blind pooled mean -> 3-class tracker probe
    h_tr = encode_histories(hist, e_msg, tr_seqs, device).cpu().numpy()
    h_ev = encode_histories(hist, e_msg, ev_seqs, device).cpu().numpy()
    seq_acc, seq_f1 = _probe(h_tr, y_tr, h_ev, y_ev)
    pool_acc, pool_f1 = _probe(_pooled(e_msg, tr_seqs), y_tr, _pooled(e_msg, ev_seqs), y_ev)

    res = {"mode": "smoke" if args.smoke else "full", "n_sequences": len(seqs),
           "held_out": int(len(y_ev)), "class_names": TRACKER_NAMES,
           "sequence": {"acc": seq_acc, "macro_f1": seq_f1},
           "pooled": {"acc": pool_acc, "macro_f1": pool_f1},
           "temporal_lift_pp": (seq_f1 - pool_f1) * 100}
    print(f"\ngpi tracker outcome (held-out {len(y_ev)}):")
    print(f"  sequence h_USR   acc {seq_acc:.3f}  macroF1 {seq_f1:.3f}")
    print(f"  pooled mean(e)   acc {pool_acc:.3f}  macroF1 {pool_f1:.3f}")
    print(f"  -> temporal lift {res['temporal_lift_pp']:+.1f} pp macroF1  "
          f"({'sequence helps' if res['temporal_lift_pp'] > 2 else 'pooled ~ sequence (short prefix)'})")
    Path(args.out).write_text(json.dumps(res, indent=2))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
