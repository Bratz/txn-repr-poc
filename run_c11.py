"""
C11 - entity-level fusion: score the CURRENT payment given the ACCOUNT's history.

PRAGMA puts everything the bank knows about the user behind every prediction;
our in-flight heads see only the current payment's prefix. C11 fuses them:

  base    _prefix_features(current payment's pre-outcome messages)   (serve_india)
  fused   base (+) h_entity - the history encoder over the account's FULL message
          timeline across its PRIOR payments (fractional-day dt - C13; recalls and
          returns of past payments are IN the history, that's the signal)

Measured on cancel/return of the current payment, held-out ACCOUNTS, on TWO
fleets from the same seed: recall_momentum ON (the signal exists - planted
account-level recall clustering) and OFF (the pre-registered NULL CONTROL:
fusion should buy ~nothing).

Pre-registered (docs/V2_DIRECTION.md, C11): pass = ON-fleet fused beats base by
>= +5pp PR-AUC on cancel or return, AND the OFF fleet shows no comparable lift.

  python run_c11.py --smoke
  python run_c11.py                 # 500x12k dense fleet, full encoder (CPU ~2h)
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent

PREFIX_TYPES = ("pain.001", "pacs.008", "pacs.009")      # pre-outcome only (no leakage)


def fleet_result(momentum, args, device):
    from data.lifelong import entity_message_timeline, entity_prefix
    from data.iso_lifecycle import MSG_TYPES
    from data.synth_india_rails import IndiaConfig, build_dataset, build_messages, build_schema
    from run_seq import embed_all_rows, encode_histories, frozen_embeddings, small_history_encoder
    from serve_india import _prefix_features

    pay, evt, accs = build_dataset(IndiaConfig(
        num_accounts=args.accounts, num_payments=args.payments, seed=11,
        recall_momentum=momentum))
    schema = build_schema(pay, accs)
    msg = build_messages(pay, evt)
    print(f"[m={momentum}] payments {len(pay):,}  msgs {len(msg):,}  "
          f"cancel prev {pay['cancel_requested'].mean():.3%}  "
          f"return prev {pay['returned'].mean():.3%}")

    encoder, vocabs, _, enc_cfg = frozen_embeddings(pay, schema, args.smoke, device)
    msg_sorted, timelines = entity_message_timeline(pay, msg)
    full_msg = vocabs.encode(msg_sorted)
    e_msg = embed_all_rows(encoder, full_msg, len(msg_sorted), device).to(device)

    # examples: every payment whose account has prior history
    pdate = dict(zip(pay["payment_id"],
                     np.asarray(np.array(pay["IntrBkSttlmDt"], dtype="datetime64[D]")
                                .astype(np.int64), dtype=float)))
    pactor = dict(zip(pay["payment_id"], pay["DbtrAcct_Id"].astype(str)))
    by_pid = {pid: g for pid, g in msg_sorted.groupby("payment_id")}
    X_base, ent_seqs, ys, actors = [], [], [], []
    for pid in pay["payment_id"]:
        actor = pactor[pid]
        tl = timelines.get(actor)         # accounts with zero VISIBLE messages skip
        if tl is None:
            continue
        seq = entity_prefix(tl, pid, pdate[pid])
        if seq is None:
            continue
        g = by_pid.get(pid)
        if g is None:
            continue
        g = g[g["msg_type"].isin(PREFIX_TYPES)].sort_values("seq")
        if not len(g):
            continue
        pool = e_msg[g.index.to_numpy()].mean(0).cpu().numpy()
        X_base.append(_prefix_features(pool, g))
        ent_seqs.append(seq)
        row = pay[pay["payment_id"] == pid].iloc[0]
        ys.append((int(row["cancel_requested"]), int(row["returned"])))
        actors.append(actor)
    X_base = np.asarray(X_base, dtype=np.float32)
    ys = np.asarray(ys)
    actors = np.asarray(actors)
    print(f"[m={momentum}] examples {len(ys):,} (accounts {len(set(actors)):,})")

    # history encoder over message embeddings, trained on TRAIN accounts only
    uniq = sorted(set(actors))
    rng = np.random.default_rng(0); rng.shuffle(uniq)
    ev_actors = set(uniq[: max(1, len(uniq) // 5)])
    ev_m = np.isin(actors, list(ev_actors)); tr_m = ~ev_m
    mt_index = {m: i for i, m in enumerate(MSG_TYPES)}
    recon = {"Ccy": vocabs.core_size("Ccy"), "msg_type": len(MSG_TYPES)}
    targets = {"Ccy": full_msg["core"]["Ccy"],
               "msg_type": torch.tensor([mt_index[t] for t in msg_sorted["msg_type"]],
                                        dtype=torch.long)}
    hist, _ = small_history_encoder(e_msg, recon, targets,
                                    [s for s, k in zip(ent_seqs, tr_m) if k], device,
                                    epochs=args.hist_epochs)
    H = encode_histories(hist, e_msg, ent_seqs, device).cpu().numpy()
    fused = np.hstack([X_base, H])

    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import average_precision_score
    out = {}
    for j, name in enumerate(("cancel", "return")):
        y = ys[:, j]
        if len(set(y[tr_m].tolist())) < 2 or y[ev_m].sum() == 0:
            out[name] = {"base": None, "fused": None, "note": "degenerate at this scale"}
            continue
        r = {}
        for label, X in (("base", X_base), ("fused", fused)):
            clf = LogisticRegression(max_iter=2000, class_weight="balanced").fit(
                X[tr_m], y[tr_m])
            r[label] = float(average_precision_score(
                y[ev_m], clf.predict_proba(X[ev_m])[:, 1]))
        r["delta_pp"] = (r["fused"] - r["base"]) * 100
        r["prevalence"] = float(y[ev_m].mean())
        out[name] = r
        print(f"[m={momentum}] {name:6} base {r['base']:.3f} -> fused {r['fused']:.3f} "
              f"({r['delta_pp']:+.1f}pp, prev {r['prevalence']:.3%})")
    return out


def main():
    ap = argparse.ArgumentParser(description="C11 - entity fusion for in-flight "
                                             "cancel/return, with momentum null control")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--accounts", type=int, default=500)
    ap.add_argument("--payments", type=int, default=12000)
    ap.add_argument("--momentum", type=float, default=1.0)
    ap.add_argument("--hist-epochs", type=int, default=2)
    ap.add_argument("--out", default=str(ROOT / "results_c11.json"))
    args = ap.parse_args()

    torch.manual_seed(0); np.random.seed(0)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if args.smoke:
        args.accounts, args.payments = 60, 1200
    print(f"device={device}  mode={'smoke' if args.smoke else 'full'}")

    res = {"mode": "smoke" if args.smoke else "full",
           "momentum": args.momentum,
           "on": fleet_result(args.momentum, args, device),
           "off": fleet_result(0.0, args, device)}

    def _delta(block, name):
        d = block.get(name, {}).get("delta_pp")
        return d if d is not None else float("nan")
    on_best = np.nanmax([_delta(res["on"], "cancel"), _delta(res["on"], "return")])
    off_best = np.nanmax([_delta(res["off"], "cancel"), _delta(res["off"], "return")])
    res["C11_pass"] = bool(on_best >= 5.0 and not (off_best >= 5.0))
    print(f"\nC11: momentum-ON best fusion delta {on_best:+.1f}pp | "
          f"OFF (null control) {off_best:+.1f}pp")
    print(f"  -> C11 {'PASS' if res['C11_pass'] else 'FAIL'} "
          f"(ON >= +5pp AND the null stays null)")
    Path(args.out).write_text(json.dumps(res, indent=2))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
