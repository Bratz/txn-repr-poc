"""
Next-payment / next-receipt forecasting on the cadence-enabled India data - heads vs the
naive baselines that MUST be beaten for the capability to be real.

Per held-out-ACTOR example (history prefix -> next event):
  occurrence  will another event arrive within the horizon?   probe on h_USR  vs  CatBoost on
              aggregates  vs  rule score (-median gap)
  when        gap to the next event (days)                    Ridge on h_USR  vs  last-gap /
              median-gap naive
  amount      next amount                                     Ridge on h_USR  vs  history-
              median naive
  payee       next counterparty                               baseline only (most-frequent
              payee top-1) - the retrieval head is deferred; the number to beat is printed.

Honesty notes: fixed monthly cadence is trivially predicted by "median gap" - the model only
earns this task on mixed/irregular cadence, which is why the baselines ride along. Receipts =
the same run with --actor CdtrAcct_Id.

  python run_next.py --smoke
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from data.next_event import windowed_examples
from data.synth_india_rails import IndiaConfig, build_dataset, build_schema
from run_seq import catboost_pr, encode_histories, frozen_embeddings, probe_pr

ROOT = Path(__file__).resolve().parent


def _reg(h_tr, y_tr, h_ev, y_ev):
    from sklearn.linear_model import Ridge
    from sklearn.metrics import mean_absolute_error
    r = Ridge().fit(h_tr, np.log1p(np.clip(y_tr, 0, None)))
    pred = np.expm1(r.predict(h_ev))
    return float(mean_absolute_error(y_ev, np.clip(pred, 0, None)))


def main():
    from encoder.history_encoder import HistoryConfig, HistoryEncoder
    from encoder.history_encoder import pretrain as hist_pretrain

    ap = argparse.ArgumentParser(description="next-event forecasting vs naive baselines")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--actor", default="DbtrAcct_Id",
                    help="DbtrAcct_Id = next payment; CdtrAcct_Id = next receipt")
    ap.add_argument("--accounts", type=int, default=500)
    ap.add_argument("--payments", type=int, default=12000)
    ap.add_argument("--cadence", type=float, default=0.5)
    ap.add_argument("--horizon", type=int, default=35)
    ap.add_argument("--hist-epochs", type=int, default=2)
    ap.add_argument("--out", default=str(ROOT / "results_next.json"))
    args = ap.parse_args()

    np.random.seed(0)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    pay, _, accs = build_dataset(IndiaConfig(
        num_accounts=args.accounts, num_payments=args.payments, seed=29,
        cadence_frac=args.cadence))
    schema = build_schema(pay, accs)
    print(f"cadence dataset {len(pay):,} payments  mix "
          f"{pay['cadence'].value_counts().to_dict()}")

    encoder, vocabs, e_pay, enc_cfg = frozen_embeddings(pay, schema, args.smoke, device)
    e_all = torch.as_tensor(e_pay).to(device)

    seqs, lab = windowed_examples(pay, actor_col=args.actor, horizon_days=args.horizon)
    actors = sorted(set(lab["actor"]))
    rng = np.random.default_rng(0); rng.shuffle(actors)
    ev_actors = set(actors[: max(1, len(actors) // 5)])
    ev_m = lab["actor"].isin(ev_actors).to_numpy(); tr_m = ~ev_m
    print(f"examples {len(lab):,}  (train {tr_m.sum():,} / held-out actors {ev_m.sum():,}; "
          f"occurrence prevalence {lab['has_next'].mean():.3f})")

    # history encoder over per-example prefixes (reuses the frozen e_t)
    recon_fields = {"Ccy": vocabs.core_size("Ccy"),
                    "identifier_type": vocabs.core_size("identifier_type")}
    full = vocabs.encode(pay)
    targets_all = {n: full["core"][n] for n in recon_fields}
    hcfg = HistoryConfig(hidden=int(e_all.shape[1]), layers=2, heads=4, ff_mult=2,
                         epochs=args.hist_epochs)
    hist = HistoryEncoder(recon_fields, hcfg).to(device)
    tr_seqs = [s for s, m in zip(seqs, tr_m) if m]
    hist_pretrain(hist, e_all, targets_all, tr_seqs, hcfg, batch_size=64)
    hist.freeze()
    H = encode_histories(hist, e_all, seqs, device).cpu().numpy()

    y_occ = lab["has_next"].to_numpy()
    res = {"actor": args.actor, "n_examples": int(len(lab)),
           "occurrence_prevalence": float(y_occ.mean())}

    # --- occurrence: model vs trees vs periodicity rule --------------------------- #
    res["occ_model_pr"] = probe_pr(H[tr_m], y_occ[tr_m], H[ev_m], y_occ[ev_m])
    agg = lab[["hist_gap_median", "hist_gap_last", "hist_amount_median",
               "n_history"]].fillna(999).to_numpy(dtype=float)
    res["occ_tree_pr"] = catboost_pr(agg[tr_m], y_occ[tr_m], agg[ev_m], y_occ[ev_m],
                                     iters=100)
    from sklearn.metrics import average_precision_score
    res["occ_rule_pr"] = float(average_precision_score(
        y_occ[ev_m], -lab["hist_gap_median"].fillna(999).to_numpy()[ev_m]))

    # --- when + amount: uncensored examples only ----------------------------------- #
    unc = lab["gap_days"].notna().to_numpy()
    for name, y, naive_cols in (
        ("gap_days", lab["gap_days"].to_numpy(dtype=float),
         ("hist_gap_last", "hist_gap_median")),
        ("next_amount", lab["next_amount"].to_numpy(dtype=float),
         ("hist_amount_median",)),
    ):
        t, e = tr_m & unc, ev_m & unc
        res[f"{name}_model_mae"] = _reg(H[t], y[t], H[e], y[e])
        for c in naive_cols:
            naive = lab[c].to_numpy(dtype=float)[e]
            res[f"{name}_naive_{c}_mae"] = float(np.nanmean(np.abs(naive - y[e])))

    # --- payee: the baseline the future retrieval head must beat ------------------- #
    e = ev_m & unc
    res["payee_baseline_top1"] = float(
        (lab["hist_top_payee"].to_numpy()[e] == lab["next_payee"].to_numpy()[e]).mean())

    print(f"\nnext-event ({'payment' if args.actor.startswith('Dbtr') else 'receipt'}, "
          f"held-out actors, horizon {args.horizon}d):")
    print(f"  occurrence  model PR {res['occ_model_pr']:.3f} | tree {res['occ_tree_pr']:.3f}"
          f" | rule(-median gap) {res['occ_rule_pr']:.3f} | prev {res['occurrence_prevalence']:.3f}")
    print(f"  when (MAE)  model {res['gap_days_model_mae']:.1f}d | naive last-gap "
          f"{res['gap_days_naive_hist_gap_last_mae']:.1f}d | naive median "
          f"{res['gap_days_naive_hist_gap_median_mae']:.1f}d")
    print(f"  amount(MAE) model {res['next_amount_model_mae']:,.0f} | naive median "
          f"{res['next_amount_naive_hist_amount_median_mae']:,.0f}")
    print(f"  payee       most-frequent-payee top-1 {res['payee_baseline_top1']:.3f} "
          f"(retrieval head deferred - this is the number to beat)")
    Path(args.out).write_text(json.dumps(res, indent=2))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
