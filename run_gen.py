"""
C8 - the TransactionGPT tier of the payment twin: GENERATE the next transaction.

TGPT (arXiv:2511.08939) predicts the next transaction's fields (time, amount, MCC/
merchant) from the history and rolls trajectories forward. Ported to our stack the
house way - frozen backbone + small heads:

  next-field heads   h_USR (frozen v2 history encoder over frozen v1 row embeddings)
                     -> gap_band (6 classes, run_c7 bands)
                     -> amount_band (currency-conditioned quantizer level // 8, 16 groups)
                     -> rail (4-way; an output field only - rail is metadata, like TGPT's MCC)
  rollout            predict fields -> synthesize the next row (copy the actor's last row,
                     advance the date by the band representative, set the amount to the
                     predicted group's grid value) -> re-encode through the FROZEN encoder
                     -> append -> repeat. Argmax by default; --sample draws from the heads.

Pre-registered (docs/V2_DIRECTION.md, C8): 1-step field accuracy vs per-actor naives
(band of median gap / band of median amount / majority rail), held-out actors. Pass =
beat the naive on >= 2 of 3 fields. Rollout distributions are reported, not thresholded.

  python run_gen.py --smoke
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from run_c7 import _GAP_EDGES, GAP_BANDS

ROOT = Path(__file__).resolve().parent

NEXT_BANDS = list(GAP_BANDS[1:])                     # no "first" for a next-gap
BAND_DAYS = np.array([0, 2, 5, 11, 22, 45], float)   # representative days per band
AMOUNT_GROUPS = 16                                   # quantizer levels // 8


def gap_band_of(gaps):
    """Days-to-next -> band index into NEXT_BANDS (same edges as run_c7)."""
    g = np.asarray(gaps, dtype=float)
    return np.searchsorted(_GAP_EDGES, g, side="left")


def amount_group_of(quantizer, amounts, ccys):
    lvl = quantizer.transform(np.asarray(amounts, float), np.asarray(ccys, object))
    return np.asarray(lvl, dtype=np.int64) // (quantizer.num_levels // AMOUNT_GROUPS)


def decode_amount(quantizer, group, ccy):
    """Predicted group -> the mid-level grid value of that group for the currency."""
    grid = quantizer.grids_.get(ccy, quantizer.grids_["__GLOBAL__"])
    per = quantizer.num_levels // AMOUNT_GROUPS
    lvl = min(int(group) * per + per // 2, len(grid) - 1)
    return float(grid[lvl])


def example_fields(pay, seqs, lab, quantizer):
    """Per-example (label, naive) pairs for the three fields. Censored rows -> masked."""
    rail = pay["rail"].astype(str).to_numpy()
    ccy = pay["Ccy"].astype(str).to_numpy()
    m = lab["next_pos"].to_numpy() >= 0
    nxt = lab["next_pos"].to_numpy()[m]
    ex_ccy = np.array([ccy[s["pos"][-1]] for s in seqs], object)

    y = {"gap": gap_band_of(lab["gap_days"].to_numpy()[m]),
         "amount": amount_group_of(quantizer, lab["next_amount"].to_numpy()[m], ex_ccy[m]),
         "rail": rail[nxt]}
    naive = {"gap": gap_band_of(lab["hist_gap_median"].fillna(999).to_numpy()[m]),
             "amount": amount_group_of(quantizer,
                                       lab["hist_amount_median"].to_numpy()[m], ex_ccy[m]),
             "rail": np.array([pd.Series(rail[s["pos"]]).mode().iloc[0]
                               for s, keep in zip(seqs, m) if keep], object)}
    return m, y, naive, ex_ccy


def fit_field_heads(H_tr, y_tr):
    # plain (unbalanced) LR: C8's metric is accuracy and the naive plays the mode -
    # balanced weights would handicap the model on the same metric.
    from sklearn.linear_model import LogisticRegression
    return {f: LogisticRegression(max_iter=2000).fit(H_tr, y)
            for f, y in y_tr.items()}


@torch.no_grad()
def simulate(hist_df, encoder, vocabs, hist, heads, quantizer, steps, device,
             max_len=64, sample=False, seed=0):
    """TGPT-style rollout: T synthetic future payments for ONE actor's history rows.
    Returns a DataFrame of the generated rows (date, amount, rail + head confidences)."""
    from data.next_event import seq_from_dates
    from run_seq import embed_all_rows, encode_histories
    rng = np.random.default_rng(seed)
    df = hist_df.sort_values("IntrBkSttlmDt").reset_index(drop=True).copy()
    e = embed_all_rows(encoder, vocabs.encode(df), len(df), device).cpu()
    out = []
    for _ in range(steps):
        tail = df.iloc[-max_len:]
        pos = np.arange(len(df) - len(tail), len(df))
        seq = seq_from_dates("roll", pos, pd.to_datetime(tail["IntrBkSttlmDt"]).values)
        H = encode_histories(hist, e.to(device), [seq], device).cpu().numpy()
        row = df.iloc[-1].copy()
        pred = {}
        for f, clf in heads.items():
            p = clf.predict_proba(H)[0]
            j = rng.choice(len(p), p=p) if sample else int(p.argmax())
            pred[f] = (clf.classes_[j], float(p[j]))
        band = int(pred["gap"][0])
        new_date = (pd.to_datetime(row["IntrBkSttlmDt"])
                    + pd.Timedelta(days=float(BAND_DAYS[band])))
        row["IntrBkSttlmDt"] = new_date.date().isoformat()
        row["IntrBkSttlmAmt"] = decode_amount(quantizer, pred["amount"][0],
                                              str(row["Ccy"]))
        if "rail" in df.columns:
            row["rail"] = str(pred["rail"][0])
        df = pd.concat([df, row.to_frame().T], ignore_index=True)
        e_new = embed_all_rows(encoder, vocabs.encode(df.iloc[[-1]]), 1, device).cpu()
        e = torch.cat([e, e_new])
        out.append({"IntrBkSttlmDt": row["IntrBkSttlmDt"],
                    "IntrBkSttlmAmt": float(row["IntrBkSttlmAmt"]),
                    "rail_pred": str(pred["rail"][0]),
                    "gap_band": NEXT_BANDS[band],
                    "conf_gap": pred["gap"][1], "conf_amount": pred["amount"][1],
                    "conf_rail": pred["rail"][1]})
    return pd.DataFrame(out)


def main():
    from data.next_event import windowed_examples
    from data.synth_india_rails import IndiaConfig, build_dataset, build_schema
    from encoders.quantizer import AdaptiveQuantizer
    from run_seq import encode_histories, frozen_embeddings, small_history_encoder

    ap = argparse.ArgumentParser(description="C8 - TGPT tier: next-field heads + rollout")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--accounts", type=int, default=500)
    ap.add_argument("--payments", type=int, default=12000)
    ap.add_argument("--cadence", type=float, default=0.5)
    ap.add_argument("--hist-epochs", type=int, default=2)
    ap.add_argument("--steps", type=int, default=5)
    ap.add_argument("--sample", action="store_true")
    ap.add_argument("--out", default=str(ROOT / "results_gen.json"))
    args = ap.parse_args()

    np.random.seed(0)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if args.smoke:
        args.accounts, args.payments = 60, 900
    pay, _, accs = build_dataset(IndiaConfig(
        num_accounts=args.accounts, num_payments=args.payments, seed=29,
        cadence_frac=args.cadence))
    schema = build_schema(pay, accs)
    print(f"device={device}  payments {len(pay):,}  cadence mix "
          f"{pay['cadence'].value_counts().to_dict()}")

    encoder, vocabs, e_pay, enc_cfg = frozen_embeddings(pay, schema, args.smoke, device)
    e_all = torch.as_tensor(e_pay).to(device)
    quantizer = AdaptiveQuantizer().fit(pay["IntrBkSttlmAmt"].to_numpy(),
                                        pay["Ccy"].to_numpy())

    seqs, lab = windowed_examples(pay)
    actors = sorted(set(lab["actor"]))
    rng = np.random.default_rng(0); rng.shuffle(actors)
    ev_actors = set(actors[: max(1, len(actors) // 5)])
    ev_m = lab["actor"].isin(ev_actors).to_numpy(); tr_m = ~ev_m
    print(f"examples {len(lab):,} (train {tr_m.sum():,} / held-out {ev_m.sum():,})")

    recon_fields = {"Ccy": vocabs.core_size("Ccy"),
                    "identifier_type": vocabs.core_size("identifier_type")}
    full = vocabs.encode(pay)
    targets_all = {n: full["core"][n] for n in recon_fields}
    tr_seqs = [s for s, k in zip(seqs, tr_m) if k]
    hist, _ = small_history_encoder(e_all, recon_fields, targets_all, tr_seqs, device,
                                    epochs=args.hist_epochs)
    H = encode_histories(hist, e_all, seqs, device).cpu().numpy()

    m, y, naive, _ = example_fields(pay, seqs, lab, quantizer)
    tr, ev = tr_m[m], ev_m[m]
    Hm = H[m]
    heads = fit_field_heads(Hm[tr], {f: v[tr] for f, v in y.items()})

    res = {"mode": "smoke" if args.smoke else "full", "n_examples": int(m.sum()),
           "fields": {}}
    wins = 0
    print("\nC8 next-field prediction (held-out actors):")
    for f in ("gap", "amount", "rail"):
        acc = float((heads[f].predict(Hm[ev]) == y[f][ev]).mean())
        nacc = float((naive[f][ev] == y[f][ev]).mean())
        wins += acc > nacc
        res["fields"][f] = {"model_acc": acc, "naive_acc": nacc}
        print(f"  {f:7} model {acc:.3f} | naive {nacc:.3f} "
              f"{'WIN' if acc > nacc else 'loss'}")
    res["C8_wins"] = int(wins)
    res["C8_pass"] = bool(wins >= 2)
    print(f"  -> C8 {'PASS' if res['C8_pass'] else 'FAIL'} ({wins}/3 fields beat naive)")

    # rollout demo: 3 held-out actors with the longest histories
    demo_actors = (lab[ev_m].groupby("actor")["n_history"].max()
                   .sort_values(ascending=False).head(3).index.tolist())
    res["rollouts"] = {}
    for a in demo_actors:
        roll = simulate(pay[pay["DbtrAcct_Id"] == a], encoder, vocabs, hist, heads,
                        quantizer, steps=args.steps, device=device, sample=args.sample)
        res["rollouts"][a] = roll.to_dict(orient="records")
        print(f"\n  rollout {a}:")
        print(roll.to_string(index=False))
    Path(args.out).write_text(json.dumps(res, indent=2, default=str))
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
