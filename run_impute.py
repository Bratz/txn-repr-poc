"""
Predict-at-initiation: what can the TFM tell us from a PARTIAL (pain.001) payment, before it
is enriched into a complete pacs.008?

For each ISO message view (pain.001 -> pacs.008; data/iso_lifecycle) we mask the not-yet-
available fields and, from that frozen representation, evaluate:
  * A  in-flight STP outcome    -> terminal_status  (settle / repair / hold / reject)
  * B  reject/hold reason       -> reject_reason     (ISO status-reason code)
  * C  ETA to settlement        -> time_to_settle_min (regression, log-space)
  * E  cancellation likelihood  -> cancel_requested  (will the originator recall it? camt.056)
  * F  return likelihood        -> returned          (will funds bounce back? pacs.004)
  * G  charges                  -> charges           (fee regression; predictable bps-of-amount)
  * imputation                  -> per-field top-1 reconstruction of the masked fields

fx_rate is deliberately NOT a head - it is market noise (unpredictable from the payment).
All should improve pain.001 -> pacs.008 as the payment enriches. pain.001 rows are ALSO mixed
into pretraining (run_seq.frozen_embeddings extra=), so the encoder has seen sparse initiation
records, not only complete ones.

ponytail: no new heads/training loop - masked reconstruction gives the imputation, and cheap
sklearn probes on the frozen f(x) give A/B/C. Trees on the COMPLETE record still win on these
feature-rule labels; the honest slice is the pain.001 view + one shared backbone.

  python run_impute.py --smoke
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from data.iso_lifecycle import ENCODER_COLS, ENRICH_ADDS, OWNED, UNKNOWN, missing_mask
from run_gpu import _to_device
from run_seq import frozen_embeddings

ROOT = Path(__file__).resolve().parent

# fields blanked in a pain.001 view (present in the complete pacs.008 record).
PAIN001_MISSING = [c for c in ENCODER_COLS if c not in OWNED["pain.001"]]


def pain001_frame(pay: pd.DataFrame) -> pd.DataFrame:
    """The pain.001 (initiation) projection of each payment: blank the clearing-resolved
    fields. Used both to mix partial rows into pretraining and to sanity-check the view."""
    df = pay.copy()
    for col in PAIN001_MISSING:
        if col in df.columns:
            df[col] = UNKNOWN
    return df


def _clf(emb, y, tr, ev):
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score, f1_score
    clf = LogisticRegression(max_iter=1000, class_weight="balanced").fit(emb[tr], y[tr])
    pred = clf.predict(emb[ev])
    return accuracy_score(y[ev], pred), f1_score(y[ev], pred, average="macro")


def _bin(emb, y, tr, ev):
    """Binary likelihood probe -> (PR-AUC, prevalence). Rare labels, so PR-AUC over accuracy;
    prevalence is the no-skill baseline. NaN if a split is single-class (too rare at this scale)."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import average_precision_score
    prev = float(y[ev].mean())
    if len(set(y[tr].tolist())) < 2 or len(set(y[ev].tolist())) < 2:
        return float("nan"), prev
    clf = LogisticRegression(max_iter=1000, class_weight="balanced").fit(emb[tr], y[tr])
    p = clf.predict_proba(emb[ev])[:, 1]
    return float(average_precision_score(y[ev], p)), prev


def _reg_mae(emb, y, tr, ev):
    """Ridge in log1p space; MAE reported back in minutes. Baseline = train log-mean."""
    from sklearn.linear_model import Ridge
    from sklearn.metrics import mean_absolute_error
    yl = np.log1p(np.clip(y, 0, None))
    reg = Ridge().fit(emb[tr], yl[tr])
    pred = np.expm1(reg.predict(emb[ev]))
    base = np.full(len(ev), np.expm1(yl[tr].mean()))
    return (float(mean_absolute_error(y[ev], pred)),
            float(mean_absolute_error(y[ev], base)))


def streaming_outcome(encoder, vocabs, msg, device):
    """D: streaming in-flight tracking. Predict 'will it get booked?' from the first k
    messages of each UETR (mean-pooled frozen embeddings). PR-AUC should rise with k as the
    lifecycle reveals itself (pain.002 RJCT at k=2, submission at k=3). Non-circular: the
    settlement-status messages (pacs.002/camt.054) that state the outcome are never in the
    prefix - max prefix is 3 (pain.001, pain.002, pacs.008).

    ponytail: 2-5 message sequences don't need the Transformer history encoder - a mean-pool
    over the frozen per-message embeddings is enough. Add the sequence encoder if prefixes
    grow long or order matters beyond 'which messages arrived'.
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import average_precision_score
    from run_seq import embed_all_rows

    msg = msg.reset_index(drop=True)
    em = embed_all_rows(encoder, vocabs.encode(msg), len(msg), device).cpu().numpy()
    groups = {}
    for i, (e2e, seq) in enumerate(zip(msg["end_to_end_id"], msg["seq"])):
        groups.setdefault(e2e, []).append((seq, i))
    booked = set(msg.loc[msg["msg_type"] == "camt.054", "end_to_end_id"])
    uetrs = list(groups)
    y = np.array([int(u in booked) for u in uetrs])
    if len(set(y.tolist())) < 2:
        return {"note": "degenerate booked label at this scale"}
    perm = np.random.permutation(len(uetrs)); cut = int(len(uetrs) * 0.8)
    tr, ev = perm[:cut], perm[cut:]
    out = {"prevalence": float(y.mean())}
    print(f"\nD  streaming booked-prediction (prev {y.mean():.2f}, PR-AUC rises as msgs arrive):")
    for k in (1, 2, 3):
        X = np.stack([em[[i for _, i in sorted(g)[:k]]].mean(0) for g in groups.values()])
        clf = LogisticRegression(max_iter=1000, class_weight="balanced").fit(X[tr], y[tr])
        pr = float(average_precision_score(y[ev], clf.predict_proba(X[ev])[:, 1]))
        out[f"prefix_{k}_pr_auc"] = pr
        print(f"  after <= {k} msg  booked PR-AUC {pr:.3f}")
    return out


def main():
    ap = argparse.ArgumentParser(description="predict-at-initiation: STP / reason / ETA + imputation")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--payments", default=str(ROOT / "data" / "india_rails_payments.parquet"))
    ap.add_argument("--messages", default=str(ROOT / "data" / "india_rails_messages.parquet"))
    ap.add_argument("--schema", default=str(ROOT / "data" / "column_schema_india.json"))
    ap.add_argument("--no-mix", action="store_true", help="don't mix pain.001 into pretraining")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--out", default=str(ROOT / "results_impute.json"))
    args = ap.parse_args()

    np.random.seed(0)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    p = Path(args.payments)
    pay = pd.read_parquet(p) if p.suffix == ".parquet" else pd.read_csv(p)
    if args.limit:
        pay = pay.head(args.limit).reset_index(drop=True)
    schema = json.loads(Path(args.schema).read_text())
    print(f"device={device}  mode={'smoke' if args.smoke else 'full'}  payments {len(pay):,}"
          f"  mix_pain001={not args.no_mix}")

    extra = None if args.no_mix else pain001_frame(pay)
    encoder, vocabs, _, _ = frozen_embeddings(pay, schema, args.smoke, device, extra=extra)
    full = _to_device(vocabs.encode(pay), device)
    recon_names = [r[0] for r in encoder.recon]
    # DELIST high-cardinality IDs from imputation: account / party ids (and UTR-like refs) are
    # ASSIGNED or DIRECTORY-LOOKED-UP, not inferable from the payment (measured top-1 ~ 0.00).
    # We still MASK them for the pain.001 view, but only report imputable structured fields.
    hicard = set(schema["buckets"]["high_card_categorical"])
    impute_fields = [r for r in recon_names if r not in hicard]
    B = len(pay)
    idx = np.random.permutation(B); cut = int(B * 0.8); tr, ev = idx[:cut], idx[cut:]

    y_status = pay["terminal_status"].to_numpy()
    y_reason = pay["reject_reason"].to_numpy()
    y_eta = pay["time_to_settle_min"].to_numpy(dtype=float)
    y_cancel = pay["cancel_requested"].to_numpy()
    y_return = pay["returned"].to_numpy()
    y_charges = pay["charges"].to_numpy(dtype=float) if "charges" in pay.columns else None

    imp, rows = {}, {}
    for si, (sname, _) in enumerate(ENRICH_ADDS):
        mask_row = missing_mask(recon_names, si)
        cm = (torch.tensor(mask_row, device=device).unsqueeze(0).expand(B, -1)
              if any(mask_row) else None)
        if cm is not None:
            for f, a in encoder.reconstruction_accuracy(full, cm).items():
                if f in impute_fields:                          # skip delisted high-card IDs
                    imp.setdefault(f, {})[sname] = a["top1"]
        with torch.no_grad():
            emb = (encoder.forward(full, cm)[0] if cm is not None
                   else encoder.encode(full)).cpu().numpy()
        a_acc, a_f1 = _clf(emb, y_status, tr, ev)                       # A
        b_acc, b_f1 = _clf(emb, y_reason, tr, ev)                       # B
        c_mae, c_base = _reg_mae(emb, y_eta, tr, ev)                    # C
        e_pr, e_prev = _bin(emb, y_cancel, tr, ev)                      # E cancellation
        f_pr, f_prev = _bin(emb, y_return, tr, ev)                      # F return
        rows[sname] = {"stp_acc": a_acc, "stp_f1": a_f1,
                       "reason_acc": b_acc, "reason_f1": b_f1,
                       "eta_mae_min": c_mae, "eta_baseline_mae_min": c_base,
                       "cancel_pr_auc": e_pr, "cancel_prevalence": e_prev,
                       "return_pr_auc": f_pr, "return_prevalence": f_prev}
        print(f"[{sname:9s}] A STP acc {a_acc:.3f}/F1 {a_f1:.3f}   "
              f"B reason acc {b_acc:.3f}/F1 {b_f1:.3f}   "
              f"C ETA MAE {c_mae:.1f} min (base {c_base:.1f})")
        print(f"{'':11s} E cancel PR-AUC {e_pr:.3f} (prev {e_prev:.3f})   "
              f"F return PR-AUC {f_pr:.3f} (prev {f_prev:.3f})")
        if y_charges is not None:
            g_mae, g_base = _reg_mae(emb, y_charges, tr, ev)           # G charges regression
            rows[sname]["charges_mae"] = g_mae
            rows[sname]["charges_baseline_mae"] = g_base
            print(f"{'':11s} G charges MAE {g_mae:.1f} (base {g_base:.1f})")

    print("\nimputation top-1  (structured fields only; high-card IDs delisted = lookup/assigned):")
    cols = [n for n, _ in ENRICH_ADDS]
    print(f"  {'field':18s} " + " ".join(f"{c[:9]:>9s}" for c in cols))
    for f in impute_fields:
        cells = [f"{imp[f][c]:.2f}" if f in imp and c in imp[f] else "-" for c in cols]
        print(f"  {f:18s} " + " ".join(f"{c:>9s}" for c in cells))
    print(f"  (delisted, not imputed: {sorted(hicard)})")

    stream = None
    mp = Path(args.messages)
    if mp.exists():
        msg = pd.read_parquet(mp) if mp.suffix == ".parquet" else pd.read_csv(mp)
        if args.limit:                                   # keep only messages for kept payments
            msg = msg[msg["payment_id"].isin(pay["payment_id"])].reset_index(drop=True)
        stream = streaming_outcome(encoder, vocabs, msg, device)

    Path(args.out).write_text(json.dumps({"predictions_by_view": rows,
                                          "imputation_top1": imp,
                                          "streaming_outcome": stream}, indent=2))
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
