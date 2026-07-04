"""
Windowed next-event examples - the forecasting harness.

For each entity (default: debtor account = payments; use the creditor column for receipts),
walk its time-ordered history and emit one example per cutoff: features = the history UP TO
the cutoff event, labels = what happens next (gap to the next event, its amount, its payee,
and whether it arrives within the horizon). A censored negative is added after the last event
when the data window provably extends past the horizon. Split is BY ACTOR (held-out accounts),
the repo's convention - it also removes any intra-account cutoff leakage.

Examples are emitted as sequence dicts compatible with data.sequence_assembly.collate /
run_seq.encode_histories, so the frozen v1 embeddings + v2 history encoder consume them as-is.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def seq_from_dates(actor, pos, ts):
    dt = np.diff(ts.astype("datetime64[D]").astype(np.int64), prepend=ts[0].astype(
        "datetime64[D]").astype(np.int64)).astype(np.float32)
    d = pd.DatetimeIndex(ts)
    return {"actor": actor, "pos": pos.astype(np.int64), "dt": dt,
            "dow": d.dayofweek.to_numpy().astype(np.int64),
            "dom": (d.day - 1).to_numpy().astype(np.int64),
            "month": (d.month - 1).to_numpy().astype(np.int64)}


def windowed_examples(pay: pd.DataFrame, actor_col="DbtrAcct_Id",
                      date_col="IntrBkSttlmDt", amount_col="IntrBkSttlmAmt",
                      payee_col="CdtrAcct_Id", horizon_days=35, min_history=3,
                      max_len=64):
    """Returns (seqs, labels) where seqs[i] is the history-prefix sequence dict and
    labels is a DataFrame: actor, has_next (within horizon), gap_days (NaN if censored),
    next_amount, next_payee, hist_gaps_median, hist_gap_last, hist_amount_median,
    hist_top_payee (the naive-baseline ingredients ride along)."""
    df = pay.reset_index(drop=True)
    dates = pd.to_datetime(df[date_col]).values
    amt = df[amount_col].to_numpy(dtype=float)
    payee = df[payee_col].astype(str).to_numpy()
    data_end = dates.max()
    seqs, rows = [], []

    def _emit(actor, hist_pos, ts_hist, has_next, gap, n_amt, n_payee):
        vals, counts = np.unique(payee[hist_pos], return_counts=True)
        seqs.append(seq_from_dates(actor, hist_pos, ts_hist))
        g = seqs[-1]["dt"][1:]
        rows.append({"actor": actor, "has_next": has_next, "gap_days": gap,
                     "next_amount": n_amt, "next_payee": n_payee,
                     "hist_gap_median": float(np.median(g)) if len(g) else np.nan,
                     "hist_gap_last": float(g[-1]) if len(g) else np.nan,
                     "hist_amount_median": float(np.median(amt[hist_pos])),
                     "hist_top_payee": str(vals[counts.argmax()]),
                     "n_history": len(hist_pos)})

    for actor, idx in df.groupby(actor_col).groups.items():
        pos = np.asarray(idx, dtype=np.int64)
        pos = pos[np.argsort(dates[pos])]
        if len(pos) <= min_history:
            continue
        ts = dates[pos]
        for k in range(min_history, len(pos)):
            hist = pos[max(0, k - max_len):k]
            gap = float((ts[k] - ts[k - 1]) / np.timedelta64(1, "D"))
            _emit(actor, hist, dates[hist], int(gap <= horizon_days), gap,
                  float(amt[pos[k]]), str(payee[pos[k]]))
        # censored negative: nothing arrived within a full horizon after the last event
        if float((data_end - ts[-1]) / np.timedelta64(1, "D")) >= horizon_days:
            hist = pos[-max_len:]
            _emit(actor, hist, dates[hist], 0, np.nan, np.nan, None)
    return seqs, pd.DataFrame(rows)
