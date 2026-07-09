"""Life-long milestones (C12) + the per-account entity message timeline (C11/C13).

PRAGMA's input contract, ported: milestones are FIRST OCCURRENCES with timestamps,
read as time-since at the evaluation point; the entity timeline is one time-ordered
stream of everything the bank has seen for an account, across payments.

No hindcasting anywhere: a payment's features use only rows STRICTLY BEFORE it in
(date, payment_id) order; an entity history for payment P contains only messages of
P's prior payments. Assumption (documented): a prior payment's outcome (recall,
return) is settled knowledge by the next payment's date - lifecycle legs land
minutes-to-hours after their payment, and consecutive payments are days apart.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

LIFELONG_COLS = ["ll_tenure_days", "ll_n_prior", "ll_days_since_xborder",
                 "ll_days_since_recall", "ll_had_xborder", "ll_had_recall"]


def lifelong_features(pay: pd.DataFrame, actor_col="DbtrAcct_Id",
                      date_col="IntrBkSttlmDt") -> pd.DataFrame:
    """Per-payment milestone features from the actor's PRIOR payments only.

    Returns a DataFrame aligned to pay's index: log1p'd tenure / days-since-first-
    crossborder / days-since-first-recall (0 where the milestone never happened,
    with had_* flags carrying the distinction), plus prior-event count."""
    df = pay.reset_index(drop=True)
    dates = pd.to_datetime(df[date_col]).values.astype("datetime64[D]").astype(np.int64)
    xb = (df["Dbtr_Ctry"].astype(str) != df["Cdtr_Ctry"].astype(str)).to_numpy()
    rc = df["cancel_requested"].to_numpy() if "cancel_requested" in df.columns \
        else np.zeros(len(df), dtype=int)
    out = np.zeros((len(df), len(LIFELONG_COLS)), dtype=np.float32)
    for _, idx in df.groupby(actor_col).groups.items():
        pos = np.asarray(idx, dtype=np.int64)
        pos = pos[np.lexsort((pos, dates[pos]))]           # (date, pid) order
        first_seen = first_xb = first_rc = None
        for k, p in enumerate(pos):
            d = dates[p]
            out[p] = [np.log1p(d - first_seen) if first_seen is not None else 0.0,
                      np.log1p(k),
                      np.log1p(d - first_xb) if first_xb is not None else 0.0,
                      np.log1p(d - first_rc) if first_rc is not None else 0.0,
                      float(first_xb is not None), float(first_rc is not None)]
            if first_seen is None:
                first_seen = d
            if xb[p] and first_xb is None:
                first_xb = d
            if rc[p] and first_rc is None:
                first_rc = d
    return pd.DataFrame(out, columns=LIFELONG_COLS)


def entity_message_timeline(pay: pd.DataFrame, msg: pd.DataFrame,
                            actor_col="DbtrAcct_Id", date_col="IntrBkSttlmDt",
                            max_len=64):
    """Per-account, time-ordered stream of ALL its (visible) messages across payments.

    Time is FRACTIONAL DAYS (C13): payment date + t_offset_min/1440, so intra-day
    lifecycle spacing survives into dt. Returns (msg_sorted, timelines) where
    timelines maps actor -> dict(pos, t_days, payment_id) with pos indexing
    msg_sorted rows; entity_prefix() slices it per evaluation payment."""
    if "msg_direction" in msg.columns:                     # the engine's information set
        msg = msg[msg["msg_direction"].notna()]
    m = msg.reset_index(drop=True).copy()
    pdate = dict(zip(pay["payment_id"],
                     pd.to_datetime(pay[date_col]).values.astype("datetime64[D]")
                     .astype(np.int64).astype(float)))
    pactor = dict(zip(pay["payment_id"], pay[actor_col].astype(str)))
    m["_t_days"] = (m["payment_id"].map(pdate)
                    + m["t_offset_min"].to_numpy(dtype=float) / 1440.0)
    m["_actor"] = m["payment_id"].map(pactor)
    m = m.sort_values(["_actor", "_t_days", "payment_id", "seq"]).reset_index(drop=True)
    timelines = {}
    for actor, idx in m.groupby("_actor").groups.items():
        pos = np.asarray(idx, dtype=np.int64)
        timelines[actor] = {"pos": pos,
                            "t_days": m["_t_days"].to_numpy()[pos],
                            "payment_id": m["payment_id"].to_numpy()[pos]}
    return m, timelines


def entity_prefix(timeline: dict, payment_id: int, payment_day: float,
                  max_len=64) -> dict | None:
    """Sequence dict of the actor's messages STRICTLY BEFORE this payment's date
    (prior payments only - the current payment's own legs are excluded even when
    same-day, via the payment_id guard). None if there is no prior history."""
    keep = (timeline["t_days"] < payment_day) & (timeline["payment_id"] != payment_id)
    pos = timeline["pos"][keep][-max_len:]
    if not len(pos):
        return None
    t = timeline["t_days"][keep][-max_len:]
    dt = np.diff(t, prepend=t[:1]).astype(np.float32)      # fractional days (C13)
    z = np.zeros(len(pos), np.int64)
    d = pd.to_datetime(np.asarray(t, dtype="int64"), unit="D")  # calendar from day part
    return {"actor": f"e{payment_id}", "pos": pos.astype(np.int64), "dt": dt,
            "dow": d.dayofweek.to_numpy().astype(np.int64),
            "dom": (d.day - 1).to_numpy().astype(np.int64),
            "month": (d.month - 1).to_numpy().astype(np.int64)}
