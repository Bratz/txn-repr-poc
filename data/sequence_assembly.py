"""
v2 / sequence assembly - turn flat transaction rows into per-entity ordered histories.

Beyond arXiv:2410.07851; see docs/V2_DIRECTION.md. The actor (the entity whose history we
model) defaults to the debtor account: its history is its outgoing payments, time-ordered.
This module groups rows by actor, orders them by settlement date, derives the inter-arrival
gap and calendar features each event needs, caps overly long histories to the most recent N
(PRAGMA keeps the most recent events), and provides a held-out-BY-ACTOR split and a collate
that pads variable-length sequences into tensors.

The held-out-by-actor split is the eval the paper's v1 walked back; it is legitimate here and
is the regime where a learned representation is expected to beat a per-id tree (the tree
memorizes account statistics that do not transfer to unseen accounts).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import torch


def assemble_sequences(df, actor_col: str = "DbtrAcct_Id",
                       date_col: str = "IntrBkSttlmDt",
                       max_len: int = 256, min_len: int = 2) -> list[dict]:
    """Group rows into per-actor, time-ordered event sequences.

    Each sequence dict holds: actor, pos (row positions into the reset-index df), dt
    (days since previous event), and dow/dom/month (0-indexed calendar features). Sequences
    shorter than min_len are dropped; longer than max_len keep the most recent max_len.
    """
    df = df.reset_index(drop=True)
    dates = pd.to_datetime(df[date_col])
    seqs: list[dict] = []
    for actor, idx in df.groupby(actor_col).groups.items():
        pos = np.asarray(idx, dtype=np.int64)
        pos = pos[np.argsort(dates.values[pos])]          # chronological
        if len(pos) > max_len:
            pos = pos[-max_len:]
        if len(pos) < min_len:
            continue
        ts = pd.to_datetime(pd.Series(dates.values[pos]))
        dt = ts.diff().dt.days.fillna(0).clip(lower=0).to_numpy().astype(np.float32)
        seqs.append({
            "actor": actor,
            "pos": pos,
            "dt": dt,
            "dow": ts.dt.dayofweek.to_numpy().astype(np.int64),       # 0..6
            "dom": (ts.dt.day - 1).to_numpy().astype(np.int64),        # 0..30
            "month": (ts.dt.month - 1).to_numpy().astype(np.int64),    # 0..11
        })
    return seqs


def split_by_actor(seqs: list[dict], frac_eval: float = 0.2, seed: int = 0):
    """Disjoint train/eval split on the ACTOR set - eval actors are unseen in training."""
    actors = sorted({s["actor"] for s in seqs})
    rng = np.random.default_rng(seed)
    rng.shuffle(actors)
    n_eval = max(1, int(len(actors) * frac_eval))
    eval_actors = set(actors[:n_eval])
    train = [s for s in seqs if s["actor"] not in eval_actors]
    ev = [s for s in seqs if s["actor"] in eval_actors]
    return train, ev


def velocity_labels(seqs: list[dict], k: int = 3, ratio: float = 0.5,
                    min_events: int = 6) -> np.ndarray:
    """1 if an actor's recent inter-arrivals BURST: mean of the last k gaps <= ratio x the
    median of the earlier gaps (i.e. >= 2x faster). A pure timing signal — order-blind pooled
    features can't see it, the time-aware history encoder can.
    ponytail: transparent rule over s['dt'] (no new data); short histories -> 0 (no evidence).
    """
    out = []
    for s in seqs:
        g = s["dt"][1:]                       # inter-arrival gaps (dt[0] is the leading 0)
        if len(g) < min_events:
            out.append(0); continue
        recent, base = float(np.mean(g[-k:])), float(np.median(g[:-k]))
        out.append(int(base > 0 and recent <= ratio * base))
    return np.asarray(out)


def collate(batch_seqs: list[dict]) -> dict:
    """Pad a list of sequences to the batch max length; build the pad mask (True = pad)."""
    B = len(batch_seqs)
    L = max(len(s["pos"]) for s in batch_seqs)
    pos = np.zeros((B, L), np.int64)
    dt = np.zeros((B, L), np.float32)
    dow = np.zeros((B, L), np.int64)
    dom = np.zeros((B, L), np.int64)
    month = np.zeros((B, L), np.int64)
    pad = np.ones((B, L), dtype=bool)
    for i, s in enumerate(batch_seqs):
        n = len(s["pos"])
        pos[i, :n] = s["pos"]
        dt[i, :n] = s["dt"]
        dow[i, :n] = s["dow"]
        dom[i, :n] = s["dom"]
        month[i, :n] = s["month"]
        pad[i, :n] = False
    return {
        "pos": torch.from_numpy(pos),
        "dt": torch.from_numpy(dt),
        "dow": torch.from_numpy(dow),
        "dom": torch.from_numpy(dom),
        "month": torch.from_numpy(month),
        "pad_mask": torch.from_numpy(pad),
    }
