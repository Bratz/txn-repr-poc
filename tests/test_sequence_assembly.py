"""Unit tests for v2 per-entity sequence assembly (data/sequence_assembly.py)."""

import numpy as np
import pandas as pd

from data.sequence_assembly import (
    assemble_sequences, collate, split_by_actor, velocity_labels,
)


def test_velocity_labels_flag_bursts_only():
    steady = {"dt": np.array([0, 10, 10, 10, 10, 10, 10, 10], np.float32)}   # constant cadence
    burst = {"dt": np.array([0, 10, 10, 10, 10, 1, 1, 1], np.float32)}        # recent speed-up
    short = {"dt": np.array([0, 1, 1], np.float32)}                           # too short -> 0
    assert velocity_labels([steady, burst, short]).tolist() == [0, 1, 0]


def test_assemble_orders_and_derives_time(sample_df):
    seqs = assemble_sequences(sample_df, actor_col="DbtrAcct_Id", min_len=2)
    assert seqs, "the sample should contain at least one multi-event debtor"
    dates = pd.to_datetime(sample_df.reset_index(drop=True)["IntrBkSttlmDt"]).values
    for s in seqs:
        assert len(s["pos"]) >= 2
        d = dates[s["pos"]]
        assert (np.diff(d) >= np.timedelta64(0, "ns")).all()   # chronological
        assert s["dt"][0] == 0 and (s["dt"] >= 0).all()
        assert (s["dow"] < 7).all() and (s["dom"] < 31).all() and (s["month"] < 12).all()


def test_max_len_keeps_most_recent():
    n = 10
    df = pd.DataFrame({
        "DbtrAcct_Id": ["A"] * n,
        "IntrBkSttlmDt": pd.date_range("2023-01-01", periods=n, freq="D").astype(str),
    })
    seqs = assemble_sequences(df, max_len=4, min_len=2)
    assert len(seqs) == 1 and len(seqs[0]["pos"]) == 4
    # the kept rows are the last four (positions 6..9)
    assert list(seqs[0]["pos"]) == [6, 7, 8, 9]


def test_split_by_actor_is_disjoint(sample_df):
    seqs = assemble_sequences(sample_df, min_len=2)
    train, ev = split_by_actor(seqs, frac_eval=0.3, seed=1)
    a_tr = {s["actor"] for s in train}
    a_ev = {s["actor"] for s in ev}
    assert a_tr and a_ev
    assert a_tr.isdisjoint(a_ev)


def test_collate_pads_and_masks():
    seqs = [
        {"pos": np.array([3, 4, 5]), "dt": np.array([0., 1., 2.], dtype=np.float32),
         "dow": np.zeros(3, np.int64), "dom": np.zeros(3, np.int64), "month": np.zeros(3, np.int64)},
        {"pos": np.array([7]), "dt": np.array([0.], dtype=np.float32),
         "dow": np.zeros(1, np.int64), "dom": np.zeros(1, np.int64), "month": np.zeros(1, np.int64)},
    ]
    b = collate(seqs)
    assert b["pos"].shape == (2, 3)
    assert b["pad_mask"][0].tolist() == [False, False, False]
    assert b["pad_mask"][1].tolist() == [False, True, True]
    assert b["pos"][1].tolist() == [7, 0, 0]            # padded slots point to row 0
