"""Tests for the forecasting harness (data/next_event.py) + the cadence generator."""

import numpy as np
import pandas as pd

from data.next_event import windowed_examples
from data.synth_india_rails import IndiaConfig, build_dataset


def _mini():
    # one actor, 6 monthly events to fixed payee; one actor with 2 events (below min history)
    rows = []
    for i, d in enumerate(pd.date_range("2023-01-01", periods=6, freq="30D")):
        rows.append({"DbtrAcct_Id": "A", "CdtrAcct_Id": "P1",
                     "IntrBkSttlmDt": d.date().isoformat(), "IntrBkSttlmAmt": 100.0 + i})
    rows.append({"DbtrAcct_Id": "B", "CdtrAcct_Id": "P2",
                 "IntrBkSttlmDt": "2023-01-01", "IntrBkSttlmAmt": 5.0})
    rows.append({"DbtrAcct_Id": "B", "CdtrAcct_Id": "P2",
                 "IntrBkSttlmDt": "2023-02-01", "IntrBkSttlmAmt": 5.0})
    return pd.DataFrame(rows)


def test_windowed_examples_labels_and_censoring():
    seqs, lab = windowed_examples(_mini(), horizon_days=35, min_history=3)
    a = lab[lab.actor == "A"]
    assert len(a) == 3                                   # cutoffs k=3,4,5 (tail < horizon)
    assert (a["gap_days"] == 30).all() and (a["has_next"] == 1).all()
    assert (a["next_payee"] == "P1").all()
    assert "B" not in set(lab["actor"])                  # below min history
    # sequences align with labels and carry timing features
    assert len(seqs) == len(lab)
    assert seqs[0]["dt"][1:].tolist() == [30.0, 30.0]    # first example: 3-event history


def test_windowed_censored_negative_when_data_extends_past_horizon():
    df = _mini()
    # add a lone late event elsewhere so data_end >> actor A's last event
    df = pd.concat([df, pd.DataFrame([{"DbtrAcct_Id": "C", "CdtrAcct_Id": "P3",
                                       "IntrBkSttlmDt": "2024-01-01",
                                       "IntrBkSttlmAmt": 1.0}])], ignore_index=True)
    _, lab = windowed_examples(df, horizon_days=35, min_history=3)
    a = lab[lab.actor == "A"]
    assert (a["has_next"] == 0).sum() == 1               # censored tail negative appended
    assert a.iloc[-1]["n_history"] == 6


def test_cadence_generator_periodic_and_default_off():
    p0, _, _ = build_dataset(IndiaConfig(num_accounts=150, num_payments=1200, seed=7))
    assert (p0["cadence"] == "none").all()               # default OFF -> constant column
    pay, _, _ = build_dataset(IndiaConfig(num_accounts=300, num_payments=6000, seed=23,
                                          cadence_frac=0.5))
    assert {"salary", "rent", "utility", "supplier", "none"} <= set(pay["cadence"])
    sal = pay[pay.cadence == "salary"]
    g = sal.groupby("CdtrAcct_Id")
    one = g.get_group(g.size().idxmax()).sort_values("IntrBkSttlmDt")
    gaps = pd.to_datetime(one["IntrBkSttlmDt"]).diff().dt.days.dropna()
    assert (gaps % 30 == 0).all()                        # monthly (missing months = 60/90)
    cv = one["IntrBkSttlmAmt"].std() / one["IntrBkSttlmAmt"].mean()
    assert cv < 0.05                                     # salary amount stability
