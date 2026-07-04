"""C7 helpers: the derived gap-band column and evenly-spread record selection."""

import numpy as np
import pandas as pd

from run_c7 import GAP_BANDS, add_gap_band
from run_gpu import _recurrence_groups


def test_add_gap_band():
    df = pd.DataFrame({
        "DbtrAcct_Id": ["A", "A", "A", "A", "B", "B"],
        "IntrBkSttlmDt": ["2023-01-01", "2023-01-01", "2023-01-04", "2023-02-15",
                          "2023-03-01", "2023-03-10"],
    })
    out = add_gap_band(df)
    a = out[out["DbtrAcct_Id"] == "A"].sort_values("IntrBkSttlmDt")["gap_band"].tolist()
    assert a == ["first", "0d", "3-7d", ">30d"]          # gaps: 0, 3, 42 days
    b = out[out["DbtrAcct_Id"] == "B"].sort_values("IntrBkSttlmDt")["gap_band"].tolist()
    assert b == ["first", "8-14d"]                       # 9 days
    assert set(out["gap_band"]) <= set(GAP_BANDS)


def test_spread_groups_cover_the_whole_history():
    # 20 events, R=5, take=spread -> first and last included, indices monotone
    df = pd.DataFrame({
        "acct": ["X"] * 20,
        "IntrBkSttlmDt": pd.date_range("2023-01-01", periods=20, freq="3D").astype(str),
        "regime_label": ["Shift"] * 20,
    })
    task = {"group_column": "acct", "label_column": "regime_label", "take": "spread"}
    groups, labels = _recurrence_groups(df, task, R=5)
    assert labels == ["Shift"] and len(groups) == 1
    g = groups[0]
    assert g[0] == 0 and g[-1] == 19 and list(g) == sorted(g)
    # default take="first" is unchanged (v1 recurrence contract)
    gf, _ = _recurrence_groups(df, {**task, "take": "first"}, R=5)
    assert list(gf[0]) == [0, 1, 2, 3, 4]
