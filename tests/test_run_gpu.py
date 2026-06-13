"""Tests for run_gpu orchestration helpers (split, json sanitize)."""

import numpy as np

from run_gpu import _sanitize, split


def test_stratified_split_keeps_rare_positive_in_eval(sample_df):
    # the committed sample has some High rows; a stratified eval must include them.
    # eval is capped at 20% of the data, so 80 < 100 is used verbatim.
    train_df, eval_df = split(sample_df, eval_rows=80, label_col="risk_label")
    assert len(eval_df) == 80
    assert len(train_df) == len(sample_df) - 80
    assert set(eval_df["risk_label"]) == set(sample_df["risk_label"])  # all classes present
    # no row overlap between train and eval
    assert not (set(eval_df.index) & set(train_df.index))


def test_sanitize_replaces_non_finite():
    obj = {"a": float("nan"), "b": float("inf"), "c": 0.5,
           "d": {"e": -float("inf")}, "f": [1.0, float("nan")]}
    out = _sanitize(obj)
    assert out["a"] is None and out["b"] is None
    assert out["c"] == 0.5
    assert out["d"]["e"] is None
    assert out["f"] == [1.0, None]


def test_sanitize_leaves_finite_untouched():
    obj = {"x": 1, "y": 2.5, "z": "ok", "w": [1, 2, 3]}
    assert _sanitize(obj) == obj
