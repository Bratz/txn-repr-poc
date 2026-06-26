"""Tests for the v2 §7 behavioural generator (data/synth_sequences.py)."""

import pandas as pd

from data.synth_sequences import V2Config, build_schema, build_v2_dataset


def test_emits_regime_and_pacs008_columns():
    df, _ = build_v2_dataset(V2Config(num_accounts=120, seed=3))
    for col in ("DbtrAcct_Id", "CdtrAcct_Id", "IntrBkSttlmAmt", "IntrBkSttlmDt",
                "risk_label", "regime_label"):
        assert col in df.columns
    assert set(df["regime_label"]) == {"Stable", "Shift"}
    # the regime label is constant within an account
    assert (df.groupby("DbtrAcct_Id")["regime_label"].nunique() == 1).all()


def test_aggregates_matched_signal_is_order_only():
    # the whole point: order-invariant summaries must NOT separate the classes,
    # so a CatBoost on aggregates (C4) and a pooled embedding (C3) are weak.
    df, _ = build_v2_dataset(V2Config(num_accounts=400, seed=11))
    rows = []
    for _, sub in df.groupby("DbtrAcct_Id"):
        sub = sub.sort_values("IntrBkSttlmDt")
        d = pd.to_datetime(sub["IntrBkSttlmDt"]).diff().dt.days.dropna().to_numpy()
        rows.append((sub["regime_label"].iloc[0], len(sub),
                     float(d.std()) if len(d) > 1 else 0.0))
    m = pd.DataFrame(rows, columns=["cls", "n", "iat_std"]).groupby("cls").mean()
    assert abs(m.loc["Stable", "iat_std"] - m.loc["Shift", "iat_std"]) < 5.0
    assert abs(m.loc["Stable", "n"] - m.loc["Shift", "n"]) < 3.0


def test_schema_exposes_entity_task():
    df, accs = build_v2_dataset(V2Config(num_accounts=80, seed=5))
    s = build_schema(df, accs)
    assert s["entity_task"]["label_column"] == "regime_label"
    assert s["entity_task"]["positive_class"] == "Shift"
    assert s["entity_task"]["actor"] == "DbtrAcct_Id"
    assert set(s["buckets"]) == {"high_card_categorical", "numerical", "core", "meta_party"}
