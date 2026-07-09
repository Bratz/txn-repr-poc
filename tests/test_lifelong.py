"""C11/C12/C13 machinery: prior-only milestones, entity timeline, momentum knob."""

import numpy as np
import pandas as pd

from data.lifelong import (LIFELONG_COLS, entity_message_timeline, entity_prefix,
                           lifelong_features)


def _toy_pay():
    return pd.DataFrame({
        "payment_id": [0, 1, 2, 3],
        "DbtrAcct_Id": ["A"] * 4,
        "IntrBkSttlmDt": ["2023-01-01", "2023-01-11", "2023-01-21", "2023-01-31"],
        "Dbtr_Ctry": ["IN"] * 4,
        "Cdtr_Ctry": ["IN", "US", "IN", "IN"],       # payment 1 is cross-border
        "cancel_requested": [0, 0, 1, 0],            # payment 2 is recalled
    })


def test_lifelong_prior_only():
    ll = lifelong_features(_toy_pay())
    c = {n: i for i, n in enumerate(LIFELONG_COLS)}
    v = ll.to_numpy()
    assert v[0].sum() == 0                                  # first payment: no history
    # payment 1: has tenure, but its OWN xborder must not count for itself
    assert v[1, c["ll_tenure_days"]] > 0 and v[1, c["ll_had_xborder"]] == 0
    # payment 2: xborder milestone now set (from payment 1); own recall not counted
    assert v[2, c["ll_had_xborder"]] == 1 and v[2, c["ll_had_recall"]] == 0
    # payment 3: recall milestone set (from payment 2), 10 days ago
    assert v[3, c["ll_had_recall"]] == 1
    assert np.isclose(v[3, c["ll_days_since_recall"]], np.log1p(10))


def test_entity_timeline_fractional_dt_and_strictly_prior():
    pay = _toy_pay()
    msg = pd.DataFrame({
        "payment_id": [0, 0, 1, 2],
        "seq": [0, 1, 0, 0],
        "msg_type": ["pacs.008", "pacs.002", "pacs.008", "pacs.008"],
        "t_offset_min": [0.0, 90.0, 0.0, 0.0],       # 90 min apart -> sub-day dt
    })
    m, tl = entity_message_timeline(pay, msg)
    t = tl["A"]
    assert len(t["pos"]) == 4
    seq = entity_prefix(t, payment_id=2, payment_day=float(
        np.datetime64("2023-01-21").astype("datetime64[D]").astype(np.int64)))
    # strictly prior: payment 2's own message excluded -> 3 rows (payments 0, 1)
    assert len(seq["pos"]) == 3
    assert 2 not in set(m["payment_id"].to_numpy()[seq["pos"]])
    # C13: the 90-minute gap survives as a FRACTIONAL day, not rounded to 0-days-int
    assert np.isclose(seq["dt"][1], 90.0 / 1440.0, atol=1e-6)
    # no history -> None
    assert entity_prefix(t, payment_id=0, payment_day=t["t_days"][0]) is None


def test_recall_momentum_plants_clustering():
    from data.synth_india_rails import IndiaConfig, build_dataset

    def repeat_rate(momentum):
        pay, _, _ = build_dataset(IndiaConfig(num_accounts=300, num_payments=15000,
                                              seed=13, recall_momentum=momentum))
        ll = lifelong_features(pay)
        had = ll["ll_had_recall"].to_numpy() == 1
        y = pay["cancel_requested"].to_numpy()
        return y[had].mean(), y[~had].mean()

    on_prior, on_none = repeat_rate(2.0)
    off_prior, off_none = repeat_rate(0.0)
    # ON: accounts with a prior recall cancel MUCH more often than those without
    assert on_prior > 1.5 * on_none
    # OFF: no comparable clustering (tolerance for sampling noise)
    assert off_prior < 1.4 * off_none
    # OFF is deterministic (same seed -> same corpus)
    p1, _, _ = build_dataset(IndiaConfig(num_accounts=50, num_payments=800, seed=3))
    p2, _, _ = build_dataset(IndiaConfig(num_accounts=50, num_payments=800, seed=3))
    assert p1.equals(p2)
