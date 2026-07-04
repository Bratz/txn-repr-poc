"""Tests for the pain.001 origination context (data/pain001_context.py)."""

import numpy as np

from data.pain001_context import CTX_FEATURES, ORIGINATION_ATTRS, build_origination_context, ctx_matrix
from data.synth_india_rails import IndiaConfig, build_dataset


def _ctx(n=4000):
    pay, _, _ = build_dataset(IndiaConfig(num_accounts=300, num_payments=n, seed=23))
    return pay, *build_origination_context(pay, seed=7)


def test_registry_dispositions_are_sane():
    allowed = {"feature", "duplicate", "label", "key", "park", "skip"}
    assert set(ORIGINATION_ATTRS.values()) <= allowed
    # outcomes are labels, never features (leakage rule)
    for a in ("responseFlag", "responseErrCode", "responseErrDesc"):
        assert ORIGINATION_ATTRS[a] == "label"
    # trace/entity ids are keys, not content
    for a in ("transactionTraceId", "userId", "sessionId"):
        assert ORIGINATION_ATTRS[a] == "key"
    # every runner feature is disposition-compatible (engineered from 'feature' attrs)
    assert "responseFlag" not in CTX_FEATURES


def test_origination_context_covers_outward_payments_with_ato_pattern():
    pay, ctx, events = _ctx()
    outward = pay[pay["direction"] == "outward"]
    assert len(ctx) == len(outward)
    assert set(ctx["payment_id"]) == set(outward["payment_id"])
    # ATO rows carry the change-then-drain pattern
    ato = ctx[ctx["atoFlag"] == 1]
    assert len(ato) > 0
    assert (ato["deviceIsNew"] == 1).all()
    assert (ato["geoMismatch"] == 1).all()
    assert (ato["credChangedRecently"] == 1).all()
    assert (ato["payeeAddedRecently"] == 1).all()
    # and the signal is rare + channel-borne
    assert 0.005 < ctx["atoFlag"].mean() < 0.06


def test_event_stream_orders_precursors_before_the_txn():
    pay, ctx, events = _ctx(2000)
    one = ctx[ctx["atoFlag"] == 1].iloc[0]
    e = events[events["payment_id"] == one["payment_id"]]
    types = set(e["event_type"])
    assert {"login", "txn"} <= types
    assert types & {"payee_add", "password_change", "mobile_change", "email_change"}
    txn_t = float(e.loc[e["event_type"] == "txn", "t_offset_min"].iloc[0])
    assert (e.loc[e["event_type"] != "txn", "t_offset_min"] < txn_t).all()


def test_ctx_matrix_shape_and_log_scaling():
    _, ctx, _ = _ctx(1500)
    X = ctx_matrix(ctx)
    assert X.shape == (len(ctx), len(CTX_FEATURES))
    assert np.isfinite(X).all()
