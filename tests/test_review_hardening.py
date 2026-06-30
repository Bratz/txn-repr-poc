"""Tests for the post-review hardening fixes (see tasks/todo.md)."""

import numpy as np
import pandas as pd

from encoders.coerce import canon_categorical
from encoders.column_assembler import build_vocabs
from encoders.party_encoder import (
    PARTY_STRUCT_ATTRS, build_field_vocabs, encode_role_parties,
)


# --- #3 canonical coercion: int / float / str forms collapse to one key --------- #

def test_canon_categorical_collapses_dtypes():
    assert list(canon_categorical(pd.Series([123, 789]))) == ["123", "789"]
    assert list(canon_categorical(pd.Series([123.0, 789.0]))) == ["123", "789"]   # no ".0"
    assert list(canon_categorical(pd.Series(["123", "789"]))) == ["123", "789"]
    assert canon_categorical(pd.Series([1e8]))[0] == "100000000"                  # no sci
    assert canon_categorical(pd.Series([123.5]))[0] == "123.5"                    # kept
    assert canon_categorical(pd.Series(["007"]))[0] == "007"                      # leading zero


def test_vocab_build_and_encode_dtype_invariant():
    schema = {"buckets": {"high_card_categorical": ["cust_id"], "numerical": ["amt"],
                          "core": ["Ccy"]}}
    train = pd.DataFrame({"cust_id": [123456, 789012], "amt": [1.0, 2.0], "Ccy": ["INR", "INR"]})
    v = build_vocabs(train, schema)
    # serve the SAME ids but as float (the dtype-drift hazard) -> must hit, not UNK
    serve = pd.DataFrame({"cust_id": [123456.0, 789012.0], "amt": [1.0, 1.0],
                          "Ccy": ["INR", "INR"]})
    enc = v.encode(serve)
    unk = v.hc_size("cust_id") - 1
    assert enc["high_card"]["cust_id"].tolist() == [0, 1]      # both resolved, neither UNK
    assert unk not in enc["high_card"]["cust_id"].tolist()


# --- P0-1 party encoder: unseen attribute -> MASK row, not NaN/out-of-bounds ----- #

def test_party_encoder_unseen_attribute_maps_to_mask_not_crash():
    role = {"key": "acct", "attrs": {a: a for a in PARTY_STRUCT_ATTRS}}
    train = pd.DataFrame({"acct": ["A", "B"], **{a: ["seen", "seen2"] for a in PARTY_STRUCT_ATTRS}})
    vocabs = build_field_vocabs(train, {"r": role})
    serve = pd.DataFrame({"acct": ["C"], **{a: ["NEVER_SEEN"] for a in PARTY_STRUCT_ATTRS}})
    keys, ids = encode_role_parties(serve, role, vocabs)       # must not raise
    assert keys == ["C"]
    for j, a in enumerate(PARTY_STRUCT_ATTRS):
        assert int(ids[0, j]) == len(vocabs[a])                # MASK row (graceful UNK)
    assert ids.dtype.is_floating_point is False                # no NaN-via-float corruption


# --- P0-2 mis-routed flag + per-class routing metrics ---------------------------- #

def test_is_mis_routed_flags_injected_gate_rows():
    from data.synth_india_rails import IndiaConfig, build_dataset
    pay, _, _ = build_dataset(IndiaConfig(num_accounts=300, num_payments=4000, seed=23))
    assert "is_mis_routed" in pay.columns
    # over-cap UPI and below-min RTGS attempts must be flagged mis-routed...
    over = pay[(pay.rail == "UPI") & (pay.IntrBkSttlmAmt > 100_000)]
    assert len(over) and (over["is_mis_routed"] == 1).all()
    # ...while a normal small UPI payment is not
    ok = pay[(pay.rail == "UPI") & (pay.IntrBkSttlmAmt < 50_000)]
    assert len(ok) and (ok["is_mis_routed"] == 0).all()


def test_rail_routing_reports_per_class_and_domestic():
    from data.synth_india_rails import IndiaConfig, build_dataset, build_schema
    from run_india import _visible_features, intake_eval
    pay, _, accs = build_dataset(IndiaConfig(num_accounts=300, num_payments=3000, seed=23))
    schema = build_schema(pay, accs)
    rng = np.random.default_rng(0)
    e = np.hstack([_visible_features(pay), rng.normal(size=(len(pay), 8))])
    idx = rng.permutation(len(pay)); cut = int(len(idx) * 0.8)
    rr = intake_eval(e, pay, schema, idx[:cut], idx[cut:])["rail_routing"]
    assert {"per_class", "domestic_probe_accuracy", "clean_probe_accuracy"} <= set(rr)
    assert "SWIFT" in rr["per_class"] and "RTGS" in rr["per_class"]
