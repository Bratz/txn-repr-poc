"""Tests for the India multi-rail generator (data/rails.py + data/synth_india_rails.py)."""

import numpy as np

from data.rails import (
    RAILS, RAIL_NAMES, below_min, choose_rail, eligible_rails, sample_identifier,
    settle_seconds, violates_cap,
)
from data.synth_india_rails import (
    EXCEPTION_CODES, IndiaConfig, WORKFLOW, build_dataset, build_schema,
)


# --------------------------------------------------------------------------- #
# rails.py - eligibility / routing logic
# --------------------------------------------------------------------------- #

def test_eligibility_respects_cap_and_min():
    assert set(eligible_rails(50_000)) == {"UPI", "IMPS", "NEFT"}      # < RTGS min
    assert set(eligible_rails(100_000)) == {"UPI", "IMPS", "NEFT"}     # == UPI cap (ok)
    assert set(eligible_rails(150_000)) == {"IMPS", "NEFT"}            # > UPI cap
    assert set(eligible_rails(300_000)) == {"RTGS", "IMPS", "NEFT"}    # >= RTGS min
    assert set(eligible_rails(600_000)) == {"RTGS", "NEFT"}           # > IMPS cap


def test_eligibility_respects_identifier():
    assert eligible_rails(50_000, "VPA") == ["UPI"]                   # VPA => UPI only
    assert eligible_rails(150_000, "VPA") == []                       # over UPI cap, no rail
    assert eligible_rails(50_000, "MMID_MOBILE") == ["IMPS"]          # MMID => IMPS only
    assert set(eligible_rails(50_000, "ACCT_IFSC")) == {"IMPS", "NEFT"}


def test_choose_rail_returns_eligible():
    rng = np.random.default_rng(0)
    for amt in (500, 5_000, 80_000, 250_000, 900_000):
        for _ in range(50):
            rail = choose_rail(amt, rng)
            assert rail in eligible_rails(amt) or rail == "NEFT"


def test_choose_rail_band_skew():
    rng = np.random.default_rng(1)
    low = [choose_rail(2_000, rng) for _ in range(400)]
    high = [choose_rail(800_000, rng) for _ in range(400)]
    assert low.count("UPI") > low.count("RTGS")                       # low value -> UPI
    assert high.count("RTGS") > high.count("UPI")                     # high value -> RTGS


def test_sample_identifier_consistent_with_rail():
    rng = np.random.default_rng(2)
    assert all(sample_identifier("UPI", rng) == "VPA" for _ in range(50))
    assert all(sample_identifier("RTGS", rng) == "ACCT_IFSC" for _ in range(50))
    assert all(sample_identifier("SWIFT", rng) == "BIC_IBAN" for _ in range(50))
    imps = {sample_identifier("IMPS", rng) for _ in range(200)}
    assert imps <= {"ACCT_IFSC", "MMID_MOBILE"} and "MMID_MOBILE" in imps


def test_swift_only_crossborder():
    assert eligible_rails(1_000_000, xborder=True) == ["SWIFT"]       # xborder => SWIFT only
    assert "SWIFT" not in eligible_rails(1_000_000)                   # domestic never SWIFT
    assert "SWIFT" not in eligible_rails(50_000, xborder=False)
    assert eligible_rails(50_000, "BIC_IBAN", xborder=True) == ["SWIFT"]
    rng = np.random.default_rng(5)
    assert all(choose_rail(a, rng, xborder=True) == "SWIFT" for a in (5_000, 5_000_000))


def test_cap_and_min_helpers():
    assert violates_cap("UPI", 150_000) and not violates_cap("UPI", 100_000)
    assert violates_cap("IMPS", 600_000) and not violates_cap("NEFT", 10_000_000)
    assert below_min("RTGS", 100_000) and not below_min("RTGS", 200_000)
    assert not below_min("UPI", 1)


def test_settle_seconds_neft_slowest():
    rng = np.random.default_rng(3)
    upi = np.mean([settle_seconds("UPI", rng) for _ in range(300)])
    neft = np.mean([settle_seconds("NEFT", rng) for _ in range(300)])
    assert neft > upi                                                 # batch wait dominates
    assert RAILS["UPI"].sla_lo <= upi <= RAILS["UPI"].sla_hi + 1


# --------------------------------------------------------------------------- #
# synth_india_rails.py - dataset
# --------------------------------------------------------------------------- #

def _small():
    return build_dataset(IndiaConfig(num_accounts=400, num_payments=4000, seed=23))


def test_emits_both_tables_with_rail_columns():
    pay, evt, accs = _small()
    assert len(pay) == 4000 and len(evt) > len(pay)
    for col in ("payment_id", "rail", "identifier_type", "settlement_kind",
                "terminal_status", "time_to_settle_min", "IntrBkSttlmAmt"):
        assert col in pay.columns
    for c in EXCEPTION_CODES:
        assert f"exc_{c}" in pay.columns
    assert set(pay["rail"]).issubset(set(RAIL_NAMES))
    assert {"exc_sla_breach", "exc_limit_exceeded"} <= set(pay.columns)


def test_all_rails_present_with_swift_crossborder():
    pay, _, _ = _small()
    assert set(pay["rail"]) == set(RAIL_NAMES)                       # incl SWIFT
    dom = pay[pay.rail != "SWIFT"]
    assert (dom["Ccy"] == "INR").all()                              # domestic rails: INR
    assert (dom["Dbtr_Ctry"] == "IN").all() and (dom["Cdtr_Ctry"] == "IN").all()
    swift = pay[pay.rail == "SWIFT"]
    assert len(swift) > 0
    assert (swift["Dbtr_Ctry"] != swift["Cdtr_Ctry"]).all()         # cross-border
    assert (swift["Dbtr_Ctry"].eq("IN") | swift["Cdtr_Ctry"].eq("IN")).all()  # one leg IN
    assert pay["Ccy"].nunique() > 1                                 # foreign ccy present


def test_over_cap_attempts_are_rejected_with_limit_exceeded():
    pay, _, _ = _small()
    bad = pay[(pay.rail == "UPI") & (pay.IntrBkSttlmAmt > RAILS["UPI"].cap)]
    assert len(bad) > 0                                              # injection produced some
    # an over-cap payment can be halted at an earlier step, but it can NEVER settle...
    assert (bad["terminal_status"] != "STP").all()
    # ...and most reach limit_check, where the cap is caught.
    assert bad["exc_limit_exceeded"].mean() > 0.7
    # any payment flagged limit_exceeded is rejected (a hard cap, never repaired).
    flagged = pay[pay["exc_limit_exceeded"] == 1]
    assert (flagged["terminal_status"] == "REJECTED").all()


def test_eta_spread_across_rails():
    pay, _, _ = _small()
    eta = pay.groupby("rail")["time_to_settle_min"].mean()
    assert eta["NEFT"] > eta["UPI"] and eta["NEFT"] > eta["IMPS"]    # batch latency shows
    assert eta["SWIFT"] > eta["NEFT"]                               # correspondent slowest


def test_rail_and_settlement_kind_not_in_feature_buckets():
    pay, _, accs = _small()
    s = build_schema(pay, accs)
    feats = sum(s["buckets"].values(), [])
    # rail is the label; settlement_kind and SttlmMtd are 1:1 consequences of it -> no leak.
    assert "rail" not in feats and "settlement_kind" not in feats and "SttlmMtd" not in feats
    assert "identifier_type" in s["buckets"]["core"]                # but instrument is a feature


def test_schema_task_manifest_and_twin_block():
    pay, _, accs = _small()
    s = build_schema(pay, accs)
    names = {t["name"] for t in s["tasks"]}
    assert {"risk", "rail_routing"} <= names
    routing = next(t for t in s["tasks"] if t["name"] == "rail_routing")
    assert routing["label_column"] == "rail" and set(routing["label_values"]) == set(RAIL_NAMES)
    t = s["twin"]
    assert set(t["rails"]) == set(RAIL_NAMES) and set(t["workflow"]) == set(WORKFLOW)
    assert t["twin_binary_tasks"] == ["exc_sla_breach", "exc_limit_exceeded"]


def test_event_log_chronological_and_references_payments():
    pay, evt, _ = _small()
    assert set(evt["payment_id"]).issubset(set(pay["payment_id"]))
    one = evt[evt.payment_id == evt.payment_id.iloc[0]]
    assert list(one["seq"]) == sorted(one["seq"])
    assert (one["t_min"].to_numpy()[1:] >= one["t_min"].to_numpy()[:-1]).all()


def test_reproducible():
    p1, _, _ = build_dataset(IndiaConfig(num_accounts=200, num_payments=1500, seed=7))
    p2, _, _ = build_dataset(IndiaConfig(num_accounts=200, num_payments=1500, seed=7))
    assert p1["rail"].tolist() == p2["rail"].tolist()
