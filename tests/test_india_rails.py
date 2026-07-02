"""Tests for the India multi-rail generator (data/rails.py + data/synth_india_rails.py)."""

import numpy as np

from data.rails import (
    RAILS, RAIL_NAMES, below_min, choose_rail, eligible_rails, sample_identifier,
    violates_cap,
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


# --------------------------------------------------------------------------- #
# synth_india_rails.py - dataset
# --------------------------------------------------------------------------- #

def _small():
    return build_dataset(IndiaConfig(num_accounts=400, num_payments=4000, seed=23))


def _with_upi():
    # UPI is off by default now; the cap / all-rails routing tests opt it back in (the
    # registry + routing still fully support it - it is a reversible run knob).
    return build_dataset(IndiaConfig(num_accounts=400, num_payments=4000, seed=23,
                                     domestic_rails=("RTGS", "NEFT", "IMPS", "UPI")))


def test_default_run_excludes_upi():
    pay, _, _ = _small()
    assert "UPI" not in set(pay["rail"])                             # dropped for now
    assert "VPA" not in set(pay["identifier_type"])                  # UPI-only instrument gone
    assert {"RTGS", "NEFT", "IMPS", "SWIFT"} == set(pay["rail"])


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
    pay, _, _ = _with_upi()
    assert set(pay["rail"]) == set(RAIL_NAMES)                       # incl UPI + SWIFT
    dom = pay[pay.rail != "SWIFT"]
    assert (dom["Ccy"] == "INR").all()                              # domestic rails: INR
    assert (dom["Dbtr_Ctry"] == "IN").all() and (dom["Cdtr_Ctry"] == "IN").all()
    swift = pay[pay.rail == "SWIFT"]
    assert len(swift) > 0
    assert (swift["Dbtr_Ctry"] != swift["Cdtr_Ctry"]).all()         # cross-border
    assert (swift["Dbtr_Ctry"].eq("IN") | swift["Cdtr_Ctry"].eq("IN")).all()  # one leg IN
    assert pay["Ccy"].nunique() > 1                                 # foreign ccy present


def test_over_cap_attempts_are_rejected_with_limit_exceeded():
    pay, _, _ = _with_upi()
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
    assert eta["NEFT"] > eta["IMPS"]                                # batch vs instant latency
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
    # deterministic gates (limit_exceeded/below_min) are DELISTED as TFM targets -> rule_computed;
    # twin binary tasks are the stochastic exceptions.
    assert t["twin_binary_tasks"] == ["exc_sla_breach", "exc_fraud_hold"]
    assert "exc_limit_exceeded" in t["rule_computed"] and "is_mis_routed" in t["rule_computed"]


def test_event_log_chronological_and_references_payments():
    pay, evt, _ = _small()
    assert set(evt["payment_id"]).issubset(set(pay["payment_id"]))
    one = evt[evt.payment_id == evt.payment_id.iloc[0]]
    assert list(one["seq"]) == sorted(one["seq"])
    assert (one["t_min"].to_numpy()[1:] >= one["t_min"].to_numpy()[:-1]).all()


def test_cancellation_and_return_labels_consistent_with_messages():
    from data.synth_india_rails import build_messages
    pay, evt, _ = _small()
    msg = build_messages(pay, evt)
    for col in ("cancel_requested", "cancel_status", "returned", "return_reason"):
        assert col in pay.columns
    by_pid = msg.groupby("payment_id")["msg_type"].agg(set)
    has = lambda t: pay["payment_id"].map(lambda p: t in by_pid.get(p, set()))
    # label <-> message presence must agree exactly
    assert (pay["cancel_requested"].astype(bool) == has("camt.056")).all()
    assert (pay["returned"].astype(bool) == has("pacs.004")).all()
    # a recall is only requested once the payment reached clearing (has a pacs.008)
    assert has("pacs.008")[pay["cancel_requested"].astype(bool)].all()


def test_swift_cover_method_emits_pacs009_alongside_pacs008():
    from data.synth_india_rails import build_messages
    pay, evt, _ = _small()
    msg = build_messages(pay, evt)
    by = msg.groupby("payment_id")["msg_type"].agg(set)
    has = lambda p, t: t in by.get(p, set())
    # every SWIFT payment that reached clearing (has a pacs.008) carries a pacs.009 cover
    for p in pay[pay.rail == "SWIFT"]["payment_id"]:
        if has(p, "pacs.008"):
            assert has(p, "pacs.009")
    # domestic rails never emit a cover
    for p in pay[pay.rail != "SWIFT"]["payment_id"]:
        assert not has(p, "pacs.009")


def test_cancellation_is_feature_modulated_by_amount():
    # cancellation is feature-modulated -> >Rs 1M payments are recalled more often. Checked on a
    # larger sample (the boosted subset is small; the effect is invisible at N=4000).
    pay, _, _ = build_dataset(IndiaConfig(num_accounts=500, num_payments=15000, seed=23))
    big = pay[pay.IntrBkSttlmAmt > 1_000_000]["cancel_requested"].mean()
    rest = pay[pay.IntrBkSttlmAmt <= 1_000_000]["cancel_requested"].mean()
    assert big > rest


def test_amount_split_models_fx_and_charges():
    pay, _, _ = _small()
    for c in ("InstdAmt", "InstdCcy", "fx_rate", "charges"):
        assert c in pay.columns
    dom = pay[pay.rail != "SWIFT"]
    assert (dom["fx_rate"] == 1.0).all()                     # domestic: no FX
    assert (dom["InstdAmt"] == dom["IntrBkSttlmAmt"]).all()  # 1:1
    sw = pay[pay.rail == "SWIFT"]
    assert (sw["InstdCcy"] != sw["Ccy"]).all()               # two currencies cross-border
    assert (sw["fx_rate"] != 1.0).mean() > 0.9               # FX applied on (almost) all
    assert (pay["charges"] > 0).all()                        # every payment carries a fee


def test_reproducible():
    p1, _, _ = build_dataset(IndiaConfig(num_accounts=200, num_payments=1500, seed=7))
    p2, _, _ = build_dataset(IndiaConfig(num_accounts=200, num_payments=1500, seed=7))
    assert p1["rail"].tolist() == p2["rail"].tolist()
