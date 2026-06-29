"""Tests for the curated cross-task golden set (data/golden_cases.py)."""

from data.golden_cases import build_golden, golden_pacs008_xml
from data.iso20022_pacs008 import parse_pacs008
from data.rails import RAIL_NAMES


def test_golden_spans_all_rails_and_labels():
    pay, evt = build_golden()
    assert set(pay["rail"]) == set(RAIL_NAMES)                    # every rail represented
    for col in ("risk_label", "geo_label", "expense_label", "terminal_status",
                "time_to_settle_min", "identifier_type", "case"):
        assert col in pay.columns
    assert len(evt) > len(pay)                                    # an event log exists
    assert set(evt["payment_id"]).issubset(set(pay["payment_id"]))


def test_deterministic_gates_have_expected_outcome():
    pay, _ = build_golden()
    g = pay.set_index("case")
    over = g.loc["upi_over_cap"]
    assert over["exc_limit_exceeded"] == 1 and over["terminal_status"] == "REJECTED"
    under = g.loc["rtgs_under_min"]
    assert under["exc_below_min"] == 1 and under["terminal_status"] == "REJECTED"


def test_swift_rows_are_crossborder_and_eta_tiered():
    pay, _ = build_golden()
    swift = pay[pay.rail == "SWIFT"]
    assert (swift["Dbtr_Ctry"] != swift["Cdtr_Ctry"]).all()
    assert (swift["geo_label"] == "International").all()
    # SWIFT clean leg settles far slower than an instant clean leg
    eta = pay.set_index("case")["time_to_settle_min"]
    assert eta["swift_xborder_clean"] > eta["upi_small_clean"]


def test_labels_are_rule_consistent_not_handtyped():
    # risk High requires factors to stack -> the large cross-border financial case is High
    pay, _ = build_golden()
    g = pay.set_index("case")
    assert g.loc["rtgs_high_risk", "risk_label"] == "High"
    assert g.loc["swift_xborder_clean", "expense_label"] == "Technology"  # creditor = Technology


def test_golden_serialises_to_pacs008_and_reparses():
    pay, _ = build_golden()
    rows = parse_pacs008(golden_pacs008_xml())
    assert len(rows) == len(pay)
    # message-native fields survive the XML round-trip
    for r, (_, p) in zip(rows, pay.iterrows()):
        assert r["IntrBkSttlmAmt"] == p["IntrBkSttlmAmt"]
        assert r["Ccy"] == p["Ccy"] and r["Dbtr_Ctry"] == p["Dbtr_Ctry"]
        assert r["CdtrAcct_Id"] == p["CdtrAcct_Id"]
