"""Tests for the ISO 20022 message lifecycle (data/iso_lifecycle.py)."""

from data.iso_lifecycle import (
    ENCODER_COLS, ENRICH_ADDS, MSG_TYPES, UNKNOWN, lifecycle_messages, missing_mask,
)

RECON = ["DbtrAcct_Id", "CdtrAcct_Id", "IntrBkSttlmAmt", "Ccy",
         "IntrBkSttlmDt", "UltmtDbtr_Id", "UltmtCdtr_Id", "identifier_type"]


def _missing(stage_idx):
    return {RECON[j] for j, m in enumerate(missing_mask(RECON, stage_idx)) if m}


def _row():
    r = {c: f"v_{c}" for c in ENCODER_COLS}
    r.update(payment_id=7, IntrBkSttlmAmt=100.0, rail="UPI", direction="outward")
    return r


def _chain(status, events):
    return [(m["msg_type"], m["tx_sts"], m["sts_reason"])
            for m in lifecycle_messages(_row(), status, events)]


def test_pain001_hides_clearing_fields_pacs008_complete():
    assert _missing(0) == {"IntrBkSttlmDt", "UltmtCdtr_Id"}
    assert not any(missing_mask(RECON, 1))
    assert _missing(1) <= _missing(0)                       # nested
    assert [n for n, _ in ENRICH_ADDS] == ["pain.001", "pacs.008"]


def test_happy_path_emits_full_five_message_chain():
    stp = _chain("STP", [("validation", "none"), ("credit", "none")])
    assert [m for m, _, _ in stp] == MSG_TYPES[:5]          # pain.001 .. camt.054 (no recall)
    assert stp[-1] == ("camt.054", "BOOK", "")


def test_pre_submission_reject_stops_at_pain002():
    pre = _chain("REJECTED", [("validation", "none"), ("limit_check", "limit_exceeded")])
    assert [m for m, _, _ in pre] == ["pain.001", "pain.002"]
    assert pre[-1] == ("pain.002", "RJCT", "AM02")           # no pacs.008


def test_clearing_reject_stops_at_pacs002():
    clr = _chain("REJECTED", [("validation", "none"), ("settlement", "settlement_fail")])
    assert [m for m, _, _ in clr] == ["pain.001", "pain.002", "pacs.008", "pacs.002"]
    assert clr[-1] == ("pacs.002", "RJCT", "ED05")           # pacs.008 sent, no camt.054


def test_manual_review_maps_to_gpi_tracker_acsp_g002_never_booked():
    mr = _chain("MANUAL_REVIEW", [("aml", "none"), ("npci_switch", "technical_decline")])
    assert [m for m, _, _ in mr][-1] == "pacs.002"
    assert mr[-1][1:] == ("ACSP", "G002")                  # gpi tracker: in repair, not booked


def test_timestamps_start_zero_end_total_monotonic():
    row = dict(_row(), time_to_settle_min=10.0)
    offs = [m["t_offset_min"] for m in lifecycle_messages(row, "STP", [("credit", "none")])]
    assert offs[0] == 0.0 and offs[-1] == 10.0
    assert all(b >= a for a, b in zip(offs, offs[1:]))


def test_event_anchored_timestamps_and_post_settlement_legs():
    row = dict(_row(), time_to_settle_min=20.0)
    ev = [("validation", "none", 1.0), ("aml", "none", 5.0),
          ("settlement", "none", 18.0), ("credit", "none", 20.0)]
    t = {m["msg_type"]: m["t_offset_min"] for m in lifecycle_messages(row, "STP", ev)}
    assert t["pain.001"] == 0.0
    assert t["pain.002"] == 5.0 and t["pacs.008"] == 5.0    # acceptance = last pre-submission
    assert t["pacs.002"] == 20.0 and t["camt.054"] == 20.0  # settlement completion
    # exception legs land AFTER settlement, in order
    crow = dict(row, cancel_requested=1, cancel_status="CNCL", return_reason="CUST")
    cm = {m["msg_type"]: m["t_offset_min"] for m in lifecycle_messages(crow, "STP", ev)}
    assert cm["camt.056"] > 20.0
    assert cm["pacs.004"] > cm["camt.029"] > cm["camt.056"]


def test_msg_direction_and_visibility_perspective():
    outm = {m["msg_type"]: m["msg_direction"]
            for m in lifecycle_messages(_row(), "STP", [("credit", "none")])}
    assert outm["pain.001"] == "IN" and outm["pacs.008"] == "OUT" and outm["camt.054"] == "IN"
    irow = dict(_row(), direction="inward")
    inm = {m["msg_type"]: (m["msg_direction"], m["visible"])
           for m in lifecycle_messages(irow, "STP", [("credit", "none")])}
    assert inm["pain.001"] == (None, 0)                      # remote debtor bank's leg
    assert inm["pacs.008"] == ("IN", 1) and inm["pacs.002"] == ("OUT", 1)


def test_cover_method_emits_pacs009_after_pacs008():
    cov = dict(_row(), SttlmMtd="COVE")
    types = [m["msg_type"] for m in lifecycle_messages(cov, "STP", [("credit", "none")])]
    assert types[:4] == ["pain.001", "pain.002", "pacs.008", "pacs.009"]
    clrg = dict(_row(), SttlmMtd="CLRG")
    assert "pacs.009" not in [m["msg_type"]
                              for m in lifecycle_messages(clrg, "STP", [("credit", "none")])]


def test_account_closed_bounces_to_return_with_reversed_parties():
    msgs = lifecycle_messages(_row(), "REJECTED", [("credit", "account_closed")])
    types = [m["msg_type"] for m in msgs]
    assert types[-2:] == ["pacs.002", "pacs.004"]           # settled, then returned
    assert msgs[-2]["tx_sts"] == "ACSC"
    ret = msgs[-1]
    assert ret["DbtrAcct_Id"] == "v_CdtrAcct_Id"            # parties reversed on the return
    assert ret["CdtrAcct_Id"] == "v_DbtrAcct_Id"


def test_accepted_recall_of_credited_payment_emits_camt056_029_and_return():
    row = dict(_row(), cancel_requested=1, cancel_status="CNCL", return_reason="CUST")
    seq = [(m["msg_type"], m["tx_sts"]) for m in
           lifecycle_messages(row, "STP", [("credit", "none")])]
    assert ("camt.056", "") in seq and ("camt.029", "CNCL") in seq
    assert seq[-1] == ("pacs.004", "")                      # funds returned after recall


def test_rejected_recall_has_resolution_but_no_return():
    row = dict(_row(), cancel_requested=1, cancel_status="RJCR")
    types = [m["msg_type"] for m in lifecycle_messages(row, "STP", [("credit", "none")])]
    assert "camt.029" in types and "pacs.004" not in types  # too late to recall


def test_message_rows_blank_unowned_columns():
    msgs = {m["msg_type"]: m for m in
            lifecycle_messages(_row(), "STP", [("credit", "none")])}
    # pain.001: creditor chain + interbank date/method not yet available
    assert msgs["pain.001"]["UltmtCdtr_Id"] == UNKNOWN
    assert msgs["pain.001"]["IntrBkSttlmDt"] == UNKNOWN
    assert msgs["pain.001"]["DbtrAcct_Id"] == "v_DbtrAcct_Id"
    # pacs.008: the complete enriched record
    assert msgs["pacs.008"]["UltmtCdtr_Id"] == "v_UltmtCdtr_Id"
    # pain.002 status message is sparse: parties blanked, amount echoed
    assert msgs["pain.002"]["DbtrAcct_Id"] == UNKNOWN
    assert msgs["pain.002"]["IntrBkSttlmAmt"] == 100.0
