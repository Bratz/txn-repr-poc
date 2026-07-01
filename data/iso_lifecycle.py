"""
ISO 20022 message lifecycle: a payment is not one record, it is a SEQUENCE OF MESSAGES from
different domains/systems, each carrying its own field set. This module models that multi-
source reality on top of the single pacs.008-style row synth_india_rails already builds.

Credit-transfer lifecycle (three ISO domains):

    pain.001  Initiation   acquisition - corporate -> debtor agent (InstdAmt, parties)
    pain.002  Initiation   status back to corporate (ACCP / RJCT)
    pacs.008  Clearing      interbank credit transfer (the ENRICHED complete record)
    pacs.002  Clearing      interbank settlement status = gpi tracker (ACSC / ACSP+G002 / RJCT)
    camt.054  Cash Mgmt     beneficiary credited (BOOK) - the true end state

Two shapes, not one nested ladder:
  * the payment RECORD enriches pain.001 -> pacs.008 (fields accumulate; nested) - this is
    what the encoder imputes (ENRICH_ADDS + missing_mask below).
  * pain.002 / pacs.002 / camt.054 are SPARSE status/notification messages that only reference
    the payment (echo amount + carry TxSts + reason) - not nested, a different source shape.

Which messages a payment emits, and their TxSts, fall straight out of where its workflow
halted (data/synth_india_rails.simulate_payment) - no new simulation.

ponytail: message rows reuse the pacs.008 columns (owned fields copied, rest UNK/NaN) so the
SAME schema encodes them - no second schema, no second encoder. Reason codes are indicative
ISO ExternalStatusReason values; refine against a real reason-code table if needed.
"""

from __future__ import annotations

import math

from data.synth_pacs008 import COLUMN_BUCKETS

UNKNOWN = "Unknown"

MSG_TYPES = ["pain.001", "pain.002", "pacs.008", "pacs.002", "camt.054",
             "camt.056", "camt.029", "pacs.004", "pacs.009"]
# ISO 20022 transaction-status codes we use (TxSts). "" = not a status message.
# The pacs.002 status IS the gpi-tracker in-flight backbone: ACSC (settled) / ACSP (in process,
# with a Gnnn subcode) / RJCT (rejected/returned). CNCL/RJCR are the camt.029 recall outcomes.
TX_STS = ["", "ACCP", "ACSC", "ACSP", "RJCT", "BOOK", "CNCL", "RJCR"]
# gpi tracker ACSP subcodes (SWIFT gpi / CBPR+). G002 is emitted for a held/in-repair payment;
# G001 (forwarded to a non-GPI agent) is supported by the vocab but not produced synthetically
# here (no agent-membership / forwarding modelled).
GPI_SUBCODES = {"G001": "forwarded to a non-GPI agent",
                "G002": "in repair / awaiting manual action"}

# All columns the encoder reads (india schema = v1 buckets + identifier_type).
_HIGH_CARD = list(COLUMN_BUCKETS["high_card_categorical"])
_NUMERIC = list(COLUMN_BUCKETS["numerical"])
_CORE = list(COLUMN_BUCKETS["core"]) + ["identifier_type"]
_META = list(COLUMN_BUCKETS["meta_party"])
ENCODER_COLS = _HIGH_CARD + _NUMERIC + _CORE + _META
_ALL = set(ENCODER_COLS)

# Field OWNERSHIP per message: which encoder columns the message legitimately carries.
# Anything not owned is blanked (UNK for categoricals, NaN for the numeric amount).
OWNED = {
    # initiation: full parties + instructed amount + instrument; NO interbank settlement
    # date/method, and the ultimate CREDITOR chain is not yet resolved (correspondent step).
    "pain.001": _ALL - {"UltmtCdtr_Id", "UltmtCdtr_Nm", "IntrBkSttlmDt", "SttlmMtd"},
    # clearing: the complete, enriched interbank record (what synth_india_rails builds).
    "pacs.008": set(_ALL),
    # status/notification: sparse - echo amount+ccy (+ settlement date once known).
    "pain.002": {"IntrBkSttlmAmt", "Ccy"},
    "pacs.002": {"IntrBkSttlmAmt", "Ccy", "IntrBkSttlmDt"},
    "camt.054": {"IntrBkSttlmAmt", "Ccy", "IntrBkSttlmDt",
                 "CdtrAcct_Id", "Cdtr_Nm", "Cdtr_Ctry"},
    # cancellation request / resolution: sparse, reference the payment by amount+ccy.
    "camt.056": {"IntrBkSttlmAmt", "Ccy"},
    "camt.029": {"IntrBkSttlmAmt", "Ccy"},
    # return: echoes the payment but with the PARTIES REVERSED (money flows back).
    "pacs.004": _ALL - {"SttlmMtd", "identifier_type"},
    # cover: interbank FI-to-FI funding leg alongside a cross-border pacs.008 (cover method).
    "pacs.009": {"IntrBkSttlmAmt", "Ccy", "IntrBkSttlmDt", "DbtrAcct_Id", "CdtrAcct_Id"},
}

# On a return (pacs.004) debtor and creditor swap: debtor↔creditor, their accounts, ultimate
# parties, names, countries and industries. Applied by _project_message(reverse=True).
REVERSAL_PAIRS = [
    ("DbtrAcct_Id", "CdtrAcct_Id"), ("UltmtDbtr_Id", "UltmtCdtr_Id"),
    ("Dbtr_Nm", "Cdtr_Nm"), ("UltmtDbtr_Nm", "UltmtCdtr_Nm"),
    ("Dbtr_Ctry", "Cdtr_Ctry"), ("Dbtr_Industry", "Cdtr_Industry"),
    ("Dbtr_SubIndustry", "Cdtr_SubIndustry"),
]

# Nested field availability for the ENRICHING record (pain.001 -> pacs.008), over the encoder's
# reconstructable columns. Used by run_impute: mask the not-yet-available fields and let the
# encoder impute them. pacs.008 owns everything, so it is the complete baseline.
ENRICH_ADDS = [
    ("pain.001", ["DbtrAcct_Id", "CdtrAcct_Id", "IntrBkSttlmAmt", "Ccy",
                  "UltmtDbtr_Id", "identifier_type"]),
    ("pacs.008", ["IntrBkSttlmDt", "UltmtCdtr_Id"]),  # clearing dates + payee chain resolved
]

# Debtor-bank pre-submission steps: a payment rejected here never reaches interbank clearing,
# so it is rejected at pain.002 (no pacs.008). Everything else is post-submission -> pacs.002.
# Documented synthetic split; reorder for a real engine's submission boundary.
PRE_SUBMISSION = {"validation", "min_amount_check", "limit_check", "enrichment",
                  "vpa_resolution", "beneficiary_resolution", "fraud_risk", "aml"}

# exception code (synth_india_rails) -> indicative ISO ExternalStatusReason code.
REASON = {
    "format_error": "FF01", "missing_field": "FF01",
    "vpa_not_found": "AC03", "beneficiary_unreachable": "BE06",
    "fraud_hold": "FRAD", "limit_exceeded": "AM02", "below_min": "AM06",
    "sanctions_hit": "RR04", "insufficient_liquidity": "AM04",
    "technical_decline": "ED05", "settlement_fail": "ED05", "batch_return": "ED05",
    "account_closed": "AC04", "fx_fail": "AM09", "no_route": "ED01", "no_cover": "ED05",
}


def stage_names():
    return [n for n, _ in ENRICH_ADDS]


def available_at(stage_idx: int) -> set:
    s = set()
    for i in range(stage_idx + 1):
        s |= set(ENRICH_ADDS[i][1])
    return s


def missing_mask(recon_names, stage_idx: int) -> list:
    """Bool list aligned to recon_names: True = field not yet available at this stage (masked,
    to be imputed). The last stage (pacs.008) owns everything -> masks nothing."""
    avail = available_at(stage_idx)
    return [n not in avail for n in recon_names]


def _project_message(pay_row, msg_type, reverse=False):
    owned = OWNED[msg_type]
    row = {}
    for col in ENCODER_COLS:
        if col in owned:
            row[col] = pay_row[col]
        else:
            row[col] = float("nan") if col in _NUMERIC else UNKNOWN
    if reverse:                                     # pacs.004: swap debtor <-> creditor
        for a, b in REVERSAL_PAIRS:
            row[a], row[b] = row[b], row[a]
    return row


def _halting_cause(events):
    """Last (step, excode) whose excode is a real exception ('none' = clean/repaired)."""
    for step, excode in reversed(list(events)):
        if excode and excode != "none":
            return step, excode
    return None, None


def lifecycle_messages(pay_row, status, events):
    """Emit the ISO message sequence for one payment as a list of encodable rows.

    events: iterable of (step, excode) for this payment (from the event log). status: the
    payment's terminal_status. Each returned row = the owned pacs.008 columns + lifecycle
    metadata (msg_type, tx_sts, sts_reason, end_to_end_id, seq).
    """
    pid = int(pay_row["payment_id"])
    e2e = f"E2E-{pid:08d}"
    halt_step, halt_exc = _halting_cause(events)
    reason = REASON.get(halt_exc, "NARR") if halt_exc else ""
    total = float(pay_row.get("time_to_settle_min", 0.0) or 0.0)  # minutes to the last message
    cancel_requested = int(pay_row.get("cancel_requested", 0))
    cancel_status = pay_row.get("cancel_status", "none")
    return_reason = pay_row.get("return_reason", "") or ""

    out, seq = [], 0

    def emit(mtype, tx_sts="", rsn="", reverse=False):
        nonlocal seq
        row = _project_message(pay_row, mtype, reverse=reverse)
        row.update(payment_id=pid, seq=seq, msg_type=mtype, tx_sts=tx_sts,
                   sts_reason=rsn, end_to_end_id=e2e,
                   rail=pay_row.get("rail", UNKNOWN),
                   direction=pay_row.get("direction", UNKNOWN))
        out.append(row)
        seq += 1

    emit("pain.001")
    # rejected at the debtor bank before interbank submission -> stops at pain.002 (no recall).
    if status == "REJECTED" and halt_step in PRE_SUBMISSION:
        emit("pain.002", "RJCT", reason)
        return _stamp_offsets(out, total)
    emit("pain.002", "ACCP")
    emit("pacs.008")
    # cross-border cover method: a pacs.009 COV funds the correspondent alongside the pacs.008.
    if pay_row.get("SttlmMtd") == "COVE":
        emit("pacs.009")

    # --- clearing outcome ------------------------------------------------------ #
    auto_return = status == "REJECTED" and halt_exc == "account_closed"
    if status == "REJECTED":
        if auto_return:                      # settled to creditor bank, then credit bounced
            emit("pacs.002", "ACSC")
            emit("pacs.004", "", return_reason or REASON["account_closed"], reverse=True)
        else:                                # rejected before/at settlement
            emit("pacs.002", "RJCT", reason)
    elif status == "MANUAL_REVIEW":          # held in repair -> gpi tracker ACSP + G002
        emit("pacs.002", "ACSP", "G002")
    else:                                    # STP / REPAIRED: settled and credited
        emit("pacs.002", "ACSC")
        emit("camt.054", "BOOK")

    # --- originator recall overlay (camt.056 request -> camt.029 resolution) ---- #
    # If the recall is accepted (CNCL) but funds were already credited, funds come back via a
    # pacs.004 return. Set on the payment by the generator; see synth_india_rails.
    if cancel_requested:
        emit("camt.056", "", "CUST")
        emit("camt.029", cancel_status if cancel_status != "none" else "RJCR")
        if cancel_status == "CNCL" and status in ("STP", "REPAIRED"):
            emit("pacs.004", "", return_reason or "CUST", reverse=True)
    return _stamp_offsets(out, total)


def _stamp_offsets(out, total):
    """Spread relative timestamps across [0, total] min: pain.001 @ 0, last message @ total."""
    n = len(out)
    for i, row in enumerate(out):
        row["t_offset_min"] = round(total * i / (n - 1), 3) if n > 1 else 0.0
    return out


if __name__ == "__main__":
    # self-check: the emission rules, field ownership and nested masks hold.
    recon = ["DbtrAcct_Id", "CdtrAcct_Id", "IntrBkSttlmAmt", "Ccy",
             "IntrBkSttlmDt", "UltmtDbtr_Id", "UltmtCdtr_Id", "identifier_type"]

    def _missing(i):
        return {recon[j] for j, m in enumerate(missing_mask(recon, i)) if m}

    # pain.001 hides exactly the two clearing-resolved fields; pacs.008 hides nothing.
    assert _missing(0) == {"IntrBkSttlmDt", "UltmtCdtr_Id"}, _missing(0)
    assert not any(missing_mask(recon, 1))
    assert _missing(1) <= _missing(0)                              # nested
    assert stage_names() == ["pain.001", "pacs.008"]

    base = {c: f"v_{c}" for c in ENCODER_COLS}
    base.update(payment_id=7, IntrBkSttlmAmt=100.0, rail="IMPS", direction="outward",
                time_to_settle_min=10.0)

    def chain(status, events):
        return [(m["msg_type"], m["tx_sts"], m["sts_reason"])
                for m in lifecycle_messages(base, status, events)]

    # happy path -> pain.001..camt.054 chain ending in a camt.054 BOOK.
    stp = chain("STP", [("validation", "none"), ("credit", "none")])
    assert [m for m, _, _ in stp] == MSG_TYPES[:5], stp
    assert stp[-1] == ("camt.054", "BOOK", "")
    # pre-submission reject (limit_check) -> stops at pain.002 RJCT, no pacs.008.
    pre = chain("REJECTED", [("validation", "none"), ("limit_check", "limit_exceeded")])
    assert [m for m, _, _ in pre] == ["pain.001", "pain.002"]
    assert pre[-1] == ("pain.002", "RJCT", "AM02"), pre
    # clearing reject (settlement) -> pacs.008 sent, then pacs.002 RJCT, no camt.054.
    clr = chain("REJECTED", [("validation", "none"), ("settlement", "settlement_fail")])
    assert [m for m, _, _ in clr] == ["pain.001", "pain.002", "pacs.008", "pacs.002"]
    assert clr[-1] == ("pacs.002", "RJCT", "ED05"), clr
    # manual review -> settlement pending, never booked.
    mr = chain("MANUAL_REVIEW", [("aml", "none"), ("npci_switch", "technical_decline")])
    assert mr[-1] == ("pacs.002", "ACSP", "G002")          # gpi tracker: in repair

    # pain.001 blanks the four not-yet-available columns; pacs.008 keeps all.
    p1 = _project_message(base, "pain.001")
    assert p1["UltmtCdtr_Id"] == UNKNOWN and p1["IntrBkSttlmDt"] == UNKNOWN
    assert p1["DbtrAcct_Id"] == "v_DbtrAcct_Id"
    assert math.isnan(_project_message(base, "pain.002")["IntrBkSttlmAmt"]) is False
    assert _project_message(base, "pacs.008")["UltmtCdtr_Id"] == "v_UltmtCdtr_Id"

    # timestamps: pain.001 @ 0, last message @ total, monotonic non-decreasing.
    msgs = lifecycle_messages(base, "STP", [("credit", "none")])
    offs = [m["t_offset_min"] for m in msgs]
    assert offs[0] == 0.0 and offs[-1] == 10.0
    assert all(b >= a for a, b in zip(offs, offs[1:]))

    # return leg: account_closed bounce -> pacs.002 ACSC then a pacs.004 with REVERSED parties.
    ar = lifecycle_messages(base, "REJECTED", [("credit", "account_closed")])
    types = [m["msg_type"] for m in ar]
    assert types[-2:] == ["pacs.002", "pacs.004"] and ar[-2]["tx_sts"] == "ACSC"
    ret = ar[-1]
    assert ret["DbtrAcct_Id"] == base["CdtrAcct_Id"] and ret["CdtrAcct_Id"] == base["DbtrAcct_Id"]

    # cancellation overlay: accepted recall of a credited payment -> camt.056, camt.029(CNCL), pacs.004.
    crow = dict(base, cancel_requested=1, cancel_status="CNCL", return_reason="CUST")
    cm = [(m["msg_type"], m["tx_sts"]) for m in lifecycle_messages(crow, "STP", [("credit", "none")])]
    assert ("camt.056", "") in cm and ("camt.029", "CNCL") in cm and cm[-1] == ("pacs.004", "")
    # rejected recall -> camt.029(RJCR), no return.
    rrow = dict(base, cancel_requested=1, cancel_status="RJCR")
    rm = [m["msg_type"] for m in lifecycle_messages(rrow, "STP", [("credit", "none")])]
    assert "camt.029" in rm and "pacs.004" not in rm

    # cover method (COVE) emits a pacs.009 alongside the pacs.008; non-cover does not.
    cov = dict(base, SttlmMtd="COVE")
    cseq = [m["msg_type"] for m in lifecycle_messages(cov, "STP", [("credit", "none")])]
    assert cseq[:4] == ["pain.001", "pain.002", "pacs.008", "pacs.009"]
    assert "pacs.009" not in [m["msg_type"] for m in
                              lifecycle_messages(dict(base, SttlmMtd="CLRG"), "STP", [("credit", "none")])]
    print("iso_lifecycle self-check OK")
