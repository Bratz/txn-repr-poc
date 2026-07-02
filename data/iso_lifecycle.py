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
    # camt.054 is strictly the CREDITOR-side notification; on the outward leg the debtor bank
    # really tracks completion via gpi trck/pacs.002 confirmations. We emit one camt.054 as the
    # booked end-state and let MSG_FLOW mark it IN on the outward perspective - a documented
    # simplification, not a message-standard claim.
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

# Per-message flow relative to OUR bank, given the payment's direction. outward = we are the
# debtor agent (pain.001 IN from the customer, clearing legs OUT, confirmations IN); inward =
# we are the creditor agent (pain.* NOT visible - they live at the remote debtor's bank).
# NOTE: camt.056 direction assumes ORIGINATOR-initiated recall (the only kind the generator
# produces); if beneficiary-initiated recalls are ever added, this map must grow a case.
MSG_FLOW = {
    "outward": {"pain.001": "IN", "pain.002": "OUT", "pacs.008": "OUT", "pacs.009": "OUT",
                "pacs.002": "IN", "camt.054": "IN", "camt.056": "OUT", "camt.029": "IN",
                "pacs.004": "IN"},
    "inward":  {"pacs.008": "IN", "pacs.009": "IN", "pacs.002": "OUT", "camt.054": "OUT",
                "camt.056": "IN", "camt.029": "OUT", "pacs.004": "OUT"},
}

# Deterministic post-settlement lags (minutes) for the exception legs - a documented synthetic
# choice (real returns/recalls take hours-days and vary).
RETURN_LAG_MIN = 60.0      # account_closed bounce: credit fails shortly after settlement
RECALL_LAG_MIN = 240.0     # originator recall: request / resolution / return, spaced 4h apart

# exception code (synth_india_rails) -> indicative ISO ExternalStatusReason code.
REASON = {
    "format_error": "FF01", "missing_field": "FF01",
    "vpa_not_found": "AC03", "beneficiary_unreachable": "BE06",
    "fraud_hold": "FRAD", "limit_exceeded": "AM02", "below_min": "AM06",
    "sanctions_hit": "RR04", "insufficient_liquidity": "AM04",
    "technical_decline": "ED05", "settlement_fail": "ED05", "batch_return": "ED05",
    "account_closed": "AC04", "fx_fail": "AM09", "no_route": "ED01", "no_cover": "ED05",
}


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
    """Last (step, excode[, t]) whose excode is a real exception ('none' = clean/repaired)."""
    for e in reversed(list(events)):
        if e[1] and e[1] != "none":
            return e[0], e[1]
    return None, None


def lifecycle_messages(pay_row, status, events):
    """Emit the ISO message sequence for one payment as a list of encodable rows.

    events: iterable of (step, excode) or (step, excode, t_min) for this payment (from the
    event log). status: the payment's terminal_status. Each returned row = the owned pacs.008
    columns + lifecycle metadata (msg_type, tx_sts, sts_reason, end_to_end_id, seq,
    t_offset_min, msg_direction, visible).

    Timestamps are ANCHORED to the workflow event times when events carry t_min: pain.001 at 0,
    pain.002/pacs.008 at debtor-bank acceptance (last pre-submission step), pacs.002/camt.054
    at settlement completion, and the exception legs (pacs.004, camt.056/029) at deterministic
    lags AFTER settlement. Without t_min (legacy 2-tuples), falls back to a linear spread.
    """
    pid = int(pay_row["payment_id"])
    e2e = f"E2E-{pid:08d}"
    ev = [(e[0], e[1], float(e[2]) if len(e) > 2 else None) for e in events]
    halt_step, halt_exc = _halting_cause(ev)
    reason = REASON.get(halt_exc, "NARR") if halt_exc else ""
    total = float(pay_row.get("time_to_settle_min", 0.0) or 0.0)  # minutes to workflow end
    anchored = bool(ev) and all(t is not None for _, _, t in ev)
    # debtor-bank acceptance time = last pre-submission step completed (0 if unknown).
    accept_t = (max((t for s, _, t in ev if s in PRE_SUBMISSION), default=0.0)
                if anchored else None)
    cancel_requested = int(pay_row.get("cancel_requested", 0))
    cancel_status = pay_row.get("cancel_status", "none")
    return_reason = pay_row.get("return_reason", "") or ""
    direction = pay_row.get("direction", UNKNOWN)

    out, seq = [], 0

    def emit(mtype, tx_sts="", rsn="", reverse=False, t=None):
        nonlocal seq
        row = _project_message(pay_row, mtype, reverse=reverse)
        flow = MSG_FLOW.get(direction, {}).get(mtype)
        row.update(payment_id=pid, seq=seq, msg_type=mtype, tx_sts=tx_sts,
                   sts_reason=rsn, end_to_end_id=e2e,
                   rail=pay_row.get("rail", UNKNOWN), direction=direction,
                   msg_direction=flow, visible=int(flow is not None),
                   t_offset_min=round(t, 3) if t is not None else None)
        out.append(row)
        seq += 1

    emit("pain.001", t=0.0 if anchored else None)
    # rejected at the debtor bank before interbank submission -> stops at pain.002 (no recall).
    if status == "REJECTED" and halt_step in PRE_SUBMISSION:
        emit("pain.002", "RJCT", reason, t=total if anchored else None)
        return _finalize_offsets(out, total)
    emit("pain.002", "ACCP", t=accept_t)
    emit("pacs.008", t=accept_t)             # submitted to clearing on acceptance
    # cross-border cover method: a pacs.009 COV funds the correspondent alongside the pacs.008.
    if pay_row.get("SttlmMtd") == "COVE":
        emit("pacs.009", t=accept_t)

    # --- clearing outcome (settlement completes at `total`) --------------------- #
    auto_return = status == "REJECTED" and halt_exc == "account_closed"
    if status == "REJECTED":
        if auto_return:                      # settled to creditor bank, then credit bounced
            emit("pacs.002", "ACSC", t=total if anchored else None)
            emit("pacs.004", "", return_reason or REASON["account_closed"], reverse=True,
                 t=total + RETURN_LAG_MIN if anchored else None)
        else:                                # rejected before/at settlement
            emit("pacs.002", "RJCT", reason, t=total if anchored else None)
    elif status == "MANUAL_REVIEW":          # held in repair -> gpi tracker ACSP + G002
        emit("pacs.002", "ACSP", "G002", t=total if anchored else None)
    else:                                    # STP / REPAIRED: settled and credited
        emit("pacs.002", "ACSC", t=total if anchored else None)
        emit("camt.054", "BOOK", t=total if anchored else None)

    # --- originator recall overlay (camt.056 request -> camt.029 resolution) ---- #
    # If the recall is accepted (CNCL) but funds were already credited, funds come back via a
    # pacs.004 return. Set on the payment by the generator; see synth_india_rails.
    if cancel_requested:
        base = total if anchored else None
        emit("camt.056", "", "CUST", t=base + RECALL_LAG_MIN if anchored else None)
        emit("camt.029", cancel_status if cancel_status != "none" else "RJCR",
             t=base + 2 * RECALL_LAG_MIN if anchored else None)
        if cancel_status == "CNCL" and status in ("STP", "REPAIRED"):
            emit("pacs.004", "", return_reason or "CUST", reverse=True,
                 t=base + 3 * RECALL_LAG_MIN if anchored else None)
    return _finalize_offsets(out, total)


def _finalize_offsets(out, total):
    """Anchored rows keep their event-derived times; legacy rows (t=None) get the old linear
    spread across [0, total] so 2-tuple callers keep working."""
    if any(r["t_offset_min"] is None for r in out):
        n = len(out)
        for i, row in enumerate(out):
            row["t_offset_min"] = round(total * i / (n - 1), 3) if n > 1 else 0.0
    return out
