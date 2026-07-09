"""
India multi-rail dataset (RTGS / NEFT / IMPS / UPI + cross-border SWIFT) - intake +
rail-conditioned workflow, for the digital twin backbone.

BEYOND arXiv:2410.07851 (a rail/twin application). ADDITIVE & PARALLEL: this is a separate
generator that REUSES v1 helpers (data/synth_pacs008.py) the same way data/synth_workflow.py
does - the paper-grounded v1 module is untouched (no flag, no risk to its task suite).

Every payment is:
  1. given an INR amount, routed onto one of the four rails by data/rails.choose_rail
     (amount-band preference over the ELIGIBLE set + instrument), and
  2. simulated traversing that rail's orchestration workflow, where each step is clean or
     throws a feature-driven exception that is repaired or halts the payment.

Two tables are emitted (same contract as synth_workflow.py):
  * payment-level (one row / payment): pacs.008-style features + rail + identifier_type +
    the labels the twin/encoder predict at INTAKE - rail (routing), which exceptions occurred
    (multi-label incl. sla_breach / limit_exceeded), terminal status, time-to-settle.
  * event-level (one row / step): the (step, outcome, exception, time) log for the in-flight
    model.

Honest scope notes
------------------
The four domestic rails are IN/INR; SWIFT is the cross-border path (one leg abroad, FX), so
cross-border/currency signals are live for SWIFT rows but degenerate among the domestic four
(recurrence is not modelled here). Meaningful targets: risk (amount+industry+cross-border),
RAIL ROUTING, sla_breach, limit_exceeded, ETA. `settlement_kind` and `rail` are CONSEQUENCES
of the routing label and are kept as metadata/label only - NOT in the feature buckets (that
would leak the rail). `identifier_type` IS a feature: the instrument is known before the rail
is chosen (VPA=>UPI, MMID=>IMPS, BIC=>SWIFT are intentionally near-deterministic, and SWIFT
is also trivially flagged by the counterparty country/currency; the real difficulty is the
domestic ACCT_IFSC majority, where the amount decides RTGS/NEFT/IMPS). Caps/min/SLA are real
RBI/NPCI/SWIFT values; the rail-mix and exception rates are documented synthetic choices.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, replace
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from data.synth_pacs008 import (
    COLUMN_BUCKETS, COUNTRIES, GenConfig, assign_expense, assign_geo, assign_risk,
    generate_accounts, project_to_pacs008, vocab_report,
)
from data.rails import (
    IDENTIFIER_TYPES, RAILS, RAIL_NAMES, below_min, choose_rail, eligible_rails,
    sample_identifier, violates_cap,
)
from data.iso_lifecycle import PRE_SUBMISSION, REASON as LIFE_REASON

# Rail-conditioned orchestration workflows (ordered steps).
WORKFLOW = {
    "UPI":  ["validation", "vpa_resolution", "fraud_risk", "limit_check",
             "npci_switch", "credit"],
    "IMPS": ["validation", "beneficiary_resolution", "fraud_risk", "limit_check",
             "npci_switch", "credit"],
    "RTGS": ["validation", "min_amount_check", "aml", "liquidity",
             "rbi_settlement", "credit"],
    "NEFT": ["validation", "enrichment", "aml", "batch_window",
             "dns_settlement", "credit"],
    # cross-border correspondent path: FX + correspondent routing + cover, slow settle.
    "SWIFT": ["validation", "enrichment", "sanctions", "fx_conversion",
              "correspondent_routing", "cover_check", "settlement", "credit"],
}
# step -> the exception it can raise.
STEP_EXCEPTION = {
    "validation": "format_error",
    "enrichment": "missing_field",
    "vpa_resolution": "vpa_not_found",
    "beneficiary_resolution": "beneficiary_unreachable",
    "fraud_risk": "fraud_hold",
    "limit_check": "limit_exceeded",
    "min_amount_check": "below_min",
    "aml": "sanctions_hit",
    "liquidity": "insufficient_liquidity",
    "npci_switch": "technical_decline",
    "rbi_settlement": "settlement_fail",
    "dns_settlement": "batch_return",
    "credit": "account_closed",
    # SWIFT cross-border steps
    "sanctions": "sanctions_hit",
    "fx_conversion": "fx_fail",
    "correspondent_routing": "no_route",
    "cover_check": "no_cover",
    "settlement": "settlement_fail",
    # batch_window raises no exception - it only adds latency (next-batch wait).
}
# sla_breach is not tied to one step; it is raised for INSTANT rails that blow their SLA.
EXCEPTION_CODES = sorted(set(STEP_EXCEPTION.values()) | {"sla_breach"})
TERMINAL_STATUS = ["STP", "REPAIRED", "MANUAL_REVIEW", "REJECTED"]

# per-step service time, seconds (lo, hi). batch_window handled separately.
SERVICE_SEC = {
    "validation": (0.5, 2), "vpa_resolution": (1, 3), "beneficiary_resolution": (1, 3),
    "enrichment": (1, 4), "fraud_risk": (1, 4), "limit_check": (0.2, 1),
    "min_amount_check": (0.2, 1), "aml": (2, 8), "liquidity": (1, 4),
    "npci_switch": (1, 5), "rbi_settlement": (5, 30), "dns_settlement": (10, 60),
    "credit": (1, 5),
    # SWIFT: FX + correspondent hops + slow settlement dominate (hours-days).
    "sanctions": (5, 30), "fx_conversion": (10, 120), "correspondent_routing": (300, 3600),
    "cover_check": (60, 600), "settlement": (3_600, 172_800),
}
NEFT_BATCH_SEC = 30 * 60        # max wait to next half-hourly batch
# probability an instant-rail switch hop times out (-> sla_breach, maybe decline)
TIMEOUT_P = {"UPI": 0.06, "IMPS": 0.04}
# probability a payment resolves (repaired) vs halts, per exception.
RESOLVE_P = {
    "format_error": 0.75, "vpa_not_found": 0.30, "beneficiary_unreachable": 0.35,
    "fraud_hold": 0.40, "limit_exceeded": 0.0, "below_min": 0.0, "sanctions_hit": 0.65,
    "insufficient_liquidity": 0.55, "technical_decline": 0.50, "settlement_fail": 0.50,
    "batch_return": 0.45, "account_closed": 0.0,
    "fx_fail": 0.50, "no_route": 0.45, "no_cover": 0.45,
}
# ISO 20022 SttlmMtd per rail (grounded: RTGS=gross via own account, rest=clearing,
# SWIFT cross-border via cover method).
RAIL_STTLM = {"RTGS": "INGA", "NEFT": "CLRG", "IMPS": "CLRG", "UPI": "CLRG",
              "SWIFT": "COVE"}


@dataclass
class IndiaConfig:
    num_accounts: int = 4000
    num_payments: int = 60000
    start_date: str = "2023-01-01"
    horizon_days: int = 365
    amount_log_mu: float = 9.0       # ~ exp(9) ~ Rs 8k base; heavy tail spans all rails
    amount_log_sigma: float = 1.6
    # share of LARGE (>cap-eligible) domestic payments deliberately mis-routed to a capped
    # rail, so limit_exceeded is a non-trivial learnable exception. Amplified vs reality
    # (real over-limit attempts are rarer) - a documented synthetic choice.
    over_cap_frac: float = 0.40
    under_min_rtgs_frac: float = 0.04  # share that attempt RTGS below the Rs 2L floor
    # base share of in-flight payments the originator recalls (camt.056). Feature-modulated:
    # higher for very large or fraud-flagged payments (so cancel-likelihood is learnable).
    cancel_frac: float = 0.03
    # C11: an account's PRIOR recalls elevate its future recall hazard (account-level
    # clustering a date-ordered entity timeline can detect). 0.0 = OFF = byte-identical
    # corpora to before the knob existed (multiplier 1, no extra rng draws).
    recall_momentum: float = 0.0
    # TEMPORAL CORRELATION hook: >0 makes an account's exception probability rise after its
    # previous payment hit an exception (heat x(1+m), decaying toward 1 when clean). Gives
    # account HISTORY predictive power over the next payment's outcome - the data change the
    # sequence-encoder eval needs. DEFAULT 0.0 = off: published artifacts stay byte-identical.
    exception_momentum: float = 0.0
    # CADENCE hook: fraction of domestic accounts that carry STANDING PATTERNS (salary credit,
    # rent/EMI, utility, weekly supplier) - periodic dates + stable amounts + fixed
    # counterparties. This is what makes next-payment/receipt prediction learnable: the
    # default uniform-random dates carry no periodicity at all. DEFAULT 0.0 = off:
    # published artifacts stay byte-identical.
    cadence_frac: float = 0.0
    inward_fraction: float = 0.45
    # domestic rails enabled for this run. UPI is off by default for now (reversible - the
    # registry still defines it); set to include "UPI" to bring the fourth rail back.
    domestic_rails: tuple = ("RTGS", "NEFT", "IMPS")
    xborder_frac: float = 0.18       # share of payments that are cross-border (-> SWIFT)
    swift_amount_log_mu: float = 11.5    # ~ exp(11.5) ~ Rs 1L; cross-border skews larger
    swift_amount_log_sigma: float = 1.3
    seed: int = 23


def india_accounts(rng, n_parents, seed):
    """v1 accounts, forced domestic (country=IN, currency=INR)."""
    accs = generate_accounts(rng, GenConfig(num_parents=n_parents, seed=seed))
    return [replace(a, country="IN", currency="INR") for a in accs]


_FOREIGN = [c for c in COUNTRIES if c != "IN"]


def foreign_accounts(rng, n_parents, seed):
    """v1 accounts forced to a non-IN country (SWIFT counterparties)."""
    accs = generate_accounts(rng, GenConfig(num_parents=n_parents, seed=seed + 1))
    out = []
    for a in accs:
        c = str(rng.choice(_FOREIGN))
        out.append(replace(a, country=c, currency=COUNTRIES[c][0]))
    return out


def _factors(src, dest, amount):
    return {
        "big": amount > 200_000, "huge": amount > 1_000_000,
        "ind": src.industry in {"Financial", "Energy"}
        or dest.industry in {"Financial", "Energy"},
    }


def _exception_prob(step, f, rng):
    p = 0.03
    if step == "validation":
        p = 0.04
    elif step == "vpa_resolution":
        p = 0.06
    elif step == "beneficiary_resolution":
        p = 0.05
    elif step == "enrichment":
        p = 0.04
    elif step == "fraud_risk":
        p = 0.03 + 0.08 * f["big"] + 0.10 * f["huge"] + 0.04 * f["ind"]
    elif step == "aml":
        p = 0.04 + 0.05 * f["ind"] + 0.04 * f["huge"]
    elif step == "liquidity":
        p = 0.05 + 0.10 * f["big"] + 0.15 * f["huge"]
    elif step == "npci_switch":
        p = 0.05            # baseline technical decline; timeout handled separately
    elif step == "rbi_settlement":
        p = 0.03
    elif step == "dns_settlement":
        p = 0.04
    elif step == "credit":
        p = 0.03
    # SWIFT cross-border steps (sanctions screening hits harder cross-border)
    elif step == "sanctions":
        p = 0.06 + 0.06 * f["ind"] + 0.05 * f["huge"]
    elif step == "fx_conversion":
        p = 0.04
    elif step == "correspondent_routing":
        p = 0.06
    elif step == "cover_check":
        p = 0.05 + 0.05 * f["huge"]
    elif step == "settlement":
        p = 0.03
    return float(np.clip(p + rng.normal(0, 0.015), 0.005, 0.95))


def _service(step, rng):
    lo, hi = SERVICE_SEC[step]
    return float(rng.uniform(lo, hi))


def simulate_payment(rail, src, dest, amount, rng, heat=1.0):
    """Traverse the rail workflow; return (events, exception_set, status, seconds).
    heat scales the random exception probabilities (account momentum; 1.0 = neutral)."""
    f = _factors(src, dest, amount)
    t = 0.0
    events, exceptions = [], set()
    repaired = False
    completed = True
    status = "STP"

    for step in WORKFLOW[rail]:
        # latency-only step: NEFT batch wait, no exception.
        if step == "batch_window":
            t += float(rng.uniform(0, NEFT_BATCH_SEC))
            events.append((step, "clean", round(t, 1)))
            continue

        t += _service(step, rng)

        # deterministic hard checks
        if step == "limit_check" and violates_cap(rail, amount):
            exceptions.add("limit_exceeded")
            events.append((step, "limit_exceeded", round(t, 1)))
            status = "REJECTED"
            completed = False
            break
        if step == "min_amount_check" and below_min(rail, amount):
            exceptions.add("below_min")
            events.append((step, "below_min", round(t, 1)))
            status = "REJECTED"
            completed = False
            break
        # limit_check / min_amount_check are pure deterministic gates: if the hard
        # check above didn't trip, they pass cleanly (no random exception draw).
        if step in ("limit_check", "min_amount_check"):
            events.append((step, "clean", round(t, 1)))
            continue

        # instant-rail switch can time out -> sla_breach (+ maybe decline)
        if step == "npci_switch" and rng.random() < TIMEOUT_P.get(rail, 0.0):
            exceptions.add("sla_breach")
            t += float(rng.uniform(60, 300))
            events.append((step, "sla_breach", round(t, 1)))
            if rng.random() < 0.5:                     # timeout becomes a decline
                exceptions.add("technical_decline")
                events.append((step, "technical_decline", round(t, 1)))
                status = "MANUAL_REVIEW"
                completed = False
                break
            repaired = True                            # delayed but settled
            continue

        # random feature-driven exception for this step (scaled by account heat)
        if rng.random() < min(0.95, _exception_prob(step, f, rng) * heat):
            exc = STEP_EXCEPTION[step]
            exceptions.add(exc)
            events.append((step, exc, round(t, 1)))
            if rng.random() < RESOLVE_P.get(exc, 0.5):
                t += float(rng.uniform(30, 300))
                events.append((step, "repaired", round(t, 1)))
                repaired = True
            else:
                status = "REJECTED" if rng.random() < 0.30 else "MANUAL_REVIEW"
                completed = False
                break
        else:
            events.append((step, "clean", round(t, 1)))

    if completed:
        status = "REPAIRED" if repaired else "STP"
    return events, exceptions, status, round(t, 1)


def _route_domestic(amount, rng, cfg):
    """Pick the DOMESTIC rail attempted, incl. injected over-cap / under-min errors.

    Returns (rail, identifier_type). Most payments route to an eligible rail; a small
    fraction deliberately attempt an over-cap or below-min rail to create the
    limit_exceeded / below_min exceptions the twin must catch. Routing is restricted to
    cfg.domestic_rails (UPI is off by default now). NOTE: with UPI off the only capped
    domestic rail is IMPS (Rs 5L), so over-cap attempts require amount > 5L and are rarer -
    limit_exceeded prevalence drops accordingly (a consequence of dropping UPI, not a bug).
    """
    allow = set(cfg.domestic_rails)
    # over-cap wrong-rail attempt: pick an enabled capped rail whose cap this amount blows.
    over = [r for r in cfg.domestic_rails
            if RAILS[r].cap is not None and amount > RAILS[r].cap]
    if over and rng.random() < cfg.over_cap_frac:
        rail = min(over, key=lambda r: RAILS[r].cap)       # tightest cap it violates
        return rail, sample_identifier(rail, rng)
    if amount < 200_000 and "RTGS" in allow and rng.random() < cfg.under_min_rtgs_frac:
        return "RTGS", "ACCT_IFSC"                         # below-floor RTGS attempt
    rail = choose_rail(amount, rng, allow=allow)
    return rail, sample_identifier(rail, rng)


def _cadence_payments(cfg, rng, in_accs, start):
    """Standing patterns for a fraction of accounts -> list of scheduled payment dicts.

    salary   monthly INWARD credit (fixed employer, fixed day, amount +-1%)
    rent     monthly outward (fixed payee, fixed day, amount +-2%)
    utility  monthly outward (fixed payee, day +-3, amount +-10%)
    supplier weekly outward (fixed payee, 7d +-1, amount +-5%)
    """
    n_in = len(in_accs)
    n_sel = int(n_in * cfg.cadence_frac)
    sel = rng.choice(n_in, size=n_sel, replace=False) if n_sel else []
    months = max(1, cfg.horizon_days // 30)
    out = []

    def _emit(pattern, src, dest, base, jitter, period, day_jitter):
        day = int(rng.integers(1, 28))
        t = day
        for k in range(months * (30 // period)):
            d = start + timedelta(days=int(t + rng.integers(-day_jitter, day_jitter + 1))
                                  if day_jitter else int(t))
            if (d - start).days >= cfg.horizon_days:
                break
            amt = round(float(base * (1 + rng.normal(0, jitter))), 2)
            out.append({"src": src, "dest": dest, "amount": max(100.0, amt),
                        "dte": d.isoformat(), "cadence": pattern})
            t += period
    for i in sel:
        me = in_accs[int(i)]
        pats = rng.choice(["salary", "rent", "utility", "supplier"],
                          size=int(rng.integers(1, 4)), replace=False)
        for p in pats:
            other = in_accs[int(rng.integers(n_in))]
            if other.account_id == me.account_id:
                continue
            if p == "salary":
                _emit("salary", other, me, float(rng.uniform(50_000, 200_000)), 0.01, 30, 0)
            elif p == "rent":
                _emit("rent", me, other, float(rng.uniform(10_000, 50_000)), 0.02, 30, 0)
            elif p == "utility":
                _emit("utility", me, other, float(rng.uniform(500, 5_000)), 0.10, 30, 3)
            else:
                _emit("supplier", me, other, float(rng.uniform(5_000, 100_000)), 0.05, 7, 1)
    rng.shuffle(out)
    # cap so ad-hoc payments still exist alongside the standing ones
    return out[: int(cfg.num_payments * 0.7)]


def _finish_row(row, pid, rail, identifier, amount, xborder, direction, status,
                events, exceptions, seconds, src, dest, cfg, rng, cadence="none",
                prior_recalls=0):
    """Shared per-payment labelling: rail/instrument metadata, amount split (FX + charges),
    mis-routed flag, reject reason, cancellation/return legs, exception flags, cadence."""
    row["payment_id"] = pid
    row["rail"] = rail
    row["identifier_type"] = identifier
    row["settlement_kind"] = RAILS[rail].settlement
    row["cadence"] = cadence
    # amount split (Track D P1): settlement leg untouched; instructed leg via FX + charges.
    # fx_rate is market noise (unpredictable); charges is bps-of-amount (regression target).
    counter_ccy = dest.currency
    if src.currency == counter_ccy:                          # domestic / same currency
        fx_rate = 1.0
        charges = min(25.0, max(5.0, row["IntrBkSttlmAmt"] * 0.0005))
    else:                                                    # cross-border FX
        fx_rate = round(float(np.exp(rng.normal(0.0, 0.4))), 4)
        charges = row["IntrBkSttlmAmt"] * float(rng.uniform(0.001, 0.005)) + 500.0
    row["InstdCcy"] = counter_ccy
    row["InstdAmt"] = round(row["IntrBkSttlmAmt"] * fx_rate, 2)
    row["fx_rate"] = fx_rate
    row["charges"] = round(charges, 2)
    # mis-routed = the ATTEMPTED rail isn't eligible (injected over-cap / under-min rows).
    _id = identifier if identifier in IDENTIFIER_TYPES else None
    row["is_mis_routed"] = int(rail not in eligible_rails(amount, _id, xborder))
    row["direction"] = direction
    row["terminal_status"] = status
    # ISO status-reason code; only non-settled payments carry one.
    halt_exc = next((e[1] for e in reversed(events) if e[1] in exceptions), None)
    halt_step = next((e[0] for e in reversed(events) if e[1] in exceptions), None)
    row["reject_reason"] = (LIFE_REASON.get(halt_exc, "NARR")
                            if status in ("REJECTED", "MANUAL_REVIEW") else "none")
    # cancellation (camt.056) + return (pacs.004) labels; recallable once past submission.
    reached_clearing = not (status == "REJECTED" and halt_step in PRE_SUBMISSION)
    p_cancel = cfg.cancel_frac * (1 + 2 * int(amount > 1_000_000)
                                  + int("fraud_hold" in exceptions)) \
        * (1 + cfg.recall_momentum * prior_recalls)
    cancel_requested = reached_clearing and rng.random() < min(p_cancel, 0.5)
    cancel_status = "none"
    if cancel_requested:
        if status == "MANUAL_REVIEW":                 # held -> easy to stop, never credited
            cancel_status = "CNCL"
        elif status in ("STP", "REPAIRED"):           # already credited -> recall may be late
            cancel_status = "CNCL" if rng.random() < 0.35 else "RJCR"
        else:                                         # REJECTED -> nothing to recall
            cancel_status = "CNCL"
    auto_return = status == "REJECTED" and halt_exc == "account_closed"
    recall_return = cancel_requested and cancel_status == "CNCL" and status in ("STP", "REPAIRED")
    row["cancel_requested"] = int(cancel_requested)
    row["cancel_status"] = cancel_status
    row["returned"] = int(auto_return or recall_return)
    row["return_reason"] = (LIFE_REASON["account_closed"] if auto_return
                            else ("CUST" if recall_return else "none"))
    row["time_to_settle_min"] = round(seconds / 60.0, 3)
    for code in EXCEPTION_CODES:
        row[f"exc_{code}"] = int(code in exceptions)
    return row


def build_dataset(cfg: IndiaConfig):
    rng = np.random.default_rng(cfg.seed)
    in_accs = india_accounts(rng, cfg.num_accounts, cfg.seed)
    fgn_accs = foreign_accounts(rng, max(1, cfg.num_accounts // 4), cfg.seed)
    start = date.fromisoformat(cfg.start_date)
    n_in, n_fgn = len(in_accs), len(fgn_accs)
    scheduled = _cadence_payments(cfg, rng, in_accs, start) if cfg.cadence_frac > 0 else []

    pay_rows, evt_rows = [], []
    acct_heat: dict = {}                    # account -> exception-momentum multiplier
    acct_recalls: dict = {}                 # account -> prior cancel_requested count (C11)
    pid = 0
    while len(pay_rows) < cfg.num_payments:
        if scheduled:                        # standing patterns first, then ad-hoc fill
            s = scheduled.pop()
            src, dest, amount, dte, cadence = s["src"], s["dest"], s["amount"], s["dte"], s["cadence"]
            rail, identifier = _route_domestic(amount, rng, cfg)
            direction = "inward" if cadence == "salary" else "outward"
            heat = acct_heat.get(src.account_id, 1.0)
            events, exceptions, status, seconds = simulate_payment(rail, src, dest, amount,
                                                                   rng, heat=heat)
            acct_heat[src.account_id] = (min(3.0, heat * (1 + cfg.exception_momentum))
                                         if exceptions else 1.0 + (heat - 1.0) * 0.5)
            row = project_to_pacs008(src, dest, amount, dte, RAIL_STTLM[rail],
                                     assign_risk(src, dest, amount, rng),
                                     assign_geo(src, dest), assign_expense(dest), "No", pid)
            _finish_row(row, pid, rail, identifier, amount, False, direction, status,
                        events, exceptions, seconds, src, dest, cfg, rng, cadence=cadence,
                        prior_recalls=acct_recalls.get(src.account_id, 0))
            acct_recalls[src.account_id] = (acct_recalls.get(src.account_id, 0)
                                            + row["cancel_requested"])
            pay_rows.append(row)
            for seq, (step, outcome, tsec) in enumerate(events):
                evt_rows.append({"payment_id": pid, "seq": seq, "step": step,
                                 "outcome": outcome,
                                 "excode": outcome if outcome in EXCEPTION_CODES else "none",
                                 "rail": rail, "t_min": round(tsec / 60.0, 3)})
            pid += 1
            continue
        xborder = rng.random() < cfg.xborder_frac
        direction = "inward" if rng.random() < cfg.inward_fraction else "outward"

        if xborder:
            # SWIFT: one leg in India, the other abroad (FX). Different pools => distinct.
            amount = round(float(np.exp(rng.normal(
                cfg.swift_amount_log_mu, cfg.swift_amount_log_sigma))), 2)
            if direction == "outward":
                src, dest = in_accs[rng.integers(n_in)], fgn_accs[rng.integers(n_fgn)]
            else:
                src, dest = fgn_accs[rng.integers(n_fgn)], in_accs[rng.integers(n_in)]
            rail, identifier = "SWIFT", "BIC_IBAN"
        else:
            src, dest = in_accs[rng.integers(n_in)], in_accs[rng.integers(n_in)]
            if dest.account_id == src.account_id:
                continue
            amount = round(float(np.exp(rng.normal(
                cfg.amount_log_mu, cfg.amount_log_sigma))), 2)
            rail, identifier = _route_domestic(amount, rng, cfg)

        dte = (start + timedelta(days=int(rng.integers(0, cfg.horizon_days)))).isoformat()

        heat = acct_heat.get(src.account_id, 1.0)
        events, exceptions, status, seconds = simulate_payment(rail, src, dest, amount, rng,
                                                               heat=heat)
        # account momentum: exceptions raise the account's heat, clean payments cool it.
        acct_heat[src.account_id] = (min(3.0, heat * (1 + cfg.exception_momentum))
                                     if exceptions else 1.0 + (heat - 1.0) * 0.5)

        row = project_to_pacs008(src, dest, amount, dte, RAIL_STTLM[rail],
                                 assign_risk(src, dest, amount, rng), assign_geo(src, dest),
                                 assign_expense(dest), "No", pid)
        _finish_row(row, pid, rail, identifier, amount, xborder, direction, status,
                    events, exceptions, seconds, src, dest, cfg, rng,
                    prior_recalls=acct_recalls.get(src.account_id, 0))
        acct_recalls[src.account_id] = (acct_recalls.get(src.account_id, 0)
                                        + row["cancel_requested"])
        pay_rows.append(row)

        for seq, (step, outcome, tsec) in enumerate(events):
            evt_rows.append({"payment_id": pid, "seq": seq, "step": step,
                             "outcome": outcome,
                             "excode": outcome if outcome in EXCEPTION_CODES else "none",
                             "rail": rail, "t_min": round(tsec / 60.0, 3)})
        pid += 1

    return pd.DataFrame(pay_rows), pd.DataFrame(evt_rows), in_accs + fgn_accs


def build_messages(pay_df, evt_df) -> pd.DataFrame:
    """Derive the ISO 20022 message-lifecycle table (multi-source) from the payment + event
    tables: one row per (payment, message) sharing an end_to_end_id. Post-hoc so build_dataset's
    signature is untouched. The halting cause per payment comes from its event log."""
    from data.iso_lifecycle import lifecycle_messages

    ev_by_pid = {pid: list(zip(g["step"], g["excode"], g["t_min"]))
                 for pid, g in evt_df.groupby("payment_id", sort=False)}
    rows = []
    for pr in pay_df.to_dict("records"):
        events = ev_by_pid.get(pr["payment_id"], [])
        rows.extend(lifecycle_messages(pr, pr["terminal_status"], events))
    return pd.DataFrame(rows)


# Downstream task manifest (read from schema; never hard-coded by trainers - paper rule).
def _tasks():
    from data.synth_pacs008 import EXPENSE_TYPES, GEO_SPANS
    return [
        {"name": "risk", "label_column": "risk_label",
         "label_values": ["Low", "Medium", "High"],
         "metric": "imbalance", "positive_class": "High", "records": "single"},
        {"name": "rail_routing", "label_column": "rail",
         "label_values": RAIL_NAMES, "metric": "multiclass", "records": "single"},
        # §5 single-record tasks. In India mode geography is limited to Asia (domestic) vs
        # International (cross-border SWIFT); expense varies by creditor industry.
        # NOTE: geo_label (assign_geo) and expense_label (assign_expense) are DETERMINISTIC
        # (no rng) - Geo-Cover is a country rule, expense an industry lookup. In the PRODUCT
        # twin these are rule/lookup-handled (delisted as TFM predictions). They are KEPT here
        # unchanged because they are the paper's §5 tagging tasks and the evidence for C2
        # (CatBoost beats the frozen adapter on exactly such feature-rule labels). Fidelity: do
        # not remove. See docs/CBPR_TWIN_GAP.md "TFM-owned vs rule/lookup/template".
        {"name": "geography", "label_column": "geo_label",
         "label_values": GEO_SPANS, "metric": "multiclass", "records": "single"},
        {"name": "expense", "label_column": "expense_label",
         "label_values": EXPENSE_TYPES, "metric": "multiclass", "records": "single"},
    ]


def _lifecycle_block(msg_df=None) -> dict:
    from data.iso_lifecycle import ENRICH_ADDS, GPI_SUBCODES, MSG_TYPES, OWNED, REASON, TX_STS
    block = {
        "msg_types": MSG_TYPES, "tx_sts": TX_STS,
        "enrich_adds": ENRICH_ADDS,                          # nested availability (imputation)
        "owned_columns": {m: sorted(OWNED[m]) for m in MSG_TYPES},
        "reason_codes": REASON,
        # pacs.002 IS the gpi tracker in-flight backbone.
        "gpi_tracker": {"status_codes": ["ACSC", "ACSP", "RJCT"], "subcodes": GPI_SUBCODES,
                        "status_message": "pacs.002", "recall_resolution": "camt.029 (CNCL/RJCR)"},
        "id_column": "end_to_end_id", "type_column": "msg_type", "status_column": "tx_sts",
    }
    if msg_df is not None and len(msg_df):
        block["msg_type_distribution"] = msg_df["msg_type"].value_counts().to_dict()
        block["tx_sts_distribution"] = msg_df["tx_sts"].value_counts().to_dict()
    return block


def build_schema(pay_df, accs, msg_df=None) -> dict:
    # identifier_type is a legitimate intake feature; rail / settlement_kind / SttlmMtd are
    # NOT features for routing (they are the label or 1:1 consequences of it - SttlmMtd is
    # derived from the rail here, so keeping it would leak the label). Drop SttlmMtd from
    # the core bucket and add identifier_type.
    core = [c for c in COLUMN_BUCKETS["core"] if c != "SttlmMtd"] + ["identifier_type"]
    buckets = {**COLUMN_BUCKETS, "core": core}
    return {
        "mode": "india_rails", "buckets": buckets,
        "label_column": "risk_label", "label_values": ["Low", "Medium", "High"],
        "tasks": _tasks(),
        "twin": {
            "rails": RAIL_NAMES, "workflow": WORKFLOW,
            "exception_codes": EXCEPTION_CODES, "terminal_status": TERMINAL_STATUS,
            "exc_columns": [f"exc_{c}" for c in EXCEPTION_CODES],
            "status_column": "terminal_status", "eta_column": "time_to_settle_min",
            "rail_column": "rail", "id_column": "payment_id",
            # Twin binary tasks are STOCHASTIC, feature-driven exceptions (genuine predictions).
            # DELISTED as TFM targets: exc_limit_exceeded / exc_below_min are deterministic hard
            # gates (violates_cap / below_min) - compute them by rule, don't predict them. So is
            # is_mis_routed (rail not in eligible_rails). sla_breach (TIMEOUT_P) and fraud_hold
            # (feature-driven probability) are real predictions and stay.
            "twin_binary_tasks": ["exc_sla_breach", "exc_fraud_hold"],
            "rule_computed": ["exc_limit_exceeded", "exc_below_min", "is_mis_routed",
                              "settlement_kind", "SttlmMtd"],
        },
        "lifecycle": _lifecycle_block(msg_df),
        # two-currency + charges metadata (labels for a future FX/charges head; not features).
        "financial_columns": ["InstdAmt", "InstdCcy", "fx_rate", "charges"],
        "n_payments": int(len(pay_df)), "n_accounts": len(accs),
        "vocab": vocab_report(pay_df),
        "rail_distribution": pay_df["rail"].value_counts().to_dict(),
        "currency_distribution": pay_df["Ccy"].value_counts().to_dict(),
        "status_distribution": pay_df["terminal_status"].value_counts().to_dict(),
        "exception_rates": {c: float(pay_df[f"exc_{c}"].mean()) for c in EXCEPTION_CODES},
        "eta_min_by_rail": pay_df.groupby("rail")["time_to_settle_min"].mean().round(2).to_dict(),
    }


def main():
    ap = argparse.ArgumentParser(
        description="India multi-rail (RTGS/NEFT/IMPS/UPI + cross-border SWIFT) generator")
    ap.add_argument("--accounts", type=int, default=IndiaConfig.num_accounts)
    ap.add_argument("--payments", type=int, default=IndiaConfig.num_payments)
    ap.add_argument("--seed", type=int, default=IndiaConfig.seed)
    ap.add_argument("--out-prefix", default="india_rails")
    ap.add_argument("--schema-out", default="column_schema_india.json")
    args = ap.parse_args()

    cfg = IndiaConfig(num_accounts=args.accounts, num_payments=args.payments, seed=args.seed)
    pay_df, evt_df, accs = build_dataset(cfg)
    msg_df = build_messages(pay_df, evt_df)
    for df, suffix in [(pay_df, "payments"), (evt_df, "events"), (msg_df, "messages")]:
        path = f"{args.out_prefix}_{suffix}.parquet"
        try:
            df.to_parquet(path, index=False)
        except Exception:
            path = path.replace(".parquet", ".csv"); df.to_csv(path, index=False)
        print(f"wrote {len(df):,} rows -> {path}")
    schema = build_schema(pay_df, accs, msg_df)
    Path(args.schema_out).write_text(json.dumps(schema, indent=2))
    print(f"rails:  {schema['rail_distribution']}")
    print(f"ccy:    {schema['currency_distribution']}")
    print(f"status: {schema['status_distribution']}")
    print(f"ETA min/rail: {schema['eta_min_by_rail']}")
    print(f"exc rates: { {k: round(v,3) for k,v in schema['exception_rates'].items()} }")


if __name__ == "__main__":
    main()
