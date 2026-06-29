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
    RAILS, RAIL_NAMES, below_min, choose_rail, sample_identifier, violates_cap,
)

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
    inward_fraction: float = 0.45
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


def simulate_payment(rail, src, dest, amount, rng):
    """Traverse the rail workflow; return (events, exception_set, status, seconds)."""
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

        # random feature-driven exception for this step
        if rng.random() < _exception_prob(step, f, rng):
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
    limit_exceeded / below_min exceptions the twin must catch.
    """
    if amount > 100_000 and rng.random() < cfg.over_cap_frac:
        rail = "UPI" if amount <= 5_000_000 else "IMPS"   # wrong-rail attempt
        return rail, sample_identifier(rail, rng)
    if amount < 200_000 and rng.random() < cfg.under_min_rtgs_frac:
        return "RTGS", "ACCT_IFSC"                         # below-floor RTGS attempt
    rail = choose_rail(amount, rng)
    return rail, sample_identifier(rail, rng)


def build_dataset(cfg: IndiaConfig):
    rng = np.random.default_rng(cfg.seed)
    in_accs = india_accounts(rng, cfg.num_accounts, cfg.seed)
    fgn_accs = foreign_accounts(rng, max(1, cfg.num_accounts // 4), cfg.seed)
    start = date.fromisoformat(cfg.start_date)
    n_in, n_fgn = len(in_accs), len(fgn_accs)

    pay_rows, evt_rows = [], []
    pid = 0
    while len(pay_rows) < cfg.num_payments:
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

        events, exceptions, status, seconds = simulate_payment(rail, src, dest, amount, rng)

        row = project_to_pacs008(src, dest, amount, dte, RAIL_STTLM[rail],
                                 assign_risk(src, dest, amount, rng), assign_geo(src, dest),
                                 assign_expense(dest), "No", pid)
        row["payment_id"] = pid
        row["rail"] = rail
        row["rail_family"] = rail                       # 1:1 in India; kept for parity
        row["identifier_type"] = identifier
        row["settlement_kind"] = RAILS[rail].settlement
        row["direction"] = direction
        row["terminal_status"] = status
        row["time_to_settle_min"] = round(seconds / 60.0, 3)
        for code in EXCEPTION_CODES:
            row[f"exc_{code}"] = int(code in exceptions)
        pay_rows.append(row)

        for seq, (step, outcome, tsec) in enumerate(events):
            evt_rows.append({"payment_id": pid, "seq": seq, "step": step,
                             "outcome": outcome,
                             "excode": outcome if outcome in EXCEPTION_CODES else "none",
                             "rail": rail, "t_min": round(tsec / 60.0, 3)})
        pid += 1

    return pd.DataFrame(pay_rows), pd.DataFrame(evt_rows), in_accs + fgn_accs


# Downstream task manifest (read from schema; never hard-coded by trainers - paper rule).
def _tasks():
    return [
        {"name": "risk", "label_column": "risk_label",
         "label_values": ["Low", "Medium", "High"],
         "metric": "imbalance", "positive_class": "High", "records": "single"},
        {"name": "rail_routing", "label_column": "rail",
         "label_values": RAIL_NAMES, "metric": "multiclass", "records": "single"},
    ]


def build_schema(pay_df, accs) -> dict:
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
            "twin_binary_tasks": ["exc_sla_breach", "exc_limit_exceeded"],
        },
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
    for df, suffix in [(pay_df, "payments"), (evt_df, "events")]:
        path = f"{args.out_prefix}_{suffix}.parquet"
        try:
            df.to_parquet(path, index=False)
        except Exception:
            path = path.replace(".parquet", ".csv"); df.to_csv(path, index=False)
        print(f"wrote {len(df):,} rows -> {path}")
    schema = build_schema(pay_df, accs)
    Path(args.schema_out).write_text(json.dumps(schema, indent=2))
    print(f"rails:  {schema['rail_distribution']}")
    print(f"ccy:    {schema['currency_distribution']}")
    print(f"status: {schema['status_distribution']}")
    print(f"ETA min/rail: {schema['eta_min_by_rail']}")
    print(f"exc rates: { {k: round(v,3) for k,v in schema['exception_rates'].items()} }")


if __name__ == "__main__":
    main()
