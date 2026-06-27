"""
Payment-level digital-twin data - synthetic workflow execution logs.

Beyond arXiv:2410.07851 (a twin application); see docs/PAYMENT_TWIN.md. ADDITIVE: the
v1 generator (synth_pacs008.py) is untouched. Each payment is simulated traversing the
orchestrator workflow - outward (-> SWIFT settlement) or inward (-> account posting) -
where every step is clean or throws a (feature-driven) exception that is then repaired
or halts the payment. Two tables are emitted:

  * payment-level (one row / payment): pacs.008 features + direction + the labels the
    twin predicts at INTAKE - which exceptions occurred (multi-label), terminal status,
    and time-to-settle.
  * event-level (one row / step): the (step, outcome, exception, time) log the twin's
    IN-FLIGHT model consumes as a sequence.

Exception probabilities are transparent functions of the payment's features (like
assign_risk), so they are learnable from the representation - and, honestly, partly
learnable by a tree on raw features too (the recurring lesson). The twin's job here is
to show the backbone CAN forecast the lifecycle, not to beat a tree.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from data.synth_pacs008 import (
    CHANNELS, COUNTRIES, GenConfig, SETTLEMENT_METHODS, assign_expense, assign_geo,
    assign_risk, generate_accounts, project_to_pacs008,
)

# Workflow definitions: ordered steps per direction.
WORKFLOW = {
    "outward": ["validation", "enrichment", "sanctions", "fraud_aml",
                "limit_liquidity", "routing", "settlement"],
    "inward": ["validation", "enrichment", "sanctions", "fraud_aml",
               "account_resolution", "posting"],
}
STEP_EXCEPTION = {
    "validation": "format_error", "enrichment": "missing_field",
    "sanctions": "sanctions_hit", "fraud_aml": "fraud_hold",
    "limit_liquidity": "insufficient_liquidity", "routing": "no_cover",
    "account_resolution": "account_not_found", "settlement": "settlement_fail",
    "posting": "posting_error",
}
EXCEPTION_CODES = sorted(set(STEP_EXCEPTION.values()))
TERMINAL_STATUS = ["STP", "REPAIRED", "MANUAL_REVIEW", "REJECTED"]
SERVICE_MIN = {"validation": 1, "enrichment": 2, "sanctions": 5, "fraud_aml": 3,
               "limit_liquidity": 2, "routing": 4, "settlement": 10,
               "account_resolution": 3, "posting": 5}
RESOLVE_P = {"sanctions_hit": 0.70, "missing_field": 0.85, "fraud_hold": 0.40,
             "format_error": 0.75, "insufficient_liquidity": 0.55, "no_cover": 0.50,
             "account_not_found": 0.45, "settlement_fail": 0.50, "posting_error": 0.55}


@dataclass
class WfConfig:
    num_accounts: int = 4000
    num_payments: int = 60000
    inward_fraction: float = 0.45
    start_date: str = "2023-01-01"
    horizon_days: int = 365
    amount_log_mu: float = 9.0
    amount_log_sigma: float = 1.3
    seed: int = 17


def _factors(src, dest, amount):
    _, sr = COUNTRIES[src.country]
    _, dr = COUNTRIES[dest.country]
    return {
        "xborder": src.country != dest.country,
        "region": (sr in {"EMEA", "Asia"}) or (dr in {"EMEA", "Asia"}),
        "ind": src.industry in {"Financial", "Energy"} or dest.industry in {"Financial", "Energy"},
        "big": amount > 50_000, "huge": amount > 250_000,
    }


def _exception_prob(step, f, rng):
    p = 0.04
    if step == "sanctions":
        p += 0.18 * f["xborder"] + 0.12 * f["region"] + 0.06 * f["ind"]
    elif step == "fraud_aml":
        p += 0.10 * f["big"] + 0.12 * f["huge"] + 0.05 * f["ind"]
    elif step == "limit_liquidity":
        p += 0.10 * f["big"] + 0.15 * f["huge"]
    elif step == "routing":
        p += 0.12 * f["xborder"]
    elif step == "enrichment":
        p += 0.04
    elif step == "account_resolution":
        p += 0.08
    return float(np.clip(p + rng.normal(0, 0.02), 0.01, 0.95))


def simulate_payment(src, dest, amount, direction, rng):
    """Traverse the workflow; return (events, exception_set, terminal_status, minutes)."""
    f = _factors(src, dest, amount)
    t = 0.0
    events, exceptions = [], set()
    repaired = False
    status = "STP"
    completed = True
    for step in WORKFLOW[direction]:
        t += SERVICE_MIN[step] * float(rng.uniform(0.7, 1.5))
        if rng.random() < _exception_prob(step, f, rng):
            exc = STEP_EXCEPTION[step]
            exceptions.add(exc)
            events.append((step, exc, round(t, 1)))
            if rng.random() < RESOLVE_P[exc]:               # repaired, continue
                t += float(rng.uniform(60, 480))
                events.append((step, "repaired", round(t, 1)))
                repaired = True
            else:                                            # halt
                status = "REJECTED" if rng.random() < 0.30 else "MANUAL_REVIEW"
                completed = False
                break
        else:
            events.append((step, "clean", round(t, 1)))
    if completed:
        status = "REPAIRED" if repaired else "STP"
    return events, exceptions, status, round(t, 1)


def build_workflow_dataset(cfg: WfConfig):
    rng = np.random.default_rng(cfg.seed)
    accs = generate_accounts(rng, GenConfig(num_parents=cfg.num_accounts, seed=cfg.seed))
    start = date.fromisoformat(cfg.start_date)
    n = len(accs)

    pay_rows, evt_rows = [], []
    pid = 0
    while len(pay_rows) < cfg.num_payments:
        src = accs[rng.integers(n)]
        dest = accs[rng.integers(n)]
        if dest.account_id == src.account_id:
            continue
        amount = round(float(np.exp(rng.normal(cfg.amount_log_mu, cfg.amount_log_sigma))), 2)
        direction = "inward" if rng.random() < cfg.inward_fraction else "outward"
        day = int(rng.integers(0, cfg.horizon_days))
        dte = (start + timedelta(days=day)).isoformat()
        channel = rng.choice(SETTLEMENT_METHODS) if rng.random() < 0.7 else rng.choice(CHANNELS)

        events, exceptions, status, minutes = simulate_payment(src, dest, amount, direction, rng)

        row = project_to_pacs008(src, dest, amount, dte, channel,
                                 assign_risk(src, dest, amount, rng), assign_geo(src, dest),
                                 assign_expense(dest), "No", pid)
        row["payment_id"] = pid
        row["direction"] = direction
        row["terminal_status"] = status
        row["time_to_settle_min"] = minutes
        for code in EXCEPTION_CODES:
            row[f"exc_{code}"] = int(code in exceptions)
        pay_rows.append(row)

        for seq, (step, outcome, tmin) in enumerate(events):
            evt_rows.append({"payment_id": pid, "seq": seq, "step": step,
                             "outcome": outcome,
                             "excode": outcome if outcome in EXCEPTION_CODES else "none",
                             "t_min": tmin})
        pid += 1

    return pd.DataFrame(pay_rows), pd.DataFrame(evt_rows), accs


def build_schema(pay_df, accs) -> dict:
    from data.synth_pacs008 import COLUMN_BUCKETS, vocab_report
    return {
        "buckets": COLUMN_BUCKETS,
        "label_column": "risk_label", "label_values": ["Low", "Medium", "High"],
        "twin": {
            "directions": list(WORKFLOW), "workflow": WORKFLOW,
            "exception_codes": EXCEPTION_CODES, "terminal_status": TERMINAL_STATUS,
            "exc_columns": [f"exc_{c}" for c in EXCEPTION_CODES],
            "status_column": "terminal_status", "eta_column": "time_to_settle_min",
            "id_column": "payment_id",
        },
        "n_payments": int(len(pay_df)), "n_accounts": len(accs),
        "vocab": vocab_report(pay_df),
        "status_distribution": pay_df["terminal_status"].value_counts().to_dict(),
        "exception_rates": {c: float(pay_df[f"exc_{c}"].mean()) for c in EXCEPTION_CODES},
    }


def main():
    ap = argparse.ArgumentParser(description="payment-level twin - workflow log generator")
    ap.add_argument("--accounts", type=int, default=WfConfig.num_accounts)
    ap.add_argument("--payments", type=int, default=WfConfig.num_payments)
    ap.add_argument("--seed", type=int, default=WfConfig.seed)
    ap.add_argument("--out-prefix", default="pacs008_twin")
    ap.add_argument("--schema-out", default="column_schema_twin.json")
    args = ap.parse_args()

    cfg = WfConfig(num_accounts=args.accounts, num_payments=args.payments, seed=args.seed)
    pay_df, evt_df, accs = build_workflow_dataset(cfg)
    for df, suffix in [(pay_df, "payments"), (evt_df, "events")]:
        path = f"{args.out_prefix}_{suffix}.parquet"
        try:
            df.to_parquet(path, index=False)
        except Exception:
            path = path.replace(".parquet", ".csv"); df.to_csv(path, index=False)
        print(f"wrote {len(df):,} rows -> {path}")
    schema = build_schema(pay_df, accs)
    Path(args.schema_out).write_text(json.dumps(schema, indent=2))
    print(f"status: {schema['status_distribution']}")
    print(f"exception rates: { {k: round(v,3) for k,v in schema['exception_rates'].items()} }")


if __name__ == "__main__":
    main()
