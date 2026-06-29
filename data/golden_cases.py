"""
Curated GOLDEN test cases - one shared, deterministic set of payments scored across ALL
tasks (the §5 single-record tasks risk / geography / expense + the India rail tasks: rail
routing, sla_breach / limit_exceeded, terminal status, ETA, in-flight next-exception).

Why a golden set: each task is otherwise exercised on its own random split. This is ONE
hand-curated bench that hits every rail (RTGS/NEFT/IMPS/UPI/SWIFT), the deterministic gate
exceptions (over-cap -> limit_exceeded, below-min -> below_min), a few stochastic exceptions
(sla_breach / sanctions / fraud / batch_return, found by seed search), and risk/expense/geo
variety - so all tasks can be eyeballed / regressed on identical inputs.

Labels are NOT hand-typed: they are computed by the real rule functions (assign_risk /
assign_geo / assign_expense) and the real lifecycle simulator (simulate_payment), so the
golden labels can never drift from the codebase. The set also serialises to pacs.008 XML
(data/iso20022_pacs008.write_pacs008) to exercise the live Layer-1 path.

NOTE: recurrence (§5) is multi-record / group-level, so it does NOT fit a per-payment golden
row and is intentionally out of this bench.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from data.synth_pacs008 import (
    Account, COUNTRIES, assign_expense, assign_geo, assign_risk, project_to_pacs008,
)
from data.synth_india_rails import (
    EXCEPTION_CODES, RAIL_STTLM, RAILS, simulate_payment,
)

DATE = "2026-06-29"


def _acct(country, industry, sub, tag):
    return Account(account_id=f"AC{tag}", account_name=f"{tag} Ops",
                   parent_id=f"PR{tag}", parent_name=f"{tag} Group",
                   industry=industry, sub_industry=sub,
                   country=country, currency=COUNTRIES[country][0])


def _has(code):
    return lambda exc, status: code in exc


def _clean(exc, status):
    return status == "STP" and not exc


# Each case: (name, src(ctry,ind,sub), dest(ctry,ind,sub), amount, rail, identifier, target)
# target is a predicate over (exception_set, status) used to seed-search a realistic,
# reproducible outcome; None = take the first seed.
CASES = [
    ("upi_small_clean",   ("IN", "Consumer", "Retail"),   ("IN", "Consumer", "Food"),
     4_800,    "UPI",  "VPA",         _clean),
    ("imps_mid_clean",    ("IN", "Consumer", "Retail"),   ("IN", "Industrials", "Logistics"),
     80_000,   "IMPS", "ACCT_IFSC",   _clean),
    ("neft_mid_clean",    ("IN", "Industrials", "Manufacturing"), ("IN", "Consumer", "Apparel"),
     150_000,  "NEFT", "ACCT_IFSC",   _clean),
    ("rtgs_large_clean",  ("IN", "Financial", "Banks"),   ("IN", "Industrials", "Construction"),
     500_000,  "RTGS", "ACCT_IFSC",   _clean),
    ("swift_xborder_clean", ("IN", "Technology", "Software"), ("US", "Technology", "Hardware"),
     125_000,  "SWIFT", "BIC_IBAN",   _clean),
    ("upi_over_cap",      ("IN", "Consumer", "Retail"),   ("IN", "Consumer", "Food"),
     150_000,  "UPI",  "VPA",         _has("limit_exceeded")),    # > Rs 1L cap
    ("rtgs_under_min",    ("IN", "Consumer", "Retail"),   ("IN", "Consumer", "Food"),
     50_000,   "RTGS", "ACCT_IFSC",   _has("below_min")),         # < Rs 2L floor
    ("upi_sla_breach",    ("IN", "Consumer", "Retail"),   ("IN", "Consumer", "Food"),
     9_500,    "UPI",  "VPA",         _has("sla_breach")),
    ("swift_sanctions",   ("IN", "Financial", "Banks"),   ("AE", "Financial", "Insurance"),
     900_000,  "SWIFT", "BIC_IBAN",   _has("sanctions_hit")),
    ("neft_batch_return", ("IN", "Industrials", "Logistics"), ("IN", "Consumer", "Retail"),
     60_000,   "NEFT", "ACCT_IFSC",   _has("batch_return")),
    ("imps_fraud",        ("IN", "Energy", "Oil"),        ("IN", "Financial", "AssetMgmt"),
     450_000,  "IMPS", "MMID_MOBILE", _has("fraud_hold")),
    ("rtgs_high_risk",    ("IN", "Financial", "Banks"),   ("IN", "Energy", "Oil"),
     3_000_000, "RTGS", "ACCT_IFSC",  _clean),
    ("swift_tech_expense", ("IN", "Consumer", "Retail"),  ("DE", "Technology", "Semiconductors"),
     220_000,  "SWIFT", "BIC_IBAN",   None),
    ("upi_capital_expense", ("IN", "Consumer", "Retail"), ("IN", "Energy", "Renewables"),
     40_000,   "UPI",  "VPA",         None),
]


def _simulate(rail, src, dest, amount, target, max_seeds=800):
    last = None
    for s in range(max_seeds):
        rng = np.random.default_rng(s)
        ev, exc, status, secs = simulate_payment(rail, src, dest, amount, rng)
        last = (ev, exc, status, secs, s)
        if target is None or target(exc, status):
            return last
    return last                                                 # best effort if not hit


def build_golden():
    """Return (pay_df, evt_df) - the curated cases with all task labels + event log."""
    pay_rows, evt_rows = [], []
    for i, (name, (sc, si, ss), (dc, di, ds), amount, rail, ident, target) in enumerate(CASES):
        src = _acct(sc, si, ss, f"S{i}")
        dest = _acct(dc, di, ds, f"D{i}")
        rng_r = np.random.default_rng(1000 + i)                 # deterministic risk noise
        risk = assign_risk(src, dest, amount, rng_r)
        ev, exc, status, secs, seed = _simulate(rail, src, dest, amount, target)

        row = project_to_pacs008(src, dest, float(amount), DATE, RAIL_STTLM[rail],
                                 risk, assign_geo(src, dest), assign_expense(dest), "No", i)
        row["case"] = name
        row["payment_id"] = i
        row["rail"] = rail
        row["rail_family"] = rail
        row["identifier_type"] = ident
        row["settlement_kind"] = RAILS[rail].settlement
        row["direction"] = "outward" if src.country == "IN" else "inward"
        row["terminal_status"] = status
        row["time_to_settle_min"] = round(secs / 60.0, 3)
        for code in EXCEPTION_CODES:
            row[f"exc_{code}"] = int(code in exc)
        pay_rows.append(row)

        for seq, (step, outcome, tsec) in enumerate(ev):
            evt_rows.append({"payment_id": i, "seq": seq, "step": step, "outcome": outcome,
                             "excode": outcome if outcome in EXCEPTION_CODES else "none",
                             "rail": rail, "t_min": round(tsec / 60.0, 3)})
    return pd.DataFrame(pay_rows), pd.DataFrame(evt_rows)


def golden_pacs008_xml():
    """The golden payments serialised as one pacs.008 message (message-native fields)."""
    from data.iso20022_pacs008 import write_pacs008
    pay, _ = build_golden()
    return write_pacs008(pay.to_dict(orient="records"))


def main():
    import argparse
    ap = argparse.ArgumentParser(description="Build the curated cross-task golden set")
    ap.add_argument("--xml-out", default=None, help="also write the set as pacs.008 XML")
    args = ap.parse_args()
    pay, evt = build_golden()
    print(f"{len(pay)} golden cases | {len(evt)} events")
    print(pay[["case", "rail", "IntrBkSttlmAmt", "Ccy", "risk_label", "geo_label",
               "expense_label", "terminal_status", "time_to_settle_min"]].to_string(index=False))
    if args.xml_out:
        from data.iso20022_pacs008 import write_pacs008
        open(args.xml_out, "w").write(write_pacs008(pay.to_dict(orient="records")))
        print(f"wrote {args.xml_out}")


if __name__ == "__main__":
    main()
