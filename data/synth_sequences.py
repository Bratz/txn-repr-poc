"""
v2 / §7 - synthetic per-account behavioural histories with a regime-change label.

ADDITIVE and SEPARATE from the v1 generator: data/synth_pacs008.py is untouched, so v1
stays byte-identical and faithful. This module reuses v1's account model and pacs.008
projection, but emits an account's payments as a coherent stream over time and tags each
account with a regime label.

The design principle (docs/V2_DIRECTION.md, and the reason C3/C4 are honest):
  * every account draws the SAME number of payments and the SAME amount distribution,
    whatever its class - so aggregate volume and amount do NOT separate the classes.
  * the signal is entirely in the TIMING. A "Stable" account pays at an even cadence; a
    "Shift" account undergoes a regime change in its timing:
        - dormancy -> reactivation : early activity, a long silence, a late cluster
        - drift / trend break       : accelerating cadence (shrinking gaps)
        - burst / spike             : an even baseline with one tight burst inserted
  * because aggregates are matched, an order-blind pooled embedding (C3 baseline) and a
    CatBoost on per-account summary stats (C4 baseline) are weak; a sequence model that
    sees the ordered timing should win. That is the claim under test, by construction.

This is single-entity behaviour only. NETWORK patterns (structuring across accounts) are
NOT modelled - PRAGMA fails those by -47% F0.5 for the same reason, and they need a graph
dimension this spec does not have.
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

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))   # repo root, for script use

from data.synth_pacs008 import (
    CHANNELS,
    COLUMN_BUCKETS,
    GenConfig,
    SETTLEMENT_METHODS,
    TASKS,
    assign_expense,
    assign_geo,
    assign_risk,
    generate_accounts,
    project_to_pacs008,
    vocab_report,
)

SHIFT_TYPES = ("dormancy", "drift", "burst")


@dataclass
class V2Config:
    num_accounts: int = 4000       # actors (debtors) whose histories we model
    creditors_per: int = 4         # counterparties each actor pays
    min_events: int = 8            # payments per account (same dist across classes)
    max_events: int = 40
    horizon_days: int = 365
    shift_fraction: float = 0.45   # share of accounts that undergo a regime change
    amount_log_mu: float = 9.0
    amount_log_sigma: float = 1.2
    amount_event_sigma: float = 0.25
    start_date: str = "2023-01-01"
    seed: int = 11


# --------------------------------------------------------------------------- #
# Timing - SAME gap multiset across classes; only the ARRANGEMENT differs.
# --------------------------------------------------------------------------- #
# This is what makes C3/C4 honest. Both classes draw their inter-arrival gaps from
# the same process, so every order-invariant summary (count, amount, gap mean/std/
# min/max) has the same distribution across classes - a pooled embedding (C3) and a
# CatBoost on aggregates (C4) cannot separate them. Only the ORDER differs:
#   * Stable : gaps interleaved small/large -> even cadence, no trend or clustering
#   * Shift  : the same gaps arranged as a regime -
#       drift   - sorted monotonically (cadence speeds up or slows down)
#       cluster - small gaps together (a burst) and large gaps together (dormancy)
# so only a model that reads the ordered, timed sequence can tell them apart.

def _gaps(rng, K, H):
    g = rng.lognormal(mean=0.0, sigma=1.0, size=max(K - 1, 1))
    return g / g.sum() * H                       # K-1 gaps spanning the horizon


def _even_order(g):
    s = np.sort(g)
    out = np.empty_like(s)
    lo, hi, i = 0, len(s) - 1, 0
    while lo <= hi:
        out[i] = s[lo]; lo += 1; i += 1
        if lo <= hi:
            out[i] = s[hi]; hi -= 1; i += 1
    return out


def _regime_order(rng, g, kind):
    s = np.sort(g)
    if kind == "drift":
        return s if rng.random() < 0.5 else s[::-1]
    small, large = s[:len(s) // 2], s[len(s) // 2:][::-1]   # burst block + dormant block
    return np.concatenate([small, large] if rng.random() < 0.5 else [large, small])


def account_stream(rng, src, creditors, cfg: V2Config):
    """Return (rows, label). Amounts and the gap MULTISET match across classes; the
    label depends only on how the gaps are ordered in time."""
    K = int(rng.integers(cfg.min_events, cfg.max_events + 1))
    mu = rng.normal(cfg.amount_log_mu, cfg.amount_log_sigma)
    amounts = np.exp(rng.normal(mu, cfg.amount_event_sigma, size=K))
    g = _gaps(rng, K, cfg.horizon_days)

    if rng.random() < cfg.shift_fraction:
        label = "Shift"
        arranged = _regime_order(rng, g, "drift" if rng.random() < 0.5 else "cluster")
    else:
        label = "Stable"
        arranged = _even_order(g)
    days = np.clip(np.concatenate([[0.0], np.cumsum(arranged)]), 0, cfg.horizon_days)

    dests = [creditors[int(rng.integers(len(creditors)))] for _ in range(K)]
    rows = [(src, dests[i], float(round(float(amounts[i]), 2)), int(days[i]))
            for i in range(K)]
    return rows, label


# --------------------------------------------------------------------------- #
# Dataset
# --------------------------------------------------------------------------- #

def build_v2_dataset(cfg: V2Config):
    rng = np.random.default_rng(cfg.seed)
    # reuse the v1 account model; treat each account as a potential debtor (actor)
    accs = generate_accounts(rng, GenConfig(num_parents=cfg.num_accounts, seed=cfg.seed))
    start = date.fromisoformat(cfg.start_date)
    n_acc = len(accs)

    records = []
    for src in accs:
        creditors = [accs[int(rng.integers(n_acc))] for _ in range(cfg.creditors_per)]
        creditors = [c for c in creditors if c.account_id != src.account_id] or [accs[0]]
        rows, label = account_stream(rng, src, creditors, cfg)
        for s, dest, amt, day in rows:
            channel = (rng.choice(SETTLEMENT_METHODS) if rng.random() < 0.7
                       else rng.choice(CHANNELS))
            dte = (start + timedelta(days=day)).isoformat()
            row = project_to_pacs008(s, dest, amt, dte, channel,
                                     assign_risk(s, dest, amt, rng), assign_geo(s, dest),
                                     assign_expense(dest), "No", src.account_id)
            row["regime_label"] = label              # per-account, repeated on each row
            records.append(row)
    return pd.DataFrame.from_records(records), accs


def build_schema(df, accs) -> dict:
    return {
        "buckets": COLUMN_BUCKETS,
        "label_column": "risk_label",                # v1-compatible single-label contract
        "label_values": ["Low", "Medium", "High"],
        "tasks": TASKS,
        "group_column": "group_id",
        # v2 entity-level task (the regime label is constant within an account)
        "entity_task": {
            "name": "regime", "label_column": "regime_label",
            "label_values": ["Stable", "Shift"], "positive_class": "Shift",
            "actor": "DbtrAcct_Id",
        },
        "n_rows": int(len(df)),
        "n_accounts": len(accs),
        "vocab": vocab_report(df),
        "regime_distribution": df.drop_duplicates("DbtrAcct_Id")["regime_label"]
            .value_counts().to_dict(),
    }


def main():
    ap = argparse.ArgumentParser(description="v2 §7 - behavioural per-account histories")
    ap.add_argument("--accounts", type=int, default=V2Config.num_accounts)
    ap.add_argument("--seed", type=int, default=V2Config.seed)
    ap.add_argument("--out", default="pacs008_seq.parquet")
    ap.add_argument("--schema-out", default="column_schema_seq.json")
    args = ap.parse_args()

    cfg = V2Config(num_accounts=args.accounts, seed=args.seed)
    df, accs = build_v2_dataset(cfg)
    try:
        df.to_parquet(args.out, index=False); written = args.out
    except Exception:
        written = args.out.replace(".parquet", ".csv"); df.to_csv(written, index=False)
    schema = build_schema(df, accs)
    Path(args.schema_out).write_text(json.dumps(schema, indent=2))

    print(f"Wrote {len(df):,} rows / {len(accs):,} accounts -> {written}")
    print(f"Regime (per account): {schema['regime_distribution']}")


if __name__ == "__main__":
    main()
