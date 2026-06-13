"""
Synthetic pacs.008 transaction generator + Layer 1 projection.

Grounded strictly to:
  Raman, Ganesh, Veloso (JPMorgan AI Research),
  "Scalable Representation Learning for Multimodal Tabular Transactions",
  arXiv:2410.07851, NeurIPS 2024 TRL workshop.

Scope of this module (v1, no extensions beyond the paper):
  - Algorithm 1 synthetic generation: M_Comp, M_Dest, M_Txns, M_Amount, M_Date.
  - CreateAccount() producing the account columns of Figure 4.
  - Layer 1 projection of each transaction into a pacs.008-typed row, with the
    four-bucket column typing used by the encoder (core / high-cardinality
    categorical / meta / numerical).
  - Risk label in the paper's risk-tag slot (Low / Medium / High).

Deliberately NOT included (walked back as extensions):
  - data-completeness feature vector
  - multi-record structuring / layering chain task
  - held-out-typology generalization split
These belong to v2, not the grounded prototype.

The ONE forced domain departure baked in here is that amounts are emitted with a
currency so the downstream adaptive quantizer can be currency-conditioned; the
generator itself samples amount exactly per Algorithm 1's M_Amount.
"""

from __future__ import annotations

import argparse
import json
import string
from dataclasses import dataclass, field, asdict
from datetime import date, timedelta

import numpy as np
import pandas as pd


# --------------------------------------------------------------------------- #
# Reference vocabularies for CreateAccount (kept small, paper-style synthetic)
# --------------------------------------------------------------------------- #

INDUSTRIES = {
    "Communications": ["Internet", "Telecom", "Media"],
    "Financial": ["Banks", "Insurance", "AssetMgmt"],
    "Industrials": ["Manufacturing", "Logistics", "Construction"],
    "Consumer": ["Retail", "Food", "Apparel"],
    "Energy": ["Oil", "Utilities", "Renewables"],
    "Technology": ["Software", "Hardware", "Semiconductors"],
}

# Country -> (currency, geographic region). Region/currency drive the risk rule.
COUNTRIES = {
    "US": ("USD", "Americas"),
    "GB": ("GBP", "EMEA"),
    "FR": ("EUR", "EMEA"),
    "DE": ("EUR", "EMEA"),
    "SG": ("SGD", "Asia"),
    "IN": ("INR", "Asia"),
    "AU": ("AUD", "Asia"),
    "AE": ("AED", "EMEA"),
    "PH": ("PHP", "Asia"),
    "JP": ("JPY", "Asia"),
}

NAME_HEADS = ["Plantations", "Far", "Vertex", "Northwind", "Acme", "Meridian",
              "Cardinal", "Solstice", "Granite", "Harbor", "Lumen", "Apex"]
NAME_FORMS = ["Ltd", "LLC", "Inc", "GmbH", "PLC", "SA", "Pte"]
NAME_TAILS = ["Sales", "Travel", "Holdings", "Trading", "Capital", "Services", ""]

SETTLEMENT_METHODS = ["INDA", "INGA", "COVE", "CLRG"]   # ISO 20022 SttlmMtd
CHANNELS = ["Swift", "Local", "OnUs"]                    # paper's Channel column

_rng_alphabet = np.array(list(string.ascii_uppercase + string.digits))


# --------------------------------------------------------------------------- #
# Probabilistic models referenced by Algorithm 1
# --------------------------------------------------------------------------- #

@dataclass
class GenConfig:
    """All knobs for Algorithm 1. Defaults give a POC-scale corpus."""
    num_parents: int = 4000          # C  (parent companies)
    num_transactions: int = 500_000  # T  (target transaction count)

    # M_Comp: accounts per company, Gaussian Mixture (K components)
    comp_means: tuple = (2.0, 8.0, 25.0)
    comp_stds: tuple = (1.0, 3.0, 8.0)
    comp_weights: tuple = (0.6, 0.3, 0.1)

    # M_Dest: number of distinct target accounts a source transacts with
    dest_mean: float = 4.0
    dest_std: float = 3.0

    # M_Txns: number of transactions between a (source, target) pair
    txns_mean: float = 3.0
    txns_std: float = 2.5

    # M_Amount: per-pair mean/var are themselves sampled, then amount ~ N(mu, var)
    amount_log_mu: float = 9.0       # ~ exp(9) ≈ 8k base scale (pre-currency)
    amount_log_sigma: float = 1.2
    amount_pair_sigma: float = 0.4   # within-pair amount spread (log space)

    # M_Date: per-pair mean/var of dates; date ~ N(mu_d, var_d) days from start
    start_date: str = "2023-01-01"
    horizon_days: int = 365
    date_pair_sigma: float = 20.0

    seed: int = 7


def _sample_gmm(rng, n, means, stds, weights):
    comp = rng.choice(len(means), size=n, p=np.asarray(weights) / np.sum(weights))
    vals = rng.normal(np.asarray(means)[comp], np.asarray(stds)[comp])
    return np.clip(np.round(vals), 1, None).astype(int)


def _rand_id(rng, length=8):
    return "".join(rng.choice(_rng_alphabet, size=length))


def _rand_name(rng):
    head = rng.choice(NAME_HEADS)
    form = rng.choice(NAME_FORMS)
    tail = rng.choice(NAME_TAILS)
    return " ".join(p for p in (head, form, tail) if p)


# --------------------------------------------------------------------------- #
# CreateAccount  (Algorithm 1 helper)
# --------------------------------------------------------------------------- #

@dataclass
class Account:
    account_id: str
    account_name: str
    parent_id: str
    parent_name: str
    industry: str
    sub_industry: str
    country: str
    currency: str


def create_account(rng, parent_id, parent_name, industry, sub_industry, country):
    cur, _ = COUNTRIES[country]
    return Account(
        account_id=_rand_id(rng, 8),
        account_name=f"{parent_name} {rng.choice(NAME_TAILS) or 'Ops'}".strip(),
        parent_id=parent_id,
        parent_name=parent_name,
        industry=industry,
        sub_industry=sub_industry,
        country=country,
        currency=cur,
    )


# --------------------------------------------------------------------------- #
# Algorithm 1
# --------------------------------------------------------------------------- #

def generate_accounts(rng, cfg: GenConfig):
    accs = []
    n_per_company = _sample_gmm(rng, cfg.num_parents,
                                cfg.comp_means, cfg.comp_stds, cfg.comp_weights)
    for c in range(cfg.num_parents):
        parent_id = _rand_id(rng, 6)
        parent_name = _rand_name(rng)
        industry = rng.choice(list(INDUSTRIES.keys()))
        sub_industry = rng.choice(INDUSTRIES[industry])
        country = rng.choice(list(COUNTRIES.keys()))
        for _ in range(int(n_per_company[c])):
            accs.append(create_account(rng, parent_id, parent_name,
                                       industry, sub_industry, country))
    return accs


def generate_transactions(rng, cfg: GenConfig, accs):
    start = date.fromisoformat(cfg.start_date)
    n_acc = len(accs)
    rows = []

    while len(rows) < cfg.num_transactions:
        src = accs[rng.integers(n_acc)]
        n_dest = max(1, int(rng.normal(cfg.dest_mean, cfg.dest_std)))
        for _ in range(n_dest):
            dest = accs[rng.integers(n_acc)]
            if dest.account_id == src.account_id:
                continue
            n_txns = max(1, int(rng.normal(cfg.txns_mean, cfg.txns_std)))

            # per-pair amount/date mean+variance, then sample each txn
            pair_log_mu = rng.normal(cfg.amount_log_mu, cfg.amount_log_sigma)
            pair_day_mu = rng.uniform(0, cfg.horizon_days)

            for _ in range(n_txns):
                amt = float(np.exp(rng.normal(pair_log_mu, cfg.amount_pair_sigma)))
                day = int(np.clip(rng.normal(pair_day_mu, cfg.date_pair_sigma),
                                  0, cfg.horizon_days))
                rows.append((src, dest, round(amt, 2),
                             (start + timedelta(days=day)).isoformat()))
                if len(rows) >= cfg.num_transactions:
                    break
            if len(rows) >= cfg.num_transactions:
                break
    return rows


# --------------------------------------------------------------------------- #
# Risk label  (paper's risk-tag slot: Low / Medium / High)
# --------------------------------------------------------------------------- #
# Grounded choice: a transparent rule over transaction features, mirroring the
# paper's "template-based instructions and desired responses" for risk tagging.
# Cross-border + large amount + a few elevated-risk industries lift the tier.
# This is a learnable signal, not an elaborate fraud-typology engine.

_HIGH_RISK_INDUSTRIES = {"Financial", "Energy"}
_HIGH_RISK_REGIONS = {"EMEA", "Asia"}


def assign_risk(src: Account, dest: Account, amount: float, rng):
    # Cross-border is the norm in this synthetic world, so it is weighted lightly;
    # High tier requires risk factors to stack (large amount + elevated industry/
    # region), keeping High a realistic minority class for imbalance-aware eval.
    score = 0.0
    if src.country != dest.country:
        score += 0.4
    if src.currency != dest.currency:
        score += 0.3
    _, src_region = COUNTRIES[src.country]
    _, dst_region = COUNTRIES[dest.country]
    if src_region in _HIGH_RISK_REGIONS and dst_region in _HIGH_RISK_REGIONS:
        score += 0.5
    if src.industry in _HIGH_RISK_INDUSTRIES or dest.industry in _HIGH_RISK_INDUSTRIES:
        score += 0.6
    if amount > 50_000:
        score += 0.7
    if amount > 250_000:
        score += 1.0
    if amount > 1_000_000:
        score += 1.0
    score += rng.normal(0, 0.25)  # label noise so the task isn't trivially separable

    if score >= 2.5:
        return "High"
    if score >= 1.6:
        return "Medium"
    return "Low"


# --------------------------------------------------------------------------- #
# Layer 1 projection: transaction -> pacs.008-typed row
# --------------------------------------------------------------------------- #
# Mapping (paper Figure 4 column -> pacs.008 element):
#   Source/Target Account ID   -> DbtrAcct / CdtrAcct        (high-card categorical)
#   Source/Target Account Name -> Dbtr/Cdtr Nm               (meta, offline party)
#   Source/Target Parent ID    -> UltmtDbtr/UltmtCdtr Id     (high-card categorical)
#   Source/Target Parent Name  -> UltmtDbtr/UltmtCdtr Nm     (meta)
#   Source/Target Industry,SubIndustry -> party meta         (meta)
#   Source/Target Country      -> party PstlAdr Ctry         (categorical, meta-ish)
#   Amount                     -> IntrBkSttlmAmt             (numerical)
#   Currency                   -> Ccy attribute of amount    (core)
#   Date                       -> IntrBkSttlmDt              (core)
#   Channel                    -> SttlmMtd / channel         (core)

COLUMN_BUCKETS = {
    # high-cardinality categorical -> §3.1 partitioning embedder
    "high_card_categorical": [
        "DbtrAcct_Id", "CdtrAcct_Id", "UltmtDbtr_Id", "UltmtCdtr_Id",
    ],
    # numerical -> §3.3 adaptive (currency-conditioned) quantizer
    "numerical": ["IntrBkSttlmAmt"],
    # core inline columns
    "core": ["Ccy", "IntrBkSttlmDt", "SttlmMtd"],
    # meta-columns -> §3.2 offline party encoder, injected as pooled summary
    "meta_party": [
        "Dbtr_Nm", "Cdtr_Nm", "UltmtDbtr_Nm", "UltmtCdtr_Nm",
        "Dbtr_Ctry", "Cdtr_Ctry",
        "Dbtr_Industry", "Cdtr_Industry",
        "Dbtr_SubIndustry", "Cdtr_SubIndustry",
    ],
}


def project_to_pacs008(src: Account, dest: Account, amount, dte, channel, risk):
    return {
        # --- high-cardinality categorical (partitioning embedder) ---
        "DbtrAcct_Id": src.account_id,
        "CdtrAcct_Id": dest.account_id,
        "UltmtDbtr_Id": src.parent_id,
        "UltmtCdtr_Id": dest.parent_id,
        # --- numerical ---
        "IntrBkSttlmAmt": amount,
        # --- core ---
        "Ccy": src.currency,
        "IntrBkSttlmDt": dte,
        "SttlmMtd": channel,
        # --- meta party (offline encoder, pooled summary at row assembly) ---
        "Dbtr_Nm": src.account_name,
        "Cdtr_Nm": dest.account_name,
        "UltmtDbtr_Nm": src.parent_name,
        "UltmtCdtr_Nm": dest.parent_name,
        "Dbtr_Ctry": src.country,
        "Cdtr_Ctry": dest.country,
        "Dbtr_Industry": src.industry,
        "Cdtr_Industry": dest.industry,
        "Dbtr_SubIndustry": src.sub_industry,
        "Cdtr_SubIndustry": dest.sub_industry,
        # --- label in the paper's risk-tag slot ---
        "risk_label": risk,
    }


# --------------------------------------------------------------------------- #
# Orchestration
# --------------------------------------------------------------------------- #

def build_dataset(cfg: GenConfig) -> pd.DataFrame:
    rng = np.random.default_rng(cfg.seed)
    accs = generate_accounts(rng, cfg)
    txns = generate_transactions(rng, cfg, accs)

    records = []
    for src, dest, amt, dte in txns:
        channel = rng.choice(SETTLEMENT_METHODS) if rng.random() < 0.7 \
            else rng.choice(CHANNELS)
        risk = assign_risk(src, dest, amt, rng)
        records.append(project_to_pacs008(src, dest, amt, dte, channel, risk))

    df = pd.DataFrame.from_records(records)
    return df, accs


def vocab_report(df: pd.DataFrame) -> dict:
    rep = {}
    for col in COLUMN_BUCKETS["high_card_categorical"]:
        rep[col] = int(df[col].nunique())
    rep["combined_account_id_vocab"] = int(
        pd.concat([df["DbtrAcct_Id"], df["CdtrAcct_Id"]]).nunique()
    )
    rep["combined_parent_id_vocab"] = int(
        pd.concat([df["UltmtDbtr_Id"], df["UltmtCdtr_Id"]]).nunique()
    )
    return rep


def main():
    ap = argparse.ArgumentParser(description="Synthetic pacs.008 generator (paper-grounded)")
    ap.add_argument("--parents", type=int, default=GenConfig.num_parents)
    ap.add_argument("--transactions", type=int, default=GenConfig.num_transactions)
    ap.add_argument("--seed", type=int, default=GenConfig.seed)
    ap.add_argument("--out", type=str, default="pacs008_synth.parquet")
    ap.add_argument("--schema-out", type=str, default="column_schema.json")
    args = ap.parse_args()

    cfg = GenConfig(num_parents=args.parents,
                    num_transactions=args.transactions,
                    seed=args.seed)

    df, accs = build_dataset(cfg)

    try:
        df.to_parquet(args.out, index=False)
        written = args.out
    except Exception:
        written = args.out.replace(".parquet", ".csv")
        df.to_csv(written, index=False)

    schema = {
        "buckets": COLUMN_BUCKETS,
        "label_column": "risk_label",
        "label_values": ["Low", "Medium", "High"],
        "n_rows": int(len(df)),
        "n_accounts": len(accs),
        "vocab": vocab_report(df),
        "risk_distribution": df["risk_label"].value_counts().to_dict(),
    }
    with open(args.schema_out, "w") as f:
        json.dump(schema, f, indent=2)

    print(f"Wrote {len(df):,} rows -> {written}")
    print(f"Accounts: {len(accs):,}")
    print(f"Vocab: {json.dumps(schema['vocab'], indent=2)}")
    print(f"Risk dist: {schema['risk_distribution']}")


if __name__ == "__main__":
    main()
