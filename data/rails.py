"""
India payment-rail registry + routing logic — four domestic INR rails + cross-border SWIFT.

BEYOND arXiv:2410.07851 (a rail/twin extension, like data/synth_workflow.py). ADDITIVE:
nothing in the paper-grounded v1 (synth_pacs008.py) is touched. This module models India's
four RBI/NPCI domestic INR rails plus cross-border SWIFT, and the (transparent,
feature-driven) logic that routes a payment onto one of them — so the encoder/twin can
learn rail behaviour.

The rails (values as of 2026; the four domestic ones run 24x7 in India today):

  RTGS  (RBI)   real-time GROSS settlement.  min Rs 2,00,000, no upper cap.    A/c+IFSC.
  NEFT  (RBI)   half-hourly BATCH (deferred net settlement). no min/cap.       A/c+IFSC.
  IMPS  (NPCI)  instant.            per-txn cap Rs 5,00,000.        A/c+IFSC or MMID+mobile.
  UPI   (NPCI)  instant.            per-txn cap Rs 1,00,000 (2-5L for some).    VPA / mobile.
  SWIFT (corr.) CROSS-BORDER correspondent. FX, no cap, hours-days.            BIC/IBAN.

The four domestic rails are all domestic INR, so they DON'T differ by currency or
cross-border — they differ by amount band, settlement mechanism, speed/SLA, value cap and
identifier, and their amount bands OVERLAP (a Rs 60,000 payment could be NEFT, IMPS or UPI),
which is what makes routing among them non-trivial. SWIFT is the cross-border path: it is
eligible ONLY when the parties are in different countries (and is then the only option), so
it is trivially separable from the domestic four by the counterparty country/currency — the
realistic situation (you know at intake whether a payment is cross-border).

Every rule here is a transparent function of features (in the spirit of assign_risk). The
caps/min/SLA are real; the rail-mix weights are a documented synthetic design choice and are
easy to tune (see BAND_WEIGHTS).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# Identifier instruments a payer can use. VPA implies UPI; MMID implies IMPS; BIC/IBAN
# implies SWIFT; an account+IFSC is rail-agnostic among RTGS/NEFT/IMPS. This is the real
# signal that separates rails at intake.
IDENTIFIER_TYPES = ["VPA", "ACCT_IFSC", "MMID_MOBILE", "BIC_IBAN"]


@dataclass(frozen=True)
class Rail:
    name: str
    operator: str          # "RBI" | "NPCI" | "SWIFT"
    settlement: str        # "gross_realtime" | "batch_dns" | "instant" | "correspondent"
    min_amount: float      # hard minimum (RTGS = 2,00,000), else 0
    cap: float | None      # per-txn cap (UPI/IMPS), None = uncapped
    identifiers: tuple     # identifier types valid for this rail
    sla_lo: float          # nominal settle time, seconds (low)
    sla_hi: float          # nominal settle time, seconds (high)
    batch_minutes: int = 0 # NEFT: wait-to-next half-hourly batch window
    scope: str = "domestic"  # "domestic" (IN<->IN) | "xborder" (IN<->foreign, SWIFT)


# Rs. All amounts in INR. SLA in seconds.
RAILS: dict[str, Rail] = {
    "RTGS": Rail("RTGS", "RBI", "gross_realtime",
                 min_amount=200_000, cap=None,
                 identifiers=("ACCT_IFSC",),
                 sla_lo=30, sla_hi=1_800),          # secs to ~30 min (RBI: credit <= 30 min)
    "NEFT": Rail("NEFT", "RBI", "batch_dns",
                 min_amount=0, cap=None,
                 identifiers=("ACCT_IFSC",),
                 sla_lo=120, sla_hi=900,            # processing once batch fires
                 batch_minutes=30),                 # + wait to next half-hourly batch
    "IMPS": Rail("IMPS", "NPCI", "instant",
                 min_amount=0, cap=500_000,
                 identifiers=("ACCT_IFSC", "MMID_MOBILE"),
                 sla_lo=5, sla_hi=60),
    "UPI": Rail("UPI", "NPCI", "instant",
                min_amount=0, cap=100_000,
                identifiers=("VPA",),
                sla_lo=2, sla_hi=30),
    "SWIFT": Rail("SWIFT", "SWIFT", "correspondent",
                  min_amount=0, cap=None,
                  identifiers=("BIC_IBAN",),
                  sla_lo=3_600, sla_hi=172_800,    # ~1 hour to ~2 days (correspondent hops)
                  scope="xborder"),
}
RAIL_NAMES = list(RAILS)
DOMESTIC_RAILS = [n for n, r in RAILS.items() if r.scope == "domestic"]

# Amount bands (INR) -> preference weights over rails. Documented synthetic choice:
# low value skews UPI/IMPS, high value skews RTGS, NEFT is the always-available middle.
# Eligibility (cap/min) is enforced separately, so weights for ineligible rails are
# dropped before sampling — the overlap that survives is what the router must learn.
_BANDS = [
    (1_000,        {"UPI": 0.70, "IMPS": 0.20, "NEFT": 0.10, "RTGS": 0.0}),
    (50_000,       {"UPI": 0.50, "IMPS": 0.30, "NEFT": 0.20, "RTGS": 0.0}),
    (100_000,      {"UPI": 0.30, "IMPS": 0.40, "NEFT": 0.30, "RTGS": 0.0}),
    (200_000,      {"UPI": 0.0,  "IMPS": 0.45, "NEFT": 0.55, "RTGS": 0.0}),
    (500_000,      {"UPI": 0.0,  "IMPS": 0.25, "NEFT": 0.25, "RTGS": 0.50}),
    (float("inf"), {"UPI": 0.0,  "IMPS": 0.0,  "NEFT": 0.30, "RTGS": 0.70}),
]


def _band_weights(amount: float) -> dict:
    for hi, w in _BANDS:
        if amount < hi:
            return w
    return _BANDS[-1][1]


def eligible_rails(amount: float, identifier_type: str | None = None,
                   xborder: bool = False) -> list[str]:
    """Rails a payment may legally use, given amount / identifier / cross-border flag.

    Scope is the first filter: a cross-border payment can only use SWIFT, a domestic one
    only the four RBI/NPCI rails. Cap/min are then hard constraints. If identifier_type is
    given it further restricts: VPA -> UPI, MMID_MOBILE -> IMPS, BIC_IBAN -> SWIFT,
    ACCT_IFSC -> RTGS/NEFT/IMPS.
    """
    want = "xborder" if xborder else "domestic"
    out = []
    for name, r in RAILS.items():
        if r.scope != want:
            continue
        if amount < r.min_amount:
            continue
        if r.cap is not None and amount > r.cap:
            continue
        if identifier_type is not None and identifier_type not in r.identifiers:
            continue
        out.append(name)
    return out


def choose_rail(amount: float, rng: np.random.Generator,
                identifier_type: str | None = None, xborder: bool = False) -> str:
    """Route a payment to a rail: amount-band preference over the ELIGIBLE set, + noise.

    Cross-border -> SWIFT (the only xborder rail). Domestic falls back to NEFT (always
    eligible, no cap/min) if nothing else qualifies.
    """
    elig = eligible_rails(amount, identifier_type, xborder)
    if not elig:
        return "SWIFT" if xborder else "NEFT"
    w = _band_weights(amount)
    p = np.array([max(w.get(name, 0.0), 1e-6) for name in elig], dtype=float)
    p = p / p.sum()
    return str(rng.choice(elig, p=p))


def sample_identifier(rail: str, rng: np.random.Generator) -> str:
    """Pick the instrument used, consistent with the rail.

    UPI -> VPA. SWIFT -> BIC/IBAN. IMPS -> mostly account+IFSC, sometimes MMID+mobile.
    RTGS/NEFT -> account+IFSC. (So VPA/MMID/BIC are rail-revealing; ACCT_IFSC is not.)
    """
    if rail == "UPI":
        return "VPA"
    if rail == "SWIFT":
        return "BIC_IBAN"
    if rail == "IMPS":
        return "MMID_MOBILE" if rng.random() < 0.25 else "ACCT_IFSC"
    return "ACCT_IFSC"


def settle_seconds(rail: str, rng: np.random.Generator) -> float:
    """Nominal end-to-end settle time in seconds, incl. NEFT wait-to-next-batch."""
    r = RAILS[rail]
    base = float(rng.uniform(r.sla_lo, r.sla_hi))
    if r.batch_minutes:
        base += float(rng.uniform(0, r.batch_minutes)) * 60.0   # queued to next batch
    return base


def violates_cap(rail: str, amount: float) -> bool:
    r = RAILS[rail]
    return r.cap is not None and amount > r.cap


def below_min(rail: str, amount: float) -> bool:
    return amount < RAILS[rail].min_amount
