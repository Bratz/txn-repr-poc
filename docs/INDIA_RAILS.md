# India multi-rail dataset — RTGS / NEFT / IMPS / UPI + cross-border SWIFT

A synthetic payment dataset spanning India's four domestic account-to-account rails **plus
cross-border SWIFT**, for the digital-twin backbone. It is a **beyond-paper extension** (like
the twin in [`PAYMENT_TWIN.md`](PAYMENT_TWIN.md)) and is **additive**: it lives in its own
modules and reuses the paper-grounded v1 helpers without touching them.

- [`data/rails.py`](../data/rails.py) — the rail registry + routing logic (pure, tested).
- [`data/synth_india_rails.py`](../data/synth_india_rails.py) — the generator (intake +
  rail-conditioned workflow), reusing `data/synth_pacs008.py` the way
  `data/synth_workflow.py` does.

## The rails

The four **domestic** rails are all INR, so currency and cross-border don't separate them —
they differ by amount band, settlement mechanism, value cap, SLA and identifier. **SWIFT** is
the **cross-border** path (one leg abroad, FX), so it re-introduces those signals for its rows:

| Rail | Operator | Settlement | Scope | Min | Per-txn cap | Identifier | Nominal SLA |
|------|----------|-----------|-------|-----|-------------|-----------|-------------|
| **RTGS** | RBI | real-time **gross** | domestic | **₹2,00,000** | none | A/c + IFSC | secs → ~30 min |
| **NEFT** | RBI | **half-hourly batch** (DNS) | domestic | none | none | A/c + IFSC | wait-to-batch + credit |
| **IMPS** | NPCI | instant | domestic | none | **₹5,00,000** | A/c+IFSC **or** MMID+mobile | seconds |
| **UPI**  | NPCI | instant | domestic | none | **₹1,00,000** (2–5L some) | **VPA** / mobile | seconds |
| **SWIFT** | correspondent | cross-border (FX) | **xborder** | none | none | **BIC / IBAN** | **hours → ~2 days** |

(The four domestic rails run 24×7 in India today.) Their amount bands **overlap** — a ₹60,000
payment is a legitimate UPI, IMPS *or* NEFT — which is what makes **rail routing** among the
domestic four non-trivial. SWIFT is trivially separable (you know at intake whether a payment
is cross-border, from the counterparty country/currency), so the routing difficulty lives in
the domestic majority.

## Generative model (rail-first, transparent rules)

Each payment is first **domestic or cross-border** (`xborder_frac`). Cross-border → SWIFT
with one leg in India and the other abroad (FX, foreign-currency counterparty). Domestic →
sample an INR amount (heavy-tailed log-normal) → `choose_rail` picks the rail by an
**amount-band preference over the eligible set** (cap/min enforced) → an identifier consistent
with that rail (`UPI⇒VPA`, `MMID⇒IMPS`, `SWIFT⇒BIC/IBAN`, else `ACCT_IFSC`). A fraction of
large domestic payments deliberately **attempt an over-cap rail** (e.g. ₹1.5L on UPI) or a
**below-floor RTGS**, to manufacture the `limit_exceeded` / `below_min` exceptions.

Each payment then traverses its **rail-specific workflow**, every step clean or raising a
feature-driven exception that is repaired or halts the payment:

```
UPI  : validation → vpa_resolution → fraud_risk → limit_check → npci_switch → credit
IMPS : validation → beneficiary_resolution → fraud_risk → limit_check → npci_switch → credit
RTGS : validation → min_amount_check → aml → liquidity → rbi_settlement → credit
NEFT : validation → enrichment → aml → batch_window → dns_settlement → credit
SWIFT: validation → enrichment → sanctions → fx_conversion → correspondent_routing
                  → cover_check → settlement → credit
```

`limit_check` / `min_amount_check` are **deterministic gates**; `batch_window` adds NEFT's
wait-to-next-batch latency (no exception); instant rails can **time out** at `npci_switch`
→ `sla_breach` (and sometimes `technical_decline`); SWIFT adds the cross-border exceptions
`fx_fail`, `no_route`, `no_cover` and a slow (hours→days) correspondent settlement.

## Two tables (same contract as the twin)

- **payment-level** — pacs.008 features + `rail`, `identifier_type`, `settlement_kind`,
  `terminal_status`, `time_to_settle_min`, and one `exc_<code>` column per exception
  (incl. `exc_sla_breach`, `exc_limit_exceeded`).
- **event-level** — one `(payment_id, seq, step, outcome, excode, rail, t_min)` row per step.

## Tasks this unlocks

| Task | Target | Notes |
|------|--------|-------|
| `risk` | risk_label | amount + industry vary; cross-border live for SWIFT rows |
| **`rail_routing`** | `rail` | predict the rail (5-class) from intake features |
| **`sla_breach`** | `exc_sla_breach` | binary twin exception (instant-rail timeout) |
| **`limit_exceeded`** | `exc_limit_exceeded` | binary twin exception (over-cap attempt) |
| **ETA** | `time_to_settle_min` | real cross-rail spread: instant (secs) ≪ NEFT batch (~15 min) ≪ SWIFT (~hours–days) |

## No-leakage rule

`rail` is the routing **label**, and `settlement_kind` is a 1:1 **consequence** of it — so
neither is placed in the feature buckets. `identifier_type` **is** a feature (the instrument
is known before the rail is chosen). `VPA⇒UPI` / `MMID⇒IMPS` / `BIC⇒SWIFT` are intentionally
near-deterministic, and SWIFT is also flagged by the counterparty country/currency (both
legitimately known at intake); the real difficulty is the domestic `ACCT_IFSC` majority,
where the **amount** decides RTGS vs NEFT vs IMPS.

## Run it

Generate, then train — **both run entirely on CPU** (no frozen-LLM path):

```bash
# 1. generate (pure NumPy/pandas, CPU)
python data/synth_india_rails.py --accounts 4000 --payments 60000 \
    --out-prefix data/india_rails --schema-out data/column_schema_india.json

# 2. train + score on held-out payments (CPU)
python run_india.py --smoke        # tiny configs, plumbing-valid numbers
python run_india.py                 # full configs (slower on CPU; better numbers)
```

Step 1 emits `india_rails_payments.parquet`, `india_rails_events.parquet`,
`column_schema_india.json`. [`run_india.py`](../run_india.py) freezes the v1 encoder, then
probes `f(payment)` for rail routing (vs a **tree baseline** on the raw visible features),
the exception likelihoods, terminal status and ETA, and trains a rail-conditioned in-flight
next-exception head. Writes `results_india.json`.

### Persist & serve (no retraining)

`run_india.py --save` writes a deployable bundle (frozen encoder + intake probes), and
[`serve_india.py`](../serve_india.py) loads it and predicts on new payment rows without
retraining — same convention as [`predict.py`](../predict.py) (the party table rides in the
encoder `state_dict`; the assembler is rebuilt with `party_store=None`).

```bash
python run_india.py --save model_india           # train + persist (encoder.pt, probes.joblib)
python serve_india.py --model-dir model_india \
    --input data/india_rails_payments.parquet --out india_predictions.csv
```

Per-payment output: predicted `rail` (+ confidence), `status`, `eta_min`, and the top
exception risk scores. The tree baseline is a **training-time diagnostic only** — its
raw-feature factorize codes aren't stable across datasets, so it is not part of the saved
model (the served predictor is the encoder + probes). A round-trip test asserts the reloaded
model reproduces the in-memory predictions exactly.

### Full CPU run (20k payments, 3-epoch encoder)

| Signal | Result | Note |
|--------|--------|------|
| rail routing | probe acc **0.75** / tree acc **0.80** / majority 0.39 | both beat majority; the tree still edges the probe (visible-feature task) |
| limit_exceeded | PR-AUC **0.55** (≈1.6% prevalence) | strongly learnable (deterministic cap rule) |
| ETA | MAE **207** vs mean baseline **400** min | rail tiers learned; ~halves the error |
| other exceptions | no_route **0.10**, below_min **0.08**, sanctions **0.07** | weaker / sparse signal |
| status | acc **0.29** (majority 0.71), macro-F1 **0.23** | STP dominates; the balanced probe trades accuracy for recall and doesn't beat majority |
| in-flight next-exception | macro-F1 **0.11**, next-any PR-AUC **0.21** (prev 0.06) | beats prevalence; sparse |

(A 1-epoch `--smoke` run gives similar-shape but weaker numbers — plumbing check only.)
The honest takeaway is the same as elsewhere in this POC: trees are strong on visible-feature
tasks; some labels (deterministic caps, rail ETA tiers) are very learnable from the
representation while the dominant-majority status label and sparse exceptions are not. The
value of the backbone is representation reuse across many tasks, not beating a tree on these
synthetic labels.

## Fidelity & honesty

Caps, minima, settlement mechanisms and nominal SLAs are **real RBI/NPCI/SWIFT values** (as
of 2026). The rail-**mix** weights, the cross-border fraction, exception **rates** (the
over-cap attempt rate is deliberately **amplified** so `limit_exceeded` is learnable), and
service times are **documented synthetic design choices** (see `_BANDS` in `data/rails.py` and
the probability tables in `synth_india_rails.py`) — they are not calibrated to any bank's
actual volumes. As with the rest of this POC, the labels are learnable feature rules, partly
learnable by a tree on raw features too; the point is that the backbone *can* model rail
behaviour, not that it beats a tree on this synthetic data.
