# CBPR+ payments hub → TFM twin: gap analysis

**Purpose.** Measure our synthetic payment twin against a real cross-border (CBPR+) payments hub, so
we know which capabilities are faithfully modelled, which are approximated, and which are missing —
and in what order to close the gaps.

**Scope & confidentiality.** This analysis is framed on the **public ISO 20022 / CBPR+ / SWIFT gpi**
standard (pacs / camt messages, gpi tracker, gCCT/gCOV/gSRP). It was informed by reviewing a vendor
reference implementation; **no vendor-proprietary content is reproduced here** — the marked-up
reference transcription is kept confidential and out of this repo. Where we cite a concept it is the
open standard, which is exactly what the twin should target.

**What "the twin" is here.** The India-rails generator + ISO lifecycle + frozen-backbone probes:
`data/synth_india_rails.py`, `data/iso_lifecycle.py`, `data/rails.py`, `run_impute.py`,
`run_india.py`, `serve_india.py`, `data/iso20022_pacs008.py`.

---

## 1. Reference model (real CBPR+ hub)

A cross-border hub processes a payment across three ISO domains and several *flows*, with the **gpi
Tracker (UETR-keyed)** as the status backbone:

- **Messages:** `pacs.008` (customer credit), `pacs.009` (FI credit), `pacs.009 COV` (cover),
  `pacs.002` (status), `pacs.004` (return), `camt.056` (cancellation request), `camt.029`
  (resolution), `camt.026/028` (investigation); **MT↔MX coexistence** (MT103/202/192/196/199).
- **Flows:** Outward · Inward · Forwarding/passthrough · Return · Cancellation · Claim · Investigation.
- **gpi services:** gCCT / gCOV / gSRP / gFIT; tracker updates carry `ACSC` / `ACSP+Gnnn` (G001 = passed
  to non-GPI agent, G002 = in repair) / `RJCT`; gSRP cancellation → `CNCL` (accepted) / `RJCR`
  (rejected), responded via `camt.029`.
- **Processing spine:** acquisition → enrichment & validation (duplicate check, Geo-Cover
  Domestic/Intl/SEPA, ON-US/OFF-US, name-number match, cover/reimbursement matching, UTR, SSI/debit
  account enrichment, cutoff/holiday warehousing) → compliance/sanctions → accounting (charges/VAT,
  FX, value-dating) → message generation → ACK/NACK → status report.
- **Exceptions:** repair queues, repair-type (auto-return / repair / reject), STP-break, factory-mode
  & mass repair, 4-eye maker-checker.

---

## 2. Gap matrix

Status: ✅ modelled · 🟡 partial · ❌ missing. **Owner** = who should produce it (see §3.5): 🧠 TFM
(ML prediction) · 📐 Rule (deterministic) · 📚 Lookup (directory/assigned) · 🧾 Template (message gen) ·
🗄️ Data (lifecycle/schema model, not itself a prediction). Effort: S/M/L. Value = usefulness.

| # | Capability (standard) | Twin today | Status | Owner | Effort | Value |
|---|---|---|---|---|---|---|
| 1 | **Message lifecycle** pain.001→…→camt.054 | `iso_lifecycle.MSG_TYPES` w/ field ownership, TxSts, timestamps | ✅ | 🗄️ Data | — | — |
| 2 | **gpi tracker status** ACSC/ACSP/RJCT (+Gnnn) | **relabelled ✅** — pacs.002 is the tracker backbone; held → `ACSP`+`G002`; schema `gpi_tracker` block | ✅ | 🧠 TFM + relabel | S | Med |
| 3 | **Reject/return reason** (ISO StsRsn) | `reject_reason` + `REASON` map | ✅ | 🧠 TFM (select) + 📚 code string | — | High |
| 4 | **pacs.009 / pacs.009 COV** (FI + cover) | **COV cover leg shipped** (every cross-border/COVE pacs.008 gets a pacs.009 cover); standalone FI-credit pacs.009 not modelled (all payments are customer transfers) | 🟡 | 🗄️ Data | M | High |
| 5 | **Cancellation (camt.056 / camt.029)** + matching | **lifecycle legs + `cancel_requested`/`cancel_status` labels shipped**; likelihood head pending | 🟡 | 🗄️ Data ✅ + 🧠 TFM (likelihood) | M | High |
| 6 | **Return (pacs.004)** + party-reversal | **lifecycle leg + reversed parties + `returned`/`return_reason` labels shipped**; likelihood head pending | 🟡 | 🗄️ Data ✅ + 🧠 TFM (likelihood) | M | High |
| 7 | **Claim mgmt** (MT191/camt.106) | none | ❌ | 🗄️ Data / 📐 Rule | L | Low |
| 8 | **Investigation** (camt.026/028/029) | none | ❌ | 🗄️ Data | L | Low |
| 9 | **Forwarding / passthrough** hops | endpoints only | ❌ | 🗄️ Data | M | Med |
| 10 | **Charges / VAT + FX + value-dating** | **amount split shipped** — `InstdAmt`/`InstdCcy`/`fx_rate`/`charges` (two-currency + fee labels); FX/charges head + value-dating pending | 🟡 | 📐 Rule (charges/VAT/dates) + 🧠 TFM (FX amount) | M | High |
| 11 | **Geo-Cover** Dom/Intl/SEPA | `geo_label` (deterministic) | 🟡 | 📐 **Rule — delisted** | S | Med |
| 12 | **ON-US / OFF-US** | not derived | ❌ | 📐 **Rule — delisted** | S | Med |
| 13 | **Duplicate validation** | none | ❌ | 📐 **Rule — delisted** | S | Med |
| 14 | **Name-number match (NNC)** | none | ❌ | 📐 **Rule — delisted** | S | Med |
| 15 | **Cover / reimbursement matching** | none | ❌ | 📚 **Lookup — delisted** | M | Med |
| 16 | **Reference directories** (SSI/RMA/BSD/IBAN) | party master only | 🟡 | 📚 Lookup → inject via §3.2 offline summary | M | Med |
| 17 | **Repair queues / repair-type / STP-break** | terminal_status + `RESOLVE_P` | 🟡 | 🧠 TFM (repair-type) + workflow | S | Med |
| 18 | **Compliance / sanctions hold** | `sanctions_hit`/`fraud_hold` → MANUAL_REVIEW | 🟡 | 📐 External screen + 🧠 TFM (pre-screen likelihood) | S | Low |
| 19 | **Cutoff / holiday / warehousing** | not modelled | ❌ | 📐 Rule (calendar) → feeds 🧠 ETA | M | Med |
| 20 | **Order mgmt** bulk SDSC/SDMC/MDMC | single payments | ❌ | 🗄️ Data | L | Low |
| 21 | **MT↔MX coexistence** (parse/emit MT) | Layer-1 pacs.008-only | ❌ | 🧾 Parser/Template | L | Low |

**Headline:** the *spine* (lifecycle, status, reason, exception→repair→terminal, imputation) is
faithful. Of the 21 rows, **6 are delisted** to Rule/Lookup (11–15, 19-calendar), several are **Data**
model work (4, 5, 6, 8, 9, 20) that merely *enable* a prediction, and the genuine **TFM/ML** gaps are
few and specific: gpi next-status (2), cancellation/return likelihood (5, 6), repair-type (17),
sanctions pre-screen (18), FX amount (10). Biggest *faithfulness* gaps remain the peripheral flows
(cancellation, return, pacs.009) and the financial detail.

---

## 3. Prediction / label opportunities the reference reveals

A real hub is a factory of labels — every queue, status, and reason is something a twin can predict
*before it happens*. Mapped to what we have / could add:

| Prediction | Label source | Status |
|---|---|---|
| STP vs repair vs hold vs reject @ pain.001 | `terminal_status` | ✅ (run_impute A) |
| Reject/hold reason (ISO code) | `reject_reason` | ✅ (run_impute B) |
| ETA to settlement @ initiation | `time_to_settle_min` | ✅ (run_impute C) |
| Streaming booked-prediction as messages arrive | camt.054 present | ✅ (run_impute D) |
| **Which repair queue / repair-type** | exception → repair-type taxonomy | ❌ new (S) |
| **Duplicate?** | duplicate flag | ❌ new (S) |
| **NNC pass/fail** | name-match score | ❌ new (S) |
| **Geo-Cover / SEPA class** | Geo-Cover label | 🟡 refine (S) |
| **Will it be cancelled/recalled?** (camt.056 likelihood) | cancellation event | ❌ new (M) |
| **Will it be returned?** (pacs.004 likelihood) | return event | ❌ new (M) |
| **gpi next tracker status** (ACSC/ACSP/RJCT) | tracker sequence | 🟡 (extend D) |
| **Charge / FX / settlement amount** | charges + fx fields | ❌ new (M, needs #10) |
| **SLA/cutoff breach** | `sla_breach` + cutoff | 🟡 extend |

The honest-value framing is unchanged: on any *single complete-record* label a gradient-boosted tree
matches or beats the frozen bottleneck. The twin earns its keep on (a) **predict-from-partial**
(pain.001 view), (b) **one shared backbone across all these labels**, and (c) **sequence/lifecycle**
regimes (streaming, cancellation-likelihood, next-tracker-status).

---

## 3.5 Capability routing: TFM-owned vs rule / lookup / template

Not every hub output is a prediction. A large fraction is **deterministic** (computable by a rule) or
an **assigned/looked-up value** (fetched from a directory or generated). These are **delisted** as TFM
prediction targets — the TFM *can* fit a head to them but shouldn't own them: a rule is exact,
auditable and free; a directory is authoritative; a template is deterministic. Route them accordingly.

### Delisted — already in the code

| Capability | Where | Why not a prediction | Route to |
|---|---|---|---|
| `is_mis_routed` | `rail not in eligible_rails(...)` | boolean rule | rule (metadata only) |
| `exc_limit_exceeded` | `violates_cap()` | deterministic cap gate | rule |
| `exc_below_min` | `below_min()` | deterministic RTGS floor | rule |
| `settlement_kind` / `SttlmMtd` / `rail_family` | `RAIL_STTLM` | 1:1 from rail (leakage) | derive from rail |
| eligibility slice of `rail_routing` | `eligible_rails()` | VPA→UPI / BIC→SWIFT / cap-eligibility | eligibility mask (done in `serve_india`) |
| high-card ID imputation (`DbtrAcct_Id`, `CdtrAcct_Id`, `UltmtDbtr_Id`, `UltmtCdtr_Id`) | `run_impute` recon | assigned/looked-up; top-1 ≈ 0.00 | directory/generator |
| `geo_label`, `expense_label` | `assign_geo` / `assign_expense` (no rng) | Geo-Cover rule / industry lookup | rule/lookup — **but kept in §5 for paper fidelity + C2 evidence; not deleted** |

Applied: `twin_binary_tasks` now = `[exc_sla_breach, exc_fraud_hold]` (stochastic); delisted gates
moved to schema `rule_computed`. `run_impute` masks the high-card IDs for the pain.001 view but no
longer reports them as imputation targets. `geo`/`expense` §5 tasks annotated, left intact.

### Delisted — upcoming (from the CBPR+ doc)

**Deterministic rules (compute):** Geo-Cover (Dom/Intl/SEPA) · ON-US/OFF-US · urgency (code words) ·
payment-type (Commercial/Treasury) · product code · duplicate validation · cap/min/eligibility ·
serial-vs-cover method · cutoff/holiday → warehousing · charge & VAT computation · gpi eligibility ·
debit-authority check · name-number match score+threshold · settlement method.

**Assigned / looked-up / generated (fetch or template):** UTR generation · SSI/debit-account
enrichment · correspondent/agent BIC resolution (BSD/RMA/SWIFT dir) · IBAN/BBAN validation & derivation ·
beneficiary-master enrichment · account/message/E2E/reference IDs · ISO reason-*code string* assignment ·
message generation (pacs/camt XML) + narrations (`write_pacs008` templating).

### Stays TFM-owned (predictions worth a head)

STP/in-flight outcome from pain.001 · reject/hold **reason selection** · ETA / SLA-breach · streaming
booked-probability · **stochastic** exceptions (fraud_hold, sanctions_hit, technical_decline, …) ·
cancellation likelihood (camt.056) · return likelihood (pacs.004) · repair-type / which-queue · gpi
next tracker status (ACSC/ACSP/RJCT, sequence) · structured-field imputation (Ccy, dates) · FX
counter-amount (borderline). `risk` is kept as a §5/C2 task though it is largely rule-derived.

**Rule of thumb.** Deterministic from ≤3 fields → rule. Assigned/looked-up → directory/generator.
Feature-rule label on a complete record → tree (TFM only for one-backbone economics). Partial input,
sequence, or "will X happen later" → TFM.

## 4. Plan — review & enhancements

Split by **owner** (per §3.5 / the matrix Owner column), because the tracks have different economics:
Rule/Lookup/Template work is cheap, deterministic and needs no training; Data work enables new
predictions; only the **TFM track** is model work. Ordered P0→P2 within each.

### Track D — Data / lifecycle model (enables the predictions)
- **P0 · Cancellation + return legs** (#5, #6) — ✅ **DONE.** `iso_lifecycle.py` emits `camt.056`
  (originator recall) → `camt.029` (CNCL/RJCR resolution) → `pacs.004` (return, parties reversed);
  `account_closed` post-settlement bounce → pacs.002 ACSC + pacs.004. Generator adds
  `cancel_requested` / `cancel_status` / `returned` / `return_reason` labels (cancellation
  feature-modulated → learnable). Label⟺message consistency + reversal unit-tested (177 pass).
- **P1 · Amount split** (#10) — ✅ **DONE (data leg).** Generator adds `InstdAmt` / `InstdCcy` /
  `fx_rate` / `charges` (additive metadata/labels; IntrBkSttlmAmt/Ccy unchanged, no encoder ripple).
  Domestic 1:1 + flat fee; cross-border FX + bps charge. `charges` is bps-of-amount (predictable →
  future regression head); `fx_rate` is market noise. Value-dating (RED/DED/CED) still pending (Rule
  track). FX/charges head is Track M P2.
- **P1 · pacs.009 / pacs.009 COV** (#4) message types.
- **P1 · Repair-type taxonomy** (#17): tag each exception auto-return / repair / reject + a queue.
- **P2** · investigation camt.026/028/029 (#8) · passthrough (#9) · bulk order files (#20).

### Track M — TFM / ML predictions (the model's actual job)
- **P0 · Relabel status → gpi tracker** (#2) — ✅ **DONE.** pacs.002 is the tracker backbone; held
  payment → `ACSP`+`G002` (was `PDNG`); `GPI_SUBCODES` (G001/G002) + a `gpi_tracker` schema block.
  Pure relabel — no new training.
- **P1 · Cancellation-likelihood + return-likelihood heads** (#5, #6) — ✅ **wired** (`run_impute` E/F):
  PR-AUC probes on `f(pain.001/pacs.008)` → "will a camt.056 / pacs.004 arrive?", NaN-guarded for rare
  labels, reported vs prevalence baseline. Smoke sits at no-skill (untrained encoder + rare labels);
  **needs the full run** for lift (the labels are feature-modulated, so lift is expected).
- **P1 · Repair-type / which-queue head** (#17): multiclass on the queue label from Track D.
- **P2 · gpi next-tracker-status** (#2, sequence) — ✅ **DONE** (`run_msgseq.py`): v2 history encoder
  over the pre-outcome message prefix (pain.001/pacs.008/pacs.009) → 3-class tracker outcome
  (ACSC/ACSP-G002/RJCT), vs an order-blind pooled baseline. **Honest result (re-tested):** sequence
  ≈ pooled (−0.6pp macro-F1 after the timestamp fix; the original +2.3pp was noise AND confounded by
  linearly-interpolated timestamps whose dt carried no information). Timestamps are now
  EVENT-ANCHORED to the workflow log and prefixes are ENGINE-VISIBILITY filtered (inward payments
  start at pacs.008 IN — their pain.* legs live at the remote bank), so the null is clean:
  within-payment prefixes are short + fixed-order and add nothing beyond the bag-of-messages. Where
  the history encoder genuinely wins is LONG per-ENTITY histories with temporal correlation
  (run_seq velocity +44pp / C3 +26pp); reproducing that for tracker outcomes needs the generator to
  correlate an account's successive payments — a future data change, documented not faked.
- **P2 · Charges regression** (#10) — ✅ **DONE** (`run_impute` head G): predicts `charges` from
  `f(x)`; beats the mean baseline even at smoke (MAE 145 vs 179) since charges is bps-of-amount and
  the encoder sees the amount. `fx_rate` is deliberately NOT a head — it is market noise.
- Already shipped: STP@pain.001 (A) · reject-reason (B) · ETA (C) · streaming booked (D).
- ✅ **In-flight scoring PRODUCTIZED**: `fit_inflight_head` trains the streaming booked head on
  message-prefix pools (all k, train-split payments only) with the SAME pooling op
  `IndiaScorer.predict_stream` applies at serve time (no train/serve skew); persisted as
  `probes["inflight_booked"]` in the bundle (`meta.inflight`), old bundles fail loudly. CLI
  auto-routes message-stream inputs (`msg_type` column) to streaming. Verified on the held-out
  stream CSV: P(booked) discriminates pre-outcome and sharpens as messages arrive.

### Capability closures (post-productization pass)
- ✅ **Velocity SERVED**: `fit_velocity` pretrains the v2 history encoder over per-account payment
  sequences at save time; bundle persists `hist.pt` + the burst head; `IndiaScorer.predict_velocity`
  + `POST /score/velocity` score an account's recent rows STATELESSLY (the engine owns the history
  store and sends the window — same pattern as /score/inflight). Response carries both the learned
  `burst_proba` and the transparent `burst_rule` for comparison.
- ✅ **Attribution SHIPPED**: `IndiaScorer.explain` — column-occlusion drivers (occlude a field →
  re-embed → probability/ETA delta), faithful by construction, no LLM. `explain: true` on
  `/score/intake` (capped at 10 payments) adds `drivers: [{field, rail_impact, risk_impact,
  eta_impact_min}]` per result.
- ✅ **Temporal-correlation data hook**: `IndiaConfig.exception_momentum` (>0 → an account's
  exception raises its next payment's exception odds, decaying when clean). Verified: P(exc | prior
  exc) exceeds P(exc | prior clean) by >5pp at momentum 0.8. **Default 0.0** so published artifacts
  stay byte-identical; the sequence-encoder-vs-trees re-test on tracker outcomes is now a config
  flag away, not a data-model change.
- ⛔ **Explicitly PARKED (L-effort, revisit at real-data phase):** MT↔MX parsing, investigation
  (camt.026/028/029), forwarding/passthrough hops, bulk order files (SDSC/SDMC/MDMC), value-dating
  (RED/DED/CED). Parked as decisions, not omissions.
- 📌 Known ceiling (documented in code): in-flight heads are linear on pooled prefixes — upgrade
  path is per-k heads or a small sequence model if real-data prefixes get longer/richer.

### Track R — Rule / Lookup / Template (delisted from ML — build once, deterministic)
- **P0 · Geo-Cover (Dom/Intl/SEPA), ON-US/OFF-US, duplicate flag** (#11–13): generator/serve rules —
  emit as *features/labels-for-audit*, not TFM heads.
- **P1 · NNC, cover/reimbursement matching, cutoff/holiday calendar** (#14, #15, #19): rule/lookup;
  the cutoff calendar *feeds* the ETA head (Track M).
- **P1 · Reference-directory enrichment** (#16): inject SSI/RMA/BSD/IBAN as §3.2 offline summaries,
  not join columns.
- **P2 · MT↔MX coexistence** (#21): MT103 parser/template in Layer-1. · Message generation stays
  `write_pacs008` templating.

### Review checklist (run alongside enhancements)
- [x] **Full (non-smoke) run** — validated at real config (8k, full 25M encoder): ETA 186 vs 232
      naive (-20%), charges MAE 97 vs 185 (-48%), streaming booked 0.84→0.87; STP/reason at/below
      naive baselines (C2 confirmed — tree territory); cancel/return heads ~chance at this rarity.
      Servable model retrained on the 20k UPI-free set (rail 0.55 probe / 0.64 tree / status 0.26;
      ETA 199 vs 392). Fresh held-out 100 (known accounts): rail 0.59 (0.64 clean) · risk 0.53.
      **Cold-start finding:** all-new accounts degrade badly (identity features dominate) — noted
      on the deck maturity line; a real-data-phase focus.
- [x] **`limit_exceeded` decision — CLOSED**: stays rule-computed (`violates_cap`), prevalence left
      realistic. It is a deterministic guard, not a prediction; no reason to inflate the data to
      make a delisted task learnable.
- [ ] **Leakage audit as columns grow**: every new column checked for being a deterministic
      consequence of a label (as `SttlmMtd` was) — report per-class / domestic-only / clean slices.
- [ ] **Sparsity guard**: as message types widen the superset schema, add per-column reconstruction
      masking so the encoder isn't rewarded for predicting structurally-UNK cells (see scaling note).
- [ ] **Delisting stays honest**: no TFM head ships for a Track-R capability; rules are unit-tested.
- [ ] **Fidelity**: §5 risk/geo/expense tasks untouched; "freeze means freeze" preserved.

---

## 5. Non-goals
This twin is a **representation-learning demonstration** on synthetic data, not a payments engine. We
do **not** implement real settlement, real SWIFT connectivity, real compliance screening, or the
vendor's specific screens/parameters. The value is showing that **one frozen backbone** predicts the
lifecycle/repair/status labels a real hub generates — especially from partial (initiation-stage) data.

---

## 5b. Second source onboarded: the eFRM channel view

The bank's eFRM request schema (~84 attributes: channel/device/session/audit/payee) is now an
ADDITIONAL SOURCE (`data/efrm_source.py`), following the same multi-source pattern as the ISO
lifecycle: its own event stream + a per-transaction channel-context frame, fused at the
embedding level — never join-widened into the pacs row (no encoder/bundle/test ripple).

**Attribute triage** (full registry in `EFRM_ATTRS`, dispositions unit-tested):
~25 **feature** (device identity/integrity, IP-vs-account geo, session timing,
failedLogins_1hr, credential-change + payee-add events — the ATO axes absent from ISO
messages) · ~20 **duplicate** of the ISO view (amount/ccy/parties/countries/instrument) ·
**label** responseFlag/ErrCode (outcomes — banned from intake features) · **key**
traceIds/userId/sessionId (entity/join keys) · **park** card block + deviceTrustLevel
(another model's output — circularity) · **skip** the 16 extensibility placeholders.

**Synthetic behaviour + measurement** (`run_efrm.py`): a rare ATO episode (new device +
credential change + payee-add + login burst → drain payment) exists only in the channel
source. Measured (held-out): ISO-view-only PR-AUC **0.018** (≈ prevalence — blind by
construction) vs channel/fused **1.00** (the synthetic pattern is deterministic, hence the
ceiling; real data will land between). The point demonstrated: this label class **requires**
the second source; fusion carries it without touching the ISO backbone. Next step on real
data: entity-level fusion through the v2 sequence encoder (userId/deviceId histories), where
the interaction signals live.

## 6. Scaling note — does the tabular method hold as we add columns/tables?

The paper's encoder is a **Transformer over columns-as-tokens**, so scaling behaves as:

- **More columns → natural axis.** Each column is one token; attention is O(n²) in *#columns* (60² is
  nothing). Adding charges/FX/value-dates/Geo-Cover/dup is just more tokens — needs only a bucket
  assignment (categorical / numeric / meta) per column. **Watch:** (a) sparsity dilution — sparse/UNK
  columns make masked-reconstruction reward trivial "predict-UNK" and let the encoder shortcut off the
  mask pattern → add per-column recon masking; (b) leakage grows with width → audit each new column.
- **More *tables* → three different mechanisms:** (1) more message *types* sharing an entity → the
  **superset schema** (what `iso_lifecycle` does), works unchanged; (2) reference/master tables
  (SSI/RMA/BSD/IBAN) → the **§3.2 offline-encode-and-inject-as-a-pooled-token** pattern — do NOT widen
  the row with join columns; (3) long per-entity histories → the **v2 sequence encoder**, not the flat
  one.
- **Breaks when:** union of many *disjoint* schemas (extreme width + sparsity) → move to per-table
  encoders + fusion (beyond the paper); or when the signal is cross-table/temporal (velocity,
  sequences) → the flat encoder can't see it regardless of width, route to v2.
- **Claim tie-in:** breadth of heads is cheap (**C1**, ~0.058 ratio holds as tasks scale); but width
  doesn't rescue **C2** (trees still win on single complete-record feature-rule labels). Width improves
  the imputation / transfer / partial-input story, which is where the backbone is differentiated.
