# Prototype Architecture — Transaction Representation Learning for Fraud/Risk

**Status:** v1 prototype (grounded). No extensions beyond the source paper.
**Use case:** Fraud / risk tagging on ISO 20022 (pacs.008) payments.
**Data:** Synthetic only (no real transaction data). Generator already built.

---

## 1. Grounding

Faithful replication of:

> Raman, Ganesh, Veloso (JPMorgan AI Research), *Scalable Representation
> Learning for Multimodal Tabular Transactions*, arXiv:2410.07851,
> NeurIPS 2024 Table Representation Learning workshop.

Every component below cites the paper section it implements. The prototype is an
**audit against the paper**, not a paper-inspired system: each deviation is
explicit and justified (see §6). If a component cannot be traced to a paper
section or a forced-departure entry, it does not belong in v1.

### Scope

In scope (replication):
- §3.1 partitioning embedder for high-cardinality identifiers
- §3.2 offline meta-column (party) encoder with pooled summary injection
- §3.3 adaptive numerical quantization
- §3.4 composite loss (reconstruction + batch-hard triplet)
- §4 / §4.1 frozen-encoder + frozen-LLM multimodal decoder with adapters

Out of scope (walked back — these are v2, not v1):
- data-completeness feature vector
- multi-record structuring / layering chain task
- held-out-typology generalization split

---

## 2. Two planes

The system splits along the paper's frozen-vs-trained boundary.

### Offline / training plane (batch, single GPU)
```
synthetic generator → Layer 1 projection
  → party encoder pretrain → party-embedding store
  → tabular encoder pretrain (frozen after this)
  → adapter instruction-tuning (only Φ, ψ, φ trained)
```

### Online / scoring plane (inference)
```
incoming pacs.008 → Layer 1 projection
  → field encoders (party store is a LOOKUP here, not compute)
  → frozen tabular encoder → frozen LLM + trained adapters
  → risk label / score → eval or alert sink
```

The asymmetry is the deployability argument: expensive party-relationship
learning is amortized offline; inference is a store lookup plus two frozen
forward passes plus a small adapter. This is what fits a payments engine's
latency budget — state it explicitly in any writeup.

---

## 3. Layers

### Layer 1 — Projection (`pacs.008 → tabular row`)
- **Paper anchor:** none. This is the data-engineering that makes ISO 20022
  conform to the paper's input contract `x = (x_c)` over `C` columns.
- **Function:** parse each `CdtTrfTxInf`, project into a fixed column schema,
  type every column into one of four buckets (below). Deterministic ETL only.
- **Contract:** consumes pacs.008 (or the generator's projected rows);
  produces a flat row + `column_schema.json` bucket manifest.
- **Status:** ✅ built (`synth_pacs008.py`, `project_to_pacs008`).

### Layer 2 — Field encoders (four parallel paths → one column-embedding set)
| Path | Paper | Fields | Notes |
|---|---|---|---|
| Partitioning embedder | §3.1 | `DbtrAcct_Id`, `CdtrAcct_Id`, `UltmtDbtr_Id`, `UltmtCdtr_Id` | B=4, α_v=−3, α_d=2.25 (paper values, do not retune) |
| Adaptive quantizer | §3.3 | `IntrBkSttlmAmt` | **currency-conditioned** (forced departure — quantize within `Ccy`) |
| Offline party encoder | §3.2 | party `*_Nm`, `*_Ctry`, `*_Industry`, `*_SubIndustry` | pre-learned offline, pooled summary injected inline; objective = masked-attribute reconstruction (chosen — see §7) |
| Standard embeddings | — | `Ccy`, `IntrBkSttlmDt`, `SttlmMtd` (core) | inline |

### Layer 3 — Tabular encoder
- **Paper anchor:** §3.4 + BERT backbone.
- Bidirectional transformer, masked-column reconstruction, **composite loss**
  = reconstruction + batch-hard triplet; positives = two perturbed views of `x`.
- Size 25M (paper "small"); 3 epochs (paper ablation sweet spot).
- **Output:** one dense transaction embedding per row. Frozen after pretraining.
- **Do not** substitute a generic contrastive loss — the triplet term is the
  claim under test (paper's *Classical Loss* ablation defends it).

### Layer 4 — Multimodal decoder
- **Paper anchor:** §4 / §4.1.
- Frozen tabular encoder `f` + frozen small LLM (Phi-class) + trainable
  `{Φ, ψ, φ}` only:
  - `Φ` — adapter layers transforming `φ(f(x))`
  - `ψ` — task embedding `Ξ_task: 1..K → ℝ^D` (task-unique + shared subspace)
  - `φ` — per-layer LLM prompt-tuning params
- Interleaving (eq. 5):
  ```
  z_i = Ξ_LLM(s(1)) ⊕ φ(f(x_i1)) ⊕ … ⊕ Ξ_LLM(t_i) ⊕ Ξ_task(k_i)
  ```
- Objective (eq. 6): `L = −Σ log P(y_i | z_i; Φ, ψ, φ)`
- Row sentinels `s(·)` = `[R1], [R2], …` are LLM-vocabulary tokens.
- v1 uses **single record per example** (mirrors paper's risk/geo/expense
  tasks). No multi-record task in v1.
- **Invariant:** if `f` or the LLM ever train, it is a different experiment.

### Layer 5 — Baselines & evaluation harness
- Runs alongside, not after.
- Baselines: CatBoost on raw flattened features; optional full fine-tune for the
  parameter-efficiency comparison.
- **Metrics:** PR-AUC, recall@fixed-FPR, F1@threshold. **Not accuracy** — the
  risk label is imbalanced (High ≈ 2%), so accuracy is misleading by
  construction. Report accuracy *alongside* only for direct comparability to the
  paper's balanced-task tables.

---

## 4. Column buckets → mechanism (the core design decision)

```
high_card_categorical → §3.1 partitioning embedder
numerical             → §3.3 adaptive (currency-conditioned) quantizer
meta_party            → §3.2 offline party encoder (pooled summary)
core                  → standard inline embeddings
label: risk_label     → §4 risk-tag slot (Low / Medium / High)
```
Authoritative bucketing lives in `column_schema.json` (emitted by Layer 1).
Every downstream layer reads buckets from there — never hard-code column lists.

---

## 5. Component contracts

| Component | Consumes | Produces |
|---|---|---|
| Generator | `GenConfig` | projected rows + `column_schema.json` |
| Layer 1 (live) | pacs.008 XML | projected rows (same schema) |
| Party encoder | `meta_party` columns | party-embedding store (keyed lookup) |
| Tabular encoder | column embeddings | row embedding `f(x)`; frozen weights |
| Decoder | `f(x)` + instruction text + task id | risk label / score |
| Eval harness | predictions + labels | PR-AUC, recall@FPR, F1 |

The **party-embedding store** is a first-class persistent store, not a cache —
it is where counterparty intelligence accumulates. The paper implies it (§3.2
offline pre-learning) but does not dwell on it.

---

## 6. Deviations ledger (keep visible in any writeup)

**Faithful replications — change nothing:**
partitioning embedder + hyperparameters; meta-column offline split; composite
loss; frozen-encoder/frozen-LLM adapter design; row sentinels; 25M size /
3-epoch sweet spot.

**Forced departures — justified, not chosen (only three):**
1. pacs.008 column schema replaces the paper's generic synthetic columns.
2. Currency-conditioned quantization — without it the quantizer is *wrong* on
   multi-currency data; this is a correctness fix, not an enhancement.
3. Imbalance-aware metrics replace accuracy — forced by the imbalanced label.

**Walked back — explicitly excluded from v1:**
completeness vector; chain/structuring task; held-out-typology split.

---

## 7. Open decision (decide deliberately, document it)

The paper says meta-columns are "pre-learned offline" but does not specify the
party encoder's training objective. **Chosen for v1:** masked-attribute
reconstruction over party blocks — most defensible, methodologically consistent
with Layer 3. This is the spot a careful reviewer will push; the choice is
documented here on purpose.

---

## 8. Tooling

PyTorch + HuggingFace; Phi-1.5-class LLM; CatBoost; single GPU. The whole
prototype is a few focused weeks, not a quarter.

Pin `column_schema.json` against the targeted pacs.008 version (`.001.08`,
`.001.10`, …); field deltas across versions must not silently drop columns.
