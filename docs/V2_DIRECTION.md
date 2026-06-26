# Layer 3b — sequence modelling over entity histories (v2 spec)

**Status:** v2 design. This is *beyond* arXiv:2410.07851 and must not change, retune, or
void anything in v1. It is a new experiment with its own falsifiable claims. v1 stays the
faithful single-record replication; v2 is the sequence extension the production transaction
foundation models (Revolut PRAGMA, arXiv:2604.08649; Visa TransactionGPT, arXiv:2511.08939)
all converge on.

The one-line thesis: the POC already has a per-transaction encoder. v2 wraps a second encoder
on top of it that reads an entity's *ordered history* of transactions and produces one
representation per entity. That is the axis trees can't see, and it's where a learned
representation finally beats a gradient-boosted baseline.

---

## Implementation status (smoke-validated, CPU)

Built and tested: event-time encoding (`encoders/time_encoding.py`), sequence assembly with a
held-out-by-actor split (`data/sequence_assembly.py`), the Layer-3b history encoder with
masked-event + CoLES objective (`encoder/history_encoder.py`), the §7 behavioural generator
(`data/synth_sequences.py`), the decoder precomputed-feature path for Option B
(`decoder/multimodal_decoder.py`), and the orchestrator `run_seq.py` that measures C3/C4/C5.
The party-store `[USR]` injection is wired with a `--no-static` ablation. 111 tests pass.

The §7 generator puts the regime signal in the **order only**: Stable vs Shift accounts share
the same gap multiset and amount distribution, so every order-invariant aggregate matches and
only a model that reads the ordered, timed sequence can separate them. On that data, the smoke
run gives (held-out accounts, prevalence ~0.44):

| Claim | Measured | Threshold | Verdict |
|---|---|---|---|
| C3 temporal lift (sequence vs order-blind pooled) | +27.7 pp (0.79 vs 0.51 PR-AUC) | `>= +10 pp` | **pass** |
| C4 held-out gen. (sequence vs CatBoost on aggregates) | +31.2 pp (0.79 vs 0.48 PR-AUC) | `>= +5 pp` | **pass** |
| C5 LLM necessity (Option A probe vs Option B LLM) | A 0.79 vs B 0.56 | drop if A within 2pp | **drop LLM** |

C4 is the first result in the project where the learned representation beats the gradient-boosted
baseline - because the signal is genuinely temporal and the aggregates are matched, so the tree
sits at chance. Caveats: these are smoke numbers (tiny encoder, few epochs, **MockLLM for C5**),
so they validate the experimental design and pipeline, not a headline claim - a full run (real
Phi for C5, paper-scale encoder) is what turns them into one. The regime is synthetic
order-structure; real behavioural data is the next wall.

---

## 0. The decision this rests on — what is the "entity"?

PRAGMA models a retail user's life as a sequence of events. Our data is pacs.008
interbank/corporate payments generated per (debtor, creditor) pair, not per-user timelines.
Before any code, pick the actor whose history we model:

- **Default: the debtor account** (`DbtrAcct_Id`). Its history = its outgoing payments,
  time-ordered. This is the cleanest "actor" in wholesale flow.
- Optional later: model both debtor- and creditor-perspectives, or the parent (`UltmtDbtr_Id`).

Everything below assumes *account-as-actor*. This reframing is the real work; the rest is
mechanical. Note it honestly: the per-entity-history framing fits retail better than wholesale,
so v2 is also a domain bet, not just an architecture change.

---

## 1. What is reused vs. what is new

| Piece | Source | v2 status |
|---|---|---|
| Per-transaction field encoders (Layer 2) | `encoders/column_assembler.py` | **reuse as-is** — this is the per-event encoder |
| Per-transaction embedding `e_t = f(x_t)` (Layer 3 `[CLS]`) | `encoder/tabular_encoder.py` | **reuse** — freeze it, it emits the per-event vector |
| Currency-conditioned amount quantizer (§3.3) | `encoders/quantizer.py` | **reuse** |
| Party store (static counterparty summary) | `encoders/party_encoder.py` | **reuse** as the `[USR]`-style static token |
| `batch_hard_triplet_loss` | `encoder/tabular_encoder.py` | **reuse** for the CoLES-style objective |
| Event-time encoding (inter-arrival + calendar) | — | **new** (Layer 2 addition) |
| History encoder over `[e_1 … e_n]` → entity vector | — | **new** (Layer 3b) |
| Whole-event masking objective | — | **new** |
| Held-out-entity eval split | — | **new** (was a walked-back v1 item; legitimate in v2) |

Roughly half of PRAGMA already exists in Layers 1–2. The new code is the sequence axis.

---

## 2. Layer 2 addition — event-time encoding

A single transaction currently carries its date only as a core categorical. A sequence needs
*time between events*. For each event `t` in an entity's ordered sequence, compute:

- `dt_t` = days since the entity's previous event (inter-arrival; `dt_0 = 0`).
- calendar features: day-of-week, day-of-month, month.

Encode them and **add** to the per-event embedding (keep width `D = config.hidden`):

```
tau(dt)      = MLP( ln(1 + dt) )            # continuous inter-arrival, log-compressed
cal(t)       = E_dow[dow] + E_dom[dom] + E_month[month]
e_t          = f(x_t) + tau(dt_t) + cal(t)  # (D,)
```

`ln(1 + dt)` mirrors PRAGMA's `8·ln(1 + t/8)` magnitude compression. The MLP is a 2-layer
`1 -> D/2 -> D` head. New module: `encoders/time_encoding.py`.

---

## 3. Layer 3b — the history encoder

**Input.** For an entity with `n` time-ordered events: the static profile token `z_USR`
(the account's party-store summary, already `D`-dim), prepended to the event embeddings:

```
Z = [ z_USR , e_1 , e_2 , … , e_n ]          # (1 + n, D)
```

**Architecture.** A transformer over `Z`. Use **bidirectional** for the representation objective
(PRAGMA's choice) — switch to causal only if you later add next-event generation
(TransactionGPT's choice). Positions: RoPE or learned, over event order; irregular spacing is
already carried by `tau(dt)`. Cap sequence length (PRAGMA keeps the most recent 6,500 events;
start with 256 for the POC scale).

**Output.** The `z_USR` output position is the entity representation `h_USR (D,)`. Per-position
outputs feed the masked-event objective.

```
H            = HistoryEncoder(Z)             # (1 + n, D)
h_USR        = H[0]                          # entity representation
h_events     = H[1:]                         # per-event contextual reps (for masking loss)
```

New module: `encoder/history_encoder.py`. Hidden `D` matches Layer 3 so `e_t` and `h` share
width. Sizing (`layers`, `heads`) is a v2 config choice, not paper-pinned.

---

## 4. Pretraining objective

Two options; build the first, keep the second as the contrastive alternative.

**(a) Whole-event masking (default, PRAGMA-style).** Mask ~15% of events (replace `e_t` with a
learned `mask_evt` vector) and reconstruct each masked event's discrete fields from context,
reusing the Layer-3 head pattern:

```
L_mask = sum over masked t of [
            CE( head_amt(h_events[t]),   amount_level_t )      # §3.3 quantizer level
          + CE( head_region(h_events[t]),creditor_region_t )
          + CE( head_ind(h_events[t]),   creditor_industry_t )
          + CE( head_mtd(h_events[t]),   settlement_method_t ) ]
```

This is the sequence analogue of masked-column reconstruction: predict a *whole transaction*
from the ones around it. It's what teaches behaviour-from-context.

**(b) CoLES-style contrastive (reuses existing code).** Take two disjoint subsequence windows of
the same entity as a positive pair; other entities are negatives; apply the existing
`batch_hard_triplet_loss` on `h_USR`. This is the sequence-level generalisation of v1's
two-masked-views triplet — a clean reuse, and consistent with §3.4's metric-learning bet.

Composite: `L = L_mask + lambda · L_triplet`, mirroring Layer 3's composite loss.

---

## 5. Training stages and the freeze boundary

Three frozen handoffs, same discipline as v1:

1. **Stage A** — pretrain Layer 2 + Layer 3 per-transaction encoder (as today). Freeze. It now
   emits `e_t` as a constant feature.
2. **Stage B** — pretrain Layer 3b over sequences of frozen `e_t` (+ time encoding) with the
   objective in §4. Freeze. It now emits `h_USR`.
3. **Downstream** — a small head on frozen `h_USR` (see §6).

The v1 freeze invariant is preserved within each stage. The only thing that ever trains
downstream is the per-task head.

---

## 6. Layer 4 fork — keep or drop the frozen LLM

This is the decision PRAGMA forces. PRAGMA gets +130% credit scoring, +40% recommendation, and
+17% fraud precision from `h_USR` plus a **linear probe or LoRA (~2–4% of weights)** — no
language model anywhere. v1 routes everything through a frozen 1.3B Phi.

- **Option A (recommended default): drop the LLM downstream.** Attach a linear probe or small
  MLP / LoRA head to `h_USR`. Cheaper, faster, simpler; this is the production recipe.
- **Option B (keep the language interface): feed `h_USR` as the `Phi(·)` record token into the
  existing Layer-4 decoder** (`decoder/multimodal_decoder.py`), unchanged except that the record
  token is `h_USR` instead of `f(x)`. Keep this only if you need to pose tasks in natural
  language and get text answers — the source paper's actual purpose, which PRAGMA gives up.

Build A first. It's the cheaper falsifiable win, and C5 below tests whether B earns its cost.

---

## 7. Data — sequence assembly (and an optional generator upgrade)

**Minimal (no generator change):** add `data/sequence_assembly.py` that groups rows by the actor
key (`DbtrAcct_Id`), sorts by `IntrBkSttlmDt`, computes `dt_t`, and caps to the most recent `N`.
The fields already exist; this is grouping, not generation.

**Richer (generator upgrade, recommended for a real signal):** extend `data/synth_pacs008.py`
so an account's stream has *learnable temporal structure* — salary-like cadence, spending drift,
a dormant-then-active reactivation, a burst. Without this the masking objective has little to
learn and the time-dependent tasks are degenerate. This also seeds the eventual structuring/graph
work (still v2+, still needs a network dimension PRAGMA itself lacks — it fails AML by −47% F0.5
for exactly that reason).

---

## 8. Evaluation — held-out-entity, and the tasks that appear

The sequence axis makes two things possible that v1 couldn't do:

- **Held-out-entity split:** train on a set of accounts, test on accounts never seen in training.
  This is the regime where a representation should beat a tree (the tree memorises per-id
  statistics that don't transfer; `h_USR` generalises from attributes + behaviour). It was a
  walked-back v1 item; in v2 it's the headline eval.
- **Per-entity tasks:** account-level risk, dormancy/churn, recurring-relationship detection done
  *temporally* (not as an unordered set), early-fraud-from-cadence.

---

## 9. New falsifiable claims (set thresholds before the run, v1-style)

| Claim | Statement | Metric | Threshold |
|---|---|---|---|
| **C3 temporal lift** | the sequence model beats the v1 single-record adapter on a time-dependent task | PR-AUC gain | `>= +10 pp` |
| **C4 held-out generalization** | on unseen accounts, `h_USR` + linear head beats CatBoost on flattened features | PR-AUC gain | `>= +5 pp` |
| **C5 LLM necessity** | dropping the LLM (linear probe on `h_USR`) stays close to the LLM-decoder on the same tasks | PR-AUC gap | within `2 pp` -> drop the LLM |

C4 is the one that matters: it's the first claim in this whole project where the representation
is *expected* to win, because it's the first task that isn't a single-record rule over visible
fields. C5 turns the §6 fork into a measured decision instead of an opinion.

---

## 10. File-level plan

| File | Change |
|---|---|
| `encoders/time_encoding.py` | new — inter-arrival + calendar encoding |
| `data/sequence_assembly.py` | new — group rows into per-entity ordered sequences |
| `encoder/history_encoder.py` | new — Layer 3b transformer + masking + CoLES loss |
| `data/synth_pacs008.py` | optional — per-account behavioural temporal structure |
| `configs/default.yaml` | new `v2_history_encoder:` block, clearly marked beyond-paper |
| `decoder/multimodal_decoder.py` | minimal — accept `h_USR` as the record token (Option B only) |
| `eval/` | held-out-entity split + per-entity task metrics |
| `run_gpu.py` | a `--seq` path: Stage A -> Stage B -> downstream |

---

## 11. Evidence map (why each decision is grounded)

- *Sequence axis, masked-event objective, frozen-encoder + light head* — PRAGMA
  (arXiv:2604.08649): two-branch encoder, multi-level masking (token/event/key), downstream via
  embedding probe or LoRA (~2–4%).
- *Static profile branch ≈ party store* — PRAGMA's profile-state `[USR]` token vs. event branch.
- *Inter-arrival time encoding* — PRAGMA's `8·ln(1+t/8)`; Time2Vec / FATA-Trans (arXiv:2310.13818).
- *CoLES contrastive option* — Babaev et al. (arXiv:2002.08232), subsequences of one entity as
  positives.
- *Generative alternative (next-event)* — TransactionGPT (arXiv:2511.08939), which predicts
  `time_gap, amount, merchant, MCC` of the future transaction.
- *Drop-the-LLM fork* — PRAGMA reaches production results with no general LLM; the LLM in v1 is
  inherited from the source paper, not from the production state of the art.
- *Graph still missing for AML* — PRAGMA underperforms its baseline by −47% F0.5 on AML because it
  "processes event histories in isolation." Sequence alone does not solve structuring; that needs a
  network dimension, which is v2+ beyond this spec.

---

## 12. Open decisions to settle first

1. **Actor:** debtor account only, or both perspectives, or parent? (Section 0.)
2. **Generator:** minimal grouping, or invest in behavioural temporal structure? (Section 7.)
   The masking objective is close to degenerate without the latter.
3. **LLM:** build Option A only, or A and B for the C5 comparison? (Section 6.)
4. **Scope honesty:** v2 is a domain bet (wholesale -> per-account histories) as much as an
   architecture change. Decide whether the synthetic wholesale data is the right testbed, or
   whether v2 waits for data with genuine per-account behaviour.
