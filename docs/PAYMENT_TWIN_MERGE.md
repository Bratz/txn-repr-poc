# The Payment Twin — three papers, one stack

## Objective (the twin's constitution)

The objective is a payment twin: one transaction foundation model that answers multiple
question types in line with Raman et al., PRAGMA, and TransactionGPT — **with no conflicts**.
The three apparent conflicts dissolve under two principles:

1. **Frozen base, richer adapters.** All three papers freeze the backbone and train
   parameter-efficient adapters; they differ only in adapter family (Raman's prompt-family
   {Φ, ψ, φ}; PRAGMA's low-rank LoRA deltas - the base stays frozen in LoRA too; TGPT's task
   heads). "LoRA vs the frozen-f invariant" was a false conflict: the invariant forbids
   fine-tuning the giants, and LoRA doesn't. C9 measures the richer adapter family on the
   SAME frozen encoder.
2. **Role and grain separation.** The LLM is the interface (Raman's claim), never the scorer
   (PRAGMA's lane: calibrated probes on frozen state) and never the simulator (TGPT's lane:
   field heads + rollout). Raman's machinery owns the record grain, PRAGMA's shape the stream
   grain, TGPT the trajectory grain - each keeps the representation its paper earned.

Question routing: classification menu → instruction decoder; calibrated PDs / ETA / in-flight
state → probes; forecasting / simulation → generative heads; brand-new question → an
instruction + ψ row, escalating to a LoRA adapter (C9) only if it needs capacity; "why" →
occlusion drivers (ours - all three papers are silent).

The twin is the merge of three architectures, each contributing the mechanism it is best at.
Every row below names the paper the mechanism comes from, where it lives in this repo, and the
measured number that says it works (or honestly doesn't). Nothing in this table is aspirational.

- **Raman et al., arXiv:2410.07851** (JPMorgan) — the replicated core: frozen tabular encoder +
  frozen-LLM instruction decoder. Contributes the **interface**.
- **PRAGMA, arXiv:2604.08649** (Revolut) — encoder-family foundation model over event histories
  with profile state. Contributes the **state-estimation shape** (implemented here convergently,
  before we read the paper — the mapping below is post-hoc and exact).
- **TransactionGPT, arXiv:2511.08939** (Visa) — generative trajectory model. Contributes the
  **simulation tier** (C8, the newest piece).

## Capability map

| Twin capability | Paper mechanism | Where in this repo | Measured |
|---|---|---|---|
| Per-payment representation | Raman §3: columns-as-tokens, §3.1 partitioned high-card embedder, §3.3 currency-conditioned quantizer | `encoder/tabular_encoder.py`, `encoders/` | C1 pass: 0.058 param ratio at matched reconstruction |
| Entity/state estimation over history | PRAGMA history encoder over per-event embeddings + time features | `encoder/history_encoder.py`, `run_seq.py` | C3 +38.5pp vs pooled; C4 +41.3pp vs CatBoost |
| Static profile in the state | PRAGMA profile-state → [USR]; Raman §3.2 party store | `party_encoder.py` store injected as `static_all` in `run_seq` | measured as part of C3/C4 path |
| In-flight lifecycle scoring | PRAGMA-shaped prefix features (ours: pool + last-message one-hots + time) | `serve_india.fit_inflight_heads` / `predict_stream` | 0.860 → 0.942 PR-AUC as messages arrive |
| Velocity / burst | PRAGMA-shaped time-aware history | `serve_india.fit_velocity` | 0.822 vs 0.222 order-blind (+60pp) |
| Forecasting (when/whether/amount) | PRAGMA-style probes on frozen state | `serve_india.fit_next_heads`, `run_next.py` | when: MAE 23.5d beats 25.9/32.5 naives; occurrence/amount lose — reported |
| **Simulation (what-if rollout)** | **TGPT: predict next transaction's fields, roll forward** | **`run_gen.py`: gap-band/amount-band/rail heads on h_USR + `simulate()` through the frozen encoder** | **C8 measured: FAIL 1/3 — gap wins (0.359 vs 0.315), amount/rail lose to per-actor naives on the cadence fleet. Rollout mechanism stands; serving stays ungated** |
| Instruction interface (the Q&A skin) | Raman §4: frozen LLM + {Φ, ψ, φ}, Eq. 5 | `serve_india.fit_instruction_decoder` / `ask`, `POST /score/ask` | C5: +1.3pp for the LLM on a fixed task; C6 pre-registered (decoder vs retired probes) |
| Paper-native sequences (fusion test) | Raman Eq. 5 as a sequence interface + PRAGMA-style time as a column | `run_c7.py` (gap_band core column, spread records) | C7 pre-registered; pass ⇒ history encoder deletable |
| Multi-source fusion | PRAGMA multi-source events; our ISO lifecycle + pain.001 origination context | `data/iso_lifecycle.py`, `data/pain001_context.py`, `run_impute.py` | ATO 0.029 ISO-only → 1.000 fused (planted signal — proves the path, not real fraud) |
| Calibrated probabilities | (none of the three papers; ours) | isotonic per head, `serve_india._fit_iso` | cancel head 0.94 → 0.19 at 3% prevalence |

## Convergences worth naming

- **Time encoding.** Our history encoder feeds `log1p(days-since-previous)` through an MLP plus
  dow/dom/month embeddings (`encoders/time_encoding.py`). PRAGMA uses `8·ln(1+t/8)` under RoPE
  plus fixed-period calendar embeddings; TGPT uses a time-gap field plus calendar fields. Same
  family, three houses. We keep ours: the C3/C4 numbers were earned with it, and PRAGMA's soft-log
  differs from ours only in scale within the compression regime that matters.
- **Discretize the numbers.** Raman quantizes amounts on the input side; TGPT buckets and
  predicts fields; PRAGMA percentile-bins numerics. C8's amount-band head and the C7 gap_band
  column apply the same philosophy to outputs and to time.
- **Nobody serves an LLM for discriminative scoring.** PRAGMA has no LLM; TGPT beats fine-tuned
  LLMs on accuracy and latency; our C5 measured +1.3pp. The LLM's place in the twin is the
  interface (Raman's actual claim), not the scorer.

## Boundaries (owned by all three papers' evidence, and ours)

- **Rules, caps, lookups** never reach a model — capability routing (`docs/CBPR_TWIN_GAP.md` §3.5).
- **Relational/AML** is out of scope for record-level models: PRAGMA reports −47% F0.5 vs a
  network-aware baseline; our capability doc says the same. A graph tier is future work everywhere.
- **Rare-event PDs at 20k scale** sit at prevalence (cancel/return) — measured null, kept visible.

## The open experiments that decide the final shape

| Claim | Question | If it passes |
|---|---|---|
| C6 | does the instruction decoder match the retired probes at full scale? | decoder-only serving for the class menu |
| C7 | do Eq. 5 sequences match the v2 history encoder? | the one non-paper component is deleted; twin = pure paper stack |
| C8 | do the generative heads beat per-actor naives at full scale? | the simulation tier is real; serving integration follows |

Runbook runs F (C6), G (C7), H (C8) — one command each on any CUDA box.
