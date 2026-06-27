# Payment-level digital twin (on the TFM backbone)

**Status:** beyond arXiv:2410.07851 — an application of the backbone, not a change to v1/v2.
This builds the **behavioural / predictive layer** of a payment-level digital twin: given a
payment, forecast the statuses and exceptions it is likely to hit across its workflow
lifecycle. It is **not** a full digital twin — the deterministic workflow logic, the rules,
and system/queue state belong to a separate process/simulation layer (see "What this is not").

## What a payment-level twin is here

The orchestrator walks a payment through a workflow; each step is **clean** or throws an
**exception**; all-clean → release to SWIFT (outward) or post/create the account (inward). The
twin predicts that trajectory:

- **At intake (v1):** the payment record → `f(x)` (frozen encoder) → frozen-rep probes for
  *exception likelihood* (multi-label), *terminal status* (STP / REPAIRED / MANUAL_REVIEW /
  REJECTED), and *time-to-settle* (ETA).
- **In-flight (v2):** as steps complete, the `(step, outcome, time)` prefix → the reused
  Layer-3b history-encoder backbone → *next-step exception* prediction, sharpening per step.

## Architecture (what it reuses)

| Piece | Reuses |
|---|---|
| Payment → `f(x)` | the frozen v1 encoder (Layers 1–3), unchanged |
| Intake heads | linear probes on `f(x)` (the Option-A pattern) |
| Step events → sequence | the v2 Layer-3b `HistoryEncoder` (architecture only) + a new `StepEmbedder` |
| Next-exception head | a linear head on the history representation, trained supervised |

New code: `data/synth_workflow.py` (the workflow-log generator), the `StepEmbedder` and
`Inflight` model + orchestration in `run_twin.py`. Nothing in v1/v2 is modified.

## Data — `data/synth_workflow.py`

Simulates payments traversing the **outward** (`validation → enrichment → sanctions →
fraud_aml → limit_liquidity → routing → settlement`) and **inward** (`… → account_resolution
→ posting`) workflows. Each step's exception probability is a transparent function of the
payment's features (cross-border, amount, region, industry) plus noise — so exceptions are
learnable from the representation. It emits two tables: **payment-level** (pacs.008 features +
`direction` + exception multi-labels + `terminal_status` + `time_to_settle_min`) and
**event-level** (`payment_id, seq, step, outcome, excode, t_min`).

## The four outputs (and the CPU smoke result)

Smoke = tiny 1-epoch encoder + CPU; numbers validate the *pipeline*, not a headline.

| Output | Stage | Smoke result | Read |
|---|---|---|---|
| Exception likelihood | intake (v1) | sanctions PR-AUC **0.41** vs ~0.35 prevalence; rare ones ≈ chance | the strong feature-driven exception is learned; the rest need a real encoder |
| Terminal status | intake (v1) | acc 0.23 vs 0.41 majority | weak — balanced head + 1-epoch encoder |
| Time-to-settle (ETA) | intake (v1) | no lift over the mean | weak — needs a trained encoder |
| Next-step exception | in-flight (v2) | next-any-exception PR-AUC **0.31** vs 0.14 prevalence (~2.2×) | genuinely above chance; loss falls 0.53→0.37 |

The two feature-/sequence-driven targets (sanctions at intake, next-any-exception in-flight)
show real signal even at smoke scale; status/ETA/rare-exceptions are limited by the smoke
encoder and should improve with a paper-scale GPU run.

## What this is NOT (the honest boundary)

- **Not the whole digital twin.** It's the learned behavioural layer. A full twin also needs
  the **deterministic process model** (the workflow DAG + rules/gates — replicated, not
  learned), live **system/queue state**, and a **discrete-event simulation** for system-level
  what-if. The twin forecasts; the orchestrator still runs the steps and makes the release gate.
- **Deterministic rule-exceptions** (missing mandatory field, bad checksum) belong to rules,
  not the model — a learned predictor adds nothing there (the v1 C2 lesson).
- **Workflow branching / causal DAG** and **multi-payment / system-state** effects are outside
  a single-payment model (the same graph/relational gap as AML).
- **Synthetic + smoke.** Real workflow logs and a GPU run are needed for headline numbers.

## Run

```bash
python data/synth_workflow.py --accounts 4000 --payments 60000 \
  --out-prefix data/pacs008_twin --schema-out data/column_schema_twin.json
python run_twin.py                 # full; --smoke for the CPU check
```

## Next steps

1. GPU run with the paper-scale encoder → real intake/in-flight numbers.
2. Train the in-flight model on **real workflow logs** (the binding constraint).
3. Add **system-state/context features** (cutoffs, queue depth, downstream status).
4. For a *system* twin: wrap the twin's stochastics in a discrete-event simulation of the
   workflow DAG + queues.
