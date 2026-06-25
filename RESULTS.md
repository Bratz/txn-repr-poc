# Results — C1 & C2 (full GPU run, 2026-06-14)

Outcome of the end-to-end run (`run_gpu.py`). Both falsifiable claims resolved
with numbers against the `configs/default.yaml` thresholds.

> **Scope note (task suite).** The numbers below are from the original
> **risk-only** decoder run. The build now covers the paper's full §5 task suite —
> risk, geography, expense (single-record) and recurrence (multi-record, Eq. 5) —
> trained jointly. Per-task metrics (geography/expense accuracy, recurrence
> PR-AUC) are emitted under `C2_per_task` in `results.json`; the multi-task GPU
> numbers will be recorded here after the next full run. C1 and the C2 risk
> headline are unchanged.

## Run configuration

| | |
|---|---|
| Data | 1,000,000 synthetic pacs.008 rows (`--parents 20000 --transactions 1000000`) |
| Realized vocab | combined account-id ≈ **119,819** (≈ the paper's ~125K), parent ≈ 19,895 |
| Risk distribution | Low 626,359 / Medium 355,478 / **High 18,163 (1.8%)** |
| GPU | 1× NVIDIA H200, **fp32** |
| Tabular encoder | pinned **25M params, 3 epochs** (§3.4), composite recon + batch-hard triplet |
| Decoder | frozen Phi-1.5 (fp32) + trainable {Φ, ψ, φ}; **prompt-mode φ**; **1 epoch** (paper §5.2) |
| Baseline | CatBoost, 300 iters, balanced class weights, raw flattened features |
| Eval | imbalance-aware; positive class = **High**; same stratified eval split for adapter & CatBoost |

## C1 — partitioning embedder (§3.1): **CONFIRMED ✅**

| metric | measured | threshold | verdict |
|---|---|---|---|
| high-card param ratio (partitioned / classical) | **0.058** | ≤ 0.55 | ✅ |
| mean top-1 recon gap (classical − partitioned), pp | **−0.012** | ≤ 1.0 | ✅ |

The partitioned embedder **matches classical masked-column reconstruction (marginally
better) at ~5.8% of the embedding-table parameters** — the paper's headline §3.1
claim, replicated at near-paper vocab scale. The negative gap means partitioned was
not worse than the dense control on the reconstruction metric.

## C2 — adapter vs baselines (§4): **param-efficient ✅, does NOT beat CatBoost ✗**

| model | PR-AUC | recall@1%FPR | F1@op | accuracy |
|---|---|---|---|---|
| CatBoost | **0.655** | 0.784 | 0.674 | 0.871 |
| Adapter (frozen f + frozen Phi + {Φ,ψ,φ}) | **0.210** | 0.243 | 0.293 | 0.841 |

| metric | measured | threshold | verdict |
|---|---|---|---|
| trainable-param ratio (adapter / full-tune) | **0.0059** (7.64M / 1.3B) | ≤ 0.10 | ✅ |
| PR-AUC gain vs CatBoost, pp | **−44.5** | ≥ +10 | ✗ |

**Interpretation (honest).** The adapter learned a genuine signal — PR-AUC 0.21 is
~10× the 0.018 random-prevalence floor — but it loses to CatBoost. This is a property
of the task, not a defect:

- The synthetic risk label is a **transparent rule over raw features** (cross-border,
  currency, region, industry, amount thresholds — see `assign_risk` in
  `data/synth_pacs008.py`), which gradient-boosted trees capture **directly**.
- The adapter must route that signal through a **frozen encoder compressed to a single
  ~D-dim token**, a **frozen LLM**, and a small prompt-tuned head trained for **1 epoch**.
  That bottleneck structurally favors GBDT on a feature-rule task.

This does **not** refute the paper, whose C2 concerns parameter-efficiency and
*rivaling full fine-tuning* on its own tasks; it shows that **our chosen
"beat CatBoost by ≥10pp" bar is not met on this synthetic, GBDT-friendly label.**
Parameter efficiency (the other half of C2) is confirmed: the adapter trains 0.59%
of a full fine-tune's parameters.

## Bottom line

- **C1 (partitioning parameter efficiency): confirmed** at scale — the prototype's
  primary, paper-faithful result.
- **C2: half confirmed** — adapter is extremely parameter-efficient, but does not beat
  CatBoost on this rule-based synthetic task.

## Reproduce

```bash
python data/synth_pacs008.py --parents 20000 --transactions 1000000 \
    --out data/pacs008_synth.parquet --schema-out data/column_schema.json
python run_gpu.py                 # full run → results.json
python run_gpu.py --limit 20000   # fast full-path check (~2-3 min on a modern GPU)
```

## Not retuned (fidelity)

These numbers are from the **pinned configuration** (B=4, α_v=−3, α_d=2.25; 25M/3-epoch
encoder; 1-epoch decoder). Per the §0 guardrails, the C2 gap was **not** closed by
retuning. Possible v2 *exploration* (would be reported as exploration, not the headline):
per-layer prefix-φ via `peft`, more decoder epochs, or a less GBDT-friendly task.
