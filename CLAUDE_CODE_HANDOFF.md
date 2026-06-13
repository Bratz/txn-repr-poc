# Claude Code Hand-off

Hand-off for executing the transaction-representation-learning prototype with
Claude Code. Read `architecture.md` first — it is the source of truth for *what*
each component is. This file covers *how to build it, in what order, and what
"done" means per phase*.

---

## 0. Read-me-first guardrails (do not violate)

This prototype's value is that it **faithfully replicates a specific paper**
(arXiv:2410.07851). Fidelity is the deliverable. Therefore:

1. **Do not extend beyond the paper in v1.** If you find yourself adding a
   feature that improves results but isn't in the paper, stop — it goes in a
   `v2/` backlog note, not the build. Three extensions were already
   deliberately walked back (completeness vector, chain/structuring task,
   held-out-typology split). Do not reintroduce them.
2. **Do not retune the paper's hyperparameters.** Partitioning embedder is fixed
   at `B=4, α_v=−3, α_d=2.25`. Encoder is 25M params / 3 epochs. These are the
   experiment, not knobs.
3. **Freeze means freeze.** In Layer 4, only `{Φ, ψ, φ}` train. The tabular
   encoder `f` and the LLM are frozen. If you unfreeze either, you have run a
   different experiment and the headline result is void.
4. **Read buckets from `column_schema.json`.** Never hard-code column name lists
   in a downstream module.
5. **Only three departures from the paper are allowed**, all already decided:
   pacs.008 schema, currency-conditioned quantization, imbalance-aware metrics.
   Any fourth departure must be raised, not silently introduced.

When in doubt, prefer the paper's choice and leave a `# PAPER: §x.y` comment so
the mapping travels with the code.

---

## 1. Repo layout (target)

```
.
├── architecture.md
├── CLAUDE_CODE_HANDOFF.md          (this file)
├── data/
│   └── synth_pacs008.py            ✅ done — Algorithm 1 generator + Layer 1
├── encoders/
│   ├── partitioning_embedder.py    Phase 2a  (§3.1)
│   ├── quantizer.py                Phase 2b  (§3.3)
│   ├── party_encoder.py            Phase 2c  (§3.2)
│   └── column_assembler.py         Phase 2d  (assembles Layer 2 → encoder input)
├── encoder/
│   └── tabular_encoder.py          Phase 2e  (§3.4 + BERT, composite loss)
├── decoder/
│   └── multimodal_decoder.py       Phase 3   (§4/§4.1, frozen f + LLM)
├── eval/
│   ├── baselines.py                Phase 4   (CatBoost, optional full-tune)
│   └── metrics.py                  Phase 4   (PR-AUC, recall@FPR, F1)
└── configs/
    └── default.yaml                shared config (paper hyperparameters pinned)
```

---

## 2. The plan (phases)

### Phase 0 — Falsifiable claims (do before any modelling)
Write the success thresholds down so the POC can't sprawl. Two claims under
test:
- **C1 (encoder):** partitioned embedder matches classical embeddings at ~½ the
  parameters (paper: 100M vs 185M, comparable reconstruction).
- **C2 (decoder):** frozen-LLM + adapter beats CatBoost and rivals full
  fine-tune at a fraction of trainable params.

Pick concrete numbers, e.g. *"adapter model beats CatBoost PR-AUC by ≥10 points
using <10% of full-tune trainable params."* Record in `configs/default.yaml`.

**Exit:** thresholds committed.

### Phase 1 — Synthetic data + projection ✅ DONE
`data/synth_pacs008.py` implements Algorithm 1 (`M_Comp` GMM, `M_Dest`,
`M_Txns`, `M_Amount`, `M_Date`, `CreateAccount`) and Layer 1 projection into
pacs.008-typed buckets. Risk label sits in the Low/Med/High risk-tag slot.

Realized at 4K parents / 200K txns: ~23.7K combined account-ID vocab, ~4K
parent-ID vocab, risk Low 62.9% / Med 35.3% / High 1.8%.

**To scale toward the paper:** `--parents ~20000 --transactions 1000000` pushes
the account vocab toward the paper's ~125K, where the partitioning embedder
actually bites.

**Exit:** ✅ generator + `column_schema.json` produced and verified.

### Phase 2 — Tabular encoder (§3.1–§3.4)
Build Layer 2 field encoders, then Layer 3.
- **2a partitioning_embedder (§3.1):** replace classical `E ∈ ℝ^{|V|×D}` with
  binned `E^b ∈ ℝ^{|V^b|×D^b}`; bins via power law; `B=4, α_v=−3, α_d=2.25`.
  Track param count vs a classical-embedding control (this *is* C1).
- **2b quantizer (§3.3):** numerical vocab `Q`, finer spacing for small values,
  assign by `argmin_i |x−Q_i|`; **conditioned on `Ccy`**.
- **2c party_encoder (§3.2):** small encoder over `meta_party` columns, objective
  = masked-attribute reconstruction; emit pooled summary → party store.
- **2d column_assembler:** read `column_schema.json`, route each column to its
  path, concatenate into the encoder's column-embedding sequence.
- **2e tabular_encoder (§3.4):** bidirectional BERT, masked-column
  reconstruction + batch-hard triplet (two perturbed views as positives); 25M;
  3 epochs.

**Exit (C1):** reconstruction accuracy on masked columns reported for partitioned
vs classical embeddings, with param counts. Partitioned should match/beat at
~½ params. Freeze the encoder.

### Phase 3 — Multimodal decoder (§4/§4.1)
- Freeze `f` (from 2e) and a Phi-class LLM.
- Implement adapters `Φ`, task embedding `Ξ_task` (`ψ`), per-layer prompt params
  (`φ`); interleave per eq. 5; train per eq. 6.
- Row sentinels `[R1]…` as LLM-vocab tokens.
- Instruction-tune on templated risk-tagging prompts; **single record/example**.

**Exit:** well-formed label predictions; trainable-param count recorded for C2.

### Phase 4 — Evaluation
- Baselines: CatBoost on raw features; optional full fine-tune (frozen encoder,
  unfrozen LLM) for the parameter-efficiency comparison.
- Metrics: PR-AUC, recall@fixed-FPR, F1@threshold. Accuracy reported *alongside*
  only for comparability to the paper's tables.

**Exit (C2):** adapter model vs CatBoost vs full-tune table; verdict against the
Phase 0 thresholds.

---

## 3. Suggested Claude Code session prompts

Drive one phase per session. Example openers:

- **Phase 2a:**
  > "Implement `encoders/partitioning_embedder.py` per architecture.md §3
  > Layer 2 and paper §3.1. Fixed hyperparameters B=4, α_v=−3, α_d=2.25. Provide
  > a `ClassicalEmbedder` control with the same interface so we can compare
  > param counts. Read vocab sizes from `column_schema.json`. Unit-test that
  > partitioned param count < classical for the realized vocab."

- **Phase 2e:**
  > "Implement `encoder/tabular_encoder.py`: bidirectional BERT over the
  > assembled column embeddings, masked-column reconstruction loss PLUS
  > batch-hard triplet loss with two perturbed views as positives (paper §3.4).
  > 25M params, 3 epochs. Do not substitute a generic contrastive loss."

- **Phase 3:**
  > "Implement `decoder/multimodal_decoder.py` per paper §4/§4.1. Freeze the
  > tabular encoder AND the Phi LLM. Only Φ (adapters), ψ (task embedding), φ
  > (per-layer prompt params) train. Interleave per eq. 5, train per eq. 6, row
  > sentinels [R1]… single record per example. Assert frozen params have
  > requires_grad=False."

Always paste the §0 guardrails into the session. Claude Code should leave
`# PAPER: §x.y` comments mapping code to paper sections.

---

## 4. Per-phase acceptance checklist

- [x] Phase 0 — C1/C2 thresholds committed to config
- [x] Phase 1 — generator + schema verified
- [~] Phase 2 — Layers 2+3 BUILT (2a–2e) and unit-tested; encoder `freeze()` ready.
      C1 param half CONFIRMED (embedding-table ratio ~0.057 << 0.55). C1 accuracy
      half (≤1pp recon give-up) PENDING the full 25M/3-epoch run — deferred to GPU
      (CPU est. ~12h for `--compare`). Run command in §6 below.
- [ ] Phase 3 — well-formed predictions; trainable-param count recorded; freeze asserted
- [ ] Phase 4 — baseline table; C2 verdict; imbalance-aware metrics

---

## 5. Definition of done (prototype)

Both claims resolved with numbers: C1 (partitioning parameter efficiency) and
C2 (adapter beats CatBoost, rivals full-tune at a fraction of trainable params),
on imbalance-aware metrics, with every component traceable to a paper section or
one of the three sanctioned departures. Anything beyond that is v2.

---

## 6. Running the C1 comparison (GPU)

The encoder + composite loss + partitioned-vs-classical harness are built and
unit-tested. To produce the C1 ACCURACY half (the headline pinned-config number),
run on a GPU box (CPU is ~12h):

```
# full pinned config: hidden=512, layers=8, heads=8, epochs=3, all 200K rows
python -m encoder.tabular_encoder --compare
```

Prints, for partitioned and classical high-card embedders: per-column top-1/top-3
masked reconstruction accuracy, the high-card param ratio (vs ≤0.55) and the mean
top-1 recon gap (vs ≤1.0pp), with a PASS/FAIL C1 verdict. Defaults read the
pinned shape from `EncoderConfig`; `--limit` caps rows (logged, not silent) for a
labelled proxy. The full run is deliberately deferred to GPU — do NOT shrink the
25M/3-epoch config to force a CPU pass (that voids the headline result, §0.2).
