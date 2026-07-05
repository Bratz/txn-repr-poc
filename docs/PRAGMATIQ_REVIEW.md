# pragmatiq vs txn-repr-poc — implementation review

Reviewed 2026-07-05 against `github.com/dynamiq-ai/pragmatiq` @ `main`
(files read: `data/tokenizer.py`, `data/schema.py`, `data/collate.py`, `models/pragmatiq.py`,
`models/heads.py`, `models/lora.py`, `training/masking.py`, `training/pretrainer.py`,
`training/probe.py`). pragmatiq is an open-source implementation of PRAGMA (arXiv:2604.08649);
this repo implements Raman et al. (arXiv:2410.07851) plus our lifecycle/serving extensions.
The two codebases answer different papers, so most differences are *paradigm*, not quality.

## Side-by-side

| Concern | pragmatiq | this repo | Verdict |
|---|---|---|---|
| Tokenization | Key–value–time: one key token per field, percentile-binned numerics (+zero bucket), categorical-vs-BPE by cardinality threshold, one shared key/value vocab (~28k), within-field positions | Columns-as-tokens over a fixed schema; §3.1 partitioned embedder for high-card ids; currency-conditioned quantizer for amounts; no free-text path | **Paradigm.** Theirs generalizes to variable-length events and text; ours is faithful to Raman's fixed-schema row and carries the C1 measurement. Adopt theirs only if free-text fields ever enter our data. |
| Time | `8·ln(1+Δt/8)` log-seconds as **RoPE positions** in profile/history encoders; calendar (hour/dow/dom) added to event vectors; lifelong items timed from first occurrence | `log1p(days)` through an MLP **added** to event embeddings + dow/dom/month embeddings (`encoders/time_encoding.py`) | **Already-equivalent family**, different mechanism. RoPE-on-time is the more principled positional treatment; ours earned C3/C4 as-is. Revisit only if C7 fails and the history encoder stays long-term. |
| Architecture | Three encoders (profile 1 / event 5 / history 2 layers at small), [USR]/[EVT] markers, block-diagonal varlen attention over packed flat buffers, no padding anywhere | Row encoder (25M) + history encoder; padded batches; [USR] via party-store injection | **Adopt-later (infra).** Their `assemble_segments` + varlen packing is the 2–5× throughput trick from the paper; ours pads. Irrelevant at 20k rows, decisive at 24B events. |
| Pretraining objective | One end-to-end MLM: token/event/key masking unioned (0.15/0.10/0.10) + 10% UNK-dropout excluded from loss; 3d head `[ẑ_e, z_h(evt), z_h(USR)] → Linear(3d→d) →` **tied** logits, label smoothing 0.1 | Two-stage: row encoder (masked-column recon + batch-hard triplet), frozen; history encoder (masked-event recon + optional CoLES triplet), frozen | **Paradigm.** Their gradients cross all levels; ours never cross the freeze line (Raman's invariant). Their key-masking mode ("hide every value of this key for this user") is a genuinely good idea with no analogue here — it forces cross-field inference. Cheap to add to our history-encoder pretraining if C7 keeps it. |
| Text values | Optional frozen Nemotron: whole text value → one frozen embedding + trainable projection; masked text reconstructed by MSE instead of CE | No text path (synthetic names are identities via party encoder) | **Reject for now** — nothing to embed until real remittance text exists. Their split of CE-for-vocab / MSE-for-frozen-embeddings is the right pattern to copy when it does. |
| Downstream adaptation | LoRA fine-tuning of the backbone (`models/lora.py`) + probes | Frozen forever + probes/calibrators + instruction decoder | **Paradigm**, and the biggest capability we lack. PRAGMA's Emb→LoRA gains (+10–70%) say end-to-end adaptability matters; Raman's freeze forbids it. A declared v3 exploration, not a patch. |
| Probes | `HistGradientBoostingClassifier` default (logistic/lightgbm selectable); same classifier for the raw-count baseline so the gap isolates the representation; no-hindcasting cutoffs per user | Calibrated LogisticRegression probes; CatBoost as adversary baseline; held-out-actor splits | **Already-equivalent discipline**, opposite defaults. Their same-classifier-for-baseline rule is cleaner than our LR-vs-CatBoost asymmetry — worth adopting in future claim runs (measure model-vs-naive under the SAME head). |
| Explanations | Integrated gradients (Streamlit demo) | Column occlusion against the frozen encoder | **Already-equivalent intent.** Occlusion is faithful-by-construction and simpler; IG is cheaper per query. Keep ours. |
| Fail-loud hygiene | Tokenizer content-hash checked at `from_pretrained` (refuses mismatched tokenizer); checkpoint format versioned; unseen values → UNK never KeyError | Bundle guards (n_features_in_, 409s, sklearn shim), seeded determinism, crash-insurance dumps | **Both houses live this.** Their tokenizer-hash-in-checkpoint is the standout: our bundles carry no vocab hash and a mismatched encoder/probes pair would fail late. **Adopt.** |
| Serving | Triton + ONNX export + `embed_records` plain-dict path | FastAPI + calibrated heads + instruction decoder | **Paradigm.** Ours serves decisions (calibrated lifecycle objects); theirs serves embeddings. Complementary, not competing. |
| Synthetic data | Generator with causal labels (credit/churn/AML arms), GraphSAGE AML ablation | Generators with causal labels + ISO 20022 lifecycle expansion, event-anchored timestamps, recall/return legs, origination context | **Ours is richer where it matters to us** — nothing in pragmatiq models a payment lifecycle. Their AML graph arm honestly loses to hand-crafted features, matching PRAGMA's −47% and our routing decision. |

## What we adopt (concrete, small)

1. **Vocab/content hash in the bundle** (their `from_pretrained` refusal pattern): store a hash of
   the fitted vocabs in `meta.json` at save; `load_india_model` refuses probes trained against a
   different encoder vocabulary. Cheap, closes a real late-failure hole.
2. **Same-classifier baselines** in future claim runs: when a claim says "beats naive," fit the
   naive's features under the same head class as the model's.
3. **Key-masking** as a third masking mode in history-encoder pretraining — only if C7 fails and
   the history encoder remains a permanent resident.

## What we deliberately do not adopt

- **End-to-end MLM + LoRA** — breaks Raman's frozen-`f` invariant, which is this repo's
  deliverable. That lane exists; it's called using pragmatiq. A PRAGMA-style twin backbone at
  production scale should start there, porting our lifecycle machinery (ISO message model,
  event-anchored timestamps, visibility filtering, capability routing) on top.
- **Varlen packing / dynamic batching** — engineering for a data scale (billions of events) three
  orders of magnitude beyond this POC's corpora.
- **Key–value tokenizer** — until variable-schema or free-text fields exist in our data, the
  fixed-schema column tokens are simpler and carry our measured claims.

## Bottom line

pragmatiq is a disciplined PRAGMA build whose engineering hygiene (tokenizer hashes, fail-loud
loads, same-classifier baselines, deterministic masking) matches this repo's values, and whose
architecture answers a different paper than ours does. The twin merge (docs/PAYMENT_TWIN_MERGE.md)
takes PRAGMA's *shape* from our own convergent implementation, not from porting pragmatiq; the
three small adoptions above are the full extent of what crossing the paradigm line is worth today.
