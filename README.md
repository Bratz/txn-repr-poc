# Transaction Representation Learning — Fraud/Risk POC

A grounded, faithful prototype of:

> Raman, Ganesh, Veloso (JPMorgan AI Research), *Scalable Representation
> Learning for Multimodal Tabular Transactions*, arXiv:2410.07851, NeurIPS 2024.

Applied to ISO 20022 (pacs.008) payments for fraud/risk tagging, on **synthetic
data only**. v1 replicates the paper — no extensions.

## Read these first (in order)
1. `architecture.md` — what each component is (source of truth).
2. `CLAUDE_CODE_HANDOFF.md` — how to build it, phase by phase, with guardrails.
3. `configs/default.yaml` — pinned hyperparameters + the Phase 0 thresholds to fill in.

## What's built
- `data/synth_pacs008.py` — Algorithm 1 generator + Layer 1 pacs.008 projection (Phase 1, ✅).
- `data/column_schema.example.json` — example bucket manifest the encoder reads.
- `data/pacs008_sample_500.csv` — 500-row sample for eyeballing.

## What's next
Phases 2–4 (encoder, decoder, eval). Folders `encoders/`, `encoder/`,
`decoder/`, `eval/` are empty build targets. Per-phase plans, acceptance
criteria, and ready-to-paste session prompts are in `CLAUDE_CODE_HANDOFF.md`.

## Setup
```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt          # Phase 1 deps only; uncomment more per phase

# regenerate data + schema (POC scale)
python data/synth_pacs008.py --parents 4000 --transactions 200000 \
  --out data/pacs008_synth.parquet --schema-out data/column_schema.json

# scale toward the paper's ~125K account vocab when ready
# python data/synth_pacs008.py --parents 20000 --transactions 1000000 ...
```

## First Claude Code session
1. Open this repo in Claude Code.
2. Paste the §0 guardrails from `CLAUDE_CODE_HANDOFF.md` into the session.
3. Do Phase 0: fill the 7 `TODO` thresholds in `configs/default.yaml`.
4. Then Phase 2a: use the partitioning-embedder prompt in the handoff §3.

## The one rule
Fidelity to the paper is the deliverable. Don't extend, don't retune pinned
hyperparameters, keep `f` and the LLM frozen in Layer 4. Only three departures
are sanctioned (pacs.008 schema, currency-conditioned quantization,
imbalance-aware metrics). A fourth must be raised, not slipped in.
