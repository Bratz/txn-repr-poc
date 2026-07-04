# vast.ai runbook — full (non-smoke) runs on GPU

Everything below was CPU-validated at smoke/limited scale; the GPU box exists to produce the
full-scale numbers fast. Any instance with **>= 12 GB VRAM** is comfortable (fp32 Phi-1.5 is
~5.7 GB; the 25M encoder is trivial). A PyTorch/CUDA template image is fine.

## 1. Setup (~3 min)

```bash
git clone https://github.com/Bratz/txn-repr-poc.git
cd txn-repr-poc && git checkout feat/india-rails
pip install -r requirements.txt          # transformers pinned <5 (5.x segfaults Phi forward)
python -m pytest tests/ -q               # ~1 min; expect all green before burning GPU time
```

All runners auto-select `cuda` when available — no flags needed.

## 2. Generate the datasets (gitignored — must be built on the box, ~2 min)

```bash
python data/synth_sequences.py --out data/pacs008_seq.parquet \
    --schema-out data/column_schema_seq.json                    # §7 behavioural corpus (run_seq)
python data/synth_india_rails.py --accounts 4000 --payments 20000 \
    --out-prefix data/india_rails --schema-out data/column_schema_india.json
```

Paths matter: `run_seq.py` reads `data/pacs008_seq.parquet`; without `--out` the generator
writes to the current directory, and a full run now fails loudly instead of silently using
the 500-row reference sample.

Sanity check before burning GPU time: the first line every runner prints is `device=cuda`.
`device=cpu` means the shell's torch has no CUDA (usually the venv wasn't activated -
`. /venv/main/bin/activate` on vast.ai base images) - fix that before proceeding.

## 3. The runs, in value order

```bash
# A. C5 with real Phi-1.5 + full-scale C3/C4/velocity (the one the CPU box kept OOM-killing).
#    Full corpus ~170k rows; on GPU expect well under an hour. Crash insurance still applies:
#    claim inputs land in data/_c5_reps.npz before the Phi load.
python -u run_seq.py --out results_seq_full.json

# B. Next-payment / next-receipt forecasting at full config (cadence data built in-run)
python -u run_next.py --out results_next.json
python -u run_next.py --actor CdtrAcct_Id --out results_next_receipts.json

# C. Predict-at-initiation A-I + origination fusion + streaming, full config on the 20k
python -u run_impute.py --out results_impute_full.json

# D. Message-lifecycle sequence (gpi tracker outcome) at full config
python -u run_msgseq.py --out results_msgseq_full.json

# E. Refresh the servable bundle (intake + inflight + velocity + next heads, calibrated)
python -u run_india.py --save model_india --out results_india.json

# F. Paper-exact serving tier (C6): instruction-tune the frozen-Phi decoder over the
#    whole classification menu and ship it INSTEAD of the classification probes.
#    Compare per-task answer quality against run E's probes on the same split.
python -u run_india.py --paper-serving --save model_india_paper --out results_india_paper.json
```

## 4. Copy back

```
results_*.json            all measured claims
model_india/              the servable bundle (encoder.pt ~180MB + hist.pt + probes.joblib)
data/_c5_reps.npz         claim-input dump (only if run A was interrupted)
```

## Known footguns (all already handled, listed so nobody re-trips them)

* transformers **must stay <5** — 5.x hard-crashes the Phi-1.5 forward (exit 139, no traceback).
* `run_seq.py` dumps `_c5_reps.npz` and frees memory *before* loading Phi; if the box is
  memory-tight anyway, `scratchpad c5_finish`-style resumption from the dump avoids repeating
  the pipeline (see run_seq docstring note).
* Datasets are seeded/deterministic: same commands -> byte-identical corpora, so numbers are
  reproducible across boxes.
* Do NOT commit generated parquet/csv/model artifacts — .gitignore already covers them.
