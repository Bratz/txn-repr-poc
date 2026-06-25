# Trained model

The full-run checkpoint (`model.pt`, ~979 MB) is too large for git, so the
weights live on the **Hugging Face Hub**; this repo carries only the metadata
(`model_meta.json`) and the loader (`predict.py`).

- **Config** (`model_meta.json`): frozen LLM `microsoft/phi-1_5`, φ=`prompt`,
  encoder hidden `512`, labels `[Low, Medium, High]`.
- **What `model.pt` contains** (see `predict.py:save_model`): the frozen tabular
  encoder `f`, the trainable adapter trio `{Φ, ψ, φ}` + `[R1…RM]`, the column
  vocabs, the quantizer grids, and — for a multi-task checkpoint — every §5 task's
  instruction/answer tokens. The LLM (Phi) is NOT in the file — it is re-downloaded
  by name at load.
- **Run that produced it:** 1M synthetic rows / ~120K account vocab; results in
  `RESULTS.md` (C1 confirmed; C2 param-efficient but below CatBoost on this data).

## Weights location

Hugging Face (**private**): `https://huggingface.co/Subratob/txn-repr-poc-model`
Downloading requires a Hugging Face token with read access to the repo.

## Use it

```bash
pip install huggingface_hub
hf auth login --token <HF_TOKEN>          # private repo → authenticate first
hf download Subratob/txn-repr-poc-model --local-dir ckpt   # gets model.pt + meta.json

# score any task by name (default: risk)
python predict.py --model-dir ckpt --input new_rows.parquet --out scored.csv --task risk
python predict.py --model-dir ckpt --input new_rows.parquet --out geo.csv    --task geography
python predict.py --model-dir ckpt --input new_rows.parquet --out rec.csv    --task recurrence
```
`new_rows.parquet` must hold the projected `column_schema.json` columns. Output
adds `p_<class>` columns + a `<task>_pred` label. Single-record tasks (risk /
geography / expense) emit **one row per input row**; the multi-record
**recurrence** task groups by `group_id` and emits **one row per
(debtor,creditor) group** with ≥`R` transactions. Phi-1.5 downloads on first load.

> The currently published checkpoint is the **risk-only** run, so it exposes just
> the `risk` task (`--task` defaults to it). A checkpoint from the four-task
> `run_gpu.py` run exposes all four; `meta.json` lists which tasks are available.
