# Trained model

The full-run checkpoint (`model.pt`, ~979 MB) is too large for git, so the
weights live on the **Hugging Face Hub**; this repo carries only the metadata
(`model_meta.json`) and the loader (`predict.py`).

- **Config** (`model_meta.json`): frozen LLM `microsoft/phi-1_5`, φ=`prompt`,
  encoder hidden `512`, labels `[Low, Medium, High]`.
- **What `model.pt` contains** (see `predict.py:save_model`): the frozen tabular
  encoder `f`, the trainable adapter trio `{Φ, ψ, φ}` + `[R1]`, the column vocabs,
  the quantizer grids, and the resolved instruction/answer tokens. The LLM (Phi)
  is NOT in the file — it is re-downloaded by name at load.
- **Run that produced it:** 1M synthetic rows / ~120K account vocab; results in
  `RESULTS.md` (C1 confirmed; C2 param-efficient but below CatBoost on this data).

## Weights location

Hugging Face: `https://huggingface.co/<your-hf-username>/txn-repr-poc-model`
(update this line with your actual repo id after upload).

## Use it

```bash
pip install huggingface_hub
hf download <your-hf-username>/txn-repr-poc-model --local-dir ckpt   # gets model.pt + meta.json
python predict.py --model-dir ckpt --input new_rows.parquet --out scored.csv
```
`new_rows.parquet` must hold the projected `column_schema.json` columns. Output
adds `p_Low/p_Medium/p_High` + a `risk_pred` label per row. Phi-1.5 downloads on
first load.
