"""
Layer 5 baselines — CatBoost on raw flattened features (handoff Phase 4).

Grounded strictly to:
  Raman, Ganesh, Veloso (JPMorgan AI Research),
  "Scalable Representation Learning for Multimodal Tabular Transactions",
  arXiv:2410.07851, NeurIPS 2024 TRL workshop.

# PAPER: §5 (baselines). CatBoost trains on the raw, flattened projected row —
# no representation learning. It is the C2 floor the adapter model must beat.
# Feature buckets are read from column_schema.json (handoff §0.4); every column
# except the single numerical amount is treated as categorical (CatBoost handles
# high-card categoricals via target statistics). The optional full fine-tune
# baseline (frozen encoder, unfrozen LLM) is the GPU comparator for C2 and is not
# produced here.
"""

from __future__ import annotations

import numpy as np


def feature_spec(schema: dict) -> tuple[list, list]:
    """Return (all_feature_columns, categorical_feature_columns) from the schema."""
    b = schema["buckets"]
    numerical = list(b["numerical"])
    categorical = list(b["high_card_categorical"]) + list(b["core"]) + list(b["meta_party"])
    categorical = [c for c in categorical if c not in numerical]
    features = numerical + categorical
    return features, categorical


def _to_catboost_frame(df, schema):
    """Raw feature frame: all columns categorical except the numerical amount."""
    features, categorical = feature_spec(schema)
    X = df[features].astype(str).copy()
    num_col = schema["buckets"]["numerical"][0]
    X[num_col] = df[num_col].astype(float).to_numpy()
    cat_idx = [X.columns.get_loc(c) for c in categorical]
    return X, cat_idx, features, categorical


def catboost_fit_predict(train_df, eval_df, schema, seed: int = 7,
                         iterations: int = 300, log=print):
    """Fit CatBoost on an explicit train frame, predict on an explicit eval frame.

    Lets the baseline share the SAME split as the encoder/decoder so the C2 table
    is comparable. Returns (y_eval_true, proba_eval, label_values, model).
    """
    from catboost import CatBoostClassifier, Pool

    label_col = schema["label_column"]
    label_values = list(schema["label_values"])
    X_tr, cat_idx, features, categorical = _to_catboost_frame(train_df, schema)
    X_te, _, _, _ = _to_catboost_frame(eval_df, schema)
    y_tr = train_df[label_col].astype(str).to_numpy()
    y_te = eval_df[label_col].astype(str).to_numpy()

    model = CatBoostClassifier(
        iterations=iterations, depth=6, learning_rate=0.1,
        loss_function="MultiClass", auto_class_weights="Balanced",
        random_seed=seed, verbose=False, allow_writing_files=False,
    )
    log(f"CatBoost: fit on {len(X_tr):,} rows, {len(features)} features "
        f"({len(categorical)} categorical), {iterations} iters ...")
    model.fit(Pool(X_tr, y_tr, cat_features=cat_idx))

    proba = np.asarray(model.predict_proba(Pool(X_te, cat_features=cat_idx)))
    classes = [str(c) for c in model.classes_]
    order = [classes.index(v) for v in label_values]
    return y_te, proba[:, order], label_values, model


def train_catboost(df, schema, test_size: float = 0.2, seed: int = 7,
                   iterations: int = 300, log=print):
    """Fit CatBoost with an internal stratified split (standalone baseline CLI)."""
    from sklearn.model_selection import train_test_split

    label_col = schema["label_column"]
    tr_idx, te_idx = train_test_split(
        np.arange(len(df)), test_size=test_size, random_state=seed,
        stratify=df[label_col].astype(str).to_numpy(),
    )
    return catboost_fit_predict(df.iloc[tr_idx], df.iloc[te_idx], schema,
                                seed=seed, iterations=iterations, log=log)


def main():
    import argparse
    import json
    from pathlib import Path

    import pandas as pd

    from eval.metrics import c2_table, evaluate

    ap = argparse.ArgumentParser(description="Layer 5 CatBoost baseline + imbalance metrics")
    root = Path(__file__).resolve().parents[1]
    ap.add_argument("--schema", default=str(root / "data" / "column_schema.json"))
    ap.add_argument("--data", default=str(root / "data" / "pacs008_synth.parquet"))
    ap.add_argument("--iterations", type=int, default=300)
    ap.add_argument("--limit", type=int, default=None, help="cap rows (logged)")
    ap.add_argument("--positive-class", default="High")
    ap.add_argument("--fixed-fpr", type=float, default=0.01)
    args = ap.parse_args()

    schema_path = Path(args.schema)
    if not schema_path.exists():
        schema_path = root / "data" / "column_schema.example.json"
    schema = json.loads(schema_path.read_text())

    path = Path(args.data)
    df = None
    if path.exists():
        try:
            df = pd.read_parquet(path) if path.suffix == ".parquet" else pd.read_csv(path)
            src = path.name
        except Exception as e:
            print(f"(could not read {path.name}: {e}; using reference sample)")
    if df is None:
        df = pd.read_csv(root / "data" / "pacs008_sample_500.csv")
        src = "pacs008_sample_500.csv (fallback)"
    if args.limit and args.limit < len(df):
        print(f"NOTE: capping rows {len(df):,} -> {args.limit:,} (--limit)")
        df = df.head(args.limit)

    print(f"Baseline eval on {src} ({len(df):,} rows), positive={args.positive_class}")
    y_true, proba, label_values, _ = train_catboost(
        df, schema, iterations=args.iterations
    )
    m = evaluate(y_true, proba, label_values, args.positive_class, args.fixed_fpr)
    print("\nCatBoost (imbalance-aware):")
    for k, val in m.items():
        print(f"  {k:<28}: {val:.4f}" if isinstance(val, float) else f"  {k:<28}: {val}")

    # C2 table with only the CatBoost row available on CPU; adapter / full_tune
    # columns fill in from the GPU decoder run.
    tbl = c2_table({"catboost": (y_true, proba)}, label_values,
                   args.positive_class, args.fixed_fpr)
    print("\nC2 table - adapter / full_tune rows pending GPU run.")
    print(f"  catboost PR-AUC: {tbl['per_model']['catboost']['pr_auc']:.4f}")


if __name__ == "__main__":
    main()
