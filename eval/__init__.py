"""Layer 5 evaluation — imbalance-aware metrics (forced departure) + baselines."""

from .metrics import (
    evaluate,
    pr_auc,
    recall_at_fixed_fpr,
    f1_at_threshold,
    c2_table,
)
from .baselines import train_catboost, feature_spec

__all__ = [
    "evaluate",
    "pr_auc",
    "recall_at_fixed_fpr",
    "f1_at_threshold",
    "c2_table",
    "train_catboost",
    "feature_spec",
]
