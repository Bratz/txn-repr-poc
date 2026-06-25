"""
Layer 5 evaluation metrics — imbalance-aware (forced departure #3).

Grounded strictly to:
  Raman, Ganesh, Veloso (JPMorgan AI Research),
  "Scalable Representation Learning for Multimodal Tabular Transactions",
  arXiv:2410.07851, NeurIPS 2024 TRL workshop.

# FORCED DEPARTURE (handoff §0.5, departures.imbalance_aware_metrics): the risk
# label is imbalanced (High ~= 2%), so ACCURACY is misleading by construction.
# Primary metrics are PR-AUC, recall@fixed-FPR, and F1@threshold; accuracy is
# reported ALONGSIDE only for comparability to the paper's balanced-task tables.
#
# recall@FPR and F1 are inherently binary, so the multiclass risk label is cast
# as a binary detection problem: positive = the rare, operationally-important
# class (default "High"). This is a documented framing choice; the positive class
# is configurable.
"""

from __future__ import annotations

import numpy as np


def _as_arrays(y_true, proba):
    return np.asarray(y_true), np.asarray(proba, dtype=np.float64)


def pr_auc(y_binary: np.ndarray, scores: np.ndarray) -> float:
    """Average precision (area under precision-recall) for the positive class."""
    from sklearn.metrics import average_precision_score
    return float(average_precision_score(y_binary, scores))


def recall_at_fixed_fpr(y_binary: np.ndarray, scores: np.ndarray,
                        fpr: float = 0.01) -> tuple[float, float]:
    """Recall (TPR) at the operating point whose FPR ≤ `fpr`, and its threshold.

    Picks the highest-recall threshold that still keeps FPR at or below the
    target false-positive budget.
    """
    from sklearn.metrics import roc_curve
    fprs, tprs, thresholds = roc_curve(y_binary, scores)
    ok = fprs <= fpr
    if not ok.any():
        return 0.0, float("inf")
    i = np.argmax(tprs * ok)            # best TPR among FPR-feasible points
    return float(tprs[i]), float(thresholds[i])


def f1_at_threshold(y_binary: np.ndarray, scores: np.ndarray,
                    threshold: float) -> float:
    from sklearn.metrics import f1_score
    return float(f1_score(y_binary, (scores >= threshold).astype(int), zero_division=0))


def evaluate(y_true, proba, label_values, positive_class: str = "High",
             fixed_fpr: float = 0.01) -> dict:
    """Imbalance-aware metric bundle from multiclass probabilities.

    Args:
        y_true: array of string labels (length N).
        proba: (N, n_classes) class probabilities, columns aligned to label_values.
        label_values: ordered class names matching proba columns.
        positive_class: rare class treated as the binary positive.
        fixed_fpr: operating FPR for recall@FPR.
    """
    y_true, proba = _as_arrays(y_true, proba)
    if positive_class not in label_values:
        raise ValueError(f"positive_class {positive_class!r} not in {label_values}")
    pos = label_values.index(positive_class)

    y_binary = (y_true == positive_class).astype(int)
    scores = proba[:, pos]

    recall, thr = recall_at_fixed_fpr(y_binary, scores, fixed_fpr)
    pred_labels = np.asarray(label_values)[proba.argmax(axis=1)]
    return {
        "positive_class": positive_class,
        "prevalence": float(y_binary.mean()),
        "pr_auc": pr_auc(y_binary, scores),
        f"recall_at_fpr_{fixed_fpr}": recall,
        "operating_threshold": thr,
        "f1_at_operating_threshold": f1_at_threshold(y_binary, scores, thr),
        # accuracy reported ALONGSIDE only (multiclass), per the departure note.
        "accuracy_multiclass": float((pred_labels == y_true).mean()),
    }


# --------------------------------------------------------------------------- #
# Balanced multiclass tasks (geography, expense) — accuracy + macro-F1
# --------------------------------------------------------------------------- #
# The paper's geography/expense tagging tasks are roughly class-balanced, so
# accuracy is meaningful here (unlike the imbalanced risk label). Macro-F1 is
# reported alongside so a degenerate majority-class predictor is still exposed.

def evaluate_multiclass(y_true, proba, label_values) -> dict:
    from sklearn.metrics import f1_score
    y_true, proba = _as_arrays(y_true, proba)
    pred = np.asarray(label_values)[proba.argmax(axis=1)]
    return {
        "accuracy": float((pred == y_true).mean()),
        "macro_f1": float(f1_score(y_true, pred, labels=label_values,
                                   average="macro", zero_division=0)),
        "n_classes": len(label_values),
    }


def evaluate_task(task: dict, y_true, proba, fixed_fpr: float = 0.01) -> dict:
    """Route a task to the right metric bundle by its `metric` field.

    imbalance / binary → PR-AUC, recall@FPR, F1 (binary positive class).
    multiclass         → accuracy + macro-F1 (balanced tasks).
    """
    metric = task.get("metric", "imbalance")
    label_values = task["label_values"]
    if metric == "multiclass":
        return {"metric": metric, **evaluate_multiclass(y_true, proba, label_values)}
    positive = task.get("positive_class", label_values[-1])
    return {"metric": metric,
            **evaluate(y_true, proba, label_values, positive, fixed_fpr)}


# --------------------------------------------------------------------------- #
# C2 comparison table
# --------------------------------------------------------------------------- #

def c2_table(results: dict, label_values, positive_class: str = "High",
             fixed_fpr: float = 0.01, trainable_params: dict | None = None,
             thresholds: dict | None = None) -> dict:
    """Assemble the C2 model-comparison table from per-model (y_true, proba).

    `results`: {model_name: (y_true, proba)}. Must include "catboost". May include
    "adapter" and "full_tune" once the GPU run produces them. `trainable_params`:
    {model_name: int}. Returns per-model metrics plus C2 verdicts for whichever
    models are present (gains/gaps are None when a comparator is missing).
    """
    thresholds = thresholds or {}
    per_model = {
        name: evaluate(yt, pr, label_values, positive_class, fixed_fpr)
        for name, (yt, pr) in results.items()
    }
    out = {"per_model": per_model, "verdict": {}}

    base = per_model.get("catboost", {}).get("pr_auc")
    ad = per_model.get("adapter", {}).get("pr_auc")
    ft = per_model.get("full_tune", {}).get("pr_auc")
    v = out["verdict"]

    if ad is not None and base is not None:
        gain = (ad - base) * 100
        thr = thresholds.get("pr_auc_gain_vs_catboost", 10.0)
        v["pr_auc_gain_vs_catboost_pp"] = gain
        v["beats_catboost"] = gain >= thr
    if ad is not None and ft is not None:
        gap = (ft - ad) * 100
        thr = thresholds.get("pr_auc_gap_vs_fulltune", 3.0)
        v["pr_auc_gap_vs_fulltune_pp"] = gap
        v["rivals_fulltune"] = gap <= thr
    if trainable_params and "adapter" in trainable_params and "full_tune" in trainable_params:
        ratio = trainable_params["adapter"] / trainable_params["full_tune"]
        thr = thresholds.get("trainable_param_ratio", 0.10)
        v["trainable_param_ratio"] = ratio
        v["param_efficient"] = ratio <= thr
    return out
