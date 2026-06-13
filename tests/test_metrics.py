"""Unit tests for the imbalance-aware metrics + C2 table."""

import numpy as np
import pytest

from eval.metrics import c2_table, evaluate, f1_at_threshold, pr_auc, recall_at_fixed_fpr

LABELS = ["Low", "Medium", "High"]


def _proba_from_pos_scores(scores, pos_idx=2, n_classes=3):
    """Build an (N, 3) proba with `scores` in the positive column, rest split."""
    p = np.zeros((len(scores), n_classes))
    p[:, pos_idx] = scores
    rest = (1 - scores) / (n_classes - 1)
    for j in range(n_classes):
        if j != pos_idx:
            p[:, j] = rest
    return p


def test_perfect_separation():
    y = np.array(["High"] * 20 + ["Low"] * 180)
    scores = np.concatenate([np.full(20, 0.99), np.full(180, 0.01)])
    assert pr_auc((y == "High").astype(int), scores) == pytest.approx(1.0)
    recall, thr = recall_at_fixed_fpr((y == "High").astype(int), scores, 0.01)
    assert recall == pytest.approx(1.0)
    assert np.isfinite(thr)


def test_pr_auc_random_near_prevalence():
    rng = np.random.default_rng(0)
    y = (rng.random(5000) < 0.02).astype(int)     # ~2% positive
    scores = rng.random(5000)                      # uninformative
    assert pr_auc(y, scores) < 0.10                # near prevalence, far below 1


def test_recall_at_fpr_in_range_and_threshold():
    rng = np.random.default_rng(1)
    y = (rng.random(2000) < 0.1).astype(int)
    scores = y * 0.5 + rng.random(2000) * 0.6      # somewhat informative
    recall, thr = recall_at_fixed_fpr(y, scores, 0.05)
    assert 0.0 <= recall <= 1.0


def test_f1_at_threshold_bounds():
    y = np.array([0, 0, 1, 1])
    scores = np.array([0.1, 0.4, 0.6, 0.9])
    assert f1_at_threshold(y, scores, 0.5) == pytest.approx(1.0)
    assert 0.0 <= f1_at_threshold(y, scores, 0.95) <= 1.0


def test_evaluate_bundle_keys_and_accuracy_alongside():
    y = np.array(["High", "Low", "Medium", "High", "Low"])
    proba = _proba_from_pos_scores(np.array([0.9, 0.1, 0.2, 0.8, 0.05]))
    m = evaluate(y, proba, LABELS, positive_class="High", fixed_fpr=0.5)
    assert m["positive_class"] == "High"
    assert 0 <= m["pr_auc"] <= 1
    assert "accuracy_multiclass" in m            # reported alongside
    assert m["prevalence"] == pytest.approx(2 / 5)


def test_evaluate_rejects_unknown_positive_class():
    y = np.array(["Low", "High"])
    proba = _proba_from_pos_scores(np.array([0.2, 0.8]))
    with pytest.raises(ValueError):
        evaluate(y, proba, LABELS, positive_class="Critical")


# --------------------------------------------------------------------------- #
# C2 table
# --------------------------------------------------------------------------- #

def _synthetic_results():
    rng = np.random.default_rng(3)
    y = np.where(rng.random(2000) < 0.05, "High", "Low")
    pos = (y == "High").astype(float)
    cat = _proba_from_pos_scores(np.clip(pos * 0.5 + rng.random(2000) * 0.4, 0, 1))
    adp = _proba_from_pos_scores(np.clip(pos * 0.8 + rng.random(2000) * 0.2, 0, 1))
    ful = _proba_from_pos_scores(np.clip(pos * 0.85 + rng.random(2000) * 0.18, 0, 1))
    return y, cat, adp, ful


def test_c2_table_full_verdict():
    y, cat, adp, ful = _synthetic_results()
    tbl = c2_table(
        {"catboost": (y, cat), "adapter": (y, adp), "full_tune": (y, ful)},
        LABELS, "High", 0.01,
        trainable_params={"adapter": 8_400_000, "full_tune": 1_300_000_000},
    )
    v = tbl["verdict"]
    assert "pr_auc_gain_vs_catboost_pp" in v
    assert "pr_auc_gap_vs_fulltune_pp" in v
    assert v["trainable_param_ratio"] == pytest.approx(8_400_000 / 1_300_000_000)
    assert v["param_efficient"] is True          # 0.0065 <= 0.10
    assert set(tbl["per_model"]) == {"catboost", "adapter", "full_tune"}


def test_c2_table_catboost_only_no_crash():
    y, cat, _, _ = _synthetic_results()
    tbl = c2_table({"catboost": (y, cat)}, LABELS, "High", 0.01)
    assert "catboost" in tbl["per_model"]
    # no comparators → no adapter/full-tune verdicts
    assert "pr_auc_gain_vs_catboost_pp" not in tbl["verdict"]
