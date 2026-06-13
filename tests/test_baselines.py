"""Tests for the CatBoost baseline + its plumbing into the metrics."""

import numpy as np
import pytest

pytest.importorskip("catboost")

from eval.baselines import feature_spec, train_catboost
from eval.metrics import evaluate


def test_feature_spec_reads_schema(schema):
    features, categorical = feature_spec(schema)
    num = schema["buckets"]["numerical"][0]
    # numerical column present and NOT categorical
    assert num in features and num not in categorical
    # every categorical feature is a real schema column, none from numerical
    assert num not in categorical
    assert set(categorical) <= set(
        schema["buckets"]["high_card_categorical"]
        + schema["buckets"]["core"]
        + schema["buckets"]["meta_party"]
    )


def test_catboost_trains_and_proba_aligned(schema, sample_df):
    y_true, proba, label_values, model = train_catboost(
        sample_df, schema, test_size=0.3, iterations=40, log=lambda *_: None
    )
    n = proba.shape[0]
    assert proba.shape == (n, len(label_values))
    # rows are valid probability distributions over the schema's label order
    np.testing.assert_allclose(proba.sum(axis=1), np.ones(n), rtol=1e-5)
    assert label_values == schema["label_values"]


def test_catboost_feeds_metrics(schema, sample_df):
    y_true, proba, label_values, _ = train_catboost(
        sample_df, schema, test_size=0.3, iterations=40, log=lambda *_: None
    )
    m = evaluate(y_true, proba, label_values, positive_class="High", fixed_fpr=0.1)
    assert 0.0 <= m["pr_auc"] <= 1.0
    assert 0.0 <= m["accuracy_multiclass"] <= 1.0
