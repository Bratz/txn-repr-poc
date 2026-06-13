"""Unit tests for the §3.3 adaptive, currency-conditioned quantizer."""

import numpy as np
import pytest
import torch

from encoders.quantizer import (
    AdaptiveQuantizer,
    geometric_levels,
    make_quantizer_embedder,
)


# --------------------------------------------------------------------------- #
# Geometric grid: finer spacing for small, coarser for large (§3.3)
# --------------------------------------------------------------------------- #

def test_geometric_gaps_increase_with_magnitude():
    g = geometric_levels(1.0, 1_000_000.0, 128)
    gaps = np.diff(g)
    assert np.all(gaps > 0)                      # strictly increasing levels
    assert np.all(np.diff(gaps) > 0)             # gaps themselves grow → finer at small
    assert g[0] == pytest.approx(1.0)
    assert g[-1] == pytest.approx(1_000_000.0)


def test_geometric_degenerate_range_is_constant():
    g = geometric_levels(500.0, 500.0, 64)
    assert g.shape == (64,)
    assert np.all(g == 500.0)


def test_geometric_requires_at_least_two_levels():
    with pytest.raises(ValueError):
        geometric_levels(1.0, 10.0, 1)


# --------------------------------------------------------------------------- #
# Assignment rule: argmin_i |x - Q_i|  (§3.3)
# --------------------------------------------------------------------------- #

def test_assignment_is_nearest_level():
    amts = np.array([10.0, 100.0, 1000.0, 9_999_999.0, 0.0001])
    q = AdaptiveQuantizer(num_levels=64, condition_on_currency=False).fit(
        np.geomspace(1.0, 1e6, 5000)
    )
    idx = q.transform(amts)
    grid = q.grids_["__GLOBAL__"]
    brute = np.array([np.argmin(np.abs(grid - x)) for x in amts])
    assert np.array_equal(idx, brute)


def test_levels_are_monotonic_in_value():
    q = AdaptiveQuantizer(num_levels=128, condition_on_currency=False).fit(
        np.geomspace(1.0, 1e6, 10000)
    )
    xs = np.sort(np.geomspace(1.0, 1e6, 500))
    idx = q.transform(xs)
    assert np.all(np.diff(idx) >= 0)             # larger value → ≥ level


def test_indices_within_vocab_and_clamp_out_of_range():
    m = 32
    q = AdaptiveQuantizer(num_levels=m, condition_on_currency=False).fit(
        np.linspace(100, 10_000, 2000)
    )
    idx = q.transform(np.array([-5.0, 1e12]))    # below min / above max
    assert idx.min() >= 0 and idx.max() < m
    assert idx[0] == 0 and idx[1] == m - 1       # clamp to endpoints


# --------------------------------------------------------------------------- #
# Currency conditioning (the forced departure)
# --------------------------------------------------------------------------- #

def test_per_currency_grids_differ_in_scale():
    rng = np.random.default_rng(0)
    usd = rng.uniform(10, 5_000, 2000)           # small-scale currency
    jpy = rng.uniform(1_000, 5_000_000, 2000)    # large-scale currency
    amounts = np.concatenate([usd, jpy])
    ccy = np.array(["USD"] * usd.size + ["JPY"] * jpy.size)
    q = AdaptiveQuantizer(num_levels=128).fit(amounts, ccy)
    assert q.grids_["JPY"][-1] > q.grids_["USD"][-1] * 100


def test_same_value_quantizes_differently_per_currency():
    # 5,000 is "large" for USD but "small" for JPY → different level bands.
    rng = np.random.default_rng(1)
    amounts = np.concatenate([rng.uniform(10, 5_000, 2000),
                              rng.uniform(1_000, 5_000_000, 2000)])
    ccy = np.array(["USD"] * 2000 + ["JPY"] * 2000)
    q = AdaptiveQuantizer(num_levels=128).fit(amounts, ccy)
    lvl = q.transform(np.array([5_000.0, 5_000.0]), np.array(["USD", "JPY"]))
    assert lvl[0] > lvl[1]                        # high band in USD, low band in JPY


def test_relative_magnitude_aligns_across_currencies():
    # The top-of-range amount in each currency should map to a high level in BOTH
    # — that is the point of conditioning (shared relative-magnitude semantics).
    rng = np.random.default_rng(2)
    usd = rng.uniform(10, 5_000, 4000)
    jpy = rng.uniform(1_000, 5_000_000, 4000)
    amounts = np.concatenate([usd, jpy])
    ccy = np.array(["USD"] * usd.size + ["JPY"] * jpy.size)
    q = AdaptiveQuantizer(num_levels=128).fit(amounts, ccy)
    lvl = q.transform(np.array([usd.max(), jpy.max()]), np.array(["USD", "JPY"]))
    assert lvl[0] > 100 and lvl[1] > 100         # both near the top of the m=128 vocab


def test_unseen_currency_falls_back_to_global():
    q = AdaptiveQuantizer(num_levels=64).fit(
        np.geomspace(10, 1e5, 1000), np.array(["USD"] * 1000)
    )
    # EUR never seen at fit time → uses global grid, must not raise.
    out = q.transform(np.array([1234.0]), np.array(["EUR"]))
    assert 0 <= out[0] < 64


def test_condition_flag_requires_currencies():
    q = AdaptiveQuantizer(condition_on_currency=True)
    with pytest.raises(ValueError):
        q.fit(np.array([1.0, 2.0, 3.0]))         # no currencies given


# --------------------------------------------------------------------------- #
# Persistence + embedding wrapper
# --------------------------------------------------------------------------- #

def test_roundtrip_dict_preserves_transform():
    rng = np.random.default_rng(3)
    amounts = rng.uniform(10, 1e6, 3000)
    ccy = rng.choice(["USD", "JPY", "GBP"], 3000)
    q = AdaptiveQuantizer(num_levels=96).fit(amounts, ccy)
    q2 = AdaptiveQuantizer.from_dict(q.to_dict())
    a = q.transform(amounts, ccy)
    b = q2.transform(amounts, ccy)
    assert np.array_equal(a, b)


def test_embedder_shape_and_param_count():
    q = AdaptiveQuantizer(num_levels=128).fit(
        np.geomspace(10, 1e6, 2000), np.array(["USD"] * 2000)
    )
    emb = make_quantizer_embedder(q, embedding_dim=64)
    out = emb(np.array([100.0, 50_000.0, 1e6]), np.array(["USD", "USD", "USD"]))
    assert out.shape == (3, 64)
    assert out.dtype == torch.float32
    assert emb.num_embedding_parameters() == 128 * 64


def test_embedder_gradients_flow():
    q = AdaptiveQuantizer(num_levels=64).fit(
        np.geomspace(10, 1e6, 1000), np.array(["USD"] * 1000)
    )
    emb = make_quantizer_embedder(q, embedding_dim=32)
    emb(np.array([100.0, 9_000.0]), np.array(["USD", "USD"])).sum().backward()
    assert emb.emb.weight.grad is not None
    assert torch.any(emb.emb.weight.grad != 0)
