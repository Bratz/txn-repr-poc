"""C1 unit tests for the §3.1 partitioning embedder.

The headline assertion (handoff Phase 2a exit): partitioned embedding-table
param count < classical, for the realized schema vocabs.
"""

import json
from pathlib import Path

import pytest
import torch

from encoders.partitioning_embedder import (
    PAPER_ALPHA_D,
    PAPER_ALPHA_V,
    PAPER_B,
    ClassicalEmbedder,
    PartitioningEmbedder,
    param_efficiency,
    power_law_partition,
)

SCHEMA_PATH = Path(__file__).resolve().parents[1] / "data" / "column_schema.json"


def _realized_vocabs():
    schema = json.loads(SCHEMA_PATH.read_text())
    return schema["vocab"]


# --------------------------------------------------------------------------- #
# power_law_partition (Eq. 2)
# --------------------------------------------------------------------------- #

def test_partition_sums_exactly():
    for total in (16, 64, 128, 3968, 23665):
        for alpha in (PAPER_ALPHA_V, PAPER_ALPHA_D):
            alloc = power_law_partition(total, PAPER_B, alpha)
            assert sum(alloc) == total
            assert len(alloc) == PAPER_B
            assert all(a >= 1 for a in alloc)


def test_vocab_monotonic_increasing():
    # α_v = -3 → weights ∝ b^3 → |V^1| ≪ … ≪ |V^B|.
    alloc = power_law_partition(23665, PAPER_B, PAPER_ALPHA_V)
    assert alloc == sorted(alloc), alloc
    assert alloc[0] < alloc[-1]


def test_dim_monotonic_decreasing():
    # α_d = 2.25 → weights ∝ b^-2.25 → D^1 ≫ … ≫ D^B.
    alloc = power_law_partition(128, PAPER_B, PAPER_ALPHA_D)
    assert alloc == sorted(alloc, reverse=True), alloc
    assert alloc[0] > alloc[-1]


def test_partition_rejects_total_below_B():
    with pytest.raises(ValueError):
        power_law_partition(3, 4, PAPER_ALPHA_V)


# --------------------------------------------------------------------------- #
# C1 headline: partitioned < classical on realized vocabs
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("dim", [64, 128, 256])
def test_partitioned_fewer_params_than_classical(dim):
    vocab = _realized_vocabs()
    targets = ["combined_account_id_vocab", "combined_parent_id_vocab",
               "DbtrAcct_Id", "CdtrAcct_Id", "UltmtDbtr_Id", "UltmtCdtr_Id"]
    for key in targets:
        v = vocab[key]
        part = PartitioningEmbedder(v, dim)
        clf = ClassicalEmbedder(v, dim)
        assert part.num_embedding_parameters() < clf.num_embedding_parameters(), key


def test_param_ratio_meets_c1_threshold():
    # C1 param_ratio threshold (configs/default.yaml) is 0.55 for the headline
    # account vocab; the table ratio is far below that.
    vocab = _realized_vocabs()
    r = param_efficiency(vocab["combined_account_id_vocab"], 128)
    assert r["param_ratio"] < 0.55
    assert r["partitioned_params"] < r["classical_params"]


def test_table_param_count_matches_formula():
    # num_embedding_parameters() == Σ_b |V^b|·D^b
    part = PartitioningEmbedder(23665, 128)
    expected = sum(v * d for v, d in zip(part.bin_vocab, part.bin_dim))
    assert part.num_embedding_parameters() == expected


# --------------------------------------------------------------------------- #
# Interface parity + forward semantics
# --------------------------------------------------------------------------- #

def test_forward_output_shape_matches_classical():
    v, d = 1000, 64
    part = PartitioningEmbedder(v, d)
    clf = ClassicalEmbedder(v, d)
    ids = torch.randint(0, v, (8, 5))
    assert part(ids).shape == (8, 5, d)
    assert clf(ids).shape == (8, 5, d)


def test_forward_handles_1d_and_scalar_batches():
    v, d = 500, 32
    part = PartitioningEmbedder(v, d)
    assert part(torch.randint(0, v, (16,))).shape == (16, d)
    assert part(torch.tensor([3])).shape == (1, d)


def test_direct_sum_placement():
    # A token in bin b must populate ONLY bin b's coordinate slice; the rest of
    # the D-dim output stays exactly zero (direct-sum, no projection).
    v, d = 2000, 64
    part = PartitioningEmbedder(v, d)
    with torch.no_grad():
        # make every table weight nonzero so "zero coords" can only come from
        # the direct-sum placement, not from an accidentally-zero parameter.
        for t in part.tables:
            t.weight.fill_(1.0)
    ids = torch.arange(v)
    out = part(ids)  # (v, d)
    for i in range(0, v, 137):  # sample tokens across the vocab
        b = int(part.token_bin[i])
        off, dim_b = part.bin_offset[b], part.bin_dim[b]
        row = out[i]
        assert torch.all(row[off:off + dim_b] != 0)
        # everything outside the bin's slice is zero
        mask = torch.ones(d, dtype=torch.bool)
        mask[off:off + dim_b] = False
        assert torch.all(row[mask] == 0)


def test_frequency_assignment_puts_frequent_token_in_bin_one():
    # Token 7 is by far the most frequent → must land in bin 1 (the small,
    # high-dim bin), regardless of its id.
    v, d = 300, 64
    freqs = torch.ones(v)
    freqs[7] = 1e6
    part = PartitioningEmbedder(v, d, token_frequencies=freqs)
    assert int(part.token_bin[7]) == 0


def test_gradients_flow_to_tables():
    v, d = 400, 64
    part = PartitioningEmbedder(v, d)
    ids = torch.randint(0, v, (32,))
    part(ids).sum().backward()
    touched = [t.weight.grad is not None and torch.any(t.weight.grad != 0)
               for t in part.tables]
    assert any(touched)
