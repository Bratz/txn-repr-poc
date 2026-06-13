"""
§3.1 Partitioning embedder for high-cardinality categorical identifiers.

Grounded strictly to:
  Raman, Ganesh, Veloso (JPMorgan AI Research),
  "Scalable Representation Learning for Multimodal Tabular Transactions",
  arXiv:2410.07851, NeurIPS 2024 TRL workshop.

# PAPER: §3.1. A classical identifier embedding E ∈ R^{|V|×D} is replaced by B
# power-law-sized bins E^b ∈ R^{|V^b|×D^b}. The shared embedding space is a
# DIRECT SUM, R^D = R^{D^1} ⊕ … ⊕ R^{D^B} (Σ_b D^b = D): a token assigned to
# bin b receives its D^b-dimensional row placed into bin b's contiguous
# coordinate slice of the D-dim output; the other coordinates stay zero. There
# is NO up-projection back to D — the paper keeps the partitioned subspaces.
#
# Bin sizing (Eq. 2 and its dimension analogue), with paper exponents:
#   |V^b| = |V| · b^{-α_v} / Σ_{j=1}^B j^{-α_v}      α_v = -3   → |V^1| ≪ … ≪ |V^B|
#   D^b   = D   · b^{-α_d} / Σ_{j=1}^B j^{-α_d}      α_d = 2.25 → D^1   ≫ … ≫ D^B
# i.e. the few frequent items land in a small, high-dimensional bin; the many
# rare items share a large, low-dimensional bin. Net params Σ_b |V^b|·D^b ≪ |V|·D.
#
# These hyperparameters (B=4, α_v=-3, α_d=2.25) are the experiment — do NOT
# retune them (handoff §0.2).
"""

from __future__ import annotations

import math
from typing import Optional, Sequence

import torch
import torch.nn as nn

# Paper-pinned hyperparameters (handoff §0.2 — frozen, do not retune).
PAPER_B = 4
PAPER_ALPHA_V = -3.0
PAPER_ALPHA_D = 2.25


# --------------------------------------------------------------------------- #
# Power-law allocation (Eq. 2 and its dimension analogue)
# --------------------------------------------------------------------------- #

def power_law_partition(total: int, B: int, alpha: float) -> list[int]:
    """Split `total` units across `B` bins with weights w_b ∝ b^{-alpha}.

    # PAPER: §3.1 Eq. 2. Used for both the vocabulary split (alpha = α_v) and
    # the embedding-dimension split (alpha = α_d). Returns B positive ints that
    # sum EXACTLY to `total` (largest-remainder rounding). Each bin is forced to
    # ≥1 unit — a real bin cannot own a zero-width subspace — which requires
    # total ≥ B.

    A negative alpha makes later bins larger (vocab, α_v=-3); a positive alpha
    makes earlier bins larger (dimension, α_d=2.25).
    """
    if B <= 0:
        raise ValueError(f"B must be positive, got {B}")
    if total < B:
        raise ValueError(f"total={total} < B={B}: cannot give every bin ≥1 unit")

    weights = [b ** (-alpha) for b in range(1, B + 1)]
    s = sum(weights)
    raw = [total * w / s for w in weights]

    alloc = [int(math.floor(x)) for x in raw]
    # Distribute the rounding remainder to the largest fractional parts.
    rem = total - sum(alloc)
    order = sorted(range(B), key=lambda b: raw[b] - math.floor(raw[b]), reverse=True)
    for i in range(rem):
        alloc[order[i % B]] += 1

    # Numerical guard: force every bin to ≥1 by stealing from the current
    # largest bin (keeps the sum invariant). Only bites when a power-law weight
    # rounds to 0; never triggers at our embedding dims (D≥64) and vocab sizes.
    for b in range(B):
        if alloc[b] == 0:
            donor = max(range(B), key=lambda k: alloc[k])
            alloc[donor] -= 1
            alloc[b] += 1

    assert sum(alloc) == total, (alloc, total)
    return alloc


# --------------------------------------------------------------------------- #
# Partitioning embedder (§3.1)
# --------------------------------------------------------------------------- #

class PartitioningEmbedder(nn.Module):
    """Power-law-partitioned embedding for one high-cardinality column.

    Interface mirrors :class:`ClassicalEmbedder` so the two are drop-in swappable
    for the C1 parameter-efficiency comparison:
      - ``forward(input_ids) -> (*input_ids.shape, embedding_dim)``
      - ``num_embedding_parameters() -> int``

    Args:
        vocab_size: |V| for this column.
        embedding_dim: D, the shared output dimension (Σ_b D^b = D).
        B, alpha_v, alpha_d: paper hyperparameters — leave at the pinned values.
        token_frequencies: optional length-|V| counts. If given, tokens are
            assigned to bins by descending frequency (most frequent → bin 1).
            If omitted, token ids are assumed already in frequency-desc order
            (id 0 = most frequent).
    """

    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int,
        B: int = PAPER_B,
        alpha_v: float = PAPER_ALPHA_V,
        alpha_d: float = PAPER_ALPHA_D,
        token_frequencies: Optional[Sequence[float]] = None,
    ):
        super().__init__()
        self.vocab_size = int(vocab_size)
        self.embedding_dim = int(embedding_dim)
        self.B = int(B)
        self.alpha_v = float(alpha_v)
        self.alpha_d = float(alpha_d)

        # |V^b| and D^b (Eq. 2 + dimension analogue).
        self.bin_vocab = power_law_partition(self.vocab_size, self.B, self.alpha_v)
        self.bin_dim = power_law_partition(self.embedding_dim, self.B, self.alpha_d)

        # Contiguous coordinate slice each bin owns in the direct-sum R^D.
        self.bin_offset: list[int] = []
        running = 0
        for d in self.bin_dim:
            self.bin_offset.append(running)
            running += d

        # Per-bin tables E^b ∈ R^{|V^b|×D^b}.
        self.tables = nn.ModuleList(
            [nn.Embedding(self.bin_vocab[b], self.bin_dim[b]) for b in range(self.B)]
        )

        self._build_assignment(token_frequencies)

    def _build_assignment(self, token_frequencies: Optional[Sequence[float]]):
        """Map each token id → (bin, local index) by frequency rank.

        # PAPER: §3.1 does not specify the assignment rule. Frequency-rank is the
        # only reading consistent with the power-law-on-frequency motivation —
        # frequent items into the small high-dimensional bin, rare items into the
        # large low-dimensional bin — so we adopt it and flag it here.
        """
        V = self.vocab_size
        if token_frequencies is None:
            rank = torch.arange(V)
        else:
            freq = torch.as_tensor(token_frequencies, dtype=torch.float)
            if freq.numel() != V:
                raise ValueError(
                    f"token_frequencies has {freq.numel()} entries, expected {V}"
                )
            order = torch.argsort(freq, descending=True)  # token ids, most→least frequent
            rank = torch.empty(V, dtype=torch.long)
            rank[order] = torch.arange(V)

        # Cumulative vocab boundaries define which rank range falls in which bin.
        starts, ends, c = [], [], 0
        for b in range(self.B):
            starts.append(c)
            c += self.bin_vocab[b]
            ends.append(c)

        token_bin = torch.empty(V, dtype=torch.long)
        token_local = torch.empty(V, dtype=torch.long)
        for b in range(self.B):
            mask = (rank >= starts[b]) & (rank < ends[b])
            token_bin[mask] = b
            token_local[mask] = rank[mask] - starts[b]

        # Buffers move with .to(device) / .cuda() and are saved in state_dict.
        self.register_buffer("token_bin", token_bin, persistent=True)
        self.register_buffer("token_local", token_local, persistent=True)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        ids = input_ids.long()
        flat = ids.reshape(-1)
        out = torch.zeros(
            flat.shape[0],
            self.embedding_dim,
            device=flat.device,
            dtype=self.tables[0].weight.dtype,
        )
        bins = self.token_bin[flat]
        locs = self.token_local[flat]
        for b in range(self.B):
            mask = bins == b
            if mask.any():
                emb = self.tables[b](locs[mask])
                off = self.bin_offset[b]
                out[mask, off:off + self.bin_dim[b]] = emb
        return out.reshape(*ids.shape, self.embedding_dim)

    def num_embedding_parameters(self) -> int:
        """Σ_b |V^b|·D^b — the embedding-table parameter count (no projections)."""
        return sum(self.bin_vocab[b] * self.bin_dim[b] for b in range(self.B))

    def extra_repr(self) -> str:
        return (
            f"vocab_size={self.vocab_size}, embedding_dim={self.embedding_dim}, "
            f"B={self.B}, bin_vocab={self.bin_vocab}, bin_dim={self.bin_dim}"
        )


# --------------------------------------------------------------------------- #
# Classical control (the C1 baseline)
# --------------------------------------------------------------------------- #

class ClassicalEmbedder(nn.Module):
    """Standard dense embedding table E ∈ R^{|V|×D} — the C1 control.

    Same interface as :class:`PartitioningEmbedder` so param counts and forward
    outputs are directly comparable.
    """

    def __init__(self, vocab_size: int, embedding_dim: int):
        super().__init__()
        self.vocab_size = int(vocab_size)
        self.embedding_dim = int(embedding_dim)
        self.emb = nn.Embedding(self.vocab_size, self.embedding_dim)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.emb(input_ids.long())

    def num_embedding_parameters(self) -> int:
        return self.vocab_size * self.embedding_dim


# --------------------------------------------------------------------------- #
# C1 reporting helper
# --------------------------------------------------------------------------- #

def param_efficiency(
    vocab_size: int,
    embedding_dim: int,
    B: int = PAPER_B,
    alpha_v: float = PAPER_ALPHA_V,
    alpha_d: float = PAPER_ALPHA_D,
) -> dict:
    """Embedding-table param counts for partitioned vs classical, at one vocab.

    NOTE (C1 scope): this is the EMBEDDING-TABLE ratio (Σ_b|V^b|D^b vs |V|D),
    which is what this module owns. The paper's headline ≈100M/185M≈0.54 is a
    TOTAL-MODEL ratio. Adjudicate C1's `param_ratio` threshold against the right
    denominator in Phase 2e — do not read the table ratio as the model ratio.
    """
    part = PartitioningEmbedder(vocab_size, embedding_dim, B, alpha_v, alpha_d)
    p = part.num_embedding_parameters()
    c = vocab_size * embedding_dim
    return {
        "vocab_size": int(vocab_size),
        "embedding_dim": int(embedding_dim),
        "bin_vocab": part.bin_vocab,
        "bin_dim": part.bin_dim,
        "partitioned_params": int(p),
        "classical_params": int(c),
        "param_ratio": p / c,
    }


# --------------------------------------------------------------------------- #
# CLI: report C1 param efficiency on the realized schema vocabs
# --------------------------------------------------------------------------- #

def main():
    import argparse
    import json
    from pathlib import Path

    ap = argparse.ArgumentParser(
        description="§3.1 partitioning embedder — C1 param-efficiency report"
    )
    ap.add_argument(
        "--schema",
        type=str,
        default=str(Path(__file__).resolve().parents[1] / "data" / "column_schema.json"),
        help="column_schema.json (authoritative bucket + vocab manifest)",
    )
    ap.add_argument("--dim", type=int, default=128,
                    help="shared embedding dim D (POC choice; ratio ~D-independent)")
    args = ap.parse_args()

    schema = json.loads(Path(args.schema).read_text())
    # Read buckets/vocab from the schema — never hard-code column lists (§0.4).
    high_card = schema["buckets"]["high_card_categorical"]
    vocab = schema["vocab"]

    print(f"Partitioning embedder C1 report (B={PAPER_B}, "
          f"alpha_v={PAPER_ALPHA_V}, alpha_d={PAPER_ALPHA_D}, D={args.dim})")
    print("NOTE: embedding-TABLE ratio; paper's 0.54 is a total-MODEL ratio.\n")
    print(f"{'vocab key':<28}{'|V|':>8}{'classical':>14}{'partitioned':>14}{'ratio':>9}")

    keys = [k for k in high_card if k in vocab]
    keys += [k for k in vocab if k not in keys]  # include combined_* vocabs too
    for k in keys:
        r = param_efficiency(vocab[k], args.dim)
        print(f"{k:<28}{r['vocab_size']:>8}{r['classical_params']:>14,}"
              f"{r['partitioned_params']:>14,}{r['param_ratio']:>9.3f}")


if __name__ == "__main__":
    main()
