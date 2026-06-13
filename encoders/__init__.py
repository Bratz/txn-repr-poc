"""Layer 2 field encoders (§3.1–§3.3) and the column assembler (§3.4 input).

# PAPER: arXiv:2410.07851. One module per field-encoder path; downstream code
# reads column buckets from column_schema.json, never from hard-coded lists.
"""

from .partitioning_embedder import (
    PartitioningEmbedder,
    ClassicalEmbedder,
    power_law_partition,
    param_efficiency,
    PAPER_B,
    PAPER_ALPHA_V,
    PAPER_ALPHA_D,
)

__all__ = [
    "PartitioningEmbedder",
    "ClassicalEmbedder",
    "power_law_partition",
    "param_efficiency",
    "PAPER_B",
    "PAPER_ALPHA_V",
    "PAPER_ALPHA_D",
]
