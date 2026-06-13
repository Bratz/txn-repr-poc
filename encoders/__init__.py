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
from .quantizer import (
    AdaptiveQuantizer,
    geometric_levels,
    make_quantizer_embedder,
    DEFAULT_NUM_LEVELS,
)
from .party_encoder import (
    PartyEncoder,
    PartyStore,
    build_party_store,
    party_roles_from_schema,
    build_field_vocabs,
    encode_role_parties,
    PARTY_STRUCT_ATTRS,
)
from .column_assembler import (
    ColumnAssembler,
    ColumnVocabs,
    build_vocabs,
    build_party_matrix,
)

__all__ = [
    "PartitioningEmbedder",
    "ClassicalEmbedder",
    "power_law_partition",
    "param_efficiency",
    "PAPER_B",
    "PAPER_ALPHA_V",
    "PAPER_ALPHA_D",
    "AdaptiveQuantizer",
    "geometric_levels",
    "make_quantizer_embedder",
    "DEFAULT_NUM_LEVELS",
    "PartyEncoder",
    "PartyStore",
    "build_party_store",
    "party_roles_from_schema",
    "build_field_vocabs",
    "encode_role_parties",
    "PARTY_STRUCT_ATTRS",
    "ColumnAssembler",
    "ColumnVocabs",
    "build_vocabs",
    "build_party_matrix",
]
