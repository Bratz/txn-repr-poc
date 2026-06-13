"""Layer 3 tabular encoder (§3.4). Frozen after pretraining for Layer 4."""

from .tabular_encoder import (
    TabularEncoder,
    EncoderConfig,
    batch_hard_triplet_loss,
)

__all__ = ["TabularEncoder", "EncoderConfig", "batch_hard_triplet_loss"]
