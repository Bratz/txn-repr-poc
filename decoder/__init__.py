"""Layer 4 multimodal decoder (§4/§4.1): frozen f + frozen LLM + {Φ, ψ, φ}."""

from .multimodal_decoder import (
    MultimodalDecoder,
    DecoderConfig,
    Adapter,
    TaskEmbedding,
    PrefixEncoder,
    MockLLM,
    HFCausalLM,
)

__all__ = [
    "MultimodalDecoder",
    "DecoderConfig",
    "Adapter",
    "TaskEmbedding",
    "PrefixEncoder",
    "MockLLM",
    "HFCausalLM",
]
