"""Low-rank adapters (LoRA, Hu et al. 2022) for the frozen tabular encoder.

The frozen-base invariant HOLDS here: LoRALinear freezes the wrapped layer's
weight/bias in its own constructor and trains only the rank-r update B @ A.
B starts at zero, so the adapted model is exactly the pretrained model at
step 0. merge() folds the update back into a plain nn.Linear for export -
serving carries no adapter machinery.

Targets for our nn.TransformerEncoder backbone: the two feed-forward linears
("linear1", "linear2") - roughly two thirds of each layer's parameters.
Attention is NOT adapted at all: nn.MultiheadAttention keeps QKV as a fused
raw parameter and reads out_proj.weight directly through the functional path,
so wrapping either would be bypassed (or crash). FFN-only LoRA is a known,
documented reduction vs LoRA-on-QKV.
"""

from __future__ import annotations

import math

import torch
from torch import nn

ENCODER_TARGETS: tuple[str, ...] = ("linear1", "linear2")


class LoRALinear(nn.Module):
    """A frozen nn.Linear plus a trainable rank-r update (alpha/r scaled)."""

    def __init__(self, base: nn.Linear, rank: int = 8, alpha: float = 8.0,
                 dropout: float = 0.0):
        super().__init__()
        self.base = base
        self.base.weight.requires_grad_(False)
        if self.base.bias is not None:
            self.base.bias.requires_grad_(False)
        self.rank = rank
        self.scaling = alpha / rank
        dev, dt = base.weight.device, base.weight.dtype
        self.lora_a = nn.Parameter(torch.zeros(rank, base.in_features, device=dev, dtype=dt))
        self.lora_b = nn.Parameter(torch.zeros(base.out_features, rank, device=dev, dtype=dt))
        self.drop = nn.Dropout(dropout)
        nn.init.kaiming_uniform_(self.lora_a, a=math.sqrt(5))   # B=0 -> identity at init

    @property
    def weight(self):
        # nn.TransformerEncoderLayer's fast-path eligibility check reads
        # .weight/.bias as raw attributes; delegate to the frozen base.
        return self.base.weight

    @property
    def bias(self):
        return self.base.bias

    def forward(self, x):
        return self.base(x) + self.scaling * (self.drop(x) @ self.lora_a.t() @ self.lora_b.t())

    def merged_linear(self) -> nn.Linear:
        m = nn.Linear(self.base.in_features, self.base.out_features,
                      bias=self.base.bias is not None,
                      device=self.base.weight.device, dtype=self.base.weight.dtype)
        with torch.no_grad():
            m.weight.copy_(self.base.weight + self.scaling * (self.lora_b @ self.lora_a))
            if self.base.bias is not None:
                m.bias.copy_(self.base.bias)
        return m


def inject_lora(model: nn.Module, rank: int = 8, alpha: float = 8.0,
                targets: tuple[str, ...] = ENCODER_TARGETS, dropout: float = 0.0) -> int:
    """Swap targeted nn.Linear children for LoRALinear in-place; returns count.

    Also disables torch's native transformer fast path (process-wide): in eval
    mode it calls F.linear on raw .weight tensors, which would silently BYPASS
    the LoRA update - trained adapters that vanish at inference. The slow path
    calls the modules, so the update is applied consistently."""
    try:
        torch.backends.mha.set_fastpath_enabled(False)
    except AttributeError:                      # older torch: no fast path to disable
        pass
    n = 0
    for name, module in list(model.named_modules()):
        for child_name, child in list(module.named_children()):
            qual = f"{name}.{child_name}" if name else child_name
            if isinstance(child, nn.Linear) and any(t in qual for t in targets):
                setattr(module, child_name, LoRALinear(child, rank, alpha, dropout))
                n += 1
    return n


def merge_lora(model: nn.Module) -> int:
    """Fold every LoRALinear back into a plain nn.Linear in-place; returns count."""
    n = 0
    for _, module in model.named_modules():
        for child_name, child in list(module.named_children()):
            if isinstance(child, LoRALinear):
                setattr(module, child_name, child.merged_linear())
                n += 1
    return n


def lora_parameters(model: nn.Module):
    for _, m in model.named_modules():
        if isinstance(m, LoRALinear):
            yield m.lora_a
            yield m.lora_b


def mark_only_lora_trainable(model: nn.Module) -> None:
    ids = {id(p) for p in lora_parameters(model)}
    for p in model.parameters():
        p.requires_grad_(id(p) in ids)
