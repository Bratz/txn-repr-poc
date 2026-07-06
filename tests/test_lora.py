"""LoRA adapters: identity at init, merge equivalence, no silent eval bypass."""

import torch
from torch import nn

from encoder.lora import (LoRALinear, inject_lora, lora_parameters, merge_lora,
                          mark_only_lora_trainable)


def test_identity_at_init_and_frozen_base():
    torch.manual_seed(0)
    base = nn.Linear(16, 8)
    x = torch.randn(4, 16)
    ref = base(x)
    lora = LoRALinear(base, rank=4)
    assert torch.allclose(lora(x), ref)                  # B=0 -> identity at step 0
    assert not lora.base.weight.requires_grad            # frozen structurally
    assert lora.lora_a.requires_grad and lora.lora_b.requires_grad


def test_merge_equivalence_after_update():
    torch.manual_seed(0)
    lora = LoRALinear(nn.Linear(16, 8), rank=4)
    with torch.no_grad():                                # simulate training
        lora.lora_b.normal_(); lora.lora_a.normal_()
    x = torch.randn(4, 16)
    adapted = lora(x)
    merged = lora.merged_linear()
    assert torch.allclose(merged(x), adapted, atol=1e-5)


def test_transformer_injection_no_eval_bypass():
    """The critical property: after injection, EVAL-mode output must reflect the
    LoRA update (torch's fast path would silently use raw weights)."""
    torch.manual_seed(0)
    layer = nn.TransformerEncoderLayer(d_model=16, nhead=2, dim_feedforward=32,
                                       batch_first=True, activation="gelu")
    enc = nn.TransformerEncoder(layer, num_layers=2)
    x = torch.randn(3, 5, 16)
    enc.eval()
    with torch.no_grad():
        ref = enc(x)
    n = inject_lora(enc, rank=4)
    assert n == 4                                        # linear1+linear2 x 2 layers
    enc.eval()
    with torch.no_grad():
        assert torch.allclose(enc(x), ref, atol=1e-5)    # identity at init, eval mode
        for p in lora_parameters(enc):                   # perturb the adapters
            p.normal_()
        out = enc(x)
    assert not torch.allclose(out, ref, atol=1e-3)       # update APPLIED in eval mode

    mark_only_lora_trainable(enc)
    trainable = [p for p in enc.parameters() if p.requires_grad]
    assert len(trainable) == 2 * n                       # A and B per adapted layer

    merged = merge_lora(enc)
    assert merged == n
    enc.eval()
    with torch.no_grad():
        assert torch.allclose(enc(x), out, atol=1e-4)    # merge preserves the function


def test_base_weights_never_move():
    torch.manual_seed(0)
    layer = nn.TransformerEncoderLayer(d_model=16, nhead=2, dim_feedforward=32,
                                       batch_first=True)
    enc = nn.TransformerEncoder(layer, num_layers=1)
    inject_lora(enc, rank=2)
    mark_only_lora_trainable(enc)
    base_before = [m.base.weight.clone() for m in enc.modules() if isinstance(m, LoRALinear)]
    opt = torch.optim.AdamW([p for p in enc.parameters() if p.requires_grad], lr=1e-2)
    loss = enc(torch.randn(2, 4, 16)).pow(2).mean()
    loss.backward(); opt.step()
    base_after = [m.base.weight for m in enc.modules() if isinstance(m, LoRALinear)]
    for b, a in zip(base_before, base_after):
        assert torch.equal(b, a)                         # the invariant, literally
