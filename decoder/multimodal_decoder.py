"""
Layer 4 multimodal decoder (§4 / §4.1) — frozen tabular encoder f + frozen LLM,
with only the adapter Φ, task embedding ψ, and per-layer prompt params φ trained.

Grounded strictly to:
  Raman, Ganesh, Veloso (JPMorgan AI Research),
  "Scalable Representation Learning for Multimodal Tabular Transactions",
  arXiv:2410.07851, NeurIPS 2024 TRL workshop.

# PAPER: §4 / §4.1.
# Interleaving (Eq. 5), single-record v1:
#     z_i = Ξ_LLM(s(1)) ⊕ Φ(f(x_i1)) ⊕ Ξ_LLM(t_i) ⊕ Ξ_task(k_i)
# Objective (Eq. 6):
#     L = − Σ_i log P(y_i | z_i ; Φ, ψ, φ)
#
# Trainable: Φ (adapter), ψ (task embedding), φ (per-layer prompt params). FROZEN:
# the tabular encoder f and the LLM (handoff §0.3 INVARIANT — if either trains it
# is a different experiment). Row sentinels s(·) = [R1], [R2], … identify records.
#
# The paper leaves adapter depth/width, the φ injection mechanism, the task
# unique/shared split, and sentinel tokenization unspecified. v1 choices
# (documented, §7-style):
#   * Φ = linear projection D_enc→D_llm + a small transformer; emits 1 soft token
#     per record (matches the single Φ(f(x)) per record in Eq. 5). # PAPER: §4.1
#   * ψ = concat(unique_table[k] ‖ shared_vector), the literal "subspace unique to
#     each task + subspace shared across tasks". # PAPER: §4.1
#   * φ = per-layer learnable prefix key/value (prefix-tuning [Li & Liang]); the
#     paper's "augment each layer … similar to prompt tuning". # PAPER: §4.1
#   * Row sentinel = a LEARNABLE soft token. The paper frames it as an LLM-vocab
#     token, but we operate in embedding space with a FROZEN LLM, so realizing it
#     as a trainable embedding keeps the freeze invariant exact (no vocab resize).
#     v1 is single-record → one sentinel, [R1].
#
# The concrete frozen LLM is swappable behind LLMInterface. A tiny self-contained
# MockLLM exercises all trainable logic on CPU; HFCausalLM wraps a real Phi-class
# model for the (GPU) instruction-tuning run — see architecture.md §8.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Protocol

import torch
import torch.nn as nn
import torch.nn.functional as F


# --------------------------------------------------------------------------- #
# Frozen-LLM interface
# --------------------------------------------------------------------------- #

class LLMInterface(Protocol):
    """Minimal contract the decoder needs from a (frozen) causal LLM."""
    hidden_size: int
    num_layers: int
    num_heads: int
    head_dim: int
    vocab_size: int

    def embed_tokens(self, ids: torch.Tensor) -> torch.Tensor: ...  # Ξ_LLM
    def forward_embeds(
        self, inputs_embeds: torch.Tensor, attention_mask: torch.Tensor,
        prefixes: Optional[list] = None,
    ) -> torch.Tensor: ...  # → logits (B, S, vocab)


# --------------------------------------------------------------------------- #
# Trainable modules: Φ (adapter), ψ (task embedding), φ (per-layer prefixes)
# --------------------------------------------------------------------------- #

class Adapter(nn.Module):
    """Φ — project the frozen row embedding f(x) into the LLM token space.

    # PAPER: §4.1 "Φ consists of a small set of transformer layers." Emits
    # `n_tokens` soft tokens per record (default 1, matching Eq. 5).
    """

    def __init__(self, d_enc: int, d_llm: int, n_layers: int = 2,
                 n_heads: int = 4, n_tokens: int = 1, dropout: float = 0.0):
        super().__init__()
        self.n_tokens = n_tokens
        self.d_enc = d_enc
        # Expand f into n_tokens soft tokens and run the small transformer at the
        # ENCODER bottleneck width (parameter-efficient — keeps Φ a small fraction
        # of trainable params for C2), then project up to the LLM width.
        self.expand = nn.Linear(d_enc, d_enc * n_tokens)
        layer = nn.TransformerEncoderLayer(
            d_model=d_enc, nhead=n_heads, dim_feedforward=4 * d_enc,
            dropout=dropout, batch_first=True, activation="gelu",
        )
        self.transformer = nn.TransformerEncoder(layer, num_layers=n_layers)
        self.out = nn.Linear(d_enc, d_llm)

    def forward(self, f: torch.Tensor) -> torch.Tensor:   # f: (B, d_enc)
        B = f.shape[0]
        x = self.expand(f).view(B, self.n_tokens, self.d_enc)
        x = self.transformer(x)
        return self.out(x)                                # (B, n_tokens, d_llm)


class TaskEmbedding(nn.Module):
    """ψ — Ξ_task: 1..K → R^{d_llm} with a per-task subspace ‖ a shared subspace."""

    def __init__(self, n_tasks: int, d_llm: int, shared_dim: Optional[int] = None):
        super().__init__()
        if shared_dim is None:
            shared_dim = d_llm // 2
        if not (0 < shared_dim < d_llm):
            raise ValueError(f"shared_dim must be in (0, {d_llm}), got {shared_dim}")
        self.unique_dim = d_llm - shared_dim
        self.unique = nn.Embedding(n_tasks, self.unique_dim)   # subspace unique per task
        self.shared = nn.Parameter(torch.randn(shared_dim) * 0.02)  # subspace shared

    def forward(self, task_ids: torch.Tensor) -> torch.Tensor:  # (B,)
        u = self.unique(task_ids)                               # (B, unique_dim)
        s = self.shared.unsqueeze(0).expand(u.shape[0], -1)     # (B, shared_dim)
        return torch.cat([u, s], dim=1)                         # (B, d_llm)


class PrefixEncoder(nn.Module):
    """φ — per-layer learnable prefix key/value (prefix-tuning).

    Produces, for each LLM layer, a (key, value) pair of shape
    (B, num_heads, prefix_len, head_dim) prepended to that layer's attention.
    """

    def __init__(self, num_layers: int, num_heads: int, head_dim: int,
                 prefix_len: int = 8):
        super().__init__()
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.prefix_len = prefix_len
        # (num_layers, 2, num_heads, prefix_len, head_dim) — 2 = key, value.
        self.prefix = nn.Parameter(
            torch.randn(num_layers, 2, num_heads, prefix_len, head_dim) * 0.02
        )

    def forward(self, batch_size: int) -> list:
        out = []
        for li in range(self.num_layers):
            k = self.prefix[li, 0].unsqueeze(0).expand(batch_size, -1, -1, -1)
            v = self.prefix[li, 1].unsqueeze(0).expand(batch_size, -1, -1, -1)
            out.append((k, v))
        return out


# --------------------------------------------------------------------------- #
# Multimodal decoder
# --------------------------------------------------------------------------- #

@dataclass
class DecoderConfig:
    n_tasks: int = 1
    adapter_layers: int = 2
    adapter_heads: int = 4
    adapter_tokens: int = 1
    prefix_len: int = 8
    task_shared_dim: Optional[int] = None


class MultimodalDecoder(nn.Module):
    """Frozen f + frozen LLM + trainable {Φ, ψ, φ}; instruction-tuned per Eq. 6."""

    def __init__(self, encoder, llm: LLMInterface, config: DecoderConfig):
        super().__init__()
        self.encoder = encoder
        self.llm = llm
        self.config = config
        d_llm = llm.hidden_size

        # Trainable {Φ, ψ, φ} + the learnable row sentinel [R1].
        self.adapter = Adapter(encoder.D, d_llm, config.adapter_layers,
                               config.adapter_heads, config.adapter_tokens)
        self.task_embedding = TaskEmbedding(config.n_tasks, d_llm, config.task_shared_dim)
        self.prefix = PrefixEncoder(llm.num_layers, llm.num_heads, llm.head_dim,
                                    config.prefix_len)
        self.row_sentinel = nn.Parameter(torch.randn(d_llm) * 0.02)  # [R1]

        self._freeze_base()

    # -- freeze invariant (handoff §0.3) ---------------------------------- #
    def _freeze_base(self):
        self.encoder.freeze()                       # tabular encoder f
        for p in self.llm_parameters():             # LLM base
            p.requires_grad_(False)

    def llm_parameters(self):
        if isinstance(self.llm, nn.Module):
            return self.llm.parameters()
        return iter(())

    def assert_frozen(self):
        """Hard-assert f and the LLM carry no trainable params (Eq. 6 invariant)."""
        assert all(not p.requires_grad for p in self.encoder.parameters()), \
            "tabular encoder f must be frozen"
        assert all(not p.requires_grad for p in self.llm_parameters()), \
            "LLM must be frozen"

    def trainable_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    # -- Eq. 5 interleaving ----------------------------------------------- #
    def build_inputs(self, batch: dict, task_ids: torch.Tensor,
                     instruction_ids: torch.Tensor):
        """z_i = Ξ_LLM(s) ⊕ Φ(f(x)) ⊕ Ξ_LLM(t) ⊕ Ξ_task(k)  → (embeds, mask)."""
        with torch.no_grad():                       # f is frozen → constant feature
            f = self.encoder.encode(batch)          # (B, D_enc)
        B = f.shape[0]

        sentinel = self.row_sentinel.view(1, 1, -1).expand(B, -1, -1)   # (B,1,Dllm)
        record = self.adapter(f)                                        # (B,n_tok,Dllm)
        instr = self.llm.embed_tokens(instruction_ids)                  # (B,S_t,Dllm)
        task = self.task_embedding(task_ids).unsqueeze(1)               # (B,1,Dllm)

        z = torch.cat([sentinel, record, instr, task], dim=1)           # (B,S_z,Dllm)
        mask = torch.ones(z.shape[0], z.shape[1], device=z.device)
        return z, mask

    # -- Eq. 6 objective -------------------------------------------------- #
    def forward(self, batch: dict, task_ids, instruction_ids, target_ids):
        z, z_mask = self.build_inputs(batch, task_ids, instruction_ids)
        y = self.llm.embed_tokens(target_ids)                           # (B,S_y,Dllm)
        inputs = torch.cat([z, y], dim=1)
        y_mask = torch.ones(target_ids.shape[0], target_ids.shape[1], device=z.device)
        mask = torch.cat([z_mask, y_mask], dim=1)

        prefixes = self.prefix(inputs.shape[0])
        logits = self.llm.forward_embeds(inputs, mask, prefixes)        # (B,S,vocab)

        # Labels: ignore the z prefix and the prefix-tuning positions; supervise
        # only the response tokens, next-token shifted.
        Sz = z.shape[1]
        labels = torch.full((target_ids.shape[0], inputs.shape[1]), -100,
                            dtype=torch.long, device=z.device)
        labels[:, Sz:] = target_ids
        # account for the φ prefix positions the LLM prepends to the logits
        pad = logits.shape[1] - inputs.shape[1]
        if pad > 0:
            labels = torch.cat(
                [torch.full((labels.shape[0], pad), -100, device=z.device), labels],
                dim=1,
            )
        shift_logits = logits[:, :-1].reshape(-1, logits.shape[-1])
        shift_labels = labels[:, 1:].reshape(-1)
        return F.cross_entropy(shift_logits, shift_labels, ignore_index=-100)

    def loss(self, *args, **kwargs):
        return self.forward(*args, **kwargs)


# --------------------------------------------------------------------------- #
# MockLLM — self-contained tiny causal LM exercising the φ prefix path (CPU test)
# --------------------------------------------------------------------------- #

class _MockBlock(nn.Module):
    def __init__(self, d: int, heads: int):
        super().__init__()
        self.h = heads
        self.hd = d // heads
        self.qkv = nn.Linear(d, 3 * d)
        self.o = nn.Linear(d, d)
        self.ln1 = nn.LayerNorm(d)
        self.ln2 = nn.LayerNorm(d)
        self.ff = nn.Sequential(nn.Linear(d, 4 * d), nn.GELU(), nn.Linear(4 * d, d))

    def forward(self, x, attn_mask, prefix=None):
        B, S, D = x.shape
        h = self.ln1(x)
        q, k, v = self.qkv(h).split(D, dim=2)
        q = q.view(B, S, self.h, self.hd).transpose(1, 2)
        k = k.view(B, S, self.h, self.hd).transpose(1, 2)
        v = v.view(B, S, self.h, self.hd).transpose(1, 2)
        Pk = 0
        if prefix is not None:                       # prepend φ prefix key/value
            pk, pv = prefix
            Pk = pk.shape[2]
            k = torch.cat([pk, k], dim=2)
            v = torch.cat([pv, v], dim=2)
        scores = (q @ k.transpose(-2, -1)) / (self.hd ** 0.5)   # (B,h,S,Pk+S)
        # causal among the S real positions; prefix fully visible.
        causal = torch.tril(torch.ones(S, S, device=x.device, dtype=torch.bool))
        allow = torch.cat([torch.ones(S, Pk, device=x.device, dtype=torch.bool),
                           causal], dim=1)            # (S, Pk+S)
        scores = scores.masked_fill(~allow.view(1, 1, S, Pk + S), float("-inf"))
        att = scores.softmax(dim=-1)
        ctx = (att @ v).transpose(1, 2).reshape(B, S, D)
        x = x + self.o(ctx)
        x = x + self.ff(self.ln2(x))
        return x


class MockLLM(nn.Module):
    """Tiny causal LM implementing LLMInterface — for CPU unit tests only."""

    def __init__(self, vocab_size=64, hidden=32, num_layers=2, num_heads=4, max_len=64):
        super().__init__()
        self.hidden_size = hidden
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.head_dim = hidden // num_heads
        self.vocab_size = vocab_size
        self.tok = nn.Embedding(vocab_size, hidden)
        self.pos = nn.Parameter(torch.randn(1, max_len, hidden) * 0.02)
        self.blocks = nn.ModuleList([_MockBlock(hidden, num_heads) for _ in range(num_layers)])
        self.ln_f = nn.LayerNorm(hidden)
        self.lm_head = nn.Linear(hidden, vocab_size)

    def embed_tokens(self, ids):
        return self.tok(ids)

    def forward_embeds(self, inputs_embeds, attention_mask, prefixes=None):
        B, S, D = inputs_embeds.shape
        x = inputs_embeds + self.pos[:, :S]
        for li, blk in enumerate(self.blocks):
            x = blk(x, attention_mask, prefix=(prefixes[li] if prefixes else None))
        return self.lm_head(self.ln_f(x))            # (B, S, vocab)


# --------------------------------------------------------------------------- #
# HF wrapper for the real (GPU) Phi run — built, not unit-tested here
# --------------------------------------------------------------------------- #

class HFCausalLM(nn.Module):
    """Wrap a HuggingFace causal LM (e.g. Phi-1.5) behind LLMInterface.

    transformers is imported lazily so the decoder module (and its MockLLM tests)
    have no hard transformers/Phi dependency. Per-layer φ prefixes are passed via
    the model's past_key_values. Used only in the deferred GPU instruction-tuning
    run (architecture.md §8); not exercised by the CPU test suite.
    """

    def __init__(self, model_name: str = "microsoft/phi-1_5"):
        super().__init__()
        from transformers import AutoModelForCausalLM, AutoTokenizer  # lazy

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name)
        cfg = self.model.config
        self.hidden_size = cfg.hidden_size
        self.num_layers = cfg.num_hidden_layers
        self.num_heads = getattr(cfg, "num_attention_heads")
        self.head_dim = self.hidden_size // self.num_heads
        self.vocab_size = cfg.vocab_size

    def embed_tokens(self, ids):
        return self.model.get_input_embeddings()(ids)

    def forward_embeds(self, inputs_embeds, attention_mask, prefixes=None):
        past = None
        if prefixes is not None:
            # HF expects past_key_values as a tuple of (key, value) per layer,
            # each (B, num_heads, prefix_len, head_dim). The attention_mask must
            # be widened to cover the prefix positions.
            past = tuple(prefixes)
            B, pk = inputs_embeds.shape[0], prefixes[0][0].shape[2]
            pad = torch.ones(B, pk, device=inputs_embeds.device, dtype=attention_mask.dtype)
            attention_mask = torch.cat([pad, attention_mask], dim=1)
        out = self.model(inputs_embeds=inputs_embeds, attention_mask=attention_mask,
                         past_key_values=past, use_cache=False)
        return out.logits


# --------------------------------------------------------------------------- #
# CLI: record the C2 trainable-param half (trio sized vs a real Phi config)
# --------------------------------------------------------------------------- #

# Reference frozen-LLM presets (dims only — no weights downloaded).
_LLM_PRESETS = {
    "mock":    dict(hidden=32,   layers=2,  heads=4,  full_tune_params=int(1e5)),
    "phi-1_5": dict(hidden=2048, layers=24, heads=32, full_tune_params=1_300_000_000),
}


def main():
    import argparse

    ap = argparse.ArgumentParser(description="Layer 4 decoder — C2 trainable-param report")
    ap.add_argument("--llm", choices=list(_LLM_PRESETS), default="phi-1_5")
    ap.add_argument("--enc-hidden", type=int, default=512, help="tabular encoder D (d_enc)")
    ap.add_argument("--adapter-layers", type=int, default=DecoderConfig.adapter_layers)
    ap.add_argument("--adapter-heads", type=int, default=DecoderConfig.adapter_heads)
    ap.add_argument("--adapter-tokens", type=int, default=DecoderConfig.adapter_tokens)
    ap.add_argument("--prefix-len", type=int, default=DecoderConfig.prefix_len)
    ap.add_argument("--n-tasks", type=int, default=3)
    args = ap.parse_args()

    p = _LLM_PRESETS[args.llm]
    d_llm, n_layers, n_heads = p["hidden"], p["layers"], p["heads"]
    head_dim = d_llm // n_heads

    # The trainable trio + sentinel — instantiated directly (no LLM weights needed).
    adapter = Adapter(args.enc_hidden, d_llm, args.adapter_layers,
                      args.adapter_heads, args.adapter_tokens)
    task = TaskEmbedding(args.n_tasks, d_llm)
    prefix = PrefixEncoder(n_layers, n_heads, head_dim, args.prefix_len)

    def n(m):
        return sum(t.numel() for t in (m.parameters() if isinstance(m, nn.Module) else [m]))

    phi = n(adapter)
    psi = n(task)
    phi_prefix = n(prefix)
    sentinel = d_llm
    trio = phi + psi + phi_prefix + sentinel
    full = p["full_tune_params"]

    print(f"Layer 4 decoder trainable params (LLM={args.llm}, d_enc={args.enc_hidden}, "
          f"d_llm={d_llm})")
    print(f"  Phi  adapter        : {phi:>12,}")
    print(f"  psi  task embedding : {psi:>12,}")
    print(f"  phi  prefix params  : {phi_prefix:>12,}")
    print(f"  [R1] row sentinel   : {sentinel:>12,}")
    print(f"  ---------------------------------")
    print(f"  trainable (trio)    : {trio:>12,}")
    print(f"  full-tune reference : {full:>12,}  ({args.llm} base)")
    ratio = trio / full
    print(f"  C2 trainable_param_ratio (trio / full-tune): {ratio:.4f}  "
          f"(threshold <= 0.10)  {'PASS' if ratio <= 0.10 else 'FAIL'}")
    print("NOTE: this is the C2 PARAM half. The PR-AUC half (vs CatBoost / full-tune)"
          " needs the GPU instruction-tuning run.")


if __name__ == "__main__":
    main()
