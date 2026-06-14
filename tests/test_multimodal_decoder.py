"""Unit tests for the Layer 4 multimodal decoder (§4/§4.1) on a MockLLM."""

import pytest
import torch

from decoder.multimodal_decoder import (
    DecoderConfig,
    MockLLM,
    MultimodalDecoder,
    PrefixEncoder,
    TaskEmbedding,
)
from encoder.tabular_encoder import EncoderConfig, build_pretraining_stack

TINY_ENC = EncoderConfig(hidden=32, layers=2, heads=2, ff_mult=2, dropout=0.0)
_RISK = {"Low": 0, "Medium": 1, "High": 2}


@pytest.fixture(scope="module")
def decoder_stack(request):
    schema = request.getfixturevalue("schema")
    df = request.getfixturevalue("sample_df")
    torch.manual_seed(0)
    enc, asm, vocabs = build_pretraining_stack(df, schema, TINY_ENC, party_epochs=1)
    llm = MockLLM(vocab_size=64, hidden=32, num_layers=2, num_heads=4)
    dec = MultimodalDecoder(enc, llm, DecoderConfig(n_tasks=3, adapter_tokens=1,
                                                    prefix_len=4, adapter_heads=4))
    head = df.head(16)
    batch = vocabs.encode(head)
    B = 16
    task_ids = torch.zeros(B, dtype=torch.long)
    instruction_ids = torch.randint(0, 64, (B, 4))
    target_ids = torch.tensor(
        [[_RISK[r]] for r in head["risk_label"]], dtype=torch.long
    )
    return {"dec": dec, "enc": enc, "llm": llm, "batch": batch, "B": B,
            "task_ids": task_ids, "instruction_ids": instruction_ids,
            "target_ids": target_ids}


# --------------------------------------------------------------------------- #
# Trainable trio: ψ subspaces + φ prefixes
# --------------------------------------------------------------------------- #

def test_task_embedding_shared_and_unique_subspaces():
    te = TaskEmbedding(n_tasks=3, d_llm=16, shared_dim=6)
    ids = torch.tensor([0, 1, 2])
    emb = te(ids)
    assert emb.shape == (3, 16)
    # the shared subspace (last 6 dims) is identical across tasks
    shared = emb[:, -6:]
    assert torch.allclose(shared[0], shared[1]) and torch.allclose(shared[1], shared[2])
    # the unique subspace differs across tasks (after init)
    assert not torch.allclose(emb[0, :10], emb[1, :10])


def test_prefix_encoder_shapes():
    pe = PrefixEncoder(num_layers=2, num_heads=4, head_dim=8, prefix_len=5)
    kv = pe(batch_size=3)
    assert len(kv) == 2
    for k, v in kv:
        assert k.shape == (3, 4, 5, 8) and v.shape == (3, 4, 5, 8)


# --------------------------------------------------------------------------- #
# Freeze invariant (handoff §0.3): only {Φ, ψ, φ} + sentinel train
# --------------------------------------------------------------------------- #

def test_encoder_and_llm_are_frozen(decoder_stack):
    dec, enc, llm = decoder_stack["dec"], decoder_stack["enc"], decoder_stack["llm"]
    dec.assert_frozen()
    assert all(not p.requires_grad for p in enc.parameters())
    assert all(not p.requires_grad for p in llm.parameters())


def test_only_adapter_task_prefix_sentinel_trainable(decoder_stack):
    dec = decoder_stack["dec"]
    trainable = {n for n, p in dec.named_parameters() if p.requires_grad}
    assert trainable, "expected some trainable params"
    for n in trainable:
        assert n.startswith(("adapter.", "task_embedding.", "prefix.", "row_sentinel")), n
    expected = sum(p.numel() for m in (dec.adapter, dec.task_embedding, dec.prefix)
                   for p in m.parameters()) + dec.row_sentinel.numel()
    assert dec.trainable_parameters() == expected


# --------------------------------------------------------------------------- #
# Eq. 5 interleaving + Eq. 6 objective
# --------------------------------------------------------------------------- #

def test_build_inputs_sequence_length(decoder_stack):
    d = decoder_stack
    z, mask = d["dec"].build_inputs(d["batch"], d["task_ids"], d["instruction_ids"])
    # 1 sentinel + 1 adapter token + 4 instruction + 1 task = 7
    assert z.shape == (d["B"], 7, d["llm"].hidden_size)
    assert mask.shape == (d["B"], 7)


def test_loss_is_finite_scalar(decoder_stack):
    d = decoder_stack
    loss = d["dec"](d["batch"], d["task_ids"], d["instruction_ids"], d["target_ids"])
    assert loss.ndim == 0 and torch.isfinite(loss)


def test_gradients_reach_trio_not_frozen_base(decoder_stack):
    d = decoder_stack
    dec = d["dec"]
    dec.zero_grad()
    loss = dec(d["batch"], d["task_ids"], d["instruction_ids"], d["target_ids"])
    loss.backward()
    # trio + sentinel receive gradient
    assert dec.row_sentinel.grad is not None
    assert any(p.grad is not None for p in dec.adapter.parameters())
    assert any(p.grad is not None for p in dec.prefix.parameters())
    assert dec.task_embedding.unique.weight.grad is not None
    # frozen base never does
    assert all(p.grad is None for p in dec.encoder.parameters())
    assert all(p.grad is None for p in dec.llm.parameters())
    dec.zero_grad()


def test_training_reduces_loss(request):
    schema = request.getfixturevalue("schema")
    df = request.getfixturevalue("sample_df")
    torch.manual_seed(0)
    enc, asm, vocabs = build_pretraining_stack(df, schema, TINY_ENC, party_epochs=1)
    llm = MockLLM(vocab_size=64, hidden=32, num_layers=2, num_heads=4)
    dec = MultimodalDecoder(enc, llm, DecoderConfig(n_tasks=3, prefix_len=4, adapter_heads=4))

    head = df.head(16)
    batch = vocabs.encode(head)
    task_ids = torch.zeros(16, dtype=torch.long)
    instr = torch.randint(0, 64, (16, 4))
    target = torch.tensor([[_RISK[r]] for r in head["risk_label"]], dtype=torch.long)

    opt = torch.optim.Adam([p for p in dec.parameters() if p.requires_grad], lr=1e-3)
    initial = dec(batch, task_ids, instr, target).item()
    for _ in range(80):
        loss = dec(batch, task_ids, instr, target)
        opt.zero_grad(); loss.backward(); opt.step()
    final = dec(batch, task_ids, instr, target).item()
    assert final < initial * 0.9, (initial, final)


def test_predict_proba_is_well_formed(decoder_stack):
    d = decoder_stack
    label_token_ids = [0, 1, 2]                  # the 3 risk-label answer tokens
    proba = d["dec"].predict_proba(d["batch"], d["task_ids"], d["instruction_ids"],
                                   label_token_ids)
    assert proba.shape == (d["B"], 3)
    torch.testing.assert_close(proba.sum(dim=1), torch.ones(d["B"]))
    assert (proba >= 0).all()


def test_prompt_mode_trains_via_inputs_embeds(request):
    # φ="prompt" must work with NO past_key_values (robust HF path): soft prompt
    # gets gradient, build_inputs grows by prefix_len, predict_proba well-formed.
    schema = request.getfixturevalue("schema")
    df = request.getfixturevalue("sample_df")
    torch.manual_seed(0)
    enc, asm, vocabs = build_pretraining_stack(df, schema, TINY_ENC, party_epochs=1)
    llm = MockLLM(vocab_size=64, hidden=32, num_layers=2, num_heads=4)
    dec = MultimodalDecoder(enc, llm, DecoderConfig(n_tasks=3, prefix_len=4,
                                                    adapter_heads=4, phi_mode="prompt"))
    assert dec.prefix is None and dec.soft_prompt is not None
    head = df.head(8)
    batch = vocabs.encode(head)
    task = torch.zeros(8, dtype=torch.long)
    instr = torch.randint(0, 64, (8, 4))
    tgt = torch.tensor([[_RISK[r]] for r in head["risk_label"]])

    z, _ = dec.build_inputs(batch, task, instr)
    assert z.shape[1] == 4 + 1 + 1 + 4 + 1     # soft(4)+sentinel+adapter+instr(4)+task

    dec.zero_grad()
    dec(batch, task, instr, tgt).backward()
    assert dec.phi_param() is dec.soft_prompt
    assert dec.soft_prompt.grad is not None and torch.any(dec.soft_prompt.grad != 0)
    proba = dec.predict_proba(batch, task, instr, [0, 1, 2])
    assert proba.shape == (8, 3)
    torch.testing.assert_close(proba.sum(dim=1), torch.ones(8))


def test_full_tune_unfreezes_llm_keeps_encoder_frozen(request):
    # C2 full-tune comparator: train_llm=True → LLM trainable, encoder STILL frozen.
    schema = request.getfixturevalue("schema")
    df = request.getfixturevalue("sample_df")
    torch.manual_seed(0)
    enc, asm, vocabs = build_pretraining_stack(df, schema, TINY_ENC, party_epochs=1)
    llm = MockLLM(vocab_size=64, hidden=32, num_layers=2, num_heads=4)
    dec = MultimodalDecoder(enc, llm, DecoderConfig(n_tasks=3, prefix_len=4,
                                                    adapter_heads=4, phi_mode="prompt",
                                                    train_llm=True))
    dec.assert_frozen()                                   # passes: encoder frozen, LLM allowed
    assert all(not p.requires_grad for p in dec.encoder.parameters())
    assert any(p.requires_grad for p in dec.llm.parameters())
    # full-tune trains far more params than the adapter-only trio
    adapter_only = sum(p.numel() for m in (dec.adapter, dec.task_embedding)
                       for p in m.parameters()) + dec.soft_prompt.numel()
    assert dec.trainable_parameters() > adapter_only + sum(
        p.numel() for p in dec.llm.parameters()) - 1

    head = df.head(8)
    batch = vocabs.encode(head)
    tgt = torch.tensor([[_RISK[r]] for r in head["risk_label"]])
    dec.zero_grad()
    dec(batch, torch.zeros(8, dtype=torch.long), torch.randint(0, 64, (8, 4)), tgt).backward()
    # gradient reaches the (now trainable) LLM, but never the frozen encoder
    assert any(p.grad is not None and torch.any(p.grad != 0) for p in dec.llm.parameters())
    assert all(p.grad is None for p in dec.encoder.parameters())
    dec.zero_grad()


def test_prefix_changes_llm_output():
    torch.manual_seed(0)
    llm = MockLLM(vocab_size=32, hidden=16, num_layers=2, num_heads=2)
    pe = PrefixEncoder(2, 2, 8, prefix_len=3)
    x = torch.randn(2, 5, 16)
    mask = torch.ones(2, 5)
    out_no = llm.forward_embeds(x, mask, prefixes=None)
    out_pf = llm.forward_embeds(x, mask, prefixes=pe(2))
    assert out_no.shape == (2, 5, 32)
    assert not torch.allclose(out_no, out_pf)   # φ prefix actually influences output
