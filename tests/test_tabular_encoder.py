"""Unit tests for the Layer 3 tabular encoder (§3.4) + composite loss."""

import pytest
import torch

from encoder.tabular_encoder import (
    EncoderConfig,
    batch_hard_triplet_loss,
    build_pretraining_stack,
)

TINY = EncoderConfig(hidden=32, layers=2, heads=2, ff_mult=2, dropout=0.0,
                     mask_prob=0.5, triplet_margin=1.0, triplet_weight=1.0)


@pytest.fixture(scope="module")
def stack(request):
    schema = request.getfixturevalue("schema")
    df = request.getfixturevalue("sample_df")
    torch.manual_seed(0)
    enc, asm, vocabs = build_pretraining_stack(df, schema, TINY, party_epochs=1)
    batch = vocabs.encode(df.head(32))
    return {"enc": enc, "asm": asm, "vocabs": vocabs, "batch": batch, "df": df,
            "schema": schema}


# --------------------------------------------------------------------------- #
# Batch-hard triplet loss
# --------------------------------------------------------------------------- #

def test_triplet_zero_when_separated():
    emb = torch.tensor([[0., 0], [9, 0], [0, 0], [9, 0]])
    lab = torch.tensor([0, 1, 0, 1])
    assert float(batch_hard_triplet_loss(emb, lab, margin=1.0)) == 0.0


def test_triplet_equals_margin_when_collapsed():
    # All points identical → d_pos = d_neg = 0 → loss = margin.
    emb = torch.zeros(6, 4)
    lab = torch.tensor([0, 1, 2, 0, 1, 2])
    assert float(batch_hard_triplet_loss(emb, lab, margin=1.0)) == pytest.approx(1.0)


# --------------------------------------------------------------------------- #
# Shapes + forward
# --------------------------------------------------------------------------- #

def test_encode_and_forward_shapes(stack):
    enc, batch = stack["enc"], stack["batch"]
    B = batch["amount"].shape[0]
    row = enc.encode(batch)
    assert row.shape == (B, enc.D)
    row2, out = enc(batch, column_mask=None)
    assert row2.shape == (B, enc.D)
    assert out.shape == (B, enc.n_tokens + 1, enc.D)   # +1 for [CLS]


def test_reconstructable_excludes_party(stack):
    enc, schema = stack["enc"], stack["schema"]
    names = [r[0] for r in enc.recon]
    # high-card + numerical + core are reconstructable; party summaries are not
    assert set(names) == set(
        schema["buckets"]["high_card_categorical"]
        + schema["buckets"]["numerical"]
        + schema["buckets"]["core"]
    )
    assert not any(n.endswith("__party") for n in names)


def test_column_mask_has_at_least_one_per_row(stack):
    enc = stack["enc"]
    m = enc.sample_column_mask(50, torch.device("cpu"))
    assert m.shape == (50, len(enc.recon))
    assert bool(m.any(dim=1).all())


# --------------------------------------------------------------------------- #
# Composite loss + training
# --------------------------------------------------------------------------- #

def test_composite_loss_components_finite(stack):
    enc, batch = stack["enc"], stack["batch"]
    loss, parts = enc.composite_loss(batch)
    assert torch.isfinite(loss)
    assert parts["recon"] >= 0 and parts["triplet"] >= 0


def test_training_reduces_composite_loss(request):
    schema = request.getfixturevalue("schema")
    df = request.getfixturevalue("sample_df")
    torch.manual_seed(0)
    enc, asm, vocabs = build_pretraining_stack(df, schema, TINY, party_epochs=1)
    batch = vocabs.encode(df.head(64))
    opt = torch.optim.Adam(enc.parameters(), lr=1e-3)

    enc.train()
    initial = enc.composite_loss(batch)[0].item()
    for _ in range(60):
        loss, _ = enc.composite_loss(batch)
        opt.zero_grad(); loss.backward(); opt.step()
    final = enc.composite_loss(batch)[0].item()
    assert final < initial, (initial, final)


def test_reconstruction_accuracy_in_range(stack):
    enc, batch = stack["enc"], stack["batch"]
    acc = enc.reconstruction_accuracy(batch)
    assert len(acc) == len(enc.recon)
    for name, row in acc.items():
        assert 0.0 <= row["top1"] <= 1.0
        assert 0.0 <= row["top3"] <= 1.0
        assert row["top3"] >= row["top1"]


# --------------------------------------------------------------------------- #
# Freeze invariant (handoff §0.3)
# --------------------------------------------------------------------------- #

def test_freeze_disables_all_grads(stack):
    enc = stack["enc"]
    enc.freeze()
    assert enc.num_trainable_parameters() == 0
    assert all(not p.requires_grad for p in enc.parameters())


# --------------------------------------------------------------------------- #
# Partitioned vs classical encoder both run (C1 plumbing)
# --------------------------------------------------------------------------- #

def test_partitioned_vs_classical_param_gap(request):
    schema = request.getfixturevalue("schema")
    df = request.getfixturevalue("sample_df")
    p_enc, p_asm, _ = build_pretraining_stack(df, schema, TINY, party_epochs=1,
                                              high_card_embedder="partitioned")
    c_enc, c_asm, _ = build_pretraining_stack(df, schema, TINY, party_epochs=1,
                                              high_card_embedder="classical")
    p_hc = sum(e.num_embedding_parameters() for e in p_asm.hc_emb.values())
    c_hc = sum(e.num_embedding_parameters() for e in c_asm.hc_emb.values())
    assert p_hc < c_hc                       # partitioned uses fewer high-card params
    # both encoders produce reconstruction accuracy without error
    batch = p_asm.vocabs.encode(df.head(16))
    assert p_enc.reconstruction_accuracy(batch)
