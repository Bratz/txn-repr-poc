"""Unit tests for the §3.2 party encoder + party-embedding store."""

import numpy as np
import pytest
import torch

from encoders.party_encoder import (
    PARTY_STRUCT_ATTRS,
    PartyEncoder,
    PartyStore,
    build_party_store,
    party_roles_from_schema,
)


def _toy_encoder(dim=64):
    return PartyEncoder({"Ctry": 10, "Industry": 6, "SubIndustry": 18},
                        embedding_dim=dim)


def _learnable_parties(n=1024, seed=0):
    """SubIndustry deterministically implies Industry; Country is independent.

    So masking Industry is reconstructable from SubIndustry → the objective has
    real signal and the loss must fall.
    """
    g = torch.Generator().manual_seed(seed)
    sub = torch.randint(0, 18, (n,), generator=g)
    industry = sub // 3                      # 18 sub-industries → 6 industries
    ctry = torch.randint(0, 10, (n,), generator=g)
    return torch.stack([ctry, industry, sub], dim=1)  # order: Ctry, Industry, SubIndustry


# --------------------------------------------------------------------------- #
# C_g < |x_g| constraint (§3.2)
# --------------------------------------------------------------------------- #

def test_summary_count_must_be_less_than_fields():
    with pytest.raises(ValueError):
        PartyEncoder({"Ctry": 10, "Industry": 6, "SubIndustry": 18}, n_summary=3)
    # C_g = 2 < 3 is allowed
    PartyEncoder({"Ctry": 10, "Industry": 6, "SubIndustry": 18}, n_summary=2)


def test_embedding_dim_divisible_by_heads():
    with pytest.raises(ValueError):
        PartyEncoder({"Ctry": 10, "Industry": 6, "SubIndustry": 18},
                     embedding_dim=65, n_heads=2)


# --------------------------------------------------------------------------- #
# Forward shapes + pooled summary
# --------------------------------------------------------------------------- #

def test_forward_shapes():
    enc = _toy_encoder(dim=32)
    ids = _learnable_parties(16)
    summary, field_out = enc(ids)
    assert summary.shape == (16, 1, 32)        # (B, C_g, D)
    assert field_out.shape == (16, 3, 32)      # (B, n_fields, D)


def test_encode_is_deterministic_in_eval():
    enc = _toy_encoder()
    ids = _learnable_parties(8)
    a = enc.encode(ids)
    b = enc.encode(ids)
    assert torch.allclose(a, b)
    assert a.shape == (8, 1, enc.embedding_dim)


def test_masking_changes_output():
    enc = _toy_encoder()
    ids = _learnable_parties(8)
    mask = torch.zeros(8, 3, dtype=torch.bool)
    mask[:, 1] = True                          # mask Industry
    _, out_plain = enc(ids)
    _, out_masked = enc(ids, mask)
    assert not torch.allclose(out_plain, out_masked)


# --------------------------------------------------------------------------- #
# Masked-attribute reconstruction objective actually learns
# --------------------------------------------------------------------------- #

def test_reconstruction_loss_is_finite_scalar():
    enc = _toy_encoder()
    loss = enc.masked_reconstruction_loss(_learnable_parties(64))
    assert loss.ndim == 0 and torch.isfinite(loss)


def test_training_reduces_reconstruction_loss():
    torch.manual_seed(0)
    enc = _toy_encoder(dim=32)
    data = _learnable_parties(1024)
    opt = torch.optim.Adam(enc.parameters(), lr=1e-3)

    enc.train()
    initial = enc.masked_reconstruction_loss(data).item()
    for _ in range(150):
        loss = enc.masked_reconstruction_loss(data)
        opt.zero_grad(); loss.backward(); opt.step()
    final = enc.masked_reconstruction_loss(data).item()
    assert final < initial * 0.9, (initial, final)


# --------------------------------------------------------------------------- #
# Party store (persistent keyed lookup)
# --------------------------------------------------------------------------- #

def test_build_lookup_and_unknown_default():
    enc = _toy_encoder(dim=16)
    keys = [f"ACC{i}" for i in range(20)]
    store = build_party_store(enc, keys, _learnable_parties(20))
    assert len(store) == 20
    assert store.dim == 16                      # C_g=1 → D
    assert store.lookup("ACC3").shape == (16,)
    assert "ACC3" in store and "NOPE" not in store
    np.testing.assert_array_equal(store.lookup("NOPE"), np.zeros(16, np.float32))


def test_store_save_load_roundtrip(tmp_path):
    enc = _toy_encoder(dim=16)
    keys = [f"ACC{i}" for i in range(10)]
    store = build_party_store(enc, keys, _learnable_parties(10))
    p = str(tmp_path / "store.npz")
    store.save(p)
    loaded = PartyStore.load(p)
    assert len(loaded) == len(store) and loaded.dim == store.dim
    for k in keys:
        np.testing.assert_allclose(loaded.lookup(k), store.lookup(k), rtol=1e-6)


# --------------------------------------------------------------------------- #
# Schema-driven role extraction (reads buckets from column_schema.json)
# --------------------------------------------------------------------------- #

def test_roles_from_schema_picks_structured_roles_only(schema):
    roles = party_roles_from_schema(schema)
    # Dbtr and Cdtr carry the full {Ctry, Industry, SubIndustry} trio + an acct key.
    assert set(roles) == {"Dbtr", "Cdtr"}
    for prefix, spec in roles.items():
        assert spec["key"] == f"{prefix}Acct_Id"
        assert set(spec["attrs"]) == set(PARTY_STRUCT_ATTRS)
    # Name-only ultimate parties are excluded (handled by §3.1 ids).
    assert "UltmtDbtr" not in roles and "UltmtCdtr" not in roles
