"""Integration tests for the Layer 2 column assembler (Eq. 4 routing)."""

import numpy as np
import pytest
import torch

from encoders.column_assembler import (
    ColumnAssembler,
    build_party_matrix,
    build_vocabs,
)
from encoders.party_encoder import (
    PARTY_STRUCT_ATTRS,
    PartyEncoder,
    build_field_vocabs,
    build_party_store,
    encode_role_parties,
    party_roles_from_schema,
)
from encoders.quantizer import AdaptiveQuantizer

D = 32


@pytest.fixture(scope="module")
def assembled(request):
    """Build a full Layer-2 assembler from the committed sample + schema."""
    schema = request.getfixturevalue("schema")
    df = request.getfixturevalue("sample_df")

    vocabs = build_vocabs(df, schema)
    quantizer = AdaptiveQuantizer().fit(
        df[vocabs.numerical_col].to_numpy(), df[vocabs.ccy_col].to_numpy()
    )

    roles = party_roles_from_schema(schema)
    pvocabs = build_field_vocabs(df, roles)
    field_vocabs = {a: len(pvocabs[a]) for a in PARTY_STRUCT_ATTRS}
    torch.manual_seed(0)
    penc = PartyEncoder(field_vocabs, embedding_dim=D)  # untrained is fine for wiring
    store_rows = {}
    for r in roles.values():
        for k, row in zip(*encode_role_parties(df, r, pvocabs)):
            store_rows[k] = row
    keys = list(store_rows)
    store = build_party_store(penc, keys, torch.stack([store_rows[k] for k in keys]))

    asm = ColumnAssembler(schema, vocabs, quantizer, store, embedding_dim=D)
    return {"asm": asm, "vocabs": vocabs, "df": df, "store": store,
            "schema": schema, "roles": roles}


# --------------------------------------------------------------------------- #
# Token routing + sequence shape (Eq. 4)
# --------------------------------------------------------------------------- #

def test_token_order_follows_schema(assembled):
    asm, schema, roles = assembled["asm"], assembled["schema"], assembled["roles"]
    expected = (
        schema["buckets"]["high_card_categorical"]
        + schema["buckets"]["numerical"]
        + schema["buckets"]["core"]
        + [f"{p}__party" for p in roles]
    )
    assert asm.token_names == expected
    # 4 high-card + 1 numerical + 3 core + 2 party = 10
    assert asm.n_tokens == 10


def test_forward_shape(assembled):
    asm, vocabs, df = assembled["asm"], assembled["vocabs"], assembled["df"]
    batch = vocabs.encode(df.head(16))
    seq = asm(batch)
    assert seq.shape == (16, asm.n_tokens, D)
    assert seq.dtype == torch.float32


# --------------------------------------------------------------------------- #
# Frozen party summaries vs trainable field encoders
# --------------------------------------------------------------------------- #

def test_party_tokens_are_frozen(assembled):
    asm = assembled["asm"]
    for emb in asm.party_emb.values():
        assert all(not p.requires_grad for p in emb.parameters())


def test_field_encoders_are_trainable(assembled):
    asm = assembled["asm"]
    assert asm.trainable_parameters() > 0
    assert asm.frozen_parameters() > 0
    # high-card, core, and quantizer-level embeddings all carry grad
    for emb in asm.hc_emb.values():
        assert any(p.requires_grad for p in emb.parameters())
    for emb in asm.core_emb.values():
        assert any(p.requires_grad for p in emb.parameters())
    assert asm.amt_emb.emb.weight.requires_grad


def test_party_token_matches_store_lookup(assembled):
    asm, vocabs, df, store = (assembled[k] for k in ("asm", "vocabs", "df", "store"))
    roles = assembled["roles"]
    batch = vocabs.encode(df.head(4))
    seq = asm(batch)
    # locate a party token in the sequence and compare to the raw store lookup
    prefix = next(iter(roles))
    key_col = roles[prefix]["key"]
    tok_idx = asm.token_names.index(f"{prefix}__party")
    for row in range(4):
        acct_id = str(df.iloc[row][key_col])
        expected = torch.from_numpy(store.lookup(acct_id))
        assert torch.allclose(seq[row, tok_idx], expected, atol=1e-6)


def test_gradients_skip_party_but_reach_field_encoders(assembled):
    asm, vocabs, df = assembled["asm"], assembled["vocabs"], assembled["df"]
    batch = vocabs.encode(df.head(8))
    asm(batch).sum().backward()
    # a frozen party table never accumulates grad
    for emb in asm.party_emb.values():
        for p in emb.parameters():
            assert p.grad is None
    # a trainable core embedding does
    some_core = next(iter(asm.core_emb.values()))
    assert some_core.weight.grad is not None
    asm.zero_grad()


# --------------------------------------------------------------------------- #
# Vocab building + UNK handling
# --------------------------------------------------------------------------- #

def test_vocab_sizes_and_unk(assembled):
    vocabs, df, schema = assembled["vocabs"], assembled["df"], assembled["schema"]
    for col in schema["buckets"]["high_card_categorical"]:
        assert vocabs.hc_size(col) == df[col].astype(str).nunique() + 1   # + UNK
        assert len(vocabs.high_card_freq[col]) == vocabs.hc_size(col)


def test_unseen_value_maps_to_unk_without_crash(assembled):
    asm, vocabs, df, schema = (assembled[k] for k in ("asm", "vocabs", "df", "schema"))
    row = df.head(1).copy()
    hc0 = schema["buckets"]["high_card_categorical"][0]
    row[hc0] = "THIS_ID_WAS_NEVER_SEEN"
    batch = vocabs.encode(row)
    assert int(batch["high_card"][hc0][0]) == len(vocabs.high_card[hc0])  # UNK index
    seq = asm(batch)
    assert seq.shape == (1, asm.n_tokens, D)


# --------------------------------------------------------------------------- #
# Guardrails on construction
# --------------------------------------------------------------------------- #

def test_party_store_dim_mismatch_raises(assembled):
    vocabs, df, schema = assembled["vocabs"], assembled["df"], assembled["schema"]
    quantizer = AdaptiveQuantizer().fit(
        df[vocabs.numerical_col].to_numpy(), df[vocabs.ccy_col].to_numpy()
    )
    # build a store with the WRONG dim
    roles = party_roles_from_schema(schema)
    pvocabs = build_field_vocabs(df, roles)
    field_vocabs = {a: len(pvocabs[a]) for a in PARTY_STRUCT_ATTRS}
    penc = PartyEncoder(field_vocabs, embedding_dim=D + 16)
    r = next(iter(roles.values()))
    keys, ids = encode_role_parties(df, r, pvocabs)
    bad_store = build_party_store(penc, keys, ids)
    with pytest.raises(ValueError):
        ColumnAssembler(schema, vocabs, quantizer, bad_store, embedding_dim=D)
