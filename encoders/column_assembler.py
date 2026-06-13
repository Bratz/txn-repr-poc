"""
Layer 2 column assembler — route each column to its field-encoder path and
concatenate into the tabular encoder's column-embedding sequence.

Grounded strictly to:
  Raman, Ganesh, Veloso (JPMorgan AI Research),
  "Scalable Representation Learning for Multimodal Tabular Transactions",
  arXiv:2410.07851, NeurIPS 2024 TRL workshop.

# PAPER: §3 Eq. 4. The encoder input is the per-column embedding sequence
#     embedding(x) = (Ξ(x_1), …, Ξ_g(x_g), …, Ξ(x_C))
# where each high-cardinality / numerical / core column contributes its own
# token and the meta-column block is replaced by the pooled party SUMMARY
# Ξ_g(x_g). This module implements that routing and concatenation; the routing
# is driven entirely by column_schema.json (handoff §0.4 — never hard-code
# column lists downstream).
#
# Routing (architecture.md §4):
#   high_card_categorical → §3.1 PartitioningEmbedder   (one per column)
#   numerical             → §3.3 currency-conditioned quantizer + level embed
#   core                  → standard inline nn.Embedding (one per column)
#   meta_party            → §3.2 party-summary token, a FROZEN store lookup
#
# Design choices (kept paper-literal):
#   * Per-column high-card embedders (NOT a merged debtor/creditor account
#     vocab). Merging same-entity-across-columns would be a modelling extension
#     beyond §3.1; per-column is the literal reading. (Sharing → v2.)
#   * The party encoder is pre-learned OFFLINE (2c) and FROZEN here: party
#     summaries are constant lookups from the store, indexed by the role's
#     account-id column. Only the Layer-2 field encoders (high-card / quantizer
#     level / core) train during Layer-3 pretraining. # PAPER: §3.2 offline.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn

from .partitioning_embedder import PartitioningEmbedder
from .party_encoder import PartyStore, party_roles_from_schema
from .quantizer import AdaptiveQuantizer, make_quantizer_embedder


# --------------------------------------------------------------------------- #
# Column vocabularies (string → int label encoders, built from data)
# --------------------------------------------------------------------------- #

@dataclass
class ColumnVocabs:
    """Per-column value→index maps. Index `len(map)` is the reserved UNK slot."""
    high_card: dict[str, dict]        # col → {value: idx}
    high_card_freq: dict[str, np.ndarray]  # col → counts aligned to idx (UNK=0)
    core: dict[str, dict]             # col → {value: idx}
    numerical_col: str
    ccy_col: str

    def hc_size(self, col: str) -> int:
        return len(self.high_card[col]) + 1   # + UNK

    def core_size(self, col: str) -> int:
        return len(self.core[col]) + 1        # + UNK

    def encode(self, df) -> dict:
        """Project a DataFrame into the tensors/arrays ColumnAssembler.forward needs."""
        def enc(col, vocab):
            unk = len(vocab)
            mapped = df[col].astype(str).map(lambda v: vocab.get(v, unk))
            return torch.as_tensor(mapped.to_numpy(), dtype=torch.long)

        return {
            "high_card": {c: enc(c, self.high_card[c]) for c in self.high_card},
            "core": {c: enc(c, self.core[c]) for c in self.core},
            "amount": df[self.numerical_col].to_numpy(dtype=np.float64),
            "ccy": df[self.ccy_col].astype(str).to_numpy(),
        }


def build_vocabs(df, schema: dict) -> ColumnVocabs:
    """Build per-column label encoders from data; buckets read from schema (§0.4)."""
    high_card, high_card_freq = {}, {}
    for col in schema["buckets"]["high_card_categorical"]:
        vc = df[col].astype(str).value_counts()        # descending frequency
        high_card[col] = {v: i for i, v in enumerate(vc.index)}
        freq = np.zeros(len(vc) + 1, dtype=np.float64)  # + UNK (freq 0)
        freq[: len(vc)] = vc.to_numpy()
        high_card_freq[col] = freq

    core = {}
    for col in schema["buckets"]["core"]:
        core[col] = {v: i for i, v in enumerate(sorted(df[col].astype(str).unique()))}

    numerical_col = schema["buckets"]["numerical"][0]
    ccy_col = "Ccy" if "Ccy" in schema["buckets"]["core"] else schema["buckets"]["core"][0]
    return ColumnVocabs(high_card, high_card_freq, core, numerical_col, ccy_col)


def build_party_matrix(store: PartyStore, vocab: dict) -> torch.Tensor:
    """Align store vectors to a role's account-id int vocab (UNK row = zeros)."""
    mat = np.zeros((len(vocab) + 1, store.dim), dtype=np.float32)
    for s, i in vocab.items():
        mat[i] = store.lookup(s)
    return torch.from_numpy(mat)


# --------------------------------------------------------------------------- #
# Column assembler (Eq. 4)
# --------------------------------------------------------------------------- #

class ColumnAssembler(nn.Module):
    """Assemble the per-column embedding sequence (B, n_tokens, D) for Layer 3."""

    def __init__(
        self,
        schema: dict,
        vocabs: ColumnVocabs,
        quantizer: AdaptiveQuantizer,
        party_store: PartyStore,
        embedding_dim: int = 128,
        partition_kwargs: dict | None = None,
    ):
        super().__init__()
        self.embedding_dim = D = int(embedding_dim)
        self.vocabs = vocabs
        pk = partition_kwargs or {}

        # Routes read from the schema (never hard-coded — §0.4).
        self.high_card_cols = list(schema["buckets"]["high_card_categorical"])
        self.numerical_cols = list(schema["buckets"]["numerical"])
        self.core_cols = list(schema["buckets"]["core"])
        if len(self.numerical_cols) != 1:
            raise NotImplementedError(
                "v1 assumes a single numerical column (the quantizer is fit for it)"
            )

        # §3.1 high-card embedders — one per column, frequency-aware bin assignment.
        self.hc_emb = nn.ModuleDict({
            col: PartitioningEmbedder(
                vocabs.hc_size(col), D,
                token_frequencies=vocabs.high_card_freq[col], **pk,
            )
            for col in self.high_card_cols
        })

        # core inline embeddings.
        self.core_emb = nn.ModuleDict({
            col: nn.Embedding(vocabs.core_size(col), D) for col in self.core_cols
        })

        # §3.3 numerical: currency-conditioned quantizer + learnable level embedding.
        self.amt_emb = make_quantizer_embedder(quantizer, D)

        # §3.2 party summaries: FROZEN store lookups per structured role, indexed
        # by that role's account-id column int vocab.
        self.party_roles = party_roles_from_schema(schema)
        if party_store.dim != D:
            raise ValueError(
                f"party_store.dim={party_store.dim} != embedding_dim={D}; rebuild "
                f"the store with C_g*D == embedding_dim"
            )
        self.party_emb = nn.ModuleDict({
            prefix: nn.Embedding.from_pretrained(
                build_party_matrix(party_store, vocabs.high_card[spec["key"]]),
                freeze=True,
            )
            for prefix, spec in self.party_roles.items()
        })

        # Token order (Eq. 4): per-column embeddings, meta block → party summaries.
        self.token_names = (
            list(self.high_card_cols)
            + list(self.numerical_cols)
            + list(self.core_cols)
            + [f"{p}__party" for p in self.party_roles]
        )

    @property
    def n_tokens(self) -> int:
        return len(self.token_names)

    def forward(self, batch: dict) -> torch.Tensor:
        toks = []
        for col in self.high_card_cols:
            toks.append(self.hc_emb[col](batch["high_card"][col]))
        for _ in self.numerical_cols:
            toks.append(self.amt_emb(batch["amount"], batch["ccy"]))
        for col in self.core_cols:
            toks.append(self.core_emb[col](batch["core"][col]))
        for prefix, spec in self.party_roles.items():
            toks.append(self.party_emb[prefix](batch["high_card"][spec["key"]]))
        return torch.stack(toks, dim=1)   # (B, n_tokens, D)

    def trainable_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def frozen_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if not p.requires_grad)


# --------------------------------------------------------------------------- #
# CLI: wire all of Layer 2 end-to-end on the realized data (smoke test)
# --------------------------------------------------------------------------- #

def main():
    import argparse
    import json
    from pathlib import Path

    import pandas as pd

    from .party_encoder import (
        PartyEncoder,
        build_field_vocabs,
        build_party_store,
        encode_role_parties,
        PARTY_STRUCT_ATTRS,
    )

    ap = argparse.ArgumentParser(description="Layer 2 column assembler smoke test")
    root = Path(__file__).resolve().parents[1]
    ap.add_argument("--schema", default=str(root / "data" / "column_schema.json"))
    ap.add_argument("--data", default=str(root / "data" / "pacs008_synth.parquet"))
    ap.add_argument("--dim", type=int, default=64)
    ap.add_argument("--party-epochs", type=int, default=3)
    ap.add_argument("--batch", type=int, default=8)
    args = ap.parse_args()

    schema_path = Path(args.schema)
    if not schema_path.exists():
        schema_path = root / "data" / "column_schema.example.json"
    schema = json.loads(schema_path.read_text())

    path = Path(args.data)
    df = None
    if path.exists():
        try:
            df = pd.read_parquet(path) if path.suffix == ".parquet" else pd.read_csv(path)
            src = path.name
        except Exception as e:
            print(f"(could not read {path.name}: {e}; using reference sample)")
    if df is None:
        df = pd.read_csv(root / "data" / "pacs008_sample_500.csv")
        src = "pacs008_sample_500.csv (fallback)"

    D = args.dim
    vocabs = build_vocabs(df, schema)
    quantizer = AdaptiveQuantizer().fit(df[vocabs.numerical_col].to_numpy(),
                                        df[vocabs.ccy_col].to_numpy())

    # Offline party encoder + store at C_g*D == D so the summary token is D-dim.
    roles = party_roles_from_schema(schema)
    pvocabs = build_field_vocabs(df, roles)
    field_vocabs = {a: len(pvocabs[a]) for a in PARTY_STRUCT_ATTRS}
    torch.manual_seed(0)
    penc = PartyEncoder(field_vocabs, embedding_dim=D)
    train_ids = torch.cat([encode_role_parties(df, r, pvocabs)[1] for r in roles.values()])
    opt = torch.optim.Adam(penc.parameters(), lr=1e-3)
    for _ in range(args.party_epochs):
        loss = penc.masked_reconstruction_loss(train_ids)
        opt.zero_grad(); loss.backward(); opt.step()
    store_rows = {}
    for r in roles.values():
        for k, row in zip(*encode_role_parties(df, r, pvocabs)):
            store_rows[k] = row
    keys = list(store_rows)
    store = build_party_store(penc, keys, torch.stack([store_rows[k] for k in keys]))

    assembler = ColumnAssembler(schema, vocabs, quantizer, store, embedding_dim=D)
    batch = vocabs.encode(df.head(args.batch))
    seq = assembler(batch)

    print(f"Column assembler on {src} (D={D})")
    print(f"tokens ({assembler.n_tokens}): {assembler.token_names}")
    print(f"sequence shape: {tuple(seq.shape)}  (batch, tokens, D)")
    print(f"trainable params: {assembler.trainable_parameters():,}  "
          f"frozen (party) params: {assembler.frozen_parameters():,}")


if __name__ == "__main__":
    main()
