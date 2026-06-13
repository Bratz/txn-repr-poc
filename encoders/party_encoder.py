"""
§3.2 Offline meta-column (party) encoder + persistent party-embedding store.

Grounded strictly to:
  Raman, Ganesh, Veloso (JPMorgan AI Research),
  "Scalable Representation Learning for Multimodal Tabular Transactions",
  arXiv:2410.07851, NeurIPS 2024 TRL workshop.

# PAPER: §3.2. Meta-columns (contextual, normalized-out attributes) are encoded
# OFFLINE by a separate function Ξ_g : x_g → R^{C_g × D}, with C_g < |x_g| so the
# result is a compact POOLED SUMMARY that occupies a single element of the
# column-embedding sequence (Eq. 4):
#     embedding(x) = (Ξ(x_1), …, Ξ_g(x_g), …, Ξ(x_C))
# The paper deliberately leaves the encoder architecture, the pooling op, and
# the training objective UNSPECIFIED.
#
# v1 decisions (architecture.md §7 — documented on purpose, the spot a reviewer
# will push):
#   * Objective = MASKED-ATTRIBUTE RECONSTRUCTION over the party block — mask one
#     attribute, reconstruct it from the rest. Methodologically consistent with
#     the Layer-3 masked-column loss (§3.4). # PAPER: §3.2 (objective unspecified)
#   * A "party" is its STRUCTURED categorical attributes {Ctry, Industry,
#     SubIndustry}. The *_Nm meta columns are party labels / identity, not
#     learnable attributes (free-text names are high-card identifiers already
#     served by the §3.1 partitioning embedder via the *_Id keys). So the store
#     is KEYED BY ACCOUNT ID and the encoder consumes the structured trio.
#   * Encoder = small Transformer over the per-attribute embedding sequence with
#     C_g learned summary tokens (CLS-style); C_g=1 by default (1 < 3). Sizes
#     (D, layers, heads) are POC choices, NOT paper-pinned.
#
# The party-embedding store (architecture.md §2/§5) is a first-class PERSISTENT,
# keyed lookup — counterparty intelligence amortized offline so that inference
# is a lookup, not a forward pass. Implemented here as PartyStore.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# Structured party attributes the §3.2 encoder consumes (de-prefixed from the
# meta_party bucket). Names are identity, not encoded attributes — see docstring.
PARTY_STRUCT_ATTRS = ("Ctry", "Industry", "SubIndustry")


# --------------------------------------------------------------------------- #
# Party encoder (Ξ_g)
# --------------------------------------------------------------------------- #

class PartyEncoder(nn.Module):
    """Small Transformer over a party's structured attributes → pooled summary.

    Args:
        field_vocabs: ordered {attribute_name: vocab_size}. The sequence of
            attributes is the dict's key order.
        embedding_dim: D (POC choice, not paper-pinned).
        n_summary: C_g, number of pooled summary tokens. Must be < #fields.
        n_layers, n_heads, ff_mult: small Transformer sizing (POC).
    """

    def __init__(
        self,
        field_vocabs: dict[str, int],
        embedding_dim: int = 64,
        n_summary: int = 1,
        n_layers: int = 2,
        n_heads: int = 2,
        ff_mult: int = 4,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.field_names = list(field_vocabs.keys())
        self.field_vocabs = dict(field_vocabs)
        self.n_fields = len(self.field_names)
        self.embedding_dim = int(embedding_dim)
        self.n_summary = int(n_summary)

        # PAPER: §3.2 constraint C_g < |x_g|.
        if not (1 <= self.n_summary < self.n_fields):
            raise ValueError(
                f"n_summary (C_g={self.n_summary}) must satisfy 1 ≤ C_g < "
                f"n_fields ({self.n_fields})"
            )
        if self.embedding_dim % n_heads != 0:
            raise ValueError(f"embedding_dim {self.embedding_dim} not divisible "
                             f"by n_heads {n_heads}")

        # Per-field embedding tables; the LAST row (index = vocab) is the [MASK]
        # token used by masked-attribute reconstruction.
        self.mask_index = {name: v for name, v in self.field_vocabs.items()}
        self.field_emb = nn.ModuleDict({
            name: nn.Embedding(v + 1, self.embedding_dim)
            for name, v in self.field_vocabs.items()
        })

        # C_g learned summary (CLS-style) query tokens, prepended to the sequence.
        self.summary_tokens = nn.Parameter(
            torch.randn(self.n_summary, self.embedding_dim) * 0.02
        )

        layer = nn.TransformerEncoderLayer(
            d_model=self.embedding_dim,
            nhead=n_heads,
            dim_feedforward=ff_mult * self.embedding_dim,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )
        self.transformer = nn.TransformerEncoder(layer, num_layers=n_layers)

        # Per-field reconstruction heads (masked-attribute objective).
        self.heads = nn.ModuleDict({
            name: nn.Linear(self.embedding_dim, v)
            for name, v in self.field_vocabs.items()
        })

    # -- core forward ----------------------------------------------------- #
    def forward(
        self,
        field_ids: torch.Tensor,            # (B, n_fields) long
        field_mask: Optional[torch.Tensor] = None,  # (B, n_fields) bool, True=mask
    ):
        B, nF = field_ids.shape
        assert nF == self.n_fields, (nF, self.n_fields)

        embs = []
        for fi, name in enumerate(self.field_names):
            ids = field_ids[:, fi]
            if field_mask is not None:
                ids = torch.where(
                    field_mask[:, fi],
                    torch.full_like(ids, self.mask_index[name]),
                    ids,
                )
            embs.append(self.field_emb[name](ids))      # (B, D)
        seq = torch.stack(embs, dim=1)                  # (B, nF, D)

        summary = self.summary_tokens.unsqueeze(0).expand(B, -1, -1)  # (B, Cg, D)
        x = torch.cat([summary, seq], dim=1)            # (B, Cg+nF, D)
        out = self.transformer(x)
        return out[:, :self.n_summary, :], out[:, self.n_summary:, :]

    # -- masked-attribute reconstruction objective (chosen, §7) ----------- #
    def masked_reconstruction_loss(self, field_ids: torch.Tensor) -> torch.Tensor:
        """Mask exactly one random attribute per row; reconstruct it.

        # PAPER: §3.2 objective is unspecified; this is the v1 choice.
        """
        device = field_ids.device
        B, nF = field_ids.shape
        masked_field = torch.randint(0, nF, (B,), device=device)
        field_mask = torch.zeros(B, nF, dtype=torch.bool, device=device)
        field_mask[torch.arange(B, device=device), masked_field] = True

        _, field_out = self.forward(field_ids, field_mask)
        masked_repr = field_out[torch.arange(B, device=device), masked_field]  # (B,D)

        total = field_ids.new_zeros((), dtype=torch.float)
        count = 0
        for fi, name in enumerate(self.field_names):
            sel = masked_field == fi
            if sel.any():
                logits = self.heads[name](masked_repr[sel])
                total = total + F.cross_entropy(
                    logits, field_ids[sel, fi], reduction="sum"
                )
                count += int(sel.sum())
        return total / max(count, 1)

    # -- pooled summary for the store (no masking) ------------------------ #
    @torch.no_grad()
    def encode(self, field_ids: torch.Tensor) -> torch.Tensor:
        """Pooled summary Ξ_g(x_g) ∈ R^{B × C_g × D} (eval, no masking)."""
        self.eval()
        summary, _ = self.forward(field_ids, field_mask=None)
        return summary


# --------------------------------------------------------------------------- #
# Party-embedding store (persistent, keyed lookup) — architecture.md §2/§5
# --------------------------------------------------------------------------- #

class PartyStore:
    """Keyed party → pooled-summary-vector lookup. Persistent, not a cache.

    Unknown keys return a configurable default (zeros) so online scoring of a
    never-before-seen counterparty degrades gracefully.
    """

    def __init__(self, dim: int, keys: Optional[list] = None,
                 vectors: Optional[np.ndarray] = None):
        self.dim = int(dim)
        self.keys: list = list(keys) if keys is not None else []
        self.vectors = (vectors if vectors is not None
                        else np.zeros((0, self.dim), dtype=np.float32))
        self._index = {k: i for i, k in enumerate(self.keys)}

    def __len__(self) -> int:
        return len(self.keys)

    def __contains__(self, key) -> bool:
        return key in self._index

    def lookup(self, key) -> np.ndarray:
        i = self._index.get(key)
        if i is None:
            return np.zeros(self.dim, dtype=np.float32)
        return self.vectors[i]

    def lookup_batch(self, keys) -> np.ndarray:
        return np.stack([self.lookup(k) for k in keys], axis=0)

    @classmethod
    def from_arrays(cls, keys, vectors: np.ndarray) -> "PartyStore":
        vectors = np.asarray(vectors, dtype=np.float32)
        return cls(dim=vectors.shape[1], keys=list(keys), vectors=vectors)

    def save(self, path: str):
        np.savez(path, keys=np.asarray(self.keys, dtype=object),
                 vectors=self.vectors, dim=np.asarray(self.dim))

    @classmethod
    def load(cls, path: str) -> "PartyStore":
        d = np.load(path, allow_pickle=True)
        return cls(dim=int(d["dim"]), keys=list(d["keys"]), vectors=d["vectors"])


def build_party_store(encoder: PartyEncoder, keys, field_ids: torch.Tensor) -> PartyStore:
    """Run the (frozen) encoder over unique parties and persist the summaries."""
    summ = encoder.encode(field_ids)               # (N, Cg, D)
    vecs = summ.reshape(summ.shape[0], -1).cpu().numpy().astype(np.float32)
    return PartyStore.from_arrays(list(keys), vecs)


# --------------------------------------------------------------------------- #
# Schema-driven party extraction (reads buckets from column_schema.json — §0.4)
# --------------------------------------------------------------------------- #

def party_roles_from_schema(schema: dict) -> dict[str, dict]:
    """Map each role prefix → {'key': key_col, 'attrs': {attr: column}}.

    Only roles that carry the full structured attribute trio are returned (the
    name-only ultimate parties are §3.1 identifiers, not structured parties).
    """
    meta = schema["buckets"]["meta_party"]
    high_card = set(schema["buckets"]["high_card_categorical"])

    grouped: dict[str, dict[str, str]] = {}
    for col in meta:
        if "_" not in col:
            continue
        prefix, attr = col.rsplit("_", 1)
        grouped.setdefault(prefix, {})[attr] = col

    roles: dict[str, dict] = {}
    for prefix, attrs in grouped.items():
        if not all(a in attrs for a in PARTY_STRUCT_ATTRS):
            continue
        key_col = next((c for c in (f"{prefix}Acct_Id", f"{prefix}_Id")
                        if c in high_card), None)
        if key_col is None:
            continue
        roles[prefix] = {
            "key": key_col,
            "attrs": {a: attrs[a] for a in PARTY_STRUCT_ATTRS},
        }
    return roles


def build_field_vocabs(df, roles: dict) -> dict[str, dict]:
    """Shared {attr: {value: idx}} over the union of all roles' columns."""
    vocabs: dict[str, dict] = {a: {} for a in PARTY_STRUCT_ATTRS}
    for role in roles.values():
        for attr, col in role["attrs"].items():
            for v in df[col].astype(str).unique():
                if v not in vocabs[attr]:
                    vocabs[attr][v] = len(vocabs[attr])
    return vocabs


def encode_role_parties(df, role: dict, vocabs: dict):
    """Dedup a role's parties by key → (keys list, field_ids LongTensor)."""
    cols = {a: role["attrs"][a] for a in PARTY_STRUCT_ATTRS}
    sub = df[[role["key"], *cols.values()]].drop_duplicates(subset=role["key"])
    keys = sub[role["key"]].tolist()
    ids = np.stack(
        [sub[cols[a]].astype(str).map(vocabs[a]).to_numpy() for a in PARTY_STRUCT_ATTRS],
        axis=1,
    ).astype(np.int64)
    return keys, torch.from_numpy(ids)


# --------------------------------------------------------------------------- #
# CLI: train the party encoder offline and build the store from realized data
# --------------------------------------------------------------------------- #

def main():
    import argparse
    import json
    from pathlib import Path

    ap = argparse.ArgumentParser(description="§3.2 party encoder — offline pretrain + store")
    root = Path(__file__).resolve().parents[1]
    ap.add_argument("--schema", default=str(root / "data" / "column_schema.json"))
    ap.add_argument("--data", default=str(root / "data" / "pacs008_synth.parquet"))
    ap.add_argument("--dim", type=int, default=64)
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--batch", type=int, default=512)
    ap.add_argument("--store-out", default=str(root / "party_store" / "party_store.npz"))
    args = ap.parse_args()

    import pandas as pd

    schema = json.loads(Path(args.schema).read_text())
    roles = party_roles_from_schema(schema)

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

    vocabs = build_field_vocabs(df, roles)
    field_vocabs = {a: len(vocabs[a]) for a in PARTY_STRUCT_ATTRS}
    print(f"Party encoder (sec 3.2) on {src}: roles={list(roles)} "
          f"attrs={field_vocabs} dim={args.dim}")

    # Training set = unique parties pooled across all roles.
    all_keys, all_ids = [], []
    for role in roles.values():
        k, ids = encode_role_parties(df, role, vocabs)
        all_keys += k
        all_ids.append(ids)
    train_ids = torch.cat(all_ids, dim=0)
    print(f"unique parties (train rows): {train_ids.shape[0]:,}")

    torch.manual_seed(0)
    enc = PartyEncoder(field_vocabs, embedding_dim=args.dim)
    opt = torch.optim.Adam(enc.parameters(), lr=1e-3)
    n = train_ids.shape[0]
    for ep in range(args.epochs):
        enc.train()
        perm = torch.randperm(n)
        tot, nb = 0.0, 0
        for s in range(0, n, args.batch):
            batch = train_ids[perm[s:s + args.batch]]
            loss = enc.masked_reconstruction_loss(batch)
            opt.zero_grad(); loss.backward(); opt.step()
            tot += loss.item(); nb += 1
        print(f"  epoch {ep+1}/{args.epochs}  masked-recon loss {tot/nb:.4f}")

    # Build + persist the store keyed by account id (dedup across roles).
    store_ids, store_keys = {}, []
    for role in roles.values():
        k, ids = encode_role_parties(df, role, vocabs)
        for kk, row in zip(k, ids):
            store_ids[kk] = row            # last wins; keys unique per account
    keys = list(store_ids.keys())
    field_ids = torch.stack([store_ids[k] for k in keys], dim=0)
    store = build_party_store(enc, keys, field_ids)

    Path(args.store_out).parent.mkdir(parents=True, exist_ok=True)
    store.save(args.store_out)
    print(f"party store: {len(store):,} parties, dim={store.dim} -> {args.store_out}")
    print(f"sample lookup {keys[0]!r}: vec[:5]={store.lookup(keys[0])[:5]}")


if __name__ == "__main__":
    main()
