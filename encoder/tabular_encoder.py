"""
Layer 3 tabular encoder (§3.4) — bidirectional BERT over the assembled column
sequence, trained with a composite (masked-column reconstruction + batch-hard
triplet) loss. Frozen after pretraining for Layer 4.

Grounded strictly to:
  Raman, Ganesh, Veloso (JPMorgan AI Research),
  "Scalable Representation Learning for Multimodal Tabular Transactions",
  arXiv:2410.07851, NeurIPS 2024 TRL workshop.

# PAPER: §3.4. "A bi-directional transformer with column masking in the standard
# BERT configuration." A [CLS] token's output is the row embedding f(x). The
# composite objective adds, to masked-column reconstruction, a BATCH-HARD TRIPLET
# loss [Hermans et al.] that pulls together two perturbed views of the same row
# and pushes apart other rows — "the triplet term is the claim under test"
# (handoff): do NOT substitute a generic contrastive loss.
#
# The paper leaves several knobs unspecified; v1 choices (documented, §7-style):
#   * Perturbation for the two positive views = independent random COLUMN MASKING
#     (reuses the BERT masking mechanism). # PAPER: §3.4 (perturbation unspecified)
#   * Composite combine = L_recon + λ·L_triplet, λ=1.0.  (weights unspecified)
#   * Triplet margin = 1.0.                              (margin unspecified)
#   * Backbone shape (layers/heads/hidden) is chosen to hit the PINNED ~25M
#     params (handoff §0.2) — the paper fixes the size, not the shape.
#
# Reconstructed columns = the high-card / numerical / core field tokens (the
# party-summary tokens are FROZEN §3.2 lookups and serve as context only).
# Reconstruction accuracy on the masked high-card columns, partitioned vs
# classical embedder, is the C1 accuracy comparison.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from encoders.column_assembler import ColumnAssembler


@dataclass
class EncoderConfig:
    # Backbone shape chosen to hit the pinned ~25M params (paper fixes size, not
    # shape). hidden == column embedding dim D.
    hidden: int = 512
    layers: int = 8
    heads: int = 8
    ff_mult: int = 4
    dropout: float = 0.1
    # Objective knobs (paper-unspecified — v1 choices).
    mask_prob: float = 0.15          # per-reconstructable-column masking prob
    triplet_margin: float = 1.0
    triplet_weight: float = 1.0      # λ in L_recon + λ·L_triplet
    # Pinned training schedule (handoff §0.2; lr from paper §5.2).
    epochs: int = 3
    lr: float = 1e-4


# --------------------------------------------------------------------------- #
# Batch-hard triplet loss (§3.4)
# --------------------------------------------------------------------------- #

def batch_hard_triplet_loss(
    embeddings: torch.Tensor, labels: torch.Tensor, margin: float = 1.0
) -> torch.Tensor:
    """Batch-hard triplet [Hermans et al.] over L2 distances.

    For every anchor: hardest (farthest) positive vs hardest (closest) negative,
    loss = relu(margin + d_pos − d_neg). `labels` mark which rows are the same
    (two views of a row share a label).
    """
    dist = torch.cdist(embeddings, embeddings)             # (N, N) euclidean
    same = labels.unsqueeze(0) == labels.unsqueeze(1)
    eye = torch.eye(len(labels), dtype=torch.bool, device=labels.device)
    pos_mask = same & ~eye
    neg_mask = ~same

    hardest_pos = dist.masked_fill(~pos_mask, float("-inf")).max(dim=1).values
    hardest_neg = dist.masked_fill(~neg_mask, float("inf")).min(dim=1).values
    valid = torch.isfinite(hardest_pos) & torch.isfinite(hardest_neg)
    if not valid.any():
        return embeddings.new_zeros(())
    loss = torch.relu(margin + hardest_pos[valid] - hardest_neg[valid])
    return loss.mean()


# --------------------------------------------------------------------------- #
# Tabular encoder
# --------------------------------------------------------------------------- #

class TabularEncoder(nn.Module):
    """BERT over the assembled column sequence; composite-loss pretraining."""

    def __init__(self, assembler: ColumnAssembler, config: EncoderConfig):
        super().__init__()
        D = assembler.embedding_dim
        if D != config.hidden:
            raise ValueError(
                f"assembler embedding_dim ({D}) must equal config.hidden "
                f"({config.hidden})"
            )
        self.assembler = assembler
        self.config = config
        self.D = D

        # Which tokens are reconstructable (everything except frozen party summaries).
        party_names = {f"{p}__party" for p in assembler.party_roles}
        self.recon = []   # list of (name, base_pos, out_pos, vocab_size)
        for base_pos, name in enumerate(assembler.token_names):
            if name in party_names:
                continue
            if name in assembler.high_card_cols:
                vocab = assembler.vocabs.hc_size(name)
            elif name in assembler.core_cols:
                vocab = assembler.vocabs.core_size(name)
            else:  # the single numerical column → quantizer level vocab
                vocab = assembler.amt_emb.quantizer.num_levels
            self.recon.append((name, base_pos, base_pos + 1, vocab))

        self.n_tokens = assembler.n_tokens
        self.cls = nn.Parameter(torch.randn(1, 1, D) * 0.02)
        self.mask_emb = nn.Parameter(torch.randn(D) * 0.02)
        self.pos = nn.Parameter(torch.randn(1, self.n_tokens + 1, D) * 0.02)

        layer = nn.TransformerEncoderLayer(
            d_model=D, nhead=config.heads, dim_feedforward=config.ff_mult * D,
            dropout=config.dropout, batch_first=True, activation="gelu",
        )
        self.transformer = nn.TransformerEncoder(layer, num_layers=config.layers)
        self.heads = nn.ModuleDict(
            {name: nn.Linear(D, vocab) for name, _, _, vocab in self.recon}
        )

    # -- targets for the reconstructable columns -------------------------- #
    def recon_targets(self, batch: dict) -> dict:
        # All targets on the encoder's device — the numerical level is built on
        # CPU from the quantizer, so it MUST be moved or it mismatches the GPU
        # mask used to index it (a CPU-only smoke test cannot catch this).
        device = self.cls.device
        t = {}
        for name, *_ in self.recon:
            if name in self.assembler.high_card_cols:
                t[name] = batch["high_card"][name].to(device)
            elif name in self.assembler.core_cols:
                t[name] = batch["core"][name].to(device)
            else:
                lvl = self.assembler.amt_emb.quantizer.transform(
                    batch["amount"], batch["ccy"]
                )
                t[name] = torch.as_tensor(lvl, dtype=torch.long, device=device)
        return t

    def sample_column_mask(self, B: int, device) -> torch.Tensor:
        """(B, n_recon) bool mask; guarantee ≥1 masked column per row."""
        n = len(self.recon)
        m = torch.rand(B, n, device=device) < self.config.mask_prob
        empty = ~m.any(dim=1)
        if empty.any():
            forced = torch.randint(0, n, (int(empty.sum()),), device=device)
            m[empty, forced] = True
        return m

    # -- forward ---------------------------------------------------------- #
    def forward(self, batch: dict, column_mask: Optional[torch.Tensor] = None):
        base = self.assembler(batch)                       # (B, T, D)
        B = base.shape[0]
        device = base.device

        if column_mask is not None:
            # Scatter the per-reconstructable mask onto the full token sequence,
            # then replace masked tokens with the learned [MASK] embedding.
            full = torch.zeros(B, self.n_tokens, dtype=torch.bool, device=device)
            for j, (_, base_pos, _, _) in enumerate(self.recon):
                full[:, base_pos] = column_mask[:, j]
            base = torch.where(full.unsqueeze(-1), self.mask_emb.view(1, 1, -1), base)

        cls = self.cls.expand(B, -1, -1)
        x = torch.cat([cls, base], dim=1) + self.pos[:, : self.n_tokens + 1]
        out = self.transformer(x)
        return out[:, 0], out                              # row embedding f(x), all tokens

    def encode(self, batch: dict) -> torch.Tensor:
        """Row embedding f(x) with no masking (for downstream / Layer 4)."""
        row_emb, _ = self.forward(batch, column_mask=None)
        return row_emb

    # -- losses ----------------------------------------------------------- #
    def reconstruction_loss(self, out, batch, column_mask):
        targets = self.recon_targets(batch)
        total = out.new_zeros(())
        count = 0
        for j, (name, _, out_pos, _) in enumerate(self.recon):
            m = column_mask[:, j]
            if m.any():
                logits = self.heads[name](out[:, out_pos][m])
                total = total + F.cross_entropy(
                    logits, targets[name][m], reduction="sum"   # already on device
                )
                count += int(m.sum())
        return total / max(count, 1)

    def composite_loss(self, batch: dict):
        """L_recon (two views) + λ·L_triplet (two views as positives)."""
        device = self.cls.device
        B = batch["amount"].shape[0]
        m1 = self.sample_column_mask(B, device)
        m2 = self.sample_column_mask(B, device)
        row1, out1 = self.forward(batch, m1)
        row2, out2 = self.forward(batch, m2)

        l_recon = self.reconstruction_loss(out1, batch, m1) + \
            self.reconstruction_loss(out2, batch, m2)

        emb = torch.cat([row1, row2], dim=0)
        labels = torch.cat([torch.arange(B, device=device)] * 2)
        l_trip = batch_hard_triplet_loss(emb, labels, self.config.triplet_margin)

        loss = l_recon + self.config.triplet_weight * l_trip
        return loss, {"recon": l_recon.detach().item(),
                      "triplet": l_trip.detach().item()}

    # -- masked-column reconstruction accuracy (C1 metric) ---------------- #
    @torch.no_grad()
    def reconstruction_accuracy(self, batch, column_mask=None, topk=(1, 3)):
        """Per-column top-k accuracy on masked positions (paper's top-3 metric)."""
        self.eval()
        device = self.cls.device
        B = batch["amount"].shape[0]
        if column_mask is None:
            column_mask = torch.ones(B, len(self.recon), dtype=torch.bool, device=device)
        _, out = self.forward(batch, column_mask)
        targets = self.recon_targets(batch)
        acc = {}
        for j, (name, _, out_pos, vocab) in enumerate(self.recon):
            m = column_mask[:, j]
            if not m.any():
                continue
            logits = self.heads[name](out[:, out_pos][m])
            tgt = targets[name][m].to(device)
            k = {kk: min(kk, vocab) for kk in topk}
            row = {}
            for kk, kc in k.items():
                top = logits.topk(kc, dim=1).indices
                row[f"top{kk}"] = float((top == tgt.unsqueeze(1)).any(1).float().mean())
            acc[name] = row
        return acc

    # -- freeze (INVARIANT: encoder frozen for Layer 4 — handoff §0.3) ----- #
    def freeze(self):
        for p in self.parameters():
            p.requires_grad_(False)
        self.eval()
        return self

    def num_trainable_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def backbone_parameters(self) -> int:
        """Transformer-only param count (the figure the paper's '25M' refers to)."""
        return sum(p.numel() for p in self.transformer.parameters())


# --------------------------------------------------------------------------- #
# Build the full Layer-2 → Layer-3 stack from data (shared by CLI + tests)
# --------------------------------------------------------------------------- #

def build_pretraining_stack(
    df, schema: dict, config: EncoderConfig,
    high_card_embedder: str = "partitioned", party_epochs: int = 3, seed: int = 0,
):
    """Wire vocabs, quantizer, offline party store, assembler, and the encoder.

    The party encoder is pre-learned offline here (§3.2) at C_g·D == config.hidden
    so its summary token matches the encoder width; it is then frozen inside the
    assembler. Returns (encoder, assembler, vocabs).
    """
    from encoders.column_assembler import build_vocabs
    from encoders.party_encoder import (
        PARTY_STRUCT_ATTRS, PartyEncoder, build_field_vocabs,
        build_party_store, encode_role_parties, party_roles_from_schema,
    )
    from encoders.quantizer import AdaptiveQuantizer

    D = config.hidden
    torch.manual_seed(seed)
    vocabs = build_vocabs(df, schema)
    quantizer = AdaptiveQuantizer().fit(
        df[vocabs.numerical_col].to_numpy(), df[vocabs.ccy_col].to_numpy()
    )

    roles = party_roles_from_schema(schema)
    pvocabs = build_field_vocabs(df, roles)
    field_vocabs = {a: len(pvocabs[a]) for a in PARTY_STRUCT_ATTRS}
    penc = PartyEncoder(field_vocabs, embedding_dim=D)
    train_ids = torch.cat([encode_role_parties(df, r, pvocabs)[1] for r in roles.values()])
    opt = torch.optim.Adam(penc.parameters(), lr=1e-3)
    for _ in range(party_epochs):
        loss = penc.masked_reconstruction_loss(train_ids)
        opt.zero_grad(); loss.backward(); opt.step()
    rows = {}
    for r in roles.values():
        for k, row in zip(*encode_role_parties(df, r, pvocabs)):
            rows[k] = row
    keys = list(rows)
    store = build_party_store(penc, keys, torch.stack([rows[k] for k in keys]))

    assembler = ColumnAssembler(
        schema, vocabs, quantizer, store, embedding_dim=D,
        high_card_embedder=high_card_embedder,
    )
    encoder = TabularEncoder(assembler, config)
    return encoder, assembler, vocabs


# --------------------------------------------------------------------------- #
# Minibatch helpers + pretraining loop
# --------------------------------------------------------------------------- #

def slice_batch(batch: dict, idx) -> dict:
    return {
        "high_card": {c: t[idx] for c, t in batch["high_card"].items()},
        "core": {c: t[idx] for c, t in batch["core"].items()},
        "amount": batch["amount"][idx],
        "ccy": batch["ccy"][idx],
    }


def pretrain(encoder: TabularEncoder, batch: dict, config: EncoderConfig,
             batch_size: int = 256, log=print) -> TabularEncoder:
    """Composite-loss pretraining for config.epochs with a cosine LR schedule."""
    n = batch["amount"].shape[0]
    opt = torch.optim.Adam(encoder.parameters(), lr=config.lr)
    steps = config.epochs * ((n + batch_size - 1) // batch_size)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(steps, 1))
    encoder.train()
    for ep in range(config.epochs):
        perm = torch.randperm(n)
        tot_r, tot_t, nb = 0.0, 0.0, 0
        for s in range(0, n, batch_size):
            idx = perm[s:s + batch_size]
            loss, parts = encoder.composite_loss(slice_batch(batch, idx))
            opt.zero_grad(); loss.backward(); opt.step(); sched.step()
            tot_r += parts["recon"]; tot_t += parts["triplet"]; nb += 1
        log(f"  epoch {ep+1}/{config.epochs}  recon {tot_r/nb:.4f}  "
            f"triplet {tot_t/nb:.4f}")
    return encoder


def main():
    import argparse
    import json
    from pathlib import Path

    import pandas as pd

    ap = argparse.ArgumentParser(description="Layer 3 tabular encoder (§3.4) pretraining / C1")
    root = Path(__file__).resolve().parents[1]
    ap.add_argument("--schema", default=str(root / "data" / "column_schema.json"))
    ap.add_argument("--data", default=str(root / "data" / "pacs008_synth.parquet"))
    ap.add_argument("--hidden", type=int, default=EncoderConfig.hidden)
    ap.add_argument("--layers", type=int, default=EncoderConfig.layers)
    ap.add_argument("--heads", type=int, default=EncoderConfig.heads)
    ap.add_argument("--epochs", type=int, default=EncoderConfig.epochs)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--limit", type=int, default=None,
                    help="cap training rows (logged); omit to use all")
    ap.add_argument("--eval-rows", type=int, default=4096)
    ap.add_argument("--compare", action="store_true",
                    help="train partitioned AND classical and print the C1 table")
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

    if args.limit and args.limit < len(df):
        print(f"NOTE: capping training rows {len(df):,} -> {args.limit:,} (--limit)")
        df = df.head(args.limit)

    cfg = EncoderConfig(hidden=args.hidden, layers=args.layers,
                        heads=args.heads, epochs=args.epochs)
    eval_n = min(args.eval_rows, max(1, len(df) // 10))
    train_df, eval_df = df.iloc[:-eval_n], df.iloc[-eval_n:]
    print(f"Tabular encoder (sec 3.4) on {src}: train={len(train_df):,} "
          f"eval={len(eval_df):,}  hidden={cfg.hidden} layers={cfg.layers} "
          f"epochs={cfg.epochs}")

    variants = ["partitioned", "classical"] if args.compare else ["partitioned"]
    results = {}
    for variant in variants:
        print(f"\n[{variant}] building + pretraining ...")
        enc, asm, vocabs = build_pretraining_stack(df, schema, cfg, high_card_embedder=variant)
        print(f"  backbone params: {enc.backbone_parameters():,}  "
              f"trainable: {enc.num_trainable_parameters():,}")
        train_batch = vocabs.encode(train_df)
        eval_batch = vocabs.encode(eval_df)
        pretrain(enc, train_batch, cfg, batch_size=args.batch)
        enc.freeze()
        acc = enc.reconstruction_accuracy(eval_batch)
        hc_params = sum(e.num_embedding_parameters() for e in asm.hc_emb.values())
        results[variant] = {"acc": acc, "hc_params": hc_params}
        print(f"  masked-column top-1 / top-3 accuracy (eval):")
        for col, row in acc.items():
            print(f"    {col:<16} top1={row['top1']:.3f}  top3={row['top3']:.3f}")

    if args.compare:
        p, c = results["partitioned"], results["classical"]
        hc_cols = schema["buckets"]["high_card_categorical"]
        gap = sum(c["acc"][k]["top1"] - p["acc"][k]["top1"] for k in hc_cols) / len(hc_cols)
        ratio = p["hc_params"] / c["hc_params"]
        print("\n=== C1 verdict (high-card columns) ===")
        print(f"  high-card param ratio (partitioned/classical): {ratio:.3f}  "
              f"(threshold <= 0.55)")
        print(f"  mean top-1 recon gap (classical - partitioned): {gap*100:.2f} pp  "
              f"(threshold <= 1.0 pp)")
        print(f"  C1 param-efficiency: {'PASS' if ratio <= 0.55 else 'FAIL'};  "
              f"C1 accuracy give-up: {'PASS' if gap <= 0.01 else 'FAIL'}")


if __name__ == "__main__":
    main()
