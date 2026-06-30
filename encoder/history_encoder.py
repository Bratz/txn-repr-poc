"""
v2 / Layer 3b - the history encoder over an entity's ordered transaction sequence.

Beyond arXiv:2410.07851; see docs/V2_DIRECTION.md. It wraps the frozen v1 per-transaction
encoder: given the sequence of per-event embeddings e_1..e_n (each the v1 [CLS] f(x_t)) plus
their inter-arrival/calendar encoding and a prepended [USR] summary token, a bidirectional
transformer produces one entity representation h_USR. This is the axis a single-record model
and a gradient-boosted tree cannot see.

Design follows the production foundation models:
  * masked-EVENT pretraining (predict a whole transaction's fields from its neighbours) -
    Revolut PRAGMA's multi-level masking (arXiv:2604.08649).
  * an optional CoLES-style contrastive term - two masked views of one entity are positives -
    which REUSES v1's batch_hard_triplet_loss (Babaev et al., arXiv:2002.08232).

Freeze discipline (docs/V2_DIRECTION.md sec. 5): the v1 encoder that produces e_t is frozen
before this stage; only the history encoder (and its time encoding) train here.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from encoder.tabular_encoder import batch_hard_triplet_loss
from encoders.time_encoding import TimeEncoding


@dataclass
class HistoryConfig:
    hidden: int = 512            # must equal the v1 encoder hidden (e_t width)
    layers: int = 4
    heads: int = 8
    ff_mult: int = 4
    dropout: float = 0.1
    mask_prob: float = 0.15      # fraction of events masked (PRAGMA-style)
    triplet_margin: float = 1.0
    triplet_weight: float = 1.0  # 0 disables the CoLES term
    epochs: int = 3
    lr: float = 1e-4


class HistoryEncoder(nn.Module):
    """[USR] + event sequence -> entity representation, with masked-event pretraining.

    Args:
        recon_fields: {field_name: vocab_size} - the discrete fields a masked event is
            reconstructed into (e.g. amount level, currency, settlement method).
        config: HistoryConfig (hidden must match the frozen e_t width).
    """

    def __init__(self, recon_fields: dict[str, int], config: HistoryConfig):
        super().__init__()
        self.config = config
        D = config.hidden
        self.D = D
        self.recon_fields = dict(recon_fields)

        self.time = TimeEncoding(D)
        self.usr_token = nn.Parameter(torch.randn(1, 1, D) * 0.02)   # [USR] summary query
        self.mask_evt = nn.Parameter(torch.randn(D) * 0.02)          # [MASK_EVT] embedding

        layer = nn.TransformerEncoderLayer(
            d_model=D, nhead=config.heads, dim_feedforward=config.ff_mult * D,
            dropout=config.dropout, batch_first=True, activation="gelu",
        )
        self.transformer = nn.TransformerEncoder(layer, num_layers=config.layers)
        self.heads = nn.ModuleDict(
            {name: nn.Linear(D, vocab) for name, vocab in self.recon_fields.items()}
        )

    # -- forward ---------------------------------------------------------- #
    def forward(self, e_seq: torch.Tensor, batch: dict,
                static: Optional[torch.Tensor] = None,
                event_mask: Optional[torch.Tensor] = None):
        """e_seq: (B, L, D) per-event embeddings. batch: dt/dow/dom/month/pad_mask (B, L).

        Returns (h_usr (B, D), h_events (B, L, D)).
        """
        x = e_seq + self.time(batch["dt"], batch["dow"], batch["dom"], batch["month"])
        if event_mask is not None:
            x = torch.where(event_mask.unsqueeze(-1), self.mask_evt.view(1, 1, -1), x)
        B = x.shape[0]
        usr = self.usr_token.expand(B, -1, -1)
        if static is not None:
            usr = usr + static.unsqueeze(1)
        z = torch.cat([usr, x], dim=1)                              # (B, 1+L, D)
        pad = torch.cat(
            [torch.zeros(B, 1, dtype=torch.bool, device=x.device), batch["pad_mask"]],
            dim=1,
        )
        out = self.transformer(z, src_key_padding_mask=pad)
        return out[:, 0], out[:, 1:]

    @torch.no_grad()
    def encode(self, e_seq, batch, static=None) -> torch.Tensor:
        """Entity representation h_USR with no masking (downstream / eval).

        Saves/restores the module's training mode so calling encode() mid-training does not
        silently leave the encoder in eval (dropout off) for the rest of the loop."""
        was_training = self.training
        self.eval()
        try:
            h_usr, _ = self.forward(e_seq, batch, static, event_mask=None)
            return h_usr
        finally:
            self.train(was_training)

    # -- objective -------------------------------------------------------- #
    def sample_event_mask(self, pad_mask: torch.Tensor) -> torch.Tensor:
        """(B, L) bool mask over non-pad events; guarantee >= 1 masked event per non-empty row."""
        valid = ~pad_mask
        m = (torch.rand_like(pad_mask, dtype=torch.float) < self.config.mask_prob) & valid
        need = valid.any(dim=1) & ~m.any(dim=1)
        for i in torch.nonzero(need, as_tuple=False).flatten().tolist():
            choices = torch.nonzero(valid[i], as_tuple=False).flatten()
            m[i, choices[torch.randint(len(choices), (1,))]] = True
        return m

    def masked_event_loss(self, h_events, targets: dict, event_mask) -> torch.Tensor:
        """Cross-entropy over masked events, averaged across positions and fields."""
        total = h_events.new_zeros(())
        n_masked = int(event_mask.sum())
        if n_masked == 0:
            return total
        for name, head in self.heads.items():
            logits = head(h_events[event_mask])                    # (M, vocab)
            total = total + F.cross_entropy(logits, targets[name][event_mask],
                                            reduction="sum")
        return total / (n_masked * len(self.heads))

    def composite_loss(self, e_seq, batch, targets, static=None):
        """L_mask (whole-event reconstruction) + weight * L_triplet (CoLES, two views)."""
        m1 = self.sample_event_mask(batch["pad_mask"])
        h1, hev1 = self.forward(e_seq, batch, static, m1)
        l_mask = self.masked_event_loss(hev1, targets, m1)

        l_trip = e_seq.new_zeros(())
        if self.config.triplet_weight > 0:
            m2 = self.sample_event_mask(batch["pad_mask"])
            h2, _ = self.forward(e_seq, batch, static, m2)
            B = e_seq.shape[0]
            emb = torch.cat([h1, h2], dim=0)
            labels = torch.cat([torch.arange(B, device=e_seq.device)] * 2)
            l_trip = batch_hard_triplet_loss(emb, labels, self.config.triplet_margin)

        loss = l_mask + self.config.triplet_weight * l_trip
        return loss, {"mask": float(l_mask.detach()), "triplet": float(l_trip.detach())}

    def freeze(self):
        for p in self.parameters():
            p.requires_grad_(False)
        self.eval()
        return self

    def num_trainable_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# --------------------------------------------------------------------------- #
# Pretraining loop (mirrors encoder.tabular_encoder.pretrain)
# --------------------------------------------------------------------------- #

def pretrain(hist: HistoryEncoder, e_all: torch.Tensor, targets_all: dict,
             train_seqs: list, config: HistoryConfig, batch_size: int = 64,
             static_all: Optional[torch.Tensor] = None, log=print) -> HistoryEncoder:
    """Train the history encoder over sequences of FROZEN per-event embeddings e_all.

    e_all: (N, D) frozen v1 embeddings for every row. targets_all: {field: (N,) long}.
    train_seqs: list of sequence dicts from data.sequence_assembly. static_all: optional
    (N, D) per-row static profile (e.g. party-store summary), gathered by the actor's rows.
    """
    from data.sequence_assembly import collate

    device = e_all.device
    opt = torch.optim.Adam(hist.parameters(), lr=config.lr)
    n = len(train_seqs)
    hist.train()
    for ep in range(config.epochs):
        perm = torch.randperm(n).tolist()
        tot_m, tot_t, nb = 0.0, 0.0, 0
        for s in range(0, n, batch_size):
            chunk = [train_seqs[i] for i in perm[s:s + batch_size]]
            b = collate(chunk)
            b = {k: v.to(device) for k, v in b.items()}
            e_seq = e_all[b["pos"]]                                 # (B, L, D)
            tgt = {name: targets_all[name].to(device)[b["pos"]] for name in hist.recon_fields}
            static = static_all[b["pos"][:, 0]] if static_all is not None else None
            loss, parts = hist.composite_loss(e_seq, b, tgt, static)
            opt.zero_grad(); loss.backward(); opt.step()
            tot_m += parts["mask"]; tot_t += parts["triplet"]; nb += 1
        log(f"  hist epoch {ep+1}/{config.epochs}  mask {tot_m/max(nb,1):.4f}  "
            f"triplet {tot_t/max(nb,1):.4f}")
    return hist
