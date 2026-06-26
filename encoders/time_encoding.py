"""
v2 / Layer 2 addition - event-time encoding for entity histories.

This is BEYOND arXiv:2410.07851 (a v1 module would cite a paper section). It belongs
to the sequence extension specified in docs/V2_DIRECTION.md and is grounded in the
production transaction foundation models:
  * inter-arrival via log compression - Revolut PRAGMA's 8*ln(1+t/8) (arXiv:2604.08649)
  * absolute calendar features - PRAGMA's hour/day/month event features

A single transaction carries its date only as a categorical core column. A *sequence*
needs the time BETWEEN events. For each event t in an entity's ordered history we encode
the inter-arrival gap dt_t (days since the previous event) and the calendar position, and
ADD the result to the per-event embedding (width D = encoder hidden), so the history
encoder sees cadence, not just order.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class TimeEncoding(nn.Module):
    """Inter-arrival + calendar encoding -> a D-dim vector added to each event embedding.

    Args:
        d: embedding width (must equal the encoder hidden size).
        n_dow / n_dom / n_month: calendar vocabulary sizes (0-indexed inputs).
    """

    def __init__(self, d: int, n_dow: int = 7, n_dom: int = 31, n_month: int = 12):
        super().__init__()
        self.d = int(d)
        # Continuous inter-arrival: log1p compresses the heavy tail (a 2-day gap and a
        # 400-day gap should not be linearly far apart). PAPER-analogue: PRAGMA log time.
        self.dt_mlp = nn.Sequential(
            nn.Linear(1, max(self.d // 2, 1)), nn.GELU(),
            nn.Linear(max(self.d // 2, 1), self.d),
        )
        self.dow = nn.Embedding(n_dow, self.d)
        self.dom = nn.Embedding(n_dom, self.d)
        self.month = nn.Embedding(n_month, self.d)

    def forward(self, dt: torch.Tensor, dow: torch.Tensor,
                dom: torch.Tensor, month: torch.Tensor) -> torch.Tensor:
        """dt: (B, L) float days since previous event (>= 0). dow/dom/month: (B, L) long.

        Returns (B, L, D).
        """
        t = torch.log1p(dt.clamp(min=0).float()).unsqueeze(-1)   # (B, L, 1)
        return self.dt_mlp(t) + self.dow(dow) + self.dom(dom) + self.month(month)
