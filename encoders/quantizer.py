"""
§3.3 Adaptive numerical quantization (currency-conditioned).

Grounded strictly to:
  Raman, Ganesh, Veloso (JPMorgan AI Research),
  "Scalable Representation Learning for Multimodal Tabular Transactions",
  arXiv:2410.07851, NeurIPS 2024 TRL workshop.

# PAPER: §3.3. A numerical column is mapped to a custom vocabulary of numerical
# tokens Q = {Q_1,…,Q_m} that "adapts to resolution, with finer spacing for
# smaller numbers and progressively larger spacing for larger numbers". A
# continuous value x is assigned to the nearest level:  argmin_i |x − Q_i|.
# The resulting level index is then embedded like a categorical token.
#
# The paper does NOT specify the spacing law, the level count m, or any
# normalization. Documented v1 choices (architecture.md §7 style):
#   * Spacing = GEOMETRIC (log-spaced) over the value range. Geometric levels
#     give absolute gaps that grow with magnitude — exactly "finer for small,
#     coarser for large", keyed to magnitude as the paper frames it (a
#     quantile grid would key off data DENSITY instead, which is a different
#     claim). # PAPER: §3.3.
#   * m = 128 levels — a POC default, NOT paper-pinned (the paper is silent).
#     Lives here, not in configs/default.yaml, alongside the other unspecified
#     sizing choices (cf. embedding dim in partitioning_embedder).
#
# FORCED DEPARTURE (handoff §0.5, departures.currency_conditioned_quantization):
# the grid is built WITHIN each currency. Amounts across currencies live on
# wildly different scales (JPY vs USD), so a single global grid mis-quantizes
# multi-currency data — this is a correctness fix, not an enhancement. The
# level-index vocabulary stays SHARED (size m): level i means "the i-th
# magnitude band for its own currency", and Ccy rides separately as a core
# column. So two currencies' level i share an embedding row (relative-magnitude
# semantics); currency identity is not folded into the numeric token.
"""

from __future__ import annotations

from typing import Optional, Sequence

import numpy as np

# POC default level count (see module docstring — not paper-pinned).
DEFAULT_NUM_LEVELS = 128
_GLOBAL_KEY = "__GLOBAL__"  # fallback grid for unseen / unconditioned currencies
_EPS = 1e-9


# --------------------------------------------------------------------------- #
# Geometric level construction
# --------------------------------------------------------------------------- #

def geometric_levels(lo: float, hi: float, m: int) -> np.ndarray:
    """m log-spaced levels in [lo, hi]  → finer gaps for small, coarser for large.

    # PAPER: §3.3 ("finer spacing for smaller numbers, progressively larger for
    # larger"). Settlement amounts are positive; we floor `lo` at a small
    # epsilon so log-spacing is well defined. Degenerate ranges (hi ≤ lo) fall
    # back to a constant grid (every value lands in one band).
    """
    if m < 2:
        raise ValueError(f"num_levels must be ≥ 2, got {m}")
    lo = max(float(lo), _EPS)
    hi = float(hi)
    if hi <= lo * (1.0 + 1e-6):
        return np.full(m, lo, dtype=np.float64)
    return np.exp(np.linspace(np.log(lo), np.log(hi), m)).astype(np.float64)


def _assign_nearest(grid: np.ndarray, x: np.ndarray) -> np.ndarray:
    """argmin_i |x − grid_i| for an ascending `grid` (vectorized; clamps ends).

    # PAPER: §3.3 assignment rule. searchsorted finds the insertion point, then
    # we pick the nearer of the two bracketing levels. Out-of-range values clamp
    # to the nearest endpoint, which is the correct nearest level.
    """
    idx = np.searchsorted(grid, x)
    idx = np.clip(idx, 1, len(grid) - 1)
    left = grid[idx - 1]
    right = grid[idx]
    choose_left = (x - left) <= (right - x)
    return np.where(choose_left, idx - 1, idx).astype(np.int64)


# --------------------------------------------------------------------------- #
# Adaptive quantizer (§3.3) — fit / transform to level indices
# --------------------------------------------------------------------------- #

class AdaptiveQuantizer:
    """Fit per-currency geometric grids; transform amounts → level indices.

    Args:
        num_levels: m, the shared level-vocabulary size.
        condition_on_currency: if True (the forced departure), one grid per
            currency; if False, a single global grid (paper's nominal behaviour).
        clip_quantile: optional (lo_q, hi_q) in [0,1] to set each grid's range
            from data percentiles instead of min/max — robustness to outliers.
            None ⇒ full [min, max] range.
    """

    def __init__(
        self,
        num_levels: int = DEFAULT_NUM_LEVELS,
        condition_on_currency: bool = True,
        clip_quantile: Optional[tuple[float, float]] = None,
    ):
        if num_levels < 2:
            raise ValueError(f"num_levels must be ≥ 2, got {num_levels}")
        self.num_levels = int(num_levels)
        self.condition_on_currency = bool(condition_on_currency)
        self.clip_quantile = clip_quantile
        self.grids_: dict[str, np.ndarray] = {}  # currency → ascending levels
        self._fitted = False

    # -- internal --------------------------------------------------------- #
    def _range(self, vals: np.ndarray) -> tuple[float, float]:
        if self.clip_quantile is not None:
            lo_q, hi_q = self.clip_quantile
            return (float(np.quantile(vals, lo_q)), float(np.quantile(vals, hi_q)))
        return (float(vals.min()), float(vals.max()))

    def _build_grid(self, vals: np.ndarray) -> np.ndarray:
        lo, hi = self._range(vals)
        return geometric_levels(lo, hi, self.num_levels)

    # -- public ----------------------------------------------------------- #
    def fit(
        self,
        amounts: Sequence[float],
        currencies: Optional[Sequence[str]] = None,
    ) -> "AdaptiveQuantizer":
        amt = np.asarray(amounts, dtype=np.float64).reshape(-1)
        if amt.size == 0:
            raise ValueError("cannot fit on empty amounts")

        # Always build a global fallback grid (used for unseen currencies and
        # when condition_on_currency is False).
        self.grids_ = {_GLOBAL_KEY: self._build_grid(amt)}

        if self.condition_on_currency:
            if currencies is None:
                raise ValueError(
                    "condition_on_currency=True requires currencies at fit time"
                )
            ccy = np.asarray(currencies, dtype=object).reshape(-1)
            if ccy.shape[0] != amt.shape[0]:
                raise ValueError("amounts and currencies length mismatch")
            for c in np.unique(ccy):
                self.grids_[str(c)] = self._build_grid(amt[ccy == c])

        self._fitted = True
        return self

    def transform(
        self,
        amounts: Sequence[float],
        currencies: Optional[Sequence[str]] = None,
    ) -> np.ndarray:
        if not self._fitted:
            raise RuntimeError("AdaptiveQuantizer must be fit before transform")
        amt = np.asarray(amounts, dtype=np.float64).reshape(-1)

        if not self.condition_on_currency:
            return _assign_nearest(self.grids_[_GLOBAL_KEY], amt)

        if currencies is None:
            raise ValueError(
                "condition_on_currency=True requires currencies at transform time"
            )
        ccy = np.asarray(currencies, dtype=object).reshape(-1)
        if ccy.shape[0] != amt.shape[0]:
            raise ValueError("amounts and currencies length mismatch")

        out = np.empty(amt.shape[0], dtype=np.int64)
        for c in np.unique(ccy):
            mask = ccy == c
            # Unseen currency → global fallback grid (no crash).
            grid = self.grids_.get(str(c), self.grids_[_GLOBAL_KEY])
            out[mask] = _assign_nearest(grid, amt[mask])
        return out

    def fit_transform(self, amounts, currencies=None) -> np.ndarray:
        return self.fit(amounts, currencies).transform(amounts, currencies)

    # -- persistence (the grids ARE the numerical vocabulary Q) ----------- #
    def to_dict(self) -> dict:
        return {
            "num_levels": self.num_levels,
            "condition_on_currency": self.condition_on_currency,
            "clip_quantile": self.clip_quantile,
            "grids": {k: v.tolist() for k, v in self.grids_.items()},
        }

    @classmethod
    def from_dict(cls, d: dict) -> "AdaptiveQuantizer":
        q = cls(
            num_levels=d["num_levels"],
            condition_on_currency=d["condition_on_currency"],
            clip_quantile=(tuple(d["clip_quantile"]) if d["clip_quantile"] else None),
        )
        q.grids_ = {k: np.asarray(v, dtype=np.float64) for k, v in d["grids"].items()}
        q._fitted = True
        return q


# --------------------------------------------------------------------------- #
# Embedding wrapper (§3.3 "embed the level like a categorical token")
# --------------------------------------------------------------------------- #

def make_quantizer_embedder(quantizer: AdaptiveQuantizer, embedding_dim: int):
    """Build a torch module that quantizes (amount, ccy) → level → embedding.

    Imported lazily so the numpy-only quantizer has no hard torch dependency.
    """
    import torch
    import torch.nn as nn

    class QuantizerEmbedder(nn.Module):
        # PAPER: §3.3 — the level index is embedded like a categorical token.
        # Shared table over the m levels (currency rides as a separate core
        # column), so level i across currencies shares a row.
        def __init__(self, q: AdaptiveQuantizer, dim: int):
            super().__init__()
            self.quantizer = q
            self.embedding_dim = int(dim)
            self.emb = nn.Embedding(q.num_levels, self.embedding_dim)

        def forward(self, amounts, currencies=None):
            idx = self.quantizer.transform(amounts, currencies)
            idx_t = torch.as_tensor(idx, dtype=torch.long, device=self.emb.weight.device)
            return self.emb(idx_t)

        def num_embedding_parameters(self) -> int:
            return self.quantizer.num_levels * self.embedding_dim

    return QuantizerEmbedder(quantizer, embedding_dim)


# --------------------------------------------------------------------------- #
# CLI: fit on the realized data and summarize per-currency grids
# --------------------------------------------------------------------------- #

def main():
    import argparse
    import json
    from pathlib import Path

    ap = argparse.ArgumentParser(description="§3.3 adaptive quantizer report")
    root = Path(__file__).resolve().parents[1]
    ap.add_argument("--schema", default=str(root / "data" / "column_schema.json"))
    ap.add_argument("--data", default=str(root / "data" / "pacs008_synth.parquet"))
    ap.add_argument("--levels", type=int, default=DEFAULT_NUM_LEVELS)
    args = ap.parse_args()

    import pandas as pd

    schema = json.loads(Path(args.schema).read_text())
    # Read the numerical column + currency from the schema — never hard-code (§0.4).
    num_col = schema["buckets"]["numerical"][0]
    ccy_col = "Ccy" if "Ccy" in schema["buckets"]["core"] else schema["buckets"]["core"][0]

    path = Path(args.data)
    df, src = None, None
    if path.exists():
        try:
            df = pd.read_parquet(path) if path.suffix == ".parquet" else pd.read_csv(path)
            src = path.name
        except Exception as e:  # e.g. parquet engine (pyarrow) not installed
            print(f"(could not read {path.name}: {e}; using reference sample)")
    if df is None:  # fall back to the committed reference sample
        df = pd.read_csv(root / "data" / "pacs008_sample_500.csv")
        src = "pacs008_sample_500.csv (fallback)"

    q = AdaptiveQuantizer(num_levels=args.levels, condition_on_currency=True)
    idx = q.fit_transform(df[num_col].to_numpy(), df[ccy_col].to_numpy())

    print(f"Adaptive quantizer (m={args.levels}, currency-conditioned) on {src}")
    print(f"column={num_col} ccy={ccy_col} rows={len(df):,} "
          f"distinct levels used={len(np.unique(idx))}/{args.levels}\n")
    print(f"{'ccy':<6}{'n':>8}{'min':>14}{'max':>16}{'Q0':>12}{'Q[-1]':>16}")
    for c in sorted(df[ccy_col].unique()):
        sub = df[df[ccy_col] == c][num_col].to_numpy()
        g = q.grids_[str(c)]
        print(f"{str(c):<6}{sub.size:>8}{sub.min():>14,.2f}{sub.max():>16,.2f}"
              f"{g[0]:>12,.2f}{g[-1]:>16,.2f}")


if __name__ == "__main__":
    main()
