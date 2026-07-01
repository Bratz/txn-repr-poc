"""
Canonical string coercion for categorical / reference-number columns.

Why: vocabs key categorical values by their *string* form. A bare `.astype(str)` is
dtype-sensitive — an id trained as int (`"123"`) but arriving at serve time as float
(`"123.0"`, because a NaN promoted the column, or CSV inferred float, or the value went
through pandas), or in scientific notation for a large int, silently mismatches the vocab
and maps to UNK with no error. This one helper, used at BOTH vocab-build and encode time,
makes int / float / str forms of the same reference number collapse to one key.

Rules:
  * integral numbers (int, or float equal to its int) -> plain integer string, no ".0",
    no scientific notation: 123, 123.0, 1e8  -> "123", "123", "100000000"
  * non-integral floats -> repr (stable round-trip): 123.5 -> "123.5"
  * genuine strings (incl. leading zeros like "007") -> preserved as-is
  * missing values -> "nan" (callers reserve a UNK/MASK slot for these)
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def _fmt_number(x):
    if pd.isna(x):
        return "nan"
    f = float(x)
    if np.isfinite(f) and f == int(f):
        return str(int(f))
    return repr(f)


def canon_categorical(s: pd.Series) -> pd.Series:
    """Stable string key for a categorical/reference-number column (see module docstring)."""
    if pd.api.types.is_bool_dtype(s):
        return s.astype(str)
    if pd.api.types.is_integer_dtype(s) or pd.api.types.is_float_dtype(s):
        return s.map(_fmt_number)
    return s.astype(str)
