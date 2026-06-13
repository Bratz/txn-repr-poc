"""Shared test setup.

Puts the repo root on sys.path so tests can `import encoders...`, and provides
schema / sample-data fixtures that work on a fresh clone (the realized
column_schema.json is gitignored; column_schema.example.json is committed).
"""

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

_DATA = ROOT / "data"


def schema_path() -> Path:
    """Prefer the realized schema; fall back to the committed example."""
    real = _DATA / "column_schema.json"
    return real if real.exists() else _DATA / "column_schema.example.json"


@pytest.fixture(scope="session")
def schema() -> dict:
    return json.loads(schema_path().read_text())


@pytest.fixture(scope="session")
def sample_df():
    """The committed 500-row reference sample (always present)."""
    import pandas as pd
    return pd.read_csv(_DATA / "pacs008_sample_500.csv")
