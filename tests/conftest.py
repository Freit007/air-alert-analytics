"""Shared test fixtures."""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

# Add project root to path
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))


FIXTURES_DIR = Path(__file__).parent / "fixtures"


@pytest.fixture(scope="session")
def raw_df() -> pd.DataFrame:
    """Load sample CSV as raw DataFrame (no transforms)."""
    from src.data.loader import load_raw
    return load_raw(FIXTURES_DIR / "sample_alerts.csv")


@pytest.fixture(scope="session")
def transformed_df(raw_df) -> pd.DataFrame:
    """Fully transformed DataFrame (all pitfalls handled)."""
    from src.data.transforms import apply_all
    return apply_all(raw_df)


@pytest.fixture(scope="session")
def analysis_df(transformed_df) -> pd.DataFrame:
    """Analysis-ready DataFrame (oblast-level, no permanent regions)."""
    from src.data.transforms import copy_for_analysis
    return copy_for_analysis(transformed_df)

