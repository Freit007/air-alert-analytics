"""Cached data loader — importable from any page without triggering app.py side effects.

[МОЄ РІШЕННЯ] Extracted from app.py to prevent st.set_page_config being called
when pages do `from app import load_data`.
"""
from __future__ import annotations

import streamlit as st


@st.cache_data(show_spinner="Завантаження датасету…", ttl=3600)
def load_data():
    """Download (if needed), validate, and transform the Vadimkin CSV dataset."""
    from src.data.loader import load
    from src.data.validators import validate
    from src.data.transforms import apply_all

    raw = load(download_if_missing=True)
    validate(raw)
    return apply_all(raw)


@st.cache_data(ttl=3600)
def load_unit_records():
    """Cached per-unit alert records (before oblast union).

    Used for coverage analysis via src.analysis.coverage.episode_unit_coverage().
    Assumes load_data() has already been called (CSV already on disk).
    """
    from src.data.loader import load_unit_records as _load
    return _load()
