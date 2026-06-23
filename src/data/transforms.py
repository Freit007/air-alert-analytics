"""Apply data-correctness rules and produce analysis-ready DataFrames.

Pitfalls handled (live-related dedup removed — project is archive-only):
  • LEVEL       — only whole-oblast alerts (done in loader.py)
  • DEDUP       — exact-duplicate rows dropped (done in loader.py)
  • TIMEZONE    — UTC → Europe/Kyiv with DST (here)
  • CENSORING   — open intervals flagged for survival analysis (here)
  • PERMANENT   — Luhansk/Crimea excluded from aggregate stats (here)

[З ДОСЛІДЖЕННЯ] Rules from data_correctness_rules section.
"""
from __future__ import annotations

import logging
from typing import Optional

import pandas as pd
import pytz

from config import (
    KYIV_TZ,
    NEAR_PERMANENT_REGIONS,
    OBLAST_ALIAS_MAP,
)

logger = logging.getLogger(__name__)

_KYIV = pytz.timezone(KYIV_TZ)


# ── Public API ────────────────────────────────────────────────────────────────

def apply_all(df: pd.DataFrame, exclude_permanent: bool = True) -> pd.DataFrame:
    """Full transform pipeline — returns a clean, analysis-ready DataFrame."""
    df = df.copy()
    df = _normalize_regions(df)
    df = _convert_to_kyiv(df)          # Timezone
    df = _mark_censored(df)            # Censoring
    df = _recompute_duration(df)       # Recompute after tz conversion
    if exclude_permanent:
        df = _exclude_permanent(df)    # Permanent regions
    df = _add_temporal_features(df)
    logger.info("Transform complete: %d rows, %d cols", len(df), len(df.columns))
    return df


def copy_for_analysis(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy of df suitable for statistical analysis.

    Callers handle censored intervals explicitly via the `censored` column.
    Returns a copy to prevent in-place mutations from affecting the cache.
    """
    return df.copy()



# ── Internal transforms ───────────────────────────────────────────────────────

def _normalize_regions(df: pd.DataFrame) -> pd.DataFrame:
    """Map variant spellings → canonical short name."""
    df["region"] = df["region"].map(
        lambda x: OBLAST_ALIAS_MAP.get(x, x) if isinstance(x, str) else x
    )
    return df


def _convert_to_kyiv(df: pd.DataFrame) -> pd.DataFrame:
    """UTC → Europe/Kyiv (full DST-aware conversion)."""
    for col in ("started_at", "finished_at"):
        if col in df.columns and pd.api.types.is_datetime64_any_dtype(df[col]):
            series = df[col]
            if series.dt.tz is None:
                series = series.dt.tz_localize("UTC")
            df[col] = series.dt.tz_convert(KYIV_TZ)
    return df


def _mark_censored(df: pd.DataFrame) -> pd.DataFrame:
    """Mark open (ongoing) intervals as right-censored."""
    df["censored"] = df["finished_at"].isna()
    return df


def _recompute_duration(df: pd.DataFrame) -> pd.DataFrame:
    """Recompute duration_min from timestamps after tz conversion."""
    complete = df["finished_at"].notna() & df["started_at"].notna()
    delta = (df.loc[complete, "finished_at"] - df.loc[complete, "started_at"])
    df.loc[complete, "duration_min"] = delta.dt.total_seconds() / 60.0
    return df


def _exclude_permanent(df: pd.DataFrame) -> pd.DataFrame:
    """Drop near-permanent regions from aggregate analysis."""
    mask = df["region"].isin(NEAR_PERMANENT_REGIONS)
    dropped = mask.sum()
    if dropped:
        logger.info(
            "Excluded %d rows from near-permanent regions: %s",
            dropped, NEAR_PERMANENT_REGIONS,
        )
    return df[~mask].copy()


def _add_temporal_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add hour/dow/month/week columns derived from Kyiv-local started_at."""
    s = df["started_at"]
    df["hour"] = s.dt.hour
    df["dow"] = s.dt.dayofweek          # 0=Monday
    df["dow_name"] = s.dt.day_name()
    df["month"] = s.dt.month
    df["month_name"] = s.dt.month_name()
    df["year"] = s.dt.year
    df["date"] = s.dt.normalize()
    df["week"] = s.dt.isocalendar().week.astype(int)
    return df


def make_binary_series(
    df: pd.DataFrame,
    freq: str = "1h",
    regions: Optional[list[str]] = None,
) -> pd.DataFrame:
    """Build a binary (0/1) time series: 1 if alert was active in that period.

    Returns a DataFrame indexed by period start, columns = regions.
    Uses alert intervals (not point-in-time) for correctness.
    """
    if regions is None:
        regions = sorted(df["region"].dropna().unique().tolist())

    t_min = df["started_at"].min().floor(freq)
    _valid_ends = df["finished_at"].dropna()
    _open_starts = df.loc[df["finished_at"].isna(), "started_at"]
    # Extend t_max to cover the start bucket of any open-ended alert that begins
    # after all closed alerts end; otherwise those alerts are silent in the series.
    _t_max_closed = _valid_ends.max() if not _valid_ends.empty else pd.NaT
    _t_max_open = (
        (_open_starts.max() + pd.tseries.frequencies.to_offset(freq))
        if not _open_starts.empty else pd.NaT
    )
    _candidates = [t for t in [_t_max_closed, _t_max_open] if pd.notna(t)]
    t_max = max(_candidates).ceil(freq) if _candidates else df["started_at"].max().ceil(freq)
    idx = pd.date_range(t_min, t_max, freq=freq, tz=KYIV_TZ)

    result = pd.DataFrame(0, index=idx, columns=regions, dtype="int8")

    for region, grp in df.groupby("region"):
        if region not in regions:
            continue
        for _, row in grp[["started_at", "finished_at"]].iterrows():
            s = row["started_at"]
            e = row["finished_at"]
            if pd.isna(e):
                e = t_max
            mask = (idx >= s.floor(freq)) & (idx < e.ceil(freq))
            result.loc[mask, region] = 1  # type: ignore[index]

    return result
