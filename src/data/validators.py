"""Data quality validation for the raw alert DataFrame.

Raises ValueError on schema violations; logs warnings for anomalies.
"""
from __future__ import annotations

import logging

import pandas as pd

from config import OBLAST_UK_NAMES, NEAR_PERMANENT_REGIONS, OBLAST_ALIAS_MAP, OFFICIAL_START_DATE

logger = logging.getLogger(__name__)

_MIN_EXPECTED_ROWS = 10_000   # sanity floor; oblast-level deduped ≈ 65k


def validate(df: pd.DataFrame) -> None:
    """Run all validation checks; raise ValueError on hard failures."""
    _check_schema(df)
    _check_row_count(df)
    _check_date_order(df)
    _check_negative_durations(df)
    _check_no_duplicates(df)
    _check_regions(df)
    _check_coverage_start(df)
    logger.info("Validation passed (%d rows).", len(df))


# ── Hard checks (raise) ───────────────────────────────────────────────────────

def _check_schema(df: pd.DataFrame) -> None:
    required = {"region", "started_at", "finished_at", "duration_min"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"[validation] Missing columns after load: {missing}")


def _check_no_duplicates(df: pd.DataFrame) -> None:
    """Insurance check: loader must have removed exact (region, start, end) dups."""
    dups = df.duplicated(subset=["region", "started_at", "finished_at"]).sum()
    if dups > 0:
        raise ValueError(
            f"[validation] {dups} duplicate (region, started_at, finished_at) rows "
            "survived loading — dedup step failed."
        )


def _check_row_count(df: pd.DataFrame) -> None:
    if len(df) < _MIN_EXPECTED_ROWS:
        raise ValueError(
            f"[validation] Only {len(df)} rows — expected ≥{_MIN_EXPECTED_ROWS}. "
            "Dataset may be corrupt or partially downloaded."
        )


def _check_date_order(df: pd.DataFrame) -> None:
    complete = df.dropna(subset=["started_at", "finished_at"])
    bad = complete[complete["finished_at"] < complete["started_at"]]
    if len(bad) > 0:
        raise ValueError(
            f"[validation] {len(bad)} rows have finished_at < started_at."
        )


def _check_negative_durations(df: pd.DataFrame) -> None:
    neg = df[df["duration_min"].notna() & (df["duration_min"] < 0)]
    if len(neg) > 0:
        raise ValueError(
            f"[validation] {len(neg)} rows have negative duration."
        )
    zero = df[df["duration_min"].notna() & (df["duration_min"] == 0)]
    if len(zero) > 0:
        logger.warning(
            "[validation] %d rows have zero duration (started_at == finished_at) "
            "— possible data artefact.",
            len(zero),
        )


# ── Soft checks (warn) ────────────────────────────────────────────────────────

def _check_coverage_start(df: pd.DataFrame) -> None:
    """Warn if data contains alerts before the official API coverage start."""
    earliest = df["started_at"].min()
    if pd.isna(earliest):
        return
    # Normalise to tz-naive date for comparison
    earliest_date = pd.Timestamp(earliest).tz_localize(None) if earliest.tzinfo is None else pd.Timestamp(earliest).tz_convert(None)
    cutoff = pd.Timestamp(OFFICIAL_START_DATE)
    if earliest_date < cutoff:
        logger.warning(
            "[validation] Earliest alert %s is before OFFICIAL_START_DATE %s "
            "(Vadimkin dataset coverage). Pre-%s data may be incomplete.",
            earliest_date.date(), OFFICIAL_START_DATE, OFFICIAL_START_DATE,
        )


def _check_regions(df: pd.DataFrame) -> None:
    # Apply the alias map first so "Харківська область" → "Харківська" before compare
    found = {OBLAST_ALIAS_MAP.get(r, r) for r in df["region"].dropna().unique()}
    known = set(OBLAST_UK_NAMES)
    unknown = found - known - NEAR_PERMANENT_REGIONS
    if unknown:
        logger.warning(
            "[validation] Unknown region names (may need alias mapping): %s",
            unknown,
        )
    missing_regions = known - NEAR_PERMANENT_REGIONS - found
    if missing_regions:
        logger.warning(
            "[validation] Expected regions absent from dataset: %s",
            missing_regions,
        )
