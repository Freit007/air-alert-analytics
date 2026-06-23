"""Descriptive statistics and EDA for historical alert data.

All functions are pure: (DataFrame, optional params) → result.
Input DataFrames must already be transformed (transforms.apply_all applied).
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd

from config import COVER_START_DATE


def alert_frequency_over_time(
    df: pd.DataFrame,
    freq: str = "W",
    regions: Optional[list[str]] = None,
) -> pd.DataFrame:
    """Count alerts per time period, optionally filtered by region.

    Returns DataFrame with columns: period, alert_count, [region if grouped].
    """
    data = df.copy()
    if regions:
        data = data[data["region"].isin(regions)]

    data = data.set_index("started_at").sort_index()
    counts = data.resample(freq).size().rename("alert_count").reset_index()
    counts.rename(columns={"started_at": "period"}, inplace=True)
    return counts


def duration_stats(df: pd.DataFrame) -> dict:
    """Summary statistics for alert duration (non-censored only).

    Requires: 'censored' column (added by apply_all).
    """
    if "censored" not in df.columns:
        raise ValueError(
            "duration_stats requires a 'censored' column; "
            "call apply_all() before passing the DataFrame."
        )
    col = df.loc[~df["censored"].fillna(False), "duration_min"].dropna()
    if col.empty:
        return {}
    return {
        "count": int(col.count()),
        "mean_min": float(col.mean()),
        "median_min": float(col.median()),
        "p25_min": float(col.quantile(0.25)),
        "p75_min": float(col.quantile(0.75)),
        "p95_min": float(col.quantile(0.95)),
        "max_min": float(col.max()),
        "std_min": float(col.std()),
        "total_hours": float(col.sum() / 60.0),
    }


def duration_histogram_data(
    df: pd.DataFrame,
    max_minutes: float = 300.0,
    n_bins: int = 60,
) -> tuple[pd.DataFrame, int, int]:
    """Return binned duration counts for histogram rendering.

    Filters to max_minutes — episodes longer than the slider are excluded,
    not clipped (clip would cause a false accumulation spike in the last bin).

    Returns (hist_df, n_total, n_excluded).
    """
    col = df.loc[~df["censored"].fillna(False), "duration_min"].dropna()
    n_total = len(col)
    col = col[col <= max_minutes]
    n_excluded = n_total - len(col)
    bins = np.linspace(0, max_minutes, n_bins + 1)
    counts, edges = np.histogram(col, bins=bins)
    hist_df = pd.DataFrame({
        "bin_left": edges[:-1],
        "bin_right": edges[1:],
        "count": counts,
    })
    return hist_df, n_total, n_excluded


def regional_breakdown(df: pd.DataFrame) -> pd.DataFrame:
    """Per-region: alert count, mean duration, total duration hours."""
    agg = (
        df.groupby("region")
        .agg(
            alert_count=("started_at", "count"),
            mean_duration_min=("duration_min", "mean"),
            total_duration_h=("duration_min", lambda x: x.sum() / 60.0),
        )
        .reset_index()
        .sort_values("alert_count", ascending=False)
    )
    agg["mean_duration_min"] = agg["mean_duration_min"].round(1)
    agg["total_duration_h"] = agg["total_duration_h"].round(1)
    return agg


def top_alert_days(df: pd.DataFrame, n: int = 15) -> pd.DataFrame:
    """Return the N days with highest alert counts across all regions."""
    day_counts = (
        df.groupby("date")
        .size()
        .rename("alert_count")
        .reset_index()
        .sort_values("alert_count", ascending=False)
        .head(n)
    )
    return day_counts


def cumulative_alerts_over_time(df: pd.DataFrame) -> pd.DataFrame:
    """Cumulative alert count by day — for a 'since war started' chart."""
    daily = (
        df.set_index("started_at")
        .resample("D")
        .size()
        .rename("daily_count")
        .reset_index()
        .rename(columns={"started_at": "date"})
    )
    daily["cumulative"] = daily["daily_count"].cumsum()
    return daily


def days_since_start(df: pd.DataFrame) -> pd.Series:
    """Number of days from invasion start (2022-02-24) — 'ndays' feature."""
    origin = pd.Timestamp(COVER_START_DATE, tz="Europe/Kyiv")
    return (df["started_at"] - origin).dt.days.rename("ndays")
