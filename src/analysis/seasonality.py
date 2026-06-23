"""Temporal pattern analysis: heatmaps by hour, day-of-week, month.

[З ДОСЛІДЖЕННЯ] Nightly Shahed attacks 20:00–06:00 Kyiv time; weekly logistics cycles.
All functions return DataFrames ready for Plotly heatmap rendering.
"""
from __future__ import annotations

import pandas as pd
import numpy as np

_DOW_ORDER = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
_DOW_SHORT = ["Пн", "Вт", "Ср", "Чт", "Пт", "Сб", "Нд"]
_DOW_EN_TO_UK = dict(zip(_DOW_ORDER, _DOW_SHORT))


def hourly_by_dow(df: pd.DataFrame) -> pd.DataFrame:
    """Count matrix: rows=hour (0–23), cols=day-of-week.

    Normalised by number of weeks in the dataset to show 'alerts per week'
    rather than raw counts (avoids dataset-length bias).
    """
    if df.empty:
        return pd.DataFrame()

    n_weeks = max(1, (df["started_at"].max() - df["started_at"].min()).days / 7)
    pivot = (
        df.groupby(["hour", "dow_name"])
        .size()
        .unstack(fill_value=0)
    )
    # Reindex BOTH axes: always return a full 24×7 matrix
    pivot = pivot.reindex(index=range(24), columns=_DOW_ORDER, fill_value=0)
    pivot = pivot.div(n_weeks).round(2)
    pivot.index.name = "hour"
    pivot.columns = [_DOW_EN_TO_UK.get(c, c) for c in pivot.columns]
    return pivot


def monthly_heatmap(df: pd.DataFrame) -> pd.DataFrame:
    """Count matrix: rows=year, cols=month (1–12)."""
    if df.empty:
        return pd.DataFrame()

    pivot = (
        df.groupby(["year", "month"])
        .size()
        .unstack(fill_value=0)
        .reindex(columns=range(1, 13), fill_value=0)
    )
    return pivot


def hourly_by_region(
    df: pd.DataFrame,
    regions: list[str] | None = None,
    normalize: bool = False,
) -> pd.DataFrame:
    """Count (or fraction) matrix: rows=hour (0–23), cols=region.

    normalize=False (default): raw counts — total alerts starting at each hour.
    normalize=True:  fraction of each region's own alerts — removes volume bias
                     so front-line regions don't visually dominate over quiet ones.
                     Each column sums to 1.0 (or to 0 if no alerts in that region).
    """
    data = df.copy()
    if regions:
        data = data[data["region"].isin(regions)]

    pivot = (
        data.groupby(["hour", "region"])
        .size()
        .unstack(fill_value=0)
    )
    if regions:
        pivot = pivot.reindex(columns=regions, fill_value=0)
    if normalize:
        col_sums = pivot.sum(axis=0)
        pivot = pivot.div(col_sums.replace(0, 1))
    return pivot


def duration_by_hour(df: pd.DataFrame) -> pd.DataFrame:
    """Mean duration (minutes) per hour-of-day — for a bar/line chart."""
    clean = df[~df["censored"].fillna(False)].copy()
    return (
        clean.groupby("hour")["duration_min"]
        .mean()
        .rename("mean_duration_min")
        .round(1)
        .reset_index()
    )


def night_vs_day_ratio(df: pd.DataFrame) -> dict:
    """Fraction of alerts that start at night (20:00–06:00 Kyiv).

    [З ДОСЛІДЖЕННЯ] Night attacks are characteristic of Shahed drones.
    """
    night = df["hour"].apply(lambda h: h >= 20 or h < 6)
    total = max(len(df), 1)
    return {
        "night_count": int(night.sum()),
        "day_count": int((~night).sum()),
        "night_fraction": float(night.sum() / total),
    }


def rolling_7d_avg(df: pd.DataFrame) -> pd.DataFrame:
    """Daily alert count + 7-day rolling average for trend chart."""
    daily = (
        df.set_index("started_at")
        .resample("D")
        .size()
        .rename("count")
        .reset_index()
        .rename(columns={"started_at": "date"})
    )
    # min_periods=7: first 6 days produce NaN — shown as a gap, not a misleading 1–6-day average
    daily["rolling_7d"] = daily["count"].rolling(7, min_periods=7).mean().round(1)
    return daily
