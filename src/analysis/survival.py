"""Survival analysis for alert duration.

[З ДОСЛІДЖЕННЯ] S(t) = probability alert is still active after t minutes.
Open intervals (finished_at is null) → right-censored observations.

Requires: lifelines
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd

try:
    from lifelines import KaplanMeierFitter
    _LIFELINES_AVAILABLE = True
except ImportError:
    _LIFELINES_AVAILABLE = False


def check_lifelines() -> None:
    if not _LIFELINES_AVAILABLE:
        raise ImportError(
            "lifelines is required for survival analysis. "
            "Install with: pip install lifelines"
        )


def kaplan_meier(
    df: pd.DataFrame,
    label: str = "all",
    timeline: Optional[np.ndarray] = None,
) -> pd.DataFrame:
    """Fit KM estimator; return DataFrame with columns:
    timeline, survival, ci_lower, ci_upper, label.

    df must contain duration_min and censored columns.
    censored=True means the interval was cut short (no all-clear).
    """
    check_lifelines()

    data = df.copy()
    data = data[data["duration_min"].notna() & (data["duration_min"] >= 0)]
    if data.empty:
        return pd.DataFrame()

    # lifelines: event=1 means the event DID happen (alert ended with all-clear)
    data["event"] = (~data["censored"].fillna(False)).astype(int)

    kmf = KaplanMeierFitter()
    kmf.fit(
        durations=data["duration_min"],
        event_observed=data["event"],
        timeline=timeline,
        label=label,
    )

    sf = kmf.survival_function_
    ci = kmf.confidence_interval_survival_function_

    result = pd.DataFrame({
        "timeline": sf.index,
        "survival": sf.iloc[:, 0].values,
        "ci_lower": ci.iloc[:, 0].values,
        "ci_upper": ci.iloc[:, 1].values,
        "label": label,
    })
    return result


def kaplan_meier_stratified(
    df: pd.DataFrame,
    strata_col: str = "hour",
    groups: Optional[list] = None,
    min_observations: int = 10,
) -> pd.DataFrame:
    """KM curves stratified by strata_col (e.g. hour, dow, region).

    Groups with fewer than min_observations entries are skipped — the KM
    estimator is unstable with very small n (wide CIs, unreliable steps).
    """
    check_lifelines()

    if groups is None:
        groups = sorted(df[strata_col].dropna().unique().tolist())

    frames = []
    for grp_val in groups:
        subset = df[df[strata_col] == grp_val]
        if len(subset) < min_observations:
            continue
        km = kaplan_meier(subset, label=str(grp_val))
        frames.append(km)

    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def hazard_at_t(km_result: pd.DataFrame, t_minutes: float) -> Optional[float]:
    """Conditional probability that alert ends in next minute given it's lasted t.

    h(t) = -d/dt ln(S(t)) approximated by finite difference.
    Returns None if t is out of range.
    """
    if km_result.empty:
        return None

    row = km_result[km_result["timeline"] <= t_minutes]
    if row.empty:
        return None

    idx = row.index[-1]
    pos = km_result.index.get_loc(idx)
    s_t = km_result.iloc[pos]["survival"]
    if pos + 1 >= len(km_result):
        return None

    s_next = km_result.iloc[pos + 1]["survival"]
    dt = km_result.iloc[pos + 1]["timeline"] - km_result.iloc[pos]["timeline"]
    if dt <= 0 or s_t <= 0 or s_next > s_t:
        return None

    return float(-(s_next - s_t) / (dt * s_t))


def median_remaining_time(km_result: pd.DataFrame, elapsed_min: float) -> Optional[float]:
    """Estimate median remaining alert duration given 'elapsed_min' have passed.

    Returns None if survival never drops below 0.5 past elapsed_min.
    """
    if km_result.empty:
        return None

    past = km_result[km_result["timeline"] >= elapsed_min]
    if past.empty:
        return None

    s_start = past.iloc[0]["survival"]
    threshold = s_start * 0.5  # half of remaining survival mass

    crossed = past[past["survival"] <= threshold]
    if crossed.empty:
        return None

    total_median = crossed.iloc[0]["timeline"]
    return float(total_median - elapsed_min)
