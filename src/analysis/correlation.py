"""Cross-regional correlation and co-occurrence analysis.

[З ДОСЛІДЖЕННЯ] Binary series per region, lag analysis, propagation vectors.
Outputs feed the heatmap and (optionally) the attack-wave feature.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def co_occurrence_matrix(df: pd.DataFrame, window_hours: float = 2.0) -> pd.DataFrame:
    """Symmetric co-occurrence matrix using a sliding time window.

    Cell[A, B] = fraction of A-episodes that have at least one B-episode starting
    within ±window_hours.  Made symmetric by averaging both directions:
    (P(B|A) + P(A|B)) / 2.  Diagonal is NaN.

    Uses numpy searchsorted on sorted int64 timestamps — fast enough for 25 regions
    × 60 k episodes without a full O(N²) scan.
    """
    regions = sorted(df["region"].dropna().unique())
    n = len(regions)
    if n == 0:
        return pd.DataFrame()

    window_ns = int(window_hours * 3600 * 1_000_000_000)

    # Per-region sorted timestamp arrays (int64 nanoseconds for searchsorted)
    region_ts: dict[str, np.ndarray] = {}
    for r in regions:
        ts = df[df["region"] == r]["started_at"].dropna().values.astype("int64")
        region_ts[r] = np.sort(ts)

    mat = np.zeros((n, n), dtype=float)

    for i, a in enumerate(regions):
        ts_a = region_ts[a]
        n_a = len(ts_a)
        if n_a == 0:
            continue
        for j, b in enumerate(regions):
            if i == j:
                continue
            ts_b = region_ts[b]
            if len(ts_b) == 0:
                continue
            # Vectorised: for every A-alert find whether any B-alert falls in the window
            lo = np.searchsorted(ts_b, ts_a - window_ns, side="left")
            hi = np.searchsorted(ts_b, ts_a + window_ns, side="right")
            mat[i, j] = np.sum(hi > lo) / n_a

    # Symmetrize by averaging both directions
    sym = (mat + mat.T) / 2
    np.fill_diagonal(sym, np.nan)

    return pd.DataFrame(sym, index=regions, columns=regions)


def lag_correlation(
    binary_series: pd.DataFrame,
    region_a: str,
    region_b: str,
    max_lag_hours: int = 6,
) -> pd.DataFrame:
    """Pearson correlation between region_a and lagged region_b.

    Returns DataFrame with columns: lag_hours, correlation.
    Positive lag → B lags behind A (alert in A precedes alert in B).
    """
    if region_a not in binary_series.columns or region_b not in binary_series.columns:
        return pd.DataFrame(columns=["lag_hours", "correlation"])

    a = binary_series[region_a]
    b = binary_series[region_b]
    results = []
    for lag in range(-max_lag_hours, max_lag_hours + 1):
        shifted = b.shift(lag)
        corr = a.corr(shifted)
        results.append({"lag_hours": lag, "correlation": round(corr, 4)})
    return pd.DataFrame(results)


def propagation_events(
    df: pd.DataFrame,
    min_regions: int = 5,
    window_hours: float = 3.0,
) -> list[dict]:
    """Identify 'wave events': bursts where ≥min_regions distinct oblasts raise
    an alert within a rolling window of `window_hours`.

    [МОЄ РІШЕННЯ] Core of the Attack Wave Replay feature.

    Algorithm — a sliding two-pointer window over ALL alert starts sorted by
    time (not grouped by calendar day, which would wrongly merge separate
    bursts):

      1. Sort every alert by started_at.
      2. For each left edge i, extend the right edge j while
         started_at[j] - started_at[i] ≤ window_hours.
      3. If the window covers ≥ min_regions DISTINCT oblasts, emit a wave using
         the first appearance of each region, then jump i past the window so the
         returned events do not overlap.

    Returns list of dicts: {date, first_region, regions_hit, duration_h,
    sequence: [{region, started_at}]}, sorted by regions_hit desc.
    """
    if df.empty:
        return []

    s = (
        df[["region", "started_at"]]
        .dropna(subset=["started_at"])
        .sort_values("started_at")
        .reset_index(drop=True)
    )
    starts = s["started_at"].tolist()
    regions = s["region"].tolist()
    n = len(s)
    window = pd.Timedelta(hours=window_hours)

    events: list[dict] = []
    i = 0
    while i < n:
        # Extend window [i, j)
        j = i
        while j < n and (starts[j] - starts[i]) <= window:
            j += 1

        # First appearance of each distinct region within [i, j)
        first_seen: dict[str, pd.Timestamp] = {}
        for k in range(i, j):
            first_seen.setdefault(regions[k], starts[k])

        if len(first_seen) >= min_regions:
            sequence = [
                {"region": r, "started_at": t}
                for r, t in sorted(first_seen.items(), key=lambda kv: kv[1])
            ]
            span_h = (sequence[-1]["started_at"] - sequence[0]["started_at"]).total_seconds() / 3600
            events.append({
                "date": str(sequence[0]["started_at"].date()),
                "first_region": sequence[0]["region"],
                "regions_hit": len(sequence),
                "duration_h": round(span_h, 2),
                "sequence": sequence,
            })
            i = j  # non-overlapping: jump past this burst
        else:
            i += 1

    return sorted(events, key=lambda x: x["regions_hit"], reverse=True)


def regional_correlation_matrix(df: pd.DataFrame, method: str = "spearman") -> pd.DataFrame:
    """Spearman rank correlation matrix from daily alert-count vectors per region.

    Spearman is more robust than Pearson for count data: major offensive spikes
    inflate Pearson values without reflecting typical co-occurrence patterns.
    method='spearman' (default) | 'pearson' | 'kendall'
    """
    daily_counts = (
        df.groupby(["date", "region"])
        .size()
        .unstack(fill_value=0)
    )
    if daily_counts.empty:
        return pd.DataFrame()
    return daily_counts.corr(method=method).round(3)
