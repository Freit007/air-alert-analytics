"""Per-episode unit-coverage metrics.

Answers: during an alert episode, how many distinct administrative units
(raions / hromadas) were actively alerting within the oblast?

§4B of the aggregation spec also calls for area_fraction and pop_fraction.
Those require an external HIERARCHY table (area_km2, population per unit)
which is not in the current dataset; the fields are omitted until an
administrative geometry / demographics source is integrated.
"""
from __future__ import annotations

import pandas as pd


def episode_unit_coverage(
    unit_records: pd.DataFrame,
    episodes: pd.DataFrame,
) -> pd.DataFrame:
    """Add unit-coverage metrics to each episode.

    Parameters
    ----------
    unit_records : output of loader.load_unit_records() — pre-union per-unit
                   records with columns: region, unit_name, started_at, finished_at
    episodes     : output of load_raw() + apply_all() — oblast episodes with
                   columns: region, started_at, finished_at

    Returns
    -------
    Copy of *episodes* with two added columns:

    unit_count    (int)
        Distinct unit names active during this episode (overlap ≥ 1 second).

    unit_fraction (float, 0–1)
        unit_count divided by the total distinct units ever seen in this
        oblast across the entire dataset.  This is a proxy denominator —
        the true administrative unit count would require external data.
    """
    if unit_records.empty or episodes.empty:
        result = episodes.copy()
        result["unit_count"] = 0
        result["unit_fraction"] = 0.0
        return result

    # Proxy denominator: total distinct units per oblast in the full dataset
    total_per_oblast: dict[str, int] = (
        unit_records.groupby("region")["unit_name"].nunique().to_dict()
    )

    # Process per-oblast to avoid re-filtering the full unit_records every row
    result_parts: list[pd.DataFrame] = []

    for oblast, ep_grp in episodes.groupby("region"):
        ur = unit_records[unit_records["region"] == oblast]
        n_total = total_per_oblast.get(oblast, 1)

        unit_counts: list[int] = []
        for _, ep in ep_grp.iterrows():
            if ur.empty:
                unit_counts.append(0)
                continue

            ep_start = ep["started_at"]
            ep_end = ep["finished_at"]

            if pd.isna(ep_end):
                # Open episode: include units that started at or after episode start
                active = ur[ur["started_at"] >= ep_start]
            else:
                # Closed episode: overlap means unit_start < ep_end AND unit_end > ep_start
                unit_ends = ur["finished_at"].fillna(pd.Timestamp.now(tz="UTC"))
                active = ur[
                    (ur["started_at"] < ep_end) &
                    (unit_ends > ep_start)
                ]

            unit_counts.append(active["unit_name"].nunique())

        part = ep_grp.copy()
        part["unit_count"] = unit_counts
        part["unit_fraction"] = (part["unit_count"] / n_total).round(4)
        result_parts.append(part)

    if not result_parts:
        result = episodes.copy()
        result["unit_count"] = 0
        result["unit_fraction"] = 0.0
        return result

    return pd.concat(result_parts).sort_index().reset_index(drop=True)
