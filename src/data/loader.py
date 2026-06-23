"""Download and ingest the Vadimkin OFFICIAL air-raid alert CSV dataset.

[З ДОСЛІДЖЕННЯ] Source: Vadimkin/ukrainian-air-raid-sirens-dataset (GitHub),
file `official_data_uk.csv` — sourced from the government alert API.

Raw schema: oblast, raion, hromada, level, started_at, finished_at, source

────────────────────────────────────────────────────────────────────────────
KEY DESIGN DECISION — "oblast alert episodes" via interval unification
────────────────────────────────────────────────────────────────────────────
The raw file has three alert levels: oblast / raion / hromada. Around November
2025 Ukraine switched from whole-oblast to per-raion alerts, so `level="oblast"`
alone makes recent data vanish. We unify OBLAST + RAION level alerts per oblast:
overlapping/touching intervals merge into single episodes, giving one consistent
series. Hromada-level alerts are deliberately excluded:

  • Hromada (village/municipality) alerts often represent small front-line zones
    (e.g. Вовчанська hromada 604 days, Нікопільський raion hromadas 20+ days)
    that are essentially occupied/evacuated — not standard air-raid sirens.
  • Including them inflates the entire oblast's "under-alert" time unfairly.
  • Oblast+raion level is the right granularity: raion alerts handle post-2025
    coverage while being meaningful enough to represent significant areas.

Also: the raw file records every alert twice → exact duplicates are dropped.
Individual alerts > MAX_RAW_ALERT_HOURS are capped before merging.

Output schema (normalised): region, started_at, finished_at, duration_min
"""
from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Optional

import pandas as pd
import requests

from config import VADIMKIN_CSV_URL, RAW_CSV_PATH, MAX_RAW_ALERT_HOURS, NEAR_PERMANENT_REGIONS, OBLAST_ALIAS_MAP

logger = logging.getLogger(__name__)

_RAW_COLUMNS = {"oblast", "raion", "hromada", "level", "started_at", "finished_at"}
_DOWNLOAD_TIMEOUT_SEC = 120
_CHUNK_SIZE = 131_072  # 128 KB


def download_csv(
    url: str = VADIMKIN_CSV_URL,
    dest: Path = RAW_CSV_PATH,
    force: bool = False,
    max_age_hours: float = 6.0,
) -> Path:
    """Download the Vadimkin CSV to *dest*.

    Skips download if the file exists AND is younger than max_age_hours.
    Pass force=True to always re-download regardless of age.
    """
    if dest.exists() and not force:
        age_h = (time.time() - dest.stat().st_mtime) / 3600
        if age_h < max_age_hours:
            logger.info("CSV cached at %s (%.1fh old, fresh)", dest, age_h)
            return dest
        logger.info("CSV at %s is %.1fh old — re-downloading", dest, age_h)

    dest.parent.mkdir(parents=True, exist_ok=True)
    logger.info("Downloading dataset from %s …", url)
    t0 = time.perf_counter()

    with requests.get(url, stream=True, timeout=_DOWNLOAD_TIMEOUT_SEC) as r:
        r.raise_for_status()
        with open(dest, "wb") as f:
            for chunk in r.iter_content(chunk_size=_CHUNK_SIZE):
                f.write(chunk)

    elapsed = time.perf_counter() - t0
    size_mb = dest.stat().st_size / 1_048_576
    logger.info("Downloaded %.1f MB in %.1f s → %s", size_mb, elapsed, dest)
    return dest


def _find_uncovered_hromada(
    df_hromada: pd.DataFrame,
    df_ro: pd.DataFrame,
) -> pd.DataFrame:
    """Return hromada alerts not covered by any raion/oblast alert in same oblast.

    A hromada alert is "covered" if at least one raion/oblast alert for the
    same oblast overlaps its time window.  Covered alerts are redundant (already
    represented by the raion/oblast layer).  Uncovered ones are real air raids
    that only appeared at the hromada level — common in early 2022 and for
    cities like Nikopol that received targeted alerts outside the oblast cycle.

    Investigation (2026-06-22): of 19,266 hromada alerts, 80.6 % are covered
    (safe to exclude); 19.4 % (3,732) are uncovered, median duration 48 min,
    max merged episode 18.7 h — no multi-day chains, safe to include.
    """
    uncovered: list[dict] = []
    cap = pd.Timedelta(hours=MAX_RAW_ALERT_HOURS)

    for oblast, grp_h in df_hromada.groupby("oblast"):
        ro = df_ro[df_ro["oblast"] == oblast]
        if ro.empty:
            uncovered.extend(grp_h.to_dict("records"))
            continue

        ro_s = ro["started_at"].values
        ro_e = ro["finished_at"].fillna(ro["started_at"] + cap).values

        for _, row in grp_h.iterrows():
            s = row["started_at"]
            e = row["finished_at"] if pd.notna(row["finished_at"]) else s + cap
            # Overlap: s < ro_e AND e > ro_s
            if not ((s.to_datetime64() < ro_e) & (e.to_datetime64() > ro_s)).any():
                uncovered.append(row.to_dict())

    return pd.DataFrame(uncovered) if uncovered else pd.DataFrame()


def _merge_oblast_episodes(df: pd.DataFrame) -> pd.DataFrame:
    """Merge all alert intervals within each oblast into unified episodes.

    Overlapping or touching [started_at, finished_at] intervals collapse into
    one episode.  Open intervals (NaT finish) are never extended by later alerts
    — they become isolated episodes.  Combined with the MAX_RAW_ALERT_HOURS
    pre-cap in _load_and_prepare this gives C-1 variant (c) defensively.

    Note: alerts.in.ua always sets finished_at before recording, so the NaT
    path is unreachable for the current official dataset (confirmed 2026-06-23,
    0 open intervals in 138 783 oblast/raion records).
    """
    group_key = "region" if "region" in df.columns else "oblast"
    records: list[dict] = []
    for region_val, grp in df.groupby(group_key, sort=False):
        intervals = sorted(
            zip(grp["started_at"], grp["finished_at"]),
            key=lambda t: t[0],
        )
        cur_s = cur_e = None
        for s, e in intervals:
            if cur_s is None:
                cur_s, cur_e = s, e
                continue
            # Merge overlapping/touching intervals.
            # pd.notna(cur_e) is False when cur_e is NaT: open episodes never
            # absorb subsequent alerts (avoids the C-1 swallowing bug).
            if pd.notna(cur_e) and s <= cur_e:
                cur_e = e if pd.isna(e) else max(cur_e, e)
            else:
                records.append({"region": region_val, "started_at": cur_s, "finished_at": cur_e})
                cur_s, cur_e = s, e
        if cur_s is not None:
            records.append({"region": region_val, "started_at": cur_s, "finished_at": cur_e})
    return pd.DataFrame(records)


def _unit_name(row: pd.Series) -> str:
    """Compute a stable unit identifier scoped within its oblast."""
    level = row.get("level") or "oblast"
    region = row.get("region") or str(row.get("oblast", ""))
    if level == "oblast":
        return region
    elif level == "raion":
        return f"{region}::{row.get('raion') or ''}"
    else:
        return f"{region}::{row.get('hromada') or ''}"


def _load_and_prepare(path: Path) -> pd.DataFrame:
    """Load CSV through dedup, timestamp parse, cap, level split, hromada inclusion.

    Returns per-unit records (before oblast union) with two added columns:
      region    : normalized oblast name (via OBLAST_ALIAS_MAP)
      unit_name : level-scoped identifier used by coverage analysis
    """
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found at {path}. "
            "Run `python scripts/download_data.py` first."
        )

    df = pd.read_csv(
        path,
        dtype={"oblast": str, "raion": str, "hromada": str, "level": str},
        parse_dates=False,
        encoding="utf-8",
        na_values=["", "None", "NaN", "null"],
    )
    df.columns = df.columns.str.strip().str.lower()

    missing = _RAW_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(f"Missing expected columns: {missing}")

    n_total = len(df)

    # Drop exact full-row duplicates (raw file double-records every alert)
    df = df.drop_duplicates(
        subset=["oblast", "raion", "hromada", "level", "started_at", "finished_at"]
    ).copy()
    n_dedup = len(df)

    # Parse timestamps as UTC
    for col in ("started_at", "finished_at"):
        df[col] = pd.to_datetime(df[col], utc=True, errors="coerce")
    n_before_nat = len(df)
    df = df.dropna(subset=["started_at"])
    n_nat_dropped = n_before_nat - len(df)
    if n_nat_dropped > 0:
        logger.warning("Dropped %d rows with unparseable started_at", n_nat_dropped)

    # Cap individual alerts that exceed MAX_RAW_ALERT_HOURS as a safety net.
    # This handles any remaining anomalously-long records at any level.
    _cap = pd.Timedelta(hours=MAX_RAW_ALERT_HOURS)
    _has_end = df["finished_at"].notna()
    _too_long = _has_end & ((df["finished_at"] - df["started_at"]) > _cap)
    if _too_long.any():
        df.loc[_too_long, "finished_at"] = df.loc[_too_long, "started_at"] + _cap
        logger.info(
            "Capped %d raw alerts exceeding %dh",
            int(_too_long.sum()), MAX_RAW_ALERT_HOURS,
        )

    # Split by level: oblast+raion is the base; hromada is handled separately.
    # Hromada-level alerts can represent:
    #   (a) Real air raids that were only declared at the hromada level —
    #       common in early 2022 (e.g. Kryvyi Rih, Nikopol, Dnipro city in
    #       March 2022 before oblast-wide alerts were in place) and later for
    #       cities under targeted attack like Nikopol.  These should be INCLUDED.
    #   (b) Continuous "front-line zone" alerts for evacuated/occupied hromadas
    #       (e.g. Вовчанська – 604 days in occupied Vovchansk; Мирівська near
    #       Nikopol reactor zone).  These are covered by overlapping raion/oblast
    #       alerts and would cause multi-month merged episodes → EXCLUDED.
    # The rule: include a hromada alert only if NO raion/oblast alert for the
    # same oblast overlaps its time window (i.e., it adds genuinely new
    # coverage not already captured by the raion/oblast layer).
    df_ro = df[df["level"].isin(["oblast", "raion"])].copy()
    df_h  = df[df["level"] == "hromada"].copy()
    logger.info("After level split: %d oblast/raion + %d hromada", len(df_ro), len(df_h))

    if not df_h.empty:
        uncov = _find_uncovered_hromada(df_h, df_ro)
        if not uncov.empty:
            logger.info("Including %d uncovered hromada alerts", len(uncov))
            df_ro = pd.concat([df_ro, uncov], ignore_index=True)

    df_ro["region"] = df_ro["oblast"].map(
        lambda x: OBLAST_ALIAS_MAP.get(x, x) if isinstance(x, str) else x
    )
    df_ro["unit_name"] = df_ro.apply(_unit_name, axis=1)

    open_count = int(df_ro["finished_at"].isna().sum())
    if open_count > 0:
        logger.warning(
            "Found %d open intervals (no finished_at) in oblast/raion records — "
            "unexpected for alerts.in.ua data; these become isolated episodes.",
            open_count,
        )

    logger.info(
        "Prepared: %d raw → %d deduped → %d unit records",
        n_total, n_dedup, len(df_ro),
    )
    return df_ro


def _filter_permanent(df_ro: pd.DataFrame) -> pd.DataFrame:
    """Remove permanent-alert regions before union (avoids giant spurious episodes)."""
    mask = df_ro["region"].isin(NEAR_PERMANENT_REGIONS)
    n = int(mask.sum())
    if n > 0:
        logger.info("Excluding %d records from permanent regions before union", n)
        return df_ro[~mask].copy()
    return df_ro


def load_raw(path: Path = RAW_CSV_PATH) -> pd.DataFrame:
    """Load the official CSV, unify per-oblast episodes, return normalised frame."""
    df_ro = _load_and_prepare(path)
    df_ro = _filter_permanent(df_ro)  # C-5: exclude before union, not after

    episodes = _merge_oblast_episodes(df_ro)
    logger.info("Built %d unified oblast episodes", len(episodes))

    episodes["duration_min"] = (
        (episodes["finished_at"] - episodes["started_at"]).dt.total_seconds().div(60.0)
    )
    episodes = episodes.sort_values("started_at").reset_index(drop=True)
    return episodes[["region", "started_at", "finished_at", "duration_min"]]


def load_unit_records(path: Path = RAW_CSV_PATH) -> pd.DataFrame:
    """Return per-unit alert records (before oblast union) for coverage analysis.

    Each row is one raw alert record with columns:
      region    : normalized oblast name
      unit_name : level-scoped identifier (e.g. "Харківська::Харківський")
      level     : "oblast" | "raion" | "hromada"
      started_at, finished_at : UTC-aware timestamps

    Permanent regions (Luhansk, Crimea) are excluded — same scope as load_raw.
    Pass to src.analysis.coverage.episode_unit_coverage() to enrich episodes.
    """
    df_ro = _load_and_prepare(path)
    df_ro = _filter_permanent(df_ro)
    return df_ro[["region", "unit_name", "level", "started_at", "finished_at"]].copy()


def load(
    path: Optional[Path] = None,
    download_if_missing: bool = True,
) -> pd.DataFrame:
    """High-level load: download if needed or stale, then return normalised DataFrame."""
    p = path or RAW_CSV_PATH
    if download_if_missing:
        download_csv(dest=p)  # handles age-check internally; re-downloads if >6h old
    elif not p.exists():
        raise FileNotFoundError(
            f"Dataset not found at {p}. "
            "Run `python scripts/download_data.py` first."
        )
    return load_raw(p)
