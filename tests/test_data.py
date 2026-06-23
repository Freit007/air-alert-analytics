"""Tests for data ingestion and transformation pipeline.

Covers the data-correctness rules for the official dataset, unified into
per-oblast alert episodes:
  • UNIFICATION  — oblast + raion alerts merge into oblast episodes
                   (hromada-level excluded: front-line village alerts skew totals)
  • DEDUP        — exact-duplicate raw rows dropped
  • TIMEZONE     — UTC → Europe/Kyiv with DST
  • CENSORING    — open intervals flagged
  • PERMANENT    — Luhansk/Crimea excluded
"""
from __future__ import annotations

import pandas as pd
import pytest
import pytz

from config import NEAR_PERMANENT_REGIONS, KYIV_TZ, OBLAST_ALIAS_MAP


# ── Loader ────────────────────────────────────────────────────────────────────

class TestLoader:
    def test_returns_dataframe(self, raw_df):
        assert isinstance(raw_df, pd.DataFrame)

    def test_expected_columns(self, raw_df):
        assert {"region", "started_at", "finished_at", "duration_min"}.issubset(
            raw_df.columns
        )

    def test_no_legacy_columns(self, raw_df):
        for col in ("naive", "level", "raion", "hromada"):
            assert col not in raw_df.columns

    def test_started_at_utc_aware(self, raw_df):
        assert pd.api.types.is_datetime64_any_dtype(raw_df["started_at"])
        assert raw_df["started_at"].dt.tz is not None

    def test_duration_min_non_negative(self, raw_df):
        valid = raw_df["duration_min"].dropna()
        assert (valid >= 0).all()

    def test_alias_normalization(self, transformed_df):
        assert "Харківська область" not in transformed_df["region"].values
        assert "Харківська" in transformed_df["region"].values

    def test_no_finished_before_started(self, raw_df):
        complete = raw_df.dropna(subset=["started_at", "finished_at"])
        bad = complete[complete["finished_at"] < complete["started_at"]]
        assert len(bad) == 0


# ── UNIFICATION: oblast + raion + hromada merge into oblast episodes ──────────

class TestUnification:
    def _kharkiv(self, raw_df):
        return raw_df[raw_df["region"] == "Харківська"].sort_values("started_at")

    def test_oblast_and_raion_merge(self, raw_df):
        """Kharkiv 2022-03-15: oblast 02:00–03:30 + raion 02:45–04:00 → merged
        02:00–04:00. The duplicate oblast row collapses.
        Hromada at 05:00–05:30 has no overlapping raion/oblast alert → it is an
        uncovered hromada alert and is correctly INCLUDED as a second episode."""
        kh = self._kharkiv(raw_df)
        first_day = kh[kh["started_at"].dt.strftime("%Y-%m-%d") == "2022-03-15"].sort_values("started_at")
        # Two episodes: merged raion/oblast (02:00-04:00) + uncovered hromada (05:00-05:30)
        assert len(first_day) == 2
        ep1 = first_day.iloc[0]
        assert ep1["started_at"].strftime("%H:%M") == "02:00"
        assert ep1["finished_at"].strftime("%H:%M") == "04:00"
        ep2 = first_day.iloc[1]
        assert ep2["started_at"].strftime("%H:%M") == "05:00"
        assert ep2["finished_at"].strftime("%H:%M") == "05:30"

    def test_raion_era_alerts_unify(self, raw_df):
        """Two overlapping Sumy RAION alerts (Dec 2025) → one oblast episode.
        This is the late-2025 raion-switch case that the unification fixes."""
        sumy = raw_df[raw_df["region"] == "Сумська"]
        assert len(sumy) == 1
        ep = sumy.iloc[0]
        assert ep["started_at"].strftime("%H:%M") == "02:05"
        assert ep["finished_at"].strftime("%H:%M") == "03:20"


# ── DEDUP ─────────────────────────────────────────────────────────────────────

class TestDedup:
    def test_no_exact_duplicate_episodes(self, raw_df):
        dups = raw_df.duplicated(subset=["region", "started_at", "finished_at"]).sum()
        assert dups == 0


# ── TIMEZONE ──────────────────────────────────────────────────────────────────

class TestTimezonePitfall:
    def test_started_at_is_kyiv(self, transformed_df):
        tz = str(transformed_df["started_at"].dt.tz)
        assert tz in (KYIV_TZ, "Europe/Kiev") or "Europe" in tz

    def test_temporal_features_exist(self, transformed_df):
        for col in ("hour", "dow", "month", "year", "date"):
            assert col in transformed_df.columns


# ── CENSORING ─────────────────────────────────────────────────────────────────

class TestCensoringPitfall:
    def test_censored_column_exists(self, transformed_df):
        assert "censored" in transformed_df.columns

    def test_closed_intervals_not_censored(self, transformed_df):
        closed = transformed_df[transformed_df["finished_at"].notna()]
        assert (~closed["censored"]).all()


# ── PERMANENT regions ─────────────────────────────────────────────────────────

class TestPermanentRegionsPitfall:
    def test_permanent_regions_excluded_before_union(self, raw_df):
        """C-5: _filter_permanent runs before _merge_oblast_episodes, so no permanent
        region episodes must appear in load_raw() output."""
        found = set(raw_df["region"]) & NEAR_PERMANENT_REGIONS
        assert not found, f"Permanent-region episodes leaked into load_raw output: {found}"

    def test_permanent_regions_excluded_from_analysis(self, transformed_df):
        remaining = transformed_df["region"].isin(NEAR_PERMANENT_REGIONS)
        assert not remaining.any()


# ── make_binary_series ────────────────────────────────────────────────────────

class TestBinarySeries:
    """make_binary_series: binary presence matrix per time bucket."""

    def _ts(self, dt_str: str) -> pd.Timestamp:
        return pd.Timestamp(dt_str, tz=KYIV_TZ)

    def _df(self, rows: list[dict]) -> pd.DataFrame:
        df = pd.DataFrame(rows)
        df["started_at"] = pd.to_datetime(df["started_at"]).dt.tz_localize(KYIV_TZ)
        df["finished_at"] = pd.to_datetime(df["finished_at"]).dt.tz_localize(KYIV_TZ)
        return df

    def test_closed_interval_marks_correct_buckets(self):
        from src.data.transforms import make_binary_series
        df = self._df([
            {"region": "Kyiv", "started_at": "2024-01-01 10:00", "finished_at": "2024-01-01 12:00"},
        ])
        result = make_binary_series(df, freq="1h")
        assert result.loc[self._ts("2024-01-01 10:00"), "Kyiv"] == 1
        assert result.loc[self._ts("2024-01-01 11:00"), "Kyiv"] == 1
        # 12:00 bucket starts at the ceil — should not be marked
        assert result.loc[self._ts("2024-01-01 12:00"), "Kyiv"] == 0

    def test_mixed_nat_and_closed_no_crash(self):
        """make_binary_series must not crash when some finished_at are NaT."""
        from src.data.transforms import make_binary_series
        df = self._df([
            {"region": "Kyiv",   "started_at": "2024-01-01 10:00", "finished_at": "2024-01-01 11:00"},
            {"region": "Lviv",   "started_at": "2024-01-01 10:30", "finished_at": None},   # NaT
        ])
        result = make_binary_series(df, freq="1h")
        assert isinstance(result, pd.DataFrame)
        assert "Kyiv" in result.columns
        assert "Lviv" in result.columns

    def test_all_nat_fallback_uses_started_at_max(self):
        """When ALL finished_at are NaT, t_max must fall back to started_at.max()."""
        from src.data.transforms import make_binary_series
        df = self._df([
            {"region": "Kyiv", "started_at": "2024-01-01 10:00", "finished_at": None},
            {"region": "Kyiv", "started_at": "2024-01-01 14:00", "finished_at": None},
        ])
        result = make_binary_series(df, freq="1h")
        # t_max derived from started_at.max()=14:00.ceil(1h)=14:00; index should exist
        assert self._ts("2024-01-01 14:00") in result.index

    def test_result_binary_values(self):
        """All values in the matrix must be 0 or 1."""
        from src.data.transforms import make_binary_series
        df = self._df([
            {"region": "Kyiv", "started_at": "2024-01-01 08:00", "finished_at": "2024-01-01 10:00"},
            {"region": "Lviv", "started_at": "2024-01-01 09:00", "finished_at": None},
        ])
        result = make_binary_series(df, freq="1h")
        assert set(result.values.flatten()).issubset({0, 1})
