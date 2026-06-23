"""Tests for analysis functions."""
from __future__ import annotations

import pandas as pd
import pytest

from src.analysis import descriptive, seasonality, correlation
from src.analysis import survival


class TestDescriptive:
    def test_alert_frequency_returns_dataframe(self, analysis_df):
        result = descriptive.alert_frequency_over_time(analysis_df, freq="D")
        assert isinstance(result, pd.DataFrame)
        assert "alert_count" in result.columns
        assert (result["alert_count"] >= 0).all()

    def test_duration_stats_keys(self, analysis_df):
        stats = descriptive.duration_stats(analysis_df)
        assert "mean_min" in stats
        assert "median_min" in stats
        assert stats["mean_min"] > 0
        assert stats["median_min"] > 0

    def test_duration_stats_order(self, analysis_df):
        stats = descriptive.duration_stats(analysis_df)
        assert stats["p25_min"] <= stats["median_min"] <= stats["p75_min"]

    def test_regional_breakdown_counts_positive(self, analysis_df):
        result = descriptive.regional_breakdown(analysis_df)
        assert (result["alert_count"] > 0).all()

    def test_regional_breakdown_no_permanent(self, analysis_df):
        from config import NEAR_PERMANENT_REGIONS
        result = descriptive.regional_breakdown(analysis_df)
        assert not result["region"].isin(NEAR_PERMANENT_REGIONS).any()

    def test_top_alert_days_length(self, analysis_df):
        result = descriptive.top_alert_days(analysis_df, n=5)
        assert len(result) <= 5

    def test_cumulative_monotone(self, analysis_df):
        result = descriptive.cumulative_alerts_over_time(analysis_df)
        assert result["cumulative"].is_monotonic_increasing


class TestSeasonality:
    def test_hourly_by_dow_shape(self, analysis_df):
        pivot = seasonality.hourly_by_dow(analysis_df)
        assert pivot.shape[0] == 24   # 24 hours
        assert pivot.shape[1] == 7    # 7 days

    def test_hourly_by_dow_non_negative(self, analysis_df):
        pivot = seasonality.hourly_by_dow(analysis_df)
        assert (pivot.values >= 0).all()

    def test_monthly_heatmap_columns(self, analysis_df):
        pivot = seasonality.monthly_heatmap(analysis_df)
        if not pivot.empty:
            assert set(pivot.columns).issubset(range(1, 13))

    def test_night_ratio_valid(self, analysis_df):
        result = seasonality.night_vs_day_ratio(analysis_df)
        assert 0 <= result["night_fraction"] <= 1
        assert result["night_count"] + result["day_count"] == len(analysis_df)

    def test_rolling_7d_monotone_index(self, analysis_df):
        result = seasonality.rolling_7d_avg(analysis_df)
        assert result["date"].is_monotonic_increasing

    def test_rolling_7d_first_six_rows_are_nan(self):
        """min_periods=7: first 6 rows must be NaN (not 1–6-day averages)."""
        from config import KYIV_TZ
        # 14 daily timestamps → 14 calendar days → 14 rows after daily resample
        dates = pd.date_range("2024-01-01", periods=14, freq="D", tz=KYIV_TZ)
        df = pd.DataFrame({
            "started_at": dates,
            "region": ["Kyiv"] * 14,
            "finished_at": dates + pd.Timedelta(hours=1),
            "duration_min": [60.0] * 14,
            "censored": [False] * 14,
        })
        result = seasonality.rolling_7d_avg(df)
        assert len(result) >= 14, f"Expected ≥14 daily rows, got {len(result)}"
        # First 6 daily bins must have NaN rolling average
        assert result["rolling_7d"].iloc[:6].isna().all(), (
            "rolling_7d_avg with min_periods=7 must produce NaN for the first 6 days"
        )
        # 7th bin (index 6) must have a value (exactly 7-day window)
        assert pd.notna(result["rolling_7d"].iloc[6])

    def test_hourly_by_region_normalized_columns_sum_to_one(self):
        """normalize=True: each region column must sum to 1.0 (or 0 if no alerts)."""
        import pytz
        from config import KYIV_TZ
        rows = []
        for h in range(24):
            rows.append({
                "started_at": pd.Timestamp("2024-01-01", tz=KYIV_TZ) + pd.Timedelta(hours=h),
                "region": "Kyiv",
                "hour": h,
            })
        for h in range(0, 12):
            rows.append({
                "started_at": pd.Timestamp("2024-01-01", tz=KYIV_TZ) + pd.Timedelta(hours=h),
                "region": "Lviv",
                "hour": h,
            })
        df = pd.DataFrame(rows)
        pivot = seasonality.hourly_by_region(df, normalize=True)
        for col in pivot.columns:
            col_sum = pivot[col].sum()
            assert abs(col_sum - 1.0) < 1e-9 or col_sum == 0.0, (
                f"Column '{col}' sums to {col_sum}, expected 1.0"
            )

    def test_hourly_by_region_raw_counts_unchanged(self):
        """normalize=False (default): values are raw integer counts, not fractions."""
        from config import KYIV_TZ
        rows = [
            {"started_at": pd.Timestamp("2024-01-01 10:00", tz=KYIV_TZ), "region": "Kyiv", "hour": 10},
            {"started_at": pd.Timestamp("2024-01-01 10:00", tz=KYIV_TZ), "region": "Kyiv", "hour": 10},
            {"started_at": pd.Timestamp("2024-01-01 11:00", tz=KYIV_TZ), "region": "Kyiv", "hour": 11},
        ]
        df = pd.DataFrame(rows)
        pivot = seasonality.hourly_by_region(df, normalize=False)
        assert pivot.loc[10, "Kyiv"] == 2
        assert pivot.loc[11, "Kyiv"] == 1


class TestCorrelation:
    def test_co_occurrence_diagonal_is_nan(self, analysis_df):
        """co_occurrence_matrix: symmetric 2-hour window co-occurrence; diagonal must be NaN."""
        mat = correlation.co_occurrence_matrix(analysis_df)
        if mat.empty:
            pytest.skip("Not enough data for co-occurrence")
        diag = [mat.loc[r, r] for r in mat.index if r in mat.columns]
        assert all(pd.isna(d) for d in diag)

    def test_co_occurrence_range(self, analysis_df):
        mat = correlation.co_occurrence_matrix(analysis_df)
        if mat.empty:
            pytest.skip("Not enough data")
        vals = mat.values.flatten()
        valid = vals[~pd.isna(vals)]
        assert (valid >= 0).all() and (valid <= 1).all()

    def test_regional_correlation_symmetric(self, analysis_df):
        mat = correlation.regional_correlation_matrix(analysis_df)
        if mat.empty:
            pytest.skip("Not enough data")
        import numpy as np
        np.testing.assert_allclose(mat.values, mat.T.values, atol=1e-10)

    def test_propagation_events_structure(self, analysis_df):
        events = correlation.propagation_events(analysis_df, min_regions=2, window_hours=10)
        for event in events:
            assert "date" in event
            assert "regions_hit" in event
            assert "sequence" in event
            assert event["regions_hit"] >= 2

    def test_propagation_window_respected(self, analysis_df):
        # Every emitted wave's sequence must span ≤ window_hours
        wh = 3.0
        events = correlation.propagation_events(analysis_df, min_regions=2, window_hours=wh)
        for event in events:
            assert event["duration_h"] <= wh + 1e-6

    def test_propagation_sequence_sorted_and_distinct(self, analysis_df):
        events = correlation.propagation_events(analysis_df, min_regions=2, window_hours=6)
        for event in events:
            times = [r["started_at"] for r in event["sequence"]]
            assert times == sorted(times)
            regs = [r["region"] for r in event["sequence"]]
            assert len(regs) == len(set(regs))  # distinct regions only

    def test_propagation_higher_threshold_fewer_events(self, analysis_df):
        lo = correlation.propagation_events(analysis_df, min_regions=2, window_hours=3)
        hi = correlation.propagation_events(analysis_df, min_regions=8, window_hours=3)
        assert len(hi) <= len(lo)

    def test_regional_correlation_spearman_symmetric(self):
        """Spearman correlation matrix must be symmetric (same invariant as Pearson)."""
        from config import KYIV_TZ
        import numpy as np
        rows = []
        for day in range(10):
            for region in ["Kyiv", "Lviv", "Odesa"]:
                rows.append({
                    "started_at": pd.Timestamp("2024-01-01", tz=KYIV_TZ) + pd.Timedelta(days=day),
                    "region": region,
                    "date": (pd.Timestamp("2024-01-01") + pd.Timedelta(days=day)).date(),
                })
        df = pd.DataFrame(rows)
        mat = correlation.regional_correlation_matrix(df, method="spearman")
        assert not mat.empty
        np.testing.assert_allclose(mat.values, mat.T.values, atol=1e-10)

    def test_regional_correlation_spearman_diagonal_is_one(self):
        """Spearman self-correlation must be 1.0 on the diagonal (requires variance)."""
        from config import KYIV_TZ
        rows = []
        # Vary alert counts per day (1–3) so Spearman has non-zero variance to rank
        for day in range(10):
            count = (day % 3) + 1  # 1, 2, 3, 1, 2, 3, ...
            for _ in range(count):
                for region in ["Kyiv", "Lviv"]:
                    rows.append({
                        "started_at": pd.Timestamp("2024-01-01", tz=KYIV_TZ) + pd.Timedelta(days=day),
                        "region": region,
                        "date": (pd.Timestamp("2024-01-01") + pd.Timedelta(days=day)).date(),
                    })
        df = pd.DataFrame(rows)
        mat = correlation.regional_correlation_matrix(df, method="spearman")
        for r in mat.index:
            assert abs(mat.loc[r, r] - 1.0) < 1e-9


# ── Known KM result used across survival tests ────────────────────────────────
# timeline:  0   30   60   90  120
# survival:  1.0  0.8  0.6  0.4  0.2
# h(t=0) = -(0.8-1.0)/(30*1.0) = 0.2/30 ≈ 0.00667
# h(t=30)= -(0.6-0.8)/(30*0.8) = 0.2/24 ≈ 0.00833
# median_remaining(elapsed=0): threshold=0.5, first crossed at t=90 → remaining=90
# median_remaining(elapsed=30): s_start=0.8, threshold=0.4, crossed at t=90 → remaining=60

_KM_SIMPLE = pd.DataFrame({
    "timeline": [0.0, 30.0, 60.0, 90.0, 120.0],
    "survival": [1.0,  0.8,  0.6,  0.4,   0.2],
    "ci_lower": [1.0,  0.7,  0.5,  0.3,   0.1],
    "ci_upper": [1.0,  0.9,  0.7,  0.5,   0.3],
    "label":    ["test"] * 5,
})


class TestHazardAtT:
    def test_empty_df_returns_none(self):
        assert survival.hazard_at_t(pd.DataFrame(), 0) is None

    def test_t_before_first_timepoint_returns_none(self):
        # timeline starts at 0; asking at t=-1 → no row with timeline <= -1
        assert survival.hazard_at_t(_KM_SIMPLE, -1) is None

    def test_t_at_last_point_returns_none(self):
        # pos+1 >= len → None (no next step to diff against)
        assert survival.hazard_at_t(_KM_SIMPLE, 120.0) is None

    def test_zero_survival_returns_none(self):
        """s_t=0 makes h(t) = -(s_next-0)/(dt*0) undefined → guard returns None."""
        km_zero = pd.DataFrame({
            "timeline": [0.0, 30.0, 60.0],
            "survival": [0.0,  0.0,  0.0],
            "ci_lower": [0.0,  0.0,  0.0],
            "ci_upper": [0.0,  0.0,  0.0],
            "label":    ["x"] * 3,
        })
        assert survival.hazard_at_t(km_zero, 0.0) is None

    def test_hazard_at_t0_value(self):
        result = survival.hazard_at_t(_KM_SIMPLE, 0.0)
        assert result is not None
        assert abs(result - 0.2 / 30.0) < 1e-9  # -(0.8-1.0)/(30*1.0)

    def test_hazard_at_t30_value(self):
        result = survival.hazard_at_t(_KM_SIMPLE, 30.0)
        assert result is not None
        assert abs(result - 0.2 / 24.0) < 1e-9  # -(0.6-0.8)/(30*0.8)

    def test_hazard_positive(self):
        # h(t) must be ≥ 0 when survival is non-increasing
        for t in [0.0, 30.0, 60.0]:
            r = survival.hazard_at_t(_KM_SIMPLE, t)
            assert r is not None and r >= 0.0

    def test_non_sequential_index_no_keyerror(self):
        # Regression for F-03: iloc[::2] creates index [0,2,4]
        km_sliced = _KM_SIMPLE.iloc[::2].copy()  # index [0,2,4]
        result = survival.hazard_at_t(km_sliced, 0.0)
        assert result is not None  # KeyError would have raised before the fix

    def test_t_between_timepoints_uses_floor(self):
        # t=45 → last row with timeline<=45 is t=30 → same as test at t=30
        r1 = survival.hazard_at_t(_KM_SIMPLE, 30.0)
        r2 = survival.hazard_at_t(_KM_SIMPLE, 45.0)
        assert r1 == r2


class TestMedianRemainingTime:
    def test_empty_df_returns_none(self):
        assert survival.median_remaining_time(pd.DataFrame(), 0) is None

    def test_elapsed_past_all_data_returns_none(self):
        assert survival.median_remaining_time(_KM_SIMPLE, 200.0) is None

    def test_survival_never_crosses_threshold_returns_none(self):
        # Survival stays high → threshold never crossed
        km_high = pd.DataFrame({
            "timeline": [0.0, 30.0, 60.0],
            "survival": [1.0,  0.9,  0.8],
            "ci_lower": [0.9,  0.8,  0.7],
            "ci_upper": [1.0,  1.0,  0.9],
            "label":    ["x"] * 3,
        })
        # s_start=1.0, threshold=0.5; survival stays at 0.8 — never crosses
        assert survival.median_remaining_time(km_high, 0.0) is None

    def test_elapsed_zero_returns_absolute_median(self):
        result = survival.median_remaining_time(_KM_SIMPLE, 0.0)
        # s_start=1.0, threshold=0.5; first row with survival<=0.5 is t=90 (s=0.4)
        assert result is not None
        assert abs(result - 90.0) < 1e-9

    def test_elapsed_30_returns_remaining(self):
        result = survival.median_remaining_time(_KM_SIMPLE, 30.0)
        # past starts at t=30 (s=0.8); threshold=0.4; crossed at t=90 → remaining=60
        assert result is not None
        assert abs(result - 60.0) < 1e-9

    def test_remaining_decreases_as_elapsed_increases(self):
        r0 = survival.median_remaining_time(_KM_SIMPLE, 0.0)
        r30 = survival.median_remaining_time(_KM_SIMPLE, 30.0)
        assert r0 is not None and r30 is not None
        assert r30 < r0  # waited 30 min → less remaining
