"""Adversarial data-pipeline tests — QA audit phase D.

Each test corresponds to a finding (D-XX). Tests are named to be maximally
self-documenting: they test DATA CORRECTNESS, not format compliance.
"""
from __future__ import annotations

import io
import os
import tempfile
from pathlib import Path

import pandas as pd
import pytest
import pytz

from config import KYIV_TZ


# ── D-01 (FIXED): make_binary_series — open alert after latest closed end ─────
class TestBinarySeriesOpenAlertAfterTmax:
    """D-01 was Critical: open-ended alert starting after t_max was silent.
    FIX: t_max = max(last_closed_end, last_open_start + 1_period), so the
    open alert's start bucket is always present in the time index.
    """

    def test_open_alert_after_closed_end_is_marked(self):
        from src.data.transforms import make_binary_series

        # Kyiv: closed 10:00–11:00
        # Lviv: open alert starting at 14:00 (after t_max_closed=11:00)
        df = pd.DataFrame({
            "region": ["Kyiv", "Lviv"],
            "started_at": [
                pd.Timestamp("2024-01-01 10:00", tz=KYIV_TZ),
                pd.Timestamp("2024-01-01 14:00", tz=KYIV_TZ),
            ],
            "finished_at": [
                pd.Timestamp("2024-01-01 11:00", tz=KYIV_TZ),
                pd.NaT,
            ],
        })
        result = make_binary_series(df, freq="1h")

        # After fix: Lviv's 14:00 bucket must be in the index and marked 1
        assert pd.Timestamp("2024-01-01 14:00", tz=KYIV_TZ) in result.index
        assert result.loc[pd.Timestamp("2024-01-01 14:00", tz=KYIV_TZ), "Lviv"] == 1

    def test_closed_alert_unaffected_by_later_open(self):
        from src.data.transforms import make_binary_series

        df = pd.DataFrame({
            "region": ["Kyiv", "Lviv"],
            "started_at": [
                pd.Timestamp("2024-01-01 10:00", tz=KYIV_TZ),
                pd.Timestamp("2024-01-01 14:00", tz=KYIV_TZ),
            ],
            "finished_at": [
                pd.Timestamp("2024-01-01 11:00", tz=KYIV_TZ),
                pd.NaT,
            ],
        })
        result = make_binary_series(df, freq="1h")
        # Kyiv's bucket is still correctly marked
        assert result.loc[pd.Timestamp("2024-01-01 10:00", tz=KYIV_TZ), "Kyiv"] == 1
        assert result.loc[pd.Timestamp("2024-01-01 11:00", tz=KYIV_TZ), "Kyiv"] == 0


# ── D-05 (CORRECT behavior documented): adjacent hromada is a new event ───────
class TestFindUncoveredHromadaBoundary:
    """D-05: a hromada alert starting exactly when the raion alert ends is
    treated as 'uncovered' (included as a new episode). This is CORRECT:
    the raion alert ended; the hromada alert is a new, separate event.

    Note: _merge_oblast_episodes uses s <= cur_e (touching = merge) because
    there the goal is to collapse a continuous burst into one span.
    _find_uncovered_hromada uses s < ro_e (strict) because its goal is to
    check whether the hromada is *during* an active raion/oblast alert.
    The two functions have different semantic purposes — not an inconsistency.
    """

    def test_adjacent_hromada_is_new_uncovered_event(self):
        from src.data.loader import _find_uncovered_hromada

        df_ro = pd.DataFrame({
            "oblast": ["Kyiv"],
            "started_at": [pd.Timestamp("2024-01-01 10:00", tz="UTC")],
            "finished_at": [pd.Timestamp("2024-01-01 12:00", tz="UTC")],
            "level": ["raion"],
        })
        df_h = pd.DataFrame({
            "oblast": ["Kyiv"],
            "started_at": [pd.Timestamp("2024-01-01 12:00", tz="UTC")],  # starts AT raion end
            "finished_at": [pd.Timestamp("2024-01-01 13:00", tz="UTC")],
            "level": ["hromada"],
        })
        uncov = _find_uncovered_hromada(df_h, df_ro)
        # Adjacent (raion just ended) → new event → correctly included
        assert len(uncov) == 1

    def test_merge_treats_touching_as_same_burst(self):
        from src.data.loader import _merge_oblast_episodes

        df = pd.DataFrame({
            "oblast": ["Kyiv", "Kyiv"],
            "started_at": [
                pd.Timestamp("2024-01-01 10:00", tz="UTC"),
                pd.Timestamp("2024-01-01 12:00", tz="UTC"),
            ],
            "finished_at": [
                pd.Timestamp("2024-01-01 12:00", tz="UTC"),
                pd.Timestamp("2024-01-01 14:00", tz="UTC"),
            ],
        })
        result = _merge_oblast_episodes(df)
        # _merge_oblast_episodes merges touching intervals (s <= cur_e)
        assert len(result) == 1, "touching episodes must merge"


# ── D-06: load_raw — invalid started_at silently dropped without log ───────────
class TestLoadRawSilentNaTDrop:
    """D-06 Major: rows with unparseable started_at are silently dropped.
    pd.to_datetime(..., errors='coerce') converts bad timestamps to NaT,
    then df.dropna(subset=['started_at']) drops them without any log.
    """

    def test_invalid_started_at_silently_dropped(self):
        from src.data.loader import load_raw

        csv_text = (
            "oblast,raion,hromada,level,started_at,finished_at,source\n"
            "Харківська область,,,oblast,2022-03-15 02:00:00+00:00,2022-03-15 03:30:00+00:00,official\n"
            "Харківська область,,,oblast,NOT_A_DATE,2022-03-15 03:30:00+00:00,official\n"
            # duplicate to satisfy dedup
            "Харківська область,,,oblast,2022-03-15 02:00:00+00:00,2022-03-15 03:30:00+00:00,official\n"
        )
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            f.write(csv_text)
            fname = f.name

        try:
            result = load_raw(Path(fname))
            # The row with NOT_A_DATE is silently gone — no warning, no counter
            # NOTE: this test DOCUMENTS the silent-drop behavior, not asserts it is correct.
            assert len(result) == 1, (
                "Row with invalid started_at 'NOT_A_DATE' is silently dropped. "
                "Expected: at minimum, a WARNING log entry with the count of dropped rows."
            )
        finally:
            os.unlink(fname)


# ── D-09 (FIXED): hazard_at_t — returns None for non-monotone KM ─────────────
class TestHazardNonMonotoneKM:
    """D-09 was Major: non-monotone KM (s_next > s_t) produced negative hazard.
    FIX: added guard `if s_next > s_t: return None`.
    """

    def test_nonmonotone_km_returns_none(self):
        from src.analysis.survival import hazard_at_t

        km_nonmono = pd.DataFrame({
            "timeline": [0.0, 30.0, 60.0],
            "survival": [1.0, 0.6, 0.8],  # illegal: 0.6 → 0.8 (increases)
            "ci_lower": [0.9, 0.5, 0.7],
            "ci_upper": [1.0, 0.7, 0.9],
            "label": ["x"] * 3,
        })
        h = hazard_at_t(km_nonmono, 30.0)
        # After fix: non-monotone → None, not a negative float
        assert h is None

    def test_monotone_km_still_returns_value(self):
        from src.analysis.survival import hazard_at_t

        km_mono = pd.DataFrame({
            "timeline": [0.0, 30.0, 60.0],
            "survival": [1.0, 0.8, 0.6],
            "ci_lower": [0.9, 0.7, 0.5],
            "ci_upper": [1.0, 0.9, 0.7],
            "label": ["x"] * 3,
        })
        h = hazard_at_t(km_mono, 30.0)
        assert h is not None and h > 0


# ── D-08 (FIXED): EXCLUDED_REGIONS dead constant removed ──────────────────────
class TestExcludedRegionsRemoved:
    """D-08 was Minor: EXCLUDED_REGIONS duplicated NEAR_PERMANENT_REGIONS exactly
    and was never used. FIX: constant removed from config.py.
    """

    def test_excluded_regions_not_in_config(self):
        import config
        assert not hasattr(config, "EXCLUDED_REGIONS"), (
            "EXCLUDED_REGIONS was a dead constant identical to NEAR_PERMANENT_REGIONS. "
            "It has been removed to prevent future divergence."
        )

    def test_near_permanent_regions_still_present(self):
        from config import NEAR_PERMANENT_REGIONS
        assert "Луганська" in NEAR_PERMANENT_REGIONS
        assert len(NEAR_PERMANENT_REGIONS) >= 2


# ── D-10: duration_stats — KeyError when censored column absent ───────────────
class TestDurationStatsContract:
    """D-10 Major: duration_stats raises KeyError when the 'censored' column
    is absent. The function's docstring does NOT document this prerequisite.
    Callers who pass a DataFrame from load_raw (pre-transform) will crash.
    """

    def test_missing_censored_column_raises_valueerror(self):
        from src.analysis.descriptive import duration_stats

        df = pd.DataFrame({"duration_min": [60.0, 120.0]})  # no censored col
        with pytest.raises(ValueError, match="censored"):
            duration_stats(df)

    def test_all_censored_returns_empty_dict(self):
        from src.analysis.descriptive import duration_stats

        df = pd.DataFrame({
            "duration_min": [60.0, 120.0],
            "censored": [True, True],
        })
        result = duration_stats(df)
        # Caller must handle empty dict — no error, but no keys either
        assert result == {}, "all-censored df must return empty dict, not partial stats"


# ── D-M5: zero-duration alert passes validation ──────────────────────────────
class TestZeroDurationValidation:
    """D-M5 Trivial: _check_negative_durations only rejects duration < 0.
    A duration_min == 0 (instantaneous alert) passes unchallenged.
    Semantically: an air-raid alert that lasted 0 minutes is likely a data artefact
    (start/end timestamps are equal) and should trigger at least a warning.
    """

    def test_zero_duration_passes_negative_check(self):
        from src.data.validators import _check_negative_durations

        df = pd.DataFrame({
            "region": ["Kyiv"],
            "started_at": [pd.Timestamp("2024-01-01 10:00", tz="UTC")],
            "finished_at": [pd.Timestamp("2024-01-01 10:00", tz="UTC")],
            "duration_min": [0.0],
        })
        # This should probably warn, but currently passes silently
        _check_negative_durations(df)  # no exception raised

    def test_negative_duration_raises(self):
        from src.data.validators import _check_negative_durations

        df = pd.DataFrame({"duration_min": [-1.0], "region": ["Kyiv"],
                           "started_at": [pd.Timestamp("2024-01-01", tz="UTC")],
                           "finished_at": [pd.Timestamp("2024-01-01", tz="UTC")]})
        with pytest.raises(ValueError):
            _check_negative_durations(df)


# ── D-M3: propagation_events — adjacent waves merged into one super-event ─────
class TestPropagationAdjacentWavesMerge:
    """D-M3 Major (analytical): two consecutive attack waves that fall within
    the same window_hours are merged into one event because i=j skips past
    the combined window after the first qualifying burst.

    Two distinct groups of 5 regions → reported as ONE event of 10 regions.
    The regions_hit count is inflated, obscuring the two separate waves.
    """

    def test_two_overlapping_waves_merged_to_one(self):
        from src.analysis.correlation import propagation_events

        base = pd.Timestamp("2024-01-01 10:00", tz=KYIV_TZ)
        rows = []
        # Wave 1: 5 regions 10:00–11:20
        for i in range(5):
            rows.append({"region": f"Oblast-{i}", "started_at": base + pd.Timedelta(minutes=i * 20)})
        # Wave 2: 5 more regions 10:30–11:50 (overlaps wave 1 in time)
        for i in range(5, 10):
            rows.append({"region": f"Oblast-{i}", "started_at": base + pd.Timedelta(minutes=30 + (i - 5) * 20)})

        df = pd.DataFrame(rows)
        evts = propagation_events(df, min_regions=5, window_hours=3.0)

        # Bug: 1 event with 10 regions reported instead of 2 events of 5 each
        assert len(evts) == 1, f"expected 1 merged event, got {len(evts)}"
        assert evts[0]["regions_hit"] == 10, (
            "Two waves of 5 regions each are merged into one 10-region event. "
            "The non-overlap i=j jump causes the second wave to be absorbed."
        )
