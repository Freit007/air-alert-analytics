"""Tests for src/ui/charts.py visual helpers."""
from __future__ import annotations

import pandas as pd
import pytest
import pytz

from config import KYIV_TZ
from src.ui.charts import gantt_timeline, km_median_bar, correlation_heatmap, heatmap_region_hour


def _make_gantt_df(n_regions: int, rows_per_region: int) -> pd.DataFrame:
    """Build a synthetic alert DataFrame with n_regions × rows_per_region rows."""
    tz = pytz.timezone(KYIV_TZ)
    records = []
    base = pd.Timestamp("2024-01-01 00:00", tz=tz)
    for r in range(n_regions):
        region = f"Область-{r:02d}"
        for i in range(rows_per_region):
            start = base + pd.Timedelta(hours=r * 2 + i * 24)
            records.append({
                "region": region,
                "started_at": start,
                "finished_at": start + pd.Timedelta(hours=1),
                "duration_min": 60.0,
            })
    return pd.DataFrame(records)


class TestGanttTimeline:
    def test_all_regions_present_when_over_limit(self):
        """With >400 total rows across 25 regions (16 rows/region), every region
        must appear in fig.layout.yaxis.ticktext after stratified downsampling."""
        n_regions = 25
        df = _make_gantt_df(n_regions=n_regions, rows_per_region=20)
        assert len(df) > 400

        fig = gantt_timeline(df, max_rows=400)

        tick_texts = list(fig.layout.yaxis.ticktext)
        expected = {f"Область-{r:02d}" for r in range(n_regions)}
        missing = expected - set(tick_texts)
        assert not missing, f"Regions missing from Gantt y-axis: {sorted(missing)}"

    def test_under_limit_keeps_all_rows(self):
        """When total rows ≤ max_rows, no downsampling occurs and all rows appear."""
        df = _make_gantt_df(n_regions=5, rows_per_region=3)
        assert len(df) <= 400

        fig = gantt_timeline(df, max_rows=400)
        assert len(fig.layout.yaxis.ticktext) == 5

    def test_empty_df_returns_figure(self):
        """Empty input must not raise — returns an empty placeholder figure."""
        fig = gantt_timeline(pd.DataFrame(
            columns=["region", "started_at", "finished_at", "duration_min"]
        ))
        assert fig is not None


def _make_km_df(labels_and_survivals: list[tuple[str, list[float]]]) -> pd.DataFrame:
    """Build a KM DataFrame from (label, [s0, s1, s2, ...]) pairs."""
    rows = []
    for label, survivals in labels_and_survivals:
        for i, s in enumerate(survivals):
            rows.append({
                "label": label,
                "timeline": float(i * 30),
                "survival": s,
                "ci_lower": s * 0.9,
                "ci_upper": min(1.0, s * 1.1),
            })
    return pd.DataFrame(rows)


class TestKmMedianBar:
    def test_uncapped_bar_shows_numeric_label(self):
        """Bar whose curve crosses 0.5 shows plain integer label (e.g. '90')."""
        # survival: 1.0, 0.8, 0.6, 0.4 → crosses 0.5 at t=90
        km = _make_km_df([("GroupA", [1.0, 0.8, 0.6, 0.4])])
        fig = km_median_bar(km)
        text_vals = list(fig.data[0].text)
        assert len(text_vals) == 1
        assert not text_vals[0].startswith("≥"), (
            f"Uncapped bar must NOT start with ≥, got: {text_vals[0]!r}"
        )
        assert text_vals[0] == "90", f"Expected '90', got {text_vals[0]!r}"

    def test_capped_bar_shows_ge_prefix(self):
        """Bar whose curve never crosses 0.5 must show '≥600' label."""
        # survival always above 0.5 → never crossed
        km = _make_km_df([("GroupB", [1.0, 0.9, 0.8, 0.7, 0.6])])
        fig = km_median_bar(km)
        text_vals = list(fig.data[0].text)
        assert len(text_vals) == 1
        assert text_vals[0].startswith("≥"), (
            f"Capped bar must start with ≥, got: {text_vals[0]!r}"
        )
        assert "600" in text_vals[0]

    def test_mixed_capped_and_uncapped(self):
        """When some groups are capped and some are not, each gets the correct label."""
        km = _make_km_df([
            ("GroupA", [1.0, 0.8, 0.6, 0.4]),   # uncapped, median at t=90
            ("GroupB", [1.0, 0.9, 0.8, 0.7]),   # capped
        ])
        fig = km_median_bar(km)
        texts = list(fig.data[0].text)
        assert len(texts) == 2
        a_text, b_text = texts[0], texts[1]
        assert not a_text.startswith("≥"), f"GroupA must not be capped, got: {a_text!r}"
        assert b_text.startswith("≥"), f"GroupB must be capped, got: {b_text!r}"

    def test_capped_bar_color_differs_from_uncapped(self):
        """Capped bars use a different color marker to visually distinguish them."""
        km = _make_km_df([
            ("GroupA", [1.0, 0.8, 0.6, 0.4]),
            ("GroupB", [1.0, 0.9, 0.8, 0.7]),
        ])
        fig = km_median_bar(km)
        colors = list(fig.data[0].marker.color)
        assert len(colors) == 2
        assert colors[0] != colors[1], (
            "Capped bar must use a different color than uncapped bar"
        )


def _hour_pivot():
    """Pivot with integer hour columns (0–23) and region rows — as heatmap_region_hour expects."""
    import numpy as np
    # rows = regions, cols = hours (transposed from hourly_by_region output)
    data = np.random.rand(3, 24)
    return pd.DataFrame(data, index=["Kyiv", "Lviv", "Odesa"], columns=range(24))


class TestHeatmapRegionHour:
    def test_raw_hover_contains_trevoh(self):
        """normalized=False → hover must say 'Тривог:' (raw count label)."""
        fig = heatmap_region_hour(_hour_pivot(), normalized=False)
        assert "Тривог:" in fig.data[0].hovertemplate

    def test_raw_title_contains_trevoh(self):
        """normalized=False → title must say 'Тривоги' (count context)."""
        fig = heatmap_region_hour(_hour_pivot(), normalized=False)
        assert "Тривоги" in fig.layout.title.text

    def test_normalized_hover_contains_chastka(self):
        """normalized=True → hover must say 'Частка:' (fraction label)."""
        fig = heatmap_region_hour(_hour_pivot(), normalized=True)
        assert "Частка:" in fig.data[0].hovertemplate

    def test_normalized_hover_uses_percent_format(self):
        """normalized=True → hover format must be ':.1%' so '0.042' renders as '4.2%'."""
        fig = heatmap_region_hour(_hour_pivot(), normalized=True)
        assert ":.1%" in fig.data[0].hovertemplate

    def test_normalized_title_differs_from_raw(self):
        """normalized=True → chart title must differ from the raw-count title."""
        fig_raw = heatmap_region_hour(_hour_pivot(), normalized=False)
        fig_norm = heatmap_region_hour(_hour_pivot(), normalized=True)
        assert fig_raw.layout.title.text != fig_norm.layout.title.text


class TestCorrelationHeatmapColorscale:
    def test_is_correlation_true_sets_zmid(self):
        """is_correlation=True must set zmid=0 (diverging scale centred at zero)."""
        df = pd.DataFrame([[1.0, 0.5], [0.5, 1.0]], columns=["A", "B"], index=["A", "B"])
        fig = correlation_heatmap(df, is_correlation=True)
        assert fig.data[0].zmid == 0, (
            f"is_correlation=True must set zmid=0, got zmid={fig.data[0].zmid!r}"
        )

    def test_is_correlation_false_sets_zmin_zmax(self):
        """Co-occurrence heatmap (is_correlation=False) must pin zmin=0, zmax=1."""
        df = pd.DataFrame([[0.0, 0.7], [0.7, 0.0]], columns=["A", "B"], index=["A", "B"])
        fig = correlation_heatmap(df, is_correlation=False)
        assert fig.data[0].zmin == 0, f"Expected zmin=0, got {fig.data[0].zmin!r}"
        assert fig.data[0].zmax == 1, f"Expected zmax=1, got {fig.data[0].zmax!r}"

    def test_is_correlation_false_does_not_set_zmid(self):
        """Co-occurrence heatmap must NOT use zmid (sequential scale has no natural centre)."""
        df = pd.DataFrame([[0.0, 0.7], [0.7, 0.0]], columns=["A", "B"], index=["A", "B"])
        fig = correlation_heatmap(df, is_correlation=False)
        # zmid should be None / not set
        assert fig.data[0].zmid is None, (
            f"is_correlation=False must not set zmid, got zmid={fig.data[0].zmid!r}"
        )
