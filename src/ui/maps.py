"""Choropleth map factory functions for Ukraine alert data.

Uses assets/ukraine_oblasts.geojson (Natural Earth admin-1, 25 features).
featureidkey = "properties.region" (our short Ukrainian name).

Implemented with go.Choroplethmapbox — renders filled oblast polygons on a dark
base map (matches the alerts.in.ua reference look) and supports click selection:
  event = st.plotly_chart(fig, on_select="rerun", key="...")
  event.selection.points[0]["location"] == clicked region's short name

NOTE on sizing: a mapbox chart placed inside a *non-active* st.tab can initialise
at the 400×300 default until that tab is shown. `_apply_responsive()` returns the
plotly config that makes the chart re-fit its container on display, and callers
pass it through st.plotly_chart(config=...).
"""
from __future__ import annotations

import json
from functools import lru_cache
from typing import Optional

import pandas as pd
import plotly.graph_objects as go

from config import OBLAST_CENTROIDS, GEOJSON_PATH

_C_SELECTED  = "#FFFFFF"   # clicked region outline + label
_C_BORDER    = "#546E7A"
_C_LABEL     = "#CFD8DC"
_C_NO_DATA   = "#37474F"   # excluded/no-data regions (Luhansk, Crimea)

_MAPBOX_STYLE = "carto-darkmatter"
_MAP_CENTER = dict(lat=49.0, lon=31.5)
_MAP_ZOOM   = 4.6

_LAYOUT_BASE = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family="system-ui, sans-serif", size=12, color=_C_LABEL),
    margin=dict(l=0, r=0, t=40, b=0),
)


def responsive_config() -> dict:
    """Plotly config that keeps the mapbox chart fit to its container width."""
    return {"responsive": True, "displayModeBar": False}


@lru_cache(maxsize=1)
def _load_geojson() -> Optional[dict]:
    """Load GeoJSON once and cache. Returns None if file missing."""
    if not GEOJSON_PATH.exists():
        return None
    with open(GEOJSON_PATH, encoding="utf-8") as f:
        return json.load(f)


def _label_traces(selected: Optional[str] = None) -> list[go.Scattermapbox]:
    """Two text-label traces (normal + selected) — Scattermapbox.textfont needs
    a scalar colour/size, so the selected region goes in its own trace."""
    norm_lat, norm_lon, norm_txt = [], [], []
    sel_lat, sel_lon, sel_txt = [], [], []
    for region, (lat, lon) in OBLAST_CENTROIDS.items():
        if region == selected:
            sel_lat.append(lat); sel_lon.append(lon); sel_txt.append(region)
        else:
            norm_lat.append(lat); norm_lon.append(lon); norm_txt.append(region)

    traces = [go.Scattermapbox(
        lat=norm_lat, lon=norm_lon, text=norm_txt,
        mode="text", textfont=dict(size=10, color=_C_LABEL),
        hoverinfo="skip", showlegend=False,
    )]
    if sel_txt:
        traces.append(go.Scattermapbox(
            lat=sel_lat, lon=sel_lon, text=sel_txt,
            mode="text", textfont=dict(size=13, color=_C_SELECTED),
            hoverinfo="skip", showlegend=False,
        ))
    return traces


def historical_choropleth(
    region_values: pd.Series,           # index=region_name, values=metric
    title: str = "Тривоги по регіонах",
    colorscale: str = "YlOrRd",
    unit: str = "",
    selected_region: Optional[str] = None,
) -> go.Figure:
    """Choropleth coloured by a continuous metric value.

    Click a region → its `location` is returned via on_select. The selected
    region gets a thick white outline + bold label.
    """
    geojson = _load_geojson()
    if geojson is None:
        return _scatter_fallback(region_values, title, unit)

    all_geojson_regions = [f["properties"]["region"] for f in geojson.get("features", [])]

    # Split into data regions (have values) and no-data regions (excluded, e.g. Crimea, Luhansk)
    data_regions = [r for r in all_geojson_regions if r in region_values.index]
    nodata_regions = [r for r in all_geojson_regions if r not in region_values.index]

    z_vals, hovers, line_widths = [], [], []
    for r in data_regions:
        val = float(region_values[r])
        z_vals.append(val)
        suffix = " ● натисніть для деталей" if r != selected_region else " ● обрано"
        hovers.append(f"<b>{r}</b><br>{val:.1f} {unit}<br><i style='color:#90A4AE'>{suffix}</i>")
        line_widths.append(3.0 if r == selected_region else 0.6)

    fig = go.Figure()

    # Gray "no data" layer for excluded regions (Луганська, АР Крим)
    if nodata_regions:
        fig.add_trace(go.Choroplethmapbox(
            geojson=geojson,
            locations=nodata_regions,
            z=[0] * len(nodata_regions),
            featureidkey="properties.region",
            colorscale=[[0, _C_NO_DATA], [1, _C_NO_DATA]],
            zmin=0, zmax=1,
            marker_opacity=0.65,
            marker_line_width=0.6,
            marker_line_color=_C_BORDER,
            showscale=False,
            hovertemplate="%{location}: немає даних (виключено зі статистики)<extra></extra>",
        ))

    fig.add_trace(go.Choroplethmapbox(
        geojson=geojson,
        locations=data_regions,
        z=z_vals,
        featureidkey="properties.region",
        colorscale=colorscale,
        marker_opacity=0.82,
        marker_line_width=line_widths,
        marker_line_color=_C_SELECTED,
        colorbar=dict(title=unit, thickness=12, len=0.7, bgcolor="rgba(0,0,0,0.3)"),
        customdata=hovers,
        hovertemplate="%{customdata}<extra></extra>",
    ))
    # No text-label traces — names appear on hover; removing them reduces clutter

    fig.update_layout(
        **_LAYOUT_BASE,
        mapbox=dict(style=_MAPBOX_STYLE, center=_MAP_CENTER, zoom=_MAP_ZOOM),
        title=title,
        height=520,
    )
    return fig


# ── Scatter fallback (only if GeoJSON file is missing) ────────────────────────

def _scatter_fallback(region_values: pd.Series, title: str, unit: str) -> go.Figure:
    lats, lons, texts, vals = [], [], [], []
    for region, val in region_values.items():
        c = OBLAST_CENTROIDS.get(region)
        if c:
            lats.append(c[0]); lons.append(c[1])
            texts.append(f"{region}<br>{val:.1f} {unit}")
            vals.append(float(val))

    fig = go.Figure(go.Scattermapbox(
        lat=lats, lon=lons, text=texts,
        mode="markers",
        marker=dict(color=vals, colorscale="YlOrRd", size=16, showscale=True),
        hovertemplate="%{text}<extra></extra>",
    ))
    fig.update_layout(
        **_LAYOUT_BASE,
        mapbox=dict(style=_MAPBOX_STYLE, center=_MAP_CENTER, zoom=_MAP_ZOOM),
        title=title,
        height=520,
    )
    return fig
