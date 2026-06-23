"""Plotly chart factory functions.

Every function is pure: (data, **opts) → plotly Figure.
Anti-alert-fatigue palette: neutral base, bright accents ONLY for anomalies.
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px

# ── Palette ───────────────────────────────────────────────────────────────────
_C_ALERT  = "#EF5350"   # active alert — used sparingly
_C_CLEAR  = "#43A047"   # all-clear
_C_ACCENT = "#1E88E5"   # primary accent
_C_MUTED  = "#546E7A"   # muted / background series
_C_WARN   = "#FFA726"   # partial / reconnecting

_LAYOUT_BASE = dict(
    template="plotly_dark",
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family="system-ui, sans-serif", size=12),
    margin=dict(l=40, r=20, t=40, b=40),
)


# ── Frequency over time ───────────────────────────────────────────────────────

def alert_frequency_bar(
    freq_df: pd.DataFrame,
    title: str = "Кількість тривог у часі",
    color: str = _C_ACCENT,
) -> go.Figure:
    """Bar chart of alert count per time period."""
    fig = go.Figure(
        go.Bar(
            x=freq_df["period"],
            y=freq_df["alert_count"],
            marker_color=color,
            hovertemplate="%{x|%d %b %Y}<br>Тривог: %{y}<extra></extra>",
        )
    )
    fig.update_layout(
        **_LAYOUT_BASE,
        title=title,
        xaxis_title="Дата",
        yaxis_title="Кількість тривог",
    )
    return fig


def rolling_trend_chart(trend_df: pd.DataFrame) -> go.Figure:
    """Daily count + 7-day rolling average."""
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=trend_df["date"], y=trend_df["count"],
        name="Щоденно", marker_color=_C_MUTED, opacity=0.5,
        showlegend=False,
        hovertemplate="%{x|%d %b %Y}: %{y}<extra></extra>",
    ))
    fig.add_trace(go.Scatter(
        x=trend_df["date"], y=trend_df["rolling_7d"],
        name="7-денне ковзне", line=dict(color=_C_ACCENT, width=2),
        showlegend=False,
        hovertemplate="%{x|%d %b %Y}: %{y:.1f}<extra></extra>",
    ))
    fig.update_layout(
        **_LAYOUT_BASE,
        title="Тренд тривог",
        xaxis_title="Дата",
        yaxis_title="Кількість тривог",
    )
    return fig


# ── Heatmaps ─────────────────────────────────────────────────────────────────

def heatmap_hour_dow(pivot: pd.DataFrame) -> go.Figure:
    """Heatmap: hour-of-day (Y) × day-of-week (X), value = avg alerts/week."""
    if pivot.empty:
        return _empty_fig("Немає даних")

    fig = go.Figure(go.Heatmap(
        z=pivot.values,
        x=pivot.columns.tolist(),
        y=[f"{h:02d}:00" for h in pivot.index],
        colorscale=[
            [0.0, "#161B22"],
            [0.3, "#1565C0"],
            [0.7, "#E53935"],
            [1.0, "#FFCDD2"],
        ],
        hoverongaps=False,
        hovertemplate="%{x}, %{y}<br>Тривог/тиждень: %{z:.2f}<extra></extra>",
    ))
    fig.update_layout(
        **_LAYOUT_BASE,
        title="Тривоги за годиною та днем тижня (в середньому на тиждень)",
        xaxis_title="День тижня",
        yaxis_title="Година",
        yaxis=dict(autorange="reversed"),
    )
    return fig


def heatmap_monthly(pivot: pd.DataFrame) -> go.Figure:
    """Heatmap: year (Y) × month (X)."""
    if pivot.empty:
        return _empty_fig("Немає даних")

    month_labels = ["Січ", "Лют", "Бер", "Квіт", "Трав", "Черв",
                    "Лип", "Серп", "Вер", "Жовт", "Лист", "Груд"]
    cols = [month_labels[c - 1] for c in pivot.columns]

    fig = go.Figure(go.Heatmap(
        z=pivot.values,
        x=cols,
        y=pivot.index.tolist(),
        colorscale="YlOrRd",
        hovertemplate="Рік %{y}, %{x}<br>Тривог: %{z}<extra></extra>",
    ))
    fig.update_layout(
        **_LAYOUT_BASE,
        title="Тривоги по місяцях та роках",
        xaxis_title="Місяць",
        yaxis_title="Рік",
    )
    return fig


def heatmap_region_hour(pivot: pd.DataFrame, normalized: bool = False) -> go.Figure:
    """Heatmap: region (Y) × hour-of-day (X).

    normalized=False: z = raw alert count  → hover "Тривог: N"
    normalized=True:  z = fraction 0–1    → hover "Частка: 4.2%"
    """
    if pivot.empty:
        return _empty_fig("Немає даних")

    if normalized:
        title = "Розподіл тривог за регіоном та годиною (частка)"
        hover = "%{y}, %{x}<br>Частка: %{z:.1%}<extra></extra>"
    else:
        title = "Тривоги за регіоном та годиною"
        hover = "%{y}, %{x}<br>Тривог: %{z}<extra></extra>"

    fig = go.Figure(go.Heatmap(
        z=pivot.values,
        x=[f"{h:02d}:00" for h in pivot.columns],
        y=pivot.index.tolist(),
        colorscale="Blues",
        hovertemplate=hover,
    ))
    fig.update_layout(
        **_LAYOUT_BASE,
        title=title,
        xaxis_title="Година",
        yaxis_title="Область",
        height=max(400, len(pivot) * 22),
    )
    return fig


# ── Duration ─────────────────────────────────────────────────────────────────

def duration_histogram(hist_df: pd.DataFrame) -> go.Figure:
    """Pre-binned duration histogram."""
    fig = go.Figure(go.Bar(
        x=(hist_df["bin_left"] + hist_df["bin_right"]) / 2,
        y=hist_df["count"],
        width=hist_df["bin_right"] - hist_df["bin_left"],
        marker_color=_C_ACCENT,
        hovertemplate="%{x:.0f} хв<br>Тривог: %{y}<extra></extra>",
    ))
    fig.update_layout(
        **_LAYOUT_BASE,
        title="Розподіл тривалості тривог (хв)",
        xaxis_title="Тривалість (хв)",
        yaxis_title="Кількість тривог",
    )
    return fig


# ── Co-occurrence / correlation ───────────────────────────────────────────────

def correlation_heatmap(
    corr_df: pd.DataFrame,
    title: str = "Кореляція між регіонами",
    is_correlation: bool = True,
    triangle: bool = True,
) -> go.Figure:
    """Symmetric correlation or co-occurrence matrix heatmap.

    is_correlation=True  (default): RdBu diverging scale (−1 to +1).
    is_correlation=False: YlOrRd sequential scale (0 to 1, shown as %).
    triangle=True: show only lower triangle to eliminate redundancy.
    """
    if corr_df.empty:
        return _empty_fig("Немає даних")

    z = corr_df.values.copy().astype(float)

    if triangle:
        # k=0 masks diagonal + upper triangle (removes self-correlation cells)
        mask = np.triu(np.ones(len(corr_df)), k=0).astype(bool)
        z[mask] = np.nan

    if is_correlation:
        colorscale_kwargs = dict(colorscale="RdBu", zmid=0, zmin=-1, zmax=1)
        hover_tmpl = (
            "<b>%{y} ↔ %{x}</b><br>"
            "ρ = %{z:.3f}<br>"
            "У дні коли %{y} активніша — %{x} теж більше тривожить<extra></extra>"
        )
    else:
        colorscale_kwargs = dict(colorscale="YlOrRd", zmin=0, zmax=1)
        hover_tmpl = (
            "<b>%{y} → %{x}</b><br>"
            "%{z:.1%}<br>"
            "Якщо у %{y} тривога, то у %{x}<br>"
            "вона з'явиться впродовж 2 год<br>"
            "з ймовірністю %{z:.1%}<extra></extra>"
        )

    fig = go.Figure(go.Heatmap(
        z=z,
        x=corr_df.columns.tolist(),
        y=corr_df.index.tolist(),
        **colorscale_kwargs,
        hovertemplate=hover_tmpl,
    ))
    fig.update_layout(
        **_LAYOUT_BASE,
        title=title,
        height=max(500, len(corr_df) * 22),
        xaxis=dict(tickangle=-45),
    )
    return fig


# ── Gantt / timeline ─────────────────────────────────────────────────────────

def gantt_timeline(
    df: pd.DataFrame,
    regions: Optional[list[str]] = None,
    max_rows: int = 400,
) -> go.Figure:
    """Event timeline (Gantt): X=time, Y=region, bars=alert intervals."""
    data = df.copy()
    if regions:
        data = data[data["region"].isin(regions)]

    if data.empty:
        return _empty_fig("Немає даних для відображення")

    # Fill open-ended alerts with "now"
    import pytz
    now = pd.Timestamp.now(tz="Europe/Kyiv")
    data["_end"] = data["finished_at"].fillna(now)

    # Stratified downsample: every region keeps at least 1 row (longest alerts first)
    # so no region silently disappears when the total exceeds max_rows.
    if len(data) > max_rows:
        n_regions = data["region"].nunique()
        per_region = max(1, max_rows // n_regions)
        data = (
            data.sort_values("duration_min", ascending=False)
            .groupby("region", group_keys=False)
            .head(per_region)
            .sort_values("started_at")
        )

    fig = go.Figure()
    region_order = sorted(data["region"].unique())

    for i, region in enumerate(region_order):
        grp = data[data["region"] == region]
        for _, row in grp.iterrows():
            fig.add_shape(
                type="rect",
                x0=row["started_at"],
                x1=row["_end"],
                y0=i - 0.4,
                y1=i + 0.4,
                fillcolor=_C_ALERT,
                line_width=0,
                opacity=0.7,
            )

    fig.update_layout(
        **_LAYOUT_BASE,
        title="Таймлайн тривог (Gantt)",
        xaxis_title="Час",
        yaxis=dict(
            tickvals=list(range(len(region_order))),
            ticktext=region_order,
            title="Область",
        ),
        height=max(400, len(region_order) * 28 + 80),
    )
    return fig


# ── Survival analysis ─────────────────────────────────────────────────────────

def survival_curve(km_df: pd.DataFrame) -> go.Figure:
    """Kaplan-Meier survival curves, optionally stratified."""
    if km_df.empty:
        return _empty_fig("Недостатньо даних для аналізу виживаності")

    fig = go.Figure()
    # Preserve insertion order from kaplan_meier_stratified (groups list order)
    labels_ordered = list(dict.fromkeys(km_df["label"]))
    n = len(labels_ordered)
    # Cyclic HSV palette for ≥12 groups (e.g., all 24 hours), Set2 otherwise
    if n >= 12:
        palette = px.colors.sample_colorscale("hsv", [i / n for i in range(n)])
    else:
        palette = px.colors.qualitative.Set2

    _CUTOFF = 0.05  # cut the curve when S(t) drops below 5%

    for i, label in enumerate(labels_ordered):
        grp = km_df[km_df["label"] == label].copy()
        # Stop drawing after the first point where S(t) < 5%
        below = (grp["survival"] < _CUTOFF).values
        if below.any():
            grp = grp.iloc[: int(below.argmax()) + 1]

        color = palette[i % len(palette)]
        group_id = f"km_group_{i}"
        fig.add_trace(go.Scatter(
            x=grp["timeline"], y=grp["survival"],
            name=str(label),
            legendgroup=group_id,
            mode="lines",
            line=dict(color=color, width=2),
            hovertemplate=f"{label}<br>t=%{{x:.0f}} хв<br>S(t)=%{{y:.3f}}<extra></extra>",
        ))
        # CI band — same legendgroup: hidden together when trace is toggled
        if len(grp) >= 2:
            fig.add_trace(go.Scatter(
                x=pd.concat([grp["timeline"], grp["timeline"].iloc[::-1]]),
                y=pd.concat([grp["ci_upper"], grp["ci_lower"].iloc[::-1]]),
                fill="toself",
                fillcolor=color,
                opacity=0.1,
                line_width=0,
                showlegend=False,
                legendgroup=group_id,
                hoverinfo="skip",
            ))

    fig.add_hline(y=0.5, line_dash="dot", line_color=_C_MUTED,
                  annotation_text="S(t)=0.5 (медіана)")

    # Cap X-axis at 600 min — meaningful range for air-raid durations
    x_max = min(600, float(km_df["timeline"].max()))
    # Vertical legend on the right for many groups; horizontal bottom otherwise
    if n >= 12:
        legend_cfg = dict(orientation="v", x=1.02, y=1, xanchor="left",
                          font=dict(size=11))
        chart_height = 620
    else:
        legend_cfg = dict(orientation="h", y=-0.2)
        chart_height = 500
    fig.update_layout(
        **_LAYOUT_BASE,
        title="Виживаність тривоги S(t): P(тривога ще активна після t хвилин)",
        xaxis_title="Тривалість (хв)",
        yaxis_title="S(t)",
        xaxis=dict(range=[0, x_max]),
        yaxis=dict(range=[0, 1.05]),
        legend=legend_cfg,
        height=chart_height,
    )
    return fig


# ── KM median bar chart ───────────────────────────────────────────────────────

def km_median_bar(km_df: pd.DataFrame, title: str = "Медіана тривалості тривоги") -> go.Figure:
    """Bar chart of P50 (median) survival time per KM group.

    Median = first timeline value where S(t) ≤ 0.5.
    Capped at 600 min when the curve never crosses 50%.
    """
    labels_ordered = list(dict.fromkeys(km_df["label"]))
    n = len(labels_ordered)
    palette = (
        px.colors.sample_colorscale("hsv", [i / n for i in range(n)])
        if n >= 12
        else list(px.colors.qualitative.Set2)
    )

    rows = []
    for label in labels_ordered:
        grp = km_df[km_df["label"] == label]
        below = grp[grp["survival"] <= 0.5]
        if not below.empty:
            med = float(below.iloc[0]["timeline"])
            is_capped = False
        else:
            med = 600.0
            is_capped = True
        rows.append({"label": str(label), "median_min": med, "capped": is_capped})
    df_med = pd.DataFrame(rows)

    bar_colors = [
        _C_MUTED if row["capped"] else palette[i % len(palette)]
        for i, row in df_med.iterrows()
    ]
    bar_texts = [
        f"≥{v:.0f}" if capped else f"{v:.0f}"
        for v, capped in zip(df_med["median_min"], df_med["capped"])
    ]

    fig = go.Figure(go.Bar(
        x=df_med["label"],
        y=df_med["median_min"],
        marker_color=bar_colors,
        hovertemplate="%{x}<br>Медіана: %{text} хв<extra></extra>",
        text=bar_texts,
        textposition="outside",
    ))
    ymax = float(df_med["median_min"].max()) * 1.2
    fig.update_layout(
        **_LAYOUT_BASE,
        title=title,
        xaxis_title="",
        yaxis_title="Медіана тривалості (хв)",
        yaxis=dict(range=[0, ymax]),
        height=420,
    )
    return fig


# ── Attack Wave Replay ────────────────────────────────────────────────────────

def attack_wave_animation(event: dict) -> go.Figure:
    """[МОЄ РІШЕННЯ] Animated CHOROPLETH replay of alert propagation.

    Each frame = 5-minute window; oblasts fill red as the alert wave arrives.
    Uses go.Choropleth (SVG geo) — same robust style as every other map.
    event = one item from correlation.propagation_events().
    """
    from src.ui.maps import _load_geojson, _MAPBOX_STYLE, _MAP_CENTER, _MAP_ZOOM

    sequence = event.get("sequence", [])
    geojson = _load_geojson()
    if not sequence or geojson is None:
        return _empty_fig("Немає даних для replay")

    regions = [f["properties"]["region"] for f in geojson.get("features", [])]

    base_time = pd.Timestamp(sequence[0]["started_at"])
    total_min = max(
        (pd.Timestamp(r["started_at"]) - base_time).total_seconds() / 60
        for r in sequence
    )
    n_frames = max(int(total_min / 5) + 2, 2)

    # For each frame, which regions are active (cumulative)
    active_at_frame: list[set] = [set() for _ in range(n_frames)]
    for rec in sequence:
        t = pd.Timestamp(rec["started_at"])
        frame_idx = int((t - base_time).total_seconds() / 300)
        for fi in range(frame_idx, n_frames):
            active_at_frame[fi].add(rec["region"])

    # 2-stop colorscale: 0 = muted (clear), 1 = alert red
    wave_colorscale = [[0.0, "#263238"], [0.49, "#263238"],
                       [0.5, _C_ALERT], [1.0, _C_ALERT]]

    def _z_for(active: set) -> list[int]:
        return [1 if r in active else 0 for r in regions]

    def _choro(z_vals: list[int]) -> go.Choroplethmapbox:
        return go.Choroplethmapbox(
            geojson=geojson,
            locations=regions,
            z=z_vals,
            featureidkey="properties.region",
            colorscale=wave_colorscale,
            zmin=0, zmax=1,
            marker_opacity=0.82,
            marker_line_width=0.6,
            marker_line_color="#546E7A",
            showscale=False,
            hovertemplate="%{location}<extra></extra>",
        )

    frames = [
        go.Frame(data=[_choro(_z_for(active))], name=str(fi))
        for fi, active in enumerate(active_at_frame)
    ]

    fig = go.Figure(data=[_choro(_z_for(set()))], frames=frames)
    fig.update_layout(
        **_LAYOUT_BASE,
        mapbox=dict(style=_MAPBOX_STYLE, center=_MAP_CENTER, zoom=_MAP_ZOOM),
        title=f"Хвиля атаки {event['date']} — {event['regions_hit']} обл. за {event['duration_h']:.1f} год",
        updatemenus=[dict(
            type="buttons",
            showactive=False,
            y=0, x=0.5, xanchor="center",
            buttons=[
                dict(label="▶ Play", method="animate",
                     args=[None, {"frame": {"duration": 600, "redraw": True},
                                  "fromcurrent": True, "transition": {"duration": 0}}]),
                dict(label="⏸ Pause", method="animate",
                     args=[[None], {"frame": {"duration": 0, "redraw": False},
                                    "mode": "immediate"}]),
            ],
        )],
        sliders=[dict(
            currentvalue=dict(prefix="Кадр: "),
            steps=[dict(args=[[str(i)], {"frame": {"duration": 0, "redraw": True},
                                         "mode": "immediate"}],
                        label=str(i), method="animate")
                   for i in range(n_frames)],
        )],
        height=520,
    )
    return fig


# ── Helpers ───────────────────────────────────────────────────────────────────

def _empty_fig(msg: str = "Немає даних") -> go.Figure:
    fig = go.Figure()
    fig.update_layout(
        **_LAYOUT_BASE,
        annotations=[dict(
            text=msg, xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False,
            font=dict(size=16, color=_C_MUTED),
        )],
    )
    return fig
