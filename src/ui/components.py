"""Reusable Streamlit UI components."""
from __future__ import annotations

from typing import Optional

import pandas as pd
import streamlit as st


def metric_row(items: list[tuple[str, str, Optional[str]]]) -> None:
    """Render a row of st.metric items. items = [(label, value, delta), ...]."""
    cols = st.columns(len(items))
    for col, (label, value, delta) in zip(cols, items):
        col.metric(label, value, delta)


def filter_sidebar(
    regions: list[str],
    default_regions: Optional[list[str]] = None,
    min_date: Optional[pd.Timestamp] = None,
    max_date: Optional[pd.Timestamp] = None,
) -> dict:
    """Render filter controls in the sidebar; return selected filter state."""
    with st.sidebar:
        st.header("Фільтри")

        selected_regions = st.multiselect(
            "Регіони",
            options=regions,
            default=default_regions or [],
            help="Залиште порожнім для всіх регіонів",
        )

        date_range = None
        if min_date is not None and max_date is not None:
            # Use today as max_value so persisted session state never exceeds the limit
            # even when the cached dataset is 1-2 days behind the current date.
            # apply_filters clamps to actual data range.
            today = pd.Timestamp.today().date()
            date_range = st.date_input(
                "Діапазон дат",
                value=(min_date.date(), max_date.date()),
                min_value=min_date.date(),
                max_value=today,
            )

        st.caption(
            "ℹ️ Луганська обл. та АР Крим виключені зі статистики "
            "(майже постійна тривога з лютого 2022)."
        )

    return {
        "regions": selected_regions,
        "date_range": date_range,
    }


def apply_filters(df: pd.DataFrame, filters: dict) -> pd.DataFrame:
    """Filter a transformed DataFrame based on filter_sidebar output."""
    data = df.copy()

    regions = filters.get("regions")
    if regions:
        data = data[data["region"].isin(regions)]

    date_range = filters.get("date_range")
    if date_range and len(date_range) == 2:
        start, end = date_range
        data = data[
            (data["started_at"].dt.date >= start) &
            (data["started_at"].dt.date <= end)
        ]

    return data


def empty_state_message(message: str = "Немає даних для відображення") -> None:
    """Show a styled empty-state message."""
    st.info(f"ℹ️ {message}", icon=None)


def data_freshness_note(df: pd.DataFrame) -> None:
    """Show when the dataset was last updated."""
    if df.empty:
        return
    last = df["started_at"].max()
    if pd.isna(last):
        return
    st.caption(f"Останній запис у датасеті: **{last.strftime('%d %b %Y %H:%M')}**")
