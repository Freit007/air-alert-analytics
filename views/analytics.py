"""Historical analytics page.

Tabs:
  1. Огляд       — frequency, rolling trend, regional breakdown
  2. Сезонність  — heatmaps (hour×dow, monthly, region×hour)
  3. Тривалість  — histogram + survival curves
  4. Кореляція   — co-occurrence + correlation matrix
  5. Attack Wave — animated propagation [МОЄ РІШЕННЯ]
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import streamlit as st

# Import from cache.py — does NOT execute app.py's module-level code
from src.data.cache import load_data
# (region metadata is used inside src.ui/src.analysis, not directly here)
from src.analysis import descriptive, seasonality, correlation, survival
from src.data.transforms import copy_for_analysis
from src.ui import charts, components, maps
from src.ui.maps import historical_choropleth

# ── Load + filter ─────────────────────────────────────────────────────────────
with st.spinner("Завантаження…"):
    try:
        df_all = load_data()
    except FileNotFoundError:
        st.error(
            "Датасет не знайдено. "
            "Запустіть `python scripts/download_data.py` для завантаження.",
            icon="🚫",
        )
        st.stop()
    except Exception as exc:
        st.error(f"Помилка завантаження датасету: {exc}", icon="🚫")
        st.stop()

filters = components.filter_sidebar(
    regions=sorted(df_all["region"].dropna().unique()),
    min_date=df_all["started_at"].min(),
    max_date=df_all["started_at"].max(),
)

df = components.apply_filters(df_all, filters)
df_analysis = copy_for_analysis(df)

if df.empty:
    components.empty_state_message("Фільтри не повертають даних. Спробуйте розширити вибір.")
    st.stop()

components.data_freshness_note(df_all)

# ── Header metrics ────────────────────────────────────────────────────────────
total = len(df)
regions_count = df["region"].nunique()
dur_stats = descriptive.duration_stats(df_analysis)
_has_kyiv_city = "м. Київ" in df["region"].values
_regions_display = f"{regions_count - 1} + м. Київ" if _has_kyiv_city else str(regions_count)

components.metric_row([
    ("Тривог (фільтр)", f"{total:,}", None),
    ("Регіонів", _regions_display, None),
    ("Медіана тривалості", f"{dur_stats.get('median_min', 0):.0f} хв", None),
    ("Всього год. тривоги", f"{dur_stats.get('total_hours', 0):,.0f}", None),
])

st.divider()

# ── Tabs ──────────────────────────────────────────────────────────────────────
tab_overview, tab_season, tab_dur, tab_corr, tab_wave = st.tabs([
    "📈 Огляд",
    "🕐 Сезонність",
    "⏱ Тривалість",
    "🔗 Кореляція",
    "🌊 Attack Wave",
])

# Mapbox charts inside a non-active st.tab initialise at the 400×300 default
# (their container has zero width while hidden). This fires a window resize a
# moment after any tab is clicked so Plotly (responsive=True) re-fits the map.
import streamlit.components.v1 as _components
_components.html(
    """
    <script>
    const doc = window.parent.document;
    doc.querySelectorAll('button[role="tab"]').forEach((t) => {
      if (t.dataset._resizeHooked) return;
      t.dataset._resizeHooked = "1";
      t.addEventListener("click", () => {
        [120, 350, 700].forEach((ms) =>
          setTimeout(() => window.parent.dispatchEvent(new Event("resize")), ms));
      });
    });
    </script>
    """,
    height=0,
)


# ════════════════════════════════════════════════════════════════════════
# Tab 1 — Огляд
# ════════════════════════════════════════════════════════════════════════
with tab_overview:
    freq_label = st.radio(
        "Агрегація",
        ["Щоденно", "Тижнево", "Місячно"],
        horizontal=True,
        index=0,
    )
    freq_map = {"Щоденно": "D", "Тижнево": "W", "Місячно": "ME"}
    freq_col = freq_map[freq_label]

    freq_df = descriptive.alert_frequency_over_time(df_analysis, freq=freq_col)
    if not freq_df.empty:
        st.plotly_chart(charts.alert_frequency_bar(freq_df), use_container_width=True)
        st.caption(
            f"**Абсолютна кількість** епізодів по всіх регіонах за обраний масштаб "
            f"({freq_label.lower()}). Перемикайте агрегацію вгорі, щоб побачити добові "
            "сплески, тижневі патерни або місячні тренди."
        )
    else:
        components.empty_state_message()

    trend_df = seasonality.rolling_7d_avg(df_analysis)
    if not trend_df.empty:
        st.plotly_chart(charts.rolling_trend_chart(trend_df), use_container_width=True)
        st.caption(
            "**Щоденні дані + 7-денне ковзне середнє.** На відміну від графіка вище, "
            "цей завжди показує щоденний розріз і додає синю лінію тренду — вона прибирає "
            "добові стрибки та показує, чи активність загалом зростає, падає або стабільна "
            "протягом обраного діапазону."
        )

    st.subheader("Розбивка по регіонах")
    reg_df = descriptive.regional_breakdown(df_analysis)
    if not reg_df.empty:
        region_total = reg_df.set_index("region")["total_duration_h"]

        # Map with click-to-select
        sel_key = "overview_map_sel"
        if sel_key not in st.session_state:
            st.session_state[sel_key] = None

        map_event = st.plotly_chart(
            historical_choropleth(
                region_total,
                title="Сумарна тривалість тривог (год) — оберіть регіон",
                unit="год",
                selected_region=st.session_state[sel_key],
            ),
            use_container_width=True,
            on_select="rerun",
            key="overview_map",
            config=maps.responsive_config(),
        )
        # Handle map click
        if map_event and map_event.selection and map_event.selection.points:
            clicked = map_event.selection.points[0].get("location")
            if clicked:
                st.session_state[sel_key] = clicked

        # Region stats panel (shown when a region is clicked)
        sel_region = st.session_state.get(sel_key)
        if sel_region:
            with st.expander(f"📍 Статистика: {sel_region}", expanded=True):
                r_data = reg_df[reg_df["region"] == sel_region]
                if not r_data.empty:
                    r = r_data.iloc[0]
                    c1, c2, c3 = st.columns(3)
                    c1.metric("Тривог", f"{int(r.get('alert_count', 0)):,}")
                    c2.metric("Середня тривалість", f"{r.get('mean_duration_min', 0):.0f} хв")
                    c3.metric("Сумарно", f"{r.get('total_duration_h', 0):.0f} год")
                if st.button("✕ Скинути вибір", key="clear_overview_map"):
                    st.session_state[sel_key] = None
                    st.rerun()

        st.dataframe(
            reg_df.rename(columns={
                "region": "Регіон",
                "alert_count": "Тривог",
                "mean_duration_min": "Середня тривалість (хв)",
                "total_duration_h": "Сумарно (год)",
            }),
            use_container_width=True,
            height=400,
            hide_index=True,
        )

    top_days = descriptive.top_alert_days(df_analysis)
    if not top_days.empty:
        st.subheader("Топ-15 днів за кількістю тривог")
        st.dataframe(
            top_days.rename(columns={"date": "Дата", "alert_count": "Тривог"}),
            use_container_width=True,
            hide_index=True,
        )


# ════════════════════════════════════════════════════════════════════════
# Tab 2 — Сезонність
# ════════════════════════════════════════════════════════════════════════
with tab_season:
    st.markdown("**Нічні атаки (20:00–06:00)**")
    night_stats = seasonality.night_vs_day_ratio(df_analysis)
    c1, c2, c3 = st.columns(3)
    c1.metric("Нічних тривог", f"{night_stats['night_count']:,}")
    c2.metric("Денних тривог", f"{night_stats['day_count']:,}")
    c3.metric("Частка нічних", f"{night_stats['night_fraction']*100:.1f}%")

    pivot_hdow = seasonality.hourly_by_dow(df_analysis)
    if not pivot_hdow.empty:
        st.plotly_chart(charts.heatmap_hour_dow(pivot_hdow), use_container_width=True)

    pivot_month = seasonality.monthly_heatmap(df_analysis)
    if not pivot_month.empty:
        st.plotly_chart(charts.heatmap_monthly(pivot_month), use_container_width=True)

    # Region×hour heatmap (for selected or all regions)
    show_regions = filters.get("regions") or sorted(df_analysis["region"].dropna().unique().tolist())
    pivot_rh = seasonality.hourly_by_region(df_analysis, regions=show_regions, normalize=True)
    if not pivot_rh.empty:
        # pivot_rh: rows=hour, cols=region → transpose for region(Y) × hour(X)
        st.plotly_chart(charts.heatmap_region_hour(pivot_rh.T, normalized=True), use_container_width=True)
        st.caption(
            "Кожна клітинка — **частка** тривог цього регіону, що розпочались у цей час доби "
            "(київський час). Значення нормалізовані по кожному регіону окремо, щоб прифронтові "
            "регіони з великим абсолютним числом тривог не домінували над тиловими."
        )

    dur_by_h = seasonality.duration_by_hour(df_analysis)
    if not dur_by_h.empty:
        import plotly.express as px
        fig_dh = px.bar(
            dur_by_h, x="hour", y="mean_duration_min",
            title="Середня тривалість за годиною доби",
            labels={"hour": "Година", "mean_duration_min": "Середня тривалість (хв)"},
            template="plotly_dark",
        )
        fig_dh.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig_dh, use_container_width=True)
        st.caption(
            "Читання: якщо тривога розпочалась о N-й годині (київський час), то в середньому "
            "вона тривала стільки хвилин до відбою. Наприклад, нічні тривоги (20–21 год) "
            "довші — це відповідає типовим патернам атак Shahed. "
            "Враховуються тільки завершені епізоди."
        )


# ════════════════════════════════════════════════════════════════════════
# Tab 3 — Тривалість
# ════════════════════════════════════════════════════════════════════════
with tab_dur:
    st.subheader("Розподіл тривалостей")
    max_min = st.slider(
        "Показати тривоги до (хв)",
        60, 720, 300, step=30,
        help="Епізоди довші за обране значення не показуються на гістограмі (їх кількість вказана нижче).",
    )
    hist_df, _n_total, _n_excl = descriptive.duration_histogram_data(
        df_analysis, max_minutes=max_min
    )
    st.plotly_chart(charts.duration_histogram(hist_df), use_container_width=True)
    if _n_excl > 0:
        st.caption(
            f"Поза графіком: {_n_excl:,} епізодів > {max_min} хв "
            f"({_n_excl / _n_total * 100:.1f}% від усіх завершених). "
            "Збільшіть повзунок, щоб побачити їх."
        )
    else:
        st.caption(f"Показано всі {_n_total:,} завершених епізодів ≤ {max_min} хв.")

    st.subheader("Скільки триває типова тривога?")
    st.markdown(
        """
**Графік виживаності S(t)** відповідає на питання: *яка ймовірність, що тривога ще
не скінчилась через t хвилин після початку?*

- **S(0) = 1.0** — усі тривоги активні на старті
- **S(60) = 0.8** означає, що 80% тривог тривають довше 60 хв
- Точка, де крива перетинає пунктир **S=0.5** — **медіана тривалості**
- Стратифікація дозволяє порівняти тривалості залежно від часу доби або дня тижня
"""
    )

    if not survival._LIFELINES_AVAILABLE:
        st.warning(
            "Бібліотека `lifelines` не встановлена. "
            "Виконайте `pip install lifelines` для аналізу виживаності.",
            icon="⚠️",
        )
    else:
        _DOW_UK = {
            "Monday": "Понеділок", "Tuesday": "Вівторок", "Wednesday": "Середа",
            "Thursday": "Четвер", "Friday": "П'ятниця",
            "Saturday": "Субота", "Sunday": "Неділя",
        }
        strat_map = {
            "Без стратифікації": None,
            "За годиною доби": "hour",
            "За днем тижня": "dow_name",
        }
        strat_label = st.selectbox(
            "Порівняння груп",
            list(strat_map.keys()),
            index=0,
            help="Порівняти тривалість тривог між різними групами (наприклад, вночі vs вдень)",
        )
        strat_col = strat_map[strat_label]

        try:
            if strat_col is None:
                km_df = survival.kaplan_meier(df_analysis, label="Всі тривоги")
            elif strat_col == "hour":
                km_df = survival.kaplan_meier_stratified(
                    df_analysis, strata_col="hour",
                    groups=list(range(24)),
                )
            else:
                km_df = survival.kaplan_meier_stratified(
                    df_analysis, strata_col="dow_name",
                    groups=["Monday", "Tuesday", "Wednesday",
                            "Thursday", "Friday", "Saturday", "Sunday"],
                )
                km_df["label"] = km_df["label"].map(lambda x: _DOW_UK.get(x, x))

            if not km_df.empty:
                if strat_col is None:
                    # Single curve — plain KM chart, no legend toggle hint needed
                    st.plotly_chart(charts.survival_curve(km_df), use_container_width=True)
                else:
                    # Stratified: median bar chart as primary, full KM curves in expander
                    _strat_display = "за стартом у конкретну годину доби" if strat_col == "hour" else "за днем тижня"
                    st.plotly_chart(
                        charts.km_median_bar(
                            km_df,
                            title=f"Медіана тривалості тривоги {_strat_display}",
                        ),
                        use_container_width=True,
                    )
                    st.caption(
                        "Медіана — час, після якого 50 % тривог у цій групі вже завершились. "
                        "Наприклад, «60 хв» означає: половина тривог цієї години тривала ≤ 60 хвилин. "
                        "Сірий стовпець з міткою **≥600** — крива не перетнула 50 % у межах 600 хв "
                        "(реальна медіана вища, але не вимірна на наявних даних). "
                        "⚠ *KM передбачає незалежність цензурування від тривалості — тривоги, що "
                        "тривають на момент завантаження датасету, вважаються рандомно цензурованими.*"
                    )
                    with st.expander("Показати криві виживаності (розширено)"):
                        st.caption(
                            "Клікніть на рядку легенди праворуч, щоб приховати або показати окрему групу. "
                            "Подвійний клік — залишити тільки цю групу."
                        )
                        st.plotly_chart(charts.survival_curve(km_df), use_container_width=True)
            else:
                components.empty_state_message("Недостатньо даних")
        except Exception as e:
            st.error(f"Помилка аналізу виживаності: {e}")


# ════════════════════════════════════════════════════════════════════════
# Tab 4 — Кореляція
# ════════════════════════════════════════════════════════════════════════
with tab_corr:
    with st.spinner("Обчислення матриці збігу…"):
        cooc_df = correlation.co_occurrence_matrix(df_analysis, window_hours=2.0)

    if not cooc_df.empty:
        st.plotly_chart(
            charts.correlation_heatmap(
                cooc_df,
                title="Збіг тривог між регіонами (вікно ±2 год)",
                is_correlation=False,
                triangle=True,
            ),
            use_container_width=True,
        )
        st.caption(
            "**Як читати:** клітинка (A, B) = % тривог регіону A, у яких впродовж 2 годин "
            "також є тривога в регіоні B. "
            "**Стратегічна цінність:** показує, які регіони атакуються в одній хвилі. "
            "Наприклад, прикордонні регіони та центр мають 85–95 % — атакуються разом; "
            "Закарпатська та Херсонська мають 40 % — зазвичай окремі удари. "
            "Матриця симетрична, показано нижній трикутник."
        )
    else:
        components.empty_state_message()

    st.divider()
    st.subheader("Кореляція добових ритмів (Спірмен)")
    st.info(
        "**Чим відрізняється від таблиці вище?**  \n"
        "Збіг вимірює **одну подію**: якщо в A тривога зараз — чи виникне вона в B впродовж 2 год?  \n"
        "Спірмен вимірює **ритм за місяці**: в дні коли A має більше тривог, B теж має їх більше?  \n"
        "Регіони можуть мати 99 % збігу (атакуються одночасно), але низьку кореляцію Спірмена "
        "(кожен регіон має свій власний добовий ритм незалежно від інших).",
        icon="ℹ️",
    )
    with st.spinner("Обчислення кореляційної матриці…"):
        corr_df = correlation.regional_correlation_matrix(df_analysis)

    if not corr_df.empty:
        st.plotly_chart(
            charts.correlation_heatmap(
                corr_df,
                title="Спірменова рангова кореляція тривог по областях",
                triangle=True,
            ),
            use_container_width=True,
        )
        st.caption(
            "**+1** — дві области мають однаковий ритм активності (разом активні, разом спокійні). "
            "**0** — незалежні."
        )


# ════════════════════════════════════════════════════════════════════════
# Tab 5 — Attack Wave Replay [МОЄ РІШЕННЯ]
# ════════════════════════════════════════════════════════════════════════
with tab_wave:
    st.subheader("🌊 Attack Wave Replay")
    st.markdown(
        """
Анімований replay поширення хвилі тривоги по регіонах.
Кожен кадр = 5-хвилинне вікно. Показує напрямок і швидкість поширення атаки.

> *Shahed-атаки* — хвиля зі Сходу/Півдня на Захід.
> *Ракетні удари* — майже одночасний охват.
        """
    )

    col_a, col_b = st.columns(2)
    with col_a:
        min_regions = st.slider("Мінімум регіонів у хвилі", 3, 12, 6)
    with col_b:
        window_hours = st.slider("Вікно поширення (год)", 1.0, 6.0, 3.0, 0.5)

    st.caption(
        "Більше «Вікно поширення» → кожна хвиля охоплює ширший часовий діапазон, "
        "але сусідні хвилі зливаються — загальна кількість подій може як зменшуватись, так і зростати. "
        "Менше «Мінімум регіонів» → поріг нижчий, подій більше."
    )

    with st.spinner("Пошук хвильових подій…"):
        wave_events = correlation.propagation_events(
            df_analysis,
            min_regions=min_regions,
            window_hours=window_hours,
        )

    if not wave_events:
        components.empty_state_message(
            "Не знайдено подій із обраними параметрами. Спробуйте зменшити поріг."
        )
    else:
        # Sort newest-first so recent events are visible (not just massive 2022 waves)
        wave_events_sorted = sorted(wave_events, key=lambda x: x["date"], reverse=True)

        st.info(f"Знайдено {len(wave_events_sorted)} хвильових подій.")

        if not wave_events_sorted:
            components.empty_state_message(
                "Немає хвиль із такою тривалістю. Змініть діапазон повзунка."
            )
        else:
            event_labels = [
                f"{e['date']} — {e['regions_hit']} обл. за {e['duration_h']:.1f} год"
                for e in wave_events_sorted[:100]
            ]
            selected_idx = st.selectbox(
                "Оберіть подію для replay",
                options=range(len(event_labels)),
                format_func=lambda i: event_labels[i],
            )

            event = wave_events_sorted[selected_idx]
            fig_wave = charts.attack_wave_animation(event)
            st.plotly_chart(
                fig_wave,
                use_container_width=True,
                key="wave_map",
                config=maps.responsive_config(),
            )

            with st.expander("Деталі послідовності"):
                seq_df = pd.DataFrame(event["sequence"])
                if not seq_df.empty:
                    seq_df["started_at"] = pd.to_datetime(seq_df["started_at"])
                    seq_df["Δ від першої (хв)"] = (
                        (seq_df["started_at"] - seq_df["started_at"].iloc[0])
                        .dt.total_seconds()
                        .div(60)
                        .round(1)
                    )
                    st.dataframe(
                        seq_df[["region", "started_at", "Δ від першої (хв)"]].rename(
                            columns={"region": "Регіон", "started_at": "Час тривоги"}
                        ),
                        use_container_width=True,
                        hide_index=True,
                    )
