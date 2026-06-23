# 🚨 Air Alert Time Series Analysis — Ukraine

Descriptive analysis of **official public air-raid alert data** for Ukraine
(March 2022 — present), at the level of the **whole oblast**.

> **Scope & ethics**: This project performs *descriptive analysis of publicly
> available historical data*. It is **not** an operational attack-prediction
> tool. Always use official sources (e.g. [alerts.in.ua](https://alerts.in.ua))
> for personal-safety decisions.

---

## Features

### 📊 Analytics
- Alert frequency over time (**daily / weekly / monthly**)
- Duration distributions + **Kaplan-Meier survival curves**
- Temporal heatmaps: hour-of-day × day-of-week, monthly patterns, region × hour
- Cross-regional **co-occurrence** and **Pearson correlation** matrices
- Regional breakdown with an **interactive choropleth map** — click an oblast to
  drill into its statistics
- **Attack Wave Replay** — animated choropleth replay of how an alert wave
  propagates across oblasts (Shahed waves spread East→West over hours; missile
  salvos light up near-simultaneously)

All maps are **filled-polygon choropleths** on a dark base, with click-to-select.

---

## Data Sources

| Source | Role |
|--------|------|
| `official_data_uk.csv` from [Vadimkin/ukrainian-air-raid-sirens-dataset](https://github.com/Vadimkin/ukrainian-air-raid-sirens-dataset) | **PRIMARY** — government alert API, has a `level` column (oblast/raion/hromada) |
| `volunteer_data_uk.csv` (same repo) | **CROSS-CHECK** — independent Telegram-based collection |

The two sources are independently collected; `scripts/cross_check.py` compares
per-oblast counts between them (Spearman ρ ≈ 0.96 — they strongly agree).

---

## Data Correctness — how the numbers are made trustworthy

The raw file mixes three alert levels (oblast / raion / hromada). Around
**November 2025 Ukraine switched from whole-oblast alerts to per-raion alerts**,
so a naive "oblast-level only" filter makes all recent data collapse to near
zero. To produce one consistent oblast-level series we **unify** alerts:

| Rule | Handling | Where |
|------|----------|-------|
| **Unification into oblast episodes** | Any alert inside an oblast (oblast/raion/hromada level) means the oblast is under alert; overlapping intervals are **merged** so simultaneous raion alerts collapse into one episode (no double-counting). | `loader.py` |
| **Exact-duplicate removal** | The raw file records every alert **twice**; identical rows are dropped. | `loader.py` |
| **UTC → Europe/Kyiv (DST)** | All timestamps converted DST-aware via `pytz`. | `transforms.py` |
| **Censoring** | Open (ongoing) intervals flagged; Kaplan-Meier accounts for right-censoring. | `transforms.py` |
| **Near-permanent regions** | Luhansk & Crimea (near-continuous alert since 2022) excluded from aggregate stats. | `transforms.py` |

An "alert count" for an oblast is therefore the number of **episodes** during
which an alert was active anywhere in it — consistent across both the
oblast-alert era and the post-2025 raion-alert era.

### Why does Kyiv city have fewer alerts than Dnipropetrovsk oblast?

This is **real, not a data error** — confirmed by cross-checking two independent
sources (`python scripts/cross_check.py`, Spearman ρ ≈ 0.92):

| Region | Official (unified episodes) | Volunteer |
|--------|---------:|----------:|
| Дніпропетровська | 3 417 | 11 658 |
| м. Київ | 2 081 | 2 191 |

Dnipropetrovsk oblast is geographically huge and on the front line, so alerts
trigger far more often. Kyiv city is one compact locality further from the
front. The volunteer source heavily over-counts Dnipropetrovsk (it doesn't merge
simultaneous raion alerts), while for Kyiv the two sources nearly match.

---

## Quick Start

```bash
git clone <your-repo-url>
cd "ai air alert"
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt

python scripts/download_data.py     # downloads official + volunteer CSVs (~45 MB)
streamlit run app.py                # opens http://localhost:8501
```

The Ukraine oblasts GeoJSON (`assets/ukraine_oblasts.geojson`, Natural Earth,
public domain) is **bundled** — no separate download. Regenerate it with
`python scripts/build_geojson.py` if needed.

---

## Project Structure

```
.
├── app.py                  # Entry point — st.navigation (Про ресурс / Аналітика)
├── config.py               # URLs, oblast list, aliases, constants
├── views/
│   ├── about.py            # "Про ресурс" — overview + data methodology
│   └── analytics.py        # Historical analysis (5 tabs)
├── src/
│   ├── data/
│   │   ├── loader.py       # Download + unify oblast episodes + dedup
│   │   ├── validators.py   # Schema / dedup / region quality checks
│   │   ├── transforms.py   # Timezone, censoring, permanent-region exclusion
│   │   └── cache.py        # @st.cache_data loader
│   ├── analysis/
│   │   ├── descriptive.py  # EDA, frequency, regional breakdown
│   │   ├── seasonality.py  # Temporal heatmaps
│   │   ├── correlation.py  # Co-occurrence + sliding-window wave detection
│   │   └── survival.py     # Kaplan-Meier (requires lifelines)
│   └── ui/
│       ├── charts.py       # Plotly figure factories + Attack Wave animation
│       ├── maps.py         # Choropleth (mapbox) + click selection
│       └── components.py   # Streamlit widgets (filters, metrics)
├── tests/
│   ├── test_data.py        # Level filter, dedup, timezone, censoring, permanent
│   ├── test_analysis.py    # EDA, seasonality, correlation, wave detection
│   └── fixtures/sample_alerts.csv
├── scripts/
│   ├── download_data.py    # Fetch both datasets
│   ├── cross_check.py      # Compare official vs volunteer (data insurance)
│   └── build_geojson.py    # Regenerate the oblasts GeoJSON
├── assets/ukraine_oblasts.geojson   # bundled (Natural Earth)
└── requirements.txt
```

---

## Run Tests

```bash
pytest tests/ -v
```

Tests cover the data-correctness rules (level filter, dedup, timezone,
censoring, permanent regions) and all analysis functions including the
sliding-window wave-detection algorithm.

---

## Advanced Feature: Attack Wave Replay

`propagation_events()` uses a **sliding two-pointer window** over all alert
starts sorted by time: it finds bursts where ≥N distinct oblasts raise an alert
within a rolling time window, emitting non-overlapping wave events. Each event is
rendered as an **animated mapbox choropleth** with 5-minute frames, revealing two
empirically distinct attack morphologies:

- **Shahed drone waves** — alerts propagate East→West over 1–3 hours
- **Missile strikes** — near-simultaneous alerts across many regions

---

## License

MIT — see [LICENSE](LICENSE).

- Dataset: [Vadimkin/ukrainian-air-raid-sirens-dataset](https://github.com/Vadimkin/ukrainian-air-raid-sirens-dataset)
- Map geometry: [Natural Earth](https://www.naturalearthdata.com/) (public domain)
