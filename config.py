"""Centralized project configuration."""
from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

# ── Paths ────────────────────────────────────────────────────────────────────
ROOT_DIR = Path(__file__).parent
DATA_DIR = Path(os.getenv("DATA_DIR", ROOT_DIR / "data"))
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
ASSETS_DIR = ROOT_DIR / "assets"

RAW_CSV_PATH = RAW_DIR / "official_data_uk.csv"
VOLUNTEER_CSV_PATH = RAW_DIR / "volunteer_data_uk.csv"
GEOJSON_PATH = ASSETS_DIR / "ukraine_oblasts.geojson"

# ── Vadimkin dataset ─────────────────────────────────────────────────────────
# PRIMARY: official data (government API via ukrainealarm), has a `level` column
# (oblast / raion / hromada) → we keep ONLY whole-oblast alerts.
# Each alert is recorded twice in the raw file → we drop exact duplicates.
VADIMKIN_OFFICIAL_CSV_URL = (
    "https://raw.githubusercontent.com/Vadimkin/"
    "ukrainian-air-raid-sirens-dataset/main/datasets/"
    "official_data_uk.csv"
)
# CROSS-CHECK: independent volunteer (Telegram-based) collection. Used by
# scripts/cross_check.py to validate per-oblast counts against the official source.
VADIMKIN_VOLUNTEER_CSV_URL = (
    "https://raw.githubusercontent.com/Vadimkin/"
    "ukrainian-air-raid-sirens-dataset/main/datasets/"
    "volunteer_data_uk.csv"
)
# Backwards-compatible alias (primary source used by the loader)
VADIMKIN_CSV_URL = VADIMKIN_OFFICIAL_CSV_URL

# ── Timezone ─────────────────────────────────────────────────────────────────
KYIV_TZ = "Europe/Kyiv"

# ── Oblast metadata (25 locations tracked by raid.fly.dev) ───────────────────
# lat/lon = approximate centroid; en = English name for display
OBLASTS: list[dict] = [
    {"uk": "Вінницька",          "en": "Vinnytsia",        "lat": 49.23, "lon": 28.47},
    {"uk": "Волинська",          "en": "Volyn",            "lat": 51.22, "lon": 24.69},
    {"uk": "Дніпропетровська",   "en": "Dnipropetrovsk",   "lat": 48.46, "lon": 35.05},
    {"uk": "Донецька",           "en": "Donetsk",          "lat": 48.03, "lon": 37.80},
    {"uk": "Житомирська",        "en": "Zhytomyr",         "lat": 50.26, "lon": 28.66},
    {"uk": "Закарпатська",       "en": "Zakarpattia",      "lat": 48.62, "lon": 22.29},
    {"uk": "Запорізька",         "en": "Zaporizhzhia",     "lat": 47.84, "lon": 35.19},
    {"uk": "Івано-Франківська",  "en": "Ivano-Frankivsk",  "lat": 48.92, "lon": 24.71},
    {"uk": "Київська",           "en": "Kyiv Oblast",      "lat": 50.51, "lon": 30.81},
    {"uk": "Кіровоградська",     "en": "Kirovohrad",       "lat": 48.51, "lon": 32.27},
    {"uk": "Луганська",          "en": "Luhansk",          "lat": 48.57, "lon": 39.33},
    {"uk": "Львівська",          "en": "Lviv",             "lat": 49.83, "lon": 23.99},
    {"uk": "Миколаївська",       "en": "Mykolaiv",         "lat": 47.06, "lon": 31.99},
    {"uk": "Одеська",            "en": "Odesa",            "lat": 46.48, "lon": 30.73},
    {"uk": "Полтавська",         "en": "Poltava",          "lat": 49.59, "lon": 34.55},
    {"uk": "Рівненська",         "en": "Rivne",            "lat": 50.62, "lon": 26.25},
    {"uk": "Сумська",            "en": "Sumy",             "lat": 51.13, "lon": 33.41},
    {"uk": "Тернопільська",      "en": "Ternopil",         "lat": 49.55, "lon": 25.59},
    {"uk": "Харківська",         "en": "Kharkiv",          "lat": 49.99, "lon": 36.23},
    {"uk": "Херсонська",         "en": "Kherson",          "lat": 46.64, "lon": 32.62},
    {"uk": "Хмельницька",        "en": "Khmelnytskyi",     "lat": 49.42, "lon": 26.98},
    {"uk": "Черкаська",          "en": "Cherkasy",         "lat": 49.44, "lon": 32.06},
    {"uk": "Чернівецька",        "en": "Chernivtsi",       "lat": 48.30, "lon": 25.93},
    {"uk": "Чернігівська",       "en": "Chernihiv",        "lat": 51.50, "lon": 31.30},
    {"uk": "м. Київ",            "en": "Kyiv (city)",      "lat": 50.45, "lon": 30.52},
]

OBLAST_UK_NAMES: list[str] = [o["uk"] for o in OBLASTS]
OBLAST_UK_TO_EN: dict[str, str] = {o["uk"]: o["en"] for o in OBLASTS}
OBLAST_CENTROIDS: dict[str, tuple[float, float]] = {
    o["uk"]: (o["lat"], o["lon"]) for o in OBLASTS
}

# Normalize common variant spellings found in the Vadimkin dataset
OBLAST_ALIAS_MAP: dict[str, str] = {
    "Вінницька область":         "Вінницька",
    "Волинська область":         "Волинська",
    "Дніпропетровська область":  "Дніпропетровська",
    "Донецька область":          "Донецька",
    "Житомирська область":       "Житомирська",
    "Закарпатська область":      "Закарпатська",
    "Запорізька область":        "Запорізька",
    "Івано-Франківська область": "Івано-Франківська",
    "Київська область":          "Київська",
    "Кіровоградська область":    "Кіровоградська",
    "Луганська область":         "Луганська",
    "Львівська область":         "Львівська",
    "Миколаївська область":      "Миколаївська",
    "Одеська область":           "Одеська",
    "Полтавська область":        "Полтавська",
    "Рівненська область":        "Рівненська",
    "Сумська область":           "Сумська",
    "Тернопільська область":     "Тернопільська",
    "Харківська область":        "Харківська",
    "Херсонська область":        "Херсонська",
    "Хмельницька область":       "Хмельницька",
    "Черкаська область":         "Черкаська",
    "Чернівецька область":       "Чернівецька",
    "Чернігівська область":      "Чернігівська",
    "Київ":                      "м. Київ",
    "місто Київ":                "м. Київ",
}

# ── Analysis constants ────────────────────────────────────────────────────────
COVER_START_DATE = "2022-02-24"   # official start of full-scale invasion
OFFICIAL_START_DATE = "2022-03-15"  # official API coverage start per Vadimkin docs

# Alerts in Luhansk & Crimea are near-permanent → flag, not filter by default
NEAR_PERMANENT_REGIONS: set[str] = {"Луганська", "АР Крим", "Крим"}

# Any single raw alert longer than this is a near-frontline/occupied-zone artefact
# (e.g. Вовчанська hromada 604 days, Нікопольський район 20+ days). Cap before
# interval merging so one multi-month hromada "alert" does not inflate the whole oblast.
MAX_RAW_ALERT_HOURS: int = 12
