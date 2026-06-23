"""One-time setup script: download the alert datasets.

Run from project root:
    python scripts/download_data.py
    python scripts/download_data.py --force      # re-download even if cached

Downloads:
  • official_data_uk.csv   — PRIMARY source (oblast-level, deduplicated at load)
  • volunteer_data_uk.csv  — CROSS-CHECK source (see scripts/cross_check.py)

The Ukraine oblasts GeoJSON (assets/ukraine_oblasts.geojson) is bundled in the
repo (Natural Earth, public domain) — no download needed.
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def download_alerts(force: bool = False) -> None:
    from src.data.loader import download_csv
    from config import (
        RAW_CSV_PATH, VADIMKIN_OFFICIAL_CSV_URL,
        VOLUNTEER_CSV_PATH, VADIMKIN_VOLUNTEER_CSV_URL,
    )

    logger.info("=== Vadimkin air-raid alerts datasets ===")
    logger.info("Downloading PRIMARY (official, oblast-level)…")
    download_csv(url=VADIMKIN_OFFICIAL_CSV_URL, dest=RAW_CSV_PATH, force=force)

    logger.info("Downloading CROSS-CHECK (volunteer)…")
    download_csv(url=VADIMKIN_VOLUNTEER_CSV_URL, dest=VOLUNTEER_CSV_PATH, force=force)

    logger.info("Download complete.")


def check_geojson() -> None:
    from config import GEOJSON_PATH
    if GEOJSON_PATH.exists():
        logger.info("GeoJSON bundled: %s", GEOJSON_PATH)
    else:
        logger.warning(
            "GeoJSON missing at %s — choropleth maps fall back to dot markers. "
            "Regenerate with scripts/build_geojson.py.",
            GEOJSON_PATH,
        )


def validate_dataset() -> None:
    from src.data.loader import load
    from src.data.validators import validate
    from src.data.transforms import apply_all

    logger.info("=== Validating dataset ===")
    raw = load(download_if_missing=False)
    validate(raw)
    transformed = apply_all(raw)
    logger.info(
        "Validation OK: %d rows after transform (%d oblast-level deduped before)",
        len(transformed), len(raw),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Download project datasets")
    parser.add_argument("--force", action="store_true", help="Re-download even if cached")
    parser.add_argument("--no-validate", action="store_true", help="Skip post-download validation")
    args = parser.parse_args()

    download_alerts(force=args.force)
    check_geojson()

    if not args.no_validate:
        validate_dataset()

    logger.info("=== Setup complete ===")


if __name__ == "__main__":
    main()
