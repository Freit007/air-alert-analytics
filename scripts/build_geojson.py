"""Regenerate assets/ukraine_oblasts.geojson from Natural Earth (public domain).

Maps each Ukraine admin-1 feature's `name_uk` to our canonical short region name
via OBLAST_ALIAS_MAP, writing `properties.region` (used as featureidkey).

Run from project root:
    python scripts/build_geojson.py
"""
from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

import requests

sys.path.insert(0, str(Path(__file__).parent.parent))
logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)

from config import GEOJSON_PATH, OBLAST_ALIAS_MAP

NE_URL = (
    "https://raw.githubusercontent.com/nvkelso/natural-earth-vector/master/"
    "geojson/ne_10m_admin_1_states_provinces.geojson"
)

# Resolve the Kiev / Kiev City ambiguity by English name
EN_OVERRIDE = {"Kiev": "Київська", "Kiev City": "м. Київ"}


def main() -> None:
    logger.info("Downloading Natural Earth admin-1…")
    data = requests.get(NE_URL, timeout=60).json()
    ukraine = [f for f in data["features"]
               if f.get("properties", {}).get("admin") == "Ukraine"]
    logger.info("Ukraine features: %d", len(ukraine))

    out = []
    for f in ukraine:
        p = f["properties"]
        name_en, name_uk = p.get("name", ""), p.get("name_uk", "")
        region = EN_OVERRIDE.get(name_en) or OBLAST_ALIAS_MAP.get(name_uk, name_uk)
        out.append({
            "type": "Feature",
            "id": region,
            "properties": {"region": region, "name_uk": name_uk, "name_en": name_en},
            "geometry": f["geometry"],
        })

    GEOJSON_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(GEOJSON_PATH, "w", encoding="utf-8") as fh:
        json.dump({"type": "FeatureCollection", "features": out},
                  fh, ensure_ascii=False, separators=(",", ":"))

    size_kb = GEOJSON_PATH.stat().st_size / 1024
    logger.info("Wrote %d features → %s (%.0f KB)", len(out), GEOJSON_PATH, size_kb)


if __name__ == "__main__":
    main()
