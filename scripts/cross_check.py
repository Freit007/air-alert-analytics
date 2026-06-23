"""Cross-check the PRIMARY (official) source against the VOLUNTEER source.

This is the data-correctness "insurance" step for the project: two independently
collected datasets are compared per-oblast. If both agree on the relative
ordering (e.g. Dnipropetrovsk >> Kyiv city), a counter-intuitive result is
real, not a bug.

Run from project root:
    python scripts/cross_check.py
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))
logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)

from config import (
    RAW_CSV_PATH, VOLUNTEER_CSV_PATH,
    OBLAST_ALIAS_MAP, NEAR_PERMANENT_REGIONS,
)


def _load_official() -> pd.DataFrame:
    """Official source, unified into per-oblast episodes (same as the app)."""
    from src.data.loader import load_raw
    df = load_raw(RAW_CSV_PATH)
    df["region"] = df["region"].map(lambda x: OBLAST_ALIAS_MAP.get(x, x))
    return df


def _load_volunteer() -> pd.DataFrame:
    df = pd.read_csv(VOLUNTEER_CSV_PATH).drop_duplicates()
    df["region"] = df["region"].map(lambda x: OBLAST_ALIAS_MAP.get(x, x))
    # Restrict to the official coverage window for a fair comparison
    df["started_at"] = pd.to_datetime(df["started_at"], utc=True)
    df = df[df["started_at"] >= pd.Timestamp("2022-03-15", tz="UTC")]
    return df


def main() -> None:
    if not RAW_CSV_PATH.exists() or not VOLUNTEER_CSV_PATH.exists():
        logger.error("Datasets missing — run scripts/download_data.py first.")
        sys.exit(1)

    off = _load_official()
    vol = _load_volunteer()

    oc = off.groupby("region").size().rename("official")
    vc = vol.groupby("region").size().rename("volunteer")

    cmp = pd.concat([oc, vc], axis=1).fillna(0).astype(int)
    cmp = cmp[~cmp.index.isin(NEAR_PERMANENT_REGIONS)]
    cmp["ratio_off_vol"] = (cmp["official"] / cmp["volunteer"].replace(0, 1)).round(2)
    cmp = cmp.sort_values("official", ascending=False)

    logger.info("=== Cross-source per-oblast alert counts (oblast-level) ===")
    print(cmp.to_string())

    # Spearman rank correlation between the two sources — should be high
    rho = cmp["official"].corr(cmp["volunteer"], method="spearman")
    logger.info("Spearman rank correlation (official vs volunteer): %.3f", rho)

    # Specific sanity statement
    for r in ("м. Київ", "Дніпропетровська"):
        if r in cmp.index:
            row = cmp.loc[r]
            logger.info("  %s: official=%d, volunteer=%d", r, row["official"], row["volunteer"])

    if rho >= 0.8:
        logger.info("✓ Sources strongly agree (ρ ≥ 0.8) — counts are trustworthy.")
    else:
        logger.warning("⚠ Sources diverge (ρ < 0.8) — investigate before trusting counts.")


if __name__ == "__main__":
    main()
