"""
Project-wide configuration constants.

All notebooks and modules should import from here rather than defining
their own copies. This is the single source of truth for baselines,
ensemble sizes, random seeds, and directory paths.
"""

import os
from pathlib import Path

# ---------------------------------------------------------------------------
# Baseline
# ---------------------------------------------------------------------------
BASELINE_YEAR: float = 2005.0
"""Reference year for sea-level and temperature anomalies."""

WAIS_ONSET_YEAR: float = 2010.0
"""Year at which WAIS contribution begins ramping.  Physically motivated
by the IMBIE structural break (~2010) when WAIS mass loss accelerated.
Distinct from BASELINE_YEAR to allow independent adjustment."""

BASELINE_WINDOW: tuple[int, int] = (1995, 2005)
"""Averaging window (inclusive start, inclusive end) used when rebasing
anomaly series to BASELINE_YEAR.  Width = BASELINE_YEAR ± 5 yr, matching
the convention established in the Frederikse and IPCC FACTS datasets."""

# ---------------------------------------------------------------------------
# Monte Carlo / sampling
# ---------------------------------------------------------------------------
N_SAMPLES: int = 2000
"""Default number of Monte Carlo ensemble members for projections."""

SEEDS: dict[str, int] = {
    "projections": 42,
    "bayesian": 42,
    "tests": 42,
}
"""Named random seeds for reproducibility.  Use as:
    rng = np.random.default_rng(SEEDS['projections'])
"""

# ---------------------------------------------------------------------------
# Directory layout
# ---------------------------------------------------------------------------
# PROJECT_ROOT is <repo>/slr_forecast/ regardless of where the caller lives.
# We locate it relative to this file: src/slr_forecast/config.py → ../../
PROJECT_ROOT: Path = Path(__file__).resolve().parents[2]

DATA_DIR: Path = PROJECT_ROOT / "data"
RAW_DATA_DIR: Path = DATA_DIR / "raw"
PROCESSED_DATA_DIR: Path = DATA_DIR / "processed"
H5_PATH: Path = PROCESSED_DATA_DIR / "slr_processed_data.h5"
FIG_DIR: Path = PROJECT_ROOT / "figures"

# ---------------------------------------------------------------------------
# SSP definitions
# ---------------------------------------------------------------------------
SSPS: list[str] = [
    "SSP1-1.9",
    "SSP1-2.6",
    "SSP2-4.5",
    "SSP3-7.0",
    "SSP5-8.5",
]
"""Available SSP scenarios in the IPCC AR6 FACTS projections."""
