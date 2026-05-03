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
BASELINE_YEAR: float = 2000.0
"""Reference year for sea-level and temperature anomalies.
Changed from 2005 to 2000 to align with the manuscript convention
of reporting 21st-century contributions relative to the year 2000."""

WAIS_ONSET_YEAR: float = 2010.0
"""Year at which WAIS contribution begins ramping.  Physically motivated
by the IMBIE structural break (~2010) when WAIS mass loss accelerated.
Distinct from BASELINE_YEAR to allow independent adjustment."""

BASELINE_WINDOW: tuple[int, int] = (1995, 2005)
"""Averaging window (inclusive start, inclusive end) used when rebasing
anomaly series to BASELINE_YEAR.  Centered on 2000 but extended to 2005
to include the transition to the Argo era."""

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

H5_OBS_PATH: Path = PROCESSED_DATA_DIR / "slr_processed_data.h5"
"""Observational data (NASA GMSL, Frederikse, Berkeley Earth, kinematics)."""

H5_COMP_PATH: Path = PROCESSED_DATA_DIR / "component_results.h5"
"""Per-component projection ensembles and fitted parameters."""

# Keep H5_PATH as alias for backward compatibility
H5_PATH: Path = H5_OBS_PATH

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

# ---------------------------------------------------------------------------
# Physical / unit constants
# ---------------------------------------------------------------------------
GT_TO_M_SLE: float = 1.0 / 362500.0
"""Gigatonnes of ice to metres of sea-level equivalent."""

PREIND_TO_BASELINE_M: float = 0.19
"""SLR from pre-industrial to BASELINE_YEAR (metres).
Frederikse et al. (2020): ~0.19 m from pre-industrial to ~2000."""

# ---------------------------------------------------------------------------
# Statistical constants
# ---------------------------------------------------------------------------
Z_94: float = 1.881
"""z-score for the 94% highest-density interval (two-tailed)."""

Z_90: float = 1.645
"""z-score for the 90% confidence interval (two-tailed)."""

# ---------------------------------------------------------------------------
# IPCC AR6 reference
# ---------------------------------------------------------------------------
IPCC_CONFIDENCE: str = "medium_confidence"
"""Confidence tier used when reading IPCC AR6 SLC NetCDF projections."""

IPCC_REF_PERIOD: tuple[int, int] = (1995, 2014)
"""IPCC AR6 reference period for sea-level projections."""

SAT_ERA_START: float = 1993.0
"""Start of the satellite altimetry era (TOPEX/Poseidon launch)."""

# ---------------------------------------------------------------------------
# Hamlington et al. (2024) satellite-era quadratic fit
# Comm. Earth Environ. 5:601, doi:10.1038/s43247-024-01761-5
# Quadratic fit to NASA altimetry 1993–2024.
# Rate at time t:  rate(t) = HAMLINGTON_ACCEL * (t - HAMLINGTON_T_REF) + HAMLINGTON_RATE
# All uncertainties are 90% CI.
# ---------------------------------------------------------------------------
HAMLINGTON_RATE: float = 3.3
"""Mean rate of GMSL rise over 1993–2024 (mm/yr), ref. to midpoint of record."""

HAMLINGTON_RATE_UNC: float = 0.3
"""90% CI half-width on HAMLINGTON_RATE (mm/yr)."""

HAMLINGTON_ACCEL: float = 0.08
"""Acceleration of GMSL rise (mm/yr²)."""

HAMLINGTON_ACCEL_UNC: float = 0.06
"""90% CI half-width on HAMLINGTON_ACCEL (mm/yr²)."""

HAMLINGTON_T_REF: float = 2008.5
"""Reference time (midpoint of 1993–2024 record)."""
