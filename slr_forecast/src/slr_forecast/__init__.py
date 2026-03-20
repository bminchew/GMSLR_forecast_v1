"""
slr_forecast — Observation-based sea-level rise projection framework
====================================================================

A three-step framework for data-driven GMSL projections:
  1. Observational extrapolation (satellite-era quadratic)
  2. Aggregate semi-empirical model (GMSL rate vs GMST)
  3. Component-wise decomposition (thermosteric, glaciers, ice sheets, TWS)

Sign convention: positive = sea level rise (SLR convention throughout).
Internal units: meters (sea level), degC (temperature anomaly), decimal years.
"""

from slr_forecast.config import (
    BASELINE_YEAR,
    BASELINE_WINDOW,
    N_SAMPLES,
    SEEDS,
    PROJECT_ROOT,
    DATA_DIR,
    RAW_DATA_DIR,
    PROCESSED_DATA_DIR,
    H5_PATH,
    FIG_DIR,
)

from slr_forecast.units import (
    M_TO_MM,
    MM_TO_M,
    GT_TO_MM_SLE,
    GT_TO_M_SLE,
    OCEAN_AREA_KM2,
    OCEAN_AREA_M2,
    SECONDS_PER_YEAR,
    gt_to_m_sle,
    m_to_mm,
    mm_to_m,
)

__version__ = "0.1.0"
