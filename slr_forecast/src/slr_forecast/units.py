"""
Unit conversion constants and helper functions.

Project-wide convention
-----------------------
- **Sign**: positive = sea level rise (SLR convention).
- **Internal units**: meters (sea level), degC (temperature anomaly),
  decimal years (time), m/yr (rate), m/yr² (acceleration).
- Convert to display units (mm, cm) only at the plotting layer.

All conversion factors are defined here.  No module should hard-code
362.5, 1000.0, or 3.625e14 for unit conversion — import from this file.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Fundamental constants
# ---------------------------------------------------------------------------
OCEAN_AREA_KM2: float = 362.5e6
"""Global ocean area, km².  Source: IPCC AR6 / Gregory et al. (2019)."""

OCEAN_AREA_M2: float = 3.625e14
"""Global ocean area, m² (= OCEAN_AREA_KM2 × 1e6)."""

SECONDS_PER_YEAR: float = 365.25 * 24 * 3600
"""Mean seconds per Julian year."""

# ---------------------------------------------------------------------------
# Sea-level conversions
# ---------------------------------------------------------------------------
M_TO_MM: float = 1000.0
"""Multiply meters by this to get millimeters."""

MM_TO_M: float = 1.0 / M_TO_MM
"""Multiply millimeters by this to get meters."""

M_TO_CM: float = 100.0
"""Multiply meters by this to get centimeters."""

CM_TO_M: float = 1.0 / M_TO_CM
"""Multiply centimeters by this to get meters."""

GT_TO_MM_SLE: float = 1.0 / 362.5
"""Multiply Gt of ice/water by this to get mm of sea-level equivalent.
   1 Gt distributed over 362.5 × 10⁶ km² ≈ 1/362.5 mm."""

GT_TO_M_SLE: float = GT_TO_MM_SLE * MM_TO_M
"""Multiply Gt of ice/water by this to get meters of sea-level equivalent."""

# ---------------------------------------------------------------------------
# Convenience functions
# ---------------------------------------------------------------------------

def gt_to_m_sle(gt: float | np.ndarray) -> float | np.ndarray:
    """Convert gigatonnes of ice/water mass change to meters SLE."""
    return gt * GT_TO_M_SLE


def m_to_mm(m: float | np.ndarray) -> float | np.ndarray:
    """Convert meters to millimeters (for display)."""
    return m * M_TO_MM


def mm_to_m(mm: float | np.ndarray) -> float | np.ndarray:
    """Convert millimeters to meters (internal units)."""
    return mm * MM_TO_M


# ---------------------------------------------------------------------------
# DataFrame metadata helpers
# ---------------------------------------------------------------------------

def tag_units(df: pd.DataFrame, units: dict[str, str]) -> pd.DataFrame:
    """Attach unit metadata to a DataFrame.

    Parameters
    ----------
    df : DataFrame
        The data to tag (modified in place).
    units : dict
        Mapping of column name → unit string, e.g.
        ``{'gmsl': 'm', 'rate': 'm/yr'}``.

    Returns
    -------
    DataFrame
        The same object, with ``df.attrs['units']`` set.
    """
    df.attrs["units"] = units
    return df


def tag_sign_convention(df: pd.DataFrame, convention: str = "slr") -> pd.DataFrame:
    """Attach sign-convention metadata to a DataFrame.

    Parameters
    ----------
    df : DataFrame
    convention : str
        ``'slr'`` (positive = sea level rise) or ``'glaciology'``
        (positive = mass gain).

    Returns
    -------
    DataFrame
    """
    if convention not in ("slr", "glaciology"):
        raise ValueError(f"Unknown sign convention: {convention!r}")
    df.attrs["sign_convention"] = convention
    return df


def tag_baseline(df: pd.DataFrame, baseline_year: float, window: tuple[int, int]) -> pd.DataFrame:
    """Attach baseline metadata to a DataFrame.

    Parameters
    ----------
    df : DataFrame
    baseline_year : float
    window : tuple[int, int]
        (start_year, end_year) of the averaging window.

    Returns
    -------
    DataFrame
    """
    df.attrs["baseline_year"] = baseline_year
    df.attrs["baseline_window"] = window
    return df


# ---------------------------------------------------------------------------
# Assertion guards
# ---------------------------------------------------------------------------

def assert_slr_convention(df: pd.DataFrame, label: str = "") -> None:
    """Raise if the DataFrame is not tagged with SLR sign convention."""
    conv = df.attrs.get("sign_convention")
    if conv != "slr":
        prefix = f"[{label}] " if label else ""
        raise ValueError(
            f"{prefix}Expected sign_convention='slr', got {conv!r}.  "
            "Tag with units.tag_sign_convention(df, 'slr') after ingestion."
        )


def assert_units_meters(values, label: str = "") -> None:
    """Warn if sea-level values look like they are in mm rather than m.

    A simple heuristic: if max |value| > 2.0, the data are probably in mm
    (no GMSL anomaly in meters should exceed ~2 m for 21st-century work).
    """
    arr = np.asarray(values)
    finite = arr[np.isfinite(arr)]
    if len(finite) == 0:
        return
    mx = np.max(np.abs(finite))
    if mx > 2.0:
        prefix = f"[{label}] " if label else ""
        raise ValueError(
            f"{prefix}Max |value| = {mx:.2f}, which exceeds 2.0 m.  "
            "Data are likely in mm — convert with mm_to_m() before use."
        )


def assert_sigma_positive(sigma, label: str = "") -> None:
    """Raise if any uncertainty values are negative (known IMBIE quirk)."""
    arr = np.asarray(sigma)
    finite = arr[np.isfinite(arr)]
    if len(finite) > 0 and np.any(finite < 0):
        prefix = f"[{label}] " if label else ""
        raise ValueError(
            f"{prefix}Negative sigma values detected.  "
            "Apply np.abs() at ingestion (known IMBIE data quirk)."
        )
