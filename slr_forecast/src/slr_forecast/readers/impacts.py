"""
SLR impact estimation functions.

These are empirical scaling relationships, not data readers per se,
but they operate on sea-level data and are used in the analysis.
"""

from __future__ import annotations

import numpy as np


def people_displaced_kulpstrauss2019(slr_m) -> np.ndarray:
    """Estimate millions of people on land below annual flood level.

    Quadratic fit to Kulp & Strauss (2019) CoastalDEM data.
    Elevation-only exposure — no coastal defenses.

    Parameters
    ----------
    slr_m : float or array
        GMSL rise in **meters** above present (~2000).

    Returns
    -------
    float or ndarray
        Millions of people exposed.

    Reference
    ---------
    Kulp & Strauss (2019), Nature Comms, 10, 4844.
    """
    slr = np.asarray(slr_m, dtype=float)
    return 17.03 * slr**2 + 145.0 * slr + 260.0


def slr_cost_jevrejeva2018(slr_m) -> np.ndarray:
    """Estimate global annual flood cost without additional adaptation.

    Piecewise-linear interpolation of Jevrejeva et al. (2018),
    Fig 4a. SSP2 socioeconomics, no additional coastal adaptation.
    Exact at all reported data points; linear extrapolation beyond
    1.8 m. Costs in billions USD/yr (2014$).

    Parameters
    ----------
    slr_m : float or array
        GMSL rise in **meters** above present (~2000).

    Returns
    -------
    float or ndarray
        Billions USD per year.

    Reference
    ---------
    Jevrejeva et al. (2018), Env. Res. Lett., 13, 074014.
    """
    _slr_pts = np.array([0.0, 0.20, 0.52, 0.63, 0.86, 1.80])
    _cost_pts = np.array([0.0, 1000.0, 10200.0, 11700.0, 14000.0, 27000.0])
    _last_slope = (_cost_pts[-1] - _cost_pts[-2]) / (_slr_pts[-1] - _slr_pts[-2])

    slr = np.asarray(slr_m, dtype=float)
    scalar = slr.ndim == 0
    slr = np.atleast_1d(slr)
    result = np.interp(slr, _slr_pts, _cost_pts)
    mask = slr > _slr_pts[-1]
    result[mask] = _cost_pts[-1] + _last_slope * (slr[mask] - _slr_pts[-1])
    result = np.where(slr < 0, 0.0, result)
    return float(result[0]) if scalar else result
