"""
SLR impact estimation functions.

These are empirical scaling relationships, not data readers per se,
but they operate on sea-level data and are used in the analysis.
"""

from __future__ import annotations

import numpy as np


# Damage function knots from Jevrejeva et al. (2018), SSP2 without
# additional adaptation. The single source for every damage calculation.
#   0.52, 0.63 m: 2100 medians for 1.5 and 2 C, US$10.2T and 11.7T (text)
#   0.86, 1.80 m: RCP8.5 median and 95th percentile, US$14.3T and 27.0T (text)
#   0.20 m: read from their Fig. 4a, where the 1.5 C, 2 C and RCP8.5
#           median curves pass through about US$2.3-2.7T
JEVREJEVA2018_SLR_M = np.array([0.00, 0.20, 0.52, 0.63, 0.86, 1.80])
JEVREJEVA2018_COST_B = np.array([0.0, 2500.0, 10200.0, 11700.0, 14300.0, 27000.0])


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

    Piecewise-linear interpolation through ``JEVREJEVA2018_SLR_M`` and
    ``JEVREJEVA2018_COST_B``, linearly extrapolated beyond 1.8 m using
    the slope of the last segment (~13,510 B$/m). SSP2 socioeconomics,
    no additional coastal adaptation. Costs in billions USD/yr (2014$).

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
    _slr_pts = JEVREJEVA2018_SLR_M
    _cost_pts = JEVREJEVA2018_COST_B
    _last_slope = (_cost_pts[-1] - _cost_pts[-2]) / (_slr_pts[-1] - _slr_pts[-2])

    slr = np.asarray(slr_m, dtype=float)
    scalar = slr.ndim == 0
    slr = np.atleast_1d(slr)
    result = np.interp(slr, _slr_pts, _cost_pts)
    mask = slr > _slr_pts[-1]
    result[mask] = _cost_pts[-1] + _last_slope * (slr[mask] - _slr_pts[-1])
    result = np.where(slr < 0, 0.0, result)
    return float(result[0]) if scalar else result
