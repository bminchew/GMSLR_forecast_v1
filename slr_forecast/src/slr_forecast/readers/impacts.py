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


# CoastalDEM hypsometry from Kulp & Strauss (2019), Supplementary Data 2,
# row "World": millions of people (2010 LandScan) on land below each height
# above present mean higher-high water (MHHW), at 1 m steps from 0 to 10 m.
KULPSTRAUSS2019_HEIGHT_M = np.arange(11.0)
KULPSTRAUSS2019_POP_MILLIONS = np.array([110.0, 230.0, 400.0, 550.0, 680.0, 770.0,
                                         840.0, 900.0, 950.0, 1000.0, 1040.0])
# Present-day population below the annual flood level (Table 1, CoastalDEM,
# RL1). The height at which the hypsometry reaches it, about 1.12 m, is the
# effective global annual-flood height above MHHW.
KULPSTRAUSS2019_RL1_PRESENT_MILLIONS = 250.0
KULPSTRAUSS2019_RL1_HEIGHT_M = float(np.interp(KULPSTRAUSS2019_RL1_PRESENT_MILLIONS,
                                               KULPSTRAUSS2019_POP_MILLIONS,
                                               KULPSTRAUSS2019_HEIGHT_M))


def people_displaced_kulpstrauss2019(slr_m) -> np.ndarray:
    """Estimate millions of people on land below annual flood level.

    Kulp & Strauss (2019) CoastalDEM hypsometry evaluated at
    ``slr_m + KULPSTRAUSS2019_RL1_HEIGHT_M``, piecewise linear between the
    1 m steps and linearly extrapolated beyond 10 m above MHHW (rise above
    about 8.9 m) using the slope of the last step. Kulp & Strauss use local
    one-year return levels (0.2-2.8 m above MHHW), so the single global
    flood height is an approximation; against the 12 scenario medians of
    their Table 1 (K14 and K17, RCP2.6/4.5/8.5, 2050 and 2100) it gives an
    rms residual of 6 M and a largest residual of 10 M.

    Elevation-only exposure: 2010 population, no coastal defenses. Kulp &
    Strauss state that their results "should therefore not be taken as
    projected impacts" and describe them as a threat indicator.

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
    _h = KULPSTRAUSS2019_HEIGHT_M
    _pop = KULPSTRAUSS2019_POP_MILLIONS
    _last_slope = (_pop[-1] - _pop[-2]) / (_h[-1] - _h[-2])

    slr = np.asarray(slr_m, dtype=float)
    scalar = slr.ndim == 0
    height = np.atleast_1d(slr) + KULPSTRAUSS2019_RL1_HEIGHT_M
    result = np.interp(height, _h, _pop)
    mask = height > _h[-1]
    result[mask] = _pop[-1] + _last_slope * (height[mask] - _h[-1])
    return float(result[0]) if scalar else result


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
