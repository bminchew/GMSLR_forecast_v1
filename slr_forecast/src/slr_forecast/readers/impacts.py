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

    Power-law fit to Jevrejeva et al. (2018). SSP2 socioeconomics,
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
    slr = np.asarray(slr_m, dtype=float)
    return np.where(slr > 0, 16153.0 * slr**0.895, 0.0)
