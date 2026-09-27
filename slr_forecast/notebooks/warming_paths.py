"""Shared within-scenario warming paths for the component projections.

The IPCC AR6 SSP GMST projections carry a 5--95% range around each median
trajectory, mostly from climate sensitivity.  That spread persists across
years (a world that runs warm stays warm), so it is represented here as one
percentile path per ensemble member rather than as independent annual noise.

Member k follows the standard-normal quantile z_k for the whole century.  The
offset from the median is

    delta_k(t) = z_k * [hw(t) - hw(t_a)]   for t > t_a,   0 otherwise,

where hw(t) = (upper - median) / 1.645 for z_k > 0 and
(median - lower) / 1.645 for z_k < 0, so the skew of the AR6 range is kept,
and t_a is WARMING_ANCHOR_YEAR (warming to date is observed).

The z vector is drawn from its own seed and is the same in every component
notebook and every SSP, so member k is one physical world throughout the
component sum.  Nothing here draws from a component's random generator.
"""

import numpy as np

try:
    from slr_forecast.config import (N_SAMPLES, SEEDS, SAMPLE_WARMING_PATHS,
                                     WARMING_ANCHOR_YEAR)
except ImportError:
    N_SAMPLES = 2000
    SEEDS = {'warming': 2027}
    SAMPLE_WARMING_PATHS = True
    WARMING_ANCHOR_YEAR = 2024.0

Z_90 = 1.645  # standard-normal quantile of the AR6 5th/95th percentiles


def warming_z(n_samples=N_SAMPLES):
    """Per-member warming quantiles z_k (zeros if SAMPLE_WARMING_PATHS is off)."""
    if not SAMPLE_WARMING_PATHS:
        return np.zeros(n_samples)
    return np.random.default_rng(SEEDS['warming']).standard_normal(n_samples)


def ssp_row_offsets(df_ssp, z, anchor=WARMING_ANCHOR_YEAR):
    """Offsets from the median path on the rows of an SSP temperature table.

    Parameters
    ----------
    df_ssp : DataFrame
        IPCC SSP GMST table with 'decimal_year', 'temperature',
        'temperature_lower' (5th) and 'temperature_upper' (95th).
    z : ndarray (n,)
        Per-member quantiles from ``warming_z``.

    Returns
    -------
    ndarray (n, n_rows), in degrees C, aligned with the rows of ``df_ssp``.
    """
    yr = df_ssp['decimal_year'].values.astype(float)
    med = df_ssp['temperature'].values
    hw_hi = (df_ssp['temperature_upper'].values - med) / Z_90
    hw_lo = (med - df_ssp['temperature_lower'].values) / Z_90
    grow_hi = np.where(yr > anchor, hw_hi - np.interp(anchor, yr, hw_hi), 0.0)
    grow_lo = np.where(yr > anchor, hw_lo - np.interp(anchor, yr, hw_lo), 0.0)
    z = np.asarray(z, dtype=float)[:, None]
    return np.where(z > 0, z * grow_hi[None, :], z * grow_lo[None, :])


def offsets_on_grid(df_ssp, z, grid, zero_through=None, anchor=WARMING_ANCHOR_YEAR):
    """Row offsets interpolated onto a time grid, shape (n, len(grid)).

    ``zero_through`` sets the offset to zero at and before that time, for
    grids spliced onto observed GMST at the end of the observed record.
    """
    rows = ssp_row_offsets(df_ssp, z, anchor)
    yr = df_ssp['decimal_year'].values.astype(float)
    out = np.array([np.interp(grid, yr, r, left=0.0) for r in rows])
    if zero_through is not None:
        out[:, np.asarray(grid) <= zero_through] = 0.0
    return out
