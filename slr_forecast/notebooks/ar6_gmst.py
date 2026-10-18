"""AR6 SSP GMST on the Berkeley Earth basis, shared by every component.

The AR6 GMST tables (`projections/temp/*` in slr_processed_data.h5) give
annual means relative to 1850-1900, stamped at 1 January. Every component
model is driven by Berkeley Earth GMST relative to 1995-2005 through the end
of the observed record (2024) and by the AR6 median path after it. Two
choices must therefore be the same in every component (OC-8 / SF-3):

- Time stamps: AR6 annual means are placed at mid-year (`midyear`), like the
  Berkeley annual means they continue.
- Offset: one scenario-independent offset puts AR6 on the Berkeley basis
  (`ar6_to_berkeley_offset`): the SSP2-4.5 median minus Berkeley Earth,
  averaged over the annual means 2015.5-2024.5, the most recent decade both
  cover. Members follow observed GMST through 2024, so the offset only sets
  where the AR6 path takes over after it.

Before 2026-10-17 the ocean used this definition (0.755 C) and the glacier,
Greenland, EAIS and Peninsula notebooks matched AR6 Historical to Berkeley
over 1995-2005 with 1 January stamps (0.726 C).
"""

import numpy as np
import pandas as pd

OFFSET_REF_KEY = 'projections/temp/SSP2_4_5'
OFFSET_WINDOW = (2015.5, 2024.5)          # mid-year labels, inclusive
BERKELEY_KEY = 'harmonized/df_berkeley_h'


def midyear(df):
    """Copy of an AR6 annual GMST table with mid-year time stamps."""
    out = df.copy()
    out['decimal_year'] = out['decimal_year'].astype(float) + 0.5
    return out


def berkeley_annual(h5_path):
    """Berkeley Earth annual means of full calendar years, mid-year labels,
    relative to its 1995-2005 mean (the basis every component uses)."""
    d = pd.read_hdf(h5_path, BERKELEY_KEY)
    t = (d.index.year + (d.index.month - 0.5) / 12.0).values
    T = d['temperature'].values
    T = T - np.nanmean(T[(t >= 1995.0) & (t < 2006.0)])
    yr = np.floor(t).astype(int)
    years, counts = np.unique(yr, return_counts=True)
    years = years[counts == 12]
    return years + 0.5, np.array([T[yr == y].mean() for y in years])


def ar6_to_berkeley_offset(h5_path):
    """Offset (degC) to subtract from AR6 GMST to put it on the Berkeley basis."""
    ref = midyear(pd.read_hdf(h5_path, OFFSET_REF_KEY))
    by, bT = berkeley_annual(h5_path)
    lo, hi = OFFSET_WINDOW
    a = ref['decimal_year'].values
    m_ar6 = (a >= lo) & (a <= hi)
    m_be = (by >= lo) & (by <= hi)
    if m_ar6.sum() != m_be.sum() or m_ar6.sum() == 0:
        raise ValueError(f'offset window {OFFSET_WINDOW} not covered by both records')
    return float(ref['temperature'].values[m_ar6].mean() - bT[m_be].mean())
