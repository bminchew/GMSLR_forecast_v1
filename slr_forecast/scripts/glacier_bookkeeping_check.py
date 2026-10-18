"""Bookkeeping check: glacier hindcast on Frederikse et al. (2020) definitions.

Our glacier component is GlaMBIE's global total, which includes the
Greenland periphery (RGI 5) and the Antarctic and subantarctic glaciers
(RGI 19). Frederikse et al. (2020) exclude both from their glacier term:
they add the Greenland periphery to the Greenland Ice Sheet and assume no
loss from the Antarctic periphery. This script measures how much of the
20th-century glacier hindcast difference is due to that definition.

For each GlaMBIE series (global; global minus RGI 5 and 19; RGI 5; RGI 19)
the rate model of component_glacier.ipynb, dH/dt = b T + c, is fitted by
weighted least squares on calendar years 2000-2023 (T: Berkeley Earth
annual GMST rel. 1995-2005), and integrated back to give the cumulative
contribution relative to 2000 (level at the start of each year, as in the
production hindcast). GlaMBIE's 95% intervals are converted to 1 sigma; the
global and global-minus-RGI-5-and-19 series also carry the production
sigma_gl,extra in quadrature (it was fitted to the global series). This is
a simplified fit (no prior on b, c free, rate space), so only its
difference between series is used: the production hindcasts are rebuilt
from component_results.h5 (glacier_zemp_diagnostic.production_hindcast_mm)
and shifted by that difference. Nothing here feeds the pipeline.

Usage:  python scripts/glacier_bookkeeping_check.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'notebooks'))
sys.path.insert(0, str(ROOT / 'scripts'))
from glacier_zemp_diagnostic import production_hindcast_mm, sigma_gl_extra_mm
GL = (ROOT / 'data/raw/glaciers/glambie/GlaMBIE_Data_DOI_10.5904_wgms-glambie-2024-07/'
      'glambie_results_20240716/calendar_years')
GT_PER_MM = 362.5
FRED = {1900: (-72.7, -40.9), 1920: (-55.2, -33.9), 1950: (-26.3, -13.2)}   # glaciers, Greenland


def read(name):
    d = pd.read_csv(GL / f'{name}.csv')
    yr = d['start_dates'].astype(int).values
    rate = -d['combined_gt'].values / GT_PER_MM                 # mm/yr SLE, + = SLR
    err = d['combined_gt_errors'].values / 1.96 / GT_PER_MM     # 95% half-widths -> 1 sigma
    return pd.DataFrame({'rate': rate, 'err': err}, index=yr)


def fit(s, T):
    yy = s.index.values
    X = np.column_stack([T.reindex(yy).values, np.ones(len(yy))])
    w = 1 / s['err'].values ** 2
    beta = np.linalg.solve(X.T @ (X * w[:, None]), X.T @ (s['rate'].values * w))
    return beta                                                  # b (mm/yr/K), c (mm/yr)


def main():
    be = pd.read_hdf(ROOT / 'data/processed/slr_processed_data.h5', 'harmonized/df_berkeley_h')
    T = be['temperature'].groupby(be.index.year).mean()
    T = T - T.loc[1995:2005].mean()

    g, r5, r19 = read('0_global'), read('5_greenland_periphery'), read('19_antarctic_and_subantarctic')
    ex = pd.DataFrame({'rate': g['rate'] - r5['rate'] - r19['rate'],
                       'err': np.sqrt(g['err'] ** 2 + r5['err'] ** 2 + r19['err'] ** 2)})
    sx = sigma_gl_extra_mm()                                     # production model-inadequacy term
    g = g.assign(err=np.hypot(g['err'], sx))
    ex = ex.assign(err=np.hypot(ex['err'], sx))
    series = {'global (as in production)': g, 'global minus RGI 5 and 19': ex,
              'RGI 5 Greenland periphery': r5, 'RGI 19 Antarctic periphery': r19}

    print(f'GlaMBIE calendar years {g.index.min()}-{g.index.max()}; mean rates (mm/yr SLE):')
    for k, s in series.items():
        print(f'  {k:28s} {s["rate"].mean():5.3f}  ({100 * s["rate"].mean() / g["rate"].mean():3.0f}% of global)')

    hind = {}
    print('\nRate model dH/dt = b T + c (WLS, 2000-2023) and hindcast relative to 2000 (mm):')
    for k, s in series.items():
        b, c = fit(s, T)
        rate = b * T + c
        hind[k] = {y: -rate.loc[y:1999].sum() for y in FRED}
        print(f'  {k:28s} b {b:5.2f}, c {c:5.2f};  '
              + ', '.join(f'{y}: {hind[k][y]:6.1f}' for y in FRED))

    prod = production_hindcast_mm(tuple(FRED))                   # (calibrated, incl. uncharted, Greenland)
    OURS = {y: (prod[y][1], prod[y][2]) for y in FRED}
    print('\nComparison on Frederikse definitions (mm rel. 2000; ours = production incl. '
          'uncharted glaciers, from component_results.h5):')
    print('  year   glaciers: ours global / ours excl. 5,19 / Frederikse    '
          'Greenland: ours / ours + RGI 5 / Frederikse')
    for y in FRED:
        d_ex = hind['global minus RGI 5 and 19'][y] - hind['global (as in production)'][y]
        print(f'  {y}   {OURS[y][0]:6.1f} / {OURS[y][0] + d_ex:6.1f} / {FRED[y][0]:6.1f}'
              f'                    {OURS[y][1]:6.1f} / {OURS[y][1] + hind["RGI 5 Greenland periphery"][y]:6.1f}'
              f' / {FRED[y][1]:6.1f}')
    print('  (ours excl. 5,19 = production value shifted by the WLS difference between the '
          'global and excluded fits; RGI 19 is dropped, as Frederikse assume no loss there)')


if __name__ == '__main__':
    main()
