"""Diagnostic: does the glacier rate model hold over 1962-2023?

The glacier component (component_glacier.ipynb) is dH/dt = b T + c, fitted
to GlaMBIE 2000-2023. Driven back to 1900 it gives almost no loss, against
about 70 mm in Frederikse et al. (2020), whose glacier term before 1961 is a
model reconstruction (Marzeion et al. 2015). Here the model is tested
against observations that reach further back: Zemp et al. (2019), global
glacier mass change for hydrological years 1962-2016 from glaciological and
geodetic data (Zenodo, doi:10.5281/zenodo.1492141; all 19 RGI regions, as
GlaMBIE's global total).

Steps
  1. Fit b, c on GlaMBIE 2000-2023 and predict Zemp 1962-1999 out of sample.
  2. Fit Zemp alone on 1962-1989 and on 1990-2016: if (b, c) differ beyond
     their uncertainty, the linear model with a constant c is misspecified
     (a delayed glacier response would show up this way).
  3. Overlap 2000-2016: mean Zemp minus GlaMBIE rate.
  4. Fit the combined record (Zemp 1962-1999, GlaMBIE 2000-2023).
  5. Cumulative hindcast relative to 2000 for each fit, against Frederikse.

Fits are generalised least squares in rate space with no prior on b.
Zemp's geodetic uncertainty (sig_Geod) is constant over long periods and is
treated as fully correlated across years; its other components, and the
GlaMBIE errors, as independent; Zemp's 95% intervals are converted to
1 sigma (slr_data_readers.read_zemp2019_global). T is Berkeley Earth annual GMST relative to
1995-2005; for Zemp's hydrological years (October to September, labelled by
the end year) T is 0.25 T(Y-1) + 0.75 T(Y). Intervals are 90%, inflated by
sqrt(chi2/dof) where that exceeds 1. Nothing here feeds the pipeline.

Usage:  python scripts/glacier_zemp_diagnostic.py
"""

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
import sys
sys.path.insert(0, str(ROOT / 'notebooks'))
from slr_data_readers import read_zemp2019_global
ZEMP = ROOT / 'data/raw/glaciers/zemp2019/Zemp_etal_results_global.csv'
GLAMBIE = (ROOT / 'data/raw/glaciers/glambie/GlaMBIE_Data_DOI_10.5904_wgms-glambie-2024-07/'
           'glambie_results_20240716/calendar_years/0_global.csv')
OUT_FIG = ROOT / 'figures/diagnostics/glacier_zemp_diagnostic.png'
GT_PER_MM = 362.5
Z90 = 1.645
FRED = {1900: -72.7, 1920: -55.2, 1950: -26.3}   # Frederikse glaciers, mm rel. 2000 (summation table)
PROD = {1900: -4.2, 1920: -9.1, 1950: -9.5}      # production glacier hindcast


def load():
    be = pd.read_hdf(ROOT / 'data/processed/slr_processed_data.h5', 'harmonized/df_berkeley_h')
    T = be['temperature'].groupby(be.index.year).mean()
    T = T - T.loc[1995:2005].mean()

    z = read_zemp2019_global(str(ZEMP)) * 1000.0                         # mm/yr SLE, 1 sigma
    z = pd.DataFrame({'rate': z['rate'], 'sig_ind': z['sigma_ind'], 'sig_cor': z['sigma_cor'],
                      'T': 0.25 * T.reindex(z.index - 1).values + 0.75 * T.reindex(z.index).values})

    g = pd.read_csv(GLAMBIE)
    g = pd.DataFrame({'rate': -g['combined_gt'].values / GT_PER_MM,
                      'sig_ind': g['combined_gt_errors'].values / GT_PER_MM,
                      'sig_cor': 0.0,
                      'T': T.reindex(g['start_dates'].astype(int)).values},
                     index=g['start_dates'].astype(int).values)
    return T, z, g


def gls(d):
    """GLS fit of rate = b T + c with the covariance described above."""
    X = np.column_stack([d['T'].values, np.ones(len(d))])
    C = np.diag(d['sig_ind'].values ** 2) + np.outer(d['sig_cor'].values, d['sig_cor'].values)
    Ci = np.linalg.inv(C)
    V = np.linalg.inv(X.T @ Ci @ X)
    beta = V @ X.T @ Ci @ d['rate'].values
    r = d['rate'].values - X @ beta
    chi2 = r @ Ci @ r / (len(d) - 2)
    V = V * max(chi2, 1.0)
    return beta, V, chi2


def show(lab, beta, V, chi2, n):
    se = np.sqrt(np.diag(V))
    print(f'  {lab:34s} b {beta[0]:5.2f} [{beta[0] - Z90 * se[0]:5.2f}, {beta[0] + Z90 * se[0]:5.2f}]'
          f'  c {beta[1]:5.2f} [{beta[1] - Z90 * se[1]:5.2f}, {beta[1] + Z90 * se[1]:5.2f}]'
          f'  chi2/dof {chi2:4.2f}  n {n}')


def hindcast(beta, T, years=(1900, 1920, 1950)):
    rate = beta[0] * T + beta[1]
    return {y: -rate.loc[y + 1:2000].sum() for y in years}


def main():
    T, z, g = load()
    print(f'Zemp et al. (2019): hydrological years {z.index.min()}-{z.index.max()}; '
          f'GlaMBIE: {g.index.min()}-{g.index.max()}')

    print('\nFits of dH/dt = b T + c (b in mm/yr/K, c in mm/yr at T = 0, i.e. 1995-2005; 90%):')
    fits = {}
    for lab, d in (('GlaMBIE 2000-2023 (as production)', g),
                   ('Zemp 1962-2016', z),
                   ('Zemp 1962-1989', z.loc[:1989]),
                   ('Zemp 1990-2016', z.loc[1990:]),
                   ('Zemp 1962-1999 + GlaMBIE 2000-2023', pd.concat([z.loc[:1999], g]))):
        fits[lab] = gls(d)
        show(lab, *fits[lab], len(d))

    # 1. Out of sample: GlaMBIE fit predicting Zemp 1962-1999
    b, V, _ = fits['GlaMBIE 2000-2023 (as production)']
    zo = z.loc[:1999]
    pred = b[0] * zo['T'] + b[1]
    X = np.column_stack([zo['T'].values, np.ones(len(zo))])
    mean_x = X.mean(axis=0)
    se_pred = np.sqrt(mean_x @ V @ mean_x)
    obs_se = np.sqrt((zo['sig_ind'] ** 2).sum() / len(zo) ** 2 + zo['sig_cor'].mean() ** 2)
    print('\n1. GlaMBIE fit predicting Zemp 1962-1999 (mean rate, mm/yr SLE):')
    print(f'   predicted {pred.mean():.2f} +/- {Z90 * se_pred:.2f} (fit), observed {zo["rate"].mean():.2f} '
          f'+/- {Z90 * obs_se:.2f}; difference {zo["rate"].mean() - pred.mean():+.2f} mm/yr, '
          f'{(zo["rate"].mean() - pred.mean()) / np.hypot(se_pred, obs_se):.1f} sigma')
    for a_, b_ in ((1962, 1975), (1976, 1987), (1988, 1999)):
        sel = zo.loc[a_:b_]
        print(f'   {a_}-{b_}: observed {sel["rate"].mean():5.2f}, predicted {(b[0] * sel["T"] + b[1]).mean():5.2f}'
              f'  (mean T {sel["T"].mean():+.2f} K)')
    cum_obs = zo['rate'].sum(); cum_pred = pred.sum()
    print(f'   cumulative 1962-1999: observed {cum_obs:.1f} mm, predicted {cum_pred:.1f} mm')

    # 2. Stationarity
    (b1, V1, _), (b2, V2, _) = fits['Zemp 1962-1989'], fits['Zemp 1990-2016']
    for k, nm in ((0, 'b'), (1, 'c')):
        dz = (b2[k] - b1[k]) / np.sqrt(V1[k, k] + V2[k, k])
        print(f'2. {nm}: 1990-2016 minus 1962-1989 = {b2[k] - b1[k]:+.2f} ({dz:+.1f} sigma)')
    # c at a common reference: rate at the early period's mean T from each fit
    Te = z.loc[:1989, 'T'].mean()
    print(f'   rate at T = {Te:+.2f} K (1962-1989 mean): early fit {b1[0] * Te + b1[1]:.2f}, '
          f'late fit {b2[0] * Te + b2[1]:.2f} mm/yr')

    # 3. Overlap offset
    yy = z.index.intersection(g.index)
    dif = z.loc[yy, 'rate'] - g.loc[yy, 'rate']
    print(f'\n3. Overlap {yy.min()}-{yy.max()}: Zemp minus GlaMBIE {dif.mean():+.2f} mm/yr '
          f'(Zemp {z.loc[yy, "rate"].mean():.2f}, GlaMBIE {g.loc[yy, "rate"].mean():.2f}); '
          f'GlaMBIE year labels are calendar years, Zemp hydrological')

    # 5. Hindcasts
    print('\n5. Cumulative glacier contribution relative to 2000 (mm):')
    print('   fit                                  1900    1920    1950')
    for lab in ('GlaMBIE 2000-2023 (as production)', 'Zemp 1962-2016', 'Zemp 1962-1989',
                'Zemp 1962-1999 + GlaMBIE 2000-2023'):
        h = hindcast(fits[lab][0], T)
        print(f'   {lab:34s} {h[1900]:6.1f}  {h[1920]:6.1f}  {h[1950]:6.1f}')
    print(f'   {"production (level space, prior)":34s} {PROD[1900]:6.1f}  {PROD[1920]:6.1f}  {PROD[1950]:6.1f}')
    print(f'   {"Frederikse (excl. RGI 5 and 19)":34s} {FRED[1900]:6.1f}  {FRED[1920]:6.1f}  {FRED[1950]:6.1f}')
    print('   Rate implied at +2 K (mm/yr): ' + ', '.join(
        f'{lab.split(" (")[0]} {fits[lab][0][0] * 2 + fits[lab][0][1]:.2f}'
        for lab in ('GlaMBIE 2000-2023 (as production)', 'Zemp 1962-2016', 'Zemp 1962-1999 + GlaMBIE 2000-2023')))

    # ── Figure ──
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))
    ax = axes[0]
    ax.errorbar(z.index, z['rate'], yerr=Z90 * np.hypot(z['sig_ind'], z['sig_cor']), fmt='o', ms=3,
                color='#0072B2', alpha=0.6, lw=0.6, label='Zemp et al. (2019)')
    ax.errorbar(g.index + 0.5, g['rate'], yerr=Z90 * g['sig_ind'], fmt='s', ms=3, color='#E69F00',
                alpha=0.7, lw=0.6, label='GlaMBIE')
    yrs = np.arange(1900, 2024)
    for lab, col in (('GlaMBIE 2000-2023 (as production)', 'k'), ('Zemp 1962-1989', '#56B4E9'),
                     ('Zemp 1990-2016', '#009E73')):
        bb = fits[lab][0]
        ax.plot(yrs, bb[0] * T.reindex(yrs).values + bb[1], color=col, lw=1.6, label=f'fit: {lab}')
    ax.axhline(0, color='0.5', lw=0.5)
    ax.set_xlim(1900, 2024); ax.set_ylabel('Glacier contribution (mm/yr SLE)')
    ax.set_title('(a) Annual rates and fitted b T + c', fontsize=10)
    ax.legend(fontsize=8); ax.grid(alpha=0.2)

    ax = axes[1]
    ax.scatter(z['T'], z['rate'], s=12, color='#0072B2', alpha=0.7, label='Zemp et al. (2019)')
    ax.scatter(g['T'], g['rate'], s=12, color='#E69F00', alpha=0.8, label='GlaMBIE')
    tt = np.linspace(-0.6, 0.8, 50)
    for lab, col in (('GlaMBIE 2000-2023 (as production)', 'k'), ('Zemp 1962-1989', '#56B4E9'),
                     ('Zemp 1990-2016', '#009E73')):
        bb = fits[lab][0]
        ax.plot(tt, bb[0] * tt + bb[1], color=col, lw=1.6)
    ax.set_xlabel('GMST anomaly rel. 1995-2005 (K)'); ax.set_ylabel('mm/yr SLE')
    ax.set_title('(b) Rate against temperature', fontsize=10)
    ax.legend(fontsize=8); ax.grid(alpha=0.2)
    OUT_FIG.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout(); plt.savefig(OUT_FIG, dpi=150, bbox_inches='tight')
    print(f'\nSaved {OUT_FIG.relative_to(ROOT)}')


if __name__ == '__main__':
    main()
