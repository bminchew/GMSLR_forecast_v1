"""Antarctic Peninsula: linear vs quadratic rate model, full likelihood (AN-6).

The production Peninsula model is linear in GMST, rate = b T + c, with the
quadratic term a_p set to zero. The notebook's own BIC comparison cannot
justify that choice: it leaves out ln|Sigma| and bounds a >= 0, so the
quadratic collapses onto the linear model and the BIC difference is -ln n by
construction. This script refits both models by maximum likelihood with the
full Gaussian likelihood of the cumulative record (annual increments with the
RMS-of-monthly sigma, as component_apeninsula, plus a fitted
sigma_p,extra), a unrestricted in sign, and prints the numbers the supplement
quotes for keeping a_p = 0:

- Delta BIC (linear minus quadratic; positive favors the quadratic) and the
  fitted a with its 90% interval;
- the same for fits ending in 2019 and 2015;
- the mean Peninsula loss rate over 2005-2014 and 2015-2023, and the share
  of the change from the IMBIE-3 SMB and dynamics partitions;
- the GMST (rel. 1995-2005) above which the quadratic gives mass gain, and
  the gain at +3 C.

Diagnostic only; the production model is unchanged. Run from the repository
root:  python scripts/peninsula_curvature_check.py
"""

import sys
from pathlib import Path

import numpy as np
from scipy.optimize import minimize_scalar

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'notebooks'))
sys.path.insert(0, str(ROOT / 'src'))

from slr_data_readers import read_imbie3                      # noqa: E402
from component_analysis import annualize_imbie, imbie_annual_rates   # noqa: E402
from ar6_gmst import berkeley_annual                          # noqa: E402

PEN = ROOT / 'data/raw/ice_sheets/imbie2026/imbie3_antarctic_peninsula_mm_partitioned.csv'
H5_OBS = ROOT / 'data/processed/slr_processed_data.h5'
GT_PER_MM = 362.5
Z_90 = 1.645


def load():
    df = read_imbie3(str(PEN), convert_to_meters=False)          # mm, SLR-positive
    yrs, rate, sig = imbie_annual_rates(df)
    lab, H, _ = annualize_imbie(df, baseline_year=2000.0)
    assert np.array_equal(yrs + 1.0, lab)
    by, bT = berkeley_annual(str(H5_OBS))
    T = np.interp(yrs + 0.5, by, bT)                              # calendar-year GMST
    return df, yrs, H, sig, rate, T


def ml_fit(H, T, sig, quadratic):
    """Maximum-likelihood fit of the cumulative record; returns -2 ln L,
    k (incl. sigma_extra), coefficients and their covariance."""
    n = len(H)
    L = np.tril(np.ones((n, n)))
    cols = ([np.cumsum(T ** 2)] if quadratic else []) + [np.cumsum(T), np.cumsum(np.ones(n))]
    X = np.column_stack(cols + [np.ones(n)])                    # last column: H0

    def gls(log_sx):
        v = sig ** 2 + np.exp(2 * log_sx)
        S = (L * v[None, :]) @ L.T
        Si = np.linalg.inv(S)
        A = X.T @ Si @ X
        beta = np.linalg.solve(A, X.T @ Si @ H)
        r = H - X @ beta
        m2 = r @ Si @ r + np.linalg.slogdet(S)[1] + n * np.log(2 * np.pi)
        return m2, beta, np.linalg.inv(A)

    res = minimize_scalar(lambda s: gls(s)[0], bounds=(np.log(1e-4), np.log(1.0)),
                          method='bounded', options={'xatol': 1e-10})
    m2, beta, cov = gls(res.x)
    return dict(m2lnL=m2, k=X.shape[1] + 1, beta=beta, cov=cov,
                sigma_extra=float(np.exp(res.x)), bic=m2 + (X.shape[1] + 1) * np.log(n))


def compare(H, T, sig):
    lin, quad = ml_fit(H, T, sig, False), ml_fit(H, T, sig, True)
    a, sa = quad['beta'][0], np.sqrt(quad['cov'][0, 0])
    return lin, quad, lin['bic'] - quad['bic'], (a, a - Z_90 * sa, a + Z_90 * sa)


def main():
    df, yrs, H, sig, rate, T = load()
    lin, quad, dbic, a = compare(H, T, sig)
    print(f'Peninsula, IMBIE-3 {yrs[0]}-{yrs[-1]} (n = {len(H)}), full likelihood, a unrestricted')
    print(f'  linear:    -2lnL = {lin["m2lnL"]:.2f}, k = {lin["k"]}, sigma_extra = {lin["sigma_extra"]:.4f} mm/yr')
    print(f'  quadratic: -2lnL = {quad["m2lnL"]:.2f}, k = {quad["k"]}, sigma_extra = {quad["sigma_extra"]:.4f} mm/yr')
    print(f'  Delta BIC (linear - quadratic) = {dbic:+.2f}')
    print(f'  a_p = {a[0]:.3f} [{a[1]:.3f}, {a[2]:.3f}] mm/yr/C^2 (90%)')
    for last in (2019, 2015):
        m = yrs <= last
        _, _, d, aa = compare(H[m], T[m], sig[m])
        print(f'  fit ending {last}: Delta BIC = {d:+.2f}, a_p = {aa[0]:.3f} [{aa[1]:.3f}, {aa[2]:.3f}]')

    # Observed rates and their IMBIE-3 partition (annual means, SLR-positive mm/yr)
    g = df.groupby(df.index.year)[['mass_balance_rate', 'smb_rate', 'dynamics_rate']].mean()
    p1, p2 = g.loc[2005:2014].mean(), g.loc[2015:2023].mean()
    d = p2 - p1
    print(f'\nMean loss rate: 2005-2014 {p1["mass_balance_rate"]:.3f}, 2015-2023 '
          f'{p2["mass_balance_rate"]:.3f} mm/yr (change {d["mass_balance_rate"]:+.3f})')
    print(f'  share of the change: SMB {100 * d["smb_rate"] / d["mass_balance_rate"]:.0f}%, '
          f'dynamics {100 * d["dynamics_rate"] / d["mass_balance_rate"]:.0f}%')

    # Where the quadratic turns to mass gain, and the gain at +3 C
    aq, bq, cq = quad['beta'][:3]
    roots = np.sort(np.roots([aq, bq, cq]).real)
    print(f'\nQuadratic rate: zero at T = {roots[1]:.2f} C (rel. 1995-2005); '
          f'GMST of {yrs[-1]}: {T[-1]:.2f} C')
    r3 = aq * 9 + bq * 3 + cq
    print(f'  at T = +3 C: {r3:.2f} mm/yr = {-r3 * GT_PER_MM:.0f} Gt/yr of mass gain')


if __name__ == '__main__':
    main()
