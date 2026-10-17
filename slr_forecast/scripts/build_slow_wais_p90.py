"""Build the p(S1) = 90% slow-WAIS ensembles used by the manuscript figures.

The slow-WAIS case presented in the paper is a mixture: continued mass loss
along the observed trajectory is very likely (p(S1) = 90%) but the accelerating
branch keeps 10% of the weight. Both branches already exist as trajectory
ensembles, so the mixture needs no re-blending and no new projection run:

    blended_stable/{ssp}/samples      total GMSL | S1   (2000 trajectories)
    blended/{ssp}/samples[idx == 1]   total GMSL | S2   (1805 trajectories)
    wais_2k/s1_samples                WAIS | S1         (2000)
    wais_2k/full_samples[idx == 1]    WAIS | S2         (1805)

Figure code takes equally weighted samples, so the mixture is realized as
20,000 trajectories: 18,000 S1 rows (exactly 90% of the ensemble) and 2,000 S2
rows. With 2000 members, the S1 rows are nine copies of each member, and the S2
rows are the 1,805 members plus 195 systematically spaced duplicates, chosen
after sorting on the 2100 value so the duplication is spread evenly through the
branch rather than concentrated in one part of it. A branch whose row count is
not a whole multiple of its size (e.g. at N_SAMPLES = 10000) is realized the
same way: every member once plus evenly spaced duplicates, or, when the branch
has more members than rows, an evenly spaced subset by rank (branch_row_index). Quantiles of the result are checked against exact weighted
quantiles of the two branches before anything is written.

Writes groups `blended_slow90` and `wais_slow90` to component_results.h5.
`blended_stable` is left untouched: the VoI and expected-loss analyses need the
conditional S1 world, not this mixture.

Run from the repository root:  python scripts/build_slow_wais_p90.py
"""

import sys
from pathlib import Path

import h5py
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
H5 = ROOT / 'data' / 'processed' / 'component_results.h5'
SSPS = ['SSP1-2.6', 'SSP2-4.5', 'SSP3-7.0', 'SSP5-8.5', 'SSP1-1.9']
P_S1 = 0.9
N_OUT = 20_000
M_TO_MM = 1000.0
TOL_MM = 2.0


def weighted_quantile(v, w, q):
    o = np.argsort(v)
    v, w = np.asarray(v)[o], np.asarray(w)[o]
    cdf = (np.cumsum(w) - 0.5 * w) / w.sum()
    return np.interp(q, cdf, v)


def branch_row_index(vals, n_rows):
    """Row picks for one branch of the mixture, given its 2100 values.

    If n_rows is a whole multiple of the branch size, every member is repeated
    the same number of times. Otherwise members are ordered by their 2100 value
    and the extra rows (or, if there are fewer rows than members, all rows) are
    taken at evenly spaced ranks, so the duplication or thinning does not
    distort the shape of the branch the way a random draw or a head-of-array
    repeat would.
    """
    n = len(vals)
    reps, rem = divmod(n_rows, n)
    if rem == 0:
        return np.tile(np.arange(n), reps)
    order = np.argsort(vals)
    extra = n_rows - n
    if extra <= 0:
        return order[np.linspace(0, n - 1, n_rows).round().astype(int)]
    picks = order[np.linspace(0, n - 1, extra).round().astype(int)]
    return np.concatenate([order, picks])


def mixture_rows(s1_vals, s2_vals):
    """Row indices into the S1 and S2 branches (2100 values given)."""
    n_s1_rows = int(round(P_S1 * N_OUT))
    return (branch_row_index(s1_vals, n_s1_rows),
            branch_row_index(s2_vals, N_OUT - n_s1_rows))


def build(s1, s2, i2100):
    """Mixture rows from two trajectory ensembles, S1 first then S2."""
    rows_s1, rows_s2 = mixture_rows(s1[:, i2100], s2[:, i2100])
    return np.vstack([s1[rows_s1], s2[rows_s2]])


def check(mix, s1, s2, i2100, label):
    """Compare the realized ensemble against exact weighted quantiles.

    The allowed deviation at each quantile is 2 mm or the gap between the two
    distinct member values that bracket the exact quantile, whichever is
    larger: in the sparse tail the realized ensemble cannot land closer than
    that spacing. Returns the largest deviation in excess of its tolerance.
    """
    v = np.concatenate([s1[:, i2100], s2[:, i2100]])
    w = np.concatenate([np.full(s1.shape[0], P_S1 / s1.shape[0]),
                        np.full(s2.shape[0], (1 - P_S1) / s2.shape[0])])
    qs = [0.05, 0.17, 0.5, 0.83, 0.95]
    exact = weighted_quantile(v, w, qs) * M_TO_MM
    got = np.percentile(mix[:, i2100], [q * 100 for q in qs]) * M_TO_MM
    dev = np.abs(got - exact)
    u = np.unique(v) * M_TO_MM
    k = np.clip(np.searchsorted(u, exact), 1, len(u) - 1)
    tol = np.maximum(TOL_MM, u[k] - u[k - 1])
    print(f'  {label}: max deviation from exact weighted quantiles '
          f'{dev.max():.2f} mm, tolerance {tol[np.argmax(dev)]:.2f} mm  '
          f'(95th: {got[-1]:.1f} vs {exact[-1]:.1f})')
    return (dev - tol).max()


def main():
    worst = -np.inf
    with h5py.File(H5, 'a') as hf:
        yrs = hf['blended/forecast_years'][:]
        i2100 = int(np.argmin(np.abs(yrs - 2100)))
        idx = hf['blended/wais_scenario_idx'][:].astype(int)

        print('Total GMSL:')
        totals = {}
        for ssp in SSPS:
            s1 = hf[f'blended_stable/{ssp}/samples'][:]
            s2 = hf[f'blended/{ssp}/samples'][:][idx == 1]
            mix = build(s1, s2, i2100)
            worst = max(worst, check(mix, s1, s2, i2100, ssp))
            totals[ssp] = mix

        print('WAIS component:')
        wyrs = hf['wais_2k/years'][:]
        iw = int(np.argmin(np.abs(wyrs - 2100)))
        w_s1 = hf['wais_2k/s1_samples'][:]
        w_s2 = hf['wais_2k/full_samples'][:][idx == 1]
        wais_mix = build(w_s1, w_s2, iw)
        worst = max(worst, check(wais_mix, w_s1, w_s2, iw, 'WAIS'))

        if worst > 0:
            raise SystemExit(f'Realized ensemble exceeds its tolerance by '
                             f'{worst:.2f} mm; not written.')

        for name in ('blended_slow90', 'wais_slow90'):
            if name in hf:
                del hf[name]

        g = hf.create_group('blended_slow90')
        g.attrs['description'] = (
            'Slow-WAIS total GMSL forecast at p(S1) = 0.9: mixture of the S1 '
            'world (blended_stable) and the S2 world (the S2 members of '
            'blended), realized as equally weighted trajectories for figure '
            'code. Presentational case only. The VoI and expected-loss '
            'analyses use blended_stable, the conditional S1 world.'
        )
        g.attrs['p_s1'] = P_S1
        g.attrs['n_samples'] = N_OUT
        g.attrs['source_groups'] = ['blended_stable', 'blended']
        g.create_dataset('forecast_years', data=yrs)
        for ssp in SSPS:
            sg = g.create_group(ssp)
            m = totals[ssp]
            sg.create_dataset('samples', data=m, compression='gzip')
            sg.create_dataset('median', data=np.median(m, axis=0))
            for p in (5, 17, 83, 95):
                sg.create_dataset(f'p{p}', data=np.percentile(m, p, axis=0))

        gw = hf.create_group('wais_slow90')
        gw.attrs['description'] = (
            'WAIS contribution at p(S1) = 0.9, mixture of wais_2k/s1_samples '
            'and the S2 members of wais_2k/full_samples.'
        )
        gw.attrs['p_s1'] = P_S1
        gw.create_dataset('years', data=wyrs)
        gw.create_dataset('samples', data=wais_mix, compression='gzip')
        gw.create_dataset('median', data=np.median(wais_mix, axis=0))
        for p in (5, 95):
            gw.create_dataset(f'p{p}', data=np.percentile(wais_mix, p, axis=0))

        sys.path.insert(0, str(ROOT / 'notebooks'))
        from component_io import stamp_derived
        stamp_derived(hf, 'blended_slow90')
        stamp_derived(hf, 'wais_slow90')

        print(f'\nWrote blended_slow90 and wais_slow90 to {H5}')

        # Numbers needed for the Fig. 4b caption: per-component variance shares
        print('\nVariance shares at 2100 (per-component, as the pies compute them):')
        comps = ['ocean', 'glacier', 'greenland']
        for ssp in SSPS[:3]:
            v = {c: float(np.var(hf[f'{c}/projections/{ssp}/samples'][:, 150]))
                 for c in comps}
            v_fast = float(np.var(hf['wais_2k/full_samples'][:, iw]))
            v_slow90 = float(np.var(wais_mix[:, iw]))
            v_s1 = float(np.var(w_s1[:, iw]))
            for label, vw in (('fast', v_fast), ('slow p(S1)=0.9', v_slow90),
                              ('slow p(S1)=1.0', v_s1)):
                tot = sum(v.values()) + vw
                print(f'  {ssp} {label:15s}: WAIS {100 * vw / tot:5.1f}%, '
                      f'non-WAIS {100 * sum(v.values()) / tot:5.1f}%')
            print()


if __name__ == '__main__':
    main()
