"""Compare our 2100 GMSL forecast against the IPCC AR6 medium- and
low-confidence total-GMSL projections at selected p(S1) mixture weights.

The stored ``blended`` group in component_results.h5 is the forecast at
p(S1) = 0.1 (the A4 scenario weights are P(S1) = 0.1, P(S2) = 0.9), and
``blended_stable`` is the same forecast with WAIS replaced by S1 for every
sample, i.e. p(S1) = 1.  Because the WAIS scenario label is an independent
categorical draw and the rate-space blend is linear in the component sum,
the total at any mixture weight p is the distribution mixture

    T(p) = p * (T | S1) + (1 - p) * (T | S2),

which we evaluate with weighted quantiles: the ``blended_stable`` samples
supply T | S1 (2000 samples) and the S2 subset of ``blended`` supplies
T | S2 (1805 samples).  No resampling, no rerun of the projection pipeline.

AR6 projections are referenced to the 1995-2014 mean; ours are referenced to
2000.  We use the same Frederikse-based offset as results_figures.ipynb.

Run from the repository root:  python scripts/ar6_confidence_comparison.py
"""

import json
import os
import sys
from pathlib import Path

import h5py
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'notebooks'))
sys.path.insert(0, str(ROOT / 'src'))

from component_projections import read_ipcc_component_nc, ipcc_extract  # noqa: E402

M_TO_MM = 1000.0
H5_COMP = ROOT / 'data' / 'processed' / 'component_results.h5'
H5_OBS = ROOT / 'data' / 'processed' / 'slr_processed_data.h5'
CONF_BASE = str(ROOT / 'data' / 'raw' / 'ipcc_ar6' / 'slr' / 'ar6' / 'global'
                / 'confidence_output_files')
SSP_TO_CODE = {
    'SSP1-2.6': 'ssp126', 'SSP2-4.5': 'ssp245',
    'SSP3-7.0': 'ssp370', 'SSP5-8.5': 'ssp585',
}
POLICY_SSPS = ['SSP1-2.6', 'SSP2-4.5', 'SSP3-7.0']
# Published (line 300) mixture weights: 'slow WAIS' p(S1)=0.9, 'fast WAIS' p(S1)=0.1.
P_S1_CASES = [0.1, 0.5, 0.9, 1.0]


def weighted_quantile(values, weights, quantiles):
    """Quantiles of a weighted empirical distribution (no resampling)."""
    values = np.asarray(values, dtype=float)
    weights = np.asarray(weights, dtype=float)
    order = np.argsort(values)
    v, w = values[order], weights[order]
    cdf = (np.cumsum(w) - 0.5 * w) / np.sum(w)
    return np.interp(np.asarray(quantiles, dtype=float), cdf, v)


def main():
    # ---- our forecast samples at 2100 -------------------------------------
    with h5py.File(H5_COMP, 'r') as hf:
        years = hf['blended/forecast_years'][:]
        i2100 = int(np.argmin(np.abs(years - 2100)))
        assert abs(years[i2100] - 2100) < 1e-6, years[i2100]
        idx = hf['blended/wais_scenario_idx'][:].astype(int)
        scen_P = {k: float(hf[f'wais/a4_scenarios/{k}'].attrs['P'])
                  for k in hf['wais/a4_scenarios']}
        full = {ssp: hf[f'blended/{ssp}/samples'][:, i2100] * M_TO_MM
                for ssp in SSP_TO_CODE}
        s1 = {ssp: hf[f'blended_stable/{ssp}/samples'][:, i2100] * M_TO_MM
              for ssp in SSP_TO_CODE}

    n_s1_draw = int(np.sum(idx == 0))
    print(f'WAIS scenario weights as stored: {scen_P}')
    print(f'blended draws: S1 {n_s1_draw}, S2 {np.sum(idx == 1)} '
          f'(=> stored mixture p(S1) = {n_s1_draw / idx.size:.3f})')

    # ---- AR6 baseline offset (same construction as results_figures.ipynb) --
    with pd.HDFStore(H5_OBS, mode='r') as store:
        df_fred = store['raw/df_frederikse']
    t = df_fred['year'].values
    g = df_fred['gmsl'].values
    offset_mm = (np.mean(g[(t >= 1995) & (t <= 2014)])
                 - np.interp(2000, t, g)) * M_TO_MM
    print(f'AR6 baseline offset (1995-2014 mean minus 2000): {offset_mm:.1f} mm')

    # ---- AR6 medium- and low-confidence totals at 2100 --------------------
    ar6 = {}
    for conf in ('medium_confidence', 'low_confidence'):
        ar6[conf] = {}
        for ssp, code in SSP_TO_CODE.items():
            data = read_ipcc_component_nc(CONF_BASE, conf, code, 'total')
            if data is None:
                continue
            ex = ipcc_extract(data, quantiles_target=(0.05, 0.17, 0.5, 0.83, 0.95))
            j = int(np.argmin(np.abs(ex['years'] - 2100)))
            ar6[conf][ssp] = {q: float(ex[q][j])
                              for q in ('q05', 'q17', 'q50', 'q83', 'q95')}
    print('\nAR6 totals at 2100 (mm, AR6 1995-2014 baseline):')
    for conf in ar6:
        for ssp, v in ar6[conf].items():
            print(f'  {conf:18s} {ssp}: median {v["q50"]:6.0f}  '
                  f'[{v["q05"]:6.0f}, {v["q95"]:6.0f}]')

    # ---- reproduce the published p(S1)=0.1 numbers -------------------------
    print('\n=== Reproduction check: stored blended forecast (p(S1)=0.1) ===')
    print(f'{"SSP":<10}{"med(2000)":>11}{"p5":>8}{"p95":>8}'
          f'{"med(AR6 base)":>15}{"AR6 mc med":>12}{"% above":>9}')
    repro = {}
    for ssp in SSP_TO_CODE:
        s = full[ssp]
        med, p5, p95 = np.percentile(s, [50, 5, 95])
        med_ar6 = med - offset_mm
        mc = ar6['medium_confidence'][ssp]['q50']
        repro[ssp] = dict(median_mm=med, p5_mm=p5, p95_mm=p95,
                          median_mm_ar6_baseline=med_ar6,
                          ar6_medium_median_mm=mc,
                          pct_above_ar6_medium=(med_ar6 / mc - 1) * 100)
        print(f'{ssp:<10}{med:11.0f}{p5:8.0f}{p95:8.0f}'
              f'{med_ar6:15.1f}{mc:12.1f}{(med_ar6 / mc - 1) * 100:9.1f}')

    # ---- mixtures at the requested p(S1) ---------------------------------
    print('\n=== Mixture forecasts by p(S1) (mm, rel. 2000 unless noted) ===')
    print(f'{"p(S1)":>6}{"SSP":<11}{"median":>9}{"p5":>8}{"p95":>8}'
          f'{"90% width":>11}{"med(AR6)":>10}')
    mix = {}
    for p in P_S1_CASES:
        mix[p] = {}
        for ssp in SSP_TO_CODE:
            s2 = full[ssp][idx == 1]
            vals = np.concatenate([s1[ssp], s2])
            w = np.concatenate([np.full(s1[ssp].size, p / s1[ssp].size),
                                np.full(s2.size, (1 - p) / max(s2.size, 1))])
            if p == 1.0:
                q = np.percentile(s1[ssp], [5, 17, 50, 83, 95])
            else:
                q = weighted_quantile(vals, w, [0.05, 0.17, 0.50, 0.83, 0.95])
            rec = dict(p5_mm=q[0], p17_mm=q[1], median_mm=q[2],
                       p83_mm=q[3], p95_mm=q[4],
                       width90_mm=q[4] - q[0],
                       median_mm_ar6_baseline=q[2] - offset_mm,
                       p95_mm_ar6_baseline=q[4] - offset_mm,
                       p5_mm_ar6_baseline=q[0] - offset_mm)
            mix[p][ssp] = rec
            print(f'{p:6.1f}{ssp:<11}{q[2]:9.0f}{q[0]:8.0f}{q[4]:8.0f}'
                  f'{q[4] - q[0]:11.0f}{q[2] - offset_mm:10.0f}')
        print()

    # ---- cross-check: does p(S1)=0.1 mixture match the stored blended? ----
    print('Consistency of the reweighting at p(S1)=0.1 (mixture minus stored, mm):')
    for ssp in SSP_TO_CODE:
        d_med = mix[0.1][ssp]['median_mm'] - repro[ssp]['median_mm']
        d_p95 = mix[0.1][ssp]['p95_mm'] - repro[ssp]['p95_mm']
        print(f'  {ssp}: median {d_med:+.1f}, p95 {d_p95:+.1f}')

    # ---- the two head-to-head comparisons --------------------------------
    print('\n=== Head-to-head: AR6 medium confidence vs our p(S1)=0.9 and 1.0 ===')
    for p in (0.9, 1.0):
        for ssp in POLICY_SSPS:
            m = mix[p][ssp]
            a = ar6['medium_confidence'][ssp]
            print(f'p(S1)={p:.1f} {ssp}: ours {m["median_mm_ar6_baseline"]:.0f} '
                  f'[{m["p5_mm_ar6_baseline"]:.0f}, {m["p95_mm_ar6_baseline"]:.0f}] '
                  f'vs AR6 mc {a["q50"]:.0f} [{a["q05"]:.0f}, {a["q95"]:.0f}] mm; '
                  f'ratio med {m["median_mm_ar6_baseline"] / a["q50"]:.2f}, '
                  f'ratio p95 {m["p95_mm_ar6_baseline"] / a["q95"]:.2f}')

    print('\n=== Head-to-head: AR6 low confidence vs our p(S1)=0.1 ===')
    for ssp in SSP_TO_CODE:
        if ssp not in ar6['low_confidence']:
            print(f'  {ssp}: no AR6 low-confidence workflow')
            continue
        m = mix[0.1][ssp]
        a = ar6['low_confidence'][ssp]
        print(f'  {ssp}: ours {m["median_mm_ar6_baseline"]:.0f} '
              f'[{m["p5_mm_ar6_baseline"]:.0f}, {m["p95_mm_ar6_baseline"]:.0f}] '
              f'vs AR6 lc {a["q50"]:.0f} [{a["q05"]:.0f}, {a["q95"]:.0f}] mm; '
              f'ratio med {m["median_mm_ar6_baseline"] / a["q50"]:.2f}, '
              f'ratio p95 {m["p95_mm_ar6_baseline"] / a["q95"]:.2f}')

    # ---- exceedance probabilities of the AR6 upper bounds ------------------
    print('\n=== P(our forecast exceeds AR6 95th percentile) at 2100 ===')
    for p in P_S1_CASES:
        for ssp in POLICY_SSPS + ['SSP5-8.5']:
            s2 = full[ssp][idx == 1]
            vals = np.concatenate([s1[ssp], s2]) - offset_mm
            w = np.concatenate([np.full(s1[ssp].size, p / s1[ssp].size),
                                np.full(s2.size, (1 - p) / max(s2.size, 1))])
            out = []
            for conf in ('medium_confidence', 'low_confidence'):
                if ssp not in ar6[conf]:
                    out.append(f'{conf.split("_")[0]}: n/a')
                    continue
                thr = ar6[conf][ssp]['q95']
                pr = float(np.sum(w[vals > thr]) / np.sum(w)) * 100
                out.append(f'{conf.split("_")[0]}: {pr:.1f}%')
            print(f'  p(S1)={p:.1f} {ssp}: ' + ', '.join(out))

    out = dict(
        offset_ar6_baseline_mm=offset_mm,
        stored_scenario_weights=scen_P,
        n_s1_draws=n_s1_draw, n_s2_draws=int(np.sum(idx == 1)),
        reproduction_p_s1_0p1=repro,
        mixtures={f'{p:g}': mix[p] for p in P_S1_CASES},
        ar6=ar6,
    )
    dest = ROOT / 'data' / 'processed' / 'ar6_confidence_comparison.json'
    with open(dest, 'w') as f:
        json.dump(out, f, indent=1, default=float)
    print(f'\nWrote {dest}')


if __name__ == '__main__':
    main()
