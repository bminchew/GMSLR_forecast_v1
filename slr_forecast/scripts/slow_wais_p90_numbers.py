"""Manuscript numbers for the slow-WAIS case at p(S1) = 90% against p(S1) = 100%.

The slow-WAIS presentation case is a mixture: with probability p the total is
drawn from the S1 (observed-trajectory) world and otherwise from the S2
(accelerating) world. `blended_stable` supplies T | S1 for all 2000 samples and
the S2 subset of `blended` supplies T | S2 for 1805, so every quantity below is
a weighted statistic of those two branches. No resampling and no re-blending:
both branches are already rate-space blended forecasts.

Covers the quantities that appear in the main text at lines 387 (exceedance
probabilities), 391 (variance and interval shares), 403 (interval widths), and
418 (the 5% exceedance design threshold and the impacts that follow from it).

Run from the repository root:  python scripts/slow_wais_p90_numbers.py
"""

import json
import sys
from pathlib import Path

import h5py
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'notebooks'))
sys.path.insert(0, str(ROOT / 'src'))

M_TO_MM = 1000.0
PREIND_M = 0.19          # observed rise, pre-industrial to the 2000 baseline
ADAPT_COST_PER_M = None  # filled from compute_evpi
SSPS = ['SSP1-2.6', 'SSP2-4.5', 'SSP3-7.0']
LABEL = {'SSP1-2.6': '2 C', 'SSP2-4.5': '3 C', 'SSP3-7.0': '4 C'}
P_CASES = [0.9, 1.0]


def weighted_quantile(v, w, q):
    o = np.argsort(v)
    v, w = np.asarray(v)[o], np.asarray(w)[o]
    cdf = (np.cumsum(w) - 0.5 * w) / w.sum()
    return np.interp(q, cdf, v)


def main():
    from compute_evpi import ADAPT_COST_PER_M as ADAPT, damage_vec
    from slr_data_readers import people_displaced_kulpstrauss2019 as kulp

    with h5py.File(ROOT / 'data' / 'processed' / 'component_results.h5', 'r') as hf:
        yrs = hf['blended/forecast_years'][:]
        i = int(np.argmin(np.abs(yrs - 2100)))
        idx = hf['blended/wais_scenario_idx'][:].astype(int)
        T1 = {s: hf[f'blended_stable/{s}/samples'][:, i] for s in SSPS}
        Tf = {s: hf[f'blended/{s}/samples'][:, i] for s in SSPS}

    T2 = {s: Tf[s][idx == 1] for s in SSPS}
    print(f'Branch sizes: S1 {len(T1[SSPS[0]])}, S2 {len(T2[SSPS[0]])}\n')

    def branches(s, p):
        v = np.concatenate([T1[s], T2[s]])
        w = np.concatenate([np.full(T1[s].size, p / T1[s].size),
                            np.full(T2[s].size, (1 - p) / T2[s].size)])
        return v, w

    out = {}

    # ---- line 387: exceedance probabilities relative to pre-industrial -----
    print('=== Exceedance probabilities at 2100, relative to pre-industrial ===')
    print(f'{"case":>7}{"scenario":<10}{"P>1 m":>8}{"P>1.5 m":>9}{"P>2 m":>8}')
    for p in P_CASES:
        for s in SSPS:
            v, w = branches(s, p)
            pi = v + PREIND_M
            pr = [100 * w[pi > t].sum() / w.sum() for t in (1.0, 1.5, 2.0)]
            out.setdefault(f'p{p:g}', {}).setdefault(s, {}).update(
                P_exceed_1m=pr[0], P_exceed_1p5m=pr[1], P_exceed_2m=pr[2])
            print(f'{p:7.1f}{LABEL[s]:<10}{pr[0]:8.1f}{pr[1]:9.1f}{pr[2]:8.1f}')
        print()

    # ---- lines 391 and 403: intervals, widths, variance and interval shares -
    print('=== Interval and variance structure at 2100 (rel. 2000) ===')
    print(f'{"case":>7}{"scenario":<10}{"median cm":>11}{"90% CI cm":>20}'
          f'{"width cm":>10}{"S2 var share":>14}{"S1 width share":>16}')
    for p in P_CASES:
        for s in SSPS:
            v, w = branches(s, p)
            q = weighted_quantile(v, w, [0.05, 0.5, 0.95]) * 100  # cm
            width = q[2] - q[0]
            v1, v2 = T1[s].var(), T2[s].var()
            m1, m2 = T1[s].mean(), T2[s].mean()
            var = p * v1 + (1 - p) * v2 + p * (1 - p) * (m1 - m2) ** 2
            s2_share = 100 * ((1 - p) * v2 + p * (1 - p) * (m1 - m2) ** 2) / var
            w1 = (np.percentile(T1[s], 95) - np.percentile(T1[s], 5)) * 100
            full = np.percentile(Tf[s], [5, 95]) * 100
            out[f'p{p:g}'][s].update(
                median_cm=q[1], p5_cm=q[0], p95_cm=q[2], width90_cm=width,
                s2_variance_share_pct=s2_share,
                s1_width_share_pct=100 * w1 / width,
                narrower_than_full_factor=(full[1] - full[0]) / width)
            print(f'{p:7.1f}{LABEL[s]:<10}{q[1]:11.1f}'
                  f'{f"[{q[0]:.1f}, {q[2]:.1f}]":>20}{width:10.1f}'
                  f'{s2_share:14.1f}{100 * w1 / width:16.1f}')
        print()

    print('Slow-WAIS interval narrower than the full forecast by a factor of:')
    for p in P_CASES:
        f = [out[f'p{p:g}'][s]['narrower_than_full_factor'] for s in SSPS]
        print(f'  p(S1)={p:.1f}: ' + ', '.join(f'{x:.1f}x' for x in f))

    print('\n2 C to 4 C shift in the median:')
    for p in P_CASES:
        d = out[f'p{p:g}']['SSP3-7.0']['median_cm'] - out[f'p{p:g}']['SSP1-2.6']['median_cm']
        pop = kulp(out[f'p{p:g}']['SSP3-7.0']['median_cm'] / 100) - \
            kulp(out[f'p{p:g}']['SSP1-2.6']['median_cm'] / 100)
        print(f'  p(S1)={p:.1f}: {d:.1f} cm, {pop:.0f} M more people, '
              f'US${ADAPT * d / 100:.0f}B/yr adaptation')

    # ---- line 418: the 5% exceedance design threshold and its impacts ------
    print('\n=== 5% exceedance threshold at 3 C (Fig. 2b stars) ===')
    fast95 = np.percentile(Tf['SSP2-4.5'], 95)
    for p in P_CASES:
        v, w = branches('SSP2-4.5', p)
        slow95 = float(weighted_quantile(v, w, [0.95])[0])
        gap = fast95 - slow95
        dpop = float(kulp(fast95) - kulp(slow95))
        dadapt = ADAPT * gap
        ddam = float(damage_vec(np.array([fast95]))[0]
                     - damage_vec(np.array([slow95]))[0])
        out[f'p{p:g}']['threshold_5pct'] = dict(
            slow_95th_m_rel2000=slow95, fast_95th_m_rel2000=float(fast95),
            gap_m=float(gap), delta_people_M=dpop,
            delta_adapt_B_per_yr=float(dadapt), delta_damage_B_per_yr=ddam)
        print(f'  p(S1)={p:.1f}: slow {slow95:.3f} m, fast {fast95:.3f} m, '
              f'gap {gap:.2f} m -> {dpop:.0f} M people, US${dadapt:.0f}B/yr '
              f'adaptation, US${ddam / 1000:.1f}T/yr damages')

    # Multiple of the observed rise over the past 125 years, as quoted at 418
    past_125yr_m = float(np.percentile(Tf['SSP2-4.5'], 95)) / 8.0
    print(f'\nObserved rise over the past 125 yr implied by the published '
          f'"eight times" ratio: {past_125yr_m:.3f} m')
    for p in P_CASES:
        slow95 = out[f'p{p:g}']['threshold_5pct']['slow_95th_m_rel2000']
        print(f'  p(S1)={p:.1f}: slow-WAIS multiple {slow95 / past_125yr_m:.1f}x')

    dest = ROOT / 'data' / 'processed' / 'slow_wais_p90_numbers.json'
    with open(dest, 'w') as f:
        json.dump(dict(adapt_cost_per_m_B=ADAPT, preindustrial_offset_m=PREIND_M,
                       cases=out), f, indent=1, default=float)
    print(f'\nWrote {dest}')


if __name__ == '__main__':
    main()
