"""Greenland sensitivity tests carried into the forecast (main text, Discussion).

component_greenland §6 (cell 26) reruns the production discharge model with
one input changed (AA at the ends of the observed window range; the EN4 layer
shifted 100 m) and saves each member's change in Greenland discharge to
greenland/sensitivity. This script passes those changes through the
production rate-space blend and reports the change at 2100 in the full
forecast (p(S1) = 10%, `blended`) and in the slow-WAIS forecast
(p(S1) = 90%, built exactly as build_slow_wais_p90.py builds it). The blend
is linear in the component sum, so the forecast change is the blend of the
Greenland change alone.

It reads `blended`, `blended_stable` and `blended_slow90`, so it runs after
`slowwais` in run_pipeline.sh. The printed "Largest change" values are the
numbers quoted in the Discussion.

Run from the repository root:  python scripts/greenland_sensitivity_forecast.py
"""

import importlib.util
import sys
from pathlib import Path

import h5py
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'notebooks'))
sys.path.insert(0, str(ROOT / 'src'))

from component_projections import blend_rate_space  # noqa: E402

_spec = importlib.util.spec_from_file_location('build_slow_wais_p90',
                                               ROOT / 'scripts' / 'build_slow_wais_p90.py')
bsw = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(bsw)

RESULTS_H5 = ROOT / 'data' / 'processed' / 'component_results.h5'
M_TO_MM = 1000.0
SSPS = ['SSP1-2.6', 'SSP2-4.5', 'SSP3-7.0']
LABEL = {'SSP1-2.6': '1.8 C', 'SSP2-4.5': '2.7 C', 'SSP3-7.0': '3.6 C'}


def q(x):
    return np.array([np.median(x), np.percentile(x, 5), np.percentile(x, 95)]) * M_TO_MM


def main():
    with h5py.File(RESULTS_H5, 'r') as hf:
        sg = hf['greenland/sensitivity']
        g_years = sg['years'][:]
        n_cases = len(sg.attrs['cases'])
        cases = {sg[f'case{i}'].attrs['name']: {s: sg[f'case{i}/{s}/delta_samples'][:] for s in SSPS}
                 for i in range(n_cases)}
        sens_saved = pd.Timestamp(sg.attrs['saved_at'])
        ba = dict(hf['blended'].attrs)
        fy = hf['blended/forecast_years'][:]
        sidx = hf['blended/wais_scenario_idx'][:].astype(int)
        full = {s: hf[f'blended/{s}/samples'][:] for s in SSPS}
        stab = {s: hf[f'blended_stable/{s}/samples'][:] for s in SSPS}
        slow_stored = {s: hf[f'blended_slow90/{s}/samples'][:] for s in SSPS}

    # The blended ensembles must come from the same run as the Greenland changes
    n = full[SSPS[0]].shape[0]
    n_sens = next(iter(cases.values()))[SSPS[0]].shape[0]
    assert n == n_sens, f'blended has {n} members, greenland/sensitivity has {n_sens}'
    assert pd.Timestamp(ba['saved_at']) > sens_saved, \
        'blended predates greenland/sensitivity: rerun component_forecast and slowwais first'

    i_f = int(np.argmin(np.abs(fy - 2100)))
    zeros = np.zeros((n, len(g_years)))

    # Slow-WAIS rows exactly as build_slow_wais_p90.py builds them
    n1 = int(round(bsw.P_S1 * bsw.N_OUT))
    rows1 = np.tile(np.arange(n), n1 // n)
    rows2 = {s: bsw.s2_row_index(full[s][sidx == 1][:, i_f], bsw.N_OUT - n1) for s in SSPS}

    def slow(s, dT):
        return np.concatenate([(stab[s] + dT)[rows1, i_f], (full[s] + dT)[sidx == 1][rows2[s], i_f]])

    for s in SSPS:   # the reconstruction reproduces the stored slow-WAIS ensemble
        assert np.allclose(np.sort(slow(s, 0.0)), np.sort(slow_stored[s][:, i_f]))

    print('Change at 2100 relative to production (mm): median [5th, 95th]')
    maxmed, maxwid = 0.0, 0.0
    for name, dG in cases.items():
        print(f'  {name}:')
        for s in SSPS:
            dT, f, _ = blend_rate_space(g_years, dG[s], zeros, zeros, g_years,
                                        ba['t_origin'], 0.0, ba['t_center'], ba['tau_blend'])
            assert np.array_equal(f, fy)
            rows = [q(full[s][:, i_f] + dT[:, i_f]) - q(full[s][:, i_f]),
                    q(slow(s, dT)) - q(slow(s, 0.0))]
            maxmed = max(maxmed, *[abs(r[0]) for r in rows])
            maxwid = max(maxwid, *[abs(r[2] - r[1]) for r in rows])
            print(f'    {LABEL[s]}: full {rows[0][0]:+5.1f} [{rows[0][1]:+5.1f}, {rows[0][2]:+5.1f}]   '
                  f'slow {rows[1][0]:+5.1f} [{rows[1][1]:+5.1f}, {rows[1][2]:+5.1f}]')
    print(f'Largest change in a forecast median: {maxmed:.1f} mm; '
          f'in a 90% interval width: {maxwid:.1f} mm')


if __name__ == '__main__':
    main()
