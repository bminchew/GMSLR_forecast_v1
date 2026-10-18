"""Manuscript numbers derived from stored pipeline results.

Prints, with the main-text passage each one supports, the quantities that
are computed from component_results.h5 and the AR6 files rather than printed
by a component notebook. Every number here comes from results the pipeline
has already written; nothing is refit. Companion scripts cover the rest:
slow_wais_p90_numbers.py (slow-WAIS exceedance, widths, impacts) and
variance_shares.py (variance shares). Writes manuscript_numbers.json.

Conventions: levels in metres relative to 2000 unless a line says otherwise.
"AR6 baseline" values subtract the 1995-2014 mean of Frederikse et al.
(2020) relative to 2000, as component_forecast.ipynb does. Component values
on the AR6 baseline subtract each member's own 1995-2014 mean.

Run from the repository root:  python scripts/manuscript_numbers.py
"""

import json
import sys
from pathlib import Path

import h5py
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'notebooks'))
sys.path.insert(0, str(ROOT / 'src'))

from component_projections import read_ipcc_component_nc, ipcc_extract  # noqa: E402

H5 = ROOT / 'data' / 'processed' / 'component_results.h5'
H5_OBS = ROOT / 'data' / 'processed' / 'slr_processed_data.h5'
REF_JSON = ROOT / 'data' / 'processed' / 'forecast_reference_models.json'
OUT_JSON = ROOT / 'data' / 'processed' / 'manuscript_numbers.json'
CONF_BASE = str(ROOT / 'data' / 'raw' / 'ipcc_ar6' / 'slr' / 'ar6' / 'global'
                / 'confidence_output_files')
M_TO_MM = 1000.0
BASELINE_YEAR = 2000.0
LABEL = {'SSP1-2.6': '1.8 C', 'SSP2-4.5': '2.7 C', 'SSP3-7.0': '3.6 C',
         'SSP5-8.5': '4.4 C'}
CODE = {'SSP1-2.6': 'ssp126', 'SSP2-4.5': 'ssp245', 'SSP3-7.0': 'ssp370',
        'SSP5-8.5': 'ssp585'}
POLICY = ['SSP1-2.6', 'SSP2-4.5', 'SSP3-7.0']


def q(x):
    """Median, 5th and 95th percentiles."""
    return [float(np.median(x)), float(np.percentile(x, 5)),
            float(np.percentile(x, 95))]


def fmt(v, scale=1.0, nd=2):
    return f'{v[0] * scale:.{nd}f} [{v[1] * scale:.{nd}f}, {v[2] * scale:.{nd}f}]'


def ar6(ssp, comp, conf='medium_confidence', year=2100):
    """AR6 (q05, q50, q95) in mm at *year* (relative to 1995-2014)."""
    d = read_ipcc_component_nc(CONF_BASE, conf, CODE[ssp], comp)
    if d is None:
        return None
    e = ipcc_extract(d)
    i = int(np.argmin(np.abs(e['years'] - year)))
    return float(e['q05'][i]), float(e['q50'][i]), float(e['q95'][i])


def main():
    out = {}

    # ── AR6 baseline offset (shared with component_forecast; SF-2) ──
    from component_analysis import ar6_baseline_offset_m
    off = ar6_baseline_offset_m(H5_OBS, BASELINE_YEAR)
    out['ar6_baseline_offset_m'] = off
    print(f'AR6 baseline (1995-2014 mean) minus 2000: {off * M_TO_MM:.1f} mm')

    with h5py.File(H5, 'r') as hf:
        from component_io import check_derived
        check_derived(hf)
        fy = hf['blended/forecast_years'][:]
        i21 = int(np.argmin(np.abs(fy - 2100)))
        fast = {s: hf[f'blended/{s}/samples'][:, i21] for s in LABEL}
        slow = {s: hf[f'blended_slow90/{s}/samples'][:, i21] for s in LABEL}
        cy = hf['proj_years'][:] if 'proj_years' in hf else np.arange(1950, 2151.)
        comp = {c: {s: hf[f'{c}/projections/{s}/samples'][:] for s in LABEL}
                for c in ['ocean', 'glacier', 'greenland', 'eais', 'apeninsula']}
        oc_layers = {s: {k: hf[f'ocean/projections/{s}/{k}_samples'][:]
                         for k in ['upper', 'twolayer']} for s in LABEL}
        w2k_y = hf['wais_2k/years'][:]
        w2k = hf['wais_2k/full_samples'][:]
        w_idx = hf['wais_2k/scenario_idx'][:].astype(int)
        wslow = hf['wais_slow90/samples'][:]
        wslow_y = hf['wais_slow90/years'][:]
        wobs_y = hf['wais/observations/years'][:]
        wobs_h = hf['wais/observations/H_obs'][:]
        preind = float(hf['blended'].attrs['preindustrial_to_baseline_m'])
    j21 = int(np.argmin(np.abs(cy - 2100)))

    # ── Storylines on the AR6 baseline (Discussion: comparison with AR6) ──
    print('\nTotal GMSL at 2100 on the AR6 baseline (m) vs AR6 medium confidence:')
    out['ar6'] = {}
    for s in LABEL:
        mc = ar6(s, 'total')
        lc = ar6(s, 'total', 'low_confidence')
        out['ar6'][s] = {'medium': mc, 'low': lc}
        sl, fa = q(slow[s] - off), q(fast[s] - off)
        out.setdefault('slow_ar6_baseline_m', {})[s] = sl
        out.setdefault('fast_ar6_baseline_m', {})[s] = fa
        line = (f'  {LABEL[s]}: slow {fmt(sl)}, fast {fmt(fa)}; AR6 mc median '
                f'{mc[1] / M_TO_MM:.3f}, 95th {mc[2] / M_TO_MM:.3f}')
        if lc is not None:
            line += f'; AR6 lc median {lc[1] / M_TO_MM:.3f}, 95th {lc[2] / M_TO_MM:.3f}'
        print(line)
        if s in POLICY:
            pct = 100 * (sl[0] / (mc[1] / M_TO_MM) - 1)
            out.setdefault('slow_pct_above_ar6', {})[s] = pct
            print(f'      slow median {pct:.1f}% above the AR6 mc median; slow 95th '
                  f'{"inside" if sl[2] < mc[2] / M_TO_MM else "outside"} the AR6 mc range')
            fpct = 100 * (fa[0] / (mc[1] / M_TO_MM) - 1)
            out.setdefault('fast_pct_above_ar6', {})[s] = fpct
            print(f'      fast median {fpct:.1f}% above the AR6 mc median')

    # ── Fast-WAIS forecast relative to 2000 (Results: first paragraph) ──
    print('\nFast-WAIS forecast at 2100 (m, rel. 2000):')
    for s in LABEL:
        v = q(fast[s])
        out.setdefault('fast_rel2000_m', {})[s] = v
        print(f'  {LABEL[s]}: {fmt(v)}')
    fr = out['fast_rel2000_m']
    width = fr['SSP2-4.5'][2] - fr['SSP2-4.5'][1]
    spread = fr['SSP3-7.0'][0] - fr['SSP1-2.6'][0]
    out['fast_width_vs_median_spread'] = {'width_m': width, 'spread_m': spread}
    print(f'  2.7 C 90% width {width:.2f} m = {width / spread:.1f}x the 1.8-3.6 C '
          f'median spread of {spread:.2f} m')

    # ── Upper tail of the full (fast) forecast ──
    print('\nFull forecast, 95th percentile minus median at 2100 (m, rel. 2000):')
    for s in POLICY:
        v = q(fast[s])
        out.setdefault('fast_tail_spread_m', {})[s] = v[2] - v[0]
        print(f'  {LABEL[s]}: {v[2] - v[0]:.3f}')

    # ── Hierarchy comparison (Discussion; Conclusions) ──
    if REF_JSON.exists():
        ref = json.loads(REF_JSON.read_text())
        quad = ref['quadratic_2100_m']
        r07 = ref['rahmstorf2007_2100_m']['SSP2-4.5']
        sl = q(slow['SSP2-4.5'])
        out['hierarchy'] = {'quadratic_m': quad, 'rahmstorf2007_m': r07, 'slow_2p7_m': sl}
        print(f'\nAt 2.7 C, 2100 (m, rel. 2000): slow {fmt(sl)}, quadratic {fmt(quad)}, '
              f'Rahmstorf (2007) {fmt(r07)}')
        print(f'  slow median minus quadratic: {(sl[0] - quad[0]) * M_TO_MM:+.0f} mm; '
              f'minus Rahmstorf: {(sl[0] - r07[0]) * M_TO_MM:+.0f} mm')
    else:
        print(f'\n{REF_JSON.name} missing: run component_forecast.ipynb first')

    # ── WAIS share of observed GMSL rise over the altimetry era (Results) ──
    # IMBIE-3 annual levels are end-of-year values (annualize_imbie: time
    # Y + 1 is the end of calendar year Y), so 1993.0 to the last IMBIE time
    # spans calendar years 1993 to the last full IMBIE year. GMSL is the
    # smoothed NASA altimetry record over the same window.
    na = pd.read_hdf(H5_OBS, '/harmonized/df_nasa_gmsl_h')
    nt, nv = na['decimal_year'].values, na['gmsl_smoothed'].values
    t0, t1 = 1993.0, float(wobs_y[-1])
    d_wais = float(np.interp(t1, wobs_y, wobs_h) - np.interp(t0, wobs_y, wobs_h))
    d_gmsl = float(np.interp(t1, nt, nv) - np.interp(t0, nt, nv))
    out['wais_share_altimetry'] = dict(first_year=int(t0), last_year=int(t1) - 1,
                                       wais_mm=d_wais * M_TO_MM, gmsl_mm=d_gmsl * M_TO_MM,
                                       share_pct=100 * d_wais / d_gmsl)
    print(f'\nWAIS share of observed GMSL rise, calendar years {int(t0)}-{int(t1) - 1}: '
          f'{d_wais * M_TO_MM:.1f} of {d_gmsl * M_TO_MM:.1f} mm = {100 * d_wais / d_gmsl:.1f}%')

    # ── Pre-industrial (1850-1900) to 2000 offset (Results; set in component_forecast) ──
    from component_analysis import preindustrial_to_baseline_m
    _, parts = preindustrial_to_baseline_m(H5_OBS, BASELINE_YEAR)
    out['preindustrial_to_baseline_m'] = dict(total=preind, **parts)
    print(f'\nPre-industrial to 2000 (m): {preind:.3f} = observed 1900-2000 '
          f'{parts["observed_1900"]:.3f} (Frederikse {parts["frederikse2020"]:.3f}, '
          f'Dangendorf {parts["dangendorf2024"]:.3f}) + 1850-1900 to 1900 '
          f'{parts["preind_to_1900"]:.3f}')

    # ── Calibrated components on both baselines (Results) ──
    ar6_name = {'ocean': 'oceandynamics', 'glacier': 'glaciers', 'greenland': 'GIS'}
    ref_mask = (cy >= 1995) & (cy <= 2014)
    print('\nCalibrated components at 2100 (mm): rel. 2000 | AR6 baseline | AR6 median | ratio')
    for c, an in ar6_name.items():
        for s in POLICY:
            S = comp[c][s] * M_TO_MM
            v0 = q(S[:, j21])
            va = q(S[:, j21] - S[:, ref_mask].mean(axis=1))
            a = ar6(s, an)[1]
            out.setdefault('components', {}).setdefault(c, {})[s] = dict(
                rel2000_mm=v0, ar6_baseline_mm=va, ar6_median_mm=a, ratio=va[0] / a)
            print(f'  {c:<9} {LABEL[s]}: {fmt(v0, nd=0)} | {fmt(va, nd=0)} | {a:.0f} | {va[0] / a:.2f}')

    # ── Thermosteric layers on the AR6 baseline (Results: divergence from AR6) ──
    # Upper layer (a_u S_u^2 + b_u S_u), deep layer (b_d S_d), and the
    # below-2000 m rate, each on the member's own 1995-2014 mean.
    print('\nThermosteric at 2100 by layer, AR6 baseline (mm, medians): '
          'upper | deep | below 2000 m | total - AR6 median')
    for s in POLICY:
        rb = lambda X: (X[:, j21] - X[:, ref_mask].mean(axis=1)) * M_TO_MM
        up = rb(oc_layers[s]['upper'])
        dp = rb(oc_layers[s]['twolayer']) - up
        bl = rb(comp['ocean'][s]) - rb(oc_layers[s]['twolayer'])
        gap = np.median(rb(comp['ocean'][s])) - ar6(s, 'oceandynamics')[1]
        out.setdefault('thermosteric_layers_mm', {})[s] = dict(
            upper=float(np.median(up)), deep=float(np.median(dp)),
            below_2000=float(np.median(bl)), gap_to_ar6=float(gap))
        print(f'  {LABEL[s]}: {np.median(up):.0f} | {np.median(dp):.0f} | '
              f'{np.median(bl):.0f} | {gap:+.0f}')

    # ── Greenland relative to 2008 (Results: Aschwanden & Brinkerhoff 2022,
    # whose simulations start in 2008) ──
    i08 = int(np.argmin(np.abs(cy - 2008)))
    print('\nGreenland at 2100 relative to 2008 (mm):')
    for s in POLICY:
        S = comp['greenland'][s] * M_TO_MM
        v = q(S[:, j21] - S[:, i08])
        out.setdefault('greenland_rel2008_mm', {})[s] = v
        print(f'  {LABEL[s]}: {fmt(v, nd=0)}')

    # ── Calibrated components vs total gap to AR6 (Discussion) ──
    # Sum of median excesses over the AR6 medians for thermosteric, glaciers
    # and Greenland, against the gap between the full and slow-WAIS
    # forecast medians and the AR6 medium-confidence total median.
    print('\nExcess over AR6 medians at 2100, AR6 baseline (mm): '
          'thermo + glaciers + Greenland | full forecast | slow-WAIS forecast')
    for s in POLICY:
        cal = sum(out['components'][c][s]['ar6_baseline_mm'][0]
                  - out['components'][c][s]['ar6_median_mm'] for c in ar6_name)
        mc = out['ar6'][s]['medium'][1]
        full_gap = out['fast_ar6_baseline_m'][s][0] * M_TO_MM - mc
        slow_gap = out['slow_ar6_baseline_m'][s][0] * M_TO_MM - mc
        out.setdefault('excess_over_ar6_mm', {})[s] = dict(
            calibrated=cal, full=full_gap, slow=slow_gap)
        print(f'  {LABEL[s]}: {cal:.0f} | {full_gap:.0f} | {slow_gap:.0f}')

    # ── Glacier comparison with GlacierMIP2 (2015-2100, rel. 2015) ──
    i15 = int(np.argmin(np.abs(cy - 2015)))
    print('\nGlaciers, 2015 level and 2015-2100 change (mm):')
    for s in POLICY:
        S = comp['glacier'][s] * M_TO_MM
        lev, d = np.median(S[:, i15]), np.median(S[:, j21] - S[:, i15])
        out.setdefault('glacier_2015', {})[s] = dict(level_2015_mm=lev, change_2015_2100_mm=d)
        print(f'  {LABEL[s]}: 2015 level {lev:.1f}, 2015-2100 median {d:.0f}')

    # ── Thermosteric fit: RMS residual and observed rise per record (Methods) ──
    with h5py.File(H5, 'r') as hf:
        g = hf['ocean/fit_diagnostic']
        Ty = g['T_annual_years'][:]
        print('\nThermosteric fit (mm): RMS residual | observed rise, first to last year')
        for lab, yk, ok, mk in [('0-700 m', 'noaa_year', 'noaa_700_corrected_mm', 'H_upper_fit_mm'),
                                ('0-2000 m', 'noaa_deep_year', 'noaa_2000_corrected_mm', 'H_total_fit_mm')]:
            y, o, m = g[yk][:], g[ok][:], g[mk][:]
            r = o - m[[int(np.argmin(np.abs(Ty - v))) for v in y]]
            out[f'thermo_fit_{lab}'] = dict(rms_mm=float(np.sqrt(np.mean(r ** 2))),
                                            rise_mm=float(o[-1] - o[0]))
            print(f'  {lab} ({y[0]:.1f}-{y[-1]:.1f}): RMS {np.sqrt(np.mean(r ** 2)):.3f} | rise {o[-1] - o[0]:.1f}')

    # ── Fitted nuisance parameters quoted in Methods (GL-2) ──
    print('\nFitted error and balance terms (median [5th, 95th]):')
    with h5py.File(H5, 'r') as hf:
        for c, lab in [('glacier', 'sigma_gl,extra'), ('apeninsula', 'sigma_p,extra')]:
            k = f'{c}/posteriors/sigma_extra_posterior'
            if k in hf:
                v = q(hf[k][:])
                out[f'{c}_sigma_extra_mm_yr'] = [x * M_TO_MM for x in v]
                print(f'  {lab} = {fmt(v, M_TO_MM)} mm/yr')
        if 'glacier/posteriors/T_eq0_posterior' in hf:
            v = q(hf['glacier/posteriors/T_eq0_posterior'][:])
            out['glacier_T_eq0_C'] = v
            print(f'  T_gl,eq = {fmt(v)} C (rel. 1995-2005)')
        # Rate coefficients of the linear Antarctic fits; columns (a, b, c), a = 0 (AN-20)
        for c, s in [('apeninsula', 'p'), ('eais', 'e')]:
            k = f'{c}/posteriors/posterior_samples'
            if k in hf:
                ps = hf[k][:]
                vb, vc = q(ps[:, 1]), q(ps[:, 2])
                out[f'{c}_b_mm_yr_C'] = [x * M_TO_MM for x in vb]
                out[f'{c}_c_mm_yr'] = [x * M_TO_MM for x in vc]
                print(f'  b_{s} = {fmt(vb, M_TO_MM, 3)} mm/yr/C | c_{s} = {fmt(vc, M_TO_MM, 3)} mm/yr')

    # ── Antarctic Peninsula + EAIS (Results) ──
    print('\nAntarctic Peninsula + EAIS at 2100 (mm, rel. 2000):')
    for s in POLICY:
        ap = comp['apeninsula'][s][:, j21] * M_TO_MM
        ea = comp['eais'][s][:, j21] * M_TO_MM
        v = q(ap + ea)
        out.setdefault('ap_plus_eais_mm', {})[s] = dict(sum=v, eais_median=float(np.median(ea)),
                                                        apeninsula_median=float(np.median(ap)))
        print(f'  {LABEL[s]}: sum {fmt(v, nd=1)}; EAIS median {np.median(ea):.1f}, '
              f'Peninsula median {np.median(ap):.1f}')

    # ── 90% interval shares (Results, Discussion; as in Fig. 4b) ──
    # Each component's own 90% interval width over the summed widths of
    # thermosteric, glaciers, Greenland and WAIS at 2100.
    w90 = lambda a: np.percentile(a, 95) - np.percentile(a, 5)
    wf = w90(w2k[:, int(np.argmin(np.abs(w2k_y - 2100)))])
    ws = w90(wslow[:, int(np.argmin(np.abs(wslow_y - 2100)))])
    print('\n90% interval shares at 2100 (%): WAIS | calibrated (thermosteric + glaciers + Greenland)')
    for s in POLICY:
        cal = sum(w90(comp[c][s][:, j21]) for c in ['ocean', 'glacier', 'greenland'])
        sh = {'fast': (100 * wf / (wf + cal), 100 * cal / (wf + cal)),
              'slow': (100 * ws / (ws + cal), 100 * cal / (ws + cal))}
        out.setdefault('interval_shares_pct', {})[s] = sh
        print(f'  {LABEL[s]}: fast {sh["fast"][0]:.1f} | {sh["fast"][1]:.1f};  '
              f'slow {sh["slow"][0]:.1f} | {sh["slow"][1]:.1f}')

    # ── S2 95th percentile against the AR6 Antarctic range (WAIS subsection) ──
    # Each S2 member on its own 1995-2014 mean (the AR6 baseline), as for
    # the components above (SF-13)
    s2 = w2k[w_idx == 1] * M_TO_MM
    s2 = s2 - s2[:, (w2k_y >= 1995) & (w2k_y <= 2014)].mean(axis=1, keepdims=True)
    s2_95 = np.percentile(s2, 95, axis=0)
    print('\nS2 95th percentile vs AR6 AIS medium-confidence 95th (mm, both on the AR6 baseline):')
    for s in ['SSP1-2.6', 'SSP2-4.5', 'SSP5-8.5']:
        d = read_ipcc_component_nc(CONF_BASE, 'medium_confidence', CODE[s], 'AIS')
        e = ipcc_extract(d)
        yrs = np.arange(2025, 2101)
        a95 = np.interp(yrs, e['years'], e['q95'])
        m95 = np.interp(yrs, w2k_y, s2_95)
        above = yrs[m95 > a95]
        first = int(above[0]) if above.size else None
        out.setdefault('s2_first_exceeds_ar6_ais95', {})[s] = first
        print(f'  {s}: first exceeds in {first}; 2045 {np.interp(2045, w2k_y, s2_95):.0f} vs '
              f'{np.interp(2045, e["years"], e["q95"]):.0f}, 2050 {np.interp(2050, w2k_y, s2_95):.0f} vs '
              f'{np.interp(2050, e["years"], e["q95"]):.0f}')

    # ── Warming-path spread at 2099 (supplement, Temperature scenarios) ──
    # A member at z = 1.645 follows the AR6 95th percentile path offset to
    # zero at WARMING_ANCHOR_YEAR; compare its 2099 excess over the median
    # with the full AR6 upper half-width.
    from warming_paths import ssp_row_offsets, WARMING_ANCHOR_YEAR
    print(f'\nUpper half-width of AR6 GMST at 2099 (95th minus median, C), '
          f'full vs warming path anchored at {WARMING_ANCHOR_YEAR:.0f}:')
    for s in ['SSP1-2.6', 'SSP2-4.5', 'SSP3-7.0']:
        df = pd.read_hdf(H5_OBS, f'projections/temp/{s.replace("-", "_").replace(".", "_")}')
        i99 = int(np.argmin(np.abs(df['decimal_year'].values - 2099)))
        full = float(df['temperature_upper'].values[i99] - df['temperature'].values[i99])
        path = float(ssp_row_offsets(df, np.array([1.645]))[0, i99])
        out.setdefault('warming_upper_halfwidth_2099_C', {})[s] = dict(full=full, path=path)
        print(f'  {LABEL[s]}: full {full:.2f}, path {path:.2f}')

    # ── Warming still to come on the AR6 median paths (main text) ──
    # Compared with the 1950-2025 validation window (component_summation).
    print('\nAR6 median GMST, warming added from 2025 to 2099 (C):')
    for s in ['SSP1-2.6', 'SSP2-4.5', 'SSP3-7.0']:
        df = pd.read_hdf(H5_OBS, f'projections/temp/{s.replace("-", "_").replace(".", "_")}')
        yrs, t = df['decimal_year'].values, df['temperature'].values
        added = float(np.interp(2099, yrs, t) - np.interp(2025, yrs, t))
        out.setdefault('warming_added_2025_2099_C', {})[s] = added
        print(f'  {LABEL[s]}: {added:.2f}')

    # ── People on land below the annual flood level per cm of GMSL rise ──
    # Secant slope of our quadratic fit to Kulp & Strauss (2019) over 0-1 m.
    from slr_data_readers import people_displaced_kulpstrauss2019 as kulp
    per_cm = float((kulp(1.0) - kulp(0.0)) / 100)
    out['kulp_people_per_cm_M'] = per_cm
    print(f'\nPeople below the annual flood level per cm of rise, 0-1 m secant '
          f'(Kulp & Strauss 2019 fit): {per_cm:.2f} million')

    OUT_JSON.write_text(json.dumps(out, indent=2, default=float))
    print(f'\nWrote {OUT_JSON}')


if __name__ == '__main__':
    main()
