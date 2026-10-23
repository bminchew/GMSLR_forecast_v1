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

from component_projections import read_ipcc_component_nc, ipcc_extract, A4_SCENARIOS  # noqa: E402

H5 = ROOT / 'data' / 'processed' / 'component_results.h5'
H5_OBS = ROOT / 'data' / 'processed' / 'slr_processed_data.h5'
REF_JSON = ROOT / 'data' / 'processed' / 'forecast_reference_models.json'
OUT_JSON = ROOT / 'data' / 'processed' / 'manuscript_numbers.json'
TRS_JSON = ROOT / 'data' / 'processed' / 'thermosteric_record_sensitivity.json'
CZ_JSON = ROOT / 'data' / 'processed' / 'cmip6_zostoga_summary.json'
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
        # 700-2000 m (0-2000 m minus 0-700 m) is fitted after removing its mean,
        # so model and data are compared about their means over the record
        for lab, yk, ok, mk in [('0-700 m', 'noaa_year', 'noaa_700_corrected_mm', 'H_upper_fit_mm'),
                                ('700-2000 m', 'noaa_deep_year', 'noaa_700_2000_corrected_mm', 'H_mid_fit_mm')]:
            y, o, m = g[yk][:], g[ok][:], g[mk][:]
            r = o - m[[int(np.argmin(np.abs(Ty - v))) for v in y]]
            if lab == '700-2000 m':
                r = r - r.mean()
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

    # ── Energetic consistency of the thermosteric projections (Supplement) ──
    # Implied ocean heat uptake = thermosteric change / expansion efficiency
    # of heat (EEH), on the AR6 window: mean(2081-2100) minus mean(1995-2014),
    # each member's own means. Literature values, Fox-Kemper et al. (2021,
    # AR6 WG1 Ch. 9, Section 9.2.2.1 and Table 9.1; Zotero ZIFW4T59):
    #   EEH 0.113 +/- 0.013 m/YJ (mean +/- sd, two-layer emulator);
    #   two-layer total OHC, median (17-83%), YJ;
    #   CMIP6 0-2000 m OHC, median (17-83%), YJ.
    # 0-2000 m EEH from Levitus et al. (2012, Zotero PCQMERBI, abstract and
    # Section 3): thermosteric trend 0.54 mm/yr over a heat-content trend of
    # 0.43e22 J/yr, 1955-2010. Both carry the same interpolation bias (Li et al.
    # 2022), so the ratio applies to the x1.14-scaled record.
    EEH, EEH_SD = 0.113, 0.013
    EEH_2000 = 0.54e-3 / 0.43e22 * 1e24
    AR6_OHC = {'SSP1-2.6': (1.18, 0.99, 1.42), 'SSP2-4.5': (1.56, 1.33, 1.86),
               'SSP3-7.0': (1.90, 1.63, 2.21), 'SSP5-8.5': (2.23, 1.92, 2.64)}
    CMIP6_OHC2000 = {'SSP1-2.6': (1.06, 0.80, 1.31), 'SSP2-4.5': (1.35, 1.08, 1.67),
                     'SSP3-7.0': (1.62, 1.37, 1.91), 'SSP5-8.5': (1.89, 1.60, 2.29)}
    from slr_forecast.config import SEEDS
    rng = np.random.default_rng(SEEDS['projections'])
    late = (cy >= 2081) & (cy <= 2100)
    win = lambda X: (X[:, late].mean(axis=1) - X[:, ref_mask].mean(axis=1)) * M_TO_MM
    print(f'\nImplied ocean heat uptake, mean 2081-2100 minus mean 1995-2014 '
          f'(EEH {EEH} +/- {EEH_SD} m/YJ; 0-2000 m EEH {EEH_2000:.3f} m/YJ):')
    print('  scenario: thermosteric (mm) | Q at 0.113 (YJ) | Q, EEH drawn (YJ) | '
          'AR6 two-layer | ratio | EEH needed | 0-2000 m thermo (mm) | '
          'Q 0-2000 (YJ) | CMIP6 0-2000 | ratio')
    for s in LABEL:
        dH = win(comp['ocean'][s])
        d2 = win(oc_layers[s]['twolayer'])
        e_draw = rng.normal(EEH, EEH_SD, dH.size)
        Q, Qd, Q2 = dH / 1e3 / EEH, dH / 1e3 / e_draw, d2 / 1e3 / EEH_2000
        a, c = AR6_OHC[s], CMIP6_OHC2000[s]
        out.setdefault('energy_check', {})[s] = dict(
            thermo_mm=q(dH), Q_YJ=q(Q), Q_drawn_YJ=q(Qd), ar6_two_layer_YJ=a,
            ratio=float(np.median(Q) / a[0]), eeh_needed=float(np.median(dH) / 1e3 / a[0]),
            thermo_0_2000_mm=q(d2), Q_0_2000_YJ=q(Q2), cmip6_0_2000_YJ=c,
            ratio_0_2000=float(np.median(Q2) / c[0]))
        print(f'  {LABEL[s]} ({s}): {fmt(q(dH), nd=0)} | {fmt(q(Q))} | {fmt(q(Qd))} | '
              f'{a[0]:.2f} ({a[1]:.2f}-{a[2]:.2f}) | {np.median(Q) / a[0]:.2f} | '
              f'{np.median(dH) / 1e3 / a[0]:.3f} | {fmt(q(d2), nd=0)} | {fmt(q(Q2))} | '
              f'{c[0]:.2f} ({c[1]:.2f}-{c[2]:.2f}) | {np.median(Q2) / c[0]:.2f}')
    e = out['energy_check']
    print(f'  ratio 4.4 C / 1.8 C: ours {e["SSP5-8.5"]["Q_YJ"][0] / e["SSP1-2.6"]["Q_YJ"][0]:.2f}, '
          f'AR6 two-layer {AR6_OHC["SSP5-8.5"][0] / AR6_OHC["SSP1-2.6"][0]:.2f}; '
          f'3.6 C / 1.8 C: ours {e["SSP3-7.0"]["Q_YJ"][0] / e["SSP1-2.6"]["Q_YJ"][0]:.2f}, '
          f'AR6 {AR6_OHC["SSP3-7.0"][0] / AR6_OHC["SSP1-2.6"][0]:.2f}')
    _ns = [(EEH - e[s]['eeh_needed']) / EEH_SD for s in e]
    print(f'  EEH needed to match AR6 lies {min(_ns):.1f} to {max(_ns):.1f} standard deviations '
          f'({EEH_SD} m/YJ) below {EEH} m/YJ')

    # ── Sensitivity to the thermosteric calibration record (Supplement) ──
    # Refits on EN4 and IAP in place of NOAA, unscaled and x1.14
    # (scripts/thermosteric_record_sensitivity.py). Trends are OLS with 90%
    # intervals from an AR(1)-adjusted effective sample size.
    trs = json.loads(TRS_JSON.read_text())
    tr = trs['trends_mm_per_yr']
    iv = lambda v: [v[0] - v[1], v[0] + v[1]]
    print(f'\nThermosteric calibration-record sensitivity')
    print(f'  IAP heat content to height (EN4 ratio, {trs["conv_window"][0]}-{trs["conv_window"][1]}): '
          f'0-700 m {trs["mm_per_zj"]["700"]:.3f}, 0-2000 m {trs["mm_per_zj"]["2000"]:.3f} mm/ZJ')
    print('  Trends (mm/yr, unscaled): NOAA | EN4 | IAP | EN4 - NOAA [90%] | IAP - NOAA [90%]')
    for k in ('0-700m_1955-2012', '0-700m_2005-2024', '0-700m_2013-2024',
              '0-2000m_2005-2024', '0-2000m_2013-2024'):
        t = tr[k]
        e, i = iv(t['en4_minus_noaa']), iv(t['iap_minus_noaa'])
        print(f'    {k}: {t["noaa"][0]:.2f} | {t["en4"][0]:.2f} | {t["iap"][0]:.2f} | '
              f'{t["en4_minus_noaa"][0]:+.2f} [{e[0]:.2f}, {e[1]:.2f}] | '
              f'{t["iap_minus_noaa"][0]:+.2f} [{i[0]:.2f}, {i[1]:.2f}]')
    fits, prod = trs['fits'], trs['production']['proj2100']
    # the JSON snapshots must come from the current component_results.h5
    for s in POLICY:
        h5_med = float(np.median(comp['ocean'][s][:, j21])) * M_TO_MM
        assert abs(prod[s][0] - h5_med) < 0.05, (
            f'thermosteric_record_sensitivity.json is stale ({s}: {prod[s][0]:.2f} vs h5 '
            f'{h5_med:.2f} mm); rerun scripts/thermosteric_record_sensitivity.py all')
    print('  2100 thermosteric rise rel. 2000 (mm), median [5th, 95th]:')
    for s in POLICY:
        print(f'    {LABEL[s]}: production {fmt(prod[s], nd=0)}; '
              + '; '.join(f'{k} {fmt(f["proj2100"][s], nd=0)}' for k, f in fits.items()))
    rec_diff = [fits[f'{r}_{x}']['proj2100'][s][0] - fits[f'noaa_{x}']['proj2100'][s][0]
                for r in ('en4', 'iap') for x in ('x100', 'x114') for s in POLICY]
    scl = [fits[f'{r}_x114']['proj2100'][s][0] - fits[f'{r}_x100']['proj2100'][s][0]
           for r in ('noaa', 'en4', 'iap') for s in POLICY]
    inside = all(prod[s][1] <= f['proj2100'][s][0] <= prod[s][2] for f in fits.values() for s in POLICY)
    print(f'  EN4/IAP minus NOAA at equal scaling, 2100 medians: {min(rec_diff):+.0f} to {max(rec_diff):+.0f} mm')
    print(f'  x1.14 minus x1.00, 2100 medians: {min(scl):.0f} to {max(scl):.0f} mm')
    print(f'  every refit median inside the production 90% interval: {inside}')
    print('  sigma_extra,u (mm) | rho, medians: '
          + '; '.join(f'{k} {f["params"]["sigma_extra_mm"][0]:.2f} | {f["params"]["rho"][0]:.2f}'
                      for k, f in fits.items()))
    out['thermo_record_sensitivity'] = dict(
        mm_per_zj=trs['mm_per_zj'], trends={k: tr[k] for k in tr},
        proj2100={k: {s: f['proj2100'][s] for s in POLICY} for k, f in fits.items()},
        record_diff_mm=[min(rec_diff), max(rec_diff)], scaling_effect_mm=[min(scl), max(scl)],
        all_inside_production_90=inside,
        sigma_extra_u_mm={k: f['params']['sigma_extra_mm'] for k, f in fits.items()},
        rho={k: f['params']['rho'] for k, f in fits.items()})

    # ---- CMIP6 zostoga comparison (scripts/cmip6_zostoga.py analyze), supp Fig. S12c,d
    cz = json.loads(CZ_JSON.read_text())
    ct, cp = cz['trends_mm_per_yr'], cz['per_degree']
    _ref = (cy >= 1995) & (cy <= 2014)
    _o = comp['ocean']['SSP2-4.5'] * M_TO_MM
    _h5 = float(np.median(_o[:, j21] - _o[:, _ref].mean(axis=1)))
    _cz = cp['ar6_reference']['production_dH_2100_rel_1995_2014_mm'][0]
    assert abs(_cz - _h5) < 0.05, (f'cmip6_zostoga_summary.json is stale ({_cz:.2f} vs h5 '
                                   f'{_h5:.2f} mm); rerun scripts/cmip6_zostoga.py analyze')
    fq = lambda q, d=2: f'{q[0]:.{d}f} [{q[1]:.{d}f}, {q[2]:.{d}f}]'
    print('\nCMIP6 zostoga (median [5th, 95th] across models; production median [90% CI])')
    print(f'  models analysed {cz["n_models"]}, reaching 2024 under SSP2-4.5 {cz["n_reaching_2024"]}, '
          f'used after 2014 {len(cz["post2014_models"])} (join-flagged: {", ".join(cz["join_flagged"])}); '
          f'drift fits {cz["drift"]["mode_counts"]}')
    for w in ('1955-2012', '2005-2024', '2013-2024'):
        r = ct[w]
        line = (f'  {w} full-depth trend (mm/yr): CMIP6 {fq(r["cmip6"])} (n={r["n"]}), '
                f'this study {fq(r["production_full"])}, this study 0-700 m {fq(r["production_0_700"])}')
        if 'obs_full_depth_est' in r:
            ob = [r['obs_full_depth_est'][k][0] for k in ('noaa', 'en4', 'iap')]
            line += f', observed 0-2000 m + below 2000 m (NOAA, EN4, IAP unscaled) {min(ob):.2f} to {max(ob):.2f}'
        print(line)
    wm, fu = cp['window_means'], cp['ssp245_2081_2100_vs_1995_2014']
    a6 = cp['ar6_reference']
    print(f'  per degree 1955-74 -> 2005-24 (mm/C): CMIP6 {fq(wm["cmip6_mm_per_K"], 0)} (n={wm["n"]}), '
          f'this study {fq(wm["production_mm_per_K"], 0)}')
    print(f'  per degree 1995-2014 -> 2081-2100 SSP2-4.5 (mm/C): CMIP6 {fq(fu["cmip6_mm_per_K"], 0)} '
          f'(n={fu["n"]}), this study {fq(fu["production_mm_per_K"], 0)} (own warming paths), '
          f'AR6 ratio of medians {a6["mm_per_K_ratio_of_medians"]:.0f}')
    out['cmip6_zostoga'] = dict(
        n_models=cz['n_models'], n_reaching_2024=cz['n_reaching_2024'],
        n_ssp245=len(cz['post2014_models']),
        trends={w: dict(cmip6=ct[w]['cmip6'], production_full=ct[w]['production_full'],
                        production_0_700=ct[w]['production_0_700'],
                        obs_full_depth_est=ct[w].get('obs_full_depth_est'))
                for w in ('1955-2012', '2005-2024', '2013-2024')},
        mm_per_K_hist=dict(cmip6=wm['cmip6_mm_per_K'], production=wm['production_mm_per_K']),
        mm_per_K_ssp245=dict(cmip6=fu['cmip6_mm_per_K'],
                             production=fu['production_mm_per_K'],
                             ar6_ratio_of_medians=a6['mm_per_K_ratio_of_medians']))

    # ── Derived values and settings quoted in the main text ──
    # Numbers that are arithmetic on inputs, analytic consequences of a
    # setting, or settings themselves, printed so that every number in the
    # .tex traces to code.
    print('\nDerived values and settings quoted in the text:')
    der = out.setdefault('derived', {})

    # AR6 central GMST paths at 2099 (rel. 1850-1900), the last AR6 year.
    t99 = {}
    for s in POLICY:
        df = pd.read_hdf(H5_OBS, f'projections/temp/{s.replace("-", "_").replace(".", "_")}')
        t99[s] = float(np.interp(2099, df['decimal_year'].values, df['temperature'].values))
    der['ar6_median_gmst_2099_C'] = t99
    print('  AR6 median GMST at 2099 (C, rel. 1850-1900): '
          + ', '.join(f'{LABEL[s]} {v:.2f}' for s, v in t99.items()))

    # Thermosteric upper-layer response-time prior (component_ocean cell 5).
    from slr_forecast.config import TAU_U_PRIOR_MEDIAN_YR, TAU_U_PRIOR_LOG_SD
    lo, hi = (TAU_U_PRIOR_MEDIAN_YR * np.exp(z * 1.6448536 * TAU_U_PRIOR_LOG_SD) for z in (-1, 1))
    der['tau_u_prior_yr'] = [TAU_U_PRIOR_MEDIAN_YR, lo, hi]
    print(f'  tau_u prior: median {TAU_U_PRIOR_MEDIAN_YR:.1f} yr, 90% [{lo:.2f}, {hi:.2f}] yr')

    # Greenland ocean-sensitivity scaling (Methods, Eqs. melt_rate to gamma_estimate).
    # Literature inputs as stated in the text: maximum melt sensitivity
    # 90-260 m/yr/C (Xu 2013, Rignot 2016, Slater 2022), face average 2.5x
    # smaller (Wood 2021), mu = 3-10 (assumed), front area 0.9-2.7 km2
    # (Rignot 2016), N = 219 outlets (Mouginot 2019).
    KMAX, FACE, MU, AF_KM2, N_OUT = (90.0, 260.0), 2.5, (3.0, 10.0), (0.9, 2.7), 219
    RHO_ICE, RHO_FW, A_OCEAN = 900.0, 1000.0, 3.625e14
    kap = tuple(k / FACE for k in KMAX)
    gt = tuple(N_OUT * m * RHO_ICE * a * 1e6 * k / (A_OCEAN * RHO_FW) * M_TO_MM
               for m, a, k in zip(MU, AF_KM2, kap))
    der['greenland_scaling'] = dict(kappa_i=kap, kappa_i_geomean=float(np.sqrt(kap[0] * kap[1])),
                                    gamma_tilde=gt, gamma_tilde_ratio=gt[1] / gt[0])
    print(f'  kappa_i = {kap[0]:.0f}-{kap[1]:.0f} m/yr/C (geometric mean {np.sqrt(kap[0] * kap[1]):.0f}); '
          f'gamma~ = {gt[0]:.3f}-{gt[1]:.2f} mm/yr/C, ratio {gt[1] / gt[0]:.0f}')

    # Glacier calibration window: GMST span of the GlaMBIE record.
    be = pd.read_hdf(H5_OBS, 'harmonized/df_berkeley_h')['temperature']
    ta = be.groupby(be.index.year).mean().loc[2000:2023]
    der['glambie_gmst_span_C'] = float(ta.max() - ta.min())
    print(f'  Annual GMST range over the GlaMBIE record 2000-2023: {ta.max() - ta.min():.2f} C')

    # Peripheral glaciers (scripts/glacier_bookkeeping_check.py).
    from glacier_bookkeeping_check import read as gl_read, fit as gl_fit
    T_be = be.groupby(be.index.year).mean()
    T_be = T_be - T_be.loc[1995:2005].mean()
    r5, r19 = gl_read('5_greenland_periphery'), gl_read('19_antarctic_and_subantarctic')
    b5, c5 = gl_fit(r5, T_be)
    r5_1950 = float(-(b5 * T_be + c5).loc[1950:1999].sum())
    der['rgi19_mean_rate_mm_yr'] = float(r19['rate'].mean())
    der['rgi5_rise_1950_2000_mm'] = r5_1950
    print(f'  Antarctic peripheral glaciers (RGI 19), mean rate {r19.index.min()}-{r19.index.max()}: '
          f'{r19["rate"].mean():.3f} mm/yr; Greenland peripheral glaciers (RGI 5), '
          f'hindcast rise 1950-2000: {-r5_1950:.1f} mm')

    # Blend weight (component_forecast): w falls from 0.9 to 0.1 over
    # t_center -/+ ln(9) tau_blend.
    with h5py.File(H5, 'r') as hf:
        tc = float(hf['blended'].attrs['t_center'])
        tb = float(hf['blended'].attrs['tau_blend'])
    half = np.log(9.0) * tb
    der['blend'] = dict(t_center=tc, tau_blend=tb, half_width_in_tau=np.log(9.0), span_yr=2 * half)
    print(f'  Blend weight 0.9 to 0.1 over t_r -/+ {np.log(9.0):.2f} tau_r = '
          f'{tc - half:.1f}-{tc + half:.1f}, a span of {2 * half:.1f} yr')

    # S2 95th percentile compared with Lin et al. (2026), whole AIS, 4.4 C, 95th = 254 mm.
    LIN2026_P95_MM = 254.0
    s2_hi = float(A4_SCENARIOS['S2_fast_wais']['high_mm'])
    der['lin2026_p95_over_s2_p95'] = LIN2026_P95_MM / s2_hi
    print(f'  Lin et al. (2026) 95th / S2 95th: {LIN2026_P95_MM:.0f} / {s2_hi:.0f} mm = '
          f'{LIN2026_P95_MM / s2_hi:.2f}')

    # Discount rate of the value-of-information analysis (compute_evpi).
    with h5py.File(H5, 'r') as hf:
        dr = float(hf['voi'].attrs['discount_rate'])
    der['discount_rate'] = dr
    print(f'  Discount rate, value of information: {100 * dr:.0f}% per year')

    OUT_JSON.write_text(json.dumps(out, indent=2, default=float))
    print(f'\nWrote {OUT_JSON}')


if __name__ == '__main__':
    main()
