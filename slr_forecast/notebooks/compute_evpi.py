"""
Compute EVPI / Value-of-Information surfaces and save to component_results.h5.

Three VOI surfaces are computed:
  1. Arrival year × target year: EVPI smoothed over target years, discounted
  2. Arrival year × P(stable): analytic EVSI as function of prior belief
  3. Arrival year × SLR threshold: exceedance probability shift and
     threshold-specific VOI

Usage:
    python compute_evpi.py

Writes to: data/processed/component_results.h5 under group 'voi/'
"""
import sys
from pathlib import Path

import h5py
import numpy as np
from scipy.ndimage import gaussian_filter

sys.path.insert(0, str(Path(__file__).resolve().parent))
from slr_data_readers import slr_cost_jevrejeva2018
from slr_forecast.config import PROCESSED_DATA_DIR

H5_COMP = PROCESSED_DATA_DIR / 'component_results.h5'

SSP = 'SSP2-4.5'
# Adaptation cost per meter of global SLR ($B/yr/m).
# Tiggeloven et al. (2020, doi:10.5194/nhess-20-1025-2020) estimate that
# 3.4% of the entire global coastline (~1.6M km) warrants dike heightening
# under a cost-benefit optimality criterion, at a unit cost of $7M per km
# per meter of dike height (based on New Orleans; Bos 2008).
# This gives 54,400 km × $7M/km/m ≈ $400B per meter of SLR as a
# one-time capital cost for dike raising alone. That covers only the
# 3.4% of coastline where dike economics are favorable; the remaining
# 96.6% still faces SLR via managed retreat, nature-based solutions,
# or absorbed losses. A conservative 10× multiplier for the full
# societal adaptation burden (seawalls, green barriers, managed retreat,
# etc.) across a broader fraction of coastline gives $4T/m = $4000B/m.
ADAPT_COST_PER_M = 4000   # $B per meter of SLR (see derivation above)
P_STABLE = 0.10           # AR6 prior
D_GRID = np.arange(0.3, 3.0, 0.05)
DR = 0.03                 # discount rate

# ── Vectorized damage function (numpy piecewise, no Python loop) ──
_SLR_KNOTS = np.array([0.00, 0.20, 0.52, 0.63, 0.86, 1.80])
_COST_KNOTS = np.array([0.0, 1000.0, 10200.0, 11700.0, 14000.0, 27000.0])


def damage_vec(slr_m):
    """Vectorized Jevrejeva (2018) damage function. Input/output: arrays."""
    slr_m = np.asarray(slr_m, dtype=float)
    result = np.interp(slr_m, _SLR_KNOTS, _COST_KNOTS)
    # Extrapolate beyond 1.80 m with final slope
    slope = (_COST_KNOTS[-1] - _COST_KNOTS[-2]) / (_SLR_KNOTS[-1] - _SLR_KNOTS[-2])
    mask = slr_m > _SLR_KNOTS[-1]
    result[mask] = _COST_KNOTS[-1] + slope * (slr_m[mask] - _SLR_KNOTS[-1])
    return result


def _expected_cost_voi(samples, design_m):
    """Adaptation cost + residual damages from overtopping: D(S - d) when S > d."""
    adapt = ADAPT_COST_PER_M * design_m
    overtopping = np.maximum(samples - design_m, 0.0)
    damages = damage_vec(overtopping)
    return adapt + np.mean(damages)


def _evpi_annual(full_s, stable_s):
    """EVPI in $B/yr for a single target-year slice of samples."""
    s1_p975 = np.percentile(stable_s, 97.5)
    unstable_s = full_s[full_s > s1_p975]
    if len(unstable_s) < 20:
        unstable_s = full_s[full_s > np.percentile(stable_s, 90)]
    if len(unstable_s) < 10:
        return 0.0
    costs_uc = np.array([_expected_cost_voi(full_s, d) for d in D_GRID])
    costs_st = np.array([_expected_cost_voi(stable_s, d) for d in D_GRID])
    costs_un = np.array([_expected_cost_voi(unstable_s, d) for d in D_GRID])
    c_star_uc = costs_uc.min()
    c_with_info = P_STABLE * costs_st.min() + (1 - P_STABLE) * costs_un.min()
    return max(c_star_uc - c_with_info, 0.0)


def _npv_factor(arrival_yr, target_yr, dr=DR):
    """NPV annuity factor: benefit horizon = target_yr - arrival_yr."""
    horizon = target_yr - arrival_yr
    if horizon <= 0:
        return 0.0
    return (1 - (1 + dr)**(-horizon)) / dr


def main():
    # ── Load full time series ──
    print('Loading blended samples ...')
    with h5py.File(str(H5_COMP), 'r') as hf:
        fc_yr = hf['blended/forecast_years'][:]
        full_all = hf[f'blended/{SSP}/samples'][:]
        stable_all = hf[f'blended_stable/{SSP}/samples'][:]

    iy2100 = np.argmin(np.abs(fc_yr - 2100))
    full_2100 = full_all[:, iy2100]
    stable_2100 = stable_all[:, iy2100]
    s1_p975 = np.percentile(stable_2100, 97.5)
    unstable_2100 = full_2100[full_2100 > s1_p975]

    # ================================================================
    # Break-even / ROI scalars (used by fig_value_of_intervention)
    # ================================================================
    print('Computing break-even scalars ...')
    cost_full = damage_vec(full_2100)
    cost_stable = damage_vec(stable_2100)
    delta_annual_B = float(np.mean(cost_full) - np.mean(cost_stable))

    # EVPI via adaptation optimization
    costs_uc = np.array([_expected_cost_voi(full_2100, d) for d in D_GRID])
    costs_st = np.array([_expected_cost_voi(stable_2100, d) for d in D_GRID])
    costs_un = np.array([_expected_cost_voi(unstable_2100, d) for d in D_GRID])
    c_star_uc = float(costs_uc.min())
    c_with_info = float(P_STABLE * costs_st.min() + (1 - P_STABLE) * costs_un.min())
    evpi_annual_B = max(c_star_uc - c_with_info, 0.0)

    print(f'  Intervention value: ${delta_annual_B:.0f} B/yr')
    print(f'  EVPI (annual): ${evpi_annual_B:.0f} B/yr')

    # ================================================================
    # Figure 1: EVPI at each target year, then VOI surface
    # ================================================================
    print('Computing Figure 1 data: arrival x target year ...')
    arr_yrs_1 = np.arange(2026, 2101, 2)
    tgt_yrs_1 = np.arange(2040, 2151, 2)

    evpi_raw = np.array([
        _evpi_annual(full_all[:, np.argmin(np.abs(fc_yr - ty))],
                     stable_all[:, np.argmin(np.abs(fc_yr - ty))])
        for ty in tgt_yrs_1
    ])
    evpi_smooth = gaussian_filter(evpi_raw, sigma=2)

    VOI_1 = np.full((len(tgt_yrs_1), len(arr_yrs_1)), np.nan)
    for j, ay in enumerate(arr_yrs_1):
        for i, ty in enumerate(tgt_yrs_1):
            if ay < ty:
                VOI_1[i, j] = evpi_smooth[i] * _npv_factor(ay, ty) / 1000
    print('  Done.')

    # ================================================================
    # Figure 2: analytic EVSI as function of P(stable) and arrival year
    # ================================================================
    print('Computing Figure 2 data: arrival year x resolution degree ...')
    p_stable_range = np.linspace(0.01, 0.99, 60)
    arr_yrs_2 = np.arange(2026, 2096, 2)

    costs_st_g = np.array([_expected_cost_voi(stable_2100, d) for d in D_GRID])
    costs_un_g = np.array([_expected_cost_voi(unstable_2100, d) for d in D_GRID])
    c_opt_st = costs_st_g.min()
    c_opt_un = costs_un_g.min()

    VOI_2 = np.full((len(p_stable_range), len(arr_yrs_2)), np.nan)
    for i, ps in enumerate(p_stable_range):
        costs_hedge = ps * costs_st_g + (1 - ps) * costs_un_g
        c_opt_hedge = costs_hedge.min()
        c_perfect = ps * c_opt_st + (1 - ps) * c_opt_un
        evsi_ann = max(c_opt_hedge - c_perfect, 0.0)
        for j, ay in enumerate(arr_yrs_2):
            VOI_2[i, j] = evsi_ann * _npv_factor(ay, 2100) / 1000
    print('  Done.')

    # ================================================================
    # Figure 3: exceedance probability shift and VOI by threshold
    # ================================================================
    print('Computing Figure 3 data: arrival year x SLR threshold ...')
    arr_yrs_3 = np.arange(2026, 2101, 2)
    slr_thresholds = np.linspace(0.2, 2.5, 60)

    delta_exceedance = np.full((len(slr_thresholds), len(arr_yrs_3)), np.nan)
    VOI_3 = np.full((len(slr_thresholds), len(arr_yrs_3)), np.nan)

    for j, ay in enumerate(arr_yrs_3):
        for i, th in enumerate(slr_thresholds):
            p_exc_full = np.mean(full_2100 > th)
            p_exc_stable = np.mean(stable_2100 > th)
            delta_exceedance[i, j] = (p_exc_full - p_exc_stable) * 100
            damage_at_th = slr_cost_jevrejeva2018(th)
            VOI_3[i, j] = (delta_exceedance[i, j] / 100
                           * damage_at_th * _npv_factor(ay, 2100) / 1000)

    p95_unconditional = np.percentile(full_2100, 95)
    p95_stable = np.percentile(stable_2100, 95)
    print('  Done.')

    # ── Print summary ──
    iy2100_tgt = np.argmin(np.abs(tgt_yrs_1 - 2100))
    print(f'\n=== EVPI summary ===')
    print(f'EVPI at 2100 (raw):      {evpi_raw[iy2100_tgt]:.1f} $B/yr')
    print(f'EVPI at 2100 (smoothed): {evpi_smooth[iy2100_tgt]:.1f} $B/yr')
    print(f'Max VOI surface 1:       {np.nanmax(VOI_1):.1f} $T')
    print(f'Max VOI surface 2:       {np.nanmax(VOI_2):.1f} $T')
    print(f'Max VOI surface 3:       {np.nanmax(VOI_3):.1f} $T')
    print(f'P95 unconditional:       {p95_unconditional:.3f} m')
    print(f'P95 stable:              {p95_stable:.3f} m')

    # ── Write to HDF5 ──
    with h5py.File(str(H5_COMP), 'a') as hf:
        grp_name = 'voi'
        if grp_name in hf:
            del hf[grp_name]
        g = hf.create_group(grp_name)

        # Attributes
        g.attrs['ssp'] = SSP
        g.attrs['adapt_cost_per_m'] = ADAPT_COST_PER_M
        g.attrs['p_stable_prior'] = P_STABLE
        g.attrs['discount_rate'] = DR
        g.attrs['damage_function'] = 'Jevrejeva2018'

        # Figure 1 surface
        g.create_dataset('arrival_years_1', data=arr_yrs_1)
        g.create_dataset('target_years_1', data=tgt_yrs_1)
        g.create_dataset('VOI_1', data=VOI_1)

        # Figure 2 surface
        g.create_dataset('arrival_years_2', data=arr_yrs_2)
        g.create_dataset('p_stable_range', data=p_stable_range)
        g.create_dataset('VOI_2', data=VOI_2)

        # Figure 3 surfaces
        g.create_dataset('arrival_years_3', data=arr_yrs_3)
        g.create_dataset('slr_thresholds', data=slr_thresholds)
        g.create_dataset('delta_exceedance', data=delta_exceedance)
        g.create_dataset('VOI_3', data=VOI_3)

        # Annotation values
        g.create_dataset('p95_unconditional', data=p95_unconditional)
        g.create_dataset('p95_stable', data=p95_stable)

        # Break-even / ROI scalars
        g.create_dataset('delta_annual_B', data=delta_annual_B)
        g.create_dataset('evpi_annual_B', data=evpi_annual_B)

        # Diagnostics
        g.create_dataset('evpi_raw', data=evpi_raw)
        g.create_dataset('evpi_smooth', data=evpi_smooth)

    print(f'\nWritten to {H5_COMP} under group "{grp_name}/"')


if __name__ == '__main__':
    main()
