"""
Compute optimal defense levels and cost curves, save to component_results.h5.

Optimal defense d* = argmin_d [ C_adapt * d + E[D(S - d) | S > d] ]
where D is the Jevrejeva (2018) damage function, computed for both
unconditional and stable-WAIS blended projections.

Usage:
    python compute_optimal_defense.py

Writes to: data/processed/component_results.h5 under group 'optimal_defense/'
"""
import sys
from pathlib import Path

import h5py
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from slr_forecast.config import PROCESSED_DATA_DIR

H5_COMP = PROCESSED_DATA_DIR / 'component_results.h5'

SSP = 'SSP2-4.5'
ADAPT_COST_PER_M = 4000

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


def total_cost_curve(samples, d_grid, damage_scale=1.0):
    """Total cost = adaptation cost + expected residual damages.

    Residual damages from overtopping only: D(max(S-d, 0)).
    Vectorized over both samples and d_grid.
    damage_scale: multiplicative factor on the damage function (for sensitivity).
    """
    # samples: (N,), d_grid: (M,) -> broadcast to (M, N)
    samples = np.asarray(samples)
    d_grid = np.asarray(d_grid)
    overtopping = np.maximum(samples[np.newaxis, :] - d_grid[:, np.newaxis], 0.0)
    # Compute damage for all (d, sample) pairs at once
    residual = damage_scale * np.mean(
        damage_vec(overtopping.ravel()).reshape(overtopping.shape), axis=1)
    adapt = ADAPT_COST_PER_M * d_grid
    return adapt + residual


def main():
    d_grid = np.arange(0.0, 4.0, 0.02)
    target_years = np.arange(2040, 2151, 2)
    panel_years = np.array([2050, 2100, 2150])

    # ── Load blended samples ──
    print(f'Loading samples from {H5_COMP} ...')
    with h5py.File(str(H5_COMP), 'r') as hf:
        fc_yr = hf['blended/forecast_years'][:]
        full_samples = hf[f'blended/{SSP}/samples'][:]
        stable_samples = hf[f'blended_stable/{SSP}/samples'][:]

    print(f'  Unconditional samples: {full_samples.shape}')
    print(f'  Stable WAIS samples:   {stable_samples.shape}')

    # ── d* vs target year ──
    dstar_uc = np.zeros(len(target_years))
    dstar_st = np.zeros(len(target_years))

    print(f'Computing d* for {len(target_years)} target years ...')
    for j, ty in enumerate(target_years):
        iy = np.argmin(np.abs(fc_yr - ty))
        c_uc = total_cost_curve(full_samples[:, iy], d_grid)
        c_st = total_cost_curve(stable_samples[:, iy], d_grid)
        dstar_uc[j] = d_grid[np.argmin(c_uc)]
        dstar_st[j] = d_grid[np.argmin(c_st)]
        if dstar_uc[j] >= d_grid[-1] - 0.02:
            print(f'  WARNING: d*(unconditional) at grid ceiling for {ty}')
        if dstar_st[j] >= d_grid[-1] - 0.02:
            print(f'  WARNING: d*(stable) at grid ceiling for {ty}')

    # ── Full cost curves at panel years ──
    cost_curves = {}
    for ty in panel_years:
        iy = np.argmin(np.abs(fc_yr - ty))
        cost_curves[('uc', ty)] = total_cost_curve(full_samples[:, iy], d_grid)
        cost_curves[('st', ty)] = total_cost_curve(stable_samples[:, iy], d_grid)

    # ── Sensitivity: d* with damage function scaled by 1/10 and 1/100 ──
    damage_scales = [0.1, 0.01]
    dstar_sens = {}  # {scale: {'uc': array, 'st': array}}

    for scale in damage_scales:
        print(f'Computing d* sensitivity with damage × {scale} ...')
        ds_uc = np.zeros(len(target_years))
        ds_st = np.zeros(len(target_years))
        for j, ty in enumerate(target_years):
            iy = np.argmin(np.abs(fc_yr - ty))
            c_uc = total_cost_curve(full_samples[:, iy], d_grid, damage_scale=scale)
            c_st = total_cost_curve(stable_samples[:, iy], d_grid, damage_scale=scale)
            ds_uc[j] = d_grid[np.argmin(c_uc)]
            ds_st[j] = d_grid[np.argmin(c_st)]
        dstar_sens[scale] = {'uc': ds_uc, 'st': ds_st}

    # ── Print summary ──
    print('\n=== Optimal defense levels (m rel. 2000) ===')
    print(f'{"Year":>6s}  {"d*(uncond)":>10s}  {"d*(stable)":>10s}  {"gap":>6s}')
    for ty in panel_years:
        j = np.argmin(np.abs(target_years - ty))
        gap = dstar_uc[j] - dstar_st[j]
        print(f'{ty:>6d}  {dstar_uc[j]:>10.2f}  {dstar_st[j]:>10.2f}  {gap:>6.2f}')

    print('\n=== Sensitivity: d* gap at 2100 ===')
    j2100 = np.argmin(np.abs(target_years - 2100))
    print(f'  D×1:    uc={dstar_uc[j2100]:.2f}, st={dstar_st[j2100]:.2f}, '
          f'gap={dstar_uc[j2100]-dstar_st[j2100]:.2f} m')
    for scale in damage_scales:
        s = dstar_sens[scale]
        print(f'  D×{scale}: uc={s["uc"][j2100]:.2f}, st={s["st"][j2100]:.2f}, '
              f'gap={s["uc"][j2100]-s["st"][j2100]:.2f} m')

    # ── Write to HDF5 ──
    with h5py.File(str(H5_COMP), 'a') as hf:
        grp_name = 'optimal_defense'
        if grp_name in hf:
            del hf[grp_name]
        g = hf.create_group(grp_name)

        g.attrs['ssp'] = SSP
        g.attrs['adapt_cost_per_m'] = ADAPT_COST_PER_M
        g.attrs['damage_function'] = 'Jevrejeva2018'

        g.create_dataset('d_grid', data=d_grid)
        g.create_dataset('target_years', data=target_years)
        g.create_dataset('dstar_unconditional', data=dstar_uc)
        g.create_dataset('dstar_stable', data=dstar_st)
        g.create_dataset('panel_years', data=panel_years)

        for ty in panel_years:
            g.create_dataset(f'cost_curve_uc_{ty}', data=cost_curves[('uc', ty)])
            g.create_dataset(f'cost_curve_st_{ty}', data=cost_curves[('st', ty)])

        # Sensitivity datasets
        g.create_dataset('damage_scales', data=np.array(damage_scales))
        for scale in damage_scales:
            tag = f'{scale:.0e}'.replace('+', '').replace('-', 'm')
            g.create_dataset(f'dstar_uc_scale_{tag}', data=dstar_sens[scale]['uc'])
            g.create_dataset(f'dstar_st_scale_{tag}', data=dstar_sens[scale]['st'])

    print(f'\nWritten to {H5_COMP} under group "{grp_name}/"')


if __name__ == '__main__':
    main()
