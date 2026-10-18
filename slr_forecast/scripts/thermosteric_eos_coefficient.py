"""Equation-of-state coefficient kappa that ties the thermosteric quadratic
a_u to b_u in the two-layer model (component_ocean.ipynb, cell 5).

Main text (Eqs. for H_u): with Delta theta(z, t) = S_u(t) phi(z) over the
upper h_u = 700 m and alpha_sw = alpha_0(z) + alpha'(z) theta',

    a_u = 1/2 int_0^h alpha'(z) phi(z)^2 dz,   b_u = int_0^h alpha_0(z) phi(z) dz.

For a profile shape g(z) with phi = A g, eliminating A gives

    a_u = kappa * b_u**2,   kappa = 1/2 int alpha' g^2 dz / (int alpha_0 g dz)^2,

so a_u is fixed by seawater physics once b_u is fitted (decision D5,
2026-10-18: a_u is not a free parameter, because the NOAA data after 1990
leave it unconstrained). kappa has units of 1/m (a_u in m/C^2, b_u in m/C).

alpha_0(z) and alpha'(z) = d alpha / d CT are TEOS-10 values (gsw) computed
in every EN4.2.2 grid cell from the 1995-2005 mean temperature and salinity
(the model's reference window), then averaged horizontally with area
weights, because a horizontally uniform warming phi(z) samples the
horizontal mean of alpha at each depth. Layer integrals use the EN4
depth_bnds, with the layer containing 700 m cut at 700 m.

Shape g(z): uniform over 0-700 m (production). The other shapes are printed
for the sensitivity statement only.

Requires gsw (TEOS-10; pip install gsw), which the pipeline itself does not
need: the printed KAPPA_EOS is copied into component_ocean.ipynb cell 5 and
scripts/ocean_early_holdout.py.

Usage:  python scripts/thermosteric_eos_coefficient.py
"""
import zipfile
from pathlib import Path

import gsw
import netCDF4 as nc
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
EN4_DIR = ROOT / 'data' / 'raw' / 'ocean_temp' / 'en4'
YEARS = range(1995, 2006)          # 1995-2005, the model's reference window
H_U = 700.0                        # m, upper-layer depth
D_CT = 0.5                         # C, half-step for the central difference


def _mean_fields():
    """1995-2005 mean EN4 potential temperature (C) and salinity, 0-700 m."""
    t_sum = s_sum = None
    n = 0
    for yr in YEARS:
        with zipfile.ZipFile(EN4_DIR / f'EN.4.2.2.analyses.g10.{yr}.zip') as zf:
            for name in sorted(zf.namelist()):
                if not name.endswith('.nc'):
                    continue
                ds = nc.Dataset('inmemory.nc', memory=zf.read(name))
                if t_sum is None:
                    depth = ds.variables['depth'][:].data
                    bnds = ds.variables['depth_bnds'][:].data
                    lat = ds.variables['lat'][:].data
                    lon = ds.variables['lon'][:].data
                    k = int(np.searchsorted(bnds[:, 0], H_U))   # levels starting above 700 m
                t = ds.variables['temperature'][0, :k].filled(np.nan) - 273.15
                s = ds.variables['salinity'][0, :k].filled(np.nan)
                ds.close()
                t_sum = t if t_sum is None else t_sum + t
                s_sum = s if s_sum is None else s_sum + s
                n += 1
    return t_sum / n, s_sum / n, depth[:k], np.minimum(bnds[:k], H_U), lat, lon, n


def main():
    pt, sp, depth, bnds, lat, lon, n_months = _mean_fields()
    p = gsw.p_from_z(-depth[:, None, None], lat[None, :, None])
    lon3, lat3 = np.meshgrid(lon, lat)
    sa = gsw.SA_from_SP(sp, p, lon3[None], lat3[None])
    ct = gsw.CT_from_pt(sa, pt)
    alpha = gsw.alpha(sa, ct, p)
    alpha_p = (gsw.alpha(sa, ct + D_CT, p) - gsw.alpha(sa, ct - D_CT, p)) / (2 * D_CT)

    w = np.broadcast_to(np.cos(np.deg2rad(lat))[None, :, None], alpha.shape)
    ok = np.isfinite(alpha) & np.isfinite(alpha_p)
    a0_z = np.nansum(np.where(ok, alpha * w, 0), axis=(1, 2)) / np.sum(np.where(ok, w, 0), axis=(1, 2))
    ap_z = np.nansum(np.where(ok, alpha_p * w, 0), axis=(1, 2)) / np.sum(np.where(ok, w, 0), axis=(1, 2))
    dz = bnds[:, 1] - bnds[:, 0]
    zc = 0.5 * (bnds[:, 0] + bnds[:, 1])

    print(f'EN4.2.2, {n_months} monthly analyses {YEARS[0]}-{YEARS[-1]}; '
          f'{len(depth)} levels to {bnds[-1, 1]:.0f} m; gsw {gsw.__version__}')
    print(f'{"z (m)":>8} {"alpha_0 (1e-4/C)":>17} {"alpha\' (1e-5/C^2)":>18}')
    for i in range(0, len(depth), 3):
        print(f'{depth[i]:8.0f} {a0_z[i] * 1e4:17.3f} {ap_z[i] * 1e5:18.3f}')

    shapes = {
        'uniform (production)': np.ones_like(zc),
        'linear, zero at 700 m': 1.0 - zc / H_U,
        'exponential, e-fold 400 m': np.exp(-zc / 400.0),
        'exponential, e-fold 200 m': np.exp(-zc / 200.0),
        'exponential, e-fold 100 m': np.exp(-zc / 100.0),
    }
    print(f'\n{"shape g(z)":28s} {"kappa (1/m)":>12} {"a_u at b_u = 50 mm/C (mm/C^2)":>31}')
    for name, g in shapes.items():
        kappa = 0.5 * np.sum(ap_z * g**2 * dz) / np.sum(a0_z * g * dz) ** 2
        print(f'{name:28s} {kappa:12.4e} {kappa * 0.050**2 * 1e3:31.3f}')
        if name.startswith('uniform'):
            kappa_prod = kappa
    print(f'\nKAPPA_EOS = {kappa_prod:.4e}  # 1/m, uniform shape; a_u = KAPPA_EOS * b_u**2')


if __name__ == '__main__':
    main()
