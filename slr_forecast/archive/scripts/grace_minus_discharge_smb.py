#!/usr/bin/env python3
"""
Compute RCM-independent Greenland SMB from GRACE total − satellite discharge.

GRACE/GRACE-FO measures total ice sheet mass balance independent of any
regional climate model. Mouginot/Mankoff provide satellite-derived ice
discharge from velocity × thickness observations. The difference is an
observationally constrained estimate of surface mass balance:

    SMB_implied = dM_total/dt − D_satellite

This script:
1. Loads IMBIE Greenland total mass balance (1992–2020, monthly, GRACE-dominated post-2002)
2. Loads Mouginot (1972–2018) and Mankoff (1986–2024) discharge rates
3. Computes implied SMB = total − discharge on overlapping annual grid
4. Loads Greenland surface temperature (Berkeley Earth gridded)
5. Regresses implied SMB against temperature to estimate C_T
6. Outputs results to CSV and prints summary

Usage:
    python scripts/grace_minus_discharge_smb.py

Output:
    data/processed/greenland_implied_smb.csv
    data/processed/greenland_ct_estimate.json

Sign convention: positive = sea level rise (mass loss from ice sheet).
    SMB rates are negative when accumulation exceeds ablation (mass gain → negative SLR).
    Discharge rates are positive (mass leaving ice sheet → positive SLR).
    Total MB rate: positive = mass loss = SLR contribution.
"""

import sys
import os
import json
import numpy as np
import pandas as pd
import statsmodels.api as sm

# Add project paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'notebooks'))
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'src'))

from slr_forecast.readers.ice_sheets import (
    read_mouginot2019_greenland,
    read_mankoff2021_greenland,
)
from slr_data_readers import read_berkeley_earth_gridded

# =========================================================================
# Paths
# =========================================================================
RAW_DIR = os.path.join(PROJECT_ROOT, 'data', 'raw')
PROC_DIR = os.path.join(PROJECT_ROOT, 'data', 'processed')
H5_PATH = os.path.join(PROC_DIR, 'slr_processed_data.h5')

IMBIE_PATH = os.path.join(RAW_DIR, 'ice_sheets', 'Gt',
                          'imbie_greenland_2021_Gt.csv')
MOUGINOT_PATH = os.path.join(RAW_DIR, 'ice_sheets', 'greenland',
                             'mouginot2019_data.xlsx')
MANKOFF_PATH = os.path.join(RAW_DIR, 'ice_sheets', 'greenland',
                            'mankoff', 'MB_SMB_D_BMB_ann.csv')
GRIDDED_T_PATH = os.path.join(RAW_DIR, 'gmst',
                              'berkEarth_Global_TAVG_Gridded_1deg.nc')

BASELINE_YEAR = 2005.0
BASELINE_WINDOW = (1995, 2005)
try:
    from slr_forecast import M_TO_MM
    from slr_forecast.config import GT_TO_M_SLE
except ImportError:
    M_TO_MM = 1000.0
    GT_TO_M_SLE = 1.0 / 362500.0

# =========================================================================
# 1. Load IMBIE Greenland total mass balance
# =========================================================================
print('1. Loading IMBIE Greenland total mass balance...')

df_imbie = pd.read_csv(IMBIE_PATH)
imbie_year = df_imbie['Year'].values
imbie_rate_gt = df_imbie['Mass balance (Gt/yr)'].values          # Gt/yr, negative = mass loss
imbie_rate_sigma_gt = df_imbie['Mass balance uncertainty (Gt/yr)'].values

# Annualise: compute annual-mean rate for each calendar year
imbie_yr_int = np.floor(imbie_year).astype(int)
imbie_unique_years = np.unique(imbie_yr_int)

imbie_annual = []
for yr in imbie_unique_years:
    mask = imbie_yr_int == yr
    n = mask.sum()
    if n < 6:
        continue   # require at least 6 months
    rate_mean = imbie_rate_gt[mask].mean()
    rate_sigma = np.sqrt(np.mean(imbie_rate_sigma_gt[mask]**2))  # combine in quadrature
    imbie_annual.append({
        'year': yr + 0.5,
        'total_rate_gt': rate_mean,
        'total_rate_sigma_gt': rate_sigma,
    })

df_imbie_ann = pd.DataFrame(imbie_annual)

# Convert to SLR convention: IMBIE negative = mass loss = positive SLR
# total_rate_gt is in Gt/yr where negative = mass loss
# For SLR: positive = mass loss, so negate
df_imbie_ann['total_rate_slr'] = -df_imbie_ann['total_rate_gt'] * GT_TO_M_SLE   # m/yr, positive = SLR
df_imbie_ann['total_rate_slr_sigma'] = df_imbie_ann['total_rate_sigma_gt'] * GT_TO_M_SLE

print(f'  IMBIE: {df_imbie_ann["year"].iloc[0]:.0f}–{df_imbie_ann["year"].iloc[-1]:.0f} '
      f'({len(df_imbie_ann)} annual pts)')
print(f'  Mean rate: {df_imbie_ann["total_rate_gt"].mean():.0f} Gt/yr '
      f'({df_imbie_ann["total_rate_slr"].mean()*M_TO_MM:.2f} mm/yr SLR)')

# =========================================================================
# 2. Load satellite-derived discharge
# =========================================================================
print('\n2. Loading satellite-derived discharge...')

# Mouginot (1972–2018): discharge_rate in m/yr SLE, positive = SLR
df_mou = read_mouginot2019_greenland(MOUGINOT_PATH)
mou_year = df_mou['decimal_year'].values.astype(float)
mou_D = df_mou['discharge_rate'].values              # m/yr SLE, positive = SLR
mou_D_sigma = df_mou['discharge_rate_sigma'].values

# Mankoff (1986–2024): same convention
df_man = read_mankoff2021_greenland(MANKOFF_PATH, obs_only=True)
man_year = df_man['decimal_year'].values.astype(float)
man_D = df_man['discharge_rate'].values
man_D_sigma = df_man['discharge_rate_sigma'].values

# Also extract Mouginot/Mankoff SMB for comparison (these ARE RCM-dependent)
mou_SMB = df_mou['smb_rate'].values                  # m/yr SLE, negative = mass gain
mou_SMB_sigma = df_mou['smb_rate_sigma'].values
man_SMB = df_man['smb_rate'].values
man_SMB_sigma = df_man['smb_rate_sigma'].values

print(f'  Mouginot D: {mou_year[0]:.0f}–{mou_year[-1]:.0f} ({len(mou_year)} pts)')
print(f'  Mankoff D:  {man_year[0]:.0f}–{man_year[-1]:.0f} ({len(man_year)} pts)')

# =========================================================================
# 3. Compute implied SMB on overlapping annual grid
# =========================================================================
print('\n3. Computing implied SMB = IMBIE total − satellite discharge...')

# Build discharge on IMBIE annual grid by preferring Mouginot (1972–2018),
# extending with Mankoff (2019–2024)
discharge_lookup = {}
discharge_sigma_lookup = {}

# Mouginot first (primary)
for i, yr in enumerate(mou_year):
    yr_key = int(round(yr))
    discharge_lookup[yr_key] = mou_D[i]
    discharge_sigma_lookup[yr_key] = mou_D_sigma[i]

# Mankoff extends beyond Mouginot (2019+)
for i, yr in enumerate(man_year):
    yr_key = int(round(yr))
    if yr_key not in discharge_lookup:
        discharge_lookup[yr_key] = man_D[i]
        discharge_sigma_lookup[yr_key] = man_D_sigma[i]

# Also build RCM-based SMB lookup for comparison
smb_rcm_lookup = {}
smb_rcm_sigma_lookup = {}
for i, yr in enumerate(mou_year):
    yr_key = int(round(yr))
    smb_rcm_lookup[yr_key] = mou_SMB[i]
    smb_rcm_sigma_lookup[yr_key] = mou_SMB_sigma[i]
for i, yr in enumerate(man_year):
    yr_key = int(round(yr))
    if yr_key not in smb_rcm_lookup:
        smb_rcm_lookup[yr_key] = man_SMB[i]
        smb_rcm_sigma_lookup[yr_key] = man_SMB_sigma[i]

# Compute implied SMB where IMBIE and discharge overlap
results = []
for _, row in df_imbie_ann.iterrows():
    yr = int(round(row['year']))
    if yr not in discharge_lookup:
        continue

    total_slr = row['total_rate_slr']           # m/yr, positive = SLR
    total_sigma = row['total_rate_slr_sigma']
    D = discharge_lookup[yr]                    # m/yr, positive = SLR
    D_sigma = discharge_sigma_lookup[yr]

    # Implied SMB = total − discharge
    # If total = SMB + D, then SMB = total − D
    # In SLR convention: SMB is negative when accumulation > ablation
    implied_smb = total_slr - D
    implied_smb_sigma = np.sqrt(total_sigma**2 + D_sigma**2)

    entry = {
        'year': yr + 0.5,
        'total_rate_slr': total_slr,
        'total_rate_slr_sigma': total_sigma,
        'discharge_rate_slr': D,
        'discharge_rate_sigma_slr': D_sigma,
        'implied_smb_rate_slr': implied_smb,
        'implied_smb_rate_sigma_slr': implied_smb_sigma,
    }

    # Add RCM SMB for comparison if available
    if yr in smb_rcm_lookup:
        entry['rcm_smb_rate_slr'] = smb_rcm_lookup[yr]
        entry['rcm_smb_rate_sigma_slr'] = smb_rcm_sigma_lookup[yr]

    results.append(entry)

df_implied = pd.DataFrame(results)

print(f'  Overlap period: {df_implied["year"].iloc[0]:.0f}–{df_implied["year"].iloc[-1]:.0f} '
      f'({len(df_implied)} annual pts)')
print(f'  Implied SMB mean: {df_implied["implied_smb_rate_slr"].mean()*M_TO_MM:.3f} mm/yr '
      f'(negative = mass gain)')
print(f'  Discharge mean:   {df_implied["discharge_rate_slr"].mean()*M_TO_MM:.3f} mm/yr')
print(f'  Total mean:       {df_implied["total_rate_slr"].mean()*M_TO_MM:.3f} mm/yr')

# Compare against RCM SMB where available
if 'rcm_smb_rate_slr' in df_implied.columns:
    overlap = df_implied.dropna(subset=['rcm_smb_rate_slr'])
    diff = overlap['implied_smb_rate_slr'] - overlap['rcm_smb_rate_slr']
    print(f'\n  RCM SMB comparison ({len(overlap)} common years):')
    print(f'    Implied SMB mean: {overlap["implied_smb_rate_slr"].mean()*M_TO_MM:.3f} mm/yr')
    print(f'    RCM SMB mean:     {overlap["rcm_smb_rate_slr"].mean()*M_TO_MM:.3f} mm/yr')
    print(f'    Mean difference:  {diff.mean()*M_TO_MM:.3f} mm/yr')
    print(f'    Std difference:   {diff.std()*M_TO_MM:.3f} mm/yr')
    print(f'    Correlation:      {overlap["implied_smb_rate_slr"].corr(overlap["rcm_smb_rate_slr"]):.3f}')

# =========================================================================
# 4. Load Greenland surface temperature
# =========================================================================
print('\n4. Loading Greenland surface temperature...')

df_gr_temp = read_berkeley_earth_gridded(GRIDDED_T_PATH)
gr_t_monthly = df_gr_temp['decimal_year'].values
gr_T_monthly = df_gr_temp['temperature'].values

# Rebaseline to 1995–2005
bl_mask = (gr_t_monthly >= BASELINE_WINDOW[0]) & (gr_t_monthly < BASELINE_WINDOW[1] + 1)
gr_T_monthly = gr_T_monthly - np.nanmean(gr_T_monthly[bl_mask])

# Annual means
gr_yr_int = np.floor(gr_t_monthly).astype(int)
gr_unique_yrs = np.unique(gr_yr_int)
gr_T_annual = {}
for yr in gr_unique_yrs:
    mask = gr_yr_int == yr
    vals = gr_T_monthly[mask]
    valid = np.isfinite(vals)
    if valid.sum() >= 10:
        gr_T_annual[yr] = vals[valid].mean()

# Match temperature to implied SMB years
T_matched = []
for _, row in df_implied.iterrows():
    yr = int(round(row['year']))
    if yr in gr_T_annual:
        T_matched.append(gr_T_annual[yr])
    else:
        T_matched.append(np.nan)

df_implied['T_greenland'] = T_matched

# Also load GMST for comparison
df_gmst = pd.read_hdf(H5_PATH, 'harmonized/df_berkeley_h')
gmst_time = (df_gmst.index.year + (df_gmst.index.month - 0.5) / 12.0).values
gmst_vals = df_gmst['temperature'].values
gmst_yr_int = np.floor(gmst_time).astype(int)
gmst_unique = np.unique(gmst_yr_int)
gmst_annual = {}
for yr in gmst_unique:
    mask = gmst_yr_int == yr
    gmst_annual[yr] = gmst_vals[mask].mean()

T_gmst_matched = []
for _, row in df_implied.iterrows():
    yr = int(round(row['year']))
    if yr in gmst_annual:
        T_gmst_matched.append(gmst_annual[yr])
    else:
        T_gmst_matched.append(np.nan)

df_implied['T_gmst'] = T_gmst_matched

valid_T = df_implied.dropna(subset=['T_greenland', 'T_gmst'])
print(f'  Greenland T matched: {len(valid_T)} years')
print(f'  T_greenland range: [{valid_T["T_greenland"].min():.2f}, '
      f'{valid_T["T_greenland"].max():.2f}] °C')
print(f'  T_GMST range:      [{valid_T["T_gmst"].min():.2f}, '
      f'{valid_T["T_gmst"].max():.2f}] °C')

# =========================================================================
# 5. Regress implied SMB against temperature
# =========================================================================
print('\n5. Regressing implied SMB against temperature...')

# Convert implied SMB to Gt/yr for interpretability
valid_T = valid_T.copy()
valid_T['implied_smb_gt'] = valid_T['implied_smb_rate_slr'] / GT_TO_M_SLE  # back to Gt/yr
valid_T['implied_smb_sigma_gt'] = valid_T['implied_smb_rate_sigma_slr'] / GT_TO_M_SLE

ct_results = {}

for T_col, T_label in [('T_greenland', 'Greenland T'), ('T_gmst', 'GMST')]:
    T = valid_T[T_col].values
    SMB = valid_T['implied_smb_gt'].values   # Gt/yr, SLR convention (positive = mass loss)
    sigma = valid_T['implied_smb_sigma_gt'].values

    # WLS: SMB = C_T * T + C_0
    X = sm.add_constant(T)
    weights = 1.0 / sigma**2
    model = sm.WLS(SMB, X, weights=weights).fit()

    C_0 = model.params[0]
    C_T = model.params[1]
    C_0_se = model.bse[0]
    C_T_se = model.bse[1]
    r2 = model.rsquared

    # Also fit quadratic: SMB = C_TT * T^2 + C_T * T + C_0
    X_quad = np.column_stack([T**2, T, np.ones(len(T))])
    model_quad = sm.WLS(SMB, X_quad, weights=weights).fit()
    C_TT = model_quad.params[0]
    C_TT_se = model_quad.bse[0]
    r2_quad = model_quad.rsquared

    # BIC comparison
    n = len(T)
    bic_lin = n * np.log(model.ssr / n) + 2 * np.log(n)
    bic_quad = n * np.log(model_quad.ssr / n) + 3 * np.log(n)
    delta_bic = bic_lin - bic_quad

    print(f'\n  {T_label}:')
    print(f'    Linear:    C_T = {C_T:.1f} ± {C_T_se:.1f} Gt/yr/°C, '
          f'C_0 = {C_0:.1f} ± {C_0_se:.1f} Gt/yr, R² = {r2:.3f}')
    print(f'    Quadratic: C_TT = {C_TT:.1f} ± {C_TT_se:.1f} Gt/yr/°C², '
          f'R² = {r2_quad:.3f}, ΔBIC = {delta_bic:+.1f}')
    print(f'    In SLE:    C_T = {C_T * GT_TO_M_SLE * M_TO_MM:.3f} mm/yr/°C')

    ct_results[T_label] = {
        'C_T_gt_per_C': float(C_T),
        'C_T_se_gt_per_C': float(C_T_se),
        'C_0_gt': float(C_0),
        'C_0_se_gt': float(C_0_se),
        'C_T_mm_sle_per_C': float(C_T * GT_TO_M_SLE * M_TO_MM),
        'r2_linear': float(r2),
        'C_TT_gt_per_C2': float(C_TT),
        'C_TT_se_gt_per_C2': float(C_TT_se),
        'r2_quadratic': float(r2_quad),
        'delta_bic': float(delta_bic),
        'n_years': int(n),
        'year_range': [float(valid_T['year'].min()), float(valid_T['year'].max())],
        'T_range_C': [float(valid_T[T_col].min()), float(valid_T[T_col].max())],
    }

# Arctic amplification implied by the two regressions
if 'Greenland T' in ct_results and 'GMST' in ct_results:
    AA_implied = (ct_results['GMST']['C_T_gt_per_C']
                  / ct_results['Greenland T']['C_T_gt_per_C'])
    print(f'\n  Implied Arctic amplification (C_T ratio): {AA_implied:.2f}')
    ct_results['arctic_amplification_implied'] = float(AA_implied)

# =========================================================================
# 6. Save outputs
# =========================================================================
print('\n6. Saving outputs...')

csv_path = os.path.join(PROC_DIR, 'greenland_implied_smb.csv')
df_implied.to_csv(csv_path, index=False, float_format='%.6f')
print(f'  CSV: {csv_path} ({len(df_implied)} rows)')

json_path = os.path.join(PROC_DIR, 'greenland_ct_estimate.json')
export = {
    'description': 'Data-driven Greenland C_T from GRACE total minus satellite discharge',
    'method': 'implied_SMB = IMBIE_total - (Mouginot+Mankoff)_discharge; WLS regression against T',
    'sign_convention': 'positive C_T means warming increases mass loss (positive SLR)',
    'data_sources': {
        'total_mass_balance': 'IMBIE Greenland (Otosaka et al. 2023), GRACE-dominated post-2002',
        'discharge': 'Mouginot et al. (2019) primary + Mankoff et al. (2021) extension',
        'temperature': 'Berkeley Earth 1° gridded (Greenland landmass) and global mean',
    },
    'baseline': '1995-2005',
    'fits': ct_results,
}

with open(json_path, 'w') as f:
    json.dump(export, f, indent=2)
print(f'  JSON: {json_path}')

# =========================================================================
# Summary
# =========================================================================
print('\n' + '='*70)
print('GREENLAND IMPLIED SMB — SUMMARY')
print('='*70)
print(f'Period: {df_implied["year"].iloc[0]:.0f}–{df_implied["year"].iloc[-1]:.0f} '
      f'({len(df_implied)} years)')
print(f'\nData-driven C_T (linear, per °C Greenland T):')
ct_gr = ct_results['Greenland T']
print(f'  C_T = {ct_gr["C_T_gt_per_C"]:.1f} ± {ct_gr["C_T_se_gt_per_C"]:.1f} Gt/yr/°C')
print(f'      = {ct_gr["C_T_mm_sle_per_C"]:.3f} mm SLE/yr/°C')
print(f'  R² = {ct_gr["r2_linear"]:.3f}')
print(f'\nData-driven C_T (linear, per °C GMST):')
ct_gm = ct_results['GMST']
print(f'  C_T = {ct_gm["C_T_gt_per_C"]:.1f} ± {ct_gm["C_T_se_gt_per_C"]:.1f} Gt/yr/°C')
print(f'      = {ct_gm["C_T_mm_sle_per_C"]:.3f} mm SLE/yr/°C')
print(f'  R² = {ct_gm["r2_linear"]:.3f}')
print(f'\nLiterature comparison (per °C local Greenland T):')
print(f'  Hanna et al. (2021): 97–114 Gt/yr/°C (MAR, 1950–2019)')
print(f'  Fettweis et al. (2013): ~100–150 Gt/yr/°C (MAR runoff)')
print(f'  Our data-driven:     {ct_gr["C_T_gt_per_C"]:.0f} ± {ct_gr["C_T_se_gt_per_C"]:.0f} Gt/yr/°C')
print('='*70)
