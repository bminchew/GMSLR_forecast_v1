"""
ISMIP6 Antarctica Data Readers
==============================

Functions to read ISMIP6 Antarctic ice sheet model outputs from the
ComputedScalarsPaper archive (Seroussi et al., 2020, The Cryosphere).

Data structure
--------------
ComputedScalarsPaper/
  {group}/
    {model}/
      {experiment}/
        computed_{variable}_AIS_{group}_{model}_{experiment}.nc
        computed_{variable}_minus_ctrl_proj_AIS_{group}_{model}_{experiment}.nc

Variables
---------
- ivaf : ice volume above floatation [m^3]
- ivol : ice volume [m^3]
- icearea : total ice area [m^2]
- iareagr : grounded ice area [m^2]
- iareafl : floating ice area [m^2]
- smb : surface mass balance (integrated) [kg/s]
- smbgr : surface mass balance over grounded ice [kg/s]
- bmbfl : basal melt rate under floating ice [kg/s]

Each variable file contains:
- Total AIS value
- region_1 (WAIS), region_2 (EAIS), region_3 (Peninsula)
- sector_1 through sector_18
- rhoi, rhow (model-specific densities)

Dependencies
------------
- xarray
- pandas
- numpy

Reference
---------
Seroussi, H., Nowicki, S., Payne, A. J., et al.: ISMIP6 Antarctica: a
multi-model ensemble of the Antarctic ice sheet evolution over the 21st
century, The Cryosphere, 14, 3033-3070, 2020.
https://doi.org/10.5194/tc-14-3033-2020
"""

import os
import numpy as np
import pandas as pd
from typing import Optional, List

# Ocean area for m^3 ice → m SLE conversion
_OCEAN_AREA_M2 = 3.625e14  # 362.5 × 10^6 km^2 = 3.625 × 10^14 m^2

# Seconds per year (for kg/s → kg/yr)
_SECONDS_PER_YEAR = 365.25 * 24 * 3600

# Available scalar variables in each experiment directory
ISMIP6_VARIABLES = {
    'ivaf': 'ice volume above floatation',
    'ivol': 'ice volume',
    'icearea': 'total ice area',
    'iareagr': 'grounded ice area',
    'iareafl': 'floating ice area',
    'smb': 'surface mass balance (integrated)',
    'smbgr': 'surface mass balance over grounded ice',
    'bmbfl': 'basal melt rate under floating ice',
}

# ISMIP6 experiment descriptions (from Nowicki et al., 2020, Table 3)
ISMIP6_EXPERIMENTS = {
    'hist_std': 'Historical, standard open/NorESM1-M forcing (2005-2015)',
    'hist_open': 'Historical, open cavity (2005-2015)',
    'ctrl_proj_std': 'Control projection, standard (2015-2100)',
    'ctrl_proj_open': 'Control projection, open cavity (2015-2100)',
    'exp01': 'NorESM1-M RCP2.6, medium melt (open)',
    'exp02': 'NorESM1-M RCP2.6, high melt (open)',
    'exp03': 'NorESM1-M RCP8.5, medium melt (open)',
    'exp04': 'NorESM1-M RCP8.5, high melt (open)',
    'exp05': 'NorESM1-M RCP8.5, medium melt (std)',
    'exp06': 'MIROC-ESM-CHEM RCP8.5, medium melt (std)',
    'exp07': 'CCSM4 RCP8.5, medium melt (std)',
    'exp08': 'HadGEM2-ES RCP8.5, medium melt (std)',
    'exp09': 'CESM2 SSP5-8.5, medium melt (std)',
    'exp10': 'CNRM-CM6 SSP1-2.6, medium melt (std)',
    'exp11': 'CNRM-CM6 SSP5-8.5, medium melt (std)',
    'exp12': 'CNRM-ESM2 SSP5-8.5, medium melt (std)',
    'exp13': 'UKESM1-0-LL SSP5-8.5, medium melt (std)',
    'expA1': 'NorESM1-M RCP2.6, PIGL-medium melt (open)',
    'expA2': 'NorESM1-M RCP2.6, PIGL-high melt (open)',
    'expA3': 'NorESM1-M RCP8.5, PIGL-medium melt (open)',
    'expA4': 'NorESM1-M RCP8.5, PIGL-high melt (open)',
    'expA5': 'NorESM1-M RCP8.5, PIGL-medium melt (std)',
    'expA6': 'MIROC-ESM-CHEM RCP8.5, PIGL-medium melt (std)',
    'expA7': 'CCSM4 RCP8.5, PIGL-medium melt (std)',
    'expA8': 'HadGEM2-ES RCP8.5, PIGL-medium melt (std)',
}


# =========================================================================
# Internal helpers
# =========================================================================

def _build_nc_path(
    base_dir: str,
    group: str,
    model: str,
    experiment: str,
    variable: str,
    ctrl_subtracted: bool = False,
) -> str:
    """Build the full path to an ISMIP6 NetCDF file.

    Parameters
    ----------
    base_dir : str
        Path to the ``ComputedScalarsPaper/`` directory.
    group, model, experiment, variable : str
        Identifiers parsed from the directory hierarchy.
    ctrl_subtracted : bool
        If True, use the ``_minus_ctrl_proj`` variant.

    Returns
    -------
    str
        Absolute file path.
    """
    if ctrl_subtracted:
        fname = f'computed_{variable}_minus_ctrl_proj_AIS_{group}_{model}_{experiment}.nc'
    else:
        fname = f'computed_{variable}_AIS_{group}_{model}_{experiment}.nc'
    return os.path.join(base_dir, group, model, experiment, fname)


def _read_ismip6_nc(
    filepath: str,
    variable: str,
    region: Optional[str] = None,
) -> pd.DataFrame:
    """Read a single ISMIP6 scalar NetCDF file into a DataFrame.

    Parameters
    ----------
    filepath : str
        Path to the ``.nc`` file.
    variable : str
        Primary variable name (e.g. ``'ivaf'``, ``'smb'``).
    region : str, optional
        If ``None``, return all available columns (total + regions + sectors).
        If ``'total'``, return only the AIS-total column.
        If ``'WAIS'``, ``'EAIS'``, or ``'Peninsula'``, return the
        corresponding region column.

    Returns
    -------
    pd.DataFrame
        DataFrame indexed by ``decimal_year`` (float).
    """
    import xarray as xr

    ds = xr.open_dataset(filepath, decode_times=False)
    time = ds['time'].values.astype(float)

    # Determine which data vars to extract
    region_map = {
        'WAIS': f'{variable}_region_1',
        'EAIS': f'{variable}_region_2',
        'Peninsula': f'{variable}_region_3',
        'total': variable,
    }

    if region is not None:
        if region in region_map:
            col_name = region_map[region]
        else:
            raise ValueError(
                f"Unknown region '{region}'. "
                f"Choose from: {list(region_map.keys())}"
            )
        data = {region: ds[col_name].values}
    else:
        # All available: total + regions + sectors
        data = {}
        data['total'] = ds[variable].values
        for rname, col in region_map.items():
            if rname == 'total':
                continue
            if col in ds:
                data[rname] = ds[col].values
        # Sectors
        for s in range(1, 19):
            scol = f'{variable}_sector_{s}'
            if scol in ds:
                data[f'sector_{s}'] = ds[scol].values

    # Extract densities
    rhoi = float(ds['rhoi'].values) if 'rhoi' in ds else np.nan
    rhow = float(ds['rhow'].values) if 'rhow' in ds else np.nan

    ds.close()

    df = pd.DataFrame(data, index=time)
    df.index.name = 'decimal_year'

    # Parse metadata from filename
    fname = os.path.basename(filepath)
    native = _native_units(variable)
    df.attrs = {
        'dataset': 'ismip6_antarctica',
        'source_file': fname,
        'variable': variable,
        'rhoi_kg_m3': rhoi,
        'rhow_kg_m3': rhow,
        'native_units': native,
        'current_units': native,
        'reference': 'Seroussi et al. (2020)',
        'doi': '10.5194/tc-14-3033-2020',
    }

    return df


def _ivaf_to_sle(ivaf_m3: np.ndarray, rhoi: float) -> np.ndarray:
    """Convert ice volume above floatation [m^3] to sea-level equivalent [m].

    SLE = -(ivaf - ivaf[0]) * rhoi / rhow_ocean / ocean_area

    where rhow_ocean = 1028 kg/m^3 (standard).  Sign convention: positive
    SLE = sea level rise (ice loss).

    Parameters
    ----------
    ivaf_m3 : array
        Ice volume above floatation in m^3.
    rhoi : float
        Ice density in kg/m^3 (model-specific).

    Returns
    -------
    array
        Change in SLE (m) relative to first time step.
    """
    rhow_ocean = 1028.0  # standard ocean water density
    delta_ivaf = ivaf_m3 - ivaf_m3[0]
    return -delta_ivaf * rhoi / rhow_ocean / _OCEAN_AREA_M2


# =========================================================================
# Public reader functions
# =========================================================================

def list_ismip6_models(base_dir: str) -> pd.DataFrame:
    """List all available ISMIP6 group/model/experiment combinations.

    Parameters
    ----------
    base_dir : str
        Path to the ``ComputedScalarsPaper/`` directory.

    Returns
    -------
    pd.DataFrame
        Table with columns: ``group``, ``model``, ``experiment``,
        ``has_open_cavity``, ``n_variables``.
    """
    records = []
    for group in sorted(os.listdir(base_dir)):
        gpath = os.path.join(base_dir, group)
        if not os.path.isdir(gpath):
            continue
        for model in sorted(os.listdir(gpath)):
            mpath = os.path.join(gpath, model)
            if not os.path.isdir(mpath):
                continue
            for exp in sorted(os.listdir(mpath)):
                epath = os.path.join(mpath, exp)
                if not os.path.isdir(epath):
                    continue
                nc_files = [f for f in os.listdir(epath) if f.endswith('.nc')]
                n_vars = len([f for f in nc_files
                              if not 'minus_ctrl_proj' in f])
                has_open = 'open' in exp
                records.append({
                    'group': group,
                    'model': model,
                    'experiment': exp,
                    'has_open_cavity': has_open,
                    'n_variables': n_vars,
                    'description': ISMIP6_EXPERIMENTS.get(exp, ''),
                })

    return pd.DataFrame(records)


def read_ismip6_scalar(
    base_dir: str,
    group: str,
    model: str,
    experiment: str,
    variable: str = 'ivaf',
    ctrl_subtracted: bool = False,
    region: Optional[str] = None,
) -> pd.DataFrame:
    """Read a single ISMIP6 scalar variable for one group/model/experiment.

    Parameters
    ----------
    base_dir : str
        Path to ``ComputedScalarsPaper/``.
    group : str
        Modeling group (e.g. ``'AWI'``, ``'UCIJPL'``).
    model : str
        Ice sheet model (e.g. ``'PISM1'``, ``'ISSM'``).
    experiment : str
        Experiment ID (e.g. ``'exp05'``, ``'ctrl_proj_std'``).
    variable : str, default ``'ivaf'``
        Scalar variable to read.  One of: ``'ivaf'``, ``'ivol'``,
        ``'icearea'``, ``'iareagr'``, ``'iareafl'``, ``'smb'``,
        ``'smbgr'``, ``'bmbfl'``.
    ctrl_subtracted : bool, default False
        If True, read the ``_minus_ctrl_proj`` variant (control-run
        drift removed).  Only available for ``exp*`` experiments.
    region : str, optional
        ``None`` for all columns, ``'total'`` for AIS total,
        ``'WAIS'``, ``'EAIS'``, ``'Peninsula'`` for sub-regions.

    Returns
    -------
    pd.DataFrame
        Indexed by ``decimal_year``.  Units are native (m^3, m^2, or
        kg/s depending on the variable).

    Raises
    ------
    FileNotFoundError
        If the NetCDF file does not exist.
    ValueError
        If the variable or region is invalid.
    """
    if variable not in ISMIP6_VARIABLES:
        raise ValueError(
            f"Unknown variable '{variable}'. "
            f"Choose from: {sorted(ISMIP6_VARIABLES.keys())}"
        )

    filepath = _build_nc_path(
        base_dir, group, model, experiment, variable, ctrl_subtracted
    )
    if not os.path.isfile(filepath):
        raise FileNotFoundError(
            f"ISMIP6 file not found: {filepath}\n"
            f"Use list_ismip6_models() to see available combinations."
        )

    df = _read_ismip6_nc(filepath, variable, region=region)
    df.attrs.update({
        'group': group,
        'model': model,
        'experiment': experiment,
        'ctrl_subtracted': ctrl_subtracted,
    })
    return df


def read_ismip6_ivaf_sle(
    base_dir: str,
    group: str,
    model: str,
    experiment: str,
    ctrl_subtracted: bool = True,
    region: Optional[str] = None,
    units: str = 'm',
) -> pd.DataFrame:
    """Read ISMIP6 ice volume above floatation as sea-level equivalent.

    Convenience wrapper around :func:`read_ismip6_scalar` that converts
    ivaf [m^3] to SLE, referenced to the first time step.

    Parameters
    ----------
    base_dir : str
        Path to ``ComputedScalarsPaper/``.
    group, model, experiment : str
        Model identifiers.
    ctrl_subtracted : bool, default True
        Read the control-subtracted variant (recommended for projections).
    region : str, optional
        ``None`` for all columns, ``'total'``, ``'WAIS'``, ``'EAIS'``,
        ``'Peninsula'``.
    units : str, default ``'m'``
        Output units: ``'m'`` for meters or ``'mm'`` for millimeters.

    Returns
    -------
    pd.DataFrame
        SLE (positive = sea level rise).  Indexed by ``decimal_year``.
    """
    if units not in ('m', 'mm'):
        raise ValueError(f"units must be 'm' or 'mm', got '{units}'")

    df_raw = read_ismip6_scalar(
        base_dir, group, model, experiment,
        variable='ivaf', ctrl_subtracted=ctrl_subtracted, region=region,
    )
    rhoi = df_raw.attrs.get('rhoi_kg_m3', 910.0)

    df_sle = df_raw.copy()
    for col in df_sle.columns:
        df_sle[col] = _ivaf_to_sle(df_raw[col].values, rhoi)

    if units == 'mm':
        for col in df_sle.columns:
            df_sle[col] = df_sle[col] * 1000.0

    df_sle.attrs = df_raw.attrs.copy()
    df_sle.attrs['current_units'] = f'{units} SLE'
    df_sle.attrs['quantity'] = 'sea_level_equivalent'
    df_sle.attrs['sign_convention'] = 'positive = sea level rise'
    return df_sle


def read_ismip6_ivaf_gmsl(
    base_dir: str,
    group: str,
    model: str,
    experiment: str,
    ctrl_subtracted: bool = True,
    region: Optional[str] = None,
) -> pd.DataFrame:
    """Read ISMIP6 ivaf as GMSL contribution in mm.

    Alias for ``read_ismip6_ivaf_sle(..., units='mm')``.
    """
    return read_ismip6_ivaf_sle(
        base_dir, group, model, experiment,
        ctrl_subtracted=ctrl_subtracted, region=region, units='mm',
    )


def read_ismip6_ensemble_sle(
    base_dir: str,
    experiment: str = 'exp05',
    ctrl_subtracted: bool = True,
    region: Optional[str] = 'total',
    units: str = 'm',
) -> pd.DataFrame:
    """Read ivaf as SLE for ALL models that ran a given experiment.

    Parameters
    ----------
    base_dir : str
        Path to ``ComputedScalarsPaper/``.
    experiment : str, default ``'exp05'``
        Experiment ID.
    ctrl_subtracted : bool, default True
        Read the control-subtracted variant.
    region : str, optional
        Region to extract (default ``'total'``).
    units : str, default ``'m'``
        Output units: ``'m'`` for meters or ``'mm'`` for millimeters.

    Returns
    -------
    pd.DataFrame
        Columns named ``'{group}/{model}'``, indexed by ``decimal_year``.
        Units: m or mm SLE (positive = sea level rise).
    """
    if units not in ('m', 'mm'):
        raise ValueError(f"units must be 'm' or 'mm', got '{units}'")

    inventory = list_ismip6_models(base_dir)
    subset = inventory[inventory['experiment'] == experiment]

    results = {}
    for _, row in subset.iterrows():
        key = f"{row['group']}/{row['model']}"
        try:
            df = read_ismip6_ivaf_sle(
                base_dir, row['group'], row['model'], experiment,
                ctrl_subtracted=ctrl_subtracted, region=region,
                units=units,
            )
            col = df.columns[0] if len(df.columns) == 1 else 'total'
            if col in df.columns:
                results[key] = df[col]
            else:
                results[key] = df.iloc[:, 0]
        except (FileNotFoundError, KeyError):
            continue

    if not results:
        raise FileNotFoundError(
            f"No models found for experiment '{experiment}' in {base_dir}"
        )

    df_ens = pd.DataFrame(results)
    df_ens.index.name = 'decimal_year'
    df_ens.attrs = {
        'dataset': 'ismip6_antarctica_ensemble',
        'variable': 'ivaf',
        'experiment': experiment,
        'ctrl_subtracted': ctrl_subtracted,
        'region': region or 'all',
        'native_units': 'm^3',
        'current_units': f'{units} SLE',
        'sign_convention': 'positive = sea level rise',
        'n_models': len(results),
        'reference': 'Seroussi et al. (2020)',
        'doi': '10.5194/tc-14-3033-2020',
    }
    return df_ens


def read_ismip6_ensemble_gmsl(
    base_dir: str,
    experiment: str = 'exp05',
    ctrl_subtracted: bool = True,
    region: Optional[str] = 'total',
) -> pd.DataFrame:
    """Read ivaf as GMSL contribution in mm for ALL models that ran an experiment.

    Alias for ``read_ismip6_ensemble_sle(..., units='mm')``.
    """
    return read_ismip6_ensemble_sle(
        base_dir, experiment=experiment,
        ctrl_subtracted=ctrl_subtracted, region=region, units='mm',
    )


def read_ismip6_smb(
    base_dir: str,
    group: str,
    model: str,
    experiment: str,
    grounded_only: bool = False,
    ctrl_subtracted: bool = False,
    region: Optional[str] = None,
    convert_to_gt_per_yr: bool = True,
) -> pd.DataFrame:
    """Read ISMIP6 surface mass balance.

    Parameters
    ----------
    base_dir : str
        Path to ``ComputedScalarsPaper/``.
    group, model, experiment : str
        Model identifiers.
    grounded_only : bool, default False
        If True, read ``smbgr`` (grounded ice only) instead of ``smb``.
    ctrl_subtracted : bool, default False
        Read the control-subtracted variant.
    region : str, optional
        ``None``, ``'total'``, ``'WAIS'``, ``'EAIS'``, ``'Peninsula'``.
    convert_to_gt_per_yr : bool, default True
        Convert from native kg/s to Gt/yr.

    Returns
    -------
    pd.DataFrame
        Indexed by ``decimal_year``.
    """
    variable = 'smbgr' if grounded_only else 'smb'
    df = read_ismip6_scalar(
        base_dir, group, model, experiment,
        variable=variable, ctrl_subtracted=ctrl_subtracted, region=region,
    )

    if convert_to_gt_per_yr:
        for col in df.columns:
            df[col] = df[col] * _SECONDS_PER_YEAR / 1e12  # kg/s → Gt/yr
        df.attrs['current_units'] = 'Gt/yr'
    return df


def read_ismip6_basal_melt(
    base_dir: str,
    group: str,
    model: str,
    experiment: str,
    ctrl_subtracted: bool = False,
    region: Optional[str] = None,
    convert_to_gt_per_yr: bool = True,
) -> pd.DataFrame:
    """Read ISMIP6 basal melt rate under floating ice.

    Parameters
    ----------
    base_dir : str
        Path to ``ComputedScalarsPaper/``.
    group, model, experiment : str
        Model identifiers.
    ctrl_subtracted : bool, default False
        Read the control-subtracted variant.
    region : str, optional
        ``None``, ``'total'``, ``'WAIS'``, ``'EAIS'``, ``'Peninsula'``.
    convert_to_gt_per_yr : bool, default True
        Convert from native kg/s to Gt/yr.

    Returns
    -------
    pd.DataFrame
        Indexed by ``decimal_year``.  Native sign convention: negative
        values = melting.  If ``convert_to_gt_per_yr``, values are still
        negative for melt.
    """
    df = read_ismip6_scalar(
        base_dir, group, model, experiment,
        variable='bmbfl', ctrl_subtracted=ctrl_subtracted, region=region,
    )

    if convert_to_gt_per_yr:
        for col in df.columns:
            df[col] = df[col] * _SECONDS_PER_YEAR / 1e12
        df.attrs['current_units'] = 'Gt/yr'
    return df


def read_ismip6_area(
    base_dir: str,
    group: str,
    model: str,
    experiment: str,
    area_type: str = 'grounded',
    ctrl_subtracted: bool = False,
    region: Optional[str] = None,
    convert_to_km2: bool = True,
) -> pd.DataFrame:
    """Read ISMIP6 ice area (total, grounded, or floating).

    Parameters
    ----------
    base_dir : str
        Path to ``ComputedScalarsPaper/``.
    group, model, experiment : str
        Model identifiers.
    area_type : str, default ``'grounded'``
        One of ``'total'``, ``'grounded'``, ``'floating'``.
    ctrl_subtracted : bool, default False
        Read the control-subtracted variant.
    region : str, optional
        ``None``, ``'total'``, ``'WAIS'``, ``'EAIS'``, ``'Peninsula'``.
    convert_to_km2 : bool, default True
        Convert from m^2 to km^2.

    Returns
    -------
    pd.DataFrame
        Indexed by ``decimal_year``.
    """
    var_map = {'total': 'icearea', 'grounded': 'iareagr', 'floating': 'iareafl'}
    if area_type not in var_map:
        raise ValueError(
            f"Unknown area_type '{area_type}'. "
            f"Choose from: {list(var_map.keys())}"
        )

    df = read_ismip6_scalar(
        base_dir, group, model, experiment,
        variable=var_map[area_type], ctrl_subtracted=ctrl_subtracted,
        region=region,
    )

    if convert_to_km2:
        for col in df.columns:
            df[col] = df[col] / 1e6
        df.attrs['current_units'] = 'km^2'
    return df


# =========================================================================
# Utility
# =========================================================================

def _native_units(variable: str) -> str:
    """Return the native unit string for a given ISMIP6 variable."""
    units = {
        'ivaf': 'm^3',
        'ivol': 'm^3',
        'icearea': 'm^2',
        'iareagr': 'm^2',
        'iareafl': 'm^2',
        'smb': 'kg/s',
        'smbgr': 'kg/s',
        'bmbfl': 'kg/s',
    }
    return units.get(variable, 'unknown')
