"""
Sea Level Rise Data Readers
===========================

This module provides functions to read and parse global mean sea level (GMSL)
and global mean surface temperature (GMST) datasets into pandas DataFrames.

All DataFrames use a datetime index and include uncertainty columns where available.

Supported Datasets
------------------
GMSL:
    - Frederikse et al. (2020): Budget-closed reconstruction 1900-2018
    - Dangendorf et al. (2019, 2024): Kalman smoother reconstruction
    - Dangendorf et al. (2024) fields: Spatial reconstruction from MATLAB files
    - Horwath et al. (2022): ESA CCI sea level budget closure
    - NASA GSFC: Satellite altimetry (TOPEX/Jason/Sentinel-6)
    - IPCC AR6 FACTS: Projected GMSL by SSP scenario (medium confidence)

Temperature:
    - Berkeley Earth: Land/Ocean temperature
    - HadCRUT5: Met Office/UEA temperature
    - NASA GISTEMP: GISS temperature analysis
    - NOAA GlobalTemp: NCEI temperature
    - IPCC AR6: Projected GMST by SSP scenario

Dependencies
------------
- pandas
- numpy
- openpyxl (for Excel files)
- xarray, netCDF4 (for NetCDF files, Dangendorf only)
- h5py (for MATLAB v7.3 files, Dangendorf fields only)

Example
-------
>>> from slr_data_readers import read_frederikse2020, read_berkeley_earth
>>> df_sl = read_frederikse2020('data/frederikse2020.xlsx')
>>> df_temp = read_berkeley_earth('data/berkeley_earth.txt')

"""
import re
import os
import zipfile
import numpy as np
import pandas as pd

from datetime import datetime
from typing import Optional, Union


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def decimal_year_to_datetime(decimal_year: float) -> datetime:
    """
    Convert decimal year to datetime object.
    
    Parameters
    ----------
    decimal_year : float
        Year as decimal (e.g., 2020.5 = July 2, 2020)
        
    Returns
    -------
    datetime
        Corresponding datetime object
        
    Examples
    --------
    >>> decimal_year_to_datetime(2020.5)
    datetime.datetime(2020, 7, 2, 12, 0, 0)
    """
    year = int(decimal_year)
    remainder = decimal_year - year
    base = datetime(year, 1, 1)
    next_year = datetime(year + 1, 1, 1)
    return base + (next_year - base) * remainder


def datetime_to_decimal_year(dt: datetime) -> float:
    """
    Convert datetime to decimal year.
    
    Parameters
    ----------
    dt : datetime
        Datetime object
        
    Returns
    -------
    float
        Decimal year representation
        
    Examples
    --------
    >>> datetime_to_decimal_year(datetime(2020, 7, 1))
    2020.4986...
    """
    year = dt.year
    base = datetime(year, 1, 1)
    next_year = datetime(year + 1, 1, 1)
    return year + (dt - base).total_seconds() / (next_year - base).total_seconds()


# =============================================================================
# UNIT CONVERSION
# =============================================================================

# ---------------------------------------------------------------------------
# Unit registry — maps unit strings to their physical dimension and a
# conversion factor to the base unit of that dimension.
#
# Base units:  length → meters,  time → years
# Temperature is special (affine, not multiplicative) and handled separately.
# ---------------------------------------------------------------------------
_UNIT_REGISTRY = {
    # Length  (base = meters)
    'm':   {'dimension': 'length', 'to_base': 1.0},
    'mm':  {'dimension': 'length', 'to_base': 0.001},
    'cm':  {'dimension': 'length', 'to_base': 0.01},
    'ft':  {'dimension': 'length', 'to_base': 0.3048},
    'in':  {'dimension': 'length', 'to_base': 0.0254},
    # Temperature  (affine — factors handled in _convert_temperature)
    'degC': {'dimension': 'temperature'},
    'degF': {'dimension': 'temperature'},
    'K':    {'dimension': 'temperature'},
    # Time  (base = years)
    'yr':    {'dimension': 'time', 'to_base': 1.0},
    'day':   {'dimension': 'time', 'to_base': 1.0 / 365.25},
    'month': {'dimension': 'time', 'to_base': 1.0 / 12.0},
    's':     {'dimension': 'time', 'to_base': 1.0 / 31557600.0},
}

# Standard unit for each dimension  (target of convert_to_standard_units)
_STANDARD_UNITS = {'length': 'm', 'temperature': 'degC', 'time': 'yr'}


def _get_dimension(unit_str: str) -> str:
    """Return the physical dimension of a *simple* (non-compound) unit string.

    Parameters
    ----------
    unit_str : str
        A unit string such as ``'m'``, ``'degC'``, ``'yr'``.

    Returns
    -------
    str
        One of ``'length'``, ``'temperature'``, ``'time'``.

    Raises
    ------
    ValueError
        If *unit_str* is not in the registry.
    """
    entry = _UNIT_REGISTRY.get(unit_str)
    if entry is None:
        raise ValueError(
            f"Unknown unit '{unit_str}'. "
            f"Supported units: {sorted(_UNIT_REGISTRY.keys())}"
        )
    return entry['dimension']


def _is_delta_column(col_name: str) -> bool:
    """Heuristic: True if column represents a *difference* quantity.

    Uncertainty, sigma, and standard-error columns are deltas — when
    converting temperature they should use the scale factor only (no
    additive offset).
    """
    _delta_tokens = ('sigma', 'unc', '_se', '_sd', 'lower', 'upper',
                     '_p05', '_p10', '_p17', '_p25', '_p75', '_p83',
                     '_p90', '_p95', 'gmsl_05', 'gmsl_17', 'gmsl_83',
                     'gmsl_95')
    cl = col_name.lower()
    return any(tok in cl for tok in _delta_tokens)


def _convert_temperature(value, from_unit: str, to_unit: str,
                         is_delta: bool = False):
    """Convert temperature values between degC, degF, and K.

    Parameters
    ----------
    value : float or array-like
        Temperature value(s).
    from_unit, to_unit : str
        Source and target unit (``'degC'``, ``'degF'``, ``'K'``).
    is_delta : bool
        If True, treat *value* as a temperature *difference* (no offset).
    """
    if from_unit == to_unit:
        return value

    # Delta conversions — only scale, no offset
    if is_delta:
        _delta_scale = {
            ('degC', 'degF'): 9.0 / 5.0,
            ('degF', 'degC'): 5.0 / 9.0,
            ('degC', 'K'):    1.0,
            ('K', 'degC'):    1.0,
            ('degF', 'K'):    5.0 / 9.0,
            ('K', 'degF'):    9.0 / 5.0,
        }
        return value * _delta_scale[(from_unit, to_unit)]

    # Absolute conversions — first to degC, then to target
    if from_unit == 'degC':
        val_c = value
    elif from_unit == 'K':
        val_c = value - 273.15
    elif from_unit == 'degF':
        val_c = (value - 32.0) * 5.0 / 9.0
    else:
        raise ValueError(f"Unknown temperature unit: {from_unit}")

    if to_unit == 'degC':
        return val_c
    elif to_unit == 'K':
        return val_c + 273.15
    elif to_unit == 'degF':
        return val_c * 9.0 / 5.0 + 32.0
    else:
        raise ValueError(f"Unknown temperature unit: {to_unit}")


def _convert_simple(value, from_unit: str, to_unit: str,
                    is_delta: bool = False):
    """Convert *value* between two units of the same dimension.

    For length and time the conversion is purely multiplicative
    (via the base unit).  For temperature ``_convert_temperature``
    is used.

    Parameters
    ----------
    value : float or array-like
    from_unit, to_unit : str
    is_delta : bool
        Relevant only for temperature; ignored for other dimensions.
    """
    if from_unit == to_unit:
        return value

    dim_from = _get_dimension(from_unit)
    dim_to   = _get_dimension(to_unit)
    if dim_from != dim_to:
        raise ValueError(
            f"Cannot convert '{from_unit}' ({dim_from}) to "
            f"'{to_unit}' ({dim_to}): different dimensions."
        )

    if dim_from == 'temperature':
        return _convert_temperature(value, from_unit, to_unit,
                                    is_delta=is_delta)

    # Multiplicative dimensions (length, time): from → base → to
    factor = (_UNIT_REGISTRY[from_unit]['to_base']
              / _UNIT_REGISTRY[to_unit]['to_base'])
    return value * factor


def _standard_unit_for(unit_str: str) -> str:
    """Return the standard unit string for the dimension of *unit_str*.

    Handles compound (rate) units like ``'mm/yr'`` → ``'m/yr'``.
    """
    if '/' in unit_str:
        parts = unit_str.split('/')
        return '/'.join(_standard_unit_for(p) for p in parts)
    dim = _get_dimension(unit_str)
    return _STANDARD_UNITS[dim]


def _check_units_are_standard(current_units: dict) -> bool:
    """Return True if every unit in *current_units* matches the standard."""
    for col, unit in current_units.items():
        try:
            if unit != _standard_unit_for(unit):
                return False
        except ValueError:
            # Unknown unit — not standard
            return False
    return True


# ---- public API -----------------------------------------------------------

def convert_to_standard_units(
    data: Union[pd.DataFrame, dict],
    inplace: bool = False,
    verbose: bool = True,
) -> Union[pd.DataFrame, dict]:
    """
    Convert DataFrame(s) to standard units: meters, Celsius, years.

    Reads ``df.attrs['current_units']`` to determine what conversions are
    needed, then applies them.  After conversion the attrs are updated and
    ``df.attrs['units_standard']`` is set to ``True``.

    Standard units
    --------------
    - Length / sea level: meters (m)
    - Temperature: degrees Celsius (degC)
    - Time: years (yr)
    - Rates: m/yr, degC/yr, etc.

    Parameters
    ----------
    data : pd.DataFrame or dict of pd.DataFrame
        Data with ``attrs['current_units']`` metadata (set by reader
        functions).  If *dict* (from projection readers), each value
        DataFrame is converted independently.
    inplace : bool, default False
        If True, modify the DataFrame(s) in place.
    verbose : bool, default True
        Print a one-line summary per DataFrame showing what was converted.

    Returns
    -------
    pd.DataFrame or dict of pd.DataFrame
        Converted data with updated ``attrs['current_units']`` and
        ``attrs['units_standard'] = True``.

    Raises
    ------
    ValueError
        If *data* lacks the required ``attrs['current_units']`` metadata.
    """
    # --- dict path: recurse on each value ---
    if isinstance(data, dict):
        result = {}
        for key, df in data.items():
            result[key] = convert_to_standard_units(
                df, inplace=inplace, verbose=verbose,
            )
        return result

    if not isinstance(data, pd.DataFrame):
        raise TypeError(f"Expected DataFrame or dict, got {type(data)}")

    if 'current_units' not in data.attrs:
        raise ValueError(
            "DataFrame lacks attrs['current_units'] metadata. "
            "Use reader functions from slr_data_readers to load data."
        )

    if data.attrs.get('units_standard', False):
        if verbose:
            dataset = data.attrs.get('dataset', '?')
            print(f"  {dataset}: already in standard units, skipping")
        return data if inplace else data.copy()

    df = data if inplace else data.copy()
    units = df.attrs['current_units'].copy()
    conversions = []

    for col, unit in list(units.items()):
        std = _standard_unit_for(unit)
        if unit != std:
            is_delta = _is_delta_column(col)
            if '/' in unit:
                # Rate: convert numerator and denominator independently.
                # For X num/denom, numerator converts normally but
                # denominator factor is inverted (per-yr → per-day
                # means *dividing* by 365.25, not multiplying).
                from_parts = unit.split('/')
                to_parts   = std.split('/')
                factor = 1.0
                for idx, (fp, tp) in enumerate(zip(from_parts, to_parts)):
                    dim = _get_dimension(fp)
                    if dim == 'temperature':
                        f = _convert_temperature(
                            1.0, fp, tp, is_delta=True,
                        )
                    else:
                        f = (_UNIT_REGISTRY[fp]['to_base']
                             / _UNIT_REGISTRY[tp]['to_base'])
                    # Denominator parts (idx > 0) get inverted
                    factor *= f if idx == 0 else (1.0 / f)
                df[col] = df[col] * factor
            else:
                df[col] = _convert_simple(df[col], unit, std,
                                          is_delta=is_delta)
            conversions.append(f"{col}: {unit} -> {std}")
            units[col] = std

    # Update attrs
    df.attrs['current_units'] = units
    df.attrs['units_standard'] = True

    if verbose:
        dataset = df.attrs.get('dataset', 'unknown')
        if conversions:
            print(f"  {dataset}: {', '.join(conversions)}")
        else:
            print(f"  {dataset}: already in standard units")

    return df


def convert_units(
    data: Union[pd.DataFrame, dict],
    target_units: dict,
    inplace: bool = False,
    verbose: bool = True,
) -> Union[pd.DataFrame, dict]:
    """
    Convert DataFrame column(s) to user-specified units.

    Reads ``df.attrs['current_units']`` to determine what each column is
    currently in, converts the requested columns to the target units,
    and updates ``attrs['current_units']``.  The ``units_standard`` flag
    is set to ``True`` only if the resulting units match the standard
    (meters, Celsius, years); otherwise it is set to ``False``.

    Supported units
    ---------------
    - Length: ``'m'``, ``'mm'``, ``'cm'``, ``'ft'``, ``'in'``
    - Temperature: ``'degC'``, ``'degF'``, ``'K'``
    - Time: ``'yr'``, ``'day'``, ``'month'``, ``'s'``
    - Rates: any compound like ``'mm/yr'``, ``'ft/day'``, ``'m/s'``, etc.

    Parameters
    ----------
    data : pd.DataFrame or dict of pd.DataFrame
        Data with ``attrs['current_units']`` metadata.
    target_units : dict
        Mapping of column names to desired unit strings.  Use the
        special key ``'_all'`` to convert every column whose current
        unit is in the same physical dimension as the target.

        Examples::

            # Specific columns
            convert_units(df, {'gmsl': 'ft', 'steric': 'mm'})

            # All length columns to feet
            convert_units(df, {'_all': 'ft'})

            # Temperature to Fahrenheit
            convert_units(df, {'temperature': 'degF'})

            # Rate conversion
            convert_units(df, {'mass_balance_rate': 'mm/day'})
    inplace : bool, default False
        If True, modify the DataFrame(s) in place.
    verbose : bool, default True
        Print a one-line summary showing what was converted.

    Returns
    -------
    pd.DataFrame or dict of pd.DataFrame
        Converted data.  ``attrs['units_standard']`` is ``True`` if all
        units match standard units; ``False`` otherwise.

    Raises
    ------
    ValueError
        If a requested conversion crosses dimensions (e.g. degC → m),
        or if *data* lacks ``attrs['current_units']`` metadata.
    """
    # --- dict path: recurse on each value ---
    if isinstance(data, dict):
        result = {}
        for key, df in data.items():
            result[key] = convert_units(
                df, target_units=target_units,
                inplace=inplace, verbose=verbose,
            )
        return result

    if not isinstance(data, pd.DataFrame):
        raise TypeError(f"Expected DataFrame or dict, got {type(data)}")

    if 'current_units' not in data.attrs:
        raise ValueError(
            "DataFrame lacks attrs['current_units'] metadata. "
            "Use reader functions from slr_data_readers to load data."
        )

    df = data if inplace else data.copy()
    units = df.attrs['current_units'].copy()
    conversions = []

    # Expand the '_all' key: apply target to every column in the same
    # dimension as the target unit.
    col_targets = {}
    if '_all' in target_units:
        target_str = target_units['_all']
        # Determine the target dimension (handle compound units)
        if '/' in target_str:
            tgt_dim = _get_dimension(target_str.split('/')[0])
        else:
            tgt_dim = _get_dimension(target_str)

        for col, cur_unit in units.items():
            # Determine dimension of current column
            if '/' in cur_unit:
                cur_dim = _get_dimension(cur_unit.split('/')[0])
            else:
                cur_dim = _get_dimension(cur_unit)
            if cur_dim == tgt_dim:
                col_targets[col] = target_str

    # Explicit per-column targets override _all
    for col, tgt in target_units.items():
        if col != '_all':
            col_targets[col] = tgt

    for col, target_str in col_targets.items():
        if col not in units:
            continue  # column not in DataFrame — skip silently

        cur_unit = units[col]
        if cur_unit == target_str:
            continue  # already in target unit

        is_delta = _is_delta_column(col)

        if '/' in cur_unit or '/' in target_str:
            # --- rate / compound unit ---
            from_parts = cur_unit.split('/')
            to_parts   = target_str.split('/')

            if len(from_parts) != len(to_parts):
                raise ValueError(
                    f"Cannot convert '{cur_unit}' to '{target_str}': "
                    f"different compound structure."
                )

            factor = 1.0
            for idx, (fp, tp) in enumerate(zip(from_parts, to_parts)):
                dim_fp = _get_dimension(fp)
                dim_tp = _get_dimension(tp)
                if dim_fp != dim_tp:
                    raise ValueError(
                        f"Cannot convert '{cur_unit}' to '{target_str}': "
                        f"component '{fp}' ({dim_fp}) vs '{tp}' ({dim_tp})."
                    )
                if dim_fp == 'temperature':
                    f = _convert_temperature(
                        1.0, fp, tp, is_delta=True,
                    )
                else:
                    f = (_UNIT_REGISTRY[fp]['to_base']
                         / _UNIT_REGISTRY[tp]['to_base'])
                # Denominator parts (idx > 0) get inverted
                factor *= f if idx == 0 else (1.0 / f)

            df[col] = df[col] * factor

        else:
            # --- simple unit ---
            df[col] = _convert_simple(df[col], cur_unit, target_str,
                                      is_delta=is_delta)

        conversions.append(f"{col}: {cur_unit} -> {target_str}")
        units[col] = target_str

    # Update attrs
    df.attrs['current_units'] = units
    df.attrs['units_standard'] = _check_units_are_standard(units)

    if verbose:
        dataset = df.attrs.get('dataset', 'unknown')
        if conversions:
            print(f"  {dataset}: {', '.join(conversions)}")
        else:
            print(f"  {dataset}: no conversions needed")

    return df


# =============================================================================
# GMSL DATA READERS
# =============================================================================

def read_nasa_gmsl(filepath: str, convert_to_meters: bool = True) -> pd.DataFrame:
    """
    Read NASA GSFC GMSL from satellite altimetry.
    
    Data from merged TOPEX/Poseidon, Jason-1/2/3, and Sentinel-6 missions.
    
    Parameters
    ----------
    filepath : str
        Path to text file (e.g., nasa_GMSL_TPJAOS_5.2.txt)
    convert_to_meters : bool, default True
        If True, convert from mm to meters.
        
    Returns
    -------
    pd.DataFrame
        DataFrame with datetime index. All values in meters (if convert_to_meters=True)
        or mm (if False).
        
        Columns:
        - gmsl: Sea level with GIA correction
        - gmsl_sigma: Standard deviation
        - gmsl_smoothed: 60-day smoothed, seasonal removed
        - gmsl_nogia: Sea level without GIA
        - decimal_year: Original timestamp
        
    Note
    ----
    Reference: TOPEX/Jason 1996-2016 collinear mean (cycles 121-858).
    This reference period does not affect analysis since data will be
    re-referenced to a common baseline.
        
    Reference
    ---------
    Beckley, B. D., et al. (2017). On the "Cal-Mode" Correction to TOPEX 
    Satellite Altimetry. JGR: Oceans, 122(11), 8371-8384.
    https://doi.org/10.1002/2017JC013090
    
    Data: https://doi.org/10.5067/GMSLM-TJ152
    """
    with open(filepath, 'r') as f:
        lines = f.readlines()
    
    # Find header end
    data_start = 0
    for i, line in enumerate(lines):
        if 'Header_End' in line:
            data_start = i + 1
            break
        if not line.startswith('HDR') and line.strip():
            try:
                float(line.split()[0])
                data_start = i
                break
            except ValueError:
                pass
    
    # Parse data
    data = []
    for line in lines[data_start:]:
        parts = line.split()
        if len(parts) >= 12:
            try:
                row = {
                    'decimal_year': float(parts[2]),
                    'gmsl_nogia': float(parts[5]),
                    'gmsl': float(parts[8]),
                    'gmsl_sigma': float(parts[9]),
                    'gmsl_smoothed': float(parts[11]),
                }
                data.append(row)
            except (ValueError, IndexError):
                continue
    
    df = pd.DataFrame(data)
    df['time'] = pd.to_datetime([decimal_year_to_datetime(y) for y in df['decimal_year']])
    df = df.set_index('time')
    df.index.name = 'time'
    
    # Convert mm to meters if requested
    if convert_to_meters:
        for col in ['gmsl', 'gmsl_sigma', 'gmsl_smoothed', 'gmsl_nogia']:
            df[col] = df[col] / 1000.0

    # Attach unit metadata
    cu = 'm' if convert_to_meters else 'mm'
    _val_cols = ['gmsl', 'gmsl_sigma', 'gmsl_smoothed', 'gmsl_nogia']
    df.attrs = {
        'dataset': 'nasa_gmsl',
        'reference': 'Beckley et al. (2017)',
        'doi': None,
        'data_doi': '10.5067/GMSLM-TJ152',
        'native_units': {c: 'mm' for c in _val_cols},
        'current_units': {**{c: cu for c in _val_cols}, 'decimal_year': 'yr'},
        'units_standard': False,
        'quantity': 'sea_level',
        'native_time_resolution': 'sub-monthly',
    }

    return df


def read_frederikse2020(filepath: str, convert_to_meters: bool = True) -> pd.DataFrame:
    """
    Read Frederikse et al. (2020) global sea level budget data.
    
    Budget-closed reconstruction decomposed into thermosteric, glaciers,
    ice sheets, and terrestrial water storage from 1900-2018.
    
    Parameters
    ----------
    filepath : str
        Path to Excel file (frederikse2020_global_timeseries.xlsx)
    convert_to_meters : bool, default True
        If True, convert from mm to meters.
        
    Returns
    -------
    pd.DataFrame
        DataFrame with datetime index (annual, July 1).
        All values in meters (if convert_to_meters=True) or mm (if False).
        
        Columns (with *_sigma, *_lower, *_upper variants):
        - gmsl: Observed GMSL
        - sum_contributors: Sum of all contributors
        - steric: Thermosteric contribution
        - glaciers: Global glaciers
        - greenland: Greenland Ice Sheet
        - antarctica: Antarctic Ice Sheet (note: total, not subdivided)
        - tws: Terrestrial water storage
        - reservoir: Reservoir impoundment
        - groundwater: Groundwater depletion
        - tws_natural: Natural TWS variability
        - altimetry: Satellite altimetry (where available)
        
    Reference
    ---------
    Frederikse, T., et al. (2020). The causes of sea-level rise since 1900.
    Nature, 584(7821), 393-397.
    https://doi.org/10.1038/s41586-020-2591-3
    """
    df = pd.read_excel(filepath, sheet_name='Global')
    
    # First column is year
    df = df.rename(columns={'Unnamed: 0': 'year'})
    
    # Create datetime index
    df['time'] = pd.to_datetime(df['year'].astype(int).astype(str) + '-07-01')
    df = df.set_index('time')
    df.index.name = 'time'
    
    # Rename columns to standard conventions
    rename_map = {
        'Observed GMSL [mean]': 'gmsl',
        'Observed GMSL [lower]': 'gmsl_lower',
        'Observed GMSL [upper]': 'gmsl_upper',
        'Sum of contributors [mean]': 'sum_contributors',
        'Sum of contributors [lower]': 'sum_contributors_lower',
        'Sum of contributors [upper]': 'sum_contributors_upper',
        'Steric [mean]': 'steric',
        'Steric [lower]': 'steric_lower',
        'Steric [upper]': 'steric_upper',
        'Glaciers [mean]': 'glaciers',
        'Glaciers [lower]': 'glaciers_lower',
        'Glaciers [upper]': 'glaciers_upper',
        'Greenland Ice Sheet [mean]': 'greenland',
        'Greenland Ice Sheet [lower]': 'greenland_lower',
        'Greenland Ice Sheet [upper]': 'greenland_upper',
        'Antarctic Ice Sheet [mean]': 'antarctica',
        'Antarctic Ice Sheet [lower]': 'antarctica_lower',
        'Antarctic Ice Sheet [upper]': 'antarctica_upper',
        'Terrestrial Water Storage [mean]': 'tws',
        'Terrestrial Water Storage [lower]': 'tws_lower',
        'Terrestrial Water Storage [upper]': 'tws_upper',
        'Reservoir impoundment [mean]': 'reservoir',
        'Reservoir impoundment [lower]': 'reservoir_lower',
        'Reservoir impoundment [upper]': 'reservoir_upper',
        'Groundwater depletion [mean]': 'groundwater',
        'Groundwater depletion [lower]': 'groundwater_lower',
        'Groundwater depletion [upper]': 'groundwater_upper',
        'Natural TWS [mean]': 'tws_natural',
        'Natural TWS [lower]': 'tws_natural_lower',
        'Natural TWS [upper]': 'tws_natural_upper',
        'Altimetry [mean]': 'altimetry',
        'Altimetry [lower]': 'altimetry_lower',
        'Altimetry [upper]': 'altimetry_upper'
    }
    df = df.rename(columns=rename_map)
    
    # Calculate 1-sigma uncertainties from 90% CI bounds
    # Frederikse lower/upper are 5th/95th percentiles (90% CI),
    # so half-width = 1.645 * sigma for a normal distribution.
    for var in ['gmsl', 'steric', 'glaciers', 'greenland', 'antarctica', 'tws',
                'sum_contributors', 'reservoir', 'groundwater', 'tws_natural', 'altimetry']:
        if f'{var}_lower' in df.columns and f'{var}_upper' in df.columns:
            df[f'{var}_sigma'] = (df[f'{var}_upper'] - df[f'{var}_lower']) / (2 * 1.645)
    
    # Convert mm to meters if requested
    if convert_to_meters:
        value_cols = [c for c in df.columns if c not in ['year', 'decimal_year']]
        df[value_cols] = df[value_cols] / 1000.0

    # Attach unit metadata
    cu = 'm' if convert_to_meters else 'mm'
    _vcols = [c for c in df.columns if c not in ['year', 'decimal_year']]
    df.attrs = {
        'dataset': 'frederikse2020',
        'reference': 'Frederikse et al. (2020)',
        'doi': '10.1038/s41586-020-2591-3',
        'data_doi': '10.5281/zenodo.3862995',
        'native_units': {c: 'mm' for c in _vcols},
        'current_units': {**{c: cu for c in _vcols}, 'year': 'yr'},
        'units_standard': False,
        'quantity': 'sea_level',
        'native_time_resolution': 'annual',
    }

    return df


def read_dangendorf2024(filepath: str) -> pd.DataFrame:
    """
    Read Dangendorf et al. Kalman Smoother GMSL reconstruction.
    
    Hybrid tide gauge reconstruction using Kalman smoother methodology,
    providing total GMSL and its steric/barystatic decomposition.
    
    Parameters
    ----------
    filepath : str
        Path to NetCDF file (dangendorf2024_KalmanSmootherHR_Global.nc)
        
    Returns
    -------
    pd.DataFrame
        DataFrame with datetime index (annual, 1900-2021). 
        All values in meters.
        
        Columns (with *_sigma uncertainty variants):
        - gmsl: Global Mean Sea Level, hybrid reconstruction
        - steric: Global Mean Steric Sea Level
        - barystatic: Global Barystatic Sea Level
        - gia: GIA correction
        - decimal_year: Original time values
        
        Note: Some fields may be NaN if empty in source file.
        
    Note
    ----
    NetCDF structure (from ncdump -h):
        dimensions: x = 122, y = 1
        variables (all in meters):
            t(y, x)        - time in decimal years (1900-2021)
            GMSLHR(y, x)   - Global Mean Sea Level (Hybrid Reconstruction)
            GMSSLHR(y, x)  - Global Mean Steric Sea Level
            GBSLHR(y, x)   - Global Barystatic Sea Level  
            GMSLHRSE(y, x) - GMSL standard error
            GMSSLHRSE(y, x)- Steric standard error
            GBSLHRSE(y, x) - Barystatic standard error
            GGIAHR(y, x)   - GIA correction
            GGIAHRSE(y, x) - GIA standard error
    
    Requires: pip install xarray netCDF4
        
    Reference
    ---------
    Dangendorf, S., et al. (2019). Persistent acceleration in global sea-level
    rise since the 1960s. Nature Climate Change, 9(9), 705-710.
    https://doi.org/10.1038/s41558-019-0531-8

    Dangendorf, S. et al. (2024). Probabilistic reconstruction of sea-level changes
    and their causes since 1900. Earth System Science Data, 16, 3471–3494,
    https://doi.org/10.5194/essd-16-3471-2024.
    """
    try:
        import xarray as xr
    except ImportError:
        raise ImportError(
            "xarray is required to read this NetCDF4/HDF5 file.\n"
            "Install with: pip install xarray netCDF4"
        )
    
    ds = xr.open_dataset(filepath)
    
    def safe_extract(var_name):
        """Extract variable, replacing fill values with NaN."""
        if var_name not in ds:
            return np.full(len(time_decimal), np.nan)
        
        data = ds[var_name].values.flatten().astype(np.float64)
        
        # Check if array is empty
        if data.size == 0:
            return np.full(len(time_decimal), np.nan)
        
        # Replace unreasonably large values (NetCDF fill value ~9.97e+36) with NaN
        data[np.abs(data) > 1e20] = np.nan
        
        # Also check for explicit fill value attributes
        if hasattr(ds[var_name], '_FillValue'):
            fv = float(ds[var_name]._FillValue)
            data[np.isclose(data, fv, rtol=1e-5)] = np.nan
        
        return data
    
    # Extract time first (required)
    time_decimal = ds['t'].values.flatten()
    time_index = pd.to_datetime([decimal_year_to_datetime(t) for t in time_decimal])
    
    # Build dataframe, safely extracting each variable
    df = pd.DataFrame({
        'gmsl': safe_extract('GMSLHR'),
        'gmsl_sigma': safe_extract('GMSLHRSE'),
        'steric': safe_extract('GMSSLHR'),
        'steric_sigma': safe_extract('GMSSLHRSE'),
        'barystatic': safe_extract('GBSLHR'),
        'barystatic_sigma': safe_extract('GBSLHRSE'),
        'gia': safe_extract('GGIAHR'),
        'gia_sigma': safe_extract('GGIAHRSE'),
        'decimal_year': time_decimal
    }, index=time_index)
    
    df.index.name = 'time'
    ds.close()
    
    # Report which fields are empty
    empty_fields = [col for col in df.columns if col != 'decimal_year' and df[col].isna().all()]
    if empty_fields:
        print(f"Note: The following fields are empty in {filepath}: {empty_fields}")

    # Attach unit metadata
    _vcols = [c for c in df.columns if c != 'decimal_year']
    df.attrs = {
        'dataset': 'dangendorf2024',
        'reference': 'Dangendorf et al. (2024)',
        'doi': '10.5194/essd-16-3471-2024',
        'data_doi': '10.5281/zenodo.10621070',
        'native_units': {c: 'm' for c in _vcols},
        'current_units': {**{c: 'm' for c in _vcols}, 'decimal_year': 'yr'},
        'units_standard': False,
        'quantity': 'sea_level',
        'native_time_resolution': 'annual',
    }

    return df


def read_dangendorf2024_fields(
    data_dir: str,
    start_year: float = 1900,
) -> pd.DataFrame:
    """
    Read Dangendorf et al. (2024) Kalman Smoother sea level fields (MATLAB .mat).

    Loads the full spatial reconstruction from two MATLAB v7.3 (HDF5) files and
    computes latitude-weighted global means for GMSL and its decomposition into
    sterodynamic, barystatic, and GIA components. This replicates Section (1) of
    the companion script ``Master_Final.m``.

    Parameters
    ----------
    data_dir : str
        Path to directory containing ``KSSLfin.mat`` and ``KSSL_SEfin.mat``.
    start_year : float, default 1900
        Subset time to years >= start_year. The raw files cover 1880-2021;
        ``Master_Final.m`` uses 1900.

    Returns
    -------
    pd.DataFrame
        DataFrame with datetime index (annual, Jan 1). All values in meters.

        Columns (with *_sigma uncertainty variants):
        - gmsl: Geocentric GMSL (latitude-weighted mean of Hybrid
          Reconstruction, ``SL_multi``). Does not include GIA.
        - gmsl_relative: Relative GMSL (= gmsl + gia). Includes GIA
          correction and is directly comparable to tide gauge observations.
        - sterodynamic: Global mean sterodynamic sea level (``SDSL``)
        - barystatic: Global mean barystatic sea level (``BarySL``)
        - gia: Global mean GIA correction (``GIA_Field_KS``)
        - gmsl_sigma: GMSL standard error (pre-computed ``GMSLHRSE``)
        - sterodynamic_sigma: Sterodynamic standard error (``GMSSLHRSE``)
        - barystatic_sigma: Barystatic standard error (``GBSLHRSE``)
        - gia_sigma: GIA standard error (``GGIAHRwmSE * t``, linear in time,
          normalized to zero at endpoint)
        - decimal_year: Original time values

    Note
    ----
    - The field data (74,742 ocean grid cells at ~0.25° resolution) are
      reduced to global means via cosine-latitude weighting::

          GMSL(t) = Σ[ field(i,t) × cos(lat_i) ] / Σ[ cos(lat_i) ]

    - Component identity: gmsl = sterodynamic + barystatic + inverse_barometer
      (the IBA term is small and absorbed into gmsl but not output separately).
    - No re-referencing is applied. Use ``harmonize_baseline`` to set a common
      reference period.
    - ``read_dangendorf2024()`` reads the pre-computed global means from a
      NetCDF file; this function recomputes them from the spatial fields and
      additionally provides the sterodynamic/barystatic decomposition.

    Requires: pip install h5py

    Reference
    ---------
    Dangendorf, S., et al. (2019). Persistent acceleration in global sea-level
    rise since the 1960s. Nature Climate Change, 9(9), 705-710.
    https://doi.org/10.1038/s41558-019-0531-8

    Dangendorf, S. et al. (2024). Probabilistic reconstruction of sea-level
    changes and their causes since 1900. Earth System Science Data, 16,
    3471-3494. https://doi.org/10.5194/essd-16-3471-2024

    Data: https://doi.org/10.5281/zenodo.10621070
    """
    try:
        import h5py
    except ImportError:
        raise ImportError(
            "h5py is required to read MATLAB v7.3 (HDF5) files.\n"
            "Install with: pip install h5py"
        )

    import os

    path_kssl = os.path.join(data_dir, 'dangendorf2024_KSSLfin.mat')
    path_se = os.path.join(data_dir, 'dangendorf2024_KSSL_SEfin.mat') 

    f_kssl = h5py.File(path_kssl, 'r')
    f_se = h5py.File(path_se, 'r')

    try:
        # Time vector (full record 1880-2021, 142 years)
        tt = f_kssl['tt'][:].flatten()

        # Subset to start_year
        s = np.where(tt >= start_year)[0]
        t = tt[s]

        # Grid coordinates: LALT is (2, 74742) in HDF5 — row 0 = lon, row 1 = lat
        lat = f_kssl['LALT'][1, :]
        cos_lat = np.cos(np.deg2rad(lat))
        cos_lat_sum = np.sum(cos_lat)

        # Read spatial fields, subsetted in time
        # HDF5 layout: (time, station) — MATLAB transposes on save
        HR = f_kssl['SL_multi'][s, :]          # Hybrid Reconstruction (geocentric)
        SDSL = f_kssl['SDSL'][s, :]            # Sterodynamic
        BarySL = f_kssl['BarySL'][s, :]        # Barystatic
        GIA = f_kssl['GIA_Field_KS'][s, :]     # GIA field

        # Compute latitude-weighted global means
        def area_mean(field):
            return np.sum(field * cos_lat[np.newaxis, :], axis=1) / cos_lat_sum

        GMSLHR = area_mean(HR)
        GMSSLHR = area_mean(SDSL)
        GBSLHR = area_mean(BarySL)
        GGIAHR = area_mean(GIA)
        GMSLHRrel = GMSLHR + GGIAHR

        # Pre-computed standard errors (global means)
        GMSLHRSE = f_se['GMSLHRSE'][:].flatten()[s]
        GMSSLHRSE = f_se['GMSSLHRSE'][:].flatten()[s]
        GBSLHRSE = f_se['GBSLHRSE'][:].flatten()[s]

        # GIA SE: linear in time, normalized to zero at record end
        # Master_Final.m: GGIAHRSE = GGIAHRwmSE' .* tt; GGIAHRSE = GGIAHRSE - GGIAHRSE(end,:)
        GGIAHRwmSE = f_se['GGIAHRwmSE'][:].flatten()[0]
        GGIAHRSE_full = GGIAHRwmSE * tt
        GGIAHRSE_full = GGIAHRSE_full - GGIAHRSE_full[-1]
        GGIAHRSE = GGIAHRSE_full[s]

    finally:
        f_kssl.close()
        f_se.close()

    # Build datetime index
    time_index = pd.to_datetime([decimal_year_to_datetime(yr) for yr in t])

    df = pd.DataFrame({
        'gmsl': GMSLHR,
        'gmsl_relative': GMSLHRrel,
        'sterodynamic': GMSSLHR,
        'barystatic': GBSLHR,
        'gia': GGIAHR,
        'gmsl_sigma': GMSLHRSE,
        'sterodynamic_sigma': GMSSLHRSE,
        'barystatic_sigma': GBSLHRSE,
        'gia_sigma': GGIAHRSE,
        'decimal_year': t,
    }, index=time_index)

    df.index.name = 'time'

    # Attach unit metadata
    _vcols = [c for c in df.columns if c != 'decimal_year']
    df.attrs = {
        'dataset': 'dangendorf2024_fields',
        'reference': 'Dangendorf et al. (2024)',
        'doi': '10.5194/essd-16-3471-2024',
        'data_doi': '10.5281/zenodo.10621070',
        'native_units': {c: 'm' for c in _vcols},
        'current_units': {**{c: 'm' for c in _vcols}, 'decimal_year': 'yr'},
        'units_standard': False,
        'quantity': 'sea_level',
        'native_time_resolution': 'annual',
    }

    return df


def read_horwath2022(filepath: str, convert_to_meters: bool = True) -> pd.DataFrame:
    """
    Read Horwath et al. (2022) ESA CCI Sea Level Budget Closure data.
    
    GRACE/GRACE-FO derived mass contributions with budget closure analysis.
    
    Parameters
    ----------
    filepath : str
        Path to CSV file (horwath2021_ESACCI_SLBC_v2_2.csv)
    convert_to_meters : bool, default True
        If True, convert from mm to meters.
        
    Returns
    -------
    pd.DataFrame
        DataFrame with datetime index. All values in meters (if convert_to_meters=True)
        or mm sea level equivalent (if False).
        
        Columns (with *_sigma uncertainty variants):
        - gmsl: Global Mean Sea Level (65°N-65°S)
        - steric_slbc: Steric contribution (SLBC_cci product)
        - steric_deep: Deep ocean steric (not in steric_slbc, but in steric_dieng)
        - steric_dieng: Steric contribution (Dieng ensemble mean)
        - omc_global: Ocean mass contribution (global, GRACE)
        - omc_65: Ocean mass contribution (65°N-65°S, GRACE)
        - glaciers: Global glaciers (excl. Greenland/Antarctica peripherals)
        - greenland_altimetry: Greenland ice sheet (altimetry)
        - greenland_peripheral: Greenland peripheral glaciers
        - greenland: Greenland total (ice sheet + peripheral, altimetry-based)
        - greenland_grace: Greenland total (GRACE)
        - antarctica_altimetry: Antarctic ice sheet (altimetry)
        - antarctica_grace: Antarctic ice sheet (GRACE)
        - tws: Terrestrial water storage (WaterGAP model)
        - sum_mass_altimetry: Sum of mass contributions (altimetry-based ice sheets)
        - sum_mass_grace: Sum of mass contributions (GRACE-based ice sheets)
        
    Note
    ----
    - Reference period: 2006-01 to 2015-12 (anomalies relative to this mean)
    - Temporal coverage: 1993-01 to 2016-12 (monthly)
    - Complete coverage: 2003-01 to 2016-08
    - Missing values (-999.99) are converted to NaN
    - Seasonal components already removed
        
    Reference
    ---------
    Horwath, M., et al. (2022). Global sea-level budget and ocean-mass budget.
    Earth System Science Data, 14(2), 411-447.
    https://doi.org/10.5194/essd-14-411-2022
    """
    with open(filepath, 'r') as f:
        lines = f.readlines()
    
    # Find data section (BADC-CSV format has 'data' marker)
    data_start = None
    for i, line in enumerate(lines):
        if line.strip() == 'data':
            data_start = i + 2  # Skip 'data' line and column index line
            break
    
    if data_start is None:
        raise ValueError("Could not find 'data' marker in BADC-CSV file")
    
    # Column mapping from header long_name definitions (1-indexed in file)
    # Converting to 0-indexed for Python
    columns = {
        0: 'decimal_year',
        1: 'gmsl',                    # Global Mean Sea Level (65°N-65°S)
        2: 'gmsl_sigma',
        3: 'steric_slbc',             # Steric (SLBC_cci product)
        4: 'steric_slbc_sigma',
        5: 'steric_deep',             # Deep ocean steric
        6: 'steric_deep_sigma',
        7: 'steric_dieng',            # Steric (Dieng ensemble mean)
        8: 'steric_dieng_sigma',
        9: 'omc_global',              # Ocean mass (global, GRACE)
        10: 'omc_global_sigma',
        11: 'omc_65',                 # Ocean mass (65°N-65°S, GRACE)
        12: 'omc_65_sigma',
        13: 'glaciers',               # Global glaciers (excl. GrIS/AIS peripherals)
        14: 'glaciers_sigma',
        15: 'greenland_altimetry',    # Greenland ice sheet (altimetry)
        16: 'greenland_altimetry_sigma',
        17: 'greenland_peripheral',   # Greenland peripheral glaciers
        18: 'greenland_peripheral_sigma',
        19: 'greenland',              # Greenland total (altimetry + peripheral)
        20: 'greenland_sigma',
        21: 'greenland_grace',        # Greenland total (GRACE)
        22: 'greenland_grace_sigma',
        23: 'antarctica_altimetry',   # Antarctic ice sheet (altimetry)
        24: 'antarctica_altimetry_sigma',
        25: 'antarctica_grace',       # Antarctic ice sheet (GRACE)
        26: 'antarctica_grace_sigma',
        27: 'tws',                    # Terrestrial water storage
        28: 'tws_sigma',
        29: 'sum_mass_altimetry',     # Sum mass (altimetry ice sheets)
        30: 'sum_mass_altimetry_sigma',
        31: 'sum_mass_grace',         # Sum mass (GRACE ice sheets)
        32: 'sum_mass_grace_sigma',
        33: 'sum_steric_dieng_mass_alt',    # Steric(Dieng) + mass(altimetry)
        34: 'sum_steric_dieng_mass_alt_sigma',
        35: 'sum_steric_slbc_mass_alt',     # Steric(SLBC+deep) + mass(altimetry)
        36: 'sum_steric_slbc_mass_alt_sigma',
        37: 'sum_steric_dieng_omc_grace',   # Steric(Dieng) + OMC(GRACE)
        38: 'sum_steric_dieng_omc_grace_sigma',
        39: 'sum_steric_slbc_omc_grace',    # Steric(SLBC+deep) + OMC(GRACE)
        40: 'sum_steric_slbc_omc_grace_sigma',
    }
    
    # Parse data rows
    data = []
    for line in lines[data_start:]:
        parts = line.strip().split(',')
        if len(parts) > 1:
            try:
                row = {}
                for idx, name in columns.items():
                    if idx < len(parts):
                        val = float(parts[idx].strip())
                        # Convert missing values (-999.99) to NaN
                        row[name] = np.nan if val <= -999 else val
                data.append(row)
            except (ValueError, IndexError):
                continue
    
    df = pd.DataFrame(data)
    
    # Convert to datetime index
    df['time'] = pd.to_datetime([decimal_year_to_datetime(y) for y in df['decimal_year']])
    df = df.set_index('time')
    df.index.name = 'time'
    
    # Convert mm to meters if requested
    if convert_to_meters:
        value_cols = [c for c in df.columns if c != 'decimal_year']
        df[value_cols] = df[value_cols] / 1000.0

    # Attach unit metadata
    cu = 'm' if convert_to_meters else 'mm'
    _vcols = [c for c in df.columns if c != 'decimal_year']
    df.attrs = {
        'dataset': 'horwath2022',
        'reference': 'Horwath et al. (2022)',
        'doi': '10.5194/essd-14-411-2022',
        'data_doi': '10.5285/17c2ce31784048de93996275ee976fff',
        'native_units': {c: 'mm' for c in _vcols},
        'current_units': {**{c: cu for c in _vcols}, 'decimal_year': 'yr'},
        'units_standard': False,
        'quantity': 'sea_level',
        'native_time_resolution': 'monthly',
    }

    return df


def read_ipcc_ar6_observed_gmsl(filepath: str, convert_to_meters: bool = True) -> pd.DataFrame:
    """
    Read IPCC AR6 observed global mean sea level data.
    
    Parameters
    ----------
    filepath : str
        Path to CSV file (global_sea_level_observed.csv)
        
    Returns
    -------
    pd.DataFrame
        DataFrame with datetime index (annual, July 1).
        All values in meters.
        
        Columns:
        - gmsl: Central estimate
        - gmsl_lower: 17th percentile (lower bound of 66% CI)
        - gmsl_upper: 83rd percentile (upper bound of 66% CI)
        - gmsl_sigma: Approximate 1-sigma uncertainty
        
    Note
    ----
    The 17%–83% range corresponds to the IPCC "likely" range (66% CI).
    gmsl_sigma is approximated as half this range, which is roughly 1-sigma
    for a normal distribution.
        
    Reference
    ---------
    IPCC, 2021: Climate Change 2021: The Physical Science Basis. 
    Contribution of Working Group I to the Sixth Assessment Report.
    Chapter 9: Ocean, Cryosphere and Sea Level Change.
    https://www.ipcc.ch/report/ar6/wg1/
    
    Fox-Kemper, B., et al. (2021). Ocean, Cryosphere and Sea Level Change.
    In Climate Change 2021: The Physical Science Basis (pp. 1211–1362).
    Cambridge University Press. https://doi.org/10.1017/9781009157896.011
    """
    if not convert_to_meters:
        raise ValueError('IPCC AR6 Observed GMSLR data are in meters by default')
        
    df = pd.read_csv(filepath, encoding='utf-8-sig')  # Handle BOM
    
    # Rename columns
    df = df.rename(columns={
        'Year': 'year',
        'Central': 'gmsl',
        '17%': 'gmsl_lower',
        '83%': 'gmsl_upper'
    })
    
    # Create datetime index (annual data at mid-year)
    df['time'] = pd.to_datetime(df['year'].astype(int).astype(str) + '-07-01')
    df = df.set_index('time')
    df.index.name = 'time'
    
    # Calculate approximate 1-sigma uncertainty
    # 17%-83% is ~1 sigma on each side for normal distribution
    df['gmsl_sigma'] = (df['gmsl_upper'] - df['gmsl_lower']) / 2
    
    # Add decimal year for convenience
    df['decimal_year'] = df['year'].astype(float) + 0.5

    # Attach unit metadata
    _sl_cols = ['gmsl', 'gmsl_lower', 'gmsl_upper', 'gmsl_sigma']
    df.attrs = {
        'dataset': 'ipcc_ar6_observed_gmsl',
        'reference': 'Fox-Kemper et al. (2021)',
        'doi': '10.1017/9781009157896.011',
        'data_doi': None,
        'native_units': {c: 'm' for c in _sl_cols},
        'current_units': {
            **{c: 'm' for c in _sl_cols},
            'year': 'yr', 'decimal_year': 'yr',
        },
        'units_standard': False,
        'quantity': 'sea_level',
        'native_time_resolution': 'annual',
    }

    return df


def read_imbie_west_antarctica(filepath: str, convert_to_meters: bool = True) -> pd.DataFrame:
    """
    Read IMBIE West Antarctica ice sheet mass balance data.

    Ice Mass Balance Inter-comparison Exercise (IMBIE) reconciled estimates
    of West Antarctic Ice Sheet mass balance from satellite observations.

    Parameters
    ----------
    filepath : str
        Path to CSV file (imbie_west_antarctica_2021_mm.csv)
    sl_unit : str, default 'meters'
        Output unit for sea level: 'meters', 'm', 'mm', 'cm'

    Returns
    -------
    pd.DataFrame
        DataFrame with datetime index (monthly, 1992-2020).

        Columns:
        - mass_balance_rate: Mass balance rate (sea level equivalent per year)
        - mass_balance_rate_sigma: Rate uncertainty
        - cumulative_mass_balance: Cumulative mass balance (sea level equivalent)
        - cumulative_mass_balance_sigma: Cumulative uncertainty
        - decimal_year: Original time values

    Source Units
    ------------
    File units: mm and mm/yr (sea level equivalent, confirmed from filename and header)

    Note
    ----
    - Temporal coverage: 1992-01 to 2020-12 (monthly)
    - Data represents West Antarctica only (not total Antarctic Ice Sheet)
    - Positive values = sea level rise contribution
    - Rate units scale with sl_unit (e.g., meters/yr if sl_unit='meters')

    Reference
    ---------
    IMBIE Team (2018). Mass balance of the Antarctic Ice Sheet from 1992 to 2017.
    Nature, 558(7709), 219-222.
    https://doi.org/10.1038/s41586-018-0179-y

    Otosaka et al. (2023) Mass balance of the Greenland and Antarctic ice sheets
    from 1992 to 2020. Earth Syst. Sci. Data, 15, 1597–1616.
    https://doi.org/10.5194/essd-15-1597-2023

    Data: https://imbie.org/data-downloads/
    """
    df = pd.read_csv(filepath)

    # Rename columns to standard conventions
    df = df.rename(columns={
        'Year': 'decimal_year',
        'Mass balance (mm/yr)': 'mass_balance_rate',
        'Mass balance uncertainty (mm/yr)': 'mass_balance_rate_sigma',
        'Cumulative mass balance (mm)': 'cumulative_mass_balance',
        'Cumulative mass balance uncertainty (mm)': 'cumulative_mass_balance_sigma'
    })

    # Create datetime index
    df['time'] = pd.to_datetime([decimal_year_to_datetime(y) for y in df['decimal_year']])
    df = df.set_index('time')
    df.index.name = 'time'

    # Convert from mm (source) to meters
    if convert_to_meters:
        value_cols = [c for c in df.columns if c != 'decimal_year']
        df[value_cols] = df[value_cols] / 1000.0

    # Attach unit metadata — mixed units: rates vs cumulative
    _len = 'm' if convert_to_meters else 'mm'
    _rate = f'{_len}/yr'
    df.attrs = {
        'dataset': 'imbie_west_antarctica',
        'reference': 'Otosaka et al. (2023)',
        'doi': '10.5194/essd-15-1597-2023',
        'data_doi': '10.5285/77B64C55-7166-4A06-9DEF-2E400398E452',
        'native_units': {
            'mass_balance_rate': 'mm/yr', 'mass_balance_rate_sigma': 'mm/yr',
            'cumulative_mass_balance': 'mm', 'cumulative_mass_balance_sigma': 'mm',
        },
        'current_units': {
            'mass_balance_rate': _rate, 'mass_balance_rate_sigma': _rate,
            'cumulative_mass_balance': _len, 'cumulative_mass_balance_sigma': _len,
            'decimal_year': 'yr',
        },
        'units_standard': False,
        'quantity': 'sea_level',
        'native_time_resolution': 'monthly',
    }

    return df

# =============================================================================
# TEMPERATURE DATA READERS
# =============================================================================

def read_berkeley_earth(filepath: str) -> pd.DataFrame:
    """
    Read Berkeley Earth global mean surface temperature.
    
    Parameters
    ----------
    filepath : str
        Path to text file (berkEarth_globalmean_airTaboveseaice.txt)
        
    Returns
    -------
    pd.DataFrame
        DataFrame with datetime index and columns:
        - temperature: Anomaly (°C, relative to 1951-1980)
        - temperature_unc: Uncertainty (°C, 95% CI)
        
    Reference
    ---------
    Rohde, R. A., & Hausfather, Z. (2020). The Berkeley Earth Land/Ocean
    Temperature Record. ESSD, 12(4), 3469-3479.
    https://doi.org/10.5194/essd-12-3469-2020
    """
    with open(filepath, 'r') as f:
        lines = f.readlines()
    
    # Find data start
    data_start = 0
    for i, line in enumerate(lines):
        if line.strip() and not line.strip().startswith('%'):
            data_start = i
            break
    
    data = []
    for line in lines[data_start:]:
        stripped = line.strip()
        if stripped and not stripped.startswith('%'):
            parts = stripped.split()
            if len(parts) >= 4:
                try:
                    data.append({
                        'year': int(parts[0]),
                        'month': int(parts[1]),
                        'temperature': float(parts[2]),
                        'temperature_unc': float(parts[3]),
                    })
                except ValueError:
                    continue
    
    df = pd.DataFrame(data)
    df['time'] = pd.to_datetime(df['year'].astype(str) + '-' + df['month'].astype(str) + '-01')
    df = df.set_index('time')[['temperature', 'temperature_unc']]
    df.index.name = 'time'

    df['temperature_sigma'] = df['temperature_unc']/1.645

    # Attach unit metadata
    df.attrs = {
        'dataset': 'berkeley_earth',
        'reference': 'Rohde & Hausfather (2020)',
        'doi': '10.5194/essd-12-3469-2020',
        'data_doi': '10.5281/zenodo.3634713',
        'native_units': {
            'temperature': 'degC', 'temperature_unc': 'degC',
            'temperature_sigma': 'degC',
        },
        'current_units': {
            'temperature': 'degC', 'temperature_unc': 'degC',
            'temperature_sigma': 'degC',
        },
        'units_standard': False,
        'quantity': 'temperature',
        'temperature_baseline': '1951-1980',
        'native_time_resolution': 'monthly',
    }

    return df


def read_hadcrut5(filepath: str) -> pd.DataFrame:
    """
    Read HadCRUT5 global mean surface temperature.
    
    Parameters
    ----------
    filepath : str
        Path to CSV file (hadcrut5_global_monthly.csv)
        
    Returns
    -------
    pd.DataFrame
        DataFrame with datetime index and columns:
        - temperature: Anomaly (°C, relative to 1961-1990)
        - temperature_unc: Symmetric uncertainty
        - temperature_lower: Lower 2.5% bound
        - temperature_upper: Upper 97.5% bound
        
    Reference
    ---------
    Morice, C. P., et al. (2021). An Updated Assessment of Near-Surface
    Temperature Change From 1850: The HadCRUT5 Data Set.
    JGR: Atmospheres, 126(3), e2019JD032361.
    https://doi.org/10.1029/2019JD032361
    """
    df = pd.read_csv(filepath)
    df['time'] = pd.to_datetime(df['Time'] + '-01')
    df = df.set_index('time')
    
    df = df.rename(columns={
        'Anomaly (deg C)': 'temperature',
        'Lower confidence limit (2.5%)': 'temperature_lower',
        'Upper confidence limit (97.5%)': 'temperature_upper'
    })
    # temperature_unc = half-width of the 95% CI (2.5%–97.5% bounds)
    df['temperature_unc'] = (df['temperature_upper'] - df['temperature_lower']) / 2
    # Convert to 1-sigma: 95% CI half-width = 1.96 * sigma
    df['temperature_sigma'] = df['temperature_unc'] / 1.96

    df.index.name = 'time'

    # Attach unit metadata
    _tcols = ['temperature', 'temperature_unc', 'temperature_sigma',
              'temperature_lower', 'temperature_upper']
    df.attrs = {
        'dataset': 'hadcrut5',
        'reference': 'Morice et al. (2021)',
        'doi': '10.1029/2019JD032361',
        'data_doi': None,
        'native_units': {c: 'degC' for c in _tcols},
        'current_units': {c: 'degC' for c in _tcols},
        'units_standard': False,
        'quantity': 'temperature',
        'temperature_baseline': '1961-1990',
        'native_time_resolution': 'monthly',
    }

    return df[['temperature', 'temperature_unc', 'temperature_sigma',
               'temperature_lower', 'temperature_upper']]


def read_nasa_gistemp(filepath: str) -> pd.DataFrame:
    """
    Read NASA GISTEMP global mean surface temperature.
    
    Parameters
    ----------
    filepath : str
        Path to CSV file (nasa_GLB_Ts_dSST.csv)
        
    Returns
    -------
    pd.DataFrame
        DataFrame with datetime index and columns:
        - temperature: Anomaly (°C, relative to 1951-1980)
        
    Reference
    ---------
    Lenssen, N. J. L., et al. (2019). Improvements in the GISTEMP 
    Uncertainty Model. JGR: Atmospheres, 124(12), 6307-6326.
    https://doi.org/10.1029/2018JD029522
    """
    df = pd.read_csv(filepath, skiprows=1)
    
    months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
              'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    
    df_long = df.melt(id_vars=['Year'], value_vars=months,
                       var_name='month', value_name='temperature')
    
    month_map = {m: i+1 for i, m in enumerate(months)}
    df_long['month_num'] = df_long['month'].map(month_map)
    df_long['time'] = pd.to_datetime(
        df_long['Year'].astype(str) + '-' + df_long['month_num'].astype(str) + '-01'
    )
    df_long['temperature'] = pd.to_numeric(df_long['temperature'], errors='coerce')
    
    df_long = df_long.set_index('time').sort_index()[['temperature']]
    df_long.index.name = 'time'

    # Attach unit metadata
    df_long.attrs = {
        'dataset': 'nasa_gistemp',
        'reference': 'Lenssen et al. (2019)',
        'doi': '10.1029/2018JD029522',
        'data_doi': None,
        'native_units': {'temperature': 'degC'},
        'current_units': {'temperature': 'degC'},
        'units_standard': False,
        'quantity': 'temperature',
        'temperature_baseline': '1951-1980',
        'native_time_resolution': 'monthly',
    }

    return df_long


def read_noaa_globaltemp(filepath: str) -> pd.DataFrame:
    """
    Read NOAA Global Temperature Anomalies.
    
    Parameters
    ----------
    filepath : str
        Path to text file (noaa_atmoT_landocean.txt)
        
    Returns
    -------
    pd.DataFrame
        DataFrame with datetime index (annual, July 1) and columns:
        - temperature: Anomaly (°C, relative to 1901-2000)
        
    Note
    ----
    This dataset is annual resolution only.
        
    Reference
    ---------
    Vose, R. S., et al. (2012). NOAA's Merged Land-Ocean Surface 
    Temperature Analysis. BAMS, 93(11), 1677-1685.
    https://doi.org/10.1175/BAMS-D-11-00241.1
    """
    with open(filepath, 'r') as f:
        lines = f.readlines()
    
    data = []
    for line in lines:
        stripped = line.strip()
        if stripped and not stripped.startswith('#'):
            parts = stripped.split(',')
            if len(parts) >= 2:
                try:
                    data.append({
                        'year': int(parts[0]),
                        'temperature': float(parts[1])
                    })
                except ValueError:
                    continue
    
    df = pd.DataFrame(data)
    df['time'] = pd.to_datetime(df['year'].astype(str) + '-07-01')
    df = df.set_index('time')[['temperature']]
    df.index.name = 'time'

    # Attach unit metadata
    df.attrs = {
        'dataset': 'noaa_globaltemp',
        'reference': 'Vose et al. (2021)',
        'doi': '10.1029/2020GL090873',
        'data_doi': '10.25921/2tj4-0e21',
        'native_units': {'temperature': 'degC'},
        'current_units': {'temperature': 'degC'},
        'units_standard': False,
        'quantity': 'temperature',
        'temperature_baseline': '1901-2000',
        'native_time_resolution': 'annual',
    }

    return df


# =============================================================================
# PROJECTION READERS
# =============================================================================


def read_ipcc_ar6_projected_temperature(
    data_dir: str,
) -> dict:
    """
    Read IPCC AR6 projected global mean surface temperature for all SSP scenarios.

    Reads CSV files from the IPCC AR6 atmospheric temperature projection directory.
    Each file contains annual GMST projections with 5th/95th percentile uncertainty
    bounds for a single SSP scenario or the historical period.

    Parameters
    ----------
    data_dir : str
        Path to directory containing the CSV files (e.g.,
        'data/raw/ipcc_ar6/atmo_temp/'). Expected files follow the naming
        convention ``tas_global_<scenario>.csv`` with columns:
        Year, 5%, Mean, 95%.

    Returns
    -------
    dict of pd.DataFrame
        Dictionary keyed by scenario name (e.g., 'Historical', 'SSP1_1_9',
        'SSP1_2_6', 'SSP2_4_5', 'SSP3_7_0', 'SSP5_8_5'). Each DataFrame
        has a datetime index (annual, Jan 1) and columns:

        - temperature: Mean projected GMST anomaly (°C)
        - temperature_lower: 5th percentile (°C)
        - temperature_upper: 95th percentile (°C)
        - temperature_sigma: Approximate 1-sigma uncertainty (°C)
        - decimal_year: Original year values

    Note
    ----
    - Temperature anomalies are relative to 1850-1900 pre-industrial baseline.
    - The 5%–95% range spans the IPCC "very likely" range (90% CI).
      temperature_sigma is approximated as half this range divided by 1.645
      (the z-score for 90% CI), giving approximate 1-sigma for a normal
      distribution.
    - Historical period covers 1950-2014; SSP projections cover 2015-2099.

    Reference
    ---------
    IPCC, 2021: Climate Change 2021: The Physical Science Basis.
    Contribution of Working Group I to the Sixth Assessment Report.
    Chapter 4: Future Global Climate.
    https://www.ipcc.ch/report/ar6/wg1/

    Lee, J.-Y., et al. (2021). Future Global Climate: Scenario-Based
    Projections and Near-Term Information. In Climate Change 2021: The
    Physical Science Basis (pp. 553-672). Cambridge University Press.
    https://doi.org/10.1017/9781009157896.006
    """
    import glob
    import os

    csv_files = sorted(glob.glob(os.path.join(data_dir, 'tas_global_*.csv')))
    if not csv_files:
        raise FileNotFoundError(
            f"No tas_global_*.csv files found in {data_dir}"
        )

    result = {}
    for fpath in csv_files:
        # Extract scenario name from filename (e.g., 'SSP2_4_5' or 'Historical')
        basename = os.path.basename(fpath)
        scenario = basename.replace('tas_global_', '').replace('.csv', '')

        df = pd.read_csv(fpath)

        # Rename columns to standard conventions
        df = df.rename(columns={
            'Year': 'decimal_year',
            '5%': 'temperature_lower',
            'Mean': 'temperature',
            '95%': 'temperature_upper',
        })

        # Create datetime index
        df['time'] = pd.to_datetime(
            [decimal_year_to_datetime(y) for y in df['decimal_year']]
        )
        df = df.set_index('time')
        df.index.name = 'time'

        # Approximate 1-sigma from 90% CI (5%-95%)
        # z-score for 90% CI = 1.645
        df['temperature_sigma'] = (
            (df['temperature_upper'] - df['temperature_lower']) / (2 * 1.645)
        )

        # Attach unit metadata to each scenario DataFrame
        _tcols = ['temperature', 'temperature_lower', 'temperature_upper',
                  'temperature_sigma']
        df.attrs = {
            'dataset': 'ipcc_ar6_projected_temperature',
            'scenario': scenario,
            'reference': 'Lee et al. (2021)',
            'doi': '10.1017/9781009157896.006',
            'data_doi': None,
            'native_units': {c: 'degC' for c in _tcols},
            'current_units': {**{c: 'degC' for c in _tcols}, 'decimal_year': 'yr'},
            'units_standard': False,
            'quantity': 'temperature',
            'temperature_baseline': '1850-1900',
            'native_time_resolution': 'annual',
        }

        result[scenario] = df

    return result


def _read_ipcc_ar6_projected_gmsl_impl(
    data_dir: str,
    confidence: str,
    convert_to_meters: bool = True,
) -> dict:
    """Internal implementation for reading IPCC AR6 FACTS projected GMSL.

    Parameters
    ----------
    data_dir : str
        Path to the confidence-level directory.
    confidence : str
        ``'medium'`` or ``'low'``.
    convert_to_meters : bool
        If True, convert from native mm to meters.
    """
    try:
        import xarray as xr
    except ImportError:
        raise ImportError(
            "xarray is required to read IPCC AR6 FACTS NetCDF files.\n"
            "Install with: pip install xarray netCDF4"
        )

    result = {}
    conf_tag = f'{confidence}_confidence'

    # Discover scenario directories
    scenario_dirs = sorted([
        d for d in os.listdir(data_dir)
        if os.path.isdir(os.path.join(data_dir, d))
    ])

    if not scenario_dirs:
        raise FileNotFoundError(
            f"No scenario directories found in {data_dir}"
        )

    # Component names and corresponding file prefixes
    components = ['oceandynamics', 'AIS', 'GIS', 'glaciers', 'landwaterstorage']

    for scenario in scenario_dirs:
        scenario_path = os.path.join(data_dir, scenario)

        # --- Read total ---
        total_file = os.path.join(
            scenario_path,
            f'total_{scenario}_{conf_tag}_values.nc'
        )
        if not os.path.exists(total_file):
            continue

        ds_total = xr.open_dataset(total_file)
        sl = ds_total['sea_level_change'].values[:, :, 0]  # (quantiles, years)
        quantiles = ds_total['quantiles'].values
        years = ds_total['years'].values

        # Truncate to years <= 2260 to avoid pandas Timestamp overflow
        yr_mask = years <= 2260
        sl = sl[:, yr_mask]
        years = years[yr_mask]

        # Find target quantile indices
        def _find_q(q_target):
            return int(np.argmin(np.abs(quantiles - q_target)))

        q05 = _find_q(0.05)
        q17 = _find_q(0.167)
        q50 = _find_q(0.5)
        q83 = _find_q(0.833)
        q95 = _find_q(0.95)

        scale = 0.001 if convert_to_meters else 1.0

        data = {
            'gmsl': sl[q50, :] * scale,
            'gmsl_lower': sl[q05, :] * scale,
            'gmsl_upper': sl[q95, :] * scale,
            'gmsl_17': sl[q17, :] * scale,
            'gmsl_83': sl[q83, :] * scale,
            'decimal_year': years.astype(float),
        }

        ds_total.close()

        # --- Read components ---
        for comp in components:
            comp_file = os.path.join(
                scenario_path,
                f'{comp}_{scenario}_{conf_tag}_values.nc'
            )
            if not os.path.exists(comp_file):
                data[comp] = np.full(len(years), np.nan)
                continue

            ds_comp = xr.open_dataset(comp_file)
            sl_comp = ds_comp['sea_level_change'].values[:, :, 0]
            q_comp = ds_comp['quantiles'].values
            years_comp = ds_comp['years'].values
            q50_comp = int(np.argmin(np.abs(q_comp - 0.5)))

            median_comp = sl_comp[q50_comp, :] * scale

            # Interpolate to total's year grid if needed
            if not np.array_equal(years_comp, years):
                data[comp] = np.interp(years, years_comp, median_comp,
                                       left=np.nan, right=np.nan)
            else:
                data[comp] = median_comp

            ds_comp.close()

        # Build DataFrame
        time_index = pd.to_datetime(
            [decimal_year_to_datetime(float(y)) for y in years]
        )

        df = pd.DataFrame(data, index=time_index)
        df.index.name = 'time'

        # Attach unit metadata
        unit = 'm' if convert_to_meters else 'mm'
        _sl_cols = ['gmsl', 'gmsl_lower', 'gmsl_upper', 'gmsl_17', 'gmsl_83']
        _comp_cols = [c for c in components if c in data]
        df.attrs = {
            'dataset': 'ipcc_ar6_projected_gmsl',
            'confidence': confidence,
            'scenario': scenario,
            'reference': 'Fox-Kemper et al. (2021)',
            'doi': '10.1017/9781009157896.011',
            'data_doi': '10.5281/zenodo.5914709',
            'native_units': {c: 'mm' for c in _sl_cols + _comp_cols},
            'current_units': {
                **{c: unit for c in _sl_cols + _comp_cols},
                'decimal_year': 'yr',
            },
            'units_standard': False,
            'quantity': 'sea_level',
            'native_time_resolution': 'decadal',
        }

        result[scenario] = df

    return result


def read_ipcc_ar6_projected_gmsl(
    data_dir: str,
    convert_to_meters: bool = True,
) -> dict:
    """
    Read IPCC AR6 projected global mean sea level (medium confidence) from FACTS.

    Reads the medium-confidence NetCDF output files from the IPCC AR6 FACTS
    framework (Garner et al., 2021). Each scenario directory contains per-component
    NetCDF files with full quantile distributions of projected sea level change.

    Parameters
    ----------
    data_dir : str
        Path to the medium-confidence directory, e.g.,
        ``data/raw/ipcc_ar6/slr/ar6/global/confidence_output_files/medium_confidence``.
        Expected structure::

            medium_confidence/
                ssp119/
                    total_ssp119_medium_confidence_values.nc
                    ...

    convert_to_meters : bool, default True
        If True, convert from native mm to meters.

    Returns
    -------
    dict of pd.DataFrame
        Dictionary keyed by scenario name. Each DataFrame has a datetime
        index (decadal, Jan 1) and columns: gmsl, gmsl_lower, gmsl_upper,
        gmsl_17, gmsl_83, oceandynamics, AIS, GIS, glaciers,
        landwaterstorage, decimal_year.

    Note
    ----
    - All values are sea level change relative to a 2005 baseline.
    - The 17%–83% range corresponds to the IPCC "likely" range (66% CI).

    Reference
    ---------
    Fox-Kemper, B., et al. (2021). https://doi.org/10.1017/9781009157896.011
    Garner, G. G., et al. (2023). https://doi.org/10.5194/gmd-16-7461-2023
    """
    return _read_ipcc_ar6_projected_gmsl_impl(
        data_dir, confidence='medium', convert_to_meters=convert_to_meters,
    )


def read_ipcc_ar6_projected_gmsl_low_confidence(
    data_dir: str,
    convert_to_meters: bool = True,
) -> dict:
    """
    Read IPCC AR6 projected GMSL (low confidence) from FACTS.

    Low-confidence projections include structured expert judgment on ice sheet
    processes that allows for higher tail risks than medium confidence.

    Parameters
    ----------
    data_dir : str
        Path to the low-confidence directory, e.g.,
        ``data/raw/ipcc_ar6/slr/ar6/global/confidence_output_files/low_confidence``.
    convert_to_meters : bool, default True
        If True, convert from native mm to meters.

    Returns
    -------
    dict of pd.DataFrame
        Structure identical to ``read_ipcc_ar6_projected_gmsl()`` but with
        low-confidence quantiles (wider tails, especially at p95).

    Note
    ----
    - Low confidence typically available for ssp126, ssp245, ssp585 only.
    - Low confidence has substantially higher p95 values, reflecting
      ice-sheet process uncertainty.

    See Also
    --------
    read_ipcc_ar6_projected_gmsl : Medium confidence projections (default).
    """
    return _read_ipcc_ar6_projected_gmsl_impl(
        data_dir, confidence='low', convert_to_meters=convert_to_meters,
    )


# =============================================================================
# OTHER DATA READERS
# =============================================================================

def read_noaa_thermosteric(zip_path, start_year=1955):
    """
    Disciplined reader for NOAA NCEI thermosteric sea level ZIP archives.
    Automatically extracts depth_range from filename and validates convention.
    """
    # 1. Extract depth_range and validate naming convention
    filename = os.path.basename(zip_path)
    # Convention: noaa_thermosteric_SL_<depth_range>-<timeresolution>*
    pattern = r"noaa_thermosteric_SL_(\d+-\d+)-(\w+)"
    match = re.search(pattern, filename)
    
    if not match:
        raise ValueError(
            f"Filename '{filename}' does not follow the required convention: "
            "noaa_thermosteric_SL_<depth_range>-<timeresolution>*"
        )
    
    depth_range = match.group(1)  # e.g., '0-700' or '0-2000'
    time_res = match.group(2)    # e.g., '3month'

    # 2. Extract the .dat file from the zip
    with zipfile.ZipFile(zip_path, 'r') as z:
        # Match file inside zip that contains the depth_range string
        file_list = [f for f in z.namelist() if f.endswith('.dat') and depth_range.replace('-', '_') in f.replace('-', '_')]
        if not file_list:
            file_list = [f for f in z.namelist() if f.endswith('.dat')]
            
        if not file_list:
            raise FileNotFoundError(f"No .dat matrix found in {filename} for depth {depth_range}.")
        
        with z.open(file_list[0]) as f:
            content = f.read().decode('utf-8')

    # 3. Parse fixed 8-character width blocks
    raw_data = []
    for line in content.splitlines():
        line = line.strip()
        # NOAA .dat files use 8-char blocks; some values may be run together
        chunks = [line[i:i+8] for i in range(0, len(line), 8)]
        for chunk in chunks:
            val_str = chunk.strip()
            if val_str:
                try:
                    raw_data.append(float(val_str))
                except ValueError:
                    continue
    
    values = np.array(raw_data)
    
    # 4. Handle missing value flags (-999.999)
    values[values == -999.999] = np.nan
    
    # 5. Construct Quarterly DatetimeIndex (Feb, May, Aug, Nov)
    #    Guard against year overflow (raw data may have trailing values)
    num_entries = len(values)
    dates = []
    for i in range(num_entries):
        year = start_year + (i // 4)
        if year > 2100:
            values = values[:i]   # truncate to valid range
            break
        month = [2, 5, 8, 11][i % 4]
        dates.append(pd.Timestamp(year=year, month=month, day=15))
        
    # 6. Standardization to Millimeters (mm)
    # Raw values are in cm (e.g., 2.065); multiply by 10 to match Project Manifest.
    col_name = f'tsl_{depth_range.replace("-", "_")}_mm'
    df = pd.DataFrame({col_name: values * 10.0}, index=dates)
    df.index.name = 'date'

    # Attach unit metadata
    df.attrs = {
        'dataset': 'noaa_thermosteric',
        'reference': 'Levitus et al. (2012)',
        'doi': '10.1029/2012GL051106',
        'data_doi': '10.7289/V53F4MVP',
        'depth_range': depth_range,
        'native_units': {col_name: 'cm'},
        'current_units': {col_name: 'mm'},
        'units_standard': False,
        'quantity': 'sea_level',
        'native_time_resolution': 'quarterly',
    }

    return df.dropna()


# =============================================================================
# SAOD (STRATOSPHERIC AEROSOL OPTICAL DEPTH) READERS
# =============================================================================

def read_glossac(filepath, wavelength=525):
    """
    Read GloSSAC stratospheric aerosol optical depth from NetCDF.

    Computes a cos(lat)-weighted global-mean SAOD at the requested wavelength
    from the ``Glossac_Aerosol_Optical_Depth`` variable.

    Parameters
    ----------
    filepath : str or Path
        Path to ``GloSSAC_V2.23_NC4.nc`` (or compatible version).
    wavelength : int, default 525
        Wavelength in nm.  Available: 386, 452, 525, 1020.
        386 and 452 are all-NaN in v2.23; use 525 (mid-visible) or 1020.

    Returns
    -------
    pd.DataFrame
        Monthly DataFrame (datetime index) with column ``saod``.
        Key values: background ~0.005–0.008; Pinatubo peak ~0.127 (525 nm).

    References
    ----------
    Thomason, L. W., et al. (2018). A new SAGE-based stratospheric aerosol
    climatology, *ESSD*, 10, 469–492, doi:10.5194/essd-10-469-2018.
    """
    import xarray as xr

    ds = xr.open_dataset(str(filepath))
    aod = ds['Glossac_Aerosol_Optical_Depth']       # (time, lat, wavelengths_glossac)
    lat = ds['lat'].values                            # 32 bins, -77.5 to 77.5
    time_ym = ds['time'].values                       # YYYYMM integers
    wl_values = ds['wavelengths_glossac'].values      # [386, 452, 525, 1020]

    # Select wavelength
    wl_idx = int(np.where(wl_values == wavelength)[0][0])
    aod_slice = aod.values[:, :, wl_idx]             # (n_time, n_lat)

    # Cos-latitude weighting
    cos_lat = np.cos(np.radians(lat))
    global_mean = np.zeros(len(time_ym))
    for i in range(len(time_ym)):
        valid = ~np.isnan(aod_slice[i])
        if valid.any():
            w = cos_lat[valid]
            global_mean[i] = np.average(aod_slice[i, valid], weights=w)
        else:
            global_mean[i] = np.nan

    # Build datetime index from YYYYMM
    dates = pd.to_datetime([
        f'{int(ym // 100)}-{int(ym % 100):02d}-15' for ym in time_ym
    ])

    df = pd.DataFrame({'saod': global_mean}, index=dates)
    df.index.name = 'time'

    ds.close()

    df.attrs = {
        'dataset': 'glossac_v2.23',
        'reference': 'Thomason et al. (2018)',
        'doi': '10.5194/essd-10-469-2018',
        'data_doi': '10.5067/GloSSAC-L3-V2.2',
        'native_units': {'saod': 'unitless'},
        'current_units': {'saod': 'unitless'},
        'units_standard': True,
        'quantity': 'aerosol_optical_depth',
        'native_time_resolution': 'monthly',
        'wavelength_nm': wavelength,
    }
    return df


def read_mauna_loa_transmission(filepath):
    """
    Read Mauna Loa Observatory apparent atmospheric transmission.

    Derives a SAOD proxy as ``-ln(transmission)`` from the morning monthly-mean
    apparent solar transmission ratio.  This record extends back to 1958,
    predating the satellite era covered by GloSSAC (1979).

    Parameters
    ----------
    filepath : str or Path
        Path to ``maunaLoa_transmission.txt``.

    Returns
    -------
    pd.DataFrame
        Monthly DataFrame (datetime index) with columns:
        - ``transmission`` : apparent solar transmission ratio (0–1)
        - ``saod`` : -ln(transmission), a proxy for aerosol optical depth

    Notes
    -----
    Major eruptions visible: Agung (~1963), El Chichón (~1982), Pinatubo (~1991).
    """
    raw = pd.read_csv(
        str(filepath),
        skiprows=2,
        sep=r'\s+',
        names=['date_label', 'decimal_year', 'transmission'],
    )

    # Build datetime index from decimal_year
    dates = [decimal_year_to_datetime(dy) for dy in raw['decimal_year']]
    df = pd.DataFrame({
        'transmission': raw['transmission'].values,
        'saod': -np.log(raw['transmission'].values),
    }, index=pd.DatetimeIndex(dates, name='time'))

    df.attrs = {
        'dataset': 'mauna_loa_transmission',
        'reference': 'NOAA Global Monitoring Laboratory',
        'doi': '',
        'data_doi': '',
        'native_units': {'transmission': 'unitless', 'saod': 'unitless'},
        'current_units': {'transmission': 'unitless', 'saod': 'unitless'},
        'units_standard': True,
        'quantity': 'aerosol_optical_depth',
        'native_time_resolution': 'monthly',
    }
    return df


# =============================================================================
# COMPONENT DECOMPOSITION READERS
# =============================================================================

# ---------------------------------------------------------------------------
# Generic IMBIE reader  (Gt-format CSV files)
# ---------------------------------------------------------------------------

def _read_imbie_gt(filepath: str, dataset_name: str, region_label: str,
                   convert_to_sle: bool = True) -> pd.DataFrame:
    """
    Internal helper to read any IMBIE Gt-format CSV file.

    All IMBIE 2021 Gt files share the same 5-column layout:
        Year, Mass balance (Gt/yr), uncertainty, Cumulative (Gt), uncertainty

    Parameters
    ----------
    filepath : str
        Path to CSV file.
    dataset_name : str
        Value for ``df.attrs['dataset']`` (e.g. ``'imbie_greenland'``).
    region_label : str
        Human-readable region (e.g. ``'Greenland Ice Sheet'``).
    convert_to_sle : bool, default True
        Convert Gt to mm SLE (÷ 362.5) then to meters (÷ 1000).
        If False, keep native Gt and Gt/yr.

    Returns
    -------
    pd.DataFrame
    """
    df = pd.read_csv(filepath)

    # Rename to standard column names
    df = df.rename(columns={
        'Year': 'decimal_year',
        'Mass balance (Gt/yr)': 'mass_balance_rate',
        'Mass balance uncertainty (Gt/yr)': 'mass_balance_rate_sigma',
        'Cumulative mass balance (Gt)': 'cumulative_mass_balance',
        'Cumulative mass balance uncertainty (Gt)': 'cumulative_mass_balance_sigma',
    })

    # Datetime index
    df['time'] = pd.to_datetime(
        [decimal_year_to_datetime(y) for y in df['decimal_year']]
    )
    df = df.set_index('time')
    df.index.name = 'time'

    # Unit conversion: Gt → mm SLE (÷ 362.5) → m SLE (÷ 1000)
    value_cols = [c for c in df.columns if c != 'decimal_year']
    if convert_to_sle:
        df[value_cols] = df[value_cols] / 362.5 / 1000.0   # Gt → m SLE
        _len = 'm'
        _rate = 'm/yr'
    else:
        _len = 'Gt'
        _rate = 'Gt/yr'

    df.attrs = {
        'dataset': dataset_name,
        'region': region_label,
        'reference': 'Otosaka et al. (2023)',
        'doi': '10.5194/essd-15-1597-2023',
        'data_doi': '10.5285/77B64C55-7166-4A06-9DEF-2E400398E452',
        'native_units': {
            'mass_balance_rate': 'Gt/yr',
            'mass_balance_rate_sigma': 'Gt/yr',
            'cumulative_mass_balance': 'Gt',
            'cumulative_mass_balance_sigma': 'Gt',
        },
        'current_units': {
            'mass_balance_rate': _rate,
            'mass_balance_rate_sigma': _rate,
            'cumulative_mass_balance': _len,
            'cumulative_mass_balance_sigma': _len,
            'decimal_year': 'yr',
        },
        'units_standard': convert_to_sle,
        'quantity': 'sea_level',
        'native_time_resolution': 'monthly',
    }
    return df


def read_imbie_greenland(filepath: str, convert_to_sle: bool = True) -> pd.DataFrame:
    """
    Read IMBIE Greenland Ice Sheet mass balance (Gt-format CSV).

    Parameters
    ----------
    filepath : str
        Path to ``imbie_greenland_2021_Gt.csv``.
    convert_to_sle : bool, default True
        If True, convert from Gt to meters sea-level equivalent
        (1 Gt = 1/362.5 mm SLE).

    Returns
    -------
    pd.DataFrame
        Monthly DataFrame (1992–2020) with columns:
        mass_balance_rate, mass_balance_rate_sigma,
        cumulative_mass_balance, cumulative_mass_balance_sigma,
        decimal_year.

    Notes
    -----
    - Total GrIS mass balance (SMB + discharge combined).
    - Positive rate → mass loss → sea level rise.
    - IMBIE GrIS includes peripheral glaciers.

    Reference
    ---------
    Otosaka et al. (2023) Earth Syst. Sci. Data, 15, 1597–1616.
    https://doi.org/10.5194/essd-15-1597-2023
    """
    return _read_imbie_gt(filepath, 'imbie_greenland', 'Greenland Ice Sheet',
                          convert_to_sle=convert_to_sle)


def read_imbie_east_antarctica(filepath: str, convert_to_sle: bool = True) -> pd.DataFrame:
    """
    Read IMBIE East Antarctic Ice Sheet mass balance (Gt-format CSV).

    Parameters
    ----------
    filepath : str
        Path to ``imbie_east_antarctica_2021_Gt.csv``.
    convert_to_sle : bool, default True

    Returns
    -------
    pd.DataFrame
        Monthly DataFrame (1992–2020).

    Notes
    -----
    - EAIS is largely stable; near mass balance or slight gain from increased snowfall.
    - Contains marine sectors (Totten, Aurora, Wilkes) but none show imminent instability.

    Reference
    ---------
    Otosaka et al. (2023) https://doi.org/10.5194/essd-15-1597-2023
    """
    return _read_imbie_gt(filepath, 'imbie_east_antarctica',
                          'East Antarctic Ice Sheet', convert_to_sle=convert_to_sle)


def read_imbie_antarctic_peninsula(filepath: str, convert_to_sle: bool = True) -> pd.DataFrame:
    """
    Read IMBIE Antarctic Peninsula mass balance (Gt-format CSV).

    Parameters
    ----------
    filepath : str
        Path to ``imbie_antarctic_peninsula_2021_Gt.csv``.
    convert_to_sle : bool, default True

    Returns
    -------
    pd.DataFrame
        Monthly DataFrame (1992–2020).

    Notes
    -----
    - The Peninsula behaves like a collection of mountain glaciers
      responding primarily to warming air temperature.
    - Larsen A (1995) and Larsen B (2002) ice shelf collapses drove
      significant glacier acceleration.

    Reference
    ---------
    Otosaka et al. (2023) https://doi.org/10.5194/essd-15-1597-2023
    """
    return _read_imbie_gt(filepath, 'imbie_antarctic_peninsula',
                          'Antarctic Peninsula', convert_to_sle=convert_to_sle)


def read_imbie_antarctica(filepath: str, convert_to_sle: bool = True) -> pd.DataFrame:
    """
    Read IMBIE total Antarctica mass balance (Gt-format CSV).

    Parameters
    ----------
    filepath : str
        Path to ``imbie_antarctica_2021_Gt.csv``.
    convert_to_sle : bool, default True

    Returns
    -------
    pd.DataFrame
        Monthly DataFrame (1992–2020).

    Notes
    -----
    Total AIS = WAIS + EAIS + Peninsula.  Useful for budget validation:
    IMBIE total should equal the sum of the three sub-components.

    Reference
    ---------
    Otosaka et al. (2023) https://doi.org/10.5194/essd-15-1597-2023
    """
    return _read_imbie_gt(filepath, 'imbie_antarctica',
                          'Antarctic Ice Sheet (total)', convert_to_sle=convert_to_sle)


def read_imbie_all(filepath: str, convert_to_sle: bool = True) -> pd.DataFrame:
    """
    Read IMBIE all ice sheets combined (Greenland + Antarctica) mass balance.

    Parameters
    ----------
    filepath : str
        Path to ``imbie_all_2021_Gt.csv``.
    convert_to_sle : bool, default True

    Returns
    -------
    pd.DataFrame
        Monthly DataFrame (1992–2020).

    Reference
    ---------
    Otosaka et al. (2023) https://doi.org/10.5194/essd-15-1597-2023
    """
    return _read_imbie_gt(filepath, 'imbie_all',
                          'All ice sheets (GrIS + AIS)', convert_to_sle=convert_to_sle)


# ---------------------------------------------------------------------------
# GlaMBIE glacier readers
# ---------------------------------------------------------------------------

_GLAMBIE_REGIONS = {
    0: 'global',
    1: 'alaska',
    2: 'western_canada_us',
    3: 'arctic_canada_north',
    4: 'arctic_canada_south',
    5: 'greenland_periphery',
    6: 'iceland',
    7: 'svalbard',
    8: 'scandinavia',
    9: 'russian_arctic',
    10: 'north_asia',
    11: 'central_europe',
    12: 'caucasus_middle_east',
    13: 'central_asia',
    14: 'south_asia_west',
    15: 'south_asia_east',
    16: 'low_latitudes',
    17: 'southern_andes',
    18: 'new_zealand',
    19: 'antarctic_and_subantarctic',
}


def read_glambie_global(filepath: str, convert_to_sle: bool = True) -> pd.DataFrame:
    """
    Read GlaMBIE global glacier consensus mass balance.

    Parameters
    ----------
    filepath : str
        Path to ``0_global_glambie_consensus.csv``.
    convert_to_sle : bool, default True
        Convert Gt to meters sea-level equivalent.

    Returns
    -------
    pd.DataFrame
        Annual DataFrame (2000–2023) with columns:
        mass_balance (Gt or m SLE), mass_balance_sigma,
        mass_balance_mwe (m.w.e.), mass_balance_mwe_sigma,
        glacier_area (km²), decimal_year.

    Notes
    -----
    - Positive mass_balance in the raw file = mass gain (opposite sign to SLR);
      this reader flips sign so that positive = sea level contribution.
    - 1 Gt water = 1/362.5 mm SLE.
    - GlaMBIE excludes ice-sheet peripheral glaciers by default, but
      Region 5 (Greenland periphery) and Region 19 (Antarctic & Subantarctic)
      overlap with IMBIE boundaries.  Check before combining with ice-sheet data.

    Reference
    ---------
    Zemp, M. et al. (2024). GlaMBIE: Global Glacier Mass Balance
    Intercomparison Exercise. World Glacier Monitoring Service (WGMS).
    https://doi.org/10.5904/wgms-glambie-2024-07
    """
    df = pd.read_csv(filepath)

    # Mid-year time coordinate
    mid_year = (df['start_dates'] + df['end_dates']) / 2.0
    df['decimal_year'] = mid_year
    df['time'] = pd.to_datetime(
        [decimal_year_to_datetime(y) for y in mid_year]
    )
    df = df.set_index('time')
    df.index.name = 'time'

    # Rename and sign-flip: GlaMBIE negative = mass loss, but we want
    # positive = sea level contribution
    df = df.rename(columns={
        'combined_gt': 'mass_balance',
        'combined_gt_errors': 'mass_balance_sigma',
        'combined_mwe': 'mass_balance_mwe',
        'combined_mwe_errors': 'mass_balance_mwe_sigma',
        'glacier_area': 'glacier_area',
    })
    # Flip sign: negative Gt = mass loss → positive SLR contribution
    df['mass_balance'] = -df['mass_balance']
    df['mass_balance_mwe'] = -df['mass_balance_mwe']

    # Drop 'region', 'start_dates', 'end_dates' columns
    df = df.drop(columns=['region', 'start_dates', 'end_dates'], errors='ignore')

    if convert_to_sle:
        # Gt → m SLE (÷ 362.5 ÷ 1000)
        for col in ['mass_balance', 'mass_balance_sigma']:
            df[col] = df[col] / 362.5 / 1000.0
        _unit = 'm/yr'
    else:
        _unit = 'Gt/yr'

    df.attrs = {
        'dataset': 'glambie_global',
        'reference': 'Zemp et al. (2024)',
        'doi': '10.5904/wgms-glambie-2024-07',
        'data_doi': '10.5904/wgms-glambie-2024-07',
        'native_units': {
            'mass_balance': 'Gt/yr', 'mass_balance_sigma': 'Gt/yr',
            'mass_balance_mwe': 'm.w.e./yr', 'mass_balance_mwe_sigma': 'm.w.e./yr',
            'glacier_area': 'km^2',
        },
        'current_units': {
            'mass_balance': _unit, 'mass_balance_sigma': _unit,
            'mass_balance_mwe': 'm.w.e./yr', 'mass_balance_mwe_sigma': 'm.w.e./yr',
            'glacier_area': 'km^2', 'decimal_year': 'yr',
        },
        'units_standard': False,
        'quantity': 'sea_level',
        'native_time_resolution': 'annual',
        'sign_convention': 'positive = sea level contribution (mass loss)',
    }
    return df


def read_glambie_regional(filepath: str, convert_to_sle: bool = True) -> pd.DataFrame:
    """
    Read a GlaMBIE regional glacier mass balance CSV.

    Parameters
    ----------
    filepath : str
        Path to a regional CSV file, e.g.
        ``glambie_results_20240716/calendar_years/1_alaska.csv``.
    convert_to_sle : bool, default True

    Returns
    -------
    pd.DataFrame
        Same structure as ``read_glambie_global()``.

    Notes
    -----
    Regional files have the same column layout as the global file.
    The region name is extracted from the 'region' column.
    """
    df = pd.read_csv(filepath)

    region_name = df['region'].iloc[0] if 'region' in df.columns else 'unknown'

    mid_year = (df['start_dates'] + df['end_dates']) / 2.0
    df['decimal_year'] = mid_year
    df['time'] = pd.to_datetime(
        [decimal_year_to_datetime(y) for y in mid_year]
    )
    df = df.set_index('time')
    df.index.name = 'time'

    df = df.rename(columns={
        'combined_gt': 'mass_balance',
        'combined_gt_errors': 'mass_balance_sigma',
        'combined_mwe': 'mass_balance_mwe',
        'combined_mwe_errors': 'mass_balance_mwe_sigma',
    })
    # Flip sign: negative Gt = mass loss → positive SLR contribution
    df['mass_balance'] = -df['mass_balance']
    df['mass_balance_mwe'] = -df['mass_balance_mwe']

    df = df.drop(columns=['region', 'start_dates', 'end_dates'], errors='ignore')

    if convert_to_sle:
        for col in ['mass_balance', 'mass_balance_sigma']:
            df[col] = df[col] / 362.5 / 1000.0
        _unit = 'm/yr'
    else:
        _unit = 'Gt/yr'

    df.attrs = {
        'dataset': f'glambie_{region_name}',
        'region': region_name,
        'reference': 'Zemp et al. (2024)',
        'doi': '10.5904/wgms-glambie-2024-07',
        'data_doi': '10.5904/wgms-glambie-2024-07',
        'native_units': {
            'mass_balance': 'Gt/yr', 'mass_balance_sigma': 'Gt/yr',
            'mass_balance_mwe': 'm.w.e./yr', 'mass_balance_mwe_sigma': 'm.w.e./yr',
            'glacier_area': 'km^2',
        },
        'current_units': {
            'mass_balance': _unit, 'mass_balance_sigma': _unit,
            'mass_balance_mwe': 'm.w.e./yr', 'mass_balance_mwe_sigma': 'm.w.e./yr',
            'glacier_area': 'km^2', 'decimal_year': 'yr',
        },
        'units_standard': False,
        'quantity': 'sea_level',
        'native_time_resolution': 'annual',
        'sign_convention': 'positive = sea level contribution (mass loss)',
    }
    return df


# ---------------------------------------------------------------------------
# IPCC AR6 FACTS Component Reader (generic)
# ---------------------------------------------------------------------------

def read_ipcc_ar6_component(
    component_dir: str,
    component_type: str,
    sub_component: Optional[str] = None,
    model: Optional[str] = None,
    scenario: str = 'ssp245',
    convert_to_meters: bool = True,
) -> pd.DataFrame:
    """
    Read an IPCC AR6 FACTS component-level projection from NetCDF.

    Generic reader for the ``dist_components/`` directory containing
    per-component, per-scenario projection distributions.

    Parameters
    ----------
    component_dir : str
        Path to the ``dist_components/`` directory containing all
        component NetCDF files.
    component_type : str
        One of: ``'glaciers'``, ``'icesheets'``, ``'landwaterstorage'``,
        ``'oceandynamics'``.
    sub_component : str, optional
        Ice-sheet sub-region: ``'GIS'``, ``'AIS'``, ``'WAIS'``, ``'EAIS'``,
        ``'PEN'``.  Required for ``component_type='icesheets'``.
        Ignored for other component types.
    model : str, optional
        Model identifier to disambiguate when multiple models exist.
        E.g. ``'ar5'``, ``'ipccar6-ismipemuicesheet'``, ``'FittedISMIP'``,
        ``'ipccar6-larmipicesheet'``, ``'ipccar6-bambericesheet'``,
        ``'dp20'``.  If None, auto-selects: prefers ``'ipccar6-'`` models
        when available.
    scenario : str, default 'ssp245'
        SSP or temperature-limit scenario (e.g. ``'ssp585'``,
        ``'tlim2.0win0.25'``).
    convert_to_meters : bool, default True
        Convert from native mm to meters.

    Returns
    -------
    pd.DataFrame
        DataFrame with integer year index and columns:
        ``median``, ``p5``, ``p17``, ``p50``, ``p83``, ``p95``,
        ``mean``, ``std``.  All in mm or m depending on
        ``convert_to_meters``.

    Notes
    -----
    - FACTS projections are relative to a 2005 baseline.
    - Native units in the NetCDF files are mm.
    - The 107 quantiles in each file span 0.000–1.000.

    Reference
    ---------
    Garner, G. G., et al. (2023). https://doi.org/10.5194/gmd-16-7461-2023
    """
    import glob as globmod

    # Build filename pattern
    # Filename convention: {type_prefix}-{model}-{scenario}[_{sub}]_globalsl.nc
    #   glaciers-ar5-glaciersgmip2-ssp245_globalsl.nc
    #   icesheets-ar5-icesheets-ssp245_WAIS_globalsl.nc
    #   landwaterstorage-ssp-landwaterstorage-ssp245_globalsl.nc
    #   oceandynamics-tlm-oceandynamics-ssp245_globalsl.nc

    if sub_component:
        pattern = os.path.join(
            component_dir,
            f'{component_type}-*-{scenario}_{sub_component}_globalsl.nc'
        )
    else:
        pattern = os.path.join(
            component_dir,
            f'{component_type}-*-{scenario}_globalsl.nc'
        )

    matches = sorted(globmod.glob(pattern))

    if not matches:
        raise FileNotFoundError(
            f"No FACTS files found for component_type='{component_type}', "
            f"sub_component='{sub_component}', scenario='{scenario}' "
            f"in {component_dir}.\nPattern: {pattern}"
        )

    # Model selection
    if model is not None:
        filtered = [m for m in matches if model in os.path.basename(m)]
        if not filtered:
            available = [os.path.basename(m) for m in matches]
            raise FileNotFoundError(
                f"Model '{model}' not found. Available files: {available}"
            )
        matches = filtered

    if len(matches) > 1:
        # Prefer ipccar6 models
        ipccar6 = [m for m in matches if 'ipccar6' in os.path.basename(m)]
        if ipccar6:
            matches = ipccar6
        # If still multiple, prefer the first alphabetically
        if len(matches) > 1:
            pass  # Take first

    nc_path = matches[0]

    # Read NetCDF
    import xarray as xr
    ds = xr.open_dataset(nc_path)

    years = ds['years'].values
    quantiles = ds['quantiles'].values
    # sea_level_change: (quantiles, years, locations=1)
    slc = ds['sea_level_change'].values[:, :, 0]  # (quantiles, years)
    ds.close()

    # Extract key quantiles
    def _get_q(q_val):
        idx = np.argmin(np.abs(quantiles - q_val))
        return slc[idx, :]

    df = pd.DataFrame({
        'p5': _get_q(0.05),
        'p17': _get_q(0.17),
        'median': _get_q(0.50),
        'p83': _get_q(0.83),
        'p95': _get_q(0.95),
        'mean': np.nanmean(slc, axis=0),
        'std': np.nanstd(slc, axis=0),
    }, index=years)
    df.index.name = 'year'

    if convert_to_meters:
        for col in df.columns:
            df[col] = df[col] / 1000.0
        _unit = 'm'
    else:
        _unit = 'mm'

    model_used = os.path.basename(nc_path).split(f'-{scenario}')[0]

    df.attrs = {
        'dataset': f'ipcc_ar6_{component_type}',
        'component': component_type,
        'sub_component': sub_component or 'total',
        'model': model_used,
        'scenario': scenario,
        'source_file': os.path.basename(nc_path),
        'reference': 'Garner et al. (2023)',
        'doi': '10.5194/gmd-16-7461-2023',
        'native_units': 'mm',
        'current_units': _unit,
        'units_standard': convert_to_meters,
        'quantity': 'sea_level',
        'baseline': 2005,
    }
    return df


def list_ipcc_ar6_components(component_dir: str) -> pd.DataFrame:
    """
    List all available IPCC AR6 FACTS component files.

    Parameters
    ----------
    component_dir : str
        Path to ``dist_components/`` directory.

    Returns
    -------
    pd.DataFrame
        Table with columns: filename, component_type, model,
        scenario, sub_component.
    """
    import glob as globmod

    files = sorted(globmod.glob(os.path.join(component_dir, '*.nc')))
    records = []
    for f in files:
        fname = os.path.basename(f)
        # Parse: {component_type}-{model_parts}-{scenario}[_{sub}]_globalsl.nc
        parts = fname.replace('_globalsl.nc', '').split('-')
        comp_type = parts[0]

        # Find scenario (starts with 'ssp' or 'tlim')
        scenario_idx = None
        for i, p in enumerate(parts):
            if p.startswith('ssp') or p.startswith('tlim'):
                scenario_idx = i
                break

        if scenario_idx is None:
            continue

        model_str = '-'.join(parts[1:scenario_idx])
        scenario_sub = parts[scenario_idx]

        # Check for sub-component (after underscore in scenario part)
        if '_' in scenario_sub:
            scenario, sub = scenario_sub.split('_', 1)
        else:
            scenario = scenario_sub
            sub = None

        records.append({
            'filename': fname,
            'component_type': comp_type,
            'model': model_str,
            'scenario': scenario,
            'sub_component': sub,
        })

    return pd.DataFrame(records)


# ---------------------------------------------------------------------------
# GRACE mascon reader  (JPL RL06.3, global gridded → land-integrated TWS)
# ---------------------------------------------------------------------------

def read_grace_tws_global(filepath: str, convert_to_sle: bool = True) -> pd.DataFrame:
    """
    Read GRACE/GRACE-FO JPL mascon and compute global land TWS anomaly.

    Integrates gridded liquid-water-equivalent (LWE) thickness over land
    (excluding Greenland and Antarctica) to produce a global terrestrial
    water storage anomaly time series in mm SLE or meters SLE.

    Parameters
    ----------
    filepath : str
        Path to ``GRCTellus.JPL.*.GLO.RL06.3M.MSCNv04CRI.nc``.
    convert_to_sle : bool, default True
        If True, convert cm LWE to meters SLE.
        If False, return integrated LWE in Gt.

    Returns
    -------
    pd.DataFrame
        Monthly DataFrame (2002–present) with columns:
        tws_anomaly, tws_anomaly_sigma, decimal_year.

    Notes
    -----
    - LWE thickness is in cm. Each 0.5°×0.5° grid cell contributes
      area × lwe_thickness × water_density to the mass anomaly.
    - Land mask from the file excludes ocean cells.
    - Greenland (lat > 60°N, lon 295–350°E) and Antarctica (lat < -60°S)
      are excluded so TWS does not double-count ice-sheet mass changes.
    - GRACE has a gap 2017-06 to 2018-05 (between GRACE and GRACE-FO).
    - 1 Gt water = 1/362.5 mm SLE.

    Reference
    ---------
    Watkins, M. M. et al. (2015). Improved methods for observing Earth's
    time variable mass distribution with GRACE. J. Geophys. Res., 120,
    2648–2671. https://doi.org/10.1002/2014JB011547
    """
    import xarray as xr

    ds = xr.open_dataset(filepath)
    lwe = ds['lwe_thickness']         # (time, lat, lon), cm
    unc = ds['uncertainty']           # (time, lat, lon), cm
    land_mask = ds['land_mask']       # (lat, lon) — 1 = land, 0 = ocean
    lat = ds['lat'].values            # degrees_north
    lon = ds['lon'].values            # degrees_east
    time_vals = ds['time'].values

    # Compute cell areas (km²) using cos(lat) weighting
    R_earth = 6371.0  # km
    dlat = np.abs(np.diff(lat).mean())  # ~0.5°
    dlon = np.abs(np.diff(lon).mean())  # ~0.5°
    lat_rad = np.deg2rad(lat)
    cell_area = (R_earth ** 2) * np.deg2rad(dlat) * np.deg2rad(dlon) * np.cos(lat_rad)
    # cell_area shape: (lat,) — broadcast over lon
    area_grid = np.outer(cell_area, np.ones(len(lon)))  # (lat, lon), km²

    # Build mask: land, excluding Greenland and Antarctica
    mask_2d = land_mask.values.copy()  # (lat, lon)
    # Exclude Antarctica: lat < -60
    mask_2d[lat < -60, :] = 0
    # Exclude Greenland: lat > 60 AND lon in [295, 350] (= [-65, -10] in 0–360)
    for i, la in enumerate(lat):
        if la > 60:
            for j, lo in enumerate(lon):
                lo360 = lo % 360
                if 295 <= lo360 <= 350:
                    mask_2d[i, j] = 0

    # Integrate: sum(lwe_cm × area_km² × mask) → total volume in cm·km²
    # Convert to Gt: 1 cm·km² = 1e-5 km³ water = 1e-5 × 1e9 tonnes = 1e4 tonnes = 0.01 Gt
    # Actually: 1 cm = 0.01 m; area in km² = 1e6 m²; volume = 0.01 × 1e6 = 1e4 m³
    # 1 m³ water = 1 tonne; 1 Gt = 1e12 kg = 1e9 tonnes
    # So 1e4 m³ = 1e4 tonnes = 1e4/1e9 Gt = 1e-5 Gt per cell
    # Total: sum(lwe_cm × area_km²) × 1e-5 = Gt
    # More precisely: cm_to_m = 0.01; km2_to_m2 = 1e6; rho_w = 1000 kg/m³
    # mass_kg = lwe_cm × 0.01 × area_km² × 1e6 × 1000
    #         = lwe_cm × area_km² × 1e7
    # mass_Gt = lwe_cm × area_km² × 1e7 / 1e12 = lwe_cm × area_km² × 1e-5

    lwe_vals = lwe.values   # (time, lat, lon)
    unc_vals = unc.values   # (time, lat, lon)

    n_time = lwe_vals.shape[0]
    tws_gt = np.full(n_time, np.nan)
    tws_gt_sigma = np.full(n_time, np.nan)

    masked_area = area_grid * mask_2d  # (lat, lon)

    for t in range(n_time):
        lwe_t = lwe_vals[t]
        unc_t = unc_vals[t]
        valid = ~np.isnan(lwe_t) & (mask_2d > 0)
        if valid.sum() > 0:
            tws_gt[t] = np.nansum(lwe_t * masked_area * valid) * 1e-5
            # Uncertainty: sum in quadrature (assumes spatially uncorrelated)
            tws_gt_sigma[t] = np.sqrt(
                np.nansum((unc_t * masked_area * valid) ** 2)
            ) * 1e-5

    ds.close()

    # Build DataFrame
    time_dt = pd.to_datetime(time_vals)
    df = pd.DataFrame({
        'tws_anomaly': tws_gt,
        'tws_anomaly_sigma': tws_gt_sigma,
    }, index=time_dt)
    df.index.name = 'time'

    # Add decimal year
    df['decimal_year'] = [datetime_to_decimal_year(t.to_pydatetime()) for t in df.index]

    if convert_to_sle:
        # Gt → m SLE: ÷ 362.5 ÷ 1000
        for col in ['tws_anomaly', 'tws_anomaly_sigma']:
            df[col] = df[col] / 362.5 / 1000.0
        _unit = 'm'
    else:
        _unit = 'Gt'

    df.attrs = {
        'dataset': 'grace_tws_global',
        'reference': 'Watkins et al. (2015)',
        'doi': '10.1002/2014JB011547',
        'data_doi': '10.5067/TEMSC-3JC63',
        'native_units': {'lwe_thickness': 'cm'},
        'current_units': {'tws_anomaly': _unit, 'tws_anomaly_sigma': _unit},
        'units_standard': convert_to_sle,
        'quantity': 'terrestrial_water_storage',
        'native_time_resolution': 'monthly',
        'excluded_regions': 'Greenland (lat>60, lon 295-350), Antarctica (lat<-60)',
        'sign_convention': 'positive = increased land storage (sea level drop)',
    }
    return df


# ---------------------------------------------------------------------------
# ENSO index readers
# ---------------------------------------------------------------------------

def read_noaa_oni(filepath: str) -> pd.DataFrame:
    """
    Read NOAA Oceanic Niño Index (ONI) from CPC.

    The ONI is a 3-month running mean of SST anomalies in the Niño 3.4
    region (5°N–5°S, 120°–170°W). El Niño: ONI ≥ +0.5 °C for ≥5
    consecutive months; La Niña: ONI ≤ −0.5 °C for ≥5 months.

    Parameters
    ----------
    filepath : str
        Path to ``noaa_oni_index.csv`` (or similar).
        Expected: Date column + ONI value.

    Returns
    -------
    pd.DataFrame
        Monthly DataFrame with column ``oni`` (°C anomaly).

    Reference
    ---------
    NOAA Climate Prediction Center.
    https://psl.noaa.gov/data/timeseries/monthly/ONI/
    """
    # Format: first row is header "Date,  ONI from CPC  missing value -99.9 ..."
    # then rows: "1950-01-01,   -1.530"
    df = pd.read_csv(filepath, skiprows=0)

    # The header line may be messy — rename
    cols = list(df.columns)
    df = df.rename(columns={cols[0]: 'date', cols[1]: 'oni'})

    df['time'] = pd.to_datetime(df['date'].str.strip())
    df = df.set_index('time')
    df.index.name = 'time'

    df['oni'] = pd.to_numeric(df['oni'], errors='coerce')
    df.loc[df['oni'] <= -99, 'oni'] = np.nan

    df = df[['oni']].copy()

    df.attrs = {
        'dataset': 'noaa_oni',
        'reference': 'NOAA Climate Prediction Center',
        'doi': '',
        'data_doi': '',
        'native_units': {'oni': 'degC'},
        'current_units': {'oni': 'degC'},
        'units_standard': True,
        'quantity': 'enso_index',
        'native_time_resolution': 'monthly',
        'description': 'Oceanic Niño Index (3-month running mean SST anomaly, Niño 3.4)',
    }
    return df


def read_noaa_mei(filepath: str) -> pd.DataFrame:
    """
    Read NOAA Multivariate ENSO Index (MEI.v2).

    The MEI combines five variables (SLP, SST, surface zonal/meridional
    winds, OLR) into a single bimonthly index via EOF analysis.

    Parameters
    ----------
    filepath : str
        Path to ``noaa_mei_index.txt``.
        Fixed-width format: YEAR followed by 12 monthly values.

    Returns
    -------
    pd.DataFrame
        Monthly DataFrame with column ``mei`` (dimensionless).

    Reference
    ---------
    Wolter, K. & Timlin, M. S. (2011). El Niño/Southern Oscillation
    behaviour since 1871. Int. J. Climatol., 31, 1074–1087.
    https://psl.noaa.gov/enso/mei/
    """
    # Fixed-width: first row = "YEAR  start  end"
    # subsequent rows = "1979  0.46  0.29  ..."
    # First line contains year range info
    with open(filepath, 'r') as f:
        lines = f.readlines()

    # First line: "1979     2025" — metadata; footer has text lines
    records = []
    for line in lines[1:]:
        parts = line.split()
        if len(parts) < 13:
            continue
        try:
            year = int(parts[0])
        except ValueError:
            continue   # skip footer text lines
        for month_idx, val_str in enumerate(parts[1:13], start=1):
            try:
                val = float(val_str)
            except ValueError:
                val = np.nan
            records.append({
                'year': year, 'month': month_idx, 'mei': val
            })

    df = pd.DataFrame(records)
    df['time'] = pd.to_datetime(
        df['year'].astype(str) + '-' + df['month'].astype(str).str.zfill(2) + '-15'
    )
    df = df.set_index('time')
    df.index.name = 'time'
    df = df[['mei']].copy()

    # Replace missing values
    df.loc[df['mei'] <= -999, 'mei'] = np.nan

    df.attrs = {
        'dataset': 'noaa_mei',
        'reference': 'Wolter & Timlin (2011)',
        'doi': '',
        'data_doi': '',
        'native_units': {'mei': 'dimensionless'},
        'current_units': {'mei': 'dimensionless'},
        'units_standard': True,
        'quantity': 'enso_index',
        'native_time_resolution': 'monthly',
        'description': 'Multivariate ENSO Index v2 (bimonthly EOF-based)',
    }
    return df


# =============================================================================
# MODULE EXPORTS
# =============================================================================

__all__ = [
    # Utilities
    'decimal_year_to_datetime',
    'datetime_to_decimal_year',
    # GMSL readers
    'read_nasa_gmsl',
    'read_frederikse2020',
    'read_dangendorf2024',
    'read_dangendorf2024_fields',
    'read_horwath2022',
    'read_ipcc_ar6_observed_gmsl',
    'read_imbie_west_antarctica',
    # Temperature readers
    'read_berkeley_earth',
    'read_hadcrut5',
    'read_nasa_gistemp',
    'read_noaa_globaltemp',
    # Projection readers
    'read_ipcc_ar6_projected_temperature',
    'read_ipcc_ar6_projected_gmsl',
    'read_ipcc_ar6_projected_gmsl_low_confidence',
    # Other readers
    'read_noaa_thermosteric',
    # SAOD readers
    'read_glossac',
    'read_mauna_loa_transmission',
    # Component decomposition readers
    'read_imbie_greenland',
    'read_imbie_east_antarctica',
    'read_imbie_antarctic_peninsula',
    'read_imbie_antarctica',
    'read_imbie_all',
    'read_glambie_global',
    'read_glambie_regional',
    'read_ipcc_ar6_component',
    'list_ipcc_ar6_components',
    'read_grace_tws_global',
    'read_noaa_oni',
    'read_noaa_mei',
    # Unit conversion
    'convert_to_standard_units',
    'convert_units',
]

if __name__ == "__main__":
    print("Sea Level Rise Data Readers")
    print("=" * 40)
    print("\nGMSL Readers:")
    print("  - read_nasa_gmsl()")
    print("  - read_frederikse2020()")
    print("  - read_dangendorf2024()")
    print("  - read_dangendorf2024_fields()")
    print("  - read_horwath2022()")
    print("  - read_ipcc_ar6_observed_gmsl()")
    print("  - read_imbie_west_antarctica()")
    print("\nTemperature Readers:")
    print("  - read_berkeley_earth()")
    print("  - read_hadcrut5()")
    print("  - read_nasa_gistemp()")
    print("  - read_noaa_globaltemp()")
    print("\nGMSL Projection Readers:")
    print("  - read_ipcc_ar6_projected_temperature()")
    print("  - read_ipcc_ar6_projected_gmsl()")
    print("\nOther Readers:")
    print("  - read_noaa_thermosteric()")
    print("\nSAOD Readers:")
    print("  - read_glossac()")
    print("  - read_mauna_loa_transmission()")
    print("\nImpact Functions:")
    print("  - people_displaced_kulpstrauss2019()")
    print("  - slr_cost_jevrejeva2018()")


# =============================================================================
# IMPACT ESTIMATION FUNCTIONS
# =============================================================================

def people_displaced_kulpstrauss2019(slr_m):
    """Estimate millions of people exposed to annual coastal flooding.

    Piecewise-linear interpolation of data extracted from Kulp & Strauss
    (2019), Nature Communications, Table 1 and main text (CoastalDEM).
    Population counts correspond to present-day (~2010 census era)
    population on land below the given water level; they do *not*
    include future population growth or coastal migration.

    Parameters
    ----------
    slr_m : float or array-like
        Global mean sea-level rise in **meters** above present (~year 2000).

    Returns
    -------
    float or np.ndarray
        Millions of people exposed to annual coastal flooding.

    Reference
    ---------
    Kulp, S. A. & Strauss, B. H. (2019). New elevation data triple
    estimates of global vulnerability to sea-level rise and coastal
    flooding. Nature Communications, 10, 4844.
    """
    # Data points: (SLR in m, millions exposed to annual flooding)
    # Extracted from Table 1 + main text (CoastalDEM, median estimates)
    slr_pts = np.array([0.00, 0.25, 0.50, 0.60, 0.80, 1.46, 2.00])
    pop_pts = np.array([250., 300., 340., 360., 400., 480., 630.])

    return np.interp(slr_m, slr_pts, pop_pts)


def slr_cost_jevrejeva2018(slr_m):
    """Estimate global annual flood cost without additional adaptation.

    Piecewise-linear interpolation of data from Jevrejeva et al. (2018),
    Environmental Research Letters, Figure 4(a) and conclusions.
    Costs are in billions of USD per year (2014 dollars), assuming the
    SSP2 socioeconomic pathway with no additional coastal adaptation
    beyond the base-year standard of protection.

    Parameters
    ----------
    slr_m : float or array-like
        Global mean sea-level rise in **meters** above present (~year 2000).

    Returns
    -------
    float or np.ndarray
        Annual global flood costs in **billions USD** per year.

    Reference
    ---------
    Jevrejeva, S., Jackson, L. P., Grinsted, A., Lincke, D., &
    Marzeion, B. (2018). Flood damage costs under the sea level rise
    with warming of 1.5 C and 2 C. Environmental Research Letters,
    13, 074014.
    """
    # Data points: (SLR in m, annual flood cost in billions USD/yr)
    # Extracted from Table/Conclusions text + Figure 4(a)
    slr_pts = np.array([0.00, 0.20, 0.52, 0.63, 0.86, 1.80])
    cost_pts = np.array([0., 1000., 10200., 11700., 14000., 27000.])

    return np.interp(slr_m, slr_pts, cost_pts)
