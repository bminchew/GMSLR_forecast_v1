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
    - Horwath et al. (2022): ESA CCI sea level budget closure
    - NASA GSFC: Satellite altimetry (TOPEX/Jason/Sentinel-6)

Temperature:
    - Berkeley Earth: Land/Ocean temperature
    - HadCRUT5: Met Office/UEA temperature
    - NASA GISTEMP: GISS temperature analysis
    - NOAA GlobalTemp: NCEI temperature

Dependencies
------------
- pandas
- numpy
- openpyxl (for Excel files)
- xarray, netCDF4 (for NetCDF files, Dangendorf only)

Example
-------
>>> from slr_data_readers import read_frederikse2020, read_berkeley_earth
>>> df_sl = read_frederikse2020('data/frederikse2020.xlsx')
>>> df_temp = read_berkeley_earth('data/berkeley_earth.txt')

"""

import numpy as np
import pandas as pd
from datetime import datetime
from typing import Optional


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
    
    # Calculate symmetric uncertainties (sigma = half the 90% CI width)
    for var in ['gmsl', 'steric', 'glaciers', 'greenland', 'antarctica', 'tws',
                'sum_contributors', 'reservoir', 'groundwater', 'tws_natural', 'altimetry']:
        if f'{var}_lower' in df.columns and f'{var}_upper' in df.columns:
            df[f'{var}_sigma'] = (df[f'{var}_upper'] - df[f'{var}_lower']) / 2
    
    # Convert mm to meters if requested
    if convert_to_meters:
        value_cols = [c for c in df.columns if c not in ['year', 'decimal_year']]
        df[value_cols] = df[value_cols] / 1000.0
    
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
    df['temperature_unc'] = (df['temperature_upper'] - df['temperature_lower']) / 2
    
    df.index.name = 'time'
    return df[['temperature', 'temperature_unc', 'temperature_lower', 'temperature_upper']]


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
    'read_horwath2022',
    'read_ipcc_ar6_observed_gmsl',
    'read_imbie_west_antarctica',
    # Temperature readers
    'read_berkeley_earth',
    'read_hadcrut5',
    'read_nasa_gistemp',
    'read_noaa_globaltemp',
]

if __name__ == "__main__":
    print("Sea Level Rise Data Readers")
    print("=" * 40)
    print("\nGMSL Readers:")
    print("  - read_nasa_gmsl()")
    print("  - read_frederikse2020()")
    print("  - read_dangendorf2024()")
    print("  - read_horwath2022()")
    print("  - read_ipcc_ar6_observed_gmsl")
    print("  - read_imbie_west_antarctica")
    print("\nTemperature Readers:")
    print("  - read_berkeley_earth()")
    print("  - read_hadcrut5()")
    print("  - read_nasa_gistemp()")
    print("  - read_noaa_globaltemp()")
