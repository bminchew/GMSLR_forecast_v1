"""
Sea Level Rise and Global Temperature Data Readers
===================================================

This module provides functions to read and parse various global mean sea level (GMSL)
and global mean surface temperature (GMST) datasets into standardized pandas DataFrames.

All DataFrames use a datetime index and include uncertainty columns where available.

Required packages:
- pandas
- numpy
- openpyxl (for Excel files)
- xarray or netCDF4 (for Dangendorf NetCDF file only)

"""

import pandas as pd
import numpy as np
from datetime import datetime
from typing import Optional, Tuple

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
    """
    year = int(decimal_year)
    rem = decimal_year - year
    base = datetime(year, 1, 1)
    next_year = datetime(year + 1, 1, 1)
    result = base + (next_year - base) * rem
    return result


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
    """
    year = dt.year
    base = datetime(year, 1, 1)
    next_year = datetime(year + 1, 1, 1)
    return year + (dt - base).total_seconds() / (next_year - base).total_seconds()


# =============================================================================
# GLOBAL MEAN SEA LEVEL (GMSL) DATA READERS
# =============================================================================

def read_nasa_gmsl(filepath: str) -> pd.DataFrame:
    """
    Read NASA GSFC GMSL from TOPEX/Poseidon, Jason-1/2/3, and Sentinel-6.
    
    This dataset provides satellite altimetry-based global mean sea level
    from the merged TOPEX/Poseidon, Jason, and Sentinel-6 missions.
    
    Parameters
    ----------
    filepath : str
        Path to the text file (nasa_GMSL_TPJAOS_5_2.txt)
        
    Returns
    -------
    pd.DataFrame
        DataFrame with datetime index. Key columns:
        - gmsl: Global mean sea level anomaly with GIA applied (mm)
        - gmsl_unc: Standard deviation (mm)
        - gmsl_smoothed: 60-day smoothed, seasonal removed (mm)
        - gmsl_nogia: GMSL without GIA correction (mm)
        - decimal_year: Original decimal year timestamp
        
    Reference
    ---------
    Beckley, B. D., Callahan, P. S., Hancock, D. W., Mitchum, G. T., & Ray, R. D. (2017).
    On the "Cal-Mode" Correction to TOPEX Satellite Altimetry and Its Effect on the 
    Global Mean Sea Level Time Series. Journal of Geophysical Research: Oceans, 122(11), 
    8371–8384. https://doi.org/10.1002/2017JC013090
    
    Data citation:
    GSFC. 2024. Global Mean Sea Level Trend from Integrated Multi-Mission Ocean Altimeters 
    TOPEX/Poseidon, Jason-1, OSTM/Jason-2, Jason-3, and Sentinel-6A Version 5.2. 
    Ver. 5.2 PO.DAAC, CA, USA. https://doi.org/10.5067/GMSLM-TJ152
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
            except:
                pass
    
    # Parse data - columns documented in header:
    # 1: altimeter type (0=dual-freq, 999=single-freq)
    # 2: merged file cycle #
    # 3: year+fraction of year (mid-cycle)
    # 4: number of observations
    # 5: number of weighted observations
    # 6: GMSL (GIA not applied) variation (mm)
    # 7: standard deviation of GMSL (GIA not applied) (mm)
    # 8: smoothed (60-day Gaussian) GMSL (GIA not applied) (mm)
    # 9: GMSL (GIA applied) variation (mm)
    # 10: standard deviation of GMSL (GIA applied) (mm)
    # 11: smoothed (60-day Gaussian) GMSL (GIA applied) (mm)
    # 12: smoothed GMSL (GIA applied); annual and semi-annual signal removed
    # 13: smoothed GMSL (GIA not applied); annual and semi-annual signal removed
    
    data = []
    for line in lines[data_start:]:
        parts = line.split()
        if len(parts) >= 12:
            try:
                row = {
                    'altimeter_type': int(parts[0]),
                    'cycle': int(parts[1]),
                    'decimal_year': float(parts[2]),
                    'n_obs': int(parts[3]),
                    'n_weighted_obs': float(parts[4]),
                    'gmsl_nogia': float(parts[5]),
                    'gmsl_nogia_std': float(parts[6]),
                    'gmsl_nogia_smoothed': float(parts[7]),
                    'gmsl': float(parts[8]),  # GIA applied
                    'gmsl_unc': float(parts[9]),
                    'gmsl_gia_smoothed': float(parts[10]),
                    'gmsl_smoothed': float(parts[11]),  # 60-day smoothed, seasonal removed
                }
                if len(parts) > 12:
                    row['gmsl_nogia_smoothed_noseasonal'] = float(parts[12])
                data.append(row)
            except (ValueError, IndexError):
                continue
    
    df = pd.DataFrame(data)
    df['time'] = pd.to_datetime([decimal_year_to_datetime(y) for y in df['decimal_year']])
    df = df.set_index('time')
    df.index.name = 'time'
    
    return df


def read_frederikse2020(filepath: str) -> pd.DataFrame:
    """
    Read Frederikse et al. (2020) global sea level budget data.
    
    This dataset provides a budget-closed reconstruction of GMSL and its components
    (thermosteric, glaciers, Greenland, Antarctica, TWS) from 1900-2018.
    
    Parameters
    ----------
    filepath : str
        Path to the Excel file (frederikse2020_global_timeseries.xlsx)
        
    Returns
    -------
    pd.DataFrame
        DataFrame with datetime index (annual, at July 1) and columns for GMSL 
        and components with associated uncertainties (lower/upper bounds).
        Key columns:
        - gmsl, gmsl_lower, gmsl_upper, gmsl_unc: Observed GMSL
        - steric, steric_unc: Thermosteric contribution
        - glaciers, glaciers_unc: Global glacier contribution
        - greenland, greenland_unc: Greenland Ice Sheet
        - antarctica, antarctica_unc: Antarctic Ice Sheet
        - tws, tws_unc: Terrestrial Water Storage
        
    Reference
    ---------
    Frederikse, T., Landerer, F., Caron, L., Adhikari, S., Parkes, D., Humphrey, V. W., 
    Dangendorf, S., Hogarth, P., Zanna, L., Cheng, L., & Wu, Y.-H. (2020). 
    The causes of sea-level rise since 1900. Nature, 584(7821), 393–397.
    https://doi.org/10.1038/s41586-020-2591-3
    """
    df = pd.read_excel(filepath, sheet_name='Global')
    
    # First column is year
    df = df.rename(columns={'Unnamed: 0': 'year'})
    
    # Create time index (annual data - use July 1)
    df['time'] = pd.to_datetime(df['year'].astype(int).astype(str) + '-07-01')
    df = df.set_index('time')
    df.index.name = 'time'
    
    # Rename columns to more convenient names
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
    
    # Calculate symmetric uncertainties as half the range
    for var in ['gmsl', 'steric', 'glaciers', 'greenland', 'antarctica', 'tws', 
                'sum_contributors', 'reservoir', 'groundwater', 'tws_natural', 'altimetry']:
        if f'{var}_lower' in df.columns and f'{var}_upper' in df.columns:
            df[f'{var}_unc'] = (df[f'{var}_upper'] - df[f'{var}_lower']) / 2
    
    return df


def read_dangendorf2024(filepath: str) -> pd.DataFrame:
    """
    Read Dangendorf et al. Kalman Smoother GMSL reconstruction.
    
    This dataset provides a hybrid tide gauge reconstruction using 
    Kalman smoother methodology, extending and updating Dangendorf et al. (2019).
    
    Parameters
    ----------
    filepath : str
        Path to the NetCDF file (dangendorf2024_KalmanSmootherHR_Global.nc)
        
    Returns
    -------
    pd.DataFrame
        DataFrame with datetime index and columns:
        - gmsl: Global mean sea level (mm)
        - gmsl_unc: Uncertainty (mm) if available
        
    Reference
    ---------
    Dangendorf, S., Hay, C., Calafat, F. M., Marcos, M., Piecuch, C. G., Berk, K., 
    & Jensen, J. (2019). Persistent acceleration in global sea-level rise since 
    the 1960s. Nature Climate Change, 9(9), 705–710.
    https://doi.org/10.1038/s41558-019-0531-8
    
    Note
    ----
    Requires xarray or netCDF4 package: pip install xarray netCDF4
    """
    try:
        import xarray as xr
    except ImportError:
        raise ImportError(
            "xarray is required to read NetCDF4 files. "
            "Install with: pip install xarray netCDF4"
        )
    
    ds = xr.open_dataset(filepath)
    
    # Find time variable
    time_var = None
    for tname in ['time', 'Time', 't', 'date']:
        if tname in ds.coords or tname in ds.dims or tname in ds.data_vars:
            time_var = tname
            break
    
    # Find sea level variable
    sl_var = None
    for slname in ['sea_level', 'gmsl', 'sl', 'GMSL', 'sea_level_anomaly', 
                   'sla', 'SSH', 'ssh', 'Global']:
        if slname in ds.data_vars:
            sl_var = slname
            break
    
    if time_var is None or sl_var is None:
        print(f"Available coordinates: {list(ds.coords)}")
        print(f"Available dimensions: {list(ds.dims)}")
        print(f"Available variables: {list(ds.data_vars)}")
        raise ValueError("Could not identify time or sea level variables. See printed info above.")
    
    # Convert to DataFrame
    time = pd.to_datetime(ds[time_var].values)
    gmsl = ds[sl_var].values.flatten()
    
    # Look for uncertainty variable
    unc_var = None
    for uname in ['uncertainty', 'error', 'se', 'std', 'sigma', 
                  'gmsl_unc', 'sea_level_uncertainty', 'err']:
        if uname in ds.data_vars:
            unc_var = uname
            break
    
    data = {'gmsl': gmsl}
    if unc_var:
        data['gmsl_unc'] = ds[unc_var].values.flatten()
    
    df = pd.DataFrame(data, index=time)
    df.index.name = 'time'
    
    ds.close()
    return df


def read_horwath2022(filepath: str) -> pd.DataFrame:
    """
    Read Horwath et al. (2022) ESA CCI Sea Level Budget Closure data.
    
    This dataset provides GRACE/GRACE-FO derived mass contributions to sea level
    and budget closure analysis from the ESA Climate Change Initiative.
    
    Parameters
    ----------
    filepath : str
        Path to the CSV file (horwath2021_ESACCI_SLBC_v2_2.csv)
        
    Returns
    -------
    pd.DataFrame
        DataFrame with datetime index and columns for GMSL and budget components.
        Key columns:
        - gmsl, gmsl_unc: Total GMSL from altimetry
        - steric_dieng, steric_dieng_unc: Steric contribution (Dieng ensemble)
        - steric_slbc, steric_slbc_unc: Steric contribution (SLBC_cci product)
        - omc_global, omc_global_unc: Ocean mass contribution (global)
        - glaciers, glaciers_unc: Global glacier contribution
        - greenland_total, greenland_total_unc: Greenland (altimetry + peripheral)
        - greenland_grace, greenland_grace_unc: Greenland (GRACE)
        - antarctica_altimetry, antarctica_altimetry_unc: Antarctica (altimetry)
        - antarctica_grace, antarctica_grace_unc: Antarctica (GRACE)
        - tws, tws_unc: Terrestrial water storage
        
    Note
    ----
    Seasonal components are already removed. Reference period: 2006-01 to 2015-12.
    Missing values are set to NaN (original file uses -999.99).
        
    Reference
    ---------
    Horwath, M., et al. (2022). Global sea-level budget and ocean-mass budget, 
    with a focus on advanced data products and uncertainty characterisation. 
    Earth System Science Data, 14(2), 411–447.
    https://doi.org/10.5194/essd-14-411-2022
    """
    with open(filepath, 'r') as f:
        lines = f.readlines()
    
    # Find the data section (BADC-CSV format has 'data' marker)
    data_start = None
    for i, line in enumerate(lines):
        if line.strip() == 'data':
            data_start = i + 2  # Skip 'data' line and column numbers line
            break
    
    if data_start is None:
        raise ValueError("Could not find 'data' marker in BADC-CSV file")
    
    # Column mapping based on long_name metadata in file header
    column_mapping = {
        0: 'decimal_year',
        1: 'gmsl',
        2: 'gmsl_unc',
        3: 'steric_slbc',
        4: 'steric_slbc_unc',
        5: 'steric_deep',
        6: 'steric_deep_unc',
        7: 'steric_dieng',
        8: 'steric_dieng_unc',
        9: 'omc_global',
        10: 'omc_global_unc',
        11: 'omc_65',
        12: 'omc_65_unc',
        13: 'glaciers',
        14: 'glaciers_unc',
        15: 'greenland_altimetry',
        16: 'greenland_altimetry_unc',
        17: 'greenland_periph',
        18: 'greenland_periph_unc',
        19: 'greenland_total',
        20: 'greenland_total_unc',
        21: 'greenland_grace',
        22: 'greenland_grace_unc',
        23: 'antarctica_altimetry',
        24: 'antarctica_altimetry_unc',
        25: 'antarctica_grace',
        26: 'antarctica_grace_unc',
        27: 'tws',
        28: 'tws_unc',
        29: 'sum_mass_altimetry',
        30: 'sum_mass_altimetry_unc',
        31: 'sum_mass_grace',
        32: 'sum_mass_grace_unc'
    }
    
    # Parse data
    data = []
    for line in lines[data_start:]:
        parts = line.strip().split(',')
        if len(parts) > 1:
            try:
                row = {}
                for idx, name in column_mapping.items():
                    if idx < len(parts):
                        val = float(parts[idx].strip())
                        # Handle missing values (file uses -999.99)
                        if val <= -999:
                            val = np.nan
                        row[name] = val
                data.append(row)
            except (ValueError, IndexError):
                continue
    
    df = pd.DataFrame(data)
    df['time'] = pd.to_datetime([decimal_year_to_datetime(y) for y in df['decimal_year']])
    df = df.set_index('time')
    df.index.name = 'time'
    
    return df


# =============================================================================
# GLOBAL MEAN SURFACE TEMPERATURE (GMST) DATA READERS
# =============================================================================

def read_berkeley_earth(filepath: str, use_air_temp: bool = True) -> pd.DataFrame:
    """
    Read Berkeley Earth global mean surface temperature data.
    
    Parameters
    ----------
    filepath : str
        Path to the text file (berkEarth_globalmean_airTaboveseaice.txt)
    use_air_temp : bool, default True
        If True, use air temperature above sea ice version.
        If False, use water temperature below sea ice version.
        (Note: Current implementation reads air temp section only)
        
    Returns
    -------
    pd.DataFrame
        DataFrame with datetime index and columns:
        - temperature: Temperature anomaly (°C, relative to 1951-1980)
        - temperature_unc: Uncertainty (°C, 95% CI)
        
    Reference
    ---------
    Rohde, R. A. & Hausfather, Z. (2020). The Berkeley Earth Land/Ocean Temperature 
    Record. Earth System Science Data, 12(4), 3469–3479.
    https://doi.org/10.5194/essd-12-3469-2020
    """
    with open(filepath, 'r') as f:
        lines = f.readlines()
    
    # Find start of data (first non-comment, non-empty line)
    data_start = 0
    for i, line in enumerate(lines):
        stripped = line.strip()
        if stripped and not stripped.startswith('%'):
            data_start = i
            break
    
    # Parse data lines
    data = []
    for line in lines[data_start:]:
        stripped = line.strip()
        if stripped and not stripped.startswith('%'):
            parts = stripped.split()
            if len(parts) >= 4:
                try:
                    year = int(parts[0])
                    month = int(parts[1])
                    anomaly = float(parts[2])
                    unc = float(parts[3])
                    data.append({
                        'year': year,
                        'month': month,
                        'temperature': anomaly,
                        'temperature_unc': unc
                    })
                except ValueError:
                    continue
    
    df = pd.DataFrame(data)
    df['time'] = pd.to_datetime(df['year'].astype(str) + '-' + df['month'].astype(str) + '-01')
    df = df.set_index('time')
    df = df[['temperature', 'temperature_unc']]
    df.index.name = 'time'
    
    return df


def read_hadcrut5(filepath: str) -> pd.DataFrame:
    """
    Read HadCRUT5 global mean surface temperature data.
    
    Parameters
    ----------
    filepath : str
        Path to the CSV file (hadcrut5_global_monthly.csv)
        
    Returns
    -------
    pd.DataFrame
        DataFrame with datetime index and columns:
        - temperature: Temperature anomaly (°C, relative to 1961-1990)
        - temperature_lower: Lower 2.5% confidence bound
        - temperature_upper: Upper 97.5% confidence bound
        - temperature_unc: Symmetric uncertainty estimate (half the 95% CI width)
        
    Reference
    ---------
    Morice, C. P., Kennedy, J. J., Rayner, N. A., Winn, J. P., Hogan, E., 
    Killick, R. E., Dunn, R. J. H., Osborn, T. J., Jones, P. D., & Simpson, I. R. (2021).
    An Updated Assessment of Near-Surface Temperature Change From 1850: The HadCRUT5 
    Data Set. Journal of Geophysical Research: Atmospheres, 126(3), e2019JD032361.
    https://doi.org/10.1029/2019JD032361
    """
    df = pd.read_csv(filepath)
    
    # Parse time column (format: YYYY-MM)
    df['time'] = pd.to_datetime(df['Time'] + '-01')
    df = df.set_index('time')
    
    # Rename columns
    df = df.rename(columns={
        'Anomaly (deg C)': 'temperature',
        'Lower confidence limit (2.5%)': 'temperature_lower',
        'Upper confidence limit (97.5%)': 'temperature_upper'
    })
    
    # Calculate symmetric uncertainty
    df['temperature_unc'] = (df['temperature_upper'] - df['temperature_lower']) / 2
    
    df = df[['temperature', 'temperature_unc', 'temperature_lower', 'temperature_upper']]
    df.index.name = 'time'
    
    return df


def read_nasa_gistemp(filepath: str) -> pd.DataFrame:
    """
    Read NASA GISTEMP global mean surface temperature data.
    
    Parameters
    ----------
    filepath : str
        Path to the CSV file (nasa_GLB_Ts_dSST.csv)
        
    Returns
    -------
    pd.DataFrame
        DataFrame with datetime index and columns:
        - temperature: Temperature anomaly (°C, relative to 1951-1980)
        
    Reference
    ---------
    GISTEMP Team (2024). GISS Surface Temperature Analysis (GISTEMP), version 4.
    NASA Goddard Institute for Space Studies.
    https://data.giss.nasa.gov/gistemp/
    
    Lenssen, N. J. L., Schmidt, G. A., Hansen, J. E., Menne, M. J., Persin, A., 
    Ruedy, R., & Zyss, D. (2019). Improvements in the GISTEMP Uncertainty Model.
    Journal of Geophysical Research: Atmospheres, 124(12), 6307–6326.
    https://doi.org/10.1029/2018JD029522
    """
    df = pd.read_csv(filepath, skiprows=1)  # Skip "Land-Ocean: Global Means" header
    
    # Melt monthly columns to long format
    months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
              'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    
    df_long = df.melt(id_vars=['Year'], value_vars=months, 
                       var_name='month', value_name='temperature')
    
    # Convert month names to numbers
    month_map = {m: i+1 for i, m in enumerate(months)}
    df_long['month_num'] = df_long['month'].map(month_map)
    
    # Create datetime
    df_long['time'] = pd.to_datetime(
        df_long['Year'].astype(str) + '-' + df_long['month_num'].astype(str) + '-01'
    )
    
    # Handle missing values (marked as ***)
    df_long['temperature'] = pd.to_numeric(df_long['temperature'], errors='coerce')
    
    df_long = df_long.set_index('time').sort_index()
    df_long = df_long[['temperature']]
    df_long.index.name = 'time'
    
    return df_long


def read_noaa_globaltemp(filepath: str) -> pd.DataFrame:
    """
    Read NOAA Global Temperature Anomalies data.
    
    Parameters
    ----------
    filepath : str
        Path to the text file (noaa_atmoT_landocean.txt)
        
    Returns
    -------
    pd.DataFrame
        DataFrame with datetime index (annual, at July 1) and columns:
        - temperature: Temperature anomaly (°C, relative to 1901-2000)
        
    Note
    ----
    This dataset has annual resolution only.
        
    Reference
    ---------
    NOAA National Centers for Environmental Information (NCEI).
    Global Surface Temperature Anomalies.
    https://www.ncei.noaa.gov/access/monitoring/global-temperature-anomalies/
    
    Vose, R. S., Arndt, D., Banzon, V. F., Easterling, D. R., Gleason, B., 
    Huang, B., Kearns, E., Lawrimore, J. H., Menne, M. J., Peterson, T. C., 
    Reynolds, R. W., Smith, T. M., Williams, C. N., & Wuertz, D. B. (2012).
    NOAA's Merged Land–Ocean Surface Temperature Analysis.
    Bulletin of the American Meteorological Society, 93(11), 1677–1685.
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
                    year = int(parts[0])
                    anomaly = float(parts[1])
                    data.append({'year': year, 'temperature': anomaly})
                except ValueError:
                    continue
    
    df = pd.DataFrame(data)
    # Annual data - assign to July 1 (middle of year)
    df['time'] = pd.to_datetime(df['year'].astype(str) + '-07-01')
    df = df.set_index('time')
    df = df[['temperature']]
    df.index.name = 'time'
    
    return df


# =============================================================================
# PREPROCESSING UTILITIES
# =============================================================================

def harmonize_baseline(df: pd.DataFrame, value_col: str, 
                       new_baseline: Tuple[int, int] = (1981, 2010)) -> pd.DataFrame:
    """
    Re-baseline temperature anomalies to a different reference period.
    
    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame with datetime index
    value_col : str
        Name of the temperature anomaly column
    new_baseline : tuple of int
        New reference period (start_year, end_year)
        
    Returns
    -------
    pd.DataFrame
        DataFrame with re-baselined anomalies
    """
    df = df.copy()
    
    # Calculate mean over new baseline period
    mask = (df.index.year >= new_baseline[0]) & (df.index.year <= new_baseline[1])
    baseline_mean = df.loc[mask, value_col].mean()
    
    # Subtract to re-baseline
    df[value_col] = df[value_col] - baseline_mean
    
    return df


def resample_to_monthly(df: pd.DataFrame, value_col: str = 'temperature',
                        unc_col: Optional[str] = None,
                        method: str = 'linear') -> pd.DataFrame:
    """
    Resample annual data to monthly frequency using interpolation.
    
    WARNING: This does not add information - use with caution and consider
    inflating uncertainties for statistical analyses.
    
    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame with datetime index (annual frequency)
    value_col : str
        Name of the value column to interpolate
    unc_col : str, optional
        Name of the uncertainty column
    method : str
        Interpolation method ('linear', 'spline', etc.)
        
    Returns
    -------
    pd.DataFrame
        Monthly resampled DataFrame
    """
    # Create monthly date range
    monthly_index = pd.date_range(df.index.min(), df.index.max(), freq='MS')
    
    # Reindex and interpolate
    df_monthly = df.reindex(df.index.union(monthly_index)).sort_index()
    df_monthly[value_col] = df_monthly[value_col].interpolate(method='time')
    
    if unc_col and unc_col in df.columns:
        # Use nearest uncertainty, but note this underestimates true uncertainty
        df_monthly[unc_col] = df_monthly[unc_col].interpolate(method='nearest')
        # Add flag column indicating interpolated points
        df_monthly['interpolated'] = ~df_monthly.index.isin(df.index)
    
    # Keep only monthly points
    df_monthly = df_monthly.loc[monthly_index]
    
    return df_monthly


def compute_thermodynamic_signal(df_frederikse: pd.DataFrame) -> pd.DataFrame:
    """
    Compute the thermodynamic sea level signal from Frederikse et al. (2020) data.
    
    Thermodynamic = Steric + Glaciers + Greenland + Antarctica - TWS
    
    Parameters
    ----------
    df_frederikse : pd.DataFrame
        DataFrame from read_frederikse2020()
        
    Returns
    -------
    pd.DataFrame
        DataFrame with thermodynamic signal and its components
    """
    df = df_frederikse.copy()
    
    # Compute thermodynamic signal
    df['barystatic'] = df['glaciers'] + df['greenland'] + df['antarctica']
    df['thermodynamic'] = df['steric'] + df['barystatic'] - df['tws']
    
    # Propagate uncertainties (assuming independence)
    df['barystatic_unc'] = np.sqrt(
        df['glaciers_unc']**2 + df['greenland_unc']**2 + df['antarctica_unc']**2
    )
    df['thermodynamic_unc'] = np.sqrt(
        df['steric_unc']**2 + df['barystatic_unc']**2 + df['tws_unc']**2
    )
    
    return df[['thermodynamic', 'thermodynamic_unc', 'steric', 'steric_unc',
               'barystatic', 'barystatic_unc', 'glaciers', 'glaciers_unc',
               'greenland', 'greenland_unc', 'antarctica', 'antarctica_unc',
               'tws', 'tws_unc']]
