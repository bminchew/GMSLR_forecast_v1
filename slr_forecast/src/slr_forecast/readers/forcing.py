"""
Climate forcing data readers.

Datasets
--------
- NOAA NCEI thermosteric sea level (quarterly, 1955–present)
- GloSSAC stratospheric aerosol optical depth (monthly, 1979–present)
- Mauna Loa Observatory transmission / SAOD proxy (monthly, 1958–present)
- NOAA ONI (Oceanic Nino Index, monthly, 1950–present)
- NOAA MEI.v2 (Multivariate ENSO Index, monthly, 1979–present)
- Volcanic effective radiative forcing, IGCC 2024 (annual, 1750–2024)
"""

from __future__ import annotations

import os
import re
import zipfile

import numpy as np
import pandas as pd

from slr_forecast.units import tag_units, tag_sign_convention
from slr_forecast.readers._utils import decimal_year_to_datetime, datetime_to_decimal_year


# =========================================================================
#  NOAA thermosteric sea level
# =========================================================================

def read_noaa_thermosteric_yearly(filepath: str) -> pd.DataFrame:
    """Read NOAA NCEI yearly global-mean thermosteric sea level.

    Reads the basin time series .dat files (e.g. ``a-mm-w0-700m.dat``)
    which contain yearly World Ocean thermosteric SL anomaly in mm.

    Returns DataFrame with columns:
        - tsl_mm : World Ocean thermosteric SL anomaly (mm)
        - tsl_se_mm : standard error (mm)
        - decimal_year : mid-year (e.g. 1955.5)

    Reference
    ---------
    Levitus et al. (2012), doi:10.1029/2012GL051106
    """
    df = pd.read_csv(filepath, sep=r'\s+')
    # Columns: YEAR WO WOse NH NHse SH SHse
    out = pd.DataFrame({
        'decimal_year': df['YEAR'].values,
        'tsl_mm': df['WO'].values,
        'tsl_se_mm': df['WOse'].values,
    })
    tag_units(out, {'tsl_mm': 'mm', 'tsl_se_mm': 'mm', 'decimal_year': 'yr'})
    out.attrs['dataset'] = 'noaa_thermosteric_yearly'
    out.attrs['reference'] = 'Levitus et al. (2012)'
    out.attrs['doi'] = '10.1029/2012GL051106'
    return out


def read_volcanic_erf(filepath, years) -> np.ndarray:
    """Volcanic effective radiative forcing (W m-2) at mid-year stamps.

    Reads the 'volcanic' column of the Indicators of Global Climate Change
    ERF aggregates file (rows at mid-year, 1750.5 onward) and returns its
    values at ``years`` (e.g. 1850.5, 1851.5, ...).  Years after the last row
    get zero, the AR6 SSP convention for future volcanic forcing; years before
    the first row raise.

    Reference
    ---------
    Forster et al. (2025), doi:10.5194/essd-17-2641-2025;
    data doi:10.5281/zenodo.15639576
    """
    erf = pd.read_csv(filepath, index_col=0)['volcanic']
    years = np.round(np.asarray(years, dtype=float), 1)
    if years.min() < erf.index.min():
        raise ValueError(f'volcanic ERF starts at {erf.index.min()}, requested {years.min()}')
    out = erf.reindex(years).to_numpy(dtype=float, copy=True)
    after = years > erf.index.max()
    out[after] = 0.0
    if np.isnan(out).any():
        raise ValueError('volcanic ERF: requested years are not mid-year stamps of the file')
    return out


def read_noaa_thermosteric(zip_path: str, start_year: int = 1955) -> pd.DataFrame:
    """Read NOAA NCEI thermosteric sea level from ZIP archive.

    Extracts depth range from filename. Returns quarterly DataFrame
    with values in **mm** (native, not yet standardised to meters —
    caller should convert if needed).

    Reference
    ---------
    Levitus et al. (2012), doi:10.1029/2012GL051106
    """
    filename = os.path.basename(zip_path)
    pattern = r"noaa_thermosteric_SL_(\d+-\d+)-(\w+)"
    match = re.search(pattern, filename)
    if not match:
        raise ValueError(
            f"Filename '{filename}' does not follow convention: "
            "noaa_thermosteric_SL_<depth_range>-<timeresolution>*"
        )

    depth_range = match.group(1)

    with zipfile.ZipFile(zip_path, "r") as z:
        file_list = [
            f for f in z.namelist()
            if f.endswith(".dat") and depth_range.replace("-", "_") in f.replace("-", "_")
        ]
        if not file_list:
            file_list = [f for f in z.namelist() if f.endswith(".dat")]
        if not file_list:
            raise FileNotFoundError(
                f"No .dat matrix found in {filename} for depth {depth_range}."
            )
        with z.open(file_list[0]) as f:
            content = f.read().decode("utf-8")

    raw_data = []
    for line in content.splitlines():
        line = line.strip()
        chunks = [line[i : i + 8] for i in range(0, len(line), 8)]
        for chunk in chunks:
            val_str = chunk.strip()
            if val_str:
                try:
                    raw_data.append(float(val_str))
                except ValueError:
                    continue

    values = np.array(raw_data)
    values[values == -999.999] = np.nan

    dates = []
    for i in range(len(values)):
        year = start_year + (i // 4)
        if year > 2100:
            values = values[:i]
            break
        month = [2, 5, 8, 11][i % 4]
        dates.append(pd.Timestamp(year=year, month=month, day=15))

    col_name = f"tsl_{depth_range.replace('-', '_')}_mm"
    df = pd.DataFrame({col_name: values * 10.0}, index=dates)  # cm → mm
    df.index.name = "date"

    tag_units(df, {col_name: "mm"})
    df.attrs["dataset"] = "noaa_thermosteric"
    df.attrs["reference"] = "Levitus et al. (2012)"
    df.attrs["doi"] = "10.1029/2012GL051106"
    df.attrs["depth_range"] = depth_range

    return df.dropna()


# =========================================================================
#  GloSSAC SAOD
# =========================================================================

def read_glossac(filepath: str, wavelength: int = 525) -> pd.DataFrame:
    """Read GloSSAC stratospheric aerosol optical depth.

    Returns monthly DataFrame with column ``saod`` (dimensionless,
    cos-lat weighted global mean).

    Reference
    ---------
    Thomason et al. (2018), ESSD, doi:10.5194/essd-10-469-2018
    """
    import xarray as xr

    ds = xr.open_dataset(str(filepath))
    aod = ds["Glossac_Aerosol_Optical_Depth"]
    lat = ds["lat"].values
    time_ym = ds["time"].values
    wl_values = ds["wavelengths_glossac"].values

    wl_idx = int(np.where(wl_values == wavelength)[0][0])
    aod_slice = aod.values[:, :, wl_idx]

    cos_lat = np.cos(np.radians(lat))
    global_mean = np.zeros(len(time_ym))
    for i in range(len(time_ym)):
        valid = ~np.isnan(aod_slice[i])
        if valid.any():
            global_mean[i] = np.average(aod_slice[i, valid], weights=cos_lat[valid])
        else:
            global_mean[i] = np.nan

    dates = pd.to_datetime([
        f"{int(ym // 100)}-{int(ym % 100):02d}-15" for ym in time_ym
    ])
    df = pd.DataFrame({"saod": global_mean}, index=dates)
    df.index.name = "time"
    ds.close()

    tag_units(df, {"saod": "unitless"})
    df.attrs["dataset"] = "glossac"
    df.attrs["reference"] = "Thomason et al. (2018)"
    df.attrs["doi"] = "10.5194/essd-10-469-2018"
    df.attrs["wavelength_nm"] = wavelength

    return df


# =========================================================================
#  Mauna Loa transmission → SAOD proxy
# =========================================================================

def read_mauna_loa_transmission(filepath: str) -> pd.DataFrame:
    """Read Mauna Loa Observatory apparent atmospheric transmission.

    Returns monthly DataFrame with columns: transmission (0–1),
    saod = -ln(transmission).

    Reference: NOAA Global Monitoring Laboratory.
    """
    raw = pd.read_csv(
        str(filepath), skiprows=2, sep=r"\s+",
        names=["date_label", "decimal_year", "transmission"],
    )
    dates = [decimal_year_to_datetime(dy) for dy in raw["decimal_year"]]
    df = pd.DataFrame(
        {
            "transmission": raw["transmission"].values,
            "saod": -np.log(raw["transmission"].values),
        },
        index=pd.DatetimeIndex(dates, name="time"),
    )

    tag_units(df, {"transmission": "unitless", "saod": "unitless"})
    df.attrs["dataset"] = "mauna_loa_transmission"
    df.attrs["reference"] = "NOAA Global Monitoring Laboratory"

    return df


# =========================================================================
#  ENSO indices
# =========================================================================

def read_noaa_oni(filepath: str) -> pd.DataFrame:
    """Read NOAA Oceanic Nino Index (ONI).

    Returns monthly DataFrame with column ``oni`` (degC anomaly).

    Reference: NOAA Climate Prediction Center.
    """
    df = pd.read_csv(filepath, skiprows=0)
    cols = list(df.columns)
    df = df.rename(columns={cols[0]: "date", cols[1]: "oni"})
    df["time"] = pd.to_datetime(df["date"].str.strip())
    df = df.set_index("time")
    df.index.name = "time"
    df["oni"] = pd.to_numeric(df["oni"], errors="coerce")
    df.loc[df["oni"] <= -99, "oni"] = np.nan
    df = df[["oni"]].copy()

    tag_units(df, {"oni": "degC"})
    df.attrs["dataset"] = "noaa_oni"
    df.attrs["reference"] = "NOAA Climate Prediction Center"

    return df


def read_noaa_mei(filepath: str) -> pd.DataFrame:
    """Read NOAA Multivariate ENSO Index (MEI.v2).

    Returns monthly DataFrame with column ``mei`` (dimensionless).

    Reference
    ---------
    Wolter & Timlin (2011), doi psl.noaa.gov/enso/mei/
    """
    with open(filepath, "r") as f:
        lines = f.readlines()

    records = []
    for line in lines[1:]:
        parts = line.split()
        if len(parts) < 13:
            continue
        try:
            year = int(parts[0])
        except ValueError:
            continue
        for month_idx, val_str in enumerate(parts[1:13], start=1):
            try:
                val = float(val_str)
            except ValueError:
                val = np.nan
            records.append({"year": year, "month": month_idx, "mei": val})

    df = pd.DataFrame(records)
    df["time"] = pd.to_datetime(
        df["year"].astype(str) + "-" + df["month"].astype(str).str.zfill(2) + "-15"
    )
    df = df.set_index("time")
    df.index.name = "time"
    df = df[["mei"]].copy()
    df.loc[df["mei"] <= -999, "mei"] = np.nan

    tag_units(df, {"mei": "dimensionless"})
    df.attrs["dataset"] = "noaa_mei"
    df.attrs["reference"] = "Wolter & Timlin (2011)"

    return df
