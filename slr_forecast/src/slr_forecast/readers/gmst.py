"""
GMST (Global Mean Surface Temperature) data readers.

All readers return DataFrames with datetime index, temperature in degC.

Datasets
--------
- Berkeley Earth (monthly, 1850–present)
- Berkeley Earth gridded (monthly, 1850–present, regional extraction from 1° NetCDF)
- HadCRUT5 (monthly, 1850–present)
- NASA GISTEMP (monthly, 1880–present)
- NOAA GlobalTemp (annual, 1850–present)
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from slr_forecast.config import Z_90
from slr_forecast.units import tag_units


# =========================================================================
#  Berkeley Earth
# =========================================================================

def read_berkeley_earth(filepath: str) -> pd.DataFrame:
    """Read Berkeley Earth global mean surface temperature.

    Returns monthly DataFrame with columns: temperature, temperature_unc,
    temperature_sigma (all degC, anomaly relative to 1951–1980).

    Reference
    ---------
    Rohde & Hausfather (2020), ESSD, doi:10.5194/essd-12-3469-2020
    """
    with open(filepath, "r") as f:
        lines = f.readlines()

    data_start = 0
    for i, line in enumerate(lines):
        if line.strip() and not line.strip().startswith("%"):
            data_start = i
            break

    data = []
    for line in lines[data_start:]:
        stripped = line.strip()
        if stripped and not stripped.startswith("%"):
            parts = stripped.split()
            if len(parts) >= 4:
                try:
                    data.append({
                        "year": int(parts[0]),
                        "month": int(parts[1]),
                        "temperature": float(parts[2]),
                        "temperature_unc": float(parts[3]),
                    })
                except ValueError:
                    continue

    df = pd.DataFrame(data)
    df["time"] = pd.to_datetime(
        df["year"].astype(str) + "-" + df["month"].astype(str) + "-01"
    )
    df = df.set_index("time")[["temperature", "temperature_unc"]]
    df.index.name = "time"

    # 95% CI → 1-sigma
    df["temperature_sigma"] = df["temperature_unc"] / Z_90

    tag_units(df, {
        "temperature": "degC",
        "temperature_unc": "degC",
        "temperature_sigma": "degC",
    })
    df.attrs["dataset"] = "berkeley_earth"
    df.attrs["reference"] = "Rohde & Hausfather (2020)"
    df.attrs["doi"] = "10.5194/essd-12-3469-2020"
    df.attrs["temperature_baseline"] = "1951-1980"

    return df


# =========================================================================
#  Berkeley Earth — gridded NetCDF (regional extraction)
# =========================================================================

def read_berkeley_earth_gridded(
    filepath: str,
    lat_bounds: tuple[float, float] = (60.0, 84.0),
    lon_bounds: tuple[float, float] = (-73.0, -12.0),
    land_threshold: float = 0.5,
    ocean_only: bool = False,
) -> pd.DataFrame:
    """Read Berkeley Earth 1° gridded NetCDF and extract a regional average.

    Returns an area-weighted monthly temperature anomaly for cells
    within the specified lat/lon box.  By default, weights by land
    fraction (land-dominated average).  With ``ocean_only=True``,
    weights by ``(1 - land_mask)`` to extract ocean surface temperature.

    Parameters
    ----------
    filepath : str or Path
        Path to ``berkEarth_Global_TAVG_Gridded_1deg.nc`` (or the
        high-resolution derivative).
    lat_bounds : tuple of float
        (lat_min, lat_max) in degrees north.  Default (60, 84) for Greenland.
    lon_bounds : tuple of float
        (lon_min, lon_max) in degrees east.  Default (-73, -12) for Greenland.
    land_threshold : float
        Minimum ``land_mask`` fraction for a cell to be included in land
        mode.  Default 0.5.  Ignored when ``ocean_only=True``.
    ocean_only : bool
        If True, weight cells by ``(1 - land_mask) * areal_weight``,
        giving an ocean-dominated average.  Cells with ``land_mask >= 1``
        are excluded.  Default False.

    Returns
    -------
    pd.DataFrame
        DataFrame with datetime index and columns:
        - temperature : area-weighted anomaly (degC, rel. 1951–1980)
        - decimal_year : fractional year for each month

    Reference
    ---------
    Rohde & Hausfather (2020), ESSD, doi:10.5194/essd-12-3469-2020
    """
    import netCDF4 as nc

    ds = nc.Dataset(str(filepath))

    lat = ds.variables["latitude"][:]
    lon = ds.variables["longitude"][:]
    time = ds.variables["time"][:]              # decimal year (mid-month)
    temp = ds.variables["temperature"][:]       # (time, lat, lon)
    land_mask = ds.variables["land_mask"][:]    # (lat, lon)
    areal_wt = ds.variables["areal_weight"][:]  # (lat, lon)
    ds.close()

    # Region mask
    lat_mask = (lat >= lat_bounds[0]) & (lat <= lat_bounds[1])
    lon_mask = (lon >= lon_bounds[0]) & (lon <= lon_bounds[1])

    lm = land_mask[np.ix_(lat_mask, lon_mask)]
    aw = areal_wt[np.ix_(lat_mask, lon_mask)]

    # Cell weighting
    if ocean_only:
        ocean_frac = 1.0 - lm
        weight = np.where(ocean_frac > 0, ocean_frac * aw, 0.0)
    else:
        weight = np.where(lm >= land_threshold, lm * aw, 0.0)

    # Area-weighted mean for each time step (NaN-safe)
    temp_region = temp[:, lat_mask, :][:, :, lon_mask]
    valid = ~np.isnan(temp_region)
    weighted_temp = np.where(valid, temp_region * weight[np.newaxis, :, :], 0.0)
    effective_weight = np.where(valid, weight[np.newaxis, :, :], 0.0)
    ts = weighted_temp.sum(axis=(1, 2)) / effective_weight.sum(axis=(1, 2))

    # Decimal year → datetime (time values are mid-month: Jan = 1/24, etc.)
    dates = pd.to_datetime([
        f"{int(y)}-{max(1, min(12, int((y - int(y)) * 12) + 1)):02d}-01"
        for y in time
    ])

    df = pd.DataFrame(
        {"temperature": ts, "decimal_year": time},
        index=dates,
    )
    df.index.name = "time"

    tag_units(df, {"temperature": "degC", "decimal_year": "yr"})
    df.attrs["dataset"] = "berkeley_earth_gridded"
    df.attrs["reference"] = "Rohde & Hausfather (2020)"
    df.attrs["doi"] = "10.5194/essd-12-3469-2020"
    df.attrs["temperature_baseline"] = "1951-1980"
    df.attrs["region_lat"] = lat_bounds
    df.attrs["region_lon"] = lon_bounds
    df.attrs["land_threshold"] = land_threshold
    df.attrs["ocean_only"] = ocean_only
    df.attrs["n_cells"] = int((weight > 0).sum())

    return df


# =========================================================================
#  HadCRUT5
# =========================================================================

def read_hadcrut5(filepath: str) -> pd.DataFrame:
    """Read HadCRUT5 global mean surface temperature.

    Returns monthly DataFrame with columns: temperature, temperature_unc,
    temperature_sigma, temperature_lower, temperature_upper (all degC,
    anomaly relative to 1961–1990).

    Reference
    ---------
    Morice et al. (2021), JGR, doi:10.1029/2019JD032361
    """
    df = pd.read_csv(filepath)
    df["time"] = pd.to_datetime(df["Time"] + "-01")
    df = df.set_index("time")

    df = df.rename(columns={
        "Anomaly (deg C)": "temperature",
        "Lower confidence limit (2.5%)": "temperature_lower",
        "Upper confidence limit (97.5%)": "temperature_upper",
    })
    df["temperature_unc"] = (df["temperature_upper"] - df["temperature_lower"]) / 2
    # 95% CI → 1-sigma (z = 1.96)
    df["temperature_sigma"] = df["temperature_unc"] / 1.96
    df.index.name = "time"

    tag_units(df, {c: "degC" for c in [
        "temperature", "temperature_unc", "temperature_sigma",
        "temperature_lower", "temperature_upper",
    ]})
    df.attrs["dataset"] = "hadcrut5"
    df.attrs["reference"] = "Morice et al. (2021)"
    df.attrs["doi"] = "10.1029/2019JD032361"
    df.attrs["temperature_baseline"] = "1961-1990"

    return df[["temperature", "temperature_unc", "temperature_sigma",
               "temperature_lower", "temperature_upper"]]


# =========================================================================
#  NASA GISTEMP
# =========================================================================

def read_nasa_gistemp(filepath: str) -> pd.DataFrame:
    """Read NASA GISTEMP global mean surface temperature.

    Returns monthly DataFrame with column: temperature (degC, anomaly
    relative to 1951–1980).

    Reference
    ---------
    Lenssen et al. (2019), JGR, doi:10.1029/2018JD029522
    """
    df = pd.read_csv(filepath, skiprows=1)
    months = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
              "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]

    df_long = df.melt(
        id_vars=["Year"], value_vars=months,
        var_name="month", value_name="temperature",
    )
    month_map = {m: i + 1 for i, m in enumerate(months)}
    df_long["month_num"] = df_long["month"].map(month_map)
    df_long["time"] = pd.to_datetime(
        df_long["Year"].astype(str) + "-" +
        df_long["month_num"].astype(str) + "-01"
    )
    df_long["temperature"] = pd.to_numeric(df_long["temperature"], errors="coerce")
    df_long = df_long.set_index("time").sort_index()[["temperature"]]
    df_long.index.name = "time"

    tag_units(df_long, {"temperature": "degC"})
    df_long.attrs["dataset"] = "nasa_gistemp"
    df_long.attrs["reference"] = "Lenssen et al. (2019)"
    df_long.attrs["doi"] = "10.1029/2018JD029522"
    df_long.attrs["temperature_baseline"] = "1951-1980"

    return df_long


# =========================================================================
#  NOAA GlobalTemp
# =========================================================================

def read_noaa_globaltemp(filepath: str) -> pd.DataFrame:
    """Read NOAA Global Temperature Anomalies.

    Returns annual DataFrame (Jul 1) with column: temperature (degC,
    anomaly relative to 1901–2000).

    Reference
    ---------
    Vose et al. (2021), doi:10.1029/2020GL090873
    """
    with open(filepath, "r") as f:
        lines = f.readlines()

    data = []
    for line in lines:
        stripped = line.strip()
        if stripped and not stripped.startswith("#"):
            parts = stripped.split(",")
            if len(parts) >= 2:
                try:
                    data.append({
                        "year": int(parts[0]),
                        "temperature": float(parts[1]),
                    })
                except ValueError:
                    continue

    df = pd.DataFrame(data)
    df["time"] = pd.to_datetime(df["year"].astype(str) + "-07-01")
    df = df.set_index("time")[["temperature"]]
    df.index.name = "time"

    tag_units(df, {"temperature": "degC"})
    df.attrs["dataset"] = "noaa_globaltemp"
    df.attrs["reference"] = "Vose et al. (2021)"
    df.attrs["doi"] = "10.1029/2020GL090873"
    df.attrs["temperature_baseline"] = "1901-2000"

    return df
