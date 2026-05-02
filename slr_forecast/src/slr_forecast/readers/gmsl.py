"""
GMSL (Global Mean Sea Level) data readers.

All readers return DataFrames in SLR convention (positive = rise),
internal units (meters), with metadata tags.

Datasets
--------
- NASA GSFC satellite altimetry (TOPEX/Jason/Sentinel-6)
- Frederikse et al. (2020) budget-closed reconstruction
- Dangendorf et al. (2024) Kalman smoother reconstruction
- Horwath et al. (2022) ESA CCI sea level budget closure
- IPCC AR6 observed GMSL
"""

from __future__ import annotations

import os
import re
from typing import Optional

import numpy as np
import pandas as pd

from slr_forecast.config import Z_90

from slr_forecast.units import (
    MM_TO_M,
    tag_units,
    tag_sign_convention,
    tag_baseline,
    assert_sigma_positive,
    assert_units_meters,
)
from slr_forecast.readers._utils import decimal_year_to_datetime, datetime_to_decimal_year


# =========================================================================
#  NASA GSFC satellite altimetry
# =========================================================================

def read_nasa_gmsl(filepath: str) -> pd.DataFrame:
    """Read NASA GSFC GMSL from satellite altimetry.

    Returns DataFrame with datetime index, all values in **meters**.
    Columns: gmsl, gmsl_sigma, gmsl_smoothed, gmsl_nogia, decimal_year.

    Reference
    ---------
    Beckley et al. (2017), doi:10.1002/2017JC013090
    Data: doi:10.5067/GMSLM-TJ152
    """
    with open(filepath, "r") as f:
        lines = f.readlines()

    # Find header end
    data_start = 0
    for i, line in enumerate(lines):
        if "Header_End" in line:
            data_start = i + 1
            break
        if not line.startswith("HDR") and line.strip():
            try:
                float(line.split()[0])
                data_start = i
                break
            except ValueError:
                pass

    data = []
    for line in lines[data_start:]:
        parts = line.split()
        if len(parts) >= 12:
            try:
                data.append({
                    "decimal_year": float(parts[2]),
                    "gmsl_nogia": float(parts[5]),
                    "gmsl": float(parts[8]),
                    "gmsl_sigma": float(parts[9]),
                    "gmsl_smoothed": float(parts[11]),
                })
            except (ValueError, IndexError):
                continue

    df = pd.DataFrame(data)
    df["time"] = pd.to_datetime(
        [decimal_year_to_datetime(y) for y in df["decimal_year"]]
    )
    df = df.set_index("time")
    df.index.name = "time"

    # Convert mm → meters
    for col in ["gmsl", "gmsl_sigma", "gmsl_smoothed", "gmsl_nogia"]:
        df[col] = df[col] * MM_TO_M

    # Metadata
    tag_sign_convention(df, "slr")
    tag_units(df, {
        "gmsl": "m", "gmsl_sigma": "m",
        "gmsl_smoothed": "m", "gmsl_nogia": "m",
        "decimal_year": "yr",
    })
    df.attrs["dataset"] = "nasa_gmsl"
    df.attrs["reference"] = "Beckley et al. (2017)"
    df.attrs["doi"] = "10.1002/2017JC013090"

    # Guards
    assert_sigma_positive(df["gmsl_sigma"], "nasa_gmsl.gmsl_sigma")
    assert_units_meters(df["gmsl"], "nasa_gmsl.gmsl")

    return df


# =========================================================================
#  Frederikse et al. (2020)
# =========================================================================

def read_frederikse2020(filepath: str) -> pd.DataFrame:
    """Read Frederikse et al. (2020) global sea level budget data.

    Returns DataFrame with datetime index (annual, Jul 1), all values
    in **meters**.  Includes component columns (steric, glaciers,
    greenland, antarctica, tws) with *_sigma uncertainty.

    Reference
    ---------
    Frederikse et al. (2020), Nature, doi:10.1038/s41586-020-2591-3
    """
    df = pd.read_excel(filepath, sheet_name="Global")
    df = df.rename(columns={"Unnamed: 0": "year"})

    df["time"] = pd.to_datetime(df["year"].astype(int).astype(str) + "-07-01")
    df = df.set_index("time")
    df.index.name = "time"

    rename_map = {
        "Observed GMSL [mean]": "gmsl",
        "Observed GMSL [lower]": "gmsl_lower",
        "Observed GMSL [upper]": "gmsl_upper",
        "Sum of contributors [mean]": "sum_contributors",
        "Sum of contributors [lower]": "sum_contributors_lower",
        "Sum of contributors [upper]": "sum_contributors_upper",
        "Steric [mean]": "steric",
        "Steric [lower]": "steric_lower",
        "Steric [upper]": "steric_upper",
        "Glaciers [mean]": "glaciers",
        "Glaciers [lower]": "glaciers_lower",
        "Glaciers [upper]": "glaciers_upper",
        "Greenland Ice Sheet [mean]": "greenland",
        "Greenland Ice Sheet [lower]": "greenland_lower",
        "Greenland Ice Sheet [upper]": "greenland_upper",
        "Antarctic Ice Sheet [mean]": "antarctica",
        "Antarctic Ice Sheet [lower]": "antarctica_lower",
        "Antarctic Ice Sheet [upper]": "antarctica_upper",
        "Terrestrial Water Storage [mean]": "tws",
        "Terrestrial Water Storage [lower]": "tws_lower",
        "Terrestrial Water Storage [upper]": "tws_upper",
        "Reservoir impoundment [mean]": "reservoir",
        "Reservoir impoundment [lower]": "reservoir_lower",
        "Reservoir impoundment [upper]": "reservoir_upper",
        "Groundwater depletion [mean]": "groundwater",
        "Groundwater depletion [lower]": "groundwater_lower",
        "Groundwater depletion [upper]": "groundwater_upper",
        "Natural TWS [mean]": "tws_natural",
        "Natural TWS [lower]": "tws_natural_lower",
        "Natural TWS [upper]": "tws_natural_upper",
        "Altimetry [mean]": "altimetry",
        "Altimetry [lower]": "altimetry_lower",
        "Altimetry [upper]": "altimetry_upper",
    }
    df = df.rename(columns=rename_map)

    # Compute 1-sigma from 90% CI (5th–95th)
    for var in [
        "gmsl", "steric", "glaciers", "greenland", "antarctica", "tws",
        "sum_contributors", "reservoir", "groundwater", "tws_natural", "altimetry",
    ]:
        lo, hi = f"{var}_lower", f"{var}_upper"
        if lo in df.columns and hi in df.columns:
            df[f"{var}_sigma"] = (df[hi] - df[lo]) / (2 * Z_90)

    # Convert mm → meters
    value_cols = [c for c in df.columns if c not in ("year", "decimal_year")]
    df[value_cols] = df[value_cols] * MM_TO_M

    # Metadata
    tag_sign_convention(df, "slr")
    tag_units(df, {c: "m" for c in value_cols})
    df.attrs["dataset"] = "frederikse2020"
    df.attrs["reference"] = "Frederikse et al. (2020)"
    df.attrs["doi"] = "10.1038/s41586-020-2591-3"

    # Guards
    assert_units_meters(df["gmsl"], "frederikse2020.gmsl")

    return df


# =========================================================================
#  Dangendorf et al. (2024) Kalman smoother
# =========================================================================

def read_dangendorf2024(filepath: str) -> pd.DataFrame:
    """Read Dangendorf et al. (2024) Kalman Smoother GMSL reconstruction.

    Returns DataFrame with datetime index (annual, 1900–2021),
    all values in **meters** (native unit).

    Columns: gmsl, gmsl_sigma, steric, steric_sigma, barystatic,
    barystatic_sigma, gia, gia_sigma, decimal_year.

    Reference
    ---------
    Dangendorf et al. (2024), ESSD, doi:10.5194/essd-16-3471-2024
    """
    import xarray as xr

    ds = xr.open_dataset(filepath)

    time_decimal = ds["t"].values.flatten()
    time_index = pd.to_datetime(
        [decimal_year_to_datetime(t) for t in time_decimal]
    )

    def _safe(var_name):
        if var_name not in ds:
            return np.full(len(time_decimal), np.nan)
        data = ds[var_name].values.flatten().astype(np.float64)
        if data.size == 0:
            return np.full(len(time_decimal), np.nan)
        data[np.abs(data) > 1e20] = np.nan
        if hasattr(ds[var_name], "_FillValue"):
            fv = float(ds[var_name]._FillValue)
            data[np.isclose(data, fv, rtol=1e-5)] = np.nan
        return data

    df = pd.DataFrame(
        {
            "gmsl": _safe("GMSLHR"),
            "gmsl_sigma": _safe("GMSLHRSE"),
            "steric": _safe("GMSSLHR"),
            "steric_sigma": _safe("GMSSLHRSE"),
            "barystatic": _safe("GBSLHR"),
            "barystatic_sigma": _safe("GBSLHRSE"),
            "gia": _safe("GGIAHR"),
            "gia_sigma": _safe("GGIAHRSE"),
            "decimal_year": time_decimal,
        },
        index=time_index,
    )
    df.index.name = "time"
    ds.close()

    # Already in meters (native NetCDF units)
    tag_sign_convention(df, "slr")
    sl_cols = ["gmsl", "gmsl_sigma", "steric", "steric_sigma",
               "barystatic", "barystatic_sigma", "gia", "gia_sigma"]
    tag_units(df, {**{c: "m" for c in sl_cols}, "decimal_year": "yr"})
    df.attrs["dataset"] = "dangendorf2024"
    df.attrs["reference"] = "Dangendorf et al. (2024)"
    df.attrs["doi"] = "10.5194/essd-16-3471-2024"

    assert_units_meters(df["gmsl"].dropna(), "dangendorf2024.gmsl")

    return df


# =========================================================================
#  Horwath et al. (2022) ESA CCI
# =========================================================================

def read_horwath2022(filepath: str) -> pd.DataFrame:
    """Read Horwath et al. (2022) ESA CCI sea level budget closure.

    Returns DataFrame with datetime index (monthly), all values in **meters**.

    Reference
    ---------
    Horwath et al. (2022), ESSD, doi:10.5194/essd-14-411-2022
    """
    with open(filepath, "r") as f:
        lines = f.readlines()

    data_start = None
    for i, line in enumerate(lines):
        if line.strip() == "data":
            data_start = i + 2
            break
    if data_start is None:
        raise ValueError("Could not find 'data' marker in BADC-CSV file")

    columns = {
        0: "decimal_year",
        1: "gmsl", 2: "gmsl_sigma",
        3: "steric_slbc", 4: "steric_slbc_sigma",
        5: "steric_deep", 6: "steric_deep_sigma",
        7: "steric_dieng", 8: "steric_dieng_sigma",
        9: "omc_global", 10: "omc_global_sigma",
        11: "omc_65", 12: "omc_65_sigma",
        13: "glaciers", 14: "glaciers_sigma",
        15: "greenland_altimetry", 16: "greenland_altimetry_sigma",
        17: "greenland_peripheral", 18: "greenland_peripheral_sigma",
        19: "greenland", 20: "greenland_sigma",
        21: "greenland_grace", 22: "greenland_grace_sigma",
        23: "antarctica_altimetry", 24: "antarctica_altimetry_sigma",
        25: "antarctica_grace", 26: "antarctica_grace_sigma",
        27: "tws", 28: "tws_sigma",
        29: "sum_mass_altimetry", 30: "sum_mass_altimetry_sigma",
        31: "sum_mass_grace", 32: "sum_mass_grace_sigma",
        33: "sum_steric_dieng_mass_alt", 34: "sum_steric_dieng_mass_alt_sigma",
        35: "sum_steric_slbc_mass_alt", 36: "sum_steric_slbc_mass_alt_sigma",
        37: "sum_steric_dieng_omc_grace", 38: "sum_steric_dieng_omc_grace_sigma",
        39: "sum_steric_slbc_omc_grace", 40: "sum_steric_slbc_omc_grace_sigma",
    }

    data = []
    for line in lines[data_start:]:
        parts = line.strip().split(",")
        if len(parts) > 1:
            try:
                row = {}
                for idx, name in columns.items():
                    if idx < len(parts):
                        val = float(parts[idx].strip())
                        row[name] = np.nan if val <= -999 else val
                data.append(row)
            except (ValueError, IndexError):
                continue

    df = pd.DataFrame(data)
    df["time"] = pd.to_datetime(
        [decimal_year_to_datetime(y) for y in df["decimal_year"]]
    )
    df = df.set_index("time")
    df.index.name = "time"

    # Convert mm → meters
    value_cols = [c for c in df.columns if c != "decimal_year"]
    df[value_cols] = df[value_cols] * MM_TO_M

    tag_sign_convention(df, "slr")
    tag_units(df, {**{c: "m" for c in value_cols}, "decimal_year": "yr"})
    df.attrs["dataset"] = "horwath2022"
    df.attrs["reference"] = "Horwath et al. (2022)"
    df.attrs["doi"] = "10.5194/essd-14-411-2022"

    return df


# =========================================================================
#  IPCC AR6 observed GMSL
# =========================================================================

def read_ipcc_ar6_observed_gmsl(filepath: str) -> pd.DataFrame:
    """Read IPCC AR6 observed global mean sea level.

    Returns DataFrame with datetime index (annual, Jul 1), values in **meters**.

    Reference
    ---------
    Fox-Kemper et al. (2021), doi:10.1017/9781009157896.011
    """
    df = pd.read_csv(filepath, encoding="utf-8-sig")
    df = df.rename(columns={
        "Year": "year",
        "Central": "gmsl",
        "17%": "gmsl_lower",
        "83%": "gmsl_upper",
    })

    df["time"] = pd.to_datetime(df["year"].astype(int).astype(str) + "-07-01")
    df = df.set_index("time")
    df.index.name = "time"

    # 17%–83% is ~1 sigma on each side for normal
    df["gmsl_sigma"] = (df["gmsl_upper"] - df["gmsl_lower"]) / 2
    df["decimal_year"] = df["year"].astype(float) + 0.5

    # Already in meters (native)
    tag_sign_convention(df, "slr")
    sl_cols = ["gmsl", "gmsl_lower", "gmsl_upper", "gmsl_sigma"]
    tag_units(df, {**{c: "m" for c in sl_cols}, "year": "yr", "decimal_year": "yr"})
    df.attrs["dataset"] = "ipcc_ar6_observed_gmsl"
    df.attrs["reference"] = "Fox-Kemper et al. (2021)"
    df.attrs["doi"] = "10.1017/9781009157896.011"

    return df
