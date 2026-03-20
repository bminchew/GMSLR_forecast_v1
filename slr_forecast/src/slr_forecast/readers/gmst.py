"""
GMST (Global Mean Surface Temperature) data readers.

All readers return DataFrames with datetime index, temperature in degC.

Datasets
--------
- Berkeley Earth (monthly, 1850–present)
- HadCRUT5 (monthly, 1850–present)
- NASA GISTEMP (monthly, 1880–present)
- NOAA GlobalTemp (annual, 1850–present)
"""

from __future__ import annotations

import numpy as np
import pandas as pd

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
    df["temperature_sigma"] = df["temperature_unc"] / 1.645

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
