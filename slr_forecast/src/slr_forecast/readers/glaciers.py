"""
Glacier mass balance data readers.

All readers return DataFrames in SLR convention (positive = sea level
rise contribution from glacier mass loss), internal units (meters).

Datasets
--------
- GlaMBIE global consensus (annual, 2000–2023)
- GlaMBIE regional by RGI region
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from slr_forecast.units import (
    GT_TO_M_SLE,
    tag_units,
    tag_sign_convention,
    assert_sigma_positive,
)
from slr_forecast.readers._utils import decimal_year_to_datetime


# RGI region lookup
GLAMBIE_REGIONS: dict[int, str] = {
    0: "global", 1: "alaska", 2: "western_canada_us",
    3: "arctic_canada_north", 4: "arctic_canada_south",
    5: "greenland_periphery", 6: "iceland", 7: "svalbard",
    8: "scandinavia", 9: "russian_arctic", 10: "north_asia",
    11: "central_europe", 12: "caucasus_middle_east",
    13: "central_asia", 14: "south_asia_west", 15: "south_asia_east",
    16: "low_latitudes", 17: "southern_andes", 18: "new_zealand",
    19: "antarctic_and_subantarctic",
}


def read_glambie_global(filepath: str) -> pd.DataFrame:
    """Read GlaMBIE global glacier consensus mass balance.

    Sign flip applied: raw negative Gt = mass loss → positive SLR.

    Returns annual DataFrame (2000–2023) with columns:
    mass_balance (m/yr SLE), mass_balance_sigma, mass_balance_mwe,
    mass_balance_mwe_sigma, glacier_area (km²), decimal_year.

    Reference
    ---------
    Zemp et al. (2024), doi:10.5904/wgms-glambie-2024-07
    """
    df = pd.read_csv(filepath)

    mid_year = (df["start_dates"] + df["end_dates"]) / 2.0
    df["decimal_year"] = mid_year
    df["time"] = pd.to_datetime(
        [decimal_year_to_datetime(y) for y in mid_year]
    )
    df = df.set_index("time")
    df.index.name = "time"

    df = df.rename(columns={
        "combined_gt": "mass_balance",
        "combined_gt_errors": "mass_balance_sigma",
        "combined_mwe": "mass_balance_mwe",
        "combined_mwe_errors": "mass_balance_mwe_sigma",
    })

    # Sign flip: negative Gt = mass loss → positive SLR contribution
    df["mass_balance"] = -df["mass_balance"]
    df["mass_balance_mwe"] = -df["mass_balance_mwe"]

    # Sigma: ensure positive
    df["mass_balance_sigma"] = np.abs(df["mass_balance_sigma"])
    df["mass_balance_mwe_sigma"] = np.abs(df["mass_balance_mwe_sigma"])

    df = df.drop(columns=["region", "start_dates", "end_dates"], errors="ignore")

    # Convert Gt → meters SLE
    for col in ["mass_balance", "mass_balance_sigma"]:
        df[col] = df[col] * GT_TO_M_SLE

    tag_sign_convention(df, "slr")
    tag_units(df, {
        "mass_balance": "m/yr", "mass_balance_sigma": "m/yr",
        "mass_balance_mwe": "m.w.e./yr", "mass_balance_mwe_sigma": "m.w.e./yr",
        "glacier_area": "km^2", "decimal_year": "yr",
    })
    df.attrs["dataset"] = "glambie_global"
    df.attrs["reference"] = "Zemp et al. (2024)"
    df.attrs["doi"] = "10.5904/wgms-glambie-2024-07"

    assert_sigma_positive(df["mass_balance_sigma"], "glambie_global.sigma")

    return df


def read_glambie_regional(filepath: str) -> pd.DataFrame:
    """Read a GlaMBIE regional glacier mass balance CSV.

    Same structure and sign convention as ``read_glambie_global()``.
    Region name extracted from the 'region' column.

    Reference
    ---------
    Zemp et al. (2024), doi:10.5904/wgms-glambie-2024-07
    """
    df = pd.read_csv(filepath)
    region_name = df["region"].iloc[0] if "region" in df.columns else "unknown"

    mid_year = (df["start_dates"] + df["end_dates"]) / 2.0
    df["decimal_year"] = mid_year
    df["time"] = pd.to_datetime(
        [decimal_year_to_datetime(y) for y in mid_year]
    )
    df = df.set_index("time")
    df.index.name = "time"

    df = df.rename(columns={
        "combined_gt": "mass_balance",
        "combined_gt_errors": "mass_balance_sigma",
        "combined_mwe": "mass_balance_mwe",
        "combined_mwe_errors": "mass_balance_mwe_sigma",
    })
    df["mass_balance"] = -df["mass_balance"]
    df["mass_balance_mwe"] = -df["mass_balance_mwe"]
    df["mass_balance_sigma"] = np.abs(df["mass_balance_sigma"])
    df["mass_balance_mwe_sigma"] = np.abs(df["mass_balance_mwe_sigma"])

    df = df.drop(columns=["region", "start_dates", "end_dates"], errors="ignore")

    for col in ["mass_balance", "mass_balance_sigma"]:
        df[col] = df[col] * GT_TO_M_SLE

    tag_sign_convention(df, "slr")
    tag_units(df, {
        "mass_balance": "m/yr", "mass_balance_sigma": "m/yr",
        "mass_balance_mwe": "m.w.e./yr", "mass_balance_mwe_sigma": "m.w.e./yr",
        "glacier_area": "km^2", "decimal_year": "yr",
    })
    df.attrs["dataset"] = f"glambie_{region_name}"
    df.attrs["region"] = region_name
    df.attrs["reference"] = "Zemp et al. (2024)"
    df.attrs["doi"] = "10.5904/wgms-glambie-2024-07"

    return df
