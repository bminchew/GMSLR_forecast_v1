"""
Ice sheet mass balance data readers.

**Sign convention fix applied here**: IMBIE Gt-format files use
glaciology convention (negative = mass loss).  All readers in this
module flip to SLR convention (positive = sea level rise) at the
point of ingestion.

IMBIE mm-format files (e.g., west_antarctica_mm.csv) are already in
SLR convention but have a known data quirk: sigma columns are stored
as negative numbers.  This is corrected with np.abs() at ingestion.

Mouginot and Mankoff use glaciology convention (negative = mass loss).
Sign is flipped to SLR convention at ingestion.

Datasets
--------
- IMBIE Greenland (Gt-format, 1992–2020)
- IMBIE East Antarctica (Gt-format, 1992–2020)
- IMBIE Antarctic Peninsula (Gt-format, 1992–2020)
- IMBIE total Antarctica (Gt-format, 1992–2020)
- IMBIE all ice sheets (Gt-format, 1992–2020)
- IMBIE West Antarctica (mm-format, 1992–2020)
- Mouginot et al. (2019) Greenland (input-output, 1972–2018)
- Mankoff et al. (2021) Greenland PROMICE (input-output, 1840–present)
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from slr_forecast.units import (
    GT_TO_M_SLE,
    MM_TO_M,
    tag_units,
    tag_sign_convention,
    assert_sigma_positive,
    assert_units_meters,
)
from slr_forecast.readers._utils import decimal_year_to_datetime


# =========================================================================
#  Generic IMBIE Gt-format reader (internal)
# =========================================================================

def _read_imbie_gt(
    filepath: str,
    dataset_name: str,
    region_label: str,
) -> pd.DataFrame:
    """Read any IMBIE Gt-format CSV and return in SLR convention, meters.

    The raw IMBIE Gt files use glaciology convention: negative mass
    balance = mass loss.  This function flips the sign so that
    positive = sea level rise contribution.

    Parameters
    ----------
    filepath : str
        Path to IMBIE Gt-format CSV.
    dataset_name : str
        Value for df.attrs['dataset'].
    region_label : str
        Human-readable region name.
    """
    df = pd.read_csv(filepath)

    df = df.rename(columns={
        "Year": "decimal_year",
        "Mass balance (Gt/yr)": "mass_balance_rate",
        "Mass balance uncertainty (Gt/yr)": "mass_balance_rate_sigma",
        "Cumulative mass balance (Gt)": "cumulative_mass_balance",
        "Cumulative mass balance uncertainty (Gt)": "cumulative_mass_balance_sigma",
    })

    df["time"] = pd.to_datetime(
        [decimal_year_to_datetime(y) for y in df["decimal_year"]]
    )
    df = df.set_index("time")
    df.index.name = "time"

    # ------------------------------------------------------------------
    # SIGN FIX: glaciology → SLR convention
    # In Gt files, negative mass_balance_rate = mass loss = SLR rise.
    # Flip sign on value columns (not sigma — uncertainty is always positive).
    # ------------------------------------------------------------------
    for col in ["mass_balance_rate", "cumulative_mass_balance"]:
        df[col] = -df[col]

    # Sigma columns: ensure positive (take abs in case of data quirks)
    for col in ["mass_balance_rate_sigma", "cumulative_mass_balance_sigma"]:
        df[col] = np.abs(df[col])

    # Unit conversion: Gt → meters SLE
    value_cols = [c for c in df.columns if c != "decimal_year"]
    df[value_cols] = df[value_cols] * GT_TO_M_SLE

    # Metadata
    tag_sign_convention(df, "slr")
    tag_units(df, {
        "mass_balance_rate": "m/yr",
        "mass_balance_rate_sigma": "m/yr",
        "cumulative_mass_balance": "m",
        "cumulative_mass_balance_sigma": "m",
        "decimal_year": "yr",
    })
    df.attrs["dataset"] = dataset_name
    df.attrs["region"] = region_label
    df.attrs["reference"] = "Otosaka et al. (2023)"
    df.attrs["doi"] = "10.5194/essd-15-1597-2023"

    # Guards
    assert_sigma_positive(
        df["mass_balance_rate_sigma"], f"{dataset_name}.mass_balance_rate_sigma"
    )
    assert_sigma_positive(
        df["cumulative_mass_balance_sigma"],
        f"{dataset_name}.cumulative_mass_balance_sigma",
    )

    return df


# =========================================================================
#  Public IMBIE Gt-format readers
# =========================================================================

def read_imbie_greenland(filepath: str) -> pd.DataFrame:
    """Read IMBIE Greenland Ice Sheet mass balance (Gt-format).

    Returns monthly DataFrame (1992–2020) in SLR convention (positive
    = sea level rise), units in meters.

    Reference: Otosaka et al. (2023), doi:10.5194/essd-15-1597-2023
    """
    return _read_imbie_gt(filepath, "imbie_greenland", "Greenland Ice Sheet")


def read_imbie_east_antarctica(filepath: str) -> pd.DataFrame:
    """Read IMBIE East Antarctic Ice Sheet mass balance (Gt-format).

    Returns monthly DataFrame (1992–2020) in SLR convention, meters.
    """
    return _read_imbie_gt(filepath, "imbie_east_antarctica", "East Antarctic Ice Sheet")


def read_imbie_antarctic_peninsula(filepath: str) -> pd.DataFrame:
    """Read IMBIE Antarctic Peninsula mass balance (Gt-format).

    Returns monthly DataFrame (1992–2020) in SLR convention, meters.
    """
    return _read_imbie_gt(filepath, "imbie_antarctic_peninsula", "Antarctic Peninsula")


def read_imbie_antarctica(filepath: str) -> pd.DataFrame:
    """Read IMBIE total Antarctica mass balance (Gt-format).

    Total AIS = WAIS + EAIS + Peninsula.
    Returns monthly DataFrame (1992–2020) in SLR convention, meters.
    """
    return _read_imbie_gt(filepath, "imbie_antarctica", "Antarctic Ice Sheet (total)")


def read_imbie_all(filepath: str) -> pd.DataFrame:
    """Read IMBIE all ice sheets combined (GrIS + AIS) mass balance.

    Returns monthly DataFrame (1992–2020) in SLR convention, meters.
    """
    return _read_imbie_gt(filepath, "imbie_all", "All ice sheets (GrIS + AIS)")


# =========================================================================
#  IMBIE West Antarctica (mm-format)
# =========================================================================

def read_imbie_west_antarctica(filepath: str) -> pd.DataFrame:
    """Read IMBIE West Antarctica mass balance (mm-format CSV).

    The mm-format file is already in SLR convention (positive = rise).
    Known data quirk: sigma columns are stored as negative numbers —
    corrected with np.abs() here.

    Returns monthly DataFrame (1992–2020) in meters.

    Reference: Otosaka et al. (2023), doi:10.5194/essd-15-1597-2023
    """
    df = pd.read_csv(filepath)

    df = df.rename(columns={
        "Year": "decimal_year",
        "Mass balance (mm/yr)": "mass_balance_rate",
        "Mass balance uncertainty (mm/yr)": "mass_balance_rate_sigma",
        "Cumulative mass balance (mm)": "cumulative_mass_balance",
        "Cumulative mass balance uncertainty (mm)": "cumulative_mass_balance_sigma",
    })

    df["time"] = pd.to_datetime(
        [decimal_year_to_datetime(y) for y in df["decimal_year"]]
    )
    df = df.set_index("time")
    df.index.name = "time"

    # Fix negative sigma quirk (known IMBIE mm-format issue)
    for col in ["mass_balance_rate_sigma", "cumulative_mass_balance_sigma"]:
        df[col] = np.abs(df[col])

    # Convert mm → meters
    value_cols = [c for c in df.columns if c != "decimal_year"]
    df[value_cols] = df[value_cols] * MM_TO_M

    # Metadata
    tag_sign_convention(df, "slr")
    tag_units(df, {
        "mass_balance_rate": "m/yr",
        "mass_balance_rate_sigma": "m/yr",
        "cumulative_mass_balance": "m",
        "cumulative_mass_balance_sigma": "m",
        "decimal_year": "yr",
    })
    df.attrs["dataset"] = "imbie_west_antarctica"
    df.attrs["region"] = "West Antarctic Ice Sheet"
    df.attrs["reference"] = "Otosaka et al. (2023)"
    df.attrs["doi"] = "10.5194/essd-15-1597-2023"

    # Guards
    assert_sigma_positive(
        df["mass_balance_rate_sigma"], "imbie_wais.mass_balance_rate_sigma"
    )
    assert_sigma_positive(
        df["cumulative_mass_balance_sigma"],
        "imbie_wais.cumulative_mass_balance_sigma",
    )
    assert_units_meters(
        df["cumulative_mass_balance"], "imbie_wais.cumulative_mass_balance"
    )

    return df


# =========================================================================
#  Mouginot et al. (2019) — Greenland input-output (1972–2018)
# =========================================================================

def read_mouginot2019_greenland(filepath: str) -> pd.DataFrame:
    """Read Mouginot et al. (2019) Greenland ice sheet mass balance.

    Input-output method: SMB from RACMO, discharge from satellite
    velocity + ice thickness.  Annual, 1972–2018, ice sheet proper
    (excludes peripheral glaciers).

    Returns annual DataFrame in SLR convention (positive = sea level
    rise), units in meters SLE.

    Columns: mb_rate, mb_rate_sigma, smb_rate, smb_rate_sigma,
    discharge_rate, discharge_rate_sigma, cumulative_mb,
    cumulative_mb_sigma, decimal_year.

    Reference
    ---------
    Mouginot et al. (2019), PNAS 116, 9239–9244.
    doi:10.1073/pnas.1904242116
    """
    import openpyxl  # noqa: F401 — needed by pandas for xlsx

    raw = pd.read_excel(filepath, sheet_name="(2) MB_GIS", header=None)

    # Layout of sheet (2) MB_GIS:
    #   Row 8:  header "D" — years in cols 15–61; "Error F" in col 68, err years cols 82–128
    #   Row 16: GIS total discharge (D) and error
    #   Row 28: GIS total SMB and error
    #   Row 38: GIS total MB (= SMB − D) and error
    #   Row 50: GIS cumulative MB and error
    # All values in Gt/yr (rates) or Gt (cumulative), glaciology convention.

    DATA_COLS = slice(15, 62)   # cols 15..61 inclusive → 47 years
    ERR_COLS = slice(82, 129)   # cols 82..128 inclusive

    years = raw.iloc[8, DATA_COLS].values.astype(float).astype(int)
    n_yr = len(years)

    # Extract GIS totals (row 16=D, 28=SMB, 38=MB, 50=cumMB)
    D = raw.iloc[16, DATA_COLS].values.astype(float)
    D_err = raw.iloc[16, ERR_COLS].values.astype(float)
    SMB = raw.iloc[28, DATA_COLS].values.astype(float)
    SMB_err = raw.iloc[28, ERR_COLS].values.astype(float)
    MB = raw.iloc[38, DATA_COLS].values.astype(float)
    MB_err = raw.iloc[38, ERR_COLS].values.astype(float)
    MB_cum = raw.iloc[50, DATA_COLS].values.astype(float)
    MB_cum_err = raw.iloc[50, ERR_COLS].values.astype(float)

    assert len(D) == n_yr == len(SMB) == len(MB) == len(MB_cum)

    # Build DataFrame — flip sign to SLR convention
    # Glaciology: MB negative = mass loss.  SLR: positive = rise.
    # D is always positive (outflow); SMB positive = gain.
    # MB = SMB - D: negative when losing mass → flip to positive for SLR.
    df = pd.DataFrame({
        "decimal_year": years.astype(float) + 0.5,
        "mb_rate": -MB * GT_TO_M_SLE,
        "mb_rate_sigma": np.abs(MB_err) * GT_TO_M_SLE,
        "smb_rate": -SMB * GT_TO_M_SLE,       # negative SMB = mass loss = positive SLR
        "smb_rate_sigma": np.abs(SMB_err) * GT_TO_M_SLE,
        "discharge_rate": D * GT_TO_M_SLE,     # D already positive = outflow = SLR
        "discharge_rate_sigma": np.abs(D_err) * GT_TO_M_SLE,
        "cumulative_mb": -MB_cum * GT_TO_M_SLE,
        "cumulative_mb_sigma": np.abs(MB_cum_err) * GT_TO_M_SLE,
    })

    df["time"] = pd.to_datetime(
        [decimal_year_to_datetime(y) for y in df["decimal_year"]]
    )
    df = df.set_index("time")
    df.index.name = "time"

    # Metadata
    tag_sign_convention(df, "slr")
    tag_units(df, {
        "mb_rate": "m/yr", "mb_rate_sigma": "m/yr",
        "smb_rate": "m/yr", "smb_rate_sigma": "m/yr",
        "discharge_rate": "m/yr", "discharge_rate_sigma": "m/yr",
        "cumulative_mb": "m", "cumulative_mb_sigma": "m",
        "decimal_year": "yr",
    })
    df.attrs["dataset"] = "mouginot2019_greenland"
    df.attrs["region"] = "Greenland Ice Sheet (excl. peripheral glaciers)"
    df.attrs["reference"] = "Mouginot et al. (2019)"
    df.attrs["doi"] = "10.1073/pnas.1904242116"
    df.attrs["method"] = "input-output (RACMO SMB + satellite discharge)"
    df.attrs["note"] = (
        "SMB from RACMO2.3p2; discharge from satellite velocity + ice thickness. "
        "Peripheral glaciers excluded. MB = SMB - D (no basal mass balance)."
    )

    # Guards
    assert_sigma_positive(df["mb_rate_sigma"], "mouginot.mb_rate_sigma")
    assert_sigma_positive(df["cumulative_mb_sigma"], "mouginot.cumulative_mb_sigma")

    return df


# =========================================================================
#  Mankoff et al. (2021) — PROMICE Greenland (1840–present)
# =========================================================================

def read_mankoff2021_greenland(filepath: str, obs_only: bool = True) -> pd.DataFrame:
    """Read Mankoff et al. (2021) PROMICE Greenland mass balance.

    Input-output method with three RCMs (HIRHAM/HARMONIE, MAR, RACMO),
    satellite-derived discharge, and basal mass balance.  Annual,
    1840–present.  Ice sheet proper (excludes peripheral glaciers).

    Parameters
    ----------
    filepath : str
        Path to ``MB_SMB_D_BMB_ann.csv``.
    obs_only : bool, default True
        If True, restrict to 1986–present (RCM + satellite era).
        Pre-1986 uses the Kjeldsen et al. (2015) reconstruction,
        which is model-based.

    Returns annual DataFrame in SLR convention (positive = sea level
    rise), units in meters SLE.

    Columns: mb_rate, mb_rate_sigma, smb_rate, smb_rate_sigma,
    discharge_rate, discharge_rate_sigma, bmb_rate, bmb_rate_sigma,
    decimal_year.

    Reference
    ---------
    Mankoff et al. (2021), ESSD 13, 5001–5025.
    doi:10.5194/essd-13-5001-2021
    Data: doi:10.22008/FK2/OHI23Z
    """
    raw = pd.read_csv(filepath)

    # Columns: time, MB, MB_err, SMB, SMB_err, D, D_err, BMB, BMB_err
    # All in Gt/yr, glaciology convention: MB negative = mass loss.
    # MB = SMB - D - BMB.
    # D and BMB are positive (outflow/melt).

    if obs_only:
        raw = raw[raw["time"] >= 1986].copy()

    years = raw["time"].values

    # Flip sign to SLR convention
    df = pd.DataFrame({
        "decimal_year": years.astype(float) + 0.5,
        "mb_rate": -raw["MB"].values * GT_TO_M_SLE,
        "mb_rate_sigma": np.abs(raw["MB_err"].values) * GT_TO_M_SLE,
        "smb_rate": -raw["SMB"].values * GT_TO_M_SLE,
        "smb_rate_sigma": np.abs(raw["SMB_err"].values) * GT_TO_M_SLE,
        "discharge_rate": raw["D"].values * GT_TO_M_SLE,
        "discharge_rate_sigma": np.abs(raw["D_err"].values) * GT_TO_M_SLE,
        "bmb_rate": raw["BMB"].values * GT_TO_M_SLE,
        "bmb_rate_sigma": np.abs(raw["BMB_err"].values) * GT_TO_M_SLE,
    })

    df["time"] = pd.to_datetime(
        [decimal_year_to_datetime(y) for y in df["decimal_year"]]
    )
    df = df.set_index("time")
    df.index.name = "time"

    # Metadata
    tag_sign_convention(df, "slr")
    tag_units(df, {
        "mb_rate": "m/yr", "mb_rate_sigma": "m/yr",
        "smb_rate": "m/yr", "smb_rate_sigma": "m/yr",
        "discharge_rate": "m/yr", "discharge_rate_sigma": "m/yr",
        "bmb_rate": "m/yr", "bmb_rate_sigma": "m/yr",
        "decimal_year": "yr",
    })
    df.attrs["dataset"] = "mankoff2021_greenland"
    df.attrs["region"] = "Greenland Ice Sheet (excl. peripheral glaciers)"
    df.attrs["reference"] = "Mankoff et al. (2021)"
    df.attrs["doi"] = "10.5194/essd-13-5001-2021"
    df.attrs["method"] = (
        "input-output (3 RCMs for SMB + satellite discharge + basal mass balance)"
    )
    df.attrs["obs_only"] = obs_only
    df.attrs["note"] = (
        "Pre-1986 uses Kjeldsen et al. (2015) reconstruction (model-based). "
        "MB = SMB - D - BMB. Peripheral glaciers excluded."
    )

    # Guards
    assert_sigma_positive(df["mb_rate_sigma"], "mankoff.mb_rate_sigma")

    return df
