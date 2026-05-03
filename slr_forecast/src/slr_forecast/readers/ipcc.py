"""
IPCC AR6 projection readers.

Datasets
--------
- Projected GMST by SSP (annual, 2015–2099)
- Projected GMSL by SSP (medium and low confidence, decadal)
- Component-level FACTS projections (glaciers, ice sheets, ocean dynamics, LWS)
"""

from __future__ import annotations

import os
from typing import Optional

import numpy as np
import pandas as pd

from slr_forecast.units import (
    MM_TO_M,
    tag_units,
    tag_sign_convention,
)
from slr_forecast.config import Z_90
from slr_forecast.readers._utils import decimal_year_to_datetime


# =========================================================================
#  Projected temperature
# =========================================================================

def read_ipcc_ar6_projected_temperature(data_dir: str) -> dict[str, pd.DataFrame]:
    """Read IPCC AR6 projected GMST for all SSP scenarios.

    Returns dict keyed by scenario name (e.g. 'SSP2_4_5'), each a
    DataFrame with datetime index and columns: temperature,
    temperature_lower, temperature_upper, temperature_sigma, decimal_year.
    All in degC, anomaly relative to 1850–1900.

    Reference
    ---------
    Lee et al. (2021), doi:10.1017/9781009157896.006
    """
    import glob as globmod

    csv_files = sorted(globmod.glob(os.path.join(data_dir, "tas_global_*.csv")))
    if not csv_files:
        raise FileNotFoundError(f"No tas_global_*.csv files found in {data_dir}")

    result = {}
    for fpath in csv_files:
        scenario = os.path.basename(fpath).replace("tas_global_", "").replace(".csv", "")
        df = pd.read_csv(fpath)
        df = df.rename(columns={
            "Year": "decimal_year",
            "5%": "temperature_lower",
            "Mean": "temperature",
            "95%": "temperature_upper",
        })
        df["time"] = pd.to_datetime(
            [decimal_year_to_datetime(y) for y in df["decimal_year"]]
        )
        df = df.set_index("time")
        df.index.name = "time"

        # 90% CI → 1-sigma
        df["temperature_sigma"] = (
            (df["temperature_upper"] - df["temperature_lower"]) / (2 * Z_90)
        )

        tag_units(df, {c: "degC" for c in [
            "temperature", "temperature_lower", "temperature_upper",
            "temperature_sigma",
        ]})
        df.attrs["dataset"] = "ipcc_ar6_projected_temperature"
        df.attrs["scenario"] = scenario
        df.attrs["reference"] = "Lee et al. (2021)"
        df.attrs["doi"] = "10.1017/9781009157896.006"
        df.attrs["temperature_baseline"] = "1850-1900"

        result[scenario] = df

    return result


# =========================================================================
#  Projected GMSL (medium / low confidence)
# =========================================================================

def _read_ipcc_ar6_projected_gmsl_impl(
    data_dir: str,
    confidence: str,
) -> dict[str, pd.DataFrame]:
    """Internal: read IPCC AR6 FACTS projected GMSL."""
    import xarray as xr

    result = {}
    conf_tag = f"{confidence}_confidence"
    components = ["oceandynamics", "AIS", "GIS", "glaciers", "landwaterstorage"]

    scenario_dirs = sorted([
        d for d in os.listdir(data_dir)
        if os.path.isdir(os.path.join(data_dir, d))
    ])
    if not scenario_dirs:
        raise FileNotFoundError(f"No scenario directories found in {data_dir}")

    for scenario in scenario_dirs:
        scenario_path = os.path.join(data_dir, scenario)
        total_file = os.path.join(
            scenario_path, f"total_{scenario}_{conf_tag}_values.nc"
        )
        if not os.path.exists(total_file):
            continue

        ds_total = xr.open_dataset(total_file)
        sl = ds_total["sea_level_change"].values[:, :, 0]
        quantiles = ds_total["quantiles"].values
        years = ds_total["years"].values

        yr_mask = years <= 2260
        sl = sl[:, yr_mask]
        years = years[yr_mask]

        def _find_q(q_target):
            return int(np.argmin(np.abs(quantiles - q_target)))

        q05, q17, q50 = _find_q(0.05), _find_q(0.167), _find_q(0.5)
        q83, q95 = _find_q(0.833), _find_q(0.95)

        data = {
            "gmsl": sl[q50, :] * MM_TO_M,
            "gmsl_lower": sl[q05, :] * MM_TO_M,
            "gmsl_upper": sl[q95, :] * MM_TO_M,
            "gmsl_17": sl[q17, :] * MM_TO_M,
            "gmsl_83": sl[q83, :] * MM_TO_M,
            "decimal_year": years.astype(float),
        }
        ds_total.close()

        for comp in components:
            comp_file = os.path.join(
                scenario_path, f"{comp}_{scenario}_{conf_tag}_values.nc"
            )
            if not os.path.exists(comp_file):
                data[comp] = np.full(len(years), np.nan)
                continue

            ds_comp = xr.open_dataset(comp_file)
            sl_comp = ds_comp["sea_level_change"].values[:, :, 0]
            q_comp = ds_comp["quantiles"].values
            years_comp = ds_comp["years"].values
            q50_comp = int(np.argmin(np.abs(q_comp - 0.5)))
            median_comp = sl_comp[q50_comp, :] * MM_TO_M

            if not np.array_equal(years_comp, years):
                data[comp] = np.interp(
                    years, years_comp, median_comp, left=np.nan, right=np.nan
                )
            else:
                data[comp] = median_comp
            ds_comp.close()

        time_index = pd.to_datetime(
            [decimal_year_to_datetime(float(y)) for y in years]
        )
        df = pd.DataFrame(data, index=time_index)
        df.index.name = "time"

        tag_sign_convention(df, "slr")
        sl_cols = ["gmsl", "gmsl_lower", "gmsl_upper", "gmsl_17", "gmsl_83"]
        comp_cols = [c for c in components if c in data]
        tag_units(df, {
            **{c: "m" for c in sl_cols + comp_cols},
            "decimal_year": "yr",
        })
        df.attrs["dataset"] = "ipcc_ar6_projected_gmsl"
        df.attrs["confidence"] = confidence
        df.attrs["scenario"] = scenario
        df.attrs["reference"] = "Fox-Kemper et al. (2021)"
        df.attrs["doi"] = "10.1017/9781009157896.011"
        df.attrs["baseline"] = 2005

        result[scenario] = df

    return result


def read_ipcc_ar6_projected_gmsl(data_dir: str) -> dict[str, pd.DataFrame]:
    """Read IPCC AR6 FACTS projected GMSL (medium confidence).

    Returns dict keyed by scenario, each a DataFrame in meters with
    columns: gmsl, gmsl_lower/upper/17/83, component medians, decimal_year.

    Reference
    ---------
    Fox-Kemper et al. (2021), doi:10.1017/9781009157896.011
    Garner et al. (2023), doi:10.5194/gmd-16-7461-2023
    """
    return _read_ipcc_ar6_projected_gmsl_impl(data_dir, confidence="medium")


def read_ipcc_ar6_projected_gmsl_low_confidence(
    data_dir: str,
) -> dict[str, pd.DataFrame]:
    """Read IPCC AR6 FACTS projected GMSL (low confidence).

    Low-confidence projections include structured expert judgment on ice
    sheet processes that allows for higher tail risks.
    """
    return _read_ipcc_ar6_projected_gmsl_impl(data_dir, confidence="low")


# =========================================================================
#  Component-level FACTS projections
# =========================================================================

def read_ipcc_ar6_component(
    component_dir: str,
    component_type: str,
    sub_component: Optional[str] = None,
    model: Optional[str] = None,
    scenario: str = "ssp245",
) -> pd.DataFrame:
    """Read an IPCC AR6 FACTS component-level projection from NetCDF.

    Parameters
    ----------
    component_dir : str
        Path to ``dist_components/`` directory.
    component_type : str
        'glaciers', 'icesheets', 'landwaterstorage', or 'oceandynamics'.
    sub_component : str, optional
        'GIS', 'AIS', 'WAIS', 'EAIS', 'PEN'. Required for icesheets.
    model : str, optional
        Model identifier for disambiguation.
    scenario : str
        SSP code (e.g. 'ssp245', 'ssp585').

    Returns
    -------
    pd.DataFrame
        Integer year index, columns: median, p5, p17, p50, p83, p95,
        mean, std. All in meters.

    Reference
    ---------
    Garner et al. (2023), doi:10.5194/gmd-16-7461-2023
    """
    import glob as globmod
    import xarray as xr

    if sub_component:
        pattern = os.path.join(
            component_dir,
            f"{component_type}-*-{scenario}_{sub_component}_globalsl.nc",
        )
    else:
        pattern = os.path.join(
            component_dir,
            f"{component_type}-*-{scenario}_globalsl.nc",
        )

    matches = sorted(globmod.glob(pattern))
    if not matches:
        raise FileNotFoundError(
            f"No FACTS files for {component_type}/{sub_component}/{scenario} "
            f"in {component_dir}. Pattern: {pattern}"
        )

    if model is not None:
        filtered = [m for m in matches if model in os.path.basename(m)]
        if not filtered:
            available = [os.path.basename(m) for m in matches]
            raise FileNotFoundError(
                f"Model '{model}' not found. Available: {available}"
            )
        matches = filtered

    if len(matches) > 1:
        ipccar6 = [m for m in matches if "ipccar6" in os.path.basename(m)]
        if ipccar6:
            matches = ipccar6

    nc_path = matches[0]
    ds = xr.open_dataset(nc_path)

    years = ds["years"].values
    quantiles = ds["quantiles"].values
    slc = ds["sea_level_change"].values[:, :, 0]
    ds.close()

    def _get_q(q_val):
        idx = np.argmin(np.abs(quantiles - q_val))
        return slc[idx, :]

    df = pd.DataFrame(
        {
            "p5": _get_q(0.05),
            "p17": _get_q(0.17),
            "median": _get_q(0.50),
            "p83": _get_q(0.83),
            "p95": _get_q(0.95),
            "mean": np.nanmean(slc, axis=0),
            "std": np.nanstd(slc, axis=0),
        },
        index=years,
    )
    df.index.name = "year"

    # Convert mm → meters
    for col in df.columns:
        df[col] = df[col] * MM_TO_M

    model_used = os.path.basename(nc_path).split(f"-{scenario}")[0]

    tag_sign_convention(df, "slr")
    tag_units(df, {c: "m" for c in df.columns})
    df.attrs["dataset"] = f"ipcc_ar6_{component_type}"
    df.attrs["component"] = component_type
    df.attrs["sub_component"] = sub_component or "total"
    df.attrs["model"] = model_used
    df.attrs["scenario"] = scenario
    df.attrs["source_file"] = os.path.basename(nc_path)
    df.attrs["reference"] = "Garner et al. (2023)"
    df.attrs["doi"] = "10.5194/gmd-16-7461-2023"
    df.attrs["baseline"] = 2005

    return df


def read_ipcc_ar6_quantiles(
    component_dir: str,
    component_type: str,
    scenario: str = "ssp245",
    sub_component: Optional[str] = None,
) -> tuple:
    """Read raw quantile arrays from an IPCC AR6 FACTS component NetCDF.

    Returns the full quantile distribution (typically 107 quantiles) for
    Monte Carlo sampling, rather than the summary percentiles returned by
    :func:`read_ipcc_ar6_component`.

    Parameters
    ----------
    component_dir : str
        Path to ``dist_components/`` directory.
    component_type : str
        'glaciers', 'icesheets', 'landwaterstorage', or 'oceandynamics'.
    scenario : str
        SSP code (e.g. 'ssp245', 'ssp585').
    sub_component : str, optional
        'GIS', 'AIS', 'WAIS', 'EAIS', 'PEN'. Required for icesheets.

    Returns
    -------
    years : ndarray, shape (n_years,)
        Integer years (e.g. 2020, 2030, ..., 2150).
    quantiles : ndarray, shape (n_quantiles,)
        Quantile levels from 0 to 1.
    slc_m : ndarray, shape (n_quantiles, n_years)
        Sea level change in **meters**, relative to 2005 baseline.
    """
    import glob as globmod
    import xarray as xr

    if sub_component:
        pattern = os.path.join(
            component_dir,
            f"{component_type}-*-{scenario}_{sub_component}_globalsl.nc",
        )
    else:
        pattern = os.path.join(
            component_dir,
            f"{component_type}-*-{scenario}_globalsl.nc",
        )

    matches = sorted(globmod.glob(pattern))
    if not matches:
        raise FileNotFoundError(
            f"No FACTS files for {component_type}/{scenario} "
            f"in {component_dir}. Pattern: {pattern}"
        )

    if len(matches) > 1:
        ipccar6 = [m for m in matches if "ipccar6" in os.path.basename(m)]
        if ipccar6:
            matches = ipccar6

    nc_path = matches[0]
    ds = xr.open_dataset(nc_path)
    years = ds["years"].values
    quantiles = ds["quantiles"].values
    slc_mm = ds["sea_level_change"].values[:, :, 0]  # (n_q, n_years)
    ds.close()

    slc_m = slc_mm * MM_TO_M
    return years, quantiles, slc_m


def list_ipcc_ar6_components(component_dir: str) -> pd.DataFrame:
    """List all available IPCC AR6 FACTS component files.

    Returns table with columns: filename, component_type, model,
    scenario, sub_component.
    """
    import glob as globmod

    files = sorted(globmod.glob(os.path.join(component_dir, "*.nc")))
    records = []
    for f in files:
        fname = os.path.basename(f)
        parts = fname.replace("_globalsl.nc", "").split("-")
        comp_type = parts[0]

        scenario_idx = None
        for i, p in enumerate(parts):
            if p.startswith("ssp") or p.startswith("tlim"):
                scenario_idx = i
                break
        if scenario_idx is None:
            continue

        model_str = "-".join(parts[1:scenario_idx])
        scenario_sub = parts[scenario_idx]

        if "_" in scenario_sub:
            scenario, sub = scenario_sub.split("_", 1)
        else:
            scenario = scenario_sub
            sub = None

        records.append({
            "filename": fname,
            "component_type": comp_type,
            "model": model_str,
            "scenario": scenario,
            "sub_component": sub,
        })

    return pd.DataFrame(records)
