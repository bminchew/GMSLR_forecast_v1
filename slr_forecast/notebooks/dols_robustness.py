#!/usr/bin/env python3
"""
DOLS Robustness Analysis: Multi-Dataset Sensitivity Matrix
============================================================

Runs DOLS on every combination of GMSL × GMST datasets to demonstrate
robustness of the quadratic temperature sensitivity (dα/dT) and linear
sensitivity (α₀).

GMSL datasets:
  - IPCC observed (total GMSL, 1950–2020)
  - IPCC observed thermodynamic (GMSL minus Frederikse TWS, 1950–2018)
  - Dangendorf (total GMSL, 1900–2021)
  - Dangendorf sterodynamic (thermodynamic analog, 1900–2021)
  - Frederikse (total GMSL, 1900–2018)
  - Frederikse thermodynamic (GMSL minus TWS, 1900–2018)
  - Horwath thermodynamic (GMSL minus TWS, monthly→annual, 1993–2016)

GMST datasets:
  - Berkeley Earth (1850–2024, monthly→annual)
  - GISTEMP (1880–2025, monthly→annual)
  - HadCRUT5 (1850–2025, monthly→annual)
  - NOAA GlobalTemp (1850–2025, annual)

Combined DOLS:
  Reports the ensemble mean ± across-dataset spread of coefficients from
  all thermodynamic GMSL × all GMST combinations.

Authors: Minchew research group, 2026
"""

import sys
import os
import warnings
from typing import Optional, Dict, Tuple, List

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable

warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# Ensure local imports
sys.path.insert(0, os.path.dirname(__file__))
import slr_analysis
from slr_analysis import calibrate_dols, DOLSResult

# Paths
DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
H5_PATH = os.path.join(DATA_DIR, "processed", "slr_processed_data.h5")
FIG_DIR = os.path.join(os.path.dirname(__file__), "..", "figures")
os.makedirs(FIG_DIR, exist_ok=True)

# Display constants
M_TO_MM = 1000.0
BASELINE_YEAR = 2005.0


# =====================================================================
#  1. LOAD AND PREPARE ALL DATASETS
# =====================================================================

def _datetime_to_decimal_year(dt_index):
    """Convert a DatetimeIndex to decimal year (float) array."""
    years = dt_index.year
    # Day of year / days in year
    day_of_year = dt_index.dayofyear
    days_in_year = np.where(dt_index.is_leap_year, 366, 365)
    return years + (day_of_year - 1) / days_in_year


def _annualize_monthly(df, col="temperature"):
    """Convert monthly series to annual means."""
    annual = df.groupby(df.index.year).mean()
    annual.index = annual.index.astype(float) + 0.5  # mid-year
    return annual


def _rebase_to_baseline(series, baseline_year=BASELINE_YEAR, window=5):
    """Rebase a series so that the baseline_year ± window/2 period averages to zero."""
    lo = baseline_year - window
    hi = baseline_year + window
    mask = (series.index >= lo) & (series.index <= hi)
    if mask.sum() == 0:
        # Fallback: nearest available years
        mask = (series.index >= lo - 5) & (series.index <= hi + 5)
    baseline_val = series[mask].mean()
    return series - baseline_val


def load_all_gmsl(start_year=1950.0):
    """Load all GMSL datasets, truncate to start_year, rebase to 2005.

    Parameters
    ----------
    start_year : float
        Truncate all datasets to begin at or after this year.
        Default 1950.0 corresponds to the IPCC observed GMSL start.

    Returns
    -------
    gmsl_datasets : dict of {name: pd.Series}
        Each series has a float index (decimal year) and values in metres.
    """
    store = pd.HDFStore(H5_PATH, mode="r")

    datasets = {}

    # --- Frederikse total GMSL ---
    df = store["/raw/df_frederikse"]
    fred_dec = _datetime_to_decimal_year(df.index)
    s = pd.Series(df["gmsl"].values, index=fred_dec, name="gmsl")
    s = _rebase_to_baseline(s)
    datasets["Frederikse"] = s

    # --- Frederikse thermodynamic (GMSL - TWS) ---
    df_th = store["/derived/df_frederikse_thermo"]
    fred_th_dec = _datetime_to_decimal_year(df_th.index)
    s = pd.Series(df_th["thermodynamic_gmsl"].values,
                  index=fred_th_dec, name="thermo_gmsl")
    s = _rebase_to_baseline(s)
    datasets["Frederikse thermo"] = s

    # --- Dangendorf total GMSL ---
    df = store["/raw/df_dangendorf"]
    if "decimal_year" in df.columns:
        dang_dec = df["decimal_year"].values
    else:
        dang_dec = _datetime_to_decimal_year(df.index)
    s = pd.Series(df["gmsl"].values, index=dang_dec, name="gmsl")
    s = _rebase_to_baseline(s)
    datasets["Dangendorf"] = s

    # --- Dangendorf sterodynamic (thermodynamic analog) ---
    s = pd.Series(df["sterodynamic"].values, index=dang_dec, name="sterodynamic")
    s = _rebase_to_baseline(s)
    datasets["Dangendorf sterodynamic"] = s

    # --- Horwath thermodynamic (GMSL - TWS, monthly → annual) ---
    df = store["/raw/df_horwath"]
    # Compute thermodynamic: gmsl - tws
    if "gmsl" in df.columns and "tws" in df.columns:
        thermo = df["gmsl"] - df["tws"]
        # Annualize
        thermo_annual = thermo.groupby(thermo.index.year).mean()
        thermo_annual.index = thermo_annual.index.astype(float) + 0.5
        s = pd.Series(thermo_annual.values, index=thermo_annual.index, name="thermo_gmsl")
        s = _rebase_to_baseline(s)
        datasets["Horwath thermo"] = s

    # --- IPCC observed total GMSL ---
    df = store["/raw/df_ipcc_observed_gmsl"]
    if "decimal_year" in df.columns:
        ipcc_dec = df["decimal_year"].values
    else:
        ipcc_dec = _datetime_to_decimal_year(df.index)
    s = pd.Series(df["gmsl"].values, index=ipcc_dec, name="gmsl")
    s = _rebase_to_baseline(s)
    datasets["IPCC observed"] = s

    # --- IPCC observed thermodynamic (subtract Frederikse TWS) ---
    # Only over the overlap period (1950-2018)
    df_fred_th = store["/derived/df_frederikse_thermo"]
    fred_th_dec2 = _datetime_to_decimal_year(df_fred_th.index)
    fred_tws = pd.Series(df_fred_th["tws"].values,
                         index=fred_th_dec2, name="tws")
    ipcc_idx = ipcc_dec
    ipcc_gmsl = df["gmsl"].values
    # Interpolate TWS to IPCC years
    tws_at_ipcc = np.interp(ipcc_idx, fred_tws.index.values, fred_tws.values,
                            left=np.nan, right=np.nan)
    ipcc_thermo = ipcc_gmsl - tws_at_ipcc
    valid = np.isfinite(ipcc_thermo)
    s = pd.Series(ipcc_thermo[valid], index=ipcc_idx[valid], name="thermo_gmsl")
    s = _rebase_to_baseline(s)
    datasets["IPCC obs thermo"] = s

    store.close()

    # Truncate all to start_year
    truncated = {}
    for name, s in datasets.items():
        s_trunc = s[s.index >= start_year]
        if len(s_trunc) >= 10:  # need at least 10 annual points for DOLS
            truncated[name] = s_trunc
        else:
            print(f"  Warning: {name} has only {len(s_trunc)} points after "
                  f"truncation to {start_year}; skipping.")

    return truncated


def load_all_gmst(start_year=1950.0):
    """Load all GMST datasets, annualize if monthly, truncate.

    Returns
    -------
    gmst_datasets : dict of {name: pd.Series}
        Each series has float index (decimal year) and temperature in °C.
    """
    store = pd.HDFStore(H5_PATH, mode="r")

    datasets = {}

    # Berkeley Earth (monthly)
    df = store["/raw/df_berkeley"]
    annual = _annualize_monthly(df, "temperature")
    s = pd.Series(annual["temperature"].values, index=annual.index, name="temperature")
    datasets["Berkeley"] = s

    # GISTEMP (monthly)
    df = store["/raw/df_gistemp"]
    annual = _annualize_monthly(df, "temperature")
    s = pd.Series(annual["temperature"].values, index=annual.index, name="temperature")
    datasets["GISTEMP"] = s

    # HadCRUT5 (monthly)
    df = store["/raw/df_hadcrut"]
    annual = _annualize_monthly(df, "temperature")
    s = pd.Series(annual["temperature"].values, index=annual.index, name="temperature")
    datasets["HadCRUT"] = s

    # NOAA GlobalTemp (already annual — one value per year)
    df = store["/raw/df_noaa"]
    noaa_dec = _datetime_to_decimal_year(df.index)
    s = pd.Series(df["temperature"].values, index=noaa_dec, name="temperature")
    datasets["NOAA"] = s

    store.close()

    # Truncate to start_year
    truncated = {}
    for name, s in datasets.items():
        s_trunc = s[s.index >= start_year]
        if len(s_trunc) >= 10:
            truncated[name] = s_trunc
    return truncated


# =====================================================================
#  2. RUN DOLS ON A SINGLE (GMSL, GMST) PAIR
# =====================================================================

def _align_and_run_dols(gmsl_series, gmst_series, order=2, n_lags=2):
    """Align two annual series by year, then run calibrate_dols.

    Parameters
    ----------
    gmsl_series : pd.Series with float index (decimal year)
    gmst_series : pd.Series with float index (decimal year)
    order : int
    n_lags : int

    Returns
    -------
    DOLSResult or None if calibration fails
    """
    # Map decimal years to integer years for alignment.
    # Use floor(x + 0.01) to avoid banker's rounding issues at exactly X.5
    gmsl_years = np.floor(gmsl_series.index.values + 0.01).astype(int)
    gmst_years = np.floor(gmst_series.index.values + 0.01).astype(int)
    common_years = np.intersect1d(gmsl_years, gmst_years)

    if len(common_years) < 10 + 2 * n_lags:
        return None

    # Build aligned series with DatetimeIndex (required by calibrate_dols)
    gmsl_vals = []
    gmst_vals = []
    dates = []
    for yr in common_years:
        # Find nearest index in each series
        gmsl_idx = np.argmin(np.abs(gmsl_series.index.values - yr))
        gmst_idx = np.argmin(np.abs(gmst_series.index.values - yr))
        gmsl_vals.append(gmsl_series.iloc[gmsl_idx])
        gmst_vals.append(gmst_series.iloc[gmst_idx])
        dates.append(pd.Timestamp(f"{yr}-07-01"))

    sl = pd.Series(gmsl_vals, index=pd.DatetimeIndex(dates), name="sea_level")
    temp = pd.Series(gmst_vals, index=pd.DatetimeIndex(dates), name="temperature")

    try:
        result = calibrate_dols(sl, temp, order=order, n_lags=n_lags)
        return result
    except Exception as e:
        return None


# =====================================================================
#  3. ROBUSTNESS MATRIX
# =====================================================================

def run_robustness_matrix(start_year=1950.0, order=2, n_lags=2, verbose=True):
    """Run DOLS on every GMSL × GMST combination.

    Parameters
    ----------
    start_year : float
        Truncate all datasets to start at or after this year.
    order : int
        Polynomial order for DOLS (1=linear, 2=quadratic).
    n_lags : int
        Number of leads/lags for DOLS.
    verbose : bool

    Returns
    -------
    results : dict
        'matrix' : dict of {(gmsl_name, gmst_name): DOLSResult}
        'alpha0_df' : pd.DataFrame (GMSL rows × GMST cols) of α₀ in mm/yr/°C
        'dalpha_dT_df' : pd.DataFrame of dα/dT in mm/yr/°C²
        'r2_df' : pd.DataFrame of R²
        'n_obs_df' : pd.DataFrame of sample sizes
        'gmsl_names' : list
        'gmst_names' : list
    """
    if verbose:
        print("=" * 70)
        print(f"DOLS ROBUSTNESS MATRIX (order={order}, n_lags={n_lags}, "
              f"start_year={start_year:.0f})")
        print("=" * 70)
        print()

    gmsl_data = load_all_gmsl(start_year=start_year)
    gmst_data = load_all_gmst(start_year=start_year)

    gmsl_names = list(gmsl_data.keys())
    gmst_names = list(gmst_data.keys())

    if verbose:
        print(f"GMSL datasets ({len(gmsl_names)}):")
        for name, s in gmsl_data.items():
            print(f"  {name}: {s.index.min():.0f}–{s.index.max():.0f} "
                  f"(n={len(s)})")
        print(f"\nGMST datasets ({len(gmst_names)}):")
        for name, s in gmst_data.items():
            print(f"  {name}: {s.index.min():.0f}–{s.index.max():.0f} "
                  f"(n={len(s)})")
        print()

    # Run all combinations
    matrix = {}
    for gmsl_name in gmsl_names:
        for gmst_name in gmst_names:
            result = _align_and_run_dols(
                gmsl_data[gmsl_name], gmst_data[gmst_name],
                order=order, n_lags=n_lags,
            )
            matrix[(gmsl_name, gmst_name)] = result
            if verbose and result is not None:
                a0 = result.alpha0 * M_TO_MM
                if order >= 2 and result.dalpha_dT is not None:
                    da = result.dalpha_dT * M_TO_MM
                    print(f"  {gmsl_name:25s} × {gmst_name:10s}: "
                          f"α₀={a0:+6.2f} mm/yr/°C, "
                          f"dα/dT={da:+6.2f} mm/yr/°C², "
                          f"R²={result.r2:.4f} (n={result.n_obs})")
                else:
                    print(f"  {gmsl_name:25s} × {gmst_name:10s}: "
                          f"α₀={a0:+6.2f} mm/yr/°C, "
                          f"R²={result.r2:.4f} (n={result.n_obs})")
            elif verbose:
                print(f"  {gmsl_name:25s} × {gmst_name:10s}: FAILED")

    # Build DataFrames
    alpha0_data = np.full((len(gmsl_names), len(gmst_names)), np.nan)
    alpha0_se_data = np.full_like(alpha0_data, np.nan)
    dalpha_data = np.full_like(alpha0_data, np.nan)
    dalpha_se_data = np.full_like(alpha0_data, np.nan)
    r2_data = np.full_like(alpha0_data, np.nan)
    n_obs_data = np.full_like(alpha0_data, np.nan)
    trend_data = np.full_like(alpha0_data, np.nan)

    for i, gmsl_name in enumerate(gmsl_names):
        for j, gmst_name in enumerate(gmst_names):
            r = matrix.get((gmsl_name, gmst_name))
            if r is not None:
                alpha0_data[i, j] = r.alpha0 * M_TO_MM
                alpha0_se_data[i, j] = r.alpha0_se * M_TO_MM
                r2_data[i, j] = r.r2
                n_obs_data[i, j] = r.n_obs
                trend_data[i, j] = r.trend * M_TO_MM
                if order >= 2 and r.dalpha_dT is not None:
                    dalpha_data[i, j] = r.dalpha_dT * M_TO_MM
                    dalpha_se_data[i, j] = r.dalpha_dT_se * M_TO_MM

    alpha0_df = pd.DataFrame(alpha0_data, index=gmsl_names, columns=gmst_names)
    alpha0_se_df = pd.DataFrame(alpha0_se_data, index=gmsl_names, columns=gmst_names)
    dalpha_df = pd.DataFrame(dalpha_data, index=gmsl_names, columns=gmst_names)
    dalpha_se_df = pd.DataFrame(dalpha_se_data, index=gmsl_names, columns=gmst_names)
    r2_df = pd.DataFrame(r2_data, index=gmsl_names, columns=gmst_names)
    n_obs_df = pd.DataFrame(n_obs_data, index=gmsl_names, columns=gmst_names)
    trend_df = pd.DataFrame(trend_data, index=gmsl_names, columns=gmst_names)

    if verbose:
        print()
        print("-" * 70)
        print("α₀ (mm/yr/°C):")
        print("-" * 70)
        print(alpha0_df.round(2).to_string())
        print()
        if order >= 2:
            print("-" * 70)
            print("dα/dT (mm/yr/°C²):")
            print("-" * 70)
            print(dalpha_df.round(2).to_string())
            print()
        print("-" * 70)
        print("R²:")
        print("-" * 70)
        print(r2_df.round(4).to_string())

    return {
        "matrix": matrix,
        "alpha0_df": alpha0_df,
        "alpha0_se_df": alpha0_se_df,
        "dalpha_dT_df": dalpha_df,
        "dalpha_dT_se_df": dalpha_se_df,
        "r2_df": r2_df,
        "n_obs_df": n_obs_df,
        "trend_df": trend_df,
        "gmsl_names": gmsl_names,
        "gmst_names": gmst_names,
        "order": order,
        "n_lags": n_lags,
        "start_year": start_year,
    }


# =====================================================================
#  4. COMBINED (ENSEMBLE) DOLS STATISTICS
# =====================================================================

def compute_combined_dols(robustness_results, thermodynamic_only=True,
                          min_n_obs=30, verbose=True):
    """Compute ensemble statistics from the robustness matrix.

    Rather than concatenating raw series (which introduces methodological
    artifacts), we treat each (GMSL, GMST) pair as an independent estimate
    and report the ensemble mean ± across-dataset spread.

    Parameters
    ----------
    robustness_results : dict from run_robustness_matrix
    thermodynamic_only : bool
        If True, restrict to thermodynamic GMSL datasets only.
    min_n_obs : int
        Minimum number of observations for a result to be included.
        Short records (e.g. Horwath, 24 yr) produce unstable quadratic
        estimates and are excluded by default.
    verbose : bool

    Returns
    -------
    combined : dict with ensemble statistics
    """
    matrix = robustness_results["matrix"]
    order = robustness_results["order"]

    # Define which GMSL datasets are "thermodynamic"
    thermo_names = ["Frederikse thermo", "Dangendorf sterodynamic",
                    "Horwath thermo", "IPCC obs thermo"]
    total_names = ["Frederikse", "Dangendorf", "IPCC observed"]

    alpha0_vals = []
    dalpha_vals = []
    trend_vals = []
    r2_vals = []
    pair_labels = []
    excluded = []

    for (gmsl_name, gmst_name), result in matrix.items():
        if result is None:
            continue
        if thermodynamic_only and gmsl_name not in thermo_names:
            continue
        # Exclude short records that produce unstable estimates
        if result.n_obs < min_n_obs:
            excluded.append(f"{gmsl_name} × {gmst_name} (n={result.n_obs})")
            continue

        alpha0_vals.append(result.alpha0 * M_TO_MM)
        trend_vals.append(result.trend * M_TO_MM)
        r2_vals.append(result.r2)
        pair_labels.append(f"{gmsl_name} × {gmst_name}")

        if order >= 2 and result.dalpha_dT is not None:
            dalpha_vals.append(result.dalpha_dT * M_TO_MM)

    alpha0_arr = np.array(alpha0_vals)
    trend_arr = np.array(trend_vals)
    r2_arr = np.array(r2_vals)

    combined = {
        "n_pairs": len(alpha0_vals),
        "pair_labels": pair_labels,
        "alpha0_mean": np.mean(alpha0_arr),
        "alpha0_median": np.median(alpha0_arr),
        "alpha0_std": np.std(alpha0_arr),
        "alpha0_min": np.min(alpha0_arr),
        "alpha0_max": np.max(alpha0_arr),
        "alpha0_values": alpha0_arr,
        "trend_mean": np.mean(trend_arr),
        "trend_std": np.std(trend_arr),
        "trend_values": trend_arr,
        "r2_mean": np.mean(r2_arr),
        "r2_min": np.min(r2_arr),
        "r2_values": r2_arr,
        "thermodynamic_only": thermodynamic_only,
    }

    if dalpha_vals:
        dalpha_arr = np.array(dalpha_vals)
        combined.update({
            "dalpha_dT_mean": np.mean(dalpha_arr),
            "dalpha_dT_median": np.median(dalpha_arr),
            "dalpha_dT_std": np.std(dalpha_arr),
            "dalpha_dT_min": np.min(dalpha_arr),
            "dalpha_dT_max": np.max(dalpha_arr),
            "dalpha_dT_values": dalpha_arr,
        })

    if verbose:
        print()
        print("=" * 70)
        subset = "thermodynamic" if thermodynamic_only else "all"
        print(f"COMBINED DOLS ENSEMBLE ({subset} GMSL datasets, "
              f"n={combined['n_pairs']} pairs, min_n_obs={min_n_obs})")
        print("=" * 70)
        if excluded:
            print(f"  Excluded (n_obs < {min_n_obs}):")
            for ex in excluded:
                print(f"    {ex}")
            print()
        print(f"  α₀:     {combined['alpha0_mean']:.2f} ± "
              f"{combined['alpha0_std']:.2f} mm/yr/°C  "
              f"[{combined['alpha0_min']:.2f}, {combined['alpha0_max']:.2f}]")
        if "dalpha_dT_mean" in combined:
            print(f"  dα/dT:  {combined['dalpha_dT_mean']:.2f} ± "
                  f"{combined['dalpha_dT_std']:.2f} mm/yr/°C²  "
                  f"[{combined['dalpha_dT_min']:.2f}, "
                  f"{combined['dalpha_dT_max']:.2f}]")
        print(f"  trend:  {combined['trend_mean']:.2f} ± "
              f"{combined['trend_std']:.2f} mm/yr")
        print(f"  R²:     {combined['r2_mean']:.4f} "
              f"[{combined['r2_min']:.4f}, {np.max(r2_arr):.4f}]")
        print()

        # Per-pair breakdown
        print("  Per-pair breakdown:")
        for i, label in enumerate(pair_labels):
            line = f"    {label:45s}: α₀={alpha0_vals[i]:+6.2f}"
            if dalpha_vals:
                line += f", dα/dT={dalpha_vals[i]:+6.2f}"
            line += f", R²={r2_vals[i]:.4f}"
            print(line)

    return combined


# =====================================================================
#  5. VISUALIZATION
# =====================================================================

def plot_coefficient_heatmaps(robustness_results, fig_path=None,
                              exclude_datasets=None):
    """Plot heatmaps of α₀ and dα/dT across the GMSL × GMST matrix.

    Creates a 2-panel figure (or 1 panel if order=1) with annotated heatmaps
    showing coefficient values ± SE in each cell.

    Parameters
    ----------
    exclude_datasets : list of str, optional
        GMSL dataset names to exclude from the figure (e.g. ["Horwath thermo"]).
    """
    alpha0_df = robustness_results["alpha0_df"]
    alpha0_se_df = robustness_results["alpha0_se_df"]
    dalpha_df = robustness_results["dalpha_dT_df"]
    dalpha_se_df = robustness_results["dalpha_dT_se_df"]
    order = robustness_results["order"]

    # Filter out excluded datasets
    if exclude_datasets:
        keep = [n for n in alpha0_df.index if n not in exclude_datasets]
        alpha0_df = alpha0_df.loc[keep]
        alpha0_se_df = alpha0_se_df.loc[keep]
        dalpha_df = dalpha_df.loc[keep]
        dalpha_se_df = dalpha_se_df.loc[keep]

    n_panels = 2 if order >= 2 else 1
    fig, axes = plt.subplots(1, n_panels, figsize=(7 * n_panels, 0.8 * len(alpha0_df) + 2))
    if n_panels == 1:
        axes = [axes]

    # --- Panel 1: α₀ ---
    ax = axes[0]
    data = alpha0_df.values
    mask = np.isnan(data)
    vmin, vmax = np.nanmin(data), np.nanmax(data)
    im = ax.imshow(data, cmap="YlOrRd", aspect="auto",
                   vmin=vmin, vmax=vmax)
    # Annotate cells
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            if not np.isnan(data[i, j]):
                se = alpha0_se_df.values[i, j]
                txt = f"{data[i,j]:.2f}\n±{se:.2f}"
                ax.text(j, i, txt, ha="center", va="center", fontsize=7,
                        color="white" if data[i, j] > (vmin + vmax) / 2 else "black")
            else:
                ax.text(j, i, "—", ha="center", va="center", fontsize=8, color="gray")

    ax.set_xticks(range(len(alpha0_df.columns)))
    ax.set_xticklabels(alpha0_df.columns, rotation=45, ha="right", fontsize=9)
    ax.set_yticks(range(len(alpha0_df.index)))
    ax.set_yticklabels(alpha0_df.index, fontsize=9)
    ax.set_title("α₀ (mm/yr/°C)", fontsize=12, fontweight="bold")
    plt.colorbar(im, ax=ax, shrink=0.7, label="mm/yr/°C")

    # --- Panel 2: dα/dT ---
    if order >= 2:
        ax = axes[1]
        data = dalpha_df.values
        vmin2, vmax2 = np.nanmin(data), np.nanmax(data)
        # Use diverging colormap centered on zero
        abs_max = max(abs(vmin2), abs(vmax2))
        im2 = ax.imshow(data, cmap="RdBu_r", aspect="auto",
                        vmin=-abs_max, vmax=abs_max)
        for i in range(data.shape[0]):
            for j in range(data.shape[1]):
                if not np.isnan(data[i, j]):
                    se = dalpha_se_df.values[i, j]
                    txt = f"{data[i,j]:.2f}\n±{se:.2f}"
                    ax.text(j, i, txt, ha="center", va="center", fontsize=7)
                else:
                    ax.text(j, i, "—", ha="center", va="center", fontsize=8,
                            color="gray")

        ax.set_xticks(range(len(dalpha_df.columns)))
        ax.set_xticklabels(dalpha_df.columns, rotation=45, ha="right", fontsize=9)
        ax.set_yticks(range(len(dalpha_df.index)))
        ax.set_yticklabels(dalpha_df.index, fontsize=9)
        ax.set_title("dα/dT (mm/yr/°C²)", fontsize=12, fontweight="bold")
        plt.colorbar(im2, ax=ax, shrink=0.7, label="mm/yr/°C²")

    plt.suptitle(f"DOLS Robustness Matrix (order={robustness_results['order']}, "
                 f"n_lags={robustness_results['n_lags']}, "
                 f"start≥{robustness_results['start_year']:.0f})",
                 fontsize=12, y=1.02)
    plt.tight_layout()

    if fig_path:
        fig.savefig(fig_path, dpi=150, bbox_inches="tight")
        print(f"Saved: {fig_path}")
    plt.close(fig)
    return fig


def plot_coefficient_forest(robustness_results, combined_thermo=None,
                            combined_all=None, fig_path=None,
                            exclude_datasets=None):
    """Forest plot showing α₀ and dα/dT ± SE for each (GMSL, GMST) pair.

    Groups by GMSL dataset (color-coded) with GMST variant as sub-rows.
    Horizontal reference lines show ensemble mean.

    Parameters
    ----------
    exclude_datasets : list of str, optional
        GMSL dataset names to exclude from the figure (e.g. ["Horwath thermo"]).
    """
    matrix = robustness_results["matrix"]
    order = robustness_results["order"]

    # Collect all valid results
    entries = []
    for (gmsl_name, gmst_name), result in matrix.items():
        if result is None:
            continue
        if exclude_datasets and gmsl_name in exclude_datasets:
            continue
        entry = {
            "gmsl": gmsl_name,
            "gmst": gmst_name,
            "label": f"{gmsl_name} × {gmst_name}",
            "alpha0": result.alpha0 * M_TO_MM,
            "alpha0_se": result.alpha0_se * M_TO_MM,
            "r2": result.r2,
            "n_obs": result.n_obs,
        }
        if order >= 2 and result.dalpha_dT is not None:
            entry["dalpha_dT"] = result.dalpha_dT * M_TO_MM
            entry["dalpha_dT_se"] = result.dalpha_dT_se * M_TO_MM
        entries.append(entry)

    # Sort by GMSL dataset, then GMST
    entries.sort(key=lambda e: (e["gmsl"], e["gmst"]))

    n = len(entries)
    n_panels = 2 if order >= 2 else 1
    fig, axes = plt.subplots(1, n_panels, figsize=(6 * n_panels, 0.4 * n + 2),
                             sharey=True)
    if n_panels == 1:
        axes = [axes]

    # Color by GMSL dataset
    gmsl_unique = list(dict.fromkeys(e["gmsl"] for e in entries))
    cmap = plt.cm.Set2
    gmsl_colors = {name: cmap(i / max(len(gmsl_unique) - 1, 1))
                   for i, name in enumerate(gmsl_unique)}

    # Identify thermodynamic datasets
    thermo_names = {"Frederikse thermo", "Dangendorf sterodynamic",
                    "Horwath thermo", "IPCC obs thermo"}

    labels = [e["label"] for e in entries]
    y_pos = np.arange(n)

    # --- Panel 1: α₀ ---
    ax = axes[0]
    for i, e in enumerate(entries):
        color = gmsl_colors[e["gmsl"]]
        marker = "o" if e["gmsl"] in thermo_names else "s"
        ax.errorbar(e["alpha0"], i, xerr=1.96 * e["alpha0_se"],
                    fmt=marker, color=color, capsize=3, markersize=5)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontsize=7)
    ax.set_xlabel("α₀ (mm/yr/°C)", fontsize=10)
    ax.set_title("Linear Sensitivity", fontsize=11)
    ax.axvline(0, color="gray", ls=":", alpha=0.5)
    if combined_thermo:
        ax.axvline(combined_thermo["alpha0_mean"], color="red", ls="--",
                   alpha=0.7, label=f"Thermo mean: {combined_thermo['alpha0_mean']:.2f}")
        ax.axvspan(combined_thermo["alpha0_mean"] - combined_thermo["alpha0_std"],
                   combined_thermo["alpha0_mean"] + combined_thermo["alpha0_std"],
                   alpha=0.1, color="red")
    ax.legend(fontsize=7, loc="lower right")
    ax.grid(True, alpha=0.3, axis="x")

    # --- Panel 2: dα/dT ---
    if order >= 2:
        ax = axes[1]
        for i, e in enumerate(entries):
            if "dalpha_dT" not in e:
                continue
            color = gmsl_colors[e["gmsl"]]
            marker = "o" if e["gmsl"] in thermo_names else "s"
            ax.errorbar(e["dalpha_dT"], i, xerr=1.96 * e["dalpha_dT_se"],
                        fmt=marker, color=color, capsize=3, markersize=5)
        ax.set_xlabel("dα/dT (mm/yr/°C²)", fontsize=10)
        ax.set_title("Quadratic Sensitivity", fontsize=11)
        ax.axvline(0, color="gray", ls=":", alpha=0.5)
        if combined_thermo and "dalpha_dT_mean" in combined_thermo:
            ax.axvline(combined_thermo["dalpha_dT_mean"], color="red", ls="--",
                       alpha=0.7,
                       label=f"Thermo mean: {combined_thermo['dalpha_dT_mean']:.2f}")
            ax.axvspan(
                combined_thermo["dalpha_dT_mean"] - combined_thermo["dalpha_dT_std"],
                combined_thermo["dalpha_dT_mean"] + combined_thermo["dalpha_dT_std"],
                alpha=0.1, color="red")
        ax.legend(fontsize=7, loc="lower right")
        ax.grid(True, alpha=0.3, axis="x")

    # Legend for GMSL dataset colors
    from matplotlib.lines import Line2D
    legend_elements = []
    for name in gmsl_unique:
        marker = "o" if name in thermo_names else "s"
        legend_elements.append(
            Line2D([0], [0], marker=marker, color="w",
                   markerfacecolor=gmsl_colors[name], markersize=7,
                   label=name))
    axes[0].legend(handles=legend_elements, fontsize=6, loc="lower right",
                   title="GMSL dataset", title_fontsize=7, ncol=1)

    plt.suptitle(f"DOLS Coefficient Estimates (95% CI)\n"
                 f"order={order}, n_lags={robustness_results['n_lags']}, "
                 f"start≥{robustness_results['start_year']:.0f}",
                 fontsize=11)
    plt.tight_layout()

    if fig_path:
        fig.savefig(fig_path, dpi=150, bbox_inches="tight")
        print(f"Saved: {fig_path}")
    plt.close(fig)
    return fig


def plot_ensemble_summary(combined_thermo, combined_all=None, fig_path=None):
    """Bar/violin plot comparing thermodynamic-only vs all-dataset ensembles.

    Shows the distribution of α₀ and dα/dT across pairs.
    """
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    # Panel 1: α₀ distribution
    ax = axes[0]
    data_list = [combined_thermo["alpha0_values"]]
    labels = ["Thermodynamic"]
    colors = ["#2166ac"]
    if combined_all is not None:
        data_list.append(combined_all["alpha0_values"])
        labels.append("All GMSL")
        colors.append("#b2182b")

    parts = ax.violinplot(data_list, positions=range(len(data_list)),
                          showmeans=True, showmedians=True)
    for i, pc in enumerate(parts["bodies"]):
        pc.set_facecolor(colors[i])
        pc.set_alpha(0.4)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, fontsize=10)
    ax.set_ylabel("α₀ (mm/yr/°C)", fontsize=10)
    ax.set_title("Linear Sensitivity Distribution", fontsize=11)
    ax.grid(True, alpha=0.3, axis="y")

    # Panel 2: dα/dT distribution
    ax = axes[1]
    if "dalpha_dT_values" in combined_thermo:
        data_list2 = [combined_thermo["dalpha_dT_values"]]
        labels2 = ["Thermodynamic"]
        colors2 = ["#2166ac"]
        if combined_all is not None and "dalpha_dT_values" in combined_all:
            data_list2.append(combined_all["dalpha_dT_values"])
            labels2.append("All GMSL")
            colors2.append("#b2182b")

        parts = ax.violinplot(data_list2, positions=range(len(data_list2)),
                              showmeans=True, showmedians=True)
        for i, pc in enumerate(parts["bodies"]):
            pc.set_facecolor(colors2[i])
            pc.set_alpha(0.4)
        ax.set_xticks(range(len(labels2)))
        ax.set_xticklabels(labels2, fontsize=10)
        ax.set_ylabel("dα/dT (mm/yr/°C²)", fontsize=10)
        ax.set_title("Quadratic Sensitivity Distribution", fontsize=11)
        ax.axhline(0, color="gray", ls=":", alpha=0.5)
        ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    if fig_path:
        fig.savefig(fig_path, dpi=150, bbox_inches="tight")
        print(f"Saved: {fig_path}")
    plt.close(fig)
    return fig


# =====================================================================
#  6. MAIN ANALYSIS
# =====================================================================

def run_analysis(start_year=1950.0, order=2, n_lags=2,
                 exclude_from_figures=None, verbose=True):
    """Run the complete DOLS robustness analysis.

    Parameters
    ----------
    start_year : float
        Truncate all datasets to start at or after this year.
        Default 1950.0 = start of IPCC observed GMSL.
    order : int
        Polynomial order for DOLS (default 2 = quadratic).
    n_lags : int
        Number of leads/lags (default 2).
    exclude_from_figures : list of str, optional
        GMSL dataset names to exclude from figures (e.g. ["Horwath thermo"]).
        These datasets still appear in the matrix printout and are still
        excluded from ensemble statistics via min_n_obs, but are dropped
        from the heatmap and forest plot for visual clarity.
    verbose : bool

    Returns
    -------
    dict with all results, combined statistics, and figure objects
    """
    if exclude_from_figures is None:
        exclude_from_figures = ["Horwath thermo"]

    # Step 1: Run the robustness matrix
    rob = run_robustness_matrix(
        start_year=start_year, order=order, n_lags=n_lags, verbose=verbose)

    # Step 2: Combined ensemble — thermodynamic datasets only
    combined_thermo = compute_combined_dols(
        rob, thermodynamic_only=True, verbose=verbose)

    # Step 3: Combined ensemble — all datasets
    combined_all = compute_combined_dols(
        rob, thermodynamic_only=False, verbose=verbose)

    # Step 4: Visualizations
    if verbose:
        print()
        print("-" * 70)
        print("FIGURES")
        if exclude_from_figures:
            print(f"  (excluding from figures: {', '.join(exclude_from_figures)})")
        print("-" * 70)

    fig_heatmap = plot_coefficient_heatmaps(
        rob,
        fig_path=os.path.join(FIG_DIR, "dols_robustness_heatmap.png"),
        exclude_datasets=exclude_from_figures,
    )

    fig_forest = plot_coefficient_forest(
        rob, combined_thermo=combined_thermo, combined_all=combined_all,
        fig_path=os.path.join(FIG_DIR, "dols_robustness_forest.png"),
        exclude_datasets=exclude_from_figures,
    )

    fig_ensemble = plot_ensemble_summary(
        combined_thermo, combined_all=combined_all,
        fig_path=os.path.join(FIG_DIR, "dols_robustness_ensemble.png"),
    )

    # Step 5: Summary
    if verbose:
        print()
        print("=" * 70)
        print("SUMMARY")
        print("=" * 70)
        print(f"\nThermodynamic ensemble ({combined_thermo['n_pairs']} pairs):")
        print(f"  α₀ = {combined_thermo['alpha0_mean']:.2f} ± "
              f"{combined_thermo['alpha0_std']:.2f} mm/yr/°C")
        if "dalpha_dT_mean" in combined_thermo:
            print(f"  dα/dT = {combined_thermo['dalpha_dT_mean']:.2f} ± "
                  f"{combined_thermo['dalpha_dT_std']:.2f} mm/yr/°C²")
        print(f"\nAll-dataset ensemble ({combined_all['n_pairs']} pairs):")
        print(f"  α₀ = {combined_all['alpha0_mean']:.2f} ± "
              f"{combined_all['alpha0_std']:.2f} mm/yr/°C")
        if "dalpha_dT_mean" in combined_all:
            print(f"  dα/dT = {combined_all['dalpha_dT_mean']:.2f} ± "
                  f"{combined_all['dalpha_dT_std']:.2f} mm/yr/°C²")

    return {
        "robustness": rob,
        "combined_thermo": combined_thermo,
        "combined_all": combined_all,
        "fig_heatmap": fig_heatmap,
        "fig_forest": fig_forest,
        "fig_ensemble": fig_ensemble,
    }


# =====================================================================
if __name__ == "__main__":
    results = run_analysis(start_year=1950.0, order=2, n_lags=2, verbose=True)
