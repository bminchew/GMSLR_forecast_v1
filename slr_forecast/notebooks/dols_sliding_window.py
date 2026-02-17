#!/usr/bin/env python3
"""
Sliding-Window DOLS: Epoch Sensitivity of α₀ and dα/dT
========================================================

Revives the "Dynamic" in DOLS by sliding a kernel-weighted window through
the observational record to estimate time-varying DOLS coefficients:
  α₀(t)     — linear temperature sensitivity (mm/yr/°C)
  dα/dT(t)  — quadratic temperature sensitivity (mm/yr/°C²)

This analysis addresses the start-date sensitivity discovered in the
multi-dataset robustness analysis (§8.4 of the LaTeX document): DOLS
coefficients trade off dramatically depending on which epoch dominates
the fit.

Additionally, this script reconsiders SAOD (volcanic forcing) in the
sliding-window context.  While SAOD was found NOT significant for the
full-record static DOLS (γ_saod t-stat = 0.26), a sliding window may
reveal epochs where volcanic forcing matters — particularly around the
Pinatubo eruption (1991) or earlier eruptions captured in the
Mauna Loa transmission record.

Key analyses:
  1. Multi-bandwidth sliding-window DOLS (h = 30, 40, 50, 60 yr)
  2. Comparison across GMSL datasets at fixed bandwidth
  3. SAOD reconsideration in sliding-window context
  4. α₀ vs dα/dT tradeoff visualization

Datasets used:
  GMSL: Frederikse total/thermo, Dangendorf total, IPCC observed total/thermo
  GMST: Berkeley Earth (primary), with GMST sensitivity check
  SAOD: GloSSAC v2.23 (1979+), Mauna Loa transmission (1958+)

Authors: Minchew research group, 2026
"""

import sys
import os
import warnings
from typing import Optional, Dict, List, Tuple

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# Ensure local imports
sys.path.insert(0, os.path.dirname(__file__))
from slr_analysis import (
    calibrate_dols_sliding,
    calibrate_dols_sliding_multibandwidth,
    SlidingDOLSResult,
)
from dols_robustness import (
    load_all_gmsl,
    load_all_gmst,
    _datetime_to_decimal_year,
    _annualize_monthly,
    M_TO_MM,
    BASELINE_YEAR,
    H5_PATH,
)

# Paths
FIG_DIR = os.path.join(os.path.dirname(__file__), "..", "figures")
os.makedirs(FIG_DIR, exist_ok=True)


# =====================================================================
#  1. DATA LOADING HELPERS
# =====================================================================

def _build_aligned_series(gmsl_series, gmst_series, saod_series=None):
    """
    Align GMSL, GMST, and optional SAOD by year and return pd.Series
    with DatetimeIndex (required by calibrate_dols_sliding).

    Parameters
    ----------
    gmsl_series : pd.Series with float index (decimal year)
    gmst_series : pd.Series with float index (decimal year)
    saod_series : pd.Series with float index (decimal year), optional

    Returns
    -------
    sl, temp, saod_out : pd.Series with DatetimeIndex
    """
    # Map to integer years
    gmsl_years = np.floor(gmsl_series.index.values + 0.01).astype(int)
    gmst_years = np.floor(gmst_series.index.values + 0.01).astype(int)
    common_years = np.intersect1d(gmsl_years, gmst_years)

    if saod_series is not None:
        saod_years = np.floor(saod_series.index.values + 0.01).astype(int)
        common_years = np.intersect1d(common_years, saod_years)

    gmsl_vals, gmst_vals, saod_vals, dates = [], [], [], []
    for yr in common_years:
        gmsl_idx = np.argmin(np.abs(gmsl_series.index.values - yr))
        gmst_idx = np.argmin(np.abs(gmst_series.index.values - yr))
        gmsl_vals.append(gmsl_series.iloc[gmsl_idx])
        gmst_vals.append(gmst_series.iloc[gmst_idx])
        dates.append(pd.Timestamp(f"{yr}-07-01"))
        if saod_series is not None:
            saod_idx = np.argmin(np.abs(saod_series.index.values - yr))
            saod_vals.append(saod_series.iloc[saod_idx])

    dt_idx = pd.DatetimeIndex(dates)
    sl = pd.Series(gmsl_vals, index=dt_idx, name="sea_level")
    temp = pd.Series(gmst_vals, index=dt_idx, name="temperature")
    saod_out = None
    if saod_series is not None:
        saod_out = pd.Series(saod_vals, index=dt_idx, name="saod")

    return sl, temp, saod_out


def load_saod_data():
    """Load SAOD datasets from H5 store.

    Returns
    -------
    glossac : pd.Series with float index (decimal year), annual SAOD
    mlo : pd.Series with float index (decimal year), annual SAOD
    """
    store = pd.HDFStore(H5_PATH, mode="r")
    saod_data = {}

    # GloSSAC v2.23 (satellite era, 1979+)
    if "/raw/df_glossac" in store:
        df = store["/raw/df_glossac"]
        col = [c for c in df.columns if 'saod' in c.lower()
               or 'optical' in c.lower()]
        if not col:
            col = [df.columns[0]]
        annual = _annualize_monthly(df, col[0])
        s = pd.Series(annual[col[0]].values, index=annual.index, name="saod")
        saod_data["GloSSAC"] = s

    # Mauna Loa transmission → SAOD proxy
    if "/raw/df_mlo_transmission" in store:
        df = store["/raw/df_mlo_transmission"]
        col = [c for c in df.columns if 'saod' in c.lower()
               or 'transmission' in c.lower() or 'optical' in c.lower()]
        if not col:
            col = [df.columns[0]]
        annual = _annualize_monthly(df, col[0])
        s = pd.Series(annual[col[0]].values, index=annual.index, name="saod")
        saod_data["MLO"] = s

    store.close()
    return saod_data


# =====================================================================
#  2. MAIN SLIDING-WINDOW ANALYSIS
# =====================================================================

def run_sliding_window_analysis(
    start_year: float = 1850.0,
    order: int = 2,
    n_lags: int = 2,
    bandwidths: List[float] = [30, 40, 50, 60],
    primary_gmst: str = "Berkeley",
    verbose: bool = True,
) -> dict:
    """
    Run the full sliding-window DOLS analysis.

    Parameters
    ----------
    start_year : float
        Use native start dates (set to 1850 to get full records).
    order : int
        Polynomial order for DOLS.
    n_lags : int
        Number of leads/lags.
    bandwidths : list of float
        Kernel bandwidths to test.
    primary_gmst : str
        Primary GMST dataset name for multi-bandwidth analysis.
    verbose : bool

    Returns
    -------
    dict with keys:
        'multi_bw'     : {gmsl_name: {bw: SlidingDOLSResult}}
        'cross_dataset': {gmsl_name: SlidingDOLSResult} at fixed bw
        'saod_compare' : {gmsl_name: {'no_saod': ..., 'glossac': ..., 'mlo': ...}}
        'datasets_used': metadata
    """
    if verbose:
        print("=" * 70)
        print("SLIDING-WINDOW DOLS ANALYSIS")
        print(f"  order={order}, n_lags={n_lags}, start≥{start_year:.0f}")
        print(f"  bandwidths: {bandwidths}")
        print("=" * 70)
        print()

    # Load data with native start dates
    gmsl_data = load_all_gmsl(start_year=start_year)
    gmst_data = load_all_gmst(start_year=start_year)
    saod_data = load_saod_data()

    # Datasets for the main analysis (exclude Horwath and Dangendorf sterodynamic)
    exclude = {"Horwath thermo", "Dangendorf sterodynamic"}
    gmsl_names = [n for n in gmsl_data if n not in exclude]

    if verbose:
        print(f"GMSL datasets ({len(gmsl_names)}):")
        for name in gmsl_names:
            s = gmsl_data[name]
            print(f"  {name}: {s.index.min():.0f}–{s.index.max():.0f} (n={len(s)})")
        print(f"\nGMST datasets:")
        for name, s in gmst_data.items():
            print(f"  {name}: {s.index.min():.0f}–{s.index.max():.0f} (n={len(s)})")
        print(f"\nSAOD datasets:")
        for name, s in saod_data.items():
            print(f"  {name}: {s.index.min():.0f}–{s.index.max():.0f} (n={len(s)})")
        print()

    primary_gmst_series = gmst_data[primary_gmst]

    # ---- A. Multi-bandwidth analysis (each GMSL × primary GMST) ----
    if verbose:
        print("-" * 70)
        print("A. MULTI-BANDWIDTH ANALYSIS")
        print("-" * 70)

    multi_bw = {}
    for gmsl_name in gmsl_names:
        sl, temp, _ = _build_aligned_series(
            gmsl_data[gmsl_name], primary_gmst_series)
        if verbose:
            print(f"\n  {gmsl_name} × {primary_gmst} "
                  f"({len(sl)} common years)")

        bw_results = calibrate_dols_sliding_multibandwidth(
            sl, temp, bandwidths=bandwidths,
            order=order, n_lags=n_lags,
        )
        multi_bw[gmsl_name] = bw_results

        if verbose:
            for bw, res in bw_results.items():
                valid = ~np.isnan(res.alpha0)
                if valid.any():
                    a0 = res.alpha0[valid] * M_TO_MM
                    print(f"    h={bw:3.0f} yr: α₀ = "
                          f"[{a0.min():.2f}, {a0.max():.2f}] mm/yr/°C "
                          f"({valid.sum()} valid fits)")

    # ---- B. Cross-dataset comparison at fixed bandwidth (h=40) ----
    fixed_bw = 40.0
    if verbose:
        print()
        print("-" * 70)
        print(f"B. CROSS-DATASET COMPARISON (h={fixed_bw:.0f} yr)")
        print("-" * 70)

    cross_dataset = {}
    for gmsl_name in gmsl_names:
        sl, temp, _ = _build_aligned_series(
            gmsl_data[gmsl_name], primary_gmst_series)
        res = calibrate_dols_sliding(
            sl, temp, span_years=fixed_bw,
            order=order, n_lags=n_lags,
        )
        cross_dataset[gmsl_name] = res

        if verbose:
            valid = ~np.isnan(res.alpha0)
            if valid.any():
                a0 = res.alpha0[valid] * M_TO_MM
                da = res.dalpha_dT[valid] * M_TO_MM if res.dalpha_dT is not None else None
                line = f"  {gmsl_name:25s}: α₀ = [{a0.min():.2f}, {a0.max():.2f}]"
                if da is not None:
                    line += f", dα/dT = [{da.min():.2f}, {da.max():.2f}]"
                print(line)

    # ---- C. SAOD reconsideration in sliding-window context ----
    if verbose:
        print()
        print("-" * 70)
        print("C. SAOD RECONSIDERATION (sliding window)")
        print("-" * 70)

    saod_compare = {}
    # Use Frederikse thermo as primary for SAOD test
    saod_test_datasets = ["Frederikse thermo", "Frederikse"]
    saod_test_datasets = [n for n in saod_test_datasets if n in gmsl_data]

    for gmsl_name in saod_test_datasets:
        saod_compare[gmsl_name] = {}

        # Without SAOD
        sl, temp, _ = _build_aligned_series(
            gmsl_data[gmsl_name], primary_gmst_series)
        res_no_saod = calibrate_dols_sliding(
            sl, temp, span_years=fixed_bw,
            order=order, n_lags=n_lags,
        )
        saod_compare[gmsl_name]["no_saod"] = res_no_saod

        # With each SAOD dataset
        for saod_name, saod_series in saod_data.items():
            sl, temp, saod_aligned = _build_aligned_series(
                gmsl_data[gmsl_name], primary_gmst_series, saod_series)
            if saod_aligned is None or len(sl) < 20:
                if verbose:
                    print(f"  {gmsl_name} × {saod_name}: insufficient overlap")
                continue
            res_saod = calibrate_dols_sliding(
                sl, temp, saod=saod_aligned,
                span_years=fixed_bw, order=order, n_lags=n_lags,
            )
            saod_compare[gmsl_name][saod_name] = res_saod

            if verbose:
                valid = ~np.isnan(res_saod.gamma_saod) if res_saod.gamma_saod is not None else np.zeros(0, dtype=bool)
                if isinstance(valid, np.ndarray) and valid.any():
                    gamma = res_saod.gamma_saod[valid] * M_TO_MM
                    gamma_se = res_saod.gamma_saod_se[valid] * M_TO_MM
                    # t-statistics
                    t_stats = gamma / gamma_se
                    sig_pct = 100 * np.mean(np.abs(t_stats) > 1.96)
                    print(f"  {gmsl_name} + {saod_name}: "
                          f"γ_saod = [{gamma.min():.3f}, {gamma.max():.3f}] mm, "
                          f"{sig_pct:.0f}% of windows significant at 95%")

    return {
        "multi_bw": multi_bw,
        "cross_dataset": cross_dataset,
        "saod_compare": saod_compare,
        "datasets_used": {
            "gmsl_names": gmsl_names,
            "gmst": primary_gmst,
            "bandwidths": bandwidths,
            "fixed_bw": fixed_bw,
            "order": order,
            "n_lags": n_lags,
            "start_year": start_year,
        },
    }


# =====================================================================
#  3. VISUALIZATION
# =====================================================================

def plot_sliding_multibandwidth(multi_bw, gmsl_name, fig_path=None):
    """
    Fig A/B: α₀(t) and dα/dT(t) vs year for multiple bandwidths,
    with ±1σ envelope.

    Parameters
    ----------
    multi_bw : dict of {bandwidth: SlidingDOLSResult}
    gmsl_name : str
        Name of the GMSL dataset (for title).
    fig_path : str, optional
    """
    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    colors = plt.cm.viridis(np.linspace(0.2, 0.9, len(multi_bw)))

    for (bw, res), color in zip(sorted(multi_bw.items()), colors):
        valid = ~np.isnan(res.alpha0)
        if not valid.any():
            continue

        t = res.time[valid]
        a0 = res.alpha0[valid] * M_TO_MM
        a0_se = res.alpha0_se[valid] * M_TO_MM

        ax = axes[0]
        ax.plot(t, a0, color=color, lw=1.5, label=f"h={bw:.0f} yr")
        ax.fill_between(t, a0 - a0_se, a0 + a0_se,
                         color=color, alpha=0.15)

        if res.dalpha_dT is not None:
            da = res.dalpha_dT[valid] * M_TO_MM
            da_se = res.dalpha_dT_se[valid] * M_TO_MM
            ax = axes[1]
            ax.plot(t, da, color=color, lw=1.5, label=f"h={bw:.0f} yr")
            ax.fill_between(t, da - da_se, da + da_se,
                             color=color, alpha=0.15)

    axes[0].set_ylabel("α₀ (mm/yr/°C)", fontsize=11)
    axes[0].set_title(f"Sliding-Window DOLS: {gmsl_name}", fontsize=13,
                       fontweight="bold")
    axes[0].axhline(0, color="gray", ls=":", alpha=0.5)
    axes[0].legend(fontsize=9, loc="upper left")
    axes[0].grid(True, alpha=0.3)

    axes[1].set_ylabel("dα/dT (mm/yr/°C²)", fontsize=11)
    axes[1].set_xlabel("Center Year", fontsize=11)
    axes[1].axhline(0, color="gray", ls=":", alpha=0.5)
    axes[1].legend(fontsize=9, loc="upper left")
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    if fig_path:
        fig.savefig(fig_path, dpi=150, bbox_inches="tight")
        print(f"Saved: {fig_path}")
    plt.close(fig)
    return fig


def plot_sliding_cross_dataset(cross_dataset, fig_path=None):
    """
    Fig C: Comparison across GMSL datasets at fixed bandwidth.
    """
    # Separate thermodynamic and total
    thermo_names = {"Frederikse thermo", "IPCC obs thermo"}
    total_names = {"Frederikse", "Dangendorf", "IPCC observed"}

    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    colors = plt.cm.Set2(np.linspace(0, 1, len(cross_dataset)))

    for (name, res), color in zip(cross_dataset.items(), colors):
        valid = ~np.isnan(res.alpha0)
        if not valid.any():
            continue

        t = res.time[valid]
        a0 = res.alpha0[valid] * M_TO_MM
        ls = "-" if name in thermo_names else "--"
        lw = 2.0 if name in thermo_names else 1.5

        axes[0].plot(t, a0, color=color, ls=ls, lw=lw, label=name)

        if res.dalpha_dT is not None:
            da = res.dalpha_dT[valid] * M_TO_MM
            axes[1].plot(t, da, color=color, ls=ls, lw=lw, label=name)

    bw = list(cross_dataset.values())[0].span_years
    axes[0].set_ylabel("α₀ (mm/yr/°C)", fontsize=11)
    axes[0].set_title(f"Cross-Dataset Sliding DOLS (h={bw:.0f} yr)",
                       fontsize=13, fontweight="bold")
    axes[0].axhline(0, color="gray", ls=":", alpha=0.5)
    axes[0].legend(fontsize=8, loc="upper left", ncol=2)
    axes[0].grid(True, alpha=0.3)

    axes[1].set_ylabel("dα/dT (mm/yr/°C²)", fontsize=11)
    axes[1].set_xlabel("Center Year", fontsize=11)
    axes[1].axhline(0, color="gray", ls=":", alpha=0.5)
    axes[1].legend(fontsize=8, loc="upper left", ncol=2)
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    if fig_path:
        fig.savefig(fig_path, dpi=150, bbox_inches="tight")
        print(f"Saved: {fig_path}")
    plt.close(fig)
    return fig


def plot_alpha_tradeoff(cross_dataset, fig_path=None):
    """
    Fig D: α₀ vs dα/dT scatterplot colored by center year.
    Shows the tradeoff discovered in §8.4.
    """
    fig, ax = plt.subplots(figsize=(8, 6))

    for name, res in cross_dataset.items():
        if res.dalpha_dT is None:
            continue
        valid = ~np.isnan(res.alpha0) & ~np.isnan(res.dalpha_dT)
        if not valid.any():
            continue

        a0 = res.alpha0[valid] * M_TO_MM
        da = res.dalpha_dT[valid] * M_TO_MM
        t = res.time[valid]

        sc = ax.scatter(a0, da, c=t, cmap="viridis", s=15, alpha=0.6,
                        label=name)

    plt.colorbar(sc, ax=ax, label="Center Year")
    ax.set_xlabel("α₀ (mm/yr/°C)", fontsize=11)
    ax.set_ylabel("dα/dT (mm/yr/°C²)", fontsize=11)
    ax.set_title("α₀ vs dα/dT Tradeoff (colored by epoch)", fontsize=13,
                  fontweight="bold")
    ax.axhline(0, color="gray", ls=":", alpha=0.5)
    ax.axvline(0, color="gray", ls=":", alpha=0.5)
    ax.legend(fontsize=8, loc="upper right")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    if fig_path:
        fig.savefig(fig_path, dpi=150, bbox_inches="tight")
        print(f"Saved: {fig_path}")
    plt.close(fig)
    return fig


def plot_saod_comparison(saod_compare, fig_path=None):
    """
    Fig E: SAOD impact on sliding-window DOLS coefficients.
    Compare α₀(t) with and without SAOD, plus γ_saod(t).
    """
    n_datasets = len(saod_compare)
    if n_datasets == 0:
        return None

    fig, axes = plt.subplots(n_datasets, 2, figsize=(14, 4 * n_datasets),
                              squeeze=False)

    for row, (gmsl_name, variants) in enumerate(saod_compare.items()):
        # Left panel: α₀ with/without SAOD
        ax = axes[row, 0]
        colors = {"no_saod": "#333333", "GloSSAC": "#d62728", "MLO": "#2ca02c"}

        for var_name, res in variants.items():
            valid = ~np.isnan(res.alpha0)
            if not valid.any():
                continue
            t = res.time[valid]
            a0 = res.alpha0[valid] * M_TO_MM
            color = colors.get(var_name, "gray")
            label = f"{'No SAOD' if var_name == 'no_saod' else var_name}"
            ax.plot(t, a0, color=color, lw=1.5, label=label)

        ax.set_ylabel("α₀ (mm/yr/°C)", fontsize=10)
        ax.set_title(f"{gmsl_name}: α₀ ± SAOD", fontsize=11)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.axhline(0, color="gray", ls=":", alpha=0.3)

        # Right panel: γ_saod(t) with significance shading
        ax = axes[row, 1]
        for var_name, res in variants.items():
            if var_name == "no_saod" or res.gamma_saod is None:
                continue
            valid = ~np.isnan(res.gamma_saod)
            if not valid.any():
                continue
            t = res.time[valid]
            gamma = res.gamma_saod[valid] * M_TO_MM
            gamma_se = res.gamma_saod_se[valid] * M_TO_MM
            color = colors.get(var_name, "gray")
            ax.plot(t, gamma, color=color, lw=1.5, label=var_name)
            ax.fill_between(t, gamma - 1.96 * gamma_se,
                             gamma + 1.96 * gamma_se,
                             color=color, alpha=0.15)
            # Mark significant windows
            t_stat = np.abs(gamma / gamma_se)
            sig = t_stat > 1.96
            if sig.any():
                ax.scatter(t[sig], gamma[sig], color=color, s=12,
                           zorder=5, marker="*")

        ax.set_ylabel("γ_saod (mm)", fontsize=10)
        ax.set_title(f"{gmsl_name}: SAOD coefficient", fontsize=11)
        ax.axhline(0, color="gray", ls=":", alpha=0.5)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    for ax in axes[-1]:
        ax.set_xlabel("Center Year", fontsize=10)

    plt.tight_layout()
    if fig_path:
        fig.savefig(fig_path, dpi=150, bbox_inches="tight")
        print(f"Saved: {fig_path}")
    plt.close(fig)
    return fig


# =====================================================================
#  4. SUMMARY TABLE
# =====================================================================

def print_summary_table(results, verbose=True):
    """
    Print a summary table of sliding-window results.
    Reports the range and median of α₀ and dα/dT across center years
    for each GMSL dataset at each bandwidth.
    """
    if not verbose:
        return

    print()
    print("=" * 90)
    print("SLIDING-WINDOW DOLS SUMMARY TABLE")
    print("=" * 90)
    print(f"{'GMSL dataset':<25s} {'BW':>4s} {'α₀ median':>10s} "
          f"{'α₀ range':>16s} {'dα/dT median':>12s} {'dα/dT range':>16s} "
          f"{'n_valid':>7s}")
    print("-" * 90)

    for gmsl_name, bw_results in results["multi_bw"].items():
        for bw, res in sorted(bw_results.items()):
            valid = ~np.isnan(res.alpha0)
            if not valid.any():
                continue

            a0 = res.alpha0[valid] * M_TO_MM
            a0_med = np.median(a0)
            a0_range = f"[{a0.min():.2f}, {a0.max():.2f}]"

            if res.dalpha_dT is not None:
                da = res.dalpha_dT[valid] * M_TO_MM
                da_med = np.median(da)
                da_range = f"[{da.min():.2f}, {da.max():.2f}]"
            else:
                da_med = float('nan')
                da_range = "—"

            print(f"{gmsl_name:<25s} {bw:4.0f} {a0_med:10.2f} "
                  f"{a0_range:>16s} {da_med:12.2f} {da_range:>16s} "
                  f"{valid.sum():7d}")

    # SAOD summary
    if results["saod_compare"]:
        print()
        print("-" * 90)
        print("SAOD SENSITIVITY SUMMARY")
        print("-" * 90)
        print(f"{'GMSL dataset':<25s} {'SAOD source':>12s} "
              f"{'γ median':>10s} {'γ range':>16s} {'% sig (95%)':>12s}")
        print("-" * 90)
        for gmsl_name, variants in results["saod_compare"].items():
            for var_name, res in variants.items():
                if var_name == "no_saod" or res.gamma_saod is None:
                    continue
                valid = ~np.isnan(res.gamma_saod)
                if not valid.any():
                    continue
                gamma = res.gamma_saod[valid] * M_TO_MM
                gamma_se = res.gamma_saod_se[valid] * M_TO_MM
                t_stat = np.abs(gamma / gamma_se)
                sig_pct = 100 * np.mean(t_stat > 1.96)
                print(f"{gmsl_name:<25s} {var_name:>12s} "
                      f"{np.median(gamma):10.3f} "
                      f"[{gamma.min():.3f}, {gamma.max():.3f}] "
                      f"{sig_pct:10.1f}%")


# =====================================================================
#  5. MAIN ENTRY POINT
# =====================================================================

def run_analysis(
    start_year: float = 1850.0,
    order: int = 2,
    n_lags: int = 2,
    bandwidths: List[float] = [30, 40, 50, 60],
    verbose: bool = True,
) -> dict:
    """
    Run the complete sliding-window DOLS analysis with all figures.

    Parameters
    ----------
    start_year : float
        Use native start dates (1850 for full records).
    order : int
        Polynomial order (default 2 = quadratic).
    n_lags : int
        Number of leads/lags (default 2).
    bandwidths : list of float
        Kernel bandwidths to test.
    verbose : bool

    Returns
    -------
    dict with analysis results and figure objects
    """
    # Run analysis
    results = run_sliding_window_analysis(
        start_year=start_year, order=order, n_lags=n_lags,
        bandwidths=bandwidths, verbose=verbose,
    )

    # Print summary table
    print_summary_table(results, verbose=verbose)

    # Generate figures
    if verbose:
        print()
        print("-" * 70)
        print("FIGURES")
        print("-" * 70)

    figs = {}

    # Fig A/B: Multi-bandwidth for each GMSL dataset
    for gmsl_name, bw_results in results["multi_bw"].items():
        safe_name = gmsl_name.lower().replace(" ", "_")
        fig_path = os.path.join(FIG_DIR,
                                f"dols_sliding_multibw_{safe_name}.png")
        figs[f"multibw_{safe_name}"] = plot_sliding_multibandwidth(
            bw_results, gmsl_name, fig_path=fig_path)

    # Fig C: Cross-dataset comparison
    figs["cross_dataset"] = plot_sliding_cross_dataset(
        results["cross_dataset"],
        fig_path=os.path.join(FIG_DIR, "dols_sliding_cross_dataset.png"),
    )

    # Fig D: α₀ vs dα/dT tradeoff
    figs["alpha_tradeoff"] = plot_alpha_tradeoff(
        results["cross_dataset"],
        fig_path=os.path.join(FIG_DIR, "dols_sliding_alpha_tradeoff.png"),
    )

    # Fig E: SAOD comparison
    if results["saod_compare"]:
        figs["saod_comparison"] = plot_saod_comparison(
            results["saod_compare"],
            fig_path=os.path.join(FIG_DIR, "dols_sliding_saod_comparison.png"),
        )

    results["figures"] = figs
    return results


# =====================================================================
if __name__ == "__main__":
    results = run_analysis(
        start_year=1850.0, order=2, n_lags=2,
        bandwidths=[30, 40, 50, 60], verbose=True,
    )
