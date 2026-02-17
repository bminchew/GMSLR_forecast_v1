#!/usr/bin/env python3
"""
Bayesian Analysis of DOLS Sea-Level Rate Model
================================================

Fits three complementary Bayesian models and generates diagnostics:
1. Static Bayesian DOLS (frequentist comparison)
2. Dynamic Linear Model (time-varying coefficients)
3. Hierarchical multi-dataset model (partial pooling)

Usage:
    python bayesian_analysis.py [--skip-dlm] [--skip-hierarchical] [--quick]

Authors: Minchew research group, 2026
"""

import sys
import os
import argparse
import warnings
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import arviz as az

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

# Local imports
sys.path.insert(0, os.path.dirname(__file__))
from slr_analysis import calibrate_dols, DOLSResult
from dols_robustness import load_all_gmsl, load_all_gmst, _align_and_run_dols
from bayesian_dols import (
    build_dols_design_matrix, fit_bayesian_dols,
    fit_bayesian_dlm, fit_hierarchical_dols,
    BayesianDOLSResult, BayesianDLMResult, HierarchicalDOLSResult,
)

# Paths
DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
H5_PATH = os.path.join(DATA_DIR, "processed", "slr_processed_data.h5")
FIG_DIR = os.path.join(os.path.dirname(__file__), "..", "figures")
os.makedirs(FIG_DIR, exist_ok=True)

M_TO_MM = 1000.0


# =====================================================================
#  HELPER: Load primary dataset (Frederikse + Berkeley)
# =====================================================================

def load_primary_dataset():
    """Load Frederikse GMSL + Berkeley Earth temperature with DatetimeIndex."""
    store = pd.HDFStore(H5_PATH, mode="r")
    df_fred = store["/raw/df_frederikse"]
    df_berk = store["/raw/df_berkeley"]
    store.close()

    sl = df_fred["gmsl"]
    sl_sigma = df_fred.get("gmsl_sigma", None)
    temp = df_berk["temperature"]
    return sl, sl_sigma, temp


def load_hierarchical_datasets(start_year=1950.0, exclude_sterodynamic=True):
    """Load GMSL datasets for hierarchical model with DatetimeIndex.

    Returns dict of {name: pd.Series} and a common temperature Series.
    """
    gmsl_all = load_all_gmsl(start_year=start_year)
    gmst_all = load_all_gmst(start_year=start_year)

    # Exclude datasets
    exclude = {'Horwath thermo'}
    if exclude_sterodynamic:
        exclude.add('Dangendorf sterodynamic')

    # Convert to DatetimeIndex (required by build_dols_design_matrix)
    datasets = {}
    for name, s in gmsl_all.items():
        if name in exclude:
            continue
        dates = pd.DatetimeIndex([pd.Timestamp(f"{int(y)}-07-01") for y in s.index])
        datasets[name] = pd.Series(s.values, index=dates, name="sea_level")

    # Common temperature: Berkeley Earth
    berk = gmst_all["Berkeley"]
    berk_dates = pd.DatetimeIndex([pd.Timestamp(f"{int(y)}-07-01") for y in berk.index])
    temp = pd.Series(berk.values, index=berk_dates, name="temperature")

    return datasets, temp


# =====================================================================
#  FIGURE 1: Bayesian Static — Posterior vs Frequentist
# =====================================================================

def plot_static_comparison(bayes_result, freq_result, save=True):
    """Corner-style comparison of Bayesian posteriors vs frequentist estimates."""
    trace = bayes_result.trace
    n_phys = bayes_result.order + 1  # order=2 → 3 params
    phys_names_display = []
    if bayes_result.order >= 3:
        phys_names_display.append(r"$d^2\alpha/dT^2$")
    if bayes_result.order >= 2:
        phys_names_display.append(r"$d\alpha/dT$")
    phys_names_display.extend([r"$\alpha_0$", "trend"])

    fig, axes = plt.subplots(1, n_phys, figsize=(4 * n_phys, 3.5), squeeze=False)
    axes = axes[0]

    var_names = list(trace.posterior.data_vars)
    for k in range(n_phys):
        ax = axes[k]
        samples = trace.posterior[var_names[k]].values.flatten() * M_TO_MM

        ax.hist(samples, bins=50, density=True, alpha=0.6, color="steelblue",
                label="Bayesian posterior")

        # Frequentist point ± 1σ
        freq_val = freq_result.physical_coefficients[k] * M_TO_MM
        freq_se = freq_result.physical_se[k] * M_TO_MM
        ax.axvline(freq_val, color="crimson", lw=2, label="Freq. estimate")
        ax.axvspan(freq_val - freq_se, freq_val + freq_se,
                   color="crimson", alpha=0.15, label="Freq. ±1σ")

        ax.set_xlabel(f"{phys_names_display[k]} (mm/yr" + ("/°C²" if k == 0 and n_phys == 3 else "/°C" if k == 1 and n_phys == 3 else "") + ")")
        ax.set_ylabel("Density" if k == 0 else "")
        ax.legend(fontsize=7)

    fig.suptitle("Bayesian Static DOLS: Posterior vs Frequentist", fontsize=12)
    plt.tight_layout()
    if save:
        path = os.path.join(FIG_DIR, "bayesian_static_posterior.png")
        fig.savefig(path, dpi=150, bbox_inches="tight")
        print(f"  Saved: {path}")
    plt.close(fig)


# =====================================================================
#  FIGURE 2: Bayesian Static — Forest Plot
# =====================================================================

def plot_static_forest(bayes_result, freq_result, save=True):
    """Forest plot comparing Bayesian HDI to frequentist CI."""
    n_phys = bayes_result.order + 1
    labels = []
    if bayes_result.order >= 3:
        labels.append(r"$d^2\alpha/dT^2$")
    if bayes_result.order >= 2:
        labels.append(r"$d\alpha/dT$")
    labels.extend([r"$\alpha_0$", "trend"])

    fig, ax = plt.subplots(figsize=(6, 2 + 0.6 * n_phys))

    y_pos = np.arange(n_phys)
    for k in range(n_phys):
        # Bayesian
        b_mean = bayes_result.physical_coefficients[k] * M_TO_MM
        b_lo, b_hi = bayes_result.physical_hdi_94[k] * M_TO_MM
        ax.errorbar(b_mean, y_pos[k] + 0.15, xerr=[[b_mean - b_lo], [b_hi - b_mean]],
                    fmt='o', color='steelblue', capsize=4, label='Bayesian 94% HDI' if k == 0 else None)

        # Frequentist
        f_mean = freq_result.physical_coefficients[k] * M_TO_MM
        f_se = freq_result.physical_se[k] * M_TO_MM
        ax.errorbar(f_mean, y_pos[k] - 0.15, xerr=1.96 * f_se,
                    fmt='s', color='crimson', capsize=4, label='Freq. 95% CI' if k == 0 else None)

    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels)
    ax.set_xlabel("Coefficient value (mm/yr units)")
    ax.axvline(0, color='gray', ls='--', lw=0.5)
    ax.legend(fontsize=8)
    ax.set_title("Static DOLS: Bayesian vs Frequentist")
    plt.tight_layout()
    if save:
        path = os.path.join(FIG_DIR, "bayesian_static_comparison.png")
        fig.savefig(path, dpi=150, bbox_inches="tight")
        print(f"  Saved: {path}")
    plt.close(fig)


# =====================================================================
#  FIGURE 3: DLM Time-Varying Coefficients
# =====================================================================

def plot_dlm_coefficients(dlm_result, freq_result=None, sliding_window=None, save=True):
    """Plot DLM time-varying coefficients with HDI bands."""
    time = dlm_result.time
    n_phys = dlm_result.coefficients_mean.shape[1]

    labels = []
    if n_phys >= 3:
        labels.append(r"$d\alpha/dT$ (mm/yr/°C²)")
    labels.append(r"$\alpha_0$ (mm/yr/°C)")
    labels.append("trend (mm/yr)")

    fig, axes = plt.subplots(n_phys, 1, figsize=(8, 3 * n_phys), sharex=True)
    if n_phys == 1:
        axes = [axes]

    for k in range(n_phys):
        ax = axes[k]
        mean_k = dlm_result.coefficients_mean[:, k] * M_TO_MM
        lo_k = dlm_result.coefficients_hdi[:, k, 0] * M_TO_MM
        hi_k = dlm_result.coefficients_hdi[:, k, 1] * M_TO_MM

        ax.plot(time, mean_k, color='steelblue', lw=1.5, label='DLM posterior mean')
        ax.fill_between(time, lo_k, hi_k, alpha=0.2, color='steelblue',
                        label='DLM 94% HDI')

        # Frequentist constant reference
        if freq_result is not None:
            f_val = freq_result.physical_coefficients[k] * M_TO_MM
            f_se = freq_result.physical_se[k] * M_TO_MM
            ax.axhline(f_val, color='crimson', ls='--', lw=1, label='Freq. DOLS')
            ax.axhspan(f_val - 1.96 * f_se, f_val + 1.96 * f_se,
                       color='crimson', alpha=0.08)

        ax.set_ylabel(labels[k])
        ax.legend(fontsize=7, loc='best')

    axes[-1].set_xlabel("Year")
    fig.suptitle("DLM: Time-Varying Coefficients", fontsize=12)
    plt.tight_layout()
    if save:
        path = os.path.join(FIG_DIR, "bayesian_dlm_coefficients.png")
        fig.savefig(path, dpi=150, bbox_inches="tight")
        print(f"  Saved: {path}")
    plt.close(fig)


# =====================================================================
#  FIGURE 4: DLM Q Posterior
# =====================================================================

def plot_dlm_Q_posterior(dlm_result, save=True):
    """Posterior distribution of evolution noise Q."""
    trace = dlm_result.trace
    var_names = list(trace.posterior.data_vars)
    n_Q = len(var_names)

    labels_Q = []
    if n_Q >= 3:
        labels_Q.append(r"$Q_{d\alpha/dT}$")
    labels_Q.append(r"$Q_{\alpha_0}$")
    labels_Q.append(r"$Q_{\mathrm{trend}}$")

    fig, axes = plt.subplots(1, n_Q, figsize=(4 * n_Q, 3.5), squeeze=False)
    axes = axes[0]

    for k in range(n_Q):
        ax = axes[k]
        samples = trace.posterior[var_names[k]].values.flatten()
        ax.hist(samples, bins=50, density=True, alpha=0.7, color="teal")
        ax.set_xlabel(f"{labels_Q[k]}")
        ax.set_ylabel("Density" if k == 0 else "")
        med = np.median(samples)
        ax.axvline(med, color='red', ls='--', label=f"median={med:.2e}")
        ax.legend(fontsize=7)

    fig.suptitle("DLM: Evolution Noise Q Posterior", fontsize=12)
    plt.tight_layout()
    if save:
        path = os.path.join(FIG_DIR, "bayesian_dlm_Q_posterior.png")
        fig.savefig(path, dpi=150, bbox_inches="tight")
        print(f"  Saved: {path}")
    plt.close(fig)


# =====================================================================
#  FIGURE 5: DLM Fixed vs Bayesian Q
# =====================================================================

def plot_dlm_comparison(dlm_bayes, dlm_fixed, save=True):
    """Compare coefficient trajectories under Bayesian vs fixed Q."""
    time = dlm_bayes.time
    n_phys = dlm_bayes.coefficients_mean.shape[1]

    labels = []
    if n_phys >= 3:
        labels.append(r"$d\alpha/dT$")
    labels.append(r"$\alpha_0$")
    labels.append("trend")

    fig, axes = plt.subplots(n_phys, 1, figsize=(8, 3 * n_phys), sharex=True)
    if n_phys == 1:
        axes = [axes]

    for k in range(n_phys):
        ax = axes[k]
        # Bayesian Q
        ax.plot(time, dlm_bayes.coefficients_mean[:, k] * M_TO_MM,
                color='steelblue', lw=1.5, label='Bayesian Q')
        ax.fill_between(time,
                        dlm_bayes.coefficients_hdi[:, k, 0] * M_TO_MM,
                        dlm_bayes.coefficients_hdi[:, k, 1] * M_TO_MM,
                        alpha=0.15, color='steelblue')
        # Fixed Q
        ax.plot(time, dlm_fixed.coefficients_mean[:, k] * M_TO_MM,
                color='darkorange', lw=1.5, ls='--', label=f'Fixed Q={dlm_fixed.Q_fixed:.1e}')
        ax.fill_between(time,
                        dlm_fixed.coefficients_hdi[:, k, 0] * M_TO_MM,
                        dlm_fixed.coefficients_hdi[:, k, 1] * M_TO_MM,
                        alpha=0.15, color='darkorange')

        ax.set_ylabel(f"{labels[k]} (mm/yr)")
        ax.legend(fontsize=7)

    axes[-1].set_xlabel("Year")
    fig.suptitle("DLM: Bayesian vs Fixed Evolution Noise", fontsize=12)
    plt.tight_layout()
    if save:
        path = os.path.join(FIG_DIR, "bayesian_dlm_fixed_vs_bayesian_Q.png")
        fig.savefig(path, dpi=150, bbox_inches="tight")
        print(f"  Saved: {path}")
    plt.close(fig)


# =====================================================================
#  FIGURE 6: Hierarchical Forest Plot
# =====================================================================

def plot_hierarchical_forest(hier_result, save=True):
    """Forest plot of dataset-specific + population coefficients."""
    names = list(hier_result.dataset_coefficients.keys())
    n_datasets = len(names)
    n_phys = len(hier_result.population_mean)

    labels = []
    if n_phys >= 3:
        labels.append(r"$d\alpha/dT$ (mm/yr/°C²)")
    labels.append(r"$\alpha_0$ (mm/yr/°C)")
    labels.append("trend (mm/yr)")

    fig, axes = plt.subplots(1, n_phys, figsize=(5 * n_phys, 3 + 0.5 * n_datasets))

    for k in range(n_phys):
        ax = axes[k]
        y_pos = np.arange(n_datasets + 1)

        # Population
        pop_val = hier_result.population_mean[k] * M_TO_MM
        pop_sd = hier_result.population_sd[k] * M_TO_MM
        ax.errorbar(pop_val, n_datasets, xerr=pop_sd,
                    fmt='D', color='black', markersize=8, capsize=5,
                    label='Population mean ±σ', zorder=10)

        # Datasets
        for i, dname in enumerate(names):
            d_val = hier_result.dataset_coefficients[dname][k] * M_TO_MM
            d_hdi = hier_result.dataset_hdi[dname][k] * M_TO_MM
            ax.errorbar(d_val, i, xerr=[[d_val - d_hdi[0]], [d_hdi[1] - d_val]],
                        fmt='o', color='steelblue', capsize=3)

        ax.set_yticks(y_pos)
        ax.set_yticklabels(names + ["Population"], fontsize=8)
        ax.axvline(0, color='gray', ls='--', lw=0.5)
        ax.set_xlabel(labels[k])
        if k == 0:
            ax.legend(fontsize=7)

    fig.suptitle("Hierarchical DOLS: Dataset-Specific vs Population Coefficients", fontsize=11)
    plt.tight_layout()
    if save:
        path = os.path.join(FIG_DIR, "bayesian_hierarchical_forest.png")
        fig.savefig(path, dpi=150, bbox_inches="tight")
        print(f"  Saved: {path}")
    plt.close(fig)


# =====================================================================
#  FIGURE 7: Hierarchical Shrinkage
# =====================================================================

def plot_hierarchical_shrinkage(hier_result, save=True):
    """Bar chart of shrinkage factors per dataset."""
    names = list(hier_result.shrinkage_factors.keys())
    shrink = [hier_result.shrinkage_factors[n] for n in names]

    fig, ax = plt.subplots(figsize=(7, 3))
    y_pos = np.arange(len(names))
    bars = ax.barh(y_pos, shrink, color='steelblue', alpha=0.7)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(names, fontsize=9)
    ax.set_xlabel("Shrinkage factor (0 = no pooling, 1 = full pooling)")
    ax.set_xlim(0, 1)
    ax.set_title("Hierarchical DOLS: Shrinkage Toward Population Mean")

    for bar, s in zip(bars, shrink):
        ax.text(bar.get_width() + 0.02, bar.get_y() + bar.get_height() / 2,
                f"{s:.2f}", va='center', fontsize=8)

    plt.tight_layout()
    if save:
        path = os.path.join(FIG_DIR, "bayesian_hierarchical_shrinkage.png")
        fig.savefig(path, dpi=150, bbox_inches="tight")
        print(f"  Saved: {path}")
    plt.close(fig)


# =====================================================================
#  MAIN
# =====================================================================

def main():
    parser = argparse.ArgumentParser(description="Bayesian DOLS analysis")
    parser.add_argument("--skip-dlm", action="store_true",
                        help="Skip DLM model fitting")
    parser.add_argument("--skip-hierarchical", action="store_true",
                        help="Skip hierarchical model fitting")
    parser.add_argument("--quick", action="store_true",
                        help="Use fewer samples for faster execution")
    args = parser.parse_args()

    if args.quick:
        n_samples = 500
        n_burnin = 300
        n_walkers_static = 32
        n_walkers_dlm = 24
    else:
        n_samples = 2000
        n_burnin = 1000
        n_walkers_static = 48
        n_walkers_dlm = 32

    print("=" * 60)
    print("Bayesian Analysis of DOLS Sea-Level Rate Model")
    print(f"  Date: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print(f"  Mode: {'quick' if args.quick else 'full'}")
    print("=" * 60)

    # ---- 1. Load primary data ----
    print("\n1. Loading data...")
    sl, sl_sigma, temp = load_primary_dataset()
    print(f"   Sea level: {len(sl)} obs, {sl.index[0]} to {sl.index[-1]}")
    print(f"   Temperature: {len(temp)} obs")

    # ---- 2. Frequentist reference ----
    print("\n2. Frequentist DOLS (reference)...")
    freq = calibrate_dols(sl, temp, gmsl_sigma=sl_sigma, order=2, n_lags=2)
    print(f"   dα/dT = {freq.dalpha_dT * M_TO_MM:.3f} ± {freq.dalpha_dT_se * M_TO_MM:.3f} mm/yr/°C²")
    print(f"   α₀    = {freq.alpha0 * M_TO_MM:.3f} ± {freq.alpha0_se * M_TO_MM:.3f} mm/yr/°C")
    print(f"   trend  = {freq.trend * M_TO_MM:.3f} ± {freq.trend_se * M_TO_MM:.3f} mm/yr")
    print(f"   R² = {freq.r2:.6f}")

    # ---- 3. Bayesian static DOLS ----
    print("\n3. Bayesian Static DOLS...")
    bayes_static = fit_bayesian_dols(
        sl, temp, gmsl_sigma=sl_sigma, order=2, n_lags=2,
        n_samples=n_samples, n_walkers=n_walkers_static, n_burnin=n_burnin,
        progress=True
    )
    print(f"   dα/dT = {bayes_static.physical_coefficients[0] * M_TO_MM:.3f} "
          f"[{bayes_static.physical_hdi_94[0, 0] * M_TO_MM:.3f}, "
          f"{bayes_static.physical_hdi_94[0, 1] * M_TO_MM:.3f}]")
    print(f"   α₀    = {bayes_static.physical_coefficients[1] * M_TO_MM:.3f} "
          f"[{bayes_static.physical_hdi_94[1, 0] * M_TO_MM:.3f}, "
          f"{bayes_static.physical_hdi_94[1, 1] * M_TO_MM:.3f}]")
    print(f"   trend  = {bayes_static.physical_coefficients[2] * M_TO_MM:.3f} "
          f"[{bayes_static.physical_hdi_94[2, 0] * M_TO_MM:.3f}, "
          f"{bayes_static.physical_hdi_94[2, 1] * M_TO_MM:.3f}]")
    print(f"   R² = {bayes_static.r2:.6f}")
    print(f"   Acceptance = {bayes_static.sampler_diagnostics['acceptance_fraction']:.3f}")

    # Figures
    print("\n   Generating static DOLS figures...")
    plot_static_comparison(bayes_static, freq)
    plot_static_forest(bayes_static, freq)

    # ---- 4. Bayesian DLM ----
    dlm_bayes = None
    dlm_fixed = None
    if not args.skip_dlm:
        print("\n4. Bayesian DLM (fully Bayesian Q)...")
        dlm_bayes = fit_bayesian_dlm(
            sl, temp, gmsl_sigma=sl_sigma, order=2, n_lags=2,
            n_samples=n_samples, n_walkers=n_walkers_dlm, n_burnin=n_burnin,
            progress=True
        )
        Q_med = dlm_bayes.Q_posterior
        print(f"   Q posterior median: {Q_med}")
        print(f"   Final coefficients: {dlm_bayes.coefficients_mean[-1] * M_TO_MM}")
        print(f"   Acceptance = {dlm_bayes.sampler_diagnostics['acceptance_fraction']:.3f}")

        # Fixed-Q comparison using posterior median
        print("\n   DLM (fixed Q = posterior median)...")
        Q_fixed_val = np.median(Q_med) if Q_med is not None else 1e-6
        dlm_fixed = fit_bayesian_dlm(
            sl, temp, gmsl_sigma=sl_sigma, order=2, n_lags=2,
            Q_fixed=Q_fixed_val, progress=False
        )
        print(f"   Fixed Q = {Q_fixed_val:.2e}")

        # Figures
        print("\n   Generating DLM figures...")
        plot_dlm_coefficients(dlm_bayes, freq)
        plot_dlm_Q_posterior(dlm_bayes)
        plot_dlm_comparison(dlm_bayes, dlm_fixed)
    else:
        print("\n4. DLM skipped.")

    # ---- 5. Hierarchical model ----
    hier_result = None
    if not args.skip_hierarchical:
        print("\n5. Hierarchical Multi-Dataset DOLS...")
        datasets, temp_hier = load_hierarchical_datasets(
            start_year=1950.0, exclude_sterodynamic=True
        )
        print(f"   Datasets: {list(datasets.keys())}")

        hier_result = fit_hierarchical_dols(
            datasets, temp_hier, order=2, n_lags=2,
            n_samples=n_samples, n_burnin=n_burnin,
            progress=True
        )
        print(f"   Population dα/dT = {hier_result.population_mean[0] * M_TO_MM:.3f} "
              f"± {hier_result.population_sd[0] * M_TO_MM:.3f} mm/yr/°C²")
        print(f"   Population α₀    = {hier_result.population_mean[1] * M_TO_MM:.3f} "
              f"± {hier_result.population_sd[1] * M_TO_MM:.3f} mm/yr/°C")
        print(f"   Population trend  = {hier_result.population_mean[2] * M_TO_MM:.3f} "
              f"± {hier_result.population_sd[2] * M_TO_MM:.3f} mm/yr")
        print(f"   Acceptance = {hier_result.sampler_diagnostics['acceptance_fraction']:.3f}")

        print("\n   Dataset coefficients (mm/yr):")
        for dname, coeffs in hier_result.dataset_coefficients.items():
            shrink = hier_result.shrinkage_factors[dname]
            print(f"     {dname:25s}: dα/dT={coeffs[0]*M_TO_MM:7.3f}  "
                  f"α₀={coeffs[1]*M_TO_MM:7.3f}  trend={coeffs[2]*M_TO_MM:7.3f}  "
                  f"shrink={shrink:.3f}")

        # Figures
        print("\n   Generating hierarchical figures...")
        plot_hierarchical_forest(hier_result)
        plot_hierarchical_shrinkage(hier_result)
    else:
        print("\n5. Hierarchical model skipped.")

    # ---- 6. Summary ----
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    print("\nCoefficient comparison (mm/yr units):")
    print(f"  {'Method':30s}  {'dα/dT':>10s}  {'α₀':>10s}  {'trend':>10s}")
    print(f"  {'-'*30}  {'-'*10}  {'-'*10}  {'-'*10}")
    print(f"  {'Frequentist DOLS':30s}  "
          f"{freq.dalpha_dT*M_TO_MM:10.3f}  "
          f"{freq.alpha0*M_TO_MM:10.3f}  "
          f"{freq.trend*M_TO_MM:10.3f}")
    print(f"  {'Bayesian Static':30s}  "
          f"{bayes_static.physical_coefficients[0]*M_TO_MM:10.3f}  "
          f"{bayes_static.physical_coefficients[1]*M_TO_MM:10.3f}  "
          f"{bayes_static.physical_coefficients[2]*M_TO_MM:10.3f}")
    if dlm_bayes is not None:
        final = dlm_bayes.coefficients_mean[-1]
        print(f"  {'DLM (final, Bayesian Q)':30s}  "
              f"{final[0]*M_TO_MM:10.3f}  "
              f"{final[1]*M_TO_MM:10.3f}  "
              f"{final[2]*M_TO_MM:10.3f}")
    if hier_result is not None:
        print(f"  {'Hierarchical (population)':30s}  "
              f"{hier_result.population_mean[0]*M_TO_MM:10.3f}  "
              f"{hier_result.population_mean[1]*M_TO_MM:10.3f}  "
              f"{hier_result.population_mean[2]*M_TO_MM:10.3f}")

    print(f"\nFigures saved to: {FIG_DIR}/")
    print("Done.")


if __name__ == "__main__":
    main()
