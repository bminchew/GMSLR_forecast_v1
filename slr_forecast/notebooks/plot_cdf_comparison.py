#!/usr/bin/env python
"""
Validate IPCC PDFs in fig2v2_ridge_exceedance.png panel (a).

The IPCC ridge PDFs in panel (a) are KDEs of samples drawn from
skew-normal fits to the 5 IPCC AR6 quantiles (5, 17, 50, 83, 95%).
This script checks whether those fitted distributions faithfully
reproduce the original IPCC quantile-defined CDF.

  Solid lines:  CDFs of the skew-normal fits used in panel (a)
  Markers:      All 107 raw IPCC AR6 medium-confidence quantiles

If the fits are accurate, the solid curves will pass through the markers.

Usage:
    python plot_cdf_comparison.py
"""

import sys
from pathlib import Path

import h5py
import numpy as np
import matplotlib.pyplot as plt
import xarray as xr
from scipy.stats import skewnorm

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from slr_forecast.config import PROCESSED_DATA_DIR

H5_IPCC = PROCESSED_DATA_DIR / "ipcc_distributions.h5"
FIG_DIR = PROJECT_ROOT / "figures"

RAW_DIR = PROJECT_ROOT / "data" / "raw"
CONF_DIR = (RAW_DIR / "ipcc_ar6" / "slr" / "ar6" / "global"
            / "confidence_output_files" / "medium_confidence")

M_TO_MM = 1000.0

# SSPs shown in fig2v2 panel (a)
SSPS = ["SSP1-2.6", "SSP2-4.5", "SSP3-7.0"]
SSP_TO_CODE = {"SSP1-2.6": "ssp126", "SSP2-4.5": "ssp245", "SSP3-7.0": "ssp370"}
SSP_COLORS = {
    "SSP1-2.6": "#1b9e77",
    "SSP2-4.5": "#d95f02",
    "SSP3-7.0": "#e7298a",
}

# Years matching fig2v2 ridge decades (2025 has no IPCC data)
TARGET_YEARS = [2050, 2100]

# Rebase offset: IPCC is relative to 1995-2014 mean (~2005);
# skew-normal fits in ipcc_distributions.h5 were rebased to 2000
# using _GMSL_2000_TO_2005_MM = 10.7 mm (see precompute_ipcc_distributions.py).
_IPCC_REBASE_MM = 10.7


def load_raw_ipcc_quantiles(ssp_code):
    """Load all 107 quantiles from the raw IPCC NetCDF (mm)."""
    nc_path = CONF_DIR / ssp_code / f"total_{ssp_code}_medium_confidence_values.nc"
    ds = xr.open_dataset(nc_path)
    years = ds["years"].values
    quantiles = ds["quantiles"].values          # (107,) from 0 to 1
    slc_mm = ds["sea_level_change"].values[:, :, 0]  # (107, n_years) in mm
    ds.close()
    return years, quantiles, slc_mm


def main():
    # ── Load skew-normal fit parameters from ipcc_distributions.h5 ──
    with h5py.File(H5_IPCC, "r") as hf:
        fit_data = {}
        for ssp in SSPS:
            grp = hf[f"params/total/{ssp}"]
            fit_data[ssp] = {
                "years": grp["years"][:],
                "alpha": grp["alpha"][:],
                "loc": grp["loc"][:],
                "scale": grp["scale"][:],
            }

    # ── Load full IPCC quantiles from raw NetCDFs ──
    raw_data = {}
    for ssp in SSPS:
        years, quantiles, slc_mm = load_raw_ipcc_quantiles(SSP_TO_CODE[ssp])
        # Rebase to 2000 baseline (same offset used in precompute script)
        raw_data[ssp] = {
            "years": years,
            "quantiles": quantiles,
            "slc_mm": slc_mm + _IPCC_REBASE_MM,
        }

    # ── Build figure ──
    fig, axes = plt.subplots(1, len(TARGET_YEARS),
                             figsize=(6 * len(TARGET_YEARS), 5),
                             sharey=True)
    if len(TARGET_YEARS) == 1:
        axes = [axes]

    for ax, yr in zip(axes, TARGET_YEARS):
        for ssp in SSPS:
            color = SSP_COLORS[ssp]
            fd = fit_data[ssp]
            rd = raw_data[ssp]

            # Year index in fitted data
            yr_idx_fit = np.argmin(np.abs(fd["years"] - yr))
            alpha = fd["alpha"][yr_idx_fit]
            loc = fd["loc"][yr_idx_fit]
            scale = fd["scale"][yr_idx_fit]

            # Skew-normal CDF (what panel (a) PDFs integrate to)
            x_lo = skewnorm.ppf(0.001, alpha, loc=loc, scale=scale)
            x_hi = skewnorm.ppf(0.999, alpha, loc=loc, scale=scale)
            x_grid = np.linspace(x_lo, x_hi, 500)
            cdf_fit = skewnorm.cdf(x_grid, alpha, loc=loc, scale=scale)

            ax.plot(x_grid, cdf_fit, color=color, lw=2, ls="-",
                    label=f"{ssp} (skew-normal fit)")

            # All 107 raw IPCC quantiles (ground truth)
            yr_idx_raw = np.argmin(np.abs(rd["years"] - yr))
            q_vals = rd["slc_mm"][:, yr_idx_raw]   # (107,) mm
            q_probs = rd["quantiles"]               # (107,) probabilities

            # Plot all quantiles as small dots
            ax.plot(q_vals, q_probs, ".", color=color, ms=4, alpha=0.6,
                    zorder=5)
            # Highlight the 5 fitted quantiles with open circles
            for p in [0.05, 0.17, 0.50, 0.83, 0.95]:
                idx = np.argmin(np.abs(q_probs - p))
                ax.plot(q_vals[idx], q_probs[idx], "o", color=color,
                        ms=8, mew=2, mfc="none", zorder=6)

        ax.set_xlabel("GMSL rise (mm, relative to 2000)")
        ax.set_title(f"{yr}")
        ax.axhline(0.5, color="gray", lw=0.5, ls=":", alpha=0.5)
        ax.grid(True, alpha=0.2)

    axes[0].set_ylabel("Cumulative probability")

    # Build legend manually (one entry per SSP + explanation)
    from matplotlib.lines import Line2D
    legend_handles = []
    for ssp in SSPS:
        legend_handles.append(
            Line2D([0], [0], color=SSP_COLORS[ssp], lw=2, ls="-",
                   label=f"{ssp} fit"))
        legend_handles.append(
            Line2D([0], [0], color=SSP_COLORS[ssp], marker=".", ls="none",
                   ms=5, label=f"{ssp} IPCC quantiles"))
    legend_handles.append(
        Line2D([0], [0], color="gray", marker="o", ls="none", ms=8,
               mew=2, mfc="none", label="5 fitted quantiles"))

    fig.legend(handles=legend_handles, loc="lower center", ncol=4,
               fontsize=8, bbox_to_anchor=(0.5, -0.10))
    fig.suptitle(
        "Validation: Skew-Normal Fits vs All IPCC AR6 Quantiles\n"
        "(medium confidence, total GMSL)",
        fontsize=12, y=1.04,
    )
    plt.tight_layout()

    out = FIG_DIR / "cdf_validation_ipcc.png"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    print(f"Saved → {out}")
    plt.close(fig)

    # ── Print numerical diagnostics at the 5 fitted quantiles ──
    ipcc_probs = np.array([0.05, 0.17, 0.50, 0.83, 0.95])
    print("\nQuantile residuals at fitted points (fit - IPCC, mm):")
    for ssp in SSPS:
        fd = fit_data[ssp]
        rd = raw_data[ssp]
        for yr in TARGET_YEARS:
            yr_idx_fit = np.argmin(np.abs(fd["years"] - yr))
            alpha = fd["alpha"][yr_idx_fit]
            loc = fd["loc"][yr_idx_fit]
            scale = fd["scale"][yr_idx_fit]
            fit_q = skewnorm.ppf(ipcc_probs, alpha, loc=loc, scale=scale)

            yr_idx_raw = np.argmin(np.abs(rd["years"] - yr))
            raw_q = np.array([
                rd["slc_mm"][np.argmin(np.abs(rd["quantiles"] - p)),
                             yr_idx_raw]
                for p in ipcc_probs
            ])
            resid = fit_q - raw_q
            print(f"  {ssp} @ {yr}: "
                  f"[{resid[0]:+.2f}, {resid[1]:+.2f}, {resid[2]:+.2f}, "
                  f"{resid[3]:+.2f}, {resid[4]:+.2f}] mm")


if __name__ == "__main__":
    main()
