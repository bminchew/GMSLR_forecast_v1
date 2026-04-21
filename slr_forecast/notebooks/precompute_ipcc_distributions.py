"""
Precompute IPCC AR6 skew-normal distribution fits and MC samples.

Fits a skew-normal distribution to the 5 key IPCC quantiles
(5, 17, 50, 83, 95) at each (component, SSP, year) combination.
This captures the genuine asymmetry (heavier upper tail) as a smooth,
unimodal distribution without p-box median-seam artifacts.

Outputs
-------
data/processed/ipcc_distributions.h5
    ├── params/{component}/{ssp}
    │     years, alpha, loc, scale    (skew-normal parameters)
    ├── samples/{component}/{ssp}
    │     samples  (N_SAMPLES, n_years) in mm
    └── attrs: n_samples, seed, baseline_year

Usage
-----
    python precompute_ipcc_distributions.py [--n-samples 2000] [--seed 777]
"""

import argparse
import os
import sys
from pathlib import Path

import h5py
import numpy as np
from scipy.optimize import minimize
from scipy.stats import skewnorm

sys.path.insert(0, os.path.dirname(__file__))
from component_projections import read_ipcc_component_nc, ipcc_extract

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
RAW_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"
PROCESSED_DIR = Path(__file__).resolve().parent.parent / "data" / "processed"
CONF_BASE = RAW_DIR / "ipcc_ar6" / "slr" / "ar6" / "global" / "confidence_output_files"
OUTPUT_PATH = PROCESSED_DIR / "ipcc_distributions.h5"

BASELINE_YEAR = 2005.0
M_TO_MM = 1000.0

SSP_TO_CODE = {
    "SSP1-2.6": "ssp126",
    "SSP2-4.5": "ssp245",
    "SSP3-7.0": "ssp370",
    "SSP5-8.5": "ssp585",
}

# Components to precompute (IPCC NetCDF names)
IPCC_COMPONENTS = [
    "total",
    "oceandynamics",
    "glaciers",
    "GIS",
    "AIS",
    "landwaterstorage",
]


# ---------------------------------------------------------------------------
# Skew-normal fitting
# ---------------------------------------------------------------------------

def fit_skewnorm_to_quantiles(q05, q17, q50, q83, q95):
    """Fit skew-normal (alpha, loc, scale) to 5 IPCC quantiles.

    Uses multiple starting points for alpha to avoid the symmetric
    local minimum.  Returns parameters for scipy.stats.skewnorm.
    """
    targets = np.array([q05, q17, q50, q83, q95])
    probs = np.array([0.05, 0.17, 0.50, 0.83, 0.95])

    # Normalize for conditioning
    t_scale = max(targets.max() - targets.min(), 1.0)
    t_center = targets.mean()
    targets_n = (targets - t_center) / t_scale

    def cost(params):
        loc_n, log_s, a = params
        return np.sum(
            (skewnorm.ppf(probs, a, loc=loc_n, scale=np.exp(log_s))
             - targets_n) ** 2
        )

    asym_sign = 1.0 if (q95 - q50) > (q50 - q05) else -1.0
    s0 = (q83 - q17) / (1.35 * t_scale)
    loc0 = (q50 - t_center) / t_scale

    best_res, best_cost = None, np.inf
    for a0 in [0.0, 2 * asym_sign, 5 * asym_sign, 10 * asym_sign]:
        res = minimize(
            cost,
            [loc0, np.log(max(s0, 0.01)), a0],
            method="Nelder-Mead",
            options={"xatol": 1e-8, "fatol": 1e-10, "maxiter": 10000},
        )
        if res.fun < best_cost:
            best_cost, best_res = res.fun, res

    loc_n, log_s, alpha = best_res.x
    return alpha, loc_n * t_scale + t_center, np.exp(log_s) * t_scale


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(n_samples=2000, seed=777, n_ridge=10000, ridge_seed=888):
    rng = np.random.default_rng(seed)
    rng_ridge = np.random.default_rng(ridge_seed)

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    with h5py.File(OUTPUT_PATH, "w") as f:
        f.attrs["n_samples"] = n_samples
        f.attrs["n_ridge"] = n_ridge
        f.attrs["seed"] = seed
        f.attrs["ridge_seed"] = ridge_seed
        f.attrs["baseline_year"] = BASELINE_YEAR

        for comp in IPCC_COMPONENTS:
            print(f"\n{comp}:")

            for ssp, code in SSP_TO_CODE.items():
                data = read_ipcc_component_nc(
                    str(CONF_BASE), "medium_confidence", code, comp
                )
                if data is None:
                    print(f"  {ssp}: not found — skipping")
                    continue

                ie = ipcc_extract(
                    data, quantiles_target=(0.05, 0.17, 0.5, 0.83, 0.95)
                )
                years = ie["years"]
                n_years = len(years)

                # Fit skew-normal at each year
                alphas = np.zeros(n_years)
                locs = np.zeros(n_years)
                scales = np.zeros(n_years)

                for j in range(n_years):
                    a, loc, scale = fit_skewnorm_to_quantiles(
                        ie["q05"][j], ie["q17"][j], ie["q50"][j],
                        ie["q83"][j], ie["q95"][j],
                    )
                    alphas[j] = a
                    locs[j] = loc
                    scales[j] = scale

                # Draw MC samples (mm) — standard count for component use
                samples_mm = np.zeros((n_samples, n_years))
                for j in range(n_years):
                    samples_mm[:, j] = skewnorm.rvs(
                        alphas[j], loc=locs[j], scale=scales[j],
                        size=n_samples, random_state=rng,
                    )

                # Draw oversampled set (mm) — for ridge plots
                ridge_mm = np.zeros((n_ridge, n_years))
                for j in range(n_years):
                    ridge_mm[:, j] = skewnorm.rvs(
                        alphas[j], loc=locs[j], scale=scales[j],
                        size=n_ridge, random_state=rng_ridge,
                    )

                # Save parameters
                grp_p = f.create_group(f"params/{comp}/{ssp}")
                grp_p.create_dataset("years", data=years)
                grp_p.create_dataset("alpha", data=alphas)
                grp_p.create_dataset("loc", data=locs)
                grp_p.create_dataset("scale", data=scales)
                # Store raw quantiles for reference
                grp_p.create_dataset("q05", data=ie["q05"])
                grp_p.create_dataset("q17", data=ie["q17"])
                grp_p.create_dataset("q50", data=ie["q50"])
                grp_p.create_dataset("q83", data=ie["q83"])
                grp_p.create_dataset("q95", data=ie["q95"])

                # Save standard samples
                grp_s = f.create_group(f"samples/{comp}/{ssp}")
                grp_s.create_dataset(
                    "samples", data=samples_mm,
                    chunks=(min(n_samples, 500), n_years),
                    compression="gzip", compression_opts=4,
                )
                grp_s.create_dataset("years", data=years)

                # Save oversampled (ridge) samples
                grp_r = f.create_group(f"ridge/{comp}/{ssp}")
                grp_r.create_dataset(
                    "samples", data=ridge_mm,
                    chunks=(min(n_ridge, 1000), n_years),
                    compression="gzip", compression_opts=4,
                )
                grp_r.create_dataset("years", data=years)

                med_2100 = ie["q50"][np.argmin(np.abs(years - 2100))]
                print(f"  {ssp}: {n_years} years, q50@2100={med_2100:.0f} mm")

    size_mb = os.path.getsize(OUTPUT_PATH) / 1024**2
    print(f"\nSaved: {OUTPUT_PATH}  ({size_mb:.1f} MB)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Precompute IPCC AR6 skew-normal fits and samples."
    )
    parser.add_argument("--n-samples", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=777)
    parser.add_argument("--n-ridge", type=int, default=10000)
    parser.add_argument("--ridge-seed", type=int, default=888)
    args = parser.parse_args()
    main(
        n_samples=args.n_samples,
        seed=args.seed,
        n_ridge=args.n_ridge,
        ridge_seed=args.ridge_seed,
    )
