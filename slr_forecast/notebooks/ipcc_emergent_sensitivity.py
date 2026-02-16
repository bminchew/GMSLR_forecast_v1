#!/usr/bin/env python3
"""
IPCC Emergent Quadratic Sensitivity Test
=========================================

Priority 2 from TODO.md: Run DOLS on IPCC projections to determine whether
the quadratic temperature sensitivity (dα/dT ≈ 3–6 mm/yr/°C²) found in
observations is also an emergent property of IPCC AR6/CMIP6 process-model
output.

Approach:
1. np.polyfit on rate-vs-T curves (linear & quadratic) with AIC/BIC/F-test
2. DOLS (calibrate_dols, n_lags=0) for consistency check
3. Decomposed component test (thermodynamic vs ice-sheet)
4. Cross-SSP coefficient comparison table
5. Rate-vs-temperature phase diagram

Authors: Minchew research group, 2026
"""

import sys
import os
import warnings
from typing import Optional

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats

warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# Ensure local imports
sys.path.insert(0, os.path.dirname(__file__))
import slr_analysis
from slr_analysis import calibrate_dols, DOLSResult, test_rate_temperature_nonlinearity

# Paths
DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
H5_PATH = os.path.join(DATA_DIR, "processed", "slr_processed_data.h5")
FIG_DIR = os.path.join(os.path.dirname(__file__), "figures")
os.makedirs(FIG_DIR, exist_ok=True)

# SSP mapping between temperature keys and SLR keys
SSP_MAP = {
    "ssp119": "SSP1_1_9",
    "ssp126": "SSP1_2_6",
    "ssp245": "SSP2_4_5",
    "ssp370": "SSP3_7_0",
    "ssp585": "SSP5_8_5",
}

# Display names
SSP_LABELS = {
    "ssp119": "SSP1-1.9",
    "ssp126": "SSP1-2.6",
    "ssp245": "SSP2-4.5",
    "ssp370": "SSP3-7.0",
    "ssp585": "SSP5-8.5",
}

SSP_COLORS = {
    "ssp119": "#1b9e77",
    "ssp126": "#2166ac",
    "ssp245": "#f57f17",
    "ssp370": "#d62728",
    "ssp585": "#7b2d8e",
}


# =====================================================================
#  1. LOAD IPCC DATA FROM H5 STORE
# =====================================================================

def load_ipcc_data():
    """Load IPCC projected GMSL and GMST from the H5 store.

    Returns
    -------
    ipcc_gmsl : dict of pd.DataFrame keyed by ssp code (e.g. 'ssp245')
    ipcc_temp : dict of pd.DataFrame keyed by SSP code (e.g. 'SSP2_4_5')
    hist_temp : pd.DataFrame for Historical period
    """
    store = pd.HDFStore(H5_PATH, mode="r")

    ipcc_gmsl = {}
    for ssp in SSP_MAP:
        key = f"/projections/gmsl/{ssp}"
        if key in store:
            ipcc_gmsl[ssp] = store[key]

    ipcc_temp = {}
    for ssp_slr, ssp_temp in SSP_MAP.items():
        key = f"/projections/temp/{ssp_temp}"
        if key in store:
            ipcc_temp[ssp_slr] = store[key]

    key_hist = "/projections/temp/Historical"
    hist_temp = store[key_hist] if key_hist in store else None

    store.close()
    return ipcc_gmsl, ipcc_temp, hist_temp


# =====================================================================
#  2. PREPARE MATCHED (T, H) PAIRS FOR EACH SSP
# =====================================================================

def prepare_ssp_data(ipcc_gmsl, ipcc_temp, hist_temp, max_year=2100):
    """Build matched annual temperature and decadal GMSL for each SSP.

    For DOLS and polyfit we need (T, H) on the same time grid.  Since SLR
    projections are decadal and temperature is annual, we:
    - Concatenate historical + SSP temperature
    - Interpolate temperature to the GMSL decadal years
    - Both are relative to the same ~2005 baseline (GMSL explicitly;
      temperature relative to 1850-1900 pre-industrial)

    Parameters
    ----------
    max_year : float, default 2100
        Truncate GMSL projections to years ≤ max_year.  Post-2100
        projections use extrapolated temperature pathways and become
        unreliable for sensitivity estimation.

    Returns
    -------
    ssp_data : dict
        ssp_code -> dict with keys:
            'years' : np.ndarray of decadal years
            'gmsl'  : np.ndarray (metres, total GMSL)
            'thermodynamic' : np.ndarray (oceandynamics + glaciers + GIS, metres)
            'temperature' : np.ndarray (°C, relative to 1850-1900)
            'gmsl_df' : pd.DataFrame with component columns
            'ais' : np.ndarray (AIS only, metres)

    Notes
    -----
    The thermodynamic component (oceandynamics + glaciers + GIS) is the
    IPCC analogue of the observational signal used to calibrate DOLS,
    which removes TWS from total GMSL to isolate the temperature-driven
    response.  Greenland is included because its mass loss is primarily a
    thermodynamic response to regional warming.  Antarctica (AIS) is
    excluded because it is dominated by marine ice-sheet dynamics and is
    not a significant contributor within the observational record used
    for calibration.
    """
    ssp_data = {}
    for ssp_code in ipcc_gmsl:
        df_sl = ipcc_gmsl[ssp_code]
        # Truncate to max_year
        df_sl = df_sl[df_sl["decimal_year"] <= max_year]
        years_sl = df_sl["decimal_year"].values

        if ssp_code not in ipcc_temp:
            continue
        df_temp = ipcc_temp[ssp_code]

        # Concatenate historical + SSP temperature
        temp_years = df_temp["decimal_year"].values
        temp_vals = df_temp["temperature"].values

        if hist_temp is not None:
            h_years = hist_temp["decimal_year"].values
            h_vals = hist_temp["temperature"].values
            # Avoid overlap: keep historical up to just before SSP start
            mask = h_years < temp_years[0]
            all_years = np.concatenate([h_years[mask], temp_years])
            all_temp = np.concatenate([h_vals[mask], temp_vals])
        else:
            all_years = temp_years
            all_temp = temp_vals

        # Interpolate temperature to GMSL decadal years
        temp_at_sl = np.interp(years_sl, all_years, all_temp)

        # Component decomposition
        # Thermodynamic = oceandynamics + glaciers + GIS
        #   (matches observational DOLS calibration: GMSL minus TWS,
        #    which retains steric + glaciers + GIS + AIS;
        #    we further exclude AIS as it is not a major thermodynamic
        #    contributor within the calibration record)
        thermo = (df_sl["oceandynamics"].values
                  + df_sl["glaciers"].values
                  + df_sl["GIS"].values)
        ais = df_sl["AIS"].values

        ssp_data[ssp_code] = {
            "years": years_sl,
            "gmsl": df_sl["gmsl"].values,
            "thermodynamic": thermo,
            "temperature": temp_at_sl,
            "gmsl_df": df_sl,
            "ais": ais,
        }

    return ssp_data


# =====================================================================
#  3. POLYFIT MODEL SELECTION (Linear vs Quadratic) — the PRIMARY method
# =====================================================================

def polyfit_model_selection(years, gmsl, temperature, label=""):
    """Fit np.polyfit linear & quadratic to IPCC rate-vs-T and compare
    using AIC, BIC, F-test, and p-value on the quadratic coefficient.

    Rates are computed by finite differences on the (smooth, decadal)
    IPCC projections.

    Parameters
    ----------
    years : array-like  (decimal years, same length as gmsl)
    gmsl  : array-like  (metres)
    temperature : array-like (°C)
    label : str  (for display)

    Returns
    -------
    dict with keys:
        'linear', 'quadratic' : sub-dicts with polyfit coeffs, predictions, etc.
        'rate', 'T_mid'       : the finite-difference rate and midpoint temperature
        'aic', 'bic'          : comparison dicts
        'f_test'              : F-test result
        'p_value_quad_coeff'  : t-test p-value on the quadratic coefficient
        'recommendation'      : str
    """
    dt = np.diff(years)
    rate = np.diff(gmsl) / dt  # m/yr
    T_mid = 0.5 * (temperature[1:] + temperature[:-1])
    n = len(rate)

    # ---- Linear fit: rate = b*T + c ----
    pf_lin = np.polyfit(T_mid, rate, 1)
    pred_lin = np.polyval(pf_lin, T_mid)
    resid_lin = rate - pred_lin
    ssr_lin = np.sum(resid_lin**2)
    k_lin = 2  # 2 parameters

    # ---- Quadratic fit: rate = a*T^2 + b*T + c ----
    pf_quad = np.polyfit(T_mid, rate, 2)
    pred_quad = np.polyval(pf_quad, T_mid)
    resid_quad = rate - pred_quad
    ssr_quad = np.sum(resid_quad**2)
    k_quad = 3  # 3 parameters

    # ---- R² ----
    sst = np.sum((rate - rate.mean()) ** 2)
    r2_lin = 1 - ssr_lin / sst if sst > 0 else np.nan
    r2_quad = 1 - ssr_quad / sst if sst > 0 else np.nan

    # ---- AIC / BIC ----
    # AIC = n * ln(SSR/n) + 2*k
    # BIC = n * ln(SSR/n) + k*ln(n)
    aic_lin = n * np.log(ssr_lin / n) + 2 * k_lin
    bic_lin = n * np.log(ssr_lin / n) + k_lin * np.log(n)
    aic_quad = n * np.log(ssr_quad / n) + 2 * k_quad
    bic_quad = n * np.log(ssr_quad / n) + k_quad * np.log(n)

    delta_aic = aic_lin - aic_quad  # positive => quadratic preferred
    delta_bic = bic_lin - bic_quad

    # ---- F-test: linear (reduced) vs quadratic (full) ----
    df_r = n - k_lin
    df_f = n - k_quad
    f_stat = ((ssr_lin - ssr_quad) / (df_r - df_f)) / (ssr_quad / df_f) \
        if ssr_quad > 0 and df_f > 0 else np.nan
    f_p = 1 - stats.f.cdf(f_stat, df_r - df_f, df_f) \
        if np.isfinite(f_stat) else np.nan

    # ---- t-test / p-value on quadratic coefficient a ----
    # Use proper OLS standard errors for the quadratic fit
    X_quad = np.column_stack([T_mid**2, T_mid, np.ones(n)])
    XtX_inv = np.linalg.inv(X_quad.T @ X_quad)
    mse_quad = ssr_quad / (n - k_quad) if n > k_quad else np.nan
    se_quad = np.sqrt(mse_quad * np.diag(XtX_inv)) if np.isfinite(mse_quad) else np.full(3, np.nan)
    t_stat_a = pf_quad[0] / se_quad[0] if se_quad[0] > 0 else np.nan
    p_value_a = 2 * (1 - stats.t.cdf(abs(t_stat_a), n - k_quad)) \
        if np.isfinite(t_stat_a) else np.nan

    # ---- Recommendation ----
    scores = {"linear": 0, "quadratic": 0}
    reasons = []

    if np.isfinite(f_p) and f_p < 0.05:
        scores["quadratic"] += 1
        reasons.append(f"F-test significant (p={f_p:.4g})")
    else:
        scores["linear"] += 1
        reasons.append(f"F-test not significant (p={f_p:.4g})")

    if np.isfinite(p_value_a) and p_value_a < 0.05:
        scores["quadratic"] += 1
        reasons.append(f"quadratic coeff t-test significant (p={p_value_a:.4g})")
    else:
        scores["linear"] += 1
        reasons.append(f"quadratic coeff t-test not significant (p={p_value_a:.4g})")

    if delta_aic > 0:
        scores["quadratic"] += 1
        reasons.append(f"AIC favors quadratic (ΔAIC={delta_aic:+.2f})")
    else:
        scores["linear"] += 1
        reasons.append(f"AIC favors linear (ΔAIC={delta_aic:+.2f})")

    if delta_bic > 0:
        scores["quadratic"] += 1
        reasons.append(f"BIC favors quadratic (ΔBIC={delta_bic:+.2f})")
    else:
        scores["linear"] += 1
        reasons.append(f"BIC favors linear (ΔBIC={delta_bic:+.2f})")

    best = max(scores, key=scores.get)
    rec = f"{best.upper()} preferred ({scores[best]}/4 criteria). " + "; ".join(reasons)

    # ---- Convert polyfit coefficients to rate-model naming ----
    # Linear: pf_lin = [b, c] where rate = b*T + c
    # Quadratic: pf_quad = [a, b, c] where rate = a*T^2 + b*T + c
    # Express dalpha_dT in mm/yr/°C² for comparison with observational DOLS
    dalpha_dT_polyfit = pf_quad[0] * 1000  # m -> mm

    return {
        "label": label,
        "linear": {
            "coeffs": pf_lin,  # [b, c] in m/yr/°C, m/yr
            "alpha0": pf_lin[0],
            "trend": pf_lin[1],
            "r2": r2_lin,
            "aic": aic_lin,
            "bic": bic_lin,
            "ssr": ssr_lin,
            "predictions": pred_lin,
            "residuals": resid_lin,
            "se": np.sqrt(ssr_lin / (n - k_lin) * np.diag(
                np.linalg.inv(np.column_stack([T_mid, np.ones(n)]).T @
                              np.column_stack([T_mid, np.ones(n)]))
            )) if n > k_lin else np.full(2, np.nan),
        },
        "quadratic": {
            "coeffs": pf_quad,  # [a, b, c] in m/yr/°C², m/yr/°C, m/yr
            "dalpha_dT": pf_quad[0],
            "alpha0": pf_quad[1],
            "trend": pf_quad[2],
            "dalpha_dT_mm": dalpha_dT_polyfit,
            "r2": r2_quad,
            "aic": aic_quad,
            "bic": bic_quad,
            "ssr": ssr_quad,
            "predictions": pred_quad,
            "residuals": resid_quad,
            "se": se_quad,
            "t_stat_a": t_stat_a,
            "p_value_a": p_value_a,
        },
        "rate": rate,
        "T_mid": T_mid,
        "n": n,
        "aic_comparison": {
            "delta_aic": delta_aic,
            "best": "quadratic" if delta_aic > 0 else "linear",
        },
        "bic_comparison": {
            "delta_bic": delta_bic,
            "best": "quadratic" if delta_bic > 0 else "linear",
        },
        "f_test": {
            "f_stat": f_stat,
            "p_value": f_p,
            "df1": df_r - df_f,
            "df2": df_f,
            "significant": f_p < 0.05 if np.isfinite(f_p) else False,
        },
        "p_value_quad_coeff": p_value_a,
        "recommendation": rec,
    }


# =====================================================================
#  4. DOLS ON IPCC PROJECTIONS (consistency check)
# =====================================================================

def run_dols_on_ipcc(ssp_data):
    """Run calibrate_dols(order=1 and order=2, n_lags=0) on each SSP.

    Uses the thermodynamic signal (oceandynamics + glaciers + GIS) to
    match the observational DOLS calibration target.

    On smooth deterministic trajectories, DOLS with n_lags=0 is equivalent
    to integral-space polynomial regression and should agree closely with
    the polyfit results.

    Returns
    -------
    dols_results : dict
        ssp_code -> dict with 'order1' and 'order2' DOLSResult objects
    """
    dols_results = {}
    for ssp_code, data in ssp_data.items():
        years = data["years"]
        thermo = data["thermodynamic"]
        temp = data["temperature"]

        # Create pd.Series with datetime index (needed by calibrate_dols)
        dates = pd.to_datetime(
            [f"{int(y)}-07-01" for y in years]
        )
        sl_series = pd.Series(thermo, index=dates, name="thermodynamic")
        temp_series = pd.Series(temp, index=dates, name="temperature")

        results = {}
        for order in [1, 2]:
            try:
                r = calibrate_dols(
                    sl_series, temp_series,
                    order=order, n_lags=0,
                )
                results[f"order{order}"] = r
            except Exception as e:
                print(f"  DOLS order={order} failed for {ssp_code}: {e}")
                results[f"order{order}"] = None

        dols_results[ssp_code] = results

    return dols_results


# =====================================================================
#  5. DECOMPOSED COMPONENT ANALYSIS
# =====================================================================

def run_component_analysis(ssp_data):
    """Run polyfit model selection separately on thermodynamic and
    AIS components.

    Thermodynamic = oceandynamics + glaciers + GIS  (continuous T-response;
        matches the observational DOLS calibration target)
    AIS = Antarctica ice sheet only  (marine ice-sheet dynamics, excluded
        from DOLS calibration)

    Returns
    -------
    component_results : dict
        ssp_code -> {'thermodynamic': polyfit_result, 'ais': polyfit_result}
    """
    component_results = {}
    for ssp_code, data in ssp_data.items():
        years = data["years"]
        temp = data["temperature"]

        comp_res = {}
        for comp_name, comp_data in [
            ("thermodynamic", data["thermodynamic"]),
            ("ais", data["ais"]),
        ]:
            comp_res[comp_name] = polyfit_model_selection(
                years, comp_data, temp,
                label=f"{SSP_LABELS[ssp_code]} {comp_name}",
            )
        component_results[ssp_code] = comp_res

    return component_results


# =====================================================================
#  6. SUMMARY TABLE
# =====================================================================

def build_summary_table(polyfit_results, dols_results, component_results):
    """Build a summary table comparing coefficients across SSPs and methods."""
    rows = []

    for ssp_code in sorted(polyfit_results.keys()):
        pf = polyfit_results[ssp_code]
        dols = dols_results.get(ssp_code, {})

        # Polyfit quadratic
        row = {
            "SSP": SSP_LABELS[ssp_code],
            # Polyfit results (rate-space, direct)
            "polyfit_dalpha_dT (mm/yr/°C²)": pf["quadratic"]["dalpha_dT_mm"],
            "polyfit_alpha0 (mm/yr/°C)": pf["quadratic"]["alpha0"] * 1000,
            "polyfit_trend (mm/yr)": pf["quadratic"]["trend"] * 1000,
            "polyfit_R²_quad": pf["quadratic"]["r2"],
            "polyfit_R²_lin": pf["linear"]["r2"],
            "F-test p-value": pf["f_test"]["p_value"],
            "quad_coeff p-value": pf["p_value_quad_coeff"],
            "ΔAIC (lin−quad)": pf["aic_comparison"]["delta_aic"],
            "ΔBIC (lin−quad)": pf["bic_comparison"]["delta_bic"],
            "Recommendation": pf["recommendation"].split(".")[0] + ".",
        }

        # DOLS results
        r2 = dols.get("order2")
        if r2 is not None:
            row["DOLS_dalpha_dT (mm/yr/°C²)"] = r2.dalpha_dT * 1000
            row["DOLS_alpha0 (mm/yr/°C)"] = r2.alpha0 * 1000
            row["DOLS_trend (mm/yr)"] = r2.trend * 1000
            row["DOLS_R²"] = r2.r2
        else:
            row["DOLS_dalpha_dT (mm/yr/°C²)"] = np.nan
            row["DOLS_alpha0 (mm/yr/°C)"] = np.nan
            row["DOLS_trend (mm/yr)"] = np.nan
            row["DOLS_R²"] = np.nan

        # Component results
        cr = component_results.get(ssp_code, {})
        th = cr.get("thermodynamic")
        if th is not None:
            row["thermo_dalpha_dT (mm/yr/°C²)"] = th["quadratic"]["dalpha_dT_mm"]
            row["thermo_recommendation"] = th["recommendation"].split(".")[0] + "."
        ais = cr.get("ais")
        if ais is not None:
            row["ais_dalpha_dT (mm/yr/°C²)"] = ais["quadratic"]["dalpha_dT_mm"]
            row["ais_recommendation"] = ais["recommendation"].split(".")[0] + "."

        rows.append(row)

    return pd.DataFrame(rows)


# =====================================================================
#  7. RATE-VS-TEMPERATURE PHASE DIAGRAM (Fig S4)
# =====================================================================

def plot_rate_vs_temperature_phase(polyfit_results, ssp_data, fig_path=None):
    """Plot IPCC-projected rate vs T for each SSP alongside polyfit curves.

    This is potentially Fig S4 in the manuscript.  Fit curves are clipped
    to ±10% beyond each SSP's data range to avoid misleading extrapolation.
    """
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))

    for ssp_code in sorted(polyfit_results.keys()):
        pf = polyfit_results[ssp_code]
        color = SSP_COLORS[ssp_code]
        label = SSP_LABELS[ssp_code]

        # Scatter the IPCC finite-difference rates
        ax.scatter(
            pf["T_mid"], pf["rate"] * 1000,
            color=color, s=40, zorder=3, alpha=0.8,
            label=f"{label}",
        )

        # Fit curve clipped to data range ± 10%
        T_lo, T_hi = pf["T_mid"].min(), pf["T_mid"].max()
        margin = 0.10 * (T_hi - T_lo) if T_hi > T_lo else 0.1
        T_curve = np.linspace(T_lo - margin, T_hi + margin, 100)
        pred = np.polyval(pf["quadratic"]["coeffs"], T_curve) * 1000
        ax.plot(T_curve, pred, color=color, ls="-", lw=1.5, alpha=0.6)

    ax.set_xlabel("Global Mean Temperature Anomaly (°C, rel. 1850–1900)", fontsize=11)
    ax.set_ylabel("Thermodynamic Rate (mm/yr)", fontsize=11)
    ax.set_title("Rate vs Temperature: IPCC AR6 Thermodynamic Projections\n"
                 "(oceandynamics + glaciers + GIS)", fontsize=11)
    ax.legend(fontsize=8, loc="upper left", ncol=2)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    if fig_path:
        fig.savefig(fig_path, dpi=150, bbox_inches="tight")
        print(f"Saved: {fig_path}")
    plt.close(fig)

    return fig


def plot_component_comparison(component_results, fig_path=None):
    """Plot rate-vs-T for thermodynamic and AIS components separately.

    Fit curves are clipped to each SSP's data range.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

    for ax, comp_name, title in zip(
        axes,
        ["thermodynamic", "ais"],
        ["Thermodynamic (ocean + glaciers + GIS)", "Antarctica (AIS)"],
    ):
        for ssp_code in sorted(component_results.keys()):
            cr = component_results[ssp_code].get(comp_name)
            if cr is None:
                continue
            color = SSP_COLORS[ssp_code]
            label = SSP_LABELS[ssp_code]

            ax.scatter(
                cr["T_mid"], cr["rate"] * 1000,
                color=color, s=40, zorder=3, alpha=0.8,
                label=label,
            )
            T_lo, T_hi = cr["T_mid"].min(), cr["T_mid"].max()
            margin = 0.10 * (T_hi - T_lo) if T_hi > T_lo else 0.1
            T_curve = np.linspace(T_lo - margin, T_hi + margin, 100)
            pred = np.polyval(cr["quadratic"]["coeffs"], T_curve) * 1000
            ax.plot(T_curve, pred, color=color, ls="-", lw=1.5, alpha=0.6)

        ax.set_xlabel("Temperature Anomaly (°C)", fontsize=10)
        ax.set_ylabel("Component Rate (mm/yr)", fontsize=10)
        ax.set_title(title, fontsize=11)
        ax.legend(fontsize=7, loc="upper left")
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    if fig_path:
        fig.savefig(fig_path, dpi=150, bbox_inches="tight")
        print(f"Saved: {fig_path}")
    plt.close(fig)

    return fig


# =====================================================================
#  8. CROSS-SSP CONSISTENCY TEST
# =====================================================================

def test_cross_ssp_consistency(polyfit_results):
    """Test whether dalpha_dT is approximately constant across SSPs.

    If the quadratic sensitivity is a true physical relationship, it
    should be approximately SSP-independent.

    Returns
    -------
    dict with mean, std, cv (coefficient of variation), and per-SSP values
    """
    vals = {}
    for ssp_code, pf in polyfit_results.items():
        vals[ssp_code] = pf["quadratic"]["dalpha_dT_mm"]

    arr = np.array(list(vals.values()))
    mean_val = np.mean(arr)
    std_val = np.std(arr)
    cv = std_val / abs(mean_val) if abs(mean_val) > 0 else np.nan

    return {
        "per_ssp": vals,
        "mean": mean_val,
        "std": std_val,
        "cv": cv,
        "is_consistent": cv < 0.5,  # <50% CV = roughly consistent
    }


# =====================================================================
#  9. STATISTICAL POWER ANALYSIS
# =====================================================================

# Observational DOLS calibration (from results_summary.json)
OBS_DALPHA_DT_MM = 5.086   # mm/yr/°C²
OBS_DALPHA_DT_SE_MM = 0.563  # mm/yr/°C²


def compute_power_analysis(polyfit_results, obs_dalpha_dT=OBS_DALPHA_DT_MM,
                           obs_dalpha_dT_se=OBS_DALPHA_DT_SE_MM, alpha=0.05):
    """Assess whether the IPCC decadal data can detect the observational
    quadratic sensitivity, given the sample size and residual variance.

    For each SSP we compute:
    1. The standard error on the quadratic coefficient from the polyfit
    2. The minimum detectable effect (MDE) at 80% power
    3. Whether the observational dα/dT falls within the ±2σ interval
       of the IPCC quadratic estimate (i.e. is it *inconsistent*?)

    The key question is: can 8 rate-vs-T points (from 9 decadal GMSL
    values) distinguish a quadratic of ~5 mm/yr/°C² from zero?

    Parameters
    ----------
    polyfit_results : dict from polyfit_model_selection
    obs_dalpha_dT : float  (observational value, mm/yr/°C²)
    obs_dalpha_dT_se : float  (observational SE, mm/yr/°C²)
    alpha : float  (significance level for MDE calculation)

    Returns
    -------
    power_results : dict
        ssp_code -> dict with 'se_a', 'mde_80', 'obs_within_2sigma',
                    'obs_within_ci', 'power_at_obs'
    """
    z_alpha = stats.norm.ppf(1 - alpha / 2)  # two-tailed critical value
    z_beta80 = stats.norm.ppf(0.80)  # 80% power

    power_results = {}
    for ssp_code, pf in polyfit_results.items():
        se_a = pf["quadratic"]["se"][0] * 1000  # m -> mm
        n = pf["n"]

        # MDE at 80% power: smallest |a| we can detect
        # MDE = (z_alpha + z_beta) * SE
        mde_80 = (z_alpha + z_beta80) * se_a

        # Is observational value within the IPCC 95% CI?
        a_hat = pf["quadratic"]["dalpha_dT_mm"]
        ci_lo = a_hat - z_alpha * se_a
        ci_hi = a_hat + z_alpha * se_a
        obs_within_ci = ci_lo <= obs_dalpha_dT <= ci_hi

        # Power to detect the observational effect size
        # power = P(reject H0 | true a = obs_dalpha_dT)
        #       = Φ((|obs| / SE) - z_alpha)
        noncentrality = abs(obs_dalpha_dT) / se_a if se_a > 0 else np.inf
        power_at_obs = stats.norm.cdf(noncentrality - z_alpha)

        power_results[ssp_code] = {
            "n_rates": n,
            "se_a_mm": se_a,
            "mde_80_mm": mde_80,
            "ipcc_dalpha_dT_mm": a_hat,
            "ipcc_ci_95": (ci_lo, ci_hi),
            "obs_dalpha_dT_mm": obs_dalpha_dT,
            "obs_within_ci": obs_within_ci,
            "power_at_obs": power_at_obs,
        }

    return power_results


# =====================================================================
#  10. MAIN ANALYSIS
# =====================================================================

def run_analysis(verbose=True):
    """Execute the full IPCC emergent sensitivity analysis.

    Returns all results for downstream use (e.g. in a notebook).
    """
    # ---- Load data ----
    if verbose:
        print("=" * 70)
        print("IPCC EMERGENT QUADRATIC SENSITIVITY TEST")
        print("=" * 70)
        print()

    ipcc_gmsl, ipcc_temp, hist_temp = load_ipcc_data()
    ssp_data = prepare_ssp_data(ipcc_gmsl, ipcc_temp, hist_temp)

    if verbose:
        print(f"Loaded {len(ssp_data)} SSP scenarios: {list(ssp_data.keys())}")
        for ssp, d in ssp_data.items():
            print(f"  {SSP_LABELS[ssp]}: T=[{d['temperature'][0]:.2f},"
                  f" {d['temperature'][-1]:.2f}]°C, "
                  f"thermo=[{d['thermodynamic'][0]*1000:.0f},"
                  f" {d['thermodynamic'][-1]*1000:.0f}] mm, "
                  f"n={len(d['years'])} points")
        print()

    # ---- Step 1: Polyfit model selection (PRIMARY method) ----
    # Fit the thermodynamic signal (oceandynamics + glaciers + GIS),
    # which is the IPCC analogue of the observational DOLS target
    # (GMSL minus TWS, excluding AIS).
    if verbose:
        print("-" * 70)
        print("STEP 1: np.polyfit Model Selection (Linear vs Quadratic)")
        print("        Target: thermodynamic signal (ocean + glaciers + GIS)")
        print("-" * 70)

    polyfit_results = {}
    for ssp_code, data in ssp_data.items():
        result = polyfit_model_selection(
            data["years"], data["thermodynamic"], data["temperature"],
            label=SSP_LABELS[ssp_code],
        )
        polyfit_results[ssp_code] = result

        if verbose:
            q = result["quadratic"]
            l = result["linear"]
            print(f"\n  {SSP_LABELS[ssp_code]}:")
            print(f"    Linear:    R²={l['r2']:.6f}  "
                  f"α₀={l['alpha0']*1000:.3f} mm/yr/°C  "
                  f"trend={l['trend']*1000:.3f} mm/yr")
            print(f"    Quadratic: R²={q['r2']:.6f}  "
                  f"dα/dT={q['dalpha_dT_mm']:.3f} mm/yr/°C²  "
                  f"α₀={q['alpha0']*1000:.3f} mm/yr/°C  "
                  f"trend={q['trend']*1000:.3f} mm/yr")
            print(f"    F-test: F={result['f_test']['f_stat']:.3f}, "
                  f"p={result['f_test']['p_value']:.4g}")
            print(f"    Quad coeff: t={q['t_stat_a']:.3f}, "
                  f"p={q['p_value_a']:.4g}")
            print(f"    ΔAIC={result['aic_comparison']['delta_aic']:+.2f}  "
                  f"ΔBIC={result['bic_comparison']['delta_bic']:+.2f}")
            print(f"    → {result['recommendation']}")

    # ---- Step 2: DOLS consistency check ----
    if verbose:
        print()
        print("-" * 70)
        print("STEP 2: DOLS Consistency Check (n_lags=0)")
        print("-" * 70)

    dols_results = run_dols_on_ipcc(ssp_data)

    if verbose:
        for ssp_code in sorted(dols_results.keys()):
            dr = dols_results[ssp_code]
            r2 = dr.get("order2")
            if r2 is not None:
                # Compare with polyfit
                pf_a = polyfit_results[ssp_code]["quadratic"]["dalpha_dT_mm"]
                dols_a = r2.dalpha_dT * 1000
                diff_pct = (dols_a - pf_a) / abs(pf_a) * 100 if pf_a != 0 else np.nan
                print(f"\n  {SSP_LABELS[ssp_code]}:")
                print(f"    DOLS order=2: dα/dT={dols_a:.3f} mm/yr/°C²  "
                      f"α₀={r2.alpha0*1000:.3f} mm/yr/°C  "
                      f"R²={r2.r2:.6f}")
                print(f"    polyfit:       dα/dT={pf_a:.3f} mm/yr/°C²")
                print(f"    Discrepancy: {diff_pct:+.1f}%")

    # ---- Step 3: Decomposed component analysis ----
    if verbose:
        print()
        print("-" * 70)
        print("STEP 3: Decomposed Component Analysis")
        print("-" * 70)

    component_results = run_component_analysis(ssp_data)

    if verbose:
        for ssp_code in sorted(component_results.keys()):
            cr = component_results[ssp_code]
            print(f"\n  {SSP_LABELS[ssp_code]}:")
            for comp_name in ["thermodynamic", "ais"]:
                c = cr[comp_name]
                q = c["quadratic"]
                print(f"    {comp_name:15s}: dα/dT={q['dalpha_dT_mm']:+.3f} mm/yr/°C², "
                      f"R²_quad={q['r2']:.4f}, "
                      f"F p={c['f_test']['p_value']:.4g}  "
                      f"→ {c['recommendation'].split('.')[0]}.")

    # ---- Step 4: Cross-SSP consistency ----
    if verbose:
        print()
        print("-" * 70)
        print("STEP 4: Cross-SSP Consistency")
        print("-" * 70)

    consistency = test_cross_ssp_consistency(polyfit_results)

    if verbose:
        print(f"\n  Thermodynamic dα/dT across SSPs:")
        for ssp, val in consistency["per_ssp"].items():
            print(f"    {SSP_LABELS[ssp]}: {val:+.3f} mm/yr/°C²")
        print(f"  Mean: {consistency['mean']:.3f} ± {consistency['std']:.3f} mm/yr/°C²")
        print(f"  CV: {consistency['cv']:.2f} "
              f"({'consistent' if consistency['is_consistent'] else 'NOT consistent'})")

    # ---- Step 5: Statistical power analysis ----
    if verbose:
        print()
        print("-" * 70)
        print("STEP 5: Statistical Power Analysis")
        print(f"        Can {8} rate-vs-T points detect obs dα/dT "
              f"= {OBS_DALPHA_DT_MM:.1f} mm/yr/°C²?")
        print("-" * 70)

    power_results = compute_power_analysis(polyfit_results)

    if verbose:
        for ssp_code in sorted(power_results.keys()):
            pr = power_results[ssp_code]
            print(f"\n  {SSP_LABELS[ssp_code]} (n={pr['n_rates']} rates):")
            print(f"    SE(dα/dT) = {pr['se_a_mm']:.2f} mm/yr/°C²")
            print(f"    MDE (80% power) = {pr['mde_80_mm']:.2f} mm/yr/°C²"
                  f"  {'< obs → detectable' if pr['mde_80_mm'] < OBS_DALPHA_DT_MM else '> obs → UNDETECTABLE'}")
            ci = pr["ipcc_ci_95"]
            print(f"    IPCC 95% CI: [{ci[0]:+.2f}, {ci[1]:+.2f}] mm/yr/°C²")
            print(f"    Obs value {OBS_DALPHA_DT_MM:.1f} within CI: "
                  f"{'YES (consistent)' if pr['obs_within_ci'] else 'NO (inconsistent)'}")
            print(f"    Power to detect obs effect: {pr['power_at_obs']:.1%}")

    # ---- Build summary table ----
    summary_df = build_summary_table(polyfit_results, dols_results, component_results)

    if verbose:
        print()
        print("-" * 70)
        print("SUMMARY TABLE")
        print("-" * 70)
        with pd.option_context("display.max_columns", 20, "display.width", 200,
                               "display.float_format", "{:.4f}".format):
            print(summary_df.to_string(index=False))

    # ---- Plots ----
    if verbose:
        print()
        print("-" * 70)
        print("FIGURES")
        print("-" * 70)

    fig_phase = plot_rate_vs_temperature_phase(
        polyfit_results, ssp_data,
        fig_path=os.path.join(FIG_DIR, "ipcc_rate_vs_temperature_phase.png"),
    )

    fig_comp = plot_component_comparison(
        component_results,
        fig_path=os.path.join(FIG_DIR, "ipcc_component_rate_vs_temperature.png"),
    )

    # ---- Return all results ----
    return {
        "ssp_data": ssp_data,
        "polyfit_results": polyfit_results,
        "dols_results": dols_results,
        "component_results": component_results,
        "consistency": consistency,
        "power_results": power_results,
        "summary_df": summary_df,
        "fig_phase": fig_phase,
        "fig_comp": fig_comp,
    }


# =====================================================================
if __name__ == "__main__":
    results = run_analysis(verbose=True)
