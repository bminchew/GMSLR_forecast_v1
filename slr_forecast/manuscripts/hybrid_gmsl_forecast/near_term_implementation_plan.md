# Implementation Plan: Near-Term Trend Constraint via Bayesian Predictive Synthesis (Revised)

## Overview

This document specifies the implementation of a near-term trend constraint module that combines a non-parametric trend forecast with the rate-and-state model projection using Bayesian Predictive Synthesis (BPS). The synthesis hyperparameters are calibrated from leave-future-out cross-validation on the satellite-era record. Their posterior uncertainty is propagated through the combined predictive, producing credible intervals that account for uncertainty in both the individual forecasts and the combination itself.

**Read the companion LaTeX document `near_term_trend_constraint_revised.tex` for the full mathematical derivation before implementing.**

## Recorded Decisions

**DECISION 1: Only WAIS is exempt from trend suppression.** WAIS dynamics are largely decoupled from GMST (driven by ocean thermal forcing at grounding lines, ice shelf buttressing loss). EAIS and Antarctic Peninsula remain in the trend-constrained total: the Peninsula has a direct surface-melting connection to GMST, and EAIS SMB is a well-behaved snowfall response to warming (Clausius-Clapeyron). IMBIE provides basin-level decomposition (WAIS, EAIS, Peninsula), so subtracting WAIS total from observations is operationally feasible. WAIS SMB is small relative to WAIS dynamics and is included in the exemption for simplicity.

**DECISION 2: BPS as post-processing (Option A).** The budget constraint operates on the uncombined Level 1 total. The BPS-combined total is a diagnostic overlay. Option B (integrating WLS rate prior into Level 1 calibration) deferred until joint MCMC is operational.

**DECISION 3: emcee primary + dynesty cross-check (mandatory).** The log-mixture likelihood can produce multimodal posteriors. Both samplers must agree before proceeding.


## Key Design Decisions

1. **Bayesian Predictive Synthesis, not MSE-optimal weighting.** The combination uses BPS (McAlinn & West, 2019) with a structured latent weight process calibrated from cross-validation. This replaces the MSE-optimal point-forecast weighting, which produced systematically overconfident predictive intervals: the linear-combination ensemble spread was w²·Var_trend + (1-w)²·Var_model, strictly narrower than either individual ensemble for 0 < w < 1. BPS produces a properly calibrated mixture predictive.

2. **Synthesis hyperparameters calibrated, not assumed.** The previous approach pegged κ to the model's own ∂r/∂T — a circular estimate where model overconfidence in its rate-temperature sensitivity accelerated its own dominance. The BPS framework calibrates the synthesis hyperparameters via leave-future-out cross-validation, replacing the assumed κ with a data-informed posterior. The posterior reports which aspects of the combination are identified by data (φ₀, κ_t) and which remain prior-dominated (κ_net at large ΔT).

3. **No double-counting of acceleration-estimation variance.** The corrected MSE uses Var[samples] + κ·ΔT²·dt² + κ_t·dt⁴, where κ_t is the pure acceleration-persistence bias.

4. **WAIS exempt from trend suppression.** Both agents subtract WAIS total (dynamics + SMB). The add-back uses WAIS total. EAIS and Peninsula remain in the trend-constrained total (they have GMST connections). This is consistent between agents and prevents the trend constraint from vetoing near-term WAIS instability scenarios.

5. **Mixture predictive for uncertainty reporting; stochastic-transition trajectories for coherence.** The combined predictive is a mixture distribution (correctly calibrated marginals). Temporally coherent trajectories for rate-of-change analysis and counterfactual queries use sample-specific synthesis weights with a smooth stochastic transition.

6. **Shared-data covariance handled by construction.** The BPS cross-validation evaluates both agents against the same held-out observations, capturing whatever covariance structure exists without requiring an explicit covariance computation.

7. **Systematic uncertainty propagated via rank-1 inflations.** GIA and WAIS subtraction systematics inflate the full 3×3 covariance (not just the rate variance).

8. **Forcing-form sensitivity requires full refit.** The default and optimistic λ forms have incompatible units. Sensitivity analysis runs a complete BPS pipeline under each form.

9. **Module placement alongside existing analysis code.** The BPS synthesis is a post-processing module, not a physical component. New code lives in `slr_forecast/notebooks/` alongside existing modules.

10. **Maximise reuse of existing code.** The project already contains WLS fitting, emcee sampling, convergence diagnostics, model selection, data readers, and projection functions. New code should wrap or call these rather than reimplementing.


## File Structure

New code goes under the existing project layout, alongside the existing analysis modules:

```
slr_forecast/notebooks/
├── slr_analysis.py               # EXISTING — compute_kinematics(), calibrate_dols(), DOLSResult
├── slr_projections.py            # EXISTING — project_gmsl_ensemble(), project_gmsl_state_ensemble()
├── bayesian_dols.py              # EXISTING — fit_bayesian_dols(), check_convergence(), build_dols_design_matrix()
├── slr_data_readers.py           # EXISTING — read_imbie_*, read_nasa_gmsl(), temperature readers
├── ipcc_emergent_sensitivity.py  # EXISTING — polyfit_model_selection() (AIC/BIC/F-test)
├── dols_robustness.py            # EXISTING — load_all_gmsl(), load_all_gmst()
│
├── trend_constraint.py           # NEW — all BPS trend constraint code (single file)
│   # Contains: compute_T_cal_max, compute_delta_T, fit_wls_quadratic,
│   #   predict_at_t_end, sample_derivatives, bma_cubic_check,
│   #   trend_projection_ensemble, BPSSynthesis class,
│   #   compute_mixture_quantiles, generate_stochastic_transition_ensemble,
│   #   compute_crossover_diagnostics, run_forcing_form_sensitivity
│
└── test_trend_constraint.py      # NEW — tests for trend_constraint.py
```

**Rationale for single file:** The BPS trend constraint is a coherent module with ~500 lines of genuinely new code (after reuse). Splitting across 6 files with 5 test files would create unnecessary navigation overhead. A single module with clear section headers is easier to maintain.

### Existing code reused (not modified)

| Existing function | Location | Used for |
|---|---|---|
| `check_convergence(trace)` | `bayesian_dols.py:274` | R-hat, ESS diagnostics for BPS posterior |
| `fit_bayesian_dols()` pattern | `bayesian_dols.py:493` | emcee walker init, burn-in, thinning pattern |
| `polyfit_model_selection()` | `ipcc_emergent_sensitivity.py:206` | Adapted for BMA quadratic-vs-cubic check |
| `calibrate_dols()` | `slr_analysis.py` | Model refitting in leave-future-out CV |
| `project_gmsl_state_ensemble()` | `slr_projections.py:453` | Model predictive for CV holdout evaluation |
| `read_imbie_*()` | `slr_data_readers.py` | WAIS basin-level data loading |
| `read_nasa_gmsl()` | `slr_data_readers.py` | Satellite-era GMSL |
| `load_all_gmsl()`, `load_all_gmst()` | `dols_robustness.py` | Multi-dataset GMSL/GMST loading |


## Section 1: Temperature Departure

Purpose: Compute the temperature departure ΔT(t) from the calibration domain for arbitrary SSP trajectories. Pure numpy, ~15 lines.

```python
def compute_T_cal_max(T_cal: np.ndarray) -> float:
    """Maximum GMST in the calibration window. Returns max(T_cal)."""

def compute_delta_T(
    T_scenario: np.ndarray, T_max_cal: float, overshoot_mode: bool = True,
) -> np.ndarray:
    """
    Temperature departure from the calibration domain.

    overshoot_mode=True (default): ΔT(t) = cummax(max(0, T(s) - T_max_cal))
    overshoot_mode=False: ΔT(t) = max(0, T(t) - T_max_cal)
    """
```

### Tests

1. **Zero departure:** If `T_scenario ≤ T_max_cal` everywhere, `delta_T` is identically zero.
2. **Monotone increasing:** Both modes give `max(0, T_scenario - T_max_cal)`.
3. **Overshoot:** `T_scenario = [1.5, 1.8, 1.6, 1.4]`, `T_max_cal = 1.2`:
   - Instantaneous: `[0.3, 0.6, 0.4, 0.2]`
   - Overshoot: `[0.3, 0.6, 0.6, 0.6]`


## Section 2: WLS Trend Model

Purpose: Fit a weighted least-squares quadratic to satellite-era GMSL altimetry and extract the posterior on (H, r, r̈) at the end of the calibration window. Includes systematic uncertainty inflations.

**New code required:** ~80 lines (statsmodels WLS + Jacobian propagation + rank-1 inflations). This is genuinely new — the existing `calibrate_dols()` fits temperature integrals, not time polynomials; `compute_kinematics()` uses kernel-weighted local regression, not parametric WLS.

**Code reuse:** The BMA cubic check adapts the BIC computation pattern from `polyfit_model_selection()` in `ipcc_emergent_sensitivity.py` (line 206). That function already computes AIC, BIC, F-test, and model selection criteria — we adapt it for quadratic-vs-cubic (instead of linear-vs-quadratic).

### Functions (not a class — functions are simpler and sufficient)

```python
def fit_wls_quadratic(
    years: np.ndarray,
    gmsl: np.ndarray,
    gmsl_uncertainty: np.ndarray | None = None,
    satellite_start: float = 1993.0,
    fix_scale: bool = True,
) -> dict:
    """
    Fit WLS quadratic H(t) = α + β·(t - t_ref) + γ·(t - t_ref)² to
    satellite-era GMSL.

    Uses statsmodels.WLS. If fix_scale=True, uses cov_type='fixed scale'.

    Returns
    -------
    dict with keys:
        "params" : np.ndarray (3,) — [α, β, γ]
        "cov_params" : np.ndarray (3, 3) — parameter covariance
        "t_ref" : float — time reference (midpoint of satellite era)
        "sigma_sq_est" : float — estimated σ̂² (for diagnostic even if fix_scale=True)
        "wls_result" : statsmodels result object
    """


def predict_at_t_end(
    wls_fit: dict,
    t_end: float,
    sigma_gia: float = 0.15,
    sigma_ais_sys: float = 0.10,
) -> dict:
    """
    Extract posterior on (H, r, r̈) at t_end via Jacobian propagation.

    J = [[1, Δ, Δ²], [0, 1, 2Δ], [0, 0, 2]]  where Δ = t_end - t_ref

    Σ_total = J @ Σ_params @ J.T + σ²_gia · v · v.T + σ²_ais · v · v.T
    where v = (Δ, 1, 0) (constant rate bias → level and rate, not acceleration).

    Returns
    -------
    dict with keys: "H_mean", "r_mean", "rdot_mean", "cov_3x3"
    """


def sample_derivatives(
    prediction: dict, n_samples: int = 10_000, seed: int | None = None,
) -> dict:
    """Draw (H, r, rdot) from MVN(mean, cov_3x3). Returns dict of arrays."""


def bma_cubic_check(
    years: np.ndarray,
    gmsl: np.ndarray,
    gmsl_uncertainty: np.ndarray | None,
    t_end: float,
    satellite_start: float = 1993.0,
) -> dict:
    """
    BMA quadratic-vs-cubic via BIC.

    REUSES the BIC computation pattern from polyfit_model_selection() in
    ipcc_emergent_sensitivity.py. Adapted: fits in time (not temperature),
    compares order 2 vs order 3.

    P(M_quad | D) ≈ exp(-BIC_q/2) / [exp(-BIC_q/2) + exp(-BIC_c/2)]

    Returns
    -------
    dict with: "bma_prob_cubic", "bma_prob_quadratic", "cubic_coeff",
               "cubic_coeff_se", "scale_diagnostic_flag" (σ̂² > 2.0)
    """
```

### Tests

1. **Synthetic recovery:** GMSL = 50·t + 3·t² + noise → verify r_mean ≈ 50 + 6·t_end, rdot_mean ≈ 6.0.
2. **GIA inflation structure:** rank-1, affects (0,0), (0,1), (1,0), (1,1); zero in row/column 2.
3. **Scale fixing:** fix_scale=True → (X^T W X)^{-1}; fix_scale=False → σ̂²(X^T W X)^{-1}.
4. **BMA cubic:** Pure quadratic data → bma_prob_cubic small. Known cubic → bma_prob_cubic large.


## Section 3: Trend Projection Ensemble

Purpose: Generate trend projection ensembles from the WLS derivative samples. ~20 lines.

```python
def trend_projection_ensemble(
    t_proj: np.ndarray,
    t_end: float,
    derivative_samples: dict,
) -> np.ndarray:
    """
    Generate trend projection ensemble. Vectorised, no loop.

    H_trend^(s)(t) = H^(s) + r^(s)·dt + 0.5·rdot^(s)·dt²

    Parameters
    ----------
    t_proj : np.ndarray, shape (n_proj,)
    t_end : float
    derivative_samples : dict — output of sample_derivatives()

    Returns
    -------
    H_ensemble : np.ndarray, shape (n_samples, n_proj)
    """
```

### Implementation notes

- Pure broadcasting: `dt = t_proj - t_end` (shape n_proj), samples are (n_samples,1) broadcasted.
- No MSE computation here. The BPS synthesis does not use a standalone MSE — it evaluates agent densities directly at holdout observations. The previous plan's `kappa`/`kappa_t` parameters for MSE_trend were diagnostic-only and created naming confusion with the BPS synthesis parameters (κ_net, κ_t). Removed for clarity. If a standalone MSE diagnostic is desired, it can be computed post-hoc from the ensemble variance.


## Section 4: BPS Synthesis

Purpose: Implement the Bayesian Predictive Synthesis framework: weight process, leave-future-out cross-validation, posterior sampling. This is the core new code (~250 lines).

### Dependencies (existing)

- `emcee` — already a project dependency (used in `bayesian_dols.py`)
- `dynesty` — new dependency for cross-check
- `check_convergence()` from `bayesian_dols.py` — **reused directly** for R-hat/ESS diagnostics
- emcee sampling pattern from `fit_bayesian_dols()` — **reused** for walker init, burn-in, thinning, arviz trace construction

### Code reuse detail

The existing `fit_bayesian_dols()` (bayesian_dols.py:493) follows this pattern:
1. Build design matrix, initialize walkers near MLE
2. Run emcee with burn-in, then production samples
3. Convert to arviz InferenceData
4. Call `check_convergence(trace)`
5. Extract posterior statistics

The BPS synthesis follows the same pattern with a different log-posterior. The sampler boilerplate (~40 lines) is identical.

### `model_refit_fn` specification (previously underspecified)

The `model_refit_fn` callback is built from existing code:

```python
def make_model_refit_fn(
    full_sl, full_temp, full_sigma, historical_temperature, historical_time,
    baseline_year, n_samples=500, order=2, n_lags=2,
):
    """
    Factory for model_refit_fn using existing calibrate_dols() +
    project_gmsl_state_ensemble().

    REUSES: calibrate_dols() from slr_analysis.py
    REUSES: project_gmsl_state_ensemble() from slr_projections.py

    For CV speed, uses n_samples=500 (not 10,000) and the frequentist
    calibrate_dols() (not full Bayesian). The synthesis posterior is
    insensitive to the CV model approximation because the BPS
    hyperparameters (φ₀, κ_net, κ_t) are low-dimensional and the CV
    log-likelihood is dominated by the trend agent at short leads.

    Returns
    -------
    callable : t_h -> {"H_predictive_mean": array, "H_predictive_std": array}
    """
    def refit_fn(t_h):
        # 1. Subset data to [1900, t_h]
        mask = full_time <= t_h
        sl_sub = full_sl[mask]
        temp_sub = full_temp[mask]
        sigma_sub = full_sigma[mask] if full_sigma is not None else None

        # 2. Refit DOLS (fast, ~0.1s)
        result = calibrate_dols(sl_sub, temp_sub, gmsl_sigma=sigma_sub,
                                order=order, n_lags=n_lags)

        # 3. Project using observed temperature for t > t_h
        #    (NOT SSP scenario — this is hindcast CV)
        holdout_mask = full_time > t_h
        holdout_temp = ...  # observed temperature trajectory
        proj_result = project_gmsl_state_ensemble(
            coefficients=result.physical_coefficients,
            coefficients_cov=result.physical_covariance,
            tau_samples=...,  # from prior or previous calibration
            temperature_projections={'holdout': holdout_temp_df},
            historical_temperature=..., historical_time=...,
            n_samples=n_samples, baseline_year=baseline_year,
        )

        # 4. Extract mean and std at holdout times
        ens = proj_result['scenarios']['holdout']
        return {
            "H_predictive_mean": ens['gmsl'].values,
            "H_predictive_std": (ens['gmsl_upper'].values - ens['gmsl_lower'].values) / 3.29,
        }
    return refit_fn
```

### Functions and class

```python
def compute_synthesis_weight(dt, delta_T, phi_0, kappa_net, kappa_t):
    """w(t) = σ(φ₀ - κ_net·ΔT²·dt² - κ_t·dt⁴). Vectorised."""

def synthesis_log_likelihood(theta, cv_data):
    """Log-sum-exp stable mixture log-likelihood over CV holdouts."""

def synthesis_log_prior(theta):
    """φ₀ ~ TruncN(2.5, 0.5², >0), κ_net ~ LogN(log 15, 0.6), κ_t ~ LogN(log 1e-3, 0.5)."""

def synthesis_log_posterior(theta, cv_data):
    """log_prior + log_likelihood."""


class BPSSynthesis:
    """Orchestrates BPS: CV data, posterior sampling, weight computation."""

    def __init__(self, n_walkers=16, n_warmup=2000, n_samples=5000, seed=None): ...

    def build_cv_data(self, years, gmsl, gmsl_uncertainty, T_obs,
                       model_refit_fn, holdout_endpoints=None, min_holdout=5):
        """
        Leave-future-out CV. For each t_h:
        1. fit_wls_quadratic() on [satellite_start, t_h]
        2. Recompute T_cal_max(t_h) — avoids information leakage
        3. model_refit_fn(t_h) → model predictive
        4. Evaluate trend/model Gaussian log-densities at holdout obs

        REUSES: fit_wls_quadratic(), predict_at_t_end() from Section 2
        """

    def fit(self, cv_data):
        """
        emcee sampling on ψ = (φ₀, log κ_net, log κ_t).

        REUSES PATTERN from fit_bayesian_dols() (bayesian_dols.py:493):
        - Walker initialisation near MLE (scipy.optimize.minimize)
        - emcee.EnsembleSampler with n_walkers, burn-in, production
        - Convert chains to arviz InferenceData
        - check_convergence(trace) from bayesian_dols.py
        """

    def fit_dynesty(self, cv_data):
        """dynesty nested sampling cross-check. Stores alongside emcee."""

    def check_sampler_agreement(self, tolerance=0.1):
        """Compare emcee/dynesty marginal quantiles. Returns dict."""

    def get_posterior_samples(self) -> np.ndarray:
        """Shape (n_total, 3): [φ₀, κ_net, κ_t]."""

    def posterior_mean_weight(self, dt, delta_T) -> np.ndarray:
        """w̄(t) = mean_s σ(φ(t; ψ^(s))). Vectorised over samples."""

    def posterior_weight_ensemble(self, dt, delta_T, n_samples=None) -> np.ndarray:
        """Per-sample weight trajectories, shape (n_psi, n_proj)."""

    def convergence_diagnostics(self):
        """
        DELEGATES to check_convergence() from bayesian_dols.py.
        Returns R-hat, ESS, acceptance fraction.
        """

    def holdout_influence_diagnostics(self, cv_data, ess_threshold=100):
        """Importance-weight pre-screen for leave-one-holdout-out."""
```

### Implementation notes

- emcee operates in transformed space: φ₀ direct (reflecting at 0), log(κ_net) and log(κ_t) sampled.
- Log-likelihood uses log-sum-exp trick for `log(w·h₁ + (1-w)·h₂)`.
- **Convergence diagnostics delegate to `check_convergence(trace)`** from bayesian_dols.py — no new code needed.
- The CV model refit uses the fast frequentist `calibrate_dols()` + `project_gmsl_state_ensemble()` — not full Bayesian. This is a deliberate approximation for CV speed.


## Section 5: Combination and WAIS Add-Back

Purpose: Generate mixture quantiles, stochastic-transition trajectories, WAIS add-back. ~100 lines.

```python
N_ENSEMBLE = 10_000  # Target ensemble size


def align_ensemble_sizes(*ensembles, target_size=N_ENSEMBLE, seed=None):
    """Resample each ensemble to target_size with replacement. ~10 lines."""


def compute_mixture_quantiles(H_trend_ens, H_model_ens, w_bar, quantiles=...):
    """
    Weighted quantiles of BPS mixture. Pool trend/model samples at each t,
    weight by w̄(t)/n_trend and (1-w̄(t))/n_model. ~20 lines.
    """


def generate_stochastic_transition_ensemble(
    H_trend_ens, H_model_ens, w_ensemble, epsilon=0.05, seed=None,
):
    """
    Temporally coherent trajectories. For each sample s:
    1. Draw u^(s) ~ U(0,1)
    2. ω^(s)(t) = σ((w^(s)(t) - u^(s)) / ε)
    3. H^(s)(t) = ω·H_trend + (1-ω)·H_model

    PREREQUISITE: All inputs same n_samples. Use align_ensemble_sizes() first.

    Returns dict: "H_combined", "u_draws", "transition_times". ~30 lines.
    """


def add_wais_component(H_comb_non_wais, H_wais):
    """H_total = H_non_wais + H_wais. 1 line."""


def rate_smoothness_diagnostic(H_combined, t_proj):
    """Finite-difference rates, check for discontinuities. ~15 lines."""
```

### Ensemble alignment protocol

1. Trend ensemble: `n_samples = N_ENSEMBLE` from `sample_derivatives()`.
2. Model ensemble: resampled to `N_ENSEMBLE` if size differs.
3. Weight ensemble: resampled from BPS posterior to `N_ENSEMBLE`.
4. Three resamplings are independent.
5. Quantile matching for trend/model (pairing by rank at each t).


## Section 6: Crossover Diagnostics

Purpose: Compute crossover time posterior from BPS. ~50 lines.

```python
def compute_crossover_diagnostics(bps, t_proj, t_end, delta_T_by_ssp,
                                   T_rate_by_ssp=None):
    """
    For each SSP and posterior sample ψ^(s), root-find φ(t; ψ^(s)) = 0.
    Overshoot SSPs (T_rate=None) skip the analytic approximation.

    Returns dict: crossover_posterior, crossover_median, crossover_90CI,
    w_bar_at_2050, w_bar_at_2100, crossover_driver, analytic_DT_star.
    """
```


## Section 7: Forcing-Form Sensitivity

Purpose: Run the full BPS pipeline under alternative λ form (ΔT⁴ instead of ΔT²). ~30 lines (orchestration only — reuses BPSSynthesis).

```python
def run_forcing_form_sensitivity(cv_data_default, cv_data_optimistic,
                                  bps_default, delta_T_by_ssp, t_proj, t_end):
    """
    Two separate BPS calibrations (incompatible κ_net units).
    Compare crossover diagnostics. This is a prior sensitivity analysis,
    meaningful only in projection (ΔT ≈ 0 in CV regime).
    """
```


## Section 8: Data Loading (no new code)

All data loading uses existing readers. No new functions needed.

### Satellite-era non-WAIS GMSL

```python
# EXISTING: slr_data_readers.py
from slr_data_readers import read_nasa_gmsl, read_imbie_*

# Satellite GMSL (1993–present)
gmsl_total = read_nasa_gmsl(filepath)  # or from HDF5 store

# WAIS basin-level (1992–present, from IMBIE)
# read_imbie_* functions already in slr_data_readers.py
wais_sl = read_imbie_wais(filepath)  # if not available, use read_imbie_antarctica()
                                      # with basin decomposition from IMBIE files

# Non-WAIS GMSL
gmsl_non_wais = gmsl_total - wais_sl  # on common time grid
```

**Consistency requirement:** The same WAIS quantity must be subtracted from both agents. EAIS and Peninsula remain in the trend-constrained total.

### SSP temperature trajectories

Already available from Level 0 (loaded via existing HDF5 readers in predictability_analysis.ipynb). The same `T_SSP(t)` arrays used for the rate-and-state projection are passed to `compute_delta_T`.

### Multi-dataset loading

For robustness checks, `load_all_gmsl()` and `load_all_gmst()` from `dols_robustness.py` provide multi-dataset access.


## Execution Workflow

```python
# ===========================================================
# Imports — existing code + new trend_constraint module
# ===========================================================
from slr_analysis import calibrate_dols
from slr_projections import project_gmsl_state_ensemble
from slr_data_readers import read_nasa_gmsl, read_imbie_antarctica  # or WAIS-specific
from bayesian_dols import check_convergence
from trend_constraint import (
    compute_T_cal_max, compute_delta_T,
    fit_wls_quadratic, predict_at_t_end, sample_derivatives,
    bma_cubic_check, trend_projection_ensemble,
    BPSSynthesis, make_model_refit_fn,
    align_ensemble_sizes, compute_mixture_quantiles,
    generate_stochastic_transition_ensemble, add_wais_component,
    rate_smoothness_diagnostic, compute_crossover_diagnostics,
    N_ENSEMBLE,
)

# ===========================================================
# Step 0: Prerequisites (already done)
# - Rate-and-state model calibrated, posterior samples available
# - Projection ensemble H_model[s, t] generated for each SSP
#   (via project_gmsl_state_ensemble() from slr_projections.py)
# - WAIS component H_wais[s, t] separated (dynamics + SMB)
# ===========================================================

# Compute non-WAIS model ensemble (WAIS only, not total AIS)
H_model_non_wais = {ssp: H_model[ssp] - H_wais[ssp] for ssp in ssps}

# ===========================================================
# Step 1: Calibration-domain temperature
# ===========================================================
T_max_cal = compute_T_cal_max(T_obs_calibration)

# ===========================================================
# Step 2: Fit WLS quadratic trend model to satellite-era non-WAIS GMSL
# ===========================================================
wls_fit = fit_wls_quadratic(years_obs, gmsl_non_wais_obs, gmsl_uncertainty_obs)
prediction = predict_at_t_end(wls_fit, t_end, sigma_gia=0.15, sigma_ais_sys=0.10)

# BMA cubic check (reuses BIC pattern from ipcc_emergent_sensitivity.py)
cubic_check = bma_cubic_check(years_obs, gmsl_non_wais_obs, gmsl_uncertainty_obs, t_end)
if cubic_check["scale_diagnostic_flag"]:
    print(f"WARNING: σ̂² = {cubic_check['sigma_sq_estimated']:.2f} > 2.0")

derivatives = sample_derivatives(prediction, n_samples=N_ENSEMBLE)

# ===========================================================
# Step 3: BPS cross-validation and posterior sampling
# ===========================================================
# Build model refit callback from existing code
model_refit_fn = make_model_refit_fn(
    full_sl=sl_series, full_temp=temp_series, full_sigma=sigma_series,
    historical_temperature=hist_temp, historical_time=hist_time,
    baseline_year=2005.0, n_samples=500,
)

bps = BPSSynthesis(n_walkers=16, n_warmup=2000, n_samples=5000, seed=42)

# Build cross-validation data (expensive: refits model for each holdout)
cv_data = bps.build_cv_data(
    years=years_obs, gmsl=gmsl_non_wais_obs,
    gmsl_uncertainty=gmsl_uncertainty_obs,
    T_obs=T_obs, model_refit_fn=model_refit_fn,
)

# Sample posterior on ψ — both samplers
bps.fit(cv_data)              # emcee (uses pattern from fit_bayesian_dols)
bps.fit_dynesty(cv_data)      # dynesty cross-check
sampler_check = bps.check_sampler_agreement()
if not sampler_check["agrees"]:
    print(f"WARNING: Samplers disagree. Using {sampler_check['recommended_sampler']}.")

# Convergence: delegates to check_convergence() from bayesian_dols.py
convergence = bps.convergence_diagnostics()
assert convergence['converged'], "MCMC did not converge"

# Holdout influence (importance-weight pre-screen)
influence = bps.holdout_influence_diagnostics(cv_data)
if influence["flagged_for_refit"]:
    print(f"Holdouts flagged for full MCMC refit: {influence['flagged_for_refit']}")

# ===========================================================
# Step 4: For each SSP
# ===========================================================
results = {}
for ssp in ssps:
    # 4a: Temperature departure
    delta_T = compute_delta_T(T_scenario[ssp], T_max_cal, overshoot_mode=True)
    dt = t_proj - t_end

    # 4b: Trend projection ensemble
    H_trend_ens = trend_projection_ensemble(t_proj, t_end, derivatives)

    # 4c: Posterior mean weight
    w_bar = bps.posterior_mean_weight(dt, delta_T)

    # 4d: Mixture predictive quantiles (PRIMARY uncertainty reporting)
    Q_mixture = compute_mixture_quantiles(H_trend_ens, H_model_non_wais[ssp], w_bar)

    # 4e: Align ensemble sizes
    H_trend_aligned, H_model_aligned = align_ensemble_sizes(
        H_trend_ens, H_model_non_wais[ssp], target_size=N_ENSEMBLE,
    )

    # 4f: Per-sample weight trajectories
    w_ensemble = bps.posterior_weight_ensemble(dt, delta_T, n_samples=N_ENSEMBLE)

    # 4g: Stochastic-transition trajectories (for rate analysis)
    transition = generate_stochastic_transition_ensemble(
        H_trend_aligned, H_model_aligned, w_ensemble, epsilon=0.05,
    )

    # 4h: Add WAIS component back
    H_comb_total = add_wais_component(transition["H_combined"], H_wais[ssp])

    # 4i: Rate smoothness diagnostic
    smoothness = rate_smoothness_diagnostic(H_comb_total, t_proj)

    results[ssp] = {
        "H_combined_non_wais": transition["H_combined"],
        "H_combined_total": H_comb_total,
        "Q_mixture_non_wais": Q_mixture,
        "w_bar": w_bar,
        "transition_times": transition["transition_times"],
        "smoothness": smoothness,
    }

# ===========================================================
# Step 5: Crossover diagnostics
# ===========================================================
delta_T_by_ssp = {ssp: compute_delta_T(T_scenario[ssp], T_max_cal) for ssp in ssps}
T_rate_by_ssp = {
    ssp: T_rate[ssp] if ssp not in ["SSP1-1.9", "SSP1-2.6"] else None
    for ssp in ssps
}
crossover = compute_crossover_diagnostics(bps, t_proj, t_end, delta_T_by_ssp, T_rate_by_ssp)
```


## Diagnostic Outputs

### Table: Crossover diagnostics

| SSP | t* median (yr) | t* 90% CI | ΔT* median (°C) | w̄(2050) | w̄(2100) | Driver |
|-----|----------------|-----------|------------------|----------|----------|--------|
| SSP1-2.6 | ... | [..., ...] | ... | ... | ... | accel_persist |
| SSP2-4.5 | ... | [..., ...] | ... | ... | ... | forcing_depart |
| SSP3-7.0 | ... | [..., ...] | ... | ... | ... | forcing_depart |
| SSP5-8.5 | ... | [..., ...] | ... | ... | ... | forcing_depart |

Note: The crossover now has a posterior distribution, not a point estimate. The 90% CI width reports the irreducible uncertainty in the combination. **Under SSP3-7.0 and SSP5-8.5, the crossover time is prior-dominated (see §CV Identifiability in the LaTeX document).**

### Table: BPS posterior summary

| Parameter | Prior median | Posterior median | Posterior 90% CI | Data-informed? |
|-----------|-------------|-----------------|------------------|----------------|
| φ₀ | 2.5 | ... | [..., ...] | Yes (well identified) |
| κ_net | 15 | ... | [..., ...] | Weakly (small ΔT only) |
| κ_t | 10⁻³ | ... | [..., ...] | Moderately |

### Table: Sampler cross-check

| Parameter | emcee median | dynesty median | Max quantile diff (IQR units) |
|-----------|-------------|----------------|-------------------------------|
| φ₀ | ... | ... | ... |
| κ_net | ... | ... | ... |
| κ_t | ... | ... | ... |

### Figure 1: Posterior weight timeseries

Plot w̄(t) vs year for all SSPs on a single panel. Shaded: 5th-95th percentile of w(t; ψ^(s)) across posterior samples.

### Figure 2: Combined projection fan chart

For one SSP (e.g., SSP2-4.5):
- Shaded 5-95% CI from model-only projection
- Shaded 5-95% CI from trend-only projection
- Shaded 5-95% CI from BPS mixture predictive (**primary**)
- Shaded 5-95% CI from stochastic-transition trajectory ensemble
- Highlight the posterior crossover region

### Figure 3: BPS posterior

Corner plot (2D marginals + 1D histograms) for (φ₀, κ_net, κ_t). Shows correlations and identifiability.

### Figure 4: Cross-validation scores

For each holdout endpoint t_h, plot the log predictive score of the trend agent, model agent, and BPS combined, as a function of lead time. Shows where and when the BPS combination improves over either individual agent.

### Figure 5: Transition time distribution

Histogram of transition times across the trajectory ensemble (from `transition_result["transition_times"]`), for each SSP. Shows the distribution of when individual trajectories switch from trend-dominated to model-dominated.

### Figure 6: Rate smoothness

For one SSP, overlay 50 combined trajectories and their rates (by finite differences). Verify smooth transitions.

### Figure 7: Forcing-form sensitivity

For each SSP, plot the posterior median crossover time under the default λ form (ΔT²·dt²) and the optimistic form (ΔT⁴·dt²). If the difference exceeds 5 years for any SSP, report prominently.


## Parameter Choices

| Parameter | Symbol | Value | Justification |
|-----------|--------|-------|---------------|
| Trend model | — | WLS quadratic (BMA with cubic if needed) | Closed-form, zero hyperparameters, model-averaging handles order uncertainty |
| Satellite start | — | 1993.0 | Start of TOPEX/Poseidon altimetry |
| GIA uncertainty | σ_GIA | 0.15 mm/yr | Half the inter-model GIA spread; rank-1 inflation of full 3×3 |
| WAIS systematic | σ_AIS,sys | 0.10 mm/yr | IMBIE inter-method rate spread for WAIS basin; rank-1 inflation of full 3×3 |
| Residual scale | σ̂² | 1.0 (fixed, with diagnostic) | Trusted observation uncertainties; flag if estimated σ̂² > 2 |
| φ₀ prior | — | N(2.5, 0.5²), φ₀ > 0 | Trend precision advantage at t_end |
| κ_net prior | — | LogNormal(log(15), 0.6) | Broad; data constrain lower bound |
| κ_t prior | — | LogNormal(log(10⁻³), 0.5) | Acceleration-persistence; pure physical bias |
| λ prior | — | LogNormal(log(1), 0.5) | Model structural bias (conservative O(ΔT²)) |
| Overshoot mode | — | True | Conservative: once system leaves calibration domain, trend does not regain credibility |
| n_samples | N_ENSEMBLE | 10,000 | Match the rate-and-state posterior ensemble size |
| MCMC walkers | — | 16 | emcee default for 3-parameter problem |
| MCMC warmup | — | 2000 | Conservative; check R_hat |
| MCMC samples | — | 5000 | Per walker; total 80,000, thinned to ~8,000 effective |
| Smoothing ε | — | 0.05 | Transition over ~2-5 years |
| CV holdout endpoints | — | {2003, 2005, ..., t_end - 5} | 2-year spacing, minimum 5 years held out |


## Prior Predictive Checks

Before running the full pipeline, verify the prior produces physically reasonable weights:

### Median checks
1. 5 yr lead, all SSPs: median w > 0.8
2. 10 yr lead, SSP2-4.5: median w ∈ [0.3, 0.8]
3. 20 yr lead, SSP2-4.5: median w < 0.3
4. 20 yr lead, SSP1-2.6 (ΔT ≈ 0): median w ∈ [0.3, 0.7]

### Spread checks (verify prior is not overly informative)
5. 20 yr lead, SSP1-2.6: 10th-90th percentile of w spans at least [0.1, 0.8]
6. 10 yr lead, SSP5-8.5: 10th-90th percentile of w spans at least [0.05, 0.7]


## Testing Strategy

All tests in a single `test_trend_constraint.py` file.

### Unit tests

1. **Temperature departure:** ΔT = 0 for T ≤ T_cal_max; overshoot mode monotone non-decreasing.
2. **WLS recovery:** Synthetic quadratic + noise → correct (r, r̈) within uncertainty.
3. **GIA/AIS inflation structure:** Rank-1, affects (H, r) cross-covariance, zero in acceleration row/column.
4. **BMA cubic:** Pure quadratic → bma_prob_cubic small. Known cubic → bma_prob_cubic large.
5. **Weight bounds:** w(t) ∈ [0, 1] for extreme parameter values.
6. **Synthesis log-likelihood:** Synthetic CV data with trend better at short leads → posterior φ₀ large and positive.
7. **Mixture quantiles:** 50th percentile between trend and model medians.
8. **Stochastic transition marginals:** Empirical CDF of H_comb at each t matches mixture quantiles (KS test).
9. **WAIS add-back:** H_total = H_non_wais + H_wais exactly.
10. **WAIS decomposition consistency:** Same WAIS subtracted from both agents; EAIS/Peninsula not subtracted.
11. **Sampler agreement:** Simple 2D problem → emcee and dynesty agree within tolerance.

### Integration test

End-to-end test with synthetic data:

1. Generate synthetic GMSL from a known rate-and-state model with known parameters.
2. Generate synthetic WAIS contribution (a known growing signal).
3. Add realistic noise.
4. Fit WLS to the non-WAIS satellite-era data.
5. Fit the rate-and-state model to the full record.
6. Run BPS cross-validation and posterior sampling (with a fast model_refit_fn that uses the known parameters).
7. Generate combined projections.
8. Verify:
   - The mixture predictive matches the trend at short leads (w̄ ≈ 1).
   - The mixture predictive matches the model at long leads (w̄ ≈ 0).
   - The crossover posterior is centered near the expected ΔT given the known physics.
   - Under a "flat temperature" scenario (ΔT = 0), the trend yields to the model within ~25 yr (governed by κ_t).
   - Under a "rapid warming" scenario, the crossover occurs within ~10 years.
   - Individual stochastic-transition trajectories are smooth.
   - The WAIS component passes through unmodified.
   - The mixture quantile CI is wider than the stochastic-transition ensemble CI.

### Regression test

Once the module is working on real data, snapshot the diagnostics table and the BPS posterior summary. Check that future code changes do not alter results beyond the expected MCMC sampling noise.


## Open Questions

1. **Cross-validation cost.** The leave-future-out CV requires refitting the rate-and-state model for each holdout endpoint (~8-10 refits). If full MCMC is too expensive, a Laplace approximation or variational inference can be used for the CV refits only (not for the final projection). The resulting synthesis posterior will be approximate, but since it's used for the weight hyperparameters (not the primary physical parameters), this is acceptable. The sensitivity of the BPS posterior to the CV model approximation should be checked by running one full-MCMC refit and comparing.

2. **Non-Gaussian agent densities.** The current implementation approximates the model agent density as Gaussian in the CV step (for computational convenience). For the final combination, the full empirical distribution is used via the weighted-pool method. If the model posterior predictive is substantially non-Gaussian (heavy tails, skewness), the Gaussian approximation in the CV could bias the synthesis calibration. A diagnostic: compare the CV log-likelihood under Gaussian vs. kernel density estimation of the model predictive. If they differ substantially, use KDE. Note: for the non-WAIS total (WAIS exempted), the Gaussian approximation is likely adequate since the remaining components (including EAIS, Peninsula) are well-behaved and GMST-connected.

3. **WAIS decomposition in the pre-GRACE era.** The WLS trend model is fit to non-WAIS GMSL, which requires subtracting the WAIS contribution. For 1993-2002 (pre-GRACE), the WAIS contribution comes from the reconciled IMBIE basin-level record, which has larger uncertainties. This uncertainty is now propagated via Σ_AIS,sys (rank-1 systematic inflation), but the per-observation uncertainty during 1993-2002 should also be increased to reflect the larger WAIS subtraction uncertainty in that period.

4. **λ estimation from Level 2.** Once the component-level models are operational, λ can be estimated empirically by comparing Level 1 and Level 2 rate predictions at the edge of the calibration domain. This provides a data-informed alternative to the subjective prior.


## Code Budget

Summary of new vs reused code:

| Section | New lines (est.) | Reused from | Notes |
|---|---|---|---|
| 1. Temperature departure | ~15 | — | Pure numpy |
| 2. WLS trend model | ~80 | `polyfit_model_selection()` pattern for BMA | statsmodels WLS + Jacobian |
| 3. Trend projection | ~20 | — | Broadcasting, no loop |
| 4. BPS synthesis | ~250 | `check_convergence()`, emcee pattern from `fit_bayesian_dols()`, `calibrate_dols()` + `project_gmsl_state_ensemble()` for model_refit_fn | Core new code |
| 5. Combination | ~80 | — | Mixture quantiles, stochastic transition |
| 6. Crossover diagnostics | ~50 | — | Root-finding on φ(t) = 0 |
| 7. Forcing-form sensitivity | ~30 | BPSSynthesis (calls itself twice) | Orchestration only |
| 8. Data loading | 0 | `read_imbie_*()`, `read_nasa_gmsl()`, `load_all_gmsl()` | All existing |
| **Total new code** | **~525** | | |
| **Tests** | **~200** | | |

**Previous plan:** 8 new files, 5 test files, ~1200 lines of new code, several functions duplicating existing infrastructure.

**This plan:** 1 new file + 1 test file, ~525 lines of new code, explicit reuse of 6 existing functions/patterns.


## Changes from Previous Plan

This section summarises the substantive changes from the original `near_term_implementation_plan.md`, motivated by the reviews in `near_term_critiques.md`, `near_term_critique2.md`, and `near_term_critique3.md`.

1. **AIS decomposition consistency (Critique 3, §1).** Both agents now subtract **WAIS total** (dynamics + SMB). The previous plan subtracted total AIS from observations but only AIS dynamics from the model, creating an inconsistency where AIS SMB was either double-counted or missing. Only WAIS is exempt from trend suppression; EAIS and Peninsula remain in the trend-constrained total because they have direct or indirect connections to GMST. Recorded as DECISION 1.

2. **GIA inflation extended to full 3×3 covariance (Critique 3, §5).** The GIA systematic now contributes a rank-1 inflation to the full covariance (level, rate, and their cross-covariance), not just the rate variance. At t_end − t_ref ≈ 18 yr, the level inflation is ~2.7 mm, which is non-negligible.

3. **WAIS subtraction systematic inflation added (Critiques 1&2, §1.2).** A second rank-1 systematic inflation σ_AIS,sys = 0.10 mm/yr accounts for temporally correlated IMBIE reconciliation biases in the WAIS basin-level record.

4. **Forcing-form sensitivity redesigned (Critiques 1&2, §1.1).** The previous `forcing_form_sensitivity` function was dimensionally incoherent (tried to substitute κ_net from one parameterization into another with different units). Now requires a full BPS refit under the alternative form. Moved to its own Module 7.

5. **emcee + dynesty cross-check (Critique 3, §8).** The log-mixture likelihood can produce multimodal posteriors. Both samplers must agree. Added `fit_dynesty()` and `check_sampler_agreement()` methods. Recorded as DECISION 3.

6. **BMA for cubic robustness (Critique 3, §10).** The binary significance threshold (5%/1%) is replaced by Bayesian model averaging via BIC. The trend agent's predictive is automatically a weighted mixture of quadratic and cubic predictions.

7. **Ensemble alignment specified (Critiques 1&2, §3.3; Critique 3, §6).** Added `align_ensemble_sizes()` utility. Target size N_ENSEMBLE = 10,000. Three resamplings are independent.

8. **Holdout influence via importance weighting (Critiques 1&2, §3.4).** Added `holdout_influence_diagnostics()` method that uses importance weighting as a pre-screen, flagging holdouts for full MCMC refit only when the importance-weight ESS is low.

9. **Prior predictive spread checks added (Critique 3, §7).** Two spread checks supplement the median checks to verify the prior is not overly concentrated.

10. **Analytic crossover domain restriction (All critiques).** The analytic formula is explicitly restricted to monotone-warming scenarios. T_rate_by_ssp is set to None for overshoot SSPs.

11. **CV T_cal_max recomputation documented (Critiques 1&2, §3.1).** The `build_cv_data` docstring now explicitly documents that T_cal_max must be recomputed for each holdout endpoint using only data available up to t_h.

12. **fix_scale diagnostic added (Critique 3, §10).** The robustness checks now include a diagnostic comparing estimated σ̂² against 1.0, flagging if observation uncertainties appear underestimated.

13. **Independence assumption stated (Critique 3, §2).** The marginal correctness proof's reliance on agent-ψ independence is now explicitly noted in the stochastic transition documentation.

14. **CV identifiability limitations documented (Critique 3, §3).** The diagnostic tables and workflow now explicitly note that high-emissions crossover times are prior-dominated.

15. **Factual error corrected.** The erroneous reference to the "2004 tsunami" has been removed.

16. **Removed Open Question 5 (>2 agents).** Premature design-for-extension is not justified by current needs. Can be revisited if a third agent is proposed.

17. **Code reuse revision (critical evaluation).** Consolidated from 8 new files + 5 test files into 1 new file + 1 test file. Eliminated ~675 lines of duplicated code by:
    - Delegating convergence diagnostics to `check_convergence()` from `bayesian_dols.py`
    - Reusing emcee sampling pattern from `fit_bayesian_dols()`
    - Adapting BIC computation from `polyfit_model_selection()` for BMA cubic check
    - Building `model_refit_fn` from existing `calibrate_dols()` + `project_gmsl_state_ensemble()`
    - Using existing data readers (`read_imbie_*`, `read_nasa_gmsl()`, `load_all_gmsl()`)
    - Replacing the WLSTrendModel class with standalone functions (simpler, sufficient)

18. **Removed MSE_trend diagnostic from workflow.** The previous plan computed `MSE_trend = Var[samples] + κ·ΔT²·dt² + κ_t·dt⁴` using undefined constants `KAPPA_MEDIAN` and `KAPPA_T_MEDIAN`. The BPS synthesis does not use this quantity (it evaluates agent densities directly). The separate κ/κ_t symbols created naming confusion with the BPS synthesis parameters (κ_net, κ_t). Removed from the workflow. Standalone MSE diagnostics can be computed post-hoc from ensemble variance if needed.

19. **model_refit_fn specified (Critiques 1&2, §3.5).** The previous plan left this as "caller provides" with no implementation path. Now specified as `make_model_refit_fn()` factory using existing `calibrate_dols()` + `project_gmsl_state_ensemble()`, with explicit speed/accuracy tradeoff documented (n_samples=500, frequentist for CV speed).

20. **File structure aligned with project.** Changed from non-existent `gmslr_framework/diagnostics/` to existing `slr_forecast/notebooks/` directory where all analysis code lives.
