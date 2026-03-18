# Implementation Plan: Near-Term Trend Constraint via Bayesian Predictive Synthesis

## Overview

This document specifies the implementation of a near-term trend constraint module that combines a non-parametric trend forecast with the rate-and-state model projection using Bayesian Predictive Synthesis (BPS). The synthesis hyperparameters are calibrated from leave-future-out cross-validation on the satellite-era record. Their posterior uncertainty is propagated through the combined predictive, producing credible intervals that account for uncertainty in both the individual forecasts and the combination itself.

**Read the companion LaTeX document `near_term_trend_constraint.tex` for the full mathematical derivation before implementing.**

## Key Design Decisions

1. **Bayesian Predictive Synthesis, not MSE-optimal weighting.** The combination uses BPS (McAlinn & West, 2019) with a structured latent weight process calibrated from cross-validation. This replaces the MSE-optimal point-forecast weighting, which produced systematically overconfident predictive intervals: the linear-combination ensemble spread was w²·Var_trend + (1-w)²·Var_model, strictly narrower than either individual ensemble for 0 < w < 1. BPS produces a properly calibrated mixture predictive.

2. **Synthesis hyperparameters calibrated, not assumed.** The previous approach pegged κ to the model's own ∂r/∂T — a circular estimate where model overconfidence in its rate-temperature sensitivity accelerated its own dominance. The BPS framework calibrates the synthesis hyperparameters via leave-future-out cross-validation, replacing the assumed κ with a data-informed posterior. The posterior reports which aspects of the combination are identified by data (φ₀, κ_t) and which remain prior-dominated (κ_net at large ΔT).

3. **No double-counting of acceleration-estimation variance.** The previous `κ_t^eff = κ_t + ¼σ²_rdot` construction double-counted: the ¼σ²_rdot·dt⁴ term was already captured by Var[H_trend samples] (computed via Monte Carlo from the WLS posterior), and then added again through κ_t^eff. The corrected MSE uses Var[samples] + κ·ΔT²·dt² + κ_t·dt⁴, where κ_t is the pure acceleration-persistence bias. The κ_t^eff concept is eliminated.

4. **AIS dynamic component exempt from trend suppression.** The BPS synthesis operates on total GMSL minus the AIS dynamic contribution. AIS dynamics are added back to the combined projection without passing through the trend-model weighting. This prevents the trend constraint from vetoing near-term WAIS instability scenarios — the exact scenarios the Arête framework needs for counterfactual intervention queries.

5. **Mixture predictive for uncertainty reporting; stochastic-transition trajectories for coherence.** The combined predictive is a mixture distribution (correctly calibrated marginals). Temporally coherent trajectories for rate-of-change analysis and counterfactual queries use sample-specific synthesis weights with a smooth stochastic transition.

6. **Shared-data covariance handled by construction.** The previous "empirical covariance diagnostic" was vacuous: it computed sample covariance between independently drawn posterior samples, which is zero by construction regardless of the true shared-data dependence. In the BPS framework, the shared-data covariance is handled implicitly through the cross-validation calibration: both agents are evaluated against the same held-out observations, and the synthesis log-likelihood captures whatever covariance structure exists in their predictions without requiring an explicit covariance computation.

7. **GIA uncertainty propagated.** The WLS rate estimate depends on the GIA correction. An additional variance σ²_GIA = (0.15 mm/yr)² is added to Var(r̂) in the propagated 3×3 covariance, representing half the inter-model GIA spread.

8. **λ functional form documented with sensitivity diagnostic.** The default model structural bias uses O(ΔT²·dt²) scaling (conservative: assumes first-order coupling may be wrong). A sensitivity diagnostic compares against O(ΔT⁴·dt²) scaling (optimistic: assumes first-order coupling is correct). The difference is 5-10 years in crossover time under SSP3-7.0.

9. **Module placement under `diagnostics/`.** The BPS synthesis is a post-processing module, not a physical component. Placing it under `components/` would create confusion about whether it participates in the budget constraint (it does not, in the current architecture).


## File Structure

All new code goes under the existing project layout:

```
gmslr_framework/
├── diagnostics/
│   └── trend_combination/
│       ├── __init__.py
│       ├── wls_trend_model.py          # WLS quadratic fit to satellite-era GMSL
│       ├── trend_projection.py         # Trend projection and MSE computation
│       ├── bps_synthesis.py            # BPS weight process, calibration, posterior sampling
│       ├── combination.py              # Mixture predictive, stochastic-transition trajectories
│       ├── crossover_diagnostics.py    # Crossover computation and reporting
│       └── tests/
│           ├── test_wls_trend_model.py
│           ├── test_trend_projection.py
│           ├── test_bps_synthesis.py
│           ├── test_combination.py
│           └── test_crossover_diagnostics.py
└── utils/
    ├── temperature_departure.py        # ΔT(t) computation from SSP trajectories
    └── tests/
        └── test_temperature_departure.py
```


## Module 1: `utils/temperature_departure.py`

Purpose: Compute the temperature departure ΔT(t) from the calibration domain for arbitrary SSP trajectories.

### Functions

```python
def compute_T_cal_max(T_cal: np.ndarray) -> float:
    """
    Maximum GMST in the calibration window.

    Parameters
    ----------
    T_cal : np.ndarray, shape (n_cal,)
        Annual GMST anomalies during the calibration period.

    Returns
    -------
    T_max_cal : float
        Maximum observed GMST (°C relative to baseline).
    """


def compute_delta_T(
    T_scenario: np.ndarray,
    T_max_cal: float,
    overshoot_mode: bool = True,
) -> np.ndarray:
    """
    Temperature departure from the calibration domain.

    Parameters
    ----------
    T_scenario : np.ndarray, shape (n_proj,)
        GMST trajectory for a given SSP, covering [t_end+1, ..., t_end+n_proj].
    T_max_cal : float
        Maximum GMST in the calibration window.
    overshoot_mode : bool
        If True, use cumulative maximum exceedance:
            ΔT(t) = max_{s in [t_end, t]} max(0, T(s) - T_max_cal)
        This prevents the trend from regaining credibility after temperature
        returns below T_max_cal (relevant for overshoot scenarios).
        If False, use instantaneous exceedance:
            ΔT(t) = max(0, T(t) - T_max_cal)

    Returns
    -------
    delta_T : np.ndarray, shape (n_proj,)
        Temperature departure at each projection time step. Units: °C.
    """
```

### Implementation notes

- `overshoot_mode=True` is the default and recommended setting.
- Both functions are pure numpy.
- `T_scenario` and `T_cal` must use the same baseline/reference period. The caller is responsible for ensuring this.

### Tests

1. **Zero departure:** If `T_scenario ≤ T_max_cal` everywhere, `delta_T` is identically zero.
2. **Monotone increasing:** If `T_scenario` is monotonically increasing, both modes give `max(0, T_scenario - T_max_cal)`.
3. **Overshoot:** If `T_scenario = [1.5, 1.8, 1.6, 1.4]` and `T_max_cal = 1.2`, then:
   - Instantaneous: `[0.3, 0.6, 0.4, 0.2]`
   - Overshoot mode: `[0.3, 0.6, 0.6, 0.6]`


## Module 2: `diagnostics/trend_combination/wls_trend_model.py`

Purpose: Fit a weighted least-squares quadratic to satellite-era GMSL altimetry and extract the posterior on the level, rate, and acceleration at the end of the calibration window. Includes GIA uncertainty inflation.

### Dependencies

- `statsmodels` (WLS regression)
- `numpy`, `scipy`

### Class

```python
class WLSTrendModel:
    """
    Weighted least-squares quadratic model for satellite-era GMSL
    trend estimation.

    Fits H(t) = α + β·(t - t_ref) + γ·(t - t_ref)² to annual-mean
    GMSL from the satellite era (1993–t_end), then extracts the
    posterior on the rate (dH/dt) and acceleration (d²H/dt²) at t_end.
    """

    def __init__(
        self,
        satellite_start: float = 1993.0,
        sigma_gia: float = 0.15,
        fix_scale: bool = True,
    ):
        """
        Parameters
        ----------
        satellite_start : float
            Start year of the satellite altimetry era.
        sigma_gia : float
            GIA inter-model rate uncertainty (mm/yr). Added in quadrature
            to Var(r_hat) in the propagated covariance. Default 0.15 mm/yr
            (half the inter-model GIA spread).
        fix_scale : bool
            If True, fix the residual scale at σ² = 1.0 (trusted
            observation uncertainties). If False, estimate σ² from
            weighted residuals.
        """

    def fit(
        self,
        years: np.ndarray,
        gmsl: np.ndarray,
        gmsl_uncertainty: np.ndarray | None = None,
    ) -> "WLSTrendModel":
        """
        Fit the WLS quadratic to satellite-era GMSL data.

        Parameters
        ----------
        years : np.ndarray, shape (n_obs,)
        gmsl : np.ndarray, shape (n_obs,)
            Observed GMSL (mm).
        gmsl_uncertainty : np.ndarray, shape (n_obs,) or None
            Per-observation 1σ uncertainty (mm). If None, OLS is used.
        """
        # Subset to satellite era
        # Center time for numerical stability: t_ref = midpoint
        # Design matrix: [1, dt, dt²]
        # Fit via statsmodels WLS
        # If fix_scale: use cov_type='fixed scale', cov_kwds={'scale': 1.0}
        # Else: use default residual variance estimation

    def predict_at_t_end(self, t_end: float) -> dict:
        """
        Extract posterior at t_end for the level, rate, and acceleration.

        The transformation from (α, β, γ) to (H, r, rdot) at t_end is:
            H = α + β·Δ + γ·Δ²     where Δ = t_end - t_ref
            r = β + 2γ·Δ
            rdot = 2γ

        The 3×3 covariance is J @ Σ_params @ J.T, where J is the Jacobian.
        GIA uncertainty is added to the (1,1) element (Var(r)).

        Returns
        -------
        result : dict with keys:
            "H_mean", "H_std" : float
            "r_mean", "r_std" : float (r_std includes GIA inflation)
            "rdot_mean", "rdot_std" : float
            "cov_r_rdot" : float
            "cov_3x3" : np.ndarray, shape (3, 3) — includes GIA inflation
        """

    def sample_derivatives(
        self, t_end: float, n_samples: int = 10000, seed: int | None = None,
    ) -> dict:
        """
        Draw joint posterior samples of (H, r, rdot) at t_end.

        Samples from the 3D multivariate normal defined by the WLS
        posterior (with GIA inflation).

        Returns
        -------
        samples : dict with keys "H", "r", "rdot", each shape (n_samples,)
        """

    def robustness_checks(
        self,
        years: np.ndarray,
        gmsl: np.ndarray,
        gmsl_uncertainty: np.ndarray | None,
        t_end: float,
        short_window_years: int = 20,
    ) -> dict:
        """
        Run cubic test and window-sensitivity robustness checks.

        Returns
        -------
        checks : dict with keys:
            "cubic_coeff" : float — estimated cubic coefficient δ
            "cubic_coeff_se" : float — standard error of δ
            "cubic_significant" : bool — True if |δ/se(δ)| > 2
            "cubic_rdot_at_tend" : float — 2γ + 6δ·(t_end - t_ref)
            "short_window_r" : float
            "short_window_rdot" : float
            "short_window_r_diff_sigma" : float
            "short_window_rdot_diff_sigma" : float
            "window_sensitive" : bool — True if either > 1σ
        """
```

### Tests

1. **Sanity on synthetic data:** Generate GMSL = 50·t + 3·t² + noise, verify `r_mean ≈ 50 + 6·t_end` and `rdot_mean ≈ 6.0` within WLS uncertainty.
2. **Covariance correctness:** On synthetic data with known noise, verify 3×3 covariance matches empirical bootstrap covariance.
3. **GIA inflation:** Verify that `r_std` with `sigma_gia > 0` exceeds `r_std` with `sigma_gia = 0` by the expected amount. Verify that `rdot_std` is unchanged.
4. **Scale fixing:** With `fix_scale=True` and accurate observation uncertainties, parameter covariance should be `(X^T W X)^{-1}`. With `fix_scale=False`, it should be `σ̂²(X^T W X)^{-1}`.
5. **Derivative variance ordering:** `r_std < rdot_std`.
6. **Exact recovery:** With zero noise, the WLS recovers exact coefficients and covariance is zero (within float tolerance).
7. **Statsmodels agreement:** Jacobian-propagated covariance matches reparametrised fit on `[1, t-t_end, (t-t_end)²]`.
8. **Cubic robustness:** On pure quadratic data, `cubic_significant == False`. On data with known cubic, detection works.
9. **Window sensitivity:** On data with stationary acceleration, `window_sensitive == False`.


## Module 3: `diagnostics/trend_combination/trend_projection.py`

Purpose: Generate trend projection ensembles and compute their prediction MSE.

### Functions

```python
def trend_projection_ensemble(
    t_proj: np.ndarray,
    t_end: float,
    derivative_samples: dict,
    delta_T: np.ndarray,
    kappa: float | np.ndarray,
    kappa_t: float | np.ndarray,
) -> dict:
    """
    Generate trend projection ensemble and compute total prediction MSE.

    The trend projection for each sample s is:
        H_trend^(s)(t) = H^(s) + r^(s)·(t - t_end) + 0.5·rdot^(s)·(t - t_end)²

    The total prediction MSE at each time step is:
        MSE_trend(t) = Var[H_trend samples](t) + κ·ΔT(t)²·(t - t_end)²
                       + κ_t·(t - t_end)⁴

    NOTE: No κ_t^eff construction. The acceleration-estimation variance is
    already captured by Var[H_trend samples]. Adding κ_t^eff would double-count.
    κ_t here is the pure acceleration-persistence bias coefficient.

    Parameters
    ----------
    t_proj : np.ndarray, shape (n_proj,)
    t_end : float
    derivative_samples : dict — output of WLSTrendModel.sample_derivatives()
    delta_T : np.ndarray, shape (n_proj,)
    kappa : float or np.ndarray, shape (n_samples,)
        Forcing-departure squared-bias coefficient.
    kappa_t : float or np.ndarray, shape (n_samples,)
        Acceleration-persistence squared-bias coefficient (pure physical
        bias, NOT κ_t^eff).

    Returns
    -------
    result : dict with keys:
        "H_ensemble" : np.ndarray, shape (n_samples, n_proj)
        "MSE_trend" : np.ndarray, shape (n_proj,)
    """
```

### Implementation notes

- The squared-bias terms are added to the sample variance to form the total MSE. They are NOT added to individual samples.
- If `kappa` or `kappa_t` is an array, marginalise: compute the expected MSE over the samples.
- The sample variance `Var[H_trend samples]` is computed along the sample axis at each time step. This captures all estimation-uncertainty contributions exactly (σ_r², ¼σ_rdot², and all cross-terms), eliminating the need for the κ_t^eff concept.


## Module 4: `diagnostics/trend_combination/bps_synthesis.py`

Purpose: Implement the Bayesian Predictive Synthesis framework: weight process, leave-future-out cross-validation, posterior sampling.

### Dependencies

- `numpy`, `scipy`
- `emcee` (MCMC sampler, lightweight, pure Python)

### Functions and Classes

```python
from dataclasses import dataclass

@dataclass
class SynthesisHyperparameters:
    """Synthesis hyperparameters ψ = (φ₀, κ_net, κ_t)."""
    phi_0: float      # Initial logit-weight (positive: trend favored)
    kappa_net: float   # Net forcing-departure degradation (logit-space)
    kappa_t: float     # Acceleration-persistence degradation (logit-space)


def compute_logit_weight(
    dt: np.ndarray,
    delta_T: np.ndarray,
    psi: SynthesisHyperparameters,
) -> np.ndarray:
    """
    Compute the logit of the synthesis weight φ(t; ψ).

    φ(t) = φ₀ - κ_net · ΔT(t)² · (t - t_end)² - κ_t · (t - t_end)⁴

    Parameters
    ----------
    dt : np.ndarray, shape (n_proj,)
        Lead times: t - t_end.
    delta_T : np.ndarray, shape (n_proj,)
        Temperature departure at each projection time.
    psi : SynthesisHyperparameters

    Returns
    -------
    phi : np.ndarray, shape (n_proj,)
    """


def compute_synthesis_weight(
    dt: np.ndarray,
    delta_T: np.ndarray,
    psi: SynthesisHyperparameters,
) -> np.ndarray:
    """
    Compute w(t; ψ) = σ(φ(t; ψ)).

    Returns
    -------
    w : np.ndarray, shape (n_proj,)
        Synthesis weight on trend agent, in [0, 1].
    """


def synthesis_log_likelihood(
    psi: SynthesisHyperparameters,
    cv_data: list[dict],
) -> float:
    """
    Compute the synthesis log-likelihood from cross-validation data.

    ℓ(ψ) = Σ_{j, Δt} log[w(Δt, ΔT; ψ)·h_trend(H_obs) + (1-w)·h_model(H_obs)]

    Parameters
    ----------
    psi : SynthesisHyperparameters
    cv_data : list of dict
        Each dict corresponds to one holdout endpoint t_h and contains:
        - "dt" : np.ndarray — lead times
        - "delta_T" : np.ndarray — temperature departures
        - "H_obs" : np.ndarray — observed GMSL at held-out times
        - "trend_log_density" : np.ndarray — log h_trend(H_obs) at each lead
        - "model_log_density" : np.ndarray — log h_model(H_obs) at each lead

    Returns
    -------
    loglik : float
    """


def synthesis_log_prior(psi: SynthesisHyperparameters) -> float:
    """
    Compute the log prior on ψ.

    φ₀ ~ TruncNormal(2.5, 0.5², lower=0)
    κ_net ~ LogNormal(log(15), 0.6)
    κ_t ~ LogNormal(log(1e-3), 0.5)

    Returns
    -------
    logprior : float (-inf if outside support)
    """


def synthesis_log_posterior(
    psi: SynthesisHyperparameters,
    cv_data: list[dict],
) -> float:
    """Log posterior: log_prior + log_likelihood."""


class BPSSynthesis:
    """
    Bayesian Predictive Synthesis for trend-model combination.

    Orchestrates the full BPS workflow: cross-validation data
    construction, posterior sampling, and weight computation.
    """

    def __init__(
        self,
        n_walkers: int = 16,
        n_warmup: int = 2000,
        n_samples: int = 5000,
        seed: int | None = None,
    ):
        """
        Parameters
        ----------
        n_walkers : int
            Number of emcee walkers.
        n_warmup : int
            Number of warm-up (burn-in) steps per walker.
        n_samples : int
            Number of sampling steps per walker (after warm-up).
        seed : int or None
        """

    def build_cv_data(
        self,
        years: np.ndarray,
        gmsl: np.ndarray,
        gmsl_uncertainty: np.ndarray | None,
        T_obs: np.ndarray,
        model_refit_fn: callable,
        holdout_endpoints: np.ndarray | None = None,
        min_holdout: int = 5,
    ) -> list[dict]:
        """
        Construct leave-future-out cross-validation data.

        For each holdout endpoint t_h:
        1. Fit WLS to [satellite_start, t_h]
        2. Call model_refit_fn(t_h) to get model posterior predictive
        3. Evaluate trend and model densities at held-out observations

        Parameters
        ----------
        years, gmsl, gmsl_uncertainty : observation arrays
        T_obs : np.ndarray — observed GMST trajectory
        model_refit_fn : callable
            Function that takes t_h (holdout endpoint) and returns a dict:
            {
                "H_predictive_mean": np.ndarray,  # shape (n_holdout,)
                "H_predictive_std": np.ndarray,    # shape (n_holdout,)
            }
            representing the model's Gaussian predictive at the held-out times.
            This is the expensive step (requires re-running the rate-and-state
            calibration for each holdout).
        holdout_endpoints : np.ndarray or None
            If None, use default: t_h ∈ {2003, 2005, ..., t_end - 5}.
        min_holdout : int
            Minimum number of held-out years required.

        Returns
        -------
        cv_data : list[dict]
            Cross-validation data suitable for synthesis_log_likelihood.
        """

    def fit(self, cv_data: list[dict]) -> "BPSSynthesis":
        """
        Sample the posterior on ψ via emcee.

        Initialises walkers near the prior mode, runs warm-up,
        then samples. Stores posterior samples.

        Returns
        -------
        self
        """

    def get_posterior_samples(self) -> np.ndarray:
        """
        Returns
        -------
        samples : np.ndarray, shape (n_total_samples, 3)
            Columns: [φ₀, κ_net, κ_t]
        """

    def posterior_mean_weight(
        self,
        dt: np.ndarray,
        delta_T: np.ndarray,
    ) -> np.ndarray:
        """
        Compute posterior mean weight w̄(t) = E_ψ[σ(φ(t; ψ))].

        Averages σ(φ(t; ψ^(s))) over posterior samples.

        Returns
        -------
        w_bar : np.ndarray, shape (n_proj,)
        """

    def posterior_weight_ensemble(
        self,
        dt: np.ndarray,
        delta_T: np.ndarray,
        n_samples: int | None = None,
    ) -> np.ndarray:
        """
        Return weight trajectories for each posterior sample.

        Parameters
        ----------
        n_samples : int or None
            If None, use all posterior samples. If specified, subsample.

        Returns
        -------
        w_ensemble : np.ndarray, shape (n_psi_samples, n_proj)
        """

    def convergence_diagnostics(self) -> dict:
        """
        MCMC convergence diagnostics.

        Returns
        -------
        diagnostics : dict with keys:
            "R_hat" : np.ndarray, shape (3,) — Gelman-Rubin for each param
            "ESS" : np.ndarray, shape (3,) — effective sample size
            "acceptance_fraction" : float
        """
```

### Implementation notes

- The cross-validation requires refitting the rate-and-state model for each holdout endpoint. This is computationally expensive. `model_refit_fn` is a callback that the caller provides, encapsulating whatever calibration machinery is appropriate (full MCMC, or a fast approximation such as Laplace or variational inference for the cross-validation only).
- For the trend agent density at each holdout, use the Gaussian predictive from the WLS posterior (analytically computed).
- For the model agent density, approximate as Gaussian with mean and std from the model posterior predictive ensemble. This is adequate for the synthesis calibration; the full empirical distribution is used in the final combination.
- The `emcee` sampler operates in a transformed space: `log(κ_net)` and `log(κ_t)` are sampled (ensuring positivity), with `φ₀` sampled directly (with a reflecting boundary at 0).
- The log-likelihood involves `log(w·h_1 + (1-w)·h_2)`. Use the log-sum-exp trick to avoid numerical underflow when one agent density is much smaller than the other.


## Module 5: `diagnostics/trend_combination/combination.py`

Purpose: Generate the combined predictive distribution (mixture quantiles) and the temporally coherent stochastic-transition trajectory ensemble.

### Functions

```python
def compute_mixture_quantiles(
    H_trend_ensemble: np.ndarray,
    H_model_ensemble: np.ndarray,
    w_bar: np.ndarray,
    quantiles: np.ndarray = np.array([0.05, 0.17, 0.50, 0.83, 0.95]),
) -> np.ndarray:
    """
    Compute quantiles of the BPS mixture predictive.

    The mixture distribution at each t is:
        p_comb(H(t)) = w̄(t)·h_trend(H(t)) + (1 - w̄(t))·h_model(H(t))

    Implementation: for each time step, pool trend and model samples,
    weight each trend sample by w̄(t)/n_trend and each model sample by
    (1-w̄(t))/n_model, and compute weighted quantiles.

    Parameters
    ----------
    H_trend_ensemble : np.ndarray, shape (n_samples, n_proj)
    H_model_ensemble : np.ndarray, shape (n_samples, n_proj)
    w_bar : np.ndarray, shape (n_proj,) — posterior mean weight
    quantiles : np.ndarray

    Returns
    -------
    Q : np.ndarray, shape (n_quantiles, n_proj)
    """


def generate_stochastic_transition_ensemble(
    H_trend_ensemble: np.ndarray,
    H_model_ensemble: np.ndarray,
    w_ensemble: np.ndarray,
    epsilon: float = 0.05,
    seed: int | None = None,
) -> dict:
    """
    Generate temporally coherent trajectory ensemble via stochastic transition.

    For each MC sample s:
    1. Use w^(s)(t) from the posterior weight ensemble (one ψ per sample).
    2. Draw u^(s) ~ Uniform(0, 1).
    3. Compute smooth transition indicator:
       ω^(s)(t) = σ((w^(s)(t) - u^(s)) / ε)
    4. Combine:
       H_comb^(s)(t) = ω^(s)(t)·H_trend^(s)(t) + (1-ω^(s)(t))·H_model^(s)(t)

    Each trajectory transitions smoothly from trend to model at a sample-
    specific time. The marginal at each t integrates to the mixture predictive.

    Parameters
    ----------
    H_trend_ensemble : np.ndarray, shape (n_samples, n_proj)
    H_model_ensemble : np.ndarray, shape (n_samples, n_proj)
    w_ensemble : np.ndarray, shape (n_samples, n_proj)
        Per-sample weight trajectories from BPSSynthesis.posterior_weight_ensemble().
        Must have n_samples matching the agent ensembles.
    epsilon : float
        Sigmoid smoothing parameter. Controls transition sharpness.
        Recommended: 0.05 (transition over ~2-5 years).
    seed : int or None

    Returns
    -------
    result : dict with keys:
        "H_combined" : np.ndarray, shape (n_samples, n_proj)
            Stochastic-transition trajectory ensemble.
        "u_draws" : np.ndarray, shape (n_samples,)
            Uniform draws (for reproducibility diagnostics).
        "transition_times" : np.ndarray, shape (n_samples,)
            Approximate transition time for each trajectory (where ω ≈ 0.5).
    """


def add_ais_component(
    H_comb_non_ais: np.ndarray,
    H_ais_dyn: np.ndarray,
) -> np.ndarray:
    """
    Add AIS dynamic component back to the combined non-AIS projection.

    H_comb_total^(s)(t) = H_comb_non_ais^(s)(t) + H_ais_dyn^(s)(t)

    Parameters
    ----------
    H_comb_non_ais : np.ndarray, shape (n_samples, n_proj)
    H_ais_dyn : np.ndarray, shape (n_samples, n_proj)

    Returns
    -------
    H_comb_total : np.ndarray, shape (n_samples, n_proj)
    """


def rate_smoothness_diagnostic(
    H_combined: np.ndarray,
    t_proj: np.ndarray,
) -> dict:
    """
    Verify temporal coherence of the combined trajectory ensemble.

    Computes rates by finite differences and checks for discontinuities.

    Returns
    -------
    diagnostic : dict with keys:
        "max_rate_jump" : float — maximum |Δr| across all samples and times
        "mean_rate_jump" : float
        "fraction_exceeding_threshold" : float — fraction of samples with
            any rate jump exceeding 1 mm/yr²
    """
```

### Implementation notes

**Why stochastic transition, not fixed-weight linear combination.** The fixed-weight linear combination `w(t)·H_trend^(s) + (1-w(t))·H_model^(s)` produces an ensemble whose spread at each t is `w²·Var_trend + (1-w)²·Var_model` — always narrower than either individual ensemble for 0 < w < 1. This is the overconfidence problem identified in the critical evaluation. The stochastic-transition approach distributes trajectories between trend and model regimes, producing spread that includes the inter-model variance. The marginals at each t integrate to the mixture predictive.

**Matching ensemble sizes.** The trend and model ensembles must have the same `n_samples`. If the model ensemble has a different size, resample with replacement before calling. Similarly, the weight ensemble from `BPSSynthesis.posterior_weight_ensemble()` must be resampled to match `n_samples` if the MCMC produced a different number of posterior draws.

**Epsilon sensitivity.** The diagnostic should report rate smoothness for `ε ∈ {0.02, 0.05, 0.10}`. If rate discontinuities are present at `ε = 0.05`, increase to 0.10.


## Module 6: `diagnostics/trend_combination/crossover_diagnostics.py`

Purpose: Compute crossover diagnostics from the posterior on synthesis hyperparameters.

### Functions

```python
def compute_crossover_diagnostics(
    bps: "BPSSynthesis",
    t_proj: np.ndarray,
    t_end: float,
    delta_T_by_ssp: dict[str, np.ndarray],
    T_rate_by_ssp: dict[str, float] | None = None,
) -> dict:
    """
    Compute crossover diagnostics from the BPS posterior.

    For each SSP and each posterior sample ψ^(s), the crossover t*^(s) is
    computed numerically by root-finding on φ(t; ψ^(s)) = 0.

    Parameters
    ----------
    bps : BPSSynthesis — fitted BPS object with posterior samples
    t_proj : np.ndarray, shape (n_proj,)
    t_end : float
    delta_T_by_ssp : dict mapping SSP name to ΔT array
    T_rate_by_ssp : dict mapping SSP name to dT/dt (for analytic approx)

    Returns
    -------
    diagnostics : dict with keys:
        "crossover_posterior" : dict[str, np.ndarray]
            For each SSP: posterior samples of t*.
        "crossover_median" : dict[str, float]
            Posterior median t* for each SSP.
        "crossover_90CI" : dict[str, tuple[float, float]]
            5th-95th percentile of t* posterior.
        "w_bar_at_2050" : dict[str, float]
        "w_bar_at_2100" : dict[str, float]
        "w_bar_timeseries" : dict[str, np.ndarray]
        "crossover_driver" : dict[str, str]
            "forcing_departure" or "acceleration_persistence" or "none"
        "analytic_DT_star" : dict[str, float | None]
    """
```


## Module 7: Data Loading

### Satellite-era non-AIS GMSL

The WLS trend model is fit to satellite-era **non-AIS** GMSL, estimated as:
```
GMSL_non_AIS(t) = GMSL_total(t) - GMSL_AIS(t)
```

where `GMSL_AIS(t)` is the AIS mass balance contribution converted to GMSL-equivalent, from the reconciled IMBIE record (1992-present) and GRACE/GraMBIE (2002-present).

**Consistency requirement:** Use the same GMSL product for the WLS fit as for the rate-and-state calibration, subsetted to the satellite era.

### SSP temperature trajectories

Already available from Level 0. The same `T_SSP(t)` arrays used for the rate-and-state projection are passed to `compute_delta_T`.


## Execution Workflow

```python
# ===========================================================
# Step 0: Prerequisites (already done)
# - Rate-and-state model calibrated, posterior samples available
# - Projection ensemble H_model[s, t] generated for each SSP
# - AIS dynamic component H_ais_dyn[s, t] separated
# ===========================================================

# Compute non-AIS model ensemble
H_model_non_ais = {ssp: H_model[ssp] - H_ais_dyn[ssp] for ssp in ssps}

# ===========================================================
# Step 1: Calibration-domain temperature
# ===========================================================
T_max_cal = compute_T_cal_max(T_obs_calibration)

# ===========================================================
# Step 2: Fit WLS quadratic trend model to satellite-era non-AIS GMSL
# ===========================================================
wls = WLSTrendModel(satellite_start=1993.0, sigma_gia=0.15, fix_scale=True)
wls.fit(years_obs, gmsl_non_ais_obs, gmsl_uncertainty_obs)
robustness = wls.robustness_checks(
    years_obs, gmsl_non_ais_obs, gmsl_uncertainty_obs, t_end
)
derivatives = wls.sample_derivatives(t_end=t_end, n_samples=N_SAMPLES)

# ===========================================================
# Step 3: BPS cross-validation and posterior sampling
# ===========================================================
bps = BPSSynthesis(n_walkers=16, n_warmup=2000, n_samples=5000, seed=42)

# Build cross-validation data (expensive: refits model for each holdout)
cv_data = bps.build_cv_data(
    years=years_obs,
    gmsl=gmsl_non_ais_obs,
    gmsl_uncertainty=gmsl_uncertainty_obs,
    T_obs=T_obs,
    model_refit_fn=model_refit_fn,  # caller provides
)

# Sample posterior on ψ
bps.fit(cv_data)
convergence = bps.convergence_diagnostics()
assert all(convergence["R_hat"] < 1.05), "MCMC did not converge"

# ===========================================================
# Step 4: For each SSP
# ===========================================================
results = {}
for ssp in ssps:
    # 4a: Temperature departure
    delta_T = compute_delta_T(T_scenario[ssp], T_max_cal, overshoot_mode=True)
    dt = t_proj - t_end

    # 4b: Trend projection ensemble and MSE
    trend_result = trend_projection_ensemble(
        t_proj, t_end, derivatives, delta_T,
        kappa=KAPPA_MEDIAN, kappa_t=KAPPA_T_MEDIAN,
    )

    # 4c: Posterior mean weight
    w_bar = bps.posterior_mean_weight(dt, delta_T)

    # 4d: Mixture predictive quantiles (PRIMARY uncertainty reporting)
    Q_mixture = compute_mixture_quantiles(
        trend_result["H_ensemble"],
        H_model_non_ais[ssp],
        w_bar,
    )

    # 4e: Per-sample weight trajectories (for coherent ensemble)
    w_ensemble = bps.posterior_weight_ensemble(dt, delta_T, n_samples=N_SAMPLES)

    # 4f: Stochastic-transition trajectory ensemble (for rate analysis)
    transition_result = generate_stochastic_transition_ensemble(
        trend_result["H_ensemble"],
        H_model_non_ais[ssp],
        w_ensemble,
        epsilon=0.05,
    )

    # 4g: Add AIS component back
    H_comb_total = add_ais_component(
        transition_result["H_combined"],
        H_ais_dyn[ssp],
    )

    # 4h: Rate smoothness diagnostic
    smoothness = rate_smoothness_diagnostic(H_comb_total, t_proj)

    # 4i: Store results
    results[ssp] = {
        "H_combined_non_ais": transition_result["H_combined"],
        "H_combined_total": H_comb_total,
        "Q_mixture_non_ais": Q_mixture,
        "w_bar": w_bar,
        "transition_times": transition_result["transition_times"],
        "smoothness": smoothness,
    }

# ===========================================================
# Step 5: Crossover diagnostics
# ===========================================================
delta_T_by_ssp = {
    ssp: compute_delta_T(T_scenario[ssp], T_max_cal) for ssp in ssps
}
crossover = compute_crossover_diagnostics(bps, t_proj, t_end, delta_T_by_ssp)
```


## Diagnostic Outputs

### Table: Crossover diagnostics

| SSP | t* median (yr) | t* 90% CI | ΔT* median (°C) | w̄(2050) | w̄(2100) | Driver |
|-----|----------------|-----------|------------------|----------|----------|--------|
| SSP1-2.6 | ... | [..., ...] | ... | ... | ... | accel_persist |
| SSP2-4.5 | ... | [..., ...] | ... | ... | ... | forcing_depart |
| SSP3-7.0 | ... | [..., ...] | ... | ... | ... | forcing_depart |
| SSP5-8.5 | ... | [..., ...] | ... | ... | ... | forcing_depart |

Note: The crossover now has a posterior distribution, not a point estimate. The 90% CI width reports the irreducible uncertainty in the combination.

### Table: BPS posterior summary

| Parameter | Prior median | Posterior median | Posterior 90% CI | Data-informed? |
|-----------|-------------|-----------------|------------------|----------------|
| φ₀ | 2.5 | ... | [..., ...] | Yes (well identified) |
| κ_net | 15 | ... | [..., ...] | Weakly (small ΔT only) |
| κ_t | 10⁻³ | ... | [..., ...] | Moderately |

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

### Figure 7: λ sensitivity

For each SSP, plot the posterior median crossover time under the default λ form (ΔT²·dt²) and the optimistic form (ΔT⁴·dt²). If the difference exceeds 5 years for any SSP, report prominently.


## Parameter Choices

| Parameter | Symbol | Value | Justification |
|-----------|--------|-------|---------------|
| Trend model | — | WLS quadratic | Closed-form, zero hyperparameters, analytically transparent |
| Satellite start | — | 1993.0 | Start of TOPEX/Poseidon altimetry |
| GIA uncertainty | σ_GIA | 0.15 mm/yr | Half the inter-model GIA spread |
| Residual scale | σ̂² | 1.0 (fixed) | Trusted observation uncertainties |
| φ₀ prior | — | N(2.5, 0.5²), φ₀ > 0 | Trend precision advantage at t_end |
| κ_net prior | — | LogNormal(log(15), 0.6) | Broad; data constrain lower bound |
| κ_t prior | — | LogNormal(log(10⁻³), 0.5) | Acceleration-persistence; pure physical bias |
| λ prior | — | LogNormal(log(1), 0.5) | Model structural bias (conservative O(ΔT²)) |
| Overshoot mode | — | True | Conservative: once system leaves calibration domain, trend does not regain credibility |
| n_samples | — | 10000 | Match the rate-and-state posterior ensemble size |
| MCMC walkers | — | 16 | emcee default for 3-parameter problem |
| MCMC warmup | — | 2000 | Conservative; check R_hat |
| MCMC samples | — | 5000 | Per walker; total 80,000, thinned to ~8,000 effective |
| Smoothing ε | — | 0.05 | Transition over ~2-5 years |
| CV holdout endpoints | — | {2003, 2005, ..., t_end - 5} | 2-year spacing, minimum 5 years held out |


## Testing Strategy

### Unit tests (per module)

Each module has its own test file. Tests should cover:

1. **Deterministic correctness:** Given known inputs, check outputs against hand-calculated values.
2. **Edge cases:** ΔT = 0 everywhere, ΔT = constant, single-point WLS fit (should raise error).
3. **Dimensional consistency:** All MSE quantities in mm², all rates in mm/yr.
4. **Weight bounds:** w(t) is always in [0, 1], including at extreme parameter values.
5. **Temporal coherence:** Rates on combined trajectories are smooth (max_rate_jump below threshold).
6. **WLS covariance propagation:** Jacobian-propagated covariance matches reparametrised fit.
7. **GIA inflation:** Only rate variance is inflated, not acceleration variance.
8. **κ_t (no κ_t^eff):** Verify that MSE_trend = Var[samples] + κ·ΔT²·dt² + κ_t·dt⁴.
9. **No double-counting:** Verify that removing rdot uncertainty from the WLS posterior (setting σ_rdot → 0) reduces Var[H_trend samples] by ¼σ_rdot²·dt⁴ and does NOT change the κ_t contribution.
10. **Synthesis log-likelihood:** On synthetic CV data where the trend is clearly better at short leads, verify that the posterior on φ₀ is large and positive.
11. **Mixture quantiles:** Verify that mixture 50th percentile lies between trend and model medians.
12. **Stochastic transition:** Verify that the marginal distribution of H_comb at each t (aggregated over all samples) is statistically consistent with the mixture quantiles.
13. **AIS add-back:** Verify H_comb_total = H_comb_non_ais + H_ais_dyn exactly (no numerical drift).

### Integration test

End-to-end test with synthetic data:

1. Generate synthetic GMSL from a known rate-and-state model with known parameters.
2. Generate synthetic AIS dynamic contribution (a known growing signal).
3. Add realistic noise.
4. Fit WLS to the non-AIS satellite-era data.
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
   - The AIS component passes through unmodified.
   - The mixture quantile CI is wider than the stochastic-transition ensemble CI (mixture has correct marginals; ensemble has coherence but slightly narrower marginals).

### Regression test

Once the module is working on real data, snapshot the diagnostics table and the BPS posterior summary. Check that future code changes do not alter results beyond the expected MCMC sampling noise.


## Open Questions

1. **Cross-validation cost.** The leave-future-out CV requires refitting the rate-and-state model for each holdout endpoint (~8-10 refits). If full MCMC is too expensive, a Laplace approximation or variational inference can be used for the CV refits only (not for the final projection). The resulting synthesis posterior will be approximate, but since it's used for the weight hyperparameters (not the primary physical parameters), this is acceptable. The sensitivity of the BPS posterior to the CV model approximation should be checked by running one full-MCMC refit and comparing.

2. **Non-Gaussian agent densities.** The current implementation approximates the model agent density as Gaussian in the CV step (for computational convenience). For the final combination, the full empirical distribution is used via the weighted-pool method. If the model posterior predictive is substantially non-Gaussian (heavy tails, skewness), the Gaussian approximation in the CV could bias the synthesis calibration. A diagnostic: compare the CV log-likelihood under Gaussian vs. kernel density estimation of the model predictive. If they differ substantially, use KDE.

3. **AIS decomposition in the pre-GRACE era.** The WLS trend model is fit to non-AIS GMSL, which requires subtracting the AIS contribution. For 1993-2002 (pre-GRACE), the AIS contribution comes from the reconciled IMBIE record, which has larger uncertainties. This uncertainty should be propagated into the WLS fit (as additional observation-level uncertainty during 1993-2002).

4. **λ estimation from Level 2.** Once the component-level models are operational, λ can be estimated empirically by comparing Level 1 and Level 2 rate predictions at the edge of the calibration domain. This provides a data-informed alternative to the subjective prior.

5. **BPS with >2 agents.** The BPS framework naturally extends to K > 2 agents. If a third agent (e.g., a GP fit, or a component-sum forecast from Level 2) is added later, the synthesis extends to a K-component mixture with K-1 weight processes. The current 2-agent implementation should be designed to accommodate this extension.
