# Review: Near-Term Trend Constraint via Bayesian Predictive Synthesis

**Scope:** Joint review of `near_term_trend_constraint_revised.tex` (mathematical derivation) and `implementation_plan_revised.pdf` (implementation specification).

**Date:** 2026-03-12

---

## 1. Structural Issues

### 1.1. `forcing_form_sensitivity` is not implementable as specified

The `forcing_form_sensitivity` function (Module 6) takes the fitted `BPSSynthesis` object and compares crossover diagnostics under the default (ΔT² · Δt²) and "optimistic" (ΔT⁴ · Δt²) forcing-departure scaling. The problem: κ_net in the default form has units logit · °C⁻² · yr⁻², while κ'_net in the optimistic form has units logit · °C⁻⁴ · yr⁻². These are dimensionally incompatible. You cannot take the posterior on κ_net from a fit to the default form and plug it into the optimistic form — the posterior median κ_net = 0.05 in the default form does not correspond to a meaningful value of κ'_net in the optimistic form.

Two options:

- **(a) Refit the full BPS (CV + MCMC) under the optimistic form with its own prior on κ'_net.** This is the principled approach. The prior on κ'_net must be specified independently (different units, different physical content). The sensitivity diagnostic then compares two fully calibrated posteriors.
- **(b) Specify a closed-form mapping between κ_net and κ'_net that preserves some crossover-equivalence condition.** For example, require that the two forms produce the same crossover time under SSP2-4.5 at the prior median: κ'_net · ΔT⁴ · Δt² = κ_net · ΔT² · Δt² → κ'_net = κ_net / ΔT², evaluated at the SSP2-4.5 crossover ΔT. This is cheaper but requires a defensible equivalence definition and is SSP-dependent.

The implementation plan does not address this. The function signature suggests it operates on the existing BPS posterior, which is incoherent. **This must be redesigned before implementation.**

### 1.2. Temporal correlation in AIS subtraction uncertainty is ignored

The WLS fit treats observations as independent (given the per-observation weights 1/σ²_eff,i). But the AIS subtraction error is temporally correlated: IMBIE reconciliation biases persist across years (systematic offsets in one or more satellite products). Under independence, the WLS correctly estimates the rate uncertainty from the scatter of individual annual observations. Under temporal correlation, the effective number of independent observations is smaller, and the rate uncertainty is biased low.

This matters quantitatively. The pre-GRACE AIS subtraction uncertainty is σ_AIS ≈ 0.5 mm/yr, but if the bias is correlated over 5-year blocks (a conservative estimate for IMBIE reconciliation systematics), the effective rate uncertainty contribution scales as σ_AIS / √n_eff rather than σ_AIS / √n, where n_eff ≪ n. The GIA inflation (σ_GIA = 0.15 mm/yr added in quadrature to Var(r̂)) addresses one systematic rate error but not the AIS subtraction systematic.

Two mitigations:

- **(a) Inflate the rate variance by an additional σ²_AIS,sys term** (analogous to the GIA inflation). This is simple and sufficient for the BPS synthesis, where the trend agent's uncertainty enters only through the ensemble variance. The magnitude should be derived from the IMBIE inter-method spread at the rate level (not the annual level).
- **(b) Model the AIS subtraction error as a correlated process** (e.g., block-diagonal or AR(1) covariance in the WLS). More principled but complicates the WLS machinery for a second-order effect.

The LaTeX derivation does not discuss this. The implementation plan propagates σ_AIS(t) into per-observation weights but not into a systematic rate-level inflation.

---

## 2. Correctness Issues

### 2.1. Minor numerical error in the Δt*_low prior 95% CI

The LaTeX (Eq. 17) states Δt*_low ~ LogNormal(log(20), 0.35) with "approximate 95% CI [10, 38]." The actual 2.5th and 97.5th percentiles are:

```
exp(log(20) - 1.96 × 0.35) = exp(2.310) ≈ 10.1
exp(log(20) + 1.96 × 0.35) = exp(3.682) ≈ 39.7
```

The 95% CI is [10, 40], not [10, 38]. The lower bound is correct; the upper bound is off by ~5%. Cosmetic but should be fixed to avoid confusion during prior predictive checks.

### 2.2. The simplification from Eq. (20) to Eq. (21) — confirmed correct

The derivation correctly shows that

```
p_comb(H(t) | D) = w̄(t) · h_trend(H(t)) + (1 − w̄(t)) · h_model(H(t))
```

where w̄(t) = E_{ψ|D}[σ(φ(t; ψ))]. This holds because h_trend and h_model are independent of ψ.

The `compute_mixture_quantiles` function uses w̄(t) as a fixed weight to pool trend and model samples, which correctly reproduces the marginal CDF of the Bayes-marginalized mixture at each t. The mixture quantiles (primary uncertainty reporting) and the stochastic-transition quantiles (trajectory coherence) are correctly distinguished in both documents. Confirmed correct.

### 2.3. The analytic crossover approximation has an unstated domain restriction

Equation (26) gives the closed-form crossover:

```
Δt* = (φ₀ / (κ_net · Ṫ² + φ₀ / (Δt*_low)⁴))^(1/4)
```

This uses ΔT(t) ≈ Ṫ · Δt, which is valid only for monotonically warming scenarios where T(t) > T^max_cal throughout. For overshoot scenarios (SSP1-2.6 with peak-and-decline), ΔT(t) in cumulative-maximum mode plateaus after the temperature peak, and the linear approximation breaks down. The implementation plan's `compute_crossover_diagnostics` correctly uses numerical root-finding as the primary method, with the analytic formula reported for intuition only. The docstring should note the monotone-warming restriction on the analytic approximation.

---

## 3. Specification Gaps

### 3.1. T^max_cal(t_h) recomputation in cross-validation is underdocumented

The LaTeX (§3.4.2) correctly states that ΔT in the CV is computed "relative to T^max_cal(t_h) = max_{t ≤ t_h} T(t)" — i.e., the calibration maximum is recomputed for each holdout endpoint using only data available up to t_h. This is necessary for the CV to be genuinely out-of-sample.

The `build_cv_data` function signature takes `T_obs` as input but does not document this recomputation in the docstring or parameter description. The implementation note on page 16 mentions it, but this should be elevated to the docstring. A naive implementation would use the global T^max_cal from the full record, which would leak future information into the CV and bias the synthesis toward trusting the trend at longer leads than the data support.

### 3.2. Definition of H_AIS,dyn is ambiguous

The LaTeX defines H_AIS,dyn(t) as "the AIS dynamic mass-loss contribution to GMSL" and exempts it from trend suppression. But the AIS has both dynamic (ice discharge) and SMB (surface mass balance) components. The exemption rationale focuses on WAIS instability scenarios, which are dynamic phenomena. EAIS SMB is mostly a snowfall response to warming — a well-behaved, trend-friendly signal.

If H_AIS,dyn means the total AIS contribution (dynamic + SMB for all three basins), then EAIS SMB is also exempted from trend anchoring, which is more conservative than necessary. If it means only the ice-dynamic discharge component, then AIS SMB is included in the non-AIS total and subject to trend anchoring, which is more physically appropriate.

The implementation needs a clear decision. The simplest approach for Phase 1: exempt the total AIS contribution (what IMBIE/GRACE measures, without dynamic/SMB decomposition), noting that this is conservative. The 7-component framework allows refining this later if the EAIS SMB exemption produces artifacts in the combined projection.

**This should be recorded as a DECISION.**

### 3.3. Ensemble-size matching is unspecified

The stochastic transition requires the trend ensemble, model ensemble, and weight ensemble to have the same n_samples. The implementation notes (page 20–21) say "Resample with replacement before calling if sizes differ." But the three sources have different natural sizes:

| Source | Natural size |
|---|---|
| Trend ensemble | n_samples from `WLSTrendModel.sample_derivatives` (default 10,000) |
| Model ensemble | Determined by rate-and-state MCMC (may differ) |
| Weight ensemble | n_walkers × n_samples_per_walker = 16 × 5000 = 80,000, thinned to ~8,000 |

The implementation plan should specify where the resampling occurs and at what target size. Natural choice: fix n_samples = 10,000 for trend and model ensembles, and thin or resample the weight posterior to 10,000 to match. `generate_stochastic_transition_ensemble` should either enforce equal sizes or raise an error on mismatch.

### 3.4. The holdout influence diagnostic could be cheaper via importance weighting

The leave-one-holdout-out influence diagnostic re-runs MCMC J times (J ≈ 8). Each run is 16 walkers × 7000 steps. An alternative: importance reweighting. For each holdout j, the log-likelihood without holdout j is ℓ(ψ) − ℓ_j(ψ). The posterior without holdout j satisfies:

```
p(ψ | D_{-j}) ∝ p(ψ | D) · exp(−ℓ_j(ψ))
```

So the importance weight for each posterior sample ψ^(s) is exp(−ℓ_j(ψ^(s))). If the effective sample size of the importance weights is reasonable (say, > 100), this gives the leave-one-out posterior without any additional MCMC runs.

This works well when no single holdout dominates — precisely the case where the influence diagnostic is not flagging anything. When a holdout does dominate (the important case), the importance weights will have low ESS, and re-running MCMC is necessary. So the importance-weighting approach is a cheap pre-screen: compute it first, and only re-run MCMC for holdouts where the importance ESS is below threshold.

The implementation plan should mention this optimization.

### 3.5. The `model_refit_fn` callback specification is underspecified

The `build_cv_data` function takes a `model_refit_fn` callback that "takes t_h and returns a dict with `H_predictive_mean` and `H_predictive_std`." But what the callback must do internally is non-trivial:

1. Refit the rate-and-state model to [1900, t_h].
2. Generate the posterior predictive for t > t_h using the actual observed temperature trajectory (not an SSP scenario).
3. Return the predictive mean and std at each held-out time.

The callback interface returns only mean and std (Gaussian approximation). The implementation note on page 16 says "approximate as Gaussian with mean and std from the model posterior predictive ensemble. This is adequate for the synthesis calibration." This is probably fine for the non-AIS total (AIS exempted, so the model predictive for non-AIS is likely well-approximated by Gaussian). But if the model predictive is significantly non-Gaussian (e.g., from strongly nonlinear rate–temperature coupling near regime boundaries), the Gaussian approximation will underestimate the tails, making the model agent look less reliable than it is and biasing the synthesis toward the trend.

Open Question #2 in the implementation plan flags the Gaussian vs. KDE comparison, which is appropriate. No immediate action needed, but the callback docstring should note the Gaussian assumption and its rationale.

### 3.6. Factual error in the holdout influence example

The LaTeX (§3.4.2 / holdout influence discussion) states: "for example, t_h = 2003 includes the post-Pinatubo recovery and 2004 tsunami in its training data."

Two problems:

1. The 2004 Indian Ocean tsunami occurs at t = 2004, which is in the held-out period for t_h = 2003, not in the training data. The sentence should say the 2004 event is in the held-out data (biasing the evaluation), not in the training data.
2. The 2004 tsunami does not materially affect global mean sea level. The sentence likely refers to the 2004/05 GMSL dip, which is more plausibly associated with La Niña–like conditions and land water storage changes, not tsunami effects.

The example is misleading and should be revised.

---

## 4. Design Judgment

### 4.1. Option A (BPS as post-processing) is correct for Phase 1

The recommendation to use BPS as post-processing rather than integrating the WLS rate prior into the Level 1 calibration (Option B) is sound. Option B's double-counting problem (satellite-era data enters both the WLS fit and the rate-and-state likelihood) is real and non-trivial to resolve. The partitioned-likelihood approach is feasible but adds complexity not justified in Phase 1.

One observation: Option A means the component sum at short leads may not match the BPS-combined total. The LaTeX identifies this as a diagnostic, which is correct. If the component sum disagrees with the BPS-combined total at short leads, that is a signal that the component models are not collectively reproducing the observed rate — a useful diagnostic in its own right.

### 4.2. The prior on φ₀ may be too tight — but this is acceptable

The prior φ₀ ~ N(2.5, 0.5²), truncated to φ₀ > 0, encodes a strong belief that the trend is much more precise than the model at t_end. The derivation uses σ_model ≈ 1.0 mm/yr and σ_r ≈ 0.3 mm/yr to get φ₀ ≈ log(1.0²/0.3²) ≈ 2.4. The σ = 0.5 prior width gives a 95% CI of roughly [1.5, 3.5], corresponding to model-to-trend MSE ratios of [e^1.5, e^3.5] = [4.5, 33].

This seems reasonable. The CV will update φ₀ since it is well-identified by short-lead holdout data, so the prior width matters less than it would for κ_net. No action needed.

### 4.3. The ε = 0.05 default for the stochastic transition is reasonable

The smoothing parameter ε = 0.05 means the transition from ω = 0.12 to ω = 0.88 occurs over a range of w of about 0.2. Given that w(t) changes by ~0.2 over 2–5 years (depending on SSP and ψ), this produces a transition window of a few years — smooth enough for rate analysis but sharp enough to preserve marginal fidelity. The diagnostic reporting results for ε ∈ {0.02, 0.05, 0.10} is appropriate. No action needed.

### 4.4. The WLS quadratic over GP argument is well-made

The LaTeX (§3.2) argues that a WLS quadratic is preferable to a GP for the trend agent because: (a) the trend projection uses only (Ĥ, r̂, r̈) at t_end plus their covariance, so the GP's ability to capture non-quadratic structure is irrelevant; (b) the GP's acceleration estimate is sensitive to the kernel lengthscale, which is an opaque tuning parameter; (c) the WLS quadratic has zero hyperparameters to optimize. This argument is correct and the decision is sound.

---

## 5. Summary of Required Actions

### Must fix before implementation

| # | Issue | Section |
|---|---|---|
| 1 | Redesign `forcing_form_sensitivity` to either refit BPS under the alternative form or specify a defensible parameter mapping. Current design is dimensionally incoherent. | §1.1 |
| 2 | Clarify definition of H_AIS,dyn (total AIS vs. dynamic-only). Record as a DECISION. | §3.2 |
| 3 | Explicitly document T^max_cal(t_h) recomputation in the `build_cv_data` docstring. | §3.1 |
| 4 | Correct the 2004 tsunami example in the holdout influence discussion. | §3.6 |

### Should fix

| # | Issue | Section |
|---|---|---|
| 5 | Address temporal correlation in AIS subtraction uncertainty, at minimum via a systematic rate-level inflation analogous to GIA inflation. | §1.2 |
| 6 | Correct Δt*_low prior 95% CI from [10, 38] to [10, 40]. | §2.1 |
| 7 | Specify ensemble-size matching protocol (where resampling occurs, target size). | §3.3 |

### Optional improvements

| # | Issue | Section |
|---|---|---|
| 8 | Add importance-weighting pre-screen for the holdout influence diagnostic. | §3.4 |
| 9 | Note monotone-warming restriction on the analytic crossover formula in the `compute_crossover_diagnostics` docstring. | §2.3 |
| 10 | Note Gaussian-approximation assumption for the model agent density in CV and its interaction with the AIS exemption in `model_refit_fn` docstring. | §3.5 |
