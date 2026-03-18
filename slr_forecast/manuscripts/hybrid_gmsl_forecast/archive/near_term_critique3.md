# Critique: Near-Term Trend Constraint via Bayesian Predictive Synthesis (Revised)

Review of `near_term_trend_constraint_revised.tex` and `implementation_plan_revised.pdf`.

---

## 1. Decomposition inconsistency between agents (logical error)

The WLS trend model is fit to observed non-AIS GMSL, defined in the LaTeX as:

$$H_{\text{non-AIS}}(t) = H_{\text{total}}(t) - H_{\text{AIS}}(t)$$

where $H_{\text{AIS}}$ is the total AIS mass balance (SMB + dynamics), since the IMBIE/GRACE data used for subtraction do not decompose into SMB and dynamics.

But the implementation plan (Step 0) defines the model agent as:

```python
H_model_non_ais = {ssp: H_model[ssp] - H_ais_dyn[ssp] for ssp in ssps}
```

This subtracts only AIS dynamics from the model total, leaving AIS SMB inside the model's "non-AIS" agent. The two agents are therefore measuring different quantities:

- Trend agent: total GMSL minus total AIS (SMB + dynamics)
- Model agent: total GMSL minus AIS dynamics only

The add-back step compounds this:

$$H_{\text{comb,total}}^{(s)} = H_{\text{comb,non-AIS}}^{(s)} + H_{\text{AIS,dyn}}^{(s)}$$

If the WLS-derived non-AIS GMSL excludes all AIS, then the combined total is missing the AIS SMB contribution entirely (it was subtracted from the observations for the trend agent, but never added back).

The correct decomposition must be consistent. Two options:

**(A)** Subtract total AIS from both agents. Define non-AIS as total minus all AIS. Then the model non-AIS agent should be `H_model[ssp] - H_ais_total[ssp]`, and the add-back should use total AIS: `H_comb_total = H_comb_non_ais + H_ais_total`. AIS SMB then passes through exempt from trend suppression, which is defensible (AIS SMB is small and temperature-driven, so it could go either way).

**(B)** Subtract only AIS dynamics from both agents. Fit the WLS to total observed GMSL minus AIS dynamics only (requiring a dynamics/SMB decomposition of the IMBIE record). AIS SMB remains in both agents and gets trend-constrained, which is physically appropriate since AIS SMB is temperature-driven and well-behaved. The add-back uses AIS dynamics only.

Option B is more principled (AIS SMB should be trend-constrained), but requires decomposing the IMBIE observational record into dynamics and SMB components. Option A is operationally simpler. Either way, the current plan is inconsistent and must be fixed.

The practical magnitude of the error: AIS SMB contributes roughly -0.1 to +0.2 mm/yr to GMSL over the satellite era. Over 2025-2100, AIS SMB projections range from near-zero (EAIS gains offset WAIS/APIS losses under low emissions) to several cm (net gains under high emissions from increased snowfall). So the inconsistency is not negligible for 2100 projections, particularly under high-emissions scenarios where EAIS SMB gains could reach ~5 cm.

---

## 2. Marginal correctness claim: valid but with an unstated assumption

The LaTeX document claims that in the $\varepsilon \to 0$ limit, the stochastic-transition marginals match the mixture predictive exactly. The argument proceeds as follows. For fixed $t$ and $\varepsilon \to 0$:

$$\omega^{(s)}(t) = \begin{cases} 1 & \text{if } w^{(s)}(t) > u^{(s)} \\ 0 & \text{if } w^{(s)}(t) < u^{(s)} \end{cases}$$

Since $u^{(s)} \sim \text{Uniform}(0,1)$ independently of everything else, $P(\omega^{(s)} = 1 \mid \psi^{(s)}) = w(t; \psi^{(s)})$. The marginal CDF of $H_{\text{comb}}^{(s)}(t)$ over $(\psi^{(s)}, u^{(s)})$ is:

$$F_{\text{comb}}(h) = \mathbb{E}_\psi\big[w(t;\psi)\,F_{\text{trend}}(h) + (1-w(t;\psi))\,F_{\text{model}}(h)\big] = \bar{w}(t)\,F_{\text{trend}}(h) + (1-\bar{w}(t))\,F_{\text{model}}(h)$$

This derivation is correct, but it relies on an unstated independence assumption: that $H_{\text{trend}}^{(s)}$ and $H_{\text{model}}^{(s)}$ are statistically independent of $\psi^{(s)}$. This holds because the agent forecast ensembles are generated from their own posteriors (WLS and rate-and-state MCMC, respectively), which do not depend on the synthesis hyperparameters. The LaTeX document should state this explicitly, since it is essential to the proof and could be violated in extensions (e.g., if the model agent's calibration were to incorporate synthesis information, as in the document's Option B for budget-constraint integration).

After quantile matching, the marginal correctness is preserved: quantile matching reorders the samples but does not change the marginal distribution of either ensemble. The joint distribution changes (introducing correlation), but the mixture-weight averaging integrates over all sample indices, so the marginal of the combined ensemble remains correct.

---

## 3. Cross-validation calibrates the synthesis in a regime where it is least needed

The document is forthright about this, but the logical structure should be made fully explicit because it determines how much weight to place on the BPS posterior.

The CV holdout periods span $\Delta t \in [1, \sim 20]$ years with $\Delta T \lesssim 0.2$°C. In this regime, the forcing-departure term contributes at most:

$$\kappa_{\text{net}} \cdot 0.04 \cdot \Delta t^2$$

At the prior median $\kappa_{\text{net}} = 0.05$ and $\Delta t = 15$ yr, this is ~0.45 logit units. The acceleration-persistence term at the same lead is $\phi_0 \cdot 15^4 / \Delta t_{\text{low}}^{*4} \approx 2.5 \cdot 50625 / 160000 \approx 0.79$ logit units. Both are modest perturbations to $\phi_0 = 2.5$.

The consequence: $\phi_0$ is well-identified (it controls behavior at all leads). $\Delta t_{\text{low}}^*$ is moderately identified (it controls the crossover under small $\Delta T$, which is exactly the CV regime). $\kappa_{\text{net}}$ is weakly identified (it is swamped by the acceleration-persistence term when $\Delta T$ is small).

The practical implication is that under SSP3-7.0 and SSP5-8.5, the crossover time is almost entirely prior-determined. The BPS posterior on $\kappa_{\text{net}}$ provides at most a weak lower bound. This is the honest state of affairs, and the document handles it correctly by reporting the posterior width and the prior-vs-posterior comparison. But the reader should understand that the "data-informed" character of the BPS framework is concentrated in the low-$\Delta T$ regime. The high-emissions crossover is essentially a prior choice, albeit a physically motivated one.

One structural consequence for the sensitivity diagnostic (Eq. 37-38 in the LaTeX): comparing $\Delta T^2 \cdot \Delta t^2$ vs. $\Delta T^4 \cdot \Delta t^2$ scaling will produce nearly identical results within the CV regime, since $\Delta T$ is small and $\Delta T^4 \ll \Delta T^2$. The sensitivity diagnostic is meaningful only in projection (where $\Delta T$ is large), and the BPS posterior provides no information to distinguish between the two functional forms. The diagnostic is therefore a prior sensitivity analysis, not a posterior diagnostic. The implementation plan should state this clearly.

---

## 4. The crossover analytic approximation has a domain restriction that is not flagged

Equation 31 in the LaTeX:

$$\Delta t^* = \left(\frac{\phi_0}{\kappa_{\text{net}} \dot{T}^2 + \phi_0/(\Delta t_{\text{low}}^*)^4}\right)^{1/4}$$

This assumes $\Delta T(t) \approx \dot{T} \cdot (t - t_{\text{end}})$, valid only for monotonically warming scenarios where $T(t)$ increases linearly from $T_{\text{cal}}^{\max}$. The approximation breaks down for:

- **Overshoot scenarios** (SSP1-1.9, SSP1-2.6 late century): $\Delta T$ saturates and the $\dot{T} \cdot \Delta t$ approximation overestimates $\Delta T$, yielding a crossover that is too short.
- **Non-linear warming** (SSP5-8.5): if the warming rate itself accelerates, $\Delta T \propto \dot{T} \cdot \Delta t$ underestimates, yielding a crossover that is too long.
- **Any scenario where $T(t) < T_{\text{cal}}^{\max}$** for an extended period post-$t_{\text{end}}$: $\Delta T = 0$ and the analytic form reduces to $\Delta t^* = \Delta t_{\text{low}}^*$, which is correct but trivially uninformative.

The numerical root-finding handles all these cases correctly. The analytic approximation should be labeled as valid only for constant-warming-rate scenarios and should not be reported for overshoot SSPs. The implementation plan's `compute_crossover_diagnostics` function should include this restriction: `T_rate_by_ssp` should be `None` for overshoot scenarios, and `analytic_crossover` should return `None` in those cases.

---

## 5. Double-counting risk in the GIA inflation

The GIA uncertainty $\sigma_{\text{GIA}} = 0.15$ mm/yr is added to $\text{Var}(\hat{r})$ in the propagated covariance. The document states this treats the GIA correction error as "uncorrelated with the estimation error." This is correct if the GIA correction is applied before the WLS fit (i.e., the GMSL data input to WLS already has a specific GIA correction applied, and $\sigma_{\text{GIA}}$ represents the uncertainty in which correction was chosen).

However, if different GIA corrections shift the entire rate by a constant amount, this should also affect $\text{Var}(\hat{H})$ through the GIA rate's accumulation over the satellite era. A constant rate error of $\delta_{\text{GIA}}$ propagates to a level error of $\delta_{\text{GIA}} \cdot (\Delta t)$ at $t_{\text{end}}$ relative to the midpoint. The off-diagonal element $\text{Cov}(\hat{H}, \hat{r})$ should also be inflated. The current implementation inflates only the (1,1) element (rate variance) of the 3×3 covariance.

To be precise: the GIA correction error enters as a systematic rate offset $\delta$ that is constant over the satellite era. In the WLS model $H(t) = \alpha + \beta \cdot (t - t_{\text{ref}}) + \gamma \cdot (t-t_{\text{ref}})^2$, a constant rate offset $\delta$ adds $\delta$ to $\beta$ without affecting $\alpha$ or $\gamma$ (since $(t-t_{\text{ref}})$ is centered). So at $t_{\text{end}}$:

- $\hat{H}$ is shifted by $\delta \cdot (t_{\text{end}} - t_{\text{ref}})$
- $\hat{r}$ is shifted by $\delta$
- $\ddot{\hat{r}}$ is unshifted

The GIA inflation should add $\sigma_{\text{GIA}}^2 \cdot (t_{\text{end}} - t_{\text{ref}})^2$ to $\text{Var}(\hat{H})$, $\sigma_{\text{GIA}}^2$ to $\text{Var}(\hat{r})$, and $\sigma_{\text{GIA}}^2 \cdot (t_{\text{end}} - t_{\text{ref}})$ to $\text{Cov}(\hat{H}, \hat{r})$. That is, the inflation should be the rank-1 outer product:

$$\Sigma_{\text{GIA}} = \sigma_{\text{GIA}}^2 \begin{pmatrix} (t_{\text{end}} - t_{\text{ref}})^2 & (t_{\text{end}} - t_{\text{ref}}) & 0 \\ (t_{\text{end}} - t_{\text{ref}}) & 1 & 0 \\ 0 & 0 & 0 \end{pmatrix}$$

With $t_{\text{ref}} \approx 2005$ and $t_{\text{end}} \approx 2023$, the level inflation is $0.15^2 \cdot 18^2 \approx 7.3 \text{ mm}^2$, which is $\sigma_H \approx 2.7$ mm. This may or may not be large relative to the WLS level uncertainty; it should be checked. The current plan inflates only the rate variance, which is incomplete.

---

## 6. Ensemble size alignment is underspecified

The stochastic transition requires three aligned sets of samples:

1. Trend ensemble: $N_{\text{trend}}$ samples from WLS posterior
2. Model ensemble: $N_{\text{model}}$ samples from rate-and-state MCMC
3. Weight ensemble: $N_\psi$ samples from BPS posterior on $\psi$

The plan sets $N_{\text{samples}} = 10{,}000$ for the trend and model ensembles. The BPS posterior produces $16 \times 5000 = 80{,}000$ raw samples, thinned to ~8,000 effective samples.

The `posterior_weight_ensemble` method accepts an `n_samples` parameter, suggesting it subsamples from the BPS posterior. But the quantile matching step (which pairs trend sample $k$ with model sample $k$) creates a fixed rank ordering. If the weight ensemble is then drawn independently for each "combined" sample, the sample index $s$ in the stochastic transition corresponds to a specific rank in the trend/model ensembles but a random draw from the $\psi$ posterior. This is fine for marginal correctness (the independence assumption in point 2 holds), but the implementation plan should specify the exact alignment procedure:

- Draw $N$ samples from the $\psi$ posterior (with replacement if $N > N_\psi^{\text{eff}}$)
- Draw $N$ samples from the trend ensemble (already $N$-sized if WLS gives exact draws)
- Draw $N$ samples from the model ensemble (resample if model MCMC size $\neq N$)
- Apply quantile matching to trend and model ensembles
- For each $s \in \{1, \ldots, N\}$, draw $u^{(s)}$ and compute $\omega^{(s)}(t)$ using $\psi^{(s)}$

The plan mentions resampling with replacement for size mismatches but does not specify whether the three resamplings should be independent (they should be) or coordinated.

---

## 7. Prior predictive checks may be too restrictive

The four prior predictive checks are:

1. 5 yr lead, all SSPs: median $w > 0.8$
2. 10 yr lead, SSP2-4.5: median $w \in [0.3, 0.8]$
3. 20 yr lead, SSP2-4.5: median $w < 0.3$
4. 20 yr lead, SSP1-2.6 ($\Delta T \approx 0$): median $w \in [0.3, 0.7]$

Check 4 deserves scrutiny. At 20 yr lead with $\Delta T = 0$, the logit is $\phi = \phi_0 - \phi_0 \cdot 20^4 / \Delta t_{\text{low}}^{*4} = \phi_0(1 - 160000/\Delta t_{\text{low}}^{*4})$. At $\Delta t_{\text{low}}^* = 20$ yr, $\phi = 0$ and $w = 0.5$. At $\Delta t_{\text{low}}^* = 15$, $\phi = \phi_0(1 - 160000/50625) = \phi_0(1 - 3.16) = -2.16\phi_0 \approx -5.4$, giving $w \approx 0.004$. At $\Delta t_{\text{low}}^* = 25$, $\phi = \phi_0(1 - 160000/390625) = \phi_0 \cdot 0.59 \approx 1.47$, giving $w \approx 0.81$.

So check 4 (median $w \in [0.3, 0.7]$ at 20 yr) is essentially a constraint that the prior median of $\Delta t_{\text{low}}^*$ is near 20 yr, which is guaranteed by construction (the prior median is 20 yr). The check is tautological: it verifies that the prior produces the behavior it was designed to produce. This is not useless (it catches implementation errors), but it is not the strong validation it appears to be.

A more informative check would test the prior predictive *spread*: e.g., that the 10th-90th percentile range of $w$ at 20 yr under SSP1-2.6 spans at least [0.1, 0.8], confirming that the prior is not overly informative about the crossover time.

---

## 8. emcee is a risky choice for this problem

Open Question 4 in the implementation plan flags multimodality but frames it as a "pilot run" check. The concern is more structural than that.

The log-likelihood (Eq. 24) is a sum of $\log(w \cdot h_1 + (1-w) \cdot h_2)$ terms. This log-mixture-density objective is well-known to produce multimodal posteriors when the two components have comparable densities at the observed data. In this problem, the transition from trend-dominated to model-dominated happens within the CV window, so there will be holdout observations where the two agents have similar predictive densities. The log-sum-exp structure creates a ridge in the likelihood along the level set where $w(t_h + \Delta t) \approx$ constant for critical $\Delta t$ values.

emcee (affine-invariant ensemble sampler) handles correlations well but can fail to traverse multimodal posteriors, especially in low dimensions where the modes may be separated by narrow valleys. The convergence diagnostic ($\hat{R}$ via split-chain) can detect some failures but not all.

Given that this is a 3-parameter problem, nested sampling (dynesty) or even grid evaluation would be more robust. The computational cost is dominated by the model refits in the CV step, not the MCMC itself. Recommendation:

- Run emcee as the primary sampler
- Run dynesty as a cross-check
- If the posteriors agree (in the sense of matching quantiles to within MCMC noise), proceed with emcee
- If they disagree, use dynesty

This should be a prerequisite, not an open question.

---

## 9. The holdout influence diagnostic is correctly identified but the flagging threshold is arbitrary

The threshold of 0.5 posterior SDs for flagging is reasonable as a starting point, but it is not grounded in any formal criterion. A more principled approach would be to report the full leave-one-out posterior for each hyperparameter and compute the effective number of holdout observations $p_{\text{loo}}$ by analogy with PSIS-LOO diagnostics. A holdout with $k > 0.7$ (in the Pareto-$k$ diagnostic sense) indicates that the posterior is sensitive to that observation in a way that the tail approximation cannot handle.

However, this level of sophistication may be unnecessary for a 3-parameter problem with ~8 holdouts. The 0.5 SD threshold is adequate for the initial implementation, provided it is treated as a screening criterion that triggers deeper investigation, not as a pass/fail test.

---

## 10. Minor issues

**Units of $\sigma_{\text{AIS}}$.** The notation table lists $\sigma_{\text{AIS}}(t)$ with units "mm" but the text specifies $\sigma_{\text{AIS}} \approx 0.5$ mm/yr (rate units). Since the AIS subtraction uncertainty enters via $\sigma_{\text{eff},i}^2 = \sigma_i^2 + \sigma_{\text{AIS}}^2(t_i)$, where $\sigma_i$ is in mm (level uncertainty per annual observation), $\sigma_{\text{AIS}}$ should also be in mm per annual observation, which is numerically the same as mm/yr. The notation should be consistent.

**The cubic robustness check.** The plan specifies that if the cubic is significant at 5% but not 1%, "report both quadratic and cubic fits and propagate the model-choice uncertainty as an additional contribution to the trend agent's predictive variance." This is ad hoc. A cleaner approach is Bayesian model averaging over the quadratic and cubic, weighted by their posterior model probabilities (BIC approximation is adequate for this). This would produce a single trend agent predictive that automatically accounts for model-choice uncertainty.

**`fix_scale=True` as default.** This assumes the observation uncertainties are trusted as absolute error bars. For satellite altimetry, this is defensible if the uncertainties are well-characterized (which they are for the modern era). But the AIS subtraction uncertainty may not be as well-characterized, especially in the pre-GRACE era where the IMBIE reconciliation uncertainties are rough estimates. A diagnostic should compare results with `fix_scale=True` and `fix_scale=False` and report the residual-variance estimate $\hat{\sigma}^2$. If $\hat{\sigma}^2 \gg 1$, the observation uncertainties are underestimated.

---

## Summary of required actions

### Must fix before implementation

1. **AIS decomposition inconsistency** between trend and model agents (Section 1)
2. **GIA inflation** should be rank-1 update to the full 3×3 covariance, not just rate variance (Section 5)

### Should fix before publication

3. State the **independence assumption** underlying marginal correctness (Section 2)
4. Label the **analytic crossover** as valid only for constant-warming-rate scenarios (Section 4)
5. Specify **ensemble alignment procedure** explicitly (Section 6)
6. Replace emcee-only plan with **emcee + dynesty cross-check** (Section 8)

### Design improvements

7. Add a **prior predictive spread check** alongside the median checks (Section 7)
8. **Bayesian model averaging** for quadratic vs. cubic trend (Section 10)
9. Diagnostic for **`fix_scale` choice** (Section 10)
