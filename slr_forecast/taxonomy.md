# Terminology Taxonomy

This document defines the terms used throughout the codebase, notebooks, and paper
to describe predictions, uncertainties, and their decomposition. All terms are
consistent with Bayesian inference and decision theory.

---

## Predictions

**Posterior predictive distribution**
The full probability distribution over future sea level, conditional on
observed data and a specified emissions scenario. This is what our framework
produces: P(H_future | data, scenario). It integrates over all uncertain
parameters via their posterior distributions.

**Conditional predictive distribution**
A posterior predictive distribution evaluated under a specific emissions
scenario (SSP). Written P(H | data, SSP). Every SSP-specific result in this
project is a conditional predictive distribution. The conditioning is on the
scenario; the integration is over parameters, model inadequacy, and deep
uncertainty.

**Marginal predictive distribution**
The predictive distribution obtained by averaging (marginalising) over
scenarios, weighted by scenario plausibility. We do not produce this — it
would require assigning probabilities to SSPs, which is a policy judgment
outside the scope of physical science. We present conditional distributions
for each SSP and leave the marginalisation to decision-makers.

**Point prediction**
A single-valued summary of a predictive distribution (e.g., posterior
median at 2100). Useful for communication but discards the uncertainty
structure. Always accompany with credible intervals.

**Hindcast**
A conditional predictive distribution evaluated on historical data that
were withheld from calibration. Used to assess predictive skill via
leave-future-out cross-validation. The hindcast is Bayesian: it is the
posterior predictive distribution for the withheld period, not a
frequentist point forecast.

---

## Uncertainty sources

All uncertainties in this framework are expressed as components of the
posterior predictive variance. The total variance decomposes additively
(with covariance terms) into the following named sources.

**Scenario uncertainty**
Variance across conditional predictive distributions for different SSPs.
Reflects societal choices about emissions, land use, and policy. Irreducible
by physical observation or modelling. Quantified by comparing predictive
distributions across SSPs at each time horizon.

**Forcing uncertainty**
Given an emissions scenario, the uncertainty in the resulting climate
forcing (global and regional temperature trajectories). Sources include
climate sensitivity, aerosol forcing, and carbon cycle feedbacks. In our
framework this enters through the spread in IPCC temperature projections
within a single SSP. Conceptually distinct from scenario uncertainty: two
societies could follow the same emissions pathway and still experience
different warming due to uncertain climate sensitivity.

**Parametric uncertainty**
Uncertainty in the rate-temperature coefficients (a, b, c, tau, gamma, etc.)
given the model structure. Quantified by the posterior distribution from
Bayesian calibration. This is the uncertainty that shrinks as we add more
observational data.

**Structural uncertainty (model inadequacy)**
Uncertainty arising from the choice of model form: linear vs quadratic,
independence of components, validity of the surface-to-ocean transfer
function, stationarity assumptions. Partially captured by the sigma_extra
(model inadequacy) parameter in each Bayesian fit, which absorbs systematic
misfit between model and observations. Also assessed via model comparison
(BIC, posterior predictive checks) and sensitivity analyses (taper, model
selection, temperature choice).

**Deep uncertainty (epistemic)**
Uncertainty about processes that are poorly constrained by available
observations and whose physics is not fully represented in any existing
model. In this project, deep uncertainty is concentrated in the West
Antarctic Ice Sheet (WAIS): the timing and magnitude of marine ice sheet
instability (MISI), marine ice cliff instability (MICI), and the bias
from using Glen's flow law with n=3 when observations support n=4.

The A4 framework addresses deep uncertainty through a discrete scenario
mixture with physically motivated ranges, not through a parametric posterior.
This is a deliberate methodological choice: the A4 scenarios represent
structurally different physical futures (no instability, MISI only,
MISI + amplifiers, MISI + MICI), not parameter variations within a
single model. The scenario weights are informed by expert judgment and
physical plausibility, not by Bayesian updating from data. As
observational constraints on WAIS dynamics improve, some scenarios may
be excluded and the weights revised.

> **TODO — A4 framework extensions.** The current A4 correction
> addresses rheology (n=3 vs n=4) only. Planned updates:
>
> 1. **Calving processes.** Calving is not well represented in any
>    IPCC ice sheet model (Aschwanden et al., 2021). Iceberg calving
>    and ice cliff failure are distinct from MISI/MICI in that they
>    can accelerate mass loss even without grounding-line instability.
>    The A4 scenario ranges should be widened or a separate
>    multiplicative correction applied to account for calving-driven
>    mass loss that is missing from the ISMIP6 baseline.
>
> 2. **Rapid grounding-line retreat.** Recent observations of rapid
>    (km/yr) grounding-line retreat at Thwaites and Pine Island
>    glaciers suggest that retreat rates may exceed those produced by
>    current process models. This is related to but distinct from
>    MISI — it concerns the rate of retreat, not just whether
>    instability is triggered. Consider adding a rate amplification
>    factor to the MISI scenarios (S2, S3) or revising the within-
>    scenario ranges upward.
>
> 3. **Initialisation uncertainty.** Ice sheet model projections are
>    sensitive to the initial state, which is itself uncertain because
>    not all aspects of an ice sheet state are observable (bed
>    topography, basal conditions, internal temperature, damage).
>    Recent work demonstrates that initialisation spread alone can
>    produce projection uncertainty comparable to or larger than the
>    inter-model spread in ISMIP6 — a source of variance that is
>    currently unquantified in IPCC assessments. If this can be
>    framed as a multiplicative or additive correction to the
>    ISMIP6-based scenario ranges, it should be incorporated into
>    the A4 framework. The challenge is that initialisation
>    uncertainty is model-specific and may not factorise cleanly
>    from the structural and parametric uncertainties already
>    represented in A4.

**Internal variability**
Stochastic fluctuations in the climate system (ENSO, NAO, volcanic
eruptions, ocean mesoscale variability) that are uncorrelated with
long-term forcing. Contributes to predictive variance at short horizons
(decadal) but is dominated by other sources at centennial horizons.
Partially absorbed by sigma_extra in the Bayesian fits; partially
captured by the residual variance in hindcast skill scores.

**Rheology bias (A1 correction)**
A specific, quantifiable component of structural uncertainty in ice sheet
process models: the systematic underestimate of ice sheet response arising
from the use of Glen's flow law exponent n=3 when observational evidence
supports n=4 (Millstein, Minchew, & Pegler, 2022; Martin et al., in press).
This bias is correctable via a multiplicative factor (A1 in the A4
framework) and applies to all ice sheet projections, not just WAIS.
Separated from generic structural uncertainty because it has a known
physical origin and a quantified magnitude (21-35% underestimate at 2100).
This is the first of several planned quantifiable corrections to the
ISMIP6 baseline (see TODO under Deep uncertainty above).

---

## Intervals

**Credible interval (CI)**
A Bayesian interval [a, b] such that P(a < theta < b | data) = p.
All uncertainty intervals in this project are credible intervals derived
from posterior or posterior predictive distributions. We report 90% CI
(5th-95th percentiles) and 66% CI (17th-83rd percentiles) unless stated
otherwise.

Not to be confused with a confidence interval, which is a frequentist
concept with a different interpretation (coverage probability over
repeated experiments, not posterior probability for the parameter).

**Highest density interval (HDI)**
The narrowest credible interval containing the specified probability mass.
Used for reporting parameter posteriors (94% HDI convention from arviz).
For symmetric posteriors, the HDI coincides with the equal-tailed CI;
for skewed posteriors (e.g., tau, WAIS contributions), the HDI is more
informative.

---

## Model comparison

**Bayes factor / evidence**
The ratio of marginal likelihoods P(data | M_1) / P(data | M_2) for two
competing models. The principled Bayesian model comparison tool, but
computationally demanding for the models in this project (requires
integration over the full parameter space).

**BIC approximation**
The Bayesian Information Criterion approximates the log marginal likelihood
as log P(data | M) ~ -BIC/2. We use delta-BIC (BIC_linear - BIC_quadratic)
as a computationally tractable proxy for the Bayes factor in the model
selection analysis (linear vs quadratic for each component). delta-BIC > 0
favours the quadratic model; delta-BIC < -2 favours linear. This is an
approximation — it assumes the posterior is approximately Gaussian near
the mode, which is reasonable for the well-constrained components.

**Posterior predictive check**
Generating simulated data from the fitted model and comparing summary
statistics with the observed data. Used to assess model adequacy (does the
model generate data that look like the observations?) rather than to
compare models against each other.

---

## Terms we avoid

**Projection** (unqualified)
Too ambiguous. In IPCC usage, "projection" means a conditional prediction
that bundles all physical uncertainties into one envelope without
decomposition. Our framework decomposes these uncertainties explicitly.
Use "conditional predictive distribution" or, where brevity is needed,
"conditional prediction."

**Forecast** (unqualified)
Implies a single predicted future without conditioning on scenarios.
Acceptable only in the specific context of "hindcast skill" (where the
conditioning on observed forcing is implicit) or when referring to the
forecasting literature (Green & Armstrong, Makridakis).

**Confidence interval**
Frequentist concept. All intervals in this project are Bayesian credible
intervals or highest density intervals.

**Significance / p-value**
Frequentist hypothesis testing concepts. We assess evidence for model
features (e.g., quadratic acceleration) via posterior probability
P(a > 0 | data) and Bayes factor approximations (delta-BIC), not via
null hypothesis significance testing.

**Best estimate**
Implies a single correct value. Use "posterior median" or "posterior mean"
with explicit uncertainty quantification.
