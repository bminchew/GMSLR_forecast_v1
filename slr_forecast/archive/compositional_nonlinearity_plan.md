# Paper Plan: The Nonlinearity in Sea-Level Rise is Compositional, Not Dynamical

## Central Thesis

The apparent quadratic relationship between aggregate GMSL and GMST is not a single physical process accelerating — it emerges from the shifting dominance of component contributions (thermosteric → land ice). Each component individually has a simpler (linear or weakly nonlinear) relationship with temperature. The aggregate polynomial is an emergent property of a time-varying mixture.

## Why This Matters

- **Semi-empirical models** fit a·T² + b·T + c to aggregate GMSL and extrapolate it. This works empirically but lacks physical transparency, making it easy for skeptics to dismiss as "just statistics."
- **IPCC process models** project each component separately but have known low biases (n=3 rheology, missing calving, biased ensembles). Their aggregate projections are lower than semi-empirical models.
- **This paper bridges the gap**: component-level semi-empirical models are physically interpretable, reveal *why* the aggregate relationship is nonlinear, and expose *which* IPCC components are biased.

---

## Phase 1: Component-Level GMSL–T Relationships (The Core Argument)

### Step 1.1: Component Rate vs Temperature Scatter Plots
- For each component (thermosteric, glaciers, Greenland, Antarctica, TWS), reconstruct the implied rate from the level-space fit and plot rate vs T
- Also plot finite-difference observed rates for validation
- Show the aggregate has clear curvature; individual components are simpler
- **Output**: Figure 2 (6-panel: 5 components + aggregate)
- **Existing code**: `fit_bayesian_level()`, component posteriors from cells 12, 41, 46
- **New code**: Rate reconstruction from level-space coefficients

### Step 1.2: Formal Model Selection (Linear vs Quadratic)
- For each component, fit:
  - M1: rate = b·T + c (linear)
  - M2: rate = a·T² + b·T + c (quadratic)
- Compare using WAIC, LOO-CV, or BIC
- **Key result**: aggregate *requires* quadratic; individual components do not (or only Greenland weakly)
- **Output**: Table of model selection criteria
- **New code**: Linear-only fit variant (modify `fit_bayesian_level` to accept `order` parameter, or use very tight prior on `a`)

---

## Phase 2: Demonstrating Compositional Emergence

### Step 2.1: Synthetic Demonstration
- Construct toy model: two linear components, one activating over time
- Show that the weighted sum appears quadratic even though neither component is
- Mathematically: if rate_total = b_A·T + w(T)·b_B·T and w(T) ~ T, then rate_total ~ T² even though each component is linear in T
- **Output**: 2-panel explanatory figure
- **New code**: ~30-50 lines, standalone

### Step 2.2: Empirical Reconstruction
- Sum individual component fits and compare against direct aggregate fit
- Compare: a_total (direct) vs a_thermo + a_glacier + a_greenland (sum of components)
- Show posterior of the difference is consistent with zero (AIS+TWS residual accounts for remainder)
- Generate ensemble curves with uncertainty envelopes
- **Output**: Figure 4 — sum of components ≈ aggregate quadratic
- **Existing code**: Cell 50 already computes coefficient sum; needs expansion into proper figure

---

## Phase 3: Sliding-Window Component Contributions

### Step 3.1: Time-Varying Fractional Contributions
- Sliding-window (20-30 yr) OLS on each component's rate
- Compute fractional contribution: f_i(t) = rate_i(t) / Σ rate_j(t)
- Show the thermosteric → land-ice shift quantitatively
- **Output**: Figure 3 — stacked area chart of fractional contributions, ~1920–2018
- **Existing code**: Cell 35 has satellite-era rates; Frederikse data has full 1900–2018 record
- **New code**: `compute_sliding_component_rates()` utility function
- **Pitfall**: Narrow windows will be noisy for Antarctica and TWS; use 20-30 yr

---

## Phase 4: Component-Level Projections

### Step 4.1: TWS Projection
- Use IPCC FACTS `landwaterstorage` projections (reader exists: `read_ipcc_ar6_component`)
- TWS is small and anthropogenic — IPCC projections are adequate
- Alternative: simple linear extrapolation of GRACE-era trend
- **New code**: `project_tws_ensemble()` — likely simple wrapper around IPCC FACTS

### Step 4.2: Antarctica Projection
- Budget-derived AIS residual from existing cell 51 provides calibration-period constraint
- Cross-validate against IMBIE direct observations
- Port A4 deep-uncertainty framework from `predictability_analysis.ipynb` (cells 32-48)
- **New code**: Factor A4 into shared function in `component_models.py` or similar
- **Pitfall**: Verify scenario weights and parameter ranges are still appropriate in component-sum context

### Step 4.3: Component-Level IPCC Comparison
- For each component, compare semi-empirical projection vs IPCC FACTS process-model projection
- Identify which components drive the aggregate discrepancy
- **Output**: Figure 5 — 5-panel comparison (+ 6th panel showing sum)
- **Key finding**: Thermosteric and glaciers broadly agree; Antarctica shows largest discrepancy
- **Existing code**: Cell 25 already compares thermosteric vs IPCC ocean dynamics; `read_ipcc_ar6_component()` handles all components

---

## Phase 5: Uncertainty Attribution

### Step 5.1: Variance Decomposition by Component
- At each future time, compute variance contributed by each component
- Must account for shared temperature forcing (components are not independent — sample from same T trajectory)
- **Output**: Figure 6 — stacked area chart of variance contribution (2020–2100)
- **Key finding**: Near-term = thermosteric + glaciers; long-term = Antarctica dominates upper tail
- **Existing code**: `predictability_analysis.ipynb` cells 24-27 have aggregate variance decomposition

### Step 5.2: Distribution Shape at 2100
- Sample from each component posterior, sum, plot total distribution
- Show the right tail comes almost entirely from Antarctic uncertainty (A4 scenarios S3/S4)
- Central estimate is well-constrained by thermosteric + glaciers + Greenland
- **Output**: Histogram/KDE at 2100, decomposed by component

---

## Phase 6: Summary and Export

### Step 6.1: Total GMSL Projection Comparison (Money Plot)
- Three approaches on one figure:
  1. Component-sum semi-empirical (this paper)
  2. Aggregate semi-empirical (polynomial, from `predictability_analysis.ipynb`)
  3. IPCC AR6 process-model total
- **Output**: Figure 7
- **Key result**: Component-sum and aggregate agree in central estimate (validates decomposition); component-sum has physically motivated uncertainty; IPCC is lower, primarily because of Antarctica

### Step 6.2: Results Export
- All component calibration posteriors (summary statistics)
- Model selection criteria
- Projection summaries at 2050, 2100 for all SSPs
- Variance decomposition table
- **Output**: `component_decomposition_results.json`

---

## Figure List

| Fig | Content | Phase |
|-----|---------|-------|
| 1 | Component budget time series (Frederikse) showing compositional shift | Existing (cell 6) |
| 2 | Component-level rate vs T scatter plots with fits | Phase 1 |
| 3 | Fractional contribution to GMSL rate over time (stacked area) | Phase 3 |
| 4 | Sum of component fits ≈ aggregate quadratic | Phase 2 |
| 5 | Component-level projections vs IPCC component projections | Phase 4 |
| 6 | Projection uncertainty decomposition by component | Phase 5 |
| 7 | Total GMSL projection comparison (component-sum vs aggregate vs IPCC) | Phase 6 |

---

## New Utility Functions

### In `bayesian_dols.py`:
- Modify `fit_bayesian_level()` to accept `order` parameter (1 = linear, 2 = quadratic)
- `compute_model_comparison(result_linear, result_quadratic)` — WAIC/BIC

### In `slr_projections.py`:
- `project_tws_ensemble()` — TWS projection using IPCC FACTS
- `project_ais_ensemble()` — Antarctica projection using A4 framework
- `project_all_components_ensemble()` — orchestrator summing all components

### In new `component_analysis.py`:
- `compute_sliding_component_rates(fred_data, window_size=20)` — rates for each component
- `compute_fractional_contributions(component_rates)` — fractional time series

---

## Methodological Concerns

1. **Shared temperature forcing**: Components are fitted independently but share T. The sum-of-components uncertainty will be too narrow if cross-correlations are ignored. Must sample from the same T trajectory for all components.
2. **Frederikse data**: Single budget-closure dataset. AIS and TWS have the largest uncertainties. Budget constraint means errors propagate between components.
3. **TWS is non-climatic**: Dominated by groundwater extraction, not temperature. This *supports* the thesis — the aggregate GMSL–T relationship is misspecified by including a non-climate component, but the component model handles it correctly.
4. **Peripheral glacier double-counting**: GlaMBIE includes Greenland/Antarctic peripheral glaciers. IMBIE GrIS includes peripherals. Frederikse handles this internally, but external comparisons need consistent boundaries.
5. **Antarctica observation length**: Shortest record, largest structural uncertainty. The budget-derived residual (1900–2018) provides longer constraint than direct observations (1992–2020) but depends on other components being correct.

---

## Sequencing

**Critical path** (sequential):
1. Phase 1 → Phase 2 → Phase 3 (establishes the core argument)

**After critical path** (can be parallelized):
- Phase 4 (component projections)
- Phase 5 (uncertainty attribution) — requires Phase 4
- Phase 6 (summary) — requires everything

---

## TODO: Revisit

- [ ] Informative ΔT₀ prior for BPS cv mode (tighter σ_log = 0.2 instead of 0.5)
- [ ] Whether BPS trend constraint adds value in the component-sum framework (anchor individual components to observed trends?)
