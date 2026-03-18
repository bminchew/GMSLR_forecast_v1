# Analysis Plan: Paper 4 — Time-Horizon Dependence of SLR Projection Methodology (Revised)

## Revision summary

This plan replaces the original 5-component version. Changes:
- Component vector: 5 → 7 (AIS split into WAIS, EAIS, APIS)
- Variance decomposition now shows WAIS as the dominant long-horizon uncertainty
- Crossover time computed for WAIS specifically, not total AIS
- Frederikse (2020) and Horwath (2022) AIS estimates used for extended observational validation
- The within-Antarctica cancellation (Paper 3) appears as a confound in the variance decomposition at the macro-component level — resolving it at sub-continental level gives cleaner regime boundaries

---

## Component vector

Same 7-component decomposition as Papers 2 and 3:

| Index | Component | Abbreviation |
|-------|-----------|-------------|
| 0 | Ocean thermal expansion | TE |
| 1 | Glaciers and ice caps | GL |
| 2 | Greenland Ice Sheet | GrIS |
| 3 | West Antarctic Ice Sheet | WAIS |
| 4 | East Antarctic Ice Sheet | EAIS |
| 5 | Antarctic Peninsula | APIS |
| 6 | Terrestrial water storage | LWS |

---

## Prerequisites

### Data files required

1. **FACTS v1.0 MC samples** with WAIS/EAIS/APIS from `full_sample_components` / `dist_components`, all 7 workflows, SSP1-2.6, SSP2-4.5, SSP5-8.5
2. **Satellite altimetry total GMSL**: monthly, 1993–present
3. **FaIR temperature projections**: GMST trajectories per SSP
4. **DOLS coefficients**: α₀ and dα/dT from the robustness ensemble
5. **Frederikse et al. (2020)**: total GMSL + total AIS, 1900–2018
6. **Horwath et al. (2022)**: total GMSL + total AIS, 1993–2016
7. **IMBIE**: WAIS, EAIS, APIS separately, 1992–2020
8. **Pre-satellite GMSL reconstruction**: Frederikse or Dangendorf, annual

---

## Project structure

```
paper4_time_horizon/
├── data/
│   ├── facts/           # Full MC samples with WAIS/EAIS/APIS
│   ├── altimetry/
│   ├── fair/
│   ├── dols/
│   ├── frederikse/
│   ├── horwath/
│   ├── imbie/
│   └── gmsl_reconstructions/
├── src/
│   ├── load_data.py
│   ├── trend_extrapolation.py
│   ├── forecast_skill.py
│   ├── variance_decomposition.py
│   ├── regime_classification.py
│   ├── figures.py
│   └── utils.py
├── outputs/
│   ├── figures/
│   └── tables/
├── tests/
│   └── test_skill.py
└── run_all.py
```

---

## Step 1: Load data

### File: `src/load_data.py`

#### 1a. Load FACTS MC samples at sub-continental resolution

```python
def load_facts_samples_7component() -> dict:
    """
    Returns dict keyed by (workflow, ssp) containing:
        np.ndarray of shape (N_samples, 7, N_times)
        
    Component axis: [TE, GL, GrIS, WAIS, EAIS, APIS, LWS]
    
    Extract WAIS/EAIS/APIS from full_sample_components for each AIS module.
    Verify WAIS + EAIS + APIS = total AIS for each sample.
    Convert cumulative to rates via finite differences.
    """
```

Store rate arrays:
```
facts_rates[workflow][ssp] = np.ndarray of shape (N_samples, 8, N_times)
# 8 = 7 components + total (= sum of components)
# Component order: [total, TE, GL, GrIS, WAIS, EAIS, APIS, LWS]
```

#### 1b. Load observational records

Same structure as Paper 2/3 load functions. Key addition for Paper 4:

- Frederikse total AIS provides an AIS constraint extending to 1900, which is valuable for the trend extrapolation analysis at long record lengths.
- Horwath total AIS provides an independent satellite-era estimate.

#### 1c. Load FaIR temperatures

Same as original plan.

---

## Step 2: Trend extrapolation models

### File: `src/trend_extrapolation.py`

Same three model classes as original (LinearTrend, QuadraticTrend, DOLSExtrapolation). No changes needed here — the trend models operate on total GMSL, not on individual components.

---

## Step 3: Pseudo-forecast skill evaluation

### File: `src/forecast_skill.py`

Same as original. The forecast skill analysis operates on total GMSL and is agnostic to the component decomposition. No changes needed.

One addition: for the extended validation window, use Frederikse total GMSL (1900-2018) to provide additional origin years beyond the satellite era. This gives lead times up to 25 years (origin 1993, validation to 2018) and allows testing whether longer fitting records improve forecast skill.

---

## Step 4: Variance decomposition of FACTS projections

### File: `src/variance_decomposition.py`

This step changes substantially. The variance decomposition now has 7 components, revealing the WAIS-specific dominance.

#### 4a. Within-scenario component variance fractions (7 components)

```python
def component_variance_fractions_7(
    mc_samples: np.ndarray,  # (N, 7, T): 7 components
) -> np.ndarray:             # (8, T): 7 components + interaction
    """
    Compute f_i(t) = Var[r_i(t)] / Var[sum_i r_i(t)] for i in 0..6
    and f_interaction(t) = 1 - sum_i f_i(t)
    """
    total = mc_samples.sum(axis=1)           # (N, T)
    total_var = np.var(total, axis=0)         # (T,)
    component_vars = np.var(mc_samples, axis=0)  # (7, T)
    fractions = component_vars / total_var[None, :]  # (7, T)
    interaction = 1.0 - fractions.sum(axis=0)    # (T,)
    return np.vstack([fractions, interaction[None, :]])  # (8, T)
```

#### 4b. Compare 7-component vs. 5-macro-component decomposition

Also compute the variance decomposition at the 5-macro-component level (TE, GL, GrIS, AIS_total, LWS) by summing the three Antarctic components before computing variances:

```python
def component_variance_fractions_5(mc_samples: np.ndarray) -> np.ndarray:
    """5-macro-component version for comparison."""
    macro = np.zeros((mc_samples.shape[0], 5, mc_samples.shape[2]))
    macro[:, 0, :] = mc_samples[:, 0, :]  # TE
    macro[:, 1, :] = mc_samples[:, 1, :]  # GL
    macro[:, 2, :] = mc_samples[:, 2, :]  # GrIS
    macro[:, 3, :] = mc_samples[:, 3:6, :].sum(axis=1)  # AIS total
    macro[:, 4, :] = mc_samples[:, 6, :]  # LWS
    return component_variance_fractions_7(macro)  # reuse function with K=5
```

The critical comparison: in the 5-component decomposition, $f_{\text{AIS,total}}$ aggregates the three Antarctic sub-components. Because EAIS and APIS have relatively small variance compared to WAIS, $f_{\text{AIS,total}} < f_{\text{WAIS}} + f_{\text{EAIS}} + f_{\text{APIS}}$ in general (the covariance between EAIS and WAIS may be negative in the FACTS samples if the modules encode an inverse precipitation-dynamics relationship). Conversely, the interaction term at the 5-component level absorbs the within-Antarctica covariance structure.

Key diagnostic: compute $f_{\text{WAIS}}$ and $f_{\text{AIS,total}}$ at each time step. The ratio $f_{\text{WAIS}} / f_{\text{AIS,total}}$ tells us what fraction of the total Antarctic variance comes from WAIS specifically. Expected: > 0.8 under SSP5-8.5 at 2100 (WAIS dominates Antarctic uncertainty).

#### 4c. Across-scenario variance (same as original)

No change. Computed from total GMSL, not from individual components.

#### 4d. Crossover time — now for WAIS specifically

```python
def crossover_time_wais(
    variance_fractions_7: np.ndarray,  # (8, T)
    times: np.ndarray,
) -> float:
    """
    Find earliest time at which f_WAIS(t) > f_i(t) for all i != WAIS.
    WAIS is index 3.
    """
    wais_frac = variance_fractions_7[3, :]
    # Compare against each non-WAIS, non-interaction component
    other_indices = [0, 1, 2, 4, 5, 6]  # TE, GL, GrIS, EAIS, APIS, LWS
    other_max = np.max(variance_fractions_7[other_indices, :], axis=0)
    exceeds = wais_frac > other_max
    if not exceeds.any():
        return np.inf
    idx = np.argmax(exceeds)
    # Interpolate
    if idx == 0:
        return times[0]
    # Linear interpolation between times[idx-1] and times[idx]
    f_prev = wais_frac[idx-1] - other_max[idx-1]
    f_curr = wais_frac[idx] - other_max[idx]
    frac = -f_prev / (f_curr - f_prev)
    return times[idx-1] + frac * (times[idx] - times[idx-1])
```

Also compute $t^*_{\text{AIS,total}}$ using the 5-component decomposition for comparison. Expected: $t^*_{\text{WAIS}} < t^*_{\text{AIS,total}}$ because the 7-component decomposition isolates the true deep-uncertainty source and is not diluted by the comparatively well-constrained EAIS and APIS.

This difference is a result: it shows that the time at which deep uncertainty dominates is earlier than traditional analyses suggest, because those analyses mask WAIS behind the aggregate AIS number.

#### 4e. Internal variability (same as original)

No change.

---

## Step 5: Regime classification

### File: `src/regime_classification.py`

#### 5a. Regime boundaries (revised)

- **Regime I → II boundary (t₁)**: Same as original (based on forecast skill).

- **Regime II → III boundary**: Now compute two versions:
  - $t_2^{\text{WAIS}}$: crossover time when WAIS dominates (7-component)
  - $t_2^{\text{AIS}}$: crossover time when total AIS dominates (5-component)
  
  Report both. The difference $t_2^{\text{AIS}} - t_2^{\text{WAIS}}$ quantifies how much the aggregation obscures the onset of the deep-uncertainty regime.

#### 5b. Meta-uncertainty (revised)

Compute the spread in $t_2^{\text{WAIS}}$ across workflows. This is more meaningful than $t_2^{\text{AIS}}$ because it targets the specific physical process (WAIS marine ice-sheet dynamics) that drives the uncertainty.

---

## Step 6: Figures

### File: `src/figures.py`

#### Figure 1: Forecast skill comparison

Same as original. No changes (this operates on total GMSL).

#### Figure 2: Variance decomposition stacked area plot — 7 components (central figure)

```
Layout: 3 panels (SSP1-2.6, SSP2-4.5, SSP5-8.5)
  - x-axis: time (2020-2150)
  - y-axis: fraction of within-scenario variance (0 to 1)
  - Stacked areas: TE (blue), GL (green), GrIS (orange), 
    WAIS (dark red), EAIS (light red/pink), APIS (salmon),
    LWS (purple), Interaction (gray)
  - Scenario variance as black hatched overlay
  
Use a red color family for the three Antarctic components so they form
a visual group while remaining individually distinguishable.
  
Annotate: vertical dashed line at t*_WAIS.
```

This is the paper's central figure. The key visual: WAIS (dark red) grows to dominate the stacked area at long horizons, while EAIS and APIS remain thin slivers. Under SSP1-2.6, TE may remain comparable to WAIS throughout.

#### Figure 3: 7-component vs. 5-component comparison (new)

```
Layout: 2 panels side by side, both under SSP5-8.5
  Left: 7-component decomposition (WAIS/EAIS/APIS separate)
  Right: 5-component decomposition (total AIS aggregated)
  
Same stacked area format.
Annotate t*_WAIS on left, t*_AIS on right.
Arrow or annotation showing: "aggregation delays apparent onset 
of deep-uncertainty regime by ~X years."
```

This figure makes the paper's secondary argument: the sub-continental decomposition reveals the onset of deep uncertainty earlier than the traditional decomposition.

#### Figure 4: Crossover time sensitivity (revised)

```
Layout: Single panel
  - x-axis: SSP scenario (categorical)
  - y-axis: crossover year
  - Points: one per workflow, jittered
  - Color: by AIS module
  - Two sets of points per workflow:
    - Circles: t*_WAIS (7-component)
    - Triangles: t*_AIS (5-component)
  - Horizontal dashed lines at key reference years (2050, 2100)
```

#### Figure 5: Three-regime schematic (TikZ)

Same as original. Update the Regime III label to specify "WAIS structural uncertainty" rather than "AIS structural uncertainty."

#### Figure 6: Variance decomposition across workflows (revised)

```
Layout: 7 panels (one per FACTS workflow), SSP5-8.5
  Each: 7-component stacked area plot
  Annotate t*_WAIS
  
Purpose: Show that WAIS dominance is robust across workflows, but the
crossover time varies with AIS module choice.
```

#### Figure 7: Internal variability vs. forced signal (same as original)

No change.

#### Figure 8: WAIS variance fraction evolution (new)

```
Layout: Single panel
  - x-axis: time (2020-2150)
  - y-axis: f_WAIS(t) — fraction of within-scenario variance from WAIS
  - Lines: one per workflow, one per SSP (use linestyle for SSP, color for workflow)
  
Purpose: Direct visualization of when and how fast WAIS comes to dominate.
Cleaner than reading this from the stacked area plots.
```

#### Table 1: Forecast skill summary

Same as original.

#### Table 2: Crossover times — both levels (revised)

| Workflow | AIS module | t*_WAIS (7-comp) | t*_AIS (5-comp) | Δt = t*_AIS - t*_WAIS |
|----------|-----------|-----------------|-----------------|----------------------|

For each SSP. The Δt column quantifies the "aggregation delay."

| SSP | Workflow | t*_WAIS | t*_AIS | Δt |
|-----|----------|--------|-------|-----|
| SSP1-2.6 | 1e | ... | ... | ... |
| ... | ... | ... | ... | ... |
| SSP5-8.5 | 4 | ... | ... | ... |

#### Table 3: Regime summary (revised)

| Regime | Time horizon | Dominant uncertainty | Best methodology | Validation |
|--------|-------------|---------------------|-----------------|------------|
| I | 0–15 yr | Internal variability | Trend extrap. | Satellite obs. |
| II | 15 yr – t*_WAIS | Scenario spread | Calibrated models | Cross-model |
| III | t*_WAIS onward | WAIS structural | Deep UQ | Internal consistency |

Note: t*_WAIS, not t*_AIS.

---

## Step 7: Validation checks

All original tests apply (updated for 7 components). Additional:

#### Test 6: WAIS crossover vs. AIS crossover ordering

Verify $t^*_{\text{WAIS}} \leq t^*_{\text{AIS}}$ for every workflow and SSP. This must hold because WAIS is a subset of AIS: the component that dominates when decomposed cannot have a later crossover than the aggregate. If this fails, there is a bug in the variance computation.

Proof sketch: $\text{Var}[\text{AIS}_{\text{total}}] = \text{Var}[\text{WAIS}] + \text{Var}[\text{EAIS}] + \text{Var}[\text{APIS}] + 2\text{Cov}[\text{WAIS}, \text{EAIS}] + \ldots$. If the covariances are negative (EAIS gain partially anticorrelated with WAIS loss), then $\text{Var}[\text{AIS}_{\text{total}}] < \text{Var}[\text{WAIS}]$, and $f_{\text{AIS,total}} < f_{\text{WAIS}}$, meaning AIS_total crosses the TE threshold later than WAIS. If covariances are positive, $f_{\text{AIS,total}} > f_{\text{WAIS}}$, but AIS_total also competes against fewer opponents in the 5-component decomposition. The ordering depends on the specific numbers. It is not guaranteed by construction. If it fails for some configuration, report this as a result (it means within-Antarctica covariance is positive and strong enough to make the aggregate AIS dominate earlier than WAIS alone).

#### Test 7: Variance decomposition sums to 1

Verify $\sum_{i=0}^{6} f_i(t) + f_{\text{interaction}}(t) = 1.0$ at every time step, for both the 7-component and 5-component versions.

---

## Step 8: Master script

```python
"""
Master script for Paper 4 analysis (revised, 7-component).

Steps:
1. Load all data (FACTS 7-component, altimetry, FaIR, DOLS, IMBIE, Frederikse, Horwath)
2. Fit trend extrapolation models at all origin years
3. Compute pseudo-forecast RMSE and coverage
4. Compute 7-component variance decompositions for all workflows × SSPs
5. Compute 5-component variance decompositions for comparison
6. Compute crossover times (t*_WAIS and t*_AIS) for all configurations
7. Classify regimes
8. Generate all figures and tables
9. Run validation tests
10. Summary report
"""
```

---

## Dependencies on other papers

- Paper 3's within-Antarctica cancellation result (EAIS gain masking WAIS loss in total AIS) is the direct motivation for why this paper must use the 7-component decomposition. Paper 4 can cite Paper 3 and state: "the aggregate AIS variance fraction conflates the dominant uncertainty source (WAIS marine ice-sheet dynamics) with comparatively well-constrained components (EAIS surface mass balance, APIS atmospheric-warming-driven retreat). Paper 3 showed that this aggregation hides compensating errors; here we show it also delays the apparent onset of the deep-uncertainty regime."

- Paper 2's budget constraint result operates more sharply on WAIS when the 7-component decomposition is used. Paper 4 can note that the budget constraint's information gain for WAIS (Paper 2) is concentrated in Regimes I and II, where the observational constraint is strong, and vanishes in Regime III, where the historical record cannot constrain the WAIS tipping threshold.

---

## Output checklist

- [ ] 8 figures (7 computed + 1 TikZ schematic)
- [ ] 3 tables
- [ ] Validation test suite passing (7 tests)
- [ ] 7-component variance decomposition arrays for all 21 configurations
- [ ] 5-component variance decomposition arrays for comparison
- [ ] Crossover times (t*_WAIS and t*_AIS) for all configurations
- [ ] Aggregation delay Δt = t*_AIS - t*_WAIS statistics
- [ ] Forecast skill metrics
- [ ] README
