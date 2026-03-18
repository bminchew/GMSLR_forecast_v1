# Analysis Plan: Paper 2 — The Sea Level Budget as an Emergent Constraint (Revised)

## Revision summary

This plan replaces the original 5-component version. Changes:
- Component vector: 5 → 7 (AIS split into WAIS, EAIS, APIS)
- All FACTS AIS modules provide WAIS/EAIS/APIS via `full_sample_components` / `dist_components`
- Frederikse et al. (2020) and Horwath et al. (2022) added as independent AIS observational constraints
- New analysis: within-Antarctica budget sub-constraint and information flow from EAIS to WAIS

---

## Component vector

Throughout this analysis, the component decomposition is:

| Index | Component | Abbreviation | Primary forcing |
|-------|-----------|-------------|-----------------|
| 0 | Ocean thermal expansion | TE | GMST |
| 1 | Glaciers and ice caps | GL | GMST |
| 2 | Greenland Ice Sheet | GrIS | GMST + North Atlantic SST |
| 3 | West Antarctic Ice Sheet | WAIS | CDW / ENSO |
| 4 | East Antarctic Ice Sheet | EAIS | Precipitation (SMB-dominated) |
| 5 | Antarctic Peninsula | APIS | Atmospheric warming (hydrofracture) |
| 6 | Terrestrial water storage | LWS | Anthropogenic + ENSO |

K = 7 components. The budget identity is:

$$r_{\text{total}}(t) = \sum_{i=0}^{6} r_i(t) + \epsilon(t)$$

An additional sub-budget holds by construction:

$$r_{\text{AIS,total}}(t) = r_{\text{WAIS}}(t) + r_{\text{EAIS}}(t) + r_{\text{APIS}}(t)$$

Both the global budget and the Antarctic sub-budget are exploitable constraints.

---

## Prerequisites

### Python environment

```bash
pip install numpy scipy pandas matplotlib seaborn xarray netCDF4 h5py tqdm
```

### Data files required

All data in `data/` relative to the project root.

1. **FACTS v1.0 Monte Carlo samples** (Kopp et al., 2023)
   - Source: Zenodo archive, or from existing pipeline
   - Required: `full_sample_components` and/or `dist_components` outputs for all workflows and SSPs
   - These provide WAIS, EAIS, APIS separately for all AIS modules (emulandice/AIS, larmip/AIS, bamber19/icesheets, deconto21/AIS)
   - Non-AIS components (TE, glaciers, GrIS, LWS) remain the same as the original plan
   - Place in `data/facts/`

2. **IMBIE reconciled ice sheet mass balance**
   - WAIS: already available as `imbie_west_antarctica_2021_mm.csv`
   - EAIS: same format, from http://imbie.org/data-downloads/
   - APIS: same format
   - GrIS: same format
   - All provide annual mass balance rate (mm/yr SLE) and cumulative (mm SLE), 1992–2020
   - Place in `data/imbie/`

3. **Frederikse et al. (2020) budget reconstruction**
   - Annual total GMSL and component contributions, 1900–2018
   - Provides total Antarctic contribution (AIS = WAIS+EAIS+APIS) as a budget component
   - The Antarctic estimate is derived from the budget closure methodology, not from direct mass balance observations before the satellite era
   - This extends AIS constraints back to 1900, far beyond IMBIE
   - Place in `data/frederikse/`

4. **Horwath et al. (2022) global sea-level budget**
   - Monthly budget, 1993–2016, with component uncertainties and error characterization
   - Provides total Antarctic contribution via gravimetry/altimetry/mass-flux reconciliation
   - Independent methodology from both IMBIE and Frederikse
   - Place in `data/horwath/`

5. **Satellite altimetry total GMSL**
   - AVISO/CNES or CSIRO, 1993–present
   - Place in `data/altimetry/`

6. **Argo-era thermosteric sea level** (for component-level validation)
   - NOAA/NCEI OHC 0-2000m
   - Place in `data/thermosteric/`

---

## Project structure

```
paper2_budget_constraint/
├── data/
│   ├── facts/           # FACTS MC samples with WAIS/EAIS/APIS
│   ├── imbie/           # WAIS, EAIS, APIS, GrIS separately
│   ├── frederikse/      # Budget reconstruction with total AIS
│   ├── horwath/         # Independent budget with total AIS
│   ├── altimetry/
│   └── thermosteric/
├── src/
│   ├── load_data.py
│   ├── rejection_sampler.py
│   ├── diagnostics.py
│   ├── figures.py
│   └── utils.py
├── outputs/
│   ├── figures/
│   └── tables/
├── tests/
│   └── test_sampler.py
└── run_all.py
```

---

## Step 1: Load and harmonize data

### File: `src/load_data.py`

#### 1a. Load FACTS MC samples at sub-continental Antarctic resolution

```python
def load_facts_samples() -> dict:
    """
    Returns dict keyed by (workflow, scenario, ais_module) containing:
        np.ndarray of shape (N_samples, 7, N_times)
        
    Component axis order: [TE, GL, GrIS, WAIS, EAIS, APIS, LWS]
    
    All values in mm relative to 1995-2014 baseline.
    N_times = decadal from 2020 to 2150 (14 time points).
    
    Sources:
        - TE, GL, GrIS, LWS: from standard FACTS component outputs
        - WAIS, EAIS, APIS: from full_sample_components or dist_components
          for the relevant AIS module
    """
```

**Critical verification**: For each sample, verify that WAIS + EAIS + APIS = total AIS (from the standard AIS output) to within numerical precision. Any discrepancy indicates a data alignment error.

If FACTS provides cumulative contributions, convert to rates:
```
rate(t) = (H(t+5) - H(t-5)) / 10  # centered finite difference, mm/yr
```

#### 1b. Load observational records

```python
def load_observations() -> pd.DataFrame:
    """
    Returns DataFrame with columns:
        year: int (1900-2023)
        
        # Total GMSL (multiple products)
        rate_total_frederikse: float (mm/yr), 1900-2018
        rate_total_frederikse_unc: float
        rate_total_altimetry: float (mm/yr), 1993-present
        rate_total_altimetry_unc: float
        rate_total_horwath: float (mm/yr), 1993-2016
        rate_total_horwath_unc: float
        
        # Antarctic sub-continental (IMBIE, 1992-2020)
        rate_wais: float (mm/yr)
        rate_wais_unc: float
        rate_eais: float
        rate_eais_unc: float
        rate_apis: float
        rate_apis_unc: float
        
        # Total AIS (multiple products for cross-validation)
        rate_ais_imbie: float       # = WAIS + EAIS + APIS from IMBIE
        rate_ais_imbie_unc: float
        rate_ais_frederikse: float  # Budget-derived, 1900-2018
        rate_ais_frederikse_unc: float
        rate_ais_horwath: float     # Independent estimate, 1993-2016
        rate_ais_horwath_unc: float
        
        # Other components
        rate_thermo: float, rate_thermo_unc: float    # Argo era (2005+)
        rate_glaciers: float, rate_glaciers_unc: float # GlaMBIE (2000+)
        rate_gris: float, rate_gris_unc: float         # IMBIE (1992+)
        rate_lws: float, rate_lws_unc: float           # Frederikse (1900+)
    """
```

**Cross-validation checks** (compute and print for the overlapping period 2002-2016):
1. `rate_ais_imbie ≈ rate_ais_frederikse ± combined_unc`
2. `rate_ais_imbie ≈ rate_ais_horwath ± combined_unc`
3. `rate_wais + rate_eais + rate_apis ≈ rate_ais_imbie` (internal consistency)
4. `sum(all_components) ≈ rate_total ± budget_closure_unc`

If Frederikse and Horwath AIS estimates disagree with IMBIE beyond combined uncertainties, this is itself a result worth reporting — it quantifies structural uncertainty in the observational AIS constraint.

---

## Step 2: Importance-weighted rejection sampling

### File: `src/rejection_sampler.py`

The sampler now operates on a 7-component vector and supports two constraint levels.

#### 2a. Global budget constraint (same logic as before, now K=7)

```python
def budget_constrained_sampling(
    component_samples: np.ndarray,   # shape (N, 7, T)
    obs_total_rate: np.ndarray,      # shape (T,)
    obs_total_unc: np.ndarray,       # shape (T,)
    sigma_budget: float = 0.3,
) -> tuple[np.ndarray, float]:
    """Same algorithm as original, but K=7 components."""
    total_implied = component_samples.sum(axis=1)  # (N, T)
    residuals = total_implied - obs_total_rate[None, :]
    sigma2 = obs_total_unc**2 + sigma_budget**2
    log_weights = -0.5 * np.sum(residuals**2 / sigma2[None, :], axis=1)
    log_weights -= log_weights.max()
    weights = np.exp(log_weights)
    ess = weights.sum()**2 / (weights**2).sum()
    return weights, ess
```

#### 2b. Antarctic sub-budget constraint (new)

An additional constraint using the independent AIS estimates from Frederikse and Horwath:

```python
def antarctic_sub_budget_constraint(
    component_samples: np.ndarray,   # shape (N, 7, T)
    obs_ais_rate: np.ndarray,        # shape (T,): Frederikse or Horwath total AIS
    obs_ais_unc: np.ndarray,         # shape (T,)
    sigma_ais_budget: float = 0.5,
) -> np.ndarray:
    """
    Additional log-likelihood from constraining WAIS+EAIS+APIS to match
    independently observed total AIS.
    
    Returns: log_weights_ais, shape (N,)
    """
    # WAIS=idx3, EAIS=idx4, APIS=idx5
    ais_implied = component_samples[:, 3, :] + component_samples[:, 4, :] + component_samples[:, 5, :]
    residuals = ais_implied - obs_ais_rate[None, :]
    sigma2 = obs_ais_unc**2 + sigma_ais_budget**2
    log_weights = -0.5 * np.sum(residuals**2 / sigma2[None, :], axis=1)
    return log_weights
```

#### 2c. Combined constraint

```python
def combined_constraint(
    component_samples: np.ndarray,
    obs_total_rate, obs_total_unc,
    obs_ais_rate, obs_ais_unc,
    sigma_budget=0.3, sigma_ais_budget=0.5,
) -> tuple[np.ndarray, float]:
    """
    Apply both global budget and Antarctic sub-budget constraints.
    The combined log-weight is the sum (product of likelihoods).
    """
    log_w_global = # ... from global budget
    log_w_ais = antarctic_sub_budget_constraint(...)
    log_w_combined = log_w_global + log_w_ais
    log_w_combined -= log_w_combined.max()
    weights = np.exp(log_w_combined)
    ess = weights.sum()**2 / (weights**2).sum()
    return weights, ess
```

#### 2d. Run configurations

Run the sampler for every combination of:

- **AIS module**: emulandice/AIS, larmip/AIS, bamber19/icesheets, deconto21/AIS
- **SSP scenario**: SSP1-2.6, SSP2-4.5, SSP5-8.5
- **Constraint type**:
  - "None": no constraint (independent baseline)
  - "Global only": global budget constraint using total GMSL
  - "AIS only": Antarctic sub-budget constraint using Frederikse/Horwath AIS
  - "Combined": global + Antarctic sub-budget
- **Constraint period**:
  - "Pre-satellite" (1900-1992): Frederikse total + Frederikse AIS only
  - "Early satellite" (1993-2002): altimetry total + Horwath AIS
  - "GRACE era" (2002-2020): altimetry + IMBIE sub-continental + Frederikse/Horwath AIS
  - "Full" (1900-2020): all available
- **AIS observational product** (for the sub-budget): Frederikse, Horwath, IMBIE (where overlapping)
- **sigma_budget**: 0.1, 0.3, 0.5, 1.0 mm/yr

Total configurations: 4 modules × 3 SSPs × 4 constraint types × 4 periods × 3 AIS products × 4 sigma values. This is a large grid. In practice, the primary runs use sigma_budget=0.3 and "Full" period; the others are sensitivity analyses that can be run in parallel.

For each run, store: importance weights, ESS, weighted means and variances per component at each time step.

---

## Step 3: Compute diagnostics

### File: `src/diagnostics.py`

#### 3a. Variance ratio (now 7 components)

$$\rho_j(t) = \frac{\text{Var}_w[r_j(t)]}{\text{Var}[r_j(t)]} \quad \text{for } j \in \{\text{TE, GL, GrIS, WAIS, EAIS, APIS, LWS}\}$$

The key comparison is now between $\rho_{\text{WAIS}}$ (which should show large variance reduction) and $\rho_{\text{EAIS}}$ (which should show moderate reduction because EAIS is comparatively well-constrained by SMB observations).

#### 3b. Induced pairwise correlations (now 7×7 matrix)

The 7×7 correlation matrix under the budget constraint. Key off-diagonal entries to examine:

- **Cor[WAIS, TE | budget]**: Should be strongly negative. The dominant information flow path: well-constrained TE tightens the total residual, which constrains WAIS.
- **Cor[WAIS, EAIS | budget]**: Should be negative, and additionally strengthened by the Antarctic sub-budget constraint. Even without the global budget, the sub-budget (WAIS + EAIS + APIS = AIS_obs) creates a collider at the sub-continental level.
- **Cor[EAIS, APIS | budget]**: Also negative via the sub-budget collider.
- **Cor[TE, EAIS | budget]**: Weak — EAIS is already moderately constrained, so the budget adds less.

#### 3c. Marginal vs. sub-budget constraint comparison

This is a new diagnostic specific to the 7-component decomposition. Compare:

1. $\rho_{\text{WAIS}}$ under "Global only" constraint
2. $\rho_{\text{WAIS}}$ under "AIS only" constraint (Frederikse/Horwath sub-budget)
3. $\rho_{\text{WAIS}}$ under "Combined" constraint

If the AIS sub-budget from Frederikse/Horwath adds substantial information beyond the global budget alone, this demonstrates the value of multiple independent budget-closure products operating at different levels of the component hierarchy.

#### 3d. Cross-validation of AIS observational products

During the overlapping period (2002-2016), three independent AIS estimates are available:
- IMBIE (direct reconciled mass balance)
- Frederikse (budget-derived)
- Horwath (independent budget)

Compute: weighted posterior WAIS+EAIS+APIS under each product as the sub-budget constraint. If the posteriors are consistent, the result is robust. If they differ, the structural uncertainty in the observational constraint itself becomes a quantifiable result.

#### 3e. Information gain (KL divergence)

Same as original plan but now computed for all 7 components.

---

## Step 4: Figures

### File: `src/figures.py`

#### Figure 1: Schematic DAG (TikZ — revised)

Two-level DAG showing:
- Top level: 7 component rate nodes → deterministic sum → total GMSL observation (global budget collider)
- Bottom level (inset or zoomed): WAIS, EAIS, APIS → deterministic sum → AIS total observation (Antarctic sub-budget collider)
- Dashed arrows indicating induced dependencies in both colliders
- Caption explaining the two-level collider structure

#### Figure 2: Variance ratio heatmap (now 7 rows)

```
Layout: 3 panels (SSP1-2.6, SSP2-4.5, SSP5-8.5)
  Each panel: heatmap with:
  - x-axis: time (2020-2150, decadal)
  - y-axis: component (TE, GL, GrIS, WAIS, EAIS, APIS, LWS)
  - Color: variance ratio ρ_j(t), scale 0 to 1
  - Horizontal line separating the three Antarctic components from the rest

Use: emulandice/AIS module, Combined constraint, sigma_budget=0.3.
```

Central expectation: WAIS row has the smallest ρ (most reduction). EAIS and APIS have moderate ρ. TE has ρ ≈ 1.

#### Figure 3: Prior vs. posterior distributions for WAIS (replaces "AIS")

```
Layout: 3×1 panels (SSP1-2.6, SSP2-4.5, SSP5-8.5)
  Each panel:
  - KDE of WAIS contribution at t=2100
  - Gray fill: prior (unweighted)
  - Red line: posterior under global budget only
  - Dark red line: posterior under combined (global + AIS sub-budget)
  - Vertical dashed lines: 17th/83rd for each distribution
  - Inset text: ρ_WAIS for each constraint level
```

The visual comparison between the two posterior lines quantifies the added value of the Frederikse/Horwath AIS sub-budget constraint.

#### Figure 4: Induced correlation matrix (now 7×7)

```
Layout: 2×3 grid (same structure as original, now 7×7 matrices)
  Columns: SSP1-2.6, SSP2-4.5, SSP5-8.5
  Top row: Prior correlation matrix at t=2100
  Bottom row: Posterior (Combined constraint) correlation matrix at t=2100
  
Labels: TE, GL, GrIS, WAIS, EAIS, APIS, LWS
Annotate the WAIS-EAIS cell — this should show strong negative correlation
from the Antarctic sub-budget collider.
```

#### Figure 5: Constraint level comparison for WAIS (new figure)

```
Layout: Single panel
  - x-axis: time (2020-2150)
  - y-axis: ρ_WAIS(t)
  - Lines:
    - "Global budget only" (dashed)
    - "AIS sub-budget only" (dotted)
    - "Combined" (solid)
  - One set of lines per SSP (color-coded)

Purpose: Quantify the marginal value of each constraint level for WAIS.
```

#### Figure 6: Sensitivity to AIS module (now showing WAIS)

```
Layout: 4 panels (one per AIS module)
  Each: WAIS posterior KDE at t=2100 under SSP5-8.5
  Gray: prior
  Colored: posterior (Combined constraint)
  Text: ρ_WAIS, ESS
```

#### Figure 7: Within-Antarctica information flow (new figure)

```
Layout: Single panel (or 3 panels for 3 SSPs)
  For t=2100:
  - Show the EAIS and APIS prior distributions
  - Show the WAIS prior distribution
  - Show how conditioning on AIS_total (Frederikse/Horwath) tightens all three
  - Then show how additionally conditioning on total GMSL further tightens WAIS

Purpose: Isolate the within-Antarctica collider mechanism from the global mechanism.
Demonstrates that even if the global budget were unavailable, the AIS sub-budget
from Frederikse/Horwath constrains WAIS through EAIS.
```

#### Figure 8: Cross-validation of AIS observational products (new figure)

```
Layout: Single panel
  - x-axis: year (1992-2020)
  - y-axis: total AIS rate (mm/yr)
  - Lines/bands: IMBIE (with uncertainty), Frederikse (with uncertainty), 
    Horwath (with uncertainty)
  - Shading: common overlap period (2002-2016)
  - Text: inter-product agreement statistics

Purpose: Establish the consistency (or inconsistency) of the three independent
AIS observational products. Inconsistency quantifies structural uncertainty
in the observational constraint itself.
```

#### Figure 9: Budget residual time series (revised from original Figure 7)

```
Layout: Single panel, 1900-2020
  - Budget residual = observed total - sum(TE + GL + GrIS + LWS)
  - This residual = WAIS + EAIS + APIS (total AIS)
  - Overlay: Frederikse AIS (1900-2018), Horwath AIS (1993-2016), 
    IMBIE total AIS (1992-2020)
  - Additionally overlay: IMBIE WAIS alone (to show WAIS dominates the residual)
  - Shading for eras: tide-gauge, early satellite, GRACE

Purpose: Show that the budget residual is consistent with direct AIS observations,
and that WAIS dominates the residual signal.
```

#### Table 1: Summary statistics (7 components)

| Component | Prior σ | Post. σ (global) | Post. σ (combined) | ρ (global) | ρ (combined) |
|-----------|---------|-------------------|---------------------|------------|--------------|
| TE | ... | ... | ... | ... | ... |
| GL | ... | ... | ... | ... | ... |
| GrIS | ... | ... | ... | ... | ... |
| WAIS | ... | ... | ... | ... | ... |
| EAIS | ... | ... | ... | ... | ... |
| APIS | ... | ... | ... | ... | ... |
| LWS | ... | ... | ... | ... | ... |

At t=2100 for each SSP, using emulandice/AIS, sigma_budget=0.3.

#### Table 2: AIS observational product cross-validation

| Period | IMBIE total | Frederikse | Horwath | Agreement |
|--------|------------|------------|---------|-----------|
| 1993-2002 | ... | ... | ... | z-score |
| 2002-2016 | ... | ... | ... | z-score |

---

## Step 5: Validation checks

### File: `tests/test_sampler.py`

All original tests apply, updated for K=7. Additional tests:

#### Test 5: Antarctic sub-budget consistency

Verify that for the combined constraint, the weighted mean of WAIS+EAIS+APIS matches the observed AIS total (from whichever product is used as the constraint) to within the tolerance sigma_ais_budget.

#### Test 6: Two-level collider correctness

Construct a synthetic example with 7 independent Gaussians, impose both a global sum constraint and a sub-sum constraint on components 3-5. Verify the analytic conditional distribution is recovered.

For the Gaussian case with two linear constraints:
- Global: $\sum_{i=0}^{6} r_i = R_{\text{total}} \pm \sigma_R$
- Antarctic: $r_3 + r_4 + r_5 = R_{\text{AIS}} \pm \sigma_A$

The conditional is multivariate Gaussian with covariance that reflects both constraints. Derive analytically and compare to MC output.

#### Test 7: Constraint ordering invariance

Verify that "global then AIS" and "AIS then global" produce identical combined weights (up to MC noise). The product of likelihoods is commutative.

---

## Step 6: Master script

```python
"""
Master script for Paper 2 analysis (revised, 7-component).

Steps:
1. Load FACTS samples with WAIS/EAIS/APIS decomposition
2. Load observations: IMBIE (sub-continental), Frederikse, Horwath (total AIS)
3. Cross-validate AIS observational products
4. Run rejection sampler: global constraint only
5. Run rejection sampler: AIS sub-budget constraint only
6. Run rejection sampler: combined constraint
7. Sensitivity: across AIS modules, SSPs, sigma values
8. Compute all diagnostics
9. Generate all figures and tables
10. Run validation tests
11. Summary report
"""
```

---

## Estimated compute time

- Data loading: ~2 min
- Sampler (full grid): ~45 min on 8 cores
- Diagnostics: ~5 min
- Figures: ~3 min
- Total: ~55 min

The grid is larger than the original plan due to the additional constraint types and AIS observational products. Restrict the full sensitivity grid to a nightly run; use the primary configuration (emulandice, Combined, Full period, sigma=0.3) for interactive development.

---

## Critical decision points

1. **If Frederikse or Horwath provide WAIS/EAIS/APIS separately (not just total AIS)**: The sub-budget constraint can be applied at each sub-continental component individually rather than only at their sum. Check the data files. If sub-continental is available, add it as an additional constraint level.

2. **If the three AIS observational products (IMBIE, Frederikse, Horwath) are inconsistent during the overlap period**: Report this as a result, not a failure. Compute the variance ratio $\rho_{\text{WAIS}}$ separately using each product as the sub-budget constraint. The spread in $\rho_{\text{WAIS}}$ across products quantifies the sensitivity of the budget-constraint information gain to structural uncertainty in the observational AIS estimate.

3. **If EAIS prior in FACTS is very narrow (already well-constrained)**: The budget constraint will add little to EAIS. The information flow will be predominantly TE → (global residual) → WAIS, with EAIS acting as a "pass-through" that is already pinned. This is fine — it means the sub-budget constraint from Frederikse/Horwath is primarily constraining WAIS via the EAIS-WAIS-APIS collider, which is the correct physics.

4. **If ESS drops below 100 for the Combined constraint**: The two constraints together may be too tight for the prior. Options: increase sigma_budget or sigma_ais_budget; report which constraint is driving the tension; interpret the low ESS as indicating prior-likelihood inconsistency (the FACTS prior distribution is inconsistent with the observations at the specified tolerance).

---

## Output checklist

- [ ] 9 figures (1 TikZ schematic, 8 computed)
- [ ] 2 tables
- [ ] Validation test suite passing (7 tests)
- [ ] AIS cross-validation statistics
- [ ] Summary statistics for all sampler configurations
- [ ] README
