# Analysis Plan: Paper 3 — Compensating Errors in Component-Level Projections (Revised)

## Revision summary

This plan replaces the original 5-component version. Changes:
- Component vector: 5 → 7 (AIS split into WAIS, EAIS, APIS)
- All FACTS AIS modules provide WAIS/EAIS/APIS via `full_sample_components` / `dist_components`
- Frederikse et al. (2020) and Horwath et al. (2022) added as independent AIS observational benchmarks
- New analysis: within-Antarctica compensating errors (EAIS gain partially masks WAIS loss)
- Cancellation index now computed at both the global level and the Antarctic sub-level

---

## Component vector

Same 7-component decomposition as Paper 2:

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

1. **FACTS v1.0 projections** with WAIS/EAIS/APIS from `full_sample_components` / `dist_components`
2. **IMBIE**: WAIS, EAIS, APIS, GrIS — reconciled mass balance, 1992–2020
3. **Frederikse et al. (2020)**: total GMSL + component contributions including total AIS, 1900–2018
4. **Horwath et al. (2022)**: independent budget with total AIS estimate, 1993–2016
5. **Argo-era thermosteric**: NOAA/NCEI OHC 0-2000m
6. **GlaMBIE**: glacier mass balance consensus, 2000–2023
7. **Satellite altimetry**: total GMSL, 1993–present
8. **DOLS coefficients**: from existing analysis

---

## Project structure

```
paper3_compensating_errors/
├── data/
│   ├── facts/
│   ├── imbie/           # WAIS, EAIS, APIS, GrIS separately
│   ├── frederikse/      # Budget-derived total AIS
│   ├── horwath/         # Independent total AIS
│   ├── thermosteric/
│   ├── glaciers/
│   ├── altimetry/
│   └── dols/
├── src/
│   ├── load_data.py
│   ├── compute_biases.py
│   ├── cancellation_index.py
│   ├── extrapolation.py
│   ├── figures.py
│   └── utils.py
├── outputs/
│   ├── figures/
│   └── tables/
├── tests/
│   └── test_biases.py
└── run_all.py
```

---

## Step 1: Load and harmonize data

### File: `src/load_data.py`

#### 1a. Observational rate time series

```python
def load_observational_rates() -> pd.DataFrame:
    """
    Returns DataFrame with columns:
        year: int (1900-2023)
        
        # Total
        rate_total, rate_total_unc (mm/yr)
        
        # Thermosteric (Argo era, 2005+)
        rate_thermo, rate_thermo_unc
        
        # Glaciers (GlaMBIE, 2000+)
        rate_glaciers, rate_glaciers_unc
        
        # Greenland (IMBIE, 1992+)
        rate_gris, rate_gris_unc
        
        # Antarctic sub-continental (IMBIE, 1992+)
        rate_wais, rate_wais_unc
        rate_eais, rate_eais_unc
        rate_apis, rate_apis_unc
        
        # Total AIS from three independent products
        rate_ais_imbie, rate_ais_imbie_unc     # = WAIS+EAIS+APIS
        rate_ais_frederikse, rate_ais_frederikse_unc  # budget-derived, 1900-2018
        rate_ais_horwath, rate_ais_horwath_unc         # independent, 1993-2016
        
        # LWS
        rate_lws, rate_lws_unc
    """
```

**Processing notes for IMBIE**:
- IMBIE reports mass change. Convert to SLR contribution: positive mass loss → positive rate_SLR.
- WAIS rate from `imbie_west_antarctica_2021_mm.csv`: use the "Mass balance (mm/yr)" column directly (already in mm SLE/yr). Verify sign convention: negative mass balance = mass loss = positive SLR contribution. Flip sign if needed.
- Compute 5-year running means for rate time series to suppress interannual noise.

**Processing notes for Frederikse AIS**:
- Frederikse provides Antarctica as a component of the budget. This is a budget-derived estimate, not a direct measurement pre-satellite.
- During 1900-1992, this is the only available AIS rate estimate. It inherits biases from all other components in the Frederikse budget. Document this caveat.
- During 1993-2020, it can be cross-validated against IMBIE and Horwath.

#### 1b. FACTS projected rates at sub-continental resolution

```python
def load_facts_rates_7component() -> dict:
    """
    Returns dict keyed by (module_combination, ssp) with values:
        pd.DataFrame with columns:
            year, component, rate_median, rate_p17, rate_p83, rate_p05, rate_p95
            
    Components: TE, GL, GrIS, WAIS, EAIS, APIS, LWS
    
    For each AIS module, extract WAIS/EAIS/APIS from full_sample_components.
    Compute rates from cumulative via finite differences.
    """
```

---

## Step 2: Compute component-level biases

### File: `src/compute_biases.py`

#### 2a. Bias computation (now 7 components)

Same algorithm as original:
$$\text{bias}_i(t) = \hat{r}_i^{\text{FACTS}}(t) - \hat{r}_i^{\text{obs}}(t)$$

But now computed for all 7 components separately. The Antarctic components have distinct expected bias patterns:

- **WAIS bias**: Expected to be negative (FACTS underestimates WAIS mass loss). ISMIP6 models systematically underestimate grounding-line retreat rates, especially for PIG and Thwaites. The Martin et al. (2025) n=3 initialization bias amplifies this.
- **EAIS bias**: Could be positive (FACTS overestimates EAIS mass gain from precipitation) or near zero. EAIS is near steady state observationally.
- **APIS bias**: Expected to be small. APIS contribution is modest and comparatively well-constrained.

#### 2b. Observational benchmarks for Antarctic sub-components

For the GRACE era (2002-2020), use IMBIE WAIS/EAIS/APIS directly.

For earlier periods:
- IMBIE starts in 1992 but relies on non-GRACE techniques pre-2002 (altimetry, IOM). Uncertainties are larger.
- Pre-1992: no direct sub-continental Antarctic observations. Frederikse provides total AIS only. The sub-continental partition must be treated as unknown. Do not compute WAIS/EAIS/APIS biases for the pre-IMBIE era.

#### 2c. Multiple AIS observational benchmarks

For total AIS bias, compute against all three products:
```python
bias_ais_vs_imbie = facts_ais_median - obs_ais_imbie
bias_ais_vs_frederikse = facts_ais_median - obs_ais_frederikse
bias_ais_vs_horwath = facts_ais_median - obs_ais_horwath
```

If the biases differ across products, the spread quantifies structural uncertainty in the bias estimate itself. Report the range.

#### 2d. Comparison periods

| Period | Years | Available AIS products | Sub-continental? |
|--------|-------|----------------------|-----------------|
| Pre-satellite | 1900-1992 | Frederikse only | No (total AIS only) |
| Early satellite | 1993-2002 | Frederikse, Horwath, IMBIE (limited) | IMBIE (larger unc.) |
| GRACE era | 2003-2016 | Frederikse, Horwath, IMBIE | IMBIE (best) |
| Recent | 2010-2020 | Frederikse (to 2018), IMBIE | IMBIE |

---

## Step 3: Cancellation index

### File: `src/cancellation_index.py`

#### 3a. Global cancellation index (same as original, now 7 components)

$$\mathcal{C}_{\text{global}}(t) = 1 - \frac{|\text{bias}_{\text{total}}(t)|}{\sum_{i=0}^{6} |\text{bias}_i(t)|}$$

#### 3b. Within-Antarctica cancellation index (new)

$$\mathcal{C}_{\text{Antarctic}}(t) = 1 - \frac{|\text{bias}_{\text{WAIS}} + \text{bias}_{\text{EAIS}} + \text{bias}_{\text{APIS}}|}
{|\text{bias}_{\text{WAIS}}| + |\text{bias}_{\text{EAIS}}| + |\text{bias}_{\text{APIS}}|}$$

This measures the degree to which EAIS overestimate (if present) masks WAIS underestimate within the total AIS number. If $\mathcal{C}_{\text{Antarctic}} > 0.3$, there is substantial within-Antarctica cancellation, meaning that anyone using the total AIS projection is seeing a partially-cancelled number that obscures the WAIS underestimate.

#### 3c. Pairwise cancellation (revised)

Compute pairwise cancellation for all $\binom{7}{2} = 21$ pairs. The primary compensating pairs to examine:

1. **(TE, WAIS)**: The dominant global compensating pair. TE biased high, WAIS biased low (loss underestimated → positive bias in SLR contribution).
2. **(WAIS, EAIS)**: The within-Antarctica compensating pair. WAIS loss underestimated, EAIS either overestimated or near zero.
3. **(TE, total AIS)**: The traditional comparison (what Tornqvist et al. examined). This masks the within-Antarctica structure.

Ranking these pairs by cancellation magnitude identifies the dominant error-cancellation pathways.

#### 3d. Hierarchical cancellation decomposition (new)

Decompose the total cancellation into two levels:

**Level 1**: Global cancellation between the 5 "macro-components" (TE, GL, GrIS, AIS_total, LWS)
**Level 2**: Within-Antarctica cancellation (WAIS, EAIS, APIS → AIS_total)

The total bias in the system is:
$$\text{bias}_{\text{total}} = \text{bias}_{\text{TE}} + \text{bias}_{\text{GL}} + \text{bias}_{\text{GrIS}} + \underbrace{(\text{bias}_{\text{WAIS}} + \text{bias}_{\text{EAIS}} + \text{bias}_{\text{APIS}})}_{\text{bias}_{\text{AIS,total}}} + \text{bias}_{\text{LWS}}$$

The Level 2 cancellation reduces $|\text{bias}_{\text{AIS,total}}|$ relative to $|\text{bias}_{\text{WAIS}}|$. This smaller $|\text{bias}_{\text{AIS,total}}|$ then participates in the Level 1 cancellation against $\text{bias}_{\text{TE}}$.

Compute: what would $\mathcal{C}_{\text{global}}$ be if we replaced the partially-cancelled $\text{bias}_{\text{AIS,total}}$ with $\text{bias}_{\text{WAIS}}$ alone? If the difference is large, within-Antarctica cancellation is a material contributor to the appearance of total-GMSL accuracy.

---

## Step 4: Forward projection of biases

### File: `src/extrapolation.py`

#### 4a. Observationally-calibrated projections (revised for sub-continental)

**Thermal expansion**: Same DOLS-based projection as original.

**WAIS**: Extrapolate the IMBIE WAIS rate time series (2002-2020). Fit:
$$\text{rate}_{\text{WAIS}}^{\text{extrap}}(t) = r_0 + \dot{r}(t - t_0) + \tfrac{1}{2}\ddot{r}(t-t_0)^2$$

This extrapolation is specific to WAIS, not total AIS. WAIS has shown clear acceleration over the IMBIE period; EAIS has not.

**EAIS**: Extrapolate the IMBIE EAIS rate. Expected to be approximately constant (near zero or slightly negative, meaning slight mass gain). A constant-rate extrapolation is the appropriate baseline.

**APIS**: Small, approximately constant. Linear extrapolation.

**Glaciers, GrIS, LWS**: Same as original.

#### 4b. Project sub-continental Antarctic biases

At each future decade:
$$\text{bias}_{\text{WAIS}}^{\text{proj}}(t) = r_{\text{WAIS}}^{\text{FACTS}}(t) - r_{\text{WAIS}}^{\text{extrap}}(t)$$

The WAIS bias is expected to grow nonlinearly because the FACTS projection (based on ISMIP6 emulation with n=3) systematically underestimates grounding-line retreat acceleration, while the observational extrapolation captures the realized acceleration.

The EAIS bias is expected to remain small and approximately constant.

#### 4c. Project both cancellation indices forward

Compute $\mathcal{C}_{\text{global}}(t)$ and $\mathcal{C}_{\text{Antarctic}}(t)$ at each future decade using the projected biases.

Key question: does $\mathcal{C}_{\text{Antarctic}}$ increase, decrease, or stay constant? If EAIS mass gain increases under warming (more precipitation) while WAIS loss accelerates, the within-Antarctica cancellation could increase — making the total AIS number look better even as WAIS gets worse. This would be a particularly insidious form of compensating error.

---

## Step 5: Figures

### File: `src/figures.py`

#### Figure 1: Component-level bias summary (now 7 panels)

```
Layout: 7 panels stacked vertically:
  TE, GL, GrIS, WAIS, EAIS, APIS, LWS
  
Each panel:
  - Solid: observed rate (with uncertainty band)
  - Dashed: FACTS module projection(s) in hindcast period
  
For WAIS/EAIS/APIS: observational benchmarks from IMBIE.
Additionally, in the WAIS panel, overlay the budget-derived AIS estimate
from Frederikse (rescaled by the observed WAIS/AIS_total fraction during
the overlap period) to give a visual sense of the longer record.
```

#### Figure 2: Bias waterfall chart (revised, 7 components)

```
Layout: 3 panels (SSPs)
  Each: waterfall bar chart with 7 component biases + total
  x-axis: TE, GL, GrIS, WAIS, EAIS, APIS, LWS, Total
  
Group the three Antarctic components visually (bracket or color family).
The WAIS bar should be the largest negative bar (underestimate).
The EAIS bar may be positive (overestimate).
The visual shows both the global cancellation and the within-Antarctica cancellation.
```

#### Figure 3: Hierarchical cancellation diagram (new, replaces original Fig 3)

```
Layout: Schematic + data hybrid
  Left side: 7 components with bias arrows (up = overestimate, down = underestimate)
  Center: Show two levels of summation:
    - WAIS + EAIS + APIS → AIS_total (Level 2 cancellation)
    - TE + GL + GrIS + AIS_total + LWS → Total (Level 1 cancellation)
  Right side: remaining bias at each level
  
Annotate with C_Antarctic and C_global values.
Show that the WAIS underestimate is partially hidden at Level 2
before it even participates in Level 1 cancellation.
```

#### Figure 4: Cancellation index time series (revised)

```
Layout: 2 panels stacked
  Top: C_global(t), 2020-2150, one line per SSP
  Bottom: C_Antarctic(t), 2020-2150, one line per SSP
  
Shading: 17-83% uncertainty from MC propagation.
Horizontal dashed line: C = 0.5
```

#### Figure 5: WAIS bias extrapolation (new, replaces original Fig 4 AIS panel)

```
Layout: 1×3 panels (SSPs)
  Each:
  - Solid: IMBIE WAIS observed rate (1992-2020), then extrapolated (dashed)
  - Colored lines: FACTS WAIS projections from each AIS module
  - Shading: growing divergence = the WAIS bias
  
Purpose: Show the WAIS-specific bias growing, unmasked by EAIS cancellation.
```

#### Figure 6: Within-Antarctica compensating errors (new)

```
Layout: Single panel
  - x-axis: year (2002-2020, observed; 2020-2100, projected)
  - y-axis: rate (mm/yr)
  - Three stacked areas: WAIS rate (red), EAIS rate (blue), APIS rate (green)
  - Two versions side by side or overlaid: 
    observed (IMBIE) vs FACTS (emulandice)
  
Purpose: Visually demonstrate that EAIS mass gain partially offsets WAIS mass loss
in the total, and that FACTS may get this partition wrong.
```

#### Figure 7: Cross-validation of AIS products (new)

```
Layout: Single panel
  - x-axis: year (1992-2020)
  - y-axis: total AIS rate (mm/yr)
  - Three lines with bands: IMBIE, Frederikse, Horwath
  - Shaded overlap region
  
Purpose: Establish reliability of the observational AIS benchmark.
```

#### Table 1: Component biases during satellite era (7 components)

| Component | Module | Obs. rate | FACTS rate | Bias | z-score |
|-----------|--------|-----------|------------|------|---------|

For all modules, GRACE era averages.

#### Table 2: Cancellation indices

| Level | Workflow | SSP | C (hindcast) | C (2050) | C (2100) |
|-------|----------|-----|-------------|----------|----------|
| Global | ... | ... | ... | ... | ... |
| Antarctic | ... | ... | ... | ... | ... |

---

## Step 6: Validation checks

All original tests apply. Additional:

#### Test 5: Sub-continental consistency
Verify WAIS + EAIS + APIS biases sum to total AIS bias (within rounding).

#### Test 6: Hierarchical cancellation arithmetic
Verify that C_global computed from 7 components equals C_global computed from 5 macro-components (TE, GL, GrIS, AIS_total, LWS) where AIS_total = WAIS + EAIS + APIS. The global cancellation index should be identical regardless of whether Antarctica is aggregated before or after the cancellation computation.

Actually, this will NOT hold in general. $\mathcal{C}$ is not linear in the component biases because of the absolute values. Verify the algebraic relationship:

$$|\text{bias}_{\text{WAIS}} + \text{bias}_{\text{EAIS}} + \text{bias}_{\text{APIS}}| \leq |\text{bias}_{\text{WAIS}}| + |\text{bias}_{\text{EAIS}}| + |\text{bias}_{\text{APIS}}|$$

with equality only when all three have the same sign. If they have mixed signs, $\sum_i |\text{bias}_i|$ at the 7-component level exceeds $\sum |\text{bias}_j|$ at the 5-macro-component level. This means $\mathcal{C}_{\text{global}}^{(7)} \geq \mathcal{C}_{\text{global}}^{(5)}$: the 7-component cancellation index is always at least as large as the 5-component version. The difference is the within-Antarctica cancellation that is hidden when you aggregate to total AIS. Report this difference explicitly — it quantifies how much cancellation is invisible at the macro-component level.

---

## Output checklist

- [ ] 7 figures
- [ ] 2 tables
- [ ] Validation test suite passing
- [ ] Component bias database (7 components × all modules × all periods × 3 SSPs)
- [ ] Both cancellation index time series (global and Antarctic) with uncertainty
- [ ] Hierarchical cancellation decomposition showing hidden within-Antarctica cancellation
- [ ] AIS cross-validation statistics (IMBIE vs Frederikse vs Horwath)
- [ ] README
