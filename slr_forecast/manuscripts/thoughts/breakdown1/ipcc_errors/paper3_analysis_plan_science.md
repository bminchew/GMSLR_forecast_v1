# Analysis Plan: Paper 3 — Compensating Errors Mask West Antarctic Ice Loss (Science format)

## Revision summary

This plan replaces `paper3_analysis_plan_v2.md`. Changes from v2:
- Restructured for Science: 4 main figures, ~10 supplementary figures, extended methods in SM
- DOLS quadratic sensitivity analysis folded in (was previously an external dependency)
- New step (Step 2) for DOLS calibration and IPCC emergent sensitivity comparison
- Figures reorganized: main text gets the 4 highest-impact figures; all others move to SM
- The DOLS coefficients are now computed in-paper, not imported from an external file

---

## Component vector

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

### Python environment

```bash
pip install numpy scipy pandas matplotlib seaborn xarray netCDF4 statsmodels tqdm
```

### Data files required

1. **FACTS v1.0 projections** with WAIS/EAIS/APIS from `full_sample_components` / `dist_components`
2. **IMBIE**: WAIS, EAIS, APIS, GrIS — reconciled mass balance, 1992–2020
3. **Frederikse et al. (2020)**: total GMSL + component contributions including total AIS, 1900–2018
4. **Horwath et al. (2022)**: independent budget with total AIS estimate, 1993–2016
5. **Argo-era thermosteric**: NOAA/NCEI OHC 0–2000m
6. **GlaMBIE**: glacier mass balance consensus, 2000–2023
7. **Satellite altimetry**: total GMSL, 1993–present
8. **GMSL reconstructions for DOLS**: Frederikse total, Frederikse thermodynamic, Dangendorf total, Dangendorf sterodynamic, IPCC observed total, IPCC observed thermodynamic, Horwath (annualized)
9. **GMST products for DOLS**: Berkeley Earth, GISTEMP, HadCRUT5, NOAA GlobalTemp
10. **IPCC AR6 FACTS component projections** (decadal, 2020–2100) for the emergent sensitivity comparison: thermal + glacier + Greenland components under all 5 SSPs
11. **FaIR temperature trajectories** per SSP (for forward extrapolation)

---

## Project structure

```
paper3_science/
├── data/
│   ├── facts/
│   ├── imbie/
│   ├── frederikse/
│   ├── horwath/
│   ├── thermosteric/
│   ├── glaciers/
│   ├── altimetry/
│   ├── gmsl_reconstructions/   # 7 GMSL records for DOLS
│   ├── gmst_products/          # 4 GMST products for DOLS
│   └── fair/                   # FaIR temperature trajectories
├── src/
│   ├── load_data.py
│   ├── dols_calibration.py     # NEW: DOLS + robustness matrix + IPCC comparison
│   ├── compute_biases.py
│   ├── cancellation_index.py
│   ├── extrapolation.py
│   ├── figures_main.py         # 4 main-text figures
│   ├── figures_supplement.py   # ~10 supplementary figures
│   └── utils.py
├── outputs/
│   ├── main_figures/
│   ├── supplementary_figures/
│   └── tables/
├── tests/
│   ├── test_dols.py
│   ├── test_biases.py
│   └── test_cancellation.py
└── run_all.py
```

---

## Step 1: Load and harmonize data

### File: `src/load_data.py`

Same as v2 plan. No changes to this step.

Additional loading functions for DOLS:

```python
def load_gmsl_records() -> dict:
    """
    Returns dict keyed by record name with values:
        pd.DataFrame with columns: year, gmsl_mm, gmsl_unc_mm
    
    Records:
        'frederikse_total': 1900-2018
        'frederikse_thermo': 1900-2018 (total minus TWS)
        'dangendorf_total': 1900-2021
        'dangendorf_sterodynamic': 1900-2021
        'ipcc_total': 1950-2020
        'ipcc_thermo': 1950-2018 (IPCC minus Frederikse TWS)
        'horwath': 1993-2016 (excluded from ensemble, too short)
    """

def load_gmst_products() -> dict:
    """
    Returns dict keyed by product name with values:
        pd.DataFrame with columns: year, gmst_C (relative to 1850-1900)
    
    Products: 'berkeley_earth', 'gistemp', 'hadcrut5', 'noaa_globaltemp'
    All annualized, common period starting 1950.
    """

def load_ipcc_thermodynamic_projections() -> dict:
    """
    Returns dict keyed by SSP with values:
        pd.DataFrame with columns: year (decadal 2020-2100),
            rate_thermo_median, rate_thermo_p17, rate_thermo_p83
    
    'Thermodynamic' = thermal expansion + glaciers + Greenland (excludes AIS).
    Constructed from FACTS component outputs.
    """
```

---

## Step 2: DOLS calibration and emergent sensitivity comparison (NEW)

### File: `src/dols_calibration.py`

This step was previously an external dependency ("DOLS coefficients from existing analysis"). It is now computed in-paper.

#### 2a. DOLS implementation

```python
def calibrate_dols(
    gmsl: pd.DataFrame,    # columns: year, gmsl_mm, gmsl_unc_mm
    gmst: pd.DataFrame,    # columns: year, gmst_C
    order: int = 2,        # polynomial order (2 = quadratic)
    n_lags: int = 2,       # leads/lags for DOLS
    start_year: int = 1950,
) -> dict:
    """
    Dynamic OLS calibration of rate-temperature relationship.
    
    Model: rate(t) = alpha_0 * T(t) + (d_alpha/dT) * T(t)^2
    
    Implementation:
    1. Compute GMSL rates via finite differences
    2. Construct regressor matrix: [T, T^2, dT/dt leads/lags]
    3. WLS with weights = 1/sigma_rate^2
    4. HAC standard errors (Newey-West)
    
    Returns:
        alpha_0: float
        alpha_0_se: float
        d_alpha_dT: float
        d_alpha_dT_se: float
        r_squared: float
        n_obs: int
        aic: float
        bic: float
    """
```

#### 2b. Multi-dataset robustness matrix

```python
def compute_robustness_matrix(
    gmsl_records: dict,    # from load_gmsl_records()
    gmst_products: dict,   # from load_gmst_products()
    start_year: int = 1950,
    n_lags: int = 2,
) -> pd.DataFrame:
    """
    Run DOLS on every (GMSL, GMST) pair.
    
    Returns DataFrame with columns:
        gmsl_name, gmst_name, alpha_0, alpha_0_se, d_alpha_dT, d_alpha_dT_se,
        r_squared, n_obs
    
    Exclude Horwath (too short for stable quadratic estimates).
    6 GMSL × 4 GMST = 24 fits.
    """
```

Compute ensemble statistics:
- Thermodynamic ensemble (8 pairs: Frederikse thermo + IPCC thermo × 4 GMST): $d\alpha/dT = 2.85 \pm 0.38$
- All-dataset ensemble (24 pairs): $d\alpha/dT = 1.83 \pm 1.09$
- Exclude Dangendorf sterodynamic from thermodynamic ensemble (explained in text)

#### 2c. IPCC emergent sensitivity comparison

```python
def ipcc_sensitivity_comparison(
    ipcc_thermo_projections: dict,   # {ssp: DataFrame}
    fair_temperatures: dict,          # {ssp: DataFrame}
) -> pd.DataFrame:
    """
    For each SSP:
    1. Construct IPCC 'thermodynamic' rate = d/dt(thermal + glaciers + Greenland)
    2. Fit linear and quadratic rate-temperature models
    3. Compare via AIC, BIC, F-test
    4. Extract alpha_0 (linear sensitivity)
    5. Compute statistical power for detecting the observational quadratic
    
    Returns DataFrame with columns:
        ssp, alpha_0_linear, alpha_0_se, d_alpha_dT_quad, d_alpha_dT_se,
        linear_preferred (bool), f_test_p, power_at_obs_effect
    """
```

Expected result: all SSPs prefer linear; $\alpha_0 \approx 1.9$–$2.5$ mm/yr/°C; observational quadratic is outside IPCC 95% CI for SSP2-4.5 through SSP5-8.5.

#### 2d. Start-date sensitivity (supplementary)

```python
def start_date_sensitivity(
    gmsl_records: dict,
    gmst_products: dict,
) -> pd.DataFrame:
    """
    For records extending before 1950 (Frederikse, Dangendorf):
    compare DOLS coefficients at 1950-start vs native-start.
    
    Demonstrates epoch dependence of the calibration.
    """
```

#### 2e. Acceptance criteria

- Thermodynamic ensemble $d\alpha/dT$ should be > 0 at 95% confidence (> 2σ from zero).
- IPCC linear sensitivity should be approximately half the observational value.
- The power analysis should confirm that the absence of quadratic detection in IPCC models is not a sample-size artifact for SSP2-4.5 through SSP5-8.5.

---

## Step 3: Compute component-level biases

### File: `src/compute_biases.py`

Same as v2 plan. No changes. Seven components, multiple AIS observational benchmarks, multiple comparison periods.

---

## Step 4: Cancellation index

### File: `src/cancellation_index.py`

Same as v2 plan. No changes. Global and Antarctic cancellation indices, hierarchical decomposition, $\Delta\mathcal{C}$ computation.

---

## Step 5: Forward projection of biases

### File: `src/extrapolation.py`

Same as v2 plan, with one change: the DOLS coefficients used for the thermosteric projection are now produced internally by Step 2, not imported from an external file.

```python
def project_thermosteric_rate(
    dols_coefficients: dict,  # from Step 2: alpha_0, d_alpha_dT, and their uncertainties
    fair_temperatures: dict,  # FaIR GMST trajectory per SSP
    n_mc: int = 10000,
) -> dict:
    """
    For each SSP, compute:
        rate_thermo_DOLS(t) = alpha_0 * T(t) + d_alpha_dT * T(t)^2
    
    Propagate coefficient uncertainty via MC draws from bivariate normal
    (alpha_0, d_alpha_dT) with covariance from DOLS.
    
    Returns dict keyed by SSP with arrays of shape (n_mc, n_times).
    """
```

Everything else (WAIS quadratic extrapolation, EAIS constant rate, APIS, GL, GrIS, LWS, MC propagation of cancellation indices) is unchanged from v2.

---

## Step 6: Figures

### Main figures (4 — for Science main text)

All main figures must be self-contained, immediately readable, and convey the central result without requiring the supplement.

#### Main Figure 1: Component-level bias waterfall (7 components)

```
Layout: Single panel, landscape orientation
  - Waterfall bar chart: TE, GL, GrIS, WAIS, EAIS, APIS, LWS, Total
  - Red bars: positive bias (FACTS overestimates)
  - Blue bars: negative bias (FACTS underestimates)
  - Final gray bar: total bias (net after cancellation)
  - Bracket grouping WAIS/EAIS/APIS with "Antarctica" label
  - Annotate: C_global and C_Antarctic values
  
Period: GRACE era (2003-2020)
Module: emulandice for all components (Workflow 1e)
SSP: SSP2-4.5 (most policy-relevant)

This is the "money figure." It must show at a glance:
  (a) TE is biased high
  (b) WAIS is biased low and is the largest individual bias
  (c) EAIS partially masks WAIS within the Antarctic sum
  (d) The total bias is small despite large component biases
```

#### Main Figure 2: Hierarchical cancellation diagram

```
Layout: Schematic with quantitative annotations
  Left: 7 component bias arrows (length proportional to |bias|)
  Center: Two-level summation:
    - Level 2: WAIS + EAIS + APIS → AIS_total (with C_Antarctic)
    - Level 1: TE + GL + GrIS + AIS_total + LWS → Total (with C_global)
  Right: Remaining bias magnitude at each level
  
Show visually how the WAIS signal is attenuated at Level 2 
before entering Level 1.

This can be a data-annotated schematic (partly TikZ, partly computed values).
The quantitative values come from the GRACE-era bias computation.
```

#### Main Figure 3: Cancellation index time series under extrapolation

```
Layout: Single panel (or 2 stacked sub-panels if space allows)
  - x-axis: year (2020-2150)
  - y-axis: Cancellation index (0-1)
  - Lines: SSP1-2.6 (blue), SSP2-4.5 (orange), SSP5-8.5 (red)
  - Shading: 17-83% MC uncertainty
  - Horizontal dashed line: C = 0.5
  - Vertical dashed line at present day
  - Left region (light gray): hindcast-validated
  - Right region: projection

If two sub-panels:
  Top: C_global
  Bottom: C_Antarctic
  
If one panel: show C_global only; C_Antarctic goes to SM.

Key feature: C_global declining under SSP5-8.5, approximately stable under SSP1-2.6.
```

#### Main Figure 4: WAIS-specific bias evolution

```
Layout: 3 sub-panels (SSP1-2.6, SSP2-4.5, SSP5-8.5) OR single panel with SSP5-8.5

Each panel:
  - x-axis: year (1992-2100)
  - y-axis: WAIS rate (mm/yr SLE)
  - Solid black line: IMBIE WAIS observed rate (1992-2020)
  - Dashed black line: quadratic extrapolation with uncertainty band
  - Colored lines: FACTS WAIS projections from 2-3 representative AIS modules
  - Shading between FACTS and observational extrapolation: the WAIS bias

Purpose: Show the WAIS-specific bias growing, unobscured by EAIS cancellation.
This is the actionable result: WAIS is the component that matters most and is
the most underestimated.
```

### Supplementary figures (~10)

#### Fig S1: 7-panel component-level rate time series

```
7 panels (TE, GL, GrIS, WAIS, EAIS, APIS, LWS)
Each: observed rate (solid with band) vs FACTS projections (dashed)
```

#### Fig S2: DOLS multi-dataset robustness matrix

```
Heatmap or dot plot showing alpha_0 and d_alpha/dT for all 24 (GMSL, GMST) pairs.
Highlight the thermodynamic ensemble and the Dangendorf sterodynamic outlier.
```

#### Fig S3: IPCC vs observational rate-temperature relationship

```
5 panels (one per SSP)
Each: rate vs T scatter from IPCC thermodynamic projections + linear fit
Overlay: observational DOLS quadratic curve with uncertainty band
Show the divergence between the two at higher temperatures.
```

#### Fig S4: Start-date sensitivity

```
Table or bar chart showing how DOLS coefficients change between 1950-start 
and native-start for each GMSL record.
```

#### Fig S5: Sliding-window DOLS coefficient evolution

```
Coefficient evolution (alpha_0(t), d_alpha/dT(t)) for multi-bandwidth 
sliding-window DOLS on Frederikse thermodynamic × Berkeley Earth.
```

#### Fig S6: AIS cross-validation (IMBIE vs Frederikse vs Horwath)

```
Single panel: total AIS rate time series from three independent products, 
1992-2020, with uncertainty bands. Show the overlap period.
```

#### Fig S7: Within-Antarctica compensating errors (stacked area)

```
Side-by-side: observed (IMBIE) vs FACTS (emulandice)
Three stacked areas: WAIS, EAIS, APIS rates.
Shows EAIS partially offsetting WAIS in both observation and projection.
```

#### Fig S8: Sensitivity to AIS module choice

```
4 panels (emulandice, larmip, bamber19, deconto21)
Each: WAIS bias for that module during GRACE era.
Shows how module choice affects the bias diagnosis.
```

#### Fig S9: C_global at 5-component vs 7-component level

```
Two lines on the same panel: C^(5) and C^(7) under SSP5-8.5.
The gap = hidden within-Antarctica cancellation.
```

#### Fig S10: Full waterfall charts for SSP1-2.6 and SSP5-8.5

```
Same as Main Fig 1 but for the other two SSPs.
```

### Supplementary tables

- **Table S1**: DOLS robustness matrix (6×4): alpha_0, d_alpha/dT for each (GMSL, GMST) pair
- **Table S2**: Start-date sensitivity of DOLS coefficients
- **Table S3**: IPCC emergent sensitivity comparison (all 5 SSPs): linear alpha_0, quadratic test, power analysis
- **Table S4**: Full 7-component bias table (all modules × all periods × 3 SSPs)
- **Table S5**: Cancellation indices at both levels (all workflows × 3 SSPs × hindcast/2050/2100)
- **Table S6**: AIS cross-validation (IMBIE vs Frederikse vs Horwath, 2002-2016)

---

## Step 7: Validation checks

### File: `tests/test_dols.py` (NEW)

#### Test D1: DOLS on synthetic linear data
Generate rate = α₀T + noise. Verify DOLS recovers α₀ and d_alpha/dT ≈ 0.

#### Test D2: DOLS on synthetic quadratic data
Generate rate = α₀T + (dα/dT)T² + noise. Verify both coefficients recovered.

#### Test D3: HAC standard errors
On synthetic AR(1) residuals, verify that HAC SEs exceed OLS SEs.

#### Test D4: IPCC sensitivity comparison
Construct a known quadratic signal, fit with linear model, verify the F-test rejects linearity at the correct significance level.

### File: `tests/test_biases.py` (same as v2)

All v2 tests apply unchanged.

### File: `tests/test_cancellation.py` (same as v2)

All v2 tests apply unchanged. Including:
- Sub-continental consistency (WAIS + EAIS + APIS = total AIS bias)
- $\mathcal{C}^{(7)} \geq \mathcal{C}^{(5)}$ inequality
- $\Delta\mathcal{C}$ quantification

---

## Step 8: Master script

### File: `run_all.py`

```python
"""
Master script for Paper 3 (Science format).

Steps:
1.  Load all data (FACTS 7-component, IMBIE, Frederikse, Horwath, 
    GMSL records, GMST products, IPCC projections, FaIR temps)
2.  DOLS calibration: robustness matrix, ensemble statistics
3.  IPCC emergent sensitivity comparison (all 5 SSPs)
4.  Start-date sensitivity and sliding-window DOLS (supplement)
5.  Compute 7-component biases (all modules × periods × SSPs)
6.  Cross-validate AIS products (IMBIE vs Frederikse vs Horwath)
7.  Compute cancellation indices (global and Antarctic, both levels)
8.  Hierarchical cancellation decomposition (Delta-C)
9.  Forward extrapolation (DOLS thermal + WAIS quadratic + MC uncertainty)
10. Project cancellation indices forward
11. Generate 4 main figures
12. Generate ~10 supplementary figures
13. Generate 6 supplementary tables
14. Run all validation tests
15. Print summary report with key numbers for main text
"""
```

---

## Estimated compute time

- Data loading: ~3 min
- DOLS robustness matrix (24 fits): ~2 min
- IPCC sensitivity comparison (5 SSPs): ~1 min
- Start-date sensitivity + sliding window: ~5 min
- Component biases (full grid): ~2 min
- Cancellation indices: ~1 min
- Forward extrapolation with MC (10,000 draws): ~10 min
- Main figures: ~2 min
- Supplementary figures: ~5 min
- Tests: ~3 min
- **Total: ~35 min**

---

## Critical decision points

1. **DOLS coefficient source**: The plan computes DOLS from scratch. If the existing pipeline already produces these numbers and they have been validated, it is acceptable to import them — but the robustness matrix and IPCC comparison must still be reproduced for the supplement to make the paper self-contained.

2. **Main Figure selection**: The 4 main figures are chosen for maximum impact in the Science format. If a reviewer requests the component-level rate time series (currently Fig S1) in the main text, it can replace Fig 2 (the schematic), since the schematic's content is also conveyed verbally in the text. Do not exceed 4 main figures.

3. **DOLS section length**: In the main text, the DOLS result gets ~500 words (the "Observational transient sensitivity" paragraph). The full methodology (lag selection, kernel weights, HAC specification) goes in Extended Methods. If the main text is over 3000 words, cut from the Implications paragraph first.

4. **Sliding-window DOLS**: This is supplementary material only. It strengthens the paper by showing the epoch-dependence, but is not essential to the central argument. If compute time is a concern, defer it.

5. **If the thermosteric bias is not significant during the hindcast period**: This is actually consistent with the narrative — the bias is in the *sensitivity* (rate of change with temperature), not in the rate during the calibration period. The DOLS result shows the sensitivity is wrong even if the current rate is approximately right. State this explicitly in the text.

---

## Key numbers for the main text

The analysis must produce these specific numbers for the ~3000-word main text:

- $d\alpha/dT$ thermodynamic ensemble value ± uncertainty
- IPCC linear sensitivity range (α₀ ≈ 1.9–2.5 mm/yr/°C)
- Factor-of-two discrepancy between observational and IPCC sensitivity
- WAIS bias magnitude during GRACE era (mm/yr)
- EAIS bias magnitude and sign
- $\mathcal{C}_{\text{global}}$ during GRACE era
- $\mathcal{C}_{\text{Antarctic}}$ during GRACE era
- $\Delta\mathcal{C} = \mathcal{C}^{(7)} - \mathcal{C}^{(5)}$ (hidden within-Antarctica cancellation)
- $\mathcal{C}_{\text{global}}$ at 2100 under SSP5-8.5 (with uncertainty range)
- WAIS bias at 2100 under SSP5-8.5 (mm/yr, with uncertainty)

These numbers are the backbone of the paper. Every sentence in the main text either motivates, presents, or interprets one of these numbers.

---

## Output checklist

Main text deliverables:
- [ ] 4 main figures (publication quality, Science dimensions)
- [ ] ~3000 words main text (LaTeX source)
- [ ] ~125 word abstract

Supplementary deliverables:
- [ ] ~10 supplementary figures
- [ ] 6 supplementary tables
- [ ] Extended methods text

Analysis deliverables:
- [ ] DOLS robustness matrix (24 fits with ensemble statistics)
- [ ] IPCC emergent sensitivity comparison (5 SSPs)
- [ ] 7-component bias database (all modules × periods × SSPs)
- [ ] AIS cross-validation statistics
- [ ] Both cancellation index time series with MC uncertainty
- [ ] Hierarchical cancellation decomposition
- [ ] Forward-projected biases and cancellation indices
- [ ] All validation tests passing
- [ ] Key numbers summary file for main text drafting
- [ ] README with data provenance and reproducibility instructions
