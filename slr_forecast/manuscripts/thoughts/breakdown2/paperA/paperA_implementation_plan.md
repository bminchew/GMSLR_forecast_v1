# Implementation Plan: Paper A — Observationally Calibrated SLR Projections

## Timeline target

Submission within 4-6 weeks. The DOLS calibration analysis exists in the project codebase. The primary new work is: (1) producing projection distributions in IPCC-comparable format, (2) the DOLS-vs-FACTS comparison, (3) the residual decomposition, (4) figures and tables.

---

## Project structure

```
paperA_dols_projections/
├── data/
│   ├── gmsl_reconstructions/   # 7 GMSL records
│   ├── gmst_products/          # 4 GMST products
│   ├── fair/                   # FaIR temperature trajectories per SSP
│   ├── facts/                  # FACTS component projections (for comparison)
│   ├── altimetry/              # Satellite GMSL for hindcast overlay
│   └── imbie/                  # IMBIE AIS+GrIS for residual validation
├── src/
│   ├── dols.py                 # DOLS calibration (existing, adapt)
│   ├── robustness_matrix.py    # Multi-dataset robustness (existing, adapt)
│   ├── projections.py          # NEW: produce projection distributions
│   ├── residual.py             # NEW: thermodynamic/residual decomposition
│   ├── comparison.py           # NEW: DOLS vs FACTS comparison
│   ├── ipcc_sensitivity.py     # IPCC emergent sensitivity (existing, adapt)
│   ├── figures_main.py         # 4 main figures
│   ├── figures_supplement.py   # ~5 supplementary figures
│   └── utils.py
├── outputs/
│   ├── projections/            # CSV projection tables for Zenodo
│   ├── main_figures/
│   ├── supplementary/
│   └── tables/
├── tests/
│   └── test_projections.py
└── run_all.py
```

---

## Step 1: Assemble existing DOLS analysis

### File: `src/dols.py`

Port or import the existing DOLS calibration code. The key function:

```python
def calibrate_dols(
    gmsl: pd.DataFrame,     # year, gmsl_mm, gmsl_unc_mm
    gmst: pd.DataFrame,     # year, gmst_C
    order: int = 2,
    n_lags: int = 2,
    start_year: int = 1950,
) -> DOLSResult:
    """
    Returns DOLSResult with fields:
        alpha_0, alpha_0_se: float
        d_alpha_dT, d_alpha_dT_se: float
        cov_matrix: np.ndarray (2x2)  # covariance of (alpha_0, d_alpha_dT)
        residuals: np.ndarray
        r_squared: float
        n_obs: int
        aic, bic: float
    """
```

### File: `src/robustness_matrix.py`

Port the existing robustness matrix computation. Produces the 6x4 grid of DOLS fits and the ensemble statistics.

```python
def compute_robustness_matrix(gmsl_records, gmst_products, start_year=1950) -> pd.DataFrame
def compute_ensemble_statistics(matrix: pd.DataFrame) -> dict
    # Returns: thermo_ensemble_mean, thermo_ensemble_std, all_ensemble_mean, all_ensemble_std
```

These functions exist. Verify they produce the numbers in the framework document:
- Thermodynamic ensemble: $d\alpha/dT = 2.85 \pm 0.38$
- All-dataset ensemble: $d\alpha/dT = 1.83 \pm 1.09$

---

## Step 2: Produce projection distributions

### File: `src/projections.py`

This is the primary new code for Paper A.

```python
def project_dols(
    dols_result: DOLSResult,
    fair_temps: pd.DataFrame,      # year, gmst_median, gmst_p17, gmst_p83
    n_mc: int = 10000,
    structural_uncertainty: dict = None,  # from robustness matrix spread
) -> ProjectionResult:
    """
    Produce GMSL projection distributions.
    
    Algorithm:
    1. Draw (alpha_0, d_alpha_dT) from bivariate normal with mean and 
       covariance from dols_result.
    2. For each draw, compute rate(t) = alpha_0 * T(t) + d_alpha_dT * T(t)^2
       at each year from 2020 to 2150.
    3. Integrate rate to get cumulative GMSL relative to baseline.
    4. Optionally: outer loop over robustness matrix entries to add 
       structural uncertainty from dataset choice.
    
    Returns ProjectionResult with fields:
        years: np.ndarray (2020-2150)
        trajectories: np.ndarray (n_mc, n_years)  # cumulative mm
        rate_trajectories: np.ndarray (n_mc, n_years)  # mm/yr
        quantiles: dict  # {5, 17, 50, 83, 95} -> np.ndarray(n_years)
    """
```

**Structural uncertainty integration**: Two approaches, implement both and compare:

**Approach 1 (simple)**: Use the primary DOLS fit (Frederikse thermodynamic × Berkeley Earth) for the coefficient posterior, then add the robustness matrix spread as an additional variance term. This is approximate but fast.

**Approach 2 (ensemble)**: For each of the 24 (GMSL, GMST) pairs, produce a projection from that pair's DOLS posterior. The ensemble of 24 projections captures the structural uncertainty from dataset choice. Weight equally or use a quality-based weighting (e.g., downweight short records). Report the multi-dataset ensemble median and spread alongside the primary projection.

Recommend Approach 2 for the paper. It is more transparent and makes the structural uncertainty explicit.

```python
def project_ensemble(
    robustness_matrix: pd.DataFrame,
    fair_temps: dict,        # {ssp: DataFrame}
    n_mc_per_fit: int = 1000,
) -> dict:
    """
    For each (GMSL, GMST) pair in the robustness matrix:
        Produce n_mc_per_fit projection trajectories.
    Pool all trajectories to form the ensemble projection.
    
    Returns dict keyed by SSP with ProjectionResult.
    """
```

**Temperature trajectory uncertainty**: FaIR provides temperature distributions, not point estimates. For each MC draw, also draw a temperature trajectory from the FaIR distribution. This propagates climate sensitivity uncertainty into the GMSL projection.

```python
def project_with_temp_uncertainty(
    dols_result: DOLSResult,
    fair_temp_samples: np.ndarray,  # (n_fair_samples, n_years)
    n_mc: int = 10000,
) -> ProjectionResult:
    """
    Joint uncertainty: (alpha_0, d_alpha_dT) × T(t) trajectory.
    Draw coefficient pair AND temperature trajectory jointly.
    """
```

#### Output format

Produce projection tables matching FACTS Table A1 format:

```python
def format_projection_table(
    projections: dict,     # {ssp: ProjectionResult}
    years: list = [2050, 2100, 2150],
) -> pd.DataFrame:
    """
    Returns table with columns:
        SSP, Year, Median (m), 17th (m), 83rd (m), 5th (m), 95th (m)
    All relative to 1995-2014 baseline, in meters.
    """
```

Export as CSV for Zenodo deposit. This is the deliverable that AR7 authors can cite directly.

---

## Step 3: Residual decomposition

### File: `src/residual.py`

```python
def compute_residual(
    gmsl_obs: pd.DataFrame,       # observed total GMSL (Frederikse or altimetry)
    dols_result: DOLSResult,
    gmst: pd.DataFrame,
) -> pd.DataFrame:
    """
    Compute: residual_rate(t) = observed_rate(t) - DOLS_predicted_rate(t)
    
    The residual represents the component of GMSL change NOT explained by
    the rate-temperature relationship. This is primarily ice-sheet dynamics
    (WAIS, GrIS discharge) and any nonlinear/threshold processes.
    
    Returns DataFrame with columns:
        year, residual_rate, residual_rate_unc,
        dols_predicted_rate, observed_rate
    """
```

**Validation**: Compare the residual against IMBIE ice-sheet contributions (AIS + GrIS discharge). During the GRACE era (2002-2020), the residual should be consistent with the sum of IMBIE AIS mass loss rate + GrIS discharge rate. If it is, the decomposition is validated. If not, the discrepancy indicates either (a) the DOLS model is capturing some non-thermodynamic signal, or (b) the ice-sheet observations are incomplete.

```python
def validate_residual(
    residual: pd.DataFrame,
    imbie_ais: pd.DataFrame,
    imbie_gris_discharge: pd.DataFrame,  # if available, else total GrIS
) -> dict:
    """
    Returns:
        mean_residual: float (mm/yr, 2002-2020)
        mean_icesheet_obs: float (mm/yr, 2002-2020)
        discrepancy: float
        discrepancy_zscore: float
    """
```

---

## Step 4: FACTS comparison

### File: `src/comparison.py`

```python
def compare_dols_facts(
    dols_projections: dict,    # {ssp: ProjectionResult}
    facts_projections: dict,   # {ssp: DataFrame with FACTS quantiles}
) -> pd.DataFrame:
    """
    Side-by-side comparison at 2050, 2100, 2150 for each SSP.
    
    Returns DataFrame with columns:
        ssp, year,
        dols_median, dols_p17, dols_p83,
        facts_median, facts_p17, facts_p83,
        difference_median, difference_pct
    """
```

### File: `src/ipcc_sensitivity.py`

Port the existing IPCC emergent sensitivity comparison. For each SSP, construct the IPCC "thermodynamic" rate (thermal + glaciers + Greenland), fit linear and quadratic models, compare via AIC/BIC/F-test, compute power analysis.

This code exists. Verify it produces the expected results:
- All SSPs prefer linear
- $\alpha_0 \approx 1.9$–$2.5$ mm/yr/°C
- Observational quadratic outside IPCC 95% CI for SSP2-4.5+

---

## Step 5: Figures

### Main figures (4)

#### Figure 1: Projection fan chart

```
Layout: Single panel (or 2x2 sub-panels for 4 SSPs)
  - x-axis: year (1950-2150)
  - y-axis: GMSL (mm relative to 1995-2014)
  - For each SSP: median line + 17-83% shading + 5-95% light shading
  - Overlay: observed GMSL (Frederikse 1950-2018 + altimetry 1993-2025)
    as black line with gray uncertainty band
  - Vertical dashed line at 2025 (present)
  
Color: SSP1-2.6 blue, SSP2-4.5 orange, SSP3-7.0 yellow, SSP5-8.5 red
```

#### Figure 2: Residual time series

```
Layout: Single panel
  - x-axis: year (1950-2020)
  - y-axis: rate (mm/yr)
  - Black line: DOLS residual rate (observed - predicted) with uncertainty
  - Red points with error bars: IMBIE total AIS rate (1992-2020)
  - Orange points: IMBIE GrIS rate (for comparison)
  - Blue shading: DOLS predicted thermodynamic rate
  
Purpose: validate the decomposition by showing the residual matches
known ice-sheet contributions.
```

#### Figure 3: DOLS vs FACTS comparison

```
Layout: 3 panels (SSP1-2.6, SSP2-4.5, SSP5-8.5)
  Each panel:
  - DOLS projection: solid line + shading
  - FACTS (Workflow 2e): dashed line + hatched shading
  - x-axis: year (2020-2150)
  - y-axis: cumulative GMSL (mm)
  
Annotate: difference at 2100 in each panel.
```

#### Figure 4: Rate-temperature relationship

```
Layout: Single panel
  - x-axis: GMST (°C above pre-industrial, 0 to 5)
  - y-axis: GMSL rate (mm/yr)
  - Observational: DOLS quadratic curve (solid) with uncertainty band
    Points: historical (T, rate) pairs from Frederikse thermo × Berkeley
  - IPCC: linear fits for each SSP (dashed lines, colored by SSP)
  - The divergence at high T is the visual summary of the sensitivity discrepancy.
```

### Supplementary figures (~5)

- Fig S1: Robustness matrix heatmap
- Fig S2: Start-date sensitivity bar chart
- Fig S3: Sliding-window coefficient evolution
- Fig S4: IPCC sensitivity per SSP with power analysis
- Fig S5: Projection density plots at 2100 (per SSP)

---

## Step 6: Validation tests

### File: `tests/test_projections.py`

```python
# Test 1: DOLS on synthetic known-quadratic data recovers coefficients
# Test 2: Projection at T=0 gives rate=0 (no sea level rise at pre-industrial)
# Test 3: Projection quantiles are monotonic in SSP forcing
#          (SSP5-8.5 median > SSP2-4.5 median > SSP1-2.6 median at all times)
# Test 4: Hindcast: DOLS prediction over 1993-2020 is consistent with
#          observed GMSL to within stated uncertainty
# Test 5: Residual during GRACE era is consistent with IMBIE AIS+GrIS
# Test 6: Projection at 2100 under SSP2-4.5 is within 50% of IPCC assessed range
#          (sanity check — should be comparable, not identical)
# Test 7: Structural uncertainty from robustness ensemble exceeds
#          single-fit coefficient uncertainty (by construction)
```

---

## Step 7: Master script

```python
"""
Paper A master script.

1. Load GMSL records, GMST products, FaIR temperatures, FACTS projections
2. Run DOLS robustness matrix (24 fits + ensemble stats)
3. Run IPCC emergent sensitivity comparison (5 SSPs)
4. Produce DOLS projections (ensemble method, all SSPs)
5. Compute residual decomposition and validate against IMBIE
6. DOLS vs FACTS comparison table
7. Generate main figures (4) and supplementary figures (5)
8. Generate tables (main: 2, supplementary: 4)
9. Export projection CSVs for Zenodo
10. Run tests
11. Print key numbers for manuscript
"""
```

---

## Key numbers for manuscript

The analysis must produce:
- $d\alpha/dT$ thermodynamic ensemble: value ± uncertainty
- $\alpha_0$ range across IPCC SSPs
- Factor-of-two sensitivity discrepancy
- DOLS median projection at 2100 for each SSP (in m, relative to 1995-2014)
- FACTS median at 2100 for each SSP
- Difference (DOLS - FACTS) at 2100 for each SSP (mm and %)
- Residual during GRACE era (mean rate ± unc) vs IMBIE AIS+GrIS

---

## Output checklist

- [ ] 4 main figures
- [ ] ~5 supplementary figures
- [ ] Table 1: Robustness matrix (condensed, main text)
- [ ] Table 2: Projection ranges (main text, IPCC-comparable format)
- [ ] Supplementary tables (full robustness, start-date, IPCC comparison, full projections)
- [ ] Projection CSV files for Zenodo (decadal, all SSPs, quantiles)
- [ ] All tests passing
- [ ] Key numbers summary
