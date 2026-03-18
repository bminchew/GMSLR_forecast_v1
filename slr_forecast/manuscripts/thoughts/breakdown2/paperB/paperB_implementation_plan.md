# Implementation Plan: Paper B — Budget-Consistent Component Projections

## Overview

Paper B has three distinct analytical components:
1. **Budget constraint** — importance-weighted rejection sampling on 7-component FACTS distributions
2. **Asymmetric ISMIP6 filtering** — quantifying which WAIS prior samples are eliminated
3. **WAIS committed + MISI mixture** — separating budget-constrained trajectory from deep-uncertainty tail

The first two components reuse code from the Paper 2 analysis plan (v2). The third is new.

---

## Component vector

| Index | Component | Abbreviation | Forcing |
|-------|-----------|-------------|---------|
| 0 | Thermal expansion | TE | GMST |
| 1 | Glaciers | GL | GMST |
| 2 | Greenland Ice Sheet | GrIS | GMST + North Atlantic SST |
| 3 | West Antarctic Ice Sheet | WAIS | CDW / ENSO |
| 4 | East Antarctic Ice Sheet | EAIS | Precipitation |
| 5 | Antarctic Peninsula | APIS | Atmospheric warming |
| 6 | Terrestrial water storage | LWS | Anthropogenic + ENSO |

---

## Data files required

1. **FACTS v1.0 full MC samples** with WAIS/EAIS/APIS from `full_sample_components` / `dist_components`, all AIS modules (emulandice, larmip, bamber19, deconto21), all SSPs
2. **IMBIE**: WAIS, EAIS, APIS, GrIS reconciled mass balance, 1992–2020
3. **Frederikse et al. (2020)**: total GMSL + total AIS component, 1900–2018
4. **Horwath et al. (2022)**: independent budget with total AIS, 1993–2016
5. **Satellite altimetry**: total GMSL, 1993–present
6. **Argo-era thermosteric**: NOAA/NCEI OHC 0–2000m
7. **GlaMBIE**: glacier mass balance, 2000–2023
8. **FaIR temperature trajectories**: per SSP
9. **Seroussi et al. (2024)**: WAIS collapse statistics from extended ISMIP6 (extract from Table 1 / Figure 3, or from uploaded PDF)
10. **Rosier et al. (2021)**: PIG tipping point analysis (for discussion, not computation)

---

## Project structure

```
paperB_budget_consistent/
├── data/
│   ├── facts/              # Full MC samples, 7-component
│   ├── imbie/              # WAIS, EAIS, APIS, GrIS
│   ├── frederikse/
│   ├── horwath/
│   ├── altimetry/
│   ├── thermosteric/
│   ├── glaciers/
│   ├── fair/
│   └── seroussi2024/       # WAIS collapse statistics
├── src/
│   ├── load_data.py
│   ├── rejection_sampler.py        # Budget constraint (from Paper 2 plan)
│   ├── ismip6_filter.py            # Asymmetric filtering analysis
│   ├── wais_committed.py           # Committed trajectory extraction
│   ├── wais_misi.py                # MISI distribution specification
│   ├── wais_mixture.py             # Mixture model and projections
│   ├── total_projections.py        # Combine all components
│   ├── diagnostics.py              # Variance ratios, correlations, ESS
│   ├── figures_main.py             # 6 main figures
│   ├── figures_supplement.py       # ~9 supplementary figures
│   └── utils.py
├── outputs/
│   ├── projections/                # CSV/NetCDF for Zenodo
│   ├── main_figures/
│   ├── supplementary/
│   └── tables/
├── tests/
│   ├── test_sampler.py
│   ├── test_filter.py
│   ├── test_mixture.py
│   └── test_projections.py
└── run_all.py
```

---

## Step 1: Load and harmonize data

### File: `src/load_data.py`

Same as Paper 2 v2 plan (7-component FACTS loading, IMBIE WAIS/EAIS/APIS, Frederikse, Horwath). No changes.

Additional loader for Seroussi et al. (2024) WAIS collapse statistics:

```python
def load_seroussi2024_collapse_stats() -> pd.DataFrame:
    """
    Extract from Seroussi et al. (2024):
    - Fraction of models showing widespread WAIS collapse at each century
      (2100, 2200, 2300) under each forcing scenario
    - WAIS mass loss distributions at 2100, 2200, 2300
    
    Source: Table 1 and/or Figure 3 of the paper (uploaded PDF).
    
    Returns DataFrame with columns:
        scenario, year, n_models, n_collapsed, fraction_collapsed,
        wais_mass_loss_median_mSLE, wais_mass_loss_p83_mSLE
    """
```

Key numbers to extract from Seroussi et al. (2024):
- Under high emissions: 30-40% of models show large-scale WAIS collapse by 2300
- WAIS contribution ranges up to 4.4 m SLE by 2300 in some models
- Ice-shelf collapse scenarios add ~1.1 m SLE on average by 2300

---

## Step 2: Budget constraint (importance sampling)

### File: `src/rejection_sampler.py`

Identical to Paper 2 v2 plan. Implements global budget constraint, Antarctic sub-budget constraint, and combined constraint. Operates on 7-component FACTS samples.

Run configurations:
- 4 AIS modules × 3 SSPs × 3 constraint types (global, AIS-sub, combined) × 3 sigma_budget values (0.1, 0.3, 0.5)
- Primary configuration: emulandice, SSP2-4.5, combined constraint, sigma_budget = 0.3

---

## Step 3: Asymmetric ISMIP6 filtering analysis

### File: `src/ismip6_filter.py`

This quantifies the budget constraint's role as a physical filter on the ISMIP6 ensemble.

```python
def analyze_wais_filtering(
    prior_samples: np.ndarray,     # (N, 7, T): unconstrained FACTS samples
    weights: np.ndarray,           # (N,): importance weights from budget constraint
    wais_index: int = 3,
    weight_threshold: float = 0.01,  # fraction of max weight below which sample is "eliminated"
) -> dict:
    """
    Analyze which WAIS prior samples survive the budget constraint.
    
    Returns:
        n_eliminated: int           # samples with w < threshold * max(w)
        frac_eliminated: float      # n_eliminated / N
        prior_wais_quantiles: dict  # {5, 17, 50, 83, 95} at each time
        posterior_wais_quantiles: dict
        median_shift: np.ndarray    # (T,): posterior_median - prior_median
        
        # Characteristics of eliminated samples:
        eliminated_wais_rates: np.ndarray   # WAIS rates of eliminated samples
        eliminated_total_rates: np.ndarray  # total rates of eliminated samples
        
        # Key diagnostic: fraction of eliminated samples that show WAIS mass gain
        frac_eliminated_with_mass_gain: float
    """
```

**Key analysis**: Partition the prior WAIS samples into three bins:
1. WAIS mass gain (rate < 0): expect high elimination rate
2. WAIS near-zero loss (rate 0 to 0.5 mm/yr): expect moderate elimination
3. WAIS substantial loss (rate > 0.5 mm/yr): expect low elimination

Compute the survival fraction in each bin. The result should show that the budget constraint preferentially eliminates bins 1 and 2 — precisely the ISMIP6 runs that show Antarctic behavior inconsistent with observations.

```python
def wais_survival_by_bin(
    prior_samples: np.ndarray,
    weights: np.ndarray,
    bins: list = [(-np.inf, 0), (0, 0.5), (0.5, np.inf)],
    wais_index: int = 3,
    time_index: int = -1,   # which time step (default: last available)
) -> pd.DataFrame:
    """
    For each WAIS rate bin, compute:
        n_samples, mean_weight, survival_fraction
    
    survival_fraction = sum(w_i for i in bin) / sum(all w)
                        normalized by n_i / N
    """
```

---

## Step 4: WAIS committed trajectory

### File: `src/wais_committed.py`

Extract the budget-constrained WAIS rate and acceleration at the end of the observational period, then extrapolate forward.

```python
def extract_committed_trajectory(
    prior_samples: np.ndarray,     # (N, 7, T)
    weights: np.ndarray,           # (N,)
    times: np.ndarray,             # (T,)
    wais_index: int = 3,
    obs_end_year: float = 2020,
) -> CommittedTrajectory:
    """
    From the budget-constrained posterior:
    1. Compute weighted mean WAIS rate at obs_end_year: r_0
    2. Compute weighted mean WAIS acceleration: dr_dt
       (finite difference of posterior mean rate over last 2 time steps)
    3. Compute posterior uncertainty on r_0 and dr_dt
       (weighted variance + covariance)
    
    Returns CommittedTrajectory with fields:
        r_0: float              # mm/yr at obs_end_year
        r_0_unc: float          # 1-sigma
        dr_dt: float            # mm/yr^2
        dr_dt_unc: float
        cov_r0_drdt: float      # covariance
    """


def project_committed(
    ct: CommittedTrajectory,
    projection_years: np.ndarray,
    n_mc: int = 10000,
) -> np.ndarray:
    """
    Forward projection of committed trajectory.
    
    rate(t) = r_0 + dr_dt * (t - t_0)
    
    No second derivative (conservative: assumes acceleration does not
    itself accelerate). This is the "no surprises" projection.
    
    Propagate uncertainty via MC draws from bivariate normal (r_0, dr_dt).
    
    Returns: np.ndarray of shape (n_mc, n_years) — WAIS rates in mm/yr
    """
```

**Important**: The committed trajectory comes from the budget-constrained posterior, not from IMBIE directly. The budget constraint shifts the WAIS rate estimate (by eliminating low-end samples), so the committed trajectory is different from a naive IMBIE extrapolation. This is a feature: the committed trajectory is anchored by the full observational record (total GMSL + component data), not just by the WAIS-specific IMBIE data.

---

## Step 5: MISI distribution specification

### File: `src/wais_misi.py`

Specify the WAIS rate distribution conditional on MISI activation.

```python
def extract_misi_distribution_from_bamber(
    bamber_wais_samples: np.ndarray,   # bamber19 WAIS rates, shape (N, T)
    committed_trajectory: np.ndarray,   # committed projection, shape (n_mc, T)
    time_index: int,                    # which time step
) -> MISIDistribution:
    """
    The Bamber et al. (2019) distribution implicitly includes MISI
    (the experts were asked about scenarios including instability).
    
    Procedure:
    1. Compute committed p83 at the specified time step.
    2. Select Bamber samples exceeding committed p83.
       These represent the expert-assessed MISI/high-end tail.
    3. Compute the excess rate: bamber_rate - committed_median.
       This is the MISI-attributable additional mass loss.
    4. Fit a shifted log-normal (or generalized Pareto) to the excess.
    
    Returns MISIDistribution with fields:
        distribution_type: str ('lognormal' or 'gpd')
        params: dict          # mu, sigma for lognormal; xi, sigma, mu for GPD
        raw_excess_samples: np.ndarray
    """


def extract_misi_distribution_from_deconto(
    deconto_wais_samples: np.ndarray,
    emulandice_wais_samples: np.ndarray,
    time_index: int,
) -> MISIDistribution:
    """
    Alternative: use the difference between deconto21 and emulandice
    WAIS distributions as the MISI-attributable component.
    
    deconto21 includes MICI; emulandice does not.
    The excess = deconto21 - emulandice represents the MICI contribution.
    """


def extract_misi_from_rheology_scaling(
    committed_trajectory: np.ndarray,
    scaling_factor: float = 0.32,   # Getraer & Morlighem: 32% more by 2100
    scaling_unc: float = 0.14,      # ± 14%
) -> MISIDistribution:
    """
    Alternative: apply n=4 rheology scaling to the committed trajectory.
    The additional mass loss from n=4 vs n=3 provides a physics-based
    estimate of the missing signal.
    
    This is not strictly MISI — it is a rheology correction. But it
    compounds with MISI (grounding-line retreat is faster with n=4,
    so MISI activates sooner and proceeds faster). Present as a
    sensitivity analysis.
    """
```

**Recommendation**: Use Bamber as primary source for $p_{\text{MISI}}$. Use DeConto and rheology scaling as sensitivity checks in supplement.

---

## Step 6: WAIS mixture model and projections

### File: `src/wais_mixture.py`

```python
def wais_mixture_projection(
    committed_rates: np.ndarray,      # (n_mc, n_years): from Step 4
    misi_distribution: MISIDistribution,  # from Step 5
    pi_values: list = [0.0, 0.05, 0.10, 0.25, 0.50],
    projection_years: np.ndarray = None,
    n_mc: int = 10000,
) -> dict:
    """
    For each pi value, produce WAIS projection distribution.
    
    Algorithm:
    For each MC draw m = 1, ..., n_mc:
        1. Draw u ~ Uniform(0, 1)
        2. If u > pi: draw from committed trajectory
           rate_WAIS^(m)(t) = committed_rates[m, :]
        3. If u <= pi: draw from committed + MISI excess
           rate_WAIS^(m)(t) = committed_rates[m, :] + misi_excess^(m)(t)
           where misi_excess is drawn from misi_distribution
    
    Returns dict keyed by pi with values:
        np.ndarray of shape (n_mc, n_years) — WAIS rates
    """
```

**Time-dependent $\pi$**: For each target $\pi(2100)$, ramp linearly from $\pi(2020) = 0$:

$$\pi(t) = \pi_{2100} \cdot \frac{t - 2020}{2100 - 2020} \quad \text{for } t \in [2020, 2100]$$

For $t > 2100$, $\pi$ continues to increase (MISI becomes more likely with continued warming). Use a logistic curve that saturates at some $\pi_{\max}$:

$$\pi(t) = \pi_{\max} \cdot \frac{1}{1 + \exp(-(t - t_{\text{mid}})/\tau)}$$

with $t_{\text{mid}}$ and $\tau$ chosen to match $\pi(2100)$ and reach saturation by $\sim$2200. For the primary analysis, use the linear ramp to 2100 and constant $\pi$ thereafter. Logistic ramp in supplement.

---

## Step 7: Total GMSL projections

### File: `src/total_projections.py`

```python
def total_gmsl_projections(
    non_wais_samples: np.ndarray,     # (N, 6, T): budget-constrained TE,GL,GrIS,EAIS,APIS,LWS
    non_wais_weights: np.ndarray,     # (N,): importance weights
    wais_mixture: dict,               # {pi: np.ndarray(n_mc, T)} from Step 6
    projection_years: np.ndarray,
) -> dict:
    """
    Combine budget-constrained non-WAIS components with WAIS mixture.
    
    For each pi:
    1. Resample non-WAIS components according to importance weights.
    2. Add WAIS mixture draws.
    3. Sum to get total GMSL rate at each time step.
    4. Integrate to get cumulative GMSL.
    5. Compute quantiles.
    
    Returns dict keyed by (ssp, pi) with ProjectionResult:
        years, trajectories, quantiles
    """
```

Format output tables to match FACTS Table A1 for direct comparison:

```python
def format_projection_tables(
    projections: dict,
    years: list = [2050, 2100, 2150],
) -> pd.DataFrame:
    """
    Columns: SSP, pi, Year, Median (m), 17th (m), 83rd (m), 5th (m), 95th (m)
    All relative to 1995-2014, in meters.
    """
```

---

## Step 8: Diagnostics

### File: `src/diagnostics.py`

From Paper 2 v2 plan:
- Variance ratio $\rho_j$ for all 7 components
- 7×7 pairwise correlation matrix (prior and posterior)
- ESS tracking
- KL divergence (prior to posterior)

Additional for Paper B:
- WAIS filtering statistics (from Step 3)
- Committed trajectory parameters (r_0, dr/dt) and their posterior uncertainty
- Comparison of WAIS projections across pi values

---

## Step 9: Figures

### Main figures (6)

#### Figure 1: WAIS prior vs posterior (the asymmetric filter)

```
Layout: 2 panels (t=2050, t=2100)
  Each panel:
  - Gray fill: WAIS prior distribution (KDE)
  - Red line: WAIS posterior (budget-constrained)
  - Shading: eliminated region (low-tail truncation)
  - Vertical lines: prior and posterior medians
  - Text: fraction eliminated, median shift
  
Module: emulandice/AIS, SSP2-4.5, combined constraint
```

#### Figure 2: Mixture model schematic

```
Layout: Single panel at t=2100, SSP5-8.5
  - Narrow blue distribution: p_committed (budget-constrained, no MISI)
  - Wide red distribution: p_MISI (Bamber tail)
  - Multiple mixture lines at pi = 0.05, 0.10, 0.25, 0.50
  - Each mixture is a weighted sum, showing how the tail extends with pi
  
This is the conceptual figure. Computed from actual distributions, not schematic.
```

#### Figure 3: WAIS projection distributions

```
Layout: 2x2 (SSP2-4.5 and SSP5-8.5 × t=2050 and t=2100)
  Each panel: overlaid density plots at different pi values
  Color gradient: pi=0 (blue) to pi=0.5 (red)
  
Show how the distribution changes with pi:
  - Low pi: narrow, budget-anchored center
  - High pi: extended right tail, MISI contribution visible
```

#### Figure 4: Total GMSL projection fan chart

```
Layout: 3 panels (SSP1-2.6, SSP2-4.5, SSP5-8.5)
  Each panel:
  - Solid line + dark shading: pi = 0.10 (moderate, 17-83%)
  - Light shading: pi = 0.25 (substantial, 5-95%)
  - Dashed line: FACTS unconstrained median for comparison
  - Black line: observed GMSL (hindcast)
  
Key visual: the budget-constrained projection with moderate MISI has
a narrower center but wider upper tail than FACTS.
```

#### Figure 5: Posterior correlation matrix (7×7)

```
Layout: 2 panels side-by-side
  Left: Prior correlations (should be ~zero, components drawn independently)
  Right: Posterior correlations (budget-constrained)
  
7×7 matrix: TE, GL, GrIS, WAIS, EAIS, APIS, LWS
Diverging colormap, annotated with values.

Key: WAIS-TE strongly negative, WAIS-EAIS negative (Antarctic sub-collider)
```

#### Figure 6: WAIS filtering by rate bin

```
Layout: Single panel (bar chart or stacked bar)
  - x-axis: WAIS rate bins (mass gain, near-zero, moderate loss, high loss)
  - y-axis: fraction of prior samples in each bin
  - Two bars per bin: prior fraction (gray) and posterior fraction (red)
  - Or: survival rate per bin
  
Purpose: Concrete visualization that the budget eliminates mass-gain 
and near-zero-loss runs.
```

### Supplementary figures (~9)

- Fig S1: AIS cross-validation (IMBIE vs Frederikse vs Horwath)
- Fig S2: Variance ratio heatmap (7 components)
- Fig S3: Constraint level comparison (global vs AIS-sub vs combined) for WAIS
- Fig S4: WAIS prior vs posterior for all 4 AIS modules
- Fig S5: MISI distribution: Bamber vs DeConto vs rheology scaling
- Fig S6: pi(t) sensitivity: total GMSL at 2100 as function of pi(2100)
- Fig S7: n=4 rheology scaling effect
- Fig S8: ESS vs sigma_budget
- Fig S9: Budget residual time series (1900-2020) with IMBIE overlay

---

## Step 10: Validation tests

### File: `tests/test_sampler.py`

From Paper 2 v2 plan:
- Known-answer Gaussian test (7 components + 2 constraints)
- Uniform weights under no constraint
- ESS monotonicity with sigma_budget
- Budget conservation (weighted total matches observed)
- Antarctic sub-budget consistency
- Constraint ordering invariance

### File: `tests/test_filter.py`

```python
# Test F1: Elimination fraction is monotonically increasing as the 
#          budget tolerance decreases (tighter constraint eliminates more)
# Test F2: All eliminated samples have total rate below observed
#          (by construction — they were eliminated because total was too low)
# Test F3: Among eliminated samples, WAIS rate is systematically lower
#          than among surviving samples
# Test F4: Under no constraint, elimination fraction = 0
```

### File: `tests/test_mixture.py`

```python
# Test M1: At pi=0, mixture distribution equals committed distribution
# Test M2: At pi=1, mixture distribution equals committed + MISI excess
# Test M3: Mixture median increases monotonically with pi
# Test M4: Mixture 95th percentile increases faster than median with pi
#          (tail extension, not symmetric shift)
# Test M5: Committed trajectory at t=2020 matches budget-constrained 
#          WAIS posterior mean at t=2020 (initial condition consistency)
# Test M6: MISI excess is non-negative (MISI adds mass loss, not reduces it)
```

### File: `tests/test_projections.py`

```python
# Test P1: Total GMSL = sum of all 7 components for every MC draw
# Test P2: Projection quantiles are monotonic in SSP 
# Test P3: At pi=0, total projection is similar to budget-constrained 
#          FACTS (no MISI, just constraint applied)
# Test P4: At pi=0.5, SSP5-8.5 95th percentile exceeds FACTS 95th 
#          (MISI extends the tail beyond what FACTS produces)
# Test P5: Total projection at 2050 is insensitive to pi choice
#          (MISI has not had time to contribute much at short horizons)
```

---

## Step 11: Master script

```python
"""
Paper B master script.

Steps:
1.  Load data: FACTS 7-component, IMBIE, Frederikse, Horwath, 
    altimetry, FaIR, Seroussi 2024 collapse stats
2.  Run budget constraint (importance sampling) for all configurations
3.  Analyze ISMIP6 filtering (Step 3): survival fractions, median shifts
4.  Extract committed trajectory from posterior (Step 4)
5.  Specify MISI distribution from Bamber tail (Step 5)
6.  [Supplement] Specify MISI from DeConto and rheology scaling
7.  Produce WAIS mixture projections for all pi values (Step 6)
8.  Produce total GMSL projections (Step 7)
9.  Compute all diagnostics (Step 8)
10. Generate 6 main figures
11. Generate 9 supplementary figures  
12. Generate tables (main: 4, supplementary: 5)
13. Export projection CSVs/NetCDFs for Zenodo
14. Run all validation tests (4 test files)
15. Print key numbers for manuscript
"""
```

---

## Key numbers for manuscript

- Thermodynamic ensemble $d\alpha/dT$ (from Paper A, cited)
- WAIS prior median and 17-83% range (emulandice, SSP2-4.5, 2100)
- WAIS posterior median and 17-83% range (budget-constrained)
- Variance ratio $\rho_{\text{WAIS}}$
- Fraction of emulandice WAIS samples effectively eliminated
- Fraction of eliminated samples showing WAIS mass gain
- Committed trajectory: r_0 ± unc, dr/dt ± unc
- WAIS projection at 2100: median (17-83) [5-95] for each SSP × pi
- Total GMSL projection at 2100: same format
- Comparison: total projection at pi=0.10 vs FACTS unconstrained

---

## Estimated compute time

- Data loading: ~3 min
- Budget constraint (all configurations): ~30 min
- ISMIP6 filter analysis: ~2 min
- Committed trajectory extraction: ~1 min
- MISI distribution fitting: ~2 min
- Mixture projections (5 pi × 3 SSP × 10,000 MC): ~10 min
- Total projections: ~5 min
- Diagnostics: ~5 min
- Figures: ~5 min
- Tests: ~5 min
- **Total: ~70 min**

---

## Output checklist

Main deliverables:
- [ ] 6 main figures
- [ ] Table 1: Component definitions
- [ ] Table 2: Variance ratios (7 components)
- [ ] Table 3: WAIS projection quantiles (SSP × pi)
- [ ] Table 4: Total GMSL projection quantiles (SSP × pi, IPCC-comparable)

Supplementary:
- [ ] 9 supplementary figures
- [ ] 5 supplementary tables
- [ ] Extended methods

Data products:
- [ ] Projection distributions (CSV/NetCDF) for Zenodo
- [ ] Budget constraint post-processing code (GitHub)

Analysis:
- [ ] All 4 test files passing
- [ ] ISMIP6 filtering statistics
- [ ] Key numbers summary
- [ ] README with data provenance
