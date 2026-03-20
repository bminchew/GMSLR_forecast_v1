# Refactor Guide: Publishable Reproducibility Container

This document catalogues the paper plan, figure/analysis inventory, notebook partitioning, code quality checks, and architectural decisions for refactoring the project into a standalone, publishable container.

---

## 0. Paper Plan

### Target
PNAS Research Article (~6 pages + SI). If short enough: Science (~3000 words + methods/SI).

### Working Title
"Data-driven sea-level projections consistently exceed process-model estimates: a three-step framework for observation-based forecasting"

### Thesis
IPCC medium-confidence SLR projections are structurally biased low because the ice sheet process models that inform them (ISMIP6) have known, quantifiable deficiencies. Data-driven approaches that do not share these biases produce higher central estimates with wider uncertainty ranges. We present a three-step framework of increasing physical complexity that (1) establishes observational baselines, (2) quantifies the aggregate temperature–sea level relationship, and (3) decomposes it into physically distinct components to identify exactly where observational constraints end and structural model uncertainty begins.

### Paper Structure

**Section 1: Introduction**
- SLR is among the most consequential impacts of climate change
- IPCC projections rely on process models with known biases (cite Aschwanden et al. 2021, Fricker et al. 2025, Martin et al. in press)
- Data-driven approaches provide an independent, complementary constraint
- We present a hierarchical three-step framework

**Section 2: Observational Extrapolation (Step 1)**
- Quadratic fit to NASA satellite altimetry (1993–2023)
- Three-component error budget following Hamlington et al. (2024)
- Extrapolation to 2050/2100 as a baseline — this is what the data say without any model
- Key result: even naive extrapolation exceeds IPCC medium-confidence for SSP2-4.5 and above

**Section 3: Aggregate Semi-Empirical Model (Step 2)**
- GMSL rate = a·T² + b·T + c, calibrated on Frederikse 1900–2018
- Bayesian level-space and rate-and-state models
- Satellite-era rate prior constrains end-of-record rate
- Key result: quadratic (dα/dT > 0) is well-supported by data
- Key insight: the quadratic arises from land ice overtaking thermosteric as the dominant driver — Simpson's paradox in the component mix

**Section 4: Component-Wise Decomposition (Step 3)**
- Thermosteric: linear or weakly quadratic in GMST (well-constrained)
- Glaciers: linear in GMST with reservoir depletion (well-constrained)
- Greenland: linear in Greenland surface temperature (moderately constrained)
- EAIS: zero contribution (consistent with observations)
- Antarctic Peninsula: slight linear (small)
- WAIS: not constrained by temperature — use Bamber et al. (2019) SEJ or scenario framework (A4)
- TWS: adopted from IPCC
- Key result: the component decomposition explains the aggregate quadratic — it is not a single process accelerating but a shifting mixture
- Key result: projection uncertainty at 2100 is dominated by WAIS, where process models are weakest and observations provide the least constraint
- Key result: compare component-wise data decomposition to IPCC component by component to show source of IPCC divergence from data 

**Section 5: Comparison with IPCC Projections**
- Side-by-side comparison at 2050, 2100 for SSP2-4.5 and SSP5-8.5
- All three steps produce higher central estimates than IPCC medium-confidence
- The discrepancy is explained by known ice sheet model biases
- Component breakdown allows for value of information estimates based on uncertainties from individual components relative to total uncertainty, including scenario uncertainty (choices societies make) 

**Section 6: Discussion**
- Data-driven approaches as a benchmark for process-model projections
- The three-step framework as a template for future assessment cycles
- Limitations: aggregate model conflates components; component model depends on WAIS scenario assumptions
- Implications for coastal planning: the "most likely" pathway tracks above IPCC medians

### Key Points for Reviewers
- The claim is NOT "IPCC is wrong" — it is "IPCC medium-confidence projections have a known low bias from ice sheet process models, and data-driven methods that avoid this bias produce higher estimates"
- The three-step hierarchy provides internal consistency checks
- The component decomposition is the scientific core: it explains why the aggregate model works, when it will fail, and where the uncertainty actually lives
- Communicate need for reduced order-data driven models because process-based models in IPCC are enormously complex -- find ways to quantify model complexity (degress of freedom, dimensions in parameter space, etc); compare SLR projection model complexity to major scientific breakthroughs like human genome project (from Claude browser interface: Cite Green & Armstrong (2015) for the 27% average error increase from complexity — it is a meta-analytic result, not a single competition, which makes it harder to dismiss. Cite Makridakis et al. (2018, M4 results, International Journal of Forecasting 36(1): 54–74) for the finding that forecast combinations outperform individual methods. Do not lean on M1–M3 alone, because a reviewer can counter with M5. Instead, frame the condition: "In data-limited, short-horizon settings — precisely the regime of decadal GMSL projection — the forecasting literature consistently finds that parsimonious methods outperform complex structural models (Green & Armstrong, 2015), and that optimal forecast combinations outperform individual methods (Makridakis et al., 2018)."

---

## 0b. Analyses, Figures, and Notebook Mapping

### Main Text Figures (target: 4–5 for PNAS, 3–4 for Nature)

| Fig | Description | Source Notebook | Status |
|-----|-------------|-----------------|--------|
| 1 | **Observations + three-step overview.** (a) Frederikse GMSL 1900–2018 with satellite-era quadratic overlay. (b) Component budget breakdown (Frederikse steric, glaciers, Greenland, Antarctica, TWS) showing the shift in dominance. | `component_decomposition.ipynb`, `playing_w_figures.ipynb` | Partially exists (`component_frederikse_overview.png`); needs panel (a) added |
| 2 | **Semi-empirical GMSL–GMST fit.** (a) Bayesian level-space fit to Frederikse GMSL (observed vs modeled). (b) Rate vs temperature showing quadratic relationship with posterior uncertainty. (c) Rate prior diagnostic: posterior rate vs satellite-era constraint. | `bayesian_ratestate.ipynb` | Components exist (`ratestate_fit_residuals.png`, `ratestate_rate_prior_diagnostic.png`); needs assembly into multi-panel |
| 3 | **Component-wise fits and projections.** (a) Thermosteric rate vs T (linear). (b) Glacier rate vs T. (c) Greenland rate vs Greenland T. (d) Combined component projections with budget closure. | `component_decomposition.ipynb` | Components exist (`component_rate_vs_temperature.png`, `component_stage1_projections.png`, etc.); needs assembly |
| 4 | **Projection comparison at 2100.** (a) Time series: IPCC medium-confidence vs three-step estimates for SSP2-4.5. (b) Probability distributions at 2100 for SSP2-4.5 and SSP5-8.5, showing IPCC, extrapolation, semi-empirical, and component-wise. (c) Variance decomposition: stacked area showing which component dominates uncertainty at each time horizon. | `predictability_analysis.ipynb`, `bayesian_ratestate.ipynb` | Components exist (`gmslr_projections_comparison.png`, `gmslr_projection_histograms.png`, `ratestate_variance_decomposition.png`); needs unification |

### Supplementary Figures

| Fig | Description | Source Notebook | Status |
|-----|-------------|-----------------|--------|
| S1 | Satellite-era quadratic fit details: NASA altimetry data, three-component error budget breakdown, comparison with Hamlington et al. (2024) | `bayesian_ratestate.ipynb` | Exists (`ratestate_satellite_era_prior.png`) |
| S2 | MCMC diagnostics: trace plots, corner plot, convergence (R-hat, ESS) for level-space and rate-and-state models | `bayesian_ratestate.ipynb` | Exists (`ratestate_traces.png`, `ratestate_corner.png`) |
| S3 | Rate-and-state model: τ posterior, state variable S(t), limiting behaviour τ→0 | `bayesian_ratestate.ipynb` | Exists (`ratestate_tau_analysis.png`, `ratestate_state_variable_demo.png`) |
| S4 | Prior sensitivity: PC prior on acceleration coefficient a — sweep over prior scales, impact on rate and projections | `bayesian_ratestate.ipynb` | Exists (`ratestate_prior_sensitivity.png`) |
| S5 | Joint rate–accel posterior vs satellite-era prior ellipse | `bayesian_ratestate.ipynb` | Exists (`ratestate_rate_accel_joint.png`) |
| S6 | Multi-dataset robustness: DOLS coefficients across 7 GMSL × 4 GMST combinations, forest plot | `slr_analysis_notebook.ipynb` or `predictability_analysis.ipynb` | Exists (`dols_robustness_forest.png`, `dols_robustness_ensemble.png`) |
| S7 | Sliding-window DOLS: time-varying coefficients, bandwidth sensitivity, SAOD comparison | `predictability_analysis.ipynb` | Exists (`dols_sliding_*.png`, 8 figures) |
| S8 | Bayesian DLM: time-varying coefficients via Kalman filter/RTS smoother, Q posterior | `bayesian_level_space.ipynb` | Exists (`bayesian_dlm_*.png`) |
| S9 | Hierarchical multi-dataset pooling: forest plot, shrinkage analysis | `bayesian_level_space.ipynb` | Exists (`bayesian_hierarchical_*.png`) |
| S10 | τ sensitivity analysis: projections for fixed τ ∈ {1, 10, 30, 50, 100, 200, 500} yr | `bayesian_ratestate.ipynb` | Exists (`ratestate_tau_sensitivity.png`) |
| S11 | Hindcast cross-validation: leave-future-out skill scores | `predictability_analysis.ipynb` | Exists (`hindcast_crossvalidation.png`) |
| S12 | Component-level model selection: BIC comparison for linear vs quadratic thermosteric | `component_decomposition.ipynb` | Exists (`component_rate_vs_T_model_selection.png`) |
| S13 | Thermosteric two-layer ocean state model: upper/deep temperature states, posteriors | `component_decomposition.ipynb` | Exists (`component_stage1b_*.png`) |
| S14 | Greenland physics decomposition: SMB vs discharge, local vs global temperature sensitivity | `component_decomposition.ipynb` | Exists (`component_greenland_*.png`) |
| S15 | IPCC component breakdown: AR6 FACTS projections by component and confidence level | `predictability_analysis.ipynb` | Exists (`ipcc_components_dual_confidence.png`) |
| S16 | WAIS scenario framework (A4): scenario weights, rheology correction, stochastic amplification | `predictability_analysis.ipynb` | Exists (`physics_informed_wais_uncertainty.png`) |
| S17 | IMBIE vs IPCC ice sheet comparison: observed mass loss vs ISMIP6 ensemble | `wais_data_ipccar6.ipynb` | Exists (`wais_data_ipccar6_*.png`) |
| S18 | Observational component rates (2002–2018): budget closure validation | `component_decomposition.ipynb` | Exists (`component_rate_timeseries_obs_vs_model.png`) |

### Notebook Partitioning

**Main text notebooks** (must run cleanly, produce publication figures):

| Notebook | Paper Section | Produces |
|----------|--------------|----------|
| `bayesian_ratestate.ipynb` | §2 (rate prior), §3 (semi-empirical) | Figs 2, S1–S5, S10 |
| `component_decomposition.ipynb` | §4 (component fits) | Figs 1b, 3, S12–S14, S18 |
| `predictability_analysis.ipynb` | §4 (WAIS), §5 (projections) | Figs 4, S6–S7, S11, S15–S16 |

**Supplementary notebooks** (support claims, not needed for main figures):

| Notebook | Role |
|----------|------|
| `bayesian_level_space.ipynb` | S8–S9 (DLM, hierarchical — methodological alternatives) |
| `slr_analysis_notebook.ipynb` | S6 (multi-dataset robustness) |
| `wais_data_ipccar6.ipynb` | S17 (IMBIE vs ISMIP6 comparison) |
| `read_process_datafiles.ipynb` | Data processing pipeline (not for publication figures) |

**Not needed for publication** (archive or remove):

| Notebook | Reason |
|----------|--------|
| `playing_w_figures.ipynb` | Exploration / figure prototyping |
| `hybrid_forecast.ipynb` / `*_executed.ipynb` | Superseded by rate-and-state framework |
| `slr_heuristic_projections.ipynb` / `*_executed.ipynb` | Superseded by three-step framework |
| `stats_v_process.ipynb` | Early exploratory analysis |
| `wais_scenario_uncertainties.ipynb` | Superseded by A4 framework in predictability_analysis |

### Analyses Still Needed (before paper)

| Analysis | Notebook | Priority | Notes |
|----------|----------|----------|-------|
| Complete component-level projections with budget closure | `component_decomposition.ipynb` | **Critical** | Must show Σcomponents ≈ total GMSL at 2100 |
| Component-level variance decomposition | `component_decomposition.ipynb` | **Critical** | Which component dominates uncertainty at each horizon |
| Side-by-side comparison of all three steps at 2050/2100 | New or `predictability_analysis.ipynb` | **Critical** | The paper's central figure |
| Sensitivity to WAIS scenario weights | `predictability_analysis.ipynb` | High | How much do results change with different P(MISI)? |
| Run rate-and-state model with NASA rate prior | `bayesian_ratestate.ipynb` | High | Already coded, needs execution |
| Validate component fits against IMBIE/GlaMBIE observations | `component_decomposition.ipynb` | High | Cross-check: model-implied cryospheric rates vs observed |

---

## 1. Sign Convention: Mass Loss vs Sea Level Rise

### The Problem
Two incompatible sign conventions coexist in the raw data and have already caused bugs:

| Convention | Positive means | Used by |
|---|---|---|
| **Glaciology** | Mass gain (ice grows) | IMBIE Gt-format CSVs, most glaciology literature |
| **Sea-level rise (SLR)** | Sea level rises | Frederikse budget, IPCC AR6 FACTS, IMBIE mm-SLE CSVs |

The IMBIE data ships in *both* formats. The Gt files (e.g., `imbie_west_antarctica_2021_Gt.csv`) use glaciology convention; the mm files (e.g., `imbie_west_antarctica_2021_mm.csv`) use SLR convention. Mixing them without explicit conversion produces silently wrong results: rates appear to decrease with warming when they should increase.

### Rules for the Refactor

1. **Adopt a single project-wide convention and document it at the top of every module.**
   The natural choice for this project is **SLR convention** (positive = sea level rise).

2. **Convert at the point of ingestion, never downstream.**
   Each data reader function must return data in SLR convention. No analysis code should ever need to negate mass-balance values.

3. **Assert sign expectations after every load.**
   For example, WAIS cumulative SLR contribution at 2020 must be positive (~5 mm). EAIS should be near zero or slightly negative. Add explicit sanity-check assertions in every reader.

4. **Tag DataFrames with convention metadata.**
   Use `df.attrs['sign_convention'] = 'slr'` so downstream code can verify programmatically.

5. **Watch for sigma sign quirks.**
   The IMBIE mm-SLE files store uncertainty values as negative numbers. This is a data formatting quirk, not a physical convention. Always apply `np.abs()` to sigma columns immediately after loading, and assert positivity.

### Known Instances
- `imbie_west_antarctica_2021_mm.csv`: all sigma values negative
- `imbie_east_antarctica_2021_mm.csv`: all sigma values negative
- `imbie_antarctic_peninsula_2021_mm.csv`: all sigma values negative
- `harmonized/df_imbie_wais_h` in HDF5: sigma negative (inherited from mm CSV during processing)
- Gt readers (`read_imbie_east_antarctica`, etc.) return glaciology convention without warning

---

## 2. Baseline Alignment and Tracking

### The Problem
All sea-level and temperature anomalies are defined relative to a baseline period. If two datasets use different baselines, their difference is meaningless. The current codebase uses `BASELINE_YEAR = 2005.0` with a 1995-2005 averaging window, but this is set in notebook cells and must be manually kept consistent across notebooks.

### Rules for the Refactor

1. **Define the baseline once, in a single config file.**
   ```python
   # config.py
   BASELINE_YEAR = 2005.0
   BASELINE_WINDOW = (1995, 2005)
   ```
   Every module imports from this file. No notebook cell should define its own baseline.

2. **Every rebaseline operation must be a named function, not inline code.**

3. **Tag DataFrames with baseline metadata.**

4. **Assert baseline consistency before combining datasets.**

5. **IPCC FACTS projections are relative to 2005 by construction.** Document the match explicitly.

6. **IMBIE records start in 1992 and are rebased to ~2005.** Document sensitivity.

---

## 3. Unit Checks and Tracking

### The Problem
The most common unit in the project is meters (sea level), but raw data arrives in mm, Gt, cm, and mm/yr. The conversion factor 362.5 (Gt ice to mm SLE, based on ocean area) appears in multiple files. Mistakes compound silently.

### Rules for the Refactor

1. **Define all conversion constants in one place** (`units.py`).

2. **Standard internal units:**
   | Quantity | Internal unit |
   |---|---|
   | Sea level | meters (m) |
   | Temperature | degrees Celsius (degC), anomaly relative to baseline |
   | Time | decimal years (yr) |
   | Rate | m/yr |
   | Acceleration | m/yr² |

3. **Convert to display units only at the plotting layer.**

4. **Tag every DataFrame with unit metadata.**

5. **Add unit-checking assertions to key functions** (e.g., warn if sea-level values > 1.0, likely mm not m).

---

## 4. Reproducibility and Determinism

1. **Pin all random seeds** in `config.py`.
2. **Pin all package versions** in `requirements.txt` with exact pins.
3. **No intermediate state on disk that is also computed in code.** Ship the processed HDF5 as fixed artifact OR generate deterministically from raw data.
4. **Every figure must be regenerable from a single command.**

---

## 5. Data Provenance and Minimality

1. **Ship only the data needed to reproduce published results.** Audit every file against actual imports.
2. **Include a data manifest with checksums.**
3. **Document every dataset's provenance** in `DATA_SOURCES.md`.
4. **No derived data in `data/raw/`.** The processed HDF5 belongs in `data/processed/` only.
5. **New data file**: `data/raw/gmslr/ablain2019_gmsl_error_covariance.nc` (5.3 MB, doi:10.17882/58344) — required for three-component error budget in satellite-era rate prior.

### Likely Unused (verify before removing)
- `data/raw/gmslr/dangendorf2024_KSSL_SEfin.mat` (653 MB) — spatial fields, not used
- `data/raw/saod/GloSSAC_V2.23_NC4.nc` (530 MB) — SAOD tested and found not significant
- `data/raw/tws/GRCTellus*.nc` (1.1 GB) — may only need pre-extracted global mean
- `data/raw/ipcc_ar6/*.pdf` (documentation, not data)
- `notebooks/archive/` (deprecated code)

---

## 6. Code Organization

### Target Structure
```
slr_forecast/
  pyproject.toml
  Dockerfile
  Makefile                    # make data, make figures, make paper
  DATA_SOURCES.md
  README.md
  src/
    slr_forecast/
      __init__.py
      config.py               # BASELINE_YEAR, N_SAMPLES, seeds, paths
      units.py                # M_TO_MM, GT_TO_M_SLE, conversion functions
      readers/
        __init__.py
        gmsl.py               # Frederikse, Dangendorf, NASA altimetry
        gmst.py               # Berkeley Earth, GISTEMP
        ice_sheets.py         # IMBIE readers (all sign-corrected to SLR)
        glaciers.py           # GlaMBIE
        ipcc.py               # AR6 FACTS component reader
        forcing.py            # SAOD, ENSO
      analysis/
        __init__.py
        dols.py               # calibrate_dols, DOLSResult
        kinematics.py         # compute_kinematics, KinematicsResult
        bayesian.py           # Bayesian models (static, DLM, level-space, rate-and-state)
        satellite_era.py      # fit_satellite_era_quadratic, error budget
        projections.py        # MC ensemble projections
      plotting/
        __init__.py
        timeseries.py
        projections.py
        components.py
  scripts/
    process_raw_data.py       # Raw → HDF5
    run_analysis.py           # Main analysis pipeline
    generate_figures.py       # All figures
  notebooks/                  # Publication notebooks (main + supplement)
    bayesian_ratestate.ipynb
    component_decomposition.ipynb
    predictability_analysis.ipynb
    bayesian_level_space.ipynb      # supplementary
    wais_data_ipccar6.ipynb         # supplementary
  tests/
    test_dols.py
    test_readers.py
    test_units.py
    test_sign_conventions.py
    test_satellite_era_fit.py       # Validates against Hamlington et al.
  data/
    raw/
    processed/
  figures/
```

---

## 7. Code Quality Checks Before Submission

### Consistency Checks
- [ ] All notebooks run top-to-bottom without error (restart kernel + run all)
- [ ] No hard-coded "magic numbers" in notebooks — all constants from config
- [ ] `sat_quad` in `bayesian_ratestate.ipynb` uses NASA altimetry with full error budget (not Frederikse)
- [ ] Rate prior values printed in notebook match Hamlington et al. within stated uncertainty
- [ ] Component projections sum to total GMSL within budget closure tolerance
- [ ] WAIS scenario weights (A4) sum to 1.0
- [ ] All sigma values are positive after loading
- [ ] All sea-level data in SLR convention (positive = rise)
- [ ] Baseline consistent across all datasets (BASELINE_YEAR = 2005.0)
- [ ] Unit consistency: all internal computation in meters; mm only for display

### Efficiency Checks
- [ ] No duplicate MCMC runs — each model is fit once, results cached
- [ ] No redundant data loads — HDF5 opened once per notebook
- [ ] MCMC convergence diagnostics pass (R-hat < 1.05, ESS > 400) for all models
- [ ] Monte Carlo ensemble size (N_SAMPLES) sufficient for stable percentiles
- [ ] Ablain covariance matrix loaded once, not per-call

### Scientific Checks
- [ ] Satellite-era quadratic fit reproduces Hamlington et al. (2024) at midpoint
- [ ] Aggregate semi-empirical rate at end of record consistent with satellite-era prior
- [ ] Component-level thermosteric sensitivity consistent with IPCC range
- [ ] Glacier sensitivity consistent with GlaMBIE consensus trends
- [ ] Greenland sensitivity consistent with IMBIE observed mass loss rates
- [ ] WAIS contribution at 2100 consistent with Bamber et al. (2019) SEJ ranges
- [ ] Total projection at 2100 brackets IPCC medium-confidence (higher central, wider CI)
- [ ] Hindcast cross-validation shows positive skill for all three steps

---

## 8. Notebook Hygiene

1. **Notebooks must not define functions that are used by other cells.** All reusable logic lives in the package.
2. **Notebooks must be runnable top-to-bottom without manual intervention.**
3. **Each notebook must declare its dependencies in cell 0.**
4. **Clear all outputs before committing.** The container regenerates them.

---

## 9. Testing

1. **Unit tests for every reader** (columns, units, sign convention, sigma positivity).
2. **Unit tests for DOLS** (round-trip synthetic data, unit consistency).
3. **Satellite-era fit validation** (Hamlington et al. comparison within tolerance).
4. **Integration tests** (raw data → figures, key numerical outputs within tolerance).
5. **Regression tests for known bugs** (IMBIE sigma sign, Gt vs mm convention, baseline alignment).

---

## 10. Docker and Execution

```makefile
data:      python scripts/process_raw_data.py
analysis:  python scripts/run_analysis.py
figures:   python scripts/generate_figures.py
paper:     pdflatex main.tex
all:       data analysis figures paper
test:      pytest tests/
```

---

## 11. Migration Checklist

- [x] Create `src/slr_forecast/` package with `__init__.py` *(done — package skeleton with readers/, analysis/, plotting/ subpackages)*
- [ ] Move remaining `.py` modules from `notebooks/` into package submodules (analysis, plotting)
- [x] Create `config.py` with all constants *(done — BASELINE_YEAR, BASELINE_WINDOW, N_SAMPLES, SEEDS, paths, SSPS)*
- [x] Create `units.py` with all conversion factors *(done — M_TO_MM, GT_TO_M_SLE, tag_units/sign_convention/baseline, assertion guards)*
- [x] Create `pyproject.toml` for editable install *(done — `pip install -e ".[dev]"` works)*
- [x] Write `tests/test_units.py` *(done — 18 tests passing)*
- [x] Refactor all readers to return SLR convention with positive sigma *(done — 28 reader functions in `src/slr_forecast/readers/`; IMBIE Gt sign flip, IMBIE mm sigma fix, GlaMBIE sign flip all applied at ingestion)*
- [x] Add `df.attrs` metadata enforcement *(done — all readers tag sign_convention, units; helpers in `units.py`)*
- [x] Add assertion guards to reader functions *(done — `assert_sigma_positive`, `assert_units_meters` called at end of readers)*
- [ ] Add assertion guards to analysis functions
- [ ] Write `DATA_SOURCES.md`
- [ ] Audit `data/raw/` for unused files
- [ ] Pin all dependencies in `requirements.txt` / `pyproject.toml`
- [x] Pin random seeds in `config.py` *(done — SEEDS dict with named keys)*
- [ ] Write Makefile
- [ ] Write integration tests
- [x] Write sign-convention regression tests *(done — `tests/test_readers.py`, 14 tests for IMBIE/GlaMBIE sign, sigma, and units)*
- [ ] Write satellite-era fit validation test
- [ ] Convert notebooks to narrative-only
- [ ] Update Dockerfile
- [ ] Test full reproduction in clean Docker environment
- [ ] Verify all figures match reference outputs
- [ ] Archive `notebooks/slr_data_readers.py` *(later — still in active use by notebooks; new package readers are independent copies with fixes applied)*
- [ ] **Complete component-level analysis** (critical path for paper)
- [ ] **Assemble multi-panel publication figures**
- [ ] **Run all three steps end-to-end and produce comparison figure**

### Migration Notes
- **Langsmith pytest plugin conflict**: The environment has a broken `langsmith` install (missing `zstandard`). Use `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest` to run tests cleanly.
- **Reader migration strategy**: New readers in `src/slr_forecast/readers/` are independent copies of the parsing logic from `notebooks/slr_data_readers.py`, with sign-convention fixes (§1), unit constants from `units.py` (§3), and metadata tagging applied at the point of ingestion. The original `notebooks/slr_data_readers.py` is untouched and still used by notebooks. When notebooks are converted to import from the package, the old file can be archived.
- **Cross-platform paths**: All file and folder paths should use `pathlib.Path` (or `os.path.join`) instead of hardcoded `/` separators. Audit notebooks and `.py` modules for string-concatenated paths (e.g., `DATA_RAW / 'ice_sheets'` is fine; `'data/raw/ice_sheets'` is not). This is required for reproducibility on Windows.
- **Next step**: Migrate analysis modules (`slr_analysis.py`, `bayesian_dols.py`, `slr_projections.py`) into `src/slr_forecast/analysis/`, or proceed with the critical-path component analysis work.
