# SLR Forecasting — TODO

*Last updated: 2026-02-16*

---

## Active Priorities

### 1. Component-Wise DOLS — Per-Component Temperature Sensitivity
**Priority: HIGH | Effort: ~1 day | Dependencies: None (data loaded)**

Fit DOLS to each SLR component independently (steric, glaciers, Greenland, Antarctica) to determine which components drive the quadratic acceleration. Replicate on IPCC projected components to compare model vs observed per-component sensitivities.

- [ ] Fit DOLS to Frederikse component columns: `steric`, `glaciers`, `greenland`, `antarctica` each regressed against GMST
- [ ] Compare component sensitivities: Which show significant quadratic? Expected: thermosteric and glaciers continuous; Antarctica poorly fit
- [ ] Replicate with IPCC projected components: `oceandynamics`, `glaciers`, `GIS`, `AIS`
- [ ] Cross-check: Do per-component α₀ values sum to total GMSL α₀?
- [ ] Variance attribution: Which component contributes most to total dα/dT?
- [ ] Horwath budget validation (1993–2016, limited statistical power)

**Data**: Frederikse has `steric`, `glaciers`, `greenland`, `antarctica` (1900–2018). IPCC has `oceandynamics`, `glaciers`, `GIS`, `AIS` (decadal, 2020–2100). Horwath has `steric_dieng`, `glaciers`, `greenland`, `antarctica_altimetry`/`antarctica_grace` (monthly, 1993–2016).

---

### 2. Stress-Test WAIS Uncertainty Approaches
**Priority: HIGH | Effort: 1–2 days | Dependencies: None (code exists)**

Reviewer-ready sensitivity analysis of the A1–A4 framework.

- [ ] **Scenario weight sensitivity:** Sweep A4 weights ±0.1 per scenario; tornado diagram of σ_ice at 2100
- [ ] **Scenario range sensitivity:** Perturb (low, high) bounds by ±20%; identify which scenario drives most variance
- [ ] **Rheology exponent:** Test n = 3.5 and n = 4.5 in A1 correction
- [ ] **Stochastic amplification magnitude:** A2 noise ±50%
- [ ] **Comparison with published estimates:** Tabulate σ_ice at 2100 from IPCC (med/low), A1–A4, Bamber (2019), DeConto (2021), Edwards (2021)
- [ ] **Tail behavior:** Compare A4 99th percentile with published worst-case estimates
- [ ] **Internal consistency:** Verify A4 convolved with DOLS reproduces IPCC low-confidence total GMSL range (approximately)

---

### 3. Greenland SLR Sensitivity to Regional Warming
**Priority: MEDIUM | Effort: 2–3 days | Dependencies: New data acquisition**

Test whether Greenland ice mass loss is more predictable when regressed against regional (Arctic) temperature rather than GMST.

- [ ] Acquire Greenland regional temperature data (ERA5, Berkeley Earth regional, DMI composites, or MAR/RACMO)
- [ ] Write reader function with `df.attrs` metadata
- [ ] Fit GrIS mass loss (IMBIE) vs regional T and compare with GMST fit
- [ ] Assess implications for variance decomposition: Does regional T shrink deep uncertainty fraction?
- [ ] If successful, construct component-level model: `GrIS_rate = α_GrIS × T_regional + trend`

---

### 4. Predictable/Unpredictable Framing
**Priority: HIGH (but comes last) | Effort: 1–2 days | Dependencies: Items 1–3**

Synthesize all results into the paper's central argument.

- [ ] Quantify reducible vs irreducible uncertainty fractions at 2050 and 2100
- [ ] Relate constrained/scenario/ice partition to decision-relevant timescales
- [ ] Connect DOLS residuals to internal variability (AMO/PDO structure?)
- [ ] Classify volcanic contribution as noise (unpredictable future eruptions)
- [ ] Incorporate IPCC cross-validation results (emergent sensitivity test)

---

### 5. Dangendorf Thermodynamic Signal
**Priority: LOW | Effort: ~0.5 day | Dependencies: None (data loaded)**

Compute Dangendorf-based thermodynamic signal as an additional robustness check.

- [ ] Approach: `thermodynamic = steric + barystatic` or generalize `compute_thermodynamic_signal()` to detect available columns
- [ ] Add to `read_process_datafiles.ipynb`
- [ ] Cross-validate vs Frederikse thermodynamic over 1900–2018 overlap

---

### 6. Publication Figures
**Priority: REQUIRED | Effort: 2–3 days | Dependencies: Items 1–4**

Journal-formatted figures for manuscript submission.

#### Planned figure list
- [ ] **Fig 1: Observational context** — GMSL records + GMST, thermodynamic overlay
- [ ] **Fig 2: DOLS calibration + hindcast** — (a) Rate-vs-T phase plot, (b) cross-validation skill
- [ ] **Fig 3: Variance decomposition** — Stacked area: constrained/scenario/ice, with and without physics-informed WAIS
- [ ] **Fig 4: GMSLR projections** — Projection envelopes (IPCC / DOLS / DOLS+WAIS), 2100 histograms
- [ ] **Fig 5: Greenland regional** — Mass loss vs regional T and vs GMST (depends on item 3)
- [ ] **Fig S1: WAIS uncertainty approaches** — A1–A4 comparison
- [ ] **Fig S2: Coefficient stability**
- [ ] **Fig S3: IPCC component decomposition**
- [ ] **Fig S4: DOLS on IPCC projections** — Rate-vs-T with IPCC SSP trajectories
- [ ] **Fig S5: Multi-dataset robustness** — Heatmap + forest plot
- [ ] **Fig S6: Sliding-window DOLS** — Multi-bandwidth coefficient evolution
- [ ] **Fig S7: Bayesian analysis** — Static posterior, DLM coefficients, hierarchical forest
- [ ] **Fig S8: WAIS stress tests** — Tornado/sensitivity diagrams

#### Formatting
- [ ] `visualization_config.py` with consistent styling: Nature single-column (89 mm), double-column (183 mm), 7–8 pt fonts, colorblind-safe palette
- [ ] All figures save as PNG (150 dpi) + PDF (vector)
- [ ] Panel labels (a, b, c) applied consistently

---

## Completed Work

<details>
<summary>Click to expand completed items</summary>

### 0. Update DOLS to WLS — DONE (2026-02-14)
Unified `calibrate_dols()` with optional `gmsl_sigma` for WLS + HAC standard errors. Unified `DOLSResult` dataclass.

### 1. Volcanic SAOD in DOLS — DONE (2026-02-14)
`read_glossac()`, `read_mauna_loa_transmission()`, SAOD integration in DOLS. Finding: SAOD NOT significant for static annual DOLS (γ_saod t=0.26); does not alias into α.

### 2. DOLS on IPCC Projections — DONE (2026-02-14)
Emergent sensitivity test. IPCC thermodynamic component is linear (α₀ ≈ 2 mm/yr/°C), not quadratic. Factor of 2 below observed DOLS sensitivity.

### 2b. Multi-Dataset Robustness — DONE (2026-02-16)
7 GMSL × 4 GMST = 28 fits. Thermodynamic ensemble (8 pairs, excl. Horwath + Dangendorf sterodynamic): dα/dT = 2.85 ± 0.38 mm/yr/°C². Script: `dols_robustness.py`.

### 2c. Sliding-Window DOLS — DONE (2026-02-17)
Multi-bandwidth (h=30,40,50,60 yr), cross-dataset. Key findings: α₀–dα/dT tradeoff confirmed; MLO SAOD significant in ~50% of windows; dα/dT increases toward present. Script: `dols_sliding_window.py`.

### 2d. Bayesian Complement — DONE (2026-02-16)
Three models: (1) Bayesian static DOLS (emcee), (2) DLM with Kalman filter + RTS smoother, (3) Hierarchical multi-dataset. Design matrix bit-for-bit equivalent to `calibrate_dols()`. Key findings: Bayesian static dα/dT ≈ 5.74 [2.86, 8.70] vs freq 5.88 ± 1.39; DLM Q ≈ 0 (coefficients constant); Hierarchical pop. dα/dT ≈ 2.57 ± 0.32 (5 datasets). Scripts: `bayesian_dols.py`, `bayesian_analysis.py`. 7 figures.

### Factorial transform bug fix (2026-02-14)
Found and fixed critical bug: regressors divided by k! AND coefficients multiplied by k!, causing (k!)² inflation. dalpha_dT was 4× inflated. Created 19-test verification suite (`test_dols.py`).

### Repository cleanup (2026-02-16)
Removed LaTeX build artifacts, PDFs, HDF5 data, duplicate notebook figures, and `.claude/` from git tracking. Updated `.gitignore`.

</details>
