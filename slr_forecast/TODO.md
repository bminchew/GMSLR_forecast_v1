# SLR Forecasting — TODO

*Consolidated 2026-03-22. Previous plans archived in `archive/done.md`.*

---

## Manuscript (00_ddpi_slrforecast2026.tex)

### Critical — must complete before submission

- [ ] **Write remaining Methods subsections**: Thermosteric, Greenland, Antarctica/Peninsula/EAIS, WAIS uncertainty
  *Effort: 1–2 days. Glaciers subsection done.*

- [ ] **Write Introduction**: SLR overview, current projection failures, forecasting principles, our approach
  *Effort: 1–2 days*

- [ ] **Write Results**: Three-step results, component fits, projections comparison with IPCC
  *Effort: 2–3 days*

- [ ] **Write Discussion and Conclusions**
  *Effort: 1–2 days*

- [ ] **Publication figures**: Journal-formatted versions of existing analysis figures
  *Effort: 2–3 days. See figure inventory in `refactor.md` §0b.*

- [ ] **Side-by-side comparison of all three steps** (naive, aggregate, component) at 2050/2100
  *Effort: 0.5 day. Central figure of the paper.*

- [ ] **Complete component-level variance decomposition**: Which component dominates uncertainty at each horizon
  *Effort: 0.5 day. Run all 6 component notebooks, then aggregate.*

- [ ] **Budget closure figure**: Σcomponents ≈ total GMSL at 2100
  *Effort: 0.5 day. Run per-component notebooks, sum projections.*

- [ ] **Data inventory spreadsheet**: Create an Excel workbook documenting all datasets used in this study. One sheet per component (Thermosteric, Glaciers, Greenland, EAIS, Peninsula, WAIS) plus sheets for GMSL, GMST, and cross-cutting data (IPCC AR6, ISMIP6). Each entry should include: dataset name, source/reference with DOI, temporal coverage, spatial coverage, how it is used (calibration/validation/diagnosis), units, file path in repo, and notes on processing. Goal: full replicability of the data pipeline.
  *Effort: 1 day*

### High — strengthens paper

- [ ] **WAIS scenario weight sensitivity**: Sweep A4 weights ±0.1; tornado diagram of σ_ice at 2100
  *Effort: 0.5 day*

- [ ] **WAIS scenario range sensitivity**: Perturb (low, high) bounds ±20%; identify dominant scenario
  *Effort: 0.5 day*

- [ ] **Rheology exponent sensitivity**: Test n = 3.5 and n = 4.5 in A1 correction
  *Effort: 0.25 day*

- [ ] **Comparison table**: σ_ice at 2100 from IPCC (med/low), A1–A4, Bamber (2019), DeConto (2021), Edwards (2021)
  *Effort: 0.5 day*

- [ ] **Populate references.bib**: Add all cited works with DOIs
  *Effort: 0.5 day*

---

## Analysis

### High — needed for paper

- [ ] **Confirm `component_eais.ipynb`**: Run with `RERUN_FITS=True` to validate SMB-based projections (C_T = +60 ± 20 Gt/yr/°C GMST from Frieler/Ligtenberg). Check that cached HDF5 round-trips correctly and projections are physically reasonable (negative SLR under warming).
  *Effort: 0.25 day*

- [ ] **Peninsula SMB+D decomposition (future work)**: Peninsula currently uses a DOLS linear fit (~45 mm at 2100 under SSP2-4.5, ~9% of total GMSL). This is adequate: the ~40 mm difference vs an SMB-only approach is well within WAIS uncertainty (~250 mm 90% CI). A Greenland-style SMB+D decomposition (truncate pre-2005 to exclude Larsen A/B collapse, fit post-collapse D) would improve physical transparency but not meaningfully change totals. Flag for follow-up paper.
  *Effort: 0.5–1 day*

- [ ] **Validate component fits against IMBIE/GlaMBIE**: Cross-check model-implied cryospheric rates vs directly observed
  *Effort: 0.25 day*

### Medium — valuable but not blocking

- [ ] **Dangendorf thermodynamic signal**: Compute as additional robustness check
  *Effort: 0.5 day. Approach: `thermodynamic = steric + barystatic` or generalize `compute_thermodynamic_signal()`.*

- [ ] **Predictable/unpredictable framing synthesis**: Quantify reducible vs irreducible uncertainty fractions at 2050 and 2100; relate to decision-relevant timescales
  *Effort: 1–2 days*

- [ ] **Rate-and-state model with NASA rate prior**: Already coded in `bayesian_ratestate.ipynb`, needs execution
  *Effort: 0.25 day*

### Low — future work or follow-up paper

- [ ] **A4 framework extensions**: Calving correction, rapid grounding-line retreat rate amplification, initialisation uncertainty (see `taxonomy.md` TODO block for details)
  *Effort: 2–3 days*

- [ ] **BPS trend constraint**: Informative ΔT₀ prior (σ_log = 0.2 instead of 0.5); assess value in component-sum framework
  *Effort: 0.5 day*

- [ ] **DOLS residual structure**: Connect to internal variability (AMO/PDO); classify volcanic as unpredictable noise
  *Effort: 1 day*

- [ ] **A4 tail behavior**: Compare 99th percentile with published worst-case estimates
  *Effort: 0.25 day*

- [ ] **A4 internal consistency**: Verify A4 convolved with DOLS reproduces IPCC low-confidence total GMSL range (approximately)
  *Effort: 0.25 day*

---

## Code Quality (from refactor.md)

### Medium — do before final submission

- [ ] **Sign convention enforcement**: Assert SLR convention (`df.attrs['sign_convention'] = 'slr'`) in all readers; add `assert_sign_consistent()` at function boundaries
  *Effort: 0.5 day*

- [ ] **Baseline tracking**: Define baseline once in `config.py`; implement `rebaseline()` and `assert_baseline_consistent()` in `units.py`
  *Effort: 0.5 day*

- [ ] **Unit assertion at function boundaries**: Implement `assert_units_consistent()` for multi-DataFrame operations
  *Effort: 0.25 day*

- [ ] **Figure formatting config**: `visualization_config.py` with Nature/PNAS sizing (89 mm / 183 mm), 7–8 pt fonts, colorblind-safe palette, PNG + PDF output
  *Effort: 0.5 day*

---

## Completed Work

<details>
<summary>Click to expand</summary>

- Refactored component decomposition into 6 per-component notebooks with standardized structure (2026-03-24)
- Added 3 new plotting functions: `plot_component_projection_twopanel()`, `plot_component_histogram()`, `plot_component_ridge()` (2026-03-24)
- Archived `component_decomposition_refactor.ipynb` and `component_decomp_sensitivities.ipynb` (2026-03-24)
- Component decomposition: all stages (1, 2a, 2b, 3, 4) implemented (Mar 2026)
- All data readers written: IMBIE ×5, GlaMBIE ×2, IPCC FACTS, GRACE TWS, ENSO indices (Mar 2026)
- Greenland joint SMB+discharge model with Mankoff/Mouginot data (Mar 2026)
- Surface-to-ocean temperature transfer function (2026-03-22)
- Projection cell rewrite with all components (2026-03-22)
- Antarctic sub-component fits: EAIS, Peninsula, WAIS/A4 (2026-03-22)
- Glaciers subsection written in manuscript (2026-03-22)
- DOLS calibration, WLS+HAC, SAOD test (2026-02-14)
- Multi-dataset robustness: 28 fits (2026-02-16)
- Sliding-window DOLS: multi-bandwidth (2026-02-17)
- Bayesian static + DLM + hierarchical (2026-02-16)
- Factorial transform bug fix + 19-test suite (2026-02-14)
- Repository cleanup + .gitignore (2026-02-16)

</details>
