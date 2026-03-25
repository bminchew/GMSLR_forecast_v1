# Next Steps: Component Decomposition

## 1. Surface-to-ocean temperature transfer function  -- DONE (2026-03-22)
Calibrated linear transfer function T_ocean = alpha * T_surface + beta from EN4/Argo (200-500 m around Greenland) vs Greenland surface T. Implemented as `fit_ocean_transfer_function()` and `project_ocean_temperature()` in `notebooks/component_analysis.py`. Propagates parameter + residual uncertainty in MC projections.

## 2. Projection cell rewrite  -- DONE (2026-03-22)
Rewrote cell 25 of `component_decomposition_refactor.ipynb`:
- Greenland: uses `fit_bayesian_greenland_joint` (separate SMB + discharge posteriors) with ocean T from the transfer function via `project_greenland_joint_ensemble()` in `notebooks/component_projections.py`
- Thermosteric, Glaciers, EAIS, Peninsula: taper-based linear fits (f_max=1) via `project_component_level_ensemble()`
- WAIS: A4 scenario framework (unchanged)
- TWS: IPCC AR6 (unchanged)
- Budget closure (cell 29) uses `taper_results['Greenland']['linear']` on the Frederikse grid (GMST-based, for budget validation only)

## 3. Pass 2 re-fit with budget rate+acceleration constraint
Add the satellite-era budget rate (0.85 ± 0.58 mm/yr) and acceleration as a constraint in `fit_bayesian_greenland_joint`. Infrastructure exists in `bayesian_dols.py` (`compute_budget_target`, `BudgetTarget`), but budget kwargs are not yet wired into the joint log-prob function.

Implementation:
- Add `budget_rate_accel` kwarg to `_greenland_joint_log_prob`
- Compute model total rate = d(H_smb + H_dyn)/dt at end of record
- Add bivariate Gaussian penalty using `_rate_accel_prior_logp`
- Do NOT use level constraint (dominated by ~16 mm Frederikse budget residual from TWS/deep ocean)

**Depends on:** Joint model Pass 1 results, NASA altimetry GMSL

## 4. Antarctic sub-component fits  -- DONE (2026-03-22)
All three sub-components implemented in `component_decomposition_refactor.ipynb`:
- **EAIS**: Linear fit (a=0, b=0 forced; trend only) in cell 17 taper results, projected with `fixed_coefficients={'a': 0.0, 'b': 0.0}` in cell 25. Data do not support temperature dependence over the 1992-2020 IMBIE record.
- **Peninsula**: Linear fit (GMST) in cell 17 taper results, projected in cell 25. Small but non-zero b.
- **WAIS**: A4 physics-informed scenario mixture (`sample_a4_wais()` in `component_projections.py`), SSP-independent. Projected in cell 25.
