# Component notebook audit: code bug report (2026-09-24)

Scope: the 8 `notebooks/component_*.ipynb` notebooks: ocean, glacier, greenland, eais, apeninsula, wais, summation and forecast. Each notebook had two independent auditors. One checked the markdown cells and the other checked the code comments and docstrings, and both logged any code bugs they found. No code was changed. Markdown and comment edits were applied separately (Appendix C).

"Confirmed" means the auditor traced the behavior in code, stored outputs or HDF5. "Suspected" means the mechanism is plausible but the auditor did not demonstrate it. Items marked (verified) were rechecked independently while compiling this report.

## 1. Cross-cutting bugs (affect more than one notebook)

### 1.1 ISMIP6 experiment → SSP mapping is wrong: HIGH, confirmed (verified)
- `notebooks/component_projections.py:1368` `ISMIP6_EXP_SSP` maps exp05/06 → "CMIP6-median", exp09 → SSP1-1.9, exp10 → SSP1-2.6, exp11 → SSP2-4.5, exp12 → SSP3-7.0 and exp13 → SSP5-8.5. The comment cites Seroussi et al. (2020) Table 1.
- Seroussi et al. (2020) Table 1 (local PDF, `data/raw/ice_sheets/ismip6/`) says otherwise. All of these are CMIP5 **RCP8.5** runs:
  - exp05 is NorESM1-M, standard melt.
  - exp06 is MIROC-ESM-CHEM.
  - exp09 and exp10 are NorESM1-M with high and low melt sensitivity.
  - exp11 and exp12 are CCSM4 with ice-shelf collapse.
  - exp13 is NorESM1-M PIGL.
  - The only RCP2.6 run in the core set is exp07, and it is not read.
- `read_ismip6.py` has a third labeling that also disagrees with the table.
- Affected: the EAIS twopanel, histogram, ridge and ismip6 figures and the cell-17 table, and the Peninsula cells 13–17. The Peninsula SSP2-4.5 comparison (cells 14–15) silently uses exp05/06, which are RCP8.5. The "+18 mm SSP3-7.0" is the CCSM4 ice-shelf-collapse run. Check any manuscript or supplement figure built from `ISMIP6_EXP_SSP`.

### 1.2 Projections flat or unforced after ~2099: MEDIUM, confirmed
The SSP temperature series end at 2099.0–2099.5, and each component handles the end differently:
- **Ocean (cell 9):** `np.interp` holds the last value, so 2100–2150 are copies of mid-2099. The SSP5-8.5 median is 494.82 mm at 2100, 2101, 2120 and 2150. The 2100 value is about 4–5 mm (~1%) low.
- **Glacier and Peninsula:** flat from about 2099 to 2150 in `component_results.h5`. For the Peninsula, "2100" is really about 2098.9, roughly 0.1 mm short.
- **Greenland (cell 15):** the delayed ocean T is NaN after about 2104 and is zero-filled. Discharge then falls to r₀ with no ocean forcing, and SMB holds T at its 2099 value. The 2100 value is unaffected.
- **Downstream:** the component_summation 2150 rows and the 2150 `Total_sum` hold these components at or near their 2100 values. The markdown "full 1950–2150 trajectory" (ocean cell 8) is not accurate until this is fixed.

### 1.3 IPCC AR6 baseline conversion: MEDIUM, confirmed (verified in summation)
- **component_summation cell 18:** `ipcc_offset_mm = (GMSL(2000) − GMSL(~2004.5))` = −11.1 mm is *added* to the AR6 values. Converting from the 1995–2014 reference to the 2000 baseline requires adding GMSL(ref) − GMSL(2000), which is +11.1 mm. As a result the IPCC curves in cells 20 and 28 sit about 22 mm too low. For example, the SSP2-4.5 median at 2020 is plotted at 39 mm when it should be 61 mm; NASA gives 67 mm.
- **component_summation cells 21 and 29:** the IPCC values get no baseline conversion, so the 2100 comparison table is about 11 mm low. The excluding-Antarctica sum in cell 21 needs a non-Antarctic offset.
- **component_forecast:** the IPCC baseline is handled three different ways.
  - Cells 16 and 21 apply no offset, although their axis and table are labeled "rel. to 2000".
  - Cell 20 uses +10.7 mm, the NASA offset from the precompute script.
  - Cell 12 uses 12.1 mm, the Frederikse offset.
  - For example, the SSP2-4.5 2100 median is 556 mm in the cell-21 table and about 567 mm in the cell-20 ridge input.
- **Known from memory ([IPCC per-component rebase bug]):** TWS from `ipcc_distributions.h5` carries the full +10.7 mm total-GMSL offset. This shifts the forecast `Total_sum` / `Total_stable` and the "Comp. sum" column. The blended forecast and the variance decomposition are unaffected.

### 1.4 IMBIE annualization: MEDIUM/LOW, confirmed
- **EAIS and Peninsula, cell 3/6:** the code takes the January value with `.first()`, on the assumption that rate and σ are constant within each year. They are not constant in 6 of 45 years (1992, 2002, 2003, 2015, 2019, 2022). The rate differs by up to 0.225 mm/yr (2022). The 1992 σ is 0.115 mm/yr, while nine months of that year report 0.241. The σ feeds the likelihood covariance.
- **`annualize_imbie` (WAIS, EAIS):** the December cumulative value is labeled yr+0.5, which places it about 0.46 yr early. This also means the rebase zero sits at the end of 2000.

### 1.5 BIC model comparison is not a like-for-like likelihood comparison: MEDIUM, confirmed/suspected
- **Peninsula and EAIS (cell 6):** the BIC uses χ² at the posterior mean against the correlated covariance. With σ_extra fitted, χ²/dof ≈ 1 by construction, and the σ_extra-dependent log-determinant is dropped. ΔBIC therefore reduces to roughly the ln(n) penalty: −3.6 is reported for the Peninsula against a penalty of −3.8, so "linear preferred" is close to predetermined. The Peninsula also converts χ²/dof back to χ² with a dof that is one smaller than the fitter's (n−n_phys−2 instead of −1).
- **Glacier cell 19 (taper sweep):** for `level_correlated`, the sweep divides cumulative-level residuals (m) by per-year rate σ (m/yr) and sums them as if independent. That is not the correlated χ² used in cell 6. The sweep gives ΔBIC −5.3 where cell 6 gives −3.1.
- **Greenland (δ selection):** each δ candidate is scored on a different number of points (45, 44, 43, 42, 41) because EN4 starts in 1970.5. The log σ² term is worth about −16.6 per point, so the BIC weights (0.995 on δ = 5) are not computed on a common dataset.

### 1.6 Independent per-year draws treated as trajectories: LOW/MEDIUM, confirmed/suspected
- **TWS in forecast cell 5 and summation cell 5:** samples are drawn independently at each IPCC year, with an adjacent-year correlation of about −0.03. The forecast differentiates these samples, which inflates the TWS rate variance. TWS is about 0.1% of the 2100 variance.
- **Ocean cell 9 (suspected):** the SSP temperature uncertainty is added as independent noise each year. The ODE averages it out, so the temperature-driven projection spread is likely understated.

## 2. Notebook-specific bugs (medium severity and notable low severity)

### component_ocean
- **Medium, confirmed:** `save_ocean_hybrid` (cell 9) deletes the whole `ocean` group, including the `ocean/posteriors` that cell 5 cached. A top-to-bottom rerun then hits a KeyError in cell 6's reload path, and cell 5's `_f['ocean']` fails on a fresh HDF5.
- **Low, confirmed:**
  - With mid-year time stamps, the nearest point to 2000 is a tie and `argmin` picks 1999.5. The value at 2000 is therefore +0.5 mm, not 0.
  - The "full-depth" observations miss the 700–2000 m change between the baseline and 2005.5, which leaves them about 1.5 mm low after 2005.
  - The rebase windows differ: [1995, 2006] gives 11 values, while column 3 uses [1995, 2005], which gives 10.
  - The "1993–2025" rate row compares a NOAA rate over 2005–2025 with the model over 1993–2025.
  - χ²/n is labeled "reduced χ²".
  - The SSP years in cell 11 are plotted without the +0.5 shift.
  - The corner plot labels the drift coefficient ϵ_u, while everything else calls it c.
  - Several plot and print labels are wrong (for example "Full depth" for 0–2000 m, and "baseline" printed at 2005).
  - Stored outputs for cells 3 and 7 predate the Dangendorf code.

### component_glacier
- **Medium, suspected:** `OBS_WINDOW = (2000, 2023)` is compared against mid-year times, so 2023.5 is dropped. That is the largest loss year in the record (548 Gt, 1.51 mm/yr), and it leaves n = 23. Temperature data exist through 2024.
- **Medium, confirmed:** the taper-sweep BIC is inconsistent (§1.5).
- **Low, confirmed:**
  - The naive `level` branch computes the centered-difference rate σ by adding two cumulative variances instead of taking their difference.
  - Error bars are 90% in panel (a) and 2σ in panel (b) and cell 17, yet all are labeled "90% CI".
  - Cell 19 is not guarded by `if REFIT:`, so it raises a NameError when REFIT=False.
  - The `bayesian_models.py` progress print says `b~HN(2.0)`, but the actual prior is Normal(0.61, 2.0) truncated at b ≥ 0.
  - "P(a>0) = 100%" is guaranteed by the prior.
- **Low, suspected:**
  - The rebase subtracts `cumsum[0]`, which already includes the year-2000 increment, so observations sit about 0.3 mm / 0.5 yr off. H0 absorbs this.
  - The cell 12 seed comment contradicts itself: `seed=400+i` draws different posterior samples for each SSP.

### component_greenland
- **Medium, suspected:** the cross-correlation diagnostic correlates the discharge *rate* with the ocean-T *rate*, but the model links rate to T *level*. The diagnostic's peak is at 11 yr, outside the δ candidate set of 4–8.
- **Medium, suspected:** the BIC over δ is computed on unequal n (§1.5).
- **Low, confirmed:**
  - The ocean T gap from 2022 to 2024 (EN4 ends in 2021) is filled by linear interpolation, although observed GMST exists for those years.
  - Cell 10 panel (c) feeds GMST to the ocean transfer function without AA. The same line is in `results_figures` cell 29.
  - The GRACE−D curve is cumulated from a rate anomaly, while the Mouginot and Mankoff curves use full rates, giving about 1 mm/yr of relative slope.
  - The best-δ centering constants are applied to every δ.
  - Baseline masks written as `<= 2005` actually cover 1995–2004.
  - Figure 10 mixes different error-bar conventions.
  - The IMBIE-3 legend says "(validation)", while the comment says it is not used for validation.
- **Low, suspected:**
  - The AA ramp assumes a present-day AA of 3.0, but `smb_projections.py` says C_T was converted with AA ≈ 2.0.
  - α and β are drawn independently, and the 0.196 °C residual is not propagated.
  - CMIP6 annual means are stamped at YYYY.0.
- **Scientific note (not a code error):** the implemented SMB_0 = 380, C_T = −300 and C_T² = −50 give an SMB zero crossing at about 1.07 °C GMST above 1995–2005. The markdown previously quoted about 2.7 °C (Noël 2021) as a "constraint". No such constraint exists in the code.

### component_eais
- **High, suspected:** the "sign-flip check" numbers in markdown cells 0, 5 and 18 cannot be reproduced under the current `level_correlated` path. They are: IMBIE-3 1992–2020 b = +0.024 to +0.026, R² ≈ 0.60; IMBIE v2021 b = −0.0295; R² 0.7402/0.7382. They likely come from the legacy path, and the conclusion that the sign flip comes from the product rather than the window is unverified until they are rerun.
- **Medium, confirmed:** the IMBIE-3 `.first()` annualization problem (§1.4).
- **Medium, confirmed:** with `symmetric_b=True` the prior printout shows `b~N(0,0.50)`, but the sampled prior has mean −0.166 mm/yr/°C.
- **Low, confirmed:**
  - The cell 3 print labels +0.166 as "(negative = SL fall)".
  - ISMIP6 series start in 2016, so the "2000 baseline" rebase anchors them at 2016. The table's `[-1]` index can land on 2101.
  - `project_smb_ensemble` is called without `baseline_year`. It happens to rebase at exactly 2000 with current data.
  - `check_convergence` is imported but never called.
  - Cell 19 is not guarded by `if REFIT:`.
- **Result change reflected in the markdown:** under the fit that actually ran, b = −0.0148 mm/yr/°C [−0.145, +0.111]. That is mass-*gain*-signed on the full record. Check whether the manuscript or supplement describes EAIS as mass-loss-signed.

### component_apeninsula
- **High, confirmed:** the ISMIP6 mapping (§1.1). SSP2-4.5 is compared against RCP8.5 runs (cells 14–15).
- **Medium, confirmed:** the cell 19 f_max sensitivity refits omit `symmetric_b=True`, so they use the truncated b ≥ 0 prior. The f_max = 1 row gives b = 0.0516, against 0.0442 from the main fit.
- **Medium:** the BIC construction (§1.5).
- **Low, confirmed:**
  - The legend has an "ISMIP6" entry, but the overlay is commented out.
  - The σ_extra corner label says mm; it should be mm/yr.
  - The CMIP offset averages 1995–2005 while Berkeley Earth averages 1995–2004.
  - `.0f` prints 1979.5–2023.5 as "1980–2024".
  - "P(a>0) = 100%" is guaranteed by the prior.
- **Note:** emcee draws from numpy's global RNG, so `seed=` fixes only the initial walkers and reruns are not bit-identical.

### component_wais
- **Medium, confirmed:** the covariance behind `S1_QUADRATIC_COV` is not positive semidefinite. It has 20 of 45 negative eigenvalues (minimum −3.0 mm², maximum 10.3 mm²). `_anchor_covariance` assumes the uncertainty is measured from the 2000 baseline, but `annualize_imbie` passes IMBIE's raw uncertainty, which grows from 0.135 to 0.702 mm. As a result the "robust" acceleration interval is narrower than the naive one (×0.92). This affects `S1_QUADRATIC_COV` and the cubic-fit intervals.
- **Medium, confirmed (rheology leftovers):**
  - Cell 6 prints "S2 post-correction" results (315 mm [107, 1670], factor 1.28) using a correction that is no longer applied.
  - Cells 24 and 31 compare n = 3 against n = 3 but label one side "n ~ N(4.1, 0.4²)". The stored output is +0.0%, and the figure shows the same curve twice.
  - Cell 31 computes `mix_n3_3070_mm` and `mix_n3_0199_mm` and never uses them.
- **Low, confirmed:**
  - `save_wais` exports `rheology_factor` 1.28/0.07 as metadata, although the factor applied is 1.0.
  - The HDF5 description calls S1 a "Linear ramp".
  - The cell 15 median β of 1.93 is diluted by S1 rows, where β = 0.
  - The H0 prior is centered on the 1979.5 observation. The effect is negligible.
  - Prints show 1980–2024 and "Anchor: year=2024".
- **Low, suspected:**
  - The S2 blend gives zero growth between 2023.5 and 2024.
  - The "AIS: LARMIP-2" curve reads the `_TOT_` file.
- **Upstream:** the S1-only and full 2k WAIS arrays are downsampled with independent indices. No current output appears affected.

### component_summation
- **Medium, confirmed:** the IPCC offset sign error and the missing offsets (§1.3).
- **Medium, confirmed:** the post-2100 flat components feed the 2150 rows (§1.2).
- **Low, confirmed:**
  - Hindcast cells 10 and 11 skip the per-sample rebase for `wais_2k` and the Greenland 1900 hindcast. Offsets are up to 1.6 mm per sample, about 0.06 mm at the median.
  - The cell 13 11-yr warming spans use a one-sided 6-yr mean at the end year because of `min_periods=6`. These spans feed numbers quoted in the manuscript.
  - The HDF5 root attribute `baseline_year` is 2005.0; it should be 2000.
  - TWS print strings say "1900–2018", but only 2003–2018 is used.
  - A print string points readers to "conversation notes".
- **Low, suspected:**
  - The cell 18 satellite-era quadratic band is rebased by the median fit's 2000 value, not each sample's own.
  - Independent per-year noise on the cumulative TWS and EAIS records understates their trend σ in the closure table.
  - The cell 24 title says "3°C warming" for SSP2-4.5.

### component_forecast
- **Medium, confirmed:** the IPCC baseline is handled inconsistently across cells 12, 16/20 and 21 (§1.3).
- **Low, confirmed:**
  - In `blend_rate_space` (cell 11), the observed GMSL at `T_ORIGIN` = 2025.34 is placed on the first grid year, 2026.0. Every blended level, the headline JSON and the HDF5 export are about 3 mm low. The "weight at origin" print reports w at 2026.0.
  - The quadratic MC draws use `cov_params`, which leave out the GIA rate uncertainty (0.15 mm/yr) that `rate_accel_cov` includes. This may be intentional.
  - The variances use `np.var` (ddof = 0), but the covariances use `np.cov` (ddof = 1).
  - The cell 12 IPCC comparison print mixes baselines: "62 vs 43.7 = 38% above" corresponds to 60.3 cm.
- **Low, suspected:** the rebase level at 2000 and `H_ORIGIN` each come from a single unsmoothed NASA sample (about ±2 mm).
- **Manuscript check:** a comment cited a "4x" within/across-scenario ratio, but the stored output gives 3.0x (1115 / 370 mm). Check whether the manuscript uses 4x.

## 3. Decisions for the user

**Resolved 2026-09-25:**
1. The deprecated files are no longer cited in the glacier, EAIS and Peninsula notebooks. `results_figures.ipynb` cells 31 and 61 still cite them in comments; they were not edited because another session was editing that notebook.
2. WAIS cells 30 and 31 (the n=3 vs n=3 comparison) were removed, along with the "Rheology exponent sensitivity" heading in cell 20. Cell 24, the scenario-weight PDF/fan, was kept; its 'current' case still duplicates 'n3_1090'.
3. The glacier `OBS_WINDOW` was changed to (2000, 2024) so the calendar-2023 point is included. In a scratch fit, b goes from 0.737 [0.297, 1.201] to 0.862 [0.431, 1.288] mm/yr/°C. c = 0.555 [0.434, 0.676] mm/yr, χ²/dof = 1.064 and ΔBIC = −3.1 are essentially unchanged. The notebook and downstream results have **not** been rerun.

Original questions:

1. **Missing files:** `plan_glacier_ratespace.md` and `handoff_glacier_ratespace.md` are cited "at the repo root" in glacier cells 1, 5 and 6, Peninsula cells 1 and 5, and EAIS cell 6. They are not on disk or in git history, and the memory index also points to them. Two edits that would re-point the Peninsula references to the `bayesian_models.py` Model 4b module comment are **held, not applied**. The options are to restore the files, re-point every reference, or remove the references.
2. **WAIS cells 24, 30 and 31:** the rheology comparison is now n = 3 vs n = 3. Delete or repurpose these cells. The markdown edits only removed the false claims.
3. **Glacier observation window:** decide whether 2023 is meant to be included.

---

## Appendix A: All bug entries by notebook (raw, both auditors)

Source tag: [md] = markdown auditor, [cm] = comments auditor. Entries found by both auditors appear twice; they are merged in the summary above.

### component_ocean.ipynb (20 entries)

- **[medium, confirmed] [cm] cell 5**: Cell 5 caches the posterior draws under ocean/posteriors, but cell 9 calls save_ocean_hybrid(), which does `del f['ocean']` and recreates the group without posteriors. After a top-to-bottom run the cache that cell 6's standalone-reload path depends on is deleted, so cell 6 fails with KeyError after a kernel restart. (The file on disk currently has ocean/posteriors, so the last write came from a later run of cell 5 alone.) Also `_f['ocean']` raises KeyError on a fresh component_results.h5 with no ocean group.
  - Expected vs actual: Expected: posteriors persist through cell 9 (write them after save_ocean_hybrid, as is done for fit_diagnostic). Actual: deleted by cell 9.
  - Excerpt: `with h5py.File('../data/processed/component_results.h5', 'a') as _f: _cg = _f['ocean'] ... _cg.create_group('posteriors')`
- **[medium, confirmed] [md] cell 9**: projections/temp/SSP* in slr_processed_data.h5 run 2015-2099 (decimal_year+0.5 -> 2015.5-2099.5), so the ODE trajectory years_proj ends at 2099.5. np.interp clamps beyond the last point: every saved ocean sample (samples, upper_samples, twolayer_samples) is constant for 2100-2150, and the value stored at 2100 is the 2099.5 value. Verified in component_results.h5: SSP5-8.5 median 489.99 mm (2099), 494.82 mm at 2100, 2101, 2120 and 2150. At the ~9.4 mm/yr SSP5-8.5 end-of-century rate the 2100 median is low by roughly 4-5 mm (~1%). Markdown cell 8 says 'The model is evaluated over the full 1950--2150 trajectory (hindcast + projection)'; that is inaccurate for 2100-2150, but no markdown edit is proposed because the fix belongs in the code (or data) and the wording should follow whichever decision is made.
  - Expected vs actual: expected: model evaluated through 2150 (or at least through 2100.0); actual: evaluated through 2099.5, then held flat to 2150
  - Excerpt: `full_samples[k, :] = np.interp(PROJ_YEARS, years_proj, H_full_k)  (PROJ_YEARS = 1950..2150; SSP GMST ends 2099.5)`
- **[medium, confirmed] [cm] cell 9**: SSP temperature files end at decimal_year 2099.0 (+0.5 -> 2099.5), so years_proj ends at 2099.5. np.interp holds the last value constant beyond it, so the saved 'samples' at PROJ_YEARS 2100 equal the mid-2099 value and every year 2101-2150 is a flat copy of it. The cell header says 'full trajectory 1950-2150'.
  - Expected vs actual: Expected: 2100 value at 2100.0 (or explicitly documented as mid-2099) and no fabricated 2101-2150 values. Actual: 2100 = H(2099.5), about half a year of rise too low (a few mm under SSP5-8.5); 2101-2150 constant.
  - Excerpt: `future_mask = ssp_years > T_annual_years[-1] ... full_samples[k, :] = np.interp(PROJ_YEARS, years_proj, H_full_k)`
- **[medium, suspected] [cm] cell 9**: GMST scenario uncertainty is added as independent year-to-year white noise with the per-year AR6 sigma. The AR6 5-95% GMST range is mostly persistent across years (climate-sensitivity spread), and independent annual noise is largely averaged out by the tau_u/tau_d integration, so the projected thermosteric spread likely understates the temperature-driven uncertainty. A coherent perturbation (one z per draw times sigma(t)) would carry it through.
  - Expected vs actual: Expected: temperature uncertainty correlated across years within a draw. Actual: independent draw each year.
  - Excerpt: `T_k[n_hist:] += rng_proj.normal(0, ssp_T_sigma[future_mask])`
- **[low, confirmed] [cm] cell 11**: The SSP temperature years used for the lower panel lack the +0.5 mid-year shift that cell 9 applies to the same data (ssp_years = decimal_year + 0.5), so the plotted SSP temperatures sit half a year early relative to the Berkeley Earth observations (T_annual_years, mid-year).
  - Expected vs actual: Expected: decimal_year + 0.5. Actual: decimal_year.
  - Excerpt: `t_years = df_t['decimal_year'].values`
- **[low, confirmed] [md] cell 3**: Stored outputs are from before the working-tree addition of the Dangendorf series: cell 3's output has no 'Dangendorf sterodynamic (validation)' line, and cell 7's output prints only the Frederikse withheld-validation lines and not the Dangendorf lines. The committed figure figures/component_ocean_fit_residuals.png is modified in the working tree, so it may not match the stored outputs either. Quoted numbers for Dangendorf could not be checked against outputs.
  - Expected vs actual: expected: outputs reflect current source; actual: outputs predate the Dangendorf code
  - Excerpt: `stored outputs of cells 3 and 7`
- **[low, confirmed] [cm] cell 3**: Inconsistent rebasing windows on mid-year stamps. The 0-700 m record, and the model in cells 5/7 (bl, bl_use), use [1995.0, 2006.0], which gives 11 annual values for 1995-2005. Column 3 (model H_full, NOAA full depth, Frederikse, Dangendorf) uses [1995.0, 2005.0], which gives 10 values for 1995-2004 and leaves out 2005. config.BASELINE_WINDOW = (1995, 2005) is documented as inclusive. Each comparison is internally consistent, so the effect is a small offset between panels. The print label 'rebased to 1995–2006' is also imprecise.
  - Expected vs actual: Expected: one window (config BASELINE_WINDOW, 1995-2005 inclusive). Actual: two different windows.
  - Excerpt: `bl_mask = (noaa_year >= 1995.0) & (noaa_year <= 2006.0)  vs  cell 7: BL_LO, BL_HI = 1995.0, 2005.0`
- **[low, confirmed] [md] cell 5**: (a) Panel title calls the 0-2000 m record 'Full depth', while the notebook elsewhere reserves 'full depth' for 0-2000 m + below-2000 m (cell 7 column 3, cell 9). (b) The titles report level-space R^2, while the same cell's comment states R^2 is not reported because it is uninformative in level space (project rule: report reduced chi-square).
  - Expected vs actual: expected: '0--2000 m' and chi2/n; actual: 'Full depth' and R^2
  - Excerpt: `ax.set_title(f'Full depth (0--2000 m), R$^2$ = {r2_2000:.4f}') and ax.set_title(f'Upper ocean (0--700 m), R$^2$ = ...')`
- **[low, confirmed] [cm] cell 5**: For the '1993-2025' row the NOAA 0-2000 m rate is fit only to 2005.5-2025.5 data (the record starts 2005.5) while the model rate is fit over 1993-2025, so 'Total: model 1.280, NOAA 1.453' compares different windows (NOAA 1.453 is identical in both rows).
  - Expected vs actual: Expected: same window for model and NOAA, or label NOAA as 2005-2025. Actual: mismatched windows under one label.
  - Excerpt: `for label, lo, hi in [('1993-2025', 1993, 2025), ...]: r2000 = np.polyfit(noaa_deep_year[m2000] - lo, ...)`
- **[low, confirmed] [cm] cell 5**: Labeled reduced chi-square but divides by n (71 and 21), not by degrees of freedom (6 parameters fit jointly to both records, plus the free offsets removed by demeaning). Cell 7 labels the same quantity 'chi2/n'. Project memory records a similar dof correction for Greenland.
  - Expected vs actual: Expected: divide by n - p or label chi2/n. Actual: chi2/n labeled reduced chi^2.
  - Excerpt: `chi2n_700 = np.mean((resid_700 / np.sqrt(...))**2); print('Reduced chi^2 (obs sigma + sigma_extra):')`
- **[low, suspected] [cm] cell 5**: Comment claim that could not be reproduced, so no edit is proposed. Recomputed noise share of var(annual differences): 0-700 m 33% (NOAA SE only) or 49% (SE plus 5% structural); 0-2000 m 34% (SE only) or 98% (SE plus structural). A 44% noise share alone would cap R^2 near 0.56, not 0.03. The model's actual rate-space R^2 is 0.04 (0-700 m, posterior medians) and -0.32 (0-2000 m). The '0.983 for a straight line on 0-2000 m' claim is verified (0.9834).
  - Expected vs actual: The 44% figure and the 'capping' logic do not match recomputation; the author should restate it.
  - Excerpt: `# ... and on annual differences the denominator is ~44% NOAA measurement noise, # capping rate-space R^2 near 0.03 for any smoothly forced model.`
- **[low, confirmed] [cm] cell 5**: Panel title calls 0-2000 m 'Full depth'; full depth in this notebook means 0-2000 m plus the below-2000 m rate (cell 7 column 3). The panel also reports R^2, which the cell's own comment says is uninformative.
  - Expected vs actual: Expected '0--2000 m'. Actual 'Full depth (0--2000 m)'.
  - Excerpt: `ax.set_title(f'Full depth (0--2000 m), R$^2$ = {r2_2000:.4f}')`
- **[low, confirmed] [md] cell 6**: Corner-plot label names the linear-drift coefficient epsilon_u, while markdown cell 4, the printed summary, the HDF5 keys (c_posterior) and save metadata (c_median_mm_yr) all call it c. Plot-label/markdown notation mismatch.
  - Expected vs actual: expected: consistent symbol (c or epsilon_u); actual: c in markdown, epsilon_u in figure
  - Excerpt: `'$ϵ_u$ (mm/yr)' in _corner_labels`
- **[low, confirmed] [md] cell 7**: Column 3 is titled 'Full depth (validation)', but its data legend says '(calibration)'. The plotted estimate is built from both calibration records plus the literature below-2000 m rate, so it is not itself a calibration series. Also axfr[1,1].legend() is called with no labeled artists (the residual-band label is commented out), producing the stored 'No artists with labels found' UserWarning.
  - Expected vs actual: expected: consistent label; actual: 'calibration' legend under 'validation' title; empty legend call
  - Excerpt: `label='NOAA full depth (calibration)' if _c == 2`
- **[low, confirmed] [cm] cell 7**: The only label in the residual row (the ±sigma band) is commented out, so this legend call has no artists; the stored output shows 'No artists with labels found to put in legend'. The call has no effect.
  - Expected vs actual: Expected: the band labeled, or the call removed. Actual: an empty legend call plus a warning.
  - Excerpt: `axfr[1, 1].legend(frameon=False, fontsize=7.2, loc='upper right', ...)`
- **[low, confirmed] [md] cell 9**: Print string labels the 2005 median (6.3 mm) as 'baseline', but projections are rebased to BASELINE_YEAR = 2000 (value 0 there).
  - Expected vs actual: expected: label '2005=' or print the 2000 value; actual: 'baseline=' on the 2005 value
  - Excerpt: `print(f'{ssp}: baseline={med_2005:.1f} mm, ...')`
- **[low, confirmed] [cm] cell 9**: years_proj is on mid-year stamps (…1999.5, 2000.5…), so |y-2000| ties at 0.5 and argmin returns 1999.5. Trajectories are zeroed at mid-1999, not 2000.0; after interpolation to integer PROJ_YEARS the value at 2000 is about +0.5 mm rather than 0. Same tie in cell 11 (idx_be_2000 -> 1999.5 for the temperature rebase).
  - Expected vs actual: Expected: zero at BASELINE_YEAR=2000.0 (e.g. interpolate to 2000.0 and subtract). Actual: zero at 1999.5.
  - Excerpt: `ib = np.argmin(np.abs(years_proj - BASELINE_YEAR)); H_full_k -= H_full_k[ib]`
- **[low, confirmed] [cm] cell 9**: The 'full-depth' observational series adds the 700-2000 m increment measured from 2005.5, so it omits the 700-2000 m change between the 1995-2005 rebasing window and 2005.5. The model's full-depth curve (b_d*S_d) does include that change. For 2005.5 onward the observational series is offset low relative to the model and to its own baseline. Using posterior-median parameters, b_d*(S_d(2005.5) - mean S_d(1995-2004)) = 1.5 mm. Affects panel (c)/(f) residuals (_rf) and the saved ocean/observations used in the two-panel figure.
  - Expected vs actual: Expected: full-depth obs and model on a common baseline. Actual: obs lack about 1.5 mm (model estimate) of pre-2005.5 700-2000 m change.
  - Excerpt: `deep_layer_increment = noaa_2000_corrected - (noaa_700_at_deep_years - noaa_700_at_deep_years[0]); obs_H[i] += np.interp(...)  (same construction in cell 7, _of)`
- **[low, confirmed] [cm] cell 9**: Print label 'baseline' reports the 2005 value (6.3 mm); BASELINE_YEAR is 2000. Stale from the 2005 baseline.
  - Expected vs actual: Expected: label '2005=' or print the 2000 value. Actual: 'baseline=6.3 mm'.
  - Excerpt: `med_2005 = proj_dict[ssp]['median'][PROJ_YEARS == 2005][0] * M_TO_MM; print(f'{ssp}: baseline={med_2005:.1f} mm, ...')`
- **[low, confirmed] [cm] cell 9**: Metadata string written to HDF5 says 2006+; the increment starts at 2005.5.
  - Expected vs actual: Expected '2005.5+' (or 2005+). Actual '2006+'.
  - Excerpt: `'obs_description': 'Full-depth estimate: 0-700m (Li 14% corrected) + 700-2000m increment (2006+) + below-2000m rate'`

### component_glacier.ipynb (14 entries)

- **[medium, confirmed] [md] cell 19**: In the taper sweep the 'level_correlated' branch shares the rate-space BIC formula. For fit_bayesian_level_annual_correlated, .residuals are LEVEL residuals (H_obs - H_model_mean, m), but sig_tapered is the per-year RATE sigma (m/yr), and the sum treats points as independent. This is neither the correlated chi-square used for the main ΔBIC in cell 6 (chi2_dof * dof) nor dimensionally consistent. The printed ΔBIC values (-2.9, -3.5, -3.8) are therefore not comparable to the cell-6 ΔBIC (-3.1), even though TAPER_REF=2000 makes the taper a no-op.
  - Expected vs actual: Expected: chi2 = res.chi2_dof * (n - n_phys - 1) (or residuals whitened by the correlated covariance). Actual: sum((level_resid / rate_sigma)^2).
  - Excerpt: `bic_q_t = np.sum((res_q.residuals / sig_tapered)**2) + 3 * np.log(n)  (level_correlated branch)`
- **[medium, confirmed] [cm] cell 19**: For 'level_correlated', res.residuals are LEVEL residuals (m) of the cumulative record, but they are divided by sig_tapered, which is the per-year RATE sigma (m/yr), and summed as if independent. This is neither the correlated chi-square used for the main ΔBIC in cell 6 (chi2_dof * dof) nor a consistent diagonal chi-square. The 'rate' branch of the same line is correct.
  - Expected vs actual: Expected: chi2 = res.chi2_dof * (n - n_phys - 1), as in cell 6. Actual: diagonal sum of level residuals over rate sigmas. In a scratch re-run at f_max=1 (a no-op taper) the two give ΔBIC = -5.3 (diagonal) vs -3.1 (correlated). The stored output (-2.9, -3.5, -3.8) therefore does not measure the same quantity as cell 6's ΔBIC = -3.1. The sign (linear preferred) does not change.
  - Excerpt: `bic_q_t = np.sum((res_q.residuals / sig_tapered)**2) + 3 * np.log(n)  (FIT_SPACE == 'level_correlated' branch)`
- **[medium, suspected] [md] cell 6**: glac_year is the mid-point of each GlaMBIE period (2000.5 ... 2023.5; 24 rows, start_dates 2000.0 to 2023.0 in the CSV). The mask on mid-year times drops the final period (2023.0-2024.0, 548 Gt, the largest annual loss in the record), giving n=23. Berkeley Earth extends through 2024-12, so temperature is available for it. The reader docstring describes the record as 'Annual DataFrame (2000-2023)' (labelled by start year, 24 rows), so 'OBS_WINDOW = (2000, 2023)' may have been intended to include all 24 periods. The markdown '2000-2023' is consistent with what the code fits (span 2000.0-2023.0), so no markdown edit proposed; flagging for the author to confirm whether the exclusion is intentional.
  - Expected vs actual: If intended to use the full record: n=24 through 2024.0. Actual: n=23 through 2023.0.
  - Excerpt: `OBS_WINDOW = (2000, 2023); mask = (glac_year >= 2000) & (glac_year <= 2023)`
- **[medium, suspected] [cm] cell 6**: glac_year is mid-interval (2000.5 ... 2023.5; the CSV has 24 rows, the last covering 2023.0-2024.0). The upper bound 2023 is compared against mid-year times, so the 2023.5 point (calendar year 2023, the largest loss in the record, 548 Gt = 1.51 mm/yr) is dropped and n = 23. The window label, the printed '2000-2023', the markdown ('2000-2023'), and the reader docstring ('Annual DataFrame (2000-2023)') all read as if calendar year 2023 is included.
  - Expected vs actual: Expected (if 2000-2023 means calendar years inclusive): 24 points 2000.5-2023.5. Actual: 23 points 2000.5-2022.5. If the exclusion is intended, the labels should say calendar years 2000-2022.
  - Excerpt: `OBS_WINDOW = (2000, 2023); mask = (glac_year >= OBS_WINDOW[0]) & (glac_year <= OBS_WINDOW[1])`
- **[low, suspected] [cm] cell 12**: The comment contradicts itself. With seed = 400 + i, project_component_level_ensemble resamples a different set of (b, c, H0) draws (rng.choice with replacement) for each SSP, so SSP-to-SSP differences include Monte Carlo resampling noise as well as the temperature difference. If the intent is that 'only the temperature trajectory should differ' (common random numbers), the code should use a single seed. If independent draws are intended, the second clause is wrong. No comment edit proposed because it is not clear which one is intended.
  - Expected vs actual: Code: independent draws per SSP. Comment's second clause: identical draws across SSPs.
  - Excerpt: `# Each SSP uses a distinct seed so that posterior draws are independent across scenarios; only the temperature trajectory should differ.  ... seed=400 + i`
- **[low, confirmed] [cm] cell 19**: These variables are defined only inside `if REFIT:` in cell 6, and cell 19 is not guarded. With REFIT=False and a successful load in cell 4, cell 19 raises NameError. Cells 9 and 10 are guarded; cells 13-15 and 20 work from the loaded projections.
  - Expected vs actual: Expected: guarded by `if REFIT:` or runnable from loaded state. Actual: NameError when REFIT=False.
  - Excerpt: `res_q = fit_bayesian_level_annual_correlated(... sigma_rate_obs=sig_tapered ...) etc. (whole cell uses rate_sig_r, yrs_r, rate_r, T_r, H_r, prior_kw_annual)`
- **[low, suspected] [cm] cell 3**: cumsum(rate)[k] is the level at the END of interval k (2001.0 + k), but it is timestamped at the interval midpoint glac_year[k] = 2000.5 + k. Subtracting glac_cumul[0] (which already contains the full year-2000 increment) zeroes the series at 2001.0 while labeling that zero as 2000.5. Each observed level is therefore plotted about 0.5 yr early and sits about 0.5 x rate (~0.3-0.4 mm) below the value at its timestamp, and the printed 'Cumulative at 2024: 17.8 mm' leaves out the year-2000 increment (0.22 mm). The fit is unaffected because H0 absorbs a constant offset. The obs-vs-projection overlays (cell 13; the projection is rebased at 2000.0 on a monthly grid) carry this small offset.
  - Expected vs actual: Expected: levels at interval ends (start_dates + 1) relative to 2000.0, i.e. cumsum without subtracting the first increment and timestamped at end of year; or midpoints consistently with a half-increment. Actual: roughly a 0.5-yr / ~0.3 mm offset. Small relative to the ~18 mm signal.
  - Excerpt: `glac_cumul = np.cumsum(glac_rate); bl_idx = argmin|glac_year - BASELINE_YEAR|; glac_rebase = glac_cumul - glac_cumul[bl_idx]`
- **[low, confirmed] [md] cell 5**: Markdown cell 5 points readers to plan_glacier_ratespace.md and handoff_glacier_ratespace.md at the repo root. Neither file exists on disk (searched both repo copies and the Dropbox SLR tree) and neither is tracked in git history. The same dangling references appear in code comments in cells 1 and 6. No edit proposed because the fix (restore the files or drop the pointers) is an author decision.
  - Expected vs actual: Expected: referenced files present at repo root. Actual: not found (only a memory note project_glacier_ratespace_handoff.md exists outside the repo).
  - Excerpt: `(see plan_glacier_ratespace.md at the repo root: ...) / see handoff_glacier_ratespace.md for why this was superseded`
- **[low, confirmed] [md] cell 6**: The progress print always labels b's prior as HalfNormal(scale) and ignores prior_b_mean. With prior_b_mean = 0.61 mm/yr/°C passed from cell 6, the prior actually applied in _level_correlated_log_prior is Normal(0.61, 2.0) truncated at b >= 0 (as the cell 5 markdown correctly states). The stored output therefore misreports the prior.
  - Expected vs actual: Expected print: b~N(0.61, 2.0) mm/yr/°C truncated at 0. Actual print: b~HN(2.0 mm/yr/°C).
  - Excerpt: `printed output 'Priors: b~HN(2.0 mm/yr/°C), ...' (from bayesian_models.fit_bayesian_level_annual_correlated, ~L3133)`
- **[low, confirmed] [cm] cell 6**: a has an Exponential prior with a >= 0 enforced in every fitter (symmetric_a=False), so P(a>0) is 100% by construction and says nothing about the data. The printed '100%' could be read as evidence for acceleration.
  - Expected vs actual: Printed diagnostic is always 100% under this prior.
  - Excerpt: `print(f'P(a>0) = {np.mean(result_quad.posterior_samples[:, 0] > 0)*100:.0f}%')  (also cell 19)`
- **[low, confirmed] [cm] cell 6**: The progress print in bayesian_models.py always labels b's prior 'HN(scale)' and ignores prior_b_mean. With prior_b_mean = 0.61 mm/yr/degC the actual prior is Normal(0.61, 2.0) truncated at b >= 0, so the printed prior is wrong. This is a print string in the module, outside this notebook.
  - Expected vs actual: Expected e.g. 'b~N(0.61, 2.0) mm/yr/°C, b>=0'. Actual 'b~HN(2.0 mm/yr/°C)'.
  - Excerpt: `stored output: 'Priors: b~HN(2.0 mm/yr/°C)' (printed by fit_bayesian_level_annual_correlated in notebooks/bayesian_models.py, b_prior_str)`
- **[low, confirmed] [cm] cell 6**: Neither file exists anywhere under the repo, and neither is tracked in git. The comments point readers to derivations they cannot find. No edit proposed: the fix (restore the files or drop the pointers) is the user's call.
  - Expected vs actual: Referenced files missing.
  - Excerpt: `comments referencing plan_glacier_ratespace.md and handoff_glacier_ratespace.md 'at the repo root' (cells 1 and 6; also markdown cell 5)`
- **[low, confirmed] [cm] cell 9**: glac_sigma is the sigma of a cumulative sum of independent increments, so H[i+w] and H[i-w] are correlated and Var(H[i+w] - H[i-w]) = sigma[i+w]^2 - sigma[i-w]^2. Adding the variances overstates the centred-difference rate error bars. This affects only the non-default FIT_SPACE='level' display.
  - Expected vs actual: Expected sqrt(sigma[i+w]^2 - sigma[i-w]^2)/dt. Actual sqrt(sigma[i+w]^2 + sigma[i-w]^2)/dt.
  - Excerpt: `rate_sigma[_i] = np.sqrt(glac_sigma[_i + _w]**2 + glac_sigma[_i - _w]**2) / _dt * M_TO_MM   (naive 'level' branch only)`
- **[low, confirmed] [cm] cell 9**: In the same figure, panel (a) observation error bars are 90% (Z_90 = 1.645 sigma) and panel (b) error bars are 2 sigma (~95%), while both model bands are labeled '90% CI'. The error-bar conventions differ between panels and are not labeled.
  - Expected vs actual: Expected: one convention (e.g. Z_90 in both). Actual: Z_90 in (a), 2 sigma in (b) and in cell 17.
  - Excerpt: `Panel A: yerr=Z_90 * _rate_scat_sig ; Panel B: yerr=2 * sig_r * M_TO_MM (also cell 17 Panel B: yerr=2 * glac_sigma)`

### component_greenland.ipynb (19 entries)

- **[medium, confirmed] [cm] cell 15**: SSP temperature series in slr_processed_data.h5 end at decimal_year 2099.0 (projections/temp/SSP*_*: 2015.0–2099.0), and t_mon = np.arange(min, max, 1/12) stops at 2098.92. The annualized ocean-T series therefore ends at 2098.5, so T_delayed is NaN for PROJ_YEARS > 2098.5+delta (≈2104–2150 for delta=5) and those NaNs are zero-filled. After ~2104 the discharge rate collapses from γ·T_ocean + r0 to r0 alone (T=0 is the 1995–2005 mean), i.e. ocean forcing is dropped. The same cell's SMB branch gets flat T(2098.9) for 2099–2150 via np.interp clamping. The 'projections' group saved to component_results.h5 spans 1950–2150, so any consumer reading post-2100 values gets these. The comment says 'leading NaNs', but there are none (pre-EN4 proxy starts 1850); only trailing NaNs occur. component_summation.ipynb uses MILESTONE_YEARS = [2050, 2100, 2150], so the 2150 Greenland discharge value is affected downstream.
  - Expected vs actual: Expected: ocean forcing continued (or projections truncated/flagged) beyond the SSP end. Actual: trailing zero-fill of the ocean-T anomaly after ~2104; 2100 values themselves unaffected (T_delayed(2100) = T_ocean(2095)).
  - Excerpt: `T_delayed = np.interp(PROJ_YEARS, t_oc_ann_proj + delta_i, T_oc_ann_proj, left=np.nan, right=np.nan); T_delayed[nan_mask] = 0.0  # 'Fill any leading NaNs'`
- **[medium, suspected] [md] cell 7**: The cross-correlation diagnostic correlates the discharge RATE with the ocean-temperature RATE. The fitted model is dH/dt = gamma*T_ocean(t-delta) + r0, which relates the discharge rate to the ocean temperature LEVEL (or, equivalently, d2H/dt2 to dT/dt). The diagnostic therefore does not test the lag structure of the fitted model. Its current output (peak 11 yr, r = 0.225) lies outside the BIC candidate set {4..8} and contradicts the markdown's 6 yr / r = 0.84 (which I could not reproduce from any stored output).
  - Expected vs actual: Expected: correlation between quantities the model links (dH/dt vs T(t-lag), or d2H/dt2 vs dT/dt). Actual: dH/dt vs dT/dt(t-lag); peak lag 11 yr, r = 0.225.
  - Excerpt: `component_analysis.fit_discharge_delay_model: dT_ocean = np.diff(T_ocean_ann)/...; dH_dyn = np.diff(H_dyn)/...; xcorr_r[k] = corrcoef(dH_dyn, dT_shifted)`
- **[medium, suspected] [md] cell 7**: BIC is compared across delay candidates that are fit to different subsets of the Mouginot record (n_valid = 45, 44, 43, 42, 41 for delta = 4..8, because the EN4 record starts in 1970.5). The log-likelihood includes the log(sigma^2) normalization term, which is about -16.6 per point for the early points (sigma_dyn ~0.23-0.25 mm), so each extra point lowers BIC by ~16.6 minus its chi^2 contribution. BIC values (and the resulting weights, e.g. w = 0.995 for delta = 5) are therefore not computed on a common data set.
  - Expected vs actual: Expected: all candidates scored on a common set of years (e.g. the subset valid for delta = 8) so BIC differences reflect fit only. Actual: n differs by one point per candidate; delta = 5 vs delta = 6 differs by 25.4 in BIC, of which ~16.8 minus the 1975.5 point's chi^2 comes from the extra point.
  - Excerpt: `component_analysis.fit_discharge_delay_model: valid = isfinite(T_shifted) per delta; log_lik = -0.5*sum(W_level*resid**2 + log(sig_v**2)); bic = 2*log(n_valid) - 2*log_lik`
- **[low, suspected] [md] cell 0**: scripts/grace_minus_discharge_smb.py uses IMBIE Greenland total mass balance (1992-2020), not GRACE alone. It is GRACE-dominated only after 2002, and IMBIE's reconciled total includes the input-output method, which uses RCM SMB. The 'GRACE − D' and 'RCM-independent' labels are therefore approximate. Not edited because the same label is used consistently in code, plot labels and the script.
  - Expected vs actual: Label: GRACE − D, RCM-independent. Actual: IMBIE total (1992-2020) − Mouginot/Mankoff discharge.
  - Excerpt: `Validated against GRACE − D implied SMB (RCM-independent)  (also cells 4, 5, 13)`
- **[low, confirmed] [cm] cell 10**: The ocean transfer function was fit with Greenland regional T (gr_temp_monthly) as the surface predictor (cell 7), but panel (c) feeds it Berkeley Earth GMST without the AA factor. Cell 15 applies T×AA for the same step. results_figures.ipynb cell 29 has the same construction (`_be_T_oc = _ot_alpha * _be_ann.values + _ot_beta`), so the comment's claim that it replicates fig3 is accurate; the error is in both. This is GMST used where Greenland regional T is required.
  - Expected vs actual: Expected: alpha*(AA*GMST)+beta (as in cell 15). Actual: alpha*GMST+beta. The splice bias-correction removes the level offset, so only the 2022→2023 increment is affected (order 0.04 mm with gamma≈0.37 mm/yr/°C); the numerical effect is small, but the construction is inconsistent.
  - Excerpt: `be_T_oc = ocean_transfer['alpha'] * be_ann.values + ocean_transfer['beta']`
- **[low, confirmed] [cm] cell 10**: The demeaning constants (int_T_mean_cal, t_mean_cal, H_mean_cal) belong to delta_best only (fit_discharge_delay_model returns best['...']), but they are applied to every posterior draw including draws at other deltas. For delta≠delta_best the fitted valid window (and therefore the integral start and means) differ, so the un-demeaned level is offset. In cell 15 the constants cancel under the BASELINE_YEAR rebase, so projections are unaffected. In the stored run, delta=4 carries weight 0.005 and the others ~0, so the effect on the diagnostic bands is negligible.
  - Expected vs actual: Expected: use fit_results[delta_i]['H_mean'/'int_T_mean'/'t_mean']. Actual: best-delta constants for all draws (diagnostics only).
  - Excerpt: `H_pred = (gamma_i * (int_T - result_discharge.int_T_mean_cal) + r0_i * (yrs_v - result_discharge.t_mean_cal) + result_discharge.H_mean_cal)  [also cell 13, and cell 10 panel (c)]`
- **[low, suspected] [cm] cell 10**: diag_years are integer years (1971..2022), whereas the calibration grid (mou_comp['time_dyn']) and T_ocean_years are mid-year (x.5). The diagnostic forward model is therefore evaluated half a year off the calibration times: int_T starts at 1976.0 rather than 1975.5, and r0*(t - t_mean_cal) uses integer t. This also feeds panel (b)/(c) and the saved greenland/discharge_diagnostic group.
  - Expected vs actual: Expected: mid-year grid consistent with calibration. Actual: 0.5-yr offset (level shift of order r0*0.5 ≈ 0.7 mm before detrending).
  - Excerpt: `diag_years = np.arange(np.floor(T_ocean_years[0]) + 1, np.floor(T_ocean_years[-1]) + 2, 1.0)`
- **[low, confirmed] [cm] cell 10**: The same figure shows 2σ observational error bars in panel (b), 1.645σ (90%) error bars in panel (c), and a panel (c) model band that combines the 5–95% discharge percentiles with ±2σ SMB, so the plotted intervals are not a single consistent confidence level.
  - Expected vs actual: Expected: one interval convention (e.g. 90%). Actual: mixed 2σ/1.645σ.
  - Excerpt: `panel (b) errorbars yerr=2*sigma; panel (c) errorbars yerr=1.645*sigma; panel (c) band totx_p5/p95 = D p5/p95 ∓ 2*totx_sig`
- **[low, confirmed] [cm] cell 10**: The adjacent comment states IMBIE-3 is 'not used for calibration or validation ... shown here purely as an additional observational cross-check', but the legend labels it '(validation)'.
  - Expected vs actual: Label and comment disagree on IMBIE-3's role.
  - Excerpt: `label='IMBIE-3 (validation)'  (panels b and c)`
- **[low, confirmed] [cm] cell 13**: Panel (a) overlays the GRACE−D implied SMB cumulated from a baseline-rate anomaly (1995–2005 mean rate removed), while the Mouginot (mou_comp['H_smb'], via _cumulate) and Mankoff (mank_smb_cumul) curves cumulate the full SMB rate. implied_smb_rate_slr is a full rate (about −1.2 mm/yr in 1992–1994 per greenland_implied_smb.csv), so the GRACE−D curve carries a relative slope of roughly +1 mm/yr compared with the other two.
  - Expected vs actual: Expected: all three curves cumulated the same way (full rate, rebased at BASELINE_YEAR). Actual: GRACE−D is an anomaly cumulative, the others are full-rate cumulatives.
  - Excerpt: `gd_rate_anom = gd_rate - gd_rate[bl_mask_gd].mean(); gd_cumul = np.cumsum(gd_rate_anom * dt_gd)`
- **[low, confirmed] [md] cell 15**: EN4 ocean T ends 2021.46, but the SSP-driven ocean T only begins at ssp_start (the last Berkeley Earth month, ~2024.96). Annual ocean T for 2022-2024 is therefore absent and filled by linear interpolation in np.interp between the 2021 EN4 value and the first SSP-derived value, even though observed Berkeley Earth GMST is available for those years.
  - Expected vs actual: Expected: 2022-2024 ocean T from observed GMST x AA through the transfer function (as done for pre-EN4 years). Actual: linear interpolation across a ~3-yr gap.
  - Excerpt: `hist_mask_oc = time_ocean_monthly < ssp_start; ... t_oc_full = concatenate([pre_en4_t, oc_hist_t, t_mon[ssp_mask]])`
- **[low, confirmed] [cm] cell 15**: EN4 ends 2021.46 but the GMST×AA transfer-function branch starts only at the last Berkeley Earth month (2024.96, df_berkeley_h ends 2024-12). Observed GMST for mid-2021 through Nov 2024 is not used, so after annualization ocean T for 2022–2023 is linearly interpolated between the 2021 EN4 (Jan–Jun only) mean and the single Dec-2024 transfer value.
  - Expected vs actual: Expected: the transfer-function branch fills from the EN4 end onward. Actual: a ~3.5-yr gap bridged by linear interpolation. Small effect on 2100 values.
  - Excerpt: `ssp_start = t_mon[searchsorted(t_mon, temp_time_monthly[-1])]; hist_mask_oc = time_ocean_monthly < ssp_start; t_oc_full = concat([pre_en4_t, oc_hist_t, t_mon[ssp_mask]])`
- **[low, suspected] [cm] cell 15**: The AA ramp assumes the literature C_T (GMST frame) implies present-day AA = 3.0, but smb_projections.py documents the GREENLAND_SMB central value as 'converted to GMST via AA~2.0'. If the embedded AA is 2.0, the pre-1960 scale should be 1/2 rather than 1/3, which changes the 20th-century SMB hindcast saved under hindcast_gmst_1900. The two sources disagree, and I could not determine from the code which is intended.
  - Expected vs actual: Notebook: AA_ref = 3.0. smb_projections.py comment: AA ~2.0.
  - Excerpt: `C_T/C_T2 are calibrated at the present-day AA (~3.0, ...); _aa_scale returns aa_t / AA with AA=3.0`
- **[low, suspected] [cm] cell 15**: The CMIP6 Historical/SSP annual means are stamped at YYYY.0 (e.g. 2015.0), not mid-year, so interpolating them onto a monthly grid and splicing with mid-month Berkeley Earth places the annual means half a year early.
  - Expected vs actual: Expected: annual means at YYYY.5. Actual: YYYY.0 (0.5-yr timing shift in the projected temperature).
  - Excerpt: `combined = pd.concat([hist_part, df_ssp]); t_mon = np.arange(combined['decimal_year'].min(), ...); np.interp(t_mon, combined['decimal_year'], ...)`
- **[low, suspected] [cm] cell 15**: alpha and beta are drawn independently, which ignores the OLS parameter covariance, and the transfer-function residual scatter (residual_std = 0.196 °C, R² = 0.379) is not propagated. The ocean-T projection uncertainty may therefore be understated. Stated as a completeness point, not a sign or unit error.
  - Expected vs actual: Expected: joint (alpha, beta) draw from the OLS covariance (and possibly residual noise). Actual: independent marginals, no residual term.
  - Excerpt: `alpha_draws = rng_dyn.normal(alpha, alpha_se); beta_draws = rng_dyn.normal(beta, beta_se)`
- **[low, confirmed] [md] cell 3**: The two temperature baselines differ: Greenland regional T uses Jan 1995-Dec 2005 (11 full years), while EN4 ocean T uses monthly times 1995.04-2004.96 (10 full years, 2005 excluded because mid-month decimal times exceed 2005.0). Cell 15 uses <= 2005 for GMST likewise. A constant offset is absorbed by beta and by rebasing, so the effect is small.
  - Expected vs actual: Expected: one baseline window definition. Actual: 1995-2005 inclusive for Greenland T; 1995-2004 for EN4 and GMST.
  - Excerpt: `bl_mask = (gr_time_raw >= 1995.0) & (gr_time_raw < 2006.0)  vs  bl_oc = (time_ocean_monthly >= 1995) & (time_ocean_monthly <= 2005)`
- **[low, confirmed] [cm] cell 3**: With mid-month decimal years, the ocean-T baseline covers Jan 1995–Dec 2004 (10 yr), while Greenland T uses Jan 1995–Dec 2005 (11 yr). df_berkeley_h is also zero-mean over Jan 1995–Dec 2005 (verified), and read_en4_regional defaults to reference_period (1995, 2006). The same `<= 2005` pattern appears in cell 15 (temp_bl for offset_gmst; mean −0.017 °C instead of 0) and in cell 13 (GRACE−D, annual, where it is inclusive).
  - Expected vs actual: Expected: a single 1995–2005 inclusive window. Actual: the ocean T and temp_bl windows drop 2005 (offsets of order 0.01–0.02 °C).
  - Excerpt: `bl_oc = (time_ocean_monthly >= 1995) & (time_ocean_monthly <= 2005)  vs  bl_mask = (gr_time_raw >= 1995.0) & (gr_time_raw < 2006.0)`
- **[low, confirmed] [md] cell 4**: Not a code bug; flagged for scientific awareness. The implemented GREENLAND_SMB values (SMB_0 = 380 Gt/yr, C_T = -300, C_T2 = -50 Gt/yr/°C, °C^2) put the zero crossing of SMB at ~1.07 °C GMST above 1995-2005 (~1.6 °C above 1951-1980), not ~2.7 °C. The markdown edit reports the implied value; whether the literature constraint should still hold is a modeling decision for the user (the code comment says C_T = -300 is 'budget-constrained').
  - Expected vs actual: Markdown: ~2.7 °C. Implied by code: 1.07 °C above 1995-2005.
  - Excerpt: `- Constraint: SMB → 0 at ~2.7°C GMST (Noël et al. 2021)`
- **[low, suspected] [md] cell 6**: Unverifiable claim: no bottom-up physical derivation of gamma exists in the notebook, notebooks/*.py, scripts/, or src/ (the only 'bottom-up' hits are this notebook and an archived sensitivity_reconciliation notebook, which does not derive gamma). Left unedited; the user should either cite the derivation or remove the clause.
  - Expected vs actual: Expected: a derivation or value to compare against. Actual: none found.
  - Excerpt: `The delay model produces $\gamma$ estimates ... consistent with the bottom-up physical derivation.`

### component_eais.ipynb (14 entries)

- **[high, confirmed] [cm] cell 13**: The ISMIP6 Antarctica experiment-to-scenario mapping is wrong. Per Seroussi et al. (2020) Table (local PDF data/raw/ice_sheets/ismip6/Seroussi et al. - 2020 ...pdf), exp05-exp13 are CMIP5 RCP experiments: exp05 NorESM1-M RCP8.5, exp06 MIROC-ESM-CHEM RCP8.5, exp07 NorESM1-M RCP2.6, exp08 CCSM4 RCP8.5, exp09 NorESM1-M RCP8.5 high melt, exp10 NorESM1-M RCP8.5 low melt, exp11 CCSM4 RCP8.5 open + shelf collapse, exp12 CCSM4 RCP8.5 standard + shelf collapse, exp13 NorESM1-M RCP8.5 PIGL. None is an SSP and exp10/exp12/exp13 are all RCP8.5.
  - Expected vs actual: Expected: ISMIP6 curves labelled by their actual forcing (all RCP8.5 except exp07 RCP2.6 / Tier-2 expA4, expA8). Actual: exp10 plotted as 'SSP1-2.6', exp12 as 'SSP3-7.0', exp13 as 'SSP5-8.5', exp05/06 as 'CMIP6 median'. Affects figures component_eais_twopanel.png, component_eais_histogram_2100.png, component_eais_ridge.png, component_eais_ismip6.png and the cell-17 comparison table (e.g. 'SSP3-7.0 ISMIP6 -52 mm' is a CCSM4 RCP8.5 shelf-collapse run). Same mapping is shared by any other notebook using ISMIP6_EXP_SSP (WAIS/Peninsula).
  - Excerpt: `EXP_GROUPS = {'SSP5-8.5': ['exp13'], 'SSP3-7.0': ['exp12'], 'SSP1-2.6': ['exp10'], 'CMIP6 median': ['exp05', 'exp06']}  (also cells 14, 15, 17; source mapping ISMIP6_EXP_SSP in component_projections.p`
- **[high, suspected] [md] cell 5**: No code or stored output in this notebook produces these windowed or IMBIE v2021 numbers. They date from the legacy FIT_SPACE='level' path (fit_bayesian_level plus robust_level_intervals). The full-record numbers from that path were also stale: b = +0.018 and R² = 0.53 in the markdown vs b = -0.0148 and R² = 0.2215 from the level_correlated run. The windowed comparison therefore likely also needs rerunning under level_correlated. The sign-flip conclusion (product vs record length) is unverified under the current method. The full-record level_correlated b is now negative, the same sign as the quoted v2021 value. A negative full-record b does not by itself contradict a positive 1992–2020 b, so these numbers were left unedited in the markdown edits.
  - Expected vs actual: Expected: windowed IMBIE-3 and v2021 fits rerun with fit_bayesian_level_annual_correlated, with the stated numbers updated. Actual: numbers carried over from the legacy method; not reproducible from this notebook.
  - Excerpt: `Sign-flip check: IMBIE-3 on 1992–2020 gives b = +0.024 to +0.026 mm/yr/°C (R² ≈ 0.60) vs b = -0.0295 (robust 90% CI halfwidth 0.147) for IMBIE v2021; v2021 quadratic two-seed R² 0.7402/0.7382. The sam`
- **[medium, confirmed] [md] cell 6**: The progress print hardcodes a zero mean for b when symmetric_b=True and ignores prior_b_mean. The notebook passes prior_b_mean = PRIOR_B_MEAN = -EAIS_SMB.C_T/362500 m/yr/°C, and _level_correlated_log_prior does use it: lp += -0.5*((b - prior_b_mean)/scale)**2.
  - Expected vs actual: Expected 'b~N(-0.17,0.50 mm/yr/°C)'. Actual 'b~N(0,0.50 mm/yr/°C)'. Only the printed label is wrong; the sampled prior is correct.
  - Excerpt: `stored output: 'Priors: b~N(0,0.50 mm/yr/°C), c~N(0.3, 1.0), H0~N(0.00, 5.00) mm' (print in bayesian_models.fit_bayesian_level_annual_correlated)`
- **[medium, confirmed] [cm] cell 6**: The comment says the IMBIE-3 rate and rate-sigma columns are constant within each calendar year ('verified: nunique()==1 per year'). They are not: nunique is 2-3 in 1992, 2002, 2003, 2015, 2019, 2022. .first() takes the January value. np.diff(eais_rebase) equals the within-year MEAN of mass_balance_rate exactly, but differs from the .first() value by up to 0.225 mm/yr (2022) and 0.06 mm/yr (1992). eais_rate_sigma (January value) builds the correlated covariance in fit_bayesian_level_annual_correlated, e.g. 1992 uses 0.115 mm/yr where 9 of 12 months report 0.241 mm/yr.
  - Expected vs actual: Expected: annual rate = within-year mean (consistent with the cumulative record), and an annual sigma built from all months, e.g. RMS or mean. Actual: January value. eais_rate only feeds labels and the Panel-A scatter (cell 9), but eais_rate_sigma sets the likelihood covariance for 6 of 45 years. The cell-6 comment's 'verified' claim needs updating with the fix.
  - Excerpt: `eais_rate = df_eais.groupby('year_int')['mass_balance_rate'].first().values eais_rate_sigma = df_eais.groupby('year_int')['mass_balance_rate_sigma'].first().values`
- **[low, confirmed] [md] cell 1**: check_convergence is imported but never called. No convergence diagnostics run on the default path, although cells 5 and 18 refer to seeds 'failing convergence diagnostics' (a legacy-path observation, now scoped in the proposed edits).
  - Expected vs actual: Expected: convergence checked, or the import removed. Actual: imported but not used.
  - Excerpt: `from bayesian_models import (..., check_convergence)`
- **[low, confirmed] [md] cell 12**: Without baseline_year, project_smb_ensemble rebases cumulative SLE to argmin(|dT|), not to BASELINE_YEAR. The saved medians are exactly zero at 2000 for SSP1-2.6 and SSP5-8.5, so the result currently matches the '2000 baseline' labels in cell 17 and its printed 'relative to 2000' table. That match is a coincidence of the temperature series; baseline_year=BASELINE_YEAR is not enforced.
  - Expected vs actual: Expected: baseline_year=BASELINE_YEAR passed explicitly. Actual: the rebase year is data-dependent and happens to equal 2000.
  - Excerpt: `project_smb_ensemble(..., T_baseline=0.0, n_samples=N_SAMPLES, seed=650)  # no baseline_year`
- **[low, confirmed] [cm] cell 12**: baseline_year is not passed, so project_smb_ensemble rebases at argmin(|dT|) rather than BASELINE_YEAR. With the current temperature data that index falls exactly at 2000.0 for all SSPs (verified, shared historical segment), so there is no numerical effect now. The 2000 anchor depends on that coincidence.
  - Expected vs actual: Expected baseline_year=BASELINE_YEAR. Actual: implicit argmin|dT| = 2000.0 (coincident).
  - Excerpt: `eais_proj = project_smb_ensemble(sensitivity=EAIS_SMB, T_proj=T_proj_annual, time_proj=PROJ_YEARS, T_baseline=0.0, n_samples=N_SAMPLES, seed=650)`
- **[low, confirmed] [cm] cell 17**: ISMIP6 regional time series start at 2016 (85-86 annual values, 2016-2100/2101), so rebasing with argmin(|t - 2000|) anchors ISMIP6 at 2016, not 2000. The ISMIP6 values are also control-run anomalies. Our projections are rebased at 2000. The label and table header state a 2000 baseline for both.
  - Expected vs actual: Expected: a common baseline, or a label that states ISMIP6 is relative to 2016. Actual: ISMIP6 relative to 2016 vs this study relative to 2000. For EAIS the 2000-2016 offset in our projection is small (order 1 mm), but the label is inaccurate. In cell 17 the ISMIP6 table column uses stats['median'][-1]; the common grid is the first run's (2016-2101 for AWI/PISM1), so '[-1]' can be 2101, not 2100.
  - Excerpt: `ax.set_ylabel('EAIS SLR (mm, 2000 baseline)'); print(f'EAIS at 2100 (mm, relative to {BASELINE_YEAR:.0f}):')  (also ismip6_ensemble_stats(..., baseline_year=BASELINE_YEAR) in cells 13-17)`
- **[low, confirmed] [cm] cell 19**: Cell 19 is not gated on REFIT but uses eais_rate_sigma, eais_rate, T_r_eais, n_eais, prior_kw_annual, all defined only inside 'if REFIT:' in cell 6. With REFIT=False it raises NameError.
  - Expected vs actual: Expected: guarded by 'if REFIT:' like cells 7, 9, 10. Actual: unguarded.
  - Excerpt: `sig_rate_tapered = apply_sigma_taper(eais_rate_sigma, ...); fit_bayesian_level_annual_correlated(..., **prior_kw_annual)`
- **[low, confirmed] [md] cell 3**: The printed value is +0.166, the mass-balance-signed conversion of C_T. The label says 'negative = SL fall', which reads as if the value were in the SLR convention. In the SLR-positive convention the sensitivity is -0.166 mm/yr/°C, as used for PRIOR_B_MEAN.
  - Expected vs actual: Expected '-0.166 (negative = SL fall)', or '+0.166 mm/yr/°C mass gain'. Actual '0.166 (negative = SL fall)'.
  - Excerpt: `print(f'  In mm SLE/yr/°C: {EAIS_SMB.C_T * 1.0/362500.0 * M_TO_MM:.3f} (negative = SL fall)') -> '0.166 (negative = SL fall)'`
- **[low, confirmed] [cm] cell 3**: The printed value is +0.166 (mass-gain sign, the same sign as C_T), but the label says '(negative = SL fall)'. In the SLR-positive convention used elsewhere this sensitivity is -0.166 mm SLE/yr/°C. The label and the sign of the printed number don't match.
  - Expected vs actual: Expected e.g. -0.166 with '(negative = SL fall)', or +0.166 labelled as mass gain. Actual: +0.166 labelled '(negative = SL fall)'.
  - Excerpt: `print(f'  In mm SLE/yr/°C: {EAIS_SMB.C_T * 1.0/362500.0 * M_TO_MM:.3f} (negative = SL fall)')`
- **[low, confirmed] [cm] cell 3**: annualize_imbie uses each calendar year's LAST monthly cumulative value (December), labels it yr+0.5, and rebases at the 2000.5 point. So the rebase zero is the end of 2000 (~2001.0), not 2000.0/mid-2000. This is internally consistent with the annual correlated fit (H_i = level at end of year i), but the stated BASELINE_YEAR anchor is offset by about 0.5-1 yr. Shared helper, same behavior in other IMBIE notebooks.
  - Expected vs actual: Expected: zero at BASELINE_YEAR=2000.0. Actual: zero at end-of-2000 cumulative value. The effect for EAIS is sub-mm.
  - Excerpt: `eais_year, eais_rebase, eais_sigma = annualize_imbie(df_eais, baseline_year=BASELINE_YEAR)`
- **[low, confirmed] [cm] cell 6**: The progress print hardcodes a zero mean for the symmetric b prior. prior_b_mean IS applied in _level_correlated_log_prior (verified), so the printed prior mean (0) disagrees with the one actually used (PRIOR_B_MEAN = -0.166 mm/yr/degC). The bug is in the .py print string, not in the notebook.
  - Expected vs actual: Expected 'b~N(-0.17,0.50 mm/yr/°C)'; printed 'b~N(0,0.50 mm/yr/°C)'.
  - Excerpt: `fit output: 'Priors: b~N(0,0.50 mm/yr/°C)'  (print string in bayesian_models.fit_bayesian_level_annual_correlated)`
- **[low, suspected] [cm] cell 7**: Legacy 'level' path only (not executed; FIT_SPACE='level_correlated'). prior_kw sets symmetric_b=True, so the legacy MCMC has no b>=0 boundary, while robust_level_intervals (per its module docstring) models a HalfNormal(b>=0) prior and takes no symmetric_b argument. The comment inherits the b>=0 description. robust_level_intervals also assumes sigma_obs is rebased (zero at the baseline anchor), but annualize_imbie returns IMBIE-3's unrebased cumulative sigma (0.61 mm at 2000).
  - Expected vs actual: Expected: the robust-SE helper and the MCMC use the same prior form and the helper gets rebased sigmas. Actual: they differ. Only matters if the legacy path is re-enabled.
  - Excerpt: `# NOTE: EAIS's b sits within about one posterior-sigma of the b>=0 boundary under this legacy path ... robust_level_intervals(..., prior_scale_b=prior_kw['prior_scale_b'], ...)`

### component_apeninsula.ipynb (18 entries)

- **[high, confirmed] [cm] cell 13**: The ISMIP6 experiment-to-scenario mapping does not match Seroussi et al. (2020) Table 1, the paper that goes with the ComputedScalarsPaper archive (see the archive README and the local PDF in data/raw/ice_sheets/ismip6/). Per that table: exp05 = NorESM1-M RCP8.5, exp06 = MIROC-ESM-CHEM RCP8.5, exp10 = NorESM1-M RCP8.5 with low gamma, exp12 = CCSM4 RCP8.5 with ice-shelf collapse, and exp13 = NorESM1-M RCP8.5 with PIGL gamma. Every group labelled 'SSP...' is therefore a CMIP5 RCP8.5 variant, and exp05/06 are CMIP5, not 'CMIP6 median'. exp12's ice-shelf collapse accounts for the out-of-pattern '+18 [-3, +32]' labelled SSP3-7.0 in the cell-17 table. exp07, the only standard-framework RCP2.6 run, is never read. component_projections.ISMIP6_EXP_SSP cites 'Seroussi et al. 2020, Table 1' but disagrees with it. read_ismip6.ISMIP6_EXPERIMENTS gives a third, also inconsistent, labelling.
  - Expected vs actual: Expected: scenario labels taken from Seroussi 2020 Table 1 (all exp05-exp13 are RCP8.5 except exp07, which is RCP2.6). Actual: exp10/exp12/exp13 are compared against this study's SSP1-2.6/SSP3-7.0/SSP5-8.5 projections in cells 13 (legend), 14, 15 and 17.
  - Excerpt: `EXP_GROUPS = {'SSP5-8.5': ['exp13'], 'SSP3-7.0': ['exp12'], 'SSP1-2.6': ['exp10'], 'CMIP6 median': ['exp05', 'exp06']} (repeated in cell 17; root cause component_projections.ISMIP6_EXP_SSP)`
- **[medium, confirmed] [cm] cell 14**: 'SSP2-4.5' is not a key in EXP_GROUPS, so the histogram (cell 14) and ridge plot (cell 15) silently fall back to exp05/exp06. Those are CMIP5 RCP8.5 runs (see the bug above), yet the plots are titled 'Antarctic Peninsula (SSP2-4.5)' and label the distribution simply 'ISMIP6'.
  - Expected vs actual: Expected: a scenario-matched ISMIP6 comparison, or a label naming the RCP8.5 fallback. Actual: RCP8.5 ISMIP6 runs are shown as the SSP2-4.5 comparison.
  - Excerpt: `ismip6_exps = EXP_GROUPS.get(HIST_SSP, ['exp05', 'exp06'])  # HIST_SSP = 'SSP2-4.5' (same in cell 15 with RIDGE_SSP)`
- **[medium, confirmed] [md] cell 19**: The sensitivity refits omit symmetric_b=True, so they use the function default symmetric_b=False (b >= 0 hard bound, HalfNormal), whereas the main fit (cell 6) uses the symmetric Normal(0, 2) prior adopted 2026-09-19. The sensitivity table therefore compares a different prior, not only a different sigma taper.
  - Expected vs actual: Expected f_max=1 to reproduce the main fit (b median 0.044). Actual f_max=1 row: b = 0.0516, matching the truncated-prior median 0.052 quoted in the cell-6 comment; dBIC -2.8 vs main -3.6.
  - Excerpt: `res_q = fit_bayesian_level_annual_correlated(... fit_sigma_extra=True, n_samples=4000, ...)  [level_correlated branch of the f_max sensitivity loop]`
- **[medium, confirmed] [cm] cell 19**: The sigma-taper sensitivity refits leave out symmetric_b=True, so they use the default HalfNormal prior (b >= 0). The main fit in cell 6 deliberately uses a symmetric prior. The sensitivity table therefore does not measure sensitivity of the adopted model. The f_max=1 row (no taper, which should reproduce the main fit) gives b = 0.0516, against the main fit's 0.0442. A truncated-b rerun of the main configuration gave 0.050, matching cell 6's comment ('0.052 (truncated)').
  - Expected vs actual: Expected: symmetric_b=True, as in cell 6, so the f_max=1 row reproduces b ~ 0.044. Actual: truncated prior; the f_max=1 row gives b = 0.0516.
  - Excerpt: `res_q = fit_bayesian_level_annual_correlated(..., fit_sigma_extra=True, n_samples=4000, ...)  /  res_l = fit_bayesian_level_annual_correlated(..., order=1, fit_sigma_extra=True, ...)`
- **[medium, confirmed] [cm] cell 6**: With fit_sigma_extra=True, each model's chi2 is computed against its own covariance, inflated by that model's posterior-mean sigma_extra. sigma_extra absorbs misfit, so chi2/dof is close to 1 for both models by construction (0.977 and 0.996). The Gaussian log-determinant term, which differs between the models because their sigma_extra differs, is dropped. As a result, 'chi2 + k ln n' is not -2 ln L_max + k ln n, and ΔBIC reduces roughly to the ln(n) penalty difference (-3.8 against the reported -3.6), so 'linear preferred' is close to predetermined. There is also a secondary dof mismatch: the fitter divides chi2 by n - n_phys - 1, but the notebook multiplies back by n - n_phys - 2 (dof_l = n-4, dof_q = n-5). Each reconstructed chi2 therefore comes out low by one chi2_dof (~1), a net effect of ~0.02 on ΔBIC.
  - Expected vs actual: Expected: BIC from the maximized full log-likelihood, including 0.5*log det Sigma(sigma_extra) (as _best_sample_fit does for the 'level' path), with chi2 rebuilt using the fitter's own dof. Actual: chi2-only BIC at the posterior mean, with an inconsistent dof back-conversion.
  - Excerpt: `chi2_l = result_lin.chi2_dof * dof_l; chi2_q = result_quad.chi2_dof * dof_q; bic_l = chi2_l + 2*np.log(n); bic_q = chi2_q + 3*np.log(n)  (same construction in cell 19)`
- **[low, confirmed] [cm] cell 10**: In fit_bayesian_level_annual_correlated, sigma_extra is added to the per-year RATE variance before cumulation (v = sigma_rate^2 + sigma_extra^2), so it has units of m/yr. After * M_TO_MM it is in mm/yr, not mm. The posterior median is ~0.05 mm/yr.
  - Expected vs actual: Expected label: (mm/yr). Actual: (mm).
  - Excerpt: `r'$\sigma_{\mathrm{extra}}$ (mm)'`
- **[low, confirmed] [cm] cell 12**: The SSP temperature series ends at decimal_year 2099.0, so T_full ends at ~2098.92. build_level_design_vectors matches each projection time to the nearest monthly index, so every PROJ_YEARS value from 2099 to 2150 maps to the last month. All 52 values from 2099 to 2150 are therefore identical, and the arrays saved to component_results.h5 through save_apeninsula are flat after 2099. Anything downstream that reads beyond 2099 gets constant values. The '2100' values are really ~2098.9, about 1 yr of rate short (~0.1 mm, ~1% of the 9-13 mm medians).
  - Expected vs actual: Expected: PROJ_YEARS limited to the temperature coverage, or the temperature extended. Actual: projections are flat from 2099 to 2150, and '2100' means ~2098.9.
  - Excerpt: `PROJ_YEARS = np.arange(1950, 2151, dtype=float); t_mon = np.arange(combined['decimal_year'].min(), combined['decimal_year'].max(), 1/12)`
- **[low, confirmed] [cm] cell 12**: The two sides of the offset use different windows. The CMIP historical side is annual and labelled at integer years, so it averages 11 years (1995-2005). The Berkeley side is monthly at mid-month decimal years (1995.04-2004.96), so it averages 10 years (1995-2004). Also, the annual CMIP/SSP means are labelled at Jan 1 (decimal_year = YYYY.0) and interpolated to monthly as if centred there, a ~0.5-yr phase shift relative to the mid-year convention used for Berkeley Earth (suspected; effect small).
  - Expected vs actual: Expected: matching averaging windows. Actual: 1995-2005 (CMIP) vs 1995-2004 (Berkeley Earth).
  - Excerpt: `overlap_mask = (df_hist['decimal_year'] >= 1995) & (df_hist['decimal_year'] <= 2005); temp_bl = np.mean(temp_monthly[(temp_time_monthly >= 1995) & (temp_time_monthly <= 2005)])`
- **[low, confirmed] [md] cell 13**: The ISMIP6 overlay in the two-panel projection plot is commented out, but the legend still carries a dashed 'ISMIP6' entry, so the figure advertises a curve it does not draw.
  - Expected vs actual: Legend entry should be removed or the overlay restored.
  - Excerpt: `ax_sl.plot([], [], 'k--', lw=1.5, alpha=0.7, label='ISMIP6')`
- **[low, confirmed] [cm] cell 13**: The ISMIP6 overlay block is commented out, but the legend still has a dashed 'ISMIP6' entry, so component_apeninsula_twopanel.png shows a legend entry for data that is not plotted.
  - Expected vs actual: Expected: no ISMIP6 legend entry while the overlay is disabled. Actual: an ISMIP6 legend entry with no matching lines.
  - Excerpt: `ax_sl.plot([], [], 'k--', lw=1.5, alpha=0.7, label='ISMIP6')`
- **[low, confirmed] [cm] cell 17**: Every ISMIP6 trajectory starts at 2016.0, so rebasing 'to BASELINE_YEAR=2000' actually rebases to 2016. This study's projections are relative to 2000.0. The cell-17 table header ('relative to 2000'), the cell-14 histogram and the cell-15 ridge plot therefore compare quantities with different baselines (~16 yr apart). Also, ismip6_ensemble_stats takes common_time from the first run; 41 runs end at 2101 and 36 at 2100, so '[-1]' in the table is 2100 or 2101 depending on dict order.
  - Expected vs actual: Expected: both sides on a common baseline, or the label saying ISMIP6 is relative to 2016. Actual: ISMIP6 relative to 2016, labelled as relative to 2000.
  - Excerpt: `bl_idx = np.argmin(np.abs(t - BASELINE_YEAR)); ... ismip6_ensemble_stats(..., baseline_year=BASELINE_YEAR); print(f'Peninsula at 2100 (mm, relative to {BASELINE_YEAR:.0f}):')`
- **[low, confirmed] [md] cell 3**: pen_year holds mid-year values 1979.5..2023.5; formatting with :.0f rounds half to even and prints '1980–2024', which disagrees with the calendar years actually covered (1979-2023) as stated in the markdown, manuscript and supplement.
  - Expected vs actual: Printed 1980–2024; data cover calendar years 1979–2023 (45 points).
  - Excerpt: `print(f'Peninsula IMBIE-3: {pen_year[0]:.0f}–{pen_year[-1]:.0f}, ...')  (also cell 6 FIT_SPACE print)`
- **[low, confirmed] [cm] cell 3**: pen_year runs from 1979.5 to 2023.5. The ':.0f' format rounds these to '1980' and '2024', so the printed range (1980-2024) disagrees with the actual record (1979-2023) and with the markdown/comments.
  - Expected vs actual: Expected: 1979-2023. Actual printed: 1980-2024.
  - Excerpt: `print(f'Peninsula IMBIE-3: {pen_year[0]:.0f}–{pen_year[-1]:.0f}, {len(pen_year)} points')  (also cell 6: f'{pen_year[0]:.0f}-{pen_year[-1]:.0f}')`
- **[low, suspected] [cm] cell 3**: In 6 of 45 years (1992, 2002, 2003, 2015, 2019, 2022), the IMBIE-3 rate and its sigma change mid-year, and the December value is taken rather than an annual representative. For example, the 2022 sigma is up to 30% different across months. The covariance is built from pen_rate_sigma_ann, so this slightly affects the likelihood. Separately, pen_rate_ann (rate_obs) is never used inside fit_bayesian_level_annual_correlated; it is only plotted in cell 9 Panel A.
  - Expected vs actual: Expected: an annual value that represents the whole year (e.g. the time-weighted mean of the monthly sigma^2) for the years that change mid-year. Actual: the December value.
  - Excerpt: `pen_rate_ann = np.array([df_pen_monthly['mass_balance_rate'].values[_yr_int_pen == y][-1] ...]); pen_rate_sigma_ann = ...[-1]`
- **[low, confirmed] [md] cell 6**: fit_bayesian_level_annual_correlated normalizes chi2_dof by (n - n_phys - 1) (sigma_extra not counted), but the notebook multiplies back by n - n_phys - 2, so the recovered chi2 is too small by one chi2_dof unit in each model.
  - Expected vs actual: chi2_l should be 0.977*42 = 41.0 (used 40.1); chi2_q 0.996*41 = 40.8 (used 39.8). Net effect on dBIC is about +0.03, so the linear-preferred conclusion is unchanged.
  - Excerpt: `dof_l = n - 2 - 1 - 1; chi2_l = result_lin.chi2_dof * dof_l  (same for dof_q) and in cell 19 dof_l_t/dof_q_t`
- **[low, suspected] [md] cell 6**: With fit_sigma_extra=True the likelihood includes the log-det of a covariance that depends on sigma_extra, but the BIC here uses only the chi2 at the posterior mean (with posterior-mean sigma_extra), dropping the log-det normalization and not using the maximized likelihood. This is the same issue the notebook's own _best_sample_fit docstring describes for the 'level' path. Size of the effect on dBIC is not quantified.
  - Expected vs actual: Expected BIC = -2 ln L_max + k ln n including log|Sigma(sigma_extra)|; actual uses chi2 at posterior mean only.
  - Excerpt: `bic_l = chi2_l + 2 * np.log(n); bic_q = chi2_q + 3 * np.log(n)`
- **[low, confirmed] [md] cell 6**: The quadratic fit uses symmetric_a=False (a ~ Exponential, a >= 0 hard bound), so P(a>0) = 100% holds by construction and carries no information from the data.
  - Expected vs actual: Printed 'P(a>0) = 100%' reads as a data result; it is a prior constraint.
  - Excerpt: `print(f'P(a>0) = ...')`
- **[low, confirmed] [cm] cell 6**: The quadratic fit uses the default symmetric_a=False, so a has an Exponential prior with a hard a >= 0 bound. P(a>0) = 100% is fixed by the prior, and the printed value says nothing about the data.
  - Expected vs actual: Expected: a diagnostic not fixed by the prior (or none). Actual: prints 100% by construction.
  - Excerpt: `print(f'P(a>0) = {np.mean(result_quad.posterior_samples[:, 0] > 0)*100:.0f}%')`

### component_wais.ipynb (19 entries)

- **[medium, confirmed] [md] cell 24**: Scenario-weight PDF/fan figure legend labels the baseline as n ~ N(4.1, 0.4^2), and the 'current' and 'n3_1090' cases are now identical (both n=3, 10/90). The figure component_wais_scenario_weight_pdf.png therefore shows two coincident curves with contradictory labels.
  - Expected vs actual: Expected: legend reflects n=3 (no rheology correction) and no duplicate case. Actual: 'Current (n ~ N(4.1, 0.4²), 10/90)' plotted over the identical 'n = 3 (fixed), 10/90' curve.
  - Excerpt: `'current': 'Current (n ~ N(4.1, 0.4\u00b2), 10/90)'; header comment 'Compares the published result (n ~ N(4.1, 0.4^2) ...'`
- **[medium, confirmed] [md] cell 31**: Rheology-sensitivity diagnostic is now a no-op: module N_OBS_MEAN=3.0, N_OBS_SIGMA=0.0, so the 'current' run and the 'n=3 fixed' run are identical. Print strings still label the current run n~N(4.1,0.4^2) and report a 'Rheology correction effect'.
  - Expected vs actual: Expected: a comparison of two distinct settings, labeled correctly. Actual: stored output shows identical results (221 mm [64, 1223] both; +0.0% median and p95; trajectory +0.0%). Delete or repurpose cell 30/31: user decision.
  - Excerpt: `print(f'  Current (n~N(4.1,0.4^2)): median=...'); _cproj.N_OBS_MEAN = 3.0 ... finally: restore`
- **[medium, confirmed] [md] cell 6**: Validation cell applies the Mode-A endpoint factor 1.28 ± 0.07 to S2 and reports it as 'S2 post-correction', but the model runs Mode B with n held at 3 (R=1). The printed post-correction S2 (median 315 mm [107, 1670]) and 'Effective rheology factor on median: 1.28' do not describe any distribution the notebook uses.
  - Expected vs actual: Expected: S2 as sampled (pre-correction = actual, 247 mm [84, 1302]). Actual: prints a 1.28x-inflated S2 labeled as the corrected result.
  - Excerpt: `rheo_draws = rng_valid.normal(RHEOLOGY_FACTOR_MEDIAN, RHEOLOGY_FACTOR_SIGMA, N_VALID) ... print('S2 post-correction: ...'); print('Effective rheology factor on median: ...')`
- **[medium, confirmed] [cm] cell 9**: component_levelspace_robust_se._anchor_covariance assumes sigma_obs is the REBASED marginal sigma (variance accumulating outward from the 2000 baseline, zero at the anchor). annualize_imbie does not rebase sigma: it passes IMBIE-3's cumulative sigma, which grows monotonically from the start of the record (0.135 mm at 1979.5, 0.395 mm at the 1999.5 anchor, 0.702 mm at 2023.5). For pre-anchor pairs the construction sets Cov(i,j)=sigma[max(i,j)]^2 > Var(i), so Sigma_data is not positive semidefinite.
  - Expected vs actual: Expected: a PSD covariance for the rebased record. Actual (read-only recomputation): 20 of 45 eigenvalues are negative (min -3.0 mm^2, max 10.3 mm^2). This feeds S1_QUADRATIC_COV (via the sandwich estimator plus _psd_sqrt_diag clipping) and the cubic-diagnostic CIs. The stored output shows the 'robust' acceleration halfwidth is x0.92 of naive, i.e. narrower, which is inconsistent with the stated purpose of widening intervals for a cumulative record. S1's 2100 spread is set mostly by the ISMIP6 extrapolation-error term, so the downstream effect is probably limited, but S1_QUADRATIC_COV is affected directly.
  - Excerpt: `_wais_robust_t = robust_level_intervals(wais_year, wais_rebase, wais_sigma, ...)  [and cell 12 _anchor_covariance(yrs_k, sig_k, baseline_year)]`
- **[low, confirmed] [cm] cell 15**: sample_a4_wais_trajectories initializes beta_arr to zeros and never assigns it for S1_status_quo rows (the S1 branch 'continue's). The printed median is taken over all 10000 samples, including ~9% zeros.
  - Expected vs actual: Expected (S2 only, lognormal(log 2, 0.3)): median ~2.00. Actual printed: 1.93, which matches the 45th percentile of S2's beta distribution after the S1 zeros are included (2*exp(0.3*z_0.449) ~ 1.92). Fix: np.median(wais_params['beta'][wais_params['scenario_idx']==1]).
  - Excerpt: `print(f'  Median beta: {np.median(wais_params["beta"]):.2f}')`
- **[low, suspected] [cm] cell 15**: With anchor_year = 2023.5 and integer PROJ_YEARS, blend_rate_space's forecast grid starts at 2024 (fmask = years >= t_origin) and sets forecast_samples[:,0] = h_origin (the 2023.5 anchor level) at 2024. S2 trajectories therefore show zero growth over the half year 2023.5 to 2024 before the rate integration begins.
  - Expected vs actual: Expected: level at 2024 = anchor + ~0.5 yr of rate. Actual: level at 2024 = anchor. Sub-mm effect at the current ~0.5 mm/yr WAIS rate.
  - Excerpt: `sample_a4_wais_trajectories(... anchor_year=anchor_year ...)  -> blend_rate_space(years, ..., anchor_year, anchor_s2, ...)`
- **[low, confirmed] [cm] cell 24**: Plot legend label states the current rheology is n ~ N(4.1, 0.4^2). The module now has N_OBS_MEAN=3.0, N_OBS_SIGMA=0.0, so the 'current' and 'n3_1090' curves are identical (the figure shows a duplicated curve under two different labels).
  - Expected vs actual: Expected label: n = 3 (fixed). Actual: 'n ~ N(4.1, 0.4^2)'. Saved to component_wais_scenario_weight_pdf.png.
  - Excerpt: `_labels_w = {'current': 'Current (n ~ N(4.1, 0.4\u00b2), 10/90)', ...}`
- **[low, confirmed] [md] cell 29**: HDF5 metadata written to component_results.h5 'wais_s1' describes S1 as a 'Linear ramp'; S1 is the quadratic-in-time IMBIE-3 posterior plus the ISMIP6 extrapolation-error term (_sample_s1_quadratic_mm).
  - Expected vs actual: Expected: 'quadratic-in-time fit to IMBIE-3 + ISMIP6 extrapolation-error term'. Actual: 'Linear ramp'.
  - Excerpt: `sg.attrs['description'] = ('S1 (no-MISI) WAIS projections: ... Linear ramp, no marine ice-sheet instability.')`
- **[low, confirmed] [md] cell 29**: Exported metadata records rheology_factor_median=1.28 / sigma=0.07 alongside rheology_mode='B'. In Mode B those constants are never used, and with n held at 3 the effective factor is 1.0. Downstream readers of the HDF5 could infer that a 1.28 correction was applied.
  - Expected vs actual: Expected: metadata reflecting the applied factor (1.0; n fixed at 3). Actual: 1.28 ± 0.07 recorded.
  - Excerpt: `save_wais(..., rheology_factor_median=RHEOLOGY_FACTOR_MEDIAN, rheology_factor_sigma=RHEOLOGY_FACTOR_SIGMA, rheology_mode=RHEOLOGY_MODE, ...)`
- **[low, confirmed] [cm] cell 29**: HDF5 metadata written to component_results.h5 says S1 is a 'Linear ramp'. S1_status_quo is sampled from the quadratic-in-time posterior (S1_QUADRATIC_MEAN/_COV) plus the ISMIP6 extrapolation-error term; there is no linear ramp.
  - Expected vs actual: Expected: 'Quadratic-in-time fit to IMBIE-3 plus ISMIP6 extrapolation-error term, no MISI.' Actual: 'Linear ramp'.
  - Excerpt: `sg.attrs['description'] = ('S1 (no-MISI) WAIS projections: 100% weight on S1_status_quo scenario. ' 'Linear ramp, no marine ice-sheet instability.')`
- **[low, confirmed] [cm] cell 29**: The wais group records rheology_factor_median=1.28 and sigma=0.07 as attributes, but in Mode B with n held at 3 these constants are not used and the applied factor is 1.0. A downstream reader could take 1.28 as the applied correction.
  - Expected vs actual: Expected: metadata reflecting that no rheology correction is applied (R=1, n=3). Actual: 1.28/0.07 stored.
  - Excerpt: `save_wais(..., rheology_factor_median=RHEOLOGY_FACTOR_MEDIAN, rheology_factor_sigma=RHEOLOGY_FACTOR_SIGMA, rheology_mode=RHEOLOGY_MODE, ...)`
- **[low, confirmed] [md] cell 3**: Annual year centers are 1979.5–2023.5; the .0f format rounds them to 1980–2024, which disagrees with the 1979–2023 record span stated in markdown (cells 0, 4) and the data (Jan 1979–Dec 2023). Same rounding in cell 15 'Anchor: year=2024' (anchor is 2023.5).
  - Expected vs actual: Expected: 1979–2023 (or 1979.5–2023.5). Actual: 1980–2024.
  - Excerpt: `print(f'WAIS IMBIE-3 (Otosaka et al. 2026): {wais_year[0]:.0f}–{wais_year[-1]:.0f}, ...')  -> '1980–2024, 45 points'`
- **[low, suspected] [md] cell 3**: annualize_imbie labels each annual value at mid-year (yr+0.5) but takes the last (December) monthly cumulative value, which is an end-of-year quantity (~yr+0.96). That is a ~0.5-yr timing offset in the level series used for the S1 fit and the anchor. The rebase point (1999.5) likewise holds the Dec-1999 value. The level effect is small (~0.25 mm at 0.5 mm/yr), but the convention looks unintended.
  - Expected vs actual: Expected: annual mean of monthly values at yr+0.5, or the Dec value labeled at yr+1.0. Actual: Dec value labeled yr+0.5.
  - Excerpt: `annualize_imbie (component_analysis.py): years[i] = yr + 0.5; H[i] = cumulative_mass_balance[mask][-1]`
- **[low, confirmed] [cm] cell 3**: annualize_imbie (component_analysis.py) takes the LAST monthly value of each calendar year (December, ~yr+0.96) but labels it years = yr + 0.5. Each annual level is therefore time-stamped ~0.46 yr early. The rebase point (argmin |years-2000| -> 1999.5) is the Dec-1999 value, the fit anchor is labeled 2023.5 but holds the Dec-2023 value, and the trajectory anchor_year=2023.5 is used for splicing and the S2 blend.
  - Expected vs actual: Expected: time stamp equals the epoch of the value (~yr+0.96), or the value is the annual mean at yr+0.5. Actual: Dec value labeled mid-year. Implies a ~0.5-yr shift in fitted v/H0 and in the anchor epoch. Module-level (not a notebook comment).
  - Excerpt: `wais_year, wais_rebase, wais_sigma = annualize_imbie(df_wais, baseline_year=BASELINE_YEAR)`
- **[low, confirmed] [cm] cell 3**: wais_year runs 1979.5-2023.5; formatting with .0f rounds to '1980–2024'. The source data span Jan 1979-Dec 2023 (read_imbie3 docstring; decimal_year 1979.04-2023.96).
  - Expected vs actual: Expected: '1979–2023' (or 1979.5–2023.5). Actual printed: '1980–2024'.
  - Excerpt: `print(f'WAIS IMBIE-3 (Otosaka et al. 2026): {wais_year[0]:.0f}–{wais_year[-1]:.0f}, ...')`
- **[low, confirmed] [cm] cell 31**: Print label states the current rheology is n~N(4.1,0.4^2). Module values are n=3 fixed; the stored output shows identical current and n=3 results (+0.0%). mix_n3_3070_mm and mix_n3_0199_mm are computed but never used in this cell.
  - Expected vs actual: Expected label: n=3 (fixed, current). Actual: 'n~N(4.1,0.4^2)'.
  - Excerpt: `print(f'  Current (n~N(4.1,0.4^2)): median=...')`
- **[low, suspected] [cm] cell 5**: The LARMIP-2 curve labeled 'AIS: LARMIP-2' reads the _TOT_ file, not the _AIS_ file. Both exist in the LARMIP output set (dist_components also has _SMB_). If TOT = AIS dynamics + SMB, the label and the ISMIP6 comparison (which uses _AIS_) are not like-for-like. I did not verify which file AR6 used for the p-box.
  - Expected vs actual: Expected: consistent AIS definition across workflows, or a label stating TOT. Actual: _TOT_ file under an 'AIS' label.
  - Excerpt: `larmip_med = read_ipcc_workflow_samples(WORKFLOW_BASE, 'wf_2e', 'ssp245', 'icesheets-ipccar6-larmipicesheet-ssp245_TOT_globalsl.nc')`
- **[low, confirmed] [cm] cell 6**: The validation cell applies a Mode-A-style rheology factor R~N(1.28, 0.07^2) and prints 'S2 post-correction: median = 315 mm [107, 1670]' and 'Effective rheology factor on median: 1.28'. The published path is Mode B with n held at 3 (N_OBS_MEAN=3, N_OBS_SIGMA=0), which applies no correction (R=1). The printed post-correction numbers do not correspond to any published result.
  - Expected vs actual: Expected: validation reflects the Mode B/n=3 path (post = pre). Actual: prints a 1.28x-inflated S2 distribution labeled as 'post-correction'.
  - Excerpt: `rheo_draws = rng_valid.normal(RHEOLOGY_FACTOR_MEDIAN, RHEOLOGY_FACTOR_SIGMA, N_VALID) ... print(f'S2 post-correction: ...') / 'Effective rheology factor on median'`
- **[low, confirmed] [cm] cell 9**: fit_bayesian_level (and robust_level_intervals, to match) center the H0 prior on H_obs[0], the first observation (1979.5, -2.06 mm). With I0 = tau = year - 2000 here, H0 is the level at 2000, not at the first observation. The prior (sigma 5 mm) therefore pulls H0 toward the 1979.5 value.
  - Expected vs actual: Expected: H0 prior centered near 0 (the level at the 2000 rebase). Actual: centered at -2.06 mm. The data dominate (posterior H0 = 0.257 mm vs WLS 0.258 mm), so the numerical impact appears negligible.
  - Excerpt: `fit_bayesian_level(..., prior_H0_sigma=PRIOR_H0_SIGMA_WAIS, ...) / robust_level_intervals(..., prior_H0_sigma=...)`

### component_summation.ipynb (19 entries)

- **[medium, confirmed] [cm] cell 16**: Stored ocean, glacier and Antarctic Peninsula projections in component_results.h5 are constant after 2099-2100 (no sample changes after 2100 for ocean, 2099 for glacier/apeninsula; checked SSP5-8.5). The 2150 milestone rows of the summary table and Total_sum at 2150 therefore freeze these components at their 2100 values while Greenland, EAIS, WAIS and TWS keep evolving. Origin is upstream (component notebooks, likely GMST trajectories ending 2100 with interp clamping), but this notebook reports 2150.
  - Expected vs actual: expected 2150 > 2100 for ocean/glacier under warming; actual identical values
  - Excerpt: `summary table 2150 rows: Thermosteric 2150 == 2100 (e.g. 143 [114, 182]), Glaciers likewise`
- **[medium, confirmed] [md] cell 18**: The IPCC-to-2000 baseline offset has the wrong sign. AR6 values are H(t) - H(ref), with ref centered at 2004.5. Converting to a 2000 baseline requires adding H(2004.5) - H(2000) = +11.1 mm. The code defines the offset as H(2000) - H(2004.5) = -11.1 mm and adds it. Numerical check: the raw AR6 SSP2-4.5 total median at 2020 is 50 mm, and NASA altimetry rebased to 2000 is +67.3 mm at 2020. The correct conversion gives 61 mm, but the code plots 39 mm.
  - Expected vs actual: Expected: IPCC + 11.1 mm (i.e. subtract ipcc_offset_mm as currently defined, or flip its definition). Actual: IPCC - 11.1 mm. The plotted IPCC total curves and 5-95% bands in cells 20 and 28 are about 22 mm too low against the 2000 baseline.
  - Excerpt: `ipcc_offset_mm = (_gmsl_at_bl - _gmsl_at_ipcc_ref) * M_TO_MM  # mm to add to IPCC  (used as `ie['q50'] + ipcc_offset_mm` in cells 20 and 28)`
- **[medium, confirmed] [cm] cell 18**: Sign error in the IPCC rebasing offset. IPCC values are V(t)=GMSL(t)-GMSL_ref(1995-2014). Expressing them relative to 2000 requires adding GMSL_ref - GMSL(2000) (about +11 mm, since sea level rose 2000->2004.5). The code adds GMSL(2000) - GMSL(2004.5) = -11.1 mm (stored output). Used in cells 20 and 28, so IPCC median/5-95 curves in component_summation_total.png and component_summation_vs_ipcc.png sit ~22 mm too low. Project memory records the correct total-GMSL offset as positive (+10.7 mm).
  - Expected vs actual: expected offset ~ +11 mm; actual -11.1 mm
  - Excerpt: `ipcc_offset_mm = (_gmsl_at_bl - _gmsl_at_ipcc_ref) * M_TO_MM  # mm to add to IPCC`
- **[medium, confirmed] [cm] cell 21**: IPCC comparisons with no baseline offset: cell 21 (IPCC sum excluding AIS) and cell 29 (2100 numerical table) use IPCC values relative to 1995-2014 while our curves/samples are relative to 2000. Cell 29 IPCC column (e.g. SSP2-4.5 556 [371, 946] mm) is therefore on a different baseline than 'Ours'. Cell 21 would need per-component offsets (not the total-GMSL one).
  - Expected vs actual: expected both on the 2000 baseline; actual IPCC on 1995-2014 baseline (~11 mm for total)
  - Excerpt: `ax.plot(d['years'], d['q50'], ...)  # IPCC excl. AIS, no offset; also cell 29 get_ipcc_stats(...) with no offset`
- **[low, confirmed] [md] cell 10**: The print reports the full Frederikse record range (1900-2018). The cell sets TWS to zero before TWS_START_YEAR = 2003 and uses only the GRACE era.
  - Expected vs actual: Expected: a print showing 2003-2018 (GRACE era), zero before. Actual: '1900–2018'.
  - Excerpt: `print(f'TWS (Frederikse): {fred_year[0]:.0f}–{fred_year[-1]:.0f}, extrapolated flat beyond 2018')`
- **[low, confirmed] [cm] cell 10**: The hindcast cells read wais_2k (and cell 11 the Greenland hindcast_gmst_1900 group) without the per-sample rebase to BASELINE_YEAR that cells 3/4 apply to the projections. Pre-anchor WAIS samples carry IMBIE measurement noise, so individual samples are up to 1.6 mm (WAIS) / 1.0 mm (Greenland) off zero at 2000; median offset ~0.06 mm. Numerically negligible, but inconsistent with the rest of the notebook.
  - Expected vs actual: expected samples zero at 2000; actual |offset| up to 1.6 mm per sample
  - Excerpt: `_wais_samp = _hf['wais_2k/full_samples'][:]  (also cell 11 w_samp_ and greenland/hindcast_gmst_1900)`
- **[low, suspected] [cm] cell 10**: Independent per-year noise is added to cumulative (integrated) TWS and EAIS records. _epoch_trend_sigma then treats these as the MC rate uncertainty, which understates trend uncertainty for these terms relative to a correlation-aware treatment (see project pitfall on cumulative records). Their contribution to the combined sigma is small but likely underestimated.
  - Expected vs actual: expected correlated (random-walk-like) perturbations on cumulative records; actual white noise per year
  - Excerpt: `tws_draw = fred_tws_rb + rng_tws.normal(0, fred_tws_sig); eais_obs_samples[:, j] = rng_eais.normal(val, sig, size=n_ea)`
- **[low, confirmed] [md] cell 11**: Same issue in the 20th-century hindcast. TWS is zero before 2003 (_grace_era20), but the print says 1900-2018.
  - Expected vs actual: Expected: 2003-2018. Actual: '1900–2018' (stored output: 'TWS: Frederikse obs 1900–2018').
  - Excerpt: `print(f'TWS: Frederikse obs {fred_year[0]:.0f}\u2013{fred_year[-1]:.0f}')`
- **[low, confirmed] [md] cell 11**: The print string points readers to 'conversation notes', which are not part of the repository.
  - Expected vs actual: Expected: a pointer to a repo location (e.g. component_greenland.ipynb), or no pointer. Actual: a reference to 'conversation notes'.
  - Excerpt: `print(f'Greenland: GMST-driven hindcast ... see conversation notes on r0 extrapolation and SMB C_T out-of-domain caveats before 1972)')`
- **[low, confirmed] [cm] cell 11**: Print strings report 1900-2018 but only the GRACE-era (2003-2018) part of the Frederikse TWS record is used; TWS is set to zero before 2003.
  - Expected vs actual: expected '2003–2018'; actual '1900–2018'
  - Excerpt: `print(f'TWS: Frederikse obs {fred_year[0]:.0f}\u2013{fred_year[-1]:.0f}')  (and cell 10: 'TWS (Frederikse): 1900–2018, extrapolated flat beyond 2018')`
- **[low, confirmed] [cm] cell 13**: With min_periods=6 the smoothed series is defined to the last year, so the end value is a one-sided 6-yr mean (e.g. 2020-2025) while the start value is a centered 11-yr mean. The comment's 'centered multi-year means' holds only at the start. This affects the 11-yr warming spans quoted for the manuscript.
  - Expected vs actual: expected centered 11-yr means at both ends; actual truncated one-sided window at the end
  - Excerpt: `_sm = _a.rolling(WARM_SMOOTH, center=True, min_periods=WARM_SMOOTH // 2 + 1).mean(); _end_sm = int(_sm.dropna().index.max())`
- **[low, suspected] [cm] cell 18**: MC band for the satellite-era quadratic subtracts the median fit's baseline value from every sample instead of each sample's own value at BASELINE_YEAR, so the band keeps intercept uncertainty at 2000 (nonzero width at the baseline), unlike the per-sample rebasing used for component ensembles. May be intentional.
  - Expected vs actual: expected per-sample rebase (zero width at 2000) if consistent with other ensembles; actual common shift
  - Excerpt: `_sq_boot_rb = _sq_boot - _sq_at_bl`
- **[low, confirmed] [md] cell 21**: The IPCC excl.-AIS component sum is plotted and printed on the AR6 1995-2014 reference (centered 2004.5) with no conversion. Our excl.-AIS sum on the same axes is rebased to 2000.
  - Expected vs actual: Expected: IPCC excl.-AIS values converted to the 2000 baseline using the non-AIS components' own 2000-to-2004.5 change, not the total-GMSL offset (and not ipcc_offset_mm as currently signed; see the cell-18 bug). Actual: no conversion, so the IPCC curves sit several mm low relative to ours. The magnitude is less than the 11.1 mm total-GMSL offset because AIS is excluded.
  - Excerpt: `ax.plot(d['years'], d['q50'], ...)  # ipcc_no_ais, no ipcc_offset_mm`
- **[low, suspected] [md] cell 24**: The plot title hardcodes '3°C warming' for SSP2-4.5. The notebook does not compute or trace this value. The stored SSP2-4.5 driving trajectory (/projections/temp/SSP2_4_5 in slr_processed_data.h5) averages 2.65 °C over 2081-2100 on that series' own baseline, which I did not check against pre-industrial.
  - Expected vs actual: Expected: a warming value traceable to the driving trajectory. Actual: a hardcoded '3°C'.
  - Excerpt: `ax.set_title(f'Stacked component means for 3$^\circ$C warming ({ssp_show})')`
- **[low, suspected] [cm] cell 24**: Plot title labels SSP2-4.5 as '3°C warming'; the AR6 best estimate for SSP2-4.5 (2081-2100 vs 1850-1900) is 2.7°C. Label only.
  - Expected vs actual: expected ~2.7°C or no temperature label; actual '3°C'
  - Excerpt: `ax.set_title(f'Stacked component means for 3$^\\circ$C warming ({ssp_show})')`
- **[low, confirmed] [md] cell 29**: get_ipcc_stats (component_projections.py:1288) returns raw AR6 slc values relative to 1995-2014 (centered 2004.5). The 'Ours' column is relative to 2000.
  - Expected vs actual: Expected: IPCC median [5,95] + 11.1 mm (H(2004.5) - H(2000)). Actual: printed unconverted, so the IPCC column is about 11 mm low relative to the 2000 baseline. Do not fix this by adding ipcc_offset_mm as currently signed (see the cell-18 bug).
  - Excerpt: `ipcc_stats = get_ipcc_stats({ssp: {'total': ipcc_total[ssp]}}, ssp, 'total', year=2100)`
- **[low, confirmed] [cm] cell 3**: component_results.h5 root attr baseline_year = 2005.0 (set at file creation, 2026-03-25), while component_io docstring and the stored data use BASELINE_YEAR=2000. Stale metadata, likely source of the '(2005)' comment in cell 3.
  - Expected vs actual: expected 2000.0; actual 2005.0
  - Excerpt: `HDF5 root attribute baseline_year (component_results.h5)`
- **[low, suspected] [md] cell 5**: TWS synthetic samples are drawn independently at each IPCC time step (2005, 2020, 2030, ...), so a single TWS trajectory has no temporal correlation. Also, AR6 landwaterstorage starts at 2005 (value 0), so np.interp clamps every year before 2005 to the 2005 draw. The 'rebase to BASELINE_YEAR' step therefore subtracts the 2005 draw (sigma floor 0.1 mm), and TWS is effectively referenced to 2005, not 2000. Per-year percentiles are unaffected and the baseline effect is about a few mm at most. Any trajectory-level statistic (rates, trends) of Total_sum inherits the uncorrelated TWS noise.
  - Expected vs actual: Expected (if trajectories are meant to be coherent): one quantile or z draw per sample applied across all years, and TWS carried back to 2000. Actual: independent draws per time step, and TWS is zero over 1950-2005.
  - Excerpt: `raw_samples_mm[:, j] = rng_tws.normal(q50[j], sigma_mm[j], N_SAMPLES)  (per IPCC time step)`
- **[low, confirmed] [cm] cell 5**: (a) IPCC landwaterstorage grid is 2005, 2020, 2030, ... with q05=q50=q95=0 at 2005; np.interp clamps all pre-2005 years to the 2005 draw, so after the 'rebase to BASELINE_YEAR' TWS is ~0 through 2005 and is effectively relative to ~2005 (IPCC ref period), not 2000; 2000-2005 TWS change is omitted. (b) Draws are independent at each IPCC time point, so per-sample TWS paths have no temporal coherence (marginal percentiles at each year are fine; per-sample rates/paths are not).
  - Expected vs actual: expected TWS relative to 2000 with coherent paths; actual relative to ~2005, independent draws per IPCC year
  - Excerpt: `raw_samples_mm[:, j] = rng_tws.normal(q50[j], sigma_mm[j], N_SAMPLES) ... np.interp(PROJ_YEARS, ipcc_years, ...)`

### component_forecast.ipynb (13 entries)

- **[medium, confirmed] [cm] cell 16**: IPCC AR6 baselines are handled three different ways in this notebook. Cells 16 (fan plot) and 21 (summary table) plot/tabulate raw NetCDF quantiles, which are relative to the 1995-2014 mean, on an axis/table otherwise relative to 2000 (no offset). Cell 20 (ridge) uses ipcc_distributions.h5, which precompute_ipcc_distributions.py shifts by a fixed +10.7 mm (NASA 2000-to-2005). Cell 12 (headline IPCC ratio) shifts our median by 12.1 mm computed from Frederikse (1995-2014 mean minus 2000).
  - Expected vs actual: Expected: one baseline treatment for IPCC across the notebook. Actual: 0 mm (cells 16, 21), +10.7 mm on IPCC (cell 20), -12.1 mm on ours (cell 12). E.g. IPCC SSP2-4.5 2100 median is 556 mm in the cell-21 table vs ~567 mm in the cell-20 ridge input.
  - Excerpt: `ax.plot(ie['years'][ipcc_mask], ie['q50'][ipcc_mask], ...)  with  ax.set_ylabel('GMSL (mm, rel. to 2000)'); also cell 21 get_ipcc_stats(...) column 'IPCC AR6 (median [5, 95] mm)'`
- **[low, confirmed] [md] cell 11**: The observed GMSL at the forecast origin (t = nasa_time[-1] = 2025.3, stored output of cell 8) is assigned to the first annual grid point at or after it (f_years[0] = 2026.0, stored output of cell 11). The ~0.7 yr of rise between 2025.3 and 2026.0 is never added, so every blended sample (both 'forecast' and 'forecast_stable', and the exported 'blended'/'blended_stable' groups and headline JSON) is shifted low by roughly 0.7 yr x ~4.5 mm/yr, about 3 mm, at all years. The printed 'Sigmoid weight at origin' is also evaluated at 2026.0, not at T_ORIGIN.
  - Expected vs actual: Expected: H(2025.3) = H_ORIGIN, then integrate the blended rate from 2025.3. Actual: H(2026.0) = H_ORIGIN.
  - Excerpt: `blend_rate_space(... T_ORIGIN, H_ORIGIN ...) -> component_projections.blend_rate_space: fmask = proj_years >= t_origin; forecast_samples[:, 0] = h_origin`
- **[low, confirmed] [cm] cell 11**: The forecast grid starts at the first integer year >= T_ORIGIN (2026.0), but the level at that grid point is set to H_ORIGIN, the observation at T_ORIGIN = 2025.34. The ~0.66 yr between the last observation and the first grid point is dropped, so every blended level (forecast and forecast_stable, and hence the headline JSON and exported 'blended'/'blended_stable' groups) is offset low by roughly rate x 0.66 yr. The print 'Sigmoid weight at origin' actually reports w at 2026.0 (0.858), not at T_ORIGIN.
  - Expected vs actual: Expected: level at 2026.0 = H_ORIGIN + integral of the blended rate from 2025.34 to 2026.0 (about +3 mm at 4.5 mm/yr). Actual: level at 2026.0 = H_ORIGIN.
  - Excerpt: `blend_rate_space(proj_years, comp_samp, ..., T_ORIGIN, H_ORIGIN, ...)  [component_projections.py: fmask = proj_years >= t_origin; forecast_samples[:, 0] = h_origin]`
- **[low, confirmed] [cm] cell 12**: Variances use np.var (ddof=0) while cross-component covariances use np.cov (ddof=1), so components + covariances + between-scenario do not sum exactly to total_var. The discrepancy is a factor N/(N-1) on the covariance term (N=2000), which is below the 0.1% print precision (sum check shows 100.0%).
  - Expected vs actual: Expected: consistent ddof. Actual: mixed ddof=0/1.
  - Excerpt: `scenario_vars.append(np.var(s)) ... comp_var_by_ssp[...].append(np.var(...)) ... cov_sum += 2 * np.cov(a, b)[0, 1]`
- **[low, confirmed] [cm] cell 12**: The printed 'our' median is our_median_cm (2000 baseline), but ratio and pct_above_ipcc are computed from our_median_cm_ar6_baseline. E.g. SSP1-2.6 prints '62 vs IPCC 43.7 cm = 1.4x (38% above)' while 62/43.7 = 1.41 (41% above); the 38% corresponds to 60.3 cm.
  - Expected vs actual: Expected: print our_median_cm_ar6_baseline alongside the ratio. Actual: prints the 2000-baseline value.
  - Excerpt: `print(f'  {ssp}: {ic["our_median_cm"]:.0f} vs IPCC {ic["ipcc_median_cm"]} cm = {ic["ratio"]:.1f}x ...')`
- **[low, confirmed] [md] cell 16**: IPCC AR6 total quantiles are read directly from the NetCDF via read_ipcc_component_nc/ipcc_extract, which do not rebase (native 1995-2014 baseline, ~2005). They are plotted on an axis labeled rel. to 2000 alongside our forecast. The same unrebased values populate the 'IPCC AR6' column of the cell 21 summary table (get_ipcc_stats). The ridge plot (cell 20) instead uses ipcc_distributions.h5 'total', which is rebased by +10.7 mm, so the three IPCC displays are not on a common baseline. Cell 12 handles this correctly by applying ipcc_baseline_offset_mm.
  - Expected vs actual: Expected: IPCC curves/table shifted to the 2000 baseline (about +10.7 to +12 mm). Actual: native AR6 baseline, about 1 cm low relative to our curves.
  - Excerpt: `ax.plot(ie['years'][ipcc_mask], ie['q50'][ipcc_mask], ...); ax.set_ylabel('GMSL (mm, rel. to 2000)')`
- **[low, confirmed] [md] cell 5**: The TWS samples come from ipcc_distributions.h5, where precompute_ipcc_distributions.py (lines 56-59, 181-184) adds the full 10.7 mm total-GMSL 2000->2005 rebase offset to every component, including landwaterstorage. TWS enters Total_sum (cell 6) with this offset, so Total_sum and Total_stable levels are raised by up to 10.7 mm (minus the true TWS 2000-2005 change). In addition, the stored AR6 years are [2005, 2020, 2030, ...], so np.interp onto PROJ_YEARS (1950-2150) holds TWS constant at its 2005 value before 2005 and interpolates linearly from 2005 to 2020. The blended forecast is unaffected (blend_rate_space uses only rates after the origin, and the variance decomposition is shift-invariant), but the level of the raw component sum is affected: the dotted 'Component sum (raw)' curves in the cell 16 figure and the 'Comp. sum' column of the cell 21 summary table.
  - Expected vs actual: Expected: TWS referenced to the 2000 baseline with its own 2000-2005 contribution. Actual: AR6 native TWS + 10.7 mm, held constant before 2005.
  - Excerpt: `load_ipcc_samples('landwaterstorage', ssp) from ipcc_distributions.h5; all_proj['tws'] = tws_proj`
- **[low, confirmed] [md] cell 5**: The TWS 'samples' are not trajectories. precompute_ipcc_distributions.py draws each native year independently, and the sample correlation between adjacent native years in ipcc_distributions.h5 samples/landwaterstorage/SSP2-4.5 is about 0 (-0.026 between columns 7 and 8). Summing these into Total_sum and differentiating them in blend_rate_space treats jumps between independent draws as a rate. The TWS contribution to the forecast spread is therefore roughly Var(TWS(Y)) + Var(TWS(origin)) instead of the variance of the change, and the linearly interpolated segments between decadal nodes create spurious per-sample rate steps. Probably small next to the WAIS spread, but it departs from the sample-by-sample coherence described in cells 0 and 10.
  - Expected vs actual: Expected: TWS sample paths with temporal correlation (e.g. a common quantile per sample across years). Actual: independent draws per native year.
  - Excerpt: `samples_m[i] = np.interp(PROJ_YEARS, ipcc_years, raw_samples_mm[i] / M_TO_MM)  (samples drawn in precompute_ipcc_distributions.py: skewnorm.rvs(...) independently at each year j)`
- **[low, confirmed] [cm] cell 5**: Previously documented issue (memory: IPCC per-component rebase bug): ipcc_distributions.h5 adds the full +10.7 mm total-GMSL rebase offset to the TWS component. Scope here: the blended forecast and all variances are unaffected (blend_rate_space uses rates only; a constant offset has no effect on variance). Total_sum and Total_stable levels are shifted by +10.7 mm, which affects the dotted component-sum curves in cell 16, the 'Comp. sum' column in cell 21, and the TWS levels in all_proj['tws']. TWS is also held flat at its 2005 value before 2005 by np.interp.
  - Expected vs actual: Expected: TWS near 0 at BASELINE_YEAR 2000. Actual: TWS includes a +10.7 mm offset.
  - Excerpt: `ipcc_years, raw_samples_mm = load_ipcc_samples('landwaterstorage', ssp)`
- **[low, confirmed] [cm] cell 5**: precompute_ipcc_distributions.py draws the TWS samples independently at each IPCC year (skewnorm.rvs per year j with a shared rng), so sample i is not a coherent trajectory; its decade-to-decade increments are random. Differencing these in blend_rate_space produces rate noise, and the blended level change from 2026 to year t carries the difference of two independent draws rather than a trajectory increment. TWS is 0.1% of the 2100 variance, so the numerical effect is small.
  - Expected vs actual: Expected: temporally coherent TWS trajectories (e.g., one quantile rank per sample across years). Actual: independent per-year draws.
  - Excerpt: `samples_m[i] = np.interp(PROJ_YEARS, ipcc_years, raw_samples_mm[i] / M_TO_MM)`
- **[low, confirmed] [cm] cell 6**: Outside this notebook: wais_2k/full_samples and wais_2k/s1_samples are downsampled with independent index sets, so sample k of Total_sum and sample k of Total_stable share the non-WAIS part but carry unrelated WAIS draws (even for scenario_idx==0 members). compute_evpi.py only uses the full and stable arrays as marginals plus a mask on the full array, so no current consumer appears affected; any future per-sample full-minus-stable difference would be.
  - Expected vs actual: Expected (if pairing is ever needed): s1 sample k corresponds to full sample k. Actual: independent.
  - Excerpt: `[component_wais.ipynb cell 29] _idx_full_ds = _rng_ds.choice(...); _idx_s1_ds = _rng_ds.choice(...)  (independent indices)`
- **[low, suspected] [cm] cell 8**: The forecast origin level is the difference of two single ~10-day NASA samples (nearest to 2000.0 and the last record point), both unsmoothed ('gmsl', not 'gmsl_smoothed'). Adjacent samples differ by several mm at both ends (e.g. last six values 76.1-80.6 mm; around 2000: -2.7 to +1.8 mm). H_ORIGIN is also treated as exact (no observational uncertainty propagated into the forecast). The quadratic samples, by contrast, are rebased to the fitted value at 2000.
  - Expected vs actual: Expected: origin/baseline levels from an averaged or fitted value with uncertainty. Actual: single-point values, deterministic, ~+/-2 mm noise each.
  - Excerpt: `idx_bl = np.argmin(np.abs(nasa_time - BASELINE_YEAR)); nasa_gmsl_rb = nasa_gmsl - nasa_gmsl[idx_bl]; H_ORIGIN = nasa_gmsl_rb[-1]`
- **[low, confirmed] [cm] cell 9**: With meas_cov_path set, fit_satellite_era_quadratic returns cov_params = cov_beta_meas + cov_beta_serial; the GIA rate uncertainty (0.15 mm/yr, 1 sigma) enters only rate_accel_cov ('GIA only enters the rate-accel covariance, not param cov'). The MC draws of the quadratic, and hence the near-term blended rate, therefore omit the GIA term. Whether this is intended is unclear.
  - Expected vs actual: Expected (if GIA should propagate): sq_rate_samples spread includes sigma_GIA. Actual: it does not.
  - Excerpt: `sq_coeff_samples = rng_sq.multivariate_normal(sat_quad.coefficients, sat_quad.cov_params, size=N_MC_QUAD)`

## Appendix B: Out-of-scope notes from auditors

### component_ocean.ipynb

- Cell 3 comment 'pre-1955 is model-constrained' (Frederikse): Frederikse et al. (2020) Methods use in situ steric from 1957; pre-1957 is the SST/reanalysis reconstruction. Same fix as proposed for markdown cell 2.
- Cell 3 comments 'loaded for reference, not used in fit' (Frederikse) and 'loaded for validation' (Dangendorf): Frederikse is now also used as withheld validation in cell 7.
- Cell 3 comment 'Rebaseline to 1995–2005 mean of the 0–700 m record cannot be computed' while the 0-700 m rebasing window in code is 1995–2006.
- Cell 5 comment 'NOAA thermosteric SL underestimates due to sparse sampling of the Southern Ocean and deep tropics': per the local Li et al. (2022) PDF (memory note project_li2022_thermosteric_bias) the 14% bias arises from linear vertical interpolation and peaks at low latitudes (~17N, 12S), not the Southern Ocean.
- Cell 6 comment 'b_u and tau_u trade off strongly (...)' describes a likelihood degeneracy without stating that tau_u's informative LogNormal(log 8, 0.5) prior breaks it (project wording rule). Same cell: 'Median [90% CI] values ... instead reported in the figure caption' but the code draws them as a table inside the figure.
- Cell 7 comment '_thin = flat[::100]  # ~5,800 draws': flat has 64 x 30000 = 1,920,000 rows, so the thinned set is 19,200 draws.
- Cell 9 comments '1956-2005: 0-700m corrected only', '2006-2026: ...', 'first 0-2000m data point (2006)': the data are at mid-year 1955.5-2004.5 and 2005.5-2025.5; the first 0-2000 m point is 2005.5 (the notebook's own printout says 1955-2004 / 2005-2025).
- Cell 7 print label '0-2000m      ' has extra padding relative to the other rows (cosmetic).
- Stored outputs are stale relative to the working tree. The cell 3 and cell 7 outputs have no Dangendorf lines (no 'Dangendorf sterodynamic (validation)' print, and no pre-/post-1957 split prints), so the uncommitted Dangendorf code has not been executed in the saved notebook. Numbers quoted from cell 7 for Dangendorf cannot be checked against outputs.
- Markdown cells were not audited (the other agent covers them). Cell 2 describes the 0-2000 m record as '2005–2025'; this matches the data.
- Li et al. (2022) attribution: memory file project_li2022_thermosteric_bias.md records that the manuscript v0 .tex files and the supplement also carry the old 'Southern Ocean sampling' wording. The proposed edit fixes only the cell 5 code comment.
- The cell 7 DANG_C color-contrast claims: the WCAG contrast of #CC7A00 on white is 3.30 (clears 3:1, verified). The deutan dE values 7.7/16.0 were not recomputed.
- RATE_BELOW_2000 = 0.07e-3 +/- 0.04e-3 m/yr is attributed to Purkey & Johnson (2010) in cells 7 and 9. The local Zotero text (K7GWHUBL) reports 0.053 +/- 0.017 mm/yr (abyssal, below 4000 m) plus 0.093 +/- 0.081 mm/yr (Southern Ocean, 1000-4000 m), about 0.1 mm/yr in total. The alternative 2000 m interface values are in their Table 1, which did not parse from the cache. The 0.07 +/- 0.04 mm/yr below-2000 m figure could not be confirmed from that text. Unverified, not flagged as a bug.
- The cell 12/13 IPCC comparison uses AR6 'oceandynamics' quantiles relative to 1995-2014 against our samples relative to 2000 (about 4.5 yr of offset). Project memory says no rebasing is needed, so this is not flagged as a bug.
- The Dangendorf comments in cells 3/7 were checked against the Dangendorf et al. (2024) ESSD text in local Zotero: the sterodynamic component comes from tide gauges, altimetry EOFs and climate-model priors, and the paper validates it against 'independent' T/S-based steric height. The 'independent of in-situ T/S' wording is consistent with that. The paper also uses altimetry covariance, which the comment does not mention.

### component_glacier.ipynb

- Cell 10 code comment says corr(b, H0) in level_correlated is '~0.2-0.3'; the module comment above fit_bayesian_level_annual_correlated in bayesian_models.py says '~0.4'. Not verifiable from stored outputs; the two comments disagree.
- Cell 10 code comment gives the rate-space b-c anti-correlation as '~-0.77'; the stored level_correlated output prints corr(b, c) = -0.820. The comment is scoped to rate space, so it may be correct for that path, but it is the only number quoted and the default path is level_correlated.
- Cell 10: halfnorm(scale=(PRIOR_SCALE_B_LEGACY if FIT_SPACE == 'level' else PRIOR_SCALE_B_LEGACY) ...) has identical branches (harmless redundancy).
- Cell 9 panel (b) and cell 17 panel B draw observation error bars as 2*sigma, while cell 9 panel (a) uses Z_90*sigma and the model band is labelled '90% CI'. Plot-convention inconsistency (code).
- Cell 17 code comment 'Rate space avoids the I1/I0 collinearity that plagued the cumulative fit' refers to an earlier IPCC cumulative fit not present in the notebook.
- Cell 18 markdown lists a '### Model selection justification' subheading; there is no dedicated cell for it. The only model-selection output in §6 is the ΔBIC printed by the taper sweep (cell 19), which is affected by the BIC bug above. No edit proposed.
- Markdown cell 0/5 describe the fit window as 2000-2023. The code fits mid-year points 2000.5-2022.5 (n=23), so the calendar-year-2023 interval is excluded (see bug 1).
- Markdown cell 5 points to plan_glacier_ratespace.md and handoff_glacier_ratespace.md, which do not exist in the repo.
- bayesian_models.py module comment (Model 4b) says corr(b, H0) 'drops to ~0.4'. A scratch re-run of the default fit gives corr(b, H0) = +0.30, which matches the notebook's cell-10 comment ('~0.2-0.3'). The module comment is outside this lane.
- slr_data_readers.read_glambie_global docstring says 'Annual DataFrame (2000-2023)'. The file has 24 annual intervals, 2000.0-2024.0.
- Cell 14/15 IPCC overlays: IPCC AR6 values are relative to the 1995-2014 mean, and this study's samples are relative to 2000.0. The baseline difference (a few mm) is not stated in the plotting comments or labels.
- Cell 20: the variable `uncapped` holds samples that were already capped in cell 12. The count is still correct because the cap clamps to exactly V_GLACIER_TOTAL_M and the test is >=. This is a naming issue only.
- Verified without edits (scratch re-run of the fitters, read-only): cell 6 exact T lookup (all mid-years hit T_annual_years); PRIOR_SCALE_A gives Exp mean 0.0434 mm/yr/°C² as printed; b-prior arithmetic (2.23-0.70)/(4-1.5) = 0.61; posterior median b at sigma=2 mm is 0.74; rate-space WLS chi2/dof = 1.02; rate-space corr(b,c) = -0.76 with the legacy prior (cell-10 comment ~-0.77); flat-prior rate b = 0.80 and level_correlated b = 0.74 (cell-1 comment 0.73 vs 0.80); dof_l/dof_q match the fitter's chi2_dof denominator (n - n_phys - 1); apply_sigma_taper is a no-op for TAPER_REF=2000 (t_ref <= years[0]=2000.5), as the cell-19 comment states; the cell-9 level_correlated design (cumsum of T_r, cumsum of ones) matches the fitter's I1/I0.

### component_greenland.ipynb

- Cell 15 code comment: 'ramped linearly ... over a fixed 60-year window ending at ... 1980' contradicts AA_RAMP_START = 1960 / AA_RAMP_END = 1980 and the inline comment '(20-yr ramp)'. The ramp is 20 years.
- Cell 22 code comment: 'In level space, GMST tends to outperform regional T' contradicts the stored output (Greenland T R² = 0.9570 > GMST R² = 0.9031; printed conclusion 'Greenland T provides better level-space fit'). Also cell 22's comment says 'linear rate–temperature model (a=0, b·T + c)', but the a-prior is still Exp(mean 10 mm/yr/°C²) and the posterior mean a = 9.755 / 10.220 is reported; a has no effect only because I2 is zeroed.
- Cell 5 code comment: 'Greenland ablation zone has no in situ weather stations' vs markdown cell 4 'limited in situ weather stations' (markdown wording is the more defensible; comment is out of my lane).
- Cell 5 code comment: GRACE − D 'Provides an observational lower bound on C_T' — the GRACE − D C_T (36 ± 15 Gt/yr/°C local T; 177 ± 65 Gt/yr/°C GMST per greenland_ct_estimate.json) is not used as a bound anywhere in the code.
- Cell 13 comment 'Annualize ocean T (may already exist from cell 7 ...)' — correct; no issue. Cell 13 and cell 10 plot legends use '90% CI', which taxonomy.md lists under terms to avoid ('Confidence interval'); plot labels are code.
- smb_projections.py GREENLAND_SMB comment block says 'Central: ... converted to GMST via AA~2.0 → ~200 Gt/yr/°C GMST', but C_T = -300; comment is stale relative to the value.
- component_analysis.fit_discharge_delay_model docstring model 'H_dyn(t) = gamma * integral(...) + r0 * t' omits the end-of-record rate-constraint equation that is appended to the WLS system (rate_window_yrs, rate_constraint_weight); the markdown (cell 6) also does not mention it, which is an omission rather than an inaccuracy.
- Cell 8 markdown: says cross-correlation 'identifies the optimal time delay δ' and BIC 'confirms the selection', but the stored output has the xcorr peak at 11 yr (r = 0.225) against a BIC-selected δ = 5 yr, so BIC does not confirm the xcorr result.
- Cell 14 markdown: the SMB description (C_T = −300 ± 80, quadratic −50 ± 30) omits SMB_0 = 380 Gt/yr, the AA ramp and the C_T2 cooling mask; the markdown agent may want to check it.
- Cells 5 and 22 print strings use 'C_T²' for the quadratic coefficient C_T2 and 'R²' for level-space fits. These are print strings (code), noted only for the markdown and print consistency review.
- Cell 15 hindcast attrs['description'] string and cell 15 ZERO_CT2_COOLING comment refer to 'conversation notes', which is not a locatable source. The attrs text is code; I left the comment unchanged because it does not misdescribe the code.
- Cell 10 IMBIE-3 comment calls IMBIE-3 'Independent'. Its CSV header says it reconciles 24 Greenland satellite surveys, including input-output estimates, so it may share discharge inputs with Mouginot/Mankoff. I did not verify this; wording left unchanged.
- Cells 5 and 13 describe GRACE−D implied SMB as 'RCM-independent'. The script uses IMBIE (2021) total MB for 1992–2020. Before GRACE (2002), that total likely includes input-output estimates that depend on RCM SMB, so RCM independence may hold only after 2002. I did not verify this against the IMBIE 2021 file; no edit proposed.
- Memory note 'AA only affects discharge, not SMB' (deferred AA=2.23 sensitivity test) holds only from 1980 on. Because _aa_scale = AA(t)/AA, the SMB scale before 1980 depends on AA (1/AA at or before 1960), so changing AA also changes the SMB hindcast (hindcast_gmst_1900).
- Not verified: the ARC discharge values and DOIs in cell 1; cell 10's claim that panel (c) 'replicates the exact construction' of fig3 (checked only against the be_T_oc line in results_figures.ipynb cell 29); and whether the h5 key harmonized/df_en4_greenland_200_500m was built with read_en4_regional's default box (58–80°N, 75–5°W, 200–500 m). The key has no attrs; the comment matches the reader defaults.

### component_eais.ipynb

- bayesian_models.fit_bayesian_level_annual_correlated docstring, prior_b_mean: says 'b >= 0 is enforced as a hard bound regardless of this value'. This is false when symmetric_b=True, as EAIS passes: _level_correlated_log_prior applies the bound only if not symmetric_b.
- Cell 6 code comment: 'b ... uses a symmetric, flat Normal(mean, PRIOR_SCALE_B) prior'. A Normal prior is not flat.
- Cell 6 code comment: 'a sweep from 0.05 to 5 mm/yr/degC moves the posterior median only from -0.003 to -0.014 mm/yr/degC (all well inside the data's own ~0.05-0.09 mm/yr/degC likelihood width)'. The current posterior 90% half-width for b is about 0.13 mm/yr/°C ([-0.145, +0.111]), and the current median is -0.0148. These sweep numbers look stale or unverified.
- Cell 7 code comment: 'main text: raw IMBIE observations are used directly in budget closure for EAIS'. Not verifiable from this notebook.
- Cell 3 code comment: 'Clausius–Clapeyron: ~5%/°C of ~1200 Gt/yr'. Same justification as the smb_projections.EAIS_SMB comment; the ~1200 Gt/yr accumulation figure is not verified here.
- Markdown cells 0, 11 and 18 call the C_T 'RCM-derived' ('RCM constraints', 'RCM-derived C_T and RACMO/literature accumulation sensitivity'). smb_projections.EAIS_SMB justifies C_T = 60 ± 20 by Clausius–Clapeyron scaling ('~5%/°C of ~1200 Gt/yr accumulation') and cites Frieler et al. (2015) and Ligtenberg et al. (2013). The 'RCM-derived' label is plausible via Ligtenberg (RACMO2), but the code does not show it. Not edited.
- Markdown cells 0 and 18 call the diagnostic 'DOLS'. The fit that runs is the Bayesian level-space fit with a correlated likelihood (fit_bayesian_level_annual_correlated). The same label appears in code (the cell 6 print 'skipping DOLS diagnostic fit' and component_io.save_dols_component), so it was not edited; a consistent one-word change is optional.
- Markdown cell 16 (§5, 'IPCC Comparison') describes only the difficulty of extracting EAIS from AR6 AIS. The code cell that follows (cell 17) compares against ISMIP6 ivaf_region_2 (ISMIP6 medians at 2100: SSP5-8.5 +9, SSP3-7.0 -52, SSP1-2.6 -14 mm). This is an omission, not an inaccurate claim, so it was not edited.
- Cell 6 code comment cites 'plan_glacier_ratespace.md at the repo root'. No such file exists in the repo root or in git (neither does handoff_glacier_ratespace.md). This is a dangling reference and I could not determine the correct target, so I proposed no edit.
- Markdown cell 0 and cell 5 (§2) report the full-record linear fit as b = +0.018 mm/yr/°C, c = +0.027 mm/yr, R² = 0.53, and the quadratic as a ≈ +0.019, R² ≈ 0.49, ΔBIC ≈ -4. The stored outputs give linear b = -0.0148 mm/yr/°C, c = 0.0071 mm/yr, R² = 0.2215, and quadratic a = -0.0061 mm/yr/°C², R² = 0.2585, ΔBIC = -3.8. The markdown numbers look like they predate the literature-centred b prior and the correlated-likelihood fit.
- Markdown cell 18 (§6) says the central estimate is 'mass-loss-signed'. The current b posterior median is negative, which is the mass-gain-with-warming sign in the SLR-positive convention. Only c (+0.007 mm/yr) is loss-signed.
- Markdown §5 (cell 16) presents the ISMIP6 comparison with SSP labels. See the high-severity bug on the experiment mapping.
- Cell 6 non-circularity argument (not edited; there is no minimal fix): the comment says citing EAIS_SMB.C_T as the b prior's justification is not circular because it is a different source from glaciers' Rounce et al. (2023) prior. The fitter docstring warns about a different circularity: within EAIS, the same C_T centres the b prior and also drives the SMB projection (cell 12), so the fit cannot independently corroborate C_T. The practical effect is small. At sigma=0.5 mm/yr/degC the analytic posterior mean is -0.017 vs -0.013 under a flat prior. The stated reason does not address this within-EAIS circularity.
- Cell 3 comment ('for DOLS diagnostic and SSP projections') and the cell 6 else-branch print ('skipping DOLS diagnostic fit') call the diagnostic 'DOLS'. The fit is the Bayesian annual correlated level-space model, not calibrate_dols. This looks like project shorthand (component_io.save_dols_component uses the same term), so I proposed no edit.
- Verified, no change needed: GT_TO_M_SLE in config.py = 1/362500, which matches the hardcoded /362500.0 in PRIOR_B_MEAN and the cell-3 print. (read_imbie3's docstring notes that IMBIE-3 itself uses 360 Gt = 1 mm; this does not affect the data, which are read in mm.) The cell-6 claim that downstream consumers rebase also checks out: component_summation cell 10 uses project_component_level_ensemble, which rebases to BASELINE_YEAR so H0 cancels, and results_figures cell 31 rebuilds the annual design anchored at EAIS's first point.

### component_apeninsula.ipynb

- Cell 1 (FIT_SPACE comment block) says 'with it, chi2/dof = 0.99'; stored output gives 0.977 for the adopted linear model (0.996 for quadratic).
- Cell 7 (cell-robust-se-corr) comment says '(chi2/dof = 0.99)'; stored value is 0.977.
- Cell 6 comment says the symmetric-prior 90% CI is '[-0.021, 0.107]'; stored output is [-0.0212, 0.1082].
- Cell 9 comment in Panel B says 'chi2/dof ~ 1.0'; accurate (0.977), no change needed.
- bayesian_models.fit_bayesian_level_annual_correlated docstring for sigma_rate_obs says 'as GlaMBIE reports them' and fit_sigma_extra 'reported GlaMBIE sigmas'; glacier-specific wording in a function also used for IMBIE-3.
- Section 5 markdown (cell 16) discusses only IPCC, while the code cell under it (cell 17) is an ISMIP6 regional comparison; the markdown does not describe that cell. Not an inaccuracy, so no edit proposed.
- Unverifiable from stored outputs (no edit proposed): reduced chi-square 2.64 without sigma_extra; truncated-prior CI [0.007, 0.110]; the 'corr(b, H0) drops sharply' statement (saved posterior gives corr(b, H0) = 0.14 under the corrected fit; the naive-fit value is not stored). 13% of b below zero verified from saved posterior (0.1305).
- Cell 1 code comment also cites plan_glacier_ratespace.md and handoff_glacier_ratespace.md 'at the repo root'; neither file exists.
- Cell 16 claim that Peninsula is evaluated in the summary notebook's budget closure verified: component_summation.ipynb cells 10-11 re-run apeninsula posteriors and sum them.
- Cell 6 comment numbers for the adopted fit ('0.044 ... [-0.021, 0.107]', 'about 13% ... negative') were reproduced exactly by a rerun (0.044 [-0.021, 0.107], P(b<0) = 0.131). The stored output's upper bound 0.1082 differs only by MCMC noise, so no edit was proposed. The truncated-prior numbers ('0.052', '[0.007, 0.110]') reran as 0.050 [0.007, 0.108], also within MCMC noise. No edit proposed. Note that emcee draws from numpy's global RNG, so seed= fixes only the walker initialization and runs are not bit-reproducible.
- Verified without change: 2.64 chi2/dof without sigma_extra (reproduced to 3 digits); PRIOR_SCALE_A Exp mean = 0.0434 mm/yr/degC^2; legacy PRIOR_SCALE_B = 0.087 mm/yr/degC; cumsum(pen_rate_ann) vs pen_rebase year-to-year difference max 0.013 mm (< 0.02 mm claim holds); read_imbie3 sign flip, mm->m and abs(sigma); annualize_imbie last-month sampling and rebase; ISMIP6 region=3 = Peninsula (README); glacier PRIOR_SCALE_B = 2.0/M_TO_MM; glacier corner cell sits in section 3; _best_sample_fit docstring claims (fit_bayesian_level .r2 from posterior mean; HalfCauchy sigma_extra prior).
- Cell 1 and cell 5 (markdown) describe the record as '1979-2023, 45 annual points', which is correct. The printed output reads 1980-2024 because of the print-format bug above.
- The ISMIP6 scenario-mapping bug also affects the markdown in cell 16 and any markdown describing the ISMIP6 comparison as SSP-matched. That is for the markdown auditor.
- Cell 9 Panel B title and cell 6 prints report R^2 (level space). Project memory says to report reduced chi-square rather than R^2 (a code/print choice; not edited).
- Cell 10 comment has a stray extra ')' in 'PRIOR_SCALE_B = 2 mm/yr/°C))'. This is typographic and was not proposed as an edit.

### component_wais.ipynb

- component_projections.py lines 73-87 (A4_SCENARIOS comment): says high_mm=1000 mm chosen to cap the mixture p95 and cites Bamber p95 ~1.19 m 'after rheology correction'; code has high_mm=1300.
- component_projections.py lines 862-865 (blend comment): endpoint described as 'S1-p99-pinned/mixture-p95-capped 94-1000 mm skew-normal'; current values are 84–1300 mm (p50-pinned).
- component_projections.py lines 56, 149-150, 166: 'component_wais.ipynb cell 11' for the quadratic fit; the fit is in cell 9 (cell 11 is the cubic-diagnostic markdown).
- component_projections.py sample_a4_wais docstring (Mode A/B): still says n_draw ~ N(4.1, 0.4²); module constants are now N(3, 0).
- component_projections.py line 333: 'Martin et al. (in press)'; the notebook references cite Martin et al. (2026) AGU Advances 7(2).
- Cell 1 comment: 'The 10%-weighted S1 and S3 scenarios get ~1000 samples each'; there is no S3, and S2 is 90%-weighted. It also says 'Mode B is used for published results' without noting that Mode B is a no-op with n=3.
- Cell 22 comment: calls 1000 mm the 'current baseline' and 1300 mm 'the notebook's previous choice ... before it was lowered'; the current baseline is 1300 mm. It also says 'S1's pinned floor of 130 mm'; the S2 floor is 84 mm. The code's '(current)' legend label is computed from A4_SCENARIOS, so it is correct.
- Cell 25 comment: 'published rheology mode (n ~ N(4.1, 0.4^2))', which is stale.
- Cell 9 comment: 'over a 28-year window near year~2000'; the IMBIE-3 record used is 45 years (1979–2023).
- Cell 17 comment: 'Shift panel (c) -- and its colorbar -- 100 pts to the right', but the code shifts 50/72 in (50 pts). Its header comment says panel (c) spans 1979-2100, but xlim is (2000, 2100).
- Cell 29 comment: 'When N_SAMPLES == N_SAMPLES_DEFAULT (the current default), this is a no-op'; N_SAMPLES=10000 here, so downsampling does occur.
- Cell 28 'Rheology exponent sensitivity' still applies R(n)=1+0.28(n-3) with sigma 0.07 as an explicit sensitivity. That is fine as a sensitivity, but its 'Base' comparison (the range-sensitivity 'Base' line) has no rheology factor, so the n=3.5–4.5 rows are all relative to an uncorrected baseline. Worth a note if the cell is kept.
- Cell 0 in-text characterizations of van den Akker et al. (2025) ('present-day forcing sufficient to deglaciate large WAIS sectors'; 'peak rate found under present-day forcing') and Naughten et al. (2023) ('≈3× historical rates') were not checked against the PDFs. The reference-list entry for van den Akker was corrected to the Zotero record (The Cryosphere 19, 283–301).
- Cell 32 reference table omits Otosaka et al. (2026), which cell 0 cites as the IMBIE-3 source. This is an omission, not an inaccuracy, so no edit was proposed.
- Cell 0 S1 table value [-13, 180] matches sample_a4_wais_endpoint (fresh N=2e6: p5 −12.6, p50 83.6, p95 180.0 mm). The stored trajectory-path output in cell 16 gives −11 mm at p5, because the anchor splice and anchor sigma differ. This is not an error.
- Unverified markdown claim (cell 0 Observations): Could not reproduce these segment values from the IMBIE-3 file with any method tried. Quadratic fits give +0.008 to +0.011 mm/yr² for 1979/1980–2010 (monthly unweighted, annual WLS, annual unweighted) and −0.010 to −0.021 for 2010–2022 depending on the end month. A rate-slope fit gives +0.010 to +0.011 and −0.013 to −0.019. 0.012 matches the full-record Bayesian acceleration (0.0122 mm/yr², cell 9), not a 1980–2010 segment. No notebook cell computes them. The peak rate ≈0.57 mm/yr around 2010–2011 does match: the 2-yr-window rate is 0.566 mm/yr at 2010.5. No markdown edit proposed because the source method is unknown.
- Cell 9 stored output (library print from robust_level_intervals) labels the acceleration slot 'b = 0.0120 mm/yr/degC' and gives halfwidths in 'mm'; for this time-based WAIS fit the units are mm/yr^2 (acceleration) and mm/yr (velocity). This is a print string in component_levelspace_robust_se.py, not in the notebook.
- component_projections.py (not notebook) comments are stale in places: S1 fit window stated as '1979-2023' and 'component_wais.ipynb cell 11' (the fit is in cell 9); the S2 blend comment says the endpoint is the 'S1-p99-pinned/mixture-p95-capped 94-1000 mm skew-normal' (current is 84-1300 mm); the S2 block says 'high_mm=1000 mm' (current 1300).
- Cell 5/17 overlay IPCC AR6 AIS p-boxes and workflow samples (relative to the 1995-2014 mean, ~2005) on WAIS values relative to 2000, under axis labels stating 'relative to 2000'. The baseline offset is small for AIS but is not stated.
- Cell 5/17 use AIS-total constituent workflow files even though WAIS-specific ISMIP6-emulator and LARMIP-2 files exist in dist_components. This is a design choice. The comments now state only that the p-box files are AIS-only.
- Cell 6 comment says S2's 95th percentile 'is pinned ~300 mm higher (to the AR6 low-confidence p95)' than the Bamber MISI-only reference; the reference quantity for '~300 mm higher' is not stated and I could not verify it. Left unedited.
- Cell 20 (markdown) lists a 'Rheology exponent sensitivity' subsection, and cell 30 (markdown) describes the published results as n ~ N(4.1, 0.4^2). Both are stale given n is now held at 3 (for the markdown auditor).
- Cell 1 comment '±11 mm stability at p95 vs ±23 mm at 2k' and cell 17 'FACTS-standard 20,000' bootstrap statement were not verified (would require sampling runs or external files). Left unedited.

### component_summation.ipynb

- cell 3 code comment: 'Rebase all components from storage baseline (2005)'. component_io.py docstring says all projection arrays are stored relative to BASELINE_YEAR (2000).
- cell 7 code comment: 'Truncate to N_SAMPLES (WAIS has 10k, everything else 2k)'. Cell 4 has already replaced WAIS with the 2000-sample wais_2k, so the truncation is a no-op for WAIS.
- cell 10 code comment header: 'compare the total against Frederikse, Dangendorf, and NASA GMSL'. The cell compares only with NASA altimetry.
- cell 10 code comments: 'ocean (hybrid NOAA+IPCC)' (lines 6 and 31). The ocean component is the two-layer model (the hybrid approach is superseded, and cell 11 uses solve_twolayer_ode with the stored ocean attrs).
- cell 10/14 code comments number the diagnostics 'Diagnostic 6' and 'Diagnostic 7'. The notebook has no Diagnostics 1-5 (leftover numbering).
- cell 11 code comment: '6. TWS: Frederikse obs (1900–2018)'. TWS is used only from 2003.
- cell 19 code comment: 'coeffs_abcd, cov_abcd, tau_samples_rs loaded in cell above (from H5_RS)'. They are loaded in cell 14 (Diagnostic 7), not in the cell above (cell 18).
- cell 27 code comment: 'loaded alongside the satellite-era quadratic fit in §4'. That load is in cell 18, which is in §5.
- Markdown cell 0, not edited: 'sum sample-by-sample to preserve correlations within each draw'. This is accurate for temporal correlation within each trajectory (except TWS, see cell-5 bug). Each component is sampled independently (separate notebooks, separate seeds, one deterministic GMST trajectory per SSP), so the sample-index pairing across components carries no cross-component correlation, and the sum is equivalent to assuming independence (consistent with the cell-10 comment 'components are calibrated independently ... combined in quadrature'). Left unchanged because the sentence does not explicitly claim cross-component correlation. The caller may want to clarify it.
- Markdown cell 0, not verified in depth: the EAIS re-inclusion rationale ('following the correlation-aware CI and b>=0 prior fix'). The re-inclusion date 2026-09-13 matches commit 5d65b8f, and exclusion was commit ed12600 (2026-08-26).
- Cell 8 (markdown) says additional diagnostics are 'in the addendum cells below, gated by RUN_ADDENDUM = True'; no such cells or variable exist in the notebook.
- Cell 0 (markdown) describes TWS only as IPCC AR6 landwaterstorage samples; the §3 hindcast/budget uses Frederikse GRACE-era TWS instead (true for projections, incomplete for diagnostics).
- Cell 8 (markdown) 'GMST-forced total hindcast — all 7 components summed': the budget closure total uses EAIS from IMBIE observations and WAIS from IMBIE-anchored stored samples, not GMST forcing.

### component_forecast.ipynb

- Cell 8 code comment: 'H_ORIGIN = nasa_gmsl_rb[-1]  # meters, relative to 2005': the rebase uses BASELINE_YEAR = 2000, so the comment should say 2000.
- Cell 9 code comment: '# Rebase to 2005' precedes a rebase to BASELINE_YEAR (2000).
- Cell 9 code comment: 'Rates: dH/dt = c1 + 2*c2*dt (same for all draws, ...)': rates differ per draw, since each draw has its own coefficients.
- Cell 2 print string / comment use British 'centred'/'centre' (also blend_rate_space docstring 'Centre', 'neighbours'); code, so not edited.
- Cell 18 top comment says '(e) Antarctic Peninsula + EAIS sum'. It is consistent with the code. Only the markdown in cell 17 was stale.
- Cell 0 lists section '7. TODO: residual attribution diagnostic (Option 3)'. No section 7 exists in the notebook. The item is labeled TODO, so it is not inaccurate and was not edited. The headline-statistics, HDF5-export and IPCC-load cells (12-14) and the summary-table cell (21) have no section headers of their own. They fall under sections 3 and 6.
- Cell 0 describes the combination as 'the component-level model sum'. The sum (cell 6) includes TWS, which is the IPCC AR6 landwaterstorage projection (cell 5) and not one of our component models. Not edited, since it is a naming generalization and not a numeric or methodological misstatement. The user may want it noted.
- Verified that fit_satellite_era_quadratic (bayesian_models.py:2151) uses coefficients [c0, c1, c2] with dt = t - t_start and accel = 2*c2. The reconstructions in cells 9 and 20 and the fan-plot accel label are consistent with it.
- Terminology (taxonomy.md 'Terms we avoid'): code comments use 'CI' (e.g. cell 2 'show CI bands', cell 18 '90% CI (p5-p95)') and unqualified 'forecast'. Pervasive and not accuracy errors; no edits proposed.
- Cell 2 print string 'Blending: sigmoid centred at ...' uses UK spelling (code string, not edited).
- Cell 12 comment previously referred to a '4x' within/across ratio; stored output is 3.0x. Any manuscript or markdown statement of a 4x ratio should be rechecked against manuscript_headline_stats.json (not verified here whether the manuscript states 4x).
- blend_rate_space docstring in component_projections.py says 'Centre of sigmoid transition' (UK spelling). Module, out of lane.
- Cell 13 HDF5 attr description for blended_stable says WAIS is 'replaced by S1 (no MISI) scenario'; accurate per component_wais cell 29, noted only because it is a code string.

## Appendix C: Applied markdown/comment edits

### component_ocean.ipynb

- [markdown] cell 2: (1) Frederikse et al. (2020) Methods: in situ steric estimates cover 1957-2018; before 1957 they use the SST/reanalysis-based reconstruction (verified in local Zotero PDF 8FJBMMDB). The notebook's own pre/post split in cell 7 also uses 1957.0. (2) The series is no longer only 'retained for reference': cell 7 plots it as 'Frederikse et al. (2020) (validation)' against the full-depth model and prints withheld-validation RMS.
- [markdown] cell 2: OMISSION (optional; does not overlap edit 1). Cell 3 loads raw/df_dangendorf (1900-2021) and cell 7 plots it as 'Dangendorf et al. (2024) (validation)' against the full-depth model; the data-loading markdown lists every other observational series loaded except this one.
- [comments] cell 3: bl_mask [1995.0, 2006.0] on mid-year stamps selects the 11 annual means 1995.5–2005.5 (calendar years 1995–2005), not a 1995–2006 mean; the comment two blocks down already calls this the 1995–2005 mean.
- [comments] cell 3: Frederikse et al. (2020) use in situ T/S products from 1957 and the SST/reanalysis-based reconstruction before 1957 (verified in local Zotero PDF); cell 7 splits its validation statistics at 1957.
- [comments] cell 5: Li et al. (2022) attribute the ~14% low bias to linear vertical interpolation vs MR-PCHIP, with thermosteric underestimates peaking near 17N and 12S, not Southern Ocean sampling (per memory file project_li2022_thermosteric_bias.md, verified against local Zotero PDF FFYGQ8XS).
- [comments] cell 5: Positivity is enforced only on a, b_u, b_d; c is unconstrained (posterior 90% CI spans zero); sigma_extra is sampled as log_sigma_extra with no Jacobian, so its prior is flat in log (1/sigma_extra), not flat in sigma_extra. The tau_u bound (0.1-500 yr) sits above this block, not below, and does not bound these parameters.
- [comments] cell 6: Cached posterior correlations: b_u-tau_u +0.93, a-c +0.59, b_u-c -0.56, a-b_u only +0.13; c multiplies elapsed time I0, not a temperature regressor. Project rule: tau_u/b_u degeneracy wording must state that tau_u's informative LogNormal(log 8, 0.5) prior breaks it.
- [comments] cell 6: The cell draws a 'median [90% CI] values' text table in the blank upper-right quadrant (fig_corner.text calls below); nothing is written to a caption.
- [comments] cell 6: Median lw = _MED_LW + 1 = 2.5; outer lw = _MED_LW/2 + 1 = 1.75, i.e. 70% of the median's, not half.
- [comments] cell 7: Column 3 now also plots and computes residuals for the Dangendorf et al. (2024) series (_dh, _rdang).
- [comments] cell 7: flat has 64 walkers x 30000 steps = 1,920,000 samples (stored output 'Samples: 1920000'); flat[::100] gives 19,200 draws.
- [comments] cell 9: harmonized/df_berkeley_h has mean 0.000 over 1995-2005 (mean over 1951-1980 is -0.549 C), so the series used here is rebased to 1995-2005.
- [comments] cell 9: noaa_year is 1955.5-2025.5 and noaa_deep_year starts 2005.5 (stored print: 'Full-depth period: 2005-2025'); the below-2000m rate is added to all years (obs_H += RATE_BELOW_2000*... for every year), not only after 2005.
- [comments] cell 9: First 0-2000 m point is 2005.5 (data file and cell 3 comment).
- [comments] cell 9: Mask is y >= noaa_deep_year[0] = 2005.5.
- [comments] cell 11: harmonized/df_berkeley_h is rebased to 1995-2005 (mean 0.000 over that window), not 1951-1980.

### component_glacier.ipynb

- [markdown] cell 5: GlaMBIE reports mass balance only, not temperature. Per the module comment above fit_bayesian_level_annual_correlated (bayesian_models.py ~L2884) and its `temperature` docstring, the annual calendar-year-mean T convention is chosen to match the rate-space fit (fit_bayesian_rate_linear), which uses the same T_r array in cell 6.
- [comments] cell 3: T_annual/T_annual_years are the temperatures entering every fit: cell 6 sets T_r = np.interp(yrs_r, T_annual_years, T_annual), which is passed to all three calibrations (and to the cell-19 taper sweep).
- [comments] cell 6: The legacy HalfNormal(0.5 mm/yr/degC) scale is stored in PRIOR_SCALE_B_LEGACY (used by prior_kw and prior_kw_rate). PRIOR_SCALE_B is the new 2.0 mm/yr/degC scale used only by the 'level_correlated' path.

### component_greenland.ipynb

- [markdown] cell 4: No constraint is imposed anywhere in smb_projections.py or the notebook. GREENLAND_SMB has SMB_0=380 Gt/yr, C_T=-300, C_T2=-50; rate = 380 - 300ΔT - 50ΔT² crosses zero at ΔT = (-300+sqrt(300²+4·50·380))/100 = 1.074 °C (GMST anomaly relative to 1995–2005, the baseline of the Berkeley Earth series used; checked mean over 1995–2005 = -0.017 °C). The ~2.7 °C figure does not follow from the implemented values under either baseline (1.07 + ~0.53 °C to 1951–1980 ≈ 1.6 °C).
- [markdown] cell 6: The surface-to-ocean transfer function (α, β) is also fit to data by OLS (fit_ocean_transfer_function: EN4 200–500 m vs Berkeley Earth Greenland regional T, R² = 0.379); only the discharge model is calibrated against mass-balance observations.
- [markdown] cell 6: δ is not fixed and cross-correlation plays no role in the fit: fit_discharge_delay_model fits each candidate, computes BIC weights, and draws the posterior as a BIC-weighted mixture (delta_posterior); the notebook redraws the same mixture with cov_robust. The cross-correlation peak (11 yr) is outside the candidate set.
- [markdown] cell 6: Stored output of cell 7: 'Cross-correlation peak: delta = 11 yr, r = 0.225' (xcorr lags 0–12 yr in fit_discharge_delay_model). See bug file: the cross-correlation definition itself is suspect, so this number may change if the code is revised.
- [markdown] cell 6: No R² = 0.995 is computed anywhere. The only R² the code reports for the delay model is the rate-space r2_rate (r2_dyn = 0.7937 at δ = 5, stored output of cell 7).
- [markdown] cell 6: Stored output of cell 7: γ posterior median 0.365 [0.314, 0.418] mm/yr/°C (5th–95th percentiles of the correlation-aware posterior; δ = 5 carries BIC weight 0.995). Per-δ point estimates span 0.340–0.465 over δ = 4–8, so 0.36–0.40 (δ = 5–6 point estimates) matches neither.
- [markdown] cell 6: fit_discharge_delay_model drops years where T_ocean(t−δ) is undefined (EN4 annual record starts 1970.5). For δ = 5 the valid Mouginot points are 1975.5–2018.5 (44 of 47); verified by recomputing the valid mask from the reader output.
- [markdown] cell 6: Cell 15 applies the transfer function to T_mon * AA_draws[i], where T_mon is spliced Berkeley Earth + CMIP6 SSP GMST and AA_draws ~ N(AA=3.0, AA_SIGMA=0.5); no SSP Greenland regional temperature product is used.
- [markdown] cell 8: Cross-correlation peaks at 11 yr (r = 0.225), outside the δ candidate set {4,…,8}; BIC selects 5 yr. BIC therefore does not confirm a cross-correlation selection, and the cross-correlation does not select δ in the code.
- [markdown] cell 14: Cell 15 draws alpha_draws, beta_draws (from alpha_se, beta_se) and AA_draws ~ N(3.0, 0.5); residual_std is never used in the projection loop (project_ocean_temperature, which would add it, is imported but not called).
- [markdown] cell 14: Cell 15: total_samples = smb samples + discharge samples, percentiles taken on the sum; no quadrature combination of σ.
- [markdown] cell 19: Section 5 contains only cell 20 (time-series comparison with IPCC AR6). No hindcast is computed or plotted in this section (the 1900–2150 hindcast arrays are only saved to HDF5 in cell 15).
- [comments] cell 1: RATE_WINDOW_YRS and RATE_CONSTRAINT_WEIGHT are passed to fit_discharge_delay_model, which adds the end-of-record rate as a weighted WLS constraint row (component_analysis.py); it changes the fitted gamma/r0, so it is not only a sanity check.
- [comments] cell 5: fit_discharge_delay_model receives only H_dyn/sigma_dyn and ocean T; mou_comp SMB is never passed to it. SMB observations are used in the diagnostics and spliced into the SMB projection (cell 15).
- [comments] cell 5: scripts/grace_minus_discharge_smb.py uses IMBIE (2021) Greenland total mass balance (imbie_greenland_2021_Gt.csv, 1992–2020, GRACE-dominated post-2002); the CSV starts in 1992, before GRACE.
- [comments] cell 9: Bars are result_discharge.bic_weights (the `bics` array is computed but not plotted); the twin-axis curve is fit_results[d]['r2_rate'], the rate-space R².
- [comments] cell 10: Figure is 3 panels (subplots(3,1)); header listed only (a) and (b). Panel (b) plots the r0-detrended cumulative discharge.
- [comments] cell 10: The code reads IMBIE-3 'cumulative_dynamics', which read_imbie3 documents as the dynamics anomaly in the SMB/dynamics partition, not total discharge (hence r0 = 0.293 mm/yr vs Mouginot's 1.383 mm/yr in the stored output).
- [comments] cell 10: The IMBIE-3 CSV header reads: 'reconciled estimates of mass balance from three independent satellite-based techniques: altimetry, gravimetry, and the input-output method', with Greenland coverage from 1971-07.
- [comments] cell 13: Panel (c) plots Mouginot total MB labelled 'cal' (its discharge part is the calibration target) and Mankoff total MB labelled 'val'.
- [comments] cell 15: AA_RAMP_START = 1960.0 and AA_RAMP_END = 1980.0, a 20-yr window (the inline comment on AA_RAMP_START already says '20-yr ramp').
- [comments] cell 15: project_smb_ensemble computes rates = C_T·aa_scale·ΔT + C_T2·aa_scale·ΔT²·ct2_mask + SMB_0 (SMB_0 = 380 Gt/yr for GREENLAND_SMB), then slr_rates = −rates·GT_TO_M_SLE. The comment omitted SMB_0 and wrote the quadratic coefficient as C_T² (reads as C_T squared).
- [comments] cell 22: Stored output: Greenland T R²=0.9570 vs GMST R²=0.9031, '→ Greenland T provides better level-space fit'. The comment asserted that GMST tends to outperform regional T. Values are left out of the comment so it does not go stale on a rerun.

### component_eais.ipynb

- [markdown] cell 0: Stored output of the default FIT_SPACE='level_correlated' linear fit (cell 6, seed 500; also the posterior saved to component_results.h5 eais/posteriors) gives median b = -0.0148 mm/yr/°C, 90% CI [-0.145, +0.111]. b < 0 in the SLR-positive convention means warming reduces SLR, i.e. a mass-gain sign. The quoted +0.018 and [-0.059, +0.095] are from the legacy 'level' path + robust_level_intervals, which did not run.
- [markdown] cell 0: The seed instability described in §2 was found with the legacy fit_bayesian_level path (the legacy branch's own print string carries the same 0.15-0.49 range). Under the default level_correlated path, ΔBIC = -3.8 in both untapered runs with different seeds (cell 6 seed 400; cell 19 f_max=1 row seed 300).
- [markdown] cell 0: Current linear posterior median b = -0.0148 mm/yr/°C (P(b<0) = 0.58 from the saved posterior), which is mass-gain-signed in the SLR-positive convention.
- [markdown] cell 5: Cell 6 stored output: b = -0.0148 mm/yr/°C, c = 0.0071 mm/yr, R² = 0.2215. 5th/95th percentiles of the saved linear posterior (64000 samples, component_results.h5 eais/posteriors/posterior_samples): b [-0.145, +0.111], c [-0.027, +0.041] mm/yr. The level_correlated likelihood uses the correlated cumulative-record covariance, so the raw posterior interval is the correlation-aware one.
- [markdown] cell 5: Cell 6 stored output: Quadratic a = -0.0061 mm/yr/°C², R² = 0.2585, ΔBIC = -3.8.
- [markdown] cell 5: Default-path code (cell 6): bic = result.chi2_dof*dof + k*np.log(n_eais), k = 2/3 (H0 excluded from both); chi2_dof is resid @ Sigma_inv @ resid at the posterior mean against the correlated covariance (fit_bayesian_level_annual_correlated). The 'heteroscedastic log-likelihood' form is _loglik_bic in the legacy 'level' branch.
- [markdown] cell 5: The 0.15-0.49 seed sweep, fixed-seed nondeterminism, and fit_bayesian_level reference describe the legacy path (its print string repeats the range). The default path gives ΔBIC = -3.8 for both untapered runs with different seeds (cell 6 quad seed 400; cell 19 f_max=1 row quad seed 300, R²_quad 0.2585 vs 0.2650); χ²_l ≈ 38.18 vs χ²_q ≈ 38.13, so ΔBIC ≈ -ln(45).
- [markdown] cell 5: Under the default path, the only in-notebook reproduction is the cell 19 f_max=1 row (apply_sigma_taper returns sigma unchanged for f_max<=1; linear seed 400 vs main seed 500): b = -0.0151 vs -0.0148 mm/yr/°C, R²_lin 0.2092 vs 0.2215. Agreement is at 2 significant figures, not 3.
- [markdown] cell 18: Full-record default-path linear b = -0.0148 mm/yr/°C (mass-gain sign). The 1992-2020 windowed value (+0.024 to +0.026) is left as stated; it is not reproducible from code in this notebook (reported in the bug file).
- [markdown] cell 18: Seed sweep (R² 0.15-0.49) and convergence failures were found with fit_bayesian_level (legacy path); check_convergence is never called in this notebook. The default level_correlated path gives ΔBIC = -3.8 in both untapered runs with different seeds.
- [markdown] cell 18: Default-path code computes bic = chi2 + k*ln(n) with chi2 against the correlated covariance (cell 6 and cell 19); component_apeninsula.ipynb cell 6 default path also uses the 'true (correlated) chi-square'.
- [markdown] cell 18: With the default FIT_SPACE the table rows use fit_bayesian_level_annual_correlated; ΔBIC is -3.8 in all three rows. The instability was documented for the legacy path.
- [comments] cell 3: component_glacier.ipynb cell 6 does not explain the calendar-year-mean vs continuous-monthly-integral difference; the explanation (~6 months of phase, shifts b by several percent) is the module comment above fit_bayesian_level_annual_correlated in bayesian_models.py.
- [comments] cell 6: The prior is not flat: it is Normal(PRIOR_B_MEAN, 0.5 mm/yr/degC), described as weakly informative a few lines below. 'Untruncated' is the distinguishing property vs HalfNormal (symmetric_b=True skips the b>=0 bound in _level_correlated_log_prior) and matches the wording in the corner-plot cell.
- [comments] cell 6: component_apeninsula.ipynb cell 6 now states there is no Peninsula-specific literature result establishing a one-signed b response, and uses symmetric_b=True for that reason. Peninsula is therefore no longer an example of a component with a physical reason to expect one-signed sensitivity.
- [comments] cell 6: PRIOR_B_MEAN = -EAIS_SMB.C_T/362500 is in m/yr/degC (as the inline comment on that line states); the fit works in meters. -0.166 mm/yr/degC = -1.66e-4 m/yr/degC.
- [comments] cell 6: The quoted -0.003 to -0.014 range reproduces a ZERO-centred prior (analytic GLS with the same correlated covariance and c/H0 priors: -0.004 at sigma=0.05, -0.013 at sigma=5). With the current prior centred at PRIOR_B_MEAN=-0.166 mm/yr/degC the same sweep gives about -0.12 at sigma=0.05 and -0.013 at sigma=5, so the sweep numbers do not describe the current prior. At the adopted sigma=0.5 the conclusion still holds (analytic -0.017 vs -0.013 flat; MCMC median -0.0148). Alternative: rerun the sweep and replace the numbers. Provenance is reconstructed analytically (Gaussian GLS approximation, not MCMC), not taken from the original sweep run.
- [comments] cell 10: In fit_bayesian_level_annual_correlated, I0 = cumsum(ones) so H(t_0) = b*T_0 + c + H0; H0 is the level before the first (1979) increment, not the level at the first point. Matches cell 6's own description and the function docstring.
- [comments] cell 10: Same as above: H0 is the level before the first annual increment, not at the 1979 point.

### component_apeninsula.ipynb

- [markdown] cell 5: Stored cell-6 output and saved HDF5 attr give linear-model chi2/dof = 0.977 (0.98; matches supplement Table value 0.98). 0.99 is stale.
- [markdown] cell 5: Stored cell-6 output: b = 0.0442 [-0.0212, 0.1082] mm/yr/°C; saved posterior in component_results.h5 gives 5-95% = [-0.0212, 0.1082]. Upper bound rounds to 0.108 (manuscript also quotes 0.108).
- [markdown] cell 16: confidence_output_files/medium_confidence/ssp245/ holds AIS, GIS, glaciers, landwaterstorage, oceandynamics, total only (no PEN), but data/raw/ipcc_ar6/slr/ar6/global/dist_components/ contains icesheets-ipccar6-ismipemuicesheet-*_PEN_globalsl.nc and icesheets-ipccar6-larmipicesheet-*_PEN_globalsl.nc for all SSPs, so the unqualified claim that no separate Peninsula projection is available is inaccurate.
- [markdown] cell 5 **(HELD, not applied)**: Neither plan_glacier_ratespace.md nor handoff_glacier_ratespace.md exists at the repo root or anywhere under global_simple_v1, and neither appears in git history; the derivation lives in the module comment block preceding fit_bayesian_level_annual_correlated in notebooks/bayesian_models.py (the reference cell 6's code comment already uses). Alternative: restore the two .md files if they exist elsewhere.
- [comments] cell 1 **(HELD, not applied)**: plan_glacier_ratespace.md and handoff_glacier_ratespace.md do not exist at the repo root, anywhere in the working tree, or in git history. The derivation is in the 'Model 4b' module comment above fit_bayesian_level_annual_correlated in bayesian_models.py.
- [comments] cell 1: The covariance in fit_bayesian_level_annual_correlated is built only from sigma_rate_obs (cumsum of sigma^2 plus sigma_extra^2). rate_obs (mass_balance_rate) is accepted but not used anywhere in the function.
- [comments] cell 1: Stored cell-6 output for the adopted linear fit reports chi2/dof = 0.977 (quadratic 0.996). A rerun of the same configuration (symmetric b, sigma_extra, seed 300) gave 0.976. 0.99 matches the older truncated-b fit (rerun: 0.987).
- [comments] cell 3: Checked against the raw IMBIE-3 Peninsula CSV via read_imbie3: mass_balance_rate and mass_balance_rate_sigma change mid-year in 6 of 45 years (e.g. 2022: -0.064 -> -0.074 -> -0.096 mm/yr). The code takes the last (December) value.
- [comments] cell 6: Stored output for this fit: chi2/dof = 0.977 (linear). Rerun gave 0.976. The 2.64 value without sigma_extra was reproduced exactly.
- [comments] cell 6: The 'level' branch (elif) is mutually exclusive with this branch and reassigns PRIOR_SCALE_B_LEGACY = calibrate_exponential_prior(0.10, 0.20/M_TO_MM) = 0.087 mm/yr/degC. Cell 7 reads it only under FIT_SPACE == 'level'. The 0.5 mm/yr/degC value is never used.
- [comments] cell 7: Stored cell-6 output: chi2/dof = 0.977 for the linear level_correlated fit.
- [comments] cell 9: The rate does not enter the likelihood. fit_bayesian_level_annual_correlated never uses rate_obs; the residual is H_obs (= pen_rebase, IMBIE-3's cumulative column) minus the model. Only sigma_rate_obs enters, through the cumulative covariance. The panel plots the rate only, with no sigma.
- [comments] cell 10: In fit_bayesian_level_annual_correlated, the modelled level at the first point is H0 + b*T_1979 + c (I1, I0 are cumsums that include the first year), so H0 is the level before the 1979 increment, as cell 6's comment states. The difference (~0.06 mm) is comparable to H0's posterior 90% width (~0.3 mm).
- [comments] cell 12: This block loads only 'projections/temp/Historical' and computes the offset from Berkeley Earth. SSP temperatures are loaded inside the REFIT loop below and in cell 13. The offset is also used by the projections in this cell, not only by plots.
- [comments] cell 13: The ISMIP6 overlay code in this cell is commented out. pen_ismip6 and EXP_GROUPS are consumed by cells 14 and 15. The stray legend entry is reported separately as a bug.
- [comments] cell 13: The ISMIP6 overlay lines are commented out, so no ISMIP6 data is plotted in this figure.

### component_wais.ipynb

- [markdown] cell 0: A4_SCENARIOS['S2_fast_wais']['low_mm'] = 84 in component_projections.py (stored cell 6 output: 'S2 bounds: 84–1300 mm'); 130 mm is not used anywhere in the sampler.
- [markdown] cell 0: A4_SCENARIOS['S2_fast_wais']['low_mm'] = 84 in component_projections.py (stored cell 6 output: 'S2 bounds: 84–1300 mm'); 130 mm is not used anywhere in the sampler.
- [markdown] cell 0: _sample_s1_quadratic_mm() adds eta*sqrt(extra_var(t)) (S1_ISMIP6_STD_COEFFS) to the quadratic draws; the S1 endpoint/trajectory are not draws from the (a, v, H0) posterior alone. The table in this same cell already states this.
- [markdown] cell 0: Code: low_mm=84 = S1 2100 median (A4_SCENARIOS comment: history 94 (p99) -> 180 (p95) -> 139 (p83) -> 84 (p50)); fresh module draw of S1 endpoint (N=2e6, seed 0) gives median 83.6 mm, p95 180.0 mm. The 'just above S1's upper tail' rationale and the 99th->95th history are false under the current rule; the ISMIP6 term is not in §3 (it lives only in component_projections.py), so the '§3' cross-refs are dropped. No new justification added. The false rationale and history are removed; nothing new is added.
- [markdown] cell 0: A4_SCENARIOS['S2_fast_wais']['low_mm'] = 84 in component_projections.py (stored cell 6 output: 'S2 bounds: 84–1300 mm'); 130 mm is not used anywhere in the sampler. Median shift: no stored output shows the alpha 0->4 range at baseline weights; a fresh module run (sample_a4_wais_endpoint, N=400,000, seed 1, alpha in {0,1,2,3,4}, p(S1)=0.10) gives medians 0.294/0.279/0.245/0.221/0.207 m, max-min = 0.087 m (0.099 m at p(S1)=0, the largest over a 0–100% p(S1) sweep). Stored cell 28 output is consistent (alpha=0: 0.30 m, alpha=3: 0.22 m). 0.13 m is stale.
- [markdown] cell 0: sample_a4_wais_trajectories: samples = anchor_draws + max(h2100 - anchor_draws, 0) * t_norm**beta, with anchor_draws ~ N(IMBIE anchor value, sigma). The ramp covers the remaining contribution above the anchor, not H_2100 * R. R (rheology factor) is now identically 1 (N_OBS_MEAN=3.0, N_OBS_SIGMA=0.0, Mode B), and it is already folded into h2100 in code, and there is no rheology section 'below' in this cell.
- [markdown] cell 0: annualize_imbie rebases to BASELINE_YEAR=2000 (stored cell 3 output: 9.0 mm). Cumulative change over the full 1979–2023 record is ≈11.1 mm (raw CSV cumulative at Dec 2023 = 11.09 mm; rebased first annual point = −2.06 mm).
- [markdown] cell 0: Local Zotero record: 'Flow laws for ice constrained by 70 years of laboratory experiments', Nature Geoscience 18, 296-304, doi 10.1038/s41561-025-01661-z; matches cell 32's entry.
- [markdown] cell 0: Local Zotero record title: 'Increasing the Glen–Nye Power-Law Exponent Accelerates Ice-Loss Projections for the Amundsen Sea Embayment, West Antarctica'; matches cell 32 entry.
- [markdown] cell 0: Local Zotero record: 'Present-day mass loss rates are a precursor for West Antarctic Ice Sheet collapse', The Cryosphere 19, 283-301, doi 10.5194/tc-19-283-2025; matches cell 32. (In-text characterizations of this paper in cell 0 were not checked against the PDF and are left unchanged.)
- [markdown] cell 4: A4_SCENARIOS['S2_fast_wais']['low_mm'] = 84 in component_projections.py (stored cell 6 output: 'S2 bounds: 84–1300 mm'); 130 mm is not used anywhere in the sampler.
- [markdown] cell 4: The ISMIP6 extrapolation-error term (S1_ISMIP6_STD_COEFFS) is defined and documented only in component_projections.py; no §3 cell discusses or computes it.
- [markdown] cell 4: A4_SCENARIOS['S2_fast_wais']['low_mm'] = 84 in component_projections.py (stored cell 6 output: 'S2 bounds: 84–1300 mm'); 130 mm is not used anywhere in the sampler. 84 mm is the S1 2100 median per the A4_SCENARIOS comment and a fresh S1 endpoint draw (median 83.6 mm).
- [markdown] cell 4: N_OBS_MEAN=3.0, N_OBS_SIGMA=0.0, N_REF=3: Mode B gives beta = beta_ref*(3+1)/(3+1) = beta_ref (lognormal, median exp(beta_loc)=2.0; stored cell 15 output 'Median beta: 1.93').
- [markdown] cell 4: Cell 0 does not mention an effective exponent; it is computed in sample_a4_wais_trajectories (_beta_eff_at) and returned as params['beta_eff_2035'/'beta_eff_2050'], referenced in the §4 cell 15 comment.
- [markdown] cell 4: component_projections.py lines 337-342: 'Rheology sensitivity deprecated (2026-09-18): n is held constant at N_REF, so Mode B gives R(n) = 1 and beta(n) = beta_ref exactly'. RHEOLOGY_MODE='B' in cell 1; RHEOLOGY_FACTOR_MEDIAN/SIGMA (1.28/0.07) are used only in Mode A. Stored cell 31 output: rheology effect +0.0%.
- [markdown] cell 8: (1) _sample_s1_quadratic_mm adds the S1_ISMIP6_STD_COEFFS term. (2) sample_a4_wais_trajectories draws quad_draws from S1_QUADRATIC_MEAN/S1_QUADRATIC_COV for S2 and feeds their rate to blend_rate_space (sigmoid centered 2035, tau 5 yr), so the fit is used for S2's trajectory; 'It is not used for S2' is false. Cell 0 ('S2 trajectory') already describes this blend.
- [markdown] cell 26: Cell 27 passes rheology_mode='B', but N_OBS_MEAN=3.0/N_OBS_SIGMA=0.0 make R(n)=1; the parenthetical otherwise implies an active correction.
- [markdown] cell 30: Current N_OBS_MEAN=3.0, N_OBS_SIGMA=0.0; published/current results no longer use n ~ N(4.1, 0.4^2). Cell 31 compares n=3 against n=3 (stored output +0.0% on median and p95), so the diagnostic is a no-op; whether to delete or repurpose cell 30/31 is the user's call (logged as a bug). This edit only removes the false claim.
- [markdown] cell 30: Same as above: cell 31 restores _cproj.N_OBS_MEAN/SIGMA to their module values (3.0/0.0), identical to the n=3 case.
- [markdown] cell 32: Literal escape sequence '\u00e9' in the markdown source renders as text, not 'é'.
- [markdown] cell 4: sample_a4_wais_trajectories builds S1 from _sample_s1_quadratic_mm(..., anchor_year=anchor_year), which adds eta*sqrt(extra_var(t)) after the anchor. This is consistent with the matching fixes in cells 0 and 8 and with the Distribution bullet in this cell.
- [markdown] cell 30: With N_OBS_MEAN=3.0 and N_OBS_SIGMA=0.0, no rheology term differs between the two runs (stored output +0.0%). Removing the clause avoids contradicting the corrected sentence above.
- [comments] cell 1: A4_SCENARIOS has only two scenarios (S1_status_quo P=0.10, S2_fast_wais P=0.90); there is no S3. Output shows S1 gets 932 of 10000 samples.
- [comments] cell 1: component_projections.py sets N_OBS_MEAN=3.0, N_OBS_SIGMA=0.0 (rheology deprecated 2026-09-18); Mode B then applies rheo=1 and beta=beta_ref exactly, while Mode A still multiplies by RHEOLOGY_FACTOR_MEDIAN=1.28. Cell 31 output confirms +0.0% effect. The existing comment implies an active correction.
- [comments] cell 5: Only the confidence_output_files p-boxes are AIS-only. AR6 dist_components does include WAIS-specific files (e.g. icesheets-ipccar6-ismipemuicesheet-ssp245_WAIS_globalsl.nc, larmipicesheet ..._WAIS_...), and component_projections.py uses the ISMIP6-emulator WAIS file for S1's extrapolation-error term.
- [comments] cell 9: IMBIE-3 WAIS record used here is 1979.5-2023.5 (45 annual points, 44-yr span; cell 3/9 output n=45). '28-year' matches the superseded IMBIE v2021 1992-2020 record.
- [comments] cell 9: bayesian_models._level_log_prior enforces b>=0 only when symmetric_b=False; fit_bayesian_level has a symmetric_b option. 'always' is inaccurate.
- [comments] cell 9: Read-only recomputation of robust_quadratic_level_fit(wais_year, wais_rebase, wais_sigma, 2000.0) on the current IMBIE-3 file gives m=0.01205 mm/yr^2, c0=0.2303 mm/yr, H0=0.2577 mm, matching the Bayesian fit (0.0122, 0.2302). The same H = 0.5*m*tau^2 + c0*tau + H0 form with free H0 is used (per its docstring); returned keys are m/c0/H0. Old values are from the superseded record.
- [comments] cell 9: Units: a is acceleration (m/yr^2), v velocity (m/yr), H0 level (m); matches component_projections.py S1_QUADRATIC_MEAN comment.
- [comments] cell 17: No cell in the notebook saves component_wais_projection.png (only component_wais_pdf_exceedance_ipcc.png, _observations, _cubic_diagnostic, _pdf_exceedance_ipcc_p_s1_sweep, _ipcc_comparison, _weight_sensitivity, _s2_bound_sensitivity, _median_heatmap_prototype, _scenario_weight_pdf, _alpha_sensitivity), so 'see those cells' points to a nonexistent cell.
- [comments] cell 17: Code plots only PROJ_YEARS >= BASELINE_YEAR (2000) and sets ax_proj.set_xlim(2000, 2100).
- [comments] cell 17: _dx_fig = (50 / 72) / fig.get_figwidth(), i.e. a 50-pt shift.
- [comments] cell 17: AR6 dist_components contains WAIS-specific constituent-workflow files (ISMIP6 emulator, LARMIP-2); only the confidence_output_files p-boxes are AIS-only.
- [comments] cell 21: alpha_values = [0.0, 2.0, baseline_alpha, 4.0, 5.0]: four alternatives to the baseline.
- [comments] cell 21: Read-only check: with alpha in {0,2,3,4,5} and 101 p(S1) values, np.linalg.matrix_rank of the full 16-column bicubic design is 16; with 3 alpha values (e.g. {0,3,4}) it is 12. Five distinct points do not make alpha^3 a linear combination of {1, alpha, alpha^2}.
- [comments] cell 21: Same as above: current sweep has 5 alpha values, so the 'no 4th alpha value' statement only held for the earlier 3-value sweep.
- [comments] cell 21: Stored output of this cell: median fit R^2=0.998 (p95 R^2=0.998).
- [comments] cell 21: p_s1_range = np.linspace(0.0, 1.0, 101) and set_xlim(0, 100): the sweep extends to p(S1)=100%.
- [comments] cell 22: A4_SCENARIOS['S2_fast_wais'] has low_mm=84 (pinned to S1's median) and high_mm=1300; there is no 130 mm floor, and 1000 mm is the previous value ('Was high_mm=1000' in component_projections.py).
- [comments] cell 22: component_projections.py A4_SCENARIOS: high_mm=1300 set 2026-09-18 as rounded AR6 low-confidence AIS p95 (1309 mm); cell's own bound_labels marks 1300 as '(current)' and cell 6 prints 'S2 bounds: 84-1300 mm'.
- [comments] cell 22: Read-only check: full bicubic design with high_mm in {500,1000,1300,1500} and 101 p(S1) values has np.linalg.matrix_rank = 16, not 15. Also removes the reference to alpha's rank deficiency, which does not hold with 5 alpha values (see cell 21 edit).
- [comments] cell 22: Stored output of this cell: median fit R^2=1.000 (p95 R^2=0.999).
- [comments] cell 23: Code uses cmap=sns.color_palette('rocket', as_cmap=True), not magma.
- [comments] cell 24: component_projections.py now sets N_OBS_MEAN=3.0, N_OBS_SIGMA=0.0; the 'current' case restores those same values, so it equals the n3_1090 case. n ~ N(4.1, 0.4^2) is the pre-2026-09-18 setting.
- [comments] cell 25: N_OBS_MEAN=3.0, N_OBS_SIGMA=0.0 in component_projections.py; Mode B with n=3 gives R=1, beta=beta_ref.
- [comments] cell 29: Cell 1 sets N_SAMPLES = 10_000; config N_SAMPLES = 2000; output shows downsampling to (2000, 201) occurred.
- [comments] cell 31: Current module values are N_OBS_MEAN=3.0, N_OBS_SIGMA=0.0; stored output shows identical results (+0.0%).

### component_summation.ipynb

- [markdown] cell 0: §3 has no residual attribution. The residual diagnostic (Diagnostic 7) is rate-and-state minus component sum, and the per-component satellite-era trend table in the Diagnostic 6 cell says it is 'descriptive: share of the component sum's own trend, not a residual attribution'.
- [markdown] cell 8: The §3 hindcast sum is not GMST-forced throughout. Only glacier, EAIS and Peninsula are re-run from posteriors with Berkeley Earth GMST. Greenland uses stored projections whose pre-splice SMB is observed Mouginot/RACMO, WAIS uses the wais_2k scenario ensemble, the budget total uses raw IMBIE observations for EAIS, and TWS uses Frederikse GRACE-era observations. In the 1900-2025 hindcast the sum is also rate-blended with altimetry.
- [markdown] cell 8: Same reason as the 'GMST-forced' change. The Diagnostic 6 cell compares the sum only with NASA altimetry (the xlim is 1990-2025; Frederikse and Dangendorf are not plotted), and the epoch budget-closure table is computed in this cell, not in an addendum.
- [markdown] cell 8: The notebook defines no RUN_ADDENDUM and has no addendum cells. Budget closure is computed in the Diagnostic 6 cell. The other §3 cells not listed above are the 1900-2025 rate-space blended hindcast (plotted against Dangendorf, Frederikse and NASA) and the warming-span cell.
- [comments] cell 3: component_io stores projections relative to BASELINE_YEAR (2000); stored medians are ~0 at 2000 and ~2-8 mm at 2005, so there is no 2005 storage baseline. Only per-sample offsets (<=1.6 mm, ocean/Greenland/WAIS obs noise) are removed. (The HDF5 root attr baseline_year=2005 is stale.)
- [comments] cell 7: Cell 4 replaces all_proj['wais'] samples with the 2000-member wais_2k ensemble; TWS is drawn with N_SAMPLES=2000; no component has 10k here.
- [comments] cell 10: Ocean attrs: model_type='twolayer_noaa', projection_source='Two-layer Geoffroy ODE + IPCC AR6 GMST trajectories' (hybrid NOAA+IPCC superseded). Greenland is also taken from stored projections via _interp_stored('greenland').
- [comments] cell 10: Same as above: stored ocean projections come from the two-layer model, not the superseded hybrid NOAA+IPCC approach.
- [comments] cell 10: eais/observations/years ends at 2023.5 in component_results.h5; eais_obs_yr[-1] drives the switch.
- [comments] cell 10: No interpolation happens at this point; ocean_samp_interp was built earlier in the cell.
- [comments] cell 10: Saved hindcast_diagnostic medians at 2025: Thermo 37.8, Glaciers 18.9, Greenland 17.2, WAIS 9.5, TWS 2.6, Peninsula 1.7, EAIS -0.7 mm; list puts TWS before WAIS.
- [comments] cell 11: blend_rate_space in component_projections.py: np.diff + midpoint averaging now at lines 1561-1567.
- [comments] cell 11: Trapezoidal integration in blend_rate_space is now at lines 1581-1587.
- [comments] cell 11: w_t = 1.0 - expit(...) is at component_projections.py line 1558.
- [comments] cell 11: Stale line reference; see above.
- [comments] cell 11: Stale line reference; see above.
- [comments] cell 11: component_greenland.ipynb pre-EN4 branch: T_ocean_pre_en4 = alpha*(GMST*AA*aa_scale)+beta, explicitly 'GMST x AA (not the real gridded Greenland regional T product)'.
- [comments] cell 11: Code zeros TWS before TWS_START_YEAR=2003 (_grace_era20 mask); only 2003-2018 Frederikse values enter.
- [comments] cell 11: Stored cell-10 output for 1993–2025 now shows Sum 3.21 vs NASA 3.29 mm/yr (3.34 predates the GRACE-era-only TWS change). The 17.1 mm (1950-1990 fall) and +0.31 mm/yr (1993-2018) figures were verified against the harmonized Frederikse data.
- [comments] cell 11: obs_rate_mc has shape (N_MC, n20) from multivariate-normal draws of the quadratic coefficients; each sample gets its own rate.
- [comments] cell 14: Per project_gmsl_state_ensemble, only c is in m/yr; a is m/yr/°C², b and d m/yr/°C. Minimal clarification.
- [comments] cell 14: Covariance entries have mixed units (products of coefficient units), not m/yr.
- [comments] cell 14: _smooth_rate fits a line (polyfit, degree 1) over a centered SMOOTH_WIN-year window; it does not use finite differences.
- [comments] cell 19: They are loaded in cell 14 (Diagnostic 7), not the cell directly above (cell 18, satellite-era quadratic / IPCC).
- [comments] cell 24: Summary table/HDF5: SSP2-4.5 2100 medians EAIS -19 mm, Peninsula +10 mm; net ~-9 mm, so the stack top differs from the Total (mean) line.
- [comments] cell 27: ipcc_total is loaded in cell 18, under the '## 5. Projection Fan Plots' header; §4 is the summary table.

### component_forecast.ipynb

- [markdown] cell 17: Cell 18 builds a 5-panel stack (panel_labels a-e); panel (e) is the Antarctic Peninsula + EAIS sum (comp_order has 4 entries). Markdown listed only b-d and three components.
- [markdown] cell 0: US spelling (style rule).
- [markdown] cell 15: US spelling (style rule).
- [comments] cell 2: US spelling (project style rule). Content is accurate: blend_rate_space uses w = 1 - expit((t - t_center)/tau).
- [comments] cell 5: precompute_ipcc_distributions.py puts 'landwaterstorage' in NORMAL_COMPONENTS and fits it with fit_normal_to_quantiles (stored alpha=0), not a skew-normal.
- [comments] cell 6: component_wais.ipynb cell 29 draws _idx_full_ds and _idx_s1_ds independently and takes scenario_idx from _idx_full_ds, so scenario_idx is index-aligned with wais_2k/full_samples only, not with s1_samples.
- [comments] cell 6: Follows from the edit above. Also, comp_projections[ssp]['WAIS'] holds the 10k-sample all_proj['wais'] array, which is not aligned with scenario_idx; only Total_sum (and forecast derived from it) uses _wais_2k_on_proj.
- [comments] cell 8: nasa_gmsl_rb is rebased to the sample nearest BASELINE_YEAR, which config.py sets to 2000.0.
- [comments] cell 9: The code interpolates each draw at BASELINE_YEAR (2000.0 in config.py), not 2005.
- [comments] cell 9: Each MC draw has its own (c1, c2), so the rate curves differ between draws; only the formula is shared.
- [comments] cell 12: Stored output of this cell gives 'Ratio (within/across): 3.0x' (1115 mm / 370 mm), not 4x.
- [comments] cell 12: From stored output (offset 12.1 mm): uncorrected vs corrected ratio differs by 2.8 (SSP1-2.6), 2.2 (SSP2-4.5), 1.8 (SSP3-7.0), 1.6 (SSP5-8.5) percentage points.
- [comments] cell 12: Only the medians are averaged; p5 is min(abs_los) and p95 is max(abs_his), an envelope across the two SSPs.
- [comments] cell 18: load_component now has an explicit branch for flat posterior datasets (component_io.py, 'e.g. ocean two-layer model: a_posterior, ...'). A read-only call load_component('ocean') in a separate process returned posteriors keys ['a_posterior', 'b_d_posterior', ...] without error, so the stated failure no longer occurs.
