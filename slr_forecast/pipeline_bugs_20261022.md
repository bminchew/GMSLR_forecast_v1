# Bug sweep 2026-10-22 (code changed since e3f9644)

Read-only sweep. 6 units, each reviewed independently by one Fable and one Opus agent (all 12 reported every changed line read), then one adversarial Opus verifier per unit. 34 findings, 0 refuted. Diffs and extracted notebook code: scratchpad bugsweep/.

**Status (2026-10-22).**
- Closed in efab5b3: ocean-pipeline-1 and the thermo-records, cmip6-zostoga and downstream fixes made before that rerun, including greenland-pipeline-2 (the `_gls_ar1` covariance scale).
- Closed after efab5b3 (uncommitted at the time of writing):
  - ocean-pipeline-2: the structural term on 700-2000 m is now referenced to 2005.5, in the notebook, the holdout script and the record-sensitivity script, followed by a rerun.
  - ocean-pipeline-5: the panel is in `fig3_calibration_validation.png` (not Fig. S0, and not in the manuscript). It now shows the fitted 700-2000 m record and the deep-layer curve.
- Closed by the peer session in 7ab8812: greenland-cmip6-1 through -6, greenland-pipeline-3 and greenland-pipeline-4. The pipeline-4 guard leaves outputs unchanged.
- Peer additions:
  - greenland-cmip6-3: the same repeated-chunk fault also hit FIO-ESM-2-0 ssp585 2045-46, GFDL-CM4 historical 1972-73 and ssp245 2037-38, and HadGEM3-GC31-MM historical 1970-71 and ssp585 2032-33. All are re-extracted and clean, and none falls in the baseline, window or junction years.
  - greenland-cmip6-4: with z measured against each model's own 5-yr variability, EC-Earth3-Veg stays below 3. The flagged models are MCM-UA-1-0, INM-CM4-8 ssp126, GISS-E2-1-G ssp245 and EC-Earth3 ssp585.

## ocean-pipeline-1 [high, CONFIRMED, found by both]
`notebooks/component_ocean.ipynb` cell 5 twolayer_log_prob (Su[0]=Sd[0]=T_annual[0], HEAD.py 302-311) vs solve_twolayer_ode calls with the volcanic-shifted target at HEAD.py 458, 826, 1140, 1177, 1436; also component_summation cell 11 (HEAD.py 970)

**Summary.** The likelihood starts both layers at the unshifted T_annual[0]. Every prediction, projection and hindcast passes T + k_v tau_u F_v to solve_twolayer_ode, which by default starts both layers at the shifted T_target[0]. The posterior is therefore applied with a different initial state from the one it was calibrated on.

**Failure.** F_v(1850.5) = +0.244 W m-2, so the projection initial condition is 0.116 C warmer at the posterior medians. The deep layer (tau_d 158 yr) still carries +0.049 C at 2000 and +0.026 C at 2100.5. The 2100 thermosteric level relative to 2000 is about 1.6 mm lower at the medians (the posterior spread reported by opus is -3.4 to -0.5 mm).

**Verification.** I read the fit loop (Su[0]=T_annual[0]; Sd[0]=T_annual[0]) and bayesian_models.solve_twolayer_ode L4724-4725 (su0 = T[0] if Su0 is None). All five ocean call sites and the summation call omit Su0/Sd0. I ran the linear perturbation with python -I from the stored h5 medians (b_u 0.04237, b_d 0.06918, tau_u 12.26, tau_d 157.86, m_v 1.0476) and the ERF reader value F_v[1850.5]=0.244142. Results: delta 0.1159 C, dSd(2000) 0.0487 C, dSd(2100.5) 0.0258 C, dH(2100.5-2000) = -1.59 mm, which reproduces both reviewers. The holdout script uses T_annual[0], so it matches the likelihood but not production.

**Manuscript.** Yes, likely. All SSP 2100 thermosteric medians and intervals shift by about -1.5 mm, and they are quoted to the mm (e.g. 160 [130,198] mm). Also affected: the printed S_d rise by 2100 (-0.02 C), the hindcast, the fit/residual figure panels, the H_mid_fit_mm RMS from manuscript_numbers.py (small), and the component_summation ocean curve.

**Fix.** Pass Su0=Sd0 equal to the first value of the unshifted forcing (T_annual[0], T_k[0], T_proj[0], _Tk[0], ode_T[0]) at every solve_twolayer_ode call that receives the shifted target. Alternatively start the likelihood and holdout at T_target[0] and refit. Then rerun ocean and downstream.

## cmip6-zostoga-4 [medium, CONFIRMED, found by opus]
`scripts/cmip6_zostoga.py` 759-766 (production_mm_per_K_at_median_dT = q3(dHf / dTf))

**Summary.** Every member's 2081-2100 rise is divided by the AR6 median warming, but each member follows its own AR6 GMST percentile path. The bracketed interval therefore mixes warming-path spread with spread in rise per degree.

**Failure.** The supplement prints '76 [61, 93] mm/°C' and Fig. S12d draws that interval as our model's 90% credible interval. Both are compared with CMIP6 ratios that use each model's own dT. With per-member warming the value is about 75 [64, 89] mm/°C.

**Verification.** Rebuilt each member's warming offset with warming_paths.warming_z(2000) and ssp_row_offsets applied to the SSP2-4.5 AR6 table. The 2081-2100 mean offset was added to dTf = 1.838 K, with 2100 held flat. Member order matches: corr(dHf, z) = 0.80. Results: per-member dT is 1.8 [1.4, 2.4] K. The current calculation gives 75.8 [61.5, 93.3] mm/K and the per-member one gives 74.7 [64.5, 89.1] mm/K. The interval narrows by about 7 mm/K in total, less than the reviewer's 'much narrower', and the median moves by 1.

**Manuscript.** Supp. text line 479 '76 [61, 93] mm °C^-1' becomes about '75 [64, 89]'. The Fig. S12d lower square and its interval also change, and the 'scaled by the median AR6 warming' wording in the caption.

**Fix.** Compute dT_i per member with warming_z and ssp_row_offsets (same seed and N_SAMPLES) and report q3(dHf/dT_i). Update the supplement text and Fig. S12d to match.

## greenland-cmip6-1 [medium, CONFIRMED, found by fable]
`scripts/cmip6_greenland_thetao.py` _covers 178-183; used at 195 (esgf_files) and in worker after esgf_files

**Summary.** _covers requires every year 2015-2100 for an SSP run. A run whose archive ends in 2099 (CAMS-CSM1-0, all three SSPs) is rejected, so the committed script cannot reproduce the existing CAMS fragments by the OPeNDAP path. It falls back to a Pangeo whole-field read, or drops the model if that read is skipped.

**Failure.** Re-extracting CAMS-CSM1-0 (or any model whose SSP files end in 209912) rejects a complete OPeNDAP set. The run then takes the Pangeo whole-field fallback or drops out of the CSV, and the CMIP6 percentiles and model count change.

**Verification.** Read the code: hi = YEARS[1] for SSPs, and all(y in yrs for y in range(lo, hi+1)). The worker raises 'do not cover the years' on both passes when _covers is False. In the CSV, CAMS-CSM1-0 ssp126/245/585 cover 2015-2099 (85 years). The CAMS ssp585 fragment JSON lists 5 files ending _208301-209912.nc and has no esgf_errors key, which the current worker always writes. So the fragment came from an earlier version of the script without this check. The README says the Pangeo fallback was not needed for any run. A rerun would therefore follow a different, slower path, or lose CAMS.

**Manuscript.** none (nothing in the .tex cites the comparison; current CSV unchanged unless re-extracted)

**Fix.** For SSP runs, require coverage through min(YEARS[1], last archived year) when that year is at least 2099, or require only the years the comparison needs. Record the actual last year in the fragment JSON.

## greenland-cmip6-3 [medium, CONFIRMED, found by opus]
`scripts/cmip6_greenland_thetao.py` box_means 347-366 / process_ds 381-395 (only screen is |T|>1e10); output cmip6_thetao_greenland_200_500m.csv

**Summary.** The CSV holds corrupt HadGEM3-GC31-MM annual values: ssp126 2025, 2026, 2030 and 2031 and ssp585 2020 and 2021 read about 0.8 degC, against about 4.5-4.8 degC in neighbouring years. Each affected block is the first 24-month chunk of a 5-year file. The extractor has no range or continuity check, so the run was logged as ok.

**Failure.** Anyone who plots or reuses the HadGEM3-GC31-MM CMIP6 series, or moves the window into the 2020s, gets spurious ~3.9 degC drops.

**Verification.** python -I on the CSV reproduces the values: ssp126 2024 4.700, 2025 0.820, 2026 0.823, 2027 4.680, 2029 4.754, 2030 0.821, 2031 0.826, 2032 4.802; ssp585 2019 4.408, 2020 0.782, 2021 0.790, 2022 4.544. The pairs line up with the step=24 chunks at the start of the 202501, 203001 and 202001 files. The only value screen is np.abs(Ti) > 1e10, so a zero-filled or partly filled server return would pass it. These years lie outside the 1995-2005 baseline and the 2081-2100 window, so the printed comparison is unchanged.

**Manuscript.** none now; a CMIP6 time-series figure, or a window that includes 2020-2031, would carry a ~3.9 degC artefact

**Fix.** Delete the affected HadGEM3-GC31-MM fragments and rows, re-extract them, and add a per-chunk sanity check (fraction of exact zeros, or a jump of more than 2 degC from the previous chunk) that triggers a retry on another replica.

## thermo-records-1 [medium, CONFIRMED, found by fable]
`scripts/thermosteric_record_sensitivity.py` _trend, lines 102-111

**Summary.** _trend uses the normal quantile 1.645 and a residual variance on n-2 dof, although the AR(1) correction leaves only neff-2 effective dof. The 90% trend-difference intervals printed into supp line 473 are too narrow.

**Failure.** For 2013-2024 (n=12), the recomputed residual lag-1 autocorrelations are 0.29/0.36/0.28/0.04 for EN4-NOAA 700, IAP-NOAA 700, EN4-NOAA 2000 and IAP-NOAA 2000, giving neff 6.6/5.7/6.7/11.0. The printed half-widths 0.109/0.104/0.115/0.079 mm/yr become 0.149/0.157/0.157/0.089 under Student-t with neff-2 dof (Santer 2008). That is 13-51% wider, and no sign flips.

**Verification.** I read lines 102-111 at HEAD. I recomputed the OLS from the JSON 'series' for the 2013-2024 differences, and the result matches the stored trends_mm_per_yr (0.203 +/- 0.109 for EN4-NOAA 700). With the t(neff-2) and Santer variance, the half-widths are as given above. manuscript_numbers.py lines 424-436 print these brackets, and supp line 473 quotes them.

**Manuscript.** Supp line 473 trend-difference brackets (e.g. EN4-NOAA 0-700 m 2013-2024 [0.09, 0.31] -> about [0.05, 0.35] mm/yr); no figure

**Fix.** Set dof = max(neff-2, 1), se = sqrt(r@r/dof/Sxx), half-width = stats.t.ppf(0.95, dof)*se. Rerun collect only, then reprint. Apply the same fix to the duplicate in cmip6_zostoga.py.

## thermo-records-2 [medium, CONFIRMED, found by both]
`scripts/thermosteric_eos_coefficient.py` lines 81-84, 103

**Summary.** kappa is built from the per-level mean of alpha over wet cells. It omits the ocean-area fraction f(z)=A_wet(z)/A_surface that a global-mean height carries. KAPPA_EOS comes out 0.2469 instead of 0.2650 (7.3% low), and en4_thermosteric.py normalises by surface area, so the two scripts are inconsistent.

**Failure.** The NOAA record that b_u is fitted to is a global-mean height. If warming is uniform over wet cells, then H = S_u * integral of f*alpha_bar*phi dz, so kappa should weight both integrals by f(z). f falls to 0.90 at 700 m. With f, kappa = 0.2650, and a_u rises by 7% (about 0.44 -> 0.48 mm/C^2 at the median b_u). The effect on 2100 is about +0.3 mm, which is below the rounding of the projection numbers.

**Verification.** Code at lines 83-84 divides each level by its own wet area (np.sum(np.where(ok,w,0),axis=(1,2))), with no surface-area normalisation. Line 103 forms kappa from those means. Two reviewers independently reran the script's fields with gsw: they reproduced 0.24689 and got 0.26497 with f(z), and f(700 m)=0.899. en4_thermosteric.py lines 205-206 divide by surf[r], which confirms the inconsistency. The script implements main-text eq. line 169 as written, so this is a derivation-level omission rather than a coding slip.

**Manuscript.** kappa_sw 0.247 -> 0.265 (main text line 169 region, supp table line 183), a_u median; projections change < 1 mm; no figure

**Fix.** Compute f_z = sum_wet(w)(z)/sum_wet_surface(w) and multiply a0_z and ap_z by f_z in the kappa integrals. Then update KAPPA_EOS in config.py, the printed kappa_sw and the a_u median. Alternatively, state the uniform-area slab approximation explicitly.

## cmip6-zostoga-1 [low, CONFIRMED, found by fable]
`scripts/cmip6_zostoga.py` 589 (load_cmip return) and 634-639 (summary dict); consumer scripts/manuscript_numbers.py 466, 483

**Summary.** post2014_models is full minus join_bad, so the count printed as 'with SSP2-4.5 to 2024' and written in the supplement as '24 of which continue under SSP2-4.5 to 2024' is one short. 25 models continue; GISS-E2-1-G is dropped for a discontinuous join.

**Failure.** The CSV has 25 of the 30 analysed models reaching 2024. load_cmip returns `full - join_bad`, which is 24. manuscript_numbers.py prints len(post2014_models) as the number that continue, and supp.tex line 477 says '24 of which continue under SSP2-4.5 to 2024'. Neither the excluded model nor the MRI-ESM2-0 re-reference shift (66.8 mm) is named.

**Verification.** Ran python -I on data/processed/cmip6_zostoga_global.csv plus the summary JSON. Of the 31 models in the CSV, 25 reach at least 2024. The summary has n_models 30, post2014_models 24 and join_flagged ['GISS-E2-1-G']. The only model that continues but is not in post is GISS-E2-1-G. The caption's '24 CMIP6 models' for the band in panel (c) is correct.

**Manuscript.** Yes, the supplement prose count at supp.tex line 477 ('24 of which continue'). The Fig. S12c caption count (24 plotted) is unchanged.

**Fix.** Emit n_with_ssp245_to_2024 = len(full) next to post2014_models, print both in manuscript_numbers.py, and reword the supplement to say 25 continue, of which GISS-E2-1-G is excluded from post-2014 metrics for a discontinuous join.

## cmip6-zostoga-2 [low, PLAUSIBLE, found by fable]
`scripts/cmip6_zostoga.py` 713-718 (window_means) and 754-759 (ssp245_2081_2100_vs_1995_2014)

**Summary.** The per-degree window means use .loc[a:b].mean() with only an isna guard, not win(). A series that ends inside a window, or has a missing year, would be averaged over fewer years without any flag.

**Failure.** Only a latent risk: a model whose ssp245 ends in 2099, or that lacks a year row, would pass the guard and be averaged over 19 years.

**Verification.** Checked every analysed model in the CSV for year gaps, a start after 1955 and NaN tas. None met any of these conditions, so the current outputs are unaffected. The asymmetry with the full-length check in win() is real in the code.

**Manuscript.** none

**Fix.** Route the Z and T window means through win(), or assert the window length.

## cmip6-zostoga-3 [low, CONFIRMED, found by fable]
`scripts/cmip6_zostoga.py` 509, 530-535 (cmd_fetch)

**Summary.** fetch --models writes the output CSV and fetch_provenance.json with only the requested models and merges nothing from the existing files. Re-fetching one model therefore truncates the ensemble that analyze sees.

**Failure.** `fetch --models MRI-ESM2-0` writes a CSV and provenance for one model only. The next analyze reports n_models 1. This would show in the printed counts.

**Verification.** Read the code. `models = sorted(hist_models if not a.models else set(a.models))`, followed by an unconditional `out.to_csv(a.out_csv)` and `meta.write_text(...)`, with no read or merge of the existing files. The current files contain 31 models, so no current output is affected.

**Manuscript.** none

**Fix.** When --models is given, merge into the existing CSV and provenance, or document that --models is for testing only.

## cmip6-zostoga-5 [low, CONFIRMED, found by opus]
`scripts/cmip6_zostoga.py` 402-414, 440-442 (drift_mode fallback / truncated)

**Summary.** For UKESM1-1-LL the drift is fitted over its whole piControl (years 2743-3204, fallback offset). It is not fitted over the parallel years as the supplement states. GISS-E2-1-G-CC is not a real exception, because its parallel segment covers its whole 1850-2014 historical run.

**Failure.** The supplement says the drift is removed 'over the years parallel to 1850-2024' for all models, but UKESM1-1-LL uses the whole-run fallback. Its fitted drift is -0.02 mm/yr, so the numerical effect is negligible.

**Verification.** From provenance: UKESM1-1-LL has drift_mode 'whole piControl', fit years [2743, 3204] and drift -0.0198 mm/yr. GISS-E2-1-G-CC is 'parallel segment (truncated)' with fit years [1850, 2014]. That model is historical-only, so its parallel span is complete and its classification is a label artifact. Neither model is in post2014_models, so neither enters the band or the per-degree values. Dropping both leaves the printed 1955-2012 median (0.376 mm/yr) unchanged. The 5th percentile moves from 0.116 to 0.143, but it is not printed.

**Manuscript.** none numerically. One sentence in the supplement describing the drift removal is inexact for one model.

**Fix.** Record drift_mode counts in the summary. Name UKESM1-1-LL as the one model with a whole-run drift fit in the supplement, or drop it.

## cmip6-zostoga-6 [low, CONFIRMED, found by opus]
`scripts/cmip6_zostoga.py` 830 (ols_dH_on_dT slope_90 = 1.645 * lr.stderr)

**Summary.** The 90% half-width of the cross-model OLS slope uses the normal quantile 1.645. With n = 24 (22 dof) the t quantile is 1.717.

**Failure.** slope_90 is 12.28 mm/K, and with t(22) it is 12.82 mm/K, about 4% wider.

**Verification.** window_means n = 24. stats.t.ppf(0.95, 22) = 1.7171, giving 12.82 versus 12.28. slope_90 is used neither by manuscript_numbers.py nor in the .tex.

**Manuscript.** none (summary JSON only)

**Fix.** Use stats.t.ppf(0.95, len(dTs)-2) * lr.stderr.

## downstream-figures-1 [low, CONFIRMED, found by both]
`notebooks/component_forecast.ipynb` cell 16 (id 2l8u55au7so), extracted HEAD lines 753-757; H_ORIGIN defined at line 251

**Summary.** The total-forecast fan plot still zeroes the NASA record at the smoothed series' value at BASELINE_YEAR, and the comment says this matches H_ORIGIN. H_ORIGIN and sq_level_samples_rb now take their zero from the satellite quadratic at 2000, so the plotted record sits about 1.7 mm below the forecast frame.

**Failure.** In component_forecast_total.png the observed NASA curve is offset by about -1.7 mm from the quadratic and forecast curves. This brings back the SF-17 mismatch that the comment says the code fixes.

**Verification.** I read the HEAD source. Line 251 has H_ORIGIN = _quad(T_ORIGIN) - _quad(BASELINE_YEAR). Lines 282-286 set sq_level_samples_rb to zero at the quadratic's value at 2000. Line 756 has nasa_gmsl_obs_rb = nasa_gmsl - gmsl_smoothed[idx_bl], and the comment at line 753 still reads 'as H_ORIGIN is'. The new comment at line 248 gives the smoothed record as +1.7 mm about the quadratic at 2000.0, so the offset is real. A grep of manuscripts/ found no reference to component_forecast_total, so this is a diagnostic figure only.

**Manuscript.** none (component_forecast_total.png is not in the main or supplement .tex)

**Fix.** Subtract _quad(BASELINE_YEAR) instead of the smoothed value at idx_bl, in the same frame as the NASA record (that is, add smoothed[idx_bl] - _quad(BASELINE_YEAR) to the current rebasing). Rewrite the comment to say what the zero is now.

## downstream-figures-2 [low, CONFIRMED, found by fable]
`notebooks/component_analysis.py` lines 740-747 (_gls_ar1.solve)

**Summary.** _gls_ar1 builds the returned (g, c) covariance from the ML plug-in variance r'r/n, not the unbiased r'r/(n-p). The transfer-function standard errors therefore come out about 2% too small (sqrt(51/49)), and the uncertainty in phi is not carried into them.

**Failure.** With n = 51 and p = 2, alpha_se goes from 0.0587 to 0.0599, and cov_by_tau feeds draw_lowpass_transfer. The quoted xi = 0.30 [0.16, 0.55] and the 2100 Greenland interval do not move at the reported precision.

**Verification.** I read the code. s2 = r @ r / n is used both in the log-likelihood (correct for the profile) and in the returned cov = s2*inv(Xs'Xs). Using the ML variance is a defensible convention, but it is not the usual GLS standard error. The size of the effect follows from the arithmetic (factor 1.0202), and the result is negligible at the precision the paper reports.

**Manuscript.** none at reported precision

**Fix.** Return r@r/(n - X.shape[1]) * inv(Xs.T@Xs) as the covariance and keep the ML s2 in the likelihood, or document that the covariance is the ML plug-in conditional on phi.

## downstream-figures-3 [low, CONFIRMED, found by fable]
`notebooks/bayesian_models.py` lines 2353-2367 (fit_satellite_era_quadratic, covariance-span scaling comment)

**Summary.** The new comment says the S_ij = r^((p_i+p_j)/2) scaling follows from random-walk altimetry errors. For a random walk, Var(c1) does scale as T^-1, but Var(c2) scales as T^-3, not the T^-2 the code applies. The rationale is therefore wrong for c2, though the code's effect is conservative because r < 1.

**Failure.** Someone checking the satellite-quadratic error budget would read the c2 scaling as derived from random-walk errors, when it actually inflates the acceleration variance relative to that assumption (by r^-1 when the fit span exceeds the covariance span). The code itself is unchanged; only the comment is new.

**Verification.** Scaling argument: for a random walk of variance q per unit time, the slope estimate goes like x(T)/T, so Var is about q/T. The curvature estimate goes like x(T)/T^2, so Var is about q/T^3. This matches the reviewer's Monte Carlo SE ratios (c1 0.719, c2 0.359 for a doubled span, compared with 0.707 and 0.354 for T^-1/2 and T^-3/2). The code's exponents give a c2 SE ratio of 0.5.

**Manuscript.** none

**Fix.** Change the comment to call the law an ad hoc interpolation that is conservative for c2 compared with a random walk, or use exponents (0,1,3)/2 if the random-walk assumption is intended and then recheck the satellite-quadratic covariance.

## greenland-cmip6-2 [low, CONFIRMED, found by both]
`scripts/cmip6_greenland_thetao.py` worker: drop_duplicates / cnt==12 filter (~484-487) and years provenance in meta (~512)

**Summary.** Years with fewer than 12 (year, month) rows are dropped without a warning, and the JSON records only [min, max] year. NESM3 loses 2069-2076 in all three SSPs, and the fragment, the log line and the README do not show it. cnt also counts month rows, not finite months, so a year with a NaN monthly box mean is kept as an 11-month mean.

**Failure.** An internal gap or mis-stamped time axis (NESM3 2069-2076) is invisible in the outputs. If the labels are shifted, NESM3's 2081-2100 window mean is computed on the wrong years.

**Verification.** In the CSV, NESM3 ssp126/245/585 each have 78 rows spanning 2015-2100 (8 years missing), while NESM3 historical is complete with 65 rows. The code matches the description: drop_duplicates, then cnt = groupby(year).month.count(), then ann = mean()[cnt == 12], and meta years = [min, max]. The claimed mis-dating of the 2041-2070 file follows from the month counts (312/360/359) but I did not demonstrate it from the time coordinate, so that part is plausible only. The 2081-2100 window is complete for NESM3, so the comparison keeps the model, but its year labels are unverified.

**Manuscript.** none

**Fix.** Write the dropped or partial years and the duplicate-stamp count into the fragment JSON and print them. Count finite months, not rows. Compare decoded months with the span implied by each file name.

## greenland-cmip6-4 [low, CONFIRMED, found by fable]
`scripts/greenland_ocean_cmip6_comparison.py` cmip6_changes 127-146

**Summary.** cmip6_changes baselines each SSP member on the historical member's 1995-2005 mean and never checks continuity at the 2014/2015 junction. EC-Earth3 and EC-Earth3-Veg have junction steps of 0.38-0.81 degC, which pass straight into their 2081-2100 dT and the CMIP6 upper percentiles.

**Failure.** The CMIP6 upper percentiles of the 2081-2100 warming include a junction discontinuity, not only forced warming.

**Verification.** Computed the 2015-2019 mean minus the 2010-2014 mean of thetao_levmean_degC from the CSV. EC-Earth3: ssp126 0.48, ssp245 0.38, ssp585 0.81 degC. EC-Earth3-Veg: 0.57, 0.59, 0.41 degC. For comparison, the SD of the year-to-year differences in the historical record is 0.13 degC (EC-Earth3) and 0.11 degC (EC-Earth3-Veg). The code applies no continuity check. Whether the step comes from the data (branching) or the extraction is not established, but either way it enters dT unflagged.

**Manuscript.** none (comparison not cited in .tex); printed CMIP6 95th percentile and 'above our 95th' fraction would shift

**Fix.** Print each model's junction step in SD units, and flag or exclude members above a threshold such as 3 SD.

## greenland-cmip6-5 [low, CONFIRMED, found by fable]
`scripts/greenland_ocean_cmip6_comparison.py` 190-195 window masks; t_mon grid at 104

**Summary.** Our '2081-2100' window is 2081-2099, and 2099 is a 5-month partial year: the monthly grid from np.arange(1950.5, 2099.5, 1/12) ends at 2099.417. The CMIP6 window is the full 2081-2100. The docstring notes the 2099 end but not that 2099 is partial.

**Failure.** Under SSP5-8.5 our window mean is biased slightly low relative to a true 2081-2100 mean, which can move printed medians in the second decimal.

**Verification.** Line 104 uses np.arange(min, max, 1/12), so the grid stops short of 2099.5. project_greenland_ocean_temperature (notebooks/greenland_discharge.py:228-240) forms annual means as bincount/n_per_yr, so 2099 is averaged over the months present. The reviewer's run showed 5 months in 2099. The docstring (line 22-23) and the printed label mention the 2099 end only. The GMST denominator uses the same months, so the ratio is self-consistent. The bias in dT is of order 0.01-0.02 degC.

**Manuscript.** none (second-decimal effect on printed comparison medians only; pipeline unaffected)

**Fix.** Drop years with n_per_yr < 12 from our window, or match both windows to 2081-2098 or 2081-2099, and say so in the docstring and the printed output.

## greenland-cmip6-6 [low, PLAUSIBLE, found by opus]
`scripts/cmip6_greenland_thetao.py` worker ~481-483 (read_pangeo called with g_holder possibly set from an OPeNDAP file)

**Summary.** When the ESGF passes fail after at least one OPeNDAP file was opened, the Pangeo fallback reuses g_holder['g'] from that netCDF file: dim names, index ranges, level slice and lev-units scaling. A zarr store laid out differently would raise an error or be subset wrongly.

**Failure.** A future run that partly reads OPeNDAP and then falls back to Pangeo on a store with different dims, level order or units would raise a KeyError or silently mis-subset the box.

**Verification.** Code confirms the path: on exception, monthly and log are reset but g_holder is not, and process_ds calls setup_grid only when g_holder.get('g') is None. read_pangeo is called with the same g_holder. This is latent: the README says no extracted run used the Pangeo fallback, so current outputs are unaffected. The effect needs a zarr layout that differs from the netCDF, which I did not demonstrate.

**Manuscript.** none

**Fix.** Reset g_holder = {} before calling read_pangeo.

## greenland-pipeline-1 [low, CONFIRMED, found by both]
`scripts/en4_greenland_profile_counts.py` cmd_figure line 383 (a.bar(..., bottom=0.7)); 387-388 log y-axis

**Summary.** In panel (b), each bar is drawn from 0.7 with height n, so on the log y-axis its top sits at n + 0.7 instead of n. Years with few profiles are drawn too high.

**Failure.** A year with 1 QC-passed profile is drawn reaching 1.7 and a year with 3 reaches 3.7. These are the sparse pre-1970 years the diagnostic is meant to judge.

**Verification.** Line 383 reads a.bar(pos.index, pos, width=0.8, color=ink, bottom=0.7), followed by set_yscale('log') and set_ylim(0.7, None). matplotlib bar draws each bar from bottom to bottom + height, so the top is n + 0.7. Nothing in the pipeline or the .tex uses this figure; it is a diagnostic only. Severity lowered from medium (fable) to low.

**Manuscript.** none (diagnostic figure only)

**Fix.** Draw a.bar(pos.index, pos - 0.7, bottom=0.7, ...) or a.vlines(pos.index, 0.7, pos).

## greenland-pipeline-2 [low, CONFIRMED, found by fable]
`notebooks/component_analysis.py` _gls_ar1, lines 744-747 (s2 = r @ r / n; cov = s2 * inv(Xs'Xs)); used by fit_ocean_lowpass_transfer cov_by_tau and draw_lowpass_transfer

**Summary.** The (g, c) covariance uses the maximum-likelihood scale r'r/n, not r'r/(n-2), and treats the profiled phi as known. The transfer-function standard errors and the per-tau normal draws are therefore about 2% too narrow.

**Failure.** With n = 51 and 2 coefficients, the SE of g and c is low by a factor of sqrt(49/51) = 0.980. g = 0.273 +/- 0.059 would read about +/- 0.060. The (g, c) draws, and so the discharge spread, are slightly narrow. Uncertainty in phi is not propagated either.

**Verification.** In _gls_ar1, solve() computes s2 = r@r/n and returns s2*inv(Xs.T@Xs). fit_ocean_lowpass_transfer stores this per tau as cov_by_tau and alpha_se. Per its docstring, draw_lowpass_transfer draws (g, c) from that tau's GLS normal. The size of the effect follows from the stored n = 51. The ML scale is a legitimate choice for the profile likelihood behind the tau weights; only its reuse for the coefficient covariance is the issue.

**Manuscript.** At most the last digit of a quoted g uncertainty; Greenland 2100 interval widths change by well under 1 mm. No median changes.

**Fix.** Use r@r/(n - X.shape[1]) for the returned covariance and keep r'r/n in the log-likelihood. Optionally say that phi uncertainty is not propagated.

## greenland-pipeline-3 [low, CONFIRMED, found by both]
`scripts/greenland_sensitivity_forecast.py` line 3 (module docstring)

**Summary.** The docstring points to 'component_greenland §6 (cell 26)', but the §6 sensitivity cell is now at 0-based index 25, and index 26 is an empty cell.

**Failure.** A reader following the pointer lands on the empty trailing cell. This is cosmetic.

**Verification.** The current notebook has 27 cells. The §6 sensitivity source is at index 25 and cells[26] is empty. At 908cac6, when the docstring was written, §6 was at 0-based index 26 of 28 cells, so the reference was 0-based and correct then. It was already at index 25 before ef757e4.

**Manuscript.** none

**Fix.** Refer to '§6 sensitivity cell' instead of a cell number, or change it to cell 25.

## greenland-pipeline-4 [low, PLAUSIBLE, found by opus]
`notebooks/greenland_discharge.py` project_greenland_ocean_temperature, lines 199-247 (en4_years/en4_ann, is_en4, fut, noise masks)

**Summary.** Projection noise is added to every year at or after floor(ssp_start), including any year that also has an EN4 value. The EN4 annual means also include partial years and use .mean() rather than nanmean. The function relies on the caller to pass full-year EN4 that ends before the branch year.

**Failure.** If EN4 reached the branch year, or a caller passed EN4 without full_years_only, AR(1) noise would be added on top of an observed value, and a partial-year mean with seasonal bias would stand in for the whole year. The new unit test cannot catch this: it uses zero noise, and its EN4 year is the branch year.

**Verification.** Traced the code. fut = unique_yrs >= floor(ssp_start) is not intersected with ~is_en4, and en4_ann has no 12-month filter. In production, component_greenland line 110 applies full_years_only to the EN4 record, which then ends in 2020. scripts/greenland_ocean_cmip6_comparison.py also applies full_years_only. The branch is in 2025, so is_en4 and fut do not overlap and the defect is not triggered.

**Manuscript.** none (latent; not triggered by production inputs)

**Fix.** Use the mask fut & ~is_en4 for projection noise, or assert that the two do not overlap. Apply the 12-month filter, or assert it, inside the function.

## greenland-pipeline-5 [low, CONFIRMED, found by opus]
`scripts/en4_greenland_profile_counts.py` module docstring line 65 ('figure' subcommand)

**Summary.** The docstring says 'Three-panel diagnostic figure', but cmd_figure draws four panels, (a) to (d).

**Failure.** The docstring does not match the output. This is cosmetic.

**Verification.** Line 65 says 'Three-panel diagnostic figure.' and line 369 creates plt.subplots(4, 1, ...).

**Manuscript.** none

**Fix.** Change the docstring to 'Four-panel'.

## ocean-pipeline-2 [low, CONFIRMED, found by both]
`notebooks/component_ocean.ipynb` cell 5 sigma_700_2000_total (HEAD.py 214-218); scripts/ocean_early_holdout.py lines 120-122

**Summary.** The 5% structural term on the 700-2000 m record is taken on |0-2000 m minus 0-700 m|. That difference has an arbitrary zero (2005.5 reference minus the 1995-2005-mean reference), so the term is largest at 2005 and near zero around 2015. This contradicts the comment that it 'vanishes at each record's reference'.

**Failure.** The structural term is 0.28 mm at 2005.5, about 0 near 2015 and 0.08 mm at 2024.5. Referenced to 2005.5 it would run from 0 to 0.36 mm. Formal errors are 0.3-1.2 mm and sigma_extra_2 is about 0.5 mm, so the effect on the weighting is small.

**Verification.** Code read: noaa_700_2000_corrected = noaa_2000_corrected - noaa_700_corrected[_k700], with the structural term on np.abs() of it. The 0-700 m reference is the 1995-2005 mean and the 0-2000 m reference is 2005.5, per the comment in the same cell. Both reviewers independently got the same values (-5.62 to +1.61 mm).

**Manuscript.** none at stated precision (posterior changes negligibly)

**Fix.** Use RESIDUAL_STRUCTURAL*np.abs(noaa_700_2000_corrected - noaa_700_2000_corrected[0]) in the notebook and the holdout script, and update the comment.

## ocean-pipeline-3 [low, CONFIRMED, found by both]
`notebooks/component_ocean.ipynb` cell 7 full-depth _sf construction (HEAD.py 874-880) and the 'Residual diagnostics quoted in the manuscript' loop (HEAD.py 999-1004); cell 9 obs_sigma (HEAD.py 1079-1086)

**Summary.** For t >= 2005.5 the full-depth series _of reduces to noaa_2000 + n7(2005.5) + below-2000 term. Its sigma, however, combines sigma_700_total with sigma_700_2000_total, and the latter already contains se_700, so the 0-700 m error is counted twice. Cell 9's saved obs_sigma uses sigma_2000_total instead, so the figure and the stored observations disagree.

**Failure.** The panel (c)/(f) error bars are too wide by a few percent. The printed full-depth chi2/n is biased low. The comment 'same construction as the projections' is false.

**Verification.** Traced the algebra: _deep_inc = noaa_2000 - (n7 - n7[0]), so _of = noaa_2000 + n7[0] + below on deep years. _sf adds sigma_700_total^2 + sigma_700_2000_total^2, and sigma_700_2000_total = sqrt(se_deep^2 + se_700^2 + struct^2). Cell 9 lines 1079-1086 use sigma_2000_total. manuscript_numbers.py reads only the 700-2000 m H_mid_fit_mm RMS, not the full-depth chi2.

**Manuscript.** Only if the full-depth chi2/n printed in the notebook is quoted in the tex (I did not find it in manuscript_numbers.py). Figure error bars change slightly.

**Fix.** Define the full-depth sigma once as sqrt(sigma_2000_total^2 + se_700(2005.5)^2 + below^2) on deep years, and use it in cells 7 and 9.

## ocean-pipeline-4 [low, CONFIRMED, found by both]
`notebooks/component_ocean.ipynb` cell 7 corner plot, except NameError branch (HEAD.py 598-604) and _TAU_D_PRIOR (HEAD.py 630)

**Summary.** The documented standalone path of the corner cell (reload posteriors after a kernel restart) raises NameError, because TAU_D_LOG_SIGMA and TAU_D_LOG_MU are defined only in the fit cell.

**Failure.** Restart the kernel and run the setup cells plus the corner cell: the except branch loads the posteriors, then lognorm(s=TAU_D_LOG_SIGMA, ...) raises NameError. The in-order pipeline run is unaffected.

**Verification.** grep shows TAU_D_LOG_MU/SIGMA assigned only at HEAD.py 172-173. The except branch imports only THERMO_VOLC_M_SD, and line 630 is outside the try/except.

**Manuscript.** none

**Fix.** Move the Geoffroy C_0/gamma arrays and TAU_D_LOG_MU/SIGMA into slr_forecast.config and import them in the fallback branch.

## ocean-pipeline-5 [low, PLAUSIBLE, found by fable]
`notebooks/results_figures.ipynb` cell 32 (HEAD.py ~3700-3719)

**Summary.** The Fig. S0 thermosteric panel still plots NOAA 0-2000 m against H_total_fit_mm. The fit now calibrates on the 700-2000 m difference record, so the figure no longer shows the calibration data as described.

**Failure.** The panel renders without error, but it shows the 0-2000 m total rather than the fitted 700-2000 m record. This is a mismatch between the caption or Methods and the figure, not a wrong number.

**Verification.** Confirmed that the keys noaa_2000_corrected_mm and H_total_fit_mm are read and the panel is labeled 'NOAA 0-2000 m'. The comparison itself is still a valid model-data comparison, so whether it is a defect depends on the caption wording, which I did not check.

**Manuscript.** Fig. S0 content/caption only; no number

**Fix.** Plot noaa_700_2000_corrected_mm against H_mid_fit_mm, or say in the caption that the 0-2000 m total is shown for reference.

## ocean-pipeline-6 [low, PLAUSIBLE, found by fable]
`scripts/manuscript_numbers.py` lines 388-396, 426-427 (TRS_JSON, CZ_JSON loads)

**Summary.** The production-side values in the record-sensitivity and CMIP6-zostoga blocks come from JSON snapshots, and nothing checks them against the current component_results.h5. After a pipeline rerun without rerunning those scripts, they go stale without warning.

**Failure.** Rerun component_ocean (for example the N=10000 run, or the fix for ocean-pipeline-1) without rerunning thermosteric_record_sensitivity.py and cmip6_zostoga.py. manuscript_numbers then prints old production values next to fresh h5 values.

**Verification.** The reviewer checked that the JSON production b_u (42.371 mm/C) equals the current h5 median, so the values are consistent now. The defect is latent, and fixing ocean-pipeline-1 would trigger it.

**Manuscript.** none currently; latent after rerun

**Fix.** Assert that the JSON production parameters match the h5 posteriors, or compute the production side from the h5.

## ocean-pipeline-7 [low, CONFIRMED, found by opus]
`src/slr_forecast/readers/forcing.py` read_volcanic_erf, lines 78-80

**Summary.** out = erf.reindex(years).values; out[after] = 0.0 writes in place to an array that is read-only under Copy-on-Write, which is the default in pandas 3.

**Failure.** After a pandas 3 upgrade, every projection call (years after 2024.5) raises 'assignment destination is read-only'.

**Verification.** The environment has pandas 2.3.2, so the code works today. With pd.options.mode.copy_on_write=True, the same reindex().values pattern raised 'assignment destination is read-only' (python -I).

**Manuscript.** none

**Fix.** out = erf.reindex(years).to_numpy(dtype=float, copy=True)

## ocean-pipeline-8 [low, CONFIRMED, found by opus]
`scripts/ocean_early_holdout.py` docstring lines 18-28; TAU_D_FIXED line ~39/78

**Summary.** The docstring says the likelihood and priors are copied from production and that the full-record fit is 'like for like', but the script fixes tau_d = 150 yr while production samples it. The script's full-record b_d, rho and sigma_extra_2 are therefore not the production posterior.

**Failure.** Quoting the script's full-record deep-layer parameters as production values would be wrong. The withheld 0-700 m metrics are unaffected.

**Verification.** The docstring contains 'copied from the production cell' and 'parameter comparison is like for like', and the script sets TAU_D_FIXED = 150.0. The stored production tau_d is 72/158/346 yr (5/50/95%).

**Manuscript.** none unless holdout deep-layer parameters are quoted

**Fix.** Qualify the docstring, or sample tau_d with the production prior.

## ocean-pipeline-9 [low, CONFIRMED, found by opus]
`tests/test_ocean.py` module docstring line 4; test_metadata_fixed_constants docstring (~line 184)

**Summary.** The module docstring still says 'five-parameter posterior ... fixed tau_d'. The new docstring says the posterior median 'stays near' the 133 yr prior median, but the stored median is 158 yr. There are no tests for read_volcanic_erf or for a = KAPPA_EOS*b_u^2.

**Failure.** The docstrings mislead readers. The 100-175 yr bound leaves little margin.

**Verification.** Read lines 1-8 and 184-185 of the test file. The stored tau_d median of 157.9 yr was reported by both reviewers' h5 checks and is consistent with fit_diagnostic tau_d_med = 157.86.

**Manuscript.** none

**Fix.** Update the docstrings. Add tests for read_volcanic_erf and for the KAPPA_EOS relation.

## thermo-records-3 [low, CONFIRMED, found by fable]
`scripts/thermosteric_record_sensitivity.py` cmd_all, lines 519-525

**Summary.** any(p.wait() for p in procs) stops at the first non-zero exit. The remaining fit subprocesses are never waited on or killed, and they keep writing their JSONs after the parent exits.

**Failure.** If one fit fails early, the parent raises SystemExit while the other five fits keep running for about 30 min. An immediate rerun of 'all' then races those processes on the same output files. No current number is affected.

**Verification.** Read lines 519-525. The any() over the generator short-circuits as described.

**Manuscript.** none

**Fix.** codes = [p.wait() for p in procs]; if any(codes): raise SystemExit(f'fits failed: {codes}')

## thermo-records-4 [low, CONFIRMED, found by fable]
`scripts/thermosteric_eos_coefficient.py` line 90

**Summary.** The f-string expression on line 90 contains a backslash escape (alpha\'), which is a SyntaxError on Python 3.11 and earlier. It works only on 3.12+ (the environment is 3.13.7).

**Failure.** Running the script under Python 3.11 or earlier fails at compile time. Results are unaffected, because KAPPA_EOS is already copied into config.py.

**Verification.** Line 90 matches the evidence, and python3 --version is 3.13.7.

**Manuscript.** none

**Fix.** Hoist the header string into a variable, or avoid the escaped quote inside the expression.

## thermo-records-5 [low, CONFIRMED, found by opus]
`scripts/en4_thermosteric.py` cmd_series lines 222-224; _year lines 211-214; cmd_annual line 247

**Summary.** The resume step skips any year whose npz exists, even when that npz came from an incomplete archive or the ctanom npy was never written. cmd_annual then silently drops years with n_months != 12.

**Failure.** Suppose a year is first processed from a partial (for example 9-month) archive. It is never reprocessed when the full archive arrives, and the annual series silently omits it. A kill between the savez and the np.save leaves the ctanom npy missing permanently. The current CSV has a complete 2025, so no present number is affected.

**Verification.** Read lines 211-214 (npz is written before npy), 222-224 (skip on npz existence only) and 247 (silent continue).

**Manuscript.** none

**Fix.** Skip a year only when its stored n_months equals the archive member count and the npy exists. Print the years dropped in annual.
