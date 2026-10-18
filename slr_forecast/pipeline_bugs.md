# Pipeline bug report (consolidated, 2026-10-17)

This file replaces `notebook_audit_bugs_2026-09-24.md`. Every item from that report has been re-checked against the current code and appears below as FIXED, STILL OPEN or NOT APPLICABLE (in the "Old-report items" section of each component part).

## How this was produced

- **Pipeline run.** A full `run_pipeline.sh` run finished cleanly on 2026-10-16 (22 of 22 steps, 6804 s). All 468 tests passed, and `component_results.h5` was unchanged by the tests.
- **Pass 1: review.** 36 read-only reviewers covered the work. Each notebook got two reviewers, one on code and data flow and one on units, baselines, time stamps and interval conventions. The rest covered the analysis modules, the data readers, the pipeline scripts, HDF5 ownership and pipeline order, duplicated constants, every number in the main text and the supplement, and the test suite.
- **Pass 2: verification.** Eight verifiers, one per area, merged the duplicates. Each one then tried to refute every finding against the code, the stored outputs, the pipeline log and the HDF5 (read-only), re-deriving numbers where feasible. Verdicts are CONFIRMED (reproduced), PLAUSIBLE (mechanism visible in the code, impact not demonstrated) or REFUTED.
- **Totals.** 212 merged entries across the eight parts, some of them duplicates across parts (cross-referenced in Part A). There are 4 distinct high-severity defects in 5 entries, because OC-1 and SF-1 are the same defect (AN-2 was withdrawn on 2026-10-17; the whole-AIS comparison is intended). About 45 entries are medium and the rest low. About ten pass-1 claims or sub-claims were refuted or corrected; each part lists them.
- **Edits.** No code or manuscript file was edited.
- **Fix status (updated 2026-10-17, after the report).** Fixed, rerun from Greenland onward, and the manuscript updated: OC-1/SF-1 (deep-ocean term before 1950), AN-7 and FT-3 (each WAIS member held at its first IMBIE-3 level, end of 1979, back to 1900 instead of zero-filled; l.359 wording), AN-5 (EAIS hindcast uses C_T in both hindcast cells), GR-2 (observed Greenland regional T outside EN4; EN4 full years only; Fig. 2c/3 extension on regional T), GR-5 (Greenland-only cell mask, scripts/make_greenland_mask.py; AA sensitivity endpoints now taken from the computed window range), SF-8 (l.373 says satellite-era rates are the unblended component sum). IN-1 = GR-3 (cell 26 now reports only the Greenland change and saves each member's change to greenland/sensitivity; the forecast changes quoted at l.433 come from scripts/greenland_sensitivity_forecast.py, step `grsens` after `slowwais`, which checks that `blended` is newer than greenland/sensitivity and has the same member count). GR-1 (Greenland discharge refit on annual increments with independent rate errors, equivalent to GLS on the cumulative record; common error scale chi2/dof in the covariance and in a profile-likelihood BIC; rate-constraint row dropped and the end-of-record rate reported as a check; sandwich covariance no longer used. Result: 98.0% weight on delta = 6 yr, gamma 0.338 [0.299, 0.376], chi2/dof 1.45; Mankoff, IMBIE-3 and the early holdout refit the same way; tex l.114/120/122/126/146/373/377/396/417/429/433/435 and supp updated). l.359 WAIS: rate set to zero through 1979. Group 1 (10k-run blockers, 2026-10-17): GR-4 (discharge ocean forcing held flat after it ends), AN-8 (slow-WAIS mixture rows from branch_row_index at any branch size; N_OUT stays 20000), MD-7 (N_SAMPLES from slr_forecast.config in every component notebook, summation and the SMB supp script; WAIS max(10000, N_SAMPLES)), IN-27 (precompute_ipcc_distributions.py is pipeline step ipccdist before forecast, sized from config; regenerates identically), MD-6 (fit_bayesian_level_annual_correlated reports autocorrelation time, R-hat and ESS over all walkers and warns below 50 tau / 5 tau burn-in; glacier cell 20 chains lengthened to 6000), IN-12 (derived groups stamped with source_saved_at; manuscript_numbers, results_figures and grsens call check_derived). Manuscript numbers unchanged. Groups 2-4 (2026-10-18, full rerun, tex number pass; uncommitted at time of writing): OC-2 (NOAA points after the forcing record dropped; fits 1955-2024 / 2005-2024), OC-3 (holdout stores random rows), OC-4 (supp holdout requoted from the pipeline run), AN-3 (imbie_annual_rates: RMS of monthly sigmas, used by WAIS, EAIS, Peninsula; test reproduces IMBIE cumulative sigma), AN-4, SF-4 (satellite quadratic fits annual and semi-annual harmonics), SF-9 (summation uses full 11-yr windows; abstract wording pending author), GL-1 (glacier_delayed_response: GlaMBIE 1-sigma plus production sigma_extra), GL-2 (sigma_extra, T_eq0 saved and printed), GR-6 (sigma_eps grid from 0), OC-11 moot (sigma_extra median now 1.970), corner contours at 1/2/3 sigma (OC-7, GL-3, MD-5, AN-9 incl. captions and p/e labels), OC-5, OC-6 (caption), OC-9, SF-7 (shared WARMING_C), FT-2 (mean split + caption), SF-5/FT-20, GR-7, AN-10, FT-4, MD-8, MD-4/FT-25 (figure consumers use temperature_unc / 1.96; stored Berkeley sigma still stale, MD-24 open), SF-10. Group 3 decisions (decisions_group3.md): SF-2 one AR6 offset, ar6_baseline_offset_m = 13.3 mm, everywhere; OC-8 one AR6-to-Berkeley GMST offset (ar6_gmst.py, 0.755 C, mid-year stamps), EAIS on observed GMST through 2024; AN-6 a_p = 0 kept with physical justification (scripts/peninsula_curvature_check.py, pipeline step pencurv); GL-4/MD-1 chi2 dof counts sigma_extra and BIC uses the full likelihood. AN-2 withdrawn (intended). AN-1 accepted as is (2026-10-17, author's decision: the inflated S1 spread stays). Low-severity sweep (2026-10-18; code only, no tex edits; 503 tests pass; full rerun started): MD-24/MD-4 (slr_processed_data.h5 stale keys regenerated by scripts/refresh_processed_keys.py, which refuses to write if any non-sigma column changes: Berkeley sigma = unc/1.96, IMBIE WAIS and Dangendorf gia_sigma and Horwath steric_deep_sigma made positive, IPCC observed sigma = half 17-83% range / 0.9542 (MD-15); guard tests in test_pipeline_guards; NASA gmsl_sigma documented as an along-track spatial SD, not renamed); two new 10k-run blockers fixed (smb_state_space_diagnostic.py had N = 2000 hardcoded; ipccdist moved before summation); smblag step added (joint_lag_fit_mar.py). Ocean: OC-12 (comment), OC-13 (attr), OC-14 (rebase at 2000.0), OC-15 (one 11-value baseline window), OC-16 to OC-22, IN-26 (NOAA read from raw files). Glacier: GL-5, GL-8, GL-9 (gain-year print), GL-10 (quadratic stationarity table), GL-12 to GL-18, GL-20, GL-21; GL-19 docstring. Greenland: GR-8, GR-15, GR-17 (SMB driver read at Y+0.5; Greenland 2100 +0.1 to +1.2 mm), GR-18 to GR-23, GR-25 to GR-28 (code strings), GR-30, GR-31 (mm print), IN-8. Antarctica: AN-11 to AN-13 (notebook parts), AN-15 (comment), AN-16, AN-17, AN-18, AN-20 (prints incl. b_p, c_p in manuscript_numbers), AN-21 (summation Peninsula hindcast anchored), AN-22 to AN-24, AN-26, AN-27; year-merging groupby(round(0)) on Y.5 stamps fixed in four notebooks. Summation/forecast: SF-11 (D_GRID from 0; VoI 2040-2050 only), SF-13, SF-14 (print), SF-15 (both conventions printed), SF-16 (Ablain fallback raises; GIA in cov_params, MD-18), SF-17, SF-18, SF-19 (verified), SF-20 (incl. IN-19), SF-21, SF-22. Shared/infra: MD-11, MD-12/IN-13, MD-13, MD-14/IN-5, MD-15, MD-17 (comment only), MD-19, MD-20 (optional anchor), MD-21, MD-22 (docstring only), MD-23/IN-3, MD-25, MD-26, MD-28 (warning), MD-29 (docstring), IN-4, IN-14 to IN-18, IN-20 to IN-22 (rate-and-state), IN-23 (s1 JSON), IN-6 to IN-11 (tests). Figures: FT-8 to FT-11 (code), FT-21 (prints), FT-22, FT-23, FT-26 (partial), FT-27, GR-24, OC-17. Still open, needing the author: every tex proposal (none applied), and the decisions OC-12, OC-13, OC-18, GL-9/GL-10 wording, GL-13, GR-12, GR-16/MD-22 (EN4 depth weighting), GR-22, GR-27, GR-28, GR-29, GR-31 units, AN-14/SF-12 (p(S1) realized at 0.0975), AN-19, AN-25, FT-8 step, FT-14/AN-13/MD-27 (360 vs 362.5 Gt/mm), FT-26, MD-17, SF-15, SF-18 Shapley, IN-16. Procedures only: IN-24 (h5repack after the 10k run), IN-28 (archive orphaned files). FT-1/IN-2: rebuild the PDFs after the tex pass.

Severity scale:

- **High:** changes a number, interval or claim in the manuscript or supplement materially, or causes a crash or silent corruption on the production path.
- **Medium:** changes a manuscript number slightly; a manuscript figure, caption or method description disagrees with the code; or a production-path defect that will bite (e.g. the planned N_SAMPLES = 10000 run).
- **Low:** diagnostics, stale comments, labels or prints, dead code, notebook-only figures.

References to `$S/...` (verification scripts, `nb_src` extracts) point to the sweep's temporary session directory and will not persist. Cell numbers are cell indices from those extracts (all cells, markdown included, counted from 0).

Line numbers refer to `manuscripts/00_ddpi_slrforecast2026/v1/01_slr_forecast_intervention2026.tex` (main) and `..._supp.tex` (supp) as of commit d0bddcd.

---

## Part A. Summary for decision-making

### A1. High-severity findings

| ID | What | Manuscript effect |
|---|---|---|
| **OC-1 = SF-1** | The pre-1950 hindcast leaves out the 0.117 mm/yr below-2000 m ocean term (summation cell 11). The stored samples from 1950 onward include it. | Main l.373: 1900–1950 rate 0.76 → about 0.88 mm/yr. 1900 gap 39–45 → about 33–40 mm. "Outside the Dangendorf 90% interval only in 1906 and 1908" no longer holds. Combined with the related items in A2, see below. |
| **AN-1** | The S1 extrapolation-error term is fitted to an ISMIP6 "std" computed as `nanstd` over 107 unevenly spaced quantile values, which inflates it by 25–30%. | Main l.307/309: S1 86 [−10, 182] mm → about 86 [9, 164] mm. Slow-WAIS tails tighten by about 3–13 mm. Slow-WAIS P(>1 m) at 3.6 °C goes from about 30% to about 29%. The supp exceedance table moves slightly. Fast-WAIS numbers are essentially unchanged. |
| **GR-1** | The Greenland δ BIC weights use a diagonal likelihood on the cumulative discharge record. With the correlated covariance, the ranking reverses: δ = 6 is preferred (weight ≥ 0.93). | Main l.120: "98.8% of the weight on δ = 5 yr", the γ interval, Fig. S1 and the δ labels in Fig. 2. The 2100 Greenland median moves by 2 mm or less. |
| **IN-1** | component_greenland cell 26 (step 5) reads `blended`, `blended_stable` and `blended_slow90`, which steps 14 and 15 write. | Main l.433 ("3 mm or less", "more than 2 mm") was computed against the previous run's ensembles. This **will crash the planned 10k run** at step 5, and would also crash a run on a fresh HDF5. |

### A2. The 1900–1950 hindcast sentence (main l.373) has four independent defects

All four move the same sentence in the same direction (higher 1900–1950 rate, smaller 1900 gap):

1. **OC-1/SF-1:** missing below-2000 m term. +0.117 mm/yr; 1900 level −5.9 mm.
2. **AN-7:** WAIS set to zero before 1950, a 2 mm step at 1950. 1900 level −2 mm.
3. **AN-5:** the EAIS hindcast uses the diagnostic regression (b = −0.043, c = +0.006), not the C_T model the forecast uses. With C_T: about +0.09 mm/yr; 1900 level about −6.8 mm.
4. **GR-2:** before 1970 the Greenland ocean forcing uses AA·GMST, although observed Greenland regional T exists. About +0.05 mm/yr; 1900 level about −3.7 mm.

Taken together, and rough because they were not re-run jointly, the 1900–1950 rate becomes about 1.0 mm/yr instead of 0.76, and the 1900 gap becomes about 21–27 mm instead of 39–45. Items 3 and 4 involve a modelling choice (see A6). The sentences at l.359 and l.361 also misdescribe the hindcast (FT-3, SF-8).

### A3. Other manuscript numbers that change (medium)

| ID | Number | Change |
|---|---|---|
| SF-2 | AR6 baseline offset ("lowers each value by 12 mm", l.429) | Four definitions are in use: 12.1, 10.7, −11.1 (wrong sign) and 0 mm. Computed consistently it is about 13.2–13.3 mm. That makes l.429 "13 mm", moves the 2.7 °C slow-WAIS 95th percentile from 0.82 to 0.81 m, and leaves the other quoted values unchanged. |
| OC-8 = SF-3 | AR6→Berkeley GMST offset | Ocean uses 0.755 °C; glacier, Greenland, EAIS and Peninsula use 0.726 °C (rate-and-state uses 0.89). With 0.726 °C everywhere, thermosteric at 2100 rises 4–5 mm (l.394). l.282's claim that every member follows one GMST path is not exact. |
| OC-2 | Thermosteric posteriors (l.196) | The 2025.5 NOAA points are fitted against the model at 2024.5 because the forcing ends 2024-12. Estimates move within their intervals. The holdout script inherits this. |
| OC-3, OC-4 | Supp holdout paragraph (l.379), Fig. S10 | The stride-96 thinning stores walkers 0 and 32 only (introduced 2026-10-16). The supp interval 15.0 [12.0, 17.6] should be 15.0 [12.2, 17.5] from the pipeline run. |
| AN-3 | EAIS b_e (l.255), skill table | IMBIE annual σ should be the RMS of the monthly σ. EAIS and Peninsula use the December value; WAIS uses the mean. b_e −0.043 → about −0.03; χ²/dof 0.94 → about 0.90. |
| AN-4 | EAIS range (l.298) | "−14 to −25 mm" → −13 to −24 mm. l.400 already says −13.1 to −24.1. |
| SF-4 = MD-3 | Satellite quadratic (l.427) | It is fitted on NASA GMSL with the seasonal cycle still in. 0.63 [0.38, 0.86] → about 0.63 [0.40, 0.86] m. |
| SF-9 | "about 1.1 °C (11-yr means)" (abstract, l.377, l.466) | The end value is a 6-yr trailing mean. Full windows give 1.00–1.07 °C. |
| GL-1 | Glacier delayed-response claims (l.433) | The script uses GlaMBIE 95% half-widths as 1σ and has no σ_extra. Corrected: "about 30 mm" holds, and "12 to 14 mm" → "11 to 13 mm". |
| GR-6 | Table S2 σ_ε 50 [25, 80] Gt/yr | The grid starts at 20. Extended toward 0, the interval becomes 50 [10, 80]. |
| GL-4 = MD-2 | Skill-table χ²/dof (supp l.371, l.400) | The denominator leaves out σ_extra. Counting it gives glacier 1.10, Peninsula 1.00, WAIS 1.01, so "0.94 to 1.05" becomes 0.94–1.10. Alternatively, state the convention. |
| AN-6 = MD-1 | Supp l.206 "a_p set to zero because BIC favors the linear model" | The BIC drops ln|Σ| and the quadratic is bounded at a ≥ 0, so ΔBIC = −ln n by construction. The unbounded maximum-likelihood ΔBIC is +4.7 in favor of a decelerating quadratic. |
| GL-2 = MD-9 | σ_gl,extra = 0.18 [0.13, 0.26] (l.233) | The value is correct but is neither saved nor printed by any pipeline output. |
| OC-11 | σ_extra "2.0 [1.7, 2.3] mm" (l.196) | The median is 1.9499, so 1.9. |

### A4. Manuscript captions and method descriptions that disagree with the code (medium)

- **OC-5:** the residual band in Fig. ocean_fit_residuals is ±1σ, but the caption says 90%.
- **OC-6:** Fig. ocean_fit_residuals panel (c) shows a full-depth composite, not "0–2000 m as in panel b".
- **OC-7 / GL-3 / AN-9 / MD-5:** all four supp corner captions say the contours are at 1, 2, 3σ. corner draws 0.5, 1, 1.5, 2σ.
- **AN-9:** in the Peninsula caption σ_extra is the "fourth" parameter, but it is third. The EAIS caption calls the prior "informative", but it is weak.
- **OC-9:** the Fig. S10 error bars leave out σ_extra, but the caption says "total point error".
- **SF-7:** the per_component_forecast legend says 2/3/4 °C, not 1.8/2.7/3.6 °C.
- **FT-2:** in Fig. 4(a), the Greenland line does not separate the discharge and SMB anomalies as the caption says (about 15 mm of committed imbalance is shown as SMB).
- **FT-3 / AN-7:** l.359 says the hindcast WAIS "follows its fit". The code uses the IMBIE-3 record, as l.344 correctly says.
- **SF-8:** the satellite-era hindcast rates (2.98–3.25) are the unblended component sum, while l.361 defines the hindcast as the blend.
- **SF-10:** l.282 leaves out the division by 1.645 in the warming-path spread. The supplement is correct.
- **SF-5 / FT-20:** the VoI caption says "present value", but the values are discounted to the arrival year. The supplement says so; the quoted 2026 values are unaffected.
- **GR-7:** Fig. 2 mixes 2σ error bars (panel b) with 1.645σ (panel c), and the caption says neither.
- **AN-10:** Fig. wais_p_s1_sweep shows the AR6 curves on the 1995–2014 baseline under a "relative to 2000" label. The LARMIP-2 curve is the dynamics+SMB total.
- **FT-4:** Table S2 defines T relative to 1850–1900, but every fit uses 1995–2005.
- **MD-8:** l.375 says "30-yr local regression", but 30 yr is the kernel half-width (60-yr support).
- **GR-5 / MD-10:** the "Greenland regional" temperature box includes Baffin Island, Iceland and Ellesmere (about 11–16% of the weight). It feeds ξ, χ and the AA range 2.1–2.7. The effect has not been computed.
- **MD-4 / FT-25:** the Berkeley Earth σ stored in `slr_processed_data.h5` predates the 95%→1σ fix (it is 1.19× too large). It feeds the GMST bands in Fig. 1(a) and in the per-component forecast figure.
- **FT-1 = IN-2:** both compiled PDFs predate this run's figures.

### A5. Production-path risks to fix before the N_SAMPLES = 10000 final run

- **IN-1:** cell 26 crashes, because the forecast groups are read before they are written.
- **AN-8 = SF-6:** `build_slow_wais_p90.py` stops on its `rem == 0` assert at 10k, or silently thins S2.
- **MD-7:** N_SAMPLES is a literal in each component notebook, not imported from config.
- **IN-27:** `ipcc_distributions.h5` is not produced by the pipeline and holds 2000 members. It fails loudly at 10k.
- **MD-6:** the fitter behind glacier, Peninsula, EAIS and WAIS runs no convergence diagnostic. `check_convergence` uses 4 of 64 walkers.
- **GR-4:** Greenland discharge ocean forcing is zero-filled after about 2103 instead of held flat. Stored values for 2104–2150 are wrong; 2100 is unaffected.
- **IN-12:** derived groups (`blended_slow90`, `wais_slow90`, `voi`, …) carry no provenance stamp, so a partial rerun can leave them stale without notice.

### A6. Decisions for the user (these are judgement calls, not mechanical fixes)

1. **EAIS in the hindcast (AN-5):** drive it with the C_T model the forecast uses (as l.359 implies), or keep the diagnostic regression and say so.
2. **Greenland δ (GR-1):** refit the δ weights with the correlated likelihood, which moves to δ = 6 and changes γ and Fig. S1, or keep the weights and rewrite l.120.
3. **S1 spread (AN-1):** fixing it changes the S1 and slow-WAIS numbers and the exceedance table, and needs a WAIS and downstream rerun.
4. **Peninsula curvature (AN-6):** the data prefer a decelerating quadratic once a may be negative. Justify a_p = 0 on other grounds, or revisit it.
5. **χ²/dof convention (GL-4/MD-2):** count σ_extra (range 0.94–1.10), or state that it is excluded.
6. **Greenland temperature box (GR-5):** mask to Greenland only and recompute ξ, χ and AA, or describe the box.
7. **AR6 offset (SF-2):** choose one definition. The recommendation is a NASA-anchored value computed once in a shared function.

### A7. Note on the Frederikse comparison (main l.373)

Frederikse et al. (2020, Methods) add the Greenland peripheral glaciers to their Greenland term and set the Antarctic periphery to zero. This study counts both peripheries in the glacier component (GlaMBIE includes RGI 5 and 19). Part of "Greenland raised it by 10 mm more than in our hindcast" is therefore this definitional difference (GL-7), and the glacier comparison is biased the opposite way. The total is unaffected.

---

## Part B. Verified findings by area

Each part below is the verifier's report as written. IDs: OC ocean, GL glacier, GR Greenland, AN Antarctic (EAIS, Peninsula, WAIS), SF summation/forecast/downstream scripts, MD shared modules/readers/IO/constants, FT figures and manuscript text, IN tests/HDF5/rate-and-state. Each part ends with its old-report items (where assigned) and cross-area notes. Items listed in Part A are merged duplicates of these entries.

---

## B1. Ocean / thermosteric (OC)

Verification scripts and outputs: `$S/verify_OC/v1.py`–`v8.py` (read-only; HDF5 opened 'r').

#### OC-1 Pre-1950 hindcast leaves out the below-2000 m term (component_summation)
- Verdict: CONFIRMED
- Severity: high
- Location: component_summation_code.py cell 11 l.966–990 (`eta_oc = oc_a * Su**2 + oc_bu * Su + oc_bd * Sd`); producer component_ocean cell 9 l.1066–1069
- Reported by: nb_summation_A, nb_summation_B, nb_ocean_A, nb_ocean_B, num_main_3
- Manuscript impact: main tex l.373: "the hindcast rate is 0.76 mm yr$^{-1}$ ... over 1900--1950" → about 0.88 (+0.117 exactly, since the rate is an OLS slope of the median and the missing term is linear). "39--45 mm above the reconstructions" (cell 30 output `Frederikse +45.4`, `Dangendorf +39.1`) → about 33–40 mm (−5.85 mm: 45.4→39.6, 39.1→33.3 = 0.117 × 50 yr). "only in 1906 and 1908" may also change (not rerun). Also affects Fig. observations (a) and component_hindcast_vs_observed.png before 1950.
- Defect: From 1950 the hindcast uses the stored ocean samples, which include `rate_deep_k*(t-2000)` (0.117 mm/yr). Before 1950 it uses the posterior-median two-layer shape without that term, offset to each member at 1950. So the thermosteric slope drops by 0.117 mm/yr before 1950, and the 1900 level is about 5.9 mm too high. The ocean notebook's own 1900–2025 validation (cell 7 `_predict`, `Hf_ = Ht_ + rate_deep*(Ty_use - BASELINE_YEAR)`) and supp l.61 ("added over the full record") both carry the term back to 1900. This is a coding inconsistency, not a design choice. The tex's "qualitative only" framing does not change that the quoted numbers are wrong.
- Evidence: I read cell 11 l.979–989. The deep term is absent, and the attr `rate_below_2000_mm_yr = 0.117` is saved in `ocean` but never read there. Summation outputs l.114–115 `1900–1950 Frederikse ... 1.39 0.76`; l.370–371 `+45.4`, `+39.1`.
- Fix: Add `hf['ocean'].attrs['rate_below_2000_mm_yr']/M_TO_MM*(ode_t - BASELINE_YEAR)` to `eta_oc` before rebasing (better: integrate each member back to 1900 with its own deep-rate draw). Then rerun summation and figures and update l.373.

#### OC-2 NOAA 2025.5 points are fitted against the model at 2024.5
- Verdict: CONFIRMED
- Severity: medium
- Location: component_ocean cell 5 l.239–244 (`obs_idx_700`, `obs_idx_2000` by argmin); cell 7 l.776–777, 797 (`_r7`, `_r2`, `_rf`); cell 5 end-to-end rate print l.474–479; scripts/ocean_early_holdout.py l.102–103
- Reported by: nb_ocean_A, num_main_1, scripts_comp
- Manuscript impact: main tex l.191 fit windows "(1955--2025)", "(2005--2025)"; l.196 posteriors (a_u 18.8, b_u 51.8, b_d 69.0 …), RMS 2.3/1.3 mm, chi2/n 1.13/0.37; supp holdout numbers. Shift is small. In a fixed-τ_u WLS proxy, a_u goes 18.4 → 16.8 mm/°C² with 2025 forcing spliced in, or → 17.8 if the 2025.5 points are dropped; b_u moves by about 0.4–0.9 and b_d by up to 1.2 mm/°C. All stay well inside the quoted 90% intervals.
- Defect: Berkeley Earth `harmonized/df_berkeley_h` ends 2024-12, so `T_annual_years[-1] = 2024.5`. Both NOAA records end 2025.5, and argmin maps that point to the 2024.5 model value. The model then gives the 2025.5 observation a value 2.26 mm (0–700 m) and 2.70 mm (0–2000 m) lower than the forcing implies, which is larger than σ_extra (1.95 mm). The cell 7 residual panels inherit this. So does the holdout, where the point has double weight in a 36-point fit and enters the reference mean.
- Evidence: v1.py: `BE last 2024-12-01`; `Ty end 2024.5 y7 ... 2025.5`; `last map 700 [2023.5 2024.5 2024.5]`; `Hu(2025.5)-Hu(2024.5) 2.257 mm, Ht 2.705 mm` (with cell 7's spliced 2025 T = 0.657 °C). v4.py WLS proxy `[18.36 51.9 69.25]` vs `[16.82 50.96 68.02]` (extended) vs `[17.78 51.54 69.04]` (drop).
- Fix: Drop NOAA points past the forcing record, or fit with the cell 7 spliced 2025 forcing. Apply the same change in the holdout script, then rerun.

#### OC-3 Holdout posterior stored with stride 96 keeps only walkers 0 and 32
- Verdict: CONFIRMED
- Severity: medium
- Location: scripts/ocean_early_holdout.py l.344 (`thin = r['flat'][::max(1, r['flat'].shape[0] // 20000)]`); used by `plot_figure` l.389–396 and `--from-posterior` l.448–454
- Reported by: scripts_comp
- Manuscript impact: Fig. S10 (fig:ocean_early_holdout) bands are always drawn from this subsample. Supp l.379 "15.0 [12.0, 17.6] mm" comes from this subsample (see OC-4).
- Defect: emcee's flat chain is step-major (row = step·64 + walker). With 1,920,000 rows the stride is 96, which keeps only walker 0 (every 3rd step) and walker 32. The stored posterior has an effective sample size of a few hundred per walker, compared with ≥8500 for the full chain. The upper bound of the withheld-level interval moves by about 0.3 mm (several times the redraw noise).
- Evidence: v2.py: `walkers kept [0 32]`; `full posterior_flat[:,0] == ocean_posteriors/a_posterior[::96]` → True. v3.py: integrated autocorrelation of each kept walker's thinned series (pooled over parameters, so not per-parameter values) is 32–53 rows, giving ESS of order 200–300 per walker. Full chain (HDF5 `level2000_model_mm`) −15.04 [−17.54, −12.24]; stored rows −14.98 [−17.64, −11.95] (with the script's seed) and −14.97 [−17.62, −11.97] (all 20000 rows).
- Fix: Thin by steps across all walkers (`sampler.get_chain(thin=k, flat=True)`) or by a random row choice. Rerun and requote.

#### OC-4 Supplement holdout level interval does not match the pipeline run
- Verdict: CONFIRMED
- Severity: medium
- Location: supp tex l.379; scripts/ocean_early_holdout.py l.223–235, 315; pipeline_full.log l.38–39
- Reported by: nb_ocean_A, scripts_comp, num_supp_2
- Manuscript impact: supp l.379 "15.0 [12.0, 17.6]~mm below the 1995--2005 mean" → 15.0 [12.2, 17.5] mm (pipeline run and HDF5). All other numbers in the paragraph match the log (b_u 52.9 [46.9, 69.0], a_u 9.1 [1.1, 23.6], 0.24 [0.17, 0.29], 0.20 [0.16, 0.24], 12.0 [11.4, 12.6]).
- Defect: The tex interval comes from the `--from-posterior` path (the 2-walker subsample of OC-3). The pipeline run uses the full chain. The two paths cannot agree because they draw from different posteriors.
- Evidence: HDF5 `ocean_early_holdout/holdout/level2000_model_mm` median −15.04, 5/95 −17.54/−12.24. My reproduction of the `--from-posterior` path gives `[-14.98 -17.64 -11.95]`, which is exactly the tex interval.
- Fix: After fixing OC-3, quote the pipeline-run values, and have manuscript_numbers.py (or the script log) print them.

#### OC-5 Fig. ocean_fit_residuals (d–f): the gray band is ±1σ, but the caption says 90% credible interval
- Verdict: CONFIRMED
- Severity: medium
- Location: component_ocean cell 7 l.774–775, 790, 872 (`_a1.fill_between(_yy, -_tt, _tt, ...)`); main tex l.204
- Reported by: nb_ocean_B, num_main_2
- Manuscript impact: main tex l.204 "with the gray band showing the 90\% credible interval on the residual". The band is ±1σ (68%) of the observational total error, not a credible interval.
- Defect: `_tt = sqrt(σ_obs² + σ_extra²)` for (d, e), and `_sf` for (f). No Z_90 factor appears anywhere in the cell. The residual error bars are ±1σ as well: σ_obs only in (d, e), σ_extra included in (f). The band's legend label is commented out (l.873).
- Evidence: Code lines quoted above. NOAA tsl_se is a 1σ standard error.
- Fix: Multiply by Z_90 and keep "90%" (and call it observational error, not credible), or change the caption to "±1σ total point error". Also say what the error bars are.

#### OC-6 Fig. ocean_fit_residuals panel (c): caption and legend misdescribe the plotted points
- Verdict: CONFIRMED
- Severity: medium
- Location: component_ocean cell 7 l.780–797 (`_of`, `_fd`), l.845 legend; main tex l.204
- Reported by: nb_ocean_B (also nb_ocean_A, as the "(calibration)" legend)
- Manuscript impact: main tex l.204 "(c) ... Also shown for reference are the 0--2000\,m observations (circles, same as in panel b)". The circles are a full-depth composite, labelled "NOAA full depth (calibration)" under a "Full depth (validation)" title.
- Defect: The panel (c) points are the 0–2000 m record + 0.117 mm/yr·(t−2000), re-zeroed on the 1995.5–2004.5 mean of a 0–700 m-only series. They are not the panel (b) points. Because the baseline window has no 700–2000 m layer, the points after 2005.5 lack the model's 700–2000 m rise from the baseline to 2005.5. That rise is 1.53 mm. Other residuals partly cancel it: the net mean panel (f) residual over 2005.5–2025.5 is +0.27 mm (RMS 1.29 mm, matching the cell 7 output), so the omission is a construction inconsistency, not a visible offset.
- Evidence: v6.py: `700-2000 model layer change baseline(1995.5-2004.5)->2005.5 = 1.53 mm`; v8.py (rebuilt from `ocean/fit_diagnostic`): `mean residual panel f (2005.5+) 0.271 mm, RMS 1.292`.
- Fix: Describe the composite in the caption and legend. Anchor the 700–2000 m increment with the model's b_d·ΔS_d from the baseline to 2005.5, or zero the composite at 2005.5 on the model.

#### OC-7 Fig. S2 caption: corner contours are not 1, 2, 3σ
- Verdict: CONFIRMED
- Severity: medium
- Location: component_ocean cell 6 l.594–604 (`corner.corner(...)`, no `levels`); supp tex l.278
- Reported by: nb_ocean_A, nb_ocean_B, num_supp_1, mod_io (also nb_glacier_A, nb_apen_A, nb_apen_B for their figures)
- Manuscript impact: supp l.278 "with contours denoting 1, 2, and 3 standard deviations". The plot actually draws four contours at 0.5, 1, 1.5 and 2σ. The same caption sentence appears for S3–S5 (other clusters).
- Defect: Installed corner 2.2.3 defaults to `levels = 1.0 - np.exp(-0.5 * np.arange(0.5, 2.1, 0.5) ** 2)` when `levels` is None.
- Evidence: Read from the installed `corner.core` source (version 2.2.3). Cell 6 passes no `levels`.
- Fix: Pass `levels=1-np.exp(-0.5*np.array([1,2,3])**2)` and re-render, or correct the caption.

#### OC-8 Thermosteric SSP forcing offset (0.755 °C) differs from the other components (0.726 °C)
- Verdict: CONFIRMED
- Severity: medium
- Location: component_ocean cell 9 l.961–968 (`T_OFFSET`, SSP2-4.5 2015.5–2024.5 overlap, applied to all SSPs); compare component_glacier cell 13 l.833–836 (Historical 1995–2005), with the same pattern in greenland, eais and apeninsula
- Reported by: xc_constants, mod_comp
- Manuscript impact: thermosteric 2100 medians (main tex: 147 / 246 / 374 mm) and the totals. Using the 0.726 °C offset raises the posterior-median ODE at 2100 by 4.3 mm (SSP2-4.5) and 5.3 mm (SSP3-7.0). Which convention is right is a choice. The defect is that summed components use different offsets.
- Defect: Ocean (and component_forecast) subtract 0.755 °C (SSP2-4.5 2015–2024 overlap with Berkeley Earth) for every SSP. The other calibrated components subtract 0.726 °C (AR6 Historical 1995–2005 mean). So the ocean is forced 0.029 °C cooler than the components it is summed with, and member warming paths are not exactly shared as the supp l.362 "same warming" text implies.
- Evidence: v5.py: `SSP2_4_5 0.7554`, `SSP5_8_5 0.7795`, `SSP1_2_6 0.7529`, `hist 1995-2005 0.7257`, BE 1995–2005 mean ≈ 0. v7.py: 2100 `[232.5 236.8]` (SSP2-4.5) and `[363.0 368.3]` (SSP3-7.0).
- Fix: Use one shared offset function (config or warming_paths) in every component, then rerun.

#### OC-9 Fig. S10 caption: error bars called "total point error" leave out σ_extra
- Verdict: CONFIRMED
- Severity: medium
- Location: scripts/ocean_early_holdout.py l.364 (`sigma_700_total_mm = sig700`), l.424 (`yerr=Z_90 * sy`); supp tex l.384
- Reported by: scripts_comp
- Manuscript impact: supp l.384 "(circles, with 90\% intervals of the total point error)". The main text (l.199) defines the point variance as including σ_extra (1.56–1.95 mm, the same size as the formal error), but the bars omit it.
- Defect: The bars are 1.645 × sqrt(formal² + (5%·|level|)²). The observed-rate and level SEs in the paragraph (0.20 [0.16, 0.24], 12.0 [11.4, 12.6]) also omit σ_extra and treat the 5% term as independent between years (see OC-12).
- Evidence: Script lines quoted above.
- Fix: Reword the caption to "formal plus 5% structural error", or add σ_extra to the bars.

#### OC-10 Supplement Table S2: "six fitted thermosteric parameters"
- Verdict: CONFIRMED
- Severity: low (a parameter count in a notation table, contradicted by the main text and the S2 caption; no result changes)
- Location: supp tex l.187
- Reported by: nb_ocean_A, nb_ocean_B, num_supp_1
- Manuscript impact: supp l.187 "six" → "five". Main l.191, supp l.278 and the code (`ndim = 5`) all have five.
- Defect: This was left over from removing the drift term c on 2026-10-16.
- Evidence: `$\boldsymbol{\phi}$ & Vector of the six fitted thermosteric parameters`; `ocean_posteriors` holds 5 arrays.
- Fix: Change to "five".

#### OC-11 σ_extra median quoted as 2.0 mm; the posterior median is 1.9499 mm
- Verdict: CONFIRMED
- Severity: low (a last-digit rounding of a nuisance parameter, 0.05 mm, far inside its interval; unlike OC-4 no endpoint is wrong)
- Location: main tex l.196; component_ocean cell 5 print (notebook JSON: `sigma_extra : 1.9499 mm [1.6804, 2.2783]`)
- Reported by: num_main_1
- Manuscript impact: main l.196 "$\sigma_{\mathrm{extra}} = 2.0$ [1.7, 2.3] mm" → 1.9 [1.7, 2.3] mm, or 1.95 [1.68, 2.28].
- Defect: 1.94989 rounds to 1.9. The holdout log's "1.950" seems to have been rounded again.
- Evidence: v1.py: `sigma_extra mm pct [1.94988675 1.68035637 2.27830901]`.
- Fix: Write 1.9 [1.7, 2.3] mm.

#### OC-12 5% residual structural error is per-point, independent, and tied to the rebase point
- Verdict: PLAUSIBLE
- Severity: low
- Location: component_ocean cell 5 l.226–232; scripts/ocean_early_holdout.py l.97–98
- Reported by: nb_ocean_A, nb_ocean_B, scripts_comp
- Manuscript impact: none demonstrated. The tex (l.199) describes the code accurately. A fully correlated ±5% scale error would add about ±12 mm (1σ) at 2100 under SSP2-4.5, in quadrature with an interval sd of about 48 mm (~3% wider).
- Defect: A residual error in the ×1.14 scale factor is fully correlated across years, but it enters as independent diagonal variance. It is proportional to |anomaly|, so it is zero at an arbitrary reference (1995–2005 for 0–700 m, 2005.5 for 0–2000 m), although the likelihood demeans each record. The 0–700 m and 0–2000 m likelihoods are summed as independent over 2005–2025 although they share the upper water column. The 0–700 m residual lag-1 autocorrelation is +0.56 (cell 7 output).
- Evidence: Code lines quoted; cell 7 output `0-700 m : RMS 2.33 mm, chi2/n 1.13, lag-1 autocorr +0.56`.
- Fix: Treat the scale uncertainty as a sampled factor per projection member (or a rank-1 covariance term), or state the approximation.

#### OC-13 Saved `ocean/observations` "full-depth estimate" lacks the 700–2000 m layer before 2005.5 and is offset after it
- Verdict: CONFIRMED
- Severity: low
- Location: component_ocean cell 9 l.987–1010, attr l.1118; consumers: cell 11 two-panel figure (notebook only), component_forecast per-component figure panel b (main tex l.387, from 2000)
- Reported by: nb_ocean_A, nb_ocean_B
- Manuscript impact: in the main-text Fig. per_component_forecast (b), the observation curve over 2000–2025 omits a 1.4 mm layer term, and its net offset relative to the model median is −0.75 mm (2005.5–2025.5). Supp l.61 says the layer is "carried ... by the deep layer of the two-layer model at earlier times", which the stored observations do not do.
- Defect: Before 2005.5 the stored series is 0–700 m + the below-2000 m rate, with no 700–2000 m layer (−7.7 mm at 1955.5 and −2.4 mm at 1990.5 relative to the model layer). After 2005.5 it misses the 1.4 mm layer rise from the baseline to 2005.5. The attr says "(2006+)", but the increment starts at 2005.5.
- Evidence: v6.py: stored obs minus model median `+5.00 mm` (1955–1975), `−0.75 mm` (2005.5+); attr string quoted. Cell 7 already restricts its own plot to ≥2005.5 for this reason.
- Fix: Store observations only from 2005.5, or anchor the increment with the model layer. Correct the attr text.

#### OC-14 Projections and the summation ODE median rebased at 1999.5 (argmin tie), not 2000.0
- Verdict: CONFIRMED
- Severity: low
- Location: component_ocean cell 9 l.1072 (`ib = np.argmin(np.abs(years_proj - BASELINE_YEAR))`), cell 15 l.1346, cell 11 l.1208; component_summation cell 11 l.982 (`ode_t` = year+0.5)
- Reported by: nb_ocean_A, nb_ocean_B, nb_summation_B, mod_io, tests
- Manuscript impact: none at quoted precision. Ocean levels relative to 2000 are about 0.4 mm high (2100 SSP2-4.5 median 246.13 vs 245.75 mm). In summation the effect cancels through the 1950 per-member offset, and only the cell 12 column moves (~0.3 mm).
- Defect: Mid-year stamps tie 1999.5/2000.5, and argmin picks 1999.5. The SSP2-4.5 median at PROJ_YEARS 2000 is +0.40 mm, while the stored observations use the 1995.5–2005.5 mean.
- Evidence: v2.py: medians `1999 −0.649, 2000 0.399, 2001 1.301` mm; median of the 1995–2005 mean 0.385 mm.
- Fix: Rebase on the value interpolated to 2000.0, or on the 1995–2005 mean.

#### OC-15 Inconsistent 1995–2005 windows: 11 values (cells 3, 5) vs 10 values (cell 7 column 3)
- Verdict: CONFIRMED
- Severity: low
- Location: component_ocean cell 3 l.99, cell 5 l.418, cell 7 l.701 (`BL_LO, BL_HI = 1995.0, 2005.0` on yyyy.5 stamps → 1995.5–2004.5)
- Reported by: nb_ocean_A, nb_ocean_B
- Manuscript impact: marginal (column 3 Frederikse/Dangendorf RMS 3.4/2.4/2.8/7.0 mm; model and data use the same 10-point window inside column 3, so this is internally consistent). The cell 3 print says "rebased to 1995–2006".
- Defect: The baseline window differs between the calibration panels and the validation panel. The cell 3 label is wrong.
- Evidence: Lines quoted.
- Fix: Use one window (config BASELINE_WINDOW with a stated mid-year convention).

#### OC-16 Stale labels and prints in component_ocean (markdown, cells 5, 7, 9)
- Verdict: CONFIRMED
- Severity: low
- Location: cell 4 l.194; cell 5 l.462–465, 471, 483–494, 505, 516; cell 7 l.845, 898; cell 9 l.1106, 1118
- Reported by: nb_ocean_A, nb_ocean_B
- Manuscript impact: none (the cell 7 legend label is in the main-text figure; see OC-6)
- Defect: (a) The markdown says the below-2000 m rate is 0.07 mm/yr; the code uses 0.117 (defined three times in the notebook and once in results_figures, all agreeing). (b) The print says "Reduced chi^2" but computes chi²/n. (c) R² still appears in plot titles and in `fit_diagnostic` attrs, and the panel is titled "Full depth (0--2000 m)". (d) The "44% noise / R² near 0.03" comment has not been checked. (e) The "1993-2025" rate row compares NOAA 0–2000 m 2005–2025 (1.453 in both rows) with the model over 1993–2025. (f) `axfr[1,1].legend()` has no labelled artists (stored UserWarning). (g) "baseline=6.6 mm" is the 2005 value. (h) The `.0f` year prints round half-to-even.
- Evidence: Stored outputs: `SSP2-4.5: baseline=6.6 mm`; notebook JSON cell 5 `Total: model 1.282, NOAA 1.453` and `Total: model 1.467, NOAA 1.453`; cell 7 `UserWarning: No artists with labels found`.
- Fix: Correct the strings and the markdown, and define RATE/SIGMA_BELOW_2000 once.

#### OC-17 Notebook-only figures with mixed baselines or time stamps (cells 11, 12, 13; results_figures fig3)
- Verdict: CONFIRMED
- Severity: low
- Location: component_ocean cell 11 l.1219 (SSP years without +0.5); cells 12–13 (AR6 oceandynamics on the 1995–2014 reference vs ours on the 2000 reference, no offset); results_figures cell 30 l.3501–3502 (`_H_full_mm = _H_total_mm + 0.117*(t-2000)` on an H_total that is zeroed at 2005.5)
- Reported by: nb_ocean_A, nb_ocean_B, mod_io
- Manuscript impact: none. component_ocean_twopanel/histogram/ridge.png and fig3_calibration_validation.png are not in either tex (grep of \includegraphics). The included fig_s0_component_fit_diagnostics.png (main l.239) has no ocean panel.
- Defect: The SSP curves are plotted half a year early. The IPCC distributions are about 6 mm too low relative to ours (our 1995–2014 mean is +6.0 mm). The fig3 "Full depth model" mixes references and hardcodes 0.117.
- Evidence: Lines quoted; tex grep.
- Fix: Add 0.5 to the SSP years, shift for the AR6 reference, and read the `rate_below_2000_mm_yr` attr.

#### OC-18 ODE initial condition: both layers start in equilibrium with 1850 GMST (not stated)
- Verdict: PLAUSIBLE
- Severity: low
- Location: component_ocean cell 5 l.268, 275; solve_twolayer_ode default; holdout script
- Reported by: nb_ocean_A
- Manuscript impact: unknown (shapes S_d, and so the 1900–1950 hindcast, given τ_d = 150 yr); the main text does not state the initial condition.
- Defect: `Su[0] = Sd[0] = T_annual[0]` (Berkeley 1850 = −1.00 °C relative to 1995–2005). A single year sets the deep-layer state.
- Evidence: Code lines quoted.
- Fix: State it in the methods, or initialise from a multi-decade mean or a spin-up.

#### OC-19 A fresh component_results.h5 would lack `proj_years` (ordering hazard)
- Verdict: PLAUSIBLE
- Severity: low
- Location: component_ocean cell 5 l.378 (`h5py.File('../data/processed/component_results.h5', 'a')`); component_io.py l.65–75 `_require_file`
- Reported by: nb_ocean_A
- Manuscript impact: none (latent; triggered only if the file is deleted before a run)
- Defect: Ocean runs first and creates the file with 'a'. `_require_file` writes `proj_years` and the file attrs only when the file does not exist, so they would never be written, and `load_all_projections` would raise KeyError.
- Evidence: `_require_file`: `if not path.exists(): ... f.create_dataset("proj_years", ...)`.
- Fix: Call `_require_file(DEFAULT_H5_PATH)` before the cache write.

#### OC-20 Stale docstrings, metadata and legacy thermosteric code
- Verdict: CONFIRMED
- Severity: low
- Location: component_io.py l.321–345 (`save_ocean_hybrid`: "projections come from IPCC AR6 oceandynamics", writes `model_type='hybrid_noaa_ipcc'` before the overwrite, `model_order = 0`); component_io module docstring (`ocean/posteriors`); bayesian_models.py l.4876 `fit_bayesian_thermosteric` and slr_projections.py l.709–850 `project_thermosteric_ensemble` (both keep the c drift term; the latter is imported in component_ocean cell 1 l.36 but never called); component_ocean cell 3 l.153 "loaded for reference, not used in fit" for Frederikse, which is used as withheld validation
- Reported by: nb_ocean_A, nb_ocean_B, mod_io, mod_bayes_B, mod_slr
- Manuscript impact: none
- Defect: The documentation describes superseded designs. The unused importable code still has the removed drift term.
- Evidence: HDF5 `ocean.attrs['model_order'] = 0`; docstring text; import line 36.
- Fix: Update the docstrings and drop or mark the legacy functions.

#### OC-21 Small traceability and wording items
- Verdict: CONFIRMED
- Severity: low
- Location: main tex l.196 ("42 and 30 mm of observed steric rise", not printed); supp l.362 ("grows from 82 to 157 mm"); component_ocean cell 3 l.86–144 (NOAA/EN4 cached into slr_processed_data.h5 on KeyError)
- Reported by: nb_ocean_A, num_main_1, num_supp_2, xc_hdf5
- Manuscript impact: none (values correct: 41.8/29.7 mm; cell 15 `SSP2-4.5: 157 | 82`)
- Defect: 42/30 mm is not printed by any code. The "82 mm" is the width with independent annual noise, not without warming spread, and the text does not say so. Once a raw NOAA key is cached it is never refreshed (the current copy matches the raw files, both ending 2025.5).
- Evidence: As reported. The cell 15 output matches.
- Fix: Print the observed rises, clarify the 82 mm baseline in the text, and document the cache.

#### OC-22 Ocean tests encode a stale baseline and retired labels
- Verdict: CONFIRMED
- Severity: low
- Location: tests/test_ocean.py l.39 (`BASELINE_YEAR = 2005.0`), l.175–178 (accepts 'physical_1layer_joint', 'hybrid_noaa_ipcc'), l.262–267 (`abs(med_mm) < 15.0` at 2005), 4-SSP list without SSP1-1.9
- Reported by: tests
- Manuscript impact: none
- Defect: The baseline check uses 2005 (actual 6.57 mm) with a 15 mm tolerance, so it cannot catch the 2000 rebase offset (OC-14) or a sign error. Nothing tests the five-parameter posterior, ×1.14, 0.117 mm/yr or τ_d = 150.
- Evidence: Lines quoted.
- Fix: Test at 2000.0 with a sub-mm tolerance, accept only 'twolayer_noaa', and add SSP1-1.9.

#### OC-23 (REFUTED) "Convergence numbers are printed only by the holdout script / cell 5 output lost"
- Verdict: REFUTED
- Severity: low
- Location: component_ocean cell 5 l.433–448; main tex l.196
- Reported by: num_main_1, nb_ocean_B
- Manuscript impact: none
- Defect: The claim is that production does not print the convergence numbers.
- Evidence: The notebook JSON cell 5 stdout contains `max tau_int = 226 steps (133x shorter than the chain); min ESS = 8504; max R_hat = 1.0037`, `0-700m: 1.13 0-2000m: 0.37`, and `sigma_extra : 1.9499 mm`. The truncation is in the extracted `_outputs.txt` only (nb_ocean_A noted this). The tex values match the production output. `progress=True` clutter is cosmetic.
- Fix: none required. Optionally set `progress=False`.

### Old-report items (component_ocean; from nb_ocean_A, spot-checked)
- save_ocean_hybrid deletes cached posteriors / KeyError on reload — FIXED — cell 5 l.378–391 writes top-level `ocean_posteriors`; cell 6 reads it.
- `_f['ocean']` KeyError on fresh HDF5 — FIXED (path gone); a related fresh-file hazard remains (OC-19).
- Ocean flat after 2099.5; 2100 = mid-2099 — FIXED — cell 9 l.1032 `ext_years = np.arange(ssp_years[-1] + 1, PROJ_YEARS[-1] + 1)`.
- SSP temperature uncertainty as independent annual noise — FIXED — cell 9 l.1051–1055 `if SAMPLE_WARMING_PATHS: T_k[n_hist:] += dT_future[k]`.
- argmin tie at 2000 → 1999.5 — STILL OPEN — cell 9 l.1072 (OC-14).
- Full-depth obs miss 700–2000 m change before 2005.5 — STILL OPEN — cell 9 l.996 (OC-13).
- Rebase windows [1995,2006] vs [1995,2005] — STILL OPEN — cell 7 l.701 (OC-15).
- "1993-2025" rate row mismatched windows — STILL OPEN — cell 5 l.483 (OC-16e).
- chi2/n labelled "reduced chi^2" — STILL OPEN — cell 5 l.471 (OC-16b).
- SSP years in cell 11 without +0.5 — STILL OPEN — l.1219 (OC-17).
- Corner drift label epsilon_u vs c — NOT APPLICABLE — drift term removed; labels l.557–560.
- "Full depth" title for 0–2000 m and R² in titles — STILL OPEN — cell 5 l.516 (OC-16c).
- 44% noise / R² cap comment — STILL OPEN — cell 5 l.464–465 (OC-16d).
- "baseline=" print at 2005 — STILL OPEN — cell 9 l.1106 (OC-16g).
- Metadata "(2006+)" — STILL OPEN — cell 9 l.1118 (OC-13).
- Cell 7 "(calibration)" legend under "validation" title; empty legend — STILL OPEN — l.845, l.898 (OC-6, OC-16f).
- Stored outputs predate Dangendorf — FIXED — cell 7 output has Dangendorf lines.
- Frederikse "pre-1955" → pre-1957 — FIXED — cell 2/3 "pre-1957".
- Frederikse "loaded for reference, not used in fit" — STILL OPEN — cell 3 l.153, 163 (OC-20).
- Cell 3 "Rebaseline ... cannot be computed" vs 1995–2006 window — NOT APPLICABLE — comment is correct.
- Li 2022 "Southern Ocean sampling" wording — FIXED — cell 5 l.219–220.
- Cell 6 degeneracy comment lacks prior; "caption" vs table — FIXED — l.531–534.
- `_thin` "~5,800 draws" — FIXED — l.752 `# 19,200 draws`.
- Cell 9 comment years 1956–2005/2006 — FIXED — l.982–985.
- Cell 7 '0-2000m      ' padding — NOT APPLICABLE — cosmetic.
- RATE_BELOW_2000 0.07 (Purkey & Johnson) — FIXED in code (0.117e-3); cell 4 markdown still says 0.07 — STILL OPEN as OC-16a.
- IPCC comparison baseline 1995–2014 vs 2000 (cells 12–13) — STILL OPEN — OC-17.
(nb_ocean_B had no old-report section; no reviewer disagreements to reconcile.)

### Cross-cluster notes
- Corner contour-level caption error also affects supp S3 (glacier), S4 (Peninsula), S5 (EAIS) — nb_glacier_A, nb_apen_A/B, num_supp_1, mod_io.
- SSP offset mismatch also hits rate-and-state (AR6 2015 matched to Berkeley 2014 by argmin tie, 0.887–0.898 °C) — nb_ratestate_A, xc_constants, mod_comp (RATESTATE/constants cluster).
- component_summation: WAIS zero before 1950 (2 mm step) — nb_summation_A/B; IPCC-to-2000 offset sign in cells 18/20/28 — nb_summation_A; Dangendorf 90% interval sigma convention (affects "1906 and 1908") — nb_summation_A; IPCC TWS 2005 reference — nb_summation_B (SUMMATION cluster). The "1906 and 1908" test also moves with OC-1.
- PDFs built before the run regenerated all figures (component_ocean_* included) — xc_hdf5 (HDF5/repo cluster).
- results_figures Greenland panel extension uses GMST, not GMST×AA — nb_figs_A (FIGURES cluster).
- Tests: HDF5 group-deletion coverage (save_ocean_hybrid vs sibling groups) untested — tests (TESTS cluster).

---

## B2. Glacier (GL)

Verification scripts and outputs: `$S/verify_GL/` (h5glac.py, sx.py, gain.py, gdr_copy.py + out_*.txt, fred.txt, imbie3.txt).

#### GL-1 glacier_delayed_response.py uses GlaMBIE 95% half-widths as 1σ and has no σ_extra; main-text l.433 numbers come from it
- Verdict: CONFIRMED
- Severity: medium
- Location: scripts/glacier_delayed_response.py:132-138 (`data()`), run in run_pipeline.sh as `glacierdelay`; same pattern in scripts/glacier_zemp_diagnostic.py:64 and scripts/glacier_bookkeeping_check.py:38 (neither in the pipeline)
- Reported by: nb_glacier_A, xc_constants, scripts_comp (cross-ref)
- Manuscript impact: main tex l.433, "fits the glacier record as well as ours when the response time is a century or more", "about 30 mm less ... at each warming level", "12 to 14 mm less if response times up to millennia are equally plausible". These are differences printed by this script (log: linear 117/142/165 vs τ-marginalised 86/111/135 → 31/31/30; vs log-uniform 105/128/151 → 12/14/14). With GlaMBIE converted to 1σ and σ_extra = 0.18 mm/yr added (the production error model), my rerun of a patched copy gives 31/32/32 mm and 11/12/13 mm, and logML(τ=101 yr) −0.1 (still "as well"). So "12 to 14" → "11 to 13"; "about 30" holds. With /1.96 alone (no σ_extra) the numbers would be 26 and 8–10 mm, so the current agreement relies on the two errors offsetting.
- Defect: The script reads the raw GlaMBIE CSV and puts `combined_gt_errors` (95% half-widths) on the covariance diagonal unconverted, while converting Zemp to 1σ via the reader. GlaMBIE variances are therefore 3.84× too large relative to Zemp, and the production σ_extra term (chi2/dof 4.1 without it) is missing. The wrong conversion partly stands in for the missing scatter term by accident.
- Evidence: `C[nz:, nz:] = np.diag((g['combined_gt_errors'].values / GT_PER_MM) ** 2)`; reader slr_data_readers.py:2799 "combined_gt_errors are 95% half-widths (1.96 sigma); converted to 1-sigma below" and `/ 1.96`. Patched copy (`GL_FAC=1.96 GL_SX=0.18`, out_1.96_0.18.txt; σ_extra added to the GlaMBIE block only, as in production, where σ_extra is fitted against GlaMBIE and Zemp is held out): linear 121/148/173, τ-marginalised 90/116/141, log-uniform 110/136/160 mm. Stationarity table (Zemp-only) is unchanged, so the supp l.379 claim about 1962–1989 vs 1990–2016 is unaffected.
- Fix: Use read_glambie_global (1σ) and add σ_extra (fixed at the production posterior median or fitted) to the GlaMBIE block; rerun and update l.433 ("11 to 13 mm"). Apply the same to glacier_zemp_diagnostic.py and glacier_bookkeeping_check.py.

#### GL-2 σ_gl,extra = 0.18 [0.13, 0.26] mm/yr (main l.233) is not saved or printed anywhere; T_eq0 not saved either
- Verdict: CONFIRMED (untraceable); the value itself reproduces
- Severity: medium (violates the numbers-traceable-to-code rule; numerically correct)
- Location: component_io.py save_glacier (~l.369-394, no `extra_posteriors`); component_glacier cell 13 save_glacier call; cell 6 summary prints (lines 384-409) lost to the emcee progress bar in stored output
- Reported by: mod_io, num_main_2, nb_glacier_A, nb_glacier_B, num_supp_1
- Manuscript impact: main l.233 "$\sigma_{\mathrm{gl,extra}} = 0.18$ [0.13, 0.26]~mm\,yr$^{-1}$" has no source. b, c and T_eq (= −c/b) do trace: HDF5 gives b 0.761 [0.601, 0.993], c 0.592 [0.518, 0.665], −c/b −0.781 [−1.027, −0.554], matching l.233.
- Defect: `glacier/posteriors` holds only `posterior_samples` (a, b, c) and `H0_posterior`. The cell-6 print of sigma_extra and T_eq0 exists in code but the stored output for cell 6 ends in a tqdm bar (`93%|█████████▎| 14878/`), and no §6 cell or manuscript_numbers.py prints σ_extra.
- Evidence: HDF5 keys `['H0_posterior', 'posterior_samples']`; outputs file CELL 6 has no "sigma_extra =" line. My approximate reconstruction (rate-space likelihood, HalfCauchy(0.2) prior, conditioned on 3000 posterior (b, c) draws, sx.py) gives 0.183 [0.129, 0.258] mm/yr, consistent with the tex.
- Fix: Pass `extra_posteriors={'sigma_extra_posterior': ..., 'T_eq0_posterior': ...}` in save_glacier (and save_apeninsula), and print σ_extra in cell 22 or manuscript_numbers.py; set `progress=False` (or flush) so cell 6 summary survives in the stored output.

#### GL-3 Glacier corner caption: contours are not at 1, 2, 3σ
- Verdict: CONFIRMED
- Severity: medium (manuscript caption wrong)
- Location: component_glacier cell 10 (`corner.corner(...)`, no `levels=`); supp l.287
- Reported by: nb_glacier_A, num_supp_1, mod_io, nb_apen_A (same defect in all four corner captions)
- Manuscript impact: supp l.287 "contours denoting 1, 2, and 3 standard deviations" is false; corner 2.2.3 draws 0.5, 1, 1.5, 2σ.
- Defect: No `levels` argument, so corner's default `1 - exp(-0.5*arange(0.5, 2.1, 0.5)**2)` applies. The other caption sub-claims hold: histograms are grey (COMP_COLORS['Glaciers'] = '#808080'), and the H0 prior N(0, 2 mm) is wide against a posterior sd of 0.25 mm. The σ_extra prior HalfCauchy(0.2 mm/yr) has a scale close to the posterior median (0.18), so "wide" is only true of its tail.
- Evidence: installed corner 2.2.3 `hist2d`: `levels = 1.0 - np.exp(-0.5 * np.arange(0.5, 2.1, 0.5) ** 2)`; call passes `labels, show_titles, color, plot_datapoints, fill_contours, hist_bin_factor, hist_kwargs` only.
- Fix: Pass `levels=1-np.exp(-0.5*np.array([1,2,3])**2)` and re-render, or change the caption to "0.5, 1, 1.5, and 2". Consider "heavy-tailed" rather than "wide" for σ_extra.

#### GL-4 Supplement χ²/dof does not count σ_extra in the degrees of freedom
- Verdict: PLAUSIBLE (convention, not stated)
- Severity: medium (supp describes it as "per degree of freedom" without saying a fitted parameter is excluded; the alternative convention changes a supp number slightly)
- Location: bayesian_models.py:3338 (`/ (n - n_phys - 1)`); supp l.371 and Table (l.400)
- Reported by: mod_bayes_A
- Manuscript impact: supp l.400 glacier "1.05" and l.371 "All four sit between 0.94 and 1.05". Counting σ_extra gives glacier 1.045×21/20 = 1.10 (Peninsula 1.00, WAIS ≈1.01).
- Defect: The denominator subtracts the physical parameters and H0 but not σ_extra. Either convention is defensible, but the supplement calls it "per degree of freedom" without saying σ_extra is excluded.
- Evidence: fitter code quoted; HDF5 attr chi2_dof = 1.04496, n = 24, n_phys = 2.
- Fix: State the convention in the caption, or count σ_extra and update to 1.10 and "0.94 and 1.10".

#### GL-5 Volume-cap claim in cell 0 contradicts cell 21 output; 2150 glacier values are cap-dominated
- Verdict: CONFIRMED
- Severity: low (not in tex; 2100 unaffected)
- Location: component_glacier cell 0 (markdown) vs cell 21 output; downstream summation cell 16 2150 rows
- Reported by: nb_glacier_A, nb_glacier_B, nb_summation_A
- Manuscript impact: none (no glacier 2150 number in tex). The VoI surface in the tex (Fig. 11c, supp l.534; compute_evpi.py VOI_2) uses 2100 samples only; the unpublished VOI_1 target-year grid runs to 2150 on SSP2-4.5, where 5.1% of glacier members are capped.
- Defect: Markdown says the 0.32 m cap "binds in at most 0.1% of samples (SSP5-8.5, 2100 and 2150)". It binds in 0% at 2100 but 5.1% (SSP2-4.5), 51.0% (SSP3-7.0) and 84.0% (SSP5-8.5) of members at 2150, because the linear model has no area shrinkage.
- Evidence: cell 21 output lines quoted above; cell 0 text.
- Fix: Correct the markdown and flag 2150 glacier values as cap-determined wherever reported.

#### GL-6 Peripheral-glacier double counting with IMBIE-3 Antarctic (RGI 19) is not checked
- Verdict: PLAUSIBLE
- Severity: low
- Location: notebooks/slr_data_readers.py:2794-2798 (docstring warning, no check); main tex l.217
- Reported by: nb_glacier_B
- Manuscript impact: unknown, small. GlaMBIE RGI 19 = 17.8 Gt/yr = 0.049 mm/yr over 2000–2023. Main l.217 addresses only the Greenland periphery.
- Defect: GlaMBIE's global total includes RGI 19. IMBIE-3 (Otosaka 2026, Methods "Accounting for the Ice Sheets' Peripheral Glaciers and Ice Caps") says all its gravimetry estimates include the ice-sheet periphery, its altimetry estimates and one input–output estimate do not, and it does not correct the Antarctic altimetry for the omission. The IMBIE-3 Peninsula/EAIS/WAIS aggregates therefore contain part of the RGI 19 signal, which is also in the glacier component. Upper bound on overlap ~0.05 mm/yr now, a few mm by 2100.
- Evidence: imbie3.txt l.420-455: "all our gravimetry estimates account for both the main ice sheets and their periphery ... this is not the case of our altimetry estimates and one of our input-output estimates"; "We thus do not attempt to correct the altimetry time-series for their omission of the GICs mass changes in Antarctica."
- Fix: Add one sentence to l.217 stating the partial overlap and its upper bound (≤0.05 mm/yr), or subtract a fraction of RGI 19.

#### GL-7 Frederikse comparison is definitionally mismatched (glacier excl. RGI 5/19; Greenland incl. RGI 5)
- Verdict: CONFIRMED
- Severity: low
- Location: component_summation cell 12 (`_comp_compare`, lines ~1376-1395); scripts/glacier_delayed_response.py:273; main tex l.373
- Reported by: nb_glacier_B, scripts_comp
- Manuscript impact: main l.373 "Greenland raised it by 10 mm more than in our hindcast" (Frederikse 1950 −13.2 vs ours −2.9 mm). Part of that 10 mm is the Greenland periphery, which Frederikse counts in Greenland and this study counts in glaciers; the glacier difference (ours −24.0 vs −26.3 mm at 1950) is biased the opposite way. The total is unaffected.
- Defect: Frederikse (2020) Methods: "we do not include these [Greenland and Antarctic peripheral] glaciers into the glacier mass balance. For Greenland, we add the peripheral glaciers to the ice-sheet contribution ... we assume no mass loss from the Antarctic peripheral glaciers." The per-component table compares unlike quantities without saying so.
- Evidence: fred.txt l.863-891 (Frederikse 2020 Zotero PDF); summation output `Glaciers (incl. uncharted) 1950 -24.0 -26.3 2.4`, `Greenland 1950 -2.9 -13.2 10.3`. Note: the memory file frederikse_greenland_sources.md says peripheral glaciers are "Excluded from greenland, counted in glaciers column", which is the reverse of the paper.
- Fix: Label the definitional difference in the print, optionally move an RGI 5 estimate between columns; consider qualifying "10 mm" in l.373. Correct the memory note.

#### GL-8 Glacier BIC: dof back-conversion mismatched (negligible) and log-det dropped, so ΔBIC ≈ −ln n by construction
- Verdict: CONFIRMED (mechanism); the dof part has negligible effect
- Severity: low (not in tex; supports cell 0 "BIC model selection confirms the linear model is preferred")
- Location: component_glacier cell 6 lines 307-313, cell 20 lines 1180-1181; bayesian_models.py:3338
- Reported by: nb_glacier_A, nb_glacier_B, mod_bayes_A, tests
- Manuscript impact: none
- Defect: The fitter divides χ² by n−n_phys−1 (21, 20), the notebook multiplies back by n−n_phys−2 (20, 19). Its effect on ΔBIC is chi2_dof_l − chi2_dof_q = 1.045 − 1.036 ≈ 0.01, so this is cosmetic. The substantive issue is that with σ_extra fitted, −2lnL includes ln|Σ(σ_extra)|, which is dropped, and χ² is taken at the posterior mean with a≈0.035 pinned near zero by its Exp prior. The quadratic and linear fits are nearly identical, so ΔBIC is essentially the k·ln n penalty (cell 20: −3.1/−3.5/−3.3 vs ln 24 = 3.18; main fit ≈ −2.0 by my arithmetic from the printed chi2_dof). The "linear preferred" verdict is close to predetermined.
- Evidence: `dof_l = n - 2 - 1 - 1`; `chi2_dof = ... / (n - n_phys - 1)`; outputs cell 6 chi2/dof 1.045 (lin), 1.036 (quad); cell 20 ΔBIC rows.
- Fix: Compute −2 max lnL including ln|Σ| from each fit, or reword the markdown to say the data do not resolve curvature. tests/test_glacier.py BIC tests use a different (RSS) formula and are vacuous.

#### GL-9 "Gain glacier mass in almost every year before about 1970" (main l.226, l.235) is untraceable and loose; code comment says 1935
- Verdict: CONFIRMED (stale comment; tex claim untraceable, approximately supported)
- Severity: low
- Location: component_glacier cell 6 comment (line 269-270); main l.226, l.235
- Reported by: num_main_2, nb_glacier_A, nb_glacier_B
- Manuscript impact: wording of l.226 ("almost every year before about 1970"). At the c-free medians (b 0.99, c 0.53) the rate is negative in 98% of 1850–1899, 88% of 1900–1949, 60% of 1950–1969 and 60% of 1970–1979; last gain year 1976.
- Defect: The comment refers to the 2026-10-07 fit ("before about 1935"). No pipeline output prints the gain years, and "almost every year" overstates 1950–1969 (12 of 20 years).
- Evidence: gain.py output; T from harmonized/df_berkeley_h (1995–2005 mean 0.0).
- Fix: Print the fraction of gain years per period in cell 22; consider "most years before the late 1970s" or "almost every year before 1950"; update the comment.

#### GL-10 Supp "nor a quadratic dependence on GMST accounts for the difference" is untraceable
- Verdict: CONFIRMED
- Severity: low
- Location: supp l.379; scripts/glacier_delayed_response.py, scripts/glacier_zemp_diagnostic.py
- Reported by: num_supp_2
- Manuscript impact: qualitative claim at supp l.379 has no pipeline source; the delayed-response part is traced (stationarity table, Zemp-only, unaffected by GL-1).
- Defect: Neither script fits a quadratic on the Zemp windows; the notebook's quadratic is fit to GlaMBIE only.
- Evidence: `grep -i quad` finds no quadratic fit in either script.
- Fix: Add a quadratic early/late test to glacier_delayed_response.py, or drop the clause.

#### GL-11 "0.21 mm/yr from the uncharted-glacier term" (main l.373) not printed as a rate
- Verdict: CONFIRMED (traceability only; value correct)
- Severity: low
- Location: component_summation cell 10/11 (prints only the 1950 level, −10.5 mm)
- Reported by: nb_summation_A, num_main_3, num_main_2
- Manuscript impact: none numerically; OLS of hindcast_20c/uncharted_glacier_med over 1950–2000 = 0.209 mm/yr.
- Defect: Derived by hand from the level.
- Evidence: my h5 check: OLS 0.209, (H2000−H1950)/50 = 0.209.
- Fix: Print the 1950–2000 rate in summation §6.

#### GL-12 Stale constants and labels in glacier scripts
- Verdict: CONFIRMED
- Severity: low
- Location: glacier_delayed_response.py:3-4, 79, 207-209 vs 280-283, 273, 279, 295, 316-318; glacier_zemp_diagnostic.py:4-5, 50; glacier_bookkeeping_check.py:31
- Reported by: nb_glacier_A, nb_glacier_B, scripts_comp
- Manuscript impact: none
- Defect: `PROD_2100 = {122, 149, 175, 199}` is printed as "production" but production is 121/147/170/193 (cell 13). `PROD`/`OURS` hindcast dicts and the "almost no loss back to 1900" docstring predate the balance prior. The "linear (as production)" row has κ free, no σ_extra, untruncated b, and Zemp in calibration, so it is not production. τ quantiles printed two ways (80 [40, 212] by weighted CDF vs 44/89/228 by sample percentiles). Panel (b) mixes Zemp at integer Y, GlaMBIE at Y+0.5, unlabelled 1σ bars.
- Evidence: pipeline log `production (level-space fit, volume cap): SSP1-2.6 122, ...`; cell 13 `121 [104, 143]`, `147 [125, 179]`, `170 [141, 213]`.
- Fix: Read production medians from component_results.h5, relabel the linear row, use one quantile method.

#### GL-13 Observation time stamps and zero offset from the projection frame; cell 3 print mislabels
- Verdict: CONFIRMED
- Severity: low
- Location: component_glacier cell 3 (lines 89-92, 113-114); HDF5 glacier/observations
- Reported by: nb_glacier_A, nb_glacier_B
- Manuscript impact: none (H0 absorbs it; overlays off by <1 mm)
- Defect: cumsum(rate)[k] is the end-of-year level but is stamped at 2000.5+k, and subtracting cumsum[0] zeroes the record at end-2000 (removing the 0.215 mm year-2000 increment). Projections are zero at 2000.0. The print "GlaMBIE: 2000–2024" and "Cumulative at 2024: 17.8 mm" round 2023.5 up and exclude 2000 (full sum 18.05 mm).
- Evidence: HDF5 years [2000.5 … 2023.5], H_obs[0] = 0, final 17.83 mm; rate2000 = 0.215 mm.
- Fix: Stamp at end of year, prepend a zero at 2000.0 (matching the IMBIE convention), print calendar years.

#### GL-14 IPCC overlays not rebased; IPCC rate fit spans flat-forcing 2100–2150
- Verdict: CONFIRMED
- Severity: low (notebook-only figures)
- Location: component_glacier cells 14-16 (ipcc_extract q50 used directly), cell 18
- Reported by: nb_glacier_A, nb_glacier_B, mod_io (shared helper)
- Manuscript impact: none (forecast tables apply the shift; tex 118/144/167)
- Defect: AR6 (1995–2014 mean) overlaid on 2000.0-based samples (~2.9 mm offset). Cell 18 regresses IPCC rates on T over decadal points to 2150 where T is held flat, giving b = −0.725 (SSP1-2.6), R² 0.003–0.35.
- Evidence: cell 15 `ipcc_med = ex['q50'][yr_idx]`; cell 18 output.
- Fix: Shift by the model's 1995–2014 mean; restrict cell 18 to ≤2100 or drop it.

#### GL-15 Error-bar conventions mixed (2σ vs Z_90) in glacier notebook figures
- Verdict: CONFIRMED
- Severity: low (notebook-only; fig_s0 in tex uses Z_90)
- Location: component_glacier cell 9 line 544 (`yerr=2 * sig_r`), cell 18 line 1103 (`yerr=2 * glac_sigma`); cell 9 line 480 uses Z_90
- Reported by: nb_glacier_A, nb_glacier_B, xc_constants, nb_figs_A
- Manuscript impact: none
- Defect: 2σ bars beside 90% bands.
- Evidence: grep of `yerr` in cell code.
- Fix: Use Z_90 throughout.

#### GL-16 Projection seeds differ by SSP (400+i); comment contradicts itself
- Verdict: CONFIRMED
- Severity: low
- Location: component_glacier cell 13 lines 843-844, 877
- Reported by: nb_glacier_A, nb_glacier_B
- Manuscript impact: none found (medians; MC noise only in cross-SSP differences)
- Defect: Member k shares its warming z_k across SSPs but not its (b, T_eq0, H0) draw, unlike EAIS/Peninsula. Comment says draws are independent and that only temperature should differ.
- Evidence: `seed=400 + i`.
- Fix: One seed for all SSPs; fix the comment.

#### GL-17 Cell 20 taper sweep not guarded by REFIT
- Verdict: CONFIRMED
- Severity: low
- Location: component_glacier cell 20
- Reported by: nb_glacier_A, nb_glacier_B
- Manuscript impact: none
- Defect: Uses names defined only under `if REFIT:` (rate_sig_r, yrs_r, prior_kw_annual); NameError when REFIT=False.
- Evidence: no guard in cell 20 (cell 22 has one).
- Fix: Wrap in `if REFIT:`.

#### GL-18 Stale comments, docstrings and markdown (σ_extra rationale, old b values, corr(b,H0), headings)
- Verdict: CONFIRMED
- Severity: low
- Location: component_glacier cell 1 (l.58-68), cell 6 (l.219-222, 269-270, rate branch l.318-325), cell 7 (l.442-444), cell 10 comment, cell 18 (l.1018), cell 19 markdown, cell 21 (`uncapped`); bayesian_models.py ~2871, ~2888-2913 and fit_bayesian_level_annual_correlated docstring
- Reported by: nb_glacier_A, nb_glacier_B, mod_bayes_A, nb_apen_B
- Manuscript impact: none
- Defect: Comments say chi2/dof ≈ 1 without σ_extra and that σ_extra is unidentified (now 4.11 without it); cell 1 "b ~ 0.85 vs ~0.91", cell 6 prior sweep "0.81/0.86/0.87" (now 0.76); 'rate' path keeps fit_sigma_extra=False on the stale 1.02 rationale; corr(b,H0) "~0.4" (module) vs "~0.2–0.3" (notebook); "I1/I0 collinearity" refers to a removed fit; "Model selection justification" heading has no cell; `uncapped` holds capped samples (count still correct).
- Evidence: cell 22 output `(a) without sigma_extra: chi2/dof = 4.11`; quoted comments.
- Fix: Update text to current values; set fit_sigma_extra=True on the 'rate' path or mark it mis-specified.

#### GL-19 apply_glacier_volume_cap docstring promises hold-from-first-exceedance; code clamps elementwise
- Verdict: CONFIRMED (no current effect)
- Severity: low
- Location: notebooks/component_projections.py:1206-1212
- Reported by: mod_comp
- Manuscript impact: none. The rate b(T − T_eq0) is negative only for T < T_eq0 ≈ −0.78 °C, which never occurs in the SSP paths, so trajectories are monotone and the two behaviours coincide.
- Defect: Docstring vs code mismatch.
- Evidence: `capped[capped > v_total] = v_total`.
- Fix: Fix the docstring or use a cumulative-max mask.

#### GL-20 Legacy 'level' branch adds cumulative variances for a centred difference
- Verdict: CONFIRMED (inactive branch)
- Severity: low
- Location: component_glacier cell 9 line 474
- Reported by: nb_glacier_A, nb_glacier_B
- Manuscript impact: none
- Defect: Var(H[i+w] − H[i−w]) for independent increments is the difference of cumulative variances, not the sum.
- Evidence: `rate_sigma[_i] = np.sqrt(glac_sigma[_i + _w]**2 + glac_sigma[_i - _w]**2) / _dt * M_TO_MM`.
- Fix: `sqrt(|σ²[i+w] − σ²[i−w]|)` or native rate σ.

#### GL-21 Glacier tests: stale baseline, vacuous BIC tests, untested 95%→1σ and balance prior
- Verdict: CONFIRMED
- Severity: low
- Location: tests/test_glacier.py:94-149 (baseline 2005 hard-coded), 234-267 (RSS-based BIC, `assert np.isfinite(bic)`, `delta_bic < 2` vs comment "negative"); reader tests
- Reported by: tests
- Manuscript impact: none
- Defect: TestGlacierCumulative rebases at 2005 (pipeline 2000). BIC tests use n·ln(RSS/n)+k ln n with k = 4/5 while the live path uses χ²+k ln n with k = 2/3. No test would fail if the /1.96 GlaMBIE/Zemp conversion were removed; the balance prior and σ_extra are untested. (The tests.md sub-claim that this class "treats the reader sigma as already 1σ" is not a defect: the reader does return 1σ.)
- Evidence: `bl_idx = np.argmin(np.abs(years - 2005.0))`; `bic = n * np.log(rss / n) + k * np.log(n)`.
- Fix: Import BASELINE_YEAR; test against the production path; add a test that read_glambie_global sigma = raw/1.96/362500.

#### GL-22 Fig. S0(a) Zemp back-run uses calendar-year GMST; the quoted validation uses hydrological-year GMST
- Verdict: CONFIRMED (construction difference; visual effect small)
- Severity: low
- Location: results_figures cell 32 `_glacier_fit_panel` (zemp branch) vs component_glacier cell 11 (`_T_hyd = 0.25 T(Y-1) + 0.75 T(Y)`)
- Reported by: nb_figs_B
- Manuscript impact: none (tex 0.30 [0.17, 0.40] comes from cell 11)
- Defect: The plotted model curve and the quoted rate come from slightly different forcing constructions.
- Evidence: per nb_figs_B; cell 11 code uses hydrological weighting (supp l.379 states it).
- Fix: Use the hydrological-year T in the figure, or note it.

### REFUTED / downgraded sub-claims
- "Corner histograms are not gray" (nb_glacier_A) — REFUTED: COMP_COLORS['Glaciers'] = '#808080' (grey), matching the caption.
- "H0 prior is described as wide but is N(0, 2 mm)" (nb_glacier_A) — REFUTED: posterior sd 0.25 mm (median −0.49 [−0.90, −0.08]), so 2 mm is wide relative to the posterior.
- "BIC dof mismatch understates χ² ~5% and so biases ΔBIC" (nb_glacier_A/B) — downgraded: the mismatch shifts ΔBIC by only chi2_dof_l − chi2_dof_q ≈ 0.01; the substantive issue is the dropped log-det (GL-8).
- "TestGlacierCumulative treats the reader sigma as already 1σ" (tests) — REFUTED: read_glambie_global returns 1σ.

### Old-report items (component_glacier, 2026-09-24 audit)
- OBS_WINDOW drops 2023.5 — FIXED — cell 6 `OBS_WINDOW = (2000, 2024)`; output n=24 (A and B agree).
- Projections flat after ~2099 (§1.2) — FIXED — HDF5 SSP2-4.5 median 147 mm (2100) → 255 mm (2150); cell 13 comment on np.interp hold.
- Taper-sweep BIC mixes level residuals and rate sigmas (§1.5) — FIXED for units; residual dof/log-det issue reported as GL-8.
- Progress print labels b prior HN — FIXED — output `b~N(0.61,2.0 mm/yr/°C) truncated b>=0`.
- "P(a>0)=100%" print — FIXED — cell 6 comment "P(a>0) is not printed".
- Dangling plan/handoff_glacier_ratespace.md references — FIXED in component_glacier (still in results_figures `_glacier_fit_panel` docstring, other cluster).
- IPCC baseline in glacier overlays (§1.3) — STILL OPEN (A said NOT APPLICABLE to numbers, B STILL OPEN; spot-checked cells 14–16: q50 used unshifted) — GL-14.
- Rebase subtracts cumsum[0] / 0.5-yr stamp — STILL OPEN — cell 3 `glac_rebase = glac_cumul - glac_cumul[bl_idx]` — GL-13.
- Seed comment contradicts `seed=400+i` — STILL OPEN — GL-16.
- Error bars 2σ vs 90% — STILL OPEN — cell 9 l.544, cell 18 l.1103 — GL-15.
- Taper cell not guarded by REFIT — STILL OPEN — cell 20 — GL-17.
- Naive 'level' centred-difference σ — STILL OPEN — cell 9 l.474 — GL-20.
- corr(b, H0) comment disagreement (~0.4 vs ~0.2–0.3) — STILL OPEN (comment) — GL-18.
- "Model selection justification" heading without cell — STILL OPEN (markdown) — GL-18.
- `uncapped` naming — STILL OPEN (naming only) — GL-18.
- "I1/I0 collinearity" comment in cell 18 — STILL OPEN (comment) — GL-18.
- read_glambie_global docstring "2000–2023" — NOT APPLICABLE — 24 calendar years 2000–2023 is correct.

### Cross-cluster notes
- IMBIE-3 Greenland now systematically includes peripheral GICs (Otosaka 2026: 8.0 ± 0.5 Gt/yr 1972–2023, 18.9 Gt/yr in the 2010s), while the Greenland model (Mouginot discharge + MAR contiguous-sheet SMB) excludes them; the Greenland-vs-IMBIE-3 validation (main l.124, supp l.379) compares unlike domains (GREENLAND cluster).
- Mouginot SMB total includes seven ICE_CAPS units, overlapping GlaMBIE RGI 5 by ~5–10 Gt/yr; reader attrs say peripheral glaciers excluded (mod_readers; GREENLAND/readers).
- fit_bayesian_level_annual_correlated computes no convergence diagnostics; glacier, Peninsula, EAIS and WAIS posteriors rely on it (mod_bayes_A; modules cluster).
- Four different AR6-GMST→Berkeley offsets; glacier uses 0.726 °C (Historical 1995–2005) vs ocean 0.755 °C, and AR6 annual means stamped at yyyy.0 (≈0.5-yr lead). For glaciers this is roughly +1.7 mm (offset) and +0.5 mm (stamp) at 2100 at b ≈ 0.76 mm/yr/°C (my rough estimate) (xc_constants, mod_comp, nb_glacier_B; constants cluster).
- Peninsula χ²/dof dof convention and the supp l.206 Peninsula BIC (mod_bayes_A, num_supp_1; Antarctic cluster).
- Corner-caption contour levels are wrong in the ocean, Peninsula and EAIS captions too (supp l.278/294/301).
- Supp Table S2 defines T relative to 1850–1900 while the glacier fit uses 1995–2005 (num_supp_1; supp-numbers cluster).
- HDF5 ordering for `glacier`: save_dols_component deletes and rewrites only the `glacier` group, and no later step writes into it (hdf5_write_map, xc_hdf5); REFIT = True makes the cached-load branch dead. Checked, no hazard.
- Memory note frederikse_greenland_sources.md states Frederikse peripheral glaciers are "counted in glaciers column"; Frederikse 2020 Methods says they are added to Greenland (and Antarctic periphery set to zero). Worth correcting.

---

## B3. Greenland (GR)

#### GR-1 δ BIC weights come from a diagonal likelihood. A correlation-aware likelihood reverses the lag ranking (δ = 6 preferred over δ = 5)
- Verdict: CONFIRMED
- Severity: high
- Location: notebooks/component_analysis.py fit_discharge_delay_model (`W_level = 1.0 / sig_v**2`; `log_lik = -0.5 * np.sum(W_level * resid**2 + np.log(sig_v**2))`); greenland_discharge.py:73–91 (keeps those weights after the robust correction); component_greenland cell 8
- Reported by: mod_comp (suspected), nb_greenland_A (old §1.5 "unequal n" FIXED; the diagonal-likelihood part was not checked)
- Manuscript impact: MAIN:120 claims. "places 98.8\% of the weight on $\delta = 5$~yr" changes to ≥ 0.93 on δ = 6. "The weights are computed before the corrections for correlated errors and excess scatter ..., so the 98.8\% is an upper bound" is literally true but misleading: the correction does not merely lower the confidence in δ = 5, it reverses the ranking. "$\gamma = 0.367$ [0.313, 0.421]" also changes, and so do Fig. S1 (δ panel, titles) and the Fig. 2 δ = 5 labels. The effect on MAIN:396 "93 [51, 132] mm" (Greenland, 2.7 °C) depends on the fix. With the production WLS estimator and the weight moved to δ = 6 (γ 0.398), discharge at 2100 is +2.0 mm (SSP2-4.5, 151.2 → 153.2) and +2.4 mm (SSP5-8.5), so Greenland goes to about 95 mm. With a GLS refit (γ 0.338 at δ = 6, 0.331 at δ = 5), discharge at 2100 is 150.1 mm, −1.1 mm against production, so about 92 mm. Either way the 2100 change is ≤ ~2 mm. The changes to the δ, weight and γ claims are what is material.
- Defect: The per-δ BIC treats the 41 cumulative discharge values as independent. The correlated covariance that robust_delay_intervals builds is applied only to the (γ, r0) spreads. The project rule says diagonal weights on a cumulative record are wrong. Under the correct covariance the ranking of lags changes, not only the confidence. The rate-constraint row also reuses the last decade of the same H_dyn points. It enters the point estimate but not the BIC.
- Evidence: Primary evidence: I re-ran calibrate_greenland_discharge read-only (weights 0.9883/0.0117 reproduced). I then computed the generalized χ² (resid' Σ_dm⁺ resid, Σ from `_increment_covariance`) of the production WLS residuals for δ = 4..8: 99.2, 75.0, **64.7**, 89.1, 152.0. The diagonal χ² is 40.0, **14.5**, 23.3, 73.6, 161.5. The log-det term is identical across δ (common window). Weights: 0.006/0.994 (δ5/δ6), or 0.069/0.929 after dividing by χ²/dof = 1.973. Scaling alone, with no correlation, would keep δ = 5 at about 0.90. Corroborating evidence: a GLS refit per δ (level rows only; it drops the rate-constraint row, so it is a different estimator from production) gives χ² 95.8, 71.1, **54.9**, 67.9, 101.6, and weights 0.004/0.985/0.011 (δ5/6/7) after scaling. Projection check: project_greenland_discharge with one median-path member at the production δ = 5 estimates reproduces the stored 151 mm.
- Fix: Compute the per-δ log-likelihood (or a GLS fit) with the correlated covariance of the cumulative record (equivalently, on annual increments), then redraw. Or keep the diagonal weights and replace the "upper bound" sentence with a statement that correlation-aware weighting prefers δ = 6.

#### GR-2 Pre-EN4 discharge forcing uses AA·GMST, not the observed Greenland regional T; the 2021.5–2024 gap and the Fig. 2c extension also bypass it
- Verdict: CONFIRMED
- Severity: medium
- Location: notebooks/greenland_discharge.py:139–162 (pre-EN4 proxy) and 130–137 (EN4 → projection splice); component_greenland cell 11 lines ~632–634 (`be_T_oc = ocean_transfer['alpha'] * be_ann.values + ocean_transfer['beta']`); same construction in results_figures cell 30
- Reported by: nb_greenland_A, nb_greenland_B (extension; gap), nb_figs_A (Fig. 3 extension)
- Manuscript impact: MAIN:373 hindcast numbers (via greenland/hindcast_gmst_1900 → component_summation). Estimated changes (fixed α = 0.148, γ = 0.367, AA = 2.5; not a rerun): "1950--2000 the hindcast rate is 1.11" → about 1.13; "Greenland raised it by 10 mm more than in our hindcast" → about 9 mm; "0.76 mm yr$^{-1}$ ... over 1900--1950" → about 0.81; "hindcast level at 1900 is 39--45 mm above" → about 35–41 mm. The partial-2021 gap moves 2100 projections by much less than 1 mm. The Fig. 2c extension changes only the 2022→2023 increment (~0.03 mm).
- Defect: This is an undocumented proxy substitution, not a contradiction of a stated method. Neither tex file describes how the ocean forcing is built before 1970 (searched). The transfer function T_ocean = ξ·T_gr + χ is fitted to Berkeley Earth Greenland-box T, which exists for 1850–2026. Before 1970, however, the code feeds it AA·GMST with AA = 2.5. The observed regional anomaly over 1900–1969 is −0.86 °C, against −1.80 °C for 2.5×GMST, so the proxy ocean T is too cold in every decade. The Fig. 2c extension uses bare GMST with neither AA nor regional T, contradicting its own comment. EN4 ends 2021-06 and the projected branch starts at 2024.96. So 2021 is a Jan–Jun mean (climatology −0.043 °C relative to the annual mean), 2022–2023 are linearly interpolated, and 2024 is December only. The transfer residual starts only in 2024. The same partial 2021 year enters the transfer fit (n = 52).
- Evidence: My recomputation with α = 0.148 and γ = 0.367: mean ocean-T error of the proxy 0.135 °C. Decadal values 0.17, 0.11, 0.12, 0.19, 0.12, 0.11, 0.14 °C for the 1900s–1960s. Cumulative discharge is understated by 2.51 mm (forcing years 1895–1944), 1.19 mm (1945–1969) and 3.70 mm in total. The 1900–1950 rate is understated by 0.050 mm/yr. EN4 months per year: `{2019: 12, 2020: 12, 2021: 6}`. GMST ends 2024-12.
- Fix: Drive the transfer function with observed Greenland regional T wherever it exists (pre-1970, 2021.5–2024, the cell 11 extension), and use AA·GMST only for scenario years. Drop or deseasonalize the partial 2021 EN4 year.

#### GR-3 §6 sensitivity (cell 26) reads `blended`, `blended_stable`, `blended_slow90`, which later pipeline steps write
- Verdict: CONFIRMED
- Severity: medium
- Location: component_greenland cell 26 (`_full = {s: _hf[f'blended/{s}/samples'][:] ...}`, `_stab`, `_slow_stored`, `_ba['t_origin']`, ...); run_pipeline.sh ENTRIES (greenland is step 5; forecast step 14 and slowwais step 15 write the groups)
- Reported by: nb_greenland_A, xc_hdf5
- Manuscript impact: SUPERSEDED by IN-1: main tex l.433 ("moves the 2100 median by 3 mm or less") does quote the cell 26 output "Largest change in a forecast median: 2.7 mm" (checked 2026-10-17). The original "none" was wrong.
- Defect: In a full run, the "full" and "slow" forecast deltas are computed against the previous run's blended ensembles. The cell's assert compares blended with blended_slow90, which are stale together, so it cannot detect this. On a fresh component_results.h5 the cell raises KeyError and the pipeline stops (set -e). At N_SAMPLES = 10000, the 2000-row stored blends would mismatch the arrays in shape. The crash scenarios follow from the code; I did not execute them.
- Evidence: `greenland.attrs['saved_at'] = 2026-10-17T03:02:53`; `blended.attrs['saved_at'] = 2026-10-17 03:49:14`. The code reads all three groups with no existence or timestamp check.
- Fix: Move the forecast part of the sensitivity into a script after `slowwais`, or have cell 26 report only the Greenland delta.

#### GR-4 Discharge ocean forcing is zero-filled after ~2103 instead of held flat
- Verdict: CONFIRMED
- Severity: medium
- Location: notebooks/greenland_discharge.py:176–179 (`left=np.nan, right=np.nan` then `T_delayed[np.isnan(T_delayed)] = 0.0`); cell 17 `t_mon = np.arange(min, max, 1/12)` ends at 2098.92
- Reported by: nb_greenland_A, nb_greenland_B (old §1.2)
- Manuscript impact: none at 2100 (VoI and all tex numbers use 2100; no tex figure goes beyond 2100). Stored greenland/projections* for 2104–2150 are wrong, as is the summation 2150 milestone table.
- Defect: The last annual ocean T is at 2098.5, so T_delayed is NaN after 2098.5 + δ. It is set to 0, the 1995–2005 baseline, which removes the ocean-forced part of the discharge rate. SMB, by contrast, holds GMST flat after 2099. The comment "zero before ocean data" describes leading NaNs, which never occur.
- Evidence: My HDF5 read of greenland/projections_discharge median annual increments (mm/yr). SSP2-4.5: 2103 1.622, **2104 1.399**, 2110 1.382, 2149 1.387. SSP5-8.5: 2103 1.875, **2104 1.401**.
- Fix: Hold the last annual ocean T flat (`right=T_oc_ann_proj[-1]`), or extend t_mon to PROJ_YEARS[-1] with the flat GMST before delaying.

#### GR-5 The "Greenland regional" surface-T box includes Baffin Island, Iceland and Ellesmere (~11% of the weight)
- Verdict: CONFIRMED (box content); impact not computed
- Severity: medium
- Location: notebooks/slr_data_readers.py:1505–1599 and src/slr_forecast/readers/gmst.py (defaults lat 60–84, lon −73 to −12, land ≥ 0.5); component_greenland cell 3 (`read_berkeley_earth_gridded(GRIDDED_PATH)  # defaults to Greenland box`)
- Reported by: mod_readers
- Manuscript impact: MAIN:296 "Greenland regional ... spanning 2.1 to 2.7" (cell 25 ratios), and ξ = 0.148, χ (supp transfer function, MAIN:296 "Greenland regional surface temperatures are used in Eq. ocean_transfer"). Values would change by an unknown amount. The tex never describes the box.
- Defect: The rectangular lat/lon box keeps every land cell with land fraction ≥ 0.5, so non-Greenland land enters the series that the paper calls Greenland regional surface temperature.
- Evidence: My read of the NetCDF land_mask × areal_weight: 774 cells, Baffin (lat < 74, lon < −60) 7.86%, Iceland 2.13%, Ellesmere 1.40%.
- Fix: Mask to Greenland only, or describe the box in the supplement. Then recompute ξ, χ and the AA window ratios.

#### GR-6 Drift-state σ_ε grid starts at 20 Gt/yr; the Table S2 5th percentile is the grid edge
- Verdict: CONFIRMED
- Severity: medium
- Location: notebooks/smb_state.py:18 (`SD_E = np.linspace(20.0, 250.0, 47)`)
- Reported by: nb_greenland_A
- Manuscript impact: SUPP:231 (Table S2) "$\sigma_\varepsilon$ ... 50 [25, 80]" → about 50 [10, 80] Gt/yr. Reviewer A said σ_ε is not quoted in the tex, which is wrong. τ (12 [6, 26]) is unchanged. The σ_s grid 95th percentile moves 180 → 186. The quoted member value 174 is drawn, so it may shift slightly. The effect on projections is small (it touches only the low tail of the weather noise).
- Defect: The posterior mass is still rising toward the lowest σ_ε bin, so the lower tail is truncated by the grid.
- Evidence: My recomputed grid_posterior with the production inputs: SD_E marginal first three bins 0.0395, 0.048, 0.059, quantiles [25, 50, 80]. With SD_E = linspace(0, 250, 51): first bins 0.024, 0.025, 0.027, quantiles [10, 50, 80]. τ quantiles [5.6, 12, 25.6] in both.
- Fix: Extend SD_E down to 0, rerun, and update Table S2.

#### GR-7 Main-text Fig. 2 mixes 2σ (panel b) and 1.645σ (panel c) error bars; caption silent
- Verdict: CONFIRMED
- Severity: medium
- Location: component_greenland cell 11 lines 587, 590, 595 (`yerr=2 * ...`) vs 699, 702, 707 (`yerr=1.645 * ...`); MAIN:114 caption
- Reported by: nb_greenland_A, nb_greenland_B, xc_constants, nb_figs_A (Fig. 3, not in tex)
- Manuscript impact: MAIN:113–114 figure component_greenland_discharge_delay_fit.png. The model bands are 90%, and the caption says nothing about the data bars.
- Defect: Panel (b) data bars are about 95%, panel (c) bars are 90%, and the model bands in both panels are 5–95%. The paper convention is 90%.
- Evidence: `yerr=2 * mou_comp['sigma_dyn'] * M_TO_MM` / `yerr=1.645 * mou_comp['sigma_mb'] * M_TO_MM`.
- Fix: Use Z_90 for every bar and state in the caption that the bars are 90%.

#### GR-8 γ interval: tex [0.313, 0.421] vs Fig. S1 title [0.313, 0.422] vs MC draws [0.315, 0.421]
- Verdict: CONFIRMED
- Severity: low
- Location: component_greenland_robust_se.py (`ci_robust` = δ = 5 estimate ± z·mixture sd); cell 13 (exact mixture quantiles in the title); cell 8 MC print
- Reported by: num_supp_1, num_main_1
- Manuscript impact: MAIN:120 "$\gamma = 0.367$ [0.313, 0.421]" cites Fig. S1, whose title reads [0.313, 0.422]
- Defect: Three definitions of the same interval are printed. num_supp_1 called the tex interval "δ = 5-only". It is actually a normal approximation with the mixture variance, centered on the δ = 5 estimate. The pure δ = 5 interval is [0.3130, 0.4209].
- Evidence: My recomputation: exact mixture 0.3131 / 0.3672 / 0.4216; ci_robust [0.3127, 0.4211]; MC draws [0.3152, 0.4209].
- Fix: Quote the exact mixture quantiles printed by cell 13 (0.422). GR-1 may supersede this.

#### GR-9 Main text describes the post-2018 drift state as a deterministic relaxation
- Verdict: CONFIRMED
- Severity: low
- Location: MAIN:151 ("this offset relaxes toward zero thereafter"); smb_state.py:207–208 (`delta[:, t] = phi * delta[:, t - 1] + q * rng.standard_normal(...)`); cell 4 markdown
- Reported by: nb_greenland_A
- Manuscript impact: wording only (SUPP:316 is accurate)
- Defect: After 2018 the state continues as a stochastic AR(1). Only its mean relaxes toward zero.
- Evidence: Code as quoted.
- Fix: Say the state continues as an AR(1) process whose mean relaxes toward zero.

#### GR-10 Main-text χ² defined as the "error-weighted sum of squared residuals"; the code uses the generalized χ²
- Verdict: CONFIRMED
- Severity: low
- Location: MAIN:120; component_greenland_robust_se.py (`chi2 = float(resid_lvl @ Sig_dm_inv @ resid_lvl)`)
- Reported by: nb_greenland_A
- Manuscript impact: wording only (2.0 on 38 dof is correct: 1.973). The diagonal χ²/dof would be 14.5/38 = 0.38.
- Defect: The value quoted is the χ² against the correlated covariance of the demeaned cumulative record, not a diagonal-weighted sum.
- Evidence: Code as quoted; my diagonal χ² for δ = 5 is 14.5.
- Fix: Define it as residuals weighted by the covariance of the cumulative record.

#### GR-11 Supplement LOMO paragraph: "As in the projections, the prediction is anchored to the withheld run's own 1995–2005 mean"
- Verdict: CONFIRMED
- Severity: low
- Location: SUPP:332; component_greenland cell 17 (`M0=SMB_M_draws` from the drift state)
- Reported by: nb_greenland_B, num_supp_2
- Manuscript impact: wording only
- Defect: Projections use the Kalman level c_gris plus s(t) (SUPP:316), not a 1995–2005 anchor. The LOMO script does use the anchor, which is a valid test design.
- Evidence: SUPP:332 quote; cell 6 `SMB_M_draws = SMB_STATE['M']` (388 [281, 522] Gt/yr vs observed 374).
- Fix: Delete "As in the projections".

#### GR-12 Drift-state out-of-sample test: the anchor is 1995–2004, and the state forecast is worse, not "no better"
- Verdict: CONFIRMED
- Severity: low
- Location: SUPP:350; scripts/smb_state_space_diagnostic.py:231–238 (`M0_h = smb_m[(yrs_m >= 1995) & (yrs_m <= HO)].mean()`, HO = 2004)
- Reported by: num_supp_2
- Manuscript impact: SUPP:350 "no better than a fixed 1995--2005 anchor" → "1995–2004", and arguably "slightly worse than"
- Defect: The anchor window ends in 2004. Mean SMB: state 379 vs anchor 361 vs observed 260 Gt/yr. Annual 90% coverage: 64% vs 100%.
- Evidence: Log: "M0 (1995-2004) 374"; "state 379 [249, 476]; production 361 [246, 470]"; "coverage ... state: 64% ... production: 100%".
- Fix: Correct the window and the wording.

#### GR-13 1972–1985 SMB departure: tex 79 Gt/yr, while two pipeline prints give 79 (difference of means) and 81 (residual about the median emulator)
- Verdict: CONFIRMED
- Severity: low
- Location: MAIN:151; log from plot_supp_greenland_smb_emulator.py ("1972-1985: emulator 387 Mouginot 466") vs smb_state_space_diagnostic.py ("1972-85 +81, 2010-18 -106"); cell 4 markdown "+80/−107"
- Reported by: num_main_1, nb_greenland_B (cell 4 markdown)
- Manuscript impact: MAIN:151 "about 79~Gt\,yr$^{-1}$" (traceable only as 466 − 387; no line prints 79)
- Defect: The same quantity is computed two ways, and the tex value is not printed directly.
- Evidence: Log lines quoted.
- Fix: Print the departure explicitly from one script, cite it, and update the cell 4 markdown.

#### GR-14 Fig. S(rcm) caption says its 2100 values "are those of" Fig. S(lomo), but two independent computations differ slightly
- Verdict: CONFIRMED
- Severity: low
- Location: SUPP:337 caption; plot_greenland_smb_mar_vs_emulator.py vs plot_supp_greenland_smb_emulator.py
- Reported by: num_supp_2
- Manuscript impact: caption wording (the tex quotes +28.7 from the LOMO script)
- Defect: The LOMO predictions are recomputed with different draws.
- Evidence: Log: UKESM SSP5-8.5 miss +28.7 (lomo) vs +28.8 (rcm); CESM2 SSP1-2.6 37.4 [13.7, 61.0] vs 37.6 [14.2, 60.9].
- Fix: Share one computation, or write "match up to sampling noise".

#### GR-15 SMB lag test (SUPP:314) is not produced by any pipeline step
- Verdict: CONFIRMED
- Severity: low
- Location: SUPP:314; scripts/joint_lag_fit_mar.py is not in run_pipeline.sh ENTRIES
- Reported by: num_supp_2
- Manuscript impact: an untraceable qualitative claim ("It did not improve the fits")
- Defect: Traceability rule violated.
- Evidence: ENTRIES lines 23–46.
- Fix: Add the script to the pipeline, or record it as an archived diagnostic.

#### GR-16 EN4 200–500 m series is an unweighted mean over depth levels
- Verdict: CONFIRMED (code); magnitude taken from the reviewer, not re-derived (raw EN4 zips not re-read)
- Severity: low
- Location: src/slr_forecast/readers/ocean_temp.py:163 (`T_depth_avg = np.nanmean(T_c, axis=0)`); also line ~361 (Argo)
- Reported by: mod_readers
- Manuscript impact: marginal (ξ, γ and δ refit; the forcing anomaly differs by ≤ ~0.045 °C)
- Defect: Levels 207–447 m with thicknesses of 25–83 m get equal weight, so the upper levels are over-weighted.
- Evidence: Code as quoted.
- Fix: Weight by depth_bnds thickness per valid column.

#### GR-17 AR6 annual GMST stamped Y.0: discharge sees a half-year lead relative to SMB
- Verdict: CONFIRMED
- Severity: low
- Location: component_greenland cell 17 (`t_mon = np.arange(...)`, `np.interp` from Y.0 stamps; SMB `_T = np.interp(_years_full, ...)` at integer Y); greenland_discharge.annualize
- Reported by: nb_greenland_A (suspected), nb_greenland_B, nb_apen_A (cross-cutting)
- Manuscript impact: negligible (my estimate is ~0.15 mm at 2100)
- Defect: Monthly interpolation between Y.0 stamps, re-annualized by calendar year, gives about (T_Y + T_{Y+1})/2 for discharge, while SMB takes T_Y.
- Evidence: `projections/temp/SSP2_4_5` decimal_year 2015.0, 2016.0, … 2099.0, index Jan 1.
- Fix: Stamp AR6 annual means at Y.5 (cross-cutting).

#### GR-18 Diagnostic forward model (Fig. 2b/c, discharge_diagnostic) is offset from the fitted model
- Verdict: PLAUSIBLE (mechanism confirmed; ~0.24 mm magnitude not re-derived)
- Severity: low
- Location: component_greenland cell 11 lines 439–462 and 640–657; cell 14 lines ~871–889
- Reported by: nb_greenland_A
- Manuscript impact: Fig. 2 panels (b, c) level, by a fraction of a mm; stored greenland/discharge_diagnostic (read by results_figures)
- Defect: The diagnostic cumulates T from the first year with delayed forcing (1976 for δ = 5 on an integer grid), while the fit cumulates from 1978.5. It still subtracts the fit's int_T_mean_cal. Best-δ constants are applied to the δ = 6 members, and the grid is integer instead of x.5.
- Evidence: `diag_years = np.arange(np.floor(T_ocean_years[0]) + 1, ...)`; `int_T = np.cumsum(T_shifted[valid] * dt_v)`; `H_pred = gamma_i * (int_T - result_discharge.int_T_mean_cal) + ...`.
- Fix: Start the integral at fit_results[δ]['yrs_valid'][0], use the per-δ constants, and use the x.5 grid.

#### GR-19 Stored greenland/observations are zero at label 1999.5, while the projections are zero at grid year 2000
- Verdict: CONFIRMED
- Severity: low
- Location: bayesian_models._cumulate (`bl_mid_idx = argmin|t − 2000|`, tie → 1999.5); component_io.save_greenland; consumers component_forecast (`_load_obs_only`, Fig. per-component) and results_figures cell 30
- Reported by: nb_greenland_B
- Manuscript impact: negligible (sub-mm offset in the overlay on component_forecast_per_component.png)
- Defect: The observations are rebased at the end of 1999 and the projections at the end of 2000 (SMB) or at 2000.0 (discharge). Consumers overlaying the two see a ≤ 1 mm baseline shift.
- Evidence: HDF5 smb H_obs around 1999.5: `[0.00086, 0.0, -0.00102]`; discharge `[-0.00123, 0.0, 0.00126]`; projections_smb median 1999/2000/2001: `[0.00102, 0.0, -0.00111]`.
- Fix: Store the rebase epoch as an attribute, or rebase the stored observations to the projection epoch. Break the tie explicitly.

#### GR-20 Stale markdown and comments in component_greenland
- Verdict: CONFIRMED
- Severity: low
- Location: cell 7 markdown ("fit over 1975–2018", "R² = 0.794", "γ 0.365 [0.314, 0.418]", equation omits the rate row); cell 12 markdown and cell 15 comment ("predict 1975–1990"); cell 4 markdown (+80/−107); cell 11 lines 538–545 (IMBIE-3 "not used for calibration or validation ... Independent" vs legend "IMBIE-3 (validation)"); cells 0, 5, 14 ("GRACE − D ... RCM-independent"); greenland_discharge.py:176 ("zero before ocean data")
- Reported by: nb_greenland_A, nb_greenland_B, num_supp_1, num_supp_2
- Manuscript impact: none (tex: 41 yr 1978–2018, dof 38, holdout 1978–1990 are correct)
- Defect: The text predates the common-window fit, the holdout window, and the role of IMBIE-3 in the tex.
- Evidence: Output cell 8 `R²(rate)=0.7930`, `γ = 0.367 [0.315, 0.421]`; cell 15 `withheld 1978.5–1990.5`.
- Fix: Update the strings.

#### GR-21 Cross-correlation "Motivation" diagnostic correlates the discharge rate with the ocean-T rate
- Verdict: CONFIRMED
- Severity: low
- Location: component_analysis.py fit_discharge_delay_model xcorr block (`np.corrcoef(dH_dyn[valid], dT_shifted[valid])`); cell 7 markdown
- Reported by: nb_greenland_A (old item)
- Manuscript impact: none
- Defect: The model links dH/dt to the T level. The diagnostic peaks at 11 yr (r = 0.225), outside the δ grid.
- Evidence: Output cell 8 "Cross-correlation peak: delta = 11 yr, r = 0.225".
- Fix: Correlate dH/dt with T(t − lag), or relabel the diagnostic.

#### GR-22 GRACE − D diagnostic: stale CSV built from IMBIE 2021, and cumulated from a rate anomaly
- Verdict: CONFIRMED
- Severity: low
- Location: data/processed/greenland_implied_smb.csv (mtime Mar 24) from scripts/grace_minus_discharge_smb.py (not in ENTRIES; reads `imbie_greenland_2021_Gt.csv`); readers component_greenland cells 5 and 14, smb_state_space_diagnostic.py:122; cell 14 `gd_rate_anom = gd_rate - gd_rate[bl_mask_gd].mean()`
- Reported by: xc_hdf5, nb_greenland_A
- Manuscript impact: none (diagnostic figure and log line "GRACE - D minus Mouginot ... +64, +30")
- Defect: The pipeline moved to IMBIE-3 but this input is still built from IMBIE 2021. In cell 14, GRACE − D has its 1995–2005 mean rate removed before cumulation, while the Mouginot and Mankoff curves use full rates.
- Evidence: mtimes (CSV Mar 24, script May 2); script line 60; cell 14 code.
- Fix: Regenerate from IMBIE-3 inside the pipeline, or drop the overlay. Cumulate full rates.

#### GR-23 Stale metadata, docstrings and script labels for the superseded SMB model
- Verdict: CONFIRMED
- Severity: low
- Location: component_io.save_greenland (`model_type = "smb_literature_plus_discharge_delay"`, docstring "SMB from literature"); smb_emulator.py docstring lines 19–22 ("M0 the observed 1995-2005 SMB, eps AR(1) weather noise"); EmulatorSummary `SMB_0` = mean of drift-state M (391), not the observed 374; cell 17 `rate_median` taken from the unspliced emulator; cell 3 print "Mankoff ... 1986–2024" (record ends 2023); cell 24 comment "(a=0, b·T + c)" while it prints the a prior and a = 9.682; scripts/plot_supp_greenland_smb_emulator.py:17, 42–43, 270 ("seed 602 ... as in the notebook", M0_SIGMA = 57) and :292 legend "Mouginot ... sets $c_{\mathrm{gris}}$" in the supp hindcast figure (the caption correctly says the 1995–2005 mean anchors it); smb_state_space_diagnostic.py "production" used for the fixed anchor in steps 4 and 7 but for the drift state in steps 8–9
- Reported by: nb_greenland_A, nb_greenland_B, mod_io
- Manuscript impact: none (the SUPP:343 caption is correct; the legend wording in that supplement figure is stale)
- Defect: Strings describe the retired C_T or fixed-anchor model.
- Evidence: HDF5 `greenland.attrs['model_type']`; quoted lines.
- Fix: Update the strings and labels; rename SMB_0 or store both values.

#### GR-24 results_figures cell 35 prints "Greenland discharge R²: N/A" (wrong key)
- Verdict: CONFIRMED
- Severity: low
- Location: results_figures cell 35 (`greenland['metadata'].get('r2_dyn', 'N/A')`); component_io.py:522–523 stores `posteriors/discharge.attrs['r2']`
- Reported by: mod_io
- Manuscript impact: none
- Defect: The key never exists, and `.get` hides the miss.
- Evidence: Output line 440 "Greenland discharge R²: N/A"; HDF5 r2 = 0.7930.
- Fix: Read `posteriors['discharge']['r2']`, or drop the print (the paper reports χ², not R²).

#### GR-25 Greenland tests exercise the retired GREENLAND_SMB (C_T = −300) model
- Verdict: CONFIRMED
- Severity: low (re-rated from medium: tests only)
- Location: tests/test_greenland.py (26 references to GREENLAND_SMB; e.g. lines 29–31, 285–313, 537, 1093–1131; IO roundtrip with GREENLAND_SMB)
- Reported by: tests
- Manuscript impact: none
- Defect: component_greenland does not import smb_projections. The live path is smb_emulator + smb_state, which test_smb_emulator.py and test_smb_state.py cover. The IO test does not round-trip the EmulatorSummary that is actually saved.
- Evidence: `grep GREENLAND_SMB nb_src/component_greenland_code.py` finds no match.
- Fix: Retire or mark the legacy tests, and point the IO test at EmulatorSummary.

#### GR-26 Notebook IPCC comparison plots (cells 19, 20, 22) mix baselines
- Verdict: CONFIRMED
- Severity: low
- Location: component_greenland cells 19–22 (`ipcc_med = ex['q50'][yr_idx]`; ipcc_extract does not rebase)
- Reported by: nb_greenland_B
- Manuscript impact: none (notebook figures; the tex converts explicitly, 71 → 68 mm)
- Defect: Our samples are relative to 2000 and AR6 is relative to 1995–2014, so the plots are offset by about 3 mm.
- Evidence: component_projections.ipcc_extract returns raw slc.
- Fix: Rebase one of the two, or label the baselines.

#### GR-27 Cell 24 Greenland-T vs GMST choice rests on level-space R² from the legacy diagonal-likelihood fit_bayesian_level
- Verdict: CONFIRMED (usage); impact not demonstrated
- Severity: low
- Location: component_greenland cell 24 (`better = 'Greenland T' if res_local.r2 > res_gmst.r2 else 'GMST'`); bayesian_models.fit_bayesian_level
- Reported by: mod_bayes_A
- Manuscript impact: none (not quoted in the tex)
- Defect: Level-space R² on a cumulative record with independent errors is the criterion that the project rules reject.
- Evidence: Output "Greenland T: R²=0.9569 / GMST: R²=0.9027".
- Fix: Use a correlation-aware likelihood comparison, or drop the cell.

#### GR-28 SMB–elevation feedback "truncated at zero" is a clip (5.1% point mass at 0)
- Verdict: CONFIRMED
- Severity: low
- Location: smb_emulator.draw_feedback_eps (`np.maximum(rng.normal(0.08, 0.05, n), 0.0)`); MAIN:151; SUPP:316; Table S2
- Reported by: num_main_1
- Manuscript impact: wording only
- Defect: The distribution is censored, not truncated.
- Evidence: Cell 6 "5.1% truncated at 0".
- Fix: Write "set to zero where negative", or resample.

#### GR-29 Mouginot reader metadata says peripheral glaciers are excluded, but the SMB total includes seven ice-cap units
- Verdict: CONFIRMED (contradiction between the docstring and attrs); the overlap magnitude is taken from the reviewer
- Severity: low
- Location: src/slr_forecast/readers/ice_sheets.py:248–251 (docstring: ice caps included) vs :327, :334 (attrs "excl. peripheral glaciers")
- Reported by: mod_readers
- Manuscript impact: none (≈5–10 Gt/yr, ~0.02 mm/yr double count with GlaMBIE RGI 5 in the hindcast)
- Defect: The attrs contradict the data, and there is possible double counting with the glacier component.
- Evidence: Quoted lines.
- Fix: Correct the attrs. Optionally subtract the ICE_CAPS rows when Mouginot is summed with GlaMBIE.

#### GR-30 Dead or latent Greenland code with stale constants
- Verdict: CONFIRMED
- Severity: low
- Location: component_projections.project_greenland_joint_ensemble (AA = 2.58, GMST×AA SMB; imported in cell 1, never called); bayesian_models.fit_bayesian_greenland (rate–accel penalty fed accel = 0); bayesian_models.prepare_mouginot_components (`'df': mou` unmasked; extend_with_arc reads its last row, latent if end_year is set); slr_projections.project_greenland_ensemble (GMST-forced docstring)
- Reported by: mod_comp, mod_bayes_B, mod_slr
- Manuscript impact: none
- Defect: Stale-constant and latent hazards if the code is reused.
- Evidence: Quoted lines; no callers in nb_src.
- Fix: Delete or mark deprecated; return `mou[mask]`.

#### GR-31 Table S2 symbols: τ vs main-text τ_s; s given in Gt/yr while the main-text equation is in mm SLE/yr
- Verdict: CONFIRMED
- Severity: low
- Location: SUPP:228–229; MAIN:151
- Reported by: num_supp_1
- Manuscript impact: notation only
- Defect: The symbol and units are inconsistent between the main text and the notation table.
- Evidence: SUPP:229 `$\tau$ & Relaxation timescale ...`; MAIN:151 `$\tau_{\mathrm{s}}$`.
- Fix: Rename to τ_s and give s in mm SLE/yr (or state 362.5 Gt/mm).

#### GR-R1 Cell 14 "90% CI" legend is stale
- Verdict: REFUTED
- Severity: low
- Location: component_greenland cell 14 (`label='90% CI (posterior)'`)
- Reported by: nb_greenland_A (inside the stale-labels finding)
- Manuscript impact: none
- Defect: Claimed to be a mislabeled interval.
- Evidence: The band is `fill_between(..., dyn_p5, dyn_p95)` from `np.nanpercentile(..., [5, 50, 95])`, which is a 90% interval. The label is correct apart from CI vs credible-interval wording. (The adjacent "+ SMB obs unc (2σ)" band does mix in 2σ. That is diagnostic only and not in the tex.)
- Fix: none needed.

### Old-report items
(component_greenland; from nb_greenland_A; nb_greenland_B has no old-report section; spot-checked items marked *)
- §1.2 Projections flat/unforced after ~2099 — STILL OPEN* — greenland_discharge.py:177–179 zero-fill; stored increment 1.62 → 1.40 mm/yr at 2104 (GR-4).
- §1.4 annualize_imbie labels — FIXED — component_analysis annualize_imbie `years[i] = yr + 1.0`.
- §1.5 δ BIC on unequal n — FIXED* — common window, n = 41 for every δ (dof 38). The diagonal-likelihood weighting remained (GR-1); fixed 2026-10-17 by the increment refit.
- xcorr rate-vs-rate — STILL OPEN* — GR-21.
- Ocean-T gap 2022–2024 by interpolation — STILL OPEN* — greenland_discharge.py:135 `hist_mask_oc = time_ocean_monthly < ssp_start`; EN4 has 6 months in 2021 (GR-2).
- Cell 10 (now 11) panel (c) GMST without AA — STILL OPEN* — cell 11 `be_T_oc = ocean_transfer['alpha'] * be_ann.values + ...` (GR-2).
- GRACE−D anomaly vs full-rate cumulation — STILL OPEN* — cell 14 `gd_rate_anom` (GR-22).
- Best-δ centering constants applied to all δ — STILL OPEN (diagnostics only; no effect on projections, which rebase at 2000, checked) — GR-18.
- Baseline masks `<= 2005` — FIXED* — cell 3 `(>= 1995.0) & (< 2006.0)`.
- Mixed error-bar conventions in Fig. 10 (now main Fig. 2) — STILL OPEN* — GR-7.
- IMBIE-3 legend "(validation)" vs comment — STILL OPEN* — the comment is now the stale side (GR-20).
- AA ramp assumes AA 3.0 vs 2.0 — NOT APPLICABLE* — `APPLY_AA_RAMP = False`.
- α, β independent; residual not propagated — FIXED* — cell 17 `multivariate_normal(..., ocean_transfer['cov'])` plus OC_NOISE AR(1).
- CMIP6 annual means stamped YYYY.0 — STILL OPEN* — GR-17.
- SMB zero-crossing note — NOT APPLICABLE — C_T model replaced by the emulator.
- Leading-NaN comment — STILL OPEN* — greenland_discharge.py:176 (GR-4/GR-20).
- GRACE − D "RCM-independent" labels — STILL OPEN* — cells 0, 5 (GR-20).
- diag_years integer grid — STILL OPEN* — cell 11 line 439 (GR-18).
- "Bottom-up physical derivation" unverifiable — FIXED — main tex gives the dimensional scaling (Eq. gamma_estimate).
- Ramp comment "60-year window" — FIXED* — cell 17 "fixed 20-year window".
- Cell 22 "GMST tends to outperform" / a-prior — FIXED for the GMST claim; STILL OPEN for the "(a=0 …)" wording vs printed a = 9.682 (GR-23).
- "No in situ weather stations" comment — NOT APPLICABLE — removed.
- GRACE−D "lower bound on C_T" — NOT APPLICABLE — removed.
- "90% CI" legends — NOT A DEFECT — the band is 5–95% (GR-R1). A listed it as STILL OPEN.
- smb_projections.py GREENLAND_SMB stale comment — NOT APPLICABLE for the notebook (module not imported), but tests still use it (GR-25).
- fit_discharge_delay_model docstring / markdown omit the rate constraint — STILL OPEN (markdown only) — GR-20.
- Cell 8 markdown "xcorr identifies δ / BIC confirms" — FIXED — cell 9 markdown is neutral.
- Cell 14 C_T description, C_T² prints, "conversation notes" attrs — NOT APPLICABLE — emulator.
- IMBIE-3 "Independent" comment — STILL OPEN* — cell 11 line 539 (GR-20).
- AA affects SMB hindcast pre-1980 — NOT APPLICABLE — SMB no longer uses AA.

### Cross-cluster notes
- Fig. 4(a) green dashed Greenland line does not split discharge/SMB anomalies as the caption says (results_figures cells 54–55; MAIN:448) — FIGS cluster (nb_figs_B, num_main_3).
- results_figures Fig. 3 (not in tex): GMST without AA in the Greenland extension, OLS IMBIE quadratic, mixed interval conventions — FIGS (nb_figs_A).
- AR6-to-Berkeley GMST offsets differ across components (Greenland/glacier/EAIS/Peninsula 0.726 °C vs ocean/forecast 0.755 vs ratestate 0.887–0.898) — constants/cross-cutting (mod_comp, xc_constants).
- N_SAMPLES hardcoded per notebook (component_greenland cell 1 included) — constants (xc_constants).
- IMBIE-3 converted at 360 Gt/mm vs pipeline 362.5 — constants (xc_constants).
- GlaMBIE global includes RGI 5 Greenland periphery and RGI 19 (overlap with Mouginot ice caps and IMBIE) — GLACIER (nb_glacier_B).
- Non-deterministic outputs (PDF CreationDate, fitted_at) touch the Greenland SMB PDFs — HDF5/repo (xc_hdf5).
- BASELINE_WINDOW docstring vs point rebase at 2000 (Greenland 1995–2005 mean of projections 0.27 mm) — IO/config (mod_io).
- Caller note (outside the repo): MEMORY.md "Greenland discharge fit window — 44 of 47 yrs (1975–2018); dof = 41" is stale. Code, output and tex now say 41 yr (1978–2018), dof 38.

---

## B4. Antarctic: EAIS, Peninsula, WAIS (AN)

#### AN-1 S1 extrapolation-error term fitted to an inflated ISMIP6 "std" (nanstd over 107 quantile values)
- Verdict: CONFIRMED
- Severity: high
- Location: notebooks/component_projections.py:359-397 (S1_ISMIP6_STD_COEFFS, comment block says coeffs = polyfit of `read_ipcc_ar6_component(...)['std']`); notebooks/slr_data_readers.py:3139 (`'std': np.nanstd(slc, axis=0)`); used by component_wais cells 5, 6, 17-19, 23-30 and every S1/slow-WAIS product
- Reported by: nb_wais_A, nb_wais_B, mod_readers, num_supp_2 (as known)
- Manuscript impact: main tex:307 and :309 "86 mm [$-10$, 182]" → about 86 [9, 164] mm; tex:309 "closely matching the ISMIP6 emulator range of [$-11$, 182] mm" no longer holds exactly once AN-1 is fixed. Slow-WAIS (p(S1)=0.9) totals at 2100 shift modestly: 5th pct +8 to +13 mm, 95th pct −3 to −11 mm (e.g. 2.7 °C slow90 [464, 827] → ≈[475, 819] mm); P(>1 m rel. pre-industrial) at 3.6 °C 29.9% → ≈29.0% (tex:417 "30%" may round to 29%); supp exceedance table (supp:421) slightly. Fast-WAIS numbers essentially unchanged.
- Defect: The `std` column is the population std of the 107 quantile values. The quantile grid is uneven (0, 0.001, 0.005, 0.01, 0.02 … 0.99, 0.995, 0.999, 1) and includes the sample extremes, so the tails are overweighted and the std is inflated by ~25–30% at every decade. The quadratic fitted to it gives the S1 extra spread 58.4 mm at 2100, where the WAIS emulator's actual distribution std is ~46.8 mm.
- Evidence: My read-only recomputation (verify_AN/s1std.py) on icesheets-ipccar6-ismipemuicesheet-ssp245_WAIS_globalsl.nc: 2100 nanstd 60.7 mm, quantile-function-integrated std 46.8 mm, (q95−q05)/3.29 = 45.6 mm; 2050: 9.6 / 7.6 / 7.6; 2080: 30.3 / 24.6 / 23.7. Coefficient curve: 2050 9.0, 2080 32.6, 2100 58.4 mm (tracks nanstd). S1 10k at 2100: [−9.9, 86.4, 182.4] mm, sd 58.7, parameter-only sd ≈ 6.1 mm. Rescaling the S1 deviations to the corrected sd (47.1 mm) gives [9.1, 86.4, 163.5] mm. Recombining blended_stable with the S2 members of blended at 0.9/0.1 reproduces stored blended_slow90 exactly (p5/50/95 at 2.7 °C 464/602/827), and with the corrected S1 gives 475/600/819 (verify_AN/s1impact.py; affine rescaling is an approximation).
- Fix: Compute std from the quantile function (integrate over quantile spacing) or from the 5–95% normal equivalent, refit S1_ISMIP6_STD_COEFFS, rerun wais and all downstream steps, and update tex:307/309/417 and the supp exceedance table. Also fix or drop the reader's `mean`/`std` columns.

#### AN-2 S1 (WAIS only, rel. 2000) is compared with the whole-AIS ISMIP6 emulator (rel. 1995–2014)
- Verdict: NOT A DEFECT (withdrawn 2026-10-17). The user confirmed the whole-Antarctica comparison is intended and is stated as such in the main text. The entry is kept for the record; the AN-1 interval change still applies to l.309.
- Severity: none (was high)
- Location: component_wais cell 32 (`sub_component='AIS'`, code l.2115); main tex:309
- Reported by: nb_wais_A, nb_wais_B, num_main_2
- Manuscript impact: tex:309 "86 mm by 2100, 1 mm above the ISMIP6 emulator median of 85 mm" and "closely matching the ISMIP6 emulator range of [$-11$, 182] mm". The WAIS emulator (the same file that sets the S1 extra-error term) gives 49 [−23, 127] mm at 2100 (SSP2-4.5), so S1 sits 37 mm above it, and after AN-1 S1 is ≈[9, 164] mm. Both the "1 mm above" and "closely matching" statements fail.
- Defect: The AIS file includes EAIS and the Peninsula. The close match is a coincidence of comparing different regions, compounded by the AN-1 inflation. Baselines also differ (1995–2014 vs 2000, a few mm).
- Evidence: cell 32 output "ISMIP6 emulator, AIS, SSP2-4.5, 2100: median 85 mm, 90% range [-11, 182] mm (rel. 1995-2014)". My interpolation of the quantile files: AIS q05/50/95 = −11/85/182; WAIS = −23/49/127 mm.
- Fix: Compare with the WAIS emulator, or say explicitly that the reference is the whole ice sheet on the AR6 baseline, and drop "closely matching". Rewrite after AN-1 is fixed.

#### AN-3 IMBIE-3 annual rate sigma: December value (EAIS, Peninsula) and monthly mean (WAIS) instead of the RMS that the cumulative record implies
- Verdict: CONFIRMED
- Severity: medium
- Location: component_eais cell 6 (`groupby('year_int')[...].last()`, l.197-199); component_apeninsula cell 3 (l.108-111, `[-1]`); notebooks/component_analysis.py:142 (`imbie_annual_rates`, `.mean()` of sigma, used by WAIS)
- Reported by: nb_eais_A, nb_eais_B, nb_apen_A, nb_apen_B, mod_comp
- Manuscript impact: tex:255 "$b_{\mathrm{e}} = -0.043$ [$-0.172$, 0.088]" and "reduced chi-square 0.94" (also supp skill table, supp:371 range "0.94 and 1.05"). My GLS refit (no prior, so approximate) moves EAIS b from −0.040 [−0.172, 0.092] (Dec σ) to −0.027 [−0.159, 0.106] (RMS σ), and χ²/dof from 0.943 to 0.896. So b_e ≈ −0.03 [−0.16, 0.10] and χ² ≈ 0.90, which would also move the low end of the supp:371 range. Peninsula changes are negligible (b +0.0450 → +0.0455 in GLS, SLR-positive). WAIS mean-vs-RMS is marginal (≤4.2% low in 6 years).
- Defect: IMBIE-3 monthly increments are independent with variance σ_m²/12 (first cumulative σ 0.063254 = 0.219118/√12), so the annual increment SD is sqrt(mean σ_m²). The cumulative σ is reproduced exactly by RMS (max rel. error 1e-5). December differs in 1992, 2002, 2003, 2015, 2019 and 2022, by up to 39% (EAIS), 20% (Peninsula) and 10% (WAIS). The monthly mean is low by up to 5.7%. December rates also differ from the true increments by up to 0.16 mm/yr (EAIS), but rate_obs is label-only in the fitter. The three Antarctic notebooks use three conventions, and the EAIS comment ("true per-year uncertainty") is wrong.
- Evidence: verify_AN/imbie_sig.py: EAIS end cumulative σ 1.203 mm; sqrt(Σσ²) with Dec 1.155, mean 1.191, RMS 1.203. Increments: max|mean_r − ΔH| = 2.8e-7, max|dec_r − ΔH| = 0.161 mm/yr. GLS: verify_AN/gls_ant.py (reads the raw CSV in mass-balance sign; values above are flipped to the pipeline SLR-positive convention).
- Fix: Use one helper for all three: rate = monthly mean, sigma = sqrt(mean σ_m²) (change `imbie_annual_rates` and call it from EAIS and the Peninsula). Rerun and update tex:255 and the skill table.

#### AN-4 Main tex l.298 EAIS 2100 range "−14 to −25 mm" is stale
- Verdict: CONFIRMED
- Severity: medium
- Location: main tex:298 (cf. tex:400)
- Reported by: nb_eais_A, nb_eais_B, num_main_2
- Manuscript impact: tex:298 "$-14$ to $-25$~mm" → "−13 to −24 mm" (tex:400 already says −13.1 to −24.1).
- Defect: The two passages disagree. The current pipeline gives EAIS 2100 medians −13.1 / −18.9 / −24.1 mm at 1.8 / 2.7 / 3.6 °C.
- Evidence: HDF5 eais/projections medians at 2100 (my read): SSP1-2.6 −13.1, SSP2-4.5 −18.9, SSP3-7.0 −24.1 mm; manuscript_numbers_output "EAIS median -13.1 … -24.1".
- Fix: Change to −13 to −24 mm and print the value from manuscript_numbers.py.

#### AN-5 EAIS hindcasts use the diagnostic rate–T fit, not the C_T model the forecast uses
- Verdict: CONFIRMED (code path); impact estimated
- Severity: medium
- Location: component_summation cell 10 (`eais_hind = project_component_level_ensemble(posterior_samples=ea_post, …)`, l.418; summed into total_samples l.555) and the 1900 hindcast (`ea_h20`, l.937, summed at l.1053)
- Reported by: nb_eais_B
- Manuscript impact: tex:373 "0.76 mm yr$^{-1}$ … over 1900--1950" and "39--45 mm above the reconstructions" at 1900. With the forecast model (b = −0.166, c = 0) instead of the diagnostic fit (b = −0.043, c = +0.006 mm/yr), EAIS adds ≈0.09 mm/yr over 1900–1950 and puts the 1900 level ≈6.8 mm lower, so roughly 0.76 → ~0.85 mm/yr and 39–45 → ~32–38 mm (my estimate from Berkeley annual T; together with AN-7, ~30–36 mm). 1950–2000 differs by ~2.3 mm. Budget closure (obs-based) is unaffected.
- Defect: The tex says the fit is a check and the forecast adopts C_T (tex:255), and says the hindcast forces "the combined GMST-sensitive model" (tex:359). The hindcast nevertheless uses the regression the paper does not adopt, so hindcast and forecast use different EAIS models. The code comment at l.464 calls eais_hind "retained only as a separate diagnostic", but it is summed into total_samples.
- Evidence: summation l.374 `ea_post = f['eais/posteriors/posterior_samples']`; HDF5 eais posterior b 5/50/95 = −0.172/−0.043/0.088, c median 0.0059 mm/yr. verify_AN/eais_hind.py: 1900–2000 rise diag 3.2 mm vs C_T 9.9 mm; 1900–1950 2.0 vs 6.4 mm.
- Fix: Drive the EAIS hindcast with the SMB model (C_T·T, c_e = 0, with C_T uncertainty), or state in tex:359 that the hindcast uses the diagnostic regression for EAIS. Then update tex:373.

#### AN-6 Peninsula "BIC favors the linear model" is set by the a ≥ 0 prior and a χ²-only BIC; an unconstrained quadratic is preferred
- Verdict: CONFIRMED
- Severity: medium
- Location: component_apeninsula cell 6 (l.282-304: quadratic fit has no `symmetric_a`, so a ~ Exponential with a ≥ 0; `bic = chi2_dof*dof + k ln n`; `dof_l = n-2-1-1` vs fitter `/(n - n_phys - 1)`); cell 0 markdown; supp:206 (a_p row)
- Reported by: nb_apen_A, nb_apen_B, mod_bayes_A, num_supp_1
- Manuscript impact: supp:206 "set to zero because BIC favors the linear model" is not supported. Maximum-likelihood ΔBIC(lin − quad) = +4.7 favours a decelerating quadratic (a ≈ −0.35 mm/yr/°C², SLR-positive). Main tex:249 makes no BIC claim. No printed number changes.
- Defect: Under a ≥ 0 the quadratic collapses onto the linear fit, so ΔBIC ≈ −ln 45 = −3.8 whatever the data say. The BIC also omits the σ_extra-dependent log|Σ| (σ_extra 0.048 vs 0.040 mm/yr), and the dof back-conversion is off by one. The production cell-6 ΔBIC print is missing from the stored output (truncated inside the tqdm stream after "Priors: a~Exp…"), so the claim cannot be traced. EAIS by contrast: a is symmetric there and ML ΔBIC = −3.0 (linear preferred), so the EAIS conclusion holds, though its printed −3.8 is also just −ln n (see AN-11).
- Evidence: My profile-likelihood GLS (verify_AN/gls_ant.py; free σ_extra, log-det included): Peninsula ΔBIC(lin−quad) = +4.74 (Dec σ), +4.61 (RMS σ), |a| = 0.35. Cell 19 stored ΔBIC −3.5/−3.1/−3.4. Stored cell 6 output ends at " 65" after the order-2 prior line.
- Fix: Compute BIC from the maximized full likelihood (as `time_model_bic` does for WAIS), with a symmetric a prior. Then either justify a_p = 0 on other grounds or state in supp:206 that the unrestricted fit prefers negative curvature. Pass progress=False so the cell-6 prints survive.

#### AN-7 1900 hindcast sets WAIS to zero before 1950 (2 mm step), and tex:359 says WAIS "follows its fit"
- Verdict: CONFIRMED
- Severity: medium
- Location: component_summation cell 11 (l.1009-1017, `wais_samp_20 = np.zeros(...)`, filled only from idx_1950); main tex:359
- Reported by: nb_wais_B, nb_summation_A, nb_summation_B, num_main_2, num_main_3
- Manuscript impact: tex:373 "39--45 mm" at 1900 is ≈2 mm too high (→ ~37–43 mm from this item alone); the 1900–1950 rate changes by <0.01 mm/yr in a level-difference sense but the 1940–1950 decade absorbs a 2 mm step. tex:359 wording is wrong.
- Defect: The WAIS member level relative to 2000 is −2.06 mm (median) from 1950 to 1980, because the sampler holds the first IMBIE level before 1979. Summation then zero-fills 1900–1949, which produces a non-physical 2 mm fall at 1950. Before 2024 WAIS is the IMBIE-3 record plus correlated errors, not a fit (tex:344 says this correctly).
- Evidence: HDF5 wais_2k/full_samples median 1950 −2.06, 1979 −2.06, 1980 −2.06, 2000 0.00, 2024 8.98 mm. Stored 1981–1983 medians −2.080/−2.016/−1.880 against obs −2.083/−2.021/−1.885 (the record, not a smooth fit).
- Fix: `np.interp` over all hind20_years (clamps to the 1950 value). Reword tex:359 to "WAIS follows the IMBIE-3 record from 1979, with zero rate before".

#### AN-8 build_slow_wais_p90.py will abort (or silently subsample) at N_SAMPLES=10000
- Verdict: CONFIRMED (by code; not executed)
- Severity: medium
- Location: scripts/build_slow_wais_p90.py:38 (`N_OUT = 20_000`), :66-74 (`assert rem == 0`), :50-63 (`s2_row_index`, `extra <= 0` branch)
- Reported by: scripts_down
- Manuscript impact: none now; all slow-WAIS numbers (blended_slow90/wais_slow90) in the planned 10k final run.
- Defect: S1 rows = 0.9·20000 = 18000 are tiled from blended_stable. With 10000 S1 members, divmod gives rem = 8000 and the assert fails. If only S2 grows above 2000 members, the S2 branch is silently thinned to 2000 rows.
- Evidence: code as quoted; at N=2000 rem = 0 (18000/2000 = 9).
- Fix: Derive row counts from branch sizes (e.g. 9·n_S1 S1 rows and n_S1 S2 rows) or store weights instead of a fixed N_OUT.

#### AN-9 Supplement corner captions (Peninsula, EAIS): contour levels, parameter order, "informative" prior, notation
- Verdict: CONFIRMED
- Severity: medium
- Location: supp:294 (Peninsula), supp:301 (EAIS) (same contour sentence at supp:279, :287); component_apeninsula cell 10 (corner.corner without `levels`, labels l.539-543); component_eais cell 10
- Reported by: nb_apen_A, nb_apen_B, mod_io, num_supp_1
- Manuscript impact: supp:294 and :301 "contours denoting 1, 2, and 3 standard deviations" → corner 2.2.3 draws 0.5, 1, 1.5, 2σ. supp:294 "The fourth parameter is … σ_extra" → it is the third (order b, c, σ_extra, H0). supp:301 "Only b_EAIS has an informative prior … C_T = 60 ± 20" → the prior is N(−0.17, 0.5²) mm/yr/°C (width ≈181 Gt/yr/°C, 9× the literature ±20), which tex:255 calls "weak". Notation b_AP/H_AP0/b_EAIS differs from b_p, H_p,0, b_e in the main text and Table S2.
- Defect: Captions describe the figures and priors inaccurately.
- Evidence: installed corner 2.2.3 `levels = 1.0 - np.exp(-0.5 * np.arange(0.5, 2.1, 0.5) ** 2)`; no `levels` in either call; Peninsula labels `[b_AP, c_AP, σ_extra, H_AP0]`.
- Fix: Pass `levels=1-np.exp(-0.5*np.array([1,2,3])**2)` (and re-render) or reword; "third parameter"; "weakly informative prior N(−0.17, 0.5²) whose mean is C_T converted"; relabel figure axes to p/e notation.

#### AN-10 Main-text Fig. wais_p_s1_sweep: AR6 AIS curves on the 1995–2014 baseline under a "relative to 2000" axis; LARMIP-2 curve is the _TOT_ (dynamics + SMB) file
- Verdict: CONFIRMED
- Severity: medium
- Location: component_wais cells 5 and 19 (xlabel 'WAIS SLR at 2100 (m; relative to 2000)'; AR6 p-box/workflow quantiles overlaid unshifted; LARMIP read from `..._TOT_globalsl.nc`); main tex:313-315 (figure and caption)
- Reported by: nb_wais_A, nb_wais_B
- Manuscript impact: tex:314 caption says "(relative to 2000)" and presents the overlays as "IPCC AR6 model ensembles for the whole Antarctic Ice Sheet" without stating their 1995–2014 baseline (offset a few mm); the LARMIP-2 curve is the workflow total (dynamics + SMB; per nb_wais_A, 137 [14, 546] mm vs 169 [49, 577] mm for the AIS-dynamics file at 2100), which the caption does not say. ±2σ IMBIE bars in panel (c) are not described.
- Defect: Overlays and labels mix baselines and content without saying so in a manuscript figure.
- Evidence: caption text at tex:314; cell 19 `xlabel = 'WAIS SLR at 2100 (m; relative to 2000)'`, `yerr=2 * wais_sigma`; LARMIP path `_TOT_` (reviewers; not re-read by me).
- Fix: State the AR6 baseline and the LARMIP total in the caption (or rebase the curves), and use or state the error-bar convention.

#### AN-11 EAIS quadratic a prior: Exponential-mean calibration used as a Normal σ; ΔBIC is −ln n by construction
- Verdict: CONFIRMED
- Severity: low
- Location: component_eais cell 6 (`PRIOR_SCALE_A = calibrate_exponential_prior(0.10, 0.10/M_TO_MM)` with `symmetric_a=True`); cell 19; bayesian_models.py:1762 (returns μ = −threshold/ln p)
- Reported by: mod_bayes_A, nb_eais_A
- Manuscript impact: none (tex makes no EAIS BIC claim; the EAIS fit is diagnostic).
- Defect: μ = 0.043 mm/yr/°C² is used as the σ of N(0, σ), giving P(a > 0.1) = 1.1% instead of the calibrated 10%. The prior is much narrower than the likelihood (GLS sd of a ≈ 0.28), so posterior a ≈ 0, χ²_q = χ²_l, and the printed ΔBIC is exactly −ln 45 = −3.8 in cell 6 and all taper rows. ML gives ΔBIC −3.0, so "linear preferred" still holds, but the print does not test it. P(a>0) prints are uninformative.
- Evidence: output "Priors: a~N(0,0.04 mm/yr/°C²)"; "ΔBIC = -3.8 (linear preferred)"; my GLS ML ΔBIC −2.98 (|a| 0.26).
- Fix: Set σ from a Normal quantile (e.g. 0.1/1.645) or compute BIC at the ML.

#### AN-12 EAIS fit print reports b prior N(0, 0.50); the sampled prior is N(−0.166, 0.50)
- Verdict: CONFIRMED
- Severity: low
- Location: notebooks/bayesian_models.py:3203 (`b_prior_str` hard-codes 0 when symmetric_b); docstring of `prior_b_mean` ("b >= 0 enforced … regardless"); component_eais cell 6 output
- Reported by: nb_eais_A, nb_eais_B, num_main_2, num_supp_1, mod_bayes_A
- Manuscript impact: none (tex:255 states N(−0.17, 0.5²) correctly).
- Defect: The stored output misreports the prior. The log-prior uses prior_b_mean.
- Evidence: output l.14 "Priors: b~N(0,0.50 mm/yr/°C)"; code `PRIOR_B_MEAN = -EAIS_SMB.C_T / 362500.0`.
- Fix: Print prior_b_mean in the symmetric branch; fix the docstring.

#### AN-13 IMBIE-3 is in mm at 360 Gt/mm while tex says 362.5 "throughout"
- Verdict: CONFIRMED
- Severity: low
- Location: main tex:102 ("1~mm SLE = 362.5 Gt throughout this study"); slr_data_readers.py:2644 (read_imbie3 docstring "360 Gt = 1 mm SLE"); component_eais cells 3, 6 (`/362500.0` hard-coded); config GT_TO_M_SLE
- Reported by: nb_eais_A, nb_eais_B, num_main_1, xc_constants
- Manuscript impact: tex:102 wording; numbers change ≤0.7% (EAIS prior mean, Davison correction).
- Defect: All IMBIE-3 series enter in mm converted by the providers at 360 Gt/mm, while Gt quantities (C_T, prior mean) are converted at 362.5 and combined with them. The notebook also duplicates the constant.
- Evidence: reader reads the mm columns; docstring quoted above.
- Fix: Qualify tex:102 ("except IMBIE-3, reported at 360 Gt/mm") or rescale IMBIE-3 by 360/362.5; import GT_TO_M_SLE.

#### AN-14 Fast-WAIS ensemble realized at p(S1) = 0.0975; full and S1 2k ensembles use independent downsample indices
- Verdict: CONFIRMED
- Severity: low
- Location: component_wais cell 31 (code l.2051-2060: `_idx_full_ds` and `_idx_s1_ds` from separate `_rng_ds.choice` calls); notebooks/compute_evpi.py:54 (`P_S1_BLENDED = 0.10`); scripts/variance_shares.py
- Reported by: nb_wais_B, nb_forecast_A, scripts_down, num_supp_2, nb_wais_A (as benign)
- Manuscript impact: small. Fast-WAIS numbers are at p(S1) = 0.0975 (195/2000; the 10k parent has 932 = 9.3%). EVPI at 2100 29.7 → 28.8–29.65 B$/yr depending on consistency (scripts_down/nb_forecast_A recomputation, not re-derived here). supp:354 fast-WAIS variance shares 98.3/97.0/93.3% (realized) vs 98.9/97.5/94.2% at exactly 0.10.
- Defect: compute_evpi documents 0.10 as "how blended was built", and the S1 rows of blended are not the rows of blended_stable (independent indices break the shared pre-anchor history pairing). Fast and slow variance shares use different methods.
- Evidence: my read `wais_2k/scenario_idx` S1 fraction 0.0975; code l.2056-2057.
- Fix: Stratify the scenario assignment (exactly 200/2000) and use one index set for both downsamples, or have consumers read the realized fraction.

#### AN-15 Post-2100 WAIS trajectories are unconstrained extrapolations
- Verdict: CONFIRMED (behaviour); manuscript impact PLAUSIBLE only through the VoI surface
- Severity: low
- Location: component_projections.sample_a4_wais_trajectories (power law continues past the 2100 pin; `_s1_ismip6_std_m` quadratic fitted on 2020–2100 only); final cap at 3.3 m
- Reported by: nb_wais_B
- Manuscript impact: none for quoted 2100 numbers; supp:534 VoI surface over arrival years to 2150 may use post-2100 values (not traced).
- Defect: S2 median at 2150 is 2.8× its 2100 value, 7.4% of all 2k members sit at the 3.3 m cap in 2150, and S1's extra sd grows by polynomial extrapolation so 13% of S1 members are below zero at 2150 (S1 2150 [−84, 177, 438] mm). WAIS is also SSP-independent after 2100, while tex:468 says emissions matter for WAIS beyond 2100.
- Evidence: my read of wais_2k / wais_s1 at 2150 as quoted.
- Fix: Mark post-2100 values as illustrative and exclude them from reported numbers, or define an explicit post-2100 rule.

#### AN-16 Rheology leftovers and Mode A defaults
- Verdict: CONFIRMED
- Severity: low
- Location: component_projections.py:582, :752, :859 (`rheology_mode='A'` defaults, R ~ N(1.28, 0.07)); component_io.py:584 (save_wais default 'A', factor 1.28); component_wais cell 6 ("S2 post-correction … 319 mm [109, 1670]", untruncated S2 draw), cell 26 legend "n ~ N(4.1, 0.4²)", cell 30 rheology sensitivity without `upper=WAIS_MAX_SLE_MM`; HDF5 wais attrs rheology_factor_median 1.28; tests/test_wais_sampling.py (fixtures use the default Mode A)
- Reported by: nb_wais_A, nb_wais_B, mod_comp, mod_io, tests
- Manuscript impact: none (all pipeline calls pass 'B'; R = 1).
- Defect: Deprecated correction remains the default and is printed and stored as if current. Tests verify the Mode A path, not production. Any new caller omitting the argument gets S2 ×1.28.
- Evidence: signatures as listed; output "S2 post-correction: median = 319 mm [109, 1670]"; test l.115/224 call without rheology_mode.
- Fix: Default to 'B' or remove Mode A; delete the cell 6/26/30 rheology prints; store R = 1; update tests.

#### AN-17 Peninsula cell 17 overwrites global `n` (45 → 38) before cell 19's BIC
- Verdict: CONFIRMED
- Severity: low
- Location: component_apeninsula cell 17 (l.997-1000 `n = 0` / `n = stats_s['n_models']`); cell 19 (l.1034-1037)
- Reported by: nb_apen_B
- Manuscript impact: none (f_max table not in tex).
- Defect: The f_max table uses dof 34/33 and ln 38 instead of 42/41 and ln 45.
- Evidence: cell 17 output last row n = 38; cell 19 uses `n`.
- Fix: Rename the loop variable or use len(pen_year).

#### AN-18 Peninsula f_max refits (cell 19) and cell 20 omit symmetric_b=True
- Verdict: CONFIRMED
- Severity: low
- Location: component_apeninsula cell 19 (l.1022-1031), cell 20 (l.1075-1079)
- Reported by: nb_apen_A, nb_apen_B
- Manuscript impact: none. tex:249 "2.64" holds (GLS with flat prior also gives 2.640; b far from the bound).
- Defect: The sensitivity runs use b ~ HalfNormal (b ≥ 0), not the production symmetric prior, so the f_max = 1 row gives b = 0.0497 instead of 0.044. Cell 20 prints "b~HN(2.0)".
- Evidence: stored cell 19 row "1 0.0155 0.0497 …"; cell 20 "Priors: b~HN(2.0 mm/yr/°C)".
- Fix: Add symmetric_b=True to both calls.

#### AN-19 S1 endpoint sampler anchors at 2023.5 and does not splice onto the observed 2024 level
- Verdict: CONFIRMED
- Severity: low
- Location: component_projections.py:395 (`S1_ISMIP6_FIT_ANCHOR_YEAR = 2023.5`), sample_a4_wais_endpoint
- Reported by: nb_wais_A, nb_wais_B, mod_comp
- Manuscript impact: sub-mm (Fig. wais_p_s1_sweep, λ sensitivity, heatmap use the endpoint path).
- Defect: The constant predates end-of-year labels (anchor now 2024.0). The endpoint path uses H_quad(2024) = 9.44 mm instead of the observed 8.98 mm.
- Evidence: constant as quoted; cell 17 "Anchor: year=2024".
- Fix: Set the constant to 2024.0 or pass the anchor explicitly.

#### AN-20 Peninsula stored cell 6 output truncated; b_p/c_p not printed by any stored output
- Verdict: CONFIRMED
- Severity: low
- Location: component_apeninsula_outputs.txt CELL 6 (ends " 65" after the quadratic prior line)
- Reported by: nb_apen_A, nb_apen_B, num_supp_1
- Manuscript impact: traceability only. tex:249 b_p = 0.044 [−0.022, 0.106], c_p = 0.056 [0.039, 0.072] match HDF5 (my read: b −0.0224/0.0440/0.1064, c 0.0393/0.0562/0.0724). Peninsula σ_extra posterior is not saved to HDF5.
- Defect: Linear/ΔBIC/P(a>0) prints are lost; numbers cannot be traced to printed output.
- Evidence: grep of the outputs file.
- Fix: progress=False in cell 6, and print b_p/c_p (and σ_p,extra) in manuscript_numbers.py; save sigma_extra_posterior.

#### AN-21 Peninsula hindcast/closure re-run without anchor_to_observed
- Verdict: CONFIRMED (code); magnitude from reviewer, consistent with the code
- Severity: low
- Location: component_summation cell 10 (`apen_hind`, l.426) and `ap_h20` (l.943)
- Reported by: nb_apen_B
- Manuscript impact: <0.1 mm (2024 median 1.58 vs anchored 1.67 mm).
- Defect: The forecast uses the anchored stored projection, while hindcast and closure use an unanchored re-run, so the Peninsula curve differs between notebooks. Stored projections are rebased to the model's own 2000 level (≈0.35 mm off the obs before 2000).
- Evidence: no anchor call in summation l.426-431.
- Fix: Read the stored samples or apply the same anchor.

#### AN-22 ISMIP6 comparisons (EAIS, Peninsula) rebased at 2016 under a "relative to 2000" label; table [-1] is 2101
- Verdict: CONFIRMED
- Severity: low
- Location: component_eais cells 13-17; component_apeninsula cells 14-17; component_projections.ismip6_ensemble_stats
- Reported by: nb_eais_A, nb_eais_B, nb_apen_A, nb_apen_B
- Manuscript impact: none (not in tex).
- Defect: ISMIP6 runs start in 2016, so argmin(|t−2000|) picks 2016; common grid from the first run ends at 2101; n counts runs, not models.
- Evidence: reviewers' time ranges {(2016, 2101, 86), (2016, 2100, 85)}; cell 17 header "relative to 2000".
- Fix: Label ISMIP6 as relative to 2016 and index with argmin(|t−2100|).

#### AN-23 SMB projection rebase relies on |ΔT| being smallest at 2000 (no baseline_year)
- Verdict: CONFIRMED (fragile; correct in this run)
- Severity: low
- Location: component_eais cell 12 (`project_smb_ensemble` without baseline_year)
- Reported by: nb_eais_A, nb_eais_B
- Manuscript impact: none (samples exactly 0 at 2000).
- Defect: The zero is placed at argmin|median ΔT|, which coincides with 2000 only for the current forcing.
- Evidence: reviewers: zero-crossing year 2000.0 (T = 0.0096 °C); my read: EAIS medians 0 at 2000.
- Fix: Pass baseline_year=BASELINE_YEAR.

#### AN-24 Stale markdown, comments, prints and metadata (Antarctic notebooks and modules)
- Verdict: CONFIRMED (spot-checked)
- Severity: low
- Location: component_wais cells 0, 4, 24 ("median 93, [-3, 191]", "93–1300 mm"); cell 17 "Median beta: 1.93" (diluted by S1 zeros); cell 9 pre/post-break print labels end-of-year years as calendar years; cell 31 wais_s1 description "Linear ramp"; cell 19 duplicate WAIS_VAF_M = 3.3 with a different citation; cell 33 reference table omits Otosaka et al. 2026; component_projections.py comment blocks (high_mm=1000, "94-1000 mm", fit_bayesian_level/robust covariance, n ~ N(4.1, 0.4²), "Martin et al. (in press)"); component_eais cells 0/5/6/10/18 (b = −0.016, R² 0.23, "mid-year centers 1979.5–2023.5", χ² "~0.91-0.93", unreproducible "sign-flip" b = +0.024…+0.026 / −0.0295, cell 3 "+0.166 (negative = SL fall)", unused check_convergence import); component_apeninsula (χ² "0.98", "BIC confirms", P(a>0)=100% fixed by prior, ISMIP6 legend entry with no curve, t_start ≈ 1979.5, glaciers "no σ_extra"); HDF5/IO model_type 'trend_only' (EAIS, test_eais.py:228 locks it) and 'linear_dols' (Peninsula); tests hard-code BASELINE_YEAR = 2005; s1_quadratic_fit.json fitted_at rewritten every run; notebook figures draw ±2σ IMBIE bars next to 90% bands; REFIT=False breaks cell 19 in EAIS and Peninsula
- Reported by: nb_eais_A, nb_eais_B, nb_apen_A, nb_apen_B, nb_wais_A, nb_wais_B, mod_comp, mod_io, tests, xc_constants
- Manuscript impact: none (tex has the current values: 86 mm, [−10, 182] subject to AN-1, 0.97, −0.043).
- Defect: Text and labels describe superseded fits, values or methods.
- Evidence: e.g. component_wais code l.16 "median 93, [-3, 191]" vs cell 18 output 0.086 [−0.010, 0.182] m; outputs "Median beta: 1.93", "Mean residual pre-break (1980-2009)"; eais code l.9-14, 147; apeninsula l.807 `ax_sl.plot([], [], 'k--', …, label='ISMIP6')`.
- Fix: Update or remove; print values from code.

#### AN-25 Wording: S2 "starts at the observed rate"
- Verdict: CONFIRMED
- Severity: low
- Location: main tex:318; component_projections S2 blend (sigmoid weight 0.90 at the 2024 anchor)
- Reported by: nb_wais_B
- Manuscript impact: tex:318 wording only.
- Defect: 10% of the power-law rate enters at the anchor and the 2100 rescale multiplies the path, so high-endpoint members start well above the observed rate.
- Evidence: my read of wais_2k: S2 first-year (2024→2025) rate percentiles 0/50/99/100 = 0.37/0.51/1.09/3.59 mm/yr vs S1 0.39/0.55/0.70.
- Fix: "starts near the observed rate", or force w(anchor) = 1.

#### AN-26 Latent: `_anchor_covariance` puts 1999 on the wrong side of the 2000 anchor with end-of-year labels
- Verdict: PLAUSIBLE (mechanism confirmed; no production caller)
- Severity: low
- Location: notebooks/component_levelspace_robust_se.py:99-100; callers only in inactive FIT_SPACE='level' branches and the uncalled results_figures `_wls_poly_fit` (l.7482/7516)
- Reported by: mod_comp
- Manuscript impact: none.
- Defect: After the zero-σ 2000 row is dropped, 1999.0 and 2001.0 tie in argmin and 1999 is picked as the anchor, so it is grouped with post-2000 points (wrong, possibly non-PSD covariance).
- Evidence: code `anchor = int(np.argmin(np.abs(years - baseline_year))); pre = np.arange(n) < anchor`.
- Fix: Split on `years < baseline_year` explicitly.

#### AN-27 CMIP/AR6 annual GMST stamped at Y.0 and interpolated onto monthly grids (≈0.5 yr phase)
- Verdict: PLAUSIBLE
- Severity: low
- Location: component_apeninsula cell 12, component_eais cell 12 (same pattern in other components)
- Reported by: nb_apen_A, nb_apen_B
- Manuscript impact: <0.1 mm for the Peninsula/EAIS; not demonstrated elsewhere.
- Defect: If the values are calendar-year means, warming is advanced by half a year relative to the BE record. Not traced to the writer of projections/temp.
- Evidence: reviewers: decimal_year integer-valued (1950.0 …, 2099.0).
- Fix: Confirm the convention; if annual means, stamp Y.5.

### Corrections to reviewer evidence (no finding was REFUTED)
- nb_eais_B's supporting claim that the IMBIE cumulative σ grows as the monthly-MEAN σ implies (1.191 vs published 1.203 mm) is incorrect: the exact match is the RMS (1.203; max rel. error 1e-5), and the mean (the WAIS convention) is itself low. nb_eais_B's December-σ defect itself is CONFIRMED (AN-3).
- num_supp_2 / nb_wais_B cite supp L362/L373 for the S1 interval [−10, 182]; the supplement does not contain that interval (grep). Only main tex:307/309 carry it. Impact narrowed accordingly in AN-1.
- nb_wais_A "independent downsample indices — NOT APPLICABLE (benign)": not benign; the pairing loss and realized p(S1) affect EVPI and variance shares slightly (AN-14).
- nb_apen_B BIC ΔBIC +4.7 and nb_apen_A: confirmed; the dof back-conversion part is Peninsula/glacier only. EAIS uses dof n−3 for linear, matching the fitter (no EAIS dof mismatch).

### Old-report items
component_eais:
- §1.1 ISMIP6 exp→SSP mapping — FIXED — component_projections.py:1714 ISMIP6_SSP_ANALOG.
- §1.2 flat after 2099 — NOT APPLICABLE (held flat to 2100 by convention; 2100 read with argmin).
- §1.4 annualization .first() — STILL OPEN as .last() (December) — cell 6 (AN-3).
- §1.4 yr+0.5 labels — FIXED — component_analysis.py:101 `yr + 1.0`.
- §1.5 BIC not like-for-like — STILL OPEN (ΔBIC = −ln n via the a prior) — AN-11.
- Sign-flip check numbers unreproducible — STILL OPEN — AN-24.
- Prior print b~N(0,…) vs −0.166 — STILL OPEN — AN-12.
- Cell 3 "+0.166 (negative = SL fall)" — STILL OPEN — AN-24.
- ISMIP6 rebased at 2016 / [-1] = 2101 — STILL OPEN — AN-22.
- project_smb_ensemble without baseline_year — STILL OPEN — AN-23.
- check_convergence imported unused — STILL OPEN — AN-24.
- Cell 19 not REFIT-guarded — STILL OPEN — AN-24.
- Stale full-record numbers in markdown — STILL OPEN — AN-24.
- Cell 18 "mass-loss-signed" — FIXED (now "mass-gain-signed").
- Docstring "b >= 0 enforced regardless" — STILL OPEN — AN-12.
- Cell 6 "symmetric, flat Normal" — FIXED.
- Cell 6 prior-sweep numbers stale — STILL OPEN — AN-24.
- plan_glacier_ratespace reference — FIXED.
- Cell 7 legacy robust_level_intervals σ — NOT APPLICABLE (legacy path inactive).
- Raw IMBIE in budget closure — FIXED (tex:344, summation obs path).
- Within-EAIS circularity comment — STILL OPEN (comment only) — AN-24.
- Markdown cell 16 omits ISMIP6 comparison — STILL OPEN (low) — AN-24.
- "DOLS" labels — NOT APPLICABLE (terminology).
- Stale EAIS R² test — FIXED (test asserts χ²); model_type 'trend_only' test remains — AN-24.
component_apeninsula (A and B agree):
- ISMIP6 mapping — FIXED. SSP2-4.5 RCP8.5 fallback — FIXED.
- §1.2 flat after 2099 — FIXED (t_mon to 2150).
- §1.4 annualization — PARTLY FIXED (labels) / STILL OPEN (December σ) — AN-3.
- §1.5 BIC construction, dof mismatch — STILL OPEN — AN-6.
- Cell 19 omits symmetric_b — STILL OPEN — AN-18.
- σ_extra corner label units — FIXED.
- CMIP vs BE offset window — FIXED; Y.0 phase — STILL OPEN (suspected) — AN-27.
- ISMIP6 legend entry — STILL OPEN — AN-24.
- Year-range print — FIXED.
- P(a>0)=100% by prior — STILL OPEN — AN-24.
- ISMIP6 2016 baseline / 2101 — STILL OPEN — AN-22.
- emcee seeding — FIXED.
- rate_obs unused — NOT APPLICABLE (label-only by design).
- Cell 7 legacy 'level' path — NOT APPLICABLE.
- plan_/handoff_ references — FIXED.
- Comment χ² numbers (0.99 → now 0.98 vs 0.975) — STILL OPEN — AN-24.
- Stale Peninsula SSP1-2.6 test — FIXED (other stale constants remain, AN-24).
component_wais (nb_wais_A only; spot-checked):
- S1_QUADRATIC_COV not PSD — FIXED (np.cov of MCMC draws).
- "Robust" interval narrower than naive — FIXED.
- Cell 6 S2 post-correction ×1.28 — STILL OPEN — AN-16.
- n ~ N(4.1, 0.4²) label (cell 26) — STILL OPEN — AN-16; old cell-31 print — FIXED.
- mix_n3_* unused — FIXED.
- save_wais rheology_factor 1.28 — STILL OPEN — AN-16.
- wais_s1 "Linear ramp" — STILL OPEN — AN-24.
- Median β 1.93 — STILL OPEN — AN-24.
- H0 prior on first observation — FIXED.
- Print years / anchor 2024 — FIXED.
- S2 blend zero growth 2023.5→2024 — NOT APPLICABLE.
- LARMIP-2 _TOT_ under AIS label — STILL OPEN — AN-10.
- S1-only vs full 2k independent indices — STILL OPEN (not benign) — AN-14.
- §1.4 December labeled yr+0.5 — FIXED.
- §1.5 BIC (WAIS) — FIXED (time_model_bic with log-det).
- component_projections stale comments (high_mm=1000, 94-1000, cell 11, n~N(4.1), Martin in press) — STILL OPEN — AN-24.
- Cell 1/22/25/9/17/29 comments, cells 20/30 markdown — FIXED except "93 mm" floor — AN-24.
- Cell 28→30 rheology sensitivity — STILL OPEN — AN-16.
- Otosaka 2026 missing from reference table — STILL OPEN — AN-24.
- Cell 0 S1 table value — STILL OPEN (93 [-3, 191]) — AN-24.
- AR6 1995–2014 baseline vs "relative to 2000" in cells 5/19 — STILL OPEN — AN-10.
- Unchecked literature/sampling claims (van den Akker/Naughten, ±11 mm stability, segment accelerations, "~300 mm higher") — NOT CHECKED.

### Cross-cluster notes
- AN-5/AN-7 feed the summation 1900 hindcast; summation cluster should merge with "pre-1950 thermosteric lacks below-2000 m term" (nb_summation_A/B, num_main_3) since all three move tex:373.
- Summation WAIS share uses clamped 2024.5 end value (nb_summation_B) and two code paths 9.1% vs 9.9% (num_main_3): summation/numbers cluster.
- manuscript_numbers S2-vs-AR6 AIS crossing on mixed baselines, "until about 2050" (scripts_down): scripts/numbers cluster (WAIS-related).
- build_slow_wais_p90 variance-share print on a third basis; slow_wais_p90_numbers stale line refs (scripts_down): scripts cluster.
- compute_evpi D_GRID lower edge (scripts_down): scripts/VoI cluster.
- results_figures: slide labels p(S1)=0.9 mixture "stable Thwaites"; fig_slr_by_component v1 "Slow WAIS" = pure S1; Fig 3 panel h OLS polyfit on IMBIE; IMBIE-3 QC block compares IMBIE-3 with itself (nb_figs_A/B): figures cluster.
- Per-component projection figure adds the total IPCC offset to every component (nb_figs_B); IPCC offset sign (xc_constants, nb_summation): baseline cluster.
- ratestate σ_ice uses WAIS samples to 2150 while model ends at 2099 (nb_ratestate_A/B): ratestate cluster.
- IMBIE-3 monthly cumulative stamped mid-month but is end-of-month (mod_readers); raw/df_imbie_wais negative sigmas (mod_readers): readers cluster (no Antarctic pipeline effect found).
- build_level_design_vectors / project_component_level_ensemble nearest-month tie at Y.0 labels (mod_bayes_A, mod_slr): modules cluster (half-month effect on EAIS/Peninsula).
- Corner "1, 2, 3 σ" captions also wrong for ocean (supp:279) and glacier (supp:287): ocean/glacier clusters.
- σ_extra not counted in χ²/dof denominator (Peninsula 0.975 → 1.00; supp:371 range) (mod_bayes_A): modules/supp cluster.
- Peninsula skill-table row omits σ_extra (num_supp_2); Table S2 β = 2 listed as fixed (num_supp_1); Table S2 T reference 1850–1900 (num_supp_1): supplement cluster.
- N_SAMPLES hard-coded per notebook (xc_constants): constants cluster; relevant to AN-8.
- Greenland §6 reads blended* before they are written (xc_hdf5): HDF5/Greenland cluster.

---

## B5. Summation, forecast and downstream scripts (SF)

#### SF-1 Pre-1950 hindcast omits the below-2000 m ocean term (and sets WAIS to 0 before 1950); 1900–1950 tex numbers change
- Verdict: CONFIRMED
- Severity: high
- Location: component_summation_code.py cell 11 l.980–988 (ocean), l.1009–1017 (WAIS); consumers cell 30 (l.2207–2217), results_figures Fig. 1 (hindcast_20c)
- Reported by: nb_summation_A, nb_summation_B, nb_ocean_A, nb_ocean_B, num_main_3, num_main_2, nb_wais_B
- Manuscript impact: main tex l.373 "0.76 mm yr$^{-1}$ ... over 1900--1950" → about 0.88; "39--45 mm above the reconstructions" → about 33–40 mm (31–38 mm if WAIS is also held at its 1950 value); "outside the 90\% interval of Dangendorf2024 only in 1906 and 1908" → no year (the sentence no longer holds). l.359 "WAIS follows its fit to the IMBIE-3 record" also does not describe the code (zero before 1950; observed IMBIE-3 levels, not the fit, 1980–2024).
- Defect: Stored ocean samples (1950+) include `rate_deep_k*(t-2000)` (0.117 mm/yr; component_ocean l.1067–1069). The pre-1950 splice uses the median two-layer shape `eta_oc = oc_a*Su**2 + oc_bu*Su + oc_bd*Sd` with no deep term, offset to each member at 1950, so the thermosteric rate drops by 0.117 mm/yr at 1950. WAIS is set to exactly 0 before 1950 although its 1950 level is −2.06 mm, a 2 mm step.
- Evidence: own recomputation from hindcast_20c and harmonized Frederikse/Dangendorf (`verify_SF/hind.py`): current 0.759 mm/yr, 1900 gaps F +45.4 / D +39.1, outside [1906, 1908] (matches cell 11/30 outputs). Adding 0.117 mm/yr×(t−1950) before 1950: 0.876 mm/yr, +39.5/+33.3, outside []. Also holding WAIS at −2.06 mm: 0.881, +37.5/+31.2, []. HDF5 `ocean.attrs['rate_below_2000_mm_yr']=0.117`; wais_2k median 1950 = 1979 = 1980 = −2.06 mm, 1985 −1.79 vs obs −1.80, 2024 8.98 vs 8.98.
- Fix: Add `rate_below_2000_mm_yr/1e3*(ode_t-BASELINE_YEAR)` to `eta_oc` (or each member's own deep rate) and use `np.interp` over all hind20_years for WAIS (clamps at the 1950 value); rerun summation and results_figures and requote l.373; reword l.359 ("WAIS follows the IMBIE-3 record from 1979, constant before").

#### SF-2 AR6 1995–2014 → 2000 offset computed four different ways (12.1, 10.7, −11.1 with wrong sign, 0)
- Verdict: CONFIRMED
- Severity: medium
- Location: component_forecast cell 12 l.617–626 (12.1 mm, also scripts/manuscript_numbers.py:70–75, results_figures cells 11/36/38); notebooks/precompute_ipcc_distributions.py:66–70 `_GMSL_2000_TO_2005_MM = 10.7` (→ ipcc_distributions.h5 total; forecast cell 20 ridge, results_figures ridge/fan, plot_cdf_comparison.py:58); component_summation cell 18 l.1722–1726 (−11.1 mm, used cells 20, 28); summation cells 21, 29 and forecast cells 16, 21 (no offset)
- Reported by: nb_summation_A, nb_summation_B, nb_forecast_A, nb_forecast_B, xc_constants, num_main_3, nb_figs_A, tests
- Manuscript impact: l.429 "lowers each value by 12 mm" → 13 mm if the offset is made consistent with the forecast's NASA-at-2000 anchor (13.3 mm). At that offset the l.429/l.405 values 0.45/0.59/0.77 m, 2–13%, 33–37% hold at quoted precision; the 2.7 °C slow 95th moves 815.0→813.8 mm (0.82→0.81 m). The plotted AR6 distributions in the manuscript fan/ridge figure (results_figures, via ipcc_distributions.h5) use 10.7 mm. Summation figures with −11.1 are not in the tex.
- Defect: (a) Summation adds G(2000)−G(2004.5) = −11.1 mm to AR6; the correct conversion adds (1995–2014 mean − G(2000)) > 0, so its IPCC curves sit ~22 mm low. (b) The forecast/MN offset interpolates Frederikse at 2000 on integer `year` labels that are annual means (stamped July 1), i.e. G(2000.5), giving 12.07 instead of 13.16 mm. (c) Precompute hard-codes 10.7 mm, which no pipeline record reproduces. (d) Several tables/plots labeled "rel. to 2000" show raw AR6.
- Evidence: own check (`verify_SF/off.py`): Frederikse code-style 12.07 mm, mid-year-stamped 13.16; NASA smoothed 1995–2014 mean − value at 1999.99 = 13.31, raw 12.83; NASA H(2004.5)−H(2000) = 11.14. Outputs: summation `IPCC baseline offset ... -11.1 mm`; forecast `12.1 mm`; h5 attrs `rebase_mm_total: 10.7`. Forecast cell 16 l.861 `ax.plot(ie['years'][ipcc_mask], ie['q50'][ipcc_mask]` under ylabel "rel. to 2000".
- Fix: One shared offset function (1995–2014 mean minus value at 2000.0, with correct time stamps, preferably from the same NASA series the forecast is anchored to), used by precompute (rerun), forecast cells 12/16/21, summation cells 18/20/21/28/29, results_figures and manuscript_numbers; requote "12 mm".

#### SF-3 AR6-to-Berkeley GMST offset differs between components summed together (0.726 vs 0.755 °C)
- Verdict: CONFIRMED (inconsistency); mm effect PLAUSIBLE (estimated)
- Severity: medium
- Location: component_glacier l.832–836, component_greenland l.1150–1153, component_eais l.594–598, component_apeninsula l.658–662 (Historical 1995–2005 − Berkeley 1995–2005); component_ocean l.961–967 and component_forecast cell 18 l.990–995 (SSP2-4.5 2015–2024 overlap, all SSPs); bayesian_ratestate and component_summation cell 19 l.1758–1764 (single-year AR6 2015 vs Berkeley 2014.5)
- Reported by: mod_comp, xc_constants, nb_ratestate_A, nb_ratestate_B, nb_summation_B
- Manuscript impact: thermosteric 2100 values l.394 "147 [109, 204], 246 [185, 342], 374 [276, 538] mm" and ratios 0.99/1.22/1.48 move by a few mm (estimated ~3–5 mm, last digit). Rate-and-state numbers are not in the tex.
- Defect: The AR6 median path spliced on after 2024 is shifted by a different offset in ocean (0.755) than in glacier/Greenland/EAIS/Peninsula (0.726), so member k is not forced by one GMST path across the sum, contrary to l.282 ("Member i of every GMST-driven component follows the same GMST path"). Glacier-type notebooks also use AR6 `decimal_year` without the +0.5 mid-year shift that ocean uses. Rate-and-state uses 0.882–0.898 °C (argmin tie picks Berkeley 2014.5).
- Evidence: own recomputation (`verify_SF/toff.py`): glacier-style 0.7257; overlap 0.7529/0.7554/0.7468/0.7795 (SSP1-2.6/2-4.5/3-7.0/5-8.5); ratestate-style 0.8867–0.8975 at BE 2014.5. Ocean output `SSP-to-Berkeley baseline offset: 0.755 C`. Estimate: (2·18.8·~2.5 + 51.8)·0.029 ≈ 4 mm upper layer at 2100.
- Fix: One offset function (e.g. in warming_paths) returning mid-year-stamped, rebased AR6 paths, used by every component and by ratestate/summation cell 19; rerun.

#### SF-4 Satellite-era quadratic: Maul–Martin rho computed on NASA GMSL with the seasonal cycle retained
- Verdict: CONFIRMED (rho, serial term); interval change PLAUSIBLE
- Severity: medium
- Location: bayesian_models.py:2336–2341; inputs component_forecast cell 8 (`nasa_gmsl = df_nasa['gmsl']`), component_summation cells 11, 18
- Reported by: mod_slr, nb_forecast_B
- Manuscript impact: l.427 "0.63 [0.38, 0.86] m" → about 0.63 [0.41, 0.86] m (mod_slr's full-budget refit); also the 5.5 mm origin σ (l.282) and early blended rates slightly.
- Defect: The fit follows Nerem 2022/Hamlington 2024, who remove the annual/semi-annual cycle, but the fitted column keeps a 4.8 mm annual cycle. Lag-1 correlation of residuals is then dominated by the deterministic cycle, inflating rho and the residual variance that set the serial-correlation covariance. H_ORIGIN uses the deseasoned `gmsl_smoothed`, so origin and quadratic come from different series.
- Evidence: own refit (`verify_SF/sq.py`): `gmsl` rho 0.886, accel 72.8; harmonics removed rho 0.798, accel 74.2, annual amplitude 4.76 mm; serial-term 1σ at 2100 0.0635 → 0.0335 m. The measurement term (0.134 m) still dominates, so the total moves less.
- Fix: Regress out annual and semi-annual harmonics inside the fit (or compute rho on deseasoned residuals); rerun forecast and requote l.427.

#### SF-5 VoI surface (Fig. 4c) is discounted to the arrival year, caption says present value
- Verdict: CONFIRMED
- Severity: medium
- Location: notebooks/compute_evpi.py:105–110 `_npv_factor`, l.197 `VOI_2[i, j] = evpi_ann * _npv_factor(ay, 2100) / 1000`; main tex l.450 caption, l.457
- Reported by: num_main_3
- Manuscript impact: Fig. 4c contours for arrival years after 2026 (for 2060 the PV-in-2026 value would be ×1.03^−34 ≈ 0.37 of what is drawn). The quoted scalars (4.7/3.9/0.9 T$, arrival 2026) are unaffected.
- Defect: The annuity runs from arrival to 2100 and is discounted to the arrival year only; nothing discounts back to 2026. The decline with arrival year therefore reflects only the shorter annuity. The annual benefit is also the 2100-distribution cost difference in every year, which the text does not state.
- Evidence: code as quoted; no `(1+DR)**-(ay-2026)` term anywhere in compute_evpi.py.
- Fix: Multiply by (1+DR)^−(ay−2026) if PV today is meant, or reword the caption to "value at the time the information arrives".

#### SF-6 build_slow_wais_p90 breaks at N_SAMPLES = 10000 (final run)
- Verdict: CONFIRMED (code path; not executed)
- Severity: medium
- Location: scripts/build_slow_wais_p90.py:38 `N_OUT = 20_000`, :66–74 `build`, :50–63 `s2_row_index`
- Reported by: scripts_down
- Manuscript impact: none now; every slow-WAIS number (blended_slow90/wais_slow90) in the final 10k run
- Defect: `divmod(18000, n_S1)` with n_S1 = 10000 leaves rem 8000 and the assert aborts. If only the S2 branch exceeds 2000 rows, the `extra <= 0` branch silently subsamples S2 to 2000 rows.
- Evidence: `reps, rem = divmod(n_s1_rows, s1.shape[0]); assert rem == 0`; `if extra <= 0: return order[np.linspace(0, n - 1, n_rows)...]`.
- Fix: Derive row counts from branch sizes (e.g. 9·n_S1 S1 rows and n_S1 S2 rows) or store weights instead of a fixed N_OUT.

#### SF-7 Per-component forecast figure legend says 2/3/4 °C
- Verdict: CONFIRMED
- Severity: medium
- Location: component_forecast_code.py cell 18 l.900–904 (used l.1043, 1079) → figures/component_forecast_per_component.png
- Reported by: nb_forecast_B, xc_constants, num_main_3
- Manuscript impact: main-text Fig. per_component_forecast (tex l.387–389); caption says legend gives AR6 central-estimate warming (1.8/2.7/3.6 °C).
- Defect: `SSP_LEGEND_LABELS` hard-codes integer labels (and no space before $^\circ$). Same stale map in compute_shapley_risk.py:26 (HDF5 attr; figure not in tex).
- Evidence: `'SSP1-2.6': r'2$^\circ$C (SSP1-2.6)', 'SSP2-4.5': r'3$^\circ$C (SSP2-4.5)', 'SSP3-7.0': r'4$^\circ$C (SSP3-7.0)'`.
- Fix: Use 1.8/2.7/3.6 $^\circ$C from one shared label map; rerun cell 18, reinstall figure, rebuild PDF.

#### SF-8 Satellite-era hindcast rates in the tex are the unblended component sum, but l.361 says the comparison uses the blend
- Verdict: CONFIRMED
- Severity: medium
- Location: component_summation cell 11 l.1201–1221, l.1286; main tex l.361, l.373, l.377
- Reported by: num_main_2, num_main_3
- Manuscript impact: description only; 2.98/3.11/3.07/3.25 mm/yr are the "Comp. sum" column; the blended values are 2.93/3.07/3.03/3.24.
- Defect: The code deliberately uses the component sum in the satellite era (prints the reason), but the tex says the level and rate comparison uses the reflected blend and calls these "the hindcast rate".
- Evidence: cell 11 output `1993–2025   Frederikse  1993–2018       3.35      2.93       2.98`; `(Satellite era: the blended hindcast contains the altimetry quadratic; use the component sum.)`.
- Fix: Add a clause at l.373/l.377 that satellite-era rates are from the unblended component sum, or quote the blended values.

#### SF-9 "about 1.1 °C (11-yr means)": end value is a 6-yr trailing mean
- Verdict: CONFIRMED
- Severity: medium
- Location: component_summation cell 13 l.1443–1445
- Reported by: nb_summation_A, nb_summation_B
- Manuscript impact: abstract l.43, l.69, l.377 "(11-yr means)", l.466. Full 11-yr windows give 1.00–1.07 °C (ending 2019/2020) instead of 1.06–1.13 °C.
- Defect: `rolling(11, center=True, min_periods=6)` makes the last value the mean of the final 6 years (centred ~2021.5–2022.5), while 1950 is a full centred 11-yr mean; the label says 11-yr means. "About 1.1" straddles the rounding boundary under the full-window convention.
- Evidence: own check (`verify_SF/warm.py`): Berkeley trunc 2024 1.098 vs full 2019 1.029; GISTEMP 1.125 vs 1.065; HadCRUT 1.060 vs 1.003.
- Fix: Use full windows (min_periods=11) and report the end year used, or describe the end value as a trailing mean; reconsider "about 1.1" vs "about 1.0–1.1".

#### SF-10 Warming-path description (main tex l.282) omits the 1/1.645 scaling
- Verdict: CONFIRMED
- Severity: medium
- Location: main tex l.282; notebooks/warming_paths.py:59–60
- Reported by: num_main_2
- Manuscript impact: l.282 "the AR6 median plus $z_i$ times the growth of the AR6 half-width since 2024" (wording; numbers unaffected). Supp l.360 is correct.
- Defect: Code uses hw = (upper − median)/1.645, so z = 1.645 reaches the 95th-percentile growth; read literally, the main text describes a spread 1.645× wider.
- Evidence: `hw_hi = (df_ssp['temperature_upper'].values - med) / Z_90`, `Z_90 = 1.645`.
- Fix: Say "the growth since 2024 of the AR6 half-width divided by 1.645" or cite the supplement.

#### SF-11 compute_evpi D_GRID starts at 0.3 m; early-target optima lie below it and only the upper edge is checked
- Verdict: CONFIRMED
- Severity: low
- Location: notebooks/compute_evpi.py:58, :76–81, :160–168
- Reported by: scripts_down
- Manuscript impact: none (tex VoI from surface 2 at 2100, S1 optimum 0.85 m); affects voi/evpi_raw, evpi_smooth, VOI_1 and fig_voi_arrival_target.png (not in tex)
- Defect: `_check_interior` raises only at the upper edge, so minima pinned at 0.3 m pass silently.
- Evidence: optimal_defense/dstar_stable 2040–2052: 0.18, 0.20, 0.22, 0.22, 0.24, 0.26, 0.28 m; dstar_unconditional 2040–2046 0.20–0.30 m (grid from 0.0).
- Fix: Start D_GRID at 0.0 and reject a minimum at either edge.

#### SF-12 Fast-WAIS ensemble realized at p(S1) = 0.0975; full and S1 WAIS rows not paired
- Verdict: CONFIRMED
- Severity: low
- Location: component_wais downsample (wais_2k); compute_evpi.py:54 `P_S1_BLENDED = 0.10`, :101, :150; variance_shares.py docstring; component_forecast cell 6 (`stable = total - full_wais + s1_wais`)
- Reported by: nb_forecast_A, scripts_down, num_supp_2, num_main_3, nb_wais_B
- Manuscript impact: negligible; supp l.354 fast shares 98.3/97.0/93.3% are from the realized mixture (analytic p=0.10: 98.9/97.5/94.2%); scalar EVPI 29.7 vs ~28.8–29.6 B$/yr (not in tex)
- Defect: 195/2000 members are S1; downstream code treats it as exactly 0.10. Independent downsample indices mean S1 rows of `blended` differ from `blended_stable`, so per-member full − stable differences carry WAIS sampling noise.
- Evidence: own read: S1 frac 0.0975; `np.allclose(full[idx==0], s1[idx==0])` False; max |diff| at 2100 221 mm.
- Fix: Stratify the scenario draw (exactly 200 S1) and reuse one index set for full and S1; or read the realized fraction from scenario_idx.

#### SF-13 S2-vs-AR6 AIS crossing compares WAIS rel. 2000 with AR6 rel. 1995–2014
- Verdict: CONFIRMED
- Severity: low
- Location: scripts/manuscript_numbers.py:256–271
- Reported by: scripts_down
- Manuscript impact: main tex l.318 "until about 2050" unchanged; SSP1-2.6 crossing 2050 → 2051 on a consistent baseline
- Defect: Elsewhere the script rebases each member on its own 1995–2014 mean; here it does not (S2 members' 1995–2014 mean is +1.27 mm), and the 2050 crossing is decided by <1 mm.
- Evidence: own recomputation: rel2000 2050/2051/2052; AR6 baseline 2051/2051/2052.
- Fix: Subtract each S2 member's 1995–2014 mean before comparing (as l.172 does).

#### SF-14 "Shortfall arises after 2018" (l.346) is not printed by any pipeline code
- Verdict: CONFIRMED (traceability only; the substantive objection in num_main_3 is REFUTED)
- Severity: low
- Location: main tex l.346; component_summation cell 10 (closure prints only 1993/2000/2010–2025)
- Reported by: num_main_3
- Manuscript impact: none on numbers; claim untraceable
- Defect: No cell prints closure for 2010–2018 vs 2018–2025. With the closure's own integer-year sampling the claim is supported: 2010–2018 NASA 4.10 vs sum 4.27 mm/yr (104.1%), 2018–2025 NASA 4.34 vs sum 3.71 (85.5%); TWS trend 0.66 → 0.00, Greenland 0.75 → 0.47 mm/yr. num_main_3's contrary result came from sampling on NASA times; the split is window/sampling sensitive.
- Evidence: `verify_SF/clos.py` on hindcast_diagnostic/total_budget_med and component medians.
- Fix: Print the 2010–2018 / 2018–2025 split (and component trends) in summation §6.

#### SF-15 Dangendorf 90%-interval test uses unrebased absolute sigma; chi2 in the same notebook uses the increment rule
- Verdict: CONFIRMED
- Severity: low
- Location: component_summation cell 30 l.2212–2217 vs cell 11 l.1255–1256
- Reported by: nb_summation_A
- Manuscript impact: l.373 "only in 1906 and 1908" (raw σ) vs 1900, 1901, 1904–1909 (increment σ relative to 2000). Moot after SF-1 (both give none).
- Defect: Two conventions for uncertainty relative to the 2000 rebase in one analysis.
- Evidence: `verify_SF/hind.py` current: raw [1906, 1908]; incr [1900, 1901, 1904, 1905, 1906, 1907, 1908, 1909].
- Fix: Pick one convention, state it in the tex.

#### SF-16 Satellite-quadratic `cov_params` omit GIA and `sigma_inflate`; Ablain span-scaling exponent is half the comment's; silent OLS+HAC fallback
- Verdict: CONFIRMED
- Severity: low
- Location: bayesian_models.py:2319–2331, 2349–2352; component_forecast cell 8, component_summation l.912/1676 (`... if os.path.exists(...) else None`)
- Reported by: nb_forecast_A, nb_forecast_B, mod_slr, mod_bayes_A
- Manuscript impact: negligible for GIA (2100 1σ 0.1480 → 0.1488 m); exponent choice affects l.427 interval (measurement term 0.134 of 0.148 m) by an unquantified amount
- Defect: MC draws use `cov_params = cov_beta_meas + cov_beta_serial` (no GIA), unlike `rate_accel_cov`. Comment says c1/c2 SE scale as 1/T and 1/T², but `r**((p_i+p_j)/2)` gives T^−1/2 and T^−1. Missing Ablain file silently switches error model.
- Evidence: code lines as quoted.
- Fix: Add σ_GIA² to cov_params[1,1]; make the scaling and its comment agree and state the assumption in the supplement; raise if the covariance file is missing.

#### SF-17 Small forecast-notebook inconsistencies
- Verdict: CONFIRMED
- Severity: low
- Location: component_forecast cell 12 l.707 (print), l.547/574 vs 581 (ddof), headline JSON `variance_decomposition`; cell 18 l.997 (argmin tie); cell 22 (blend-midpoint print); cell 0 "7. TODO"; cell 16 obs rebased on raw `gmsl`
- Reported by: nb_forecast_A, nb_forecast_B, num_main_3
- Manuscript impact: none (tex "3 mm or less" matches stored output, max over policy SSPs 3.0 mm, but the cell prints "Largest change: 8.2 mm")
- Defect: Print shows rel-2000 median next to AR6-baseline ratio ("60 vs IPCC 43.7 cm = 1.3x (34% above)"); np.var ddof=0 vs np.cov ddof=1; JSON variance block is a third share basis (WAIS 88.0%); GMST zero picks 1999.5 (tie; 1999 and 2000 annual means happen to be equal); stale TODO.
- Evidence: outputs and code lines as cited.
- Fix: Print `our_median_cm_ar6_baseline`; use consistent ddof; rename or drop the JSON block; pick 2000.5 explicitly; print the policy-SSP maximum.

#### SF-18 Variance-share prints on bases the paper does not use
- Verdict: CONFIRMED
- Severity: low
- Location: scripts/build_slow_wais_p90.py:167–181; scripts/slow_wais_p90_numbers.py ("S2 var share"); compute_shapley_risk.py:72–78
- Reported by: scripts_down, num_main_3
- Manuscript impact: none (tex shares from variance_shares.py and manuscript_numbers interval shares)
- Defect: build_slow prints "for the Fig. 4b caption ... as the pies compute them" on var(c)/Σvar without EAIS/Peninsula/covariance (fast 99.2/98.3/96.2, slow 95.7/91.1/81.3), but the pies use 90% interval widths; hard-coded `[:, 150]`. Shapley fractions omit TWS so they sum to 97% of the total risk (log "Frac sum 12505" vs total 12866).
- Evidence: code and pipeline_full.log lines as cited.
- Fix: Delete or relabel these prints; normalize Shapley fractions by the component sum or add a TWS row.

#### SF-19 No provenance across derived groups; ipcc_distributions.h5 outside the pipeline at fixed 2000 members
- Verdict: PLAUSIBLE (hazard; not triggered in this full run)
- Severity: low
- Location: component_forecast cell 13; build_slow_wais_p90.py, compute_evpi/defense/shapley, manuscript_numbers.py; run_pipeline.sh ENTRIES (no precompute); component_projections.py:1521 (raises on size mismatch)
- Reported by: nb_forecast_A, nb_forecast_B, scripts_down, xc_hdf5
- Manuscript impact: none now; partial reruns or the 10k run
- Defect: `blended_slow90`, `wais_slow90`, `voi`, `optimal_defense`, `risk_shapley` carry no source stamp, so rerunning `forecast` alone leaves them stale. Precompute is not in run_pipeline.sh; the 10k run will fail loudly at forecast cell 5 unless rerun by hand.
- Evidence: no `saved_at` copy in derived groups; ENTRIES list lacks precompute; h5 attrs n_samples=2000.
- Fix: Copy `blended.attrs['saved_at']` into derived groups and assert on read; add precompute (with N_SAMPLES) before summation.

#### SF-20 Summation-notebook diagnostics (not in tex)
- Verdict: CONFIRMED
- Severity: low
- Location: component_summation cells 14, 18, 19, 29, 30, 11 (l.982), 5; component_results.h5 root attrs
- Reported by: nb_summation_A, nb_summation_B, nb_ratestate_A, nb_ratestate_B, mod_io, xc_hdf5, num_main_3
- Manuscript impact: none (tex l.417 "9%" comes from manuscript_numbers 9.1%; "0.21 mm/yr" l.373 verified 0.209 but not printed)
- Defect: Cell 14 rate-state hindcast draws MVN coefficients plus independent τ (no `posterior_samples`, although `posterior/coefficients` exists) and is zeroed at 1999.5; cell 18 sat-quad band rebased by the median fit's 2000 value (629 [380, 870] vs forecast 627 [381, 864]); cell 19 AR6 splice at Berkeley 2014.5 and "2100" = 2099; cell 30 WAIS share evaluates IMBIE at 2024.5 beyond its 2024.0 end (9.9% vs MN 9.1%); ocean ODE median rebased at 1999.5 (+0.34 mm at 2000); projected TWS zero through 2005 (2000–2005 change omitted, ~1–2 mm, documented); root attrs `baseline_year 2005.0`; uncharted-glacier 0.21 mm/yr not printed.
- Evidence: code lines and HDF5 reads as cited.
- Fix: Pass the joint posterior; rebase per sample; print the uncharted rate and the 2010–2018/2018–2025 split; refresh root attrs.

#### SF-21 Stale comments, prints and labels (summation)
- Verdict: CONFIRMED
- Severity: low
- Location: component_summation l.354 ("Frederikse, Dangendorf, and NASA"), l.465 ("2023.5"), l.544/1035 ("TWS ... 1900–2018"), l.873–874/1062/1080 (stale line refs), l.2094 ("3$^\circ$C warming"), Diagnostic 6/7 numbering, cell 0 TWS description
- Reported by: nb_summation_A, nb_summation_B
- Manuscript impact: none
- Defect: Text disagrees with code/data and the decimal warming labels.
- Evidence: spot-checked l.354, l.465, l.2094, l.352/1480.
- Fix: Update strings.

#### SF-22 Tests encode stale forecast/summation conventions
- Verdict: CONFIRMED
- Severity: low
- Location: tests/test_summation.py:44 `BASELINE_YEAR = 2005.0`; tests/test_forecast.py:237–292 (inline blend copy, `forecast_samples[:, 0] = h_origin`), :526–542 (IPCC on two baselines)
- Reported by: tests
- Manuscript impact: none
- Defect: Summation test checks near-zero at 2005 with loose bounds; forecast test exercises a stale copy of blend_rate_space lacking the off-grid origin step; IPCC test compares rel-2000 median with AR6 rel-1995–2014.
- Evidence: lines as cited.
- Fix: Import BASELINE_YEAR and blend_rate_space from production; compare `our_median_cm_ar6_baseline`.

### Old-report items
(component_summation from nb_summation_A; component_forecast from nb_forecast_A; B reviewers had none assigned; no disagreements found)
- Summation §1.3 cell 18 IPCC offset sign — STILL OPEN — cell 18 l.1726 `ipcc_offset_mm = (_gmsl_at_bl - _gmsl_at_ipcc_ref)`, output −11.1 mm (SF-2)
- Summation §1.3 cells 21, 29 IPCC unconverted — STILL OPEN — cell 29 `get_ipcc_stats(...)` raw; cell 21 no offset (SF-2)
- §1.3 TWS carries +10.7 mm total offset — FIXED — precompute `_IPCC_REBASE_COMPONENTS = {"total"}`; landwaterstorage q50(2005) = 0
- §1.2 cell 16 post-2100 flat components feed 2150 rows — FIXED — ocean SSP5-8.5 medians differ 2100 vs 2150
- §1.6 cell 5 TWS independent per-year draws — FIXED — load_ipcc_tws_projection "each member holding one quantile across all years"
- §1.4 annualize_imbie yr+0.5 labeling — FIXED — component_analysis.py:101 `years[i] = yr + 1.0`
- Hindcast cells skip per-sample rebase for wais_2k / Greenland 1900 — FIXED (moot) — both exactly zero at 2000
- Cell 13 min_periods=6 one-sided 11-yr mean — STILL OPEN — l.1443 (SF-9)
- Root attr baseline_year 2005 — STILL OPEN — root attrs `baseline_year 2005.0` (SF-20)
- TWS prints "1900–2018" (cells 10, 11) — STILL OPEN — l.544, l.1035 (SF-21)
- "conversation notes" print — FIXED — cell 11 l.1005–1007
- Cell 18 sat-quad band rebased by median fit — STILL OPEN — l.1698 `_sq_boot_rb = _sq_boot - _sq_at_bl` (SF-20)
- Independent per-year noise on cumulative TWS/EAIS in closure — FIXED — `cumulative_error_paths` l.490, l.534 (verified)
- Cell 24 "3°C warming" title — STILL OPEN — l.2094 (SF-21)
- Cell 5 TWS referenced to 2005 — STILL OPEN (documented) — load_ipcc_tws_projection docstring (SF-20)
- Comment cell 3 "storage baseline (2005)" — FIXED — l.120 (verified)
- Comment cell 7 "WAIS has 10k" — FIXED — l.241
- Comment cell 10 header "Frederikse, Dangendorf, and NASA" — STILL OPEN — l.354 (SF-21)
- Comment "ocean (hybrid NOAA+IPCC)" — FIXED — l.357 "ocean (two-layer model)"
- Comment "Diagnostic 6/7" numbering — STILL OPEN — l.352, l.1480 (SF-21)
- Comment cell 11 "TWS: Frederikse obs (1900–2018)" — FIXED — l.1019 "GRACE era only (2003–2018)"
- Comment cell 19 "loaded in cell above" — FIXED — l.1733
- Comment cell 27 "§4" — FIXED — l.2137 "§5"
- Markdown cell 0 "preserve correlations within each draw" — NOT APPLICABLE — shared warming_z paths give real cross-component correlation
- Markdown cell 8 RUN_ADDENDUM — FIXED — cell 8 revised
- Markdown cell 0 TWS only as IPCC — STILL OPEN — l.15–16 (SF-21)
- Markdown cell 8 "GMST-forced total hindcast" — FIXED — l.270 "Total hindcast"
- Forecast §1.3 IPCC baseline handled three ways (cells 12, 16/20, 21) — STILL OPEN — 12.1 / 10.7 / 0 (SF-2)
- Forecast: TWS carries +10.7 mm — FIXED — see above
- Forecast §1.6 TWS per-year draws — FIXED — one quantile per member
- Forecast §1.2 flat after 2099 — NOT APPLICABLE — AR6 held flat to 2100 by design (2026-10-15)
- Origin placed on first grid year — FIXED — component_projections.py:1939–1951; output `w=0.873` at 2025.34
- Quadratic MC omits GIA (cov_params) — STILL OPEN — cell 9 `multivariate_normal(..., sat_quad.cov_params ...)` (SF-16)
- ddof mix np.var / np.cov — STILL OPEN — cell 12 l.547/581 (SF-17)
- IPCC print mixes baselines — STILL OPEN — l.707 (SF-17)
- Single unsmoothed NASA sample for rebase/H_ORIGIN — FIXED for forecast — cell 8 uses `gmsl_smoothed`; only cell 16 plotted obs use raw (SF-17)
- "4x" within/across ratio — FIXED — output "3.5x"
- wais_2k full/s1 independent index sets — STILL OPEN — `np.allclose(...)` False (SF-12)
- Cell 8/9 comments "relative to 2005" — FIXED — "relative to BASELINE_YEAR (2000)"
- Cell 0 "7. TODO" — STILL OPEN — forecast l.28 (SF-17)

### Cross-cluster notes
- GREENLAND/INFRA: component_greenland cell 26 reads `blended`, `blended_stable`, `blended_slow90` before forecast/slowwais write them (stale in a full run; KeyError on fresh HDF5) (xc_hdf5, nb_greenland_A)
- OCEAN: NOAA 2025.5 points fitted against model 2024.5 (nb_ocean_A, num_main_1, scripts_comp); ocean projections zeroed at 1999.5 (nb_ocean_A/B); holdout 2-walker thinning and supp 15.0 [12.0, 17.6] mm (scripts_comp, num_supp_2)
- RATESTATE: projections zeroed at 2015 but labeled rel. 2000; 2099 as 2100; τ/MVN draws (nb_ratestate_A/B)
- FIGURES-TEX: results_figures uses 10.7 mm ridge/fan AR6 offset vs 12.1 text (nb_figs_A); per-component IPCC figure adds total offset to each component (nb_figs_B); Fig 1a forecast line +6 mm (nb_figs_A, num_main_3); Fig 4 caption "variance decomposition" vs interval shares (nb_figs_B); integer warming labels in results_figures cells 41/52 (nb_figs_B, xc_constants)
- READERS: IPCC AR6 observed GMSL σ = half 17–83% range (4.6% small; used in summation chi2) (mod_readers); NASA `gmsl_sigma` is spatial scatter (mod_readers)
- ANTARCTIC: S1 vs whole-AIS ISMIP6 comparison; post-2100 WAIS extrapolation (feeds VOI_1 to 2150) (nb_wais_A/B)
- GLACIER: 84% of SSP5-8.5 glacier members at the 0.32 m cap by 2150 (nb_summation_A)
- INFRA: N_SAMPLES hard-coded in each notebook incl. summation `N_MC = 2000` (xc_constants)
- COMPONENTS: plot_component_projection_twopanel overlays AR6 without rebasing (mod_io)

---

## B6. Shared modules, readers, IO, constants (MD)

No finding in the MODULES cluster reaches high. Scripts used for the re-derivations are in `$S/verify_MD/` (be.py, imbie.py, satq.py). The two BIC scripts are the reviewer's, rerun from `$S`.

#### MD-1 Linear-vs-quadratic BIC rebuilt from `chi2_dof` leaves out ln|Σ| and uses a different dof from the fitter
- Verdict: CONFIRMED
- Severity: medium
- Location: notebooks/bayesian_models.py:3331-3338 (`chi2_dof`); component_glacier cell 6 (code l.305-313); component_apeninsula cell 6 (l.297-304); f_max robustness cells (glacier ~l.1180, apeninsula ~l.1036)
- Reported by: mod_bayes_A, nb_glacier_A, nb_glacier_B, nb_apen_A, nb_apen_B, nb_eais_A, num_supp_1
- Manuscript impact: supp.tex:206 "set to zero because BIC favors the linear model". The conclusion holds only because of the a ≥ 0 prior bound, not because the BIC as computed measures the data. With a unbounded the properly computed ΔBIC(lin−quad) is +4.74, which favors the quadratic. No printed number changes.
- Defect: The fitter returns only `r'Σ̄⁻¹r/(n − n_phys − 1)` at the posterior mean. The notebooks set BIC = chi2 + k ln n. When σ_extra is fitted, the two models have different Σ, so the dropped ln|Σ| term does not cancel in ΔBIC. The notebooks also multiply back by `n − n_phys − 2`, not the fitter's `n − n_phys − 1`. That error mostly cancels in ΔBIC (about ±1 on each side). Every stored ΔBIC is about −ln n (EAIS −3.8 against ln 45 = 3.81; glacier f_max rows −3.1/−3.5/−3.3; Peninsula −3.1/−3.5/−3.4). So chi2_q ≈ chi2_l, and the "BIC" is only the parameter penalty.
- Evidence: code `chi2_dof = float((resid @ Sigma_inv_mean @ resid) / (n - n_phys - 1))` vs glacier `dof_l = n - 2 - 1 - 1`. I reran `$S/bic_check_modbayesA.py` (Peninsula, unconstrained MLE). Lin: σ_extra 0.0479, ln|Σ| −246.84, chi2 44.87. Quad: a = −0.351, σ_extra 0.0399, ln|Σ| −255.53, chi2 45.02. Proper ΔBIC +4.74, chi2-only −3.96. `$S/bic_check2_modbayesA.py` (MAP under the notebook priors) gives a = 0, proper −3.81 and chi2-only −3.75. `component_analysis.time_model_bic` (WAIS) already uses the full −2 ln L.
- Fix: Return −2 ln L (chi2 + ln|Σ|) from the fitter, at the max-likelihood draw, and use it in the notebooks as `time_model_bic` does. State in the supplement that the linear choice rests on the a ≥ 0 bound.

#### MD-2 χ²/dof denominators count H0 but not σ_extra; the supplement range 0.94–1.05 depends on that unstated convention
- Verdict: CONFIRMED
- Severity: medium
- Location: notebooks/bayesian_models.py:3338; consumers: glacier/apeninsula cell 6 prints, component_wais l.626 (via `fit_imbie_time_model`)
- Reported by: mod_bayes_A
- Manuscript impact: supp.tex:371 "All four sit between 0.94 and 1.05" → "0.94 and 1.10" if σ_extra is counted. Table supp.tex:400 Glaciers 1.05 → 1.10. Peninsula (l.402) 0.97 → 1.00. WAIS (l.404) 0.99 → 1.01. EAIS 0.94 is unchanged (no σ_extra).
- Defect: The denominator is n − n_phys − 1. It counts H0 but not the fitted σ_extra, while `time_model_bic` in the same codebase counts σ_extra in k. Glacier n = 24, n_phys = 2 (b, T_eq0): 1.045 × 21/20 = 1.097. Peninsula n = 45, n_phys = 2: 0.975 × 42/41 = 0.999. WAIS n = 45, order = 1 → n_phys = 2: 0.99 × 42/41 = 1.014. The supplement caption says only "per degree of freedom".
- Evidence: the stored prints `chi2/dof = 1.045` (glacier out l.220), `0.975` (apen out l.285), `0.99` (wais out l.35); `time_model_bic`: `k = (tm['order'] + 1) + 2 + ...` ("k counts the rate coefficients, H0, sigma_extra").
- Fix: Either count σ_extra (and φ) in the denominator and update the table, or state in the caption that variance parameters are not counted.

#### MD-3 Satellite-era quadratic is fitted to NASA column 9, which keeps the annual and semi-annual cycle; this widens the 2100 interval
- Verdict: CONFIRMED
- Severity: medium
- Location: notebooks/slr_data_readers.py:654 (`'gmsl': float(parts[8])` = file column 9); bayesian_models.py:2337-2342 (Maul–Martin ρ on those residuals); callers component_forecast cell 8, component_summation l.293/908/1672, bayesian_ratestate
- Reported by: mod_slr, nb_forecast_B, mod_readers (cross-ref)
- Manuscript impact: main.tex:427 "0.63 [0.38, 0.86] m" → about 0.63 [0.40–0.41, 0.86] m. Fig. 1b satellite diamonds and the ratestate rate prior shift slightly.
- Defect: The NASA header defines column 9 as unsmoothed GIA-applied GMSL. Only column 12 has the annual and semi-annual signals removed. The fitted cycles (4.8 mm annual, 1.6 mm semi-annual) raise the lag-1 residual autocorrelation and the residual variance in the serial-correlation term. The Ablain covariance refers to a series with the cycle removed.
- Evidence: I ran `fit_satellite_era_quadratic` read-only (`verify_MD/satq.py`) with the production arguments, then MVN draws of `cov_params` (seed 42, 20k). On column 9: rate SE 0.611 mm/yr, H(2100)−H(2000) 0.629 [0.382, 0.872] m (stored forecast output: 0.627 [0.381, 0.864]). With the harmonics regressed out: rate SE 0.570 mm/yr, 0.634 [0.405, 0.861] m. Annual amplitude 4.76 mm, semi-annual 1.63 mm.
- Fix: Remove the harmonics inside the fit (or fit them jointly), then recompute the tex interval. Column 12 is not a drop-in replacement, because its 60-day smoothing makes ρ ≈ 0.99.

#### MD-4 slr_processed_data.h5 still holds Berkeley Earth σ from before the 95% fix (σ = unc/1.645, 1.19× too large)
- Verdict: CONFIRMED
- Severity: medium
- Location: data/processed/slr_processed_data.h5 `raw/df_berkeley`, `harmonized/df_berkeley_h` (built by read_process_datafiles.ipynb, not in run_pipeline.sh); consumers component_forecast cell 18 (l.978-982, band `±1.645*T_annual_sigma`, l.1034) and results_figures cell 15/18 (l.880-899, fill l.1218)
- Reported by: mod_readers. Contradicted by nb_forecast_B ("Berkeley 95% CI / 1.96 × 1.645, which is correct") and nb_figs_A ("Berkeley converts 95% to 1σ"). Both checked the reader, not the stored table.
- Manuscript impact: no number. The observed-GMST band in main Fig. `per_component_forecast`(a) (main.tex:387) is the full 95% half-width, drawn where the code intends 90%. The caption states no level for it. In Fig. 1(a) (main.tex:366) results_figures divides by √12 first, so the drawn band is narrow overall and the 1.19 partly offsets that. No calibration uses this σ (the ocean l.1340 `temperature_sigma` is the AR6 frame).
- Defect: The reader was fixed to `/ Z_95` in commit 053eca5 (2026-05-24). The stored tables predate it (file mtime 2026-03-22) and are never regenerated by the pipeline.
- Evidence: `verify_MD/be.py`: `temperature_unc/temperature_sigma` = 1.645 (min = max) in both keys. The raw file header says "Uncertainties represent the 95% confidence". Reader l.1480: `df['temperature_unc'] / Z_95`.
- Fix: Regenerate the Berkeley keys (and the other stale keys, MD-24) with the current readers. Add read_process_datafiles, or a check that stored σ matches the reader, to run_pipeline.sh.

#### MD-5 Supplement corner-plot captions say contours at 1, 2, 3σ, but corner draws its default 0.5, 1, 1.5, 2σ
- Verdict: CONFIRMED
- Severity: medium
- Location: corner.corner calls in component_ocean l.594, component_glacier l.670, component_apeninsula l.567, component_eais l.492 (and bayesian_ratestate l.757); supp.tex:278, 287, 294, 301
- Reported by: mod_io, nb_glacier_A, nb_apen_A, nb_apen_B, nb_ocean_A, nb_ocean_B, num_supp_1
- Manuscript impact: four supp captions "contours denoting 1, 2, and 3 standard deviations" are wrong as drawn
- Defect: None of the calls passes `levels=`. corner 2.2.3's default is `1 - exp(-0.5*arange(0.5, 2.1, 0.5)**2)`, i.e. 0.5, 1, 1.5 and 2σ.
- Evidence: there is no `levels` within 14 lines of any of the four calls. The default is printed from `inspect.getsource(corner.core.hist2d)`.
- Fix: Pass `levels=1-np.exp(-0.5*np.array([1,2,3])**2)` and regenerate the figures, or reword the captions.

#### MD-6 `fit_bayesian_level_annual_correlated` (all glacier, Peninsula, EAIS and WAIS posteriors) runs no convergence diagnostic
- Verdict: CONFIRMED
- Severity: medium
- Location: notebooks/bayesian_models.py:3347-3395; callers glacier/apeninsula/eais cell 6, component_analysis.fit_imbie_time_model
- Reported by: mod_bayes_A, nb_eais_A (check_convergence imported, unused)
- Manuscript impact: unknown. Convergence of four production posteriors is not demonstrated anywhere.
- Defect: Every other fitter calls `check_convergence`. This one builds the arviz trace and returns, and `sampler_diagnostics` has only the acceptance fraction and settings. No notebook calls `check_convergence` or `get_autocorr_time` on these fits. Burn-in is fixed (2000–4000 steps).
- Evidence: grep: `check_convergence` appears at bayesian_models l.600/907/1176/1580/2807/3744/4466/5248/6028 but not between 3061 and 3395. In nb_src it appears only on import lines (glacier l.28, greenland l.42).
- Fix: Compute `sampler.get_autocorr_time(quiet=True)` over all walkers inside the fitter, store it, and warn when burn-in < ~20τ.

#### MD-7 N_SAMPLES is a literal in each component notebook and in summation, not imported from config
- Verdict: CONFIRMED
- Severity: medium (will bite the planned N_SAMPLES = 10000 final run)
- Location: component_ocean l.56, glacier l.48, greenland l.75, eais l.69, apeninsula l.45 (`N_SAMPLES = 2000`); component_summation l.77 `N_MC = 2000`, l.1796 `n_samples=2000`; bayesian_ratestate l.799/811 `n_samples=2000`; config.py:33
- Reported by: xc_constants (related: scripts_down build_slow_wais 10k assert, nb_forecast_B/xc_hdf5 precompute not in pipeline)
- Manuscript impact: none now (all agree at 2000)
- Defect: Changing config alone leaves the component files at 2000 rows while forecast, summation and results_figures expect config.N_SAMPLES. Member alignment across components relies on every notebook using the same N.
- Evidence: grep output as listed. `component_wais` l.2049 imports `N_SAMPLES as N_SAMPLES_DEFAULT  # 2000`.
- Fix: Import N_SAMPLES from slr_forecast.config everywhere, and keep the WAIS 10k as an explicit multiple.

#### MD-8 "30-yr local regression" is a tricube kernel with a 30-yr half-width (60-yr support)
- Verdict: CONFIRMED
- Severity: medium (the method description in the tex disagrees with the code)
- Location: notebooks/slr_analysis.py:900-909 (`h = span_years`, `u = (time - t0)/h`, tricube on |u| ≤ 1); docstring l.798-799, 839-841; results_figures l.914 `RATE_SPAN_YEARS = 30`
- Reported by: mod_slr
- Manuscript impact: main.tex:375 "the same 30-yr local regression" (and the Fig. 1b caption) → the kernel half-width is 30 yr, the full support 60 yr and the FWHM about 35 yr. The docstring's "edge effects within span_years/2" should read "within span_years". All Frederikse rates after about 1988 come from one-sided windows.
- Evidence: code as quoted. `SlidingDOLSResult` (l.1352) documents span_years as a half-width.
- Fix: Reword the tex as "tricube kernel with a 30-yr half-width", or set h = 15. Fix the docstring.

#### MD-9 The glacier σ_gl,extra posterior quoted in the tex is neither saved nor printed
- Verdict: CONFIRMED
- Severity: medium (a tex number not traceable to code output, against the project rule)
- Location: notebooks/component_io.py:369-394 (`save_glacier` passes no `extra_posteriors`); component_glacier cell 13 (l.894-914); `save_apeninsula` likewise
- Reported by: mod_io, nb_glacier_A, nb_apen_B
- Manuscript impact: main.tex:233 "$\sigma_{\mathrm{gl,extra}} = 0.18$ [0.13, 0.26] mm/yr" has no source. σ_p,extra is described (l.249) with no number, so only the glacier value is affected. T_gl,eq can be recovered as −c/b.
- Defect: `glacier/posteriors` holds only `H0_posterior` and `posterior_samples`, and no stored output prints σ_extra quantiles.
- Evidence: HDF5 `glacier/posteriors` = ['H0_posterior', 'posterior_samples']; apeninsula the same. Grep of glacier outputs, manuscript_numbers output and the pipeline log finds no glacier σ_extra print (the only `sigma_extra (mm)` line in the log is ocean).
- Fix: Save `sigma_extra_posterior` (and `T_eq0_samples`) via `extra_posteriors`, and print the median and 90% interval in the §6 appendix or in manuscript_numbers.py.

#### MD-10 The "Greenland regional" Berkeley Earth temperature box includes Baffin Island, Iceland and Ellesmere
- Verdict: CONFIRMED (mechanism); impact on ξ, χ and AA not computed
- Severity: medium (the series is described in the tex as Greenland temperature)
- Location: notebooks/slr_data_readers.py:1505-1599 and src/slr_forecast/readers/gmst.py:90 (defaults lat 60–84, lon −73 to −12, land ≥ 0.5); component_greenland cell 3
- Reported by: mod_readers
- Manuscript impact: unknown. It feeds the ocean-transfer fit, the AA range "2.1 to 2.7" (main.tex:296) and the GMST-vs-regional sensitivity.
- Defect: A rectangular lat/lon box keeps every land cell, so non-Greenland land carries a material share of the weight.
- Evidence: my check (774 cells, cos-lat weights on the boolean mask, rough boxes): Baffin (lat < 74, lon < −60) 12.4%, Iceland 3.6%. The reviewer, using land_mask × areal_weight, got Baffin 7.9%, Iceland 2.1% and Ellesmere 1.4%. The cell count agrees.
- Fix: Mask to Greenland (a polygon or the Greenland-only file `berkEarth_greenland_surfaceT.txt`), then recompute ξ, χ and the AA ratios (Greenland cluster).

#### MD-11 `check_convergence` sees 4 of 32–64 walkers, `converged` ignores tail ESS, and the ratestate print reads keys that do not exist
- Verdict: CONFIRMED
- Severity: low
- Location: notebooks/bayesian_models.py:275-338; `n_chains_arviz = min(4, n_walkers)` at l.583, 898, 1164, 1571, 2800, 3348, 3737, 4458, 5241, 6021; bayesian_ratestate l.777-779
- Reported by: mod_bayes_A, nb_ratestate_A, nb_ratestate_B
- Manuscript impact: none
- Defect: R-hat and ESS are computed on 4 emcee walkers, which are not independent chains. `converged` checks R-hat and bulk ESS only, although the docstring says "all ESS > 100". The ratestate cell prints `conv.get('rhat_max')` and `conv.get('ess_min')`, but the dict returns `rhat`, `ess_bulk` and `ess_tail`.
- Evidence: ratestate output l.1238 `R-hat max: N/A`, l.1240 `Converged: True`.
- Fix: Use all walkers or the emcee autocorrelation time, include tail ESS in `converged`, and fix the print keys.

#### MD-12 component_results.h5 root attributes say baseline_year = 2005, n_samples = 2000
- Verdict: CONFIRMED
- Severity: low (stale metadata and a print; nothing computes from it)
- Location: notebooks/component_io.py:65-75 (`_require_file` writes attrs only on creation), 868-880 (`list_components`); results_figures cell 3 output
- Reported by: mod_io, nb_summation_A, nb_summation_B, xc_hdf5
- Manuscript impact: none
- Defect: The file was created on 2026-03-25 under the 2005 baseline, and later saves open it with mode 'a'. results_figures prints "Baseline year: 2005.0" next to its own 2000.0. n_samples is also wrong for WAIS (10000) and the slow90 groups (20000).
- Evidence: root attrs `{'baseline_year': 2005.0, 'created': '2026-03-25T00:11:58...', 'n_samples': 2000}`.
- Fix: Refresh the root attrs on every save, or write baseline_year and n_samples per group.

#### MD-13 Stale model descriptions and silent drops in component_io
- Verdict: CONFIRMED
- Severity: low
- Location: notebooks/component_io.py:8-31 (layout docstring), 315-345 (`save_ocean_hybrid` doc "projections come from IPCC AR6 oceandynamics", `model_order=0`), 378 (`save_glacier` default `'linear_dols'`, overridden), 409 (`'trend_only'`), 440 (`save_apeninsula` `'linear_dols'`, stored), 475/507 (`'smb_literature_plus_discharge_delay'`, stored), 584 (`save_wais` `rheology_mode='A'`, 1.28); 103-108 and 194-199 (silent drops)
- Reported by: mod_io, nb_apen_B, nb_greenland_B, nb_wais_A, tests
- Manuscript impact: none
- Defect: The stored `model_type` strings describe superseded models: apeninsula `linear_dols`, greenland `smb_literature_plus_discharge_delay`, eais `trend_only` (locked in by a test), and ocean `model_order = 0`. `_save_projection_ssp` silently skips arrays with ndim > 2, and the metadata loops silently drop None, NaN and list values.
- Evidence: HDF5 attrs read: `apeninsula model_type 'linear_dols'`, `greenland 'smb_literature_plus_discharge_delay'`, `eais 'trend_only'`, `ocean model_order 0`, `wais rheology_factor_median 1.28`. Code: `if val.ndim <= 2: sg.create_dataset(...)`, `elif np.isscalar(v) and np.isfinite(v)`.
- Fix: Update the strings and docstrings, and warn on any skipped key.

#### MD-14 A4 WAIS samplers (and `save_wais`) default to rheology_mode = 'A', which still multiplies S2 endpoints by R ~ N(1.28, 0.07)
- Verdict: CONFIRMED
- Severity: low (every pipeline caller passes 'B')
- Location: notebooks/component_projections.py:582, 752, 859 (defaults), 713-718 (Mode A draw); component_io.py:584; tests/test_wais_sampling.py
- Reported by: mod_comp, mod_io, tests, nb_wais_A, nb_wais_B
- Manuscript impact: none
- Defect: The rheology correction is deprecated, but omitting the argument applies a 28% S2 inflation. The comment `# rheo = 1 in Mode B` sits inside the Mode A branch. The tests exercise Mode A.
- Evidence: component_wais l.241 `RHEOLOGY_MODE = 'B'`; results_figures l.3889 `_rheo_mode = 'B'`. No caller relies on the default.
- Fix: Default to 'B' or delete Mode A, and update the tests.

#### MD-15 IPCC AR6 observed GMSL: half the 17–83% range is used as 1σ (4.6% too small)
- Verdict: CONFIRMED
- Severity: low
- Location: notebooks/slr_data_readers.py:1299; src/slr_forecast/readers/gmsl.py:379; results_figures l.1278 (`yerr=Z_90*ipcc_obs_sigma`)
- Reported by: mod_readers
- Manuscript impact: the Fig. 1(a) caption (main.tex:366) says "5–95% range bars". The drawn bars cover 1.645 × 0.954σ = 1.57σ, an 88.3% interval, so they are 4.6% short. component_summation uses `ar6_gmsl_sig` only in an unused tuple slot (l.1196).
- Defect: For a normal distribution the 17–83% half-range is 0.954σ.
- Evidence: `df['gmsl_sigma'] = (df['gmsl_upper'] - df['gmsl_lower']) / 2`, and the stored ratio is exactly 1.0.
- Fix: Divide by 2 × 0.9542, or plot the native 17–83% bounds scaled to 5–95%.

#### MD-16 `imbie_annual_rates` uses the mean of the monthly σ; the increment SD implied by IMBIE-3's cumulative σ is the RMS
- Verdict: CONFIRMED
- Severity: low
- Location: notebooks/component_analysis.py:142 (WAIS)
- Reported by: mod_comp
- Manuscript impact: negligible (WAIS σ_extra and the (a, v) covariance, at the 0.1% level)
- Defect: The IMBIE-3 cumulative σ equals sqrt(cumsum(σ_m²/12)) to about 3e-6 relative, so year Y's increment SD is sqrt(mean σ_m²).
- Evidence: `verify_MD/imbie.py`. WAIS: max (rms−mean)/mean 4.4%, mean over years 0.13%. EAIS: 6.0% and 0.23%. Peninsula: 1.0% and 0.06%.
- Fix: `np.sqrt(np.mean(sigma_m**2))`. Use one function in all three IMBIE notebooks (see the cross-cluster note on the December σ).

#### MD-17 Satellite-quadratic measurement-covariance span scaling: the code exponent is half the one in the comment
- Verdict: CONFIRMED (code vs comment); which law is right is not established, and the impact is PLAUSIBLE
- Severity: low
- Location: notebooks/bayesian_models.py:2319-2331
- Reported by: mod_bayes_A, mod_slr
- Manuscript impact: plausibly the width of main.tex:427 "0.63 [0.38, 0.86] m". The measurement term is the dominant part (the reviewer gives 0.134 of 0.148 m 1σ). Not re-derived.
- Defect: The comment says the SE scales as 1/T for c1 and 1/T² for c2. The code scales the variance by r^((p_i+p_j)/2), so the SE scales as T^(−1/2) and T^(−1). The Ablain matrix ends at 2018.58 and the fit runs to 2025.34 (r ≈ 0.79).
- Evidence: `scale_mat = r ** ((pwr[:, None] + pwr[None, :]) / 2.0)` under the comment `# on c1 (trend) scales as ~1/T and on c2 (acceleration) as ~1/T²`.
- Fix: Choose one law, make the code and comment agree, and state the extrapolation assumption in the supplement.

#### MD-18 Satellite-quadratic `cov_params` omit GIA and `sigma_inflate`, which `rate_accel_cov` includes
- Verdict: CONFIRMED
- Severity: low
- Location: notebooks/bayesian_models.py:2347-2352; consumers component_forecast cell 9, component_summation l.917
- Reported by: mod_slr
- Manuscript impact: negligible (reviewer: sd 0.1480 → 0.1488 m at 2100 with GIA added)
- Defect: The MC level bands and the rate prior use different error models.
- Evidence: `cov_params = cov_beta_meas + cov_beta_serial  # (GIA only enters the rate-accel covariance, not param cov)`.
- Fix: Add σ_GIA² to the (c1, c1) element and apply sigma_inflate², or document the difference.

#### MD-19 `_anchor_covariance` breaks the 1999.0/2001.0 tie toward 1999 and puts 1999 on the post-baseline side
- Verdict: CONFIRMED
- Severity: low (no production caller; used only in the inactive `FIT_SPACE='level'` branches)
- Location: notebooks/component_levelspace_robust_se.py:99-114
- Reported by: mod_comp
- Manuscript impact: none
- Defect: Once the zero-σ 2000.0 row is dropped from end-of-year IMBIE labels, `argmin` returns 1999.0, and `pre = arange(n) < anchor` excludes 1999 from the pre group. The covariance is then wrong and can be indefinite. `_psd_sqrt_diag` clips the negative eigenvalues silently.
- Evidence: code as quoted. All three notebooks set `FIT_SPACE = 'level_correlated'` (apen l.71, glacier l.74, eais l.123).
- Fix: Use `pre = years < baseline_year`, and raise on a materially negative eigenvalue.

#### MD-20 S1 endpoint sampler and trajectory sampler use different anchors (2023.5 default vs 2024.0 splice)
- Verdict: CONFIRMED (sub-mm)
- Severity: low
- Location: notebooks/component_projections.py:395, 457-458, 829-833
- Reported by: mod_comp
- Manuscript impact: none
- Defect: `sample_a4_wais_endpoint` uses the unspliced quadratic (H0 = 0) with the 2023.5 default anchor. The trajectories splice onto the observed 9.0 mm at 2024.0, while the quadratic gives 9.44 mm there. The constant is documented as the fit-time centring of the ISMIP6 σ polynomial, so the claim that it is "stale" is only partly right.
- Evidence: extra σ at 2100 is 58.3549 mm (anchor 2023.5) vs 58.3562 mm (2024.0), a 0.001 mm difference. Location difference: 9.44 − 9.0 ≈ 0.4 mm. wais output l.73 `Anchor: year=2024, IMBIE=9.0 ± 0.6 mm`.
- Fix: Pass the splice anchor and the observed level to the endpoint sampler as well.

#### MD-21 `apply_glacier_volume_cap` docstring promises hold-from-first-exceedance; the code clamps element-wise
- Verdict: CONFIRMED (no current effect)
- Severity: low
- Location: notebooks/component_projections.py:1206-1212
- Reported by: mod_comp
- Manuscript impact: none
- Defect: `capped[capped > v_total] = v_total` would let a trajectory fall back below the cap.
- Evidence: in the HDF5, members at the cap are 0% (SSP1-1.9 and SSP1-2.6), 5.1% (SSP2-4.5), 51% (SSP3-7.0) and 84% (SSP5-8.5). Members that hit the cap and later fall below it: 0 in every SSP.
- Fix: Correct the docstring, or apply `np.maximum.accumulate` before clamping.

#### MD-22 EN4 depth average is an unweighted mean over levels
- Verdict: CONFIRMED (code); magnitude PLAUSIBLE (reviewer's recompute, not re-derived)
- Severity: low
- Location: src/slr_forecast/readers/ocean_temp.py:163 (also l.361, Argo) → `harmonized/df_en4_greenland_200_500m` (component_greenland l.123)
- Reported by: mod_readers
- Manuscript impact: marginal (the discharge forcing anomaly changes by up to ~0.045 °C per the reviewer)
- Defect: `T_depth_avg = np.nanmean(T_c, axis=0)` weights levels 207–447 m equally, although their thicknesses range from 25 to 83 m.
- Fix: Weight by the `depth_bnds` thickness over valid levels, then rerun the Greenland discharge fit.

#### MD-23 `project_*` functions pin the baseline by `argmin` with no range check
- Verdict: CONFIRMED
- Severity: low here (module root cause of the ratestate/summation "zeroed at 2015" item, owned elsewhere)
- Location: notebooks/slr_projections.py:185, 368, 546, 794, 953, 1170
- Reported by: mod_slr (also the ratestate and summation reviewers)
- Manuscript impact: none directly
- Defect: If baseline_year lies outside the projection grid, the series is zeroed at the nearest grid point with no warning.
- Fix: Raise when |time[idx] − baseline_year| > dt/2.

#### MD-24 Other stale or misleading reader outputs
- Verdict: CONFIRMED (first and second items read from the HDF5); PLAUSIBLE (the rest, not re-derived)
- Severity: low
- Location: slr_processed_data.h5 `raw/df_imbie_wais` (negative σ; the np.abs fix 71b79dc came after the build) and `kinematics/gmst/berkeley` (built with the old σ); slr_data_readers.py:1042-1044 (Dangendorf `gia_sigma`); l.655 / gmsl.py:79 (NASA `gmsl_sigma` = column 10, along-track SD of about 89 mm); l.2700 (IMBIE-3 monthly cumulative stamped mid-month but representing end of month); l.180-185 (`_is_delta_column` treats `*_lower`/`*_upper` as differences); l.3136-3142 (`read_ipcc_ar6_component` mean/std over unevenly spaced quantiles, p17 = 0.17 rather than 0.167); ice_sheets.py:328/334 (Mouginot "peripheral glaciers excluded", while the total includes 7 ice-cap rows)
- Reported by: mod_readers
- Manuscript impact: none found (none of these columns is used numerically in the pipeline, except `read_ipcc_ar6_component` std; see cross-cluster note)
- Evidence: Dangendorf `gia_sigma` min −0.000642 m, and 99% of rows are negative.
- Fix: Regenerate with MD-4, apply abs() to the Dangendorf σ, rename `gmsl_sigma` to `gmsl_spatial_sd`, and fix the attrs.

#### MD-25 Legacy and dead functions carry superseded models or wrong priors
- Verdict: CONFIRMED (no pipeline callers)
- Severity: low
- Location: bayesian_models.py:4876 `fit_bayesian_thermosteric` (keeps drift c; init `theta0*(1+0.05N)` collapses ≈0 dimensions); slr_projections.py:709 `project_thermosteric_ensemble` (`c·(t−t₀)`, baseline 2005; imported by component_ocean l.36, never called); bayesian_models.py:6237 and 7190 (`- log_tau` spurious Jacobian with τ sampled in log space: prior median lowered by e^{−σ²}); 5578 (`_rate_accel_prior_logp(rate_model_b, 0.0, ...)` scores accel = 0); component_projections.py:1219-1383 (`project_greenland_joint_ensemble`, AA = 2.58, AA×GMST); slr_projections.py:592-702 (`project_gmsl_tau_sensitivity` resamples unseeded per τ); slr_analysis `calibrate_dols` silent alignment and kwargs drop; component_plotting dead helpers (±2σ labelled "90% CI", "rel. to 2005"); read_ismip6.py rhow 1028; models 1–4 static/DLM/hierarchical (N(0,10²) on standardized coefficients)
- Reported by: mod_bayes_A, mod_bayes_B, mod_slr, mod_comp, mod_io, mod_readers, xc_constants
- Manuscript impact: none
- Evidence: `lp += -0.5 * ((log_tau - mu_lt) / sig_lt) ** 2 - log_tau` at both lines, with `tau = np.exp(log_tau)`. A grep of scripts/, notebooks/*.py, tests and nb_src finds no callers except the unused import.
- Fix: Move them to archive/ or delete them. At minimum mark them deprecated, and drop the `- log_tau`.

#### MD-26 Stale docstrings and comments in shared modules
- Verdict: CONFIRMED
- Severity: low
- Location: bayesian_models.py:2874-2913, 3119-3122, 3146-3150 (σ_extra default "chi2/dof = 1.02 no excess scatter", against glacier 4.11 now; "b ≥ 0 enforced regardless" is false with symmetric_b; "do not reuse for a window not starting at baseline"); 2794 (`obs_times` overwritten in `fit_bayesian_level`); 3846-3853 and 4552-4553 (`solve_state_ode`/`solve_twolayer_ode` "exact for linear T", but the scheme uses an endpoint mean held constant); component_projections.py:74-88, 152-170, 208-216, 606-622 (high_mm 1000 → 1300; S1 from `fit_bayesian_level` cell 11 → `fit_imbie_time_model`; n ~ N(4.1, 0.4²)); component_levelspace_robust_se.py:12-53; config.BASELINE_WINDOW docstring (never imported by the pipeline)
- Reported by: mod_bayes_A, mod_bayes_B, mod_slr, mod_comp, mod_io, nb_glacier_A
- Manuscript impact: none
- Evidence: glacier output l.1246 `chi2/dof = 4.110` vs the docstring "chi2/dof = 1.02 with no sigma_extra". `# branches. Its 95th-percentile bound (high_mm=1000 mm)`.
- Fix: Update the text. No code change is needed.

#### MD-27 Duplicated literals (all agree today) and the Gt-to-mm convention mismatch
- Verdict: CONFIRMED
- Severity: low
- Location: listed in xc_constants (RATE_BELOW_2000 ×4, TAU_D/STRUCTURAL_FRAC in ocean_early_holdout, WAIS_VAF_M vs WAIS_MAX_SLE_MM with different citations, blend T_CENTER/TAU, adaptation economics in 8+ places, P_S1 copies, M_TO_MM/BASELINE_YEAR re-hardcoded, 362.5/362500 literals); IMBIE-3 native 360 Gt/mm vs pipeline 362.5 (eais cell 6 `PRIOR_B_MEAN = -EAIS_SMB.C_T / 362500.0`)
- Reported by: xc_constants, mod_io
- Manuscript impact: none (the Gt/mm mismatch is 0.7% on the EAIS C_T prior and the Davison correction)
- Fix: Define each constant once and import it. Use 360 Gt/mm for quantities applied to IMBIE-3 mm series, or note the difference.

#### MD-28 `build_level_design_vectors` snaps to the nearest month with no bounds check
- Verdict: CONFIRMED (latent)
- Severity: low
- Location: bayesian_models.py:1955-1958; used via slr_projections.py:1137-1144
- Reported by: mod_bayes_A, mod_slr
- Manuscript impact: none (current callers extend the grid to PROJ_YEARS[-1] + 1/12)
- Defect: Out-of-range times silently take the end values. End-of-year labels (Y.0) tie between mid-December and mid-January, and the tie resolves to December.
- Fix: Raise when a time is out of range, and interpolate the integrals.

#### MD-29 Minor numerical conventions (suspected or diagnostic)
- Verdict: PLAUSIBLE
- Severity: low
- Location: slr_projections.py:1170-1171 (projections zeroed at 2000.0 on integer-year grid vs observations' 1995–2005 mid-year mean; "2100" = level at 2100.0, ~0.5 yr below an annual mean); slr_analysis.py:932-937 (`compute_kinematics` default SEs treat kernel weights as inverse variances, ignore serial correlation; diagnostics only); component_plotting.py:914-926 (`plot_component_projection_twopanel` overlays unrebased AR6 quantiles; notebook-only figures)
- Reported by: mod_slr, mod_io
- Manuscript impact: unknown, at most a few mm at 2100 for the first item; none for the others
- Fix: Evaluate projections at yyyy.5 or as annual means. Document the SEs as nominal. Add an IPCC offset argument to the plotting helper.

#### MD-R1 "model_ensemble_draws is unseeded, so component-notebook p50 curves are not reproducible"
- Verdict: REFUTED (impact); the unseeded call is real but latent
- Severity: low
- Location: notebooks/component_analysis.py:391
- Reported by: mod_comp
- Manuscript impact: none
- Defect: In glacier (l.519/533), EAIS (l.411) and Peninsula (l.482) the calls sit only in the inactive `'rate'`/`'level'` branches. The production `'level_correlated'` branches build `_H_ens` from all posterior draws. The only active caller, results_figures l.4167, calls `np.random.seed(42)` first. component_ocean imports the function but never calls it.
- Fix: Optionally add an `rng` argument.

#### MD-R2 "Berkeley σ in the forecast panel (a) is 95%/1.96 × 1.645, which is correct" (nb_forecast_B) and "Berkeley converts 95% to 1σ" (nb_figs_A)
- Verdict: REFUTED
- Severity: n/a
- Location: see MD-4
- Reported by: nb_forecast_B, nb_figs_A
- Manuscript impact: see MD-4
- Defect: Both checked the reader code. The stored HDF5 that the figures read has unc/σ = 1.645, not 1.96.
- Fix: See MD-4.

### Cross-cluster notes
- EAIS/Peninsula use the December monthly σ as the annual increment σ. My check: it differs from the RMS (the value the IMBIE-3 cumulative σ implies) by up to 39% in a year (EAIS) and 20% (Peninsula), 10% (WAIS, which uses the mean instead). That is larger than MD-16. Antarctic cluster.
- Peninsula a ≥ 0 prior decides "BIC favors linear" (nb_apen_A/B, num_supp_1); EAIS Exponential mean used as a Normal σ for a (mod_bayes_A, nb_eais_A). Antarctic cluster. MD-1 is the module side.
- Rate-and-state: S0 initial condition differs between fit and projection; diagonal likelihood on Frederikse; τ drawn independently of (a, b, c, d) in the MVN branch; "94% HDI" label; level-space R²; `time=I0_obs` (mod_bayes_B, mod_slr). Ratestate cluster.
- Three to four AR6-GMST→Berkeley offsets (0.726/0.755/0.89 °C), and IPCC 1995–2014→2000 GMSL offsets 12.1/10.7/−11.1 mm with the summation sign error (mod_comp, xc_constants). Summation/forecast cluster.
- Greenland: `_cumulate` argmin tie at 1999.5 and end-of-year values at mid-year labels; `prepare_mouginot_components` returns the unmasked df; `extend_with_arc` trapezoids and correlated increment errors; δ mixture weights from a diagonal likelihood with the rate pseudo-observation counted twice (mod_bayes_B, mod_comp). Greenland discharge figure 2σ vs 1.645σ (xc_constants, nb_greenland_B). Legacy `fit_bayesian_level` (diagonal likelihood, level R²) used in Greenland cell 24 for the Greenland-T vs GMST choice and in ratestate cell 13 (mod_bayes_A). Greenland and ratestate clusters.
- results_figures cell 35 reads `metadata['r2_dyn']` (always N/A) and prints rheology 1.28 (mod_io, nb_figs_B). Figures cluster.
- Warning-label maps 2/3/4 °C in forecast cell 18, results_figures cells 41/52 and compute_shapley (xc_constants). Figures cluster.
- glacier_delayed_response.py treats GlaMBIE 95% as 1σ and holds a stale PROD_2100 (xc_constants). Glacier/scripts cluster.
- Glacier cap binds in 5–84% of members by 2150, against the markdown's 0.1% (nb_glacier_B, nb_summation_A). Glacier cluster.
- `read_ipcc_ar6_component` std (moments of the quantile values) feeds S1_ISMIP6_STD_COEFFS (nb_wais_B). WAIS cluster.
- build_slow_wais_p90 assert at 10k; precompute_ipcc_distributions not in run_pipeline.sh (scripts_down, nb_forecast_B, xc_hdf5). Tests/HDF5 cluster.
- Ocean: Berkeley ends at 2024.5 but NOAA at 2025.5 (argmin), and projections are zeroed at 1999.5 (nb_ocean_A/B). Ocean cluster.

---

## B7. Figures and manuscript text (FT)

Verification scripts and outputs: `$S/verify_FT/` (f1–f5.py, otosaka.txt). MAIN = `01_slr_forecast_intervention2026.tex`, SUPP = `..._supp.tex` (line numbers as of the current files).

#### FT-1 Compiled manuscript and supplement PDFs predate this run's figures
- Verdict: CONFIRMED
- Severity: medium
- Location: manuscripts/00_ddpi_slrforecast2026/v1/01_slr_forecast_intervention2026.pdf, ..._supp.pdf
- Reported by: xc_hdf5
- Manuscript impact: every \includegraphics figure in both compiled PDFs (pre-run versions)
- Defect: main PDF was built Oct 16 17:47:00 (main .tex 17:45:03), supp PDF Oct 16 21:52:58. Every pipeline figure they include was rewritten between 22:28 and 23:50. The compiled PDFs therefore show pre-run figures. The main .tex text also predates the run, although the number sweeps found its values match this run.
- Evidence: `stat`: component_ocean_posterior_corner.png 22:28:02, component_ocean_fit_residuals.png 22:28:34, component_greenland_discharge_delay_fit.png 23:02:29, fig1v2_observations.png 23:50:03, fig2v6_fan_exceedance2100.png 23:50:09, fig_slr_by_component_v2.png 23:50:30, component_ocean_early_holdout.png 22:57:14, supp SMB PDFs 23:04. The one exception is fig1_spectrum_schematic.pdf (Oct 13). It is hand-made and no pipeline step regenerates it.
- Fix: rebuild both PDFs after the run (install-regenerated-figures rule).

#### FT-2 Fig. 4(a): the Greenland dashed line does not split the discharge and SMB anomalies as the caption says
- Verdict: CONFIRMED
- Severity: medium
- Location: results_figures:cell 55 (same in cell 54); MAIN:448 caption
- Reported by: nb_figs_B, num_main_3
- Manuscript impact: MAIN:448 "Green dashed lines within Greenland bars separate the discharge anomaly (below) from the SMB anomaly (above)". This is a caption/figure mismatch; no number changes.
- Defect: below the line is the median discharge minus its 2000 rate carried forward. Above the line is the ensemble-mean total Greenland minus that quantity. The upper segment is therefore the SMB anomaly plus the 2000 net imbalance carried forward (0.147 mm/yr × (t−2000)), not the SMB anomaly alone. The construction also mixes a median (discharge) with a mean (bar).
- Evidence: `_d_rate_0 = (disch[_baseline_idx + 1] - disch[_baseline_idx - 1]) / 2.0`; `y_line = grl_bottoms_v2[...] + max(d_anom, 0)`. My recomputation for SSP2-4.5 from HDF5: discharge rate at 2000 1.213, SMB −1.064, total-mean 0.147 mm/yr. At 2050/2075/2100 the drawn upper segment is 21.2/38.7/62.9 mm, against SMB anomalies (relative to the 2000 SMB rate) of 13.8/27.7/48.0 mm. About 15 mm at 2100 is committed imbalance labeled as SMB.
- Fix: subtract both 2000 rates and use the same statistic (mean) for both parts, or reword the caption, e.g. "discharge above its 2000 rate (below) and all other Greenland change (above)".

#### FT-3 Main-text hindcast says WAIS "follows its fit", but the code uses the IMBIE-3 record (contradicts l.344)
- Verdict: CONFIRMED
- Severity: medium
- Location: MAIN:359; component_summation cell 11 (`wais_2k/full_samples`)
- Reported by: num_main_2, nb_wais_B, num_main_3 (pre-1950 part; see Cross-cluster)
- Manuscript impact: MAIN:359 "WAIS follows its fit to the IMBIE-3 record" is a wrong method description and disagrees with MAIN:344 ("enter as the IMBIE-3 records ... through 2023, continued by the WAIS fit"). No number changes.
- Defect: through 2024 the hindcast WAIS trajectories reproduce the observed IMBIE-3 cumulative record, held constant before the record starts. They are not the quadratic fit.
- Evidence: median of wais_2k/full_samples vs wais/observations/H_obs (mm): 1981 −2.080 vs −2.083; 1982 −2.016 vs −2.021; 1983 −1.880 vs −1.885; 2015 5.356 vs 5.354; 2024 8.981 vs 8.985. The level is flat at −2.055 for years ≤ 1980.
- Fix: MAIN:359: "WAIS follows the IMBIE-3 record (held at its first value before 1979)".

#### FT-4 Table S2 defines T relative to 1850–1900; every component model uses T relative to 1995–2005
- Verdict: CONFIRMED
- Severity: medium
- Location: SUPP:140 (T row), SUPP:141 (ΔT row)
- Reported by: num_supp_1
- Manuscript impact: SUPP:140 "GMST anomaly, referenced to 1850--1900" (should be 1995–2005 for the model equations). SUPP:141 ΔT row (EAIS) defines a symbol that MAIN does not use. MAIN uses ΔT only as ΔT_ocean in eqs. 128–138, and its EAIS equation uses T.
- Defect: the T that forces the ocean, glacier, Peninsula, EAIS and SMB fits is rebased to 1995–2005. MAIN:151 says so, as do the table's own rows for c_gl, c_p and c_e ("rate at the 1995--2005 mean GMST"). 1850–1900 applies only to the scenario labels (MAIN:80). Read as written, the table changes the meaning of every intercept.
- Evidence: harmonized/df_berkeley_h annual mean 1995–2005 = 4.3e-17 °C; 1850–1900 = −0.872 °C. The glacier prior T_eq0 ~ N(−0.857, 0.15) is negative, so it is on the 1995–2005 baseline.
- Fix: define T as the GMST anomaly relative to 1995–2005, note that the scenario labels use 1850–1900, and drop the ΔT row.

#### FT-5 Table S2 lists symbols the main text does not use, misses some it does, and says "six" thermosteric parameters
- Verdict: CONFIRMED
- Severity: low
- Location: SUPP:115 (intro: "lists every symbol that appears in the main text ... Symbols used only in this supplement ... are not repeated here"), SUPP:121 caption, rows 187, 206, 232, 234, 235, 258, etc.
- Reported by: num_supp_1
- Manuscript impact: SUPP:187 "Vector of the six fitted thermosteric parameters" → five (MAIN:191 φ = (a_u, b_u, b_d, τ_u, σ_extra); five arrays in ocean_posteriors). Rows with no occurrence in MAIN (grep -F count 0): R^2 (SUPP:258), \epsilon_{fb}, C_{T^2}, \dot{M}_0, \sigma_s, \sigma_\varepsilon, H_{e,0}, a_p, and ΔT in its EAIS meaning. MAIN symbols missing from the table: \alpha_{0,sw} (MAIN:162, 167, 174), \varphi_d and \Delta\theta_d (MAIN:174), \sigma_{\varepsilon_{go}} (MAIN:294), \tau_s (FT-7).
- Defect: the table breaks its own stated rule in both directions. The R² row survives although R² was dropped from the paper.
- Evidence: grep counts listed above.
- Fix: change "six" to "five", drop the supplement-only rows (or soften the intro sentence), and add the missing main-text symbols.

#### FT-6 Table S2 gives β = 2 as a fixed value; it is drawn lognormal with median 2
- Verdict: CONFIRMED
- Severity: low
- Location: SUPP:221; notebooks/component_projections.py:141, 722, 737
- Reported by: num_supp_1
- Manuscript impact: SUPP:221 value "2" → "median 2 (lognormal, log-sd 0.3)". MAIN says "median exponent".
- Defect: β is sampled as `crng.lognormal(s['beta_loc'], s['beta_scale'])` with beta_loc = ln 2 and beta_scale = 0.3. It is not fixed.
- Evidence: `'beta_loc': np.log(2.0), 'beta_scale': 0.3`.
- Fix: as above.

#### FT-7 Table S2 (and the supplement text) name the drift timescale τ; the main text calls it τ_s; s is given in Gt/yr but added to a mm/yr equation
- Verdict: CONFIRMED
- Severity: low
- Location: SUPP:229 (τ row), SUPP:228 (s units), SUPP:350 (text uses "\tau"); MAIN:151 (`\tau_{\mathrm{s}}` three times; the equation is in mm SLE/yr)
- Reported by: num_supp_1
- Manuscript impact: notation only
- Defect: the symbol in the table and the supplement prose does not match the main text. The units of s are inconsistent with the main-text equation it enters.
- Evidence: SUPP:229 `$\tau$ & Relaxation timescale of the drift state`; MAIN:151 `$\dot{H}_{\mathrm{smb}} = a_{\mathrm{gris}}T^2 + ... + s(t)$`.
- Fix: rename to τ_s throughout the supplement; give s in mm SLE/yr or state the 362.5 Gt/mm conversion.

#### FT-8 Fig. 1(a) dashed "median forecast" is shifted +6 mm from the stored blended median
- Verdict: CONFIRMED
- Severity: low
- Location: results_figures:cell 18 (`_fc_offset`, code l.1296–1302); MAIN:368 caption
- Reported by: nb_figs_A, num_main_3
- Manuscript impact: figure only. The caption calls it "the median forecast"; the drawn 2100 value is 758 mm against the quoted 0.75 m (752 mm). The rounding is unchanged.
- Defect: the line is the blended SSP2-4.5 median plus a constant that anchors it to the blended hindcast's 2025 level plus one year of its rate. That hides a 4.4 mm mismatch between the hindcast end (87.9 mm at 2025) and the forecast origin (83.5 mm at 2025.34).
- Evidence: f1.py: `offset mm 5.999`, `2100 752.09 → 758.09`, `h20 2025 87.95`, `h_origin 83.52`.
- Fix: plot the stored median unshifted from (t_origin, h_origin_m), or state the anchoring in the caption.

#### FT-9 Fig. 1(a) observational series use mixed zero points; IPCC-obs is re-zeroed on an argmin-tie sample
- Verdict: CONFIRMED (magnitude corrected)
- Severity: low
- Location: results_figures:cell 15 (`nasa_gmsl_rb`, l.857–859), cell 18 (`ipcc_obs_rb`, l.1199–1200)
- Reported by: nb_figs_A
- Manuscript impact: figure only (MAIN:368)
- Defect: the harmonized Frederikse and Dangendorf series are plotted as stored, zero on their 1995–2005 mean. NASA and IPCC-obs are re-zeroed on a single sample nearest 2000. IPCC-obs is annual at yyyy.5, so `argmin(|t−2000|)` ties between 1999.5 and 2000.5 and returns 1999.5. That shifts IPCC-obs up by 3.2 mm (the reviewer's 2.2 mm used the 2000.5 value). NASA shifts up by 0.5 mm.
- Evidence: f2.py: IPCC-obs harmonized 1995–2005 mean = 0; values 1999.5 −3.245, 2000.5 −2.175 mm. NASA nearest sample 1999.990, −0.520 mm.
- Fix: leave the harmonized series as stored (they already share the 1995–2005 baseline).

#### FT-10 Fig. 1(a) GMST projection line is not in the caption; it is spliced to one noisy annual value
- Verdict: CONFIRMED
- Severity: low
- Location: results_figures:cell 18 (l.1307–1321); MAIN:368 caption
- Reported by: nb_figs_A
- Manuscript impact: caption omits a plotted, legended element ("GMST projection (IPCC AR6 2.7 °C)"). The caption says only "GMST (red) ... is referenced to the right-hand axis".
- Defect: the offset is `np.interp(2020, yrs_ann, temp_ann)`, a single unsmoothed annual Berkeley value. Interannual noise in 2020 therefore sets the line's vertical position.
- Evidence: code quoted.
- Fix: mention the projection in the caption, and splice to `temp_smooth` or a multi-year mean.

#### FT-11 Fig. 1(b) legend calls the Hamlington quadratic-rate diamonds "GMSL (satellite altimetry)"
- Verdict: CONFIRMED
- Severity: low
- Location: results_figures:cell 18 (`_labels_b2`, l.1549–1550; diamonds from `sq_rate`, l.1412–1418)
- Reported by: nb_figs_A
- Manuscript impact: figure legend in MAIN Fig. 8b. The caption (MAIN:369) is correct.
- Defect: the diamonds are `HAMLINGTON_ACCEL*(t−T_REF)+HAMLINGTON_RATE`, the rate of the published quadratic, not rates estimated from altimetry.
- Evidence: code quoted.
- Fix: label "Satellite-era quadratic (Hamlington et al., 2024)".

#### FT-12 Fig. 4 caption title says "variance decomposition"; panel (b) shows 90%-interval shares
- Verdict: CONFIRMED
- Severity: low
- Location: MAIN:447; results_figures:cell 55
- Reported by: nb_figs_B
- Manuscript impact: MAIN:447 title "Component contributions, variance decomposition, and value of information". The panel (b) text in the caption is correct.
- Defect: the pies use `_w90 = p95 − p5`, and the code comment says variance is deliberately not used.
- Evidence: cell 55 `_w90 = lambda a: np.percentile(a, 95) - np.percentile(a, 5)`, "Variance is not used here".
- Fix: e.g. "Component contributions, interval shares, and value of information".

#### FT-13 "30-yr local regression" is a tricube kernel with a 30-yr half-width (60-yr support)
- Verdict: CONFIRMED
- Severity: low
- Location: notebooks/slr_analysis.py:886, 901, 909 (docstring l.798, 839); results_figures cell 15 `RATE_SPAN_YEARS = 30`; MAIN:369, 375
- Reported by: mod_slr
- Manuscript impact: wording, MAIN:369 and 375 ("30-yr local regression")
- Defect: `h = span_years`, `u = (t−t0)/h`, and the tricube is nonzero for |u| ≤ 1. Each local fit therefore uses ±30 yr (FWHM ≈ 35 yr). The docstring says edge effects begin within span_years/2; they actually begin within span_years. So every Frederikse rate after about 1988 is one-sided.
- Evidence: code quoted.
- Fix: "a local quadratic regression with a 30-yr tricube half-width"; fix the docstring.

#### FT-14 "1 mm SLE = 362.5 Gt throughout this study" does not hold for the IMBIE-3 inputs (360 Gt/mm)
- Verdict: CONFIRMED (against the primary source)
- Severity: low
- Location: MAIN:102; SUPP:165; notebooks/slr_data_readers.py:2644, 2685–2687
- Reported by: num_main_1, xc_constants, nb_eais_B
- Manuscript impact: MAIN:102 statement. All IMBIE-3-derived series (WAIS/EAIS/Peninsula, Greenland validation) are about 0.7% smaller than at 362.5.
- Defect: the reader takes the provider's mm columns directly. The pipeline's Gt-based priors and corrections (EAIS C_T prior, Davison correction) use 362.5.
- Evidence: Otosaka et al. 2026 PDF (data/raw/ice_sheets/imbie2026): "one with the data provided in ... millimetres (mm) (assuming that 360 Gt of ice lost is equivalent to 1 mm sea level rise)". The reader reads 'Mass balance (mm/yr)' and 'Cumulative mass balance anomaly (mm)'.
- Fix: rescale IMBIE-3 mm by 360/362.5 on read, or qualify the sentence.

#### FT-15 SMB–elevation feedback "truncated at zero" is a clip (point mass at zero)
- Verdict: CONFIRMED
- Severity: low
- Location: MAIN:151; SUPP:232, SUPP:316; notebooks/smb_emulator.py:250–252
- Reported by: num_main_1, num_supp_2
- Manuscript impact: wording only (the effect on numbers is negligible)
- Defect: `np.maximum(rng.normal(0.08, 0.05, n), 0.0)` sets 5.1% of members to exactly zero. That is a censored normal, not a truncated one. The docstring says "truncated" too.
- Evidence: greenland output "Elevation feedback eps: median 0.080, 5.1% truncated at 0".
- Fix: "with negative draws set to zero", or resample negatives.

#### FT-16 Supplement drift-state out-of-sample test: the anchor is 1995–2004, and the state forecast is worse, not "no better"
- Verdict: CONFIRMED
- Severity: low
- Location: SUPP:350; scripts/smb_state_space_diagnostic.py:231, 239
- Reported by: num_supp_2
- Manuscript impact: SUPP:350 "no better than a fixed 1995--2005 anchor" → "slightly worse than a fixed 1995–2004 anchor"
- Defect: for the 1972–2004 fit, the anchor is `smb_m[(yrs_m >= 1995) & (yrs_m <= 2004)].mean()`. The state forecast's mean is further from the observed value, and its band covers fewer years.
- Evidence: log l.204–210: "M0 (1995-2004) 374"; "mean SMB: observed 260; state 379 [249, 476]; production 361 [246, 470]"; "annual coverage ... state: 64%" vs "production: 100%"; both at the 7th percentile of the cumulative.
- Fix: as above.

#### FT-17 Supplement claims with no pipeline output behind them
- Verdict: CONFIRMED
- Severity: low
- Location: SUPP:379 ("nor a quadratic dependence on GMST accounts for the difference"); SUPP:350 ("widens its 90% interval"); SUPP:314 (SMB lag test "did not improve the fits")
- Reported by: num_supp_2
- Manuscript impact: three qualitative claims are untraceable
- Defect: (i) glacier_delayed_response.py (log l.75–80) and scripts/glacier_zemp_diagnostic.py test only linear and delayed models on the Zemp 1962–1989 / 1990–2016 split. Neither contains a quadratic fit, and the notebook's quadratic is on GlaMBIE only. (ii) For the drift state, the log prints the change against the anchor (−2.0 [−41.3, 32.6] mm) and M 388 [281, 522] vs M0 374 [283, 463], but no 2100 90% width with and without the state. (iii) scripts/joint_lag_fit_mar.py is not in run_pipeline.sh.
- Evidence: grep for "quadratic" in both glacier scripts gives no fit; run_pipeline.sh has no joint_lag entry.
- Fix: add the quadratic test and the width print to the pipeline scripts and add the lag script to the pipeline, or drop or soften the claims.

#### FT-18 Fig. S(rcm) caption says its 2100 values "are those of" Fig. S(lomo); two independent computations differ slightly
- Verdict: CONFIRMED
- Severity: low
- Location: SUPP:337 caption; scripts/plot_greenland_smb_mar_vs_emulator.py vs scripts/plot_supp_greenland_smb_emulator.py
- Reported by: num_supp_2
- Manuscript impact: caption wording. The tex quotes the LOMO values (+28.7 mm).
- Defect: the two scripts recompute the LOMO prediction separately.
- Evidence: log l.134/163 CESM2 SSP1-2.6 37.4 [13.7, 61.0] vs 37.6 [14.2, 60.9]; l.143/172 UKESM SSP5-8.5 miss +28.7 vs +28.8.
- Fix: share one computation, or write "match, up to sampling noise,".

#### FT-19 Opening of the supplement posterior section overstates its coverage and what the projections carry
- Verdict: CONFIRMED
- Severity: low
- Location: SUPP:266
- Reported by: num_supp_1
- Manuscript impact: SUPP:266 "collects the parameter posteriors of every calibrated component model" and "they are the values carried into the projections"
- Defect: the section has no posterior figure for the WAIS quadratic-in-time MCMC fit (with σ_extra) or for the Greenland SMB drift state (τ, σ_s, σ_ε). The projections carry full posterior draws, not the summary medians and intervals.
- Evidence: SUPP figure labels 272–302 cover only Greenland discharge, ocean, glacier, Peninsula and EAIS. The HDF5 holds the posterior draw arrays (e.g. ocean_posteriors 1.92M draws; glacier/posteriors/posterior_samples 64000×3).
- Fix: "of the calibrated rate–temperature and discharge models"; "the full posterior draws are carried into the projections".

#### FT-20 VoI "present values" in the main caption are present values at the arrival year
- Verdict: CONFIRMED (main-text wording only; the supplement states it)
- Severity: low
- Location: MAIN:450 caption, MAIN:457; notebooks/compute_evpi.py:105–110, 197; SUPP:527
- Reported by: num_main_3
- Manuscript impact: none for quoted values (arrival 2026). The Fig. 4(c) surface for later arrivals is PV at arrival, not PV in 2026.
- Defect: `VOI_2 = evpi_ann * _npv_factor(ay, 2100)` is discounted to the arrival year. SUPP:527 says so explicitly ("present value at the year of arrival"), but the main caption's "present values ... at a 3% discount rate" implies today's value. Each year's benefit is the 2100-distribution annual EVPI, which the main text does not state. The qualitative claim (value declines with later arrival) holds either way.
- Evidence: code quoted; SUPP:527–529.
- Fix: main caption: "present values at the year of arrival", or multiply by (1.03)^−(t_a−2026).

#### FT-21 Main/supp numbers with no printed pipeline source (consolidated list)
- Verdict: CONFIRMED (values correct where recomputed; not traceable)
- Severity: low
- Location: MAIN:80 (1.7/2.7/3.9 °C in 2099); MAIN:104, 136 (219 glaciers, 107, 85%); MAIN:131–146 and SUPP:156–160 (κ 36–104 median 60, μ 3–10, A_f, γ̃ 0.05–1.5, "factor of 30"); MAIN:191 (τ_u prior [3.5, 18] yr); MAIN:196 (42/30 mm observed steric rise; convergence 226 steps/1/133/≥8500 ESS printed only by the oceanholdout script)
- Reported by: num_main_1, num_supp_1
- Manuscript impact: traceability (feedback_numbers_traceable_to_code)
- Defect: none of these is printed by manuscript_numbers.py, a §6 cell or a pipeline script. Reviewers recomputed them and they match: 1.724/2.748/3.909 °C; γ̃ 0.053–1.53; exp(ln 8 ± 1.645·0.5) = 3.51/18.2; 41.8/29.7 mm. I confirmed that manuscript_numbers.py prints only the 2099 half-width and the warming added, not the 2099 levels.
- Fix: add these to manuscript_numbers.py (or a §6 cell).

#### FT-22 Slide and ridge "2025"/"2075" IPCC distributions are IPCC 2020/2070 (argmin tie on decadal years)
- Verdict: CONFIRMED
- Severity: low (re-rated from medium: no figure in the main or supp tex uses them)
- Location: results_figures:cell 20 (l.1664–1665), feeding cells 22 and 27
- Reported by: nb_figs_A
- Manuscript impact: none (presentation slide fig_gmsl_pdf_2025_slide.png, fig2v2)
- Defect: ridge years are [2020, 2030, …], so `argmin(|yrs−2025|)` returns 2020 and 2075 returns 2070. The ≤5 tolerance accepts both.
- Evidence: f3.py: 2025 → 2020, median 60.7 mm (printed "IPCC AR6: 61 mm"); interpolated to 2025 it would be 83.0 mm. 2075 → 2070, 347.9 mm vs 382.1 interpolated. The 2025 slide's IPCC deficit is about 22 mm too large.
- Fix: interpolate the quantiles to the target year, or raise when the matched year differs.

#### FT-23 Unused figures and prints carry stale labels and metadata (results_figures)
- Verdict: CONFIRMED
- Severity: low
- Location: results_figures cells 3, 11, 12, 22, 26, 28, 35, 41, 52–57, 61, 62; risk_shapley attr
- Reported by: nb_figs_A, nb_figs_B, mod_io, num_main_3
- Manuscript impact: none (none of these figures or prints is in the tex)
- Defect: (a) cell 28 slide labels `blended_slow90` (p(S1)=0.9 mixture) "stable Thwaites", and its comment says "S1 only". (b) cell 54 v1 labels pure `wais_s1` "Slow WAIS". (c) integer 2/3/4 °C labels in cells 22, 26 (`r'2$^\circ$C'`), 41 (`SSP_LABELS_ORT`), 52 (`_WARMING_MAP`, used in 53/56/57) and the risk_shapley attr. (d) Stale prints: cell 3 "Baseline year: 2005.0" (stale root attr); cell 11 "Headline stats (uncorrected baseline; ...)" over corrected values; cell 12 "Ratio: 4x" (tex 3.5) and a 91.5% vs 91.7% "check" of different quantities; cell 35 "WAIS rheology factor: 1.28", "Glacier R²", discharge R² always "N/A" (wrong key `r2_dyn`); cell 61 model_type linear_dols/trend_only; cell 62 IMBIE QC compares IMBIE-3 with itself (old 8.60 = new 8.60). (e) cells 56/57 comments say "$200B/yr per meter" (the code uses 134.67). (f) cell 41 set_xticklabels without set_xticks (matplotlib warning).
- Evidence: outputs l.10, 224, 241, 436, 443, 678–688; code l.3293, 3319.
- Fix: relabel or delete the unused variants, and fix the prints and comments.

#### FT-24 IPCC-to-2000 rebase differs between figure data (10.7 mm) and text numbers (12.1 mm); cell 36 adds the total offset to each AR6 component
- Verdict: CONFIRMED
- Severity: low
- Location: data/processed/ipcc_distributions.h5 attr (mtime Oct 13 23:31, precompute not in run_pipeline.sh), used in results_figures cells 20, 26–28; cell 36 (l.116–118 of the cell, `ie['q50'] + ipcc_offset_mm`)
- Reported by: nb_figs_A, nb_figs_B, num_main_3
- Manuscript impact: none. The fig1v2 and fig2v6 code paths do not read IPCC distributions; the text uses 12.1 mm (MN).
- Defect: these are duplicated constants that disagree by 1.4 mm. Cell 36 applies the GMSL offset to glaciers/GIS/ocean/AIS individually, the per-component rebase pattern fixed on 2026-10-14.
- Evidence: attrs `{'rebase_mm_total': 10.7, 'baseline_year': 2000.0}`; MN "12.1 mm".
- Fix: regenerate ipcc_distributions.h5 with the shared offset (and add it to the pipeline), and drop the per-component offset in cell 36.

#### FT-25 Fig. 1(a) GMST 90% band: annual σ assumes independent monthly errors, and the stored Berkeley σ is 1.19× too large
- Verdict: PLAUSIBLE
- Severity: low
- Location: results_figures:cell 15 (l.880–883), cell 18 (l.1218–1219); slr_processed_data.h5 /harmonized/df_berkeley_h
- Reported by: nb_figs_A (√12), mod_readers (stale σ)
- Manuscript impact: the width of the GMST band in MAIN Fig. 8a; no numbers
- Defect: `sqrt(sum(s²))/len(s)` divides the monthly σ by about √12. Berkeley's monthly errors are largely correlated within a year, so this likely understates. Separately, the stored σ is unc/1.645, not unc/1.96 (I verified the ratio is 1.645 for every row). The two errors partly offset, and the net direction depends on the monthly error correlation, which is not demonstrated.
- Evidence: `(temperature_unc / temperature_sigma)` = 1.645 (min = max).
- Fix: use Berkeley's annual uncertainty, and refresh the stored Berkeley σ (readers cluster).

#### FT-26 Fig. 3 (cell 30, not in tex) uses inline reconstructions with mixed conventions
- Verdict: CONFIRMED
- Severity: low
- Location: results_figures:cell 30 (l.3569 `_be_T_oc = _ot_alpha * GMST + _ot_beta`; l.4001 `np.polyfit` IMBIE quadratic), cell 29 markdown
- Reported by: nb_figs_A, nb_ocean_A, nb_ocean_B
- Manuscript impact: none (fig3_calibration_validation.png is not \includegraphics'd)
- Defect: the Greenland extension maps raw GMST through a transfer function fitted to Greenland regional T (no AA factor). The WAIS "extrapolated observations" use an unweighted OLS quadratic, not the S1 fit. Interval conventions are mixed (1σ/2σ/IQR/17–83%), "Slow WAIS" is pure S1, and the markdown describes 6 panels while the code draws 8. Panel (a)'s full-depth curve mixes a 2005.5 anchor with a 2000-referenced rate term.
- Evidence: code quoted.
- Fix: fix the figure if it is revived; otherwise delete it.

#### FT-27 Fig. S0(a) Zemp back-run uses calendar-year GMST; the validation numbers use hydrological-year GMST
- Verdict: CONFIRMED (construction); visual effect not quantified
- Severity: low
- Location: results_figures:cell 32 `_glacier_fit_panel` (zemp branch); component_glacier l.772–773 (`_T_hyd = 0.25 T(Y−1) + 0.75 T(Y)`)
- Reported by: nb_figs_B
- Manuscript impact: none (the tex rates come from component_glacier)
- Defect: the plotted out-of-sample curve and the quoted 0.30 [0.17, 0.40] mm/yr come from slightly different temperature constructions.
- Fix: use the hydrological-year T in the figure.

### REFUTED / corrected
- nb_figs_A "checked consistent: Berkeley converts 95% to 1σ" for the Fig. 1(a) GMST band. REFUTED for the stored data: the HDF5 table was built with /1.645 (ratio 1.645 on every row). The reader is fixed, but the table is stale. See FT-25 and mod_readers.
- nb_figs_A Fig. 1(a) IPCC-obs shift "+2.2 mm". The magnitude is wrong: the argmin tie selects 1999.5, so the shift is +3.2 mm (FT-9).
- nb_figs_A "ridge tie: medium". Re-rated to low, because no manuscript figure uses it (FT-22).
- num_main_3 VoI "PV at arrival ... as the caption implies". This is not undisclosed: SUPP:527 states it explicitly. It remains a main-caption wording issue only (FT-20).

### Old-report items
Not assigned.

### Cross-cluster notes
- Satellite-era hindcast rates 2.98/3.11/3.07/3.25 are the unblended component sum, while MAIN:361 says blended (num_main_2, num_main_3). Summation.
- The "shortfall arises after 2018" attribution at MAIN:346 is unsupported (num_main_3). Summation.
- Pre-1950 hindcast ocean lacks the below-2000 m term (0.76 mm/yr, 39–45 mm at 1900) (num_main_3). Summation/ocean.
- WAIS set to zero before 1950 in the 1900 hindcast, a 2 mm step (nb_summation_A/B, nb_wais_B, num_main_3). Summation.
- σ_extra thermosteric 2.0 vs 1.95 rounding; ocean residual band ±1σ vs "90%" caption; fit window 2025.5 vs forcing 2024.5 (num_main_1, num_main_2, nb_ocean_A/B). Ocean.
- Glacier σ_gl,extra 0.18 [0.13, 0.26] is untraceable (not saved to HDF5); "fiftyfold unchanged"; stale comment about mass gain before 1935/1970 (num_main_2, num_supp_1). Glacier.
- EAIS "−14 to −25 mm" → −13 to −24; EAIS prior print N(0, 0.5); EAIS caption "informative" (num_main_2, num_supp_1). EAIS.
- ISMIP6 whole-AIS comparison for S1 and its baseline (num_main_2). WAIS.
- Warming-path description omits the 1/1.645 scaling (MAIN:282) (num_main_2). Forecast/warming paths.
- Per-component forecast legend 2/3/4 °C (nb_forecast_B, xc_constants, num_main_3). Forecast.
- AR6-baseline offset 12.1/11.1/10.7 mm across paths; satellite-quadratic 627/629/631; WAIS share 9.1 vs 9.9%; three variance-share computations (num_main_3, scripts_down). Forecast/summation/scripts.
- Corner-plot contour levels "1, 2, 3 σ" vs corner defaults; Peninsula σ_extra is the third, not the fourth, parameter; AP/EAIS subscript notation (num_supp_1, mod_io, nb_*). Component notebooks.
- Table S2 a_p row "BIC favors linear" (prior a ≥ 0 makes ΔBIC ≈ −ln 45) (num_supp_1, nb_apen_A). Peninsula.
- γ interval 0.421 (main) vs Fig. S1 title 0.422; 79 vs 81 Gt/yr departure; stale 1975–2018 markdown (num_main_1, num_supp_1, num_supp_2). Greenland.
- Ocean early-holdout level 15.0 [12.0, 17.6] vs pipeline 15.0 [12.2, 17.5]; warming-path 82 → 157 mm comparator wording (num_supp_2). Ocean.
- LOMO "as in the projections" anchor wording (num_supp_2); Peninsula skill-table row omits σ_extra (num_supp_2). Greenland/Peninsula.
- Fast vs slow variance-share methods (realized vs analytic) (num_supp_2). Scripts.
- Stale Berkeley σ in slr_processed_data.h5 also affects the component_forecast per-component figure band (mod_readers). Readers.
- Stale root attrs of component_results.h5 (baseline_year 2005) (mod_io, xc_hdf5). IO.

---

## B8. Tests, HDF5 data flow, rate-and-state (IN)

**Does any rate-and-state value reach the manuscript or supplement? No.**
- No hits for rate-and-state/ratestate in either .tex file.
- No \includegraphics target comes from bayesian_ratestate.
- No script reads `bayesian_ratestate_posterior.h5`. grep of scripts/, notebooks/*.py and src finds only the defining modules bayesian_models.py and slr_projections.py.
- Fig. 1b (`fig1v2_observations.png`, results_figures cell 18): the rate-and-state scatter is commented out (l.1397–1399). `h_rs2` is only a legend handle for the component-model gradient. The R07 line uses the literal `a_R07 = 3.4`.
- Tex 375 ("slope of the observed rate ... agrees with 3.4") is a visual claim against that literal line. It is not the rate-and-state "Record-averaged sensitivity 3.5 [3.0, 4.2]" (results_figures cell 6 print). num_main_2's mapping of that sentence to the 3.5 is loose, though it does not matter.
- No sentence in either tex covers the rate-doubling cell (results_figures cell 13).
- Tex 427's semi-empirical 0.73 [0.67, 0.82] m comes from Rahmstorf (2007) via manuscript_numbers/forecast cell 23, not from this model.
- In component_summation, the rate-and-state output is used only in cells 14 and 19–21: the diagnostic figures and prints `component_summation_total/no_ais.png`. Cell 10 makes the manuscript figure `component_hindcast_vs_observed.png`, and it does not use the rate-and-state output.

So all rate-and-state items below are notebook/diagnostic-only (manuscript impact: none).

#### IN-1 Greenland §6 sensitivity reads `blended`, `blended_stable`, `blended_slow90` before the steps that write them; tex 433 numbers come from it
- Verdict: CONFIRMED (ordering and tex link); numeric impact PLAUSIBLE (small)
- Severity: high (certain crash of the planned 10k final run; tex 433 numbers computed against stale ensembles)
- Location: component_greenland cell 26 (extracted l.1775–1781); run_pipeline.sh ENTRIES (greenland is #5; forecast #14 and slowwais #15 write the groups); `set -euo pipefail` (l.16)
- Reported by: xc_hdf5, nb_greenland_A
- Manuscript impact: main tex 433: "moves the 2100 median by 3 mm or less" and "Neither test changes the width of the 90% credible interval by more than 2 mm". These come from cell 26's output `Largest change in a forecast median: 2.7 mm; in a 90% interval width: 1.8 mm`. The reviewers said "no tex impact". That is wrong: the numbers are in the text. They were computed by adding this run's Greenland perturbation δG to the previous run's `blended` ensembles, which predate the 2026-10-16 drift-term removal. The deltas are differences, so the rounded values (3 mm, 2 mm) probably do not change. That was not demonstrated.
- Defect: cell 26 builds its "full" and "slow" deltas on the ensembles from the previous run. Its assertion compares only blended with blended_slow90, and those two are stale together, so it cannot detect this. On a fresh component_results.h5 the cell raises KeyError and the pipeline stops at step 5. On the planned N_SAMPLES=10000 run the crash is certain. At step 5, `blended` in the file still holds the previous 2k members, so `_full[s][:, _if] + _dT[:, _if]` (2000 vs 10000 rows) and `_rows1`/`_stab[s][_rows1]` fail, and `set -e` stops the pipeline.
- Evidence: code `_full = {s: _hf[f'blended/{s}/samples'][:] ...}; _slow_stored = {s: _hf[f'blended_slow90/{s}/samples'][:] ...}`. HDF5 `greenland.saved_at 2026-10-17T03:02:53Z`, `blended.saved_at 2026-10-17 03:49:14Z`.
- Fix: Move the sensitivity block to a script that runs after `slowwais` (and add it to run_pipeline.sh before `numbers`). Then rerun it and recheck tex 433.

#### IN-2 Compiled manuscript and supplement PDFs predate this run's figures
- Verdict: CONFIRMED
- Severity: medium
- Location: manuscripts/00_ddpi_slrforecast2026/v1/01_slr_forecast_intervention2026.pdf (Oct 16 17:47:00), ..._supp.pdf (Oct 16 21:52:58)
- Reported by: xc_hdf5
- Manuscript impact: every pipeline figure in both PDFs, e.g. component_ocean_posterior_corner.png and fig2v6_fan_exceedance2100.png, is the pre-run version. The .tex text is unaffected.
- Defect: the run wrote the figures between 22:28 and 23:50 local, after both PDFs were built.
- Evidence: `ls -laT` mtimes as above. The run's figures carry later mtimes (xc_hdf5 table, spot-checked).
- Fix: Rebuild both PDFs (install-regenerated-figures rule).

#### IN-3 Rate-and-state projections are zeroed at 2015.0 but labelled, printed and stored as "rel. 2000"
- Verdict: CONFIRMED
- Severity: medium (mislabelled pipeline output file; diagnostic only)
- Location: bayesian_ratestate cells 17, 19, 21, 23; slr_projections.py:546 (`baseline_idx = np.argmin(np.abs(time_proj - baseline_year))`), :368 (same in project_gmsl_ensemble); no range check in either
- Reported by: nb_ratestate_A, nb_ratestate_B, mod_slr
- Manuscript impact: none
- Defect: `time_proj` is the SSP grid only (2015.0–2099.0). The 1850+ history is used only for the S(t) spin-up, so argmin returns index 0 and H(2015)=0. The cell 17/23 tables, the cell 19 y-labels, the root attr `baseline_year=2000`, and `median_at_2100_m` (printed by results_figures cell 5) therefore all omit the 2000–2015 rise, roughly 45 mm. The summation cell 19 overlay is different: it starts its grid at BASELINE_YEAR and so is anchored near 2000.
- Evidence: HDF5 `projections/SSP2-4.5/time` 2015.0…2099.0, `gmsl_median[:3] = [0, 0.00388, 0.00793]`, file attr `baseline_year 2000.0`. Output header `Projection comparison at 2050, 2100 (mm rel. 2000)`.
- Fix: Integrate H over the spliced historical+SSP grid and anchor it at the 1995–2005 mean. Make all `project_*` functions raise when `|time[baseline_idx] − baseline_year| > 0.5 yr`.

#### IN-4 Rate-and-state SSP→Berkeley temperature offset is a single-year match chosen by an argmin tie (2014), 0.89–0.90 °C instead of ≈0.75 °C
- Verdict: CONFIRMED
- Severity: medium (biases all rate-and-state projections; diagnostic only)
- Location: bayesian_ratestate cell 3 (l.151–157); same pattern in component_summation cell 19 (`idx_ov = argmin(|temp_annual['year'] - ipcc_first_year|)`)
- Reported by: nb_ratestate_A, nb_ratestate_B, nb_summation_B, mod_comp (cross-ref), xc_constants (cross-ref)
- Manuscript impact: none
- Defect: Berkeley annual labels are yyyy.5 and AR6 starts at 2015.0, so argmin ties between 2014.5 and 2015.5 and picks 2014.5, a cold year (0.213 °C). The comment says the target is the "1995–2005 baseline", but no multi-year mean is used. Each SSP gets its own single-year offset, and the print shows only the last SSP's (SSP5-8.5) value. SSP temperatures come out about 0.14 °C too cold relative to the component pipeline's forcing.
- Evidence (my recomputation, verify_IN/rs1.py): offsets SSP1-2.6 0.887, SSP2-4.5 0.891, SSP3-7.0 0.882, SSP5-8.5 0.898 (picked 2014.5). Matching 2015.5 instead gives 0.742–0.753. A 2015–2024 overlap mean gives 0.747–0.780 (SSP2-4.5 0.755 = component_ocean T_OFFSET). Stored output: `Temperature baseline offset ...: 0.898 °C`, `Verification at 2015: ... diff = -0.0722 °C`.
- Fix: Use one shared offset helper (the same one the components use) for all SSPs, and stamp AR6 years consistently.

#### IN-5 WAIS production samplers default to deprecated rheology Mode A; trajectory/endpoint/cap tests exercise only Mode A
- Verdict: CONFIRMED
- Severity: medium (latent silent fallback in production functions; tests give false assurance)
- Location: notebooks/component_projections.py:582, 752, 859 (`rheology_mode='A'` defaults); tests/test_wais_sampling.py:220–233 (fixture, no rheology_mode), 1043–1052, 113–187, 402–460
- Reported by: tests
- Manuscript impact: none (component_wais passes `RHEOLOGY_MODE='B'` explicitly)
- Defect: Any caller that omits the argument gets the ×N(1.28, 0.07) correction that the paper does not use. The trajectory fixture and the cap-every-year test check only that path. Several tests assert properties of the deprecated correction itself.
- Evidence: `def sample_a4_wais_trajectories(n_samples, rng, years, rheology_mode='A',`. The fixture calls `sample_a4_wais_trajectories(500, rng, years, anchor_year=2020.0, ...)` with no mode. Only lines 196/207/865 use 'B'.
- Fix: Change the production defaults to 'B' (or make the argument required), pass 'B' in the fixture and cap test, and retire the Mode-A-only tests.

#### IN-6 Tests hardcode BASELINE_YEAR = 2005 with tolerances wider than the 2000→2005 rise
- Verdict: CONFIRMED
- Severity: low
- Location: tests/test_ocean.py:39, 262–267; test_apeninsula.py:34, 209–215; test_eais.py:38, 319–324; test_summation.py:44, 104–113, 328
- Reported by: tests
- Manuscript impact: none
- Defect: The "near zero at baseline" tests check 2005 rather than the pipeline's 2000 (1995–2005 mean), so a mistaken 2005 rebase would still pass. test_forecast.py imports `slr_forecast.config.BASELINE_YEAR`, so the tests are also inconsistent with each other.
- Evidence: HDF5 SSP2-4.5 medians at 2000/2005 (mm): ocean 0.399/6.566 (bound `abs(med_mm) < 15.0`), Peninsula 0.000/0.303 (`< 1.0`), EAIS 0.000/0.037 (`< 2.0`).
- Fix: Import BASELINE_YEAR from config. Assert the 1995–2005 mean ≈ 0 within ~1 mm.

#### IN-7 test_forecast tests a stale inline copy of blend_rate_space
- Verdict: CONFIRMED
- Severity: low
- Location: tests/test_forecast.py:236–324 (`_make_blend_func`); production notebooks/component_projections.py:1949–1951
- Reported by: tests
- Manuscript impact: none
- Defect: The test redefines the function. Its copy sets `forecast_samples[:, 0] = h_origin`, while production integrates a partial first step from an off-grid t_origin. The production function used by component_forecast is never tested.
- Evidence: test copy l.~260 `forecast_samples[:, 0] = h_origin`. Production `forecast_samples[:, 0] = (h_origin + 0.5 * (blended_rate_o + blended_rate[:, 0]) * (f_years[0] - t_origin))`.
- Fix: `from component_projections import blend_rate_space`. Add an off-grid t_origin case.

#### IN-8 Greenland SMB tests exercise the retired C_T = −300 literature model
- Verdict: CONFIRMED
- Severity: low
- Location: tests/test_greenland.py:31–33 (imports GREENLAND_SMB, project_smb_ensemble), 284–313, 320–453, 533–537, 1093–1131, TestGreenlandIO
- Reported by: tests
- Manuscript impact: none
- Defect: About 20 tests check the superseded SMB model, which component_greenland no longer imports (the live path is smb_emulator + smb_state). The IO roundtrip writes `smb_sensitivity=GREENLAND_SMB`, but the notebook writes an EmulatorSummary. One comment contradicts its own assert ("-200 Gt/yr/C" above `-300.0 / 362500.0`).
- Evidence: test_greenland.py:533–537 quoted above. No `GREENLAND_SMB` in nb_src/*_code.py.
- Fix: Delete them or mark them legacy. Point TestGreenlandIO at EmulatorSummary.

#### IN-9 Vacuous and re-implementing tests
- Verdict: CONFIRMED
- Severity: low
- Location: tests/test_summation.py:68–72, 382–395; test_glacier.py:237–256 (BIC with k=4/5 vs live chi2+k·log n, k=2/3), 94–149; test_summation.py:276–308 (TWS sampler "same logic as notebook"); test_greenland.py:925–1049; test_ratestate.py:548–588 (TestSensitivity: literal arithmetic only); test_wais_sampling.py:936–989
- Reported by: tests
- Manuscript impact: none
- Defect: Some tests are self-referential: `total1 += s; total2 += s; assert allclose`, `'eais' in SUMMED_COMPONENTS` (a test-local list), and checks on literals that never call production code. One has its comment and assert reversed: test_bic_prefers_simpler_model says "should be negative" but asserts `delta_bic < 2`. Others re-implement logic the pipeline has since changed.
- Evidence: quoted lines checked in the files.
- Fix: Delete these, or assert against production functions and HDF5 outputs.

#### IN-10 Untested pipeline behaviors (coverage gaps)
- Verdict: CONFIRMED (my grep of tests/*.py for `ocean_posteriors`, `save_ocean_hybrid`, `ocean_early_holdout`, `rebase_mm_total`, `1.96` finds no hits; `blended` appears only in the inline blend copy and a docstring)
- Severity: low
- Location: tests/
- Reported by: tests
- Manuscript impact: none
- Defect: Nothing tests the following: the group-deletion side effects of save_* on sibling groups (`ocean_posteriors`, `ocean_early_holdout`, `blended*`), the IPCC offset on the total only, forcing held flat to 2100, `[-1]`≠2100 indexing, the 95%→1σ conversion in the GlaMBIE/Zemp readers, the absence of `c` in the two-layer posterior, the glacier balance prior/sigma_extra, SAMPLE_WARMING_PATHS sharing, variance_shares.py, or SSP1-1.9 for Peninsula/EAIS/ocean.
- Evidence: tests.md "Untested pipeline behaviors" list.
- Fix: Add regression tests for the 2026-10-14…16 changes first: the IPCC total-only offset, the drift-term removal, the 2100 hold and the GlaMBIE ÷1.96.

#### IN-11 Stale docstrings/labels in tests and the EAIS/Peninsula `model_type` labels
- Verdict: CONFIRMED
- Severity: low
- Location: tests/test_eais.py:227–232; test_apeninsula.py:48, 73, 120–123, 188–193; notebooks/component_io.py:409 (`"model_type": "trend_only"`), 440 (`'linear_dols'`); test_forecast.py:526–542 (compares an ours-rel-2000 value with IPCC rel-1995–2014)
- Reported by: tests, nb_apen_B, mod_io
- Manuscript impact: none
- Defect: The docstrings still say "AR(1) increments" and "mid-year 1979.5–2023.5" labels. The EAIS fit is labelled 'trend_only' though it is linear in T, and the test locks that label in. The IPCC comparison test skips the AR6-baseline offset, but the margin (752 vs 556 mm) makes the outcome insensitive.
- Evidence: HDF5 `eais` attrs `model_type 'trend_only', model_order 1`.
- Fix: Relabel to 'linear_level_correlated', update the docstrings, and compare `our_median_cm_ar6_baseline`.

#### IN-12 No provenance stamps on derived groups; partial reruns leave downstream groups silently stale
- Verdict: CONFIRMED (absence of stamps); hazard PLAUSIBLE (not triggered in this full run)
- Severity: low
- Location: scripts/build_slow_wais_p90.py:129–163; notebooks/compute_evpi.py, compute_optimal_defense.py, compute_shapley_risk.py; scripts/manuscript_numbers.py (no check)
- Reported by: xc_hdf5, scripts_down, nb_forecast_A
- Manuscript impact: none in the current run (all steps ran in order). Only after a partial rerun such as `./run_pipeline.sh forecast`.
- Defect: `blended_slow90`, `wais_slow90` and `voi` carry no `saved_at` or source stamp, and no reader compares stamps. IN-1 is the in-pipeline case of the same hazard.
- Evidence: HDF5 attrs `blended_slow90.saved_at None`, `wais_slow90 None`, `voi None`; `blended.saved_at 2026-10-17 03:49:14`.
- Fix: Copy `blended.attrs['saved_at']` into each derived group as `source_saved_at`, and assert it matches in manuscript_numbers and results_figures.

#### IN-13 component_results.h5 root attributes stale (baseline_year 2005, n_samples 2000) and printed as current
- Verdict: CONFIRMED
- Severity: low
- Location: notebooks/component_io.py:65–75 (`_require_file`, writes only at creation), 878–879 (`list_components`); results_figures cell 3 output
- Reported by: xc_hdf5, mod_io, nb_summation_A, nb_summation_B
- Manuscript impact: none (print only)
- Defect: The root attrs date from file creation on 2026-03-25. WAIS has 10000 members and blended_slow90/wais_slow90 have 20000.
- Evidence: root `{'baseline_year': 2005.0, 'created': '2026-03-25T00:11:58...', 'n_samples': 2000}`.
- Fix: Refresh the root attrs on every save, or drop them.

#### IN-14 Rate-and-state "2100" is 2099; variance-decomposition rows 2125/2150 repeat 2099 and add WAIS σ to a total that already contains ice
- Verdict: CONFIRMED
- Severity: low
- Location: bayesian_ratestate cells 17, 21, 23 (HDF5 attrs `median_at_2100_m` etc.); component_summation cell 19 prints
- Reported by: nb_ratestate_A, nb_ratestate_B, nb_summation_B
- Manuscript impact: none
- Defect: The AR6 GMST ends in 2099 and is not held flat to 2100 here, unlike the 2026-10-15 convention. In cell 21, argmin clamps 2125/2150 to 2099 for σ_constr and σ_scenario, while σ_ice comes from WAIS samples that do reach 2150. Adding WAIS σ in quadrature also double-counts ice, because the rate-and-state GMSL is a total.
- Evidence: HDF5 `time[-1]=2099.0`, `gmsl_median[-1] == median_at_2100_m` (SSP1-2.6 0.77909). Cell 21 output rows `2100 220.2 744.0 430.6`, `2125 220.2 744.0 706.8`, `2150 220.2 744.0 913.9`.
- Fix: Hold the forcing flat to 2100. Restrict target_years to the grid (or assert `|t[idx]-yr|<0.5`). Drop or document σ_ice.

#### IN-15 Rate-and-state projections omit within-SSP warming-path uncertainty
- Verdict: CONFIRMED
- Severity: low
- Location: bayesian_ratestate cell 17; slr_projections.project_gmsl_state_ensemble (reads only `temperature`)
- Reported by: nb_ratestate_B
- Manuscript impact: none
- Defect: `temperature_lower/upper` are rebased in cell 3 but never sampled, and there is no warming_paths/SAMPLE_WARMING_PATHS. The printed 5–95% (e.g. SSP2-4.5 1301 [1044, 1550] mm) is parameter-only, which also understates σ_constrained in cell 21.
- Evidence: no `temperature_sigma`/`warming` in bayesian_ratestate_code.py. HDF5 SSP2-4.5 p5/p95 1.044/1.550 m.
- Fix: Sample per-member AR6 percentile paths as the components do, or label the intervals "parameter uncertainty only".

#### IN-16 Rate-and-state level likelihood is diagonal on the cumulative Frederikse record
- Verdict: CONFIRMED (code); size of the effect not measured
- Severity: low
- Location: notebooks/bayesian_models.py:4107–4109 (`_state_level_log_prob`); also `fit_bayesian_level` (instantaneous initializer); bayesian_ratestate cell 11 `compute_kinematics` without error_cov
- Reported by: nb_ratestate_A, nb_ratestate_B, mod_bayes_B, mod_bayes_A
- Manuscript impact: none
- Defect: The likelihood is iid per-year σ (plus σ_extra) on an integrated GMSL level record. The project rule says this understates intervals by about 3×. The correlated-error fitter `fit_bayesian_level_annual_correlated` already exists.
- Evidence: `sigma_total = np.sqrt(sigma_obs_fixed**2 + sigma_extra**2)`; `ll = -0.5 * np.sum((resid / sigma_total)**2 + 2.0 * np.log(sigma_total))`.
- Also: cell 11 prints `Frederikse kinematics (20-yr) at 2018.5: 6.504 ± 0.824 mm/yr`. This is a one-sided local-quadratic endpoint estimate with diagonal SE (`compute_kinematics` without `error_cov`), whereas results_figures uses `cumulative_record_cov`. It appears only as a comparison line in ratestate_satellite_era_prior.png (reported by nb_ratestate_A).
- Fix: Use a correlation-aware covariance, or state that the posterior assumes independent errors. Pass `error_cov=cumulative_record_cov(...)` in cell 11.

#### IN-17 Rate prior evaluated at a single monthly temperature (July 2018)
- Verdict: CONFIRMED
- Severity: low
- Location: bayesian_models.py:4115–4119 (`T_end = T_monthly[idx_eval]`), 2670–2671 (`fit_bayesian_level`)
- Reported by: nb_ratestate_A
- Manuscript impact: none
- Defect: The smooth 30-yr satellite rate (3.94 ± 0.65 mm/yr) is compared with the model rate at one month's GMST. That month is about 0.13 °C below the ±5-yr trend value at the same epoch used for dT/dt. With dRate/dT ≈ 8 mm/yr/°C, the error is about 1 mm/yr, larger than the prior σ.
- Evidence (verify_IN/rs3.py): T(2018.54) = 0.244 °C; ±5-yr linear fit at 2018.54 = 0.380 °C (slope 15.8 m°C/yr); 2018 annual mean 0.322 °C. Stored output `T_end=0.244 °C`.
- Fix: Take T_end (and S_end) from the same ±5-yr fit used for dT/dt.

#### IN-18 State-variable initial condition: calibration S0 = T(1900.54) vs projection spin-up from 1850 (plus a third Euler variant in results_figures)
- Verdict: CONFIRMED (code); impact small for τ≈15 yr and growing toward the tail
- Severity: low
- Location: bayesian_models.py:4091 (`solve_state_ode(T_monthly, time_monthly, tau)`, S0=T[0]); bayesian_ratestate cell 13 (fit grid starts at obs_years[0]); slr_projections.py:539–541, 557 (history from 1850); results_figures cell 13 `_solve_state_ode`
- Reported by: nb_ratestate_A, nb_ratestate_B, mod_bayes_B, mod_slr
- Manuscript impact: none
- Defect: The forward runs use an S(t) history that differs from the one d and τ were fitted with. mod_bayes_B computed Δ ≈ 0.07–0.09 °C at 2000–2018 for τ = 76 yr.
- Evidence: projection `T_full = np.concatenate([historical_temperature[hist_mask], T_proj])` with history from 1850.0; calibration `S[0] = temperature[0]`. T(1900.54) = −0.754 °C (my check).
- Fix: Spin up S from 1850 in the calibration too, and reuse `solve_state_ode` in results_figures.

#### IN-19 component_summation cell 14 hindcast draws MVN (a,b,c,d) with independent τ and anchors at 1999.5
- Verdict: CONFIRMED
- Severity: low
- Location: component_summation cell 14 (l.1489–1528) and cell 19 (l.1787–1798); slr_projections.py:517–518
- Reported by: nb_ratestate_A, nb_ratestate_B, mod_slr
- Manuscript impact: none (the "missing memory" residual diagnostic and the cell 19–21 overlays are not in the tex)
- Defect: The cell passes `calibration/coefficients` and `covariance` but not `posterior/coefficients`, so τ is bootstrapped independently. This breaks corr(τ, c) = 0.38 and allows d < 0: with the MVN mean 0.65 and sd 0.60 mm/yr/°C, P(d<0) ≈ 14%, although the exported joint posterior has 0% negative d. The hindcast grid is yyyy.5, so argmin(|t−2000|) picks 1999.5, while the component sum is zeroed at 2000.0. The residual at 2000 is +1.0 mm (cell 14 output), consistent with that offset. The ratestate notebook itself (cell 17) does pass the joint `posterior_samples`.
- Evidence: HDF5 calib coef d = 0.000651, sd 0.000598 m/yr/°C. Joint posterior negative-d fraction 0.0. corr(τ, coefs) = [−0.02, 0.15, 0.38, −0.12].
- Fix: Pass `posterior_samples=hf['posterior/coefficients'][:]` with the paired τ, and re-zero after interpolating onto `hind_years`.

#### IN-20 Rate-and-state convergence print reads nonexistent keys; diagnostics use 4 of 64 walkers
- Verdict: CONFIRMED
- Severity: low
- Location: bayesian_ratestate cell 15 (l.777–778); bayesian_models.py:583 (`n_chains_arviz = min(4, n_walkers)`), `check_convergence` (returns `rhat`, `ess_bulk`, `ess_tail`, `converged`)
- Reported by: nb_ratestate_A, nb_ratestate_B, mod_bayes_A
- Manuscript impact: none
- Defect: The print always shows N/A, and "Converged: True" is computed from the first 4 emcee walkers only. The 7-D rate-and-state per-parameter diagnostics are not visible in the stored output, because the progress bar truncates it.
- Evidence: output `R-hat max: N/A` / `ESS min:   N/A` / `Converged: True`.
- Fix: Print `max(conv['rhat'].values())` and `min(conv['ess_bulk'].values())`. Use all walkers or `get_autocorr_time`.

#### IN-21 Rate-and-state labels, units and hard-coded record-averaged endpoints
- Verdict: CONFIRMED
- Severity: low
- Location: bayesian_ratestate cell 8 markdown (τ prior "[5, 81 yr]", "data determine the posterior"), cells 13/23 prints ("94% HDI" for 3rd–97th percentiles; all coefficients in "mm/yr"; "Calibration (1901–2019)" from `:.0f` on 1900.54/2018.54), cell 23 `T_fred_start, T_fred_end = -0.6, 0.244`; results_figures cell 5 comment units
- Reported by: nb_ratestate_A, nb_ratestate_B, mod_bayes_B
- Manuscript impact: none. The record-averaged 3.5 [3.0, 4.2] mm/yr/°C is printed next to Rahmstorf 3.4 in results_figures cell 6 but is not in the tex (see the header).
- Defect: The code gives τ prior [5, 79] yr. The posterior τ 15.7 [3.9, 76.2] yr is essentially the prior. −0.6 °C matches no Berkeley value (1900 annual −0.70, 1900.54 monthly −0.754).
- Evidence: outputs `τ: LogNormal(median=20 yr, 95% CI [5, 79] yr)`, `dalpha_dT: 4.565 [3.099, 6.053] mm/yr`, `Calibration (1901–2019, n=119)`. My check: Berkeley 1900 annual −0.7006, 1900.54 −0.7540.
- Also: only 4 SSPs are exported (`ssp_keys` in cell 3 has no SSP1-1.9). Downstream readers hard-code the same 4, so nothing breaks (nb_ratestate_A, nb_ratestate_B; xc_constants treats it as by design).
- Fix: Correct the strings, label the intervals equal-tailed, and compute the endpoints from the data. Loop over PROJ_SSPS.

#### IN-22 Unseeded prior draws make figures/ratestate_priors.png change on every run
- Verdict: CONFIRMED
- Severity: low
- Location: bayesian_ratestate cell 9 (l.359, 368, 383: `np.random.lognormal/exponential`, no seed anywhere in the notebook)
- Reported by: nb_ratestate_A
- Manuscript impact: none (figure not in tex)
- Defect: The figure is non-reproducible, which shows up as a spurious git diff.
- Evidence: `git diff --stat`: `figures/ratestate_priors.png | Bin 47894 -> 47548 bytes`. `grep np.random.seed` finds nothing.
- Fix: Use `np.random.default_rng(seed)`.

#### IN-23 Other non-deterministic tracked outputs: s1_quadratic_fit.json `fitted_at`, PDF /CreationDate
- Verdict: CONFIRMED
- Severity: low
- Location: component_projections.save_s1_quadratic; matplotlib PDF savefig (scripts/plot_supp_greenland_smb_emulator.py, plot_greenland_smb_mar_vs_emulator.py)
- Reported by: xc_hdf5, nb_wais_A
- Manuscript impact: none
- Defect: Identical results produce git diffs, which hide real changes.
- Evidence: `git diff notebooks/s1_quadratic_fit.json` changes only `fitted_at` 2026-10-15T05:39 → 2026-10-17T03:12. greenland_smb_mar_vs_emulator.pdf and supp_greenland_smb_lomo.pdf have byte sizes identical to HEAD (65764, 22736), with /CreationDate D:20261014… → D:20261016….
- Fix: `metadata={'CreationDate': None}` on PDF savefig. Rewrite the JSON only when mean/cov change.

#### IN-24 component_results.h5 never compacted (1627 MB file, 364 MB of data)
- Verdict: CONFIRMED
- Severity: low
- Location: data/processed/component_results.h5
- Reported by: xc_hdf5
- Manuscript impact: none
- Defect: Each run's `del` + `create_group` leaves free space that HDF5 never reclaims, so the file grows every run.
- Evidence: sum of `get_storage_size()` = 364.3 MB; file size 1627.4 MB (verify_IN/h1.py).
- Fix: `h5repack` periodically, or write to a fresh file and swap.

#### IN-25 Stale external input: greenland_implied_smb.csv built from IMBIE 2021 by a script outside the pipeline
- Verdict: CONFIRMED
- Severity: low
- Location: data/processed/greenland_implied_smb.csv (Mar 24); writer scripts/grace_minus_discharge_smb.py:59–60 (`imbie_greenland_2021_Gt.csv`, script mtime May 2); readers component_greenland cells 5, 14 and scripts/smb_state_space_diagnostic.py (step 8)
- Reported by: xc_hdf5
- Manuscript impact: none (diagnostic print and a figure not in the tex)
- Defect: The diagnostic compares against a superseded IMBIE release, while the rest of the pipeline uses IMBIE-3.
- Evidence: mtimes as above; grep of the script line 60.
- Fix: Regenerate from IMBIE-3 inside the pipeline, or drop the overlay.

#### IN-26 Pipeline writes into its own input file on a missing key (component_ocean cell 3)
- Verdict: CONFIRMED (code); not triggered in this run
- Severity: low
- Location: component_ocean cell 3 (`except KeyError: ... to_hdf(H5_PATH, ..., mode='a')` for NOAA TSL, NOAA deep and EN4 global)
- Reported by: xc_hdf5
- Manuscript impact: none
- Defect: Once a key is cached, later updates to the raw .dat files are never re-read.
- Evidence: the three `to_hdf(..., mode='a')` lines in the extracted code.
- Fix: Read the raw files directly, or compare a hash or mtime.

#### IN-27 ipcc_distributions.h5 not produced by the pipeline; fixed at n_samples=2000
- Verdict: CONFIRMED (contents current: `rebase_mm_total 10.7`, `n_samples 2000`)
- Severity: low (the 10k run fails loudly; component_projections raises on the size mismatch)
- Location: notebooks/precompute_ipcc_distributions.py (not in run_pipeline.sh)
- Reported by: xc_hdf5 (memory note: known)
- Manuscript impact: none for N=2000
- Defect: This step must be run by hand before the final 10k run.
- Evidence: the attrs listed above.
- Fix: Add it to run_pipeline.sh, keyed on N_SAMPLES.

#### IN-28 Orphaned files in data/processed and fig1_spectrum_schematic.pdf not regenerated by the pipeline
- Verdict: CONFIRMED
- Severity: low
- Location: data/processed/{bayesian_ratestate_projections.npz, bayesian_ratestate_results.json (May 2), results_summary.json, stage1_thermosteric_results.json, ridge_distributions.json (194 MB), greenland_ct_estimate.json, bps_cv_cache_*.npz, mouginot_tw_glaciers_aggregate.csv}; figures/fig1_spectrum_schematic.pdf (Oct 13 16:22, main tex 84; source v1/fig1_design/fig1_current/gen.py)
- Reported by: xc_hdf5
- Manuscript impact: none. The schematic's hard-coded labels 1.8/2.7/3.6 °C match the current conventions.
- Defect: The orphaned files can be mistaken for current results. The old ratestate npz/json in particular sit next to the live posterior .h5. The schematic must be regenerated by hand if its labels change.
- Evidence: `ls -la` dates; no pipeline .py references `fig1_spectrum_schematic`.
- Fix: Archive the orphans and remove the unused JSON_GREENLAND_CT constant in results_figures cell 2.

### REFUTED / corrected claims
No merged finding was fully refuted. The lines below correct reviewer sub-claims only.
- "Greenland cell 26 numbers are not in the tex" (xc_hdf5, nb_greenland_A): REFUTED. Main tex 433 quotes them ("3 mm or less", "more than 2 mm"). See IN-1.
- "results_figures computes `rs_rate_pred` but does not plot it; Fig. 1b uses rate-and-state" (concern): no rate-and-state reaches Fig. 1b. The scatter is commented out, and `h_rs2` is a component-model legend handle.
- num_main_2 maps tex 375 "agrees with ... 3.4" to the rate-and-state 3.5 [3.0, 4.2]. The sentence is about the observed slope against the literal R07 line, so the rate-and-state value is not its source. No defect, but the trace in num_main_2 is wrong.

### Cross-cluster notes
- mod_readers: slr_processed_data.h5 holds a Berkeley Earth sigma from before the 95% fix (1.19× too large); two main-text figures use it.
- mod_comp / xc_constants: three different SSP→Berkeley temperature offset methods (0.726 vs 0.755 °C) across the components reach every component's 2100 value (the components' version of IN-4).
- mod_comp: `model_ensemble_draws` uses the unseeded global RNG (component-notebook figure medians).
- mod_bayes_B: `_cumulate` single-point rebase at 1999.5 via an argmin tie; spurious −log τ Jacobian in unused Greenland joint/discharge fitters.
- mod_slr: the NASA rate-prior residual-autocorrelation item affects tex 427 "0.63 [0.38, 0.86] m" (quadratic extrapolation), not rate-and-state.
- nb_summation_B: ocean ODE median rebased at 1999.5 (+0.34 mm at 2000.0); WAIS satellite-era share uses a clamped end value.
- mod_io: results_figures cell 35 Greenland R² key lookup always prints N/A; corner-plot contour levels vs supplement captions.
