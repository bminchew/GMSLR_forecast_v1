# Group 3 decisions: AR6 offset, Peninsula curvature, χ²/dof convention, GMST offset

Read-only analysis, 2026-10-17, at commit 5433eee. Code and tex were read at HEAD (`git show HEAD:`), not from the working copy, which holds another session's uncommitted edits. All numbers come from copies of `component_results.h5` and `slr_processed_data.h5` taken before the current pipeline rerun, plus raw data. Nothing in the repo was changed except this file. Scripts and full write-ups (`result.md` per item) are in this session's scratchpad under `g3/{sf2,an6,gl4,oc8}/`.

Line numbers are HEAD line numbers in the main text (`01_slr_forecast_intervention2026.tex`) or the supplement (`..._supp.tex`). The other session is editing the tex, so find sentences by their quoted content. Under the numbers-traceable-to-code rule, every new number below must print from pipeline code before it goes into the tex. Numbers marked "estimate" were not computed through the pipeline.

**Interactions.** Decisions 1 and 4 both move the totals by a few mm. They act on the same two boundary digits in l.429 (the 1.8 °C slow-WAIS median on the AR6 baseline, 445.9 mm now, and the 2.7 °C slow-WAIS 95th percentile, 814.4 mm now). The Monte Carlo error of a median at 2000 members is about 2–3 mm. Requote all affected numbers from the N_SAMPLES = 10000 final run, not from the estimates here. All four decisions can be folded into that run at no extra runtime.

**Errors already in the tex (independent of these decisions).** These were judged against the HDF5 copy, which matches the abstract and l.394 values to the quoted digit:
- l.429 95th-percentile list says "0.82" at 2.7 °C. The pipeline gives 814.4 mm (0.81 m), which matches "0.59 [0.45, 0.81] m" in the same paragraph.
- l.405 says "[0.53, 1.73]" and l.466 says "0.54 to 1.73" for the same interval (pipeline: 0.5349).
- The slow-WAIS gap to the trend extrapolation is "27 mm" at l.417 and "25 mm" at l.427 (stored JSON gives 27 mm).
- Supplement glacier χ²/dof "1.05" is 1.04496, a double rounding of the printed 1.045 (see Decision 3).

---

## 1. SF-2: AR6 baseline offset

**Question.** To compare with AR6, the forecast is shifted from its year-2000 zero to the AR6 1995–2014 baseline. The code uses four different values for that shift. Which one should be used, and where should it be computed?

**What I verified.**

Only the 12.1 mm value reaches the manuscript:
- It enters through `scripts/manuscript_numbers.py` (l.70–75, applied at l.106). That sets every AR6-baseline total in l.429 and "2 to 13%" in l.429 and l.466.
- It also enters through component_forecast cell 12. That sets the fast-WAIS percentages: l.405 "34%/33%/37%" and "33 to 37%" in l.429.

The other three values reach no manuscript number or figure:

| Value | Where | Feeds |
|---|---|---|
| 12.1 mm | results_figures cells 11, 36, 38 | prints and figures not in the tex |
| 10.7 mm | `precompute_ipcc_distributions.py` (`_GMSL_2000_TO_2005_MM`), `plot_cdf_comparison.py` l.58 | `ipcc_distributions.h5` → forecast cell 20 ridge (no saved figure) and results_figures fig2v2 (not in tex). No record reproduces 10.7 mm |
| −11.1 mm | component_summation cell 18 (used in cells 20, 28) | wrong sign; AR6 curves sit about 24 mm low; figures not in tex |
| 0 | summation cells 21, 29; forecast cells 16, 21 | raw AR6 plotted or printed under "rel. to 2000" labels |

The forecast zero is the NASA altimetry level. It is the `gmsl_smoothed` value at the sample nearest 2000 (t = 1999.9905), set in forecast cell 8 (`H_ORIGIN`). Frederikse plays no part. The stored forecast starts in 2026, so a single scalar offset is needed for the total. The per-member method at l.383 applies only to the components.

**Corrections to pipeline_bugs.md:**
- "The manuscript fan/ridge figure uses 10.7 mm" is wrong. The tex figure (fig2v6, results_figures cell 24) draws no AR6 data.
- "Leaves the other quoted values unchanged" is wrong. At 13.3 mm the 1.8 °C slow-WAIS median becomes 0.44 m.
- "815.0 → 813.8 mm" does not match the current HDF5. The current value is 814.4 → 813.2 mm, so "0.82" is already wrong today.
- The summation error is about 24 mm, not about 22 mm.

**Options** (offset = 1995–2014 mean minus value at 2000):

| Definition | Offset | 1.8 °C slow median (AR6 baseline) | 2.7 °C slow 95th | Ranges in l.429 |
|---|---|---|---|---|
| Frederikse, integer-year stamps (current) | 12.07 mm | 445.9 mm → 0.45 m | 814.4 mm → 0.81 m | 2–13%, 33–37% |
| Frederikse, mid-year stamps | 13.16 mm | 444.8 mm → 0.44 m | 813.3 mm → 0.81 m | 2–13%, 33–37% |
| **NASA smoothed, at the forecast's own 2000 sample** | **13.32 mm** | 444.6 mm → 0.44 m | 813.2 mm → 0.81 m | 2–13%, 33–37% |
| NASA raw | 12.86–13.62 mm | 0.45 / 0.44 m | 0.81 m | 2–13%, 33–37% |
| AR6 observational compilation | 15.07–15.61 mm | 0.44 m | 0.81 m | 1–13%, 32–36%; 3.6 °C median 0.76 m |
| Dangendorf | 15.28–17.19 mm | 0.44 m | 0.81 m | 1–13%, 32–36% |

All other quoted values in l.429 are unchanged under 12.1–13.3 mm. These include the 2.7 °C and 3.6 °C intervals, fast 0.74/1.72 m at 2.7 °C and 1.10/2.09 m at 4.4 °C, and l.405 34/33/37%. At 13.32 mm the 1.8 °C fast value is exactly 34.00%. Which record is used moves the offset more than the time stamps do.

**Tex sentences that change** (l.429, with 13.3 mm):
- "To compare with AR6, we shift our forecast to the AR6 1995--2014 baseline, which lowers each value by 12 mm." → "13 mm". Optionally name the altimetry record that sets the forecast's starting level.
- "our slow-WAIS forecast gives 0.45 [0.33, 0.63] m at 1.8 $^\circ$C" → "0.44 [0.33, 0.63] m".
- "(0.63, 0.82, and 1.06 m, compared with 0.78, 0.95, and 1.11 m)" → "0.81". This value is wrong under any offset.

**Recommendation: 13.3 mm.** Compute it as the NASA `gmsl_smoothed` mean over [1995, 2015) minus its value at the same sample the forecast uses as its zero. The offset converts our forecast, so it should come from the series that defines our zero. That makes it consistent by construction.
- Put it in one function, for example `ar6_baseline_offset_m()` in `component_analysis.py` next to `preindustrial_to_baseline_m`, and store it as a `blended` attribute.
- Use it everywhere the offset appears: forecast cells 12, 16 and 21; `manuscript_numbers.py`; results_figures cells 11, 36 and 38; summation cells 18, 20, 21, 28 and 29; and `precompute_ipcc_distributions.py`. Rerun the precompute with `--n-samples 10000`, because it is not in `run_pipeline.sh`.
- Forecast cell 12 and `manuscript_numbers.py` must change together. Otherwise the fast and slow l.429 numbers end up on different baselines.

---

## 2. AN-6 = MD-1: Antarctic Peninsula curvature (a_p)

**Question.** The supplement table says a_p is "set to zero because BIC favors the linear model". Is that statement true? If not, should the quadratic be adopted, or should a_p = 0 be kept with a different justification?

**What I verified.**
- **How the current BIC is built.** The notebook BIC is χ² at the posterior mean plus k ln n, with no ln|Σ| term. The quadratic fit has a ≥ 0, from an Exponential prior with no `symmetric_a`. Under that bound the quadratic collapses onto the linear fit, so ΔBIC = −ln 45 = −3.81 whatever the data say. The dof back-conversion is also off by one.
- **Independent maximum-likelihood fit.** I refit with the full Gaussian likelihood, ln|Σ| included, σ_extra free and a unbounded. It reproduces pipeline_bugs.md to two decimals:

| σ convention | Model | a (mm yr⁻¹ °C⁻²) | b | c | σ_extra (mm/yr) | −2 ln L | k | BIC |
|---|---|---|---|---|---|---|---|---|
| December (HEAD) | linear | 0 | 0.045 | 0.056 | 0.048 | −119.27 | 4 | −104.04 |
| December (HEAD) | quadratic | −0.351 | 0.071 | 0.080 | 0.040 | −127.81 | 5 | −108.78 |
| RMS of monthly (working copy) | linear | 0 | 0.046 | 0.056 | 0.048 | −119.43 | 4 | −104.20 |
| RMS of monthly (working copy) | quadratic | −0.348 | 0.071 | 0.080 | 0.040 | −127.84 | 5 | −108.81 |

- ΔBIC(lin − quad) is +4.74 with December σ and +4.61 with RMS σ, so the quadratic is preferred. The likelihood-ratio p is 0.0035. The 90% confidence interval for a is [−0.54, −0.16] mm yr⁻¹ °C⁻². All of the gain is in ln|Σ|: the χ² values are equal, but the quadratic needs a smaller σ_extra.
- **The preference rests on 2015–2023.** Ending the fit in 2019 gives ΔBIC +0.4, and ending it in 2015 gives −1.9.
  - The Peninsula rate fell from 0.117 mm/yr (2005–2014 mean) to 0.014 mm/yr (2015–2023).
  - In the IMBIE-3 partition about half of that drop is more snowfall (single-year SMB events in 2016 and 2021) and half is slower discharge.
  - The calibration range is −0.42 to +0.66 °C relative to 1995–2005, and only 2015–2023 lie above +0.30 °C.
- **Extrapolating the quadratic is implausible.**
  - The rate peaks at +0.10 °C relative to 1995–2005 and turns to mass gain above +0.59 °C, a level already passed in 2023.
  - At +3 °C (about 3.7 °C above pre-industrial) it gives about −2.9 mm/yr, roughly 1,040 Gt/yr of mass gain. The local slope there is −2.0 mm yr⁻¹ °C⁻¹, about 12 times the snowfall sensitivity the paper adopts for the whole EAIS.
  - It also implies mass gain over 1900–1950, which moves the 1900–2000 Peninsula hindcast by −14.6 mm.

**Corrections and additions to pipeline_bugs.md:**
- AN-6 says the cell-6 ΔBIC print is missing. At HEAD it is present ("ΔBIC = −3.8 (linear preferred)"), but it measures only −ln n.
- Passing `symmetric_a=True` alone would apply a N(0, 0.043) prior. The fitted |a| is about 8 times that scale, so the prior would hold a near zero. Any quadratic refit needs a wider prior on a.

**Options** (Peninsula contribution at 2100 relative to 2000):

| Scenario | (a) a_p = 0, stored pipeline | (b) quadratic, estimate on the AR6 median path |
|---|---|---|
| 1.8 °C | 9.1 [3.9, 14.3] mm | about −13 mm |
| 2.7 °C | 10.6 [3.1, 18.0] mm | about −44 mm |
| 3.6 °C | 11.9 [2.3, 21.4] mm | about −88 mm |

- **Effect of (b) on the totals (estimate).** The slow-WAIS medians 0.46/0.60/0.78 m would become about 0.44/0.55/0.68 m. The blend with the satellite quadratic damps the Peninsula shift only to −20/−52/−97 mm. I did not compute intervals for (b).
- **Option (a)** changes no main-text number.

**Tex sentences each option changes.**

*(a) Keep a_p = 0.* Only the supplement changes. Main l.249 ("We choose the linear rate--temperature model $\dot{H}_{\mathrm{p}} = b_{\mathrm{p}}T + c_{\mathrm{p}}$, which captures the combined response of SMB and ice discharge to warming.") makes no BIC claim and can stay.
- Supplement table row (l.206): "Quadratic sensitivity to warming, set to zero because BIC favors the linear model" → "Quadratic sensitivity to warming, set to zero (see text)".
- Draft supplement paragraph:

> We also fitted a quadratic rate model, $\dot{H}_{\mathrm{p}} = a_{\mathrm{p}}T^2 + b_{\mathrm{p}}T + c_{\mathrm{p}}$, with $a_{\mathrm{p}}$ unrestricted in sign. With the BIC computed from the maximized likelihood, including the fitted $\sigma_{\mathrm{p,extra}}$, the quadratic is preferred ($\Delta\mathrm{BIC} = 4.7$), with negative curvature, $a_{\mathrm{p}} = -0.35$ [$-0.54$, $-0.16$] mm yr$^{-1}$ $^\circ$C$^{-2}$. This preference comes from the warmest years of the record. The Peninsula loss rate averaged 0.12 mm yr$^{-1}$ over 2005--2014 and 0.01 mm yr$^{-1}$ over 2015--2023, and in the IMBIE-3 partition about half of this decline comes from increased snowfall and half from slower ice discharge \citep{Otosaka2026}. Fitted to 1979--2019 alone, the quadratic is no longer clearly preferred ($\Delta\mathrm{BIC} = 0.4$). Extrapolated, the quadratic would make the Peninsula gain mass whenever GMST is more than 0.6 $^\circ$C above its 1995--2005 mean, a level reached in 2023, and at 3 $^\circ$C above that mean the gain would be about 1000 Gt yr$^{-1}$. A recent snowfall anomaly and a slowdown in discharge do not support a response that grows with the square of warming, so we keep the linear model and set $a_{\mathrm{p}} = 0$.

*(b) Adopt the quadratic.*
- Supplement table row → "Quadratic sensitivity to warming, preferred by BIC from the maximized likelihood" with value "calibrated".
- Main text changes:
  - l.43 and l.466 medians "0.46, 0.60, and 0.78 m".
  - l.249 (the model statement and its b_p, c_p values).
  - l.400: "An Antarctic Peninsula contribution that rises from $+9.1$ to $+11.9$ mm partly offsets this", plus the combined Peninsula + EAIS medians.
  - l.417: "0.75--1.09 m" and all exceedance probabilities.
  - l.427: "0.60 [0.46, 0.83] m".
  - l.439 and l.468: "adds 0.32 m" becomes about 0.24 m, and the "53 million" figure changes.
  - The l.373 hindcast.
- Figures 8–11, the variance shares and the VoI would all need a rerun.

**Recommendation: (a), keep a_p = 0, and replace the BIC justification with the physical one above.** The computed BIC claim is an artifact of the a ≥ 0 bound, and the correctly computed BIC favors the quadratic. That preference rests on nine years of snowfall and discharge anomalies, does not survive dropping 2020–2023, and extrapolates to large Peninsula mass gain far outside the calibrated range.

Separately, fix MD-1 in code. The fitter should return the full −2 ln L, and the notebook BIC should use it, as `time_model_bic` already does for WAIS. The new supplement numbers (4.7, −0.35 [−0.54, −0.16], 0.12/0.01 mm/yr, about half, 0.4, 0.6 °C, 1000 Gt/yr) need a §6 appendix cell or a `manuscript_numbers.py` entry before they go in.

---

## 3. GL-4 = MD-2: χ²/dof convention in the supplement skill table

**Question.** The χ²/dof denominator (n − n_phys − 1) counts the rate parameters and the initial level H0, but not the fitted σ_extra. Should σ_extra be counted, or should the caption state the convention?

**What I verified.** Full-precision values: glacier, Peninsula and EAIS are from HDF5 attributes. WAIS and Greenland come from my read-only reproductions of the notebook fits, which match the notebook prints exactly.

| Row | n | χ² | (A) current: σ_extra not counted | (B) σ_extra counted | Table now |
|---|---|---|---|---|---|
| Glaciers | 24 | 21.944 | /21 = 1.0450 → **1.04** | /20 = 1.0972 → 1.10 | 1.05 |
| Peninsula | 45 | 40.934 | /42 = 0.9746 → 0.97 | /41 = 0.9984 → 1.00 | 0.97 |
| EAIS (no σ_extra) | 45 | 39.620 | 0.9433 → 0.94 | 0.9433 → 0.94 | 0.94 |
| WAIS | 45 | 41.557 | /42 = 0.9895 → 0.99 | /41 = 1.0136 → 1.01 | 0.99 |
| Greenland discharge | 40 increments | 54.911 | /38 = 1.445 → 1.45 | /37 = 1.484 (see below) | 1.45 |
| Thermosteric 0–700 / 0–2000 m | 71 / 21 | — | χ²/n, unaffected | χ²/n, unaffected | 1.13 / 0.37 |

- **Arithmetic.** The pipeline_bugs.md figures (1.10, 1.00, 1.01) hold at full precision.
- **Glacier rounding.** Under the current convention the glacier value rounds to 1.04, not 1.05. The 1.05 comes from rounding the printed 1.045 a second time. Neither bug entry notes this.
- **Greenland.** Keep 1.45 on 38 dof under either option. The reported value is the error scale itself, and the scale is not inside the χ² it is computed from. Counting it would give 1.48 but break the statement that it is "the factor by which the parameter covariance is inflated" (supp l.371, main l.120).
- **Thermosteric.** Keep these rows as χ²/n with the existing caption. A per-record dof is not well defined because the parameters are shared between the two records.
- **Separately, AN-3 (the RMS σ, already in the working copy) moves EAIS from 0.94 to about 0.90 under either option.** It also changes main l.255 "(reduced chi-square 0.94)".
- **Not in GL-4 or MD-2: other affected text.** Under option B, two more passages change: main l.249 ("lowers the reduced chi-square from 2.64 to 0.97") and supp l.373 ("its reduced chi-square of 0.99").
- **Not in GL-4 or MD-2: verdicts disagree.** GL-4 is marked PLAUSIBLE and MD-2 CONFIRMED for the same issue.
- **Traceability.** WAIS chi2_dof and the glacier and Peninsula σ_extra posteriors are not stored in the HDF5. The Peninsula σ_extra is not printed anywhere.
- **Outside scope.** The thermosteric 1.13 and 0.37 use residuals rebased for plotting. The demeaned residuals the likelihood uses give 1.10 and 0.34.

**Range sentence** ("All four sit between 0.94 and 1.05"):
- (A): "0.94 and 1.04" with corrected rounding.
- (B): "0.94 and 1.10".
- With AN-3, the low end becomes 0.90 under either option.

**Tex sentences each option changes.**

Current caption sentence: "For glaciers, the Peninsula, EAIS, and WAIS, $\chi^2/\mathrm{dof}$ is evaluated against the full covariance of the cumulative record."

*(A) State the convention (text only, no rerun).*
- Caption → "... evaluated against the full covariance of the cumulative record, including the fitted model-inadequacy term where there is one. The degrees of freedom are the number of annual values minus the fitted rate parameters and the initial level. The model-inadequacy term is not counted."
- Body, replace "All four sit between 0.94 and 1.05." with "The degrees of freedom are the number of annual values minus the fitted rate parameters and the initial level, and the model-inadequacy terms are not counted. All four sit between 0.94 and 1.04."
- Glacier row 1.05 → 1.04.

*(B) Count σ_extra (needs a fitter change and a rerun of glacier, Peninsula and WAIS).*
- Code: change the denominator in `fit_bayesian_level_annual_correlated` to `n - n_phys - 1 - int(fit_sigma_extra) - int(fit_ar1)`. This also removes the GL-8 back-conversion mismatch.
- Caption → "... evaluated against the full covariance of the cumulative record, including the fitted model-inadequacy term where there is one. The degrees of freedom are the number of annual values minus all fitted parameters, including the initial level and the model-inadequacy term."
- Add to the Greenland caption sentence: "Its degrees of freedom count only $\gamma$ and $r_0$, because the value reported is the error scale itself."
- Body → "The degrees of freedom are the number of annual values minus all fitted parameters, including the initial level and, where fitted, the model-inadequacy term. All four sit between 0.94 and 1.10."
- Table rows: Glaciers 1.10, Peninsula 1.00, WAIS 1.01.
- Supp l.373: "its reduced chi-square of 0.99" → "1.01".
- Main l.249: "from 2.64 to 0.97" → "from 2.64 to 1.00". The 2.64 and the glacier 4.1 are fits without σ_extra and do not change.

**Recommendation: (B).** It matches what the rest of the code already does:
- `time_model_bic` counts σ_extra in k (k = 4 for WAIS).
- The Greenland BIC counts the error scale.
- The glacier and Peninsula notebooks define dof with σ_extra included.
- The table's "Two-layer, 5 fitted" reaches 5 only by counting σ_extra.

Only the one fitter line leaves it out. σ_extra is estimated from the same residuals, so it uses a degree of freedom. The rerun fits inside the planned 10k run. If no rerun is wanted, (A) is the fallback, and it must also correct the glacier entry to 1.04.

---

## 4. OC-8 = SF-3: AR6-to-Berkeley GMST offset

**Question.** AR6 GMST paths are put on the Berkeley Earth 1995–2005 basis by subtracting an offset. The ocean uses a different offset from the other components summed with it. Which is right, and what does unifying them change?

**What I verified.** In every component the offset is subtracted from the AR6 median path first. That path is then switched in after observed Berkeley GMST at the end of 2024, and each member's warming offset (zero through 2024) is added after that.

| Site | Role | Definition | Value |
|---|---|---|---|
| component_ocean cell 9 | forcing | SSP2-4.5 minus Berkeley, 2015.5–2024.5, applied to all SSPs; mid-year stamps | 0.7554 °C |
| glacier, Greenland, EAIS, Peninsula | forcing | AR6 Historical 1995–2005 minus Berkeley 1995–2005; AR6 annual means stamped at 1 January | 0.7257 °C |
| component_forecast cell 18 | figure only (temperature panel) | as ocean | 0.7554 °C |
| component_forecast cell 23 | Rahmstorf reference, feeds l.427 "0.73 m" | per-SSP overlap | 0.7529 / 0.7554 / 0.7468 °C |
| bayesian_ratestate; summation cell 19; results_figures cell 13 | diagnostics, not in the tex | AR6 2015 minus Berkeley 2014.5 | 0.8818–0.8975 °C |

- **Neither is uniquely right.** The offset depends on the matching window:

| Matching window | Offset |
|---|---|
| 1995–2005 | 0.726 °C |
| 2015–2024 | 0.755 °C |
| 1995–2014, AR6's own reference period | 0.791 °C |
| Equal warming since 1850–1900 | 0.872 °C |

  So the offset is uncertain by about ±0.03 °C, as large as the gap between the two values in use. This amounts to about ±4 mm of thermosteric rise at 2100 that the forecast does not carry.
- **The step where observed GMST hands over to AR6 is small either way.** At 2.7 °C, measured against the Berkeley 2015–2024 trend, it is +0.017 °C with 0.726 and −0.013 °C with 0.755.
- **A second mismatch, not in pipeline_bugs.md.** The glacier-type notebooks stamp AR6 annual means at 1 January, so their forcing runs 0.5 yr ahead of the ocean's. That is about 0.01 °C warmer near term, worth at most about 1 mm at 2100.
- **EAIS never switches to observed GMST.** It runs on the AR6 Historical and SSP path throughout. The effect is under 0.1 mm, but l.282 is not literally true.

**Corrections to pipeline_bugs.md:**
- OC-8's 4.3 and 5.3 mm are confirmed.
- OC-8 cites supp l.362; at HEAD it is l.361.
- The full rate-and-state range is 0.8818–0.8975 °C, not 0.8867–0.8975.
- Forecast cell 18 feeds only a figure, not the forcing.
- Neither entry lists forecast cell 23, results_figures cell 13, EAIS not following observed GMST, the window dependence, or that the blend damps the effect on totals.

**Options** (2100 relative to 2000):

| | Option A: ocean moves to 0.726 °C | Option B: other components move to 0.755 °C |
|---|---|---|
| How computed | Exact rebuild of component_ocean cell 9; matches the stored samples to 0.0 mm. These are pipeline-equivalent numbers | Estimate: sensitivity × 0.0297 °C × 75 yr |
| Thermosteric | 147 [109, 204] → 150 [112, 208] mm; 246 [185, 342] → 250 [189, 348] mm; 374 [276, 538] → 379 [280, 545] mm | unchanged |
| Other components | unchanged | glaciers −1.7 mm; Greenland about −1 to −2 mm; Peninsula −0.1 mm; EAIS +0.4 mm |
| Totals after blending | +2.2 / +3.1 / +4.0 mm | about −2.2 to −2.9 mm |
| Abstract medians (slow 0.46/0.60/0.78 m, fast 0.60/0.75/0.94 m) | unchanged | unchanged |

**Tex sentences that change.**

*Under A* (computed member by member):
- l.394: "The thermosteric contribution is 147 [109, 204], 246 [185, 342], and 374 [276, 538] mm ... our medians are 141, 240, and 367 mm, which are 0.99, 1.22, and 1.48 times" → 150 [112, 208], 250 [189, 348], 379 [280, 545] mm; 144, 245, 373 mm; 1.02, 1.24, 1.50.
- l.405:
  - [0.53, 1.73] → [0.54, 1.74] m.
  - 4.4 °C 95th 2.10 → 2.11 m.
  - "34%" → 35% at 1.8 °C.
- l.417: P(>1 m above pre-industrial) fast 67% → 68% and slow 29% → 30%.
- l.427 and l.466: [0.46, 0.83] → [0.47, 0.83] m. If cell 23 also moves, the Rahmstorf value 0.73 → 0.74 m.
- l.429:
  - 0.59 → 0.60 in the 3.6 °C interval.
  - The 2.7 °C 95th becomes 0.82 m, which makes the current "0.82" correct.
  - "2 to 13%" → "3 to 14%" (2.53%, close to the boundary).
  - 4.4 °C fast 1.10/2.09 → 1.11/2.10 m.
- Supp l.361: "82 to 157 mm" → 83 to 159 mm.

*Under B* (estimates):
- l.429:
  - 1.8 °C slow median 0.45 → 0.44 m.
  - "2 to 13%" → "1 to 13%".
  - Possibly "33%" → 32% at 2.7 °C.
- l.398, glacier: "121 [104, 143], 147 [125, 179], and 170 [141, 213] mm" → about 119/145/168 mm.
- l.396, Greenland: about 70/92/115 → 69/90/114 mm.

*Under either option:*
- l.282: "Member $i$ of every GMST-driven component follows the same GMST path. ... Through 2024, all members follow observed GMST." This becomes true only after unification and after EAIS switches to observed GMST through 2024.
- Neither tex file says how AR6 is put on the Berkeley basis. Add one Methods sentence.

**Recommendation: one shared function returning a single, scenario-independent offset, with mid-year stamps everywhere. I lean toward 0.755 °C (option B), the SSP2-4.5 2015–2024 overlap.**
- Every component already follows observed Berkeley GMST through 2024, so the 1995–2005 anchor the paper states is carried by the observations, not by the offset.
- The offset only sets where the AR6 path picks up after 2024, and the most recent decade is the natural match for that.
- Use the function in all five components and in forecast cells 18 and 23.
- Add the +0.5 yr stamp in the glacier-type notebooks.
- Optionally switch EAIS to observed GMST through 2024.
- Add the Methods sentence. Optionally note the ±0.03 °C window sensitivity.

What needs fixing is the inconsistency, more than the choice between the two values. Option A is equally defensible if the author prefers to keep the 1995–2005 matching window stated in the paper. Both need the downstream rerun that the 10k final run already includes.

**Combined effect with Decision 1.** Under B, the 13.3 mm offset and the −2 to −3 mm shift from B push the 1.8 °C slow-WAIS AR6-baseline median the same way, to 0.44 m, and "2 to 13%" toward "1 to 13%". Under A they roughly cancel for that value. Requote from the 10k run.
