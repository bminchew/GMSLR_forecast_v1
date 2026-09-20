# Downstream updates needed after the glacier calibration change (2026-09-19)

What changed: `component_glacier.ipynb` moved from a naive level-space fit
(H0 anchored at 1850, independent-per-point likelihood, chi2/dof ~0.11-0.14)
to `fit_bayesian_level_annual_correlated` (H0 anchored at the record's own
first point = BASELINE_YEAR 2000, true correlated covariance built from
native GlaMBIE per-year rate uncertainties, calendar-year temperature
matching, `b` prior = Normal(0.61, 2.0) mm/yr/°C truncated at b≥0, centered
on Rounce et al. 2023 and weakly informative). Result: `b = 0.745 [0.297,
1.201]` mm/yr/°C, `c = 0.574 [0.452, 0.695]` mm/yr, chi2/dof = 1.038 — a
material change from both the old naive-level value (b≈0.82) and the
briefly-considered rate-space value (b≈0.62 at the old prior, ≈0.80 flat).
`component_glacier.ipynb` itself is fully up to date (executed, H5 saved,
figures regenerated, no errors) — see below for what still depends on it.

**Update 2026-09-19 (later same day): EAIS and Antarctic Peninsula are now
done too**, via two parallel background agents, both validated with the same
rigor as glaciers (flat-prior limit cross-checked against native
drop-first-point WLS; corr(b,H0) confirmed much lower than the old fit).

- **EAIS**: `b_e = -0.015 [-0.144, 0.116]` mm/yr/°C, chi2/dof=0.91, prior
  `Normal(-0.166, 0.5)` centered on an independent literature SMB
  sensitivity (Frieler 2015 / Ligtenberg 2013's C_T), not truncated (b can
  plausibly be either sign for EAIS). The old seed-dependent quadratic-fit
  instability (R² swinging 0.15-0.49) is gone. **EAIS's forecast projections
  are unaffected** — they come from a separate literature SMB model
  (`project_smb_ensemble`), not this regression; only the diagnostic fit
  panel and one methods paragraph change. One genuine (not fatal) data
  quirk found on independent cross-check: EAIS's cumulative and native-rate
  IMBIE-3 columns disagree by up to 0.22mm/yr at two likely data-product
  splice years (1992.5, 2022.5) — `H_obs` isn't exactly `cumsum(rate_obs)`
  the way it is for glaciers, which is plausibly why chi2/dof is 0.91
  rather than closer to 1.0. Documented as a minor caveat, not fixed.
- **Antarctic Peninsula**: `b_p = 0.044 [-0.020, 0.108]` mm/yr/°C, `c_p =
  0.056 [0.039, 0.072]` mm/yr, chi2/dof=0.98 **with a fitted sigma_extra**
  (unlike glaciers/EAIS — Peninsula shows real excess annual scatter beyond
  IMBIE-3's reported uncertainty: chi2/dof=2.64 without it). Old interval
  width 0.003 → new width 0.13, consistent with (though not directly
  measured against) the same H0-anchor artifact diagnosed for glaciers.
  **Revised 2026-09-20**: the fork's initial pass kept the prior's original
  non-negative (truncated at b_p≥0) family, reasoning it was "the same
  physical logic as glaciers." Checked whether that truncation was doing
  real work before accepting it: refitting with a symmetric prior of the
  same width moved the median from 0.052→0.044 and widened the 90% CI from
  [0.007, 0.110] to [-0.020, 0.108] — about 13% of the untruncated
  posterior mass is negative, so the truncation was the difference between
  "barely excludes zero" and "consistent with zero." Unlike glaciers
  (global-aggregate mountain-glacier mass loss has no plausible reverse
  mechanism), no Peninsula-specific literature result was found strong
  enough to justify a one-signed prior, so it was switched to symmetric
  (`symmetric_b=True`) and the wider, zero-spanning interval is now what's
  reported — a truncation chosen because it produced a "cleaner" result
  would have been picking the answer, not reporting it. No literature-
  grounded nonzero prior mean was found for Peninsula's combined
  SMB+dynamics sensitivity either — prior stays zero-centered.
  Minor cosmetic issue: the saved H5 metadata still shows the stale
  `model_type='linear_dols'` (harmless — confirmed nothing branches on this
  string; just a mislabeled diagnostic field, matching the pre-existing
  default in `component_io.save_apeninsula`).

`results_figures.ipynb` cell 30 now has `_glacier_fit_panel`,
`_eais_fit_panel`, and `_peninsula_fit_panel`, all following the identical
pattern; `fig_s0_component_fit_diagnostics.png` regenerated with all three
corrected panels (WAIS panel (d) still untouched, as it should be).

**Manuscript**: proposed paragraphs for both EAIS and Peninsula exist (from
the agents' reports) but are NOT applied — manuscript edits need explicit
per-change permission, and only the glacier paragraph was explicitly
authorized so far.

Everything below is now unblocked and can proceed in the order given.

---

## 1. Must rerun, in this order, once EAIS/Peninsula land

1. **`notebooks/component_summation.ipynb`** — combines all component rates
   for the budget-closure diagnostic. Currently reflects the *pre-Rounce-prior*
   exploratory glacier fit (modified in git status from earlier this session,
   before today's changes) — stale twice over now. Must rerun after EAIS/
   Peninsula are done, not before, so it isn't rerun three times.
2. **`./run_pipeline.sh forecast`** — the "blended" forecast in
   `component_results.h5` (feeds Figure 1v2, headline stats, per-component
   projection panels) has not been rerun since before this entire glacier
   investigation started (predates even the original rate-space attempt).
3. **`./run_pipeline.sh figures`** — regenerates the full figure set from the
   rerun forecast. The ~23 figure files already showing as modified in `git
   status` (from before today) are from a stale intermediate state and should
   not be committed or trusted as final until this rerun happens.
4. **`data/processed/manuscript_headline_stats.json`** — precomputed
   headline numbers consumed by `results_figures.ipynb` §1; regenerate as
   part of the `figures` pipeline stage or by hand if it's a separate step.

## 2. `results_figures.ipynb` — one cell fixed, one cell flagged

- **Cell 30** (`fig_s0_component_fit_diagnostics.png`): DONE for all three —
  glacier (a), Peninsula (b), and EAIS (c) panels all rebuilt to use their
  own fit's posterior directly (median and band from the same ensemble, no
  more mismatch); `_glacier_fit_panel`/`_peninsula_fit_panel`/
  `_eais_fit_panel` all follow the identical pattern. WAIS panel (d)
  untouched. PNG regenerated and visually confirmed.
- **Cell 60** (IMBIE-3 comparison figure): still calls the OLD glacier code
  path. Its own comment says this is deliberate ("The glacier panel uses
  GlaMBIE, not IMBIE, and is left unchanged" — it was never about glacier's
  own calibration, only about overlaying IMBIE-3 on the ice-sheet panels).
  That reasoning still holds for *why* it doesn't need IMBIE-3 data added,
  but it will now render an inconsistent (old-methodology) glacier panel
  next to updated Peninsula/EAIS panels once those land. Decide: leave as a
  frozen historical comparison (re-caption to say so explicitly), or update
  it to match cell 30. Not fixed in this pass — flagging only.

## 3. Manuscript (`01_slr_forecast_intervention2026.tex`)

- **Glaciers subsubsection**: DONE (edited today) — equation description,
  prior, posterior numbers (b=0.75, c=0.57, chi2/dof=1.04), and the
  `fig:glacier_corner` caption (now correctly describes 3 parameters and the
  2000-year anchor) are all updated and the manuscript recompiles cleanly
  (0 LaTeX errors, 0 undefined references — checked, not assumed).
- **Antarctic Peninsula / East Antarctica subsubsections**: DONE — both
  applied and manuscript recompiles cleanly (full bibtex cycle, 0 errors, 0
  undefined references, including the two new citations `Frieler2015`/
  `Ligtenberg2013` for EAIS's prior and `Scambos2004` for Peninsula's
  ice-shelf-collapse mechanism — all three verified present in
  `references.bib` with complete author lists and DOIs before use).
  Peninsula's `b_p` prior was revised a second time after applying the
  agent's initial (truncated) version — see §1's Peninsula entry above for
  why — so the manuscript now reports the symmetric-prior numbers
  (`b_p = 0.044 [-0.020, 0.108]`), not the fork's original proposal.
- **"Total GMSL budget closure"** (§Results/Validation, currently 97-106%
  closure with 0.70σ/1.70σ/0.70σ residuals): these numbers were computed
  under the OLD glacier fit and are now stale regardless of what happens to
  EAIS/Peninsula. Do not touch until §1's reruns are done — recomputing this
  from a partially-updated component set would just create a third stale
  version.
- **Headline forecast numbers** (0.65/0.82/1.02/1.21 m at 2-5°C, §Component
  projections and §Full model projections): not touched, and probably not
  very sensitive to a single small component's revision (glaciers are a
  modest contributor next to WAIS/thermosteric), but that is an assumption,
  not a verified fact — check after the forecast rerun rather than asserting
  it's unaffected.
- **`***cite \citep{Aschwanden2022}` placeholder** (line ~410): unrelated to
  this work, still present, still compiles (key resolves) — separate,
  pre-existing cleanup item, not touched here.

## 4. Repo housekeeping

- **`notebooks/component_summation_executed.ipynb`**: still shows as deleted
  in git status with an untracked copy under `notebooks/archive/` — the
  notebook self-archives its own stale snapshot on execution (noted in the
  original handoff, still unresolved, still low-priority relative to the
  above).
- **Markdown files now stale, candidates for deletion** (flagging only, not
  deleting — see the accompanying message to the user):
  - `handoff_glacier_ratespace.md` — recommended rate space as the fix;
    superseded by the level-space-with-corrected-likelihood approach that
    was actually implemented. Its diagnostic content (the H0-anchor-at-1850
    root cause, the mis-specified-likelihood finding) is preserved more
    durably in `bayesian_models.py`'s new module-level comment above
    `fit_bayesian_level_annual_correlated` and in the manuscript text itself.
  - `plan_glacier_ratespace.md` — captured the investigation that led to the
    level-space fix (the H0 re-referencing test, the T-matching convention
    finding, the reparameterization-invariance debugging). Superseded by
    what was actually built; its content is preserved in the same places as
    above plus this document.
  - Both are untracked (`git status` shows `??`), so deleting them has no
    git-history cost — they were never committed.

## 5. Verified NOT to need changes

- `component_io.py`'s `save_glacier` — accepts the new result object and
  `extra_metadata` unchanged; confirmed the saved H5 attrs show
  `model_type='linear_level_correlated'`, `fit_space='level_correlated'`,
  `chi2_dof=1.038` correctly.
- No other code branches on the literal `model_type` string value anywhere
  in the codebase (checked) — it's informational/printed only, so the new
  `'linear_level_correlated'` value is safe.
- `component_greenland.ipynb`, `component_wais.ipynb`, `component_ocean.ipynb`
  and their manuscript sections are unrelated to this change and untouched.
