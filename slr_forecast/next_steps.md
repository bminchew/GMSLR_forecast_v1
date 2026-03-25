# Next Steps: Component Decomposition

Updated 2026-03-23 after major architectural revisions.

## Completed (this session)

### Infrastructure
- Budget rate+accel constraint wired into `_greenland_joint_log_prob` and `fit_bayesian_greenland_joint` (not used as constraint — diagnostic only, consistent with Pass 1)
- `prepare_greenland_components` merges Mankoff + Mouginot (superseded by Mouginot-only approach)
- `prepare_mouginot_components` for Mouginot-only data preparation
- `fit_bayesian_greenland_discharge` — discharge-only model (6 params, no SMB fit)
- `BayesianGreenlandDischargeResult` — result dataclass with SMB pass-through
- `smb_projections.py` — SMB projection module using literature sensitivities (C_T, C_T2)
  - `SMBSensitivity` dataclass with published values for GrIS, EAIS, Peninsula
  - `project_smb_ensemble()` — MC projection under SSP scenarios
  - `project_smb_at_warming_levels()` — comparison table at specific warming levels
- `read_noaa_thermosteric_yearly` reader for NOAA basin time series
- `read_berkeley_earth_gridded` extended with `ocean_only=True` for SST extraction
- `fit_bayesian_thermosteric` extended with joint ocean T calibration (κ, δ, σ_ocean)
- NOAA thermosteric (0-700m, 0-2000m) and EN4 global (0-700m) cached to HDF5
- Per-glacier SMB-discharge feedback analysis (`diagnostics/smb_discharge_feedback.py`)
- TW glacier aggregate data extracted (`data/processed/mouginot_tw_glaciers_aggregate.csv`)

### Key decisions
- Thermosteric: use NOAA + EN4 global (not Frederikse pre-1955)
- Single-layer ODE sufficient (two-layer deep ocean <5% of signal)
- Greenland: calibrate on Mouginot only, Mankoff withheld for validation
- Greenland SMB: use RACMO observations directly, NOT polynomial fit to temperature
- Same approach for EAIS and Peninsula: use IMBIE observations, not polynomial fits
- SMB-discharge feedback: not supported by per-glacier data (r=+0.12, p=0.08, wrong sign)

### Manuscript updates
- Greenland discharge model physical justification (SMB-D independence, per-glacier analysis)
- Baseline corrected to 1995-2005 (was incorrectly stated as 1850-1900)
- Calibration data changed to Mouginot with Mankoff validation

## Remaining work

### 1. Greenland discharge-only fit — DONE
Implemented `fit_bayesian_greenland_discharge` (6 params: γ_atm, γ_ocean, log_τ, D₀, log_σ_dyn, H₀_dyn). Cell 47 updated to use discharge-only fit with Mouginot D. SMB passes through as fixed RACMO observations.

### 2. SMB projection module — DONE but C_T values need revision
Implemented `smb_projections.py` with:
- `SMBSensitivity` dataclass with published values for GrIS, EAIS, Peninsula
- `project_smb_ensemble()` — MC projection propagating C_T uncertainty
- `project_smb_at_warming_levels()` — comparison at specific warming levels

**C_T revision needed (Mar 2026 literature review):**
- Old GrIS value: -125 ± 25 Gt/yr/°C GMST (from Fettweis 2013, which is per °C *local* T, not GMST)
- Updated: -200 ± 80 Gt/yr/°C GMST at current warming (Hanna 2021, Fettweis 2013, corrected for AA~2.0)
- Should add quadratic term: ~-50 Gt/yr/°C² GMST for ablation zone expansion (Noël 2021, Sellevold 2020)
- Physical constraint: SMB → 0 at ~2.7°C GMST (Noël et al. 2021)
- Structural uncertainty: factor ~1.8 between RACMO and MAR (Glaude et al. 2024)
- Data-driven C_T from GRACE - D (see new item 7 below) may supersede literature values
- See memory: topics/project_greenland_smb_sensitivity.md

### 3. Thermosteric fit with NOAA + EN4
Run cell 28 with the updated data (NOAA thermosteric + EN4 global subsurface T). The code is written but hasn't been run with the new data sources yet. Expect τ_u better constrained by global (not regional) subsurface T.

### 4. Surface-to-ocean transfer function for Greenland discharge projections
The thermosteric S_u state variable provides a global ocean state. For Greenland discharge projections, need to connect SSP temperatures to regional ocean T. Options:
- Use κ·S_u + δ from the joint thermosteric fit (global → regional scaling)
- Fit a regional transfer function from GMST → EN4 Greenland peripheral T
- The sinusoidal pattern seen with regional data suggests a lag; global S_u may work better

### 5. Projection cell rewrite
Replace the deprecated taper-based projection cell with current models:
- Thermosteric: use `result_phys` (single-layer + ocean T joint fit)
- Glaciers: use `result_glacier` (Stage 2a, unchanged)
- Greenland SMB: literature sensitivity C_T with MC uncertainty
- Greenland D: calibrated ODE with projected ocean T
- Antarctica: EAIS/Peninsula literature sensitivities + WAIS A4 framework
- TWS: IPCC median (as before)

**Depends on:** Steps 1-4 above

### 7. Data-driven Greenland C_T from GRACE − Discharge
Compute implied SMB = GRACE total mass balance − satellite-derived discharge (Mouginot/Mankoff).
This is an RCM-independent pan-Greenland SMB estimate (2002–2024). Regress against Greenland surface T
to obtain a purely observational C_T. No one has published this.
- GRACE total: from IMBIE Greenland (dominated by GRACE post-2002)
- Discharge: Mouginot (1972–2018) + Mankoff (2019–2023), satellite velocities × ice thickness
- Greenland T: Berkeley Earth gridded
- Script: `scripts/grace_minus_discharge_smb.py`
- Compare against RCM values (MAR ~100–150, RACMO similar, per °C local) and our literature C_T

### 8. IPCC hindcast notebook
`notebooks/ipcc_hindcast.ipynb` — extract IPCC-implied sensitivities, run backward through observed T,
compare against observations. Infrastructure built; needs running and interpretation.
- ISMIP6 Greenland data downloaded (Zenodo DOI:10.5281/zenodo.3939037, 21 models)
- SMB/D decomposition verified (smb + dyn = total, closes budget exactly)

### 9. Reviewer 2 items (from self-review)
- Minor 4: Check γ_atm / D₀ degeneracy in joint posterior
- Minor 8: Verify R² comparison (Greenland T vs GMST) uses same model structure
- Major 1: Quantify melt-elevation feedback bias for projections (~3-5% by 2100)
- Major 2: Clarify D_eff physical interpretation in manuscript
- Major 3: Improve per-glacier analysis with BedMachine thickness normalization and regional partial correlations
