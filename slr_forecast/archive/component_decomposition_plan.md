# Component Decomposition Plan for Hierarchical SLR Framework

## 1. Motivation

The aggregate GMSL–GMST model (Bayesian level-space, including rate-and-state extension) has reached its natural ceiling. The quadratic rate coefficient `a` aliases physically distinct processes — thermosteric expansion, glacier loss, Greenland surface melt, Greenland discharge, and Antarctic dynamics — that respond to warming at different rates, saturate at different temperatures, and have fundamentally different extrapolation behaviour. A single polynomial cannot resolve these contributions, and the result is unreasonably large projections at high T because the model extrapolates the *total* quadratic acceleration rather than the sum of component-specific curves.

Moving to component-wise estimation allows:
1. **Component-appropriate functional forms**: Linear for thermosteric, self-limiting for glaciers, split SMB/discharge for Greenland, budget-constrained residual for Antarctica.
2. **Budget constraint as regulariser**: Σ components = total GMSL provides cross-component information flow.
3. **Physically defensible extrapolation**: Each component is extrapolated according to its own physics rather than a single polynomial.
4. **Transparent uncertainty attribution**: We can identify which component drives projection uncertainty at each time horizon.

---

## 2. Component Structure

Following the hierarchical framework (see `hierarchical_slr_framework.tex`, §3):

```
GMSL_total = GMSL_thermo + GMSL_glaciers + GMSL_GrIS + GMSL_AIS + GMSL_TWS
```

### 2.1 Thermosteric Expansion (Stage 1 — first priority)

**Physics**: Ocean absorbs heat → water expands. Response is approximately linear in GMST because the thermal expansion coefficient varies slowly over the relevant temperature range. Best-constrained component. Expected: `a_thermo ≈ 0`, response dominated by `b_thermo · T + c_thermo`.

**Observational record**:
- Argo float array: 2005–present, near-global below 2000 m (best data)
- NOAA thermosteric reconstructions: 0–700 m (1955–present) and 0–2000 m (2005–present)
- Deep ocean (below 2000 m): poorly observed, contributes ~10% of total thermosteric

**Model**: `rate_thermo(T) ≈ b_thermo · T + c_thermo`
PC prior on `a_thermo` toward zero (expect linear response).

### 2.2 Glaciers and Small Ice Caps (Stage 2a)

**Physics**: Mountain glaciers and ice caps respond to temperature through surface energy balance. Response may be nonlinear because: (a) small glaciers disappear entirely, reducing total glacier area (self-limiting); (b) elevation–temperature feedback amplifies warming at glacier surfaces. Total glacier reservoir is ~0.32 m SLE — glaciers *cannot* contribute more than this.

**Observational record**:
- GlaMBIE consensus: 2000–2023, global and 19 RGI regions, annual, Gt and m.w.e.
- GRACE/GRACE-FO: 2002–present (must separate glacier signal from ice sheets)
- Geodetic mass balance: Global coverage, multi-decadal

**Model**: `rate_glaciers(T) = a_glaciers · T² + b_glaciers · T + c_glaciers` with physical constraint that cumulative loss cannot exceed remaining glacier volume (~320 mm SLE).

**Key distinction from ice sheets**: Greenland peripheral glaciers (GlaMBIE region 5) must either be included here or in GrIS, not both. IMBIE GrIS includes peripheral glaciers; GlaMBIE Region 19 (Antarctic and Subantarctic) overlaps with Antarctic Peninsula glaciers. We need to define clean boundaries.

### 2.3 Greenland Ice Sheet (Stage 2b)

**Physics**: Greenland loses mass through two distinct channels with very different temperature sensitivities:
- **Surface mass balance (SMB, ~60% of total loss)**: Strongly and rapidly responsive to GMST. As temperature rises, the ablation zone expands inland and to higher elevations, driving nonlinear acceleration of surface melt. The melt–elevation feedback (lowering surface → warmer temperatures → more melt) amplifies this response.
- **Ice discharge (~40% of total loss)**: Driven by ocean–ice interaction at marine-terminating glacier fronts. Warm ocean water (often subsurface Atlantic Water / CDW-origin) melts the underside of floating ice tongues and glacier faces, reducing buttressing and accelerating flow. Response to GMST is indirect and delayed, mediated through ocean circulation changes. Some discharge increase is a fast response to GMST (retreat of marine-terminating outlets), but the coupling is not as direct as SMB.

**Constraints from Aschwanden et al. (2019, Science Advances)**: Comprehensive ensemble of ice-sheet simulations predicts GrIS contributing 5–33 cm of SLR by 2200 under RCP 8.5, with discharge constituting 8–45% of total mass loss. Key finding: SMB dominates in all scenarios, but discharge fraction varies widely depending on ocean forcing assumptions.

**Observational record**:
- IMBIE 2021: 1992–2020, monthly, total GrIS mass balance (Gt/yr) — does not separate SMB from discharge
- GRACE/GRACE-FO: 2002–present, spatial patterns allow partial SMB/discharge separation via regional decomposition
- SMB models (RACMO, MAR): gridded surface mass balance, driven by reanalysis, 1958–present
- Ice discharge: Measured from satellite velocity × ice thickness at flux gates, annual updates

**Model**: Two-component structure:
```
rate_GrIS(T) = rate_SMB(T) + rate_discharge(t, T_ocean?)
rate_SMB(T) = b_SMB · T + c_SMB   (possibly with a_SMB · T² for melt–elevation feedback)
rate_discharge(t) ≈ trend + adjustment   (weakly temperature-dependent, time-series-based)
```

### 2.4 Antarctic Ice Sheet (Stage 4 — final, residual-based)

**Physics**: Antarctica is three very different ice sheets that must be treated separately:

**2.4.1 Antarctic Peninsula**
- Effectively a collection of mountain glaciers on a steep spine, responding to warming air temperature much like other glacier systems.
- Rapid warming in the late 20th century drove collapse of ice shelves (Larsen A in 1995, Larsen B in 2002) and subsequent glacier acceleration.
- Current contribution: small relative to WAIS.
- Model: similar to glaciers, `rate_Peninsula(T) ≈ b_Pen · T + c_Pen`.

**2.4.2 East Antarctic Ice Sheet (EAIS)**
- Huge, largely stable ice sheet whose mass *gains* through increased snowfall have historically offset some of the mass loss from other regions.
- Contains marine sectors (e.g., Totten, Aurora, Wilkes) that could potentially retreat, but none are showing concerning signs of imminent instability over the next 100 years.
- Recent IMBIE data shows EAIS near mass balance or slightly gaining.
- Model: `rate_EAIS ≈ c_EAIS` (approximately constant, possibly slightly negative = mass gain under warming due to increased precipitation).

**2.4.3 West Antarctic Ice Sheet (WAIS) — largest source of uncertainty**
- Marine ice sheet with ~4 m of SLR potential.
- The Amundsen Sea Sector (Thwaites Glacier, Pine Island Glacier) is the primary concern.
- WAIS dynamics are controlled by warm Circumpolar Deep Water (CDW) delivery to the continental shelf, modulated by ENSO and Southern Annular Mode.
- Key processes: Marine Ice Sheet Instability (MISI) — once grounding lines retreat onto retrograde bed slopes, retreat may become self-sustaining and potentially irreversible; Marine Ice Cliff Instability (MICI) — hypothesised but debated mechanism for rapid collapse of tall ice cliffs.
- Observational record is short (~30 years), the system may not yet have crossed irreversibility thresholds, and the future trajectory depends on poorly constrained ocean circulation changes.
- **This is where the A4 deep-uncertainty framework (Level 3) applies.**

**Strategy**: AIS is the *residual* component in the budget constraint:
```
rate_AIS = rate_total − rate_thermo − rate_glaciers − rate_GrIS − rate_TWS
```
This gives an observationally-derived bound on the historical AIS contribution without requiring an ice-sheet model. The A4 scenario mixture then extends the projection envelope beyond what observations can constrain.

### 2.5 Terrestrial Water Storage (Stage 3)

**Physics**: Both secular (dam impoundment, groundwater depletion) and climate-driven (ENSO-modulated precipitation) components. Secular TWS contributes primarily to the background rate `c`. Climate-driven TWS introduces interannual variability that is correlated with WAIS forcing through the ENSO–CDW teleconnection.

**Model**: `rate_TWS(t) = c_TWS,secular + rate_TWS,climate(ENSO(t))`

---

## 3. Data Inventory

### 3.1 Data Available Locally (high confidence)

| Dataset | Location | Coverage | Format | Reader exists? |
|---------|----------|----------|--------|---------------|
| **Thermosteric** | | | | |
| NOAA thermosteric 0–700 m | `data/raw/steric/noaa_thermosteric_SL_0-700-3month-*.zip` | 1955–present, quarterly | NetCDF (zipped) | `read_noaa_thermosteric()` |
| NOAA thermosteric 0–2000 m | `data/raw/steric/noaa_thermosteric_SL_0-2000-3month-*.zip` | 2005–present, quarterly | NetCDF (zipped) | `read_noaa_thermosteric()` |
| Frederikse thermodynamic component | `derived/df_frederikse_thermo` in HDF5 | 1900–2018, annual | HDF5 | In HDF5 store |
| **Glaciers** | | | | |
| GlaMBIE global consensus | `data/raw/glaciers/0_global_glambie_consensus.csv` | 2000–2023, annual | CSV | **No** |
| GlaMBIE 19 RGI regions | `data/raw/glaciers/glambie/.../glambie_results_20240716/calendar_years/` | 2000–2023, annual | CSV (19 files) | **No** |
| IPCC AR6 glacier projections | `data/raw/ipcc_ar6/slr/.../glaciers-*.nc` | 2020–2100/2150 | NetCDF (20 files) | **No** |
| **Greenland** | | | | |
| IMBIE Greenland (total) | `data/raw/ice_sheets/Gt/imbie_greenland_2021_Gt.csv` | 1992–2020, monthly | CSV | **No** |
| IPCC AR6 GIS projections | `data/raw/ipcc_ar6/slr/.../icesheets-*_GIS_globalsl.nc` | 2020–2100/2150 | NetCDF (~25 files) | **No** |
| **Antarctica** | | | | |
| IMBIE WAIS | `data/raw/ice_sheets/Gt/imbie_west_antarctica_2021_Gt.csv` + HDF5 | 1992–2020, monthly | CSV/HDF5 | `read_imbie_west_antarctica()` |
| IMBIE EAIS | `data/raw/ice_sheets/Gt/imbie_east_antarctica_2021_Gt.csv` | 1992–2020, monthly | CSV | **No** |
| IMBIE Peninsula | `data/raw/ice_sheets/Gt/imbie_antarctic_peninsula_2021_Gt.csv` | 1992–2020, monthly | CSV | **No** |
| IMBIE Antarctica (total) | `data/raw/ice_sheets/Gt/imbie_antarctica_2021_Gt.csv` | 1992–2020, monthly | CSV | **No** |
| IMBIE all ice sheets | `data/raw/ice_sheets/Gt/imbie_all_2021_Gt.csv` | 1992–2020, monthly | CSV | **No** |
| IPCC AR6 AIS projections | `data/raw/ipcc_ar6/slr/.../icesheets-*_{AIS,WAIS,EAIS}_globalsl.nc` | 2020–2100/2150 | NetCDF (~60 files) | **No** |
| **TWS** | | | | |
| GRACE mascon (global) | `data/raw/tws/GRCTellus.JPL.*.GLO.RL06.3M.MSCNv04CRI.nc` | 2002–2025, monthly | NetCDF | **No** |
| GRACE mascon (land only) | `data/raw/tws/GRCTellus_JPL_LND_RL06.nc` | 2002–present | NetCDF | **No** |
| IPCC AR6 landwater projections | `data/raw/ipcc_ar6/slr/.../landwaterstorage-*.nc` | 2020–2100/2150 | NetCDF (5 files) | **No** |
| **Ocean dynamics** | | | | |
| IPCC AR6 ocean dynamics projections | `data/raw/ipcc_ar6/slr/.../oceandynamics-tlm-*.nc` | 2020–2100/2150 | NetCDF (~10 files) | **No** |
| **Temperature** | | | | |
| Berkeley Earth, HadCRUT5, GISTEMP, NOAA | HDF5 store | 1850–present | HDF5 | ✓ all 4 readers |
| IPCC AR6 temperature projections | HDF5 store | 2020–2100+ | HDF5 | `read_ipcc_ar6_projected_temperature()` |
| **ENSO** | | | | |
| NOAA ONI index | `data/raw/enso/noaa_oni_index.csv` | Historical | CSV | **No** |
| NOAA MEI index | `data/raw/enso/noaa_mei_index.txt` | Historical | Text | **No** |
| **GMSL** | | | | |
| Frederikse, Dangendorf, NASA, Horwath, IPCC | HDF5 store | Various (1900–2018+) | HDF5 | ✓ all readers |
| **Total GMSL projections** | HDF5 store | 5 SSPs + 5 tlim | HDF5 | ✓ readers |

### 3.2 Data That Need New Readers (available locally, no reader function)

These files exist on disk but have no reader in `slr_data_readers.py`:

| Dataset | Priority | Effort | Notes |
|---------|----------|--------|-------|
| IMBIE Greenland | **High** (Stage 2b) | Low | Same CSV format as WAIS; adapt `read_imbie_west_antarctica()` |
| IMBIE EAIS | **High** (Stage 4) | Low | Same format |
| IMBIE Peninsula | Medium (Stage 4) | Low | Same format |
| IMBIE Antarctica total | Medium (validation) | Low | Same format |
| IMBIE all ice sheets | Low (validation) | Low | Same format |
| GlaMBIE global consensus | **High** (Stage 2a) | Low | Simple CSV with Gt and m.w.e. |
| GlaMBIE regional (19 files) | Medium (diagnostics) | Medium | Same format, need loop over regions |
| NOAA ONI / MEI | Medium (Stage 3) | Low | Simple CSV/text |
| GRACE mascon (global) | Medium (Stage 3) | Medium | NetCDF, need spatial integration for SLE |
| GRACE mascon (land only) | Medium (Stage 3) | Medium | NetCDF |
| IPCC FACTS component NetCDFs | **High** (all stages) | Medium | Generic reader for ~224 files across all component types |

### 3.3 Data to Download

| Dataset | Source | Priority | Why needed |
|---------|--------|----------|------------|
| **Greenland SMB (RACMO/MAR)** | RACMO output via PANGAEA or direct; MAR via Fettweis group | **High** | Separate SMB from discharge in GrIS budget. Needed for Stage 2b two-component model |
| **Greenland ice discharge** | Mouginot et al. (2019) or King et al. (2020), available via NSIDC/PANGAEA | **High** | Discharge time series at flux gates. Needed to isolate the ~40% non-SMB contribution |
| **Argo-based steric (IPCC-recommended)** | WCRP Global Sea Level Budget Group or individual products (e.g., IAP, Ishii, EN4) | Medium | Cross-validate NOAA thermosteric; extend depth coverage. NOAA 0–2000 m may suffice for Stage 1 |
| **Dam impoundment time series** | Chao et al. (2008) or ICOLD database | Low | Secular TWS component. Can use IPCC AR6 assessed values as a prior instead |
| **Groundwater depletion estimates** | Wada et al. (2012, 2016) or Döll et al. | Low | Secular TWS component. Same as above — prior from literature may suffice |
| **Updated IMBIE (post-2020)** | IMBIE website (imbie.org) | Low | Extend ice sheet records beyond 2020. Current 1992–2020 is adequate for initial implementation |

**Note on Aschwanden et al. (2019):** The paper provides ensemble projections for GrIS under various RCP scenarios. These serve as *prior information* for the Greenland model (constraining the plausible range of GrIS contribution and the SMB/discharge partition) rather than as direct observational input. The key result — GrIS contributes 5–33 cm by 2200 under RCP 8.5, with discharge fraction 8–45% — will be used to set physically informed prior bounds.

---

## 4. Implementation Plan

### Stage 1: Thermosteric Decomposition

**Goal**: Fit `rate_thermo(T) = a_thermo · T² + b_thermo · T + c_thermo` using steric sea level observations, establishing the best-constrained component of the budget.

**Steps**:
1. **Data preparation**
   - Read NOAA thermosteric 0–2000 m (2005–present) and 0–700 m (1955–present) — reader already exists
   - Read Frederikse thermodynamic component from HDF5 (already in store)
   - Cross-validate: NOAA 0–2000 m vs Frederikse thermodynamic over overlap period
   - Harmonize to common time axis and units (meters, rebased to 2005)
   - Note: Frederikse "thermodynamic" includes steric + barystatic ocean dynamics; NOAA is purely thermosteric. Need to understand the difference and choose appropriately.

2. **Bayesian calibration**
   - Adapt `fit_bayesian_level()` for component-specific calibration
   - PC (Exponential) prior on `a_thermo` → expect near zero (linear response)
   - Weakly informative priors on `b_thermo`, `c_thermo`
   - Likelihood: thermosteric observations with their reported uncertainties

3. **Residual computation**
   - `rate_cryospheric+TWS(t) = rate_total(t) − rate_thermo(t)`
   - This residual constrains the *sum* of all non-thermosteric contributions
   - Propagate uncertainty from both the total and thermosteric fits

4. **Deliverables**
   - Posterior on (a_thermo, b_thermo, c_thermo) with uncertainty
   - Thermosteric projection envelope under all SSPs
   - Cryospheric residual time series with uncertainty bands
   - Comparison with IPCC AR6 thermosteric projections (ocean dynamics component)

### Stage 2a: Glaciers

**Goal**: Fit glacier contribution using GlaMBIE observations, accounting for the self-limiting nature of glacier mass loss.

**Steps**:
1. **Data preparation**
   - Write reader for GlaMBIE global consensus CSV
   - Convert Gt → mm SLE (1 Gt = 1/362.5 mm SLE)
   - Write reader for IPCC FACTS glacier projection NetCDFs
   - Handle boundary with GrIS peripheral glaciers and Antarctic/Subantarctic glaciers:
     * GlaMBIE Region 5 = Greenland periphery → include in glaciers OR GrIS (not both)
     * GlaMBIE Region 19 = Antarctic and Subantarctic → include in glaciers OR AIS (not both)
     * Decision: follow IPCC convention where glaciers exclude ice sheet peripheral glaciers. Check IMBIE vs GlaMBIE boundary definitions.

2. **Bayesian calibration**
   - `rate_glaciers(T) = a_glaciers · T² + b_glaciers · T + c_glaciers`
   - Physical constraint: cumulative glacier SLE loss ≤ ~320 mm (total glacier reservoir)
   - Prior: allow `a_glaciers > 0` but bounded by reservoir depletion physics
   - Short observational record (2000–2023) means wide posteriors; budget constraint from Stage 1 helps

3. **Deliverables**
   - Posterior on (a_glaciers, b_glaciers, c_glaciers)
   - Glacier projection with self-limiting saturation
   - Updated residual: `rate_ice_sheets+TWS = rate_total − rate_thermo − rate_glaciers`

### Stage 2b: Greenland Ice Sheet

**Goal**: Fit GrIS contribution with explicit SMB/discharge separation.

**Steps**:
1. **Data preparation**
   - Write reader for IMBIE Greenland (adapt from `read_imbie_west_antarctica()`)
   - **Download** Greenland SMB model output (RACMO or MAR) — needed to separate SMB from discharge
   - **Download** ice discharge time series — needed for the non-SMB component
   - Write reader for IPCC FACTS GIS projection NetCDFs
   - Cross-validate: IMBIE total ≈ SMB + discharge records

2. **Two-component model**
   ```
   rate_GrIS(T) = rate_SMB(T) + rate_discharge(t)
   rate_SMB(T) = b_SMB · T + c_SMB     [possibly + a_SMB · T² for melt–elevation feedback]
   rate_discharge(t) = c_discharge + trend_discharge · (t − t₀)
   ```
   - SMB component: strongly temperature-dependent, dominates (~60% of total loss)
   - Discharge component: weakly temperature-dependent, driven by ocean forcing and glacier dynamics
   - Aschwanden et al. (2019) constraints inform priors: discharge = 8–45% of total under RCP 8.5

3. **Bayesian calibration**
   - Joint fit of SMB parameters (b_SMB, c_SMB, possibly a_SMB) and discharge parameters
   - IMBIE total provides the combined constraint
   - If SMB/discharge separation data are available: use as additional likelihood terms
   - If not: use Aschwanden et al. priors on discharge fraction

4. **Deliverables**
   - Posteriors on SMB and discharge parameters
   - GrIS projection with explicit SMB/discharge decomposition
   - Updated residual: `rate_AIS+TWS = rate_total − rate_thermo − rate_glaciers − rate_GrIS`

### Stage 3: Terrestrial Water Storage

**Goal**: Estimate the secular and climate-driven TWS contributions.

**Steps**:
1. **Data preparation**
   - Write reader for GRACE mascon land-only data
   - Write reader for NOAA ONI/MEI indices
   - Write reader for IPCC FACTS landwater storage projections
   - Separate GRACE TWS from ice-sheet signals (GRACE mascon already separates land/ocean, but Antarctic and Greenland ice sheets must be excluded from the TWS integral)

2. **Model**
   ```
   rate_TWS(t) = c_TWS_secular + f(ENSO(t))
   ```
   - Secular component: literature-informed prior from dam impoundment + groundwater depletion estimates
   - Climate-driven component: regression on ENSO index
   - Budget provides additional constraint

3. **Deliverables**
   - TWS secular rate estimate
   - ENSO-driven variability characterization
   - Final residual: `rate_AIS = rate_total − Σ(other components)`

### Stage 4: Antarctic Ice Sheet (Budget Residual + Deep Uncertainty)

**Goal**: Constrain AIS contribution observationally via the budget residual; extend projection uncertainty via the A4 framework.

**Steps**:
1. **Historical AIS from budget**
   - `rate_AIS(t) = rate_total(t) − rate_thermo(t) − rate_glaciers(t) − rate_GrIS(t) − rate_TWS(t)`
   - Cross-validate against IMBIE Antarctica (direct observations)
   - Decompose into Peninsula + EAIS + WAIS using IMBIE sub-components

2. **Antarctic sub-components**
   - Write readers for IMBIE EAIS, Peninsula
   - WAIS reader already exists
   - Validate: IMBIE total Antarctica ≈ EAIS + WAIS + Peninsula
   - Validate: budget residual ≈ IMBIE total Antarctica (to within uncertainties)

3. **Projection**
   - Peninsula: treat as glaciers, `rate_Pen(T) ≈ b_Pen · T + c_Pen`
   - EAIS: approximately constant, `rate_EAIS ≈ c_EAIS` (possibly slightly negative = mass gain)
   - WAIS: this is where the A4 scenario mixture (already implemented) provides the deep-uncertainty extension:
     * S1: Status quo (P=0.10)
     * S2: MISI (P=0.55)
     * S3: MISI + amplifiers (P=0.25)
     * S4: MISI + MICI (P=0.10)
   - The budget-constrained WAIS rate provides a reality check on A4 weights

4. **Deliverables**
   - Observationally-derived AIS contribution (from budget)
   - Sub-component decomposition (Peninsula, EAIS, WAIS)
   - Full projection with A4 deep-uncertainty module for WAIS
   - Comparison: budget-derived AIS vs IMBIE direct observations vs IPCC AR6

---

## 5. Code Architecture

### 5.1 New Reader Functions (in `slr_data_readers.py`)

```python
# Ice sheets — adapt from read_imbie_west_antarctica()
read_imbie_greenland()           # IMBIE GrIS total mass balance
read_imbie_east_antarctica()     # IMBIE EAIS mass balance
read_imbie_antarctic_peninsula() # IMBIE Peninsula mass balance
read_imbie_antarctica()          # IMBIE total Antarctica (validation)

# Glaciers
read_glambie_global()            # GlaMBIE global consensus
read_glambie_regional(region)    # GlaMBIE by RGI region number (0–19)

# IPCC FACTS components (generic)
read_ipcc_ar6_component(component_type, sub_component=None, scenario=None)
    # component_type: 'glaciers', 'icesheets', 'landwaterstorage', 'oceandynamics'
    # sub_component: 'GIS', 'AIS', 'WAIS', 'EAIS', 'PEN', etc.
    # scenario: 'ssp126', 'ssp245', etc.

# TWS
read_grace_mascon_land()         # GRACE JPL mascon, land-only TWS

# ENSO
read_noaa_oni()                  # Oceanic Niño Index
read_noaa_mei()                  # Multivariate ENSO Index
```

### 5.2 New Analysis Module: `component_models.py`

```python
# Component-specific Bayesian fits
fit_bayesian_thermosteric(...)     # Linear-in-T, PC prior on a_thermo
fit_bayesian_glaciers(...)         # Quadratic with reservoir limit
fit_bayesian_greenland(...)        # Two-component SMB + discharge
fit_bayesian_tws(...)              # Secular + ENSO regression

# Budget constraint
compute_budget_residual(total_fit, component_fits)  # AIS = total - Σ(others)
validate_budget_closure(components, total)           # Check Σ = total

# Component projections
project_component_ensemble(component_fit, temperature_projections, ...)
project_all_components_ensemble(all_fits, temperature_projections, ...)
```

### 5.3 New Notebook: `component_decomposition.ipynb`

Structure follows the staged implementation:
- Cells 0–3: Data loading and harmonization
- Cells 4–9: Stage 1 — Thermosteric
- Cells 10–15: Stage 2a — Glaciers
- Cells 16–21: Stage 2b — Greenland
- Cells 22–25: Stage 3 — TWS
- Cells 26–31: Stage 4 — Antarctic (budget residual + A4)
- Cells 32–35: Full budget validation and variance decomposition
- Cells 36–39: Projections with component breakdown
- Cells 40–43: Summary figures and export

---

## 6. Key Scientific Decisions

### 6.1 Glacier/Ice Sheet Boundaries
- **Decision needed**: How to handle GlaMBIE Region 5 (Greenland periphery) and Region 19 (Antarctic/Subantarctic).
- **Recommendation**: Follow IPCC AR6 convention. Check FACTS component definitions for consistency.

### 6.2 Thermosteric vs. Steric vs. Sterodynamic
- Frederikse "thermodynamic" component includes both thermal expansion and ocean dynamic redistribution.
- NOAA provides purely thermosteric (thermal expansion only).
- For the budget, we want the component that sums cleanly with the cryospheric + TWS terms to give total GMSL. This is thermosteric + ocean dynamics, not thermosteric alone.
- **Recommendation**: Use the IPCC "ocean" component (ocean dynamics + thermal expansion) rather than pure thermosteric, OR use Frederikse thermodynamic.

### 6.3 SMB/Discharge Separation for Greenland
- If we can download SMB model output (RACMO/MAR) + discharge data: fit the two-component model directly.
- If we cannot: use Aschwanden et al. (2019) to set prior on discharge fraction (8–45%), fit total IMBIE GrIS, and partition based on the prior.
- **Recommendation**: Download the data if possible (Stage 2b is significantly more informative with direct SMB/discharge observations). If not feasible initially, use the prior-based approach and flag it for future improvement.

### 6.4 TWS Complexity
- Full ENSO-coupled TWS model (Stage 3) is scientifically interesting but may be overkill for the initial budget.
- **Recommendation**: Start with a simple secular TWS estimate from IPCC AR6 assessed values, validate against GRACE, then add ENSO coupling later if needed.

---

## 7. Verification Strategy

1. **Budget closure**: At every stage, verify Σ(resolved components) + residual = total GMSL within uncertainties.
2. **Cross-validation against IMBIE**: Budget-derived AIS contribution should be consistent with IMBIE direct observations (1992–2020 overlap).
3. **IPCC consistency**: Component projections should be compared with (not necessarily match) IPCC AR6 component-level projections.
4. **Extrapolation sanity**:
   - Thermosteric: should not exceed ~50 cm by 2100 under any scenario (ocean heat capacity limit)
   - Glaciers: cumulative loss must stay below ~320 mm SLE
   - GrIS: Aschwanden et al. constrain 5–33 cm by 2200 (not 2100), so 2100 values should be smaller
   - WAIS: A4 framework provides the physically-motivated uncertainty envelope
5. **Posterior predictive checks**: At each stage, check that model can reproduce the observational record within the stated uncertainties.

---

## 8. Priority Order and Dependencies

```
[Stage 1: Thermosteric]  ←  Needs: NOAA steric reader (exists), Frederikse thermo (in HDF5)
         │                    Downloads: None (may want Argo products later)
         │                    New readers: IPCC FACTS ocean dynamics
         │
         ├──→  [Stage 2a: Glaciers]  ←  Needs: GlaMBIE reader (new)
         │              │                 Downloads: None
         │              │                 New readers: GlaMBIE, IPCC FACTS glaciers
         │              │
         │              ├──→  [Stage 2b: Greenland]  ←  Needs: IMBIE GrIS reader (new)
         │              │              │                  Downloads: RACMO/MAR SMB, discharge data
         │              │              │                  New readers: IMBIE GrIS, IPCC FACTS GIS
         │              │              │
         │              │              ├──→  [Stage 3: TWS]  ←  Needs: GRACE reader (new)
         │              │              │           │              Downloads: None (or dam/GW data)
         │              │              │           │              New readers: GRACE, ONI/MEI, IPCC LWS
         │              │              │           │
         │              │              │           └──→  [Stage 4: AIS]  ←  Budget residual
         │              │              │                                     IMBIE EAIS/Pen readers (new)
         │              │              │                                     A4 framework (exists)
```

**Minimum viable path (if Greenland SMB/discharge data are hard to get):**
1. Thermosteric (Stage 1) — no downloads needed
2. Glaciers (Stage 2a) — no downloads needed
3. Greenland total (Stage 2b, simplified) — no downloads needed, use IMBIE total + Aschwanden priors
4. TWS (Stage 3, simplified) — use IPCC assessed secular rate as prior
5. AIS (Stage 4) — budget residual + A4

This minimum path requires **zero downloads** and only new *reader functions* for local data.

---

## 9. Timeline Estimate

| Task | Effort | Dependencies |
|------|--------|-------------|
| Write IMBIE readers (GrIS, EAIS, Peninsula) | 1 hour | None |
| Write GlaMBIE reader | 1 hour | None |
| Write IPCC FACTS component reader | 2 hours | None |
| Write GRACE/ENSO readers | 1 hour | None |
| Stage 1: Thermosteric calibration + notebook | 3–4 hours | Readers |
| Stage 2a: Glacier calibration + notebook | 3–4 hours | Stage 1, GlaMBIE reader |
| Stage 2b: Greenland calibration + notebook | 4–6 hours | Stage 2a, IMBIE GrIS reader, (optional: SMB download) |
| Stage 3: TWS + notebook | 2–3 hours | Stage 2b, GRACE reader |
| Stage 4: AIS budget + A4 integration + notebook | 3–4 hours | Stage 3, IMBIE EAIS/Pen readers |
| Full budget validation + projection figures | 3–4 hours | All stages |
| LaTeX documentation | 4–6 hours | All stages |

**Total**: ~25–35 hours of implementation work.

**Recommended first session**: Write all readers (IMBIE × 3, GlaMBIE, IPCC FACTS components) + Stage 1 thermosteric calibration.
