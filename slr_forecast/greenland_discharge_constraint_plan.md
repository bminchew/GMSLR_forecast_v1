# Discharge-Constrained Greenland Model — Plan for Review

*Saved 2026-03-02 for later consideration. Not yet implemented.*

## Problem

The physics-informed Greenland model (cells 44-48 of `component_decomposition.ipynb`) fits a two-component model separating SMB (instantaneous, quadratic in T) from discharge (lagged ODE). The initial run revealed an **identifiability problem**:

| Quantity | Current value | Expected |
|----------|--------------|----------|
| Discharge fraction | -17% (unphysical) | Positive; ~50-66% historically but decreasing over time |
| Secular drift c | 0.40 mm/yr (absorbs 88% of signal) | < 0.15 mm/yr |
| SSP5-8.5 projection at 2100 | 688 mm | ~90-290 mm (Aschwanden) |
| R² vs Frederikse | 0.9683 | Good, but misleading — c does the fitting |

**Root cause**: With only total Greenland observations (Frederikse), the model cannot distinguish SMB from discharge. The prior on γ (discharge sensitivity) was too tight (σ = 1 mm/yr/°C) and the prior on c (secular drift) too loose (σ = 0.5 mm/yr), so c absorbed the physics signal.

## Key Physical Insight

Discharge is geometrically constrained by bed topography — it cannot scale arbitrarily with warming. SMB becomes an increasingly large fraction of total mass loss as temperature rises (quadratic response via melt-elevation feedback). Historical discharge fractions (Mouginot 2019: 66%, IMBIE 2023: 50%) will **decrease** over time as SMB dominates. This is what Aschwanden et al. 2019 captures in their projections.

**Therefore**: Constraining the discharge *fraction* of total would be physically wrong. The constraint should be on the **absolute magnitude** of cumulative discharge.

## Proposed Fixes

### Fix 1: Prior adjustments (no external data needed)

| Parameter | Current prior | Proposed | Rationale |
|-----------|--------------|----------|-----------|
| γ (discharge sensitivity) | HalfNormal(σ=0.001) | HalfNormal(σ=**0.005**) | Allow substantial discharge; 5× wider |
| c (secular drift) | Normal(0.0002, 0.0005) | Normal(0.0002, **0.0002**) | Prevent drift from absorbing physics |

No Beta prior on discharge fraction — see physical reasoning above.

### Fix 2: Cumulative discharge observations in likelihood

- **Data source**: Mouginot et al. 2019 (PNAS), cumulative GrIS discharge 1972-2018, 260 outlet glaciers
- **Dual-target likelihood**: L = L_total(Frederikse, 1900-2018) + λ · L_discharge(Mouginot, 1992-2018)
- Constrains γ and τ_dyn from observed discharge magnitude
- Does NOT constrain future fraction — that evolves from the physics
- λ (weight) default 1.0; controls relative influence

**Fallback**: If data unavailable, Fix 1 alone should help substantially.

## Model Structure (unchanged)

```
H_gris(t) = H_smb(t) + H_dyn(t) + c·(t−t₀) + H₀

SMB (instantaneous):   rate_smb = a_eff·T² + b_eff·T
                       H_smb = a_eff·I₂ + b_eff·I₁

Discharge (lagged):    dD_eff/dt = (T − D_eff) / τ_dyn
                       H_dyn = γ · ∫D_eff dτ

Parameters: [a_eff, b_eff, γ, τ_dyn, c, σ_extra, H₀]  — 7 sampled
```

## Code Changes Required

### `bayesian_dols.py`
- `_greenland_log_prior()` (line 3605): No structural changes — just pass different prior_scale values
- `_greenland_log_prob()` (line 3657): Add optional args for cumulative discharge observations; when provided, adds discharge likelihood term after computing H_dyn_cum
- `fit_bayesian_greenland()` (line 3716): Add optional `discharge_obs`, `discharge_obs_sigma`, `discharge_obs_times`, `discharge_rebase_year`, `discharge_weight` kwargs; all default None for backward compatibility

### `slr_data_readers.py` (if data acquired)
- Add `read_mouginot2019_greenland()` reader, ~40 lines, following `_read_imbie_gt()` pattern

### `component_decomposition.ipynb` (cells 45-47)
- Cell 45: Update prior constants, add optional Mouginot data loading with try/except fallback
- Cell 46: Show discharge fraction as diagnostic (not constraint); add discharge data comparison
- Cell 47: Overlay Mouginot cumulative discharge on decomposition figure

## Verification Targets

1. Cumulative H_dyn positive and physically reasonable
2. Secular drift c < 0.15 mm/yr
3. R² ≥ 0.96 against Frederikse
4. Projections closer to Aschwanden ranges
5. Discharge fraction *decreases* in high-emission scenarios (SMB dominates)
6. Backward compatibility with cells 49-53

## Open Questions

- Is the two-component decomposition (SMB + discharge) the right level of complexity, or should we simplify to a single-component model with better priors?
- How sensitive are projections to the choice of τ_dyn prior?
- Should we explore RACMO/MAR SMB data as an alternative constraint (SMB observations rather than discharge)?
- Is there a cleaner way to break the c/γ degeneracy without external data?

## References

- Mouginot et al. (2019), PNAS — 46 years of GrIS mass balance decomposition
- IMBIE (Otosaka et al. 2023) — total GrIS mass balance 1992-2020
- Aschwanden et al. (2019), Science Advances — PISM projections showing discharge 8-45% by 2100
- Mankoff et al. (2020) — updated solid ice discharge estimates
