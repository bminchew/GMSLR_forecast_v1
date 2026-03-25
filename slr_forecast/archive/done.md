# Archived Plans and Notes

*Consolidated 2026-03-22. These plans have been fully or largely executed.*
*Kept for historical reference only — do not update.*

---

# 1. Component Decomposition Plan

*Originally: `component_decomposition_plan.md`*
*Status: FULLY EXECUTED — all stages, readers, and notebook cells implemented.*

## Motivation

The aggregate GMSL–GMST model has reached its natural ceiling. The quadratic rate coefficient `a` aliases physically distinct processes that respond to warming at different rates. Moving to component-wise estimation allows component-appropriate functional forms, budget constraint as regulariser, physically defensible extrapolation, and transparent uncertainty attribution.

## Component Structure

- GMSL_total = GMSL_thermo + GMSL_glaciers + GMSL_GrIS + GMSL_AIS + GMSL_TWS
- Thermosteric (Stage 1): linear in T, PC prior on a_thermo ✓
- Glaciers (Stage 2a): quadratic with GlaMBIE validation ✓
- Greenland (Stage 2b): joint SMB + discharge with ocean T ODE ✓
- TWS (Stage 3): IPCC assessed secular rate ✓
- Antarctica (Stage 4): EAIS trend-only, Peninsula linear, WAIS A4 framework ✓

## Readers Written

All planned readers implemented in `slr_data_readers.py`:
- IMBIE: Greenland, EAIS, Peninsula, Antarctica total, all ice sheets ✓
- GlaMBIE: global consensus, regional ✓
- IPCC FACTS: generic component reader ✓
- GRACE TWS, NOAA ONI/MEI ✓

---

# 2. Greenland Discharge Constraint Plan

*Originally: `greenland_discharge_constraint_plan.md`*
*Status: SUPERSEDED — joint model with Mankoff/Mouginot data implemented.*

## Problem

Initial two-component model had identifiability problem: discharge fraction = -17% (unphysical), secular drift c absorbed 88% of signal.

## Solution Implemented

Joint SMB + discharge fit to Mankoff observational data (R²=0.95). Ocean temperature ODE for discharge with EN4/Argo transfer function. Budget closure via rate+accel constraint (not level).

---

# 3. Compositional Nonlinearity Plan

*Originally: `compositional_nonlinearity_plan.md`*
*Status: LARGELY EXECUTED — core argument implemented, now being written as `00_ddpi_slrforecast2026.tex`.*

## Central Thesis

The apparent quadratic GMSL-GMST relationship is compositional (shifting component dominance), not dynamical (single process accelerating). Each component individually has a simpler relationship with temperature.

## Phases Completed

- Phase 1 (component rate vs T): Implemented in component_decomposition notebooks ✓
- Phase 2 (compositional emergence): Coefficient sum comparison done ✓
- Phase 3 (sliding-window contributions): Component rates computed ✓
- Phase 4 (component projections): All components projected ✓
- Phase 5 (uncertainty attribution): Variance decomposition done ✓
- Phase 6 (summary): Being written as manuscript ✓

## Open items (migrated to TODO.md)

- Informative ΔT₀ prior for BPS cv mode
- Whether BPS trend constraint adds value in component-sum framework

---

# 4. Publication Strategies

*Originally: `PubStrategiesClaude.md`*
*Status: SUPERSEDED — paper plan evolved into refactor.md §0 and the actual manuscript.*

## Options Considered

- Option A: Single comprehensive paper
- Option B: Two papers (methods + WAIS)
- Option C: Two papers (story + methods) ← Recommended

## Current Direction

Single paper targeting PNAS/Science with three-step framework (naive → aggregate → component-wise). Working title: "Data indicate that sea level rise projections are too low by at least half"

---

# 5. Next Steps (Component Decomposition)

*Originally: `next_steps.md`*
*Status: 3/4 items DONE.*

## Completed

1. Surface-to-ocean temperature transfer function (2026-03-22) ✓
2. Projection cell rewrite (2026-03-22) ✓
4. Antarctic sub-component fits (2026-03-22) ✓

## Open item (migrated to TODO.md)

3. Pass 2 re-fit with budget rate+acceleration constraint for Greenland joint model
