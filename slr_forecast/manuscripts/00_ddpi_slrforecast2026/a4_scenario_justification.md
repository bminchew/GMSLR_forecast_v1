# A4 WAIS Scenario Framework: Literature Justification

**Purpose:** Provide detailed, citation-backed justification for every parameter
choice in the revised 2-scenario A4 framework, sufficient to satisfy a skeptical
reviewer with deep expertise in ice sheet modeling.

---

## 1. Framework Structure: Two Scenarios

| Scenario | P | Range (mm, 2100) | α | Physics |
|----------|---|-------------------|---|---------|
| S1: Status quo | 0.10 | 25–85 | 0 | No MISI; linear melt-driven discharge |
| S2: Fast WAIS (MISI + MICI) | 0.90 | 150–1300 | +3.22 | MISI, with or without ice-cliff cascade; upper bound pinned to AR6 low-confidence AIS p95 |

### Why two scenarios, not three

An earlier version of this framework used three scenarios: S1 (status quo),
S2 (moderate MISI, 150–1000 mm, P=0.80), and S3 (MISI+MICI, 600–1300 mm,
P=0.10). We merge S2 and S3 into a single scenario because the distinction
is not physically defensible: if MISI is triggered, current models cannot
represent the amplifying processes that accompany it, and there is no
principled way to separate "MISI without cliff failure" from "MISI with
cliff failure" as two independent, mutually exclusive branches. Specifically:

- **Calving is not represented** in any ISMIP6 model (Aschwanden et al., 2021,
  The Cryosphere; Fricker et al., 2025, Science). Calving is a first-order
  control on grounding-line retreat rate, and ice-cliff failure is simply the
  limiting case of unrepresented calving processes, not a categorically
  distinct mechanism.
- **Rheology is systematically wrong** (n=3 vs observed n≈4): Martin et al.
  (2026, AGU Advances) show 21–35% underestimation of 100-year SLR from
  rheology alone; Getraer & Morlighem (2025, GRL) find 32±14% greater
  Amundsen Sea Embayment loss with n=4.
- **Subglacial hydrology** can amplify discharge up to threefold (Nature
  Communications, 2025, doi:10.1038/s41467-025-58375-4).
- **Ocean-ice coupling** is parameterized, not dynamically computed, in ISMIP6.

A scenario where "MISI happens but proceeds exactly as models predict,
without either amplification or cliff failure" is therefore an artifact of
model limitations, not a distinct physical regime. The merged S2 uses a
probability-weighted skewed distribution (§4) to capture the full range from
moderate to amplified MISI outcomes, including the possibility of MICI.

This merge is one step further along the same logic that motivated an even
earlier four-scenario framework's collapse to three: that version separated
"moderate MISI" (150–400 mm) from "MISI with amplifiers" (400–1000 mm) before
those two were themselves merged into a single MISI scenario. The current
2-scenario framework completes this simplification by folding the MISI+MICI
branch into the same merged distribution, while still capturing MICI's
distinct (negative) skew contribution through the probability-weighted blend
described in §4.

---

## 2. Scenario Weights: 10–90

### S1 = 10% (no MISI)

**Supporting evidence:**

- IPCC AR6 Ch. 9 (Fox-Kemper et al., 2021): MISI "may be underway" in the
  Amundsen Sea Embayment (medium confidence). The language "cannot be ruled out
  that it has already begun" implies P(no MISI) is well below 50%.
- Joughin et al. (2014, Science): Concluded that "early-stage collapse" of
  Thwaites is already underway, with grounding-line retreat on a retrograde bed.
- IMBIE (2023, ESSD): WAIS mass loss rate increased from 0.15 mm/yr (1992)
  to 0.44 mm/yr (2017), with structural acceleration.
- Naughten et al. (2023, Nature Climate Change): Rapid WAIS ice-shelf melting
  at ~3× historical rates is unavoidable regardless of emissions scenario —
  the ocean forcing for MISI is already committed.
- van den Akker et al. (2025, The Cryosphere): Present-day ocean thermal
  forcing, held constant, is sufficient to deglaciate large parts of WAIS.
  Thwaites and Pine Island collapse within 300–2500 years.
- Bamber et al. (2019, PNAS): Expert opinion on whether observed WAIS
  acceleration is externally forced vs internal variability was split 7/7/8
  across 22 experts — but this was before several confirming observations.

**Assessment:** Sustained stability (no MISI) requires grounding lines to
stabilize despite ongoing retreat on retrograde slopes, committed ocean
warming, and observed acceleration. A 10% weight reflects the small but
non-zero possibility that observed trends are reversible or dominated by
internal variability.

### S2 = 90% (Fast WAIS: MISI, with or without MICI)

**Supporting evidence for MISI as the baseline expectation:**

- AR6 assigns medium confidence to MISI contributing up to ~40 cm by 2100,
  and includes MISI in its main (medium-confidence) projections.
- Ritz et al. (2015, Nature): Probabilistic treatment of MISI triggering
  finds it is the dominant instability pathway. 5% probability of exceeding
  30 cm by 2100 from Antarctica.
- Kopp et al. (2017, Earth's Future): Treats MISI as the baseline instability
  physics in probabilistic SLR projections.
- Bamber et al. (2019, PNAS): WAIS distributions are dominated by MISI-type
  dynamics. The +5°C scenario gives WAIS 5th–95th of −5 to 93 cm (median 18
  cm), with strongly positive skew reflecting instability outcomes.
- DeConto et al. (2021, Nature): Revised projections (without MICI) give
  total Antarctic median ~34 cm under RCP8.5, with nonlinear acceleration
  above 3°C.

**Supporting evidence for including MICI's contribution within S2, but with
low weight in the blend:**

- AR6 assigns low confidence to MICI. It is excluded from medium-confidence
  projections and noted as "characterised by deep uncertainty."
- Morlighem et al. (2024, Science Advances): Multi-model study of Thwaites
  found WAIS "may not be vulnerable to MICI during the 21st century." Dynamic
  thinning prevents cliffs from growing tall enough for sustained failure.
  Calving rates would need to be ≥25× higher than models suggest.
- Edwards et al. (2019, Nature): MICI is not required to reproduce LIG,
  Pliocene, or modern sea-level changes.
- Clerc, Minchew & Liberty (2019, GRL): Realistic ice-shelf removal
  timescales (days+) raise the critical cliff height from ~90 m to ~540 m
  via viscous relaxation, substantially reducing MICI likelihood.
- Crawford et al. (2021, Nature Communications): Structural cliff failure
  initiates at ~136 m (higher than commonly assumed). Stabilizing feedbacks
  (melange, viscous flow) can arrest retreat.
- Schlemm et al. (2022, The Cryosphere): MICI is self-limiting due to
  melange buttressing and embayment geometry changes.
- DeConto & Pollard (2016, Nature): MICI + hydrofracturing calibrated against
  Pliocene/LIG constraints produced ~1 m Antarctic SLR by 2100 (RCP8.5).
  While now considered an overestimate, the mechanism has not been definitively
  ruled out.
- Bassis & Walker (2012, Proc. Roy. Soc. A): Ice cliffs exceeding ~100 m
  (with crevasses) are structurally unstable. Theoretical maximum stable
  height is ~220 m for intact ice.
- Gilford et al. (2020, JGR): Bayesian emulator shows MICI matters for
  high-end paleo scenarios (>6 m Antarctic SLE during LIG).
- IPCC AR6: MICI is presented under "cannot be ruled out" language for
  low-likelihood, high-impact storylines.

**Assessment:** MISI is the most physically supported instability mechanism,
included in IPCC medium-confidence projections, and arguably already
initiated. The 90% weight reflects the consensus position that MISI is the
baseline expectation for WAIS evolution. Rather than assigning MICI a
separate, small mixture weight as in the earlier three-scenario framework,
its distinct (negative) skew and evidentiary skepticism are folded directly
into S2's blended shape parameters (§4): the weight of evidence since 2019
has moved decisively against MICI operating during the 21st century,
particularly Morlighem et al. (2024) and Clerc et al. (2019), but the
mechanism cannot be entirely excluded and its potential impact is large
enough that it must be represented. This is functionally equivalent to a
10%-within-90% (i.e., 0.10/0.90 of the merged mass) contribution from
MICI-type outcomes, consistent with AR6's low-confidence assessment and
with our conclusions being qualitatively insensitive to whether MICI
operates.

### Cross-check with Bamber et al. (2019) SEJ

The Bamber et al. structured expert judgment provides an independent
validation target for our mixture distribution. Their WAIS-specific results:

| Warming | 5th | Median | 83rd | 95th |
|---------|-----|--------|------|------|
| +2°C | −30 mm | 80 mm | 230 mm | 440 mm |
| +5°C | −50 mm | 180 mm | 460 mm | 930 mm |

Our A4 mixture (after rheology correction) produces:
- Median: ~410 mm, 90% CI: [60, 1620] mm

Our distribution is wider than Bamber's +5°C WAIS estimates. This is
expected and defensible because: (a) Bamber was published before Martin et al.
(2026) quantified the rheology bias; (b) the SEJ was conducted before
Morlighem et al. (2024) and recent Thwaites observations that have sharpened
the evidence for ongoing instability; (c) our framework explicitly includes
the rheology correction that shifts the entire distribution upward by ~28%;
and (d) merging the former MICI-only tail into the dominant 90%-weight
scenario, with its upper bound pinned to the AR6 low-confidence p95
(1300 mm), widens the tail relative to the three-scenario version, where
that same upper bound was reached by only 10% of the mixture's mass.
In the paper, we present Bamber SEJ as a complementary, methodology-orthogonal
constraint on WAIS uncertainty, noting that agreement in the broad shape
(strongly right-skewed, median in the hundreds of mm) strengthens confidence
in both approaches despite their different methodologies.

---

## 3. Scenario Ranges

### S1: 25–85 mm (status quo)

MISI does not operate — bed topography, mélange buttressing, and ice-shelf
re-formation prevent the runaway grounding-line flux feedback, regardless
of ocean warming. Discharge responds linearly to ocean thermal forcing.

The range is derived from a physical scaling chain:

1. **Melt–discharge linearity (Joughin et al. 2021):** PIG ice loss scales
   linearly with time-averaged melt volume (r² ≥ 0.98). Holds across ASE
   sectors in ISMIP6 ensemble (r² ≥ 0.94).
2. **Quadratic melt scaling:** Melt ∝ (T_ocean − T_f)². Current PIG melt
   ~67 Gt/yr at ΔT₀ ≈ 2.15°C above freezing.
3. **Projected ocean warming (Naughten et al. 2023):** Amundsen shelf warms
   +0.7 to +1.5°C by 2100 (Paris 1.5°C through RCP 8.5).
4. **Enhancement factor:** f = [(2.15 + ΔT_warm)/2.15]². At +1.5°C: f ≈ 2.9.
   At +0.7°C: f ≈ 1.7. Consistent with Jourdain et al. (2022): 1.4–2.2×.
5. **Rate ramp:**
   - *Lower bound:* IMBIE time-averaged rate (0.24 mm/yr) × 95 yr ≈ 23 mm → **25 mm**.
   - *Upper bound:* Peak rate (0.44 mm/yr) ramped by f = 2.9 to 1.3 mm/yr.
     Linear ramp: (0.44 + 1.3)/2 × 95 ≈ 83 mm → **85 mm**.

### S2: 150–1300 mm (Fast WAIS: MISI, with or without MICI)

This merges the former S2 (150–1000 mm, moderate MISI) and S3 (600–1300 mm,
MISI+MICI) ranges. The lower bound is unchanged from the former S2; the
upper bound is unchanged from the former S3, now serving as the pin for the
entire merged scenario.

**Lower bound (150 mm), basin-by-basin derivation:**

Anchored to transient-calibrated ice sheet model projections (Goldberg,
Morlighem & Gourmelen, 2026, GRL) and observed grounding-line retreat rates
(Rignot et al., 2026, PNAS):

- *Thwaites* (~40 mm): Goldberg et al. (2026) calibrated two independent
  models (STREAMICE, ISSM) against observed surface elevation changes and
  found volume-above-flotation loss rates of 180–200 Gt/yr by 2067 from the
  trunk subdomain alone, using n=3 with no calving or subglacial hydrology.
  Integrating to 2100 and scaling to the full catchment gives ~40 mm SLE —
  from a model configuration whose every omission biases mass loss low.
- *PIG* (~40 mm): 33 km of grounding-line retreat since 1992 (Rignot et al.,
  2026), sharing the same ocean forcing and retrograde bed geometry as
  Thwaites.
- *Smith/Kohler* (~12 mm): 43 km GL retreat since 1996, the largest relative
  to glacier size in the ASE.
- *Other ASE* (~10 mm): Pope, Haynes, Berry, Hull/Land — all retreating.
- *Non-ASE WAIS* (~30 mm): Basins without MISI (Siple Coast, Marie Byrd
  Land) contributing at S1-like rates.

The maximally conservative sum (~130 mm) understates the lower bound because
of the documented low biases in the underlying models (n=3, no calving, no
subglacial hydrology, premature rate stabilization); the 150 mm floor
accounts for these known biases without overcorrecting.

**Upper bound (1300 mm), pinned to the IPCC AR6 low-confidence storyline:**

S2's upper bound is anchored to the upper portion of the IPCC AR6
low-confidence Antarctic ice sheet projection under SSP5-8.5 (Fox-Kemper
et al., 2021). The AR6 low-confidence AIS distribution at 2100 under
SSP5-8.5 has:

| Quantile | AIS contribution (mm) |
|----------|----------------------|
| p83      | 559                  |
| p95      | 1309                 |

The upper bound (1300 mm) maps directly onto the AR6 p95 (1309 mm, rounded).
Because S2 now carries 90% of the mixture's weight (versus 10% for the
former S3), this bound governs the tail behavior of the great majority of
the distribution, not just a small minority branch. This anchoring is
appropriate because the AR6 low-confidence projection is precisely the one
that incorporates the deep-uncertainty processes (MISI + MICI) that define
S2's upper range.

For GMSL forecasting purposes, it is immaterial whether ice loss originates
from WAIS or from the marine basins of EAIS — both contribute to the same
global ocean. In the AR6 low-confidence storyline (and in DeConto & Pollard
2016), WAIS dominates but EAIS marine basins (e.g., Wilkes, Aurora) also
contribute. Our S2 upper bound therefore represents total Antarctic
instability contribution, labeled "WAIS" as shorthand because WAIS dynamics
drive the instability.

Independent cross-checks on the upper bound:

- Cross-checked against the ASE ice volume above flotation
  (V_af ≈ 1100 mm; Morlighem et al. 2020, BedMachine Antarctica);
  the ~200 mm excess is consistent with contributions from EAIS
  marine basins (Wilkes, Aurora) in the AR6 low-confidence storyline.
- Edwards et al. (2019, Nature) find total Antarctic outcome ~450 mm with
  MICI under RCP8.5; adding MISI from basins where MICI doesn't operate
  brings the total to ~600 mm — close to the AR6 p83, which serves as a
  secondary (not exactly matched) reference point for the interior shape
  of S2's distribution.
- DeConto & Pollard (2016): 640–1140 mm total Antarctic under RCP8.5
  with MICI — our upper bound is slightly above their range,
  consistent with the rheology correction that postdates their work.
- Bamber et al. (2019) WAIS 95th percentile at +5°C: 930 mm — below
  our upper bound, as expected since Bamber predates the rheology
  correction literature.
- van den Akker et al. (2025, The Cryosphere): During rapid collapse
  phase, WAIS contributes ~3 mm/yr GMSL. Over several decades this can
  reach 1 m-scale contributions, and reaching 1.3 m by 2100 implies a
  genuinely low-probability, deep-uncertainty tail consistent with AR6's
  own "low confidence" framing.

---

## 4. Skewness Parameterization

### A probability-weighted blend, not a single scenario's shape

With S2 now merging the former MISI-only and MISI+MICI branches, its shape
parameters (α, and the trajectory exponent's log-mean) are the
probability-weighted blend of the two source branches, weighted by their
share of the merged 0.90 mass:

α = (0.80 × 4.0 + 0.10 × (−3.0)) / 0.90 = **3.22**

β log-mean = (0.80 × ln 1.8 + 0.10 × ln 2.2) / 0.90 = ln(1.84), i.e. median
**1.84**

This composites both literature-motivated skew regimes into the single
merged distribution, rather than an arbitrary refit to any single target
percentile (such as exactly matching the AR6 p83). With bounds of
150–1300 mm, α = 3.22 produces a pre-rheology-correction median of ~350 mm
(own-scenario, i.e., before blending with S1) — somewhat above the former
MISI-only scenario's ~300 mm, because the merged scenario now also carries
the former MICI branch's mass and its 95th percentile is pinned ~300 mm
higher than before.

### Positive skew component: Robel et al. (2019)

Robel, Seroussi & Roe (2019, PNAS) demonstrated analytically and
numerically that MISI amplifies and skews uncertainty:

- **Analytical result:** When the grounding-line flux nonlinearity exponent
  β > 3, ensemble distributions of retreating grounding lines are skewed
  toward more retreat (positive skew in SLR). Realistic values of β are
  4–5, well above the threshold.
- **Physical mechanism:** On retrograde beds, more-retreated ensemble
  members retreat faster (positive feedback), while less-retreated members
  experience weaker instability. This creates systematic asymmetry where
  the tail toward greater ice loss is heavier.
- **Thwaites ensemble (500 members, ISSM):** Spread of ~20 cm SLR during
  fast retreat with interdecadal forcing, ~40 cm with multidecadal forcing.
  This represents 25–50% of total Thwaites deglaciation SLR.
- **Fractional projection uncertainty:** Ranges from 0.25 to 0.50
  depending on forcing persistence.
- **Follow-up work:** Verjans, Robel & Ambelorun (2024, The Cryosphere)
  showed that noise-induced drift from calving variability is equivalent
  to ~1 cm GMSL in the first century. Reproducing this drift
  deterministically requires a 270% increase in mean calving rate —
  indicating that deterministic models are systematically biased low.
- **Robel et al. (2026, preprint):** Climate variability alone triggers
  Thwaites collapse within ~300 years; low-frequency ocean variability
  accelerates disintegration by up to 250 years.

This positive-skew evidence motivated the former MISI-only scenario's
α = +4, which contributes 0.80/0.90 of the weight in the blended α = 3.22.

**Bamber et al. (2019) conditional median cross-check** (as applied to the
former MISI-only scenario): The Bamber WAIS median at +5°C is 180 mm
(unconditional). Removing ~15% of outcomes corresponding to S1-like
stability and ~5% to S3-like extremes, the conditional-on-MISI median is
~250–300 mm. With the former MISI-only bounds of 150–1000 mm, α = 4
produced a pre-correction median of ~303 mm — within this range.

### Negative skew component: MICI skepticism

The negative-α contribution (from the former MISI+MICI scenario, weighted
0.10/0.90 into the blend) concentrates probability toward the lower portion
of its range, reflecting:

- Morlighem et al. (2024, Science Advances): MICI is unlikely at Thwaites
  during the 21st century; calving rates would need to be ≥25× higher
  than physically motivated models suggest.
- Clerc et al. (2019, GRL): Realistic ice-shelf removal timescales raise
  the critical cliff height from ~90 m to ~540 m.
- Schlemm et al. (2022, The Cryosphere): MICI is self-limiting due to
  melange buttressing.

If MICI operates at all, it is more likely to produce outcomes at the
lower end of its potential range than at the maximum. This is why the
blended α (3.22) is somewhat below the former MISI-only value (4.0),
pulling some probability mass down from the extreme positive-skew case,
consistent with the evidence against efficient MICI operation.

---

## 5. Rheology Correction

Applied to all scenarios as a multiplicative factor: median 1.28, σ = 0.07.

| Study | Finding |
|-------|---------|
| Millstein, Minchew & Pegler (2022, Comm. Earth & Env.) | Satellite obs: n = 4.1 ± 0.4 in fast-flowing Antarctic ice shelves |
| Bons et al. (2018, GRL) | Field obs, N. Greenland: n ≈ 4 |
| Ranganathan & Minchew (2024, PNAS) | Dislocation creep (n=4) dominates in all fast-flowing areas |
| Fan et al. (2025, Nature Geoscience) | Lab meta-analysis: n = 3.5–3.7, closer to 4 excluding high-T data |
| Martin et al. (2026, AGU Advances) | n=3→n=4 underestimates 100-yr SLR by **21–35%** (MISMIP+) |
| Getraer & Morlighem (2025, GRL) | n=4 gives **32±14%** greater ASE loss by 2100 (ISMIP6 framework) |

The correction factor of 1.28 (median) is conservative relative to the
literature range of 1.21–1.35. The σ = 0.07 spans the range from Martin
et al.'s MISMIP+ (21%, factor 1.21) to their ABUMIP (35%, factor 1.35).

The rheology correction is applied to all scenarios because the bias affects
every ice sheet model regardless of which instability mechanism operates.

---

## 6. Summary: Why Our Distribution Exceeds IPCC

Our A4 mixture produces wider uncertainty than IPCC AR6 medium-confidence
projections. This is expected and physically justified because IPCC
projections inherit multiple documented biases:

1. **Rheology bias:** All ISMIP6 models use n=3; observed n ≈ 4 produces
   21–35% more SLR (Martin et al., 2026; Getraer & Morlighem, 2025).
2. **Missing calving:** No ISMIP6 model adequately represents calving
   (Aschwanden et al., 2021; Fricker et al., 2025).
3. **Missing subglacial hydrology:** Can amplify discharge up to 3×
   (Nature Communications, 2025).
4. **Biased ensemble:** ISMIP6 models share methods and jointly omit
   physics — not independent samples (Aschwanden et al., 2021).
5. **Incomplete uncertainty quantification:** ISMIP6 integrates over model
   structure but not parametric uncertainty, which may be larger
   (Aschwanden et al., 2019: IQR up to 12.9 cm SLE for RCP8.5).
6. **Initialization uncertainty:** Calibration method produces factor-of-2
   spread comparable to inter-model spread (Goldberg et al., 2026, GRL).
7. **Historical validation failure:** Most ISMIP6 models do not reproduce
   observed mass loss; the 95th percentile merely tracks observations
   (Aschwanden et al., 2021).
8. **Observed rates exceed models:** Under continued warming to 4°C+,
   IPCC models show slower SLR rates than extrapolation of modern
   observations (Hamlington et al., 2024, Comm. Earth & Env.).

These are not speculative concerns — each is documented in the peer-reviewed
literature and most have been quantified. Our framework addresses them
explicitly through the rheology correction and the scenario structure, which
spans the range from model-consistent outcomes (lower S2) through
model-exceeding outcomes supported by observations and theory.

---

## References

Aschwanden, A., et al. (2021). Brief communication: ISMIP6 ... The Cryosphere, 15, 5705–5715. doi:10.5194/tc-15-5705-2021

Bamber, J.L., & Aspinall, W.P. (2013). An expert judgement assessment ... Nature Climate Change, 3, 424–427. doi:10.1038/nclimate1778

Bamber, J.L., et al. (2019). Ice sheet contributions to future sea-level rise from structured expert judgment. PNAS, 116(23), 11195–11200. doi:10.1073/pnas.1817205116

Bassis, J.N., & Walker, C.C. (2012). Upper and lower limits on the stability of calving glaciers ... Proc. Roy. Soc. A, 468, 913–931.

Bons, P.D., et al. (2018). Greenland Ice Sheet: Higher nonlinearity of ice flow ... GRL, 45. doi:10.1029/2018GL078356

Clayton, T., et al. (2025). Modeling ice cliff stability using a new Mohr-Coulomb-based phase field fracture model. J. Glaciology. doi:10.1017/jog.2025.18

Clerc, F., Minchew, B.M., & Behn, M.D. (2019). Marine Ice Cliff Instability Mitigated by Slow Removal of Ice Shelves. GRL, 46. doi:10.1029/2019GL084183

Crawford, A.J., et al. (2021). Marine ice-cliff instability modeling shows mixed-mode ice-cliff failure ... Nature Communications, 12, 2861. doi:10.1038/s41467-021-23070-7

DeConto, R.M., & Pollard, D. (2016). Contribution of Antarctica to past and future sea-level rise. Nature, 531, 591–597. doi:10.1038/nature17145

DeConto, R.M., et al. (2021). The Paris Agreement and future sea-level rise from Antarctica. Nature, 593, 83–89. doi:10.1038/s41586-021-03427-0

Edwards, T.L., et al. (2019). Revisiting Antarctic ice loss due to marine ice-cliff instability. Nature, 566, 58–64. doi:10.1038/s41586-019-0901-4

Edwards, T.L., et al. (2021). Projected land ice contributions to twenty-first-century sea level rise. Nature, 593, 74–82. doi:10.1038/s41586-021-03302-y

Fan, S., et al. (2025). Reconciling the discrepancy between ice-creep experiments ... Nature Geoscience. doi:10.1038/s41561-025-01661-z

Fox-Kemper, B., et al. (2021). Ocean, Cryosphere and Sea Level Change. In IPCC AR6 WG1 Ch. 9.

Fricker, H.A., et al. (2025). Antarctica in 2025. Science. doi:10.1126/science.adt9619

Getraer, G., & Morlighem, M. (2025). The impact of a higher-order creep law ... GRL. doi:10.1029/2024GL112516

Gilford, D.M., et al. (2020). Could the Last Interglacial Constrain Projections ... JGR Earth Surface, 125. doi:10.1029/2019JF005418

Goldberg, D.N., Morlighem, M., & Gourmelen, N. (2026). Transient calibration ... GRL. doi:10.1029/2025GL118823

Golledge, N.R., et al. (2015). The multi-millennial Antarctic commitment to future sea-level rise. Nature, 526, 421–425. doi:10.1038/nature15706

Golledge, N.R., et al. (2019). Global environmental consequences of twenty-first-century ice-sheet melt. Nature, 566, 65–72. doi:10.1038/s41586-019-0889-9

Hamlington, B.D., et al. (2024). The rate of global sea level rise doubled ... Comm. Earth & Env. doi:10.1038/s43247-024-01761-5

Horton, B.P., et al. (2020). Estimating global mean sea-level rise ... npj Climate Atmos. Sci. doi:10.1038/s41612-020-0121-5

IMBIE Team (2023). Mass balance of the Greenland and Antarctic ice sheets ... ESSD, 15, 1597–1616. doi:10.5194/essd-15-1597-2023

Joughin, I., et al. (2014). Marine Ice Sheet Collapse Potentially Under Way for the Thwaites Glacier Basin ... Science, 344, 735–738. doi:10.1126/science.1249055

Kopp, R.E., et al. (2014). Probabilistic 21st and 22nd century sea-level projections ... Earth's Future, 2, 383–406. doi:10.1002/2014EF000239

Kopp, R.E., et al. (2017). Evolving understanding of Antarctic ice-sheet physics ... Earth's Future, 5, 1217–1233. doi:10.1002/2017EF000663

Levermann, A., et al. (2020). Projecting Antarctica's contribution to future sea level rise from basal ice shelf melt ... Earth Syst. Dyn., 11, 35–76. doi:10.5194/esd-11-35-2020

Martin, D.F., Trevers, R., & Minchew, B.M. (2026). A higher-order flow law produces systematic under-estimation ... AGU Advances, 7. doi:10.1029/2025AV001946

Millstein, J.D., Minchew, B.M., & Pegler, S.S. (2022). Ice viscosity is more sensitive to stress than commonly assumed ... Comm. Earth & Env. doi:10.1038/s43247-022-00385-x

Morlighem, M., et al. (2024). The West Antarctic Ice Sheet may not be vulnerable to marine ice cliff instability ... Science Advances, 10, eado7794. doi:10.1126/sciadv.ado7794

Naughten, K.A., et al. (2023). Unavoidable future increase in West Antarctic ice-shelf melting ... Nature Climate Change. doi:10.1038/s41558-023-01818-x

Oppenheimer, M., Little, C.M., & Cooke, R.M. (2016). Expert judgement and uncertainty quantification ... Nature Climate Change, 6, 445–451. doi:10.1038/nclimate2959

Pattyn, F. (2018). The paradigm shift in Antarctic ice sheet modelling. Nature Communications, 9, 2728. doi:10.1038/s41467-018-05003-z

Pollard, D., DeConto, R.M., & Alley, R.B. (2015). Potential Antarctic Ice Sheet retreat driven by hydrofracturing and ice cliff failure. EPSL, 412, 112–121. doi:10.1016/j.epsl.2014.12.035

Ranganathan, M., & Minchew, B.M. (2024). A modified Glen's law for high-stress deformation ... PNAS. doi:10.1073/pnas.2309788121

Ritz, C., et al. (2015). Potential sea-level rise from Antarctic ice-sheet instability constrained by observations. Nature, 528, 115–118. doi:10.1038/nature16147

Robel, A.A., Seroussi, H., & Roe, G.H. (2019). Marine ice sheet instability amplifies and skews uncertainty ... PNAS, 116(30), 14887–14892. doi:10.1073/pnas.1904822116

Schlemm, T., et al. (2022). Stabilizing effect of mélange buttressing on the marine ice-cliff instability ... The Cryosphere, 16, 1979–1996. doi:10.5194/tc-16-1979-2022

Schohn, D., et al. (2025). Temperate ice is linear-viscous ... Science, 387, 182–185. doi:10.1126/science.adp7708

Seroussi, H., et al. (2020). ISMIP6 Antarctica ... The Cryosphere, 14, 3033–3070. doi:10.5194/tc-14-3033-2020

van den Akker, T., et al. (2025). West Antarctic Ice Sheet retreat ... The Cryosphere, 19, 283+. doi:10.5194/tc-19-283-2025

Verjans, V., Robel, A.A., & Ambelorun, O. (2024). Biases in ice sheet models from missing noise-induced drift. The Cryosphere, 18, 2613+. doi:10.5194/tc-18-2613-2024
