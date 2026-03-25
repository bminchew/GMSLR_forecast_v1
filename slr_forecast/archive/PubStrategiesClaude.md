# Publication Strategy Notes

*Working document — not for sharing with collaborators*
*Last updated: 2026-02-16*

---

## The Core Body of Work

This project develops a hierarchical sea-level rise forecasting framework that:
1. Calibrates a semi-empirical rate-temperature relationship (DOLS) on observed GMSL and GMST
2. Decomposes projection uncertainty into constrained (DOLS parameter), scenario (SSP spread), and deep (ice sheet) components
3. Constructs physics-informed Antarctic ice-sheet uncertainty that corrects three systematic biases in IPCC AR6 projections (rheology, stochastic amplification, missing processes)
4. Demonstrates that IPCC process models underestimate the observed thermodynamic sensitivity
5. Tests robustness across multiple GMSL/GMST datasets, time windows, and Bayesian frameworks

---

## Option A: Single Comprehensive Paper

### Structure
One paper that tells the full story: calibration, uncertainty decomposition, WAIS physics, robustness, and implications.

### Title (working)
"Hierarchical sea-level rise forecasting: separating predictable thermodynamic response from deep Antarctic ice-sheet uncertainty"

### Main questions
1. How much of future SLR uncertainty is constrained by the observed temperature-SLR relationship vs. dominated by unknowable ice-sheet dynamics?
2. Does the IPCC AR6 framework understate Antarctic ice-sheet uncertainty, and by how much?
3. Is the semi-empirical temperature sensitivity robust across observational datasets, calibration windows, and statistical frameworks?

### Methods
- DOLS on multiple GMSL x GMST combinations with WLS + HAC standard errors
- Sliding-window DOLS for time-varying coefficients
- Bayesian static, DLM, and hierarchical multi-dataset models
- Four physics-informed WAIS uncertainty approaches (A1-A4)
- Emergent sensitivity test: DOLS on IPCC thermodynamic projections
- Monte Carlo projection ensemble with variance decomposition

### Conclusions
- The observed quadratic temperature sensitivity (dα/dT ≈ 2.9 mm/yr/°C² thermodynamic ensemble) is robust across datasets but absent from IPCC process models, which show only linear thermodynamic sensitivity (α₀ ≈ 2 mm/yr/°C)
- IPCC AR6 understates AIS uncertainty by 2-3x due to systematic rheology bias (n=3 vs n=4), omitted stochastic amplification, and missing processes
- The predictable fraction (thermodynamic + scenario) dominates through 2100, but ice-sheet deep uncertainty becomes the primary source of irreducible risk
- Bayesian analysis confirms coefficient stability (DLM Q ≈ 0) and population dα/dT ≈ 2.6 ± 0.3 mm/yr/°C² across 5 datasets

### Target journal
- **Nature Geoscience** — fits their scope (Earth system + climate + policy relevance), appropriate length (3000 words + methods + extended data), high impact
- **AGU Advances** — broader scope, allows longer format, strong SLR readership, faster review
- **Nature Climate Change** — if the policy framing (predictable vs unpredictable uncertainty for adaptation) is foregrounded

### Pros
- Tells a complete, self-contained story
- Stronger impact: the combination of semi-empirical calibration + physics-informed ice uncertainty + IPCC discrepancy is more compelling together than apart
- The variance decomposition is the unifying thread that ties everything together
- Avoids the "salami slicing" concern

### Cons
- Long and dense — may be difficult to fit in Nature-family format (3000 words)
- Reviewers may find the scope too broad or want deeper treatment of individual topics
- The WAIS uncertainty framework and the DOLS methodology serve different audiences (glaciology vs statistics/climate)
- Risk of "trying to do too much" — diluting the message

---

## Option B: Two Papers

### Paper 1: "Semi-empirical sea-level forecasting: a Bayesian hierarchical framework"

**Focus**: The DOLS methodology, its robustness, Bayesian extensions, and the emergent sensitivity discrepancy with IPCC process models.

**Main questions**:
1. Can a simple quadratic rate-temperature model, calibrated on observations, outperform process models for near-term SLR prediction?
2. How robust is the calibrated sensitivity across datasets, time windows, and statistical frameworks?
3. Why do observations show a quadratic temperature sensitivity that IPCC process models do not?

**Key results**:
- DOLS calibration with multi-dataset robustness (24 fits, thermodynamic ensemble dα/dT = 2.85 ± 0.38)
- Sliding-window analysis showing epoch-dependent coefficient evolution
- Bayesian DLM confirming time-invariant coefficients (Q ≈ 0)
- Hierarchical model: population dα/dT = 2.57 ± 0.32 across 5 datasets
- Emergent sensitivity test: IPCC thermodynamic component is linear, not quadratic, with α₀ ≈ 2 mm/yr/°C (factor of 2 below observed)
- TSLS connection to Grinsted & Christensen (2021) and Jevrejeva et al. (2021)

**Target journal**:
- **Journal of Climate** — methodological home, allows thorough treatment, strong readership
- **Earth's Future** — interdisciplinary, policy-relevant, AGU flagship
- **Nature Geoscience** — if the IPCC discrepancy angle is sharpened into a headline result

### Paper 2: "Physics-informed Antarctic ice-sheet uncertainty for sea-level projections"

**Focus**: The WAIS uncertainty framework (A1-A4), its physical basis, and its impact on the predictability partition.

**Main questions**:
1. By how much does the IPCC AR6 understate Antarctic ice-sheet uncertainty?
2. How do rheology bias, stochastic amplification, and missing processes individually and jointly affect the uncertainty distribution?
3. What fraction of future SLR uncertainty is fundamentally irreducible vs. reducible through better observations or models?

**Key results**:
- Four complementary approaches converging on σ_ice ≈ 320-491 mm at 2100 (vs IPCC 150-290 mm)
- Rheology correction: n=3→n=4 adds 21-35% depending on forcing (Martin et al.) and 32±14% pan-Antarctic (Getraer & Morlighem)
- Stochastic amplification: 20-60 cm irreducible uncertainty from MISI (Robel et al. 2019)
- Variance decomposition: ice-sheet fraction rises from 2% (IPCC medium) to 18% (A4)
- The clean separation between emission-dependent (thermodynamic) and emission-independent (WAIS) uncertainty components

**Target journal**:
- **The Cryosphere** — topical home for ice-sheet projections, strong community
- **Science** — if framed as "IPCC systematically underestimates Antarctic uncertainty" (but very competitive)
- **Nature Geoscience** — if the predictability angle and policy relevance are foregrounded

### Pros of two papers
- Each paper has a cleaner, more focused narrative
- Different audiences can engage with the part most relevant to them
- More publication volume (two papers > one for career/visibility)
- Allows deeper treatment of each topic within page limits
- The DOLS methodology paper can stand alone as a statistical contribution; the WAIS paper can stand alone as a glaciology/risk contribution
- Reduces reviewer fatigue

### Cons of two papers
- The two halves are most powerful together — the variance decomposition is the connective tissue
- Risk that reviewers of Paper 2 want to see the full DOLS calibration (which is in Paper 1)
- Sequencing: Paper 1 ideally published first (Paper 2 references its calibrated constrained uncertainty)
- The emergent sensitivity discrepancy (IPCC underestimates thermodynamic sensitivity) straddles both papers — it motivates Paper 2's focus on ice sheets but is methodologically part of Paper 1

---

## Option C: Two Papers (Alternative Split)

### Paper 1: "Predictable and unpredictable components of 21st-century sea-level rise"

**Focus**: The framing paper. Develops the full hierarchy and demonstrates the separation. High-level, broad audience.

**Key elements**:
- DOLS calibration (concise — details in supplement)
- Variance decomposition (the core figure)
- WAIS uncertainty (A4, presented as the recommended estimate with alternatives in supplement)
- The "what can we predict?" message for policy

**Target**: Nature Climate Change, Nature Geoscience, or Science

### Paper 2: "Bayesian semi-empirical sea-level sensitivity: multi-dataset robustness and time-varying calibration"

**Focus**: The technical paper. Deep methodological treatment for the statistics/climate modeling community.

**Key elements**:
- Full DOLS theory and implementation
- Multi-dataset robustness matrix
- Sliding-window analysis
- Bayesian static + DLM + hierarchical models
- IPCC emergent sensitivity comparison

**Target**: Journal of Climate, Journal of Geophysical Research - Oceans

### Assessment
This split is more natural: Paper 1 is the "story" paper aimed at a broad audience; Paper 2 is the "methods" paper aimed at specialists. Paper 1 can reference Paper 2 as "companion" or "submitted" for methodological details. The risk is that reviewers of Paper 1 may want more methodological depth than is appropriate for the format.

---

## Recommendation

**Option C (two papers with story/methods split) is the strongest strategy**, for three reasons:

1. **The central message — "here is what we can and cannot predict about sea-level rise" — is powerful and timely**, and deserves a high-profile venue where the audience includes policymakers and adaptation planners, not just climate scientists. This message gets diluted if buried in 30 pages of methodology.

2. **The methodology is genuinely novel and deep enough to warrant its own paper.** The combination of DOLS, Bayesian extensions, multi-dataset robustness, sliding-window evolution, and the IPCC emergent sensitivity discrepancy is a substantial methodological contribution. Journal of Climate or JGR-Oceans readers will want the full treatment.

3. **Sequencing works naturally.** The methods paper can be submitted first (or simultaneously), and the story paper can reference it. Alternatively, the story paper can include essential methodology in a supplement and reference the companion paper as "in preparation."

---

## Remaining Analyses That Would Strengthen Either Paper

| Analysis | Strengthens | Effort | Priority |
|----------|------------|--------|----------|
| Component-wise DOLS (5c) | Both — reveals which component drives the quadratic | 1 day | High |
| Stress-test WAIS (4) | Paper 2 / WAIS paper — reviewer robustness | 1-2 days | High for submission |
| Greenland regional (5a) | Paper 1 / story paper — "we can predict more if we use the right temperature" | 2-3 days | Medium |
| Predictable/unpredictable framing (5b) | Paper 1 — the concluding synthesis | 1-2 days | High but comes last |
| Publication figures (6) | Both | 2-3 days | Required for submission |

---

## Timeline Sketch

Assuming Option C:

1. **Now → +1 week**: Component-wise DOLS (5c) + WAIS stress tests (4)
2. **+1–2 weeks**: Greenland regional analysis (5a) if data acquisition is quick
3. **+2–3 weeks**: Finalize framing and narrative (5b), draft Paper 1 (story paper)
4. **+3–4 weeks**: Publication figures (6), finalize Paper 2 (methods paper)
5. **+4–5 weeks**: Internal review, circulate to collaborators
6. **+6 weeks**: Submit Paper 1 to Nature Climate Change / Nature Geoscience; submit Paper 2 to Journal of Climate
