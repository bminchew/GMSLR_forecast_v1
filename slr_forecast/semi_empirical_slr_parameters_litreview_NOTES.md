# Semi-Empirical SLR Model Parameters: Companion Notes (v2 — Verified)

## Verification Status

Parameters verified from original publications via web search on 2026-03-08:

| Study | Status | Source |
|-------|--------|--------|
| R07   | **VERIFIED** | Full text from PIK PDF: a=3.4 mm/yr/K, T0=-0.5K below 1951-1980, slope robust 3.2-3.5 for embedding 2-17yr |
| VR09  | **VERIFIED** | PNAS full text + PMC: T0=-0.41±0.03K, a=0.56±0.05 cm/a/K (=5.6±0.5 mm/yr/K), b=-4.9±1.0 cm/K (=-49±10 mm/K) |
| GMJ10 | **PARTIALLY VERIFIED** | Table 1 Moberg variant from MetaSD replication: a=1290mm/K, tau=208yr, b=770mm, T0=-0.59K. Historical variant: tau~1190yr, a~5030mm. Did not access Springer full text. |
| JMG10 | **DOI VERIFIED** | 10.1029/2010GL042947 confirmed. Forcing-based model — parameters not temperature-based, so not directly usable with SSP GMST. |
| RPV12 | **DOI VERIFIED** | 10.1007/s00382-011-1226-7 confirmed from multiple sources |
| S12   | **DOI VERIFIED** | 10.1038/nclimate1584 confirmed. Parameters not extracted. |
| J14   | **DOI VERIFIED** | 10.1088/1748-9326/9/10/104008 confirmed. Parameters not extracted. |
| K16   | **DOI VERIFIED** | 10.1073/pnas.1517056113 confirmed from PNAS |
| M16   | **DOI VERIFIED** | 10.1073/pnas.1500515113. T_eq,GSIC = -0.15K from citing paper. Other component params not extracted. |
| G21   | **VERIFIED** | Ocean Sci., 17, 181-186. DOI: 10.5194/os-17-181-2021. TSLS metric defined. |
| G22   | **VERIFIED** | Earth's Future, 10, e2022EF002696. DOI: 10.1029/2022EF002696. Component TSLS values from abstract. |

---

## Projection Recipes for SSP GMST Inputs

IPCC AR6 SSP temperature projections use a 1850-1900 baseline. All studies below use a 1951-1980 baseline (NASA GISS convention). The offset is approximately 0.4°C.

### R07: Simple integration (VERIFIED)

```python
# T_ssp[t] = SSP temperature anomaly relative to 1850-1900 (K)
a = 3.4  # mm/yr/K
T0_ssp = -0.1  # = -0.5 + 0.4, relative to 1850-1900

# Rate at time t:
rate_t = a * (T_ssp[t] - T0_ssp)  # mm/yr

# Cumulative from reference year t_ref:
H = H_ref + a * trapz(T_ssp - T0_ssp, t)  # mm
```

### VR09: Integration + level term (VERIFIED)

```python
a = 5.6    # mm/yr/K
b = -49.0  # mm/K (NEGATIVE)
T0_ssp = -0.01  # = -0.41 + 0.4, relative to 1850-1900

# Rate at time t:
rate_t = a * (T_ssp[t] - T0_ssp) + b * dT_dt[t]  # mm/yr

# Cumulative (the b*dT/dt term integrates analytically):
H = H_ref + a * trapz(T_ssp - T0_ssp, t) + b * (T_ssp[t] - T_ssp[t_ref])  # mm
```

**Key subtlety**: The `b` term is negative, so it acts as: when dT/dt > 0 (warming), b*dT/dt < 0, which *reduces* the rate. This seems counterintuitive but reflects that rapid warming produces a transient sea-level response that *lags* the equilibrium response — the b term corrects for this lag. When warming stops (dT/dt → 0), the b term vanishes and the full a*(T-T0) rate applies.

**Connection to framework**: In the rate-and-state formulation (bayesian_level_space_methods.tex), b_VR = -d*tau. The VR09 b = -49 mm/K with a/b ratio gives an effective tau ≈ |b|/a ≈ 49/5.6 ≈ 8.75 yr — a very short response timescale, consistent with rapid ocean mixed-layer equilibration.

### GMJ10: ODE integration (PARTIALLY VERIFIED — Moberg variant)

```python
a_eq = 1290.0  # mm/K  (equilibrium sensitivity)
tau = 208.0     # yr
b_offset = 770.0  # mm
T0_ssp = -0.59 + 0.4  # = -0.19 K, relative to 1850-1900

# Equilibrium sea level for temperature T:
S_eq = a_eq * (T_ssp[t] - T0_ssp) + b_offset  # mm

# ODE (exact exponential for piecewise-linear T):
dt = 1.0  # yr
T_bar = 0.5 * (T_ssp[t] + T_ssp[t+1])
S_eq_bar = a_eq * (T_bar - T0_ssp) + b_offset
H[t+1] = S_eq_bar + (H[t] - S_eq_bar) * exp(-dt/tau)
```

**WARNING**: The GMJ10 parameters are from the Moberg paleo-temperature variant. The Historical variant gives radically different values (tau~1190yr, a~5030mm). These two variants produce very different projections because the short-tau variant responds quickly to warming while the long-tau variant has large committed rise. The full posterior distribution (not just the mode) is needed for proper UQ. The a-tau posterior ridge is a fundamental identifiability issue.

### JMG10: NOT directly usable with SSP GMST

This is a forcing-based model. To use it, you would need to convert SSP GMST to radiative forcing components (solar, volcanic, GHG+aerosol) or obtain the SSP forcing timeseries directly from the IPCC/CMIP archive.

### M16: Multi-component (NOT reproducible from this CSV)

Each SLR component has its own response function. Consult Supplementary Table S1-S4 of the original paper. The thermal expansion component uses a relaxation model structurally similar to GMJ10 but calibrated only on the steric signal.

---

## Comparison Points for DOLS

The framework's DOLS uses rate(t) = a*T² + b*T + c (quadratic), with:
- a = dα/dT (quadratic sensitivity, mm/yr/°C²)
- b = α₀ (linear sensitivity, mm/yr/°C)
- c = background rate at T=0

The semi-empirical models above are **linear** in T (R07, VR09) or nonlinear only through the relaxation dynamics (GMJ10). None include an explicit T² term.

To compare:
1. Evaluate the DOLS rate at a specific temperature, e.g., rate(T=1°C) = a + b + c
2. Compare with R07's rate at T=1°C: 3.4 * (1.0 - T0)
3. For T0_SSP = -0.1K: R07 rate at 1°C = 3.4 * 1.1 = 3.74 mm/yr
4. DOLS all-dataset ensemble (1950-start): rate at 1°C = 1.83*1 + 1.73*1 + c ≈ 3.56 + c mm/yr

These are broadly consistent, which makes sense: the DOLS quadratic captures the same aggregate signal as the linear semi-empirical models, but partitions it differently between linear and quadratic terms.

The key discriminant is behavior at **high temperatures** (>2°C above baseline): the quadratic DOLS predicts accelerating sensitivity, while R07/VR09 predict linear extrapolation. This is precisely the regime relevant to SSP3-7.0 and SSP5-8.5 late-century projections.
