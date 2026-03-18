# Implementation Plan: Paper C — Hierarchical Bayesian Framework

## Overview

Paper C is the architecture paper. It presents the full framework design and demonstrates it through Stage 2 (thermosteric + glaciers + Greenland). It depends on Papers A and B for the empirical results (DOLS projections, budget constraint, WAIS treatment) but provides the unified architecture that subsumes both.

The implementation has two parts: (1) the Bayesian inference engine (MCMC with budget constraint), which is the core new code; (2) the component models, which are relatively simple parametric forms with priors.

---

## Timeline

This paper is on a longer timeline than A and B (3-6 months after them). The MCMC engine is the bottleneck. If a working MCMC posterior can be produced within 2 months, the paper can be submitted within 4 months of starting.

---

## Project structure

```
paperC_framework/
├── gmslr_framework/                # The framework code (reusable beyond the paper)
│   ├── core/
│   │   ├── dag.py                  # DAG specification and validation
│   │   ├── budget_constraint.py    # Rate-level budget as likelihood term
│   │   └── types.py                # Shared types (ComponentModel, Prior, etc.)
│   ├── components/
│   │   ├── thermosteric.py         # Stage 1: linear-in-T with PC prior
│   │   ├── glaciers.py             # Stage 2a: quadratic + volume depletion
│   │   ├── greenland.py            # Stage 2b: SMB + discharge two-component
│   │   ├── wais.py                 # Stage 4 (stub for Paper C; full in Paper B)
│   │   ├── eais.py                 # Stage 4 (stub)
│   │   ├── apis.py                 # Stage 4 (stub)
│   │   ├── tws.py                  # Stage 3 (stub)
│   │   └── residual.py             # Budget residual (absorbs unresolved components)
│   ├── inadequacy/
│   │   ├── scalar_discrepancy.py   # Phase 1: sigma_extra per component
│   │   └── gp_discrepancy.py       # Phase 2: GP delta_i(t) (stub)
│   ├── inference/
│   │   ├── joint_model.py          # Full joint log-posterior
│   │   ├── sampler.py              # MCMC sampler (NUTS via NumPyro or PyMC)
│   │   └── convergence.py          # R-hat, ESS, trace diagnostics
│   ├── projections/
│   │   ├── project.py              # Forward projection from posterior
│   │   └── variance_decomposition.py  # Nested LTV decomposition
│   ├── interventions/
│   │   ├── do_calculus.py          # Graph surgery for counterfactuals
│   │   └── scenarios.py            # SAI, CDR, Thwaites stabilization specs
│   └── utils/
│       ├── constants.py
│       └── plotting.py
├── data/                           # Same data as Papers A and B
│   ├── gmsl_reconstructions/
│   ├── gmst_products/
│   ├── fair/
│   ├── thermosteric/
│   ├── glaciers/
│   ├── imbie/
│   ├── frederikse/
│   ├── altimetry/
│   └── facts/
├── paper/
│   ├── figures_main.py             # 8 main figures
│   ├── figures_supplement.py       # ~7 supplementary figures
│   └── tables.py
├── tests/
│   ├── test_dag.py
│   ├── test_budget.py
│   ├── test_components.py
│   ├── test_inference.py
│   ├── test_projections.py
│   └── test_interventions.py
└── run_paper.py                    # Master script for paper results
```

---

## Step 1: Core infrastructure

### File: `gmslr_framework/core/dag.py`

```python
@dataclass
class DAGNode:
    name: str
    symbol: str
    node_type: str      # 'forcing', 'mediating', 'component_rate', 
                        # 'constraint', 'deep_uncertainty', 'observation'
    parents: list[str]
    description: str

@dataclass  
class DAGEdge:
    source: str
    target: str
    edge_type: str      # 'causal', 'deterministic', 'observation', 'dashed_phase2'
    justification: str

class FrameworkDAG:
    """
    Full DAG specification for the hierarchical SLR framework.
    Supports:
    - Validation (no cycles, all nodes reachable)
    - d-separation queries
    - Intervention (graph surgery): delete incoming edges to intervened node
    - CI relation extraction
    """
    
    def __init__(self):
        self.nodes = {}    # name -> DAGNode
        self.edges = []    # list of DAGEdge
    
    def add_node(self, node: DAGNode) -> None: ...
    def add_edge(self, edge: DAGEdge) -> None: ...
    def validate(self) -> list[str]: ...            # returns list of warnings
    def d_separated(self, x, y, given) -> bool: ... # d-separation query
    def intervene(self, node_name, value) -> 'FrameworkDAG':
        """Graph surgery: return modified DAG with incoming edges to node_name removed."""
    def get_ci_relations(self) -> list[tuple]: ...   # implied CI relations
    
    def build_default(self) -> None:
        """Construct the 7-component framework DAG with all nodes and edges."""
        # Forcing nodes
        self.add_node(DAGNode('SSP', 'SSP', 'forcing', [], 'Emission scenario'))
        self.add_node(DAGNode('F', 'F(t)', 'forcing', ['SSP'], 'Radiative forcing'))
        self.add_node(DAGNode('T_GMST', 'T(t)', 'forcing', ['F'], 'Global mean temperature'))
        
        # Mediating nodes
        self.add_node(DAGNode('T_NA', 'T_NA(t)', 'mediating', ['T_GMST'], 'North Atlantic SST'))
        self.add_node(DAGNode('E', 'E(t)', 'mediating', [], 'ENSO index'))  # exogenous Phase 1
        self.add_node(DAGNode('T_CDW', 'T_CDW(t)', 'mediating', ['E'], 'CDW temperature'))
        
        # Component rate nodes
        self.add_node(DAGNode('r_th', 'r_th(t)', 'component_rate', ['T_GMST'], 'Thermosteric'))
        self.add_node(DAGNode('r_gl', 'r_gl(t)', 'component_rate', ['T_GMST'], 'Glaciers'))
        self.add_node(DAGNode('r_SMB', 'r_SMB(t)', 'component_rate', ['T_GMST'], 'GrIS SMB'))
        self.add_node(DAGNode('r_dis', 'r_dis(t)', 'component_rate', ['T_NA'], 'GrIS discharge'))
        self.add_node(DAGNode('r_WAIS', 'r_WAIS(t)', 'component_rate', ['T_CDW'], 'WAIS'))
        self.add_node(DAGNode('r_EAIS', 'r_EAIS(t)', 'component_rate', ['T_GMST'], 'EAIS'))
        self.add_node(DAGNode('r_APIS', 'r_APIS(t)', 'component_rate', ['T_GMST'], 'APIS'))
        self.add_node(DAGNode('r_TWS_s', 'r_TWS_s(t)', 'component_rate', [], 'TWS secular'))
        self.add_node(DAGNode('r_TWS_c', 'r_TWS_c(t)', 'component_rate', ['E'], 'TWS climate'))
        
        # Budget constraint node (deterministic)
        all_rates = ['r_th','r_gl','r_SMB','r_dis','r_WAIS','r_EAIS','r_APIS','r_TWS_s','r_TWS_c']
        self.add_node(DAGNode('r_total', 'r_total(t)', 'constraint', all_rates, 'Total rate'))
        
        # ... observation nodes, deep uncertainty nodes, edges
```

### File: `gmslr_framework/core/budget_constraint.py`

```python
def budget_log_likelihood(
    component_rates: dict[str, np.ndarray],  # {component_name: rate_array(T,)}
    obs_total_rate: np.ndarray,               # (T,)
    obs_total_unc: np.ndarray,                # (T,)
    sigma_extra: dict[str, float],            # {component_name: sigma_extra_i}
) -> float:
    """
    Compute log-likelihood of the budget constraint.
    
    sigma_budget^2(t) = sigma_obs^2(t) + sum_i sigma_extra_i^2
    
    Note: sigma_budget changes at each MCMC step because sigma_extra_i are
    sampled parameters. This function must be called inside the MCMC loop.
    """
    total_model = sum(component_rates.values())  # sum over components
    sigma_budget_sq = obs_total_unc**2 + sum(s**2 for s in sigma_extra.values())
    residuals = total_model - obs_total_rate
    return -0.5 * np.sum(residuals**2 / sigma_budget_sq)
```

---

## Step 2: Component models

### File: `gmslr_framework/components/thermosteric.py`

```python
class ThermostericModel:
    """
    rate_thermo(t) = a_thermo * T(t)^2 + b_thermo * T(t) + c_thermo
    
    Priors:
        a_thermo ~ Exponential(lambda_a)    # PC prior toward zero
        b_thermo ~ Normal(mu_b, sigma_b)    # weakly informative
        c_thermo ~ Normal(0, sigma_c)       # intercept
    """
    
    def __init__(self, pc_prior_lambda: float = 1.0):
        self.pc_prior_lambda = pc_prior_lambda
    
    def rate(self, T: np.ndarray, params: dict) -> np.ndarray:
        return params['a_thermo'] * T**2 + params['b_thermo'] * T + params['c_thermo']
    
    def log_prior(self, params: dict) -> float:
        # PC prior on a_thermo: exponential toward zero
        # Normal priors on b_thermo, c_thermo
        ...
    
    def get_prior_specs(self) -> dict:
        """Return prior specifications for MCMC setup."""
        ...
```

### File: `gmslr_framework/components/glaciers.py`

```python
class GlacierModel:
    """
    rate_gl(t) = (a_gl * T^2 + b_gl * T + c_gl) * [V_gl(t) / V_gl_0]^p
    
    Volume depletion: V_gl(t) = V_gl_0 - integral_0^t rate_gl(tau) d_tau
    This is an ODE that must be solved forward in time.
    V_gl_0 = 0.32 m SLE (total glacier reservoir).
    
    Priors:
        a_gl ~ Normal(0, sigma_a)
        b_gl ~ Normal(mu_b, sigma_b)  # informed by GlaMBIE
        c_gl ~ Normal(0, sigma_c)
        p ~ LogNormal(mu_p, sigma_p)  # from Marzeion/Edwards
    """
    
    def rate(self, T: np.ndarray, params: dict, dt: float = 1.0) -> np.ndarray:
        """Forward integration with volume depletion."""
        V = params.get('V_gl_0', 320.0)  # mm SLE
        rates = np.zeros_like(T)
        for i, t in enumerate(T):
            uncorrected = params['a_gl'] * t**2 + params['b_gl'] * t + params['c_gl']
            rates[i] = uncorrected * (V / params.get('V_gl_0', 320.0))**params['p']
            V -= rates[i] * dt
            V = max(V, 0.0)  # volume cannot go negative
        return rates
```

### File: `gmslr_framework/components/greenland.py`

```python
class GreenlandModel:
    """
    Two-component model:
    rate_GrIS(t) = [b_SMB * T(t) + c_SMB] + [beta_dis * T_NA(t) + c_dis]
    
    SMB responds to GMST. Discharge responds to North Atlantic ocean temperature.
    T_NA may be latent (pre-Argo): parameterized as gamma_NA * T_GMST + epsilon_NA.
    """
    
    def rate(self, T_GMST, T_NA, params):
        smb = params['b_SMB'] * T_GMST + params['c_SMB']
        discharge = params['beta_dis'] * T_NA + params['c_dis']
        return smb + discharge
```

### File: `gmslr_framework/components/residual.py`

```python
class ResidualComponent:
    """
    At any stage, unresolved components are in the residual:
    rate_residual(t) = rate_total(t) - sum(resolved component rates)
    
    The residual has no parametric model — it is diagnosed from the
    budget constraint. Its uncertainty is inherited from the total
    and the resolved components.
    """
    
    def compute(self, total_rate, resolved_rates):
        return total_rate - sum(resolved_rates.values())
```

---

## Step 3: Model inadequacy

### File: `gmslr_framework/inadequacy/scalar_discrepancy.py`

```python
class ScalarDiscrepancy:
    """
    Phase 1 model inadequacy: sigma_extra_i per component.
    
    Augments the observation variance:
    sigma_i^2(t) = sigma_obs_i^2(t) + sigma_extra_i^2
    
    Prior: sigma_extra_i ~ HalfCauchy(0, s_i)
    where s_i is scaled to expected magnitude of unmodeled variability.
    """
    
    def __init__(self, component_name: str, scale: float):
        self.component_name = component_name
        self.scale = scale
    
    def log_prior(self, sigma_extra: float) -> float:
        # HalfCauchy(0, scale)
        return -np.log(1 + (sigma_extra / self.scale)**2) - np.log(np.pi * self.scale / 2)
    
    def augmented_variance(self, sigma_obs: np.ndarray, sigma_extra: float) -> np.ndarray:
        return sigma_obs**2 + sigma_extra**2
```

---

## Step 4: Joint model and inference

### File: `gmslr_framework/inference/joint_model.py`

```python
class JointModel:
    """
    The full joint log-posterior for the hierarchical framework.
    
    log p(theta | D) propto 
        sum_i log p(theta_i)                     # component priors
      + sum_i log p(sigma_extra_i)                # inadequacy priors  
      + sum_i log p(D_i | theta_i, sigma_extra_i) # component likelihoods
      + log p(H_obs | {rate_i(theta_i)}, sigma_budget)  # budget constraint
    
    where sigma_budget depends on {sigma_extra_i} (coupled).
    """
    
    def __init__(self, stage: int = 2):
        """
        stage=0: total GMSL only (DOLS)
        stage=1: + thermosteric
        stage=2: + glaciers + Greenland
        """
        self.stage = stage
        self.components = {}     # name -> ComponentModel
        self.discrepancies = {}  # name -> ScalarDiscrepancy
        self.data = {}           # name -> observational data
        self._setup_stage(stage)
    
    def _setup_stage(self, stage):
        if stage >= 1:
            self.components['thermo'] = ThermostericModel()
            self.discrepancies['thermo'] = ScalarDiscrepancy('thermo', scale=0.5)
        if stage >= 2:
            self.components['glaciers'] = GlacierModel()
            self.components['greenland'] = GreenlandModel()
            self.discrepancies['glaciers'] = ScalarDiscrepancy('glaciers', scale=0.3)
            self.discrepancies['greenland'] = ScalarDiscrepancy('greenland', scale=0.5)
    
    def log_posterior(self, params: dict) -> float:
        """Evaluate the full joint log-posterior at params."""
        lp = 0.0
        
        # Component priors
        for name, model in self.components.items():
            lp += model.log_prior(params[name])
        
        # Inadequacy priors
        for name, disc in self.discrepancies.items():
            lp += disc.log_prior(params[f'sigma_extra_{name}'])
        
        # Component likelihoods (against component-specific data)
        component_rates = {}
        for name, model in self.components.items():
            rate = model.rate(self.data['T_GMST'], params[name])
            component_rates[name] = rate
            sigma_aug = self.discrepancies[name].augmented_variance(
                self.data[f'sigma_obs_{name}'], params[f'sigma_extra_{name}']
            )
            lp += -0.5 * np.sum((self.data[f'rate_obs_{name}'] - rate)**2 / sigma_aug)
        
        # Budget constraint
        lp += budget_log_likelihood(
            component_rates, 
            self.data['rate_total_obs'],
            self.data['sigma_total_obs'],
            {n: params[f'sigma_extra_{n}'] for n in self.discrepancies}
        )
        
        return lp
```

### File: `gmslr_framework/inference/sampler.py`

```python
def run_mcmc(
    model: JointModel,
    n_warmup: int = 2000,
    n_samples: int = 5000,
    n_chains: int = 4,
    backend: str = 'numpyro',   # or 'pymc'
) -> MCMCResult:
    """
    Run NUTS MCMC on the joint model.
    
    If backend='numpyro':
        Use NumPyro's NUTS sampler with JAX compilation.
        Requires model.log_posterior to be JAX-compatible.
        
    If backend='pymc':
        Use PyMC's NUTS sampler.
        Requires model to be specified as a PyMC model.
    
    Returns MCMCResult with fields:
        samples: dict[str, np.ndarray]  # parameter samples (n_chains * n_samples, ...)
        log_likelihood: np.ndarray      # per-sample log-likelihood
        diagnostics: dict               # R-hat, ESS, divergences
    """
```

**Implementation choice**: NumPyro is strongly preferred for this problem. The joint model has ~15-25 parameters (depending on stage), moderate dimensionality for NUTS. JAX compilation provides 10-100x speedup over PyMC for this problem size. The main implementation effort is making the component models JAX-compatible (replace numpy with jax.numpy, ensure differentiability).

If JAX compatibility is too costly to achieve quickly, fall back to PyMC.

---

## Step 5: Projections and variance decomposition

### File: `gmslr_framework/projections/project.py`

```python
def project_from_posterior(
    posterior_samples: dict,     # from MCMC
    component_models: dict,     # {name: ComponentModel}
    fair_temps: dict,           # {ssp: temperature trajectory}
    projection_years: np.ndarray,
) -> dict:
    """
    For each posterior sample and each SSP:
    1. Evaluate each component rate at the projection temperatures.
    2. Sum components + residual to get total rate.
    3. Integrate to get cumulative GMSL.
    
    Returns dict keyed by SSP with:
        trajectories: np.ndarray (n_samples, n_years)
        component_trajectories: dict[str, np.ndarray]
        quantiles: dict
    """
```

### File: `gmslr_framework/projections/variance_decomposition.py`

```python
def variance_decomposition(
    component_trajectories: dict,  # {ssp: {component: np.ndarray(n_samples, n_years)}}
) -> dict:
    """
    Nested law of total variance:
    1. Within-scenario: component variance fractions f_i(t, S)
    2. Cross-component covariance C_cross(t, S)
    3. Across-scenario: Var_S[E[H|S]]
    4. R_sep diagnostic
    
    Returns dict with all decomposition arrays.
    """
```

---

## Step 6: Intervention engine

### File: `gmslr_framework/interventions/do_calculus.py`

```python
def do_intervention(
    dag: FrameworkDAG,
    posterior_samples: dict,
    component_models: dict,
    intervention: dict,            # {node_name: intervention_value_or_function}
    projection_years: np.ndarray,
    fair_temps_baseline: dict,
    fair_temps_intervened: dict,   # modified temperature trajectory
) -> dict:
    """
    Perform graph surgery and compute counterfactual projection.
    
    1. Modify the DAG: delete incoming edges to intervened nodes.
    2. Replace intervened node values with the intervention specification.
    3. Propagate through the modified DAG to get counterfactual component rates.
    4. Compute the difference: H_counterfactual(t) - H_baseline(t) = effect.
    
    Returns:
        baseline_projection: ProjectionResult
        counterfactual_projection: ProjectionResult
        effect: np.ndarray (n_samples, n_years)  # SLR avoided/added
        effect_quantiles: dict
    """
```

For Paper C, demonstrate with one or two interventions:

**SAI scenario**: Replace $F(t)$ with a modified forcing trajectory $F_{\text{SAI}}(t)$ that reduces radiative forcing by $\Delta F$ after 2040. Propagate through GMST → component rates → GMSL. Report the SLR difference.

**Thwaites stabilization** (conceptual): Replace $r_{\text{WAIS}}$ with a stabilized trajectory. At Stage 2, WAIS is in the residual, so this intervention operates on the residual directly. The effect on total GMSL is the integral of the difference in WAIS rates.

---

## Step 7: Figures for the paper

### Main figures (8)

1. **Hierarchical structure schematic** (4 levels, conceptual)
2. **DAG** (TikZ, full specification)
3. **Stage 1 results**: thermosteric rate fit + budget residual
4. **Stage 2 results**: glacier + GrIS fits + narrowed residual
5. **Projection comparison across stages** (Stage 0 → 1 → 2 uncertainty narrowing)
6. **Counterfactual: SAI scenario** (total GMSL with and without SAI)
7. **Variance decomposition** (7-component stacked area)
8. **Posterior correlation matrix** (7×7, showing budget-induced dependence)

### Supplementary figures (7)

- S1: Full DAG (larger)
- S2: Prior sensitivity
- S3: MCMC trace plots and convergence
- S4: Posterior predictive checks (per component)
- S5: Residual evolution Stage 0 → 1 → 2
- S6: sigma_extra posteriors
- S7: Durbin-Watson diagnostics

---

## Step 8: Validation tests

### `tests/test_dag.py`

```python
# Test: DAG has no cycles
# Test: All nodes reachable from forcing nodes
# Test: d-separation queries match hand-computed CI relations
# Test: Intervention removes correct edges
# Test: CI relations consistent with framework document
```

### `tests/test_budget.py`

```python
# Test: Budget log-likelihood on synthetic data recovers known answer
# Test: sigma_budget updates correctly when sigma_extra changes
# Test: Budget constraint with sigma_budget → infinity gives flat likelihood
```

### `tests/test_components.py`

```python
# Test: Thermosteric rate at T=0 gives c_thermo
# Test: Glacier volume depletion: cumulative loss cannot exceed V_gl_0
# Test: Greenland two-component: sum of SMB + discharge = total
# Test: Each component model is differentiable (JAX grad works)
```

### `tests/test_inference.py`

```python
# Test: Known-answer test on synthetic data with known posterior
#       (linear model + Gaussian prior + Gaussian likelihood → known Gaussian posterior)
# Test: R-hat < 1.01 for all parameters
# Test: Bulk ESS > 400 for all parameters
# Test: No divergences
# Test: Posterior predictive covers observed data (coverage > 0.80 for 90% CI)
```

### `tests/test_projections.py`

```python
# Test: Projection quantiles monotonic in SSP forcing
# Test: Stage 2 projection is tighter than Stage 1 (narrower CI at all times)
# Test: Variance decomposition sums to 1.0
# Test: R_sep diagnostic is computed and reported
```

### `tests/test_interventions.py`

```python
# Test: do(F=0) → all component rates go to their intercepts
# Test: Intervention effect is non-zero for SAI with Delta_F > 0
# Test: Intervention on a non-ancestor of a component has no effect
#       (e.g., do(T_NA = ...) does not affect r_thermo)
```

---

## Step 9: Master script

```python
"""
Paper C master script.

Steps:
1.  Load all data
2.  Build DAG and validate
3.  Set up joint model at Stage 1
4.  Run MCMC (Stage 1): thermosteric + budget constraint
5.  Diagnose convergence
6.  Posterior predictive checks (Stage 1)
7.  Compute Stage 1 projections
8.  Set up joint model at Stage 2
9.  Run MCMC (Stage 2): + glaciers + Greenland
10. Diagnose convergence
11. Posterior predictive checks (Stage 2)
12. Compute Stage 2 projections
13. Compare Stage 0 → 1 → 2 projections
14. Run intervention analysis (SAI scenario)
15. Variance decomposition
16. Generate all figures (8 main + 7 supplement)
17. Generate all tables (4 main + 4 supplement)
18. Export projection data
19. Run all tests
20. Summary report
"""
```

---

## Estimated compute time

- Data loading: ~3 min
- DAG construction and validation: ~1 min
- MCMC Stage 1 (4 chains × 7000 samples): ~30-60 min (NumPyro), ~2-4 hr (PyMC)
- MCMC Stage 2 (4 chains × 7000 samples): ~60-120 min (NumPyro), ~4-8 hr (PyMC)
- Projections: ~10 min
- Intervention analysis: ~5 min
- Variance decomposition: ~5 min
- Figures: ~10 min
- Tests: ~10 min
- **Total: ~2-4 hours (NumPyro) or ~6-12 hours (PyMC)**

NumPyro is strongly recommended. The MCMC is the bottleneck and JAX compilation provides order-of-magnitude speedup.

---

## Critical decision points

1. **MCMC backend**: NumPyro vs PyMC. NumPyro is faster but requires JAX-compatible model code (no Python control flow that depends on parameter values in differentiable regions). The glacier volume-depletion ODE may need special handling (use `jax.lax.scan` instead of Python for-loop). If this is too complex, use PyMC for initial development and port to NumPyro later.

2. **Stage 2 identifiability**: With thermosteric, glaciers, and Greenland all responding to GMST, the budget constraint may not be sufficient to separate them pre-satellite. The identifiability depends on the component-specific datasets (Argo, GlaMBIE, IMBIE GrIS). If the posterior is multimodal or has strong parameter correlations, add informative priors from process models (RACMO for GrIS SMB, GlacierMIP for glacier sensitivity).

3. **North Atlantic SST for GrIS discharge**: Pre-Argo, T_NA is poorly observed. Options: (a) treat as latent and parameterize from GMST with large uncertainty, (b) use EN4 or similar reanalysis, (c) fit a simple linear T_NA = gamma * T_GMST + noise model and sample gamma. Option (c) is recommended for Phase 1.

4. **If MCMC does not converge**: Simplify. Remove the quadratic term from glaciers (set a_gl = 0). Remove the discharge component from Greenland (fit total GrIS only). These simplifications reduce the parameter count and improve identifiability. They can be relaxed once the simpler model converges.

5. **Scope for paper**: The paper must demonstrate Stages 1-2 with working MCMC. Stage 0 (DOLS) is already done (Paper A). Stages 3-4 are described as architecture with planned implementation. If Stage 2 MCMC proves too difficult within the timeline, fall back to Stage 1 only (thermosteric decomposition). Stage 1 alone, with the budget constraint, model inadequacy, DAG, and intervention engine, is sufficient for a framework paper.

---

## Dependencies on Papers A and B

- Paper A provides the Stage 0 result (DOLS projections) and the IPCC sensitivity comparison. Paper C cites this and uses the DOLS calibration as the Level 1 constraint.
- Paper B provides the budget constraint demonstration and the WAIS mixture model. Paper C cites Paper B for the WAIS treatment and incorporates the mixture model as the Level 3 (A4) module for WAIS.
- Paper C provides the architecture that unifies Papers A and B. It is the "theory" paper; Papers A and B are the "empirical" papers.

---

## Output checklist

- [ ] Framework code (gmslr_framework package, documented, tested)
- [ ] 8 main figures
- [ ] 7 supplementary figures
- [ ] 4 main tables + 4 supplementary tables
- [ ] MCMC convergence diagnostics (R-hat, ESS, trace plots)
- [ ] Posterior predictive checks for all components
- [ ] Projection tables (Stages 0-2, all SSPs, decadal)
- [ ] Counterfactual projection under SAI scenario
- [ ] Variance decomposition
- [ ] All test suites passing (6 test files)
- [ ] Framework code on GitHub
- [ ] README with installation and usage instructions
