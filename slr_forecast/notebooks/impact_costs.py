"""Coastal adaptation cost and flood damage functions quoted in the paper.

The main text converts the gap between the fast-WAIS and slow-WAIS 95th
percentiles into an annual adaptation cost and an annual flood damage
(scripts/slow_wais_p90_numbers.py); the supplement gives the derivation
(sections "Adaptation cost" and "Flood damages").

Adaptation cost. Tiggeloven et al. (2020, doi:10.5194/nhess-20-1025-2020)
estimate that 3.4% of the global coastline (~1.6M km) warrants dike
heightening under a cost-benefit optimality criterion, at a unit cost of
$7M per km per meter of dike height (New Orleans; Bos 2008). That gives
54,400 km x $7M/km/m ~ $400B per meter of rise as a one-time capital cost
for dike raising alone. A conservative 10x multiplier for the full societal
adaptation burden (seawalls, nature-based solutions, managed retreat,
infrastructure relocation) across a broader fraction of the coastline gives
$4T/m. The capital cost is annualized with the capital recovery factor
CRF = r (1 + r)^n / ((1 + r)^n - 1) at the real discount rate r over the
planning horizon n.

Flood damages. damage_vec is the Jevrejeva et al. (2018) annual global flood
cost without further adaptation ($B/yr), piecewise linear in GMSL rise (m);
see slr_forecast.readers.impacts.slr_cost_jevrejeva2018.
"""
from slr_forecast.readers.impacts import slr_cost_jevrejeva2018 as damage_vec  # noqa: F401

ADAPT_CAPITAL_PER_M = 4000  # $B total capital per meter of SLR
DR = 0.03                   # real discount rate
HORIZON = 75                # planning horizon (years)

# Annualize capital cost using the capital recovery factor:
#   CRF = r(1+r)^n / ((1+r)^n - 1)
_CRF = DR * (1 + DR)**HORIZON / ((1 + DR)**HORIZON - 1)
ADAPT_COST_PER_M = ADAPT_CAPITAL_PER_M * _CRF  # $B/yr per meter of SLR
