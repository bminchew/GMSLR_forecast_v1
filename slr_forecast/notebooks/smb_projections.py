"""SMB projection module for ice sheet components.

Projects surface mass balance contributions to sea level using
physically-grounded sensitivities from the literature rather than
fitting polynomial models to noisy temperature data.

Physical basis: Cuffey & Paterson (2010) Eq. 4.8:
    Δb_n = C_T · ΔT + C_P · ΔP + ...

For ice sheets, the SMB response to warming is the sum of:
- Increased melt (C_T < 0, dominates in ablation zone)
- Increased precipitation via Clausius-Clapeyron (C_P > 0, interior)

References
----------
- Cuffey & Paterson (2010), The Physics of Glaciers, 4th ed., Eq. 4.8-4.10
- Fettweis et al. (2013), The Cryosphere, 7, 469-489 (GrIS SMB sensitivity)
- Gregory & Huybrechts (2006), Phil. Trans. R. Soc. A, 364, 1709-1731
- Frieler et al. (2015), The Cryosphere, 9, 1039-1062 (Antarctic accumulation)
- Ligtenberg et al. (2013), Climate Dynamics, 41, 3283-3299 (Antarctic SMB)
"""

import numpy as np
from dataclasses import dataclass
from typing import Optional


@dataclass
class SMBSensitivity:
    """SMB sensitivity parameters for a single ice sheet component.

    SMB_rate(T) = C_T · ΔT + C_T2 · ΔT² + SMB_0
    where ΔT is temperature anomaly relative to baseline.

    Units: Gt/yr for rates, °C for temperature.
    Convert to m SLE via / 362500.
    """
    C_T: float          # linear sensitivity (Gt/yr/°C), negative = mass loss
    C_T_sigma: float    # uncertainty on C_T (Gt/yr/°C)
    C_T2: float = 0.0   # quadratic sensitivity (Gt/yr/°C²), for ablation zone expansion
    C_T2_sigma: float = 0.0
    SMB_0: float = 0.0  # baseline SMB rate anomaly (Gt/yr) at ΔT = 0
    reference: str = ''  # citation
    temperature_frame: str = 'GMST'  # 'GMST' or 'local'
    AA_factor: float = 1.0  # Arctic/Antarctic amplification (GMST → local)


# ── Published sensitivities ──

# Greenland SMB sensitivity — RCM-derived with inter-RCM structural uncertainty.
# Central: mean across MAR (~100-150 Gt/yr/°C local) and RACMO (~100 Gt/yr/°C local),
# converted to GMST via AA~2.0 → ~200 Gt/yr/°C GMST.
# Structural uncertainty: factor ~1.8 between RACMO and MAR/HIRHAM (Glaude et al. 2024).
# Quadratic: sensitivity doubles between current warming and +3°C (Noël 2021, Sellevold 2020).
# Validated: GRACE − D implied SMB gives data-driven C_T = 36 ± 15 Gt/yr/°C local T
# (observational lower bound; R² = 0.18 from high interannual variability).
GREENLAND_SMB = SMBSensitivity(
    C_T=-200.0,          # Gt/yr per °C GMST (RCM mean ~110 local × AA~2.0)
    C_T_sigma=80.0,      # 1σ: inter-RCM spread + parametric
    C_T2=-50.0,          # Gt/yr per °C² GMST (ablation zone expansion)
    C_T2_sigma=30.0,
    SMB_0=0.0,           # anomaly at baseline T (by definition)
    reference='Hanna et al. (2021); Fettweis et al. (2013); Glaude et al. (2024)',
    temperature_frame='GMST',
    AA_factor=1.0,       # sensitivity already in GMST frame
)

EAIS_SMB = SMBSensitivity(
    C_T=60.0,            # Gt/yr per °C GMST — positive = mass GAIN (more snowfall)
    C_T_sigma=20.0,      # Clausius-Clapeyron: ~5%/°C of ~1200 Gt/yr accumulation
    C_T2=0.0,
    C_T2_sigma=0.0,
    SMB_0=0.0,
    reference='Frieler et al. (2015); Ligtenberg et al. (2013)',
    temperature_frame='GMST',
    AA_factor=1.0,
)

PENINSULA_SMB = SMBSensitivity(
    C_T=-15.0,           # Gt/yr per °C GMST — small net loss (melt > accumulation gain)
    C_T_sigma=10.0,
    C_T2=0.0,
    C_T2_sigma=0.0,
    SMB_0=0.0,
    reference='Estimated from IMBIE trend and RACMO sensitivity',
    temperature_frame='GMST',
    AA_factor=1.0,
)


GT_TO_M_SLE = 1.0 / 362500.0  # 1 Gt = 1/362500 m SLE


def project_smb_ensemble(
    sensitivity: SMBSensitivity,
    T_proj: dict,
    time_proj: np.ndarray,
    T_baseline: float = 0.0,
    n_samples: int = 2000,
    seed: Optional[int] = None,
    volume_cap_m: Optional[float] = None,
    baseline_year: Optional[float] = None,
) -> dict:
    """Project SMB contribution to SLE under multiple SSP scenarios.

    Parameters
    ----------
    sensitivity : SMBSensitivity
        Published sensitivity parameters.
    T_proj : dict
        ``{ssp_name: ndarray}`` — annual temperature anomaly (°C)
        relative to the project baseline (1995–2005).
        Must be in the frame specified by sensitivity.temperature_frame.
    time_proj : ndarray
        Annual projection times (decimal years).
    T_baseline : float
        Temperature at the baseline epoch (should be ~0 if T_proj is
        relative to 1995–2005).
    n_samples : int
        Number of MC samples for uncertainty propagation.
    seed : int or None
    volume_cap_m : float or None
        If provided, cap cumulative SLE at this value (meters).
        For glaciers/EAIS where mass gain is bounded.
    baseline_year : float or None
        If provided, rebase cumulative SLE so that it is zero at this
        year.  If None (default), rebases to the year where dT ≈ 0.

    Returns
    -------
    dict
        ``{ssp_name: {'samples': (n_samples, n_times),
                       'median': ..., 'p5': ..., 'p17': ...,
                       'p83': ..., 'p95': ...,
                       'rate_median': ...}}``
    """
    rng = np.random.default_rng(seed)
    n_times = len(time_proj)
    dt = np.diff(time_proj, prepend=time_proj[0] - 1.0)

    # Draw sensitivity parameters
    C_T_draws = rng.normal(sensitivity.C_T, sensitivity.C_T_sigma, n_samples)
    C_T2_draws = rng.normal(sensitivity.C_T2, sensitivity.C_T2_sigma, n_samples)

    results = {}

    for ssp_name, T_ssp in T_proj.items():
        # Temperature anomaly relative to baseline
        dT = T_ssp - T_baseline

        # SMB rate for each sample: (n_samples, n_times)
        # rate_i(t) = C_T_i · dT(t) + C_T2_i · dT(t)² + SMB_0
        rates = (C_T_draws[:, None] * dT[None, :]
                 + C_T2_draws[:, None] * dT[None, :] ** 2
                 + sensitivity.SMB_0)

        # Convert Gt/yr → m SLE/yr
        rates_m = rates * GT_TO_M_SLE

        # Cumulative SLE (trapezoidal integration)
        # Convention: positive rate = positive SLR contribution
        # For SMB: negative SMB (mass loss) → positive SLR
        # rates are in Gt/yr where negative = mass loss
        # We want SLR contribution: -SMB_rate * GT_TO_M_SLE
        slr_rates = -rates_m  # flip sign: mass loss → positive SLR

        cumulative = np.cumsum(slr_rates * dt[None, :], axis=1)

        # Rebase to baseline epoch
        if baseline_year is not None:
            baseline_idx = np.argmin(np.abs(time_proj - baseline_year))
        else:
            baseline_idx = np.argmin(np.abs(dT))
        cumulative -= cumulative[:, baseline_idx:baseline_idx + 1]

        # Apply volume cap if specified
        if volume_cap_m is not None:
            cumulative = np.minimum(cumulative, volume_cap_m)

        results[ssp_name] = {
            'samples': cumulative,
            'median': np.median(cumulative, axis=0),
            'p5': np.percentile(cumulative, 5, axis=0),
            'p17': np.percentile(cumulative, 17, axis=0),
            'p83': np.percentile(cumulative, 83, axis=0),
            'p95': np.percentile(cumulative, 95, axis=0),
            'rate_median': np.median(slr_rates, axis=0),
            'times': time_proj,
        }

    return results


def project_smb_at_warming_levels(
    sensitivity: SMBSensitivity,
    warming_levels: np.ndarray,
    n_samples: int = 2000,
    seed: Optional[int] = None,
) -> dict:
    """Project SMB rate at specific warming levels (for comparison tables).

    Parameters
    ----------
    sensitivity : SMBSensitivity
    warming_levels : ndarray
        GMST anomalies in °C (e.g., [1.5, 2.0, 3.0, 4.0]).
    n_samples : int
    seed : int or None

    Returns
    -------
    dict with keys:
        'warming_levels' : ndarray
        'rate_median' : ndarray (Gt/yr)
        'rate_p5', 'rate_p95' : ndarray
        'slr_rate_median' : ndarray (mm/yr SLE)
    """
    rng = np.random.default_rng(seed)
    C_T_draws = rng.normal(sensitivity.C_T, sensitivity.C_T_sigma, n_samples)
    C_T2_draws = rng.normal(sensitivity.C_T2, sensitivity.C_T2_sigma, n_samples)

    rates = np.zeros((n_samples, len(warming_levels)))
    for j, dT in enumerate(warming_levels):
        rates[:, j] = C_T_draws * dT + C_T2_draws * dT ** 2 + sensitivity.SMB_0

    return {
        'warming_levels': warming_levels,
        'rate_median': np.median(rates, axis=0),
        'rate_p5': np.percentile(rates, 5, axis=0),
        'rate_p95': np.percentile(rates, 95, axis=0),
        'slr_rate_median': np.median(-rates * GT_TO_M_SLE * 1000, axis=0),
    }
