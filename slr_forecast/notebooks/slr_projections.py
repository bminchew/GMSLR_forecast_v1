import numpy as np
import pandas as pd
from typing import Optional

'''
Various tools to project SLR

# Single projection with best-fit coefficients
# Rate model: rate = 0.1*T² + 3.0*T + 1.5 (mm/yr)
coeffs = np.array([0.1, 3.0, 1.5])
gmsl_projection = project_gmsl_from_temperature(
    coefficients=coeffs,
    temperature_projection=df_ipcc_temperature_ssp245,
    baseline_year=2000.0,
    baseline_gmsl=0.0,
    temp_col='temperature',
    temp_sigma_col='temperature_sigma'  # optional
)

# Ensemble projection with coefficient uncertainty
coeffs = np.array([0.1, 3.0, 1.5])
cov = np.array([
    [0.01**2, 0, 0],
    [0, 0.3**2, 0],
    [0, 0, 0.2**2]
])  # Covariance matrix

scenarios = {
    'SSP1-2.6': df_ssp126,
    'SSP2-4.5': df_ssp245,
    'SSP3-7.0': df_ssp370,
    'SSP5-8.5': df_ssp585
}

results = project_gmsl_ensemble(
    coefficients=coeffs,
    coefficients_cov=cov,
    temperature_projections=scenarios,
    baseline_year=2000.0,
    n_samples=1000
)

# Access results
df_ssp245_gmsl = results['scenarios']['SSP2-4.5']

# With SAOD (for hindcasting with observed volcanic forcing):
results = project_gmsl_ensemble(
    coefficients=coeffs,
    coefficients_cov=cov,
    temperature_projections={'observed': df_temp_obs},
    baseline_year=2005.0,
    gamma_saod=result.gamma_saod,
    saod_scenario={'observed': saod_series},
)
'''

def datetime_to_decimal_year(dt):
    """Convert a datetime object to a decimal year."""
    year = dt.year
    start = pd.Timestamp(year=year, month=1, day=1)
    end = pd.Timestamp(year=year + 1, month=1, day=1)
    return year + (dt - start).total_seconds() / (end - start).total_seconds()


def project_gmsl_from_temperature(
    coefficients: np.ndarray,
    temperature_projection: pd.DataFrame,
    baseline_year: float = 2000.0,
    baseline_gmsl: float = 0.0,
    temp_col: str = 'temperature',
    temp_sigma_col: str = None,
    gamma_saod: Optional[float] = None,
    saod_series: Optional[pd.Series] = None,
) -> pd.DataFrame:
    """
    Project GMSL from temperature using a polynomial rate-temperature relationship.

    The model assumes:
        dGMSL/dt = (1/2)(d²α/dT²)T² + α₀T + c  [+ γ × SAOD]

    where α=d(dGMSLR/dt)/DT, T is temperature anomaly, and the coefficients
    describe how the rate of sea level rise depends on temperature.

    Parameters
    ----------
    coefficients : np.ndarray
        Polynomial coefficients in numpy convention [a, b, c] where:
        - a = dα/dT : Quadratic term (rate acceleration with temperature)
        - b = α₀ : Linear sensitivity (units/yr per °C at T=0)
        - c = constant : Background rate at T=0
        Rate model: rate = a*T² + b*T + c

    temperature_projection : pd.DataFrame
        DataFrame with datetime index containing temperature projections.
        Must have column specified by `temp_col`.

    baseline_year : float, default 2000.0
        Reference year where GMSL = baseline_gmsl

    baseline_gmsl : float, default 0.0
        GMSL value at baseline_year (in same units as coefficients)

    temp_col : str, default 'temperature'
        Column name for temperature values

    temp_sigma_col : str, optional
        Column name for temperature uncertainty. If provided, propagates
        uncertainty to GMSL projection.

    gamma_saod : float, optional
        SAOD coefficient from ``calibrate_dols(..., saod=...)``.  When
        provided together with *saod_series*, the rate is augmented by
        ``gamma_saod × SAOD(t)``.  For future projections without
        predicted volcanic forcing, omit or set to None — the SAOD term
        contributes nothing (equivalent to SAOD = 0).

    saod_series : pd.Series, optional
        Stratospheric aerosol optical depth time series with datetime
        index.  Required when *gamma_saod* is not None.  The series is
        reindexed to the temperature projection dates; missing values are
        filled with 0 (background / no eruption).

    Returns
    -------
    pd.DataFrame
        DataFrame with datetime index containing:
        - temperature: Input temperature projection
        - rate: Instantaneous rate of GMSL rise (units/yr)
        - gmsl: Cumulative GMSL (integrated from baseline)
        - decimal_year: Time in decimal years

        If temp_sigma_col provided, also includes:
        - temperature_sigma: Input uncertainty
        - rate_sigma: Rate uncertainty from temperature uncertainty
        - gmsl_sigma: Cumulative GMSL uncertainty

    Example
    -------
    >>> coeffs = np.array([0.1, 3.0, 1.5])
    >>> gmsl = project_gmsl_from_temperature(coeffs, df_temp_projection)

    Notes
    -----
    Integration is performed using the trapezoidal rule:
        GMSL(t) = GMSL(t₀) + ∫[t₀ to t] rate(τ) dτ

    For uncertainty propagation, we use:
        σ_rate = |drate/dT| × σ_T = |2aT + b| × σ_T
        σ_GMSL is accumulated in quadrature over the integration

    For future projections, SAOD is assumed zero (no predicted eruptions)
    unless an explicit *saod_series* is provided (e.g. for hindcasting
    with observed volcanic forcing).
    """
    a, b, c = coefficients

    # Extract temperature
    T = temperature_projection[temp_col].values

    # Get time in decimal years
    if 'decimal_year' in temperature_projection.columns:
        time_years = temperature_projection['decimal_year'].values
    else:
        time_years = np.array([
            datetime_to_decimal_year(t) for t in temperature_projection.index
        ])

    # Compute instantaneous rate: rate = a*T² + b*T + c [+ gamma*SAOD]
    rate = a * T**2 + b * T + c

    # Add SAOD contribution if provided
    if gamma_saod is not None and saod_series is not None:
        # Normalize SAOD index to month-start for alignment
        saod_ms = saod_series.copy()
        saod_ms.index = saod_ms.index.to_period('M').to_timestamp()
        saod_ms = saod_ms[~saod_ms.index.duplicated(keep='first')]

        # Align to temperature projection dates (fill missing with 0)
        temp_idx_ms = temperature_projection.index.to_period('M').to_timestamp()
        saod_aligned = saod_ms.reindex(temp_idx_ms, fill_value=0.0).values
        rate = rate + gamma_saod * saod_aligned

    # Integrate to get cumulative GMSL using trapezoidal rule
    # Find baseline index
    baseline_idx = np.argmin(np.abs(time_years - baseline_year))

    # Time steps in years
    dt = np.diff(time_years)

    # Integrate forward and backward from baseline
    gmsl = np.zeros_like(rate)
    gmsl[baseline_idx] = baseline_gmsl

    # Forward integration (baseline to end)
    for i in range(baseline_idx, len(gmsl) - 1):
        gmsl[i + 1] = gmsl[i] + 0.5 * (rate[i] + rate[i + 1]) * dt[i]

    # Backward integration (baseline to start)
    for i in range(baseline_idx, 0, -1):
        gmsl[i - 1] = gmsl[i] - 0.5 * (rate[i] + rate[i - 1]) * dt[i - 1]

    # Build output DataFrame
    result = pd.DataFrame({
        'temperature': T,
        'rate': rate,
        'gmsl': gmsl,
        'decimal_year': time_years
    }, index=temperature_projection.index)
    result.index.name = 'time'

    # Uncertainty propagation if temperature uncertainty provided
    if temp_sigma_col is not None and temp_sigma_col in temperature_projection.columns:
        T_sigma = temperature_projection[temp_sigma_col].values

        # Rate uncertainty: σ_rate = |drate/dT| × σ_T = |2aT + b| × σ_T
        drate_dT = np.abs(2 * a * T + b)
        rate_sigma = drate_dT * T_sigma

        # GMSL uncertainty: accumulate in quadrature
        gmsl_sigma = np.zeros_like(rate)
        gmsl_sigma[baseline_idx] = 0.0

        # Forward propagation
        for i in range(baseline_idx, len(gmsl) - 1):
            step_sigma = 0.5 * np.sqrt(rate_sigma[i]**2 + rate_sigma[i + 1]**2) * dt[i]
            gmsl_sigma[i + 1] = np.sqrt(gmsl_sigma[i]**2 + step_sigma**2)

        # Backward propagation
        for i in range(baseline_idx, 0, -1):
            step_sigma = 0.5 * np.sqrt(rate_sigma[i]**2 + rate_sigma[i - 1]**2) * dt[i - 1]
            gmsl_sigma[i - 1] = np.sqrt(gmsl_sigma[i]**2 + step_sigma**2)

        result['temperature_sigma'] = T_sigma
        result['rate_sigma'] = rate_sigma
        result['gmsl_sigma'] = gmsl_sigma

    return result


def project_gmsl_ensemble(
    coefficients: np.ndarray,
    coefficients_cov: np.ndarray,
    temperature_projections: dict,
    baseline_year: float = 2000.0,
    baseline_gmsl: float = 0.0,
    n_samples: int = 1000,
    temp_col: str = 'temperature',
    seed: int = None,
    gamma_saod: Optional[float] = None,
    saod_scenarios: Optional[dict] = None,
    posterior_samples: Optional[np.ndarray] = None,
) -> dict:
    """
    Project GMSL ensemble from multiple temperature scenarios with
    coefficient uncertainty.

    Parameters
    ----------
    coefficients : np.ndarray
        Best-fit polynomial coefficients [a, b, c] where
        rate = a*T² + b*T + c  (physical, not regression).

    coefficients_cov : np.ndarray
        3×3 covariance matrix for coefficient uncertainty.

    temperature_projections : dict
        Dictionary of {scenario_name: DataFrame} with temperature
        projections.
        E.g., {'SSP1-2.6': df_ssp126, 'SSP2-4.5': df_ssp245, ...}

    baseline_year : float, default 2000.0
        Reference year where GMSL = baseline_gmsl

    baseline_gmsl : float, default 0.0
        GMSL value at baseline_year

    n_samples : int, default 1000
        Number of Monte Carlo samples for uncertainty

    temp_col : str, default 'temperature'
        Column name for temperature values

    seed : int, optional
        Random seed for reproducibility

    gamma_saod : float, optional
        SAOD coefficient from ``calibrate_dols(..., saod=...)``.
        When provided together with *saod_scenarios*, the rate is
        augmented by ``gamma_saod × SAOD(t)`` for each scenario.
        For future SSP projections (no predicted eruptions), omit this
        parameter — the SAOD term contributes nothing.

    saod_scenarios : dict, optional
        Dictionary of {scenario_name: pd.Series} with SAOD time series.
        Keys must match *temperature_projections*.  Missing scenarios
        are treated as SAOD = 0.  Missing time steps within a scenario
        are filled with 0 (background / no eruption).

    posterior_samples : np.ndarray, optional
        Pre-drawn coefficient samples, shape (n_posterior, n_coeffs).
        When provided, ``n_samples`` are drawn (with replacement) from
        these instead of from MVN(coefficients, coefficients_cov).
        This preserves non-Gaussianity (e.g., from half-normal priors
        in Bayesian rate-space calibration).  ``coefficients`` and
        ``coefficients_cov`` are still stored in the output metadata.

    Returns
    -------
    dict
        Dictionary with keys:
        - 'scenarios': dict of {scenario_name: DataFrame} with columns:
            - temperature, rate, gmsl (median)
            - rate_lower, rate_upper (5-95% CI)
            - gmsl_lower, gmsl_upper (5-95% CI)
        - 'coefficients': input coefficients
        - 'coefficients_cov': input covariance
        - 'n_samples': number of Monte Carlo samples

    Example
    -------
    >>> coeffs = np.array([0.1, 3.0, 1.5])
    >>> cov = np.diag([0.01, 0.1, 0.05])**2
    >>> scenarios = {'SSP2-4.5': df_ssp245, 'SSP5-8.5': df_ssp585}
    >>> results = project_gmsl_ensemble(coeffs, cov, scenarios)
    """
    rng = np.random.default_rng(seed)

    # Ensure covariance matrix is positive semi-definite
    cov = np.array(coefficients_cov, dtype=np.float64)
    eigvals = np.linalg.eigvalsh(cov)
    if np.any(eigvals < -1e-10 * np.max(np.abs(eigvals))):
        # Force PSD via nearest PSD projection
        eigvals_clip = np.maximum(eigvals, 0)
        eigvecs = np.linalg.eigh(cov)[1]
        cov = eigvecs @ np.diag(eigvals_clip) @ eigvecs.T
        cov = 0.5 * (cov + cov.T)  # enforce symmetry

    # Sample coefficients
    if posterior_samples is not None:
        # Draw from provided posterior samples (preserves non-Gaussianity)
        idx = rng.choice(len(posterior_samples), n_samples, replace=True)
        coeff_samples = posterior_samples[idx]
    else:
        # Draw from multivariate normal
        coeff_samples = rng.multivariate_normal(
            coefficients, cov, n_samples
        )

    results = {
        'scenarios': {},
        'coefficients': coefficients,
        'coefficients_cov': coefficients_cov,
        'n_samples': n_samples,
    }

    for scenario_name, temp_df in temperature_projections.items():
        T = temp_df[temp_col].values

        # Get time
        if 'decimal_year' in temp_df.columns:
            time_years = temp_df['decimal_year'].values
        else:
            time_years = np.array([
                datetime_to_decimal_year(t) for t in temp_df.index
            ])

        dt = np.diff(time_years)
        baseline_idx = np.argmin(np.abs(time_years - baseline_year))

        # Align SAOD if provided
        saod_vals = np.zeros(len(T))  # default: no volcanic forcing
        if gamma_saod is not None and saod_scenarios is not None:
            if scenario_name in saod_scenarios:
                s = saod_scenarios[scenario_name]
                s_ms = s.copy()
                s_ms.index = s_ms.index.to_period('M').to_timestamp()
                s_ms = s_ms[~s_ms.index.duplicated(keep='first')]
                t_idx_ms = temp_df.index.to_period('M').to_timestamp()
                saod_vals = s_ms.reindex(t_idx_ms, fill_value=0.0).values

        # Run ensemble
        rate_ensemble = np.zeros((n_samples, len(T)))
        gmsl_ensemble = np.zeros((n_samples, len(T)))

        for k, (a, b, c_coeff) in enumerate(coeff_samples):
            # Rate = polynomial + SAOD
            rate = a * T**2 + b * T + c_coeff
            if gamma_saod is not None:
                rate = rate + gamma_saod * saod_vals
            rate_ensemble[k] = rate

            # Integrate
            gmsl = np.zeros(len(T))
            gmsl[baseline_idx] = baseline_gmsl

            for i in range(baseline_idx, len(T) - 1):
                gmsl[i + 1] = (
                    gmsl[i]
                    + 0.5 * (rate_ensemble[k, i] + rate_ensemble[k, i + 1]) * dt[i]
                )
            for i in range(baseline_idx, 0, -1):
                gmsl[i - 1] = (
                    gmsl[i]
                    - 0.5 * (rate_ensemble[k, i] + rate_ensemble[k, i - 1]) * dt[i - 1]
                )

            gmsl_ensemble[k] = gmsl

        # Compute percentiles
        result_df = pd.DataFrame(
            {
                'temperature': T,
                'rate': np.median(rate_ensemble, axis=0),
                'rate_lower': np.percentile(rate_ensemble, 5, axis=0),
                'rate_upper': np.percentile(rate_ensemble, 95, axis=0),
                'gmsl': np.median(gmsl_ensemble, axis=0),
                'gmsl_lower': np.percentile(gmsl_ensemble, 5, axis=0),
                'gmsl_upper': np.percentile(gmsl_ensemble, 95, axis=0),
                'decimal_year': time_years,
            },
            index=temp_df.index,
        )
        result_df.index.name = 'time'

        results['scenarios'][scenario_name] = result_df

    return results


# ====================================================================
# Rate-and-State Projection Functions
# ====================================================================

def _solve_state_ode_projection(temperature, time_years, tau, S0=None):
    """Solve dS/dt = (T - S)/τ for projection (standalone, no import needed).

    Duplicates the analytical exponential integrator from bayesian_dols.py
    to keep slr_projections.py self-contained.
    """
    n = len(temperature)
    S = np.empty(n)
    if tau < 0.01:
        return temperature.copy()
    S[0] = temperature[0] if S0 is None else S0
    for i in range(n - 1):
        dt = time_years[i + 1] - time_years[i]
        T_avg = 0.5 * (temperature[i] + temperature[i + 1])
        decay = np.exp(-dt / tau)
        S[i + 1] = T_avg + (S[i] - T_avg) * decay
    return S


def project_gmsl_state_ensemble(
    coefficients: np.ndarray,
    coefficients_cov: np.ndarray,
    tau_samples: np.ndarray,
    temperature_projections: dict,
    historical_temperature: np.ndarray,
    historical_time: np.ndarray,
    baseline_year: float = 2000.0,
    baseline_gmsl: float = 0.0,
    n_samples: int = 1000,
    temp_col: str = 'temperature',
    seed: int = None,
    posterior_samples: Optional[np.ndarray] = None,
) -> dict:
    """Project GMSL ensemble using the rate-and-state model.

    For each Monte Carlo sample, solves the state-variable ODE and
    integrates rate = a·T² + b·T + c + d·(S−T) to obtain cumulative
    GMSL.  Historical temperature is prepended to each projection so
    the state variable S(t) has the correct initial condition.

    Parameters
    ----------
    coefficients : np.ndarray, shape (4,)
        Best-fit [a, b, c, d].
    coefficients_cov : np.ndarray, shape (4, 4)
        Covariance matrix.
    tau_samples : np.ndarray, shape (n_posterior,)
        Posterior samples of the relaxation time τ (years).
    temperature_projections : dict
        {scenario_name: DataFrame} with column ``temp_col``.
    historical_temperature : np.ndarray
        Monthly historical temperature used during calibration, for
        ODE spin-up before the projection starts.
    historical_time : np.ndarray
        Decimal years matching ``historical_temperature``.
    baseline_year, baseline_gmsl, n_samples, temp_col, seed :
        Same as ``project_gmsl_ensemble()``.
    posterior_samples : np.ndarray, optional
        Joint posterior samples, shape (n_posterior, 4) for [a, b, c, d].
        When provided, ``n_samples`` are drawn (with replacement) from
        these + ``tau_samples`` jointly.

    Returns
    -------
    dict  (same structure as ``project_gmsl_ensemble()``)
    """
    rng = np.random.default_rng(seed)

    # Sample coefficients and τ jointly
    if posterior_samples is not None:
        n_post = len(posterior_samples)
        idx = rng.choice(n_post, n_samples, replace=True)
        coeff_samp = posterior_samples[idx]      # (n_samples, 4)
        tau_samp = tau_samples[idx]              # (n_samples,)
    else:
        # MVN coefficients + independent τ draws
        cov = np.array(coefficients_cov, dtype=np.float64)
        eigvals = np.linalg.eigvalsh(cov)
        if np.any(eigvals < -1e-10 * np.max(np.abs(eigvals))):
            eigvals_clip = np.maximum(eigvals, 0)
            eigvecs = np.linalg.eigh(cov)[1]
            cov = eigvecs @ np.diag(eigvals_clip) @ eigvecs.T
            cov = 0.5 * (cov + cov.T)
        coeff_samp = rng.multivariate_normal(coefficients, cov, n_samples)
        tau_samp = rng.choice(tau_samples, n_samples, replace=True)

    results = {
        'scenarios': {},
        'coefficients': coefficients,
        'coefficients_cov': coefficients_cov,
        'n_samples': n_samples,
    }

    for scenario_name, temp_df in temperature_projections.items():
        T_proj = temp_df[temp_col].values
        if 'decimal_year' in temp_df.columns:
            time_proj = temp_df['decimal_year'].values
        else:
            time_proj = np.array([
                datetime_to_decimal_year(t) for t in temp_df.index
            ])

        # Prepend historical temperature for ODE spin-up
        # Trim historical to avoid overlap: keep only before projection start
        t_proj_start = time_proj[0]
        hist_mask = historical_time < t_proj_start
        T_full = np.concatenate([historical_temperature[hist_mask], T_proj])
        time_full = np.concatenate([historical_time[hist_mask], time_proj])
        n_hist = hist_mask.sum()
        n_proj = len(T_proj)

        dt_proj = np.diff(time_proj)
        baseline_idx = np.argmin(np.abs(time_proj - baseline_year))

        # Ensemble arrays (projection portion only)
        rate_ensemble = np.zeros((n_samples, n_proj))
        gmsl_ensemble = np.zeros((n_samples, n_proj))

        for k in range(n_samples):
            a_k, b_k, c_k, d_k = coeff_samp[k]
            tau_k = tau_samp[k]

            # Solve ODE on full (historical + projection) timeline
            S_full = _solve_state_ode_projection(T_full, time_full, tau_k)

            # Extract projection portion
            S_proj = S_full[n_hist:]
            diseq = S_proj - T_proj

            # Rate = a·T² + b·T + c + d·(S − T)
            rate = a_k * T_proj**2 + b_k * T_proj + c_k + d_k * diseq
            rate_ensemble[k] = rate

            # Integrate from baseline
            gmsl = np.zeros(n_proj)
            gmsl[baseline_idx] = baseline_gmsl
            for i in range(baseline_idx, n_proj - 1):
                gmsl[i + 1] = gmsl[i] + 0.5 * (rate[i] + rate[i + 1]) * dt_proj[i]
            for i in range(baseline_idx, 0, -1):
                gmsl[i - 1] = gmsl[i] - 0.5 * (rate[i] + rate[i - 1]) * dt_proj[i - 1]
            gmsl_ensemble[k] = gmsl

        result_df = pd.DataFrame({
            'temperature': T_proj,
            'rate': np.median(rate_ensemble, axis=0),
            'rate_lower': np.percentile(rate_ensemble, 5, axis=0),
            'rate_upper': np.percentile(rate_ensemble, 95, axis=0),
            'gmsl': np.median(gmsl_ensemble, axis=0),
            'gmsl_lower': np.percentile(gmsl_ensemble, 5, axis=0),
            'gmsl_upper': np.percentile(gmsl_ensemble, 95, axis=0),
            'decimal_year': time_proj,
        }, index=temp_df.index)
        result_df.index.name = 'time'
        results['scenarios'][scenario_name] = result_df

    return results


def project_gmsl_tau_sensitivity(
    coefficients: np.ndarray,
    coefficients_cov: np.ndarray,
    tau_values: np.ndarray,
    temperature_projections: dict,
    historical_temperature: np.ndarray,
    historical_time: np.ndarray,
    baseline_year: float = 2000.0,
    baseline_gmsl: float = 0.0,
    n_samples: int = 500,
    temp_col: str = 'temperature',
    seed: int = None,
    posterior_samples: Optional[np.ndarray] = None,
    d_value: float = None,
) -> dict:
    """Test sensitivity of GMSL projections to fixed values of τ.

    Runs ``project_gmsl_state_ensemble`` for each fixed τ in
    ``tau_values``, sampling only (a, b, c, d) from the posterior.
    This isolates the effect of the relaxation timescale on projections.

    Parameters
    ----------
    coefficients : np.ndarray, shape (4,)
        Best-fit [a, b, c, d].
    coefficients_cov : np.ndarray, shape (4, 4)
        Covariance matrix for [a, b, c, d].
    tau_values : np.ndarray
        Array of fixed τ values (years) to test, e.g.
        [1, 10, 30, 50, 100, 200, 500].
    temperature_projections : dict
        {scenario_name: DataFrame} with temperature projections.
    historical_temperature : np.ndarray
        Monthly historical temperature for ODE spin-up.
    historical_time : np.ndarray
        Decimal years matching historical_temperature.
    baseline_year, baseline_gmsl, n_samples, temp_col, seed :
        Same as ``project_gmsl_state_ensemble()``.
    posterior_samples : np.ndarray, optional
        Joint [a, b, c, d] posterior samples.
    d_value : float, optional
        If provided, override d with this fixed value for all samples
        (useful for testing d=0 → instantaneous limit).

    Returns
    -------
    dict
        Keys are τ values (float), values are dicts with same structure
        as ``project_gmsl_state_ensemble()`` output.
        Also includes a 'summary' key with a DataFrame of GMSL at
        key years (2050, 2100, 2150) for each τ × scenario.
    """
    rng = np.random.default_rng(seed)

    # Sample coefficients once (reuse across all τ values)
    if posterior_samples is not None:
        n_post = len(posterior_samples)
        idx = rng.choice(n_post, n_samples, replace=True)
        coeff_samp = posterior_samples[idx].copy()
    else:
        cov = np.array(coefficients_cov, dtype=np.float64)
        eigvals = np.linalg.eigvalsh(cov)
        if np.any(eigvals < -1e-10 * np.max(np.abs(eigvals))):
            eigvals_clip = np.maximum(eigvals, 0)
            eigvecs = np.linalg.eigh(cov)[1]
            cov = eigvecs @ np.diag(eigvals_clip) @ eigvecs.T
            cov = 0.5 * (cov + cov.T)
        coeff_samp = rng.multivariate_normal(coefficients, cov, n_samples)

    if d_value is not None:
        coeff_samp[:, 3] = d_value

    all_results = {}
    summary_rows = []

    for tau_val in tau_values:
        # Create constant τ array
        tau_fixed = np.full(n_samples, tau_val)

        result = project_gmsl_state_ensemble(
            coefficients=coefficients,
            coefficients_cov=coefficients_cov,
            tau_samples=tau_fixed,
            temperature_projections=temperature_projections,
            historical_temperature=historical_temperature,
            historical_time=historical_time,
            baseline_year=baseline_year,
            baseline_gmsl=baseline_gmsl,
            n_samples=n_samples,
            temp_col=temp_col,
            seed=None,  # use pre-sampled coefficients
            posterior_samples=coeff_samp,
        )
        all_results[tau_val] = result

        # Extract summary at key years
        for ssp_name, df_ssp in result['scenarios'].items():
            for yr in [2050, 2100, 2150]:
                idx_yr = np.argmin(np.abs(df_ssp['decimal_year'].values - yr))
                if abs(df_ssp['decimal_year'].values[idx_yr] - yr) < 2.0:
                    summary_rows.append({
                        'tau': tau_val,
                        'scenario': ssp_name,
                        'year': yr,
                        'gmsl_median': df_ssp['gmsl'].iloc[idx_yr],
                        'gmsl_lower': df_ssp['gmsl_lower'].iloc[idx_yr],
                        'gmsl_upper': df_ssp['gmsl_upper'].iloc[idx_yr],
                    })

    all_results['summary'] = pd.DataFrame(summary_rows)
    return all_results


# ====================================================================
# Physically-Motivated Thermosteric Projection
# ====================================================================

def project_thermosteric_ensemble(
    posterior_samples: np.ndarray,
    tau_u_samples: np.ndarray,
    tau_d_samples: Optional[np.ndarray],
    temperature_projections: dict,
    historical_temperature: np.ndarray,
    historical_time: np.ndarray,
    baseline_year: float = 2005.0,
    baseline_steric: float = 0.0,
    n_samples: int = 2000,
    n_layers: int = 1,
    temp_col: str = 'temperature',
    seed: Optional[int] = None,
) -> dict:
    """Project thermosteric sea level using the physically-motivated model.

    For each MC sample, solves the two-layer cascade ODE on the combined
    historical + projection temperature trajectory, then evaluates:

        η(t) = a·S_u² + b_u·S_u [+ b_d·S_d] + c·(t − t₀) + H₀

    No integration is needed — the model predicts sea level directly.

    Parameters
    ----------
    posterior_samples : (n_post, n_phys)
        Physical coefficients: [a, b, c] (1-layer) or [a, b_u, b_d, c] (2-layer).
    tau_u_samples : (n_post,)
        Upper-ocean relaxation time samples (years).
    tau_d_samples : (n_post,) or None
        Deep-ocean relaxation time samples.  None for single-layer.
    temperature_projections : dict
        {scenario_name: DataFrame} with columns [temp_col, 'decimal_year'].
    historical_temperature : (n_hist,)
        Monthly historical GMST (°C) for ODE spin-up.
    historical_time : (n_hist,)
        Monthly decimal years for historical period.
    baseline_year : float
        Year at which projected steric = baseline_steric.
    baseline_steric : float
        Steric sea level at baseline year (meters).
    n_samples : int
        Number of MC samples to draw from posterior.
    n_layers : {1, 2}
    temp_col : str
        Column name for temperature in projection DataFrames.
    seed : int or None

    Returns
    -------
    dict with keys:
        'scenarios': {ssp_name: DataFrame with 'decimal_year', 'steric',
                      'steric_lower', 'steric_upper', 'temperature'}
        'n_samples': int
        'n_layers': int
    """
    from bayesian_dols import solve_twolayer_ode

    rng = np.random.default_rng(seed)
    n_post = len(posterior_samples)
    idx = rng.choice(n_post, size=min(n_samples, n_post), replace=False)

    results = {'scenarios': {}, 'n_samples': n_samples, 'n_layers': n_layers}

    for ssp_name, df_proj in temperature_projections.items():
        T_proj = df_proj[temp_col].values
        time_proj = df_proj['decimal_year'].values
        n_proj = len(T_proj)

        # Combine historical + projection for ODE spin-up
        T_full = np.concatenate([historical_temperature, T_proj])
        time_full = np.concatenate([historical_time, time_proj])
        n_hist = len(historical_temperature)

        # Time offset from start of historical record
        t0 = historical_time[0]

        # Find baseline index in projection
        baseline_idx = np.argmin(np.abs(time_proj - baseline_year))

        steric_ens = np.zeros((len(idx), n_proj))

        for ii, k in enumerate(idx):
            if n_layers == 1:
                a_k, b_k, c_k = posterior_samples[k, :3]
                tau_u_k = tau_u_samples[k]
                tau_d_k = np.inf
            else:
                a_k, bu_k, bd_k, c_k = posterior_samples[k, :4]
                tau_u_k = tau_u_samples[k]
                tau_d_k = tau_d_samples[k] if tau_d_samples is not None else np.inf

            # Solve ODE on full timeline
            S_u_full, S_d_full = solve_twolayer_ode(
                T_full, time_full, tau_u_k, tau_d_k
            )

            # Extract projection portion
            S_u_proj = S_u_full[n_hist:]
            S_d_proj = S_d_full[n_hist:]
            I0_proj = time_proj - t0

            # Evaluate level model
            if n_layers == 1:
                eta = a_k * S_u_proj**2 + b_k * S_u_proj + c_k * I0_proj
            else:
                eta = (a_k * S_u_proj**2 + bu_k * S_u_proj
                       + bd_k * S_d_proj + c_k * I0_proj)

            # Rebase to baseline
            eta -= eta[baseline_idx]
            eta += baseline_steric

            steric_ens[ii] = eta

        # Summary statistics
        median = np.median(steric_ens, axis=0)
        p5 = np.percentile(steric_ens, 5, axis=0)
        p95 = np.percentile(steric_ens, 95, axis=0)
        p17 = np.percentile(steric_ens, 17, axis=0)
        p83 = np.percentile(steric_ens, 83, axis=0)

        results['scenarios'][ssp_name] = pd.DataFrame({
            'decimal_year': time_proj,
            'temperature': T_proj,
            'steric': median,
            'steric_lower': p5,
            'steric_upper': p95,
            'steric_p17': p17,
            'steric_p83': p83,
        })

    return results


# ════════════════════════════════════════════════════════════════════
#  Greenland physics-informed projection
# ════════════════════════════════════════════════════════════════════

def project_greenland_ensemble(
    posterior_samples: np.ndarray,
    tau_dyn_samples: np.ndarray,
    H0_samples: np.ndarray,
    temperature_projections: dict,
    historical_temperature: np.ndarray,
    historical_time: np.ndarray,
    baseline_year: float = 2005.0,
    n_samples: int = 2000,
    temp_col: str = 'temperature',
    seed: Optional[int] = None,
) -> dict:
    """Project Greenland SLR using the physics-informed two-component model.

    For each MC sample, builds a combined historical + projection
    temperature timeline, computes:

        H_smb = a_eff·I₂ + b_eff·I₁               (pre-computed integrals)
        dD_eff/dt = (T − D_eff) / τ_dyn             (discharge ODE)
        H_dyn = γ · ∫₀ᵗ D_eff(τ) dτ                (cumulative discharge)
        H_gris = H_smb + H_dyn + c·(t − t₀) + H₀

    Returns decomposed projections (total, SMB, discharge) with
    uncertainty bands.

    Parameters
    ----------
    posterior_samples : (n_post, 4)
        Physical coefficients [a_eff, b_eff, γ, c].
    tau_dyn_samples : (n_post,)
        Ice-dynamic relaxation time (years).
    H0_samples : (n_post,)
        Baseline sea-level offset (meters).
    temperature_projections : dict
        {scenario_name: DataFrame} with columns [temp_col, 'decimal_year'].
        Monthly resolution recommended.
    historical_temperature : (n_hist,)
        Monthly historical GMST (°C) for ODE spin-up.
    historical_time : (n_hist,)
        Monthly decimal years for historical period.
    baseline_year : float
        Year at which projected SLR = 0.
    n_samples : int
        Number of MC samples to draw from posterior.
    temp_col : str
        Column name for temperature in projection DataFrames.
    seed : int or None

    Returns
    -------
    dict with keys:
        'scenarios': {ssp_name: DataFrame with 'decimal_year',
            'greenland', 'greenland_lower', 'greenland_upper',
            'greenland_smb', 'greenland_smb_lower', 'greenland_smb_upper',
            'greenland_dyn', 'greenland_dyn_lower', 'greenland_dyn_upper',
            'temperature'}
        'n_samples': int
    """
    from bayesian_dols import solve_twolayer_ode

    rng = np.random.default_rng(seed)
    n_post = len(posterior_samples)
    idx = rng.choice(n_post, size=min(n_samples, n_post), replace=False)

    results = {'scenarios': {}, 'n_samples': n_samples}

    for ssp_name, df_proj in temperature_projections.items():
        T_proj = df_proj[temp_col].values
        time_proj = df_proj['decimal_year'].values
        n_proj = len(T_proj)

        # Combine historical + projection for ODE spin-up
        T_full = np.concatenate([historical_temperature, T_proj])
        time_full = np.concatenate([historical_time, time_proj])
        n_hist = len(historical_temperature)
        t0 = historical_time[0]

        # ── Design vectors on full monthly grid (once per SSP) ──
        dt_full = np.diff(time_full)
        T2_mid = 0.5 * (T_full[:-1]**2 + T_full[1:]**2)
        T1_mid = 0.5 * (T_full[:-1] + T_full[1:])
        I2_full = np.concatenate([[0], np.cumsum(T2_mid * dt_full)])
        I1_full = np.concatenate([[0], np.cumsum(T1_mid * dt_full)])
        I0_full = time_full - t0

        # Extract projection portion
        I2_proj = I2_full[n_hist:]
        I1_proj = I1_full[n_hist:]
        I0_proj = I0_full[n_hist:]

        # Baseline index in projection
        baseline_idx = np.argmin(np.abs(time_proj - baseline_year))

        green_ens = np.zeros((len(idx), n_proj))
        smb_ens = np.zeros((len(idx), n_proj))
        dyn_ens = np.zeros((len(idx), n_proj))

        for ii, k in enumerate(idx):
            a_k, b_k, gamma_k, c_k = posterior_samples[k]
            tau_k = tau_dyn_samples[k]
            H0_k = H0_samples[k]

            # Discharge ODE on full timeline
            D_eff, _ = solve_twolayer_ode(T_full, time_full, tau_k, np.inf)

            # Cumulative ∫D_eff (vectorised trapezoidal)
            D_mid = 0.5 * (D_eff[:-1] + D_eff[1:])
            cum_D = np.concatenate([[0], np.cumsum(D_mid * dt_full)])
            cum_D_proj = cum_D[n_hist:]

            # Components
            H_smb = a_k * I2_proj + b_k * I1_proj
            H_dyn = gamma_k * cum_D_proj
            H_total = H_smb + H_dyn + c_k * I0_proj + H0_k

            # Rebase to baseline
            H_total -= H_total[baseline_idx]
            H_smb -= H_smb[baseline_idx]
            H_dyn -= H_dyn[baseline_idx]

            green_ens[ii] = H_total
            smb_ens[ii] = H_smb
            dyn_ens[ii] = H_dyn

        # Summary statistics
        results['scenarios'][ssp_name] = pd.DataFrame({
            'decimal_year': time_proj,
            'temperature': T_proj,
            'greenland': np.median(green_ens, axis=0),
            'greenland_lower': np.percentile(green_ens, 5, axis=0),
            'greenland_upper': np.percentile(green_ens, 95, axis=0),
            'greenland_p17': np.percentile(green_ens, 17, axis=0),
            'greenland_p83': np.percentile(green_ens, 83, axis=0),
            'greenland_smb': np.median(smb_ens, axis=0),
            'greenland_smb_lower': np.percentile(smb_ens, 5, axis=0),
            'greenland_smb_upper': np.percentile(smb_ens, 95, axis=0),
            'greenland_dyn': np.median(dyn_ens, axis=0),
            'greenland_dyn_lower': np.percentile(dyn_ens, 5, axis=0),
            'greenland_dyn_upper': np.percentile(dyn_ens, 95, axis=0),
        })

    return results