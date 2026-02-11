import numpy as np
import pandas as pd

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
'''

def project_gmsl_from_temperature(
    coefficients: np.ndarray,
    temperature_projection: pd.DataFrame,
    baseline_year: float = 2000.0,
    baseline_gmsl: float = 0.0,
    temp_col: str = 'temperature',
    temp_sigma_col: str = None
) -> pd.DataFrame:
    """
    Project GMSL from temperature using a quadratic GMLSRrate-temperature relationship.
    
    The model assumes:
        dGMSL/dt = (1/2)(d²α/dT²)T² + α₀T + c
    
    where α=d(dGMSLR/dt)/DT, T is temperature anomaly, and the coefficients describe how the
    rate of sea level rise depends on temperature.
    
    Parameters
    ----------
    coefficients : np.ndarray
        Quadratic coefficients in numpy convention [a, b, c] where:
        - a = (1/2)(d²α/dT²) : Quadratic term (rate acceleration with temperature)
        - b = α₀ : Linear sensitivity (mm/yr per °C at T=0)
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
    >>> # Coefficients from quadratic fit: rate = 0.1*T² + 3.0*T + 1.5 (mm/yr)
    >>> coeffs = np.array([0.1, 3.0, 1.5])
    >>> gmsl = project_gmsl_from_temperature(coeffs, df_temp_projection)
    
    Notes
    -----
    Integration is performed using the trapezoidal rule:
        GMSL(t) = GMSL(t₀) + ∫[t₀ to t] rate(τ) dτ
    
    For uncertainty propagation, we use:
        σ_rate = |drate/dT| × σ_T = |2aT + b| × σ_T
        σ_GMSL is accumulated in quadrature over the integration
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
    
    # Compute instantaneous rate: rate = a*T² + b*T + c
    rate = a * T**2 + b * T + c
    
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
        # Trapezoidal rule: integral += 0.5 * (rate[i] + rate[i+1]) * dt
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
            # Uncertainty in integral step
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
    seed: int = None
) -> dict:
    """
    Project GMSL ensemble from multiple temperature scenarios with coefficient uncertainty.
    
    Parameters
    ----------
    coefficients : np.ndarray
        Best-fit quadratic coefficients [a, b, c]
        
    coefficients_cov : np.ndarray
        3x3 covariance matrix for coefficient uncertainty
        
    temperature_projections : dict
        Dictionary of {scenario_name: DataFrame} with temperature projections.
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
    >>> cov = np.diag([0.01, 0.1, 0.05])**2  # Diagonal covariance
    >>> scenarios = {'SSP2-4.5': df_ssp245, 'SSP5-8.5': df_ssp585}
    >>> results = project_gmsl_ensemble(coeffs, cov, scenarios)
    """
    if seed is not None:
        np.random.seed(seed)
    
    # Sample coefficients from multivariate normal
    coeff_samples = np.random.multivariate_normal(coefficients, coefficients_cov, n_samples)
    
    results = {'scenarios': {}, 'coefficients': coefficients, 
               'coefficients_cov': coefficients_cov, 'n_samples': n_samples}
    
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
        
        # Run ensemble
        rate_ensemble = np.zeros((n_samples, len(T)))
        gmsl_ensemble = np.zeros((n_samples, len(T)))
        
        for k, (a, b, c) in enumerate(coeff_samples):
            # Rate
            rate_ensemble[k] = a * T**2 + b * T + c
            
            # Integrate
            gmsl = np.zeros(len(T))
            gmsl[baseline_idx] = baseline_gmsl
            
            for i in range(baseline_idx, len(T) - 1):
                gmsl[i + 1] = gmsl[i] + 0.5 * (rate_ensemble[k, i] + rate_ensemble[k, i + 1]) * dt[i]
            for i in range(baseline_idx, 0, -1):
                gmsl[i - 1] = gmsl[i] - 0.5 * (rate_ensemble[k, i] + rate_ensemble[k, i - 1]) * dt[i - 1]
            
            gmsl_ensemble[k] = gmsl
        
        # Compute percentiles
        result_df = pd.DataFrame({
            'temperature': T,
            'rate': np.median(rate_ensemble, axis=0),
            'rate_lower': np.percentile(rate_ensemble, 5, axis=0),
            'rate_upper': np.percentile(rate_ensemble, 95, axis=0),
            'gmsl': np.median(gmsl_ensemble, axis=0),
            'gmsl_lower': np.percentile(gmsl_ensemble, 5, axis=0),
            'gmsl_upper': np.percentile(gmsl_ensemble, 95, axis=0),
            'decimal_year': time_years
        }, index=temp_df.index)
        result_df.index.name = 'time'
        
        results['scenarios'][scenario_name] = result_df
    
    return results