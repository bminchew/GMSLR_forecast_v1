"""
Sea Level Rise Semi-Empirical Analysis
======================================

This module implements methods for:
1. Local polynomial regression for estimating rates and accelerations
2. Baseline alignment across datasets
3. Dynamic OLS (DOLS) calibration for climate sensitivity estimation

The semi-empirical framework follows:
    dH/dt = α × T(t) + ε

where:
    H = sea level (mm)
    T = temperature anomaly (°C or K)
    α = transient climate sensitivity (mm/yr/K)

References
----------
Rahmstorf, S. (2007). A Semi-Empirical Approach to Projecting Future 
    Sea-Level Rise. Science, 315(5810), 368-370.
    
Vermeer, M., & Rahmstorf, S. (2009). Global sea level linked to global 
    temperature. PNAS, 106(51), 21527-21532.

Stock, J. H., & Watson, M. W. (1993). A Simple Estimator of Cointegrating 
    Vectors in Higher Order Integrated Systems. Econometrica, 61(4), 783-820.
"""

import os
import warnings
from dataclasses import dataclass
from typing import Optional, Tuple, Dict, Union

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.interpolate import CubicSpline

# Plot style - use seaborn-v0_8-poster for publication-quality figures
import matplotlib.pyplot as plt
try:
    plt.style.use("seaborn-v0_8-poster")
except OSError:
    plt.style.use("seaborn-poster")  # Fallback for older matplotlib


# =============================================================================
# CONFIGURATION & DATA STRUCTURES
# =============================================================================

@dataclass
class KinematicsResult:
    """
    Container for local regression kinematics results.
    
    Attributes
    ----------
    rate : np.ndarray
        Rate of change (velocity) at each time point [value_units/year]
    rate_se : np.ndarray
        Standard error of rate estimates
    accel : np.ndarray
        Acceleration (second derivative) at each time point [value_units/year²]
    accel_se : np.ndarray
        Standard error of acceleration estimates
    n_effective : np.ndarray
        Effective number of observations used at each point
    time : np.ndarray
        Time values (decimal years)
    span_years : float
        Bandwidth used for the regression
    """
    rate: np.ndarray
    rate_se: np.ndarray
    accel: np.ndarray
    accel_se: np.ndarray
    n_effective: np.ndarray
    time: np.ndarray
    span_years: float
    
    def to_dataframe(self) -> pd.DataFrame:
        """Convert results to pandas DataFrame."""
        return pd.DataFrame({
            'rate': self.rate,
            'rate_se': self.rate_se,
            'accel': self.accel,
            'accel_se': self.accel_se,
            'n_effective': self.n_effective,
            'decimal_year': self.time
        }, index=pd.Index(self.time, name='time'))


@dataclass 
class DOLSResult:
    """
    Container for DOLS calibration results.
    
    Attributes
    ----------
    alpha : float
        Estimated climate sensitivity [mm/yr/K]
    alpha_se : float
        Standard error of alpha (HAC-robust)
    alpha_ci : Tuple[float, float]
        95% confidence interval for alpha
    equilibrium_temp : float
        Implied equilibrium temperature (intercept term)
    trend : float
        Background trend coefficient [mm/yr]
    model : sm.regression.linear_model.RegressionResultsWrapper
        Full statsmodels results object
    diagnostics : Dict
        Model diagnostics (R², residual stats, etc.)
    """
    alpha: float
    alpha_se: float
    alpha_ci: Tuple[float, float]
    equilibrium_temp: float
    trend: float
    model: object
    diagnostics: Dict


# =============================================================================
# SECTION 1: LOCAL POLYNOMIAL REGRESSION (KINEMATICS)
# =============================================================================

def compute_kinematics(
    time: np.ndarray,
    value: np.ndarray,
    sigma: np.ndarray,
    span_years: float,
    weights: Optional[np.ndarray] = None,
    min_effective_obs: int = 12,
    kernel: str = 'tricube'
) -> KinematicsResult:
    """
    Estimate rate and acceleration using kernel-weighted local polynomial regression.
    
    This function implements locally weighted scatterplot smoothing (LOWESS/LOESS)
    with a quadratic polynomial, providing estimates of the first and second 
    derivatives of the time series at each point.
    
    The local model at each target point t₀ is:
    
        y(t) = β₀ + β₁(t - t₀) + ½β₂(t - t₀)² + ε
    
    where β₁ = rate (dy/dt) and β₂ = acceleration (d²y/dt²).
    
    Weighting scheme:
        w_i = K((t_i - t₀)/h) × (1/σ_i²) × p_i
    
    where K is the kernel function, h is the bandwidth, σ_i is the measurement
    uncertainty, and p_i is an optional provenance weight.
    
    Parameters
    ----------
    time : np.ndarray
        Time values as decimal years. Must be monotonically increasing.
    value : np.ndarray
        Observed values (e.g., sea level in mm).
    sigma : np.ndarray
        1-sigma measurement uncertainty for each observation.
        Must be positive.
    span_years : float
        Bandwidth of the kernel in years. Points beyond this distance
        from the target receive zero weight (for compact kernels).
        Typical values: 20-30 years for climate applications.
    weights : np.ndarray, optional
        Additional weights (e.g., for interpolated data). Default is 1.0.
    min_effective_obs : int, default 12
        Minimum effective observations required for regression.
        For quadratic fit, recommend ≥10-12.
    kernel : str, default 'tricube'
        Kernel function: 'tricube', 'gaussian', or 'epanechnikov'.
    
    Returns
    -------
    KinematicsResult
        Dataclass containing rate, acceleration, standard errors, and metadata.
    
    Raises
    ------
    ValueError
        If inputs have inconsistent lengths or invalid values.
    
    Examples
    --------
    >>> result = compute_kinematics(
    ...     time=decimal_years,
    ...     value=gmsl_mm,
    ...     sigma=gmsl_unc,
    ...     span_years=30
    ... )
    >>> print(f"Current rate: {result.rate[-1]:.2f} ± {result.rate_se[-1]:.2f} mm/yr")
    
    Notes
    -----
    Edge effects: Near the boundaries, the kernel becomes asymmetric. The
    method naturally handles this, but estimates within span_years/2 of
    the edges have higher uncertainty and potential bias.
    
    References
    ----------
    Cleveland, W. S. (1979). Robust Locally Weighted Regression and Smoothing
        Scatterplots. JASA, 74(368), 829-836.
    """
    # =========================
    # Input validation
    # =========================
    time = np.asarray(time, dtype=np.float64)
    value = np.asarray(value, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    
    n = len(time)
    
    if not (len(value) == n and len(sigma) == n):
        raise ValueError(
            f"Inconsistent input lengths: time={n}, value={len(value)}, sigma={len(sigma)}"
        )
    
    if np.any(sigma <= 0):
        raise ValueError("All sigma values must be positive")
    
    if np.any(np.isnan(value)):
        warnings.warn(
            f"Input contains {np.sum(np.isnan(value))} NaN values. "
            "These points will be excluded from local fits."
        )
    
    if weights is None:
        weights = np.ones(n)
    else:
        weights = np.asarray(weights, dtype=np.float64)
        if len(weights) != n:
            raise ValueError(f"weights length ({len(weights)}) must match time length ({n})")
    
    # =========================
    # Kernel function selection
    # =========================
    kernel_functions = {
        'tricube': lambda u: np.where(np.abs(u) <= 1, (1 - np.abs(u)**3)**3, 0),
        'gaussian': lambda u: np.exp(-0.5 * u**2),
        'epanechnikov': lambda u: np.where(np.abs(u) <= 1, 0.75 * (1 - u**2), 0)
    }
    
    if kernel not in kernel_functions:
        raise ValueError(f"Unknown kernel '{kernel}'. Options: {list(kernel_functions.keys())}")
    
    kernel_func = kernel_functions[kernel]
    
    # =========================
    # Initialize output arrays
    # =========================
    rate = np.full(n, np.nan)
    rate_se = np.full(n, np.nan)
    accel = np.full(n, np.nan)
    accel_se = np.full(n, np.nan)
    n_effective = np.full(n, np.nan)
    
    h = span_years  # Bandwidth
    
    # =========================
    # Main regression loop
    # =========================
    for i in range(n):
        t0 = time[i]
        
        # Skip if target value is NaN
        if np.isnan(value[i]):
            continue
        
        # Normalized distance from target point
        u = (time - t0) / h
        
        # Kernel weights
        k_weights = kernel_func(u)
        
        # Combined weights: kernel × inverse-variance × provenance
        # The 1/σ² weighting is optimal for heteroscedastic data
        combined_weights = k_weights * (1.0 / sigma**2) * weights
        
        # Mask for valid observations (non-zero weight, non-NaN value)
        valid = (combined_weights > 1e-12) & (~np.isnan(value))
        
        # Effective sample size (accounts for varying weights)
        n_eff = np.sum(weights[valid])
        n_effective[i] = n_eff
        
        if n_eff < min_effective_obs:
            continue
        
        # Extract local data
        t_local = time[valid]
        y_local = value[valid]
        w_local = combined_weights[valid]
        
        # Center time at target for numerical stability
        dt = t_local - t0
        
        # Design matrix for quadratic: y = β₀ + β₁·dt + ½β₂·dt²
        # The ½ factor means β₂ directly equals the second derivative
        X = np.column_stack([
            np.ones(len(dt)),   # Intercept
            dt,                  # Linear (rate)
            0.5 * dt**2         # Quadratic (acceleration)
        ])
        
        # Weighted least squares
        try:
            model = sm.WLS(y_local, X, weights=w_local).fit()
            
            rate[i] = model.params[1]
            rate_se[i] = model.bse[1]
            accel[i] = model.params[2]
            accel_se[i] = model.bse[2]
            
        except (np.linalg.LinAlgError, ValueError):
            # Singular matrix or numerical issues - skip this point
            continue
    
    return KinematicsResult(
        rate=rate,
        rate_se=rate_se,
        accel=accel,
        accel_se=accel_se,
        n_effective=n_effective,
        time=time,
        span_years=span_years
    )


# =============================================================================
# SECTION 2: BASELINE ALIGNMENT
# =============================================================================

def align_to_baseline(
    df: pd.DataFrame,
    value_col: str,
    time_col: str = 'time',
    start_year: float = 1993,
    end_year: float = 2009,
    inclusive: str = 'left'
) -> pd.DataFrame:
    """
    Shift a dataset so its mean over a reference period is zero.
    
    Baseline alignment is essential for semi-empirical modeling to ensure
    physical consistency when combining multiple datasets or when the
    intercept has a meaningful interpretation (e.g., equilibrium temperature).
    
    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame. NOT modified in place.
    value_col : str
        Column name containing values to align.
    time_col : str, default 'time'
        Column name containing time values (decimal years).
    start_year : float, default 1993
        Start of reference period (inclusive).
    end_year : float, default 2009
        End of reference period. Interpretation depends on `inclusive`.
    inclusive : str, default 'left'
        How to handle boundaries:
        - 'left': start_year ≤ t < end_year (use end_year=2009 for 1993-2008)
        - 'both': start_year ≤ t ≤ end_year
        - 'right': start_year < t ≤ end_year
        - 'neither': start_year < t < end_year
    
    Returns
    -------
    pd.DataFrame
        New DataFrame with aligned values. Original is not modified.
    
    Raises
    ------
    ValueError
        If no data falls within the reference period.
    
    Examples
    --------
    >>> # Align to satellite era (1993-2008 inclusive)
    >>> df_aligned = align_to_baseline(
    ...     df, 'gmsl', 'decimal_year',
    ...     start_year=1993, end_year=2009, inclusive='left'
    ... )
    
    Notes
    -----
    The default reference period (1993-2008) corresponds to the early
    satellite altimetry era, commonly used for sea level studies.
    """
    # Create copy to avoid modifying original
    df = df.copy()
    
    # Build mask based on inclusivity
    t = df[time_col]
    
    if inclusive == 'left':
        mask = (t >= start_year) & (t < end_year)
    elif inclusive == 'both':
        mask = (t >= start_year) & (t <= end_year)
    elif inclusive == 'right':
        mask = (t > start_year) & (t <= end_year)
    elif inclusive == 'neither':
        mask = (t > start_year) & (t < end_year)
    else:
        raise ValueError(f"inclusive must be 'left', 'right', 'both', or 'neither', got '{inclusive}'")
    
    if mask.sum() == 0:
        raise ValueError(
            f"No data in reference period [{start_year}, {end_year}). "
            f"Data range: [{t.min():.1f}, {t.max():.1f}]"
        )
    
    # Compute and subtract baseline mean
    baseline_mean = df.loc[mask, value_col].mean()
    df[value_col] = df[value_col] - baseline_mean
    
    # Store alignment metadata
    df.attrs['baseline_period'] = (start_year, end_year)
    df.attrs['baseline_mean_removed'] = baseline_mean
    
    return df


# =============================================================================
# SECTION 3: DOLS CALIBRATION FOR CLIMATE SENSITIVITY
# =============================================================================

def calibrate_alpha_dols(
    time: np.ndarray,
    gmsl: np.ndarray,
    gmsl_sigma: np.ndarray,
    temperature: np.ndarray,
    n_lags: int = 2,
    include_trend: bool = True,
    hac_maxlags: int = 24
) -> DOLSResult:
    """
    Estimate climate sensitivity (α) using Dynamic Ordinary Least Squares.
    
    The semi-empirical sea level model relates sea level to integrated temperature:
    
        H(t) = H₀ + α × ∫T(τ)dτ + β×t + ε(t)
    
    where:
        H(t) = sea level at time t
        T(t) = temperature anomaly
        α = climate sensitivity [mm/yr/K]
        β = background trend (e.g., ongoing GIA adjustment)
    
    DOLS addresses non-stationarity and serial correlation by augmenting the
    regression with leads and lags of the first-differenced regressor:
    
        H(t) = α × ∫T(τ)dτ + β×t + Σᵢ γᵢ × ΔT(t+i) + ε(t)
    
    This yields a consistent and asymptotically efficient estimator for α
    even when H and ∫T are cointegrated I(1) processes.
    
    Parameters
    ----------
    time : np.ndarray
        Time values (decimal years, monthly resolution assumed).
    gmsl : np.ndarray
        Global mean sea level anomaly (mm).
    gmsl_sigma : np.ndarray
        Uncertainty in GMSL (mm). Used for WLS weighting.
    temperature : np.ndarray
        Global mean surface temperature anomaly (K or °C).
    n_lags : int, default 2
        Number of leads AND lags of ΔT to include.
        Total dynamic terms = 2 × n_lags + 1.
        For monthly data, n_lags=2 means ±2 months.
    include_trend : bool, default True
        Whether to include a linear time trend.
    hac_maxlags : int, default 24
        Maximum lags for HAC (Newey-West) standard error estimation.
        Default of 24 months accounts for ~2 years of autocorrelation.
    
    Returns
    -------
    DOLSResult
        Dataclass containing:
        - alpha: Climate sensitivity estimate
        - alpha_se: HAC-robust standard error
        - alpha_ci: 95% confidence interval
        - model: Full statsmodels results
        - diagnostics: Model fit statistics
    
    Raises
    ------
    ValueError
        If inputs have inconsistent lengths or insufficient valid observations.
    
    Examples
    --------
    >>> result = calibrate_alpha_dols(
    ...     time=decimal_years,
    ...     gmsl=gmsl_mm,
    ...     gmsl_sigma=gmsl_unc,
    ...     temperature=gmst_anomaly,
    ...     n_lags=2
    ... )
    >>> print(f"α = {result.alpha:.2f} ± {result.alpha_se:.2f} mm/yr/K")
    
    Notes
    -----
    Interpretation of α:
        α ≈ 3-4 mm/yr/K is typical for thermosteric + glacier response
        α ≈ 8-12 mm/yr/K when including ice sheet contributions
    
    The trend term β captures processes not related to temperature:
        - Glacial isostatic adjustment (GIA)
        - Long-term ice sheet dynamics
        - Groundwater depletion
    
    References
    ----------
    Stock, J. H., & Watson, M. W. (1993). A Simple Estimator of Cointegrating
        Vectors in Higher Order Integrated Systems. Econometrica, 61(4), 783-820.
        
    Vermeer, M., & Rahmstorf, S. (2009). Global sea level linked to global 
        temperature. PNAS, 106(51), 21527-21532.
    """
    # =========================
    # Input validation
    # =========================
    time = np.asarray(time, dtype=np.float64)
    gmsl = np.asarray(gmsl, dtype=np.float64)
    gmsl_sigma = np.asarray(gmsl_sigma, dtype=np.float64)
    temperature = np.asarray(temperature, dtype=np.float64)
    
    n = len(time)
    if not (len(gmsl) == n and len(gmsl_sigma) == n and len(temperature) == n):
        raise ValueError("All input arrays must have the same length")
    
    if np.any(gmsl_sigma <= 0):
        raise ValueError("All gmsl_sigma values must be positive")
    
    # =========================
    # Construct regressors
    # =========================
    
    # Time step (for numerical integration)
    dt = np.gradient(time)
    
    # Integrated temperature: ∫T(τ)dτ (cumulative trapezoidal)
    integrated_T = np.cumsum(temperature * dt)
    
    # First difference of temperature: ΔT
    delta_T = np.gradient(temperature)
    
    # Build design matrix
    regressors = {
        'intercept': np.ones(n),
        'integrated_T': integrated_T
    }
    
    if include_trend:
        # Center time to reduce collinearity with intercept
        regressors['trend'] = time - time.mean()
    
    # Add leads and lags of ΔT
    delta_T_series = pd.Series(delta_T)
    for lag in range(-n_lags, n_lags + 1):
        col_name = f'delta_T_lag{lag:+d}'
        regressors[col_name] = delta_T_series.shift(-lag).values  # Note: shift sign convention
    
    # Create design matrix
    X = pd.DataFrame(regressors)
    y = gmsl
    w = 1.0 / (gmsl_sigma ** 2)
    
    # Remove rows with NaN (from leads/lags at edges)
    valid_mask = X.notna().all(axis=1) & ~np.isnan(y) & ~np.isnan(w)
    
    n_valid = valid_mask.sum()
    n_dropped = n - n_valid
    
    if n_valid < len(regressors) + 10:
        raise ValueError(
            f"Insufficient valid observations ({n_valid}) for {len(regressors)} parameters. "
            f"Consider reducing n_lags."
        )
    
    if n_dropped > 0:
        warnings.warn(f"Dropped {n_dropped} observations due to NaN values (from leads/lags)")
    
    X_valid = X.loc[valid_mask].values
    y_valid = y[valid_mask]
    w_valid = w[valid_mask]
    
    # =========================
    # Fit DOLS model
    # =========================
    
    # Use HAC (Newey-West) standard errors to account for residual autocorrelation
    model = sm.WLS(y_valid, X_valid, weights=w_valid).fit(
        cov_type='HAC',
        cov_kwds={'maxlags': hac_maxlags}
    )
    
    # =========================
    # Extract results
    # =========================
    
    # Parameter indices
    idx_integrated_T = list(regressors.keys()).index('integrated_T')
    idx_intercept = list(regressors.keys()).index('intercept')
    idx_trend = list(regressors.keys()).index('trend') if include_trend else None
    
    alpha = model.params[idx_integrated_T]
    alpha_se = model.bse[idx_integrated_T]
    alpha_ci = model.conf_int()[idx_integrated_T]
    
    equilibrium_temp = model.params[idx_intercept]
    trend = model.params[idx_trend] if include_trend else 0.0
    
    # Diagnostics
    diagnostics = {
        'r_squared': model.rsquared,
        'r_squared_adj': model.rsquared_adj,
        'n_observations': n_valid,
        'n_parameters': len(regressors),
        'n_lags': n_lags,
        'hac_maxlags': hac_maxlags,
        'residual_std': np.std(model.resid),
        'durbin_watson': sm.stats.durbin_watson(model.resid),
        'aic': model.aic,
        'bic': model.bic
    }
    
    return DOLSResult(
        alpha=alpha,
        alpha_se=alpha_se,
        alpha_ci=tuple(alpha_ci),
        equilibrium_temp=equilibrium_temp,
        trend=trend,
        model=model,
        diagnostics=diagnostics
    )


# =============================================================================
# SECTION 4: CONVENIENCE FUNCTIONS
# =============================================================================

def compute_kinematics_multi_bandwidth(
    time: np.ndarray,
    value: np.ndarray,
    sigma: np.ndarray,
    bandwidths: list = [15, 20, 30, 50],
    **kwargs
) -> Dict[float, KinematicsResult]:
    """
    Compute kinematics for multiple bandwidths to assess sensitivity.
    
    Parameters
    ----------
    time, value, sigma : np.ndarray
        Input data (see compute_kinematics).
    bandwidths : list of float
        Bandwidths to evaluate (in years).
    **kwargs
        Additional arguments passed to compute_kinematics.
    
    Returns
    -------
    Dict[float, KinematicsResult]
        Dictionary mapping bandwidth to results.
    """
    results = {}
    for bw in bandwidths:
        results[bw] = compute_kinematics(time, value, sigma, span_years=bw, **kwargs)
    return results


def ensure_directories(base_path: str = "..") -> None:
    """Create standard project directory structure."""
    dirs = [
        f"{base_path}/data/raw",
        f"{base_path}/data/processed",
        f"{base_path}/figures",
        f"{base_path}/results"
    ]
    for d in dirs:
        os.makedirs(d, exist_ok=True)


# =============================================================================
# MODULE EXECUTION
# =============================================================================

if __name__ == "__main__":
    print("Sea Level Analysis Module")
    print("=" * 50)
    print("\nAvailable functions:")
    print("  - compute_kinematics(): Local polynomial regression")
    print("  - align_to_baseline(): Reference period alignment")
    print("  - calibrate_alpha_dols(): Climate sensitivity estimation")
    print("\nExample usage:")
    print("  from slr_semiempirical import compute_kinematics, calibrate_alpha_dols")
    print("  result = compute_kinematics(time, gmsl, sigma, span_years=30)")
