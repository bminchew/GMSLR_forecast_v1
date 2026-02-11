"""
Sea Level Rise Analysis Module
==============================

This module provides functions for analyzing global mean sea level (GMSL) and
global mean surface temperature (GMST) time series, including:

1. **Preprocessing**: Resampling, baseline alignment, uncertainty handling
2. **Kinematics**: Local polynomial regression for rates and accelerations
3. **Calibration**: Semi-empirical model fitting (DOLS) for climate sensitivity

Designed for peer-reviewed research with emphasis on:
- Rigorous uncertainty quantification
- Proper handling of mixed-resolution data
- Transparent methodology with detailed documentation

Dependencies
------------
- numpy
- pandas
- statsmodels

Example Workflow
----------------
>>> from slr_data_readers import read_frederikse2020, read_berkeley_earth
>>> from slr_analysis import (
...     align_to_baseline, resample_to_monthly,
...     compute_kinematics, calibrate_alpha_dols
... )
>>>
>>> # Load data
>>> df_sl = read_frederikse2020('data/frederikse2020.xlsx')
>>> df_temp = read_berkeley_earth('data/berkeley_earth.txt')
>>>
>>> # Align to common baseline
>>> df_sl = align_to_baseline(df_sl, 'gmsl', start_year=1993, end_year=2009)
>>>
>>> # Compute rates and accelerations
>>> result = compute_kinematics(time, gmsl, sigma, span_years=30)

References
----------
Cleveland, W. S. (1979). Robust Locally Weighted Regression and Smoothing
    Scatterplots. JASA, 74(368), 829-836.

Rahmstorf, S. (2007). A Semi-Empirical Approach to Projecting Future
    Sea-Level Rise. Science, 315(5810), 368-370.

Stock, J. H., & Watson, M. W. (1993). A Simple Estimator of Cointegrating
    Vectors in Higher Order Integrated Systems. Econometrica, 61(4), 783-820.
"""

import warnings
from dataclasses import dataclass
from datetime import datetime
from typing import Optional, Tuple, Dict, List, Union

import numpy as np
import pandas as pd
import statsmodels.api as sm


# =============================================================================
# DATA STRUCTURES
# =============================================================================

@dataclass
class KinematicsResult:
    """
    Container for local polynomial regression results.
    
    Attributes
    ----------
    rate : np.ndarray
        Rate of change (first derivative) at each time point.
        Units: [value_units]/year (e.g., mm/yr for sea level)
    rate_se : np.ndarray
        Standard error of rate estimates.
    accel : np.ndarray
        Acceleration (second derivative) at each time point.
        Units: [value_units]/year² (e.g., mm/yr² for sea level)
    accel_se : np.ndarray
        Standard error of acceleration estimates.
    n_effective : np.ndarray
        Effective number of observations used at each point,
        accounting for provenance weights.
    time : np.ndarray
        Time values (decimal years).
    span_years : float
        Bandwidth used for the local regression.
    kernel : str
        Kernel function used ('tricube', 'gaussian', 'epanechnikov').
    """
    rate: np.ndarray
    rate_se: np.ndarray
    accel: np.ndarray
    accel_se: np.ndarray
    n_effective: np.ndarray
    time: np.ndarray
    span_years: float
    kernel: str = 'tricube'
    
    def to_dataframe(self) -> pd.DataFrame:
        """Convert results to pandas DataFrame with datetime index."""
        time_index = pd.to_datetime([
            _decimal_year_to_datetime(t) for t in self.time
        ])
        return pd.DataFrame({
            'rate': self.rate,
            'rate_se': self.rate_se,
            'accel': self.accel,
            'accel_se': self.accel_se,
            'n_effective': self.n_effective,
            'decimal_year': self.time
        }, index=pd.Index(time_index, name='time'))
    
    def __repr__(self) -> str:
        valid_rate = ~np.isnan(self.rate)
        return (
            f"KinematicsResult(\n"
            f"  time range: {self.time.min():.4f} - {self.time.max():.4f}\n"
            f"  span_years: {self.span_years}\n"
            f"  kernel: {self.kernel}\n"
            f"  valid estimates: {valid_rate.sum()} / {len(self.rate)}\n"
            f"  mean rate: {np.nanmean(self.rate):.6f} ± {np.nanmean(self.rate_se):.6f}\n"
            f"  mean accel: {np.nanmean(self.accel):.6f} ± {np.nanmean(self.accel_se):.6f}\n"
            f")"
        )


@dataclass
class DOLSResult:
    """
    Container for Dynamic OLS calibration results.
    
    Attributes
    ----------
    alpha : float
        Estimated climate sensitivity parameter.
        Units: [sea_level_units]/year/[temperature_units] (e.g., mm/yr/K)
    alpha_se : float
        HAC-robust standard error of alpha.
    alpha_ci : Tuple[float, float]
        95% confidence interval for alpha.
    equilibrium_temp : float
        Implied equilibrium/baseline temperature (intercept term).
    trend : float
        Background linear trend coefficient.
        Units: [sea_level_units]/year (e.g., mm/yr)
    model : object
        Full statsmodels RegressionResults object.
    diagnostics : Dict
        Model diagnostics including R², AIC, BIC, Durbin-Watson, etc.
    n_lags : int
        Number of leads/lags used in DOLS specification.
    """
    alpha: float
    alpha_se: float
    alpha_ci: Tuple[float, float]
    equilibrium_temp: float
    trend: float
    model: object
    diagnostics: Dict
    n_lags: int
    
    def __repr__(self) -> str:
        return (
            f"DOLSResult(\n"
            f"  alpha: {self.alpha:.3f} ± {self.alpha_se:.3f} "
            f"[{self.alpha_ci[0]:.3f}, {self.alpha_ci[1]:.3f}]\n"
            f"  trend: {self.trend:.4f}\n"
            f"  R²: {self.diagnostics['r_squared']:.4f}\n"
            f"  n_obs: {self.diagnostics['n_observations']}\n"
            f"  n_lags: {self.n_lags}\n"
            f")"
        )

@dataclass
class DOLSQuadraticResult:
    """
    Results from quadratic Dynamic OLS calibration.
    
    Model: H(t) = α₀×∫T(τ)dτ + (1/2)(dα/dT)×∫T²(τ)dτ + β×t + Σγᵢ×ΔT(t+i) + ε
    
    Attributes
    ----------
    alpha0 : float
        Linear sensitivity coefficient (sea level rise per °C)
    alpha0_se : float
        Standard error of alpha0
    dalpha_dT : float
        Quadratic sensitivity coefficient (change in sensitivity per °C)
    dalpha_dT_se : float
        Standard error of dalpha_dT
    trend : float
        Linear trend coefficient (non-temperature-driven SLR)
    trend_se : float
        Standard error of trend
    coefficients : np.ndarray
        All regression coefficients [dalpha_dT/2, alpha0, trend, gamma_lags...]
    covariance : np.ndarray
        Full covariance matrix of coefficients
    r2 : float
        Coefficient of determination
    r2_adj : float
        Adjusted R²
    aic : float
        Akaike Information Criterion
    bic : float
        Bayesian Information Criterion
    n_obs : int
        Number of observations used
    n_lags : int
        Number of lead/lag terms
    residuals : np.ndarray
        Model residuals
    fitted : np.ndarray
        Fitted values
    time : np.ndarray
        Time values used
    """
    alpha0: float
    alpha0_se: float
    dalpha_dT: float
    dalpha_dT_se: float
    trend: float
    trend_se: float
    coefficients: np.ndarray
    covariance: np.ndarray
    r2: float
    r2_adj: float
    aic: float
    bic: float
    n_obs: int
    n_lags: int
    residuals: np.ndarray
    fitted: np.ndarray
    time: np.ndarray
    
    def __repr__(self) -> str:
        return (
            f"DOLSQuadraticResult(\n"
            f"  α₀ = {self.alpha0:.3f} ± {self.alpha0_se:.3f} (linear sensitivity)\n"
            f"  dα/dT = {self.dalpha_dT:.3f} ± {self.dalpha_dT_se:.3f} (quadratic sensitivity)\n"
            f"  trend = {self.trend:.4f} ± {self.trend_se:.4f}\n"
            f"  R² = {self.r2:.3f}, R²_adj = {self.r2_adj:.3f}\n"
            f"  AIC = {self.aic:.1f}, BIC = {self.bic:.1f}\n"
            f"  n_obs = {self.n_obs}, n_lags = {self.n_lags}\n"
            f")"
        )
    
    def predict_rate(self, temperature: np.ndarray) -> np.ndarray:
        """
        Predict GMSL rate for given temperature.
        
        rate = dα/dT × T² + α₀ × T + trend
        """
        return self.dalpha_dT * temperature**2 + self.alpha0 * temperature + self.trend
    
    def predict_rate_ci(self, temperature: np.ndarray, 
                        confidence: float = 0.95) -> Tuple[np.ndarray, np.ndarray]:
        """
        Predict GMSL rate with confidence interval.
        
        Returns (lower, upper) bounds.
        """
        from scipy import stats
        
        rate = self.predict_rate(temperature)
        
        # Gradient of rate w.r.t. [dalpha_dT/2, alpha0, trend]: [T², T, 1]
        # Note: coefficient is dalpha_dT/2, so gradient for dalpha_dT is T²/2... 
        # Actually we store dalpha_dT directly, need to be careful here
        # The covariance is for [coeff of ∫T², coeff of ∫T, trend, ...]
        # coeff of ∫T² = dalpha_dT/2, so var(dalpha_dT) = 4*var(coeff)
        
        # For prediction: rate = dalpha_dT*T² + alpha0*T + trend
        # Jacobian w.r.t. [dalpha_dT, alpha0, trend] = [T², T, 1]
        
        # Build Jacobian for each temperature point
        n = len(temperature)
        J = np.column_stack([temperature**2, temperature, np.ones(n)])
        
        # Extract 3x3 covariance for [dalpha_dT, alpha0, trend]
        # Need to transform from coefficient covariance
        # cov[0,0] is var(dalpha_dT/2), so var(dalpha_dT) = 4*cov[0,0]
        cov_params = np.zeros((3, 3))
        cov_params[0, 0] = 4 * self.covariance[0, 0]  # var(dalpha_dT)
        cov_params[0, 1] = 2 * self.covariance[0, 1]  # cov(dalpha_dT, alpha0)
        cov_params[0, 2] = 2 * self.covariance[0, 2]  # cov(dalpha_dT, trend)
        cov_params[1, 0] = cov_params[0, 1]
        cov_params[1, 1] = self.covariance[1, 1]  # var(alpha0)
        cov_params[1, 2] = self.covariance[1, 2]  # cov(alpha0, trend)
        cov_params[2, 0] = cov_params[0, 2]
        cov_params[2, 1] = cov_params[1, 2]
        cov_params[2, 2] = self.covariance[2, 2]  # var(trend)
        
        # Prediction variance: var(rate) = J @ cov @ J.T (diagonal elements)
        var_rate = np.sum((J @ cov_params) * J, axis=1)
        se_rate = np.sqrt(var_rate)
        
        # Critical value
        alpha = 1 - confidence
        z = stats.norm.ppf(1 - alpha / 2)
        
        return rate - z * se_rate, rate + z * se_rate


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def _decimal_year_to_datetime(decimal_year: float) -> datetime:
    """Convert decimal year to datetime object."""
    year = int(decimal_year)
    remainder = decimal_year - year
    base = datetime(year, 1, 1)
    next_year = datetime(year + 1, 1, 1)
    return base + (next_year - base) * remainder


def _datetime_to_decimal_year(dt: datetime) -> float:
    """Convert datetime object to decimal year."""
    year = dt.year
    base = datetime(year, 1, 1)
    next_year = datetime(year + 1, 1, 1)
    return year + (dt - base).total_seconds() / (next_year - base).total_seconds()


def _estimate_resolution_months(index: pd.DatetimeIndex) -> float:
    """
    Estimate the temporal resolution of a datetime index in months.
    
    Returns approximate months between observations:
    - 1.0 for monthly
    - 3.0 for quarterly
    - 12.0 for annual
    """
    if len(index) < 2:
        return 1.0
    
    # Median time difference in days
    diffs = np.diff(index.values).astype('timedelta64[D]').astype(float)
    median_days = np.median(diffs)
    months = median_days / 30.44  # Average days per month
    
    # Round to common resolutions
    if months < 1.5:
        return 1.0
    elif months < 4.5:
        return 3.0
    elif months < 9:
        return 6.0
    else:
        return 12.0


# =============================================================================
# SECTION 1: PREPROCESSING
# =============================================================================

def resample_to_monthly(
    df: pd.DataFrame,
    value_col: str,
    unc_col: Optional[str] = None,
    method: str = 'time',
    inflate_uncertainty: bool = True,
    track_provenance: bool = True
) -> pd.DataFrame:
    """
    Resample lower-frequency (e.g., annual) data to monthly resolution.
    
    IMPORTANT STATISTICAL CONSIDERATIONS
    ------------------------------------
    Temporal interpolation increases the number of data points but does NOT
    increase the information content of the dataset. When annual data is
    interpolated to monthly resolution:
    
    1. The 12 monthly values derived from one annual observation are NOT
       independent measurements.
    2. Standard errors computed from interpolated data will be artificially
       small by approximately √n, where n is the interpolation factor.
    3. Degrees of freedom should reflect the original sample size.
    
    This function addresses these issues by:
    - Tracking which points are original vs. interpolated
    - Inflating uncertainties for interpolated points by √(interpolation_factor)
    - Providing weights for use in subsequent weighted regression
    
    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame with datetime index.
    value_col : str
        Column containing values to interpolate.
    unc_col : str, optional
        Column containing uncertainty estimates.
    method : str, default 'time'
        Interpolation method: 'time', 'linear', 'spline', 'pchip'.
    inflate_uncertainty : bool, default True
        If True, inflate uncertainties for interpolated points.
    track_provenance : bool, default True
        If True, add columns tracking data provenance:
        - 'is_original': Boolean flag for original observations
        - 'weight': Statistical weight (1.0 for original, 1/n for interpolated)
        - 'interpolation_factor': Local interpolation ratio
    
    Returns
    -------
    pd.DataFrame
        Monthly-resolution DataFrame with additional metadata columns.
    
    Examples
    --------
    >>> df_monthly = resample_to_monthly(
    ...     df_annual, 
    ...     value_col='gmsl',
    ...     unc_col='gmsl_sigma'
    ... )
    >>> # Use weights in regression
    >>> model = sm.WLS(y, X, weights=df_monthly['weight']).fit()
    
    See Also
    --------
    merge_multiresolution_data : Alternative that preserves native resolution
    """
    df = df.copy()
    
    # Store original info
    original_times = df.index.copy()
    original_resolution = _estimate_resolution_months(df.index)
    
    # Create monthly date range
    monthly_index = pd.date_range(
        df.index.min().replace(day=1),
        df.index.max() + pd.DateOffset(months=1),
        freq='MS'
    )
    
    # Reindex and interpolate
    combined_index = df.index.union(monthly_index)
    df_reindexed = df.reindex(combined_index).sort_index()
    df_reindexed[value_col] = df_reindexed[value_col].interpolate(method=method)
    
    # Handle uncertainty
    if unc_col and unc_col in df.columns:
        df_reindexed[f'{unc_col}_original'] = df_reindexed[unc_col].copy()
        df_reindexed[unc_col] = df_reindexed[unc_col].interpolate(method='nearest')
        df_reindexed[f'{unc_col}_original'] = df_reindexed[f'{unc_col}_original'].interpolate(method='nearest')
    
    # Keep only monthly points
    df_monthly = df_reindexed.loc[monthly_index].copy()
    
    if track_provenance:
        # Identify original points (within 15 days of original timestamp)
        df_monthly['is_original'] = False
        for orig_time in original_times:
            time_diff = np.abs((df_monthly.index - orig_time).days)
            df_monthly.loc[time_diff <= 15, 'is_original'] = True
        
        # Interpolation factor and weights
        df_monthly['interpolation_factor'] = original_resolution
        df_monthly.loc[df_monthly['is_original'], 'interpolation_factor'] = 1.0
        df_monthly['weight'] = 1.0 / df_monthly['interpolation_factor']
        df_monthly['effective_n'] = df_monthly['weight']
    
    # Inflate uncertainties
    if inflate_uncertainty and unc_col and unc_col in df.columns:
        if track_provenance:
            inflation = np.sqrt(df_monthly['interpolation_factor'])
            df_monthly[unc_col] = df_monthly[f'{unc_col}_original'] * inflation
        else:
            df_monthly[unc_col] = df_monthly[unc_col] * np.sqrt(original_resolution)
    
    df_monthly.index.name = 'time'
    return df_monthly


def merge_multiresolution_data(
    datasets: List[pd.DataFrame],
    value_col: str = 'gmsl',
    unc_col: Optional[str] = 'gmsl_sigma',
    resolution_months: Optional[List[float]] = None
) -> pd.DataFrame:
    """
    Merge datasets with different temporal resolutions without interpolation.
    
    This preserves native resolution and assigns statistical weights based on
    temporal coverage. An annual observation receives weight=12 (equivalent to
    12 months of information), while monthly observations receive weight=1.
    
    Parameters
    ----------
    datasets : List[pd.DataFrame]
        DataFrames to merge, each with datetime index.
    value_col : str, default 'gmsl'
        Column name for values.
    unc_col : str, optional
        Column name for uncertainties.
    resolution_months : List[float], optional
        Resolution of each dataset in months. If None, auto-detected.
    
    Returns
    -------
    pd.DataFrame
        Merged DataFrame with columns:
        - value_col: Values
        - unc_col: Uncertainties (if provided)
        - 'resolution': Temporal resolution in months
        - 'weight': Statistical weight
        - 'source': Index of source dataset
    
    Examples
    --------
    >>> df_merged = merge_multiresolution_data(
    ...     [df_frederikse, df_altimetry],
    ...     resolution_months=[12, 1]  # Annual, Monthly
    ... )
    """
    if len(datasets) == 0:
        raise ValueError("At least one dataset required")
    
    parts = []
    for i, df in enumerate(datasets):
        part = pd.DataFrame(index=df.index)
        part[value_col] = df[value_col].values
        
        if unc_col and unc_col in df.columns:
            part[unc_col] = df[unc_col].values
        elif unc_col:
            part[unc_col] = np.nan
        
        # Determine resolution
        if resolution_months is not None:
            res = resolution_months[i]
        else:
            res = _estimate_resolution_months(df.index)
        
        part['resolution'] = res
        part['weight'] = res  # Weight proportional to temporal coverage
        part['source'] = i
        parts.append(part)
    
    merged = pd.concat(parts).sort_index()
    merged.index.name = 'time'
    return merged


def align_to_baseline(
    df: pd.DataFrame,
    value_col: str,
    time_col: Optional[str] = None,
    start_year: float = 1993,
    end_year: float = 2009,
    inclusive: str = 'left'
) -> pd.DataFrame:
    """
    Shift a dataset so its mean over a reference period is zero.
    
    Baseline alignment ensures physical consistency when combining datasets
    or when the intercept has a meaningful interpretation.
    
    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame. NOT modified in place.
    value_col : str
        Column to align.
    time_col : str, optional
        Column containing time values. If None, uses the index.
    start_year : float, default 1993
        Start of reference period.
    end_year : float, default 2009
        End of reference period.
    inclusive : str, default 'left'
        Boundary handling: 'left' (default), 'right', 'both', 'neither'.
        'left' means start_year ≤ t < end_year.
    
    Returns
    -------
    pd.DataFrame
        Aligned DataFrame (copy of original).
    
    Examples
    --------
    >>> # Align to satellite era (1993-2008)
    >>> df_aligned = align_to_baseline(df, 'gmsl', start_year=1993, end_year=2009)
    """
    df = df.copy()
    
    # Get time values
    if time_col is not None:
        t = df[time_col]
    elif isinstance(df.index, pd.DatetimeIndex):
        t = pd.Series([_datetime_to_decimal_year(dt) for dt in df.index], index=df.index)
    else:
        t = pd.Series(df.index, index=df.index)
    
    # Build mask
    if inclusive == 'left':
        mask = (t >= start_year) & (t < end_year)
    elif inclusive == 'both':
        mask = (t >= start_year) & (t <= end_year)
    elif inclusive == 'right':
        mask = (t > start_year) & (t <= end_year)
    elif inclusive == 'neither':
        mask = (t > start_year) & (t < end_year)
    else:
        raise ValueError(f"inclusive must be 'left', 'right', 'both', or 'neither'")
    
    if mask.sum() == 0:
        raise ValueError(
            f"No data in reference period [{start_year}, {end_year}). "
            f"Data range: [{t.min():.1f}, {t.max():.1f}]"
        )
    
    baseline_mean = df.loc[mask, value_col].mean()
    df[value_col] = df[value_col] - baseline_mean
    
    # Store metadata
    df.attrs['baseline_period'] = (start_year, end_year)
    df.attrs['baseline_mean_removed'] = baseline_mean
    
    return df


def harmonize_baseline(
    df: pd.DataFrame, 
    value_col: str,
    new_baseline: Tuple[int, int] = (1995, 2005)
) -> pd.DataFrame:
    """
    Re-baseline anomalies to a different reference period.
    
    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame with datetime index.
    value_col : str
        Column containing anomaly values.
    new_baseline : Tuple[int, int], default (1981, 2010)
        New reference period (start_year, end_year).
    
    Returns
    -------
    pd.DataFrame
        DataFrame with re-baselined values.
    """
    df = df.copy()
    mask = (df.index.year >= new_baseline[0]) & (df.index.year <= new_baseline[1])
    
    if mask.sum() == 0:
        raise ValueError(f"No data in baseline period {new_baseline}")
    
    baseline_mean = df.loc[mask, value_col].mean()
    df[value_col] = df[value_col] - baseline_mean

    known_suffixes = ['_lower', '_upper', '_smoothed']
    
    for suffix in known_suffixes:
        target_col = value_col + suffix
        if target_col in df.columns:
            df[target_col] = df[target_col] - baseline_mean
    
    return df


def compute_thermodynamic_signal(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute thermodynamic sea level signal from budget components.
    
    Thermodynamic signal = Steric + Barystatic - TWS
    
    where Barystatic = Glaciers + Greenland + Antarctica
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing columns: 'steric', 'glaciers', 'greenland',
        'antarctica', 'tws' and their uncertainties (*_sigma).
    
    Returns
    -------
    pd.DataFrame
        DataFrame with thermodynamic signal and components.
    """
    df = df.copy()
    
    # Compute components
    df['barystatic'] = df['glaciers'] + df['greenland'] + df['antarctica'] + df['tws']
    df['sum_tws_comps'] = df['reservoir'] + df['groundwater'] + df['tws_natural']
    
    df['thermodynamic_sum'] = df['sum_contributors'] - df['tws']
    df['thermodynamic_gmsl'] = df['gmsl'] - df['tws']
    
    # Propagate uncertainties (assuming independence)
    if 'glaciers_sigma' in df.columns:
        df['barystatic_sigma'] = np.sqrt(
            df['glaciers_sigma']**2 + 
            df['greenland_sigma']**2 + 
            df['antarctica_sigma']**2 +
            df['tws_sigma']**2
        )
        df['thermodynamic_sum_sigma'] = np.sqrt(
            df['sum_contributors_sigma']**2 + 
            df['tws_sigma']**2
        )
        df['thermodynamic_gmsl_sigma'] = np.sqrt(
            df['gmsl_sigma']**2 +
            df['tws_sigma']**2
        )
    
    return df


# =============================================================================
# SECTION 2: KINEMATICS (LOCAL POLYNOMIAL REGRESSION)
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
    
    METHODOLOGY
    -----------
    At each target point t₀, fits a weighted quadratic polynomial:
    
        y(t) = β₀ + β₁(t - t₀) + ½β₂(t - t₀)² + ε
    
    The coefficients have physical interpretations:
        β₁ = rate of change (first derivative) at t₀
        β₂ = acceleration (second derivative) at t₀
    
    Weights combine three factors:
        w_i = K((t_i - t₀)/h) × (1/σ_i²) × p_i
    
    where:
        K(u) = kernel function (default: tricube)
        h = bandwidth (span_years)
        σ_i = measurement uncertainty
        p_i = provenance weight (for interpolated data)
    
    KERNEL FUNCTIONS
    ----------------
    - 'tricube': K(u) = (1 - |u|³)³ for |u| ≤ 1, else 0
      Compact support, smooth, recommended for most applications.
    
    - 'gaussian': K(u) = exp(-u²/2)
      Infinite support, very smooth, includes all data with decaying weights.
    
    - 'epanechnikov': K(u) = 0.75(1 - u²) for |u| ≤ 1
      Compact support, optimal for density estimation.
    
    BANDWIDTH SELECTION
    -------------------
    The bandwidth controls the bias-variance tradeoff:
    
        Smaller bandwidth → More variance, less bias
            - Captures rapid changes, noisier estimates
            - Recommended: 10-15 years for detecting recent changes
        
        Larger bandwidth → Less variance, more bias
            - Smoother estimates, may miss rapid changes
            - Recommended: 30-50 years for long-term trends
    
    Parameters
    ----------
    time : np.ndarray
        Time values as decimal years (e.g., 2020.5 for mid-2020).
    value : np.ndarray
        Observed values (e.g., sea level in mm).
    sigma : np.ndarray
        1-sigma measurement uncertainty. Must be positive.
    span_years : float
        Bandwidth in years. Typical values: 20-30 for climate applications.
    weights : np.ndarray, optional
        Provenance weights (e.g., from resample_to_monthly). Default: 1.0.
    min_effective_obs : int, default 12
        Minimum effective observations for regression.
    kernel : str, default 'tricube'
        Kernel function: 'tricube', 'gaussian', 'epanechnikov'.
    
    Returns
    -------
    KinematicsResult
        Dataclass with rate, accel, standard errors, and metadata.
        Call .to_dataframe() to convert to pandas DataFrame.
    
    Examples
    --------
    >>> result = compute_kinematics(
    ...     time=decimal_years,
    ...     value=gmsl_mm,
    ...     sigma=gmsl_sigma,
    ...     span_years=30
    ... )
    >>> print(result)
    >>> df = result.to_dataframe()
    
    Notes
    -----
    Edge effects: Estimates within span_years/2 of record boundaries have
    asymmetric kernel support. The method handles this naturally, but
    uncertainties are larger and potential bias exists near edges.
    
    References
    ----------
    Cleveland, W. S. (1979). Robust Locally Weighted Regression and Smoothing
        Scatterplots. JASA, 74(368), 829-836.
    
    Church, J. A., & White, N. J. (2006). A 20th century acceleration in global
        sea-level rise. Geophysical Research Letters, 33, L01602.
    """
    # Input validation
    time = np.asarray(time, dtype=np.float64)
    value = np.asarray(value, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    
    n = len(time)
    if not (len(value) == n and len(sigma) == n):
        raise ValueError(
            f"Inconsistent lengths: time={n}, value={len(value)}, sigma={len(sigma)}"
        )
    
    if np.any(sigma <= 0):
        raise ValueError("All sigma values must be positive")
    
    if np.any(np.isnan(value)):
        warnings.warn(
            f"Input contains {np.sum(np.isnan(value))} NaN values. "
            "These will be excluded from local fits."
        )
    
    if weights is None:
        weights = np.ones(n)
    else:
        weights = np.asarray(weights, dtype=np.float64)
        if len(weights) != n:
            raise ValueError(f"weights length ({len(weights)}) must match data ({n})")
    
    # Kernel functions
    kernels = {
        'tricube': lambda u: np.where(np.abs(u) <= 1, (1 - np.abs(u)**3)**3, 0),
        'gaussian': lambda u: np.exp(-0.5 * u**2),
        'epanechnikov': lambda u: np.where(np.abs(u) <= 1, 0.75 * (1 - u**2), 0)
    }
    if kernel not in kernels:
        raise ValueError(f"Unknown kernel '{kernel}'. Options: {list(kernels.keys())}")
    kernel_func = kernels[kernel]
    
    # Initialize outputs
    rate = np.full(n, np.nan)
    rate_se = np.full(n, np.nan)
    accel = np.full(n, np.nan)
    accel_se = np.full(n, np.nan)
    n_effective = np.full(n, np.nan)
    
    h = span_years
    
    # Main loop
    for i in range(n):
        if np.isnan(value[i]):
            continue
        
        t0 = time[i]
        u = (time - t0) / h
        
        # Combined weights
        k_weights = kernel_func(u)
        combined = k_weights * (1.0 / sigma**2) * weights
        
        # Valid mask
        valid = (combined > 1e-12) & (~np.isnan(value))
        n_eff = np.sum(weights[valid])
        n_effective[i] = n_eff
        
        if n_eff < min_effective_obs:
            continue
        
        # Local regression
        t_loc = time[valid]
        y_loc = value[valid]
        w_loc = combined[valid]
        
        dt = t_loc - t0
        X = np.column_stack([np.ones(len(dt)), dt, 0.5 * dt**2])
        
        try:
            model = sm.WLS(y_loc, X, weights=w_loc).fit()
            rate[i] = model.params[1]
            rate_se[i] = model.bse[1]
            accel[i] = model.params[2]
            accel_se[i] = model.bse[2]
        except (np.linalg.LinAlgError, ValueError):
            continue
    
    return KinematicsResult(
        rate=rate,
        rate_se=rate_se,
        accel=accel,
        accel_se=accel_se,
        n_effective=n_effective,
        time=time,
        span_years=span_years,
        kernel=kernel
    )


def compute_kinematics_multibandwidth(
    time: np.ndarray,
    value: np.ndarray,
    sigma: np.ndarray,
    bandwidths: List[float] = [15, 20, 30, 50],
    **kwargs
) -> Dict[float, KinematicsResult]:
    """
    Compute kinematics for multiple bandwidths to assess sensitivity.
    
    Parameters
    ----------
    time, value, sigma : np.ndarray
        Input data (see compute_kinematics).
    bandwidths : List[float], default [15, 20, 30, 50]
        Bandwidths to evaluate (in years).
    **kwargs
        Additional arguments passed to compute_kinematics.
    
    Returns
    -------
    Dict[float, KinematicsResult]
        Mapping from bandwidth to results.
    
    Examples
    --------
    >>> results = compute_kinematics_multibandwidth(time, value, sigma)
    >>> for bw, res in results.items():
    ...     print(f"{bw}yr: rate = {res.rate[-1]:.2f} mm/yr")
    """
    results = {}
    for bw in bandwidths:
        results[bw] = compute_kinematics(time, value, sigma, span_years=bw, **kwargs)
    return results


# =============================================================================
# SECTION 3: SEMI-EMPIRICAL CALIBRATION (DOLS)
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
    Estimate climate sensitivity (α) using Dynamic Least Squares.
    
    SEMI-EMPIRICAL MODEL
    --------------------
    The relationship between sea level and temperature is modeled as:
    
        H(t) = H₀ + α × ∫T(τ)dτ + β×t + ε(t)
    
    where:
        H(t) = sea level at time t [mm]
        T(t) = temperature anomaly [K or °C]
        α = climate sensitivity [mm/yr/K]
        β = background trend [mm/yr]
        H₀ = reference sea level
    
    DOLS METHODOLOGY
    ----------------
    Dynamic LS addresses cointegration between non-stationary series by
    augmenting the regression with leads and lags of ΔT:
    
        H(t) = α × ∫T(τ)dτ + β×t + Σᵢ γᵢ × ΔT(t+i) + ε(t)
    
    Yields consistent, asymptotically efficient estimates even when
    H and ∫T are I(1) cointegrated processes.

    Uses HAC (Newey-West) standard errors to account for residual autocorrelation
    
    Parameters
    ----------
    time : np.ndarray
        Time values (decimal years).
    gmsl : np.ndarray
        Global mean sea level anomaly [mm].
    gmsl_sigma : np.ndarray
        GMSL uncertainty [mm]. Used for WLS weighting.
    temperature : np.ndarray
        Temperature anomaly [K or °C].
    n_lags : int, default 2
        Number of leads AND lags of ΔT. Total terms = 2×n_lags + 1.
    include_trend : bool, default True
        Include linear time trend.
    hac_maxlags : int, default 24
        Maximum lags for HAC (Newey-West) standard errors.
    
    Returns
    -------
    DOLSResult
        Dataclass containing alpha, standard error, confidence interval,
        trend, full model, and diagnostics.
    
    Examples
    --------
    >>> result = calibrate_alpha_dols(
    ...     time=decimal_years,
    ...     gmsl=gmsl_mm,
    ...     gmsl_sigma=gmsl_sigma,
    ...     temperature=temperature_anomaly
    ... )
    >>> print(f"α = {result.alpha:.2f} ± {result.alpha_se:.2f} mm/yr/K")
    
    References
    ----------
    Stock, J. H., & Watson, M. W. (1993). A Simple Estimator of Cointegrating
        Vectors in Higher Order Integrated Systems. Econometrica, 61(4), 783-820.
    
    Vermeer, M., & Rahmstorf, S. (2009). Global sea level linked to global
        temperature. PNAS, 106(51), 21527-21532.
    """
    # Input validation
    time = np.asarray(time, dtype=np.float64)
    gmsl = np.asarray(gmsl, dtype=np.float64)
    gmsl_sigma = np.asarray(gmsl_sigma, dtype=np.float64)
    temperature = np.asarray(temperature, dtype=np.float64)
    
    n = len(time)
    if not all(len(x) == n for x in [gmsl, gmsl_sigma, temperature]):
        raise ValueError("All inputs must have the same length")
    
    if np.any(gmsl_sigma <= 0):
        raise ValueError("All gmsl_sigma values must be positive")
    
    # Construct regressors
    dt = np.gradient(time)
    integrated_T = np.cumsum(temperature * dt)
    delta_T = np.gradient(temperature)
    
    # Build design matrix
    regressors = {'intercept': np.ones(n), 'integrated_T': integrated_T}
    
    if include_trend:
        regressors['trend'] = time - time.mean()
    
    # Add leads and lags
    delta_T_series = pd.Series(delta_T)
    for lag in range(-n_lags, n_lags + 1):
        regressors[f'delta_T_lag{lag:+d}'] = delta_T_series.shift(-lag).values
    
    X = pd.DataFrame(regressors)
    y = gmsl
    w = 1.0 / (gmsl_sigma ** 2)
    
    # Remove NaN rows
    valid = X.notna().all(axis=1) & ~np.isnan(y) & ~np.isnan(w)
    n_valid = valid.sum()
    
    if n_valid < len(regressors) + 10:
        raise ValueError(f"Insufficient observations ({n_valid}) for {len(regressors)} parameters")
    
    if n - n_valid > 0:
        warnings.warn(f"Dropped {n - n_valid} observations due to NaN")
    
    # Fit model with HAC (Newey-West) standard errors to account for residual autocorrelation
    model = sm.WLS(y[valid], X.loc[valid].values, weights=w[valid]).fit(
        cov_type='HAC',
        cov_kwds={'maxlags': hac_maxlags}
    )
    
    # Extract results
    param_names = list(regressors.keys())
    idx_alpha = param_names.index('integrated_T')
    idx_intercept = param_names.index('intercept')
    idx_trend = param_names.index('trend') if include_trend else None
    
    alpha = model.params[idx_alpha]
    alpha_se = model.bse[idx_alpha]
    alpha_ci = tuple(model.conf_int()[idx_alpha])
    
    diagnostics = {
        'r_squared': model.rsquared,
        'r_squared_adj': model.rsquared_adj,
        'n_observations': n_valid,
        'n_parameters': len(regressors),
        'residual_std': np.std(model.resid),
        'durbin_watson': sm.stats.durbin_watson(model.resid),
        'aic': model.aic,
        'bic': model.bic
    }
    
    return DOLSResult(
        alpha=alpha,
        alpha_se=alpha_se,
        alpha_ci=alpha_ci,
        equilibrium_temp=model.params[idx_intercept],
        trend=model.params[idx_trend] if include_trend else 0.0,
        model=model,
        diagnostics=diagnostics,
        n_lags=n_lags
    )

def calibrate_alpha_dols_quadratic(
    sea_level: pd.Series,
    temperature: pd.Series,
    n_lags: int = 2,
    time_unit: str = 'year',
    hac_kernel: str = 'bartlett',
    hac_bandwidth: int = None
) -> DOLSQuadraticResult:
    """
    Calibrate quadratic sea level sensitivity using Dynamic OLS.
    
    Estimates the relationship:
        H(t) = (dα/dT)/2 × ∫T²(τ)dτ + α₀ × ∫T(τ)dτ + β×t + Σγᵢ×ΔT(t+i) + ε
    
    where:
        - H(t) is sea level at time t
        - T(t) is temperature anomaly
        - α₀ is the linear sensitivity (SLR per °C)
        - dα/dT is how sensitivity changes with temperature
        - β is a linear trend (non-temperature SLR)
        - γᵢ are lead/lag coefficients for ΔT
    
    This implies a rate model:
        dH/dt = (dα/dT) × T² + α₀ × T + β
    
    Parameters
    ----------
    sea_level : pd.Series
        Sea level time series with datetime index
    temperature : pd.Series
        Temperature anomaly time series with datetime index
    n_lags : int, default 2
        Number of lead and lag terms for ΔT (total 2*n_lags + 1 terms)
    time_unit : str, default 'year'
        Time unit for integration ('year' or 'month')
    hac_kernel : str, default 'bartlett'
        Kernel for HAC standard errors ('bartlett', 'parzen', 'quadratic')
    hac_bandwidth : int, optional
        Bandwidth for HAC. If None, uses Newey-West automatic selection.
        
    Returns
    -------
    DOLSQuadraticResult
        Dataclass containing estimates, uncertainties, and diagnostics
        
    Example
    -------
    >>> result = calibrate_alpha_dols_quadratic(df['gmsl'], df['temperature'])
    >>> print(f"α₀ = {result.alpha0:.2f} ± {result.alpha0_se:.2f}")
    >>> print(f"dα/dT = {result.dalpha_dT:.2f} ± {result.dalpha_dT_se:.2f}")
    
    Notes
    -----
    The quadratic term captures accelerating sensitivity as temperatures rise,
    consistent with nonlinear ice sheet dynamics and thermal expansion.
    
    HAC (Heteroskedasticity and Autocorrelation Consistent) standard errors
    are used to account for residual autocorrelation.
    
    References
    ----------
    Stock, J. H., & Watson, M. W. (1993). A simple estimator of cointegrating
    vectors in higher order integrated systems. Econometrica, 61(4), 783-820.
    """
    # Align series
    common_index = sea_level.index.intersection(temperature.index)
    H = sea_level.loc[common_index].values.astype(float)
    T = temperature.loc[common_index].values.astype(float)
    
    # Convert index to decimal years
    time_years = np.array([
        t.year + (t.month - 1) / 12 + (t.day - 1) / 365.25
        for t in common_index
    ])
    
    # Time step
    if time_unit == 'year':
        dt = np.median(np.diff(time_years))
    else:  # month
        dt = 1 / 12
    
    n = len(H)
    
    # Compute cumulative integrals using trapezoidal rule
    # ∫T(τ)dτ and ∫T²(τ)dτ
    integral_T = np.zeros(n)
    integral_T2 = np.zeros(n)
    for i in range(1, n):
        integral_T[i] = integral_T[i-1] + 0.5 * (T[i] + T[i-1]) * dt
        integral_T2[i] = integral_T2[i-1] + 0.5 * (T[i]**2 + T[i-1]**2) * dt
    
    # Compute ΔT for leads and lags
    delta_T = np.diff(T, prepend=T[0])
    
    # Build regressor matrix
    # [∫T², ∫T, t, ΔT_{t-n_lags}, ..., ΔT_{t}, ..., ΔT_{t+n_lags}]
    regressors = [integral_T2, integral_T, time_years - time_years[0]]
    
    for lag in range(-n_lags, n_lags + 1):
        if lag < 0:
            shifted = np.concatenate([np.full(-lag, np.nan), delta_T[:lag]])
        elif lag > 0:
            shifted = np.concatenate([delta_T[lag:], np.full(lag, np.nan)])
        else:
            shifted = delta_T
        regressors.append(shifted)
    
    X = np.column_stack(regressors)
    
    # Remove rows with NaN (from leads/lags)
    valid = ~np.any(np.isnan(X), axis=1)
    X_valid = X[valid]
    H_valid = H[valid]
    time_valid = time_years[valid]
    n_valid = len(H_valid)
    
    # Add constant
    X_valid = np.column_stack([X_valid, np.ones(n_valid)])
    
    # OLS estimation
    XtX = X_valid.T @ X_valid
    XtX_inv = np.linalg.inv(XtX)
    coeffs = XtX_inv @ (X_valid.T @ H_valid)
    
    # Fitted values and residuals
    fitted = X_valid @ coeffs
    residuals = H_valid - fitted
    
    # HAC covariance estimation
    if hac_bandwidth is None:
        # Newey-West automatic bandwidth selection
        hac_bandwidth = int(np.floor(4 * (n_valid / 100) ** (2/9)))
    
    # Compute kernel weights
    if hac_kernel == 'bartlett':
        kernel_weights = lambda j: 1 - j / (hac_bandwidth + 1) if j <= hac_bandwidth else 0
    elif hac_kernel == 'parzen':
        def kernel_weights(j):
            z = j / (hac_bandwidth + 1)
            if z <= 0.5:
                return 1 - 6*z**2 + 6*z**3
            elif z <= 1:
                return 2 * (1 - z)**3
            else:
                return 0
    else:  # quadratic spectral
        def kernel_weights(j):
            z = j / (hac_bandwidth + 1)
            if z == 0:
                return 1
            return 3 * (np.sin(np.pi*z)/(np.pi*z) - np.cos(np.pi*z)) / (np.pi*z)**2
    
    # HAC sandwich estimator
    k = X_valid.shape[1]
    S = np.zeros((k, k))
    
    for j in range(hac_bandwidth + 1):
        weight = kernel_weights(j)
        if j == 0:
            Gamma_j = sum(residuals[t] * residuals[t] * np.outer(X_valid[t], X_valid[t]) 
                         for t in range(n_valid)) / n_valid
            S += weight * Gamma_j
        else:
            Gamma_j = sum(residuals[t] * residuals[t-j] * np.outer(X_valid[t], X_valid[t-j]) 
                         for t in range(j, n_valid)) / n_valid
            S += weight * (Gamma_j + Gamma_j.T)
    
    # HAC covariance matrix
    cov_hac = n_valid * XtX_inv @ S @ XtX_inv
    se = np.sqrt(np.diag(cov_hac))
    
    # Extract key parameters
    # coeffs[0] = coefficient on ∫T² = (dα/dT)/2
    # coeffs[1] = coefficient on ∫T = α₀
    # coeffs[2] = trend coefficient
    dalpha_dT = 2 * coeffs[0]
    dalpha_dT_se = 2 * se[0]
    alpha0 = coeffs[1]
    alpha0_se = se[1]
    trend = coeffs[2]
    trend_se = se[2]
    
    # Model fit statistics
    ss_res = np.sum(residuals**2)
    ss_tot = np.sum((H_valid - np.mean(H_valid))**2)
    r2 = 1 - ss_res / ss_tot
    r2_adj = 1 - (1 - r2) * (n_valid - 1) / (n_valid - k)
    
    # Information criteria
    log_likelihood = -n_valid/2 * (np.log(2*np.pi) + np.log(ss_res/n_valid) + 1)
    aic = -2 * log_likelihood + 2 * k
    bic = -2 * log_likelihood + k * np.log(n_valid)
    
    return DOLSQuadraticResult(
        alpha0=alpha0,
        alpha0_se=alpha0_se,
        dalpha_dT=dalpha_dT,
        dalpha_dT_se=dalpha_dT_se,
        trend=trend,
        trend_se=trend_se,
        coefficients=coeffs,
        covariance=cov_hac,
        r2=r2,
        r2_adj=r2_adj,
        aic=aic,
        bic=bic,
        n_obs=n_valid,
        n_lags=n_lags,
        residuals=residuals,
        fitted=fitted,
        time=time_valid
    )



def test_rate_temperature_nonlinearity(
    rate: np.ndarray,
    temperature: np.ndarray,
    rate_sigma: np.ndarray = None,
    alpha: float = 0.05
) -> dict:
    """
    Test whether a quadratic model (rate vs temperature) is significantly
    better than a linear model.
    
    Fits two models:
        Linear:    rate = b₁×T + b₀
        Quadratic: rate = a₂×T² + a₁×T + a₀
    
    Parameters
    ----------
    rate : np.ndarray
        GMSL rate (e.g., from compute_kinematics)
    temperature : np.ndarray
        Temperature anomaly (must be same length as rate)
    rate_sigma : np.ndarray, optional
        Uncertainty in rate estimates. If provided, uses weighted least squares.
    alpha : float, default 0.05
        Significance level for hypothesis tests
        
    Returns
    -------
    dict with keys:
        'linear': dict with 'coeffs', 'se', 'r2', 'aic', 'bic', 'residuals'
        'quadratic': dict with 'coeffs', 'se', 'r2', 'aic', 'bic', 'residuals'
        'f_test': dict with 'f_stat', 'p_value', 'significant'
        'aic_comparison': dict with 'delta_aic', 'preferred_model'
        'bic_comparison': dict with 'delta_bic', 'preferred_model'
        'recommendation': str summarizing results
        
    Example
    -------
    >>> result = test_rate_temperature_nonlinearity(rate, temperature)
    >>> print(result['recommendation'])
    >>> print(f"F-test p-value: {result['f_test']['p_value']:.4f}")
    
    Notes
    -----
    - F-test: Tests H₀: quadratic term = 0. Significant p-value suggests nonlinearity.
    - AIC/BIC: Lower is better. ΔAIC > 2 suggests meaningful improvement.
    - For autocorrelated residuals (likely with time series), p-values may be 
      anti-conservative. Consider this a screening test.
    """
    from scipy import stats
    
    # Remove NaN values
    valid = ~(np.isnan(rate) | np.isnan(temperature))
    rate = rate[valid]
    temperature = temperature[valid]
    n = len(rate)
    
    if rate_sigma is not None:
        rate_sigma = rate_sigma[valid]
        weights = 1.0 / rate_sigma**2
    else:
        weights = np.ones(n)
    
    # Normalize weights
    weights = weights / weights.sum() * n
    
    # --- Linear fit: rate = b₁×T + b₀ ---
    X_lin = np.column_stack([temperature, np.ones(n)])
    W = np.diag(weights)
    
    # Weighted least squares: (X'WX)^(-1) X'Wy
    XtWX_lin = X_lin.T @ W @ X_lin
    XtWy_lin = X_lin.T @ W @ rate
    coeffs_lin = np.linalg.solve(XtWX_lin, XtWy_lin)
    
    # Predictions and residuals
    pred_lin = X_lin @ coeffs_lin
    resid_lin = rate - pred_lin
    
    # Weighted SSR and SST
    ssr_lin = np.sum(weights * resid_lin**2)
    sst = np.sum(weights * (rate - np.average(rate, weights=weights))**2)
    r2_lin = 1 - ssr_lin / sst
    
    # Standard errors
    k_lin = 2  # number of parameters
    mse_lin = ssr_lin / (n - k_lin)
    cov_lin = mse_lin * np.linalg.inv(XtWX_lin)
    se_lin = np.sqrt(np.diag(cov_lin))
    
    # AIC and BIC (using RSS, assuming Gaussian errors)
    # AIC = n*log(RSS/n) + 2k
    # BIC = n*log(RSS/n) + k*log(n)
    aic_lin = n * np.log(ssr_lin / n) + 2 * k_lin
    bic_lin = n * np.log(ssr_lin / n) + k_lin * np.log(n)
    
    # --- Quadratic fit: rate = a₂×T² + a₁×T + a₀ ---
    X_quad = np.column_stack([temperature**2, temperature, np.ones(n)])
    
    XtWX_quad = X_quad.T @ W @ X_quad
    XtWy_quad = X_quad.T @ W @ rate
    coeffs_quad = np.linalg.solve(XtWX_quad, XtWy_quad)
    
    # Predictions and residuals
    pred_quad = X_quad @ coeffs_quad
    resid_quad = rate - pred_quad
    
    # Weighted SSR
    ssr_quad = np.sum(weights * resid_quad**2)
    r2_quad = 1 - ssr_quad / sst
    
    # Standard errors
    k_quad = 3
    mse_quad = ssr_quad / (n - k_quad)
    cov_quad = mse_quad * np.linalg.inv(XtWX_quad)
    se_quad = np.sqrt(np.diag(cov_quad))
    
    # AIC and BIC
    aic_quad = n * np.log(ssr_quad / n) + 2 * k_quad
    bic_quad = n * np.log(ssr_quad / n) + k_quad * np.log(n)
    
    # --- F-test for nested models ---
    # H₀: quadratic coefficient = 0 (linear model is sufficient)
    # F = [(SSR_reduced - SSR_full) / (df_reduced - df_full)] / [SSR_full / df_full]
    df_lin = n - k_lin
    df_quad = n - k_quad
    
    f_stat = ((ssr_lin - ssr_quad) / (df_lin - df_quad)) / (ssr_quad / df_quad)
    p_value = 1 - stats.f.cdf(f_stat, df_lin - df_quad, df_quad)
    
    # --- T-test on quadratic coefficient ---
    t_stat_quad = coeffs_quad[0] / se_quad[0]
    p_value_t = 2 * (1 - stats.t.cdf(np.abs(t_stat_quad), df_quad))
    
    # --- AIC/BIC comparison ---
    delta_aic = aic_lin - aic_quad  # positive = quadratic is better
    delta_bic = bic_lin - bic_quad
    
    # --- Build recommendation ---
    reasons = []
    prefer_quadratic = 0
    
    if p_value < alpha:
        reasons.append(f"F-test significant (p={p_value:.4f})")
        prefer_quadratic += 1
    else:
        reasons.append(f"F-test not significant (p={p_value:.4f})")
    
    if delta_aic > 2:
        reasons.append(f"AIC favors quadratic (ΔAIC={delta_aic:.1f})")
        prefer_quadratic += 1
    elif delta_aic < -2:
        reasons.append(f"AIC favors linear (ΔAIC={delta_aic:.1f})")
    else:
        reasons.append(f"AIC inconclusive (ΔAIC={delta_aic:.1f})")
    
    if delta_bic > 2:
        reasons.append(f"BIC favors quadratic (ΔBIC={delta_bic:.1f})")
        prefer_quadratic += 1
    elif delta_bic < -2:
        reasons.append(f"BIC favors linear (ΔBIC={delta_bic:.1f})")
    else:
        reasons.append(f"BIC inconclusive (ΔBIC={delta_bic:.1f})")
    
    r2_improvement = r2_quad - r2_lin
    reasons.append(f"R² improvement: {r2_improvement:.4f} ({r2_lin:.3f} → {r2_quad:.3f})")
    
    if prefer_quadratic >= 2:
        recommendation = "QUADRATIC model preferred. " + "; ".join(reasons)
    elif prefer_quadratic == 0:
        recommendation = "LINEAR model preferred. " + "; ".join(reasons)
    else:
        recommendation = "INCONCLUSIVE - consider additional diagnostics. " + "; ".join(reasons)
    
    return {
        'linear': {
            'coeffs': coeffs_lin,  # [b₁, b₀]
            'se': se_lin,
            'r2': r2_lin,
            'aic': aic_lin,
            'bic': bic_lin,
            'residuals': resid_lin,
            'predictions': pred_lin
        },
        'quadratic': {
            'coeffs': coeffs_quad,  # [a₂, a₁, a₀]
            'se': se_quad,
            'r2': r2_quad,
            'aic': aic_quad,
            'bic': bic_quad,
            'residuals': resid_quad,
            'predictions': pred_quad,
            't_stat': t_stat_quad,
            'p_value_t': p_value_t
        },
        'f_test': {
            'f_stat': f_stat,
            'p_value': p_value,
            'df1': df_lin - df_quad,
            'df2': df_quad,
            'significant': p_value < alpha
        },
        'aic_comparison': {
            'delta_aic': delta_aic,
            'preferred_model': 'quadratic' if delta_aic > 2 else ('linear' if delta_aic < -2 else 'inconclusive')
        },
        'bic_comparison': {
            'delta_bic': delta_bic,
            'preferred_model': 'quadratic' if delta_bic > 2 else ('linear' if delta_bic < -2 else 'inconclusive')
        },
        'n_observations': n,
        'recommendation': recommendation
    }
    
# =============================================================================
# MODULE INFO
# =============================================================================

__all__ = [
    # Data structures
    'KinematicsResult',
    'DOLSResult',
    # Preprocessing
    'resample_to_monthly',
    'merge_multiresolution_data', 
    'align_to_baseline',
    'harmonize_baseline',
    'compute_thermodynamic_signal',
    # Kinematics
    'compute_kinematics',
    'compute_kinematics_multibandwidth',
    # Calibration
    'test_rate_temperature_nonlinearity',
    'calibrate_alpha_dols',
]

if __name__ == "__main__":
    print("Sea Level Rise Analysis Module")
    print("=" * 50)
    print("\nPreprocessing:")
    print("  - resample_to_monthly()")
    print("  - merge_multiresolution_data()")
    print("  - align_to_baseline()")
    print("  - harmonize_baseline()")
    print("  - compute_thermodynamic_signal()")
    print("\nKinematics:")
    print("  - compute_kinematics()")
    print("  - compute_kinematics_multibandwidth()")
    print("\nCalibration:")
    print("  - test_rate_temperature_nonlinearity")
    print("  - calibrate_alpha_dols()")
