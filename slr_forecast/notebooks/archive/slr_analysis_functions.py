"""
Sea Level Analysis Functions: Resampling and Local Regression
==============================================================

This module provides functions for:
1. Resampling time series between different temporal resolutions with proper
   uncertainty quantification
2. Local polynomial regression for estimating rates and accelerations

These functions are designed for sea level and climate time series analysis,
with particular attention to statistical rigor required for peer-reviewed
research.

Key methodological considerations:
- Interpolation does not create new information; uncertainties must be inflated
- Mixed-resolution data requires careful weighting in regression
- Local polynomial regression provides time-varying rate/acceleration estimates

References
----------
Cleveland, W. S., & Devlin, S. J. (1988). Locally Weighted Regression: An 
    Approach to Regression Analysis by Local Fitting. Journal of the American 
    Statistical Association, 83(403), 596–610.
    
Fan, J., & Gijbels, I. (1996). Local Polynomial Modelling and Its Applications.
    Chapman & Hall/CRC.

Author: Climate Analysis Toolkit
"""

import pandas as pd
import numpy as np
import statsmodels.api as sm
from typing import Optional, Tuple, Union
from datetime import datetime


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
       independent measurements
    2. Standard errors computed from interpolated data will be artificially
       small by a factor of approximately √n, where n is the interpolation
       factor (n=12 for annual→monthly)
    3. Degrees of freedom in statistical tests should reflect the original
       sample size, not the interpolated sample size
    
    This function addresses these issues by:
    - Tracking which points are original vs. interpolated
    - Inflating uncertainties for interpolated points
    - Providing weights for use in subsequent weighted regression
    
    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame with datetime index. Can be at any temporal resolution
        (annual, quarterly, etc.)
    value_col : str
        Name of the column containing the values to interpolate
    unc_col : str, optional
        Name of the column containing uncertainty estimates. If provided,
        uncertainties will be interpolated and optionally inflated.
    method : str, default 'time'
        Interpolation method passed to pandas interpolate(). Options include:
        - 'time': Linear interpolation weighted by time distance (recommended)
        - 'linear': Simple linear interpolation by index position
        - 'spline': Cubic spline (use with caution; can introduce artifacts)
        - 'pchip': Piecewise Cubic Hermite Interpolating Polynomial
    inflate_uncertainty : bool, default True
        If True, inflate uncertainties for interpolated points to account for
        the fact that interpolation does not add information. The inflation
        factor is √(months_per_original_interval).
    track_provenance : bool, default True
        If True, add columns tracking data provenance:
        - 'is_original': Boolean flag for original vs. interpolated points
        - 'weight': Statistical weight (1.0 for original, 1/n for interpolated)
        - 'effective_n': Effective sample size contribution
    
    Returns
    -------
    pd.DataFrame
        Monthly-resolution DataFrame with columns:
        - Original value column (interpolated)
        - Original uncertainty column (if provided, with inflation)
        - '{unc_col}_original': Pre-inflation uncertainty (if unc_col provided)
        - 'is_original': True for original data points (if track_provenance)
        - 'weight': Statistical weight for regression (if track_provenance)
        - 'interpolation_factor': Local interpolation ratio (if track_provenance)
    
    Examples
    --------
    >>> # Resample annual sea level data to monthly
    >>> df_monthly = resample_to_monthly(
    ...     df_annual, 
    ...     value_col='gmsl',
    ...     unc_col='gmsl_unc',
    ...     inflate_uncertainty=True
    ... )
    >>> 
    >>> # Use weights in subsequent regression
    >>> model = sm.WLS(y, X, weights=df_monthly['weight']).fit()
    
    Notes
    -----
    For sea level budget analysis, it is often preferable to:
    1. Keep data at native resolution when possible
    2. Perform separate analyses for different temporal resolution datasets
    3. Use the weights provided by this function in weighted least squares
    
    The uncertainty inflation assumes that the original uncertainty represents
    the standard error of the measurement. If the original uncertainty 
    represents something else (e.g., a confidence interval), adjust accordingly.
    
    See Also
    --------
    merge_multiresolution_data : Alternative approach preserving native resolution
    compute_local_regression : Weighted local regression using these outputs
    """
    df = df.copy()
    
    # Store original timestamps for provenance tracking
    original_times = df.index.copy()
    original_resolution = _estimate_resolution_months(df.index)
    
    # Create monthly date range spanning the data
    monthly_index = pd.date_range(
        df.index.min().replace(day=1),  # Start at first of month
        df.index.max() + pd.DateOffset(months=1),
        freq='MS'  # Month Start frequency
    )
    
    # Reindex to include both original and monthly timestamps
    combined_index = df.index.union(monthly_index)
    df_reindexed = df.reindex(combined_index).sort_index()
    
    # Perform interpolation on value column
    df_reindexed[value_col] = df_reindexed[value_col].interpolate(method=method)
    
    # Handle uncertainty column if provided
    if unc_col and unc_col in df.columns:
        # Store original uncertainty before inflation
        df_reindexed[f'{unc_col}_original'] = df_reindexed[unc_col].copy()
        
        # Interpolate uncertainty using nearest-neighbor
        # (linear interpolation of uncertainty is not statistically meaningful)
        df_reindexed[unc_col] = df_reindexed[unc_col].interpolate(method='nearest')
        df_reindexed[f'{unc_col}_original'] = df_reindexed[f'{unc_col}_original'].interpolate(method='nearest')
    
    # Keep only the monthly timestamps
    df_monthly = df_reindexed.loc[monthly_index].copy()
    
    if track_provenance:
        # Identify original vs. interpolated points
        # A point is "original" if it falls within 15 days of an original timestamp
        df_monthly['is_original'] = False
        for orig_time in original_times:
            time_diff = np.abs((df_monthly.index - orig_time).days)
            df_monthly.loc[time_diff <= 15, 'is_original'] = True
        
        # Calculate interpolation factor (how many monthly points per original)
        df_monthly['interpolation_factor'] = original_resolution
        df_monthly.loc[df_monthly['is_original'], 'interpolation_factor'] = 1.0
        
        # Statistical weight: interpolated points contribute less information
        # Weight = 1/interpolation_factor, so 12 monthly points from 1 annual
        # observation collectively have the same weight as the original
        df_monthly['weight'] = 1.0 / df_monthly['interpolation_factor']
        
        # Effective sample size contribution
        df_monthly['effective_n'] = df_monthly['weight']
    
    # Inflate uncertainties for interpolated points
    if inflate_uncertainty and unc_col and unc_col in df.columns:
        if track_provenance:
            # Inflate by sqrt(interpolation_factor) for interpolated points
            inflation = np.sqrt(df_monthly['interpolation_factor'])
            df_monthly[unc_col] = df_monthly[f'{unc_col}_original'] * inflation
        else:
            # If not tracking provenance, inflate all by sqrt of estimated resolution
            df_monthly[unc_col] = df_monthly[unc_col] * np.sqrt(original_resolution)
    
    df_monthly.index.name = 'time'
    
    return df_monthly


def _estimate_resolution_months(index: pd.DatetimeIndex) -> float:
    """
    Estimate the typical temporal resolution of a datetime index in months.
    
    Parameters
    ----------
    index : pd.DatetimeIndex
        DateTime index to analyze
        
    Returns
    -------
    float
        Estimated resolution in months (e.g., 12.0 for annual, 1.0 for monthly)
    """
    if len(index) < 2:
        return 1.0
    
    # Calculate median time difference in days
    diffs = np.diff(index.values).astype('timedelta64[D]').astype(float)
    median_days = np.median(diffs)
    
    # Convert to months (approximate)
    months = median_days / 30.44  # Average days per month
    
    # Round to common resolutions
    if months < 1.5:
        return 1.0   # Monthly
    elif months < 4.5:
        return 3.0   # Quarterly
    elif months < 9:
        return 6.0   # Semi-annual
    else:
        return 12.0  # Annual


def compute_local_regression(
    time: np.ndarray,
    value: np.ndarray,
    uncertainty: np.ndarray,
    span_years: float,
    weights: Optional[np.ndarray] = None,
    min_effective_obs: int = 10,
    kernel: str = 'tricube',
    return_fit: bool = False
) -> pd.DataFrame:
    """
    Compute time-varying rates and accelerations using local quadratic regression.
    
    METHODOLOGY
    -----------
    This function implements locally weighted polynomial regression (LOESS/LOWESS
    variant) with a quadratic (degree-2) polynomial. At each time point t₀, a
    weighted least squares regression is performed:
    
        y(t) = β₀ + β₁(t - t₀) + ½β₂(t - t₀)² + ε
    
    The coefficients have physical interpretations:
        β₀ = Value at time t₀
        β₁ = Rate of change (first derivative) at t₀
        β₂ = Acceleration (second derivative) at t₀
    
    Weights are assigned based on three factors:
    1. Kernel weight: Points closer to t₀ receive higher weight (tricube kernel)
    2. Measurement uncertainty: Points with smaller σ receive higher weight (1/σ²)
    3. Data provenance: Interpolated points receive reduced weight
    
    The combined weight for observation i at target point t₀ is:
    
        w_i = K((t_i - t₀)/h) × (1/σ_i²) × p_i
    
    where:
        K(u) = (1 - |u|³)³ for |u| ≤ 1, else 0  (tricube kernel)
        h = bandwidth (span_years)
        σ_i = measurement uncertainty
        p_i = provenance weight (1.0 for original, 1/n for interpolated)
    
    KERNEL FUNCTIONS
    ----------------
    The kernel function determines how weight decreases with distance from the
    target point. The tricube kernel is recommended because:
    - Compact support: Points beyond the bandwidth receive zero weight
    - Smooth: Continuous second derivative, yielding smooth rate estimates
    - Efficient: Robust to outliers at the edge of the window
    
    Available kernels:
        'tricube': K(u) = (1 - |u|³)³  [default, recommended]
        'gaussian': K(u) = exp(-u²/2)   [infinite support, very smooth]
        'epanechnikov': K(u) = (1 - u²) [optimal for density estimation]
    
    BANDWIDTH SELECTION
    -------------------
    The bandwidth (span_years) controls the bias-variance tradeoff:
    
        Smaller bandwidth → Higher variance, lower bias
            - Captures rapid changes
            - Noisier estimates
            - Suitable for detecting recent acceleration
        
        Larger bandwidth → Lower variance, higher bias  
            - Smoother estimates
            - May miss rapid transitions
            - Suitable for long-term trends
    
    Recommended values for sea level analysis:
        - 10-15 years: Resolving decadal variability (noisy)
        - 20-30 years: Climate-scale signals (recommended default)
        - 40-50 years: Long-term acceleration (very smooth)
    
    Parameters
    ----------
    time : np.ndarray
        Time values as decimal years (e.g., 2020.5 for mid-2020).
        Must be monotonically increasing.
    value : np.ndarray
        Observed values (e.g., sea level in mm, temperature in °C).
        Same length as time.
    uncertainty : np.ndarray
        Standard uncertainty (1σ) for each observation.
        Used as weights in regression: w ∝ 1/σ².
        Same length as time.
    span_years : float
        Bandwidth of the local regression window in years.
        Points within this distance of the target receive non-zero kernel weight.
    weights : np.ndarray, optional
        Additional weights for each observation, typically from data provenance
        (e.g., interpolated vs. original data). Values should be in (0, 1].
        If None, all observations receive weight 1.0.
    min_effective_obs : int, default 10
        Minimum effective number of observations required for regression.
        The effective count accounts for provenance weights:
            n_eff = Σ(weights) for points with non-zero kernel weight
        If n_eff < min_effective_obs, the estimate is set to NaN.
        For quadratic regression, minimum ~10 recommended for stable estimates.
    kernel : str, default 'tricube'
        Kernel function for distance weighting. Options:
        - 'tricube': (1 - |u|³)³, compact support, smooth [recommended]
        - 'gaussian': exp(-u²/2), infinite support, very smooth
        - 'epanechnikov': (1 - u²), compact support, optimal MSE
    return_fit : bool, default False
        If True, also return the fitted values at each time point.
    
    Returns
    -------
    pd.DataFrame
        DataFrame indexed by time with columns:
        - 'rate': Estimated rate of change (dy/dt) at each time point
        - 'rate_se': Standard error of rate estimate
        - 'accel': Estimated acceleration (d²y/dt²) at each time point
        - 'accel_se': Standard error of acceleration estimate
        - 'n_effective': Effective number of observations used at each point
        - 'fitted': Fitted values at each point (if return_fit=True)
        
        Units: rate is in [value_units]/year, accel is in [value_units]/year²
        For sea level in mm: rate in mm/yr, acceleration in mm/yr²
    
    Examples
    --------
    >>> # Basic usage with annual sea level data
    >>> results = compute_local_regression(
    ...     time=decimal_years,
    ...     value=gmsl_mm,
    ...     uncertainty=gmsl_unc_mm,
    ...     span_years=30
    ... )
    >>> print(f"Current rate: {results['rate'].iloc[-1]:.2f} ± "
    ...       f"{results['rate_se'].iloc[-1]:.2f} mm/yr")
    
    >>> # With interpolated data - use provenance weights
    >>> df_monthly = resample_to_monthly(df_annual, 'gmsl', 'gmsl_unc')
    >>> results = compute_local_regression(
    ...     time=df_monthly['decimal_year'].values,
    ...     value=df_monthly['gmsl'].values,
    ...     uncertainty=df_monthly['gmsl_unc'].values,
    ...     span_years=30,
    ...     weights=df_monthly['weight'].values  # From resampling function
    ... )
    
    Notes
    -----
    Statistical interpretation of outputs:
    
    1. Rate (β₁): The instantaneous rate of change at each time point.
       For sea level, this represents how fast sea level is rising at that
       moment, in mm/year.
    
    2. Acceleration (β₂): The rate of change of the rate. Positive acceleration
       means the rate is increasing over time. For sea level, acceleration
       of 0.1 mm/yr² means the rate increases by 0.1 mm/yr each year.
    
    3. Standard errors: Derived from the weighted least squares covariance
       matrix. These assume the model is correctly specified and residuals
       are uncorrelated. For time series with autocorrelation, consider
       inflating standard errors or using HAC (Newey-West) corrections.
    
    4. Edge effects: Estimates near the beginning and end of the record
       have higher uncertainty due to one-sided data availability. The
       standard errors account for this, but interpretation should be
       cautious within span_years/2 of the record boundaries.
    
    References
    ----------
    Cleveland, W. S. (1979). Robust Locally Weighted Regression and Smoothing
        Scatterplots. Journal of the American Statistical Association, 74(368),
        829–836.
    
    Church, J. A., & White, N. J. (2006). A 20th century acceleration in global
        sea-level rise. Geophysical Research Letters, 33, L01602.
        https://doi.org/10.1029/2005GL024826
    
    Dangendorf, S., et al. (2019). Persistent acceleration in global sea-level
        rise since the 1960s. Nature Climate Change, 9, 705–710.
        https://doi.org/10.1038/s41558-019-0531-8
    
    See Also
    --------
    resample_to_monthly : Prepare data with proper uncertainty inflation
    """
    # Input validation
    time = np.asarray(time, dtype=float)
    value = np.asarray(value, dtype=float)
    uncertainty = np.asarray(uncertainty, dtype=float)
    
    n = len(time)
    if not (len(value) == n and len(uncertainty) == n):
        raise ValueError("time, value, and uncertainty must have the same length")
    
    if weights is None:
        weights = np.ones(n)
    else:
        weights = np.asarray(weights, dtype=float)
        if len(weights) != n:
            raise ValueError("weights must have the same length as time")
    
    # Validate uncertainty values
    if np.any(uncertainty <= 0):
        raise ValueError("All uncertainty values must be positive")
    
    # Initialize output arrays
    results = {
        'rate': np.full(n, np.nan),
        'rate_se': np.full(n, np.nan),
        'accel': np.full(n, np.nan),
        'accel_se': np.full(n, np.nan),
        'n_effective': np.full(n, np.nan)
    }
    if return_fit:
        results['fitted'] = np.full(n, np.nan)
    
    # Bandwidth
    h = span_years
    
    # Select kernel function
    kernel_funcs = {
        'tricube': lambda u: np.where(np.abs(u) <= 1, (1 - np.abs(u)**3)**3, 0),
        'gaussian': lambda u: np.exp(-0.5 * u**2),
        'epanechnikov': lambda u: np.where(np.abs(u) <= 1, 1 - u**2, 0)
    }
    if kernel not in kernel_funcs:
        raise ValueError(f"Unknown kernel '{kernel}'. Choose from: {list(kernel_funcs.keys())}")
    kernel_func = kernel_funcs[kernel]
    
    # Main loop: compute local regression at each time point
    for i in range(n):
        t0 = time[i]
        
        # Compute distance from target point, normalized by bandwidth
        u = (time - t0) / h
        
        # Compute kernel weights
        kernel_weights = kernel_func(u)
        
        # Combined weights: kernel × (1/σ²) × provenance_weight
        # The 1/σ² weighting is standard for heteroscedastic WLS
        combined_weights = kernel_weights * (1 / uncertainty**2) * weights
        
        # Identify observations with non-negligible weight
        mask = combined_weights > 1e-12
        
        # Check effective sample size
        n_eff = np.sum(weights[mask])
        results['n_effective'][i] = n_eff
        
        if n_eff < min_effective_obs:
            # Insufficient data for reliable quadratic fit
            continue
        
        # Extract data within the window
        t_local = time[mask]
        y_local = value[mask]
        w_local = combined_weights[mask]
        
        # Center time at target point for numerical stability
        dt = t_local - t0
        
        # Design matrix for quadratic polynomial:
        # y = β₀ + β₁(t-t₀) + β₂(t-t₀)²/2
        # The factor of 1/2 on the quadratic term means β₂ directly
        # represents the second derivative (acceleration)
        X = np.column_stack([
            np.ones(len(dt)),    # Intercept (β₀)
            dt,                   # Linear term (β₁ = rate)
            0.5 * dt**2          # Quadratic term (β₂ = acceleration)
        ])
        
        # Perform weighted least squares regression
        try:
            model = sm.WLS(y_local, X, weights=w_local).fit()
            
            # Extract rate and acceleration with standard errors
            results['rate'][i] = model.params[1]
            results['rate_se'][i] = model.bse[1]
            results['accel'][i] = model.params[2]
            results['accel_se'][i] = model.bse[2]
            
            if return_fit:
                results['fitted'][i] = model.params[0]  # Value at t0
                
        except Exception:
            # Regression failed (e.g., singular matrix)
            continue
    
    # Create output DataFrame
    # Convert time to datetime for index
    time_index = pd.to_datetime([_decimal_year_to_datetime(t) for t in time])
    df_results = pd.DataFrame(results, index=time_index)
    df_results.index.name = 'time'
    
    # Add decimal_year as a column for convenience
    df_results['decimal_year'] = time
    
    return df_results


def _decimal_year_to_datetime(decimal_year: float) -> datetime:
    """Convert decimal year to datetime object."""
    year = int(decimal_year)
    rem = decimal_year - year
    base = datetime(year, 1, 1)
    next_year = datetime(year + 1, 1, 1)
    return base + (next_year - base) * rem


def merge_multiresolution_data(
    datasets: list,
    value_col: str = 'gmsl',
    unc_col: Optional[str] = 'gmsl_unc',
    resolution_months: Optional[list] = None
) -> pd.DataFrame:
    """
    Merge multiple datasets with different temporal resolutions without interpolation.
    
    This function provides an alternative to resampling that preserves the native
    resolution of each dataset. Instead of interpolating, it assigns appropriate
    statistical weights based on each dataset's temporal resolution.
    
    METHODOLOGY
    -----------
    For datasets with different temporal resolutions, the information content
    of each observation differs. An annual average contains information from
    12 months, while a monthly observation contains information from 1 month.
    
    This function assigns weights proportional to the temporal coverage:
        weight = resolution_months / reference_resolution
    
    For example, if the reference is monthly (1 month):
        - Monthly observations: weight = 1.0
        - Annual observations: weight = 12.0
    
    These weights can be used in weighted regression to properly account for
    the different information content of each observation.
    
    Parameters
    ----------
    datasets : list of pd.DataFrame
        List of DataFrames to merge. Each must have a datetime index and
        contain the specified value column.
    value_col : str, default 'gmsl'
        Name of the value column present in all datasets.
    unc_col : str, optional, default 'gmsl_unc'
        Name of the uncertainty column. If None or not present in a dataset,
        uncertainty is set to NaN for that dataset.
    resolution_months : list of float, optional
        Temporal resolution of each dataset in months. If None, resolution
        is estimated automatically from the data.
    
    Returns
    -------
    pd.DataFrame
        Merged DataFrame with columns:
        - value_col: Merged values
        - unc_col: Merged uncertainties (if provided)
        - 'resolution': Temporal resolution in months
        - 'weight': Statistical weight for regression
        - 'source': Index of source dataset (0, 1, 2, ...)
    
    Examples
    --------
    >>> # Merge annual reconstruction with monthly altimetry
    >>> df_merged = merge_multiresolution_data(
    ...     datasets=[df_frederikse, df_altimetry],
    ...     value_col='gmsl',
    ...     unc_col='gmsl_unc',
    ...     resolution_months=[12, 1]  # Annual, Monthly
    ... )
    
    Notes
    -----
    When datasets overlap in time, this function keeps all observations.
    For analysis, you may want to:
    1. Use both with appropriate weights
    2. Prefer one dataset in overlap periods
    3. Compute a weighted average
    
    See Also
    --------
    resample_to_monthly : Alternative approach using interpolation
    compute_local_regression : Use weights in local polynomial regression
    """
    if len(datasets) == 0:
        raise ValueError("At least one dataset must be provided")
    
    merged_parts = []
    
    for i, df in enumerate(datasets):
        df_part = pd.DataFrame(index=df.index)
        df_part[value_col] = df[value_col].values
        
        # Handle uncertainty
        if unc_col and unc_col in df.columns:
            df_part[unc_col] = df[unc_col].values
        else:
            df_part[unc_col] = np.nan
        
        # Determine resolution
        if resolution_months is not None:
            res = resolution_months[i]
        else:
            res = _estimate_resolution_months(df.index)
        
        df_part['resolution'] = res
        df_part['weight'] = res  # Weight proportional to temporal coverage
        df_part['source'] = i
        
        merged_parts.append(df_part)
    
    # Concatenate and sort by time
    df_merged = pd.concat(merged_parts).sort_index()
    df_merged.index.name = 'time'
    
    return df_merged


def compute_rate_of_change(
    df: pd.DataFrame,
    value_col: str,
    unc_col: Optional[str] = None,
    window_years: float = 1.0,
    method: str = 'centered'
) -> pd.DataFrame:
    """
    Compute simple rate of change (first derivative) using finite differences.
    
    This is a simpler alternative to local polynomial regression when only
    the rate (not acceleration) is needed, or for comparison purposes.
    
    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame with datetime index.
    value_col : str
        Name of the value column.
    unc_col : str, optional
        Name of the uncertainty column.
    window_years : float, default 1.0
        Time window for computing differences (in years).
    method : str, default 'centered'
        Differencing method:
        - 'centered': (y[t+Δ] - y[t-Δ]) / (2Δ)
        - 'forward': (y[t+Δ] - y[t]) / Δ
        - 'backward': (y[t] - y[t-Δ]) / Δ
    
    Returns
    -------
    pd.DataFrame
        DataFrame with columns:
        - 'rate': Estimated rate of change
        - 'rate_unc': Uncertainty in rate (if unc_col provided)
    """
    from slr_data_readers import datetime_to_decimal_year
    
    df = df.copy()
    
    # Convert datetime index to decimal years
    decimal_years = np.array([datetime_to_decimal_year(t) for t in df.index])
    values = df[value_col].values
    
    n = len(df)
    rates = np.full(n, np.nan)
    rate_uncs = np.full(n, np.nan)
    
    for i in range(n):
        t0 = decimal_years[i]
        
        if method == 'centered':
            # Find points closest to t0 - Δ and t0 + Δ
            mask_before = decimal_years < t0
            mask_after = decimal_years > t0
            
            if not (np.any(mask_before) and np.any(mask_after)):
                continue
            
            # Find closest points within window
            idx_before = np.argmin(np.abs(decimal_years[mask_before] - (t0 - window_years/2)))
            idx_after = np.argmin(np.abs(decimal_years[mask_after] - (t0 + window_years/2)))
            
            # Convert to absolute indices
            idx_before = np.where(mask_before)[0][idx_before]
            idx_after = np.where(mask_after)[0][idx_after]
            
            dt = decimal_years[idx_after] - decimal_years[idx_before]
            dy = values[idx_after] - values[idx_before]
            
            if dt > 0:
                rates[i] = dy / dt
                
                if unc_col and unc_col in df.columns:
                    unc = df[unc_col].values
                    rate_uncs[i] = np.sqrt(unc[idx_after]**2 + unc[idx_before]**2) / dt
    
    result = pd.DataFrame({
        'rate': rates,
        'rate_unc': rate_uncs
    }, index=df.index)
    
    return result
