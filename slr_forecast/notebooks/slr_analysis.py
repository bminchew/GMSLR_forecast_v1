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
    Unified container for Dynamic OLS calibration results.

    Supports linear (order=1), quadratic (order=2), and cubic (order=3)
    models, with an optional SAOD (volcanic) forcing term.

    Physical parameters are stored in **rate-model** form so that the
    covariance matrix is already transformed and ready for use in
    projections — no manual ``diag([2,1,1])`` transform is needed.

    Rate model (order=2 example):
        dH/dt = dα/dT × T² + α₀ × T + trend  (+ γ_saod × SAOD if included)

    Attributes
    ----------
    order : int
        Polynomial order (1, 2, or 3).
    physical_coefficients : np.ndarray
        Rate-model coefficients [a_n, …, a_1, trend].
        order=1: [alpha0, trend]
        order=2: [dalpha_dT, alpha0, trend]
        order=3: [d2alpha_dT2, dalpha_dT, alpha0, trend]
    physical_covariance : np.ndarray
        Covariance matrix of ``physical_coefficients``.
    physical_se : np.ndarray
        Standard errors (sqrt of diagonal of ``physical_covariance``).
    gamma_saod, gamma_saod_se : float or None
        SAOD coefficient and its SE (None if SAOD not included).
    alpha0, alpha0_se : float
        Linear sensitivity and SE (always present).
    dalpha_dT, dalpha_dT_se : float or None
        Quadratic sensitivity and SE (order ≥ 2).
    d2alpha_dT2, d2alpha_dT2_se : float or None
        Cubic sensitivity and SE (order ≥ 3).
    trend, trend_se : float
        Background linear trend and SE.
    regression_coefficients, regression_covariance : np.ndarray
        Raw regression-level parameters and HAC covariance.
    r2, r2_adj : float
        R² and adjusted R².
    aic, bic : float
        Information criteria.
    n_obs, n_lags : int
        Number of observations and lead/lag terms.
    has_saod : bool
        Whether SAOD was included as a regressor.
    residuals, fitted, time : np.ndarray
        Regression diagnostics arrays.
    model : object
        Full statsmodels RegressionResults object.
    """
    order: int
    physical_coefficients: np.ndarray
    physical_covariance: np.ndarray
    physical_se: np.ndarray
    # SAOD
    gamma_saod: Optional[float]
    gamma_saod_se: Optional[float]
    # Named accessors
    alpha0: float
    alpha0_se: float
    dalpha_dT: Optional[float]
    dalpha_dT_se: Optional[float]
    d2alpha_dT2: Optional[float]
    d2alpha_dT2_se: Optional[float]
    trend: float
    trend_se: float
    # Regression level
    regression_coefficients: np.ndarray
    regression_covariance: np.ndarray
    # Fit statistics
    r2: float
    r2_adj: float
    aic: float
    bic: float
    n_obs: int
    n_lags: int
    has_saod: bool
    # Arrays
    residuals: np.ndarray
    fitted: np.ndarray
    time: np.ndarray
    model: object

    # ----- backward-compat alias (old DOLSQuadraticResult stored raw) -----
    @property
    def covariance(self) -> np.ndarray:
        """Alias for regression_covariance (backward compatibility)."""
        return self.regression_covariance

    @property
    def coefficients(self) -> np.ndarray:
        """Alias for regression_coefficients (backward compatibility)."""
        return self.regression_coefficients

    def __repr__(self) -> str:
        lines = [f"DOLSResult(order={self.order}, saod={self.has_saod})"]
        if self.order >= 3:
            lines.append(
                f"  d²α/dT² = {self.d2alpha_dT2:.6f} ± {self.d2alpha_dT2_se:.6f}"
            )
        if self.order >= 2:
            lines.append(
                f"  dα/dT   = {self.dalpha_dT:.6f} ± {self.dalpha_dT_se:.6f}"
            )
        lines.append(f"  α₀      = {self.alpha0:.6f} ± {self.alpha0_se:.6f}")
        lines.append(f"  trend   = {self.trend:.6f} ± {self.trend_se:.6f}")
        if self.has_saod:
            lines.append(
                f"  γ_saod  = {self.gamma_saod:.6f} ± {self.gamma_saod_se:.6f}"
            )
        lines.append(
            f"  R² = {self.r2:.4f}, R²_adj = {self.r2_adj:.4f}"
        )
        lines.append(
            f"  AIC = {self.aic:.1f}, BIC = {self.bic:.1f}"
        )
        lines.append(f"  n_obs = {self.n_obs}, n_lags = {self.n_lags}")
        return "\n".join(lines)

    def predict_rate(self, temperature: np.ndarray) -> np.ndarray:
        """
        Predict SLR rate at given temperatures.

        rate = a_n × T^n + … + a_1 × T + trend

        Does **not** include the SAOD term (which requires an SAOD value).
        """
        temperature = np.asarray(temperature, dtype=np.float64)
        rate = np.full_like(temperature, self.trend)
        # physical_coefficients = [a_n, …, a_1, trend]
        # iterate from highest power down to linear
        poly_coeffs = self.physical_coefficients[:-1]  # exclude trend
        n_poly = len(poly_coeffs)
        for i, a in enumerate(poly_coeffs):
            power = n_poly - i
            rate = rate + a * temperature ** power
        return rate

    def predict_rate_ci(
        self, temperature: np.ndarray, confidence: float = 0.95
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Predict SLR rate with confidence interval.

        Returns (lower, upper) bounds using the delta method.
        """
        from scipy import stats

        temperature = np.asarray(temperature, dtype=np.float64)
        rate = self.predict_rate(temperature)

        # Jacobian: d(rate)/d(physical_coefficients)
        # physical_coefficients = [a_n, …, a_1, trend]
        n_phys = len(self.physical_coefficients)
        n_poly = n_phys - 1  # exclude trend
        n_pts = len(temperature)
        J = np.ones((n_pts, n_phys))
        for i in range(n_poly):
            power = n_poly - i
            J[:, i] = temperature ** power
        # J[:, -1] = 1 (trend, already set)

        var_rate = np.sum((J @ self.physical_covariance) * J, axis=1)
        se_rate = np.sqrt(np.maximum(var_rate, 0.0))

        z = stats.norm.ppf(1 - (1 - confidence) / 2)
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
    
    # Store metadata (preserve existing attrs from reader functions)
    existing = df.attrs.copy()
    existing['baseline_period'] = (start_year, end_year)
    existing['baseline_mean_removed'] = baseline_mean
    df.attrs = existing

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

    # Store metadata (preserve existing attrs from reader functions)
    existing = df.attrs.copy()
    existing['harmonized_baseline'] = new_baseline
    existing['harmonized_baseline_mean'] = float(baseline_mean)
    df.attrs = existing

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

def calibrate_dols(
    sea_level: pd.Series,
    temperature: pd.Series,
    gmsl_sigma: Optional[pd.Series] = None,
    saod: Optional[pd.Series] = None,
    order: int = 2,
    n_lags: int = 2,
    hac_maxlags: Optional[int] = None,
) -> DOLSResult:
    """
    Estimate sea-level sensitivity using Dynamic Ordinary Least Squares.

    Fits a polynomial relationship between sea level and integrated
    temperature, with optional SAOD (volcanic) forcing:

        H(t) = Σₖ aₖ × ∫Tᵏ(τ)dτ  +  β×t  [+ γ×∫SAOD(τ)dτ]
               + Σᵢ δᵢ ΔT(t+i)  [+ Σᵢ ηᵢ ΔSAOD(t+i)]  + const + ε

    This implies a rate model:

        dH/dt = aₙ Tⁿ + … + a₁ T + β  [+ γ × SAOD]

    The function addresses cointegration between non-stationary sea-level
    and temperature series by augmenting the regression with leads and lags
    of ΔT (and ΔSAOD when applicable).  HAC (Newey–West) standard errors
    account for residual autocorrelation.

    Parameters
    ----------
    sea_level : pd.Series
        Sea level with datetime index (e.g. metres).
    temperature : pd.Series
        Temperature anomaly with datetime index (°C).
    gmsl_sigma : pd.Series, optional
        1-σ uncertainty of sea level.  When provided, WLS weights
        ``1 / sigma²`` are used; otherwise OLS (uniform weights).
    saod : pd.Series, optional
        Stratospheric aerosol optical depth with datetime index.
        When provided, adds ∫SAOD + SAOD leads/lags to the model.
    order : int, default 2
        Polynomial order: 1 (linear), 2 (quadratic), or 3 (cubic).
    n_lags : int, default 2
        Number of leads **and** lags of ΔT (total 2×n_lags + 1).
    hac_maxlags : int, optional
        Maximum lags for HAC covariance.  If None, uses the Newey–West
        automatic selection ``floor(4 × (n/100)^(2/9))``.

    Returns
    -------
    DOLSResult
        Unified result dataclass with physical coefficients already
        transformed to rate-model form plus HAC covariance.

    Examples
    --------
    >>> result = calibrate_dols(sl, temp, gmsl_sigma=sigma, order=2)
    >>> print(result)
    >>> coeffs = result.physical_coefficients  # [dalpha_dT, alpha0, trend]
    >>> cov    = result.physical_covariance     # 3×3, already transformed

    References
    ----------
    Stock, J. H. & Watson, M. W. (1993). Econometrica 61(4), 783–820.
    """

    if order not in (1, 2, 3):
        raise ValueError(f"order must be 1, 2, or 3, got {order}")

    # ---- 1. Align series on common datetime index ----
    # Normalise all indices to month-start so that series with different
    # day-of-month conventions (e.g. 1st vs 15th) can be matched.
    def _to_month_start(s: pd.Series) -> pd.Series:
        """Snap datetime index to the first of each month."""
        new_idx = s.index.to_period('M').to_timestamp()
        out = s.copy()
        out.index = new_idx
        # Drop any duplicates that arise from snapping (keep first)
        return out[~out.index.duplicated(keep='first')]

    sl_ms   = _to_month_start(sea_level)
    temp_ms = _to_month_start(temperature)
    saod_ms = _to_month_start(saod) if saod is not None else None

    common = sl_ms.index.intersection(temp_ms.index)
    if saod_ms is not None:
        common = common.intersection(saod_ms.index)
    common = common.sort_values()

    H = sl_ms.loc[common].values.astype(np.float64)
    T = temp_ms.loc[common].values.astype(np.float64)
    S = saod_ms.loc[common].values.astype(np.float64) if saod_ms is not None else None

    has_sigma = gmsl_sigma is not None
    if has_sigma:
        sig_ms = _to_month_start(gmsl_sigma)
        sigma = sig_ms.reindex(common).values.astype(np.float64)
        # Fill any NaN sigma with the median (don't let a few gaps kill WLS)
        nan_sig = np.isnan(sigma)
        if nan_sig.any():
            sigma[nan_sig] = np.nanmedian(sigma)
    else:
        sigma = None

    # ---- 2. Decimal-year time vector ----
    time_years = np.array([
        t.year + (t.month - 1) / 12 + (t.day - 1) / 365.25
        for t in common
    ])
    n = len(H)
    dt = np.median(np.diff(time_years))

    # ---- 3. Trapezoidal integrals ∫Tᵏ ----
    # Use raw integrals (no factorial scaling) so that the regression
    # coefficient c_k directly equals the physical rate coefficient a_k:
    #   H(t) = c_k × ∫T^k  →  dH/dt = c_k × T^k  →  a_k = c_k
    integrals = []  # will contain [∫T^order, …, ∫T]
    for k in range(order, 0, -1):
        Tk = T ** k
        integral_Tk = np.zeros(n)
        for i in range(1, n):
            integral_Tk[i] = integral_Tk[i - 1] + 0.5 * (Tk[i] + Tk[i - 1]) * dt
        integrals.append(integral_Tk)

    # ---- 4. Optional: ∫SAOD ----
    if S is not None:
        integral_S = np.zeros(n)
        for i in range(1, n):
            integral_S[i] = integral_S[i - 1] + 0.5 * (S[i] + S[i - 1]) * dt

    # ---- 5. ΔT (and ΔSAOD) for leads / lags ----
    delta_T = np.diff(T, prepend=T[0])
    if S is not None:
        delta_S = np.diff(S, prepend=S[0])

    # ---- 6. Build design matrix ----
    # Column order: [∫T^order, …, ∫T, time_trend, ∫SAOD?, ΔT_lags, ΔSAOD_lags?]
    cols = list(integrals)  # polynomial integrals (high → low)
    cols.append(time_years - time_years[0])  # trend
    n_phys = len(cols)  # number of physical parameters (order + 1)

    if S is not None:
        cols.append(integral_S)
        idx_saod = len(cols) - 1
    else:
        idx_saod = None

    # Temperature leads/lags
    for lag in range(-n_lags, n_lags + 1):
        if lag < 0:
            shifted = np.concatenate([np.full(-lag, np.nan), delta_T[:lag]])
        elif lag > 0:
            shifted = np.concatenate([delta_T[lag:], np.full(lag, np.nan)])
        else:
            shifted = delta_T.copy()
        cols.append(shifted)

    # SAOD leads/lags
    if S is not None:
        for lag in range(-n_lags, n_lags + 1):
            if lag < 0:
                shifted = np.concatenate([np.full(-lag, np.nan), delta_S[:lag]])
            elif lag > 0:
                shifted = np.concatenate([delta_S[lag:], np.full(lag, np.nan)])
            else:
                shifted = delta_S.copy()
            cols.append(shifted)

    X = np.column_stack(cols)

    # ---- 7. Drop NaN rows (from leads/lags) ----
    valid = ~np.any(np.isnan(X), axis=1) & ~np.isnan(H)
    if has_sigma:
        valid = valid & ~np.isnan(sigma)
    X_v = X[valid]
    H_v = H[valid]
    time_v = time_years[valid]
    n_valid = int(valid.sum())

    # Add constant (intercept)
    X_v = sm.add_constant(X_v, prepend=False)  # constant is last column

    # ---- 8. Fit with WLS/OLS + HAC ----
    if hac_maxlags is None:
        hac_maxlags = int(np.floor(4 * (n_valid / 100) ** (2 / 9)))

    if has_sigma:
        weights = 1.0 / sigma[valid] ** 2
        model = sm.WLS(H_v, X_v, weights=weights).fit(
            cov_type='HAC', cov_kwds={'maxlags': hac_maxlags}
        )
    else:
        model = sm.OLS(H_v, X_v).fit(
            cov_type='HAC', cov_kwds={'maxlags': hac_maxlags}
        )

    reg_coeffs = model.params
    reg_cov = model.cov_params()
    reg_se = model.bse

    # ---- 9. Physical (rate-model) parameters ----
    # Regressors are raw integrals [∫T^order, …, ∫T, trend], so the
    # regression coefficients **directly** equal the physical rate-model
    # coefficients:  H = c_k × ∫T^k  →  dH/dt = c_k × T^k  →  a_k = c_k
    # No factorial transform needed.
    phys_coeffs = reg_coeffs[:n_phys].copy()
    phys_cov = reg_cov[:n_phys, :n_phys].copy()
    phys_se = np.sqrt(np.diag(phys_cov))

    # ---- 10. SAOD coefficient ----
    if idx_saod is not None:
        gamma_saod = float(reg_coeffs[idx_saod])
        gamma_saod_se = float(reg_se[idx_saod])
    else:
        gamma_saod = None
        gamma_saod_se = None

    # ---- 11. Named accessors ----
    # phys_coeffs = [a_order, …, a_1, trend]
    trend_val = float(phys_coeffs[-1])
    trend_se_val = float(phys_se[-1])
    alpha0_val = float(phys_coeffs[-2])
    alpha0_se_val = float(phys_se[-2])
    dalpha_dT_val = float(phys_coeffs[-3]) if order >= 2 else None
    dalpha_dT_se_val = float(phys_se[-3]) if order >= 2 else None
    d2alpha_dT2_val = float(phys_coeffs[-4]) if order >= 3 else None
    d2alpha_dT2_se_val = float(phys_se[-4]) if order >= 3 else None

    return DOLSResult(
        order=order,
        physical_coefficients=phys_coeffs,
        physical_covariance=phys_cov,
        physical_se=phys_se,
        gamma_saod=gamma_saod,
        gamma_saod_se=gamma_saod_se,
        alpha0=alpha0_val,
        alpha0_se=alpha0_se_val,
        dalpha_dT=dalpha_dT_val,
        dalpha_dT_se=dalpha_dT_se_val,
        d2alpha_dT2=d2alpha_dT2_val,
        d2alpha_dT2_se=d2alpha_dT2_se_val,
        trend=trend_val,
        trend_se=trend_se_val,
        regression_coefficients=reg_coeffs,
        regression_covariance=reg_cov,
        r2=model.rsquared,
        r2_adj=model.rsquared_adj,
        aic=model.aic,
        bic=model.bic,
        n_obs=n_valid,
        n_lags=n_lags,
        has_saod=(saod is not None),
        residuals=model.resid,
        fitted=model.fittedvalues,
        time=time_v,
        model=model,
    )


# ---- Backward-compatibility wrappers (deprecated) ----

DOLSQuadraticResult = DOLSResult  # type alias

def calibrate_alpha_dols_quadratic(
    sea_level: pd.Series,
    temperature: pd.Series,
    n_lags: int = 2,
    **kwargs,
) -> DOLSResult:
    """Deprecated — use ``calibrate_dols(order=2)``."""
    warnings.warn(
        "calibrate_alpha_dols_quadratic is deprecated. "
        "Use calibrate_dols(order=2) instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    return calibrate_dols(sea_level, temperature, order=2, n_lags=n_lags)


def calibrate_alpha_dols(
    time: np.ndarray,
    gmsl: np.ndarray,
    gmsl_sigma: np.ndarray,
    temperature: np.ndarray,
    n_lags: int = 2,
    include_trend: bool = True,
    hac_maxlags: int = 24,
) -> DOLSResult:
    """Deprecated — use ``calibrate_dols(order=1)``."""
    warnings.warn(
        "calibrate_alpha_dols is deprecated. "
        "Use calibrate_dols(order=1) instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    # Convert numpy arrays to pd.Series for the new API
    dates = pd.to_datetime([
        _decimal_year_to_datetime(t) for t in time
    ])
    sl = pd.Series(gmsl, index=dates)
    temp = pd.Series(temperature, index=dates)
    sig = pd.Series(gmsl_sigma, index=dates)
    return calibrate_dols(
        sl, temp, gmsl_sigma=sig, order=1,
        n_lags=n_lags, hac_maxlags=hac_maxlags,
    )



def test_rate_temperature_nonlinearity(
    rate: np.ndarray,
    temperature: np.ndarray,
    rate_sigma: np.ndarray = None,
    alpha: float = 0.05
) -> dict:
    """
    Test whether polynomial rate–temperature models of increasing order
    significantly improve fit.

    Fits three models:
        Linear:    rate = b₁×T + b₀
        Quadratic: rate = a₂×T² + a₁×T + a₀
        Cubic:     rate = c₃×T³ + c₂×T² + c₁×T + c₀

    Parameters
    ----------
    rate : np.ndarray
        GMSL rate (e.g. from compute_kinematics).
    temperature : np.ndarray
        Temperature anomaly (same length as rate).
    rate_sigma : np.ndarray, optional
        1-σ uncertainty in rate.  If provided, uses WLS.
    alpha : float, default 0.05
        Significance level for hypothesis tests.

    Returns
    -------
    dict
        Keys: 'linear', 'quadratic', 'cubic' (each with coeffs, se, r2,
        aic, bic, residuals, predictions), 'f_tests' (pairwise),
        'f_test' (backward-compat alias → linear_vs_quadratic),
        'aic_comparison', 'bic_comparison', 'recommendation'.
    """
    from scipy import stats

    # --- Helpers ---
    def _fit(X, y, w, n):
        k = X.shape[1]
        W = np.diag(w)
        XtWX = X.T @ W @ X
        coeffs = np.linalg.solve(XtWX, X.T @ W @ y)
        pred = X @ coeffs
        resid = y - pred
        ssr = np.sum(w * resid ** 2)
        sst_val = np.sum(w * (y - np.average(y, weights=w)) ** 2)
        r2 = 1 - ssr / sst_val
        mse = ssr / (n - k)
        cov = mse * np.linalg.inv(XtWX)
        se = np.sqrt(np.diag(cov))
        aic = n * np.log(ssr / n) + 2 * k
        bic = n * np.log(ssr / n) + k * np.log(n)
        return {
            'coeffs': coeffs, 'se': se, 'r2': r2, 'aic': aic, 'bic': bic,
            'residuals': resid, 'predictions': pred, 'ssr': ssr, 'k': k,
        }

    def _f_test(res_reduced, res_full, n, alpha_val):
        ssr_r, k_r = res_reduced['ssr'], res_reduced['k']
        ssr_f, k_f = res_full['ssr'], res_full['k']
        df_r, df_f = n - k_r, n - k_f
        f_stat = ((ssr_r - ssr_f) / (df_r - df_f)) / (ssr_f / df_f)
        p = 1 - stats.f.cdf(f_stat, df_r - df_f, df_f)
        return {'f_stat': f_stat, 'p_value': p,
                'df1': df_r - df_f, 'df2': df_f,
                'significant': p < alpha_val}

    # --- Data prep ---
    valid = ~(np.isnan(rate) | np.isnan(temperature))
    rate = rate[valid]
    temperature = temperature[valid]
    n = len(rate)

    if rate_sigma is not None:
        rate_sigma = rate_sigma[valid]
        weights = 1.0 / rate_sigma ** 2
    else:
        weights = np.ones(n)
    weights = weights / weights.sum() * n

    T = temperature

    # --- Fit models ---
    X_lin  = np.column_stack([T,    np.ones(n)])
    X_quad = np.column_stack([T**2, T, np.ones(n)])
    X_cub  = np.column_stack([T**3, T**2, T, np.ones(n)])

    res_lin  = _fit(X_lin,  rate, weights, n)
    res_quad = _fit(X_quad, rate, weights, n)
    res_cub  = _fit(X_cub,  rate, weights, n)

    # t-tests on highest-order coefficient
    df_quad = n - res_quad['k']
    t_stat_quad = res_quad['coeffs'][0] / res_quad['se'][0]
    p_t_quad = 2 * (1 - stats.t.cdf(np.abs(t_stat_quad), df_quad))
    res_quad['t_stat'] = t_stat_quad
    res_quad['p_value_t'] = p_t_quad

    df_cub = n - res_cub['k']
    t_stat_cub = res_cub['coeffs'][0] / res_cub['se'][0]
    p_t_cub = 2 * (1 - stats.t.cdf(np.abs(t_stat_cub), df_cub))
    res_cub['t_stat'] = t_stat_cub
    res_cub['p_value_t'] = p_t_cub

    # --- F-tests ---
    f_lq = _f_test(res_lin, res_quad, n, alpha)
    f_qc = _f_test(res_quad, res_cub, n, alpha)
    f_lc = _f_test(res_lin, res_cub, n, alpha)

    # --- AIC / BIC comparison ---
    aic_vals = {'linear': res_lin['aic'], 'quadratic': res_quad['aic'], 'cubic': res_cub['aic']}
    bic_vals = {'linear': res_lin['bic'], 'quadratic': res_quad['bic'], 'cubic': res_cub['bic']}

    best_aic = min(aic_vals, key=aic_vals.get)
    best_bic = min(bic_vals, key=bic_vals.get)
    delta_aic = {m: aic_vals[m] - aic_vals[best_aic] for m in aic_vals}
    delta_bic = {m: bic_vals[m] - bic_vals[best_bic] for m in bic_vals}

    # --- Recommendation ---
    reasons = []
    scores = {'linear': 0, 'quadratic': 0, 'cubic': 0}

    if f_lq['significant']:
        reasons.append(f"F lin→quad significant (p={f_lq['p_value']:.4f})")
        scores['quadratic'] += 1
    else:
        reasons.append(f"F lin→quad not significant (p={f_lq['p_value']:.4f})")
        scores['linear'] += 1

    if f_qc['significant']:
        reasons.append(f"F quad→cub significant (p={f_qc['p_value']:.4f})")
        scores['cubic'] += 1
    else:
        reasons.append(f"F quad→cub not significant (p={f_qc['p_value']:.4f})")

    reasons.append(f"AIC best: {best_aic} (Δ={delta_aic})")
    scores[best_aic] += 1
    reasons.append(f"BIC best: {best_bic} (Δ={delta_bic})")
    scores[best_bic] += 1

    reasons.append(
        f"R²: lin={res_lin['r2']:.3f}, quad={res_quad['r2']:.3f}, cub={res_cub['r2']:.3f}"
    )

    best = max(scores, key=scores.get)
    if scores[best] >= 2:
        recommendation = f"{best.upper()} model preferred. " + "; ".join(reasons)
    else:
        recommendation = "INCONCLUSIVE — consider additional diagnostics. " + "; ".join(reasons)

    # Remove internal helper keys before returning
    for res in (res_lin, res_quad, res_cub):
        res.pop('ssr', None)
        res.pop('k', None)

    return {
        'linear': res_lin,
        'quadratic': res_quad,
        'cubic': res_cub,
        'f_tests': {
            'linear_vs_quadratic': f_lq,
            'quadratic_vs_cubic': f_qc,
            'linear_vs_cubic': f_lc,
        },
        'f_test': f_lq,  # backward compatibility
        'aic_comparison': {
            'values': aic_vals,
            'best_model': best_aic,
            'delta_aic': delta_aic,
            # backward-compat scalar
            'delta_aic_lq': aic_vals['linear'] - aic_vals['quadratic'],
            'preferred_model': best_aic,
        },
        'bic_comparison': {
            'values': bic_vals,
            'best_model': best_bic,
            'delta_bic': delta_bic,
            'delta_bic_lq': bic_vals['linear'] - bic_vals['quadratic'],
            'preferred_model': best_bic,
        },
        'n_observations': n,
        'recommendation': recommendation,
    }


def test_saod_ic(
    sea_level: pd.Series,
    temperature: pd.Series,
    saod: pd.Series,
    gmsl_sigma: Optional[pd.Series] = None,
    order: int = 2,
    n_lags: int = 2,
    alpha: float = 0.05,
) -> dict:
    """
    Test whether adding SAOD improves the DOLS model.

    Compares:
        Model A: DOLS without SAOD
        Model B: DOLS with SAOD

    Uses AIC, BIC, and an F-test (nested models, 1 extra parameter
    for ∫SAOD plus 2×n_lags+1 extra ΔSAOD lag terms).

    Parameters
    ----------
    sea_level, temperature, saod : pd.Series
        Input series with datetime index.
    gmsl_sigma : pd.Series, optional
        Sea-level uncertainty for WLS.
    order : int, default 2
        Polynomial order.
    n_lags : int, default 2
        Number of leads/lags.
    alpha : float, default 0.05
        Significance level.

    Returns
    -------
    dict
        Keys: 'without_saod', 'with_saod' (summaries), 'f_test',
        'aic_comparison', 'bic_comparison', 'recommendation'.
    """
    from scipy import stats

    # Both models must be calibrated on the SAME time range.
    # Normalise to month-start so different day-of-month conventions match.
    def _to_month_start(s: pd.Series) -> pd.Series:
        new_idx = s.index.to_period('M').to_timestamp()
        out = s.copy()
        out.index = new_idx
        return out[~out.index.duplicated(keep='first')]

    sl_ms   = _to_month_start(sea_level)
    temp_ms = _to_month_start(temperature)
    saod_ms = _to_month_start(saod)

    common = sl_ms.index.intersection(temp_ms.index).intersection(saod_ms.index)
    sl_c   = sl_ms.loc[common]
    temp_c = temp_ms.loc[common]
    saod_c = saod_ms.loc[common]
    sig_c  = _to_month_start(gmsl_sigma).reindex(common) if gmsl_sigma is not None else None

    res_a = calibrate_dols(sl_c, temp_c, gmsl_sigma=sig_c, saod=None,
                           order=order, n_lags=n_lags)
    res_b = calibrate_dols(sl_c, temp_c, gmsl_sigma=sig_c, saod=saod_c,
                           order=order, n_lags=n_lags)

    # F-test for nested models
    ssr_a = float(np.sum(res_a.residuals ** 2))
    ssr_b = float(np.sum(res_b.residuals ** 2))
    k_a = len(res_a.regression_coefficients)
    k_b = len(res_b.regression_coefficients)
    df_extra = k_b - k_a  # number of extra SAOD parameters
    df_b = res_b.n_obs - k_b

    f_stat = ((ssr_a - ssr_b) / df_extra) / (ssr_b / df_b)
    p_value = 1 - stats.f.cdf(f_stat, df_extra, df_b)

    # SAOD coefficient t-test
    gamma = res_b.gamma_saod
    gamma_se = res_b.gamma_saod_se
    t_saod = gamma / gamma_se if gamma_se > 0 else np.inf
    p_saod = 2 * (1 - stats.t.cdf(np.abs(t_saod), df_b))

    # AIC / BIC
    d_aic = res_a.aic - res_b.aic  # positive → SAOD model better
    d_bic = res_a.bic - res_b.bic

    # Recommendation
    reasons = []
    score_saod = 0
    if p_value < alpha:
        reasons.append(f"F-test significant (p={p_value:.4f})")
        score_saod += 1
    else:
        reasons.append(f"F-test not significant (p={p_value:.4f})")
    if d_aic > 2:
        reasons.append(f"AIC favors SAOD (ΔAIC={d_aic:.1f})")
        score_saod += 1
    elif d_aic < -2:
        reasons.append(f"AIC favors no-SAOD (ΔAIC={d_aic:.1f})")
    else:
        reasons.append(f"AIC inconclusive (ΔAIC={d_aic:.1f})")
    if d_bic > 2:
        reasons.append(f"BIC favors SAOD (ΔBIC={d_bic:.1f})")
        score_saod += 1
    elif d_bic < -2:
        reasons.append(f"BIC favors no-SAOD (ΔBIC={d_bic:.1f})")
    else:
        reasons.append(f"BIC inconclusive (ΔBIC={d_bic:.1f})")

    reasons.append(
        f"γ_saod = {gamma:.6f} ± {gamma_se:.6f} (t={t_saod:.2f}, p={p_saod:.4f})"
    )
    reasons.append(
        f"R² without={res_a.r2:.4f}, with={res_b.r2:.4f}"
    )

    if score_saod >= 2:
        rec = "INCLUDE SAOD. " + "; ".join(reasons)
    elif score_saod == 0:
        rec = "EXCLUDE SAOD. " + "; ".join(reasons)
    else:
        rec = "INCONCLUSIVE. " + "; ".join(reasons)

    return {
        'without_saod': {
            'r2': res_a.r2, 'aic': res_a.aic, 'bic': res_a.bic,
            'n_params': k_a,
            'physical_coefficients': res_a.physical_coefficients,
        },
        'with_saod': {
            'r2': res_b.r2, 'aic': res_b.aic, 'bic': res_b.bic,
            'n_params': k_b,
            'physical_coefficients': res_b.physical_coefficients,
            'gamma_saod': gamma,
            'gamma_saod_se': gamma_se,
            'gamma_saod_t': t_saod,
            'gamma_saod_pvalue': p_saod,
        },
        'f_test': {
            'f_stat': f_stat,
            'p_value': p_value,
            'df1': df_extra,
            'df2': df_b,
            'significant': p_value < alpha,
        },
        'aic_comparison': {
            'delta_aic': d_aic,
            'preferred_model': 'with_saod' if d_aic > 2 else (
                'without_saod' if d_aic < -2 else 'inconclusive'),
        },
        'bic_comparison': {
            'delta_bic': d_bic,
            'preferred_model': 'with_saod' if d_bic > 2 else (
                'without_saod' if d_bic < -2 else 'inconclusive'),
        },
        'recommendation': rec,
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
    'calibrate_dols',
    'test_rate_temperature_nonlinearity',
    'test_saod_ic',
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
    print("  - calibrate_dols()")
    print("  - test_rate_temperature_nonlinearity()")
    print("  - test_saod_ic()")
