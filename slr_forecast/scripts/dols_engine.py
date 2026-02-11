import numpy as np
import statsmodels.api as sm

def saod_constrained_dols(time, sl, se, temp, saod, p_order=1):
    '''
    Performs a Dynamic Ordinary Least Squares (DOLS) regression to estimate 
    sea-level sensitivity (alpha) constrained by volcanic forcing (gamma).
    
    Parameters:
    -----------
    time : array-like
        Decimal year vector.
    sl   : array-like
        Sea level anomalies (mm).
    se   : array-like
        Standard error/uncertainty of sea level measurements.
    temp : array-like
        Temperature anomalies (K or °C).
    saod : array-like
        Stratospheric Aerosol Optical Depth (or proxy from MLO).
    p_order : int
        The number of leads and lags to include (corrects for autocorrelation).
    '''
    
    # 1. PREPARE THERMODYNAMIC AND VOLCANIC DRIVERS
    # Calculate the time step (typically 1/12 for monthly data)
    dt = np.mean(np.diff(time))
    
    # Integrated Temperature (it): Represents the cumulative heat uptake
    it = (temp * dt).cumsum()
    dit = temp * dt # The 'rate' of integration (used for DOLS leads/lags)
    
    # Integrated SAOD (isaod): Represents the cumulative volcanic forcing impact
    isaod = (saod * dt).cumsum()
    dsaod = saod * dt # The 'rate' of integration (used for DOLS leads/lags)
    
    # 2. CONSTRUCT LEADS AND LAGS (The 'Dynamic' in DOLS)
    # These terms account for endogenous relationships and autocorrelation 
    # in the residual error term, ensuring unbiased estimates of Alpha.
    ll_t = [np.roll(dit, -j) for j in range(-p_order, p_order + 1)]
    ll_s = [np.roll(dsaod, -j) for j in range(-p_order, p_order + 1)]
    
    # 3. BUILD THE DESIGN MATRIX (X)
    # Column order: [Intercept, Alpha term, Beta (Linear trend), Gamma (Volcanic), Leads/Lags]
    X = sm.add_constant(np.column_stack([it, time, isaod, np.column_stack(ll_t + ll_s)]))
    
    # 4. TRIM BOUNDARIES
    # We must slice the data to remove 'p_order' points from start and end 
    # due to the shifting required by leads and lags.
    v = slice(p_order, -p_order)
    
    # 5. WEIGHTED LEAST SQUARES (WLS) WITH HAC ERRORS
    # - Weights: Inverse of the variance (1/se^2) to prioritize more accurate data.
    # - HAC: Newey-West covariance estimator to ensure robust p-values/confidence intervals.
    model = sm.WLS(sl[v], X[v], weights=1/(se[v]**2)).fit(
        cov_type='HAC', 
        cov_kwds={'maxlags': 12}
    )
    
    return model
