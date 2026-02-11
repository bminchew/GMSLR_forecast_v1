import os

# Define the directory structure
base_dir = "slr_forecast"
folders = [
    "data/raw", "data/processed", "data/metadata",
    "notebooks", "scripts", "results/figures", "results/tables",
    "environment"
]

for folder in folders:
    os.makedirs(os.path.join(base_dir, folder), exist_ok=True)

# 1. Create the Dockerfile
docker_content = """FROM jupyter/scipy-notebook:latest
WORKDIR /home/jovyan/work
COPY --chown=jovyan:users . /home/jovyan/work
RUN mamba install --yes h5py seaborn statsmodels
EXPOSE 8888
"""
with open(os.path.join(base_dir, "environment/Dockerfile"), "w") as f:
    f.write(docker_content)

# 2. Create the DOLS Engine Script
script_content = """import numpy as np
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
"""
with open(os.path.join(base_dir, "scripts/dols_engine.py"), "w") as f:
    f.write(script_content)

# 3. Create a README template
readme_content = f"# Sea Level Rise Forecast\\n\\nBuilt on January 16, 2026\\n\\nTo run: `docker build -t slr-forecast ./environment && docker run -p 8888:8888 slr-forecast`"
with open(os.path.join(base_dir, "README.md"), "w") as f:
    f.write(readme_content)

print("Folder structure and core scripts created.")
