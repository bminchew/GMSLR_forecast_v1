# Sea Level Rise Forecasting: A Hierarchical Framework

A physics-informed framework for probabilistic sea-level rise (SLR) projections that separates **predictable** thermodynamic responses from **deeply uncertain** Antarctic ice-sheet dynamics.

## Overview

This project implements a hierarchical SLR forecasting framework that decomposes total projection uncertainty into three components:

$$\sigma^2_{\text{total}}(t) = \sigma^2_{\text{constrained}}(t) + \sigma^2_{\text{scenario}}(t) + \sigma^2_{\text{ice}}(t)$$

- **Constrained uncertainty** ($\sigma_{\text{constrained}}$): Calibrated from the observational record using Dynamic Ordinary Least Squares (DOLS), capturing thermodynamic sensitivity (steric expansion + glaciers + Greenland)
- **Scenario uncertainty** ($\sigma_{\text{scenario}}$): Spread across SSP emission pathways (societal choices)
- **Ice-sheet deep uncertainty** ($\sigma_{\text{ice}}$): Physics-informed West Antarctic Ice Sheet (WAIS) uncertainty, correcting systematic biases in IPCC AR6 projections

### Key Results

- DOLS calibration: quadratic rate-temperature model `dH/dt = (dα/dT) × T² + α₀ × T + trend`
- Physics-informed WAIS corrections increase ice-sheet uncertainty from ~150 mm (IPCC medium confidence) to ~491 mm at 2100
- The ice-sheet fraction of total variance rises from 2% to 18% with physics-informed corrections
- Multi-dataset robustness analysis confirms positive quadratic sensitivity across all observational records
- IPCC process models show systematically lower thermodynamic sensitivity than observations

## Project Structure

```
slr_forecast/
├── README.md                          # This file
├── TODO.md                            # Research task tracking
├── physics_informed_wais_uncertainty.tex  # LaTeX document (supplementary material)
│
├── notebooks/                         # Analysis code
│   ├── predictability_analysis.ipynb  # PRIMARY: DOLS calibration, projections, WAIS framework
│   ├── read_process_datafiles.ipynb   # Data loading and preprocessing pipeline
│   ├── slr_analysis_notebook.ipynb    # Exploratory analysis and visualization
│   │
│   ├── slr_analysis.py               # Core DOLS engine + kinematics + statistical tests
│   ├── slr_data_readers.py            # 14+ data reader functions with unit conversion
│   ├── slr_projections.py             # Monte Carlo projection ensemble generation
│   ├── preprocessing_functions.py     # Data preprocessing utilities
│   ├── visualization_cells.py         # Plotting helper functions
│   ├── test_dols.py                   # 19-test verification suite for DOLS
│   │
│   ├── dols_robustness.py             # Multi-dataset robustness matrix (7 GMSL × 4 GMST)
│   ├── ipcc_emergent_sensitivity.py   # DOLS applied to IPCC projections
│   ├── dols_sliding_window.py         # Sliding-window epoch sensitivity + SAOD analysis
│   │
│   └── archive/                       # Deprecated code (kept for reference)
│
├── data/
│   ├── raw/                           # Immutable observational records (see data/raw/README.md)
│   │   ├── gmslr/                     # Sea level reconstructions
│   │   ├── gmst/                      # Temperature products
│   │   ├── ice_sheets/                # IMBIE mass balance
│   │   ├── glaciers/                  # GlaMBIE glacier data
│   │   ├── saod/                      # Volcanic aerosol optical depth
│   │   ├── tws/                       # GRACE terrestrial water storage
│   │   ├── steric/                    # Steric sea level
│   │   ├── enso/                      # Climate indices
│   │   └── ipcc_ar6/                  # IPCC AR6 FACTS projections (NetCDF)
│   └── processed/
│       ├── slr_processed_data.h5      # All preprocessed data (53 HDF5 keys)
│       └── results_summary.json       # Machine-readable key results
│
├── figures/                           # Generated figures (PNG, 150 dpi)
├── scripts/                           # Standalone execution scripts
│   ├── dols_engine.py                 # DOLS execution engine
│   └── check_readiness.py            # Data readiness validation
└── environment/                       # Docker reproducibility setup
```

## Quick Start

### Prerequisites
- Python 3.9+
- Core dependencies: `numpy`, `pandas`, `scipy`, `statsmodels`, `matplotlib`, `h5py`

### Docker (recommended)
```bash
docker build -t slr-forecast ./environment
docker run -p 8888:8888 slr-forecast
```

### Local setup
```bash
pip install numpy pandas scipy statsmodels matplotlib h5py tables openpyxl netCDF4
```

### Workflow

1. **Data preprocessing**: Run `read_process_datafiles.ipynb` to load all raw data, standardize units, harmonize baselines, and save to `slr_processed_data.h5`
2. **Main analysis**: Run `predictability_analysis.ipynb` for DOLS calibration, Monte Carlo projections, variance decomposition, and physics-informed WAIS uncertainty
3. **Robustness checks**: Run the standalone scripts:
   ```bash
   cd notebooks
   python dols_robustness.py        # Multi-dataset coefficient stability
   python ipcc_emergent_sensitivity.py  # DOLS vs IPCC process models
   python dols_sliding_window.py    # Epoch sensitivity + SAOD reconsideration
   ```

## Core Analysis Modules

### `slr_analysis.py` — DOLS Engine

The central analysis module implementing:

- **`calibrate_dols()`**: Unified DOLS function supporting polynomial orders 1-3, optional WLS (weighted least squares with measurement uncertainties), optional SAOD (volcanic forcing), and HAC (Newey-West) standard errors for robust inference
- **`calibrate_dols_sliding()`**: Sliding-window (kernel-weighted) DOLS for estimating time-varying coefficients α₀(t) and dα/dT(t)
- **`compute_kinematics()`**: Kernel-weighted local polynomial regression for rates and accelerations
- **`test_rate_temperature_nonlinearity()`**: Model selection (linear/quadratic/cubic) via F-tests, AIC, BIC
- **`test_saod_ic()`**: Information-criterion test for volcanic forcing significance

### `slr_data_readers.py` — Data I/O

14+ reader functions, each attaching standardized metadata (`df.attrs`) including dataset name, reference, DOI, native and current units. Includes:

- **`convert_to_standard_units()`**: Converts to meters, Celsius, years (idempotent)
- **`convert_units()`**: General-purpose unit conversion (m/mm/cm/ft, degC/degF/K, compound rates)

### `slr_projections.py` — Forward Projections

Monte Carlo ensemble generation under SSP scenarios with optional SAOD for hindcasting.

## Datasets

### Sea Level (GMSL)
| Dataset | Period | Type | Reference |
|---------|--------|------|-----------|
| Frederikse et al. | 1900-2018 | Budget reconstruction | Nature (2020) |
| Dangendorf et al. | 1900-2021 | Kalman smoother | Nature Climate Change (2019) |
| Horwath et al. | 1993-2016 | ESA CCI budget | ESSD (2022) |
| NASA GSFC | 1993-present | Satellite altimetry | — |
| IPCC AR6 observed | 1950-2020 | Composite | IPCC AR6 WG1 |

### Temperature (GMST)
Berkeley Earth, GISTEMP v4, HadCRUT5, NOAA GlobalTemp

### IPCC AR6 Projections
Component-resolved FACTS projections (medium + low confidence) for SSP1-1.9 through SSP5-8.5, with ocean dynamics, glaciers, Greenland, and Antarctic contributions.

## Key Findings

### DOLS Calibration
- Quadratic rate model preferred over linear (F-test, AIC, BIC)
- SAOD (volcanic) forcing does NOT alias into temperature sensitivity (γ_saod t-stat = 0.26)
- Multi-dataset ensemble (excluding Dangendorf sterodynamic): dα/dT = 2.85 ± 0.38 mm/yr/°C²

### IPCC vs Observations
- IPCC thermodynamic sensitivity is ~2× lower than observational DOLS estimate
- All SSPs prefer linear (not quadratic) rate-temperature relationship
- Consistent with Grinsted & Christensen (2021) transient sea level sensitivity framework

### Physics-Informed WAIS Uncertainty
Four complementary approaches (A1-A4) for correcting IPCC AIS projections:
- A1: Rheology correction (Glen's flow law n=3 → n=4)
- A2: Stochastic amplification during marine ice-sheet instability
- A3: Process-informed discrete scenario mixture
- A4 (recommended): Combined framework — σ_ice ≈ 491 mm at 2100

### Sliding-Window DOLS
- Coefficients α₀ and dα/dT trade off in an epoch-dependent manner
- MLO SAOD becomes significant in ~50% of sliding windows (vs 0% in static DOLS)
- Sensitivity to start date: Frederikse thermo dα/dT drops from 2.65 (1950+) to 0.19 (1900+)

## References

- Frederikse, T., et al. (2020). The causes of sea-level rise since 1900. *Nature*, 584, 393-397.
- Dangendorf, S., et al. (2019). Persistent acceleration in global sea-level rise since the 1960s. *Nature Climate Change*, 9, 705-710.
- Grinsted, A. & Christensen, J.H. (2021). The transient sensitivity of sea level rise. *Ocean Science*, 17, 181-186.
- Robel, A.A., Seroussi, H. & Roe, G.H. (2019). Marine ice sheet instability amplifies and skews uncertainty. *PNAS*, 116(30), 14887-14892.
- Martin, D.F., et al. (in review). Impact of the stress exponent on ice sheet simulations. *AGU Advances*.
- Getraer, R.D. & Morlighem, M. (2025). Increasing the Glen-Nye power-law exponent accelerates ice-loss projections. *GRL*, 52.
- Fricker, H.A., et al. (2025). Antarctica in 2025: Drivers of deep uncertainty in projected ice loss. *Science*, 387(6736), 758-765.

