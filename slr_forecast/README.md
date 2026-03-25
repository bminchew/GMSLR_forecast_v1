# Sea Level Rise Forecasting: A Hierarchical Framework

A physics-informed framework for probabilistic sea-level rise (SLR) projections that decomposes total GMSL into physical components, calibrates each against independent observations, and propagates all uncertainties through conditional SSP scenarios.

## Overview

This project implements a hierarchical SLR forecasting framework with three tiers of increasing physical complexity:

1. **Naive extrapolation**: Satellite-era rate + acceleration (lower bound)
2. **Aggregate semi-empirical**: Bayesian rate-state model relating GMSL rate to GMST
3. **Component decomposition**: Separate physics-informed models for each SLR contributor

The component decomposition is the primary framework. Each contributor is modeled independently with component-specific physics, calibrated against dedicated observational datasets, and validated against withheld data:

| Component | Model | Calibration data | Key parameters |
|-----------|-------|-------------------|---------------|
| Thermosteric | Single-layer ODE, joint calibration | NOAA TSL (1955-2025) + EN4 subsurface T (1970-2021) | a, b, c, tau_u, kappa, delta |
| Glaciers | Bayesian linear DOLS + volume cap | GlaMBIE consensus (2000-2024) | b, c, H0 |
| Greenland SMB | RCM-derived sensitivity | RACMO via Mouginot (1972-2018) | C_T, C_T2 (literature) |
| Greenland discharge | ODE with ocean thermal lag | Mouginot discharge + EN4 ocean T | gamma_atm, gamma_ocean, tau |
| EAIS | Trend-only (literature SMB sensitivity) | IMBIE (1992-2020) | b, c |
| Peninsula | Bayesian linear DOLS | IMBIE (1992-2020) | b, c, H0 |
| WAIS | A4 deep-uncertainty scenario mixture | Expert judgment + rheology correction | Scenario weights, skew-normal params |

## Project Structure

```
slr_forecast/
├── README.md
├── TODO.md                            # Research task tracking
├── taxonomy.md                        # Bayesian terminology, uncertainty decomposition
├── refactor.md                        # Sign conventions, units, baseline tracking
│
├── notebooks/                         # Analysis code
│   ├── component_ocean.ipynb          # Thermosteric: joint NOAA+EN4 physical ODE
│   ├── component_glacier.ipynb        # Glaciers: linear DOLS on GlaMBIE
│   ├── component_greenland.ipynb      # Greenland: SMB (literature) + discharge ODE
│   ├── component_eais.ipynb           # East Antarctica: trend-only + ISMIP6 comparison
│   ├── component_apeninsula.ipynb     # Antarctic Peninsula: linear DOLS + ISMIP6
│   ├── component_wais.ipynb           # WAIS: A4 scenario framework
│   │
│   ├── predictability_analysis.ipynb  # Aggregate DOLS calibration and projections
│   ├── read_process_datafiles.ipynb   # Data loading and preprocessing pipeline
│   ├── slr_analysis_notebook.ipynb    # Exploratory analysis and visualization
│   │
│   ├── bayesian_dols.py              # Bayesian fitting: level-space, rate-state,
│   │                                  #   thermosteric physical, Greenland discharge
│   ├── slr_analysis.py               # Core DOLS engine + kinematics
│   ├── slr_data_readers.py           # 14+ data reader functions
│   ├── slr_projections.py            # MC projection ensemble generation
│   ├── smb_projections.py            # SMB sensitivity projections (Greenland, EAIS)
│   ├── component_analysis.py         # Fitting helpers, transfer functions
│   ├── component_projections.py      # IPCC/ISMIP6 readers, A4 WAIS framework
│   ├── component_plotting.py         # Projection plots, histograms, ridge plots
│   ├── component_io.py               # HDF5 I/O for component results
│   │
│   ├── archive/                      # Superseded notebooks
│   └── diagnostics/                  # Supplementary diagnostic notebooks
│
├── src/slr_forecast/
│   ├── config.py                     # BASELINE_YEAR, N_SAMPLES, SEEDS, paths
│   └── readers/                      # Package-level data readers
│       ├── forcing.py                # NOAA thermosteric, volcanic AOD
│       ├── gmst.py                   # Berkeley Earth, GISTEMP
│       ├── ocean_temp.py             # EN4 regional subsurface temperature
│       └── ice_sheets.py             # Mouginot, Mankoff, IMBIE
│
├── data/
│   ├── raw/                          # Immutable observational records
│   │   ├── gmslr/                    # Sea level reconstructions
│   │   ├── gmst/                     # Temperature products
│   │   ├── steric/                   # NOAA thermosteric, EN4
│   │   ├── ocean_temp/               # EN4 subsurface gridded T
│   │   ├── ice_sheets/               # IMBIE, Mouginot, Mankoff
│   │   │   └── ismip6/               # ISMIP6 Antarctica (13 models, regional ivaf)
│   │   ├── glaciers/                 # GlaMBIE consensus
│   │   ├── ipcc_ar6/                 # IPCC AR6 FACTS projections (NetCDF)
│   │   └── ...                       # SAOD, TWS, ENSO indices
│   └── processed/
│       ├── slr_processed_data.h5     # All preprocessed data (53 HDF5 keys)
│       └── component_results.h5      # Fitted parameters + projections per component
│
├── figures/                          # Generated figures (PNG, 150 dpi)
├── manuscripts/
│   └── 00_ddpi_slrforecast2026/      # Main paper (LaTeX)
└── scripts/                          # Standalone utilities
```

## Quick Start

### Prerequisites
- Python 3.10+
- Core: `numpy`, `pandas`, `scipy`, `statsmodels`, `matplotlib`, `h5py`, `netCDF4`
- Bayesian: `emcee`, `arviz`
- Optional: `tables` (HDF5 via pandas), `openpyxl`

### Workflow

1. **Data preprocessing**: Run `read_process_datafiles.ipynb` to load all raw data, standardize units, harmonize baselines, and save to `slr_processed_data.h5`

2. **Component fitting**: Run each `component_*.ipynb` notebook with `REFIT = True`. Each notebook:
   - Loads observational data
   - Fits a Bayesian model (MCMC via emcee)
   - Generates SSP projections with full MC uncertainty
   - Saves results to `component_results.h5`
   - Compares against IPCC AR6 and/or ISMIP6

3. **Subsequent runs**: Set `REFIT = False` to skip expensive fitting and load from HDF5. The switch auto-falls back to `REFIT = True` if no saved results exist.

4. **Aggregate analysis**: Run `predictability_analysis.ipynb` for the aggregate DOLS framework and three-step comparison.

### Component I/O

All component notebooks write to a shared HDF5 file (`data/processed/component_results.h5`) via `component_io.py`:

```python
from component_io import load_all_projections, list_components

# Inspect what's saved
list_components()

# Load all projections for aggregation
proj_years, all_proj = load_all_projections()
# all_proj = {'ocean': {'SSP2-4.5': {'samples': (2000, 201), 'median': ..., ...}}, ...}
```

## Component Models

### Thermosteric (Ocean Thermal Expansion)
Single-layer physical ODE with joint NOAA + EN4 calibration:
- `dS_u/dt = (T - S_u) / tau_u` (ocean thermal lag)
- `eta(t) = a*S_u^2 + b*S_u + c*t + H0` (steric expansion)
- Transfer function `T_sub = kappa*S_u + delta` couples to Greenland discharge

### Glaciers
Bayesian linear rate-temperature fit to GlaMBIE:
- `dH/dt = b*T + c` (linear selected over quadratic by BIC)
- Volume cap at 0.32 m SLE (Farinotti et al. 2019)

### Greenland
- **SMB**: Literature-derived C_T sensitivities from RCM ensemble (RACMO, MAR, HIRHAM)
- **Discharge**: ODE driven by subsurface ocean temperature via jointly-calibrated transfer function

### WAIS
A4 deep-uncertainty scenario mixture with skew-normal distributions in log-space (Robel et al. 2019), rheology correction (n=3 to n=4, Martin et al.), SSP-independent.

## Key Constants

| Constant | Value | Description |
|----------|-------|-------------|
| BASELINE_YEAR | 2005.0 | Reference epoch for anomalies |
| N_SAMPLES | 2000 | MC ensemble size |
| M_TO_MM | 1000.0 | Unit conversion |
| PROJ_YEARS | 1950-2150 | Annual projection grid |

## Datasets

### Calibration Data
| Component | Dataset | Period | Reference |
|-----------|---------|--------|-----------|
| Thermosteric | NOAA TSL 0-700m | 1955-2025 | Levitus et al. (2012) |
| Thermosteric | EN4 global 0-700m | 1970-2021 | Good et al. (2013) |
| Glaciers | GlaMBIE consensus | 2000-2024 | GlaMBIE (2024) |
| Greenland SMB | Mouginot/RACMO | 1972-2018 | Mouginot et al. (2019) |
| Greenland discharge | Mouginot + EN4 | 1972-2018 | Mouginot et al. (2019) |
| EAIS, Peninsula, WAIS | IMBIE-2 | 1992-2020 | Otosaka et al. (2023) |

### Validation Data
| Dataset | Period | Reference |
|---------|--------|-----------|
| Frederikse steric | 1900-2018 | Frederikse et al. (2020) |
| Mankoff discharge | 1986-2023 | Mankoff et al. (2021) |
| ISMIP6 Antarctica | 2016-2101 | Seroussi et al. (2020) |
| IPCC AR6 FACTS | 2020-2150 | Fox-Kemper et al. (2021) |

### Temperature Forcing
Berkeley Earth monthly GMST (1850-present), SSP projections from IPCC AR6 FACTS.

## References

- Frederikse, T., et al. (2020). The causes of sea-level rise since 1900. *Nature*, 584, 393-397.
- Seroussi, H., et al. (2020). ISMIP6 Antarctica. *The Cryosphere*, 14, 3033-3070.
- Mouginot, J., et al. (2019). Forty-six years of Greenland Ice Sheet mass balance. *PNAS*, 116(19), 9239-9244.
- Robel, A.A., Seroussi, H. & Roe, G.H. (2019). Marine ice sheet instability amplifies and skews uncertainty. *PNAS*, 116(30), 14887-14892.
- Martin, D.F., et al. (in review). Impact of the stress exponent on ice sheet simulations. *AGU Advances*.
- Fricker, H.A., et al. (2025). Antarctica in 2025. *Science*, 387(6736), 758-765.
- Good, S.A., et al. (2013). EN4: Quality controlled ocean temperature and salinity profiles. *JGR Oceans*, 118, 6704-6716.
- Levitus, S., et al. (2012). World ocean heat content and thermosteric sea level change. *GRL*, 39, L10603.
