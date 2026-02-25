# Sea Level Rise Forecasting: A Hierarchical Uncertainty Framework

A physics-informed probabilistic framework for global mean sea-level rise (GMSLR) projections that separates **predictable** thermodynamic responses from **deeply uncertain** Antarctic ice-sheet dynamics.

## Motivation

Current GMSLR projections conflate well-constrained thermodynamic processes (thermal expansion, glaciers) with deeply uncertain ice-sheet contributions. This framework decomposes total projection uncertainty into three components:

$$\sigma^2_{\text{total}}(t) = \sigma^2_{\text{constrained}}(t) + \sigma^2_{\text{scenario}}(t) + \sigma^2_{\text{ice}}(t)$$

| Component | Source | Character |
|-----------|--------|-----------|
| Constrained | Observational calibration (DOLS) | Reducible with longer records |
| Scenario | SSP emission pathway spread | Societal choice, irreducible by science |
| Ice sheet | WAIS dynamics beyond process models | Deep uncertainty, physics-informed bounds |

## Methods

**Dynamic OLS (DOLS)** calibrates a quadratic rate-temperature model against the observational record:

```
dH/dt = (d alpha/dT) * T^2 + alpha_0 * T + trend
```

with WLS weighting, HAC standard errors, and optional volcanic (SAOD) forcing. Multi-dataset robustness testing spans 7 GMSL reconstructions x 4 temperature products. Bayesian extensions (static, DLM, hierarchical) provide complementary uncertainty quantification via `emcee`.

**Physics-informed WAIS uncertainty** corrects systematic biases in IPCC AR6 ice-sheet projections through four approaches addressing rheology (Glen's law n=3 vs n=4), stochastic amplification during marine ice-sheet instability, and missing-process enumeration.

## Repository Structure

```
.
├── slr_forecast/
│   ├── notebooks/          # Analysis code (~11k lines)
│   │   ├── slr_analysis.py             # DOLS engine
│   │   ├── slr_data_readers.py         # 14+ data readers with metadata
│   │   ├── slr_projections.py          # Monte Carlo ensemble projections
│   │   ├── bayesian_dols.py            # Bayesian DOLS / DLM / hierarchical
│   │   ├── dols_robustness.py          # Multi-dataset robustness matrix
│   │   ├── dols_sliding_window.py      # Time-varying coefficient analysis
│   │   ├── ipcc_emergent_sensitivity.py  # DOLS applied to IPCC projections
│   │   └── test_dols.py               # 19-test verification suite
│   │
│   ├── data/
│   │   ├── raw/            # Observational records (gitignored)
│   │   └── processed/      # HDF5 preprocessed data (gitignored)
│   │
│   ├── figures/            # Generated analysis figures
│   ├── scripts/            # Standalone execution scripts
│   ├── environment/        # Dockerfile for reproducibility
│   ├── README.md           # Detailed documentation
│   └── TODO.md             # Research priorities
│
└── papers/                 # Related manuscripts and references
```

## Quick Start

### Prerequisites

Python 3.9+ with:
```bash
pip install numpy pandas scipy statsmodels matplotlib h5py tables openpyxl netCDF4 xarray emcee arviz seaborn
```

### Docker (recommended)
```bash
docker build -t slr-forecast ./slr_forecast/environment
docker run -p 8888:8888 slr-forecast
```

### Workflow

1. **Preprocess data** -- Run `notebooks/read_process_datafiles.ipynb` to load raw observations and save to `slr_processed_data.h5`
2. **Main analysis** -- Run `notebooks/predictability_analysis.ipynb` for DOLS calibration, projections, and variance decomposition
3. **Robustness** -- Run standalone scripts:
   ```bash
   cd slr_forecast/notebooks
   python dols_robustness.py
   python dols_sliding_window.py
   python ipcc_emergent_sensitivity.py
   python bayesian_analysis.py
   ```
4. **Tests** -- `python test_dols.py` (19 tests covering coefficient recovery, WLS, SAOD, polynomial orders)

See [`slr_forecast/README.md`](slr_forecast/README.md) for detailed documentation of modules, datasets, and findings.

## Data

Raw observational data is not tracked in this repository. Place data files in `slr_forecast/data/raw/` following the directory structure documented in [`slr_forecast/data/raw/README.md`](slr_forecast/data/raw/README.md).

**Sea level**: Frederikse (2020), Dangendorf (2019/2024), Horwath (2022), NASA GSFC, IPCC AR6
**Temperature**: Berkeley Earth, GISTEMP v4, HadCRUT5, NOAA GlobalTemp
**Ice sheets**: IMBIE-3 (2023) -- Greenland, WAIS, EAIS, Peninsula
**Glaciers**: GlaMBIE (2024) consensus
**Volcanic**: GloSSAC v2.2, Mauna Loa transmission
**Projections**: IPCC AR6 FACTS (SSP1-1.9 through SSP5-8.5)

## References

- Frederikse, T., et al. (2020). The causes of sea-level rise since 1900. *Nature*, 584, 393-397.
- Dangendorf, S., et al. (2019). Persistent acceleration in global sea-level rise since the 1960s. *Nature Climate Change*, 9, 705-710.
- Grinsted, A. & Christensen, J.H. (2021). The transient sensitivity of sea level rise. *Ocean Science*, 17, 181-186.
- Robel, A.A., Seroussi, H. & Roe, G.H. (2019). Marine ice sheet instability amplifies and skews uncertainty. *PNAS*, 116(30), 14887-14892.
- Fricker, H.A., et al. (2025). Antarctica in 2025: Drivers of deep uncertainty in projected ice loss. *Science*, 387(6736), 758-765.
