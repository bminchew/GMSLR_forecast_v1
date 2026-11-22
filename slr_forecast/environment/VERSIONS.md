# Software versions: macOS development environment vs. container

The results of record (the archived numbers and figures) come from the final run in the linux/arm64 container on an Apple Silicon Mac. The container's package versions are pinned to those of the macOS development environment, `python2025`, in which the pipeline was developed and run. In this file, "production" means that macOS environment and its runs.

## Production environment (macOS)

The 10k-ensemble runs on macOS used the conda env `python2025`:

- Python 3.13.7 (conda-forge build `h5eba815_100_cp313`).
- Platform **osx-64**: x86_64 binaries running under Rosetta 2 on an Apple M1 Max (10 cores). The env is not native arm64.
- BLAS/LAPACK: conda-forge **OpenBLAS 0.3.30**, OpenMP threading layer (`libopenblas 0.3.30 openmp_h6006d49_4`). Kernel selected at run time: `Nehalem`. Threads: 10 (default, no thread variables set). numpy and scipy were linked against this OpenBLAS, not Apple Accelerate.
- numpy SIMD under Rosetta: baseline SSE–SSSE3, with SSE4.1, SSE4.2 and POPCNT found. AVX, AVX2, FMA3 and AVX-512 were not available.
- The env mixed conda-forge and PyPI packages. From PyPI wheels: h5py, tables (PyTables), xarray, arviz, emcee, corner, openpyxl, pytest, numexpr, blosc2, h5netcdf, xarray-einstats, iniconfig, pluggy and packaging, and, for the preprocessing scripts, gsw, cartopy, fsspec, aiohttp, requests, zarr, numcodecs, dask and plotly. Everything else came from conda-forge.
- The slr_forecast package was installed editable.

The pins were taken from `conda list`, `pip freeze` and a runtime probe of the production env (`__version__`/`__file__`, `show_config()` and thread pools). Where `conda list` and `pip freeze` disagree because a pip install shadowed a conda package (fsspec, click), the pins follow the version actually imported.

## Container environment

- Base image: `mambaorg/micromamba:2.10.0-debian12-slim`, pinned by its multi-arch index digest (Debian 12.15, glibc 2.36).
- Conda packages come from `conda-linux-64.lock` or `conda-linux-aarch64.lock`. These are explicit, md5-checked locks generated with conda-lock 4.1.0 from `environment.yml`.
- Every direct dependency, and every transitive package that also existed in production, is pinned to the production version. The exceptions are ca-certificates, certifi and openssl, which handle TLS only, and multidict and propcache (see deviation 8).
- Each lock has 225 packages. 201 match the versions production imported. 5 differ: the 3 TLS packages, multidict and propcache. fsspec and click match the imported pip versions (2026.2.0, 8.3.0), not the shadowed conda records (2026.3.0, 8.3.1). 19 are Linux-only or packaging metapackages with no production counterpart: libgcc-ng, libstdcxx(-ng), ld_impl, keyutils, libattr, attr, libuuid, libcrc32c, c-blosc2, colorama, cpython/python-gil, `_python_abi3_support`, msgpack-python, deprecated, setuptools-scm, vcs_versioning, and dask-core (production had the `dask` 2025.9.1 wheel).
- `environment/check_env.py` checks the direct dependencies at run time. All match production (see the tables).

| Package | Production | Container | Note |
|---|---|---|---|
| python | 3.13.7 | 3.13.7 | conda-forge |
| numpy | 2.3.2 | 2.3.2 | |
| scipy | 1.16.1 | 1.16.1 | |
| pandas | 2.3.2 | 2.3.2 | |
| matplotlib | 3.10.6 | 3.10.6 (`matplotlib-base`) | metapackage dropped (GUI toolkits only) |
| seaborn | 0.13.2 | 0.13.2 | |
| statsmodels | 0.14.6 | 0.14.6 | |
| xarray | 2026.2.0 | 2026.2.0 | PyPI → conda-forge |
| netCDF4 / libnetcdf | 1.7.2 / 4.9.3 | 1.7.2 / 4.9.3 | |
| cftime | 1.6.5 | 1.6.5 | |
| h5py | 3.15.1 (HDF5 1.14.6, bundled) | 3.15.1 (HDF5 1.14.6) | PyPI → conda-forge |
| tables (PyTables) | 3.10.2 (HDF5 **1.14.3**, bundled) | 3.10.2 (HDF5 **1.14.6**) | PyPI → conda-forge; shared HDF5 |
| numexpr | 2.14.1 | 2.14.1 | PyPI → conda-forge |
| openpyxl | 3.1.5 | 3.1.5 | PyPI → conda-forge |
| tqdm | 4.67.3 | 4.67.3 | |
| emcee | 3.1.6 | 3.1.6 | PyPI → conda-forge |
| arviz | 0.23.4 | 0.23.4 | PyPI → conda-forge |
| corner | 2.2.3 | 2.2.3 | PyPI → conda-forge |
| jupyter_core | 5.9.1 | 5.9.1 | |
| nbconvert | 7.17.0 | 7.17.0 (`nbconvert-core`) | |
| nbclient / nbformat | 0.10.4 / 5.10.4 | 0.10.4 / 5.10.4 | |
| ipykernel / ipython | 7.2.0 / 9.10.0 | 7.2.0 / 9.10.0 | |
| jupyter_client | 8.8.0 | 8.8.0 | |
| pytest | 9.0.2 | 9.0.2 | PyPI → conda-forge |
| libopenblas | 0.3.30 openmp | 0.3.30 openmp | same version and threading layer |
| llvm-openmp | 21.1.8 | 21.1.8 | |
| libgfortran5 | 15.2.0 | 15.2.0 | |

Packages needed **only by the preprocessing and data-download scripts**, which are shipped but not run by `run_pipeline.sh`:

| Package | Production | Container | Used by |
|---|---|---|---|
| gsw | 3.6.23 | 3.6.23 | `scripts/en4_thermosteric.py`, `scripts/thermosteric_eos_coefficient.py` |
| cartopy (+ pyproj 3.7.2, pyshp 2.3.1, PROJ 9.7.1) | 0.25.0 | 0.25.0 | `scripts/make_greenland_mask.py` (Natural Earth coastline; downloads on first use) |
| shapely (GEOS 3.14.1) | 2.1.2 | 2.1.2 | `scripts/make_greenland_mask.py` |
| fsspec | 2026.2.0 | 2026.2.0 | `scripts/extract_mar_protect_smb.py` (HTTP byte-range reads); zarr over HTTPS |
| aiohttp | 3.12.15 | 3.12.15 | fsspec HTTP backend |
| requests | 2.34.2 | 2.34.2 | fsspec HTTP; catalog queries |
| zarr (+ numcodecs 0.16.5) | 3.1.6 | 3.1.6 | `xarray.open_zarr` in `scripts/cmip6_zostoga.py`, `scripts/gcm_greenland_temperature.py` |
| dask | 2025.9.1 | 2025.9.1 (`dask-core`) | `xarray.open_mfdataset` in `scripts/cmip6_zostoga.py` |
| plotly | 6.5.2 | 6.5.2 | `notebooks/read_process_datafiles.ipynb` |

## Deviations from production

1. **Operating system and C library.** Production ran on macOS (osx-64), and the container runs on Linux (glibc 2.36). The conda-forge builds have the same versions but different build strings and compilers.
2. **Package source for the former PyPI packages.** h5py, PyTables, xarray, arviz, emcee, corner, openpyxl, numexpr, h5netcdf, xarray-einstats, pytest, the preprocessing packages listed above, and their small helpers now come from conda-forge, at the same versions. In production, the h5py and PyTables wheels each bundled their own HDF5, and PyTables' was 1.14.3. In the container both share conda-forge HDF5 1.14.6. PyTables uses the conda `c-blosc2` 2.23.1 C library; production used the bundled library in the `blosc2` 4.0.0 Python wheel.
3. **`matplotlib-base` instead of `matplotlib`, and `nbconvert-core` instead of `nbconvert`.** Both are the same code without GUI or optional extras. The `jupyter` metapackage (JupyterLab, Notebook server) is not installed, because `jupyter nbconvert --execute` does not need it.
4. **Packages present in production but not imported by any shipped file are omitted.** These include scikit-learn, geopandas, pytorch, jupyter_ai and others. A scan of the imports of every file in this repository (ENTRIES notebooks and scripts, their local helpers, preprocessing scripts, `read_process_datafiles.ipynb`, `src/` and `tests/`) found no use of them. `dask` is installed as `dask-core`, without `distributed`, which no shipped file uses.
5. **Fonts.** The production figures used fonts installed on the Mac. The arete style resolved to Helvetica Neue, which is not redistributable. The image installs Debian `fonts-liberation2` 2.1.5-1 (Liberation Sans, metric-compatible with Arial), and the arete style resolves to Liberation Sans in the container. apt fetches it from the live Debian 12 mirror at build time: the package version is pinned, the mirror is not, so a build fails if the mirror no longer carries 2.1.5-1. Fig. 9 (`component_forecast_per_component.png`) is drawn with the arete style as well. The other figures of `component_forecast.ipynb`, which the paper does not use, keep matplotlib's default style and its bundled DejaVu Sans. Liberation Sans has Arial's metrics, but Helvetica Neue's are close and not identical, so the figures differ in typeface and text widths and positions can shift slightly. The plotted content is unaffected.
6. **Plot style of `component_forecast.ipynb`.** The production run imported a plot-style package that is not on PyPI or conda-forge, and its `use('poster')` call left matplotlib's defaults unchanged. The shipped notebook does not import it. It draws Fig. 9 (`component_forecast_per_component.png`) with the arete `'poster'` style inside `plt.rc_context`, so that figure differs in style from the production file; its other figures, which the paper does not use, keep matplotlib's defaults.
7. **Threads.** The production run used 10 OpenBLAS/OpenMP threads. The image sets `OMP_NUM_THREADS=OPENBLAS_NUM_THREADS=MKL_NUM_THREADS=NUMEXPR_NUM_THREADS=1` (see below).
8. **multidict 6.6.4 and propcache 0.3.2.** These are aiohttp helpers, used only by the download scripts. At these versions they do not solve on conda-forge for linux-aarch64, so the locks have 6.9.1 and 0.5.4.
9. **multiprocessing start method.** `scripts/ocean_early_holdout.py` uses `Pool(2)`. On Linux with Python 3.13 the default start method is `fork`, and on macOS it is `spawn`. Each worker seeds its own generators, so this does not change the draws.

## Numerical parity: expect statistical reproduction, not bitwise identity

Bitwise identity with the macOS production run is **not expected**. Only statistical reproduction is expected: the same distributions within Monte Carlo noise. The paper's numbers are therefore taken from the container run of record (`data/processed/manuscript_numbers.json`), not from a macOS run. A cheap check with seeded numpy, scipy and emcee operations was run in production and in both container architectures:

| Operation | prod, 10 thr | prod, 1 thr | linux-64 (Rosetta) | linux-aarch64 |
|---|---|---|---|---|
| `default_rng(42).normal` | ref | = | = | = |
| `x.sum()`, `x.mean()` | ref | = | = | = |
| `np.linalg.solve` 300×300 | ref | = | = | differs |
| `A @ A` 300×300 | ref | **differs** | differs | differs |
| Cholesky | ref | **differs** | differs | differs |
| `skewnorm.ppf` | ref | = | = | differs |
| `np.exp`, `np.sin` (1000 values) | ref | = | differs | differs |
| emcee, 500 steps, Gaussian target | ref | = | = | = |

What the check shows:

- The numpy random streams (PCG64) and seeded emcee chains with a cheap log-probability are identical everywhere. The random draws that drive the ensembles are therefore the same.
- BLAS level-3 results differ in the last bits with the **thread count, even within production** (10 threads vs. 1). They also differ with the OpenBLAS kernel: production and linux-64 under Rosetta both select `Nehalem`, and linux-aarch64 selects `neoversen1`. Native x86 hosts were not tested; OpenBLAS picks a kernel to match the CPU, so they will likely differ from Nehalem. Transcendental functions differ between the macOS and glibc math libraries and numpy's SIMD dispatch.
- Over long MCMC chains (emcee with BLAS-dependent likelihoods), last-bit differences grow into different chains and different samples. Posteriors and projection quantiles should agree to within Monte Carlo sampling error, not to the last digit.
- linux/amd64 is numerically the closer match to production (same instruction set, and the same OpenBLAS kernel under Rosetta), but on Apple Silicon with Docker Desktop's Rosetta emulation the Jupyter kernels in the amd64 image start and then hang, so no notebook step runs; the scripts and pytest are unaffected. The results of record therefore come from the native linux/arm64 image. The amd64 image is meant for native x86_64 hosts. `check_env.py` starts a kernel to catch this.

### Thread settings (Dockerfile `ENV`)

| Variable | Value | Purpose |
|---|---|---|
| `OMP_NUM_THREADS` | 1 | OpenMP pool (OpenBLAS is the OpenMP build) |
| `OPENBLAS_NUM_THREADS` | 1 | OpenBLAS pool |
| `MKL_NUM_THREADS` | 1 | harmless if MKL is absent (the lock pins `nomkl`) |
| `NUMEXPR_NUM_THREADS`, `NUMEXPR_MAX_THREADS` | 1 | PyTables/numexpr |
| `PYTHONHASHSEED` | 0 | fixed `str` hash order, so any set iteration is repeatable |

With one thread, results do not depend on the host's core count, and the container is repeatable with itself on a given architecture. To trade that for speed, override at run time, e.g. `-e OMP_NUM_THREADS=4 -e OPENBLAS_NUM_THREADS=4`. The run-time cost of one thread compared with production's 10 has not been measured. The nbconvert per-cell timeouts in `run_pipeline.sh` (up to 240 min for `component_ocean`) were set for production speed, so a slower single-thread or emulated run could hit them. `NB_TIMEOUT_SCALE` multiplies them and `NB_TIMEOUT=-1` removes them (see `README.md`).

## Regenerating the locks

```bash
# from environment/, in a throwaway container (nothing installed on the host)
docker run --rm -v "$PWD":/lock -w /lock \
  mambaorg/micromamba:2.10.0-debian12-slim@sha256:bd7f1077166c1c71e9375a0e54835f151d5f216b96301b75f888c53c46d8e78b \
  bash -c "micromamba create -y -q -n cl -c conda-forge conda-lock=4.1.0 && \
           micromamba run -n cl conda-lock lock --micromamba -f environment.yml \
             -p linux-64 -p linux-aarch64 --kind explicit --filename-template 'conda-{platform}.lock'"
```

The image also records what it actually installed, in `/opt/spectrum/installed-explicit.txt` and `/opt/spectrum/installed-pip-freeze.txt` (outside `/work`, so a repository mounted there does not hide them).
