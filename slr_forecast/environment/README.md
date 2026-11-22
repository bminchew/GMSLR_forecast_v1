# SPECTRUM runtime environment

This directory holds a pinned software environment and Docker image that run the full SPECTRUM sea-level forecast pipeline (`run_pipeline.sh`).

| File | Purpose |
|---|---|
| `environment.yml` | conda-forge spec. Direct dependencies and transitive packages are pinned to the versions of the macOS development environment. |
| `conda-linux-64.lock`, `conda-linux-aarch64.lock` | explicit, md5-checked locks (conda-lock 4.1.0). The image installs from these. |
| `Dockerfile` | image definition: micromamba 2.10.0 on Debian 12, pinned by digest. |
| `entrypoint.sh` | default command. It runs `bash ./run_pipeline.sh "$@"`. |
| `check_env.py` | import and version check. It needs no data. |
| `VERSIONS.md` | macOS development environment vs. container versions, deviations, and numerical parity. |

## Build

Run the build from the repository root, the directory that holds `run_pipeline.sh`. `.dockerignore` keeps `data/`, HDF5/netCDF files and Python caches out of the build context.

```bash
# from the repository root
docker build -f environment/Dockerfile -t spectrum:1.0 .                         # native architecture
docker build -f environment/Dockerfile --platform linux/amd64 -t spectrum:1.0-amd64 .  # x86_64
```

Each build takes about 2 minutes with a warm base image and a fast network. The image uses about 1.9 GB on disk (about 435 MB compressed).

## Run

Data are not in the image. Mount them:

| Container path | Mode | Contents |
|---|---|---|
| `/work/data/raw` | read-only (`:ro`) | raw observational and model inputs: about 530 MB is read directly by the pipeline; the larger raw archives are needed only for preprocessing (`data/data_manifest.md`, section 3) |
| `/work/data/processed` | read-write | HDF5 intermediates and results (`slr_processed_data.h5`, `component_results.h5`, ...) |
| `/work/figures` | read-write, optional | figures written by the notebooks and scripts; without a mount they stay inside the container |

The mounts must already hold the inputs that `run_pipeline.sh` reads but does not create: the processed bundle of `data/data_manifest.md` (section 3, and sections 1 and 2a for the files). In `data/processed/` these are `slr_processed_data.h5`, which `notebooks/read_process_datafiles.ipynb` builds from `data/raw`; the CSVs shipped in the repository (`greenland_mask_berkeley_1deg.csv` and `en4_greenland_peripheral_*m.csv`); and the `thermorec` and `zostoga` inputs `en4_thermosteric_annual.csv`, `iap_v42_ohc_global_monthly.csv`, `cmip6_zostoga_global.csv` and `cmip6_zostoga_ar6_tlm_ssp245.h5`. Under `data/raw/` they are `steric/cmip6_zostoga/fetch_provenance.json`, `steric/cmip6_zostoga/zelinka_ecs.json` and the derived CSVs under `ice_sheets/greenland/` (`mar_protect/mar_protect_integrated_smb.csv`, `mar_protect/gcm_temperature.csv`, `glaude2024/glaude2024_integrated_smb.csv`). `.dockerignore` keeps all of `data/` out of the image, and the mount hides the repository's `data/processed/`, so the shipped CSVs must also be in the mounted directory.

Run the full pipeline. The results of record come from a run of the linux/arm64 image on an Apple Silicon Mac; the macOS environment is the development environment whose package versions the image pins. At the production ensemble size of 10000, a run in the macOS environment took about 2 h 40 min (10 threads); a 500-member run of the arm64 image took 2 h 15 min. The script entries run with `$PYTHON` (`python3` in the image) and the notebooks in the `python3` Jupyter kernel; `run_pipeline.sh` prints both interpreters before the first step. In the image they are the same interpreter.

```bash
docker run --rm \
  -v /path/to/data/raw:/work/data/raw:ro \
  -v /path/to/data/processed:/work/data/processed \
  -v /path/to/figures:/work/figures \
  spectrum:1.0
```

Other uses:

```bash
docker run --rm spectrum:1.0 --list                         # list the pipeline entries
docker run --rm spectrum:1.0 --help                         # run_pipeline.sh usage
docker run --rm -v ...mounts... spectrum:1.0 ocean glacier  # run selected entries
docker run --rm -e SLR_N_SAMPLES=500 -v ...mounts... spectrum:1.0  # reduced ensemble (test runs)
docker run --rm -e NB_TIMEOUT_SCALE=4 -v ...mounts... spectrum:1.0  # notebook timeouts x4
docker run --rm spectrum:1.0 exec python environment/check_env.py  # check the environment
docker run --rm spectrum:1.0 exec python -m pytest -rs tests      # tests; data-dependent ones skip
docker run --rm -it spectrum:1.0 exec bash                        # shell inside the environment
```

Notes:

- **Ensemble size.** `src/slr_forecast/config.py` reads `SLR_N_SAMPLES`. Unset means 10000, the production value. A reduced run overwrites `data/processed/` in place, so point the processed mount at a scratch copy for test runs.
- **Writable tree.** `nbconvert --inplace` rewrites the notebooks in `/work/notebooks`, and `component_wais` rewrites `notebooks/s1_quadratic_fit.json`. Both happen inside the container. To keep the executed notebooks, mount a copy of `notebooks/` at `/work/notebooks`.
- **Linux hosts.** Add `--user "$(id -u):$(id -g)"` so that files written to the mounted directories belong to you. The tree in the image is world-writable for this reason. Docker Desktop on macOS maps ownership itself.
- **Threads.** The image sets `OMP_NUM_THREADS`, `OPENBLAS_NUM_THREADS`, `MKL_NUM_THREADS` and `NUMEXPR_NUM_THREADS` to 1, so results do not depend on the host's core count. Override with `-e OMP_NUM_THREADS=4 -e OPENBLAS_NUM_THREADS=4` to trade that for speed. See `VERSIONS.md`.
- **Notebook timeouts.** The per-cell nbconvert timeouts in `run_pipeline.sh` (up to 240 min for `component_ocean`) were set for the macOS development machine (10 threads). The container runs one thread, so a run can hit them. `-e NB_TIMEOUT_SCALE=4` multiplies every notebook timeout by 4; `-e NB_TIMEOUT=-1` removes them (`NB_TIMEOUT` sets every notebook timeout in seconds and takes precedence over `NB_TIMEOUT_SCALE`).
- **Preprocessing scripts.** Some shipped scripts download source data: `extract_mar_protect_smb.py`, `gcm_greenland_temperature.py`, `iap_ohc_global.py` and `cmip6_zostoga.py fetch`, plus cartopy's Natural Earth coastline in `make_greenland_mask.py`. These need network access and are not part of `run_pipeline.sh`. The pipeline step `zostoga` runs `cmip6_zostoga.py analyze`, which needs no network: it reads the inputs that `fetch` cached upstream.
- **Architecture.** On Apple Silicon the native image is linux/arm64, and the results of record come from it. Under Docker Desktop's Rosetta emulation the linux/amd64 image cannot run the notebooks (Jupyter kernels hang after starting; `check_env.py` tests this), so use amd64 only on native x86_64 hosts. Neither image is bitwise identical to a macOS run (see `VERSIONS.md`).

## Regenerating the locks

See `VERSIONS.md`, section "Regenerating the locks". After changing `environment.yml`, regenerate both locks and rebuild.
