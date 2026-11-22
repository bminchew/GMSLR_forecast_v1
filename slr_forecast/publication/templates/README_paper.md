# SPECTRUM: global mean sea-level forecast

<!-- DRAFT for the paper repository (bminchew/spectrum-gmsl). Items in
     [square brackets] are pending decisions. This comment block and the
     brackets go before v1.0. -->

SPECTRUM forecasts global mean sea level to 2150 by modeling each contributor
(ocean thermal expansion, glaciers, the Greenland ice sheet, East and West
Antarctica, the Antarctic Peninsula, and land water storage) as a calibrated
statistical model driven by global mean surface temperature. The component
forecasts are summed with their uncertainties and checked against the observed
sea-level budget, without fitting to the global mean sea-level record.

This repository holds the code that produces the numbers and figures in
[citation of the paper, DOI pending]. Fig. 1 is a hand-drawn schematic; the
code does not produce it.

## Install (Docker)

The pinned environment and image are in `environment/` (see
`environment/README.md` for details, lockfiles and architecture notes).

```bash
docker build -f environment/Dockerfile -t spectrum:1.0 .
docker run --rm spectrum:1.0 exec python environment/check_env.py   # environment check, no data needed
```

## Data

Data are not stored in this repository. `data/data_manifest.md` lists every
input the pipeline reads, its source, its citation, and how to obtain it
[Zenodo data bundle, DOI pending]. Place the files under `data/raw/` and
`data/processed/` with the relative paths given there. The small processed CSV
files in `data/processed/` are included. The Docker run mounts a host directory
at `/work/data/processed`, which hides these copies, so the mounted directory
must also hold them (they are also in the Zenodo bundle).

`data/processed/slr_processed_data.h5` is built from `data/raw/` by
`notebooks/read_process_datafiles.ipynb` (see the manifest for the run order of
the preprocessing scripts).

## Run the full pipeline

```bash
docker run --rm \
  -v /path/to/data/raw:/work/data/raw:ro \
  -v /path/to/data/processed:/work/data/processed \
  -v /path/to/figures:/work/figures \
  spectrum:1.0
```

This runs `run_pipeline.sh`, which executes the notebooks and scripts listed in
its `ENTRIES` array in order (`--list` shows them; naming entries runs only
those). Outside Docker, run `./run_pipeline.sh` from the repository root in an
environment built from `environment/environment.yml`, with the package
installed (`pip install -e .`). The script entries run with `$PYTHON` (default
`python3`) and the notebooks in the `python3` Jupyter kernel; both must be the
same environment, and `run_pipeline.sh` prints both before the first step.

The production ensemble has 10,000 members. Set `SLR_N_SAMPLES` for a reduced
run, for example `-e SLR_N_SAMPLES=500` with Docker or
`SLR_N_SAMPLES=500 ./run_pipeline.sh`. A reduced run overwrites
`data/processed/` in place, so use a scratch copy. These draw counts do not
follow `SLR_N_SAMPLES`: the WAIS mixture in `component_wais.ipynb` (at least
10,000, downsampled to `SLR_N_SAMPLES` for export), the slow-WAIS mixture in
`scripts/build_slow_wais_p90.py` (20,000), the ridge plots of
`component_forecast.ipynb` and `notebooks/precompute_ipcc_distributions.py`
(10,000 each), and the satellite-era quadratic interval in
`component_summation.ipynb` (10,000). Fixed draw counts for bootstraps,
posterior subsamples and diagnostic sweeps also do not follow it
(`src/slr_forecast/config.py` lists them).

The results in the paper come from a run of the linux/arm64 image on an Apple
Silicon Mac. The image pins the package versions of the macOS environment in
which the pipeline was developed; a 10,000-member run there took about
2 h 40 min (10 threads). A 500-member run of the arm64 image took 2 h 15 min. The notebook timeouts in
`run_pipeline.sh` were set for that machine; for a slower run, scale them with
`NB_TIMEOUT_SCALE` or remove them with `NB_TIMEOUT=-1` (see
`environment/README.md`).

## Outputs

- `data/processed/component_results.h5`: posterior samples and projections of
  every component and of the total.
- `data/processed/manuscript_numbers.json` and the other JSON summaries in
  `data/processed/`: the numbers quoted in the paper.
- `figures/` and `figures/diagnostics/`: main-text, supplement and diagnostic
  figures; `notebooks/figures/`: figures from the preprocessing notebook.
- The executed notebooks in `notebooks/`, with their outputs.

## Tests

```bash
docker run --rm spectrum:1.0 exec python -m pytest tests
```

Tests that need data skip when the data are absent.

## Provenance

`PROVENANCE` records the development commit this repository was exported
from and the sha256 of every file.

## License

Code: BSD-3-Clause (`LICENSE`). Data bundle on Zenodo: CC-BY-4.0, except
third-party datasets, which keep their own terms (see `data/data_manifest.md`).

## Citation

[CITATION.cff pending]. Please cite the paper [DOI pending], this code
[Zenodo software DOI pending], and the data bundle [Zenodo data DOI pending].
