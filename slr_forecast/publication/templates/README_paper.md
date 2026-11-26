# SPECTRUM: global mean sea-level forecast

SPECTRUM forecasts global mean sea level (GMSL) to 2100 by modeling each
contributor separately. Ocean thermal expansion, glaciers, the Antarctic
Peninsula, and Greenland ice discharge are calibrated by Bayesian inference
against observational records. Greenland surface mass balance is emulated from
regional climate model simulations. East Antarctic surface mass balance and
land water storage use published sensitivities and assessments. West
Antarctica follows two storylines, slow and fast. Each temperature-sensitive
component is forced by global mean surface temperature, observed to 2025 and
then following the IPCC AR6 warming paths. The components are summed member by
member over a 10,000-member ensemble. Observed GMSL is withheld from every
calibration and used to test the sum.

This repository holds the code that produces the numbers and figures in

> Minchew, B., Meyer, C., Tiecke, T., Gros, A., White, B., Johnson, L., and
> Bartholomaus, T. A Data-Driven, Component-Level Projection Framework for
> 21st-Century Sea-Level Rise. Manuscript in preparation.

Fig. 1 of the paper is a schematic drawn separately; the code does not produce
it.

## Install (Docker)

The pinned environment and image are in `environment/` (see
`environment/README.md` for details, lockfiles and architecture notes).

```bash
docker build -f environment/Dockerfile -t spectrum:1.0 .
docker run --rm spectrum:1.0 exec python environment/check_env.py   # environment check, no data needed
```

## Data

Data are not stored in this repository. `data/data_manifest.md` lists every
input the pipeline reads, its source, its citation, its license, and how to obtain it. The inputs
that may be redistributed, and the results of the paper, will be archived on
Zenodo; the link will be added here. Place the files under `data/raw/` and
`data/processed/` with the relative paths given there. The small processed CSV
files in `data/processed/` are included. The Docker run mounts a host directory
at `/work/data/processed`, which hides these copies, so the mounted directory
must also hold them (they are also in the Zenodo bundle).

Inputs whose licenses do not permit redistribution must be downloaded from
their providers (route "source" in the manifest). Two of them are processed
files that the full run needs:

- `data/raw/ice_sheets/greenland/mar_protect/mar_protect_integrated_smb.csv`
  (MARv3.12 Greenland surface mass balance). `scripts/extract_mar_protect_smb.py`
  downloads and integrates the model output, which takes about 57 h.
- `data/processed/iap_v42_ohc_global_monthly.csv` (IAP ocean heat content),
  read only by the calibration-record sensitivity test (`thermorec`).
  `scripts/iap_ohc_global.py` builds it. IAP updates its files without changing
  the version label: the paper used files downloaded on 22 October 2026, and
  IAP replaced them on 5 November 2026. A later download, or a later
  recalibration by IAP, can therefore change the `thermorec` results slightly
  from those in the paper.

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

The results in the paper come from a 10,000-member run of the linux/arm64
image on an Apple Silicon Mac, which took 2 h 25 min. The image pins the
package versions of the macOS environment in which the pipeline was developed.
A 500-member run of the same image took 2 h 15 min, because of the fixed draw
counts listed above. The notebook timeouts in
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
third-party datasets and the files derived from non-commercial datasets, which
keep their own terms (see `data/data_manifest.md`).

Four small data files in `data/processed/` are derived from datasets licensed
for non-commercial use only. They are not covered by the BSD license and keep
the terms of their sources:

- `en4_greenland_peripheral_*.csv` (three files) are regional averages of the
  Met Office EN.4.2.2 objective analyses (Good, S. A., M. J. Martin, and
  N. A. Rayner, 2013, J. Geophys. Res. Oceans, doi:10.1002/2013JC009067).
  EN.4.2.2 data were obtained from https://www.metoffice.gov.uk/hadobs/en4/
  and are British Crown Copyright, Met Office, provided under a Non-Commercial
  Government Licence
  (http://www.nationalarchives.gov.uk/doc/non-commercial-government-licence/version/2/).
- `greenland_mask_berkeley_1deg.csv` lists the cells of the Berkeley Earth
  1° grid that lie in Greenland. Berkeley Earth data are licensed under
  CC BY-NC 4.0 (https://berkeleyearth.org/data/).

`data/data_manifest.md` gives the license of every input, including the
Berkeley Earth and EN4 material in the Zenodo data bundle.

## Citation

Please cite the paper and this code (see `CITATION.cff`). The paper and the
Zenodo archives of the code and data do not have DOIs yet; they will be added
here and in `CITATION.cff` when they do.
