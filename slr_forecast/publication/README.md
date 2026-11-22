# Publication export

This folder builds the public repository for the SPECTRUM sea-level paper
(AGU Earth's Future), `bminchew/spectrum-gmsl`, from the development
repository. The development repository stays as it is for future work. The
public repository is a curated copy with no dev history.

## Flow (one way)

```
dev repo (this one) --export_publication.sh--> clone of spectrum-gmsl --commit / tag / push--> GitHub --> Zenodo
```

- All changes are made in the dev repo, then re-exported. Never edit the paper
  repo by hand and never copy anything back.
- The export holds exactly the files that `allowlist.txt` names, plus a fixed
  set of generated files (placeholders, `data/data_manifest.md`, `.gitignore`,
  `PROVENANCE`). A directory entry ships only the files git tracks under it.
- Data are not exported, apart from the small files in allowlist group (e).
  `data_manifest.md` lists every input; the export copies it to
  `data/data_manifest.md`.

## Release flow

1. **Iterate** with working-tree exports into a scratch directory:

   ```bash
   PYTHON=/path/to/project/env/python publication/export_publication.sh /tmp/spectrum-export
   ```

   Modified and untracked shipped files are allowed and listed in
   `PROVENANCE`; personal paths (`/Users/`, `Dropbox`, the home directory)
   only produce warnings.

2. **Final production run** (10,000 members) in the linux/amd64 image, with
   the dev repo mounted at `/work` so notebook outputs, figures and
   `data/processed/` update in place. This run gives the results of record.
   `run_pipeline.sh` runs only the paper's steps; the development-only steps
   are in `run_extras.sh`, which does not ship. Commit the dev repo first.
   The steps add to `data/processed/component_results.h5` and create it only
   if it is absent, so groups from an earlier run (for example those written
   by `run_extras.sh`) would survive into the archived file. Move the old file
   aside first, and keep it until the new run is checked. Then, from the
   project root:

   ```bash
   mv data/processed/component_results.h5 /path/outside/the/repo/component_results.before.h5
   docker build -f environment/Dockerfile --platform linux/amd64 -t spectrum:1.0-amd64 .
   docker run --rm --platform linux/amd64 -e NB_TIMEOUT=-1 \
     -v "$PWD":/work -v "$PWD/data/raw":/work/data/raw:ro \
     spectrum:1.0-amd64
   ```

   Review the diff (numbers, figures, notebook outputs) and commit the outputs.
   Then re-sweep every number in the manuscript from
   `data/processed/manuscript_numbers.json` of this run. The container run
   reproduces the macOS runs within Monte Carlo noise, not digit for digit
   (`environment/VERSIONS.md`), so numbers taken from a macOS run do not carry
   over.

3. **Final export from a commit** into a clone of the paper repo. Commit the
   dev repo first (shipped files and the release tooling in `publication/`),
   then:

   ```bash
   git clone git@github.com:bminchew/spectrum-gmsl.git /path/to/spectrum-gmsl
   PYTHON=/path/to/project/env/python publication/export_publication.sh \
       --from-commit <dev-commit> --force /path/to/spectrum-gmsl
   ```

   `--from-commit` reads every shipped file, the allowlist, the data manifest
   and the templates from that commit (`git cat-file`), so the working tree
   does not matter. The running exporter and checker must be identical to
   their committed versions. `--force` accepts the clone only if it has no
   commits yet or holds a `PROVENANCE` written by this exporter, and only if
   its tracked files are unmodified; so create the GitHub repository empty
   (no README, LICENSE or `.gitignore`). Files tracked in the clone but not in
   the export are deleted, and the exporter lists them. Release mode fails on
   personal paths, so the notebooks must carry outputs from the container
   rerun, or be exported with `--clear-outputs`. It also fails unless the
   export has a top-level `README.md` without draft markers (see below).

4. **Tag**: in the clone, review `git status`, commit, tag `v1.0` and push.
   Put the dev commit (the `source_commit` line of `PROVENANCE`) in the tag
   message.

5. **Zenodo**: the GitHub-Zenodo integration archives the tagged release and
   mints the software DOI. Add the DOI to the README and `CITATION.cff`, and
   link the data bundle DOI.

`--require-clean` is the working-tree alternative to step 3: it exports the
working tree at HEAD and fails unless every shipped file and every release
tool file is tracked, unmodified and not ignored.

**Pending user decisions before v1.0.** The paper repo needs its own top-level
`README.md`, `LICENSE` and `CITATION.cff`. `LICENSE` ships from
`templates/LICENSE` (allowlist line `publication/templates/LICENSE -> LICENSE`).
A draft README is in `templates/README_paper.md`; it ships once the allowlist
has the line `publication/templates/README_paper.md -> README.md`. Until then,
a release export (`--from-commit` or `--require-clean`) fails: release mode
requires a top-level `README.md` with no `[... pending]` placeholder and no
`<!-- DRAFT` comment. `CITATION.cff` does not exist yet.

## Files

| File | Purpose |
|---|---|
| `allowlist.txt` | Paths that ship, grouped and commented. A commented-out path does not ship. `SRC -> DEST` ships the file SRC as DEST. |
| `data_manifest.md` | Every data file the pipeline reads: size, reader, class, source, citation key, route. Ships as `data/data_manifest.md`. |
| `export_publication.sh` | Builds the export in a staging directory, validates it, then installs it into the target. |
| `check_export.py` | The single validator: exact inventory, symlinks, denylist, imports, spec paths, pipeline ENTRIES, executable bits, mentions of the development-only steps, personal paths, and (release mode) the top-level README. |
| `test_export.py` | Release-gating tests (pytest). |
| `templates/gitignore` | Installed as `.gitignore` in the export. |
| `templates/README_paper.md` | Draft top-level README of the paper repo. |
| `templates/LICENSE` | Installed as `LICENSE` in the export (allowlist `SRC -> DEST` line). |

## What the exporter does

1. **Target checks** before anything is written. It refuses a target that
   contains a `..` component, is the filesystem root, is inside the dev git
   repo, or is an ancestor of it. The checks compare file identity
   (`[ a -ef b ]`) of the target's existing ancestors, so both
   `~/Documents/...` and `~/Dropbox/Documents/...` spellings are caught, and
   also compare the logical and physical path strings component by component.
   A non-empty target needs `--force` and must be either a git repository
   with no commits or an earlier export (a `PROVENANCE` written by this
   exporter), with no symlinks outside `.git`. If it is a git repository, its
   tracked files must be unmodified (`git status --untracked-files=no` is
   empty).
2. **Expansion.** Every allowlist entry becomes a list of files. A directory
   entry expands to `git ls-files` (or `git ls-tree` with `--from-commit`).
   Untracked, non-ignored files in an allowlisted directory are an error and
   are listed; ignored files never ship. Symlinks are refused: an allowlisted
   path that is, or passes through, a symlink, any symlink inside an
   allowlisted directory, and symlink or submodule modes in git.
3. **Staging.** The files are copied into a fresh temporary directory,
   together with the generated files. macOS extended attributes are stripped
   (`xattr -cr`). `--clear-outputs` clears notebook outputs in the staging
   copies only. Placeholders are created for the output directories the code
   writes into without creating them: `data/raw/`, `data/processed/`,
   `figures/`, `figures/diagnostics/` and `notebooks/figures/`.
4. **Validation.** `check_export.py` runs on the staging tree with the exact
   expected inventory. Any failure stops the export.
5. **Install.** A new or empty target gets the staging tree by copy and
   rename. An existing clone gets `rsync -a --delete --exclude /.git`, so
   files dropped from the allowlist disappear and `.git` is kept. Before that,
   the exporter refuses if `--delete` would remove a file the clone did not
   ship (not tracked in its index, or not listed in its old `PROVENANCE`),
   such as downloaded data. A dry run after the install must find no
   difference.

On any failure the target is left as it was, and the staging directory is
removed (EXIT trap). One exception: if `rsync` itself fails halfway through a
`--force` install, the clone can be partly updated; `git status` in the clone
shows it. A rerun refuses the modified clone, so first restore its tracked
files (`git checkout -- .`) and remove the untracked files `git status`
lists, then rerun.

`PROVENANCE` records the dev commit, the export mode, the git status of the
shipped files only (paths relative to the project root), the date, the sha256
of the allowlist (and its path, `publication/allowlist.txt`), data manifest,
gitignore template, exporter and checker, the options, and the sha256 of every
exported file. It records no remote URL, branch or path outside the shipped
set. `EXPORT_ALLOWLIST` (an override allowlist, for iteration only) is refused
in release mode and recorded as an override otherwise.

## Checking and testing

```bash
# Static checks of an export (also run by the exporter on its staging tree).
# Run from outside the dev repo with -I so the dev tree cannot shadow it.
cd /tmp && python -I /path/to/slr_forecast/publication/check_export.py \
    /path/to/export --dev-root /path/to/slr_forecast [--mode release]

# Release-gating tests: exports the real working tree to a tmp dir and
# exercises the refusal cases on a throwaway fixture repo.
PYTHONDONTWRITEBYTECODE=1 /Users/minchew/opt/anaconda3/envs/python2025/bin/python \
    -m pytest -p no:cacheprovider publication/test_export.py

# Tests of the exported code, not the editable dev install.
cd /path/to/export && PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH="$PWD/src:$PWD/notebooks" python -m pytest tests/ -p no:cacheprovider
#    Bare 'pytest' from the export root collects only tests/ as well
#    (pyproject.toml sets testpaths); scripts/test_scenario_transfer_mar.py is
#    a pipeline module, not a test file.
```

`check_export.py` reports UNRESOLVED for non-local imports that the running
interpreter cannot find, and fails on them, so run it (and the exporter, via
`PYTHON`) with the project environment. `--allow-unresolved` downgrades them
to warnings for a quick check with another interpreter.

The real-tree tests fail, with the exporter's own message, while the dev tree
is not exportable (for example, untracked files in an allowlisted directory).
`TEST_EXPORT_ALLOWLIST=/path/to/allowlist` runs them with an iteration-only
override.

## Updating the allowlist

When a pipeline entry, helper module or data file is added, add it to
`allowlist.txt` and `data_manifest.md`, commit it, re-export and run the
tests. `MISSING-LOCAL` or `MISSING-MEMBER` in the checker output means that
the export imports something the allowlist does not ship.

## Running the exported pipeline

`run_pipeline.sh` runs the ENTRIES in order, from the ocean component to the
manuscript numbers (`numbers`) and the paper figures (`figures`). The
post-processing JSON files are ENTRIES steps: `thermorec`
(`scripts/thermosteric_record_sensitivity.py`) and `zostoga`
(`scripts/cmip6_zostoga.py analyze`), both between `forecast` and `numbers`.
`zostoga` reads only cached inputs; the network download (`fetch`) and the
AR6 extract (`extract-ar6`) that build them run upstream, like the
preprocessing below.

Steps outside `run_pipeline.sh` (allowlist group a4; `data_manifest.md` gives
the run order):

- `notebooks/read_process_datafiles.ipynb` and `scripts/refresh_processed_keys.py`
  build `data/processed/slr_processed_data.h5`;
- the other preprocessing scripts build the PREP files listed in
  `data_manifest.md`, and `scripts/cmip6_zostoga.py fetch` and `extract-ar6`
  build the cached inputs of `zostoga`.

The dev repo also has `run_extras.sh` (not shipped), which runs the
development-only steps (value of information, optimal defense, Shapley risk,
`results_figures_extra.ipynb`) after a completed `run_pipeline.sh`. It sources
`run_pipeline.sh` for the runner machinery; `run_pipeline.sh` itself does not
know about it.

## Maintainer notes (dev repo only; this folder does not ship)

See also `data_manifest_maintainer_notes.md`.


- **Copyrighted PDFs inside `data/raw/`**: `gmslr/papers/` (34 MB),
  `ice_sheets/*.pdf`, `ice_sheets/greenland/*.pdf`, `ice_sheets/imbie2026/*.pdf`,
  `ice_sheets/ismip6/*.pdf`, `glaciers/*.pdf`, and the IPCC and AR5 PDFs in
  `ipcc_ar6/`. Build the Zenodo bundle from `data_manifest.md`, never by
  zipping `data/raw/`.
- **Broken symlink**: `data/raw/gmslr/matlab` points outside the repo. No
  shipped code reads it.
- **Stale README**: `data/raw/README.md` describes a DOLS calibration and a
  1993–2008 baseline. `data/raw/forcing/README.md` and
  `data/raw/greenland_ocean_temp_sources.md` are current and worth adding to
  the bundle.
- **h5 keys not written by `read_process_datafiles.ipynb`**: the notebook
  opens the h5 with `mode='w'`, which drops `harmonized/df_noaa_tsl_0_700m`,
  `harmonized/df_noaa_tsl_0_2000m`, `harmonized/df_en4_greenland_200_500m` and
  `harmonized/df_en4_global_0_700m`. `scripts/refresh_processed_keys.py
  --write`, run after it, adds them (`data_manifest.md`, sections 1a and 4.1).
  The pipeline reads the NOAA `.dat` files directly; the NOAA keys are read
  only by `tests/test_ocean.py`.
- **`component_results.h5` has no commit attribute.** Record the producing
  commit in the Zenodo metadata.
- **Dependencies of the preprocessing scripts** (group a4) and of
  `cmip6_zostoga.py fetch` (gsw, cartopy, shapely, fsspec, aiohttp, zarr,
  dask, plotly) are in `environment/environment.yml` and the locks; they are
  not needed by `run_pipeline.sh`.
