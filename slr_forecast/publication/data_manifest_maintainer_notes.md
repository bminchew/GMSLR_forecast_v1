# Data manifest: maintainer notes (dev repository only; does not ship)

Companion to `data_manifest.md`, which ships as `data/data_manifest.md`.
Items here are open decisions or facts about the development copy that the
public manifest should not carry.

## Before the Zenodo bundle

- **Licenses not yet checked against the provider's terms.** The public
  manifest says "provider terms" (or names the provider) for these:
  NOAA NCEI thermosteric files (US Government work assumed), Berkeley Earth,
  GlaMBIE (WGMS), Zemp et al. 2019, Mouginot et al. 2019 supplement (PNAS),
  Mankoff et al. 2021 (CC-BY-4.0 assumed), IMBIE-3, ISMIP6, Ablain et al. 2019,
  MARv3.12 PROTECT (derived CSV), CMIP6 (derived CSV), Glaude et al. 2024
  (derived CSV), EN4 (Met Office license; the three shipped
  `en4_greenland_peripheral_*.csv` files and two h5 keys are EN4-derived),
  IAP v4.2, the 2021 IMBIE releases and the Horwath and AR6
  observed-GMSL files.
- **Frederikse et al. 2020: checked 2026-11-24.** The Zenodo record
  (10.5281/zenodo.3862995) is `other-closed`, "(c) 2020 All Rights Reserved".
  The pipeline now reads the NASA PO.DAAC copy (JPL_RECON_GMSL v1.0,
  10.5067/GMSLT-FJPL1), whose netCDF says `license: Public Domain` and whose
  values equal the Zenodo Global sheet to float32 precision (max 5.3e-5 mm).
  The seven Frederikse keys of `slr_processed_data.h5` were rebuilt from it.
  Do not ship the Zenodo xlsx.
- **Do not archive `data/raw/` whole.** The development copy also holds
  publisher PDFs (for example the Mouginot et al. 2019 PDFs in
  `data/raw/ice_sheets/greenland/`) and unrelated files.
- **`data/raw/gmslr/matlab`** is a symlink to
  `~/Documents/Research/SLR/forecasting/data_and_projections/sealevelrise/dangendorf2024/matlab`.
  It resolves only on this machine. No shipped code reads it; leave it out.
- **`download_data.sh`** for the "source" route is not written.

## Citation keys

Confirmed by the reader docstrings in `notebooks/slr_data_readers.py`:
Frederikse2020, Dangendorf2024, Horwath2022, Beckley2017, Otosaka2023,
Morice2021. The other keys in the manifest were matched to the dataset by
title. No bib key was found for the Mauna Loa transmission record, CMIP6 or
the Zelinka et al. 2020 ECS table.

## Processed-input rebuild (checked 2026-11-21)

- Fresh export, `read_process_datafiles.ipynb` (53 s) then
  `refresh_processed_keys.py --write` (2 min 34 s): 55 of the 56 data keys and
  `config` equal to the distributed `slr_processed_data.h5` (`assert_frame_equal`,
  `check_exact=True`); `metadata.created` differs; the 56th, the Argo key below,
  is not rebuilt. The notebook alone already reproduces its 51 data keys and
  `config` exactly, so the refresh changes none of them and only adds the four
  NOAA and EN4 keys.
- The notebook failed in a fresh export (no `notebooks/figures/`) at its first
  `savefig`, before writing the h5; it now creates that folder (rerun: 18 s,
  same 51 keys exact).
- `harmonized/df_argo_greenland_200_500m` is in the distributed file but no
  shipped code writes or reads it; a rebuild drops it.
- `extract_en4_layers.py` (3 min 42 s): both layer CSVs byte-identical.
- `integrate_glaude2024_smb.py` (67 s): CSV byte-identical.
- `extract_mar_protect_smb.py`: 3 run-years (CESM2 historical 2000, CESM2
  ssp585 2090, IPSL-CM6A-LR ssp585 2050) identical to the CSV rows (network,
  97-463 s each); a full rebuild is about 57 h (the script's 'per run-year'
  figure is wall time with 4 workers running).
- `en4_greenland_peripheral_200_500m.csv` (March 2026) holds absolute
  temperature from an earlier reader that did not re-baseline. No shipped
  script writes it. The `greenland` fallback that reads it was broken (it
  built `decimal_year` from `year` and `month` columns the file does not have)
  and is fixed; it now gives the key's forcing to 1.1e-15 °C.
- EN4 year range: the 1900–1969 archives in `data/raw/ocean_temp/en4/` were
  added on 2026-10-21, after the stored keys (March 2026) and layer CSVs
  (2026-10-14) were built. Every shipped EN4 regional extraction now passes
  `year_range=(1970, 2021)`; without it the reader would return 1900–2021.
- `harmonized/df_en4_global_0_700m` is not a global mean. `read_en4_regional`
  maps `lon_bounds=(-180, 180)` to 180–180, so the key averages the 180° E
  meridian only (n_valid = 166). It feeds only diagnostic plots in
  `component_ocean.ipynb`; it is reproduced as stored. Fixing the reader would
  change that plot and the key.
- `scripts/en4_thermosteric.py` takes its archive directories on the command
  line (`en4/` and `en4/analyses_2021_2025/`); `en4_thermosteric_annual.csv`
  covers 1950-2025. Its year handling with the 1900-1949 archives now in
  `en4/` was not rechecked, nor were the Greenland mask, `gcm_temperature.csv`,
  the IAP CSV and the zostoga outputs.
- The distributed `cmip6_zostoga_summary.json` (a CACHE file, archived with the
  results) matches only the 10,000-member ocean run. A full
  `SLR_N_SAMPLES=500 ./run_pipeline.sh` (advertised in the runner header) rewrites it at the `zostoga` step, which runs `cmip6_zostoga.py
  analyze` from the cached inputs (`fetch` needs network and is upstream), so
  `numbers` and `figures` pass their zostoga staleness checks. Running selected
  steps after `ocean` without `zostoga` stops `numbers` and `figures` on that
  check; run `./run_pipeline.sh zostoga` first.
