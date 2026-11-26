# Data manifest: maintainer notes (dev repository only; does not ship)

Companion to `data_manifest.md`, which ships as `data/data_manifest.md`.
Items here are open decisions or facts about the development copy that the
public manifest should not carry.

## Before the Zenodo bundle

- **Licenses checked 2026-11-25** against each provider's own pages, DOI
  records (DataCite, Zenodo API) or license files.
  - **Compatible with the CC-BY-4.0 bundle** (CC BY 4.0, CC0/US public domain,
    or UK OGL v3), attribution required as stated by each provider:
    - NOAA NCEI thermosteric (US public domain, NCEI open data policy);
    - NASA GSFC altimetry GMSL v5.2 (EOSDIS guidance, CC0);
    - Dangendorf 2024 KSSL files (Zenodo 10.5281/zenodo.10621070, CC BY);
    - AR6 observed GMSL and AR6 GMST projections: both are the Figure SPM.8
      data (Fyfe et al. 2021, CEDA 10.5285/98af2184e13e4b91893ab72f301790db),
      CC BY; cite that dataset, not only Fox-Kemper/Lee;
    - AR6 FACTS projections incl. `tlmdata/twolayer_SSPs.h5` (CC BY; CITATION
      file obliges citing Fox-Kemper 2021, Kopp 2023 and the version);
    - IGCC 2024 (CC BY); HadCRUT5 (OGL v3); GISTEMP v4 and NOAAGlobalTemp
      (US public domain, no dataset license); GloSSAC v2.23 (NASA, open);
      NOAA GML Mauna Loa transmission (public domain);
    - GlaMBIE (WGMS policy, "corresponds to" CC BY 4.0); Zemp 2019 (Zenodo
      1492141, v1.0.0, CC BY; v1.1.0 corrects Iceland and the global sum);
      Mankoff 2021 (GEUS Dataverse, CC BY; required phrase "PROMICE mass
      balance product (Larsen et al., 2026, Mankoff et al. 2021)");
    - IMBIE-3 and IMBIE 2021 (UK Polar Data Centre, OGL v3; IMBIE 2021
      dataset citation is Shepherd et al. 2021, PDC);
    - ISMIP6 ComputedScalarsPaper (Zenodo 3940765, CC BY; acknowledge CMIP6,
      ISMIP6 and the modeling groups);
    - Glaude 2024 (CC BY); Noël 2021 mask (CC BY; cite GIMP DEM too);
    - CMIP6 (Terms of Use 6.2: 125 models CC BY 4.0, 6 GISS CC0; check each
      model used against the per-model license page);
    - ESA SLBC_cci v2.2 (ESA terms: "any user for any purpose";
      redistribution not named; medium confidence).
  - **Non-commercial only; must be carved out of the CC-BY grant and labeled:**
    - Berkeley Earth (CC BY-NC 4.0; berkeleyearth.org/data). Affects the
      gridded and annual files, `greenland_mask_berkeley_1deg.csv`, and the
      Berkeley keys of `slr_processed_data.h5`.
    - EN4.2.2 (UK NCGL v2): the three `en4_greenland_peripheral_*.csv`, the
      EN4 keys of the h5, `en4_thermosteric_annual.csv`.
    - Ablain 2019 covariance (SEANOE 10.17882/58344, CC BY-NC 4.0).
  - **No license found; ask the producer or leave to the "source" route:**
    - MARv3.12 PROTECT (ftp.climato.be; no license anywhere): ask Fettweis.
      The derived CSV takes ~57 h to rebuild, so permission matters most here.
    - Mouginot 2019 PNAS Dataset S2 (`sd02.xlsx`; article CC BY-NC-ND; no
      SI license): ask Mouginot/Rignot, or users download from PNAS.
    - IAP v4.2 OHC (usage statement only; asks users to submit results to the
      platform): ask Cheng, or users rebuild with prep:iap.
    - Zelinka ECS (GitHub repo has no LICENSE; Zenodo 6647291 "other-open"):
      ask Zelinka, or prep:zfetch downloads it.
  - **Decision (user, 2026-11-25): option (a).** The bundle stays CC-BY-4.0
    with the Berkeley Earth and EN4 derived files carved out as non-commercial
    (listed at the top of `data_manifest.md`); no permission requests will be
    sent; inputs without a license go to the "source" route.
  - **Open:** IAP (the archived CSV is the input of record, because a fresh
    download no longer reproduces it) and MAR PROTECT (57 h to rebuild; check
    whether ISMIP7 releases MAR under a license).
  - **MAR check (2026-11-25):** ISMIP7 does not use MAR for Greenland
    (its forcing is SDBN1 and dEBM2 downscaling; no data license stated). The
    PROTECT MARv3.12 runs are archived under CC-BY-4.0 only as Goelzer's
    processed ISMIP6-format forcing (NIRD doi:10.11582/2025.prm9am5n, 357 GB,
    all 13 of our scenario runs plus 3 more): annual aSMB anomalies relative
    to 1960-1989, regridded to the ISMIP6 1 km grid and extended beyond the MAR
    ice mask; the reference climatology is not included. The emulator uses
    anomalies relative to the history run's 1995-2005 mean, so the product
    could in principle replace the ftp files, but integrated values would
    differ (regridding, extension, mask) and the emulator would need a refit.
  - **`harmonized/df_argo_greenland_200_500m`** is in the h5 but no shipped
    code reads it; drop it from the distributed copy rather than check Argo
    terms.
  - **Manifest corrections found during the check:** Ablain license is CC
    BY-NC 4.0; the AR6 observed GMSL and GMST files are SPM.8 data; the
    Mouginot file is Dataset S2; record the Zemp (v1.0.0) and Mankoff versions;
    the Berkeley high-res annual file asks for Rohde, Calvert & Hausfather
    (ESSD Discuss. 2026). `data/raw/gmst/noaa_atmoT_landocean.txt` is an
    **October-only** Climate at a Glance series, not annual; only the
    preprocessing overview plot reads it (labeled "annual"); the paper does
    not use NOAAGlobalTemp, GISTEMP or HadCRUT.
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
