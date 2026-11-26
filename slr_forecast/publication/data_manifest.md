# SPECTRUM data manifest

Every data file or directory the pipeline reads, with its size, the step that
reads it, its class, its upstream source and how it is distributed. Paths are
relative to the repository root; place files at exactly these paths. Sizes
are on-disk sizes on 2026-11-21.

Data are not stored in the git repository, apart from the four small files in
`data/processed/` marked **ship** below. Those four are also in the Zenodo
bundle, because the Docker image mounts `data/` at run time and does not see
the copies in the repository.

License: the Zenodo data bundle is distributed under CC-BY-4.0, except
third-party datasets, which keep their own terms (License column of §2a),
and the files derived from non-commercial datasets, which keep those terms:

- from Berkeley Earth (CC BY-NC 4.0, non-commercial only):
  `data/processed/greenland_mask_berkeley_1deg.csv` and the Berkeley Earth
  keys of `slr_processed_data.h5` (`raw/df_berkeley`,
  `harmonized/df_berkeley_h`, `kinematics/gmst/berkeley`);
- from EN.4.2.2 (British Crown Copyright, Met Office, UK Non-Commercial
  Government Licence v2, non-commercial only):
  `data/processed/en4_greenland_peripheral_*.csv`,
  `data/processed/en4_thermosteric_annual.csv` and the EN4 keys of
  `slr_processed_data.h5` (`harmonized/df_en4_*`).

`slr_processed_data.h5` therefore contains non-commercial material; its other
keys are CC-BY-4.0 or carry the terms of their sources. Inputs with no license
that permits redistribution are not in the bundle and must be downloaded from
the provider (Route "source").

## Classes

| Class | Meaning |
|---|---|
| **RAW** | Observational or model input, used as downloaded (or unzipped). |
| **CACHE** | Derived file that a `run_pipeline.sh` step rebuilds. Not needed before a run. |
| **PREP** | Processed input that the pipeline reads but does not rebuild. Its build code is in the repository (`notebooks/read_process_datafiles.ipynb` or `scripts/`), outside `run_pipeline.sh`. Section 4 gives the build order and run times. |

## Distribution routes

| Route | Meaning |
|---|---|
| **ship** | Small; committed in the repository. |
| **zenodo** | In the Zenodo processed bundle (one DOI for the paper). |
| **source** | Download from the original provider. |

## Pipeline steps

These names are used in the "Read by" column. They follow `run_pipeline.sh`.

- ENTRIES: `ocean`, `oceanholdout`, `glacier`, `glacierdelay`, `greenland`, `smbsupp`, `smbrcm`, `smbstate`, `smblag`, `eais`, `peninsula`, `pencurv`, `wais`, `ipccdist`, `summation`, `forecast`, `thermorec`, `zostoga`, `slowwais`, `grsens`, `slownums`, `varshares`, `numbers`, `figures`.
- Preprocessing: `prep:read_process` (read_process_datafiles.ipynb), `prep:refresh` (refresh_processed_keys.py), `prep:en4layers` (extract_en4_layers.py), `prep:mask` (make_greenland_mask.py), `prep:glaude` (integrate_glaude2024_smb.py), `prep:mar` (extract_mar_protect_smb.py), `prep:gcmT` (gcm_greenland_temperature.py), `prep:en4thermo` (en4_thermosteric.py), `prep:iap` (iap_ohc_global.py), `prep:eos` (thermosteric_eos_coefficient.py), `prep:zfetch` (cmip6_zostoga.py fetch), `prep:zar6` (cmip6_zostoga.py extract-ar6).

`thermorec` is `scripts/thermosteric_record_sensitivity.py`. It refits the
ocean model to NOAA, EN4 and IAP (six fits in parallel, about 13 min) and
writes `thermosteric_record_sensitivity.json`. `zostoga` is
`scripts/cmip6_zostoga.py` without arguments (its `analyze` subcommand, about
5 s). It compares CMIP6 thermosteric sea level with the observations and the
ocean results and writes `cmip6_zostoga_summary.json`, from cached inputs
only (§4.2).

---

## 1. Processed files in `data/processed/`

| Path | Size | Read by | Class | Built by | Route |
|---|---|---|---|---|---|
| `data/processed/slr_processed_data.h5` | 3.8 MB | all components, oceanholdout, glacierdelay, smbsupp, smbstate, pencurv, ipccdist, summation, forecast, thermorec, zostoga, slownums, numbers, figures, prep:en4layers | **PREP** (keys in §1a) | prep:read_process, then prep:refresh | zenodo |
| `data/processed/component_results.h5` | 2.9 GB | written by every component, then read by all later steps | **CACHE** | the pipeline (10,000-sample production run) | zenodo (the results of record) |
| `data/processed/ipcc_distributions.h5` | 83 MB | forecast, summation (via component_projections) | **CACHE** | ipccdist (`notebooks/precompute_ipcc_distributions.py`) | rebuilt; zenodo optional |
| `data/processed/manuscript_headline_stats.json` | 24 KB | figures (cross-check of the exceedance table) | **CACHE** | forecast | rebuilt; zenodo with the results |
| `data/processed/forecast_reference_models.json` | 4 KB | numbers | **CACHE** | forecast | rebuilt; zenodo with the results |
| `data/processed/slow_wais_p90_numbers.json` | 4 KB | (output) | **CACHE** | slownums | rebuilt; zenodo with the results |
| `data/processed/manuscript_numbers.json` | 28 KB | (output) | **CACHE** | numbers | rebuilt; zenodo with the results |
| `notebooks/s1_quadratic_fit.json` | 4 KB | wais (at import), forecast, figures, other `component_projections` users | **CACHE**, but must exist before `wais` runs: `load_s1_quadratic()` raises if it is absent | wais | **ship** |
| `data/processed/thermosteric_record_sensitivity.json` | 32 KB | zostoga, numbers, figures | **CACHE** | thermorec | rebuilt; zenodo with the results |
| `data/processed/thermosteric_record_fits/` | 24 KB | thermorec (per-fit output, collected into the JSON) | **CACHE** | thermorec | rebuilt; zenodo with the results |
| `data/processed/cmip6_zostoga_summary.json` | 48 KB | numbers, figures | **CACHE** | zostoga | rebuilt; zenodo with the results |
| `data/processed/cmip6_zostoga_global.csv` | 492 KB | zostoga | **PREP** | prep:zfetch (Pangeo and ESGF downloads; see §4.2) | zenodo |
| `data/processed/cmip6_zostoga_ar6_tlm_ssp245.h5` | 11 MB | zostoga | **PREP** | prep:zar6: the `ssp245/surface_temperature`, `ssp245/ECS` and `ssp245/TCR` datasets of the AR6 `tlmdata/twolayer_SSPs.h5` (574 MB), copied unchanged. Without it, `zostoga` reads the full file. | zenodo |
| `data/processed/greenland_mask_berkeley_1deg.csv` | 12 KB | greenland | **PREP** | prep:mask, from `berkEarth_Global_TAVG_Gridded_1deg.nc` (cartopy, shapely) | **ship**, and zenodo |
| `data/processed/en4_greenland_peripheral_100_400m.csv` | 44 KB | greenland (ocean-layer sensitivity) | **PREP** | prep:en4layers, from EN4 | **ship**, and zenodo |
| `data/processed/en4_greenland_peripheral_300_600m.csv` | 44 KB | greenland (ocean-layer sensitivity) | **PREP** | prep:en4layers, from EN4 | **ship**, and zenodo |
| `data/processed/en4_greenland_peripheral_200_500m.csv` | 44 KB | greenland, only when the h5 key `harmonized/df_en4_greenland_200_500m` is absent | **PREP** (shortcut) | the same EN4 extraction as that key, stored as absolute temperature (the key holds anomalies relative to 1995–2005). After the 1995–2005 re-baselining in `greenland`, the two give the same forcing to within 1.1e-15 °C. No shipped script writes this file; prep:refresh builds the key from EN4. | **ship**, and zenodo |
| `data/processed/en4_thermosteric_annual.csv` | 12 KB | thermorec | **PREP** | prep:en4thermo, from EN4 and TEOS-10 (gsw) | zenodo |
| `data/processed/iap_v42_ohc_global_monthly.csv` | 40 KB | thermorec | **PREP** | prep:iap (downloads IAP v4.2). The paper's CSV was built from files downloaded on 2026-10-22. On 2026-11-05 IAP replaced every v4.2 file under the same version label and URL, so a later download gives slightly different values (January 2020, 0–300 m: 105.40 ZJ, compared with 106.01 ZJ in the paper's CSV), and the thermorec results may not match the paper exactly. | source (prep:iap; step 9 below) |

`numbers` and `figures` check that both JSON files match the current
`component_results.h5` (the 2100 thermosteric medians must agree to 0.05 mm),
and stop with the name of the script to rerun when a file is missing or
stale. `thermorec` and `zostoga` rebuild their JSON files on every full run;
after a partial rerun that includes `ocean`, rerun both before `numbers`.

### 1a. Keys of `slr_processed_data.h5`

`read_process_datafiles.ipynb` opens the file with `mode='w'` and writes the
first group of keys below. `refresh_processed_keys.py --write` must follow
it: it rewrites five tables with the current readers and adds the keys the
notebook does not write. Rebuilt this way from the raw files, the 55 data keys
listed below and `config` equal those of the distributed file exactly (pandas
`assert_frame_equal`, `check_exact=True`); only the creation time in
`metadata` differs.

| Key(s) | Read by | Written by |
|---|---|---|
| `raw/df_{frederikse, dangendorf, horwath, nasa_gmsl, ipcc_observed_gmsl, imbie_wais, berkeley, hadcrut, gistemp, noaa, glossac, mlo_transmission}` | components, figures, ar6_gmst.py | prep:read_process; prep:refresh rewrites berkeley, dangendorf, imbie_wais, horwath, ipcc_observed_gmsl |
| `harmonized/df_*_h` (10 tables, baseline 1995–2005) | components, scripts, figures | prep:read_process; prep:refresh (5 of them) |
| `derived/df_frederikse_thermo`, `derived/df_frederikse_thermo_h` | ocean, figures | prep:read_process |
| `kinematics/gmsl/*`, `kinematics/gmst/berkeley` | figures, forecast | prep:read_process; prep:refresh (berkeley, ipcc) |
| `projections/temp/{Historical, SSP1_1_9, …, SSP5_8_5}` (AR6 GMST) | every component, warming paths, thermorec, zostoga | prep:read_process |
| `projections/gmsl/*`, `projections/gmsl_low/*` (AR6 GMSL) | figures, forecast | prep:read_process |
| `config`, `metadata` | (record of the build) | prep:read_process |
| `harmonized/df_noaa_tsl_0_700m`, `harmonized/df_noaa_tsl_0_2000m` | `tests/test_ocean.py` (0–700 m). The pipeline reads the NOAA `.dat` files directly (ocean, oceanholdout, thermorec). | prep:refresh, from `data/raw/steric/noaa_tsl_yearly_0-*m.dat` |
| `harmonized/df_en4_greenland_200_500m` | greenland (the Greenland discharge forcing); prep:en4layers (reference check) | prep:refresh, from EN4 1970–2021 (58–80° N, 75–5° W, 200–500 m) |
| `harmonized/df_en4_global_0_700m` | ocean (diagnostic plots only). If the key is absent, `ocean` computes it from EN4 with the same arguments and stores it in this file. | prep:refresh, from EN4 1970–2021 |
| `harmonized/df_argo_greenland_200_500m` | not read by shipped code | not rebuilt (present in the distributed file only) |

The EN4 keys use the yearly archives 1970–2021 only (`year_range=(1970, 2021)`
in the code); the EN4 folder may also hold earlier years.

---

## 2. Raw inputs in `data/raw/`

"Read directly" means that a `run_pipeline.sh` step reads the file. Files
needed only by preprocessing are in §2b.

### 2a. Read directly by `run_pipeline.sh` steps

| Path | Size | Read by | Source (citation key) | License | Route |
|---|---|---|---|---|---|
| `data/raw/steric/noaa_tsl_yearly_0-700m.dat` | 8 KB | ocean, oceanholdout, thermorec; prep:refresh | NOAA NCEI thermosteric sea level, 0–700 m, yearly (Levitus2012) | US public domain (NCEI open data policy) | zenodo or source |
| `data/raw/steric/noaa_tsl_yearly_0-2000m.dat` | 4 KB | ocean, oceanholdout, thermorec; prep:refresh | NOAA NCEI thermosteric sea level, 0–2000 m (Levitus2012) | US public domain (NCEI open data policy) | zenodo or source |
| `data/raw/gmst/berkEarth_Global_TAVG_annual_highres_prelim.txt` | 16 KB | ocean | Berkeley Earth global annual mean (Rohde2020) | **CC BY-NC 4.0** (Berkeley Earth; non-commercial only) | source |
| `data/raw/gmst/berkEarth_Global_TAVG_Gridded_1deg.nc` | 408 MB | greenland; prep:mask | Berkeley Earth 1° gridded TAVG (Rohde2020) | **CC BY-NC 4.0** (Berkeley Earth; non-commercial only) | source |
| `data/raw/forcing/igcc2024/ERF_best_aggregates_1750-2024.csv` | 64 KB | ocean, oceanholdout, summation, thermorec (`config.VOLC_ERF_CSV`) | Indicators of Global Climate Change 2024, v2025.06.11, doi:10.5281/zenodo.15639576 (Forster2025) | CC-BY-4.0 (`LICENSE` in archive) | zenodo |
| `data/raw/forcing/README.md` | 4 KB | (documentation) | Provenance of the forcing files: release, checksum, columns, the AR6 volcanic convention, and how the pipeline uses the file | CC-BY-4.0 | zenodo |
| `data/raw/ipcc_ar6/slr/ar6/global/confidence_output_files/` | 1.4 MB | ocean, glacier, greenland, eais, peninsula, wais, ipccdist, summation, forecast, thermorec, numbers, figures; prep:read_process | IPCC AR6 sea-level projections, FACTS (Garner2023; Fox-Kemper2021) | CC-BY-4.0 (`LICENSE` in the AR6 folder; cite Fox-Kemper2021, Kopp2023 and the version) | zenodo or source |
| `data/raw/ipcc_ar6/slr/ar6/global/dist_components/` | 4 MB | wais | as above | CC-BY-4.0 | zenodo or source |
| `data/raw/ipcc_ar6/slr/ar6/global/full_sample_workflows/{wf_1e,wf_2e,wf_3e,wf_4}/…_globalsl.nc` (4 files) | 2.1 MB (folder: 147 MB) | wais | as above | CC-BY-4.0 | zenodo (the 4 files) or source |
| `data/raw/glaciers/0_global_glambie_consensus.csv` | 4 KB | glacier | GlaMBIE global consensus, doi:10.5904/wgms-glambie-2024-07 (GlaMBIE2025) | WGMS data policy (corresponds to CC-BY-4.0) | zenodo or source |
| `data/raw/glaciers/glambie/GlaMBIE_Data_DOI_10.5904_wgms-glambie-2024-07/` | 3.2 MB | glacierdelay, numbers (via glacier_bookkeeping_check, glacier_zemp_diagnostic) | GlaMBIE regional results (GlaMBIE2025) | WGMS data policy (corresponds to CC-BY-4.0) | zenodo or source |
| `data/raw/glaciers/zemp2019/Zemp_etal_results_global.csv` | 8 KB | glacier, glacierdelay, numbers, figures | Zemp et al. 2019 (Zemp2019) | CC-BY-4.0 (Zenodo 10.5281/zenodo.1492141, version 1.0.0) | zenodo or source |
| `data/raw/ice_sheets/greenland/mouginot2019_data.xlsx` | 1.6 MB | greenland, smbsupp, smbstate | Mouginot et al. 2019 PNAS supplement, Dataset S2 (`sd02.xlsx`, renamed) (Mouginot2019) | no license found for the supplement (the article is CC BY-NC-ND 4.0) | source |
| `data/raw/ice_sheets/greenland/mankoff/MB_SMB_D_BMB_ann.csv` | 20 KB | greenland, smbsupp, smbstate | Mankoff et al. 2021 Greenland mass balance (Mankoff2021) | CC-BY-4.0 (GEUS Dataverse doi:10.22008/FK2/OHI23Z; cite as "PROMICE mass balance product (Larsen et al., Mankoff et al. 2021)"; record the version) | zenodo or source |
| `data/raw/ice_sheets/imbie2026/imbie3_greenland_mm_partitioned.csv` | 104 KB | greenland, smbstate | IMBIE-3 (Otosaka2026) | UK Open Government Licence v3.0 (UK Polar Data Centre) | zenodo or source |
| `data/raw/ice_sheets/imbie2026/imbie3_east_antarctica_mm_partitioned.csv` | 88 KB | eais | IMBIE-3 (Otosaka2026) | UK Open Government Licence v3.0 (UK Polar Data Centre) | zenodo or source |
| `data/raw/ice_sheets/imbie2026/imbie3_antarctic_peninsula_mm_partitioned.csv` | 88 KB | peninsula, pencurv | IMBIE-3 (Otosaka2026) | UK Open Government Licence v3.0 (UK Polar Data Centre) | zenodo or source |
| `data/raw/ice_sheets/imbie2026/imbie3_west_antarctica_mm_partitioned.csv` | 88 KB | wais | IMBIE-3 (Otosaka2026) | UK Open Government Licence v3.0 (UK Polar Data Centre) | zenodo or source |
| `data/raw/ice_sheets/ismip6/ComputedScalarsPaper/` | 103 MB | eais, peninsula | ISMIP6 Antarctica scalars (Seroussi2020) | CC-BY-4.0 (Zenodo 10.5281/zenodo.3940766; acknowledge CMIP6, ISMIP6 and the modeling groups) | source |
| `data/raw/gmslr/ablain2019_gmsl_error_covariance.nc` | 5.3 MB | summation, forecast | Altimetry GMSL error covariance (Ablain2019) | **CC BY-NC 4.0** (SEANOE doi:10.17882/58344; non-commercial only) | zenodo or source |
| `data/raw/ice_sheets/greenland/mar_protect/mar_protect_integrated_smb.csv` | 88 KB | greenland, smbsupp, smbrcm and smbstate (via `smb_emulator.py`), smblag | **PREP**, derived: MARv3.12 PROTECT multi-GCM SMB, integrated by prep:mar | no license found (ftp.climato.be) | source (prep:mar rebuilds it from ftp.climato.be, about 57 h; step 6 below) |
| `data/raw/ice_sheets/greenland/mar_protect/gcm_temperature.csv` | 80 KB | greenland, smbsupp, smbrcm and smbstate (via `smb_emulator.py`), smblag | **PREP**, derived: CMIP6 GMST and Greenland temperature from Pangeo, built by prep:gcmT | CMIP6 Terms of Use: CC-BY-4.0, or CC0 for the GISS models | zenodo |
| `data/raw/ice_sheets/greenland/glaude2024/glaude2024_integrated_smb.csv` | 16 KB | smbrcm | **PREP**, derived: Glaude et al. 2024 RCM SMB (doi:10.5281/zenodo.13270554; Glaude2024), built by prep:glaude from `data/raw/glaude2024/` | CC-BY-4.0 (Zenodo 10.5281/zenodo.13270554) | zenodo |
| `data/raw/ipcc_ar6/slr/ar6/global/full_sample_components/oceandynamics-tlm-oceandynamics-ssp245_globalsl.nc` | 612 KB | zostoga | IPCC AR6 FACTS thermal-expansion samples, SSP2-4.5 (Garner2023) | CC-BY-4.0 | zenodo or source |
| `data/raw/steric/cmip6_zostoga/fetch_provenance.json` | 44 KB | zostoga | **PREP**, derived: members, data sources and drift fits of the CMIP6 zostoga download, written by prep:zfetch | CMIP6 Terms of Use: CC-BY-4.0, or CC0 for the GISS models | zenodo |
| `data/raw/steric/cmip6_zostoga/zelinka_ecs.json` | 80 KB | zostoga | ECS of the CMIP6 models (Zelinka et al. 2020, GitHub `mzelinka/cmip56_forcing_feedback_ecs`), downloaded by prep:zfetch | no license found (GitHub repository has none) | source (prep:zfetch downloads it) |

The three derived CSVs under `data/raw/ice_sheets/greenland/` and
`fetch_provenance.json` are derived files stored under `data/raw/`. Their
class is PREP, not RAW.

### 2b. Read only by preprocessing (inputs of `slr_processed_data.h5` and the PREP files)

| Path | Size | Read by | Source (citation key) | Route |
|---|---|---|---|---|
| `data/raw/gmslr/JPL_RECON_GMSL_global_timeseries_measures.nc` | 86 KB | prep:read_process | Frederikse et al. 2020 (Frederikse2020), as distributed by NASA PO.DAAC: JPL_RECON_GMSL v1.0, `global_timeseries_measures.nc` renamed, doi:10.5067/GMSLT-FJPL1, public domain | source or zenodo |
| `data/raw/gmslr/dangendorf2024_KSSLfin.mat`, `dangendorf2024_KSSL_SEfin.mat` | 426 MB + 227 MB | prep:read_process, prep:refresh | Dangendorf et al. 2024 (Dangendorf2024) | source |
| `data/raw/gmslr/horwath2021_ESACCI_SLBC_v2.2.csv` | 112 KB | prep:read_process, prep:refresh | ESA CCI sea-level budget closure (Horwath2022) | source or zenodo |
| `data/raw/gmslr/nasa_GMSL_TPJAOS_5.2.txt` | 148 KB | prep:read_process | NASA GSFC altimetry GMSL v5.2 (Beckley2017) | source |
| `data/raw/gmslr/ipccar6_global_sea_level_observed.csv` | 4 KB | prep:read_process, prep:refresh | IPCC AR6 Figure SPM.8 data, observed GMSL (Fyfe2021data; CC-BY-4.0, CEDA doi:10.5285/98af2184e13e4b91893ab72f301790db) | source or zenodo |
| `data/raw/ice_sheets/antarctica/imbie_west_antarctica_2021_mm.csv` | 12 KB | prep:read_process, prep:refresh | IMBIE 2021 release (Otosaka2023) | source or zenodo |
| `data/raw/gmst/berkEarth_globalmean_airTaboveseaice.txt` | 208 KB | prep:read_process, prep:refresh | Berkeley Earth (Rohde2020) | source |
| `data/raw/gmst/hadcrut5_global_monthly.csv` | 88 KB | prep:read_process | HadCRUT5 (Morice2021) | source |
| `data/raw/gmst/nasa_GLB.Ts+dSST.csv` | 16 KB | prep:read_process | GISTEMP v4 (Lenssen2019) | source |
| `data/raw/gmst/noaa_atmoT_landocean.txt` | 4 KB | prep:read_process | NOAAGlobalTemp (Vose2021) | source |
| `data/raw/saod/GloSSAC_V2.23_NC4.nc` | 530 MB | prep:read_process | GloSSAC v2.23 (Thomason2018) | source |
| `data/raw/saod/maunaLoa_transmission.txt` | 28 KB | prep:read_process | NOAA GML Mauna Loa apparent transmission | source |
| `data/raw/ipcc_ar6/atmo_temp/` | 44 KB | prep:read_process (`projections/temp/*`) | IPCC AR6 Figure SPM.8 data, GMST projections (Fyfe2021data, Lee2021; CC-BY-4.0, CEDA doi:10.5285/98af2184e13e4b91893ab72f301790db) | zenodo or source |
| `data/raw/ocean_temp/en4/EN.4.2.2.analyses.g10.{1970…2021}.zip` | 15 GB | prep:refresh, prep:en4layers, prep:eos; ocean only if `harmonized/df_en4_global_0_700m` is missing from the h5 | Met Office EN4.2.2 objective analyses (Good2013) | source |
| `data/raw/ocean_temp/en4/EN.4.2.2.analyses.g10.{1950…2020}.zip` and `data/raw/ocean_temp/en4/analyses_2021_2025/EN.4.2.2.analyses.g10.{2021…2025}.zip` | 21 GB + 1.5 GB (includes the 1970–2020 files above) | prep:en4thermo (`--zip-dirs` takes both folders) | as above | source |
| `data/raw/glaude2024/{MAR,HIRHAM,RACMO}_SMB_1km.nc` | 9.6 GB | prep:glaude | Glaude et al. 2024 (Glaude2024), doi:10.5281/zenodo.13270554 | source |
| `data/raw/ice_sheets/greenland/noel2021/Icemask_Topography_lon_lat_average_1km_GrIS.nc` | 62 MB | prep:glaude, prep:mar | Noël et al. 2021 (Noel2021), doi:10.5281/zenodo.4289959 | source |
| `data/raw/ipcc_ar6/slr/ar6/tlmdata/twolayer_SSPs.h5` | 574 MB | prep:zar6; zostoga only if `cmip6_zostoga_ar6_tlm_ssp245.h5` is absent | IPCC AR6 FACTS two-layer model (Garner2023) | source |
| `data/raw/steric/cmip6_zostoga/` (rest of the folder) | 139 MB | prep:zfetch (download cache: Pangeo catalogs, ESGF files, `tas_global_annual/`) | CMIP6 zostoga and tas (Pangeo, ESGF) | source (the script downloads it) |
| `data/raw/ocean_temp/cmip6_greenland/cmip6_thetao_greenland_200_500m.csv` | in 6.7 MB folder | prep:zfetch (optional reuse of GMST) | CMIP6 (Pangeo) | not needed |

### 2c. The 59 GB `data/raw/ocean_temp/`

| Subfolder | Size | Used? |
|---|---|---|
| `en4/` | 52 GB | Preprocessing only (§2b), and the `ocean` fallback above. A full `run_pipeline.sh` run with the processed bundle does **not** need it. |
| `ecco_v4r4/` | 6.0 GB | Not read by shipped code. |
| `argo/` | 1.2 GB | Not read by shipped code (`readers/ocean_temp.py` has an Argo reader that no pipeline step calls). |
| `cmip6_greenland/` | 6.7 MB | Optional reuse by prep:zfetch; not needed. |

### 2d. Read only by the tests

Tests that need data skip when the data are absent: with no `data/`, 266
tests pass and 224 skip (2026-11-21).

| Path | Size | Source | Route |
|---|---|---|---|
| `data/raw/ice_sheets/Gt/imbie_*_2021_Gt.csv` (6 files) | 116 KB | IMBIE 2021 release, Gt format (Otosaka2023) | zenodo or source |
| `data/raw/ice_sheets/antarctica/imbie_west_antarctica_2021_mm.csv`, `data/raw/glaciers/0_global_glambie_consensus.csv`, `data/raw/gmslr/ipccar6_global_sea_level_observed.csv` | listed above | as above | as above |

### 2e. Files in `data/raw/` that no shipped code reads

`data/raw/tws/` (537 MB), `data/raw/enso/` (85 MB), `data/raw/adaptation/`
(45 MB), `data/raw/ismip6_greenland/` (56 MB),
`data/raw/ice_sheets/imbie2026/imbie3_antarctica_mm_partitioned.csv`,
`data/raw/gmslr/frederikse2020_global_basin_timeseries.xlsx`,
`data/raw/gmslr/frederikse2020_global_timeseries.xlsx` (the Zenodo copy of
the same numbers, "All Rights Reserved"; do not redistribute),
`data/raw/gmslr/matlab/` and the rest of `data/raw/ipcc_ar6/`. None of
these is needed.

---

## 3. Size summary

| What | Size |
|---|---|
| Processed bundle needed to run `run_pipeline.sh` (`slr_processed_data.h5`, the PREP files above, the three derived CSVs under `data/raw/ice_sheets/greenland/`, the zostoga inputs under `data/raw/steric/cmip6_zostoga/`) | ≈ 17 MB, of which 11 MB is `cmip6_zostoga_ar6_tlm_ssp245.h5` |
| Raw files read directly by `run_pipeline.sh` steps (§2a) | ≈ 530 MB, of which 408 MB is the Berkeley Earth gridded file and 103 MB is ISMIP6 |
| Production results (`component_results.h5`) | 2.9 GB |
| Raw files needed only to rebuild the processed inputs (§2b) | ≈ 35 GB, dominated by EN4 1950–2025 (22.5 GB) and the Glaude 2024 fields (9.6 GB) |

---

## 4. Rebuilding the processed inputs

Run from the repository root unless noted. Each step writes only the files
named. Run times are wall-clock times on a 10-core MacBook Pro (2021) with the
data on local disk; "not timed" steps were not rerun for this manifest.

### 4.1 Order

| Step | Command | Writes | Needs | Time |
|---|---|---|---|---|
| 1 | `cd notebooks && jupyter nbconvert --to notebook --execute --output read_process_datafiles.executed.ipynb read_process_datafiles.ipynb` | `slr_processed_data.h5` (deletes all keys first); three overview figures in `notebooks/figures/`; the executed notebook copy | §2b files | 1 min |
| 2 | `python scripts/refresh_processed_keys.py --write` | 16 keys of `slr_processed_data.h5`, 4 of them new | §2b files, NOAA `.dat`, EN4 1970–2021 | 2.5 min |
| 3 | `python scripts/extract_en4_layers.py` | `en4_greenland_peripheral_{100_400,300_600}m.csv` | step 2, EN4 | 4 min |
| 4 | `python scripts/make_greenland_mask.py` | `greenland_mask_berkeley_1deg.csv` | Berkeley Earth gridded file | not timed |
| 5 | `python scripts/integrate_glaude2024_smb.py` | `glaude2024_integrated_smb.csv` | `data/raw/glaude2024/`, Noël mask | 1 min |
| 6 | `python scripts/extract_mar_protect_smb.py --tier 1`, then `--tier 2` and `--tier 3` | `mar_protect_integrated_smb.csv` (appends; skips finished run-years) | network (ftp.climato.be), Noël mask; imports step 5's script | about 57 h for all 1,364 run-years (about 150 s of wall time per run-year with the default 4 workers) |
| 7 | `python scripts/gcm_greenland_temperature.py` | `gcm_temperature.csv` | network (Pangeo CMIP6) | not timed |
| 8 | `python -I scripts/en4_thermosteric.py clim`, `series`, then `annual` (see its docstring) | `en4_thermosteric_annual.csv` | EN4 1950–2025, gsw | not timed |
| 9 | `python -I scripts/iap_ohc_global.py --csv … --tmp …` (see its docstring) | `iap_v42_ohc_global_monthly.csv` | network (IAP v4.2) | not timed |
| 10 | `python scripts/thermosteric_eos_coefficient.py` | nothing; prints `KAPPA_EOS` (set in `src/slr_forecast/config.py`) | EN4 | not timed |

Steps 1 and 2 reproduce every key of the distributed `slr_processed_data.h5`
that shipped code reads, exactly. Steps 3 and 5 reproduce their files byte for
byte. Step 6 reproduces each run-year it extracts, as checked on a subset.

### 4.2 CMIP6 zostoga comparison (`zostoga`)

`scripts/cmip6_zostoga.py` has three subcommands. `analyze` is the
`run_pipeline.sh` entry `zostoga` (the default when the script is run
without arguments). It reads only local files: `cmip6_zostoga_global.csv`,
`cmip6_zostoga_ar6_tlm_ssp245.h5`, `fetch_provenance.json` and
`zelinka_ecs.json` (§1, §2a), the AR6 thermal-expansion samples (§2a),
`thermosteric_record_sensitivity.json` (`thermorec`), `slr_processed_data.h5`
and `component_results.h5`. It stops with the command to run when one of the
cached inputs is missing. Its inputs are built upstream, once:

| Step | Command | Writes | Needs |
|---|---|---|---|
| prep:zfetch | `python -I scripts/cmip6_zostoga.py fetch --download-dir data/raw/steric/cmip6_zostoga --out-csv data/processed/cmip6_zostoga_global.csv` | `cmip6_zostoga_global.csv`; `fetch_provenance.json`, `zelinka_ecs.json` and the download cache in `data/raw/steric/cmip6_zostoga/` | network (Pangeo CMIP6 cloud catalog, ESGF, GitHub) |
| prep:zar6 | `python -I scripts/cmip6_zostoga.py extract-ar6` | `cmip6_zostoga_ar6_tlm_ssp245.h5` | `tlmdata/twolayer_SSPs.h5` (574 MB) |

`zostoga` runs after `thermorec` and before `numbers`, so a full run always
rebuilds `cmip6_zostoga_summary.json` from the current ocean results. Built
from the distributed inputs and the production `component_results.h5`, it
reproduces the distributed summary: every value is identical.

---

## 5. Environment variables

| Variable | Read by | Effect |
|---|---|---|
| `SLR_N_SAMPLES` | `src/slr_forecast/config.py` (every step) | Ensemble size; default 10000 (the production ensemble size). |
| `PYTHON` | `run_pipeline.sh` | Interpreter for the script entries; default `python3`. |
| `NB_TIMEOUT_SCALE` | `run_pipeline.sh` | Multiplies every notebook's nbconvert timeout; default 1. |
| `NB_TIMEOUT` | `run_pipeline.sh` | Every notebook's nbconvert timeout in seconds, `-1` for no limit; overrides `NB_TIMEOUT_SCALE`. |
| `OCEAN_HOLDOUT_H5` | `scripts/ocean_early_holdout.py` | HDF5 file for the `ocean_early_holdout` group; default `data/processed/component_results.h5`. |
| `OCEAN_HOLDOUT_FIG` | `scripts/ocean_early_holdout.py` | Output figure; default `figures/component_ocean_early_holdout.png`. |
| `OCEAN_HOLDOUT_QUICK` | `scripts/ocean_early_holdout.py` | If set, 200 + 400 MCMC steps instead of 6000 + 30000 (smoke test only; results are not comparable). |

---

## 6. Notes

- Citation keys refer to the paper's `references.bib`.
- Build the Zenodo bundle from the tables above, not by archiving `data/raw/`
  whole.
