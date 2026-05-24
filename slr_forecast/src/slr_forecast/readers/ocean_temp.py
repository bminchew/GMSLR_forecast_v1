"""
Ocean temperature data readers.

All readers return DataFrames with datetime index, temperature in degC.

Datasets
--------
- EN4 (Met Office, monthly gridded, 1900–present)
- Argo Roemmich-Gilson climatology (monthly gridded, 2004–present)
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from slr_forecast.units import tag_units


# =========================================================================
#  EN4 — Met Office Hadley Centre subsurface temperature
# =========================================================================

def read_en4_regional(
    en4_dir: str | Path,
    lat_bounds: tuple[float, float] = (58.0, 80.0),
    lon_bounds: tuple[float, float] = (-75.0, -5.0),
    depth_bounds: tuple[float, float] = (200.0, 500.0),
    year_range: tuple[int, int] | None = None,
    reference_period: tuple[float, float] = (1995.0, 2006.0),
) -> pd.DataFrame:
    """Read EN4 analyses and extract a regional subsurface temperature series.

    Extracts monthly depth-averaged, area-weighted potential temperature
    for the specified lat/lon/depth box from yearly EN4 zip archives.

    Default region: Greenland-peripheral waters (58–80°N, 75–5°W,
    200–500 m depth), capturing subsurface Atlantic Water in the
    Irminger Sea, Davis Strait, Baffin Bay, and NE Greenland shelf.

    Parameters
    ----------
    en4_dir : str or Path
        Directory containing ``EN.4.2.2.analyses.g10.YYYY.zip`` files.
    lat_bounds : tuple of float
        (lat_min, lat_max) in degrees north.
    lon_bounds : tuple of float
        (lon_min, lon_max) in degrees east.  Negative values are
        converted internally to EN4's 0–360° convention.
    depth_bounds : tuple of float
        (depth_min, depth_max) in metres.
    year_range : tuple of int or None
        (start_year, end_year) inclusive.  If None, reads all available
        zip files.
    reference_period : tuple of float
        (start, end) decimal years for computing anomalies.
        Temperature is returned as anomaly relative to the mean
        over this period.  Default (1995, 2006) spans the project
        baseline window with exclusive upper bound.

    Returns
    -------
    pd.DataFrame
        Monthly DataFrame with columns:
        - temperature : depth/area-weighted mean anomaly (degC)
        - temperature_sigma : spatial standard deviation (degC)
        - decimal_year : fractional year
        - n_valid : number of valid ocean cells used

    Reference
    ---------
    Good et al. (2013), J. Geophys. Res., doi:10.1002/2013JC009067
    """
    import netCDF4 as nc
    import zipfile

    en4_dir = Path(en4_dir)

    # Convert lon bounds to EN4 convention (0–360)
    lon_min_en4 = lon_bounds[0] % 360
    lon_max_en4 = lon_bounds[1] % 360

    # Find available zip files
    zips = sorted(en4_dir.glob("EN.4.2.2.analyses.g10.*.zip"))
    if not zips:
        raise FileNotFoundError(f"No EN4 zip files found in {en4_dir}")

    if year_range is not None:
        zips = [z for z in zips
                if year_range[0] <= int(z.stem.split(".")[-1]) <= year_range[1]]

    records = []
    grid_info = {}

    try:
        from tqdm.auto import tqdm
        zip_iter = tqdm(zips, desc="EN4", unit="yr")
    except ImportError:
        zip_iter = zips

    for zpath in zip_iter:
        with zipfile.ZipFile(zpath) as zf:
            nc_names = sorted(n for n in zf.namelist() if n.endswith(".nc"))
            for nc_name in nc_names:
                # Parse year/month from filename
                parts = nc_name.replace(".nc", "").split(".")
                yyyymm = parts[-1]  # e.g. "202001"
                year = int(yyyymm[:4])
                month = int(yyyymm[4:6])

                with zf.open(nc_name) as f:
                    ds = nc.Dataset("mem", memory=f.read())

                    # Build masks on first file
                    if not grid_info:
                        lat = ds.variables["lat"][:]
                        lon = ds.variables["lon"][:]
                        depth = ds.variables["depth"][:]

                        lat_mask = ((lat >= lat_bounds[0])
                                    & (lat <= lat_bounds[1]))
                        if lon_min_en4 <= lon_max_en4:
                            lon_mask = ((lon >= lon_min_en4)
                                        & (lon <= lon_max_en4))
                        else:
                            # Wraps around 0°
                            lon_mask = ((lon >= lon_min_en4)
                                        | (lon <= lon_max_en4))
                        depth_mask = ((depth >= depth_bounds[0])
                                      & (depth <= depth_bounds[1]))

                        # Cosine latitude weighting
                        lat_sub = lat[lat_mask]
                        cos_wt = np.cos(np.radians(lat_sub))

                        grid_info = {
                            "lat_mask": lat_mask,
                            "lon_mask": lon_mask,
                            "depth_mask": depth_mask,
                            "cos_wt": cos_wt,
                            "n_depth": int(depth_mask.sum()),
                        }

                    # Extract region: (1, depth, lat, lon) → (depth, lat, lon)
                    T_full = ds.variables["temperature"][0, :, :, :]
                    ds.close()

                T_region = T_full[grid_info["depth_mask"], :, :][
                    :, grid_info["lat_mask"], :
                ][:, :, grid_info["lon_mask"]]

                # Convert to Celsius (EN4 stores as Kelvin)
                if hasattr(T_region, "mask"):
                    valid = ~T_region.mask
                    T_c = np.where(valid, T_region.data - 273.15, np.nan)
                else:
                    T_c = T_region - 273.15
                    valid = ~np.isnan(T_c)

                # Depth-average first, then area-weighted spatial average
                T_depth_avg = np.nanmean(T_c, axis=0)  # (lat, lon)

                cos_2d = grid_info["cos_wt"][:, np.newaxis]
                valid_2d = ~np.isnan(T_depth_avg)
                wt = np.where(valid_2d, cos_2d, 0.0)
                wt_sum = wt.sum()

                if wt_sum > 0:
                    T_mean = np.nansum(T_depth_avg * wt) / wt_sum
                    T_std = np.sqrt(
                        np.nansum(wt * (T_depth_avg - T_mean) ** 2) / wt_sum
                    )
                    n_valid = int(valid_2d.sum())
                else:
                    T_mean = np.nan
                    T_std = np.nan
                    n_valid = 0

                records.append({
                    "year": year,
                    "month": month,
                    "temperature": T_mean,
                    "temperature_sigma": T_std,
                    "decimal_year": year + (month - 0.5) / 12.0,
                    "n_valid": n_valid,
                })

    df = pd.DataFrame(records)
    df["time"] = pd.to_datetime(
        df["year"].astype(str) + "-" + df["month"].astype(str).str.zfill(2) + "-01"
    )
    df = df.set_index("time").sort_index()
    df = df[["temperature", "temperature_sigma", "decimal_year", "n_valid"]]
    df.index.name = "time"

    # Rebaseline to reference period
    ref_mask = ((df["decimal_year"] >= reference_period[0])
                & (df["decimal_year"] < reference_period[1]))
    if ref_mask.sum() > 0:
        ref_mean = df.loc[ref_mask, "temperature"].mean()
        df["temperature"] = df["temperature"] - ref_mean
    else:
        ref_mean = np.nan

    tag_units(df, {
        "temperature": "degC",
        "temperature_sigma": "degC",
        "decimal_year": "yr",
    })
    df.attrs["dataset"] = "en4_regional"
    df.attrs["reference"] = "Good et al. (2013)"
    df.attrs["doi"] = "10.1002/2013JC009067"
    df.attrs["en4_version"] = "EN.4.2.2"
    df.attrs["bias_correction"] = "g10"
    df.attrs["region_lat"] = lat_bounds
    df.attrs["region_lon"] = lon_bounds
    df.attrs["region_depth"] = depth_bounds
    df.attrs["reference_period"] = reference_period
    df.attrs["reference_mean_degC"] = float(ref_mean)

    return df


# =========================================================================
#  Argo Roemmich-Gilson — gridded subsurface temperature anomalies
# =========================================================================

def read_argo_rg_regional(
    argo_dir: str | Path,
    lat_bounds: tuple[float, float] = (58.0, 80.0),
    lon_bounds: tuple[float, float] = (-75.0, -5.0),
    depth_bounds: tuple[float, float] = (200.0, 500.0),
    reference_period: tuple[float, float] = (1995.0, 2006.0),
    en4_baseline_offset: float | None = None,
) -> pd.DataFrame:
    """Read Argo Roemmich-Gilson gridded data for a regional subsurface T series.

    Reads monthly gzipped NetCDF files (``RG_ArgoClim_YYYYMM_2019.nc.gz``)
    and the climatology file to produce absolute temperature, then
    rebaselines to anomalies relative to *reference_period*.

    Default region matches :func:`read_en4_regional`: Greenland-peripheral
    waters (58–80°N, 75–5°W, 200–500 dbar).

    Parameters
    ----------
    argo_dir : str or Path
        Directory containing ``RG_ArgoClim_YYYYMM_2019.nc.gz`` files
        and ``RG_ArgoClim_Temperature_2019.nc.gz`` (climatology).
    lat_bounds : tuple of float
        (lat_min, lat_max) in degrees north.
    lon_bounds : tuple of float
        (lon_min, lon_max) in degrees east.  Negative values are
        converted to the Argo 20.5–379.5° convention.
    depth_bounds : tuple of float
        (depth_min, depth_max) in dbar (≈ metres).
    reference_period : tuple of float
        (start, end) decimal years for computing anomalies.
        Default (1995, 2006) matches the project baseline (1995–2005).
        If Argo data do not cover this period (typical: Argo starts
        2019), ``en4_baseline_offset`` must be provided.
    en4_baseline_offset : float or None
        Offset to convert from the Argo 2004–2018 climatological
        mean to the target reference_period baseline.  Computed as::

            offset = EN4_mean(reference_period) - EN4_mean(2004, 2019)

        When provided, anomalies are: T - clim_mean - offset.
        When None, the reader attempts to compute the reference
        mean from available Argo data (only works if Argo covers
        the reference_period).

    Returns
    -------
    pd.DataFrame
        Monthly DataFrame with columns:
        - temperature : depth/area-weighted mean anomaly (degC)
        - temperature_sigma : spatial standard deviation (degC)
        - decimal_year : fractional year
        - n_valid : number of valid ocean cells used

    Reference
    ---------
    Roemmich & Gilson (2009), Prog. Oceanogr., doi:10.1016/j.pocean.2009.03.004
    """
    import gzip
    import netCDF4 as nc4

    argo_dir = Path(argo_dir)

    # ── Load climatological mean ──
    clim_path = argo_dir / "RG_ArgoClim_Temperature_2019.nc.gz"
    if not clim_path.exists():
        raise FileNotFoundError(
            f"Climatology file not found: {clim_path}")

    with gzip.open(clim_path) as f:
        ds = nc4.Dataset("mem", memory=f.read())
        lat = np.array(ds.variables["LATITUDE"][:])
        lon = np.array(ds.variables["LONGITUDE"][:])
        pres = np.array(ds.variables["PRESSURE"][:])
        T_clim_raw = ds.variables["ARGO_TEMPERATURE_MEAN"][:]
        T_clim = np.ma.filled(T_clim_raw, fill_value=np.nan)  # (pres, lat, lon)
        bathy_mask = np.array(ds.variables["BATHYMETRY_MASK"][:])     # (pres, lat, lon)
        ds.close()

    # ── Build spatial masks ──
    lat_mask = (lat >= lat_bounds[0]) & (lat <= lat_bounds[1])

    # Argo lon convention: 20.5–379.5
    lon_min_argo = lon_bounds[0] % 360
    lon_max_argo = lon_bounds[1] % 360
    if lon_min_argo <= lon_max_argo:
        lon_mask = (lon >= lon_min_argo) & (lon <= lon_max_argo)
    else:
        lon_mask = (lon >= lon_min_argo) | (lon <= lon_max_argo)

    pres_mask = (pres >= depth_bounds[0]) & (pres <= depth_bounds[1])

    # Cosine latitude weighting
    cos_wt = np.cos(np.radians(lat[lat_mask]))

    # Extract climatological mean for the region
    T_clim_region = T_clim[pres_mask, :, :][:, lat_mask, :][:, :, lon_mask]

    # ── Read monthly anomaly files ──
    monthly_files = sorted(argo_dir.glob("RG_ArgoClim_2*_2019.nc.gz"))
    if not monthly_files:
        raise FileNotFoundError(
            f"No monthly Argo files found in {argo_dir}")

    records = []

    try:
        from tqdm.auto import tqdm
        file_iter = tqdm(monthly_files, desc="Argo", unit="mo")
    except ImportError:
        file_iter = monthly_files

    for fpath in file_iter:
        # Parse YYYYMM from filename
        stem = fpath.name.split("_")[2]  # e.g. "202001"
        year = int(stem[:4])
        month = int(stem[4:6])

        with gzip.open(fpath) as f:
            ds = nc4.Dataset("mem", memory=f.read())
            T_anom_raw = ds.variables["ARGO_TEMPERATURE_ANOMALY"][0, :, :, :]
            T_anom = np.ma.filled(T_anom_raw, fill_value=np.nan)
            ds.close()

        # Extract region
        T_anom_region = T_anom[pres_mask, :, :][:, lat_mask, :][:, :, lon_mask]

        # Absolute T = climatology + anomaly (both NaN where invalid)
        T_abs = T_clim_region + T_anom_region

        # Depth-average first, then area-weighted spatial mean
        T_depth_avg = np.nanmean(T_abs, axis=0)  # (lat, lon)

        cos_2d = cos_wt[:, np.newaxis]
        valid_2d = ~np.isnan(T_depth_avg)
        wt = np.where(valid_2d, cos_2d, 0.0)
        wt_sum = wt.sum()

        if wt_sum > 0:
            T_mean = float(np.nansum(T_depth_avg * wt) / wt_sum)
            T_std = float(np.sqrt(
                np.nansum(wt * (T_depth_avg - T_mean) ** 2) / wt_sum))
            n_valid = int(valid_2d.sum())
        else:
            T_mean = np.nan
            T_std = np.nan
            n_valid = 0

        records.append({
            "year": year,
            "month": month,
            "temperature": T_mean,
            "temperature_sigma": T_std,
            "decimal_year": year + (month - 0.5) / 12.0,
            "n_valid": n_valid,
        })

    df = pd.DataFrame(records)
    df["time"] = pd.to_datetime(
        df["year"].astype(str) + "-"
        + df["month"].astype(str).str.zfill(2) + "-01")
    df = df.set_index("time").sort_index()
    df = df[["temperature", "temperature_sigma", "decimal_year", "n_valid"]]
    df.index.name = "time"

    # Rebaseline to reference period.
    # Argo climatological regional mean (area-weighted mean over climatology period).
    T_clim_depth_avg = np.nanmean(T_clim_region, axis=0)
    cos_2d_clim = cos_wt[:, np.newaxis]
    valid_clim = ~np.isnan(T_clim_depth_avg)
    wt_clim = np.where(valid_clim, cos_2d_clim, 0.0)
    clim_regional_mean = float(
        np.nansum(T_clim_depth_avg * wt_clim) / wt_clim.sum())

    ref_mask = ((df["decimal_year"] >= reference_period[0])
                & (df["decimal_year"] < reference_period[1]))
    if ref_mask.sum() > 0:
        # Argo covers the reference period — compute directly
        ref_mean = df.loc[ref_mask, "temperature"].mean()
    elif en4_baseline_offset is not None:
        # Use EN4 to bridge: ref_mean = clim_mean + offset
        # where offset = EN4_mean(ref_period) - EN4_mean(climatology_period)
        ref_mean = clim_regional_mean + en4_baseline_offset
    else:
        raise ValueError(
            f"Argo data do not cover reference_period "
            f"{reference_period}, and en4_baseline_offset was not "
            f"provided.  Compute it from EN4 as: "
            f"EN4_mean(ref_period) - EN4_mean(climatology_period).")

    df["temperature"] = df["temperature"] - ref_mean

    tag_units(df, {
        "temperature": "degC",
        "temperature_sigma": "degC",
        "decimal_year": "yr",
    })
    df.attrs["dataset"] = "argo_rg_regional"
    df.attrs["reference"] = "Roemmich & Gilson (2009)"
    df.attrs["doi"] = "10.1016/j.pocean.2009.03.004"
    df.attrs["climatology_period"] = "2004-2018"
    df.attrs["region_lat"] = lat_bounds
    df.attrs["region_lon"] = lon_bounds
    df.attrs["region_depth"] = depth_bounds
    df.attrs["reference_period"] = reference_period
    df.attrs["reference_mean_degC"] = float(ref_mean)

    return df
