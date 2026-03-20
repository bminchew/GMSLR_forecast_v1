"""
Terrestrial water storage reader.

Datasets
--------
- GRACE/GRACE-FO JPL mascon (monthly, 2002–present)
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from slr_forecast.units import GT_TO_M_SLE, tag_units, tag_sign_convention
from slr_forecast.readers._utils import datetime_to_decimal_year


def read_grace_tws_global(filepath: str) -> pd.DataFrame:
    """Read GRACE/GRACE-FO JPL mascon global land TWS anomaly.

    Integrates gridded LWE thickness over land (excluding Greenland
    and Antarctica) and converts to meters SLE.

    Returns monthly DataFrame with columns: tws_anomaly (m SLE),
    tws_anomaly_sigma (m SLE), decimal_year.

    Note: positive tws_anomaly = increased land storage = sea level DROP.
    The sign convention tag reflects this (attrs['tws_sign'] = 'positive=storage').
    Downstream code must negate when adding to GMSL budget.

    Reference
    ---------
    Watkins et al. (2015), doi:10.1002/2014JB011547
    """
    import xarray as xr

    ds = xr.open_dataset(filepath)
    lwe = ds["lwe_thickness"]
    unc = ds["uncertainty"]
    land_mask = ds["land_mask"]
    lat = ds["lat"].values
    lon = ds["lon"].values
    time_vals = ds["time"].values

    # Cell areas (km²)
    R_earth = 6371.0
    dlat = np.abs(np.diff(lat).mean())
    dlon = np.abs(np.diff(lon).mean())
    lat_rad = np.deg2rad(lat)
    cell_area = (R_earth**2) * np.deg2rad(dlat) * np.deg2rad(dlon) * np.cos(lat_rad)
    area_grid = np.outer(cell_area, np.ones(len(lon)))

    # Mask: land, excluding Greenland and Antarctica
    mask_2d = land_mask.values.copy()
    mask_2d[lat < -60, :] = 0
    for i, la in enumerate(lat):
        if la > 60:
            for j, lo in enumerate(lon):
                lo360 = lo % 360
                if 295 <= lo360 <= 350:
                    mask_2d[i, j] = 0

    # Integrate LWE → Gt
    # 1 cm·km² = 1e-5 Gt (see derivation in slr_data_readers.py)
    lwe_vals = lwe.values
    unc_vals = unc.values
    n_time = lwe_vals.shape[0]
    tws_gt = np.full(n_time, np.nan)
    tws_gt_sigma = np.full(n_time, np.nan)
    masked_area = area_grid * mask_2d

    for t in range(n_time):
        lwe_t = lwe_vals[t]
        unc_t = unc_vals[t]
        valid = ~np.isnan(lwe_t) & (mask_2d > 0)
        if valid.sum() > 0:
            tws_gt[t] = np.nansum(lwe_t * masked_area * valid) * 1e-5
            tws_gt_sigma[t] = (
                np.sqrt(np.nansum((unc_t * masked_area * valid) ** 2)) * 1e-5
            )

    ds.close()

    time_dt = pd.to_datetime(time_vals)
    df = pd.DataFrame(
        {"tws_anomaly": tws_gt, "tws_anomaly_sigma": tws_gt_sigma},
        index=time_dt,
    )
    df.index.name = "time"
    df["decimal_year"] = [
        datetime_to_decimal_year(t.to_pydatetime()) for t in df.index
    ]

    # Convert Gt → meters SLE
    for col in ["tws_anomaly", "tws_anomaly_sigma"]:
        df[col] = df[col] * GT_TO_M_SLE

    tag_sign_convention(df, "slr")
    tag_units(df, {
        "tws_anomaly": "m",
        "tws_anomaly_sigma": "m",
        "decimal_year": "yr",
    })
    df.attrs["dataset"] = "grace_tws_global"
    df.attrs["reference"] = "Watkins et al. (2015)"
    df.attrs["doi"] = "10.1002/2014JB011547"
    df.attrs["tws_sign"] = "positive=storage (negate for SLR contribution)"
    df.attrs["excluded_regions"] = (
        "Greenland (lat>60, lon 295-350), Antarctica (lat<-60)"
    )

    return df
