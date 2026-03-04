"""
Readers for IPCC AR6 FACTS full-sample ensemble NetCDF files.

Reads the raw 20,000-member Monte Carlo ensembles from the FACTS
framework used in IPCC AR6 Chapter 9 sea-level projections.

Two data sources are supported:

  1. Component files  (full_sample_components/):
     Individual SLR contributions from specific models and sub-components
     (glaciers, ice sheets by region). Each file contains 20,000 samples
     for one model × one scenario × one sub-component.

  2. Workflow totals  (full_sample_workflows/):
     Total GMSL from pre-defined model combinations (wf_1e through wf_4).
     Each ``total-workflow.nc`` is the sum of all components in that
     workflow, with sample correlations preserved across components.

File naming convention (components)::

    {component_type}-{version}-{model}-{scenario}[_{sub_component}]_globalsl.nc

Examples::

    glaciers-ipccar6-gmipemuglaciers-ssp245_globalsl.nc
    icesheets-ipccar6-ismipemuicesheet-ssp245_AIS_globalsl.nc

Workflow directories (wf_1e … wf_4) each contain per-scenario
subdirectories with a ``total-workflow.nc`` plus the individual
component files that were summed.

NetCDF structure (all files)::

    Dimensions:   (samples=20000, years=9..29, locations=1)
    Variables:    sea_level_change  int16  (samples, years, locations)  [mm]
                  lat               float32 (locations)
                  lon               float32 (locations)
    Attributes:   baseyear=2005, scenario, source, description, history

Reference
---------
Fox-Kemper, B., et al. (2021). https://doi.org/10.1017/9781009157896.011
Garner, G. G., et al. (2023). https://doi.org/10.5194/gmd-16-7461-2023
Data: https://doi.org/10.5281/zenodo.5914709
"""

import os
import numpy as np
import pandas as pd


# ====================================================================
# Workflow metadata — describes the model combination in each workflow
# ====================================================================
WORKFLOW_INFO = {
    'wf_1e': {
        'label': 'ISMIP6-emu',
        'confidence': 'medium',
        'ice_ais': 'ISMIP6 emulated',
        'ice_gis': 'ISMIP6 emulated',
        'glaciers': 'GMIP emulated',
    },
    'wf_1f': {
        'label': 'FittedISMIP + AR5',
        'confidence': 'medium',
        'ice_ais': 'AR5',
        'ice_gis': 'FittedISMIP',
        'glaciers': 'AR5 GMIP2',
    },
    'wf_2e': {
        'label': 'LARMIP + ISMIP6-emu',
        'confidence': 'medium',
        'ice_ais': 'LARMIP2 (total)',
        'ice_gis': 'ISMIP6 emulated',
        'glaciers': 'GMIP emulated',
    },
    'wf_2f': {
        'label': 'LARMIP + FittedISMIP',
        'confidence': 'medium',
        'ice_ais': 'LARMIP2 (total)',
        'ice_gis': 'FittedISMIP',
        'glaciers': 'AR5 GMIP2',
    },
    'wf_3e': {
        'label': 'DP16 + ISMIP6-emu',
        'confidence': 'medium',
        'ice_ais': 'DeConto & Pollard (2016)',
        'ice_gis': 'ISMIP6 emulated',
        'glaciers': 'GMIP emulated',
    },
    'wf_3f': {
        'label': 'DP16 + FittedISMIP',
        'confidence': 'medium',
        'ice_ais': 'DeConto & Pollard (2016)',
        'ice_gis': 'FittedISMIP',
        'glaciers': 'AR5 GMIP2',
    },
    'wf_4': {
        'label': 'Bamber SEJ',
        'confidence': 'low',
        'ice_ais': 'Bamber et al. (2019) SEJ',
        'ice_gis': 'Bamber et al. (2019) SEJ',
        'glaciers': 'AR5 GMIP2',
    },
}


# ====================================================================
# Internal helpers
# ====================================================================

def _parse_component_filename(filename: str) -> dict:
    """Parse a full_sample_components filename into metadata.

    Parameters
    ----------
    filename : str
        e.g. ``'icesheets-ipccar6-ismipemuicesheet-ssp245_AIS_globalsl.nc'``

    Returns
    -------
    dict
        Keys: component_type, version, model, scenario, sub_component.
    """
    stem = filename.replace('_globalsl.nc', '')

    sub_component = None
    for sub in ['WAIS', 'EAIS', 'AIS', 'GIS', 'PEN', 'SMB', 'TOT']:
        if stem.endswith(f'_{sub}'):
            sub_component = sub
            stem = stem[:-len(f'_{sub}')]
            break

    parts = stem.split('-', 3)
    if len(parts) != 4:
        raise ValueError(f"Cannot parse component filename: {filename}")

    return {
        'component_type': parts[0],
        'version': parts[1],
        'model': parts[2],
        'scenario': parts[3],
        'sub_component': sub_component,
    }


def _read_netcdf_samples(filepath: str, convert_to_meters: bool) -> pd.DataFrame:
    """Read a single FACTS full-sample NetCDF and return a DataFrame.

    Parameters
    ----------
    filepath : str
        Path to ``.nc`` file with ``sea_level_change`` variable.
    convert_to_meters : bool
        If True, scale int16 mm → float64 meters.

    Returns
    -------
    pd.DataFrame
        Shape ``(n_samples, n_years)``.  Columns are integer years
        (e.g. 2020, 2030, …).  Index is sample number ``0 … n-1``.
        The ``.attrs`` dict is populated by the caller.
    """
    try:
        import xarray as xr
    except ImportError:
        raise ImportError(
            "xarray is required to read FACTS NetCDF files.\n"
            "Install with: pip install xarray netCDF4"
        )

    ds = xr.open_dataset(filepath)

    slc = ds['sea_level_change'].values[:, :, 0]   # (n_samples, n_years)
    years = ds['years'].values.astype(int)
    samples = ds['samples'].values

    lat = float(ds['lat'].values[0]) if 'lat' in ds else np.nan
    lon = float(ds['lon'].values[0]) if 'lon' in ds else np.nan

    nc_attrs = dict(ds.attrs)
    ds.close()

    scale = 0.001 if convert_to_meters else 1.0
    unit = 'm' if convert_to_meters else 'mm'

    df = pd.DataFrame(
        slc.astype(np.float64) * scale,
        index=pd.RangeIndex(len(samples), name='sample'),
        columns=years,
    )
    df.columns.name = 'year'

    # Base attrs populated here; callers extend with domain-specific keys
    df.attrs = {
        'native_units': 'mm',
        'current_units': unit,
        'conversion_factor': scale,
        'baseyear': int(nc_attrs.get('baseyear', 2005)),
        'scenario': nc_attrs.get('scenario', ''),
        'description': nc_attrs.get('description', ''),
        'source': nc_attrs.get('source', ''),
        'history': nc_attrs.get('history', ''),
        'comment': nc_attrs.get('comment', ''),
        'lat': lat,
        'lon': lon,
        'n_samples': int(slc.shape[0]),
        'n_years': int(slc.shape[1]),
        'years': years.tolist(),
    }

    return df


# ====================================================================
# Public API — component files
# ====================================================================

def read_ar6_full_sample_component(
    filepath: str,
    convert_to_meters: bool = True,
) -> pd.DataFrame:
    """Read a single FACTS full-sample component NetCDF file.

    Parameters
    ----------
    filepath : str
        Full path to a ``*_globalsl.nc`` file in ``full_sample_components/``.
    convert_to_meters : bool, default True
        If True, convert from native int16 mm to float64 meters.

    Returns
    -------
    pd.DataFrame
        Shape ``(n_samples, n_years)``.  Columns are integer years
        (2020, 2030, …, up to 2300).  Index is sample number.

        ``.attrs`` contains:

        ============== ================================================
        Key            Description
        ============== ================================================
        dataset        ``'ipcc_ar6_full_sample_component'``
        filename       Original filename
        component_type ``'glaciers'`` or ``'icesheets'``
        version        Model suite (e.g. ``'ipccar6'``, ``'ar5'``)
        model          Specific model (e.g. ``'ismipemuicesheet'``)
        scenario       SSP or temperature-limit scenario
        sub_component  ``'AIS'``, ``'GIS'``, …, or ``None`` (glaciers)
        native_units   ``'mm'``
        current_units  ``'m'`` or ``'mm'``
        conversion_factor  0.001 (m) or 1.0 (mm)
        baseyear       2005
        description    NetCDF description attribute
        source         NetCDF source attribute
        history        NetCDF creation timestamp
        comment        NetCDF comment (e.g. included GCMs)
        lat, lon       Global-mean location coordinates
        n_samples      Number of Monte Carlo samples (20000)
        n_years        Number of decadal time steps
        years          List of year values
        reference      Citation
        doi            Chapter DOI
        data_doi       Data archive DOI
        quantity       ``'sea_level_change'``
        ============== ================================================
    """
    filename = os.path.basename(filepath)
    meta = _parse_component_filename(filename)
    df = _read_netcdf_samples(filepath, convert_to_meters)

    df.attrs.update({
        'dataset': 'ipcc_ar6_full_sample_component',
        'filename': filename,
        'component_type': meta['component_type'],
        'version': meta['version'],
        'model': meta['model'],
        'sub_component': meta['sub_component'],
        'reference': 'Fox-Kemper et al. (2021)',
        'doi': '10.1017/9781009157896.011',
        'data_doi': '10.5281/zenodo.5914709',
        'quantity': 'sea_level_change',
    })

    return df


def list_ar6_full_sample_components(data_dir: str) -> pd.DataFrame:
    """Catalog all component files in ``full_sample_components/``.

    Parameters
    ----------
    data_dir : str
        Path to the ``full_sample_components/`` directory.

    Returns
    -------
    pd.DataFrame
        One row per file.  Columns: ``filename``, ``filepath``,
        ``component_type``, ``version``, ``model``, ``scenario``,
        ``sub_component``, ``model_label``.

        ``.attrs`` contains ``data_dir`` and ``n_files``.
    """
    records = []
    for f in sorted(os.listdir(data_dir)):
        if not f.endswith('_globalsl.nc'):
            continue
        try:
            meta = _parse_component_filename(f)
        except ValueError:
            continue
        meta['filename'] = f
        meta['filepath'] = os.path.join(data_dir, f)
        meta['model_label'] = f"{meta['version']}-{meta['model']}"
        records.append(meta)

    df = pd.DataFrame(records)
    df.attrs = {
        'dataset': 'ipcc_ar6_full_sample_components_catalog',
        'data_dir': data_dir,
        'n_files': len(records),
    }
    return df


def read_ar6_full_sample_components(
    data_dir: str,
    scenario: str | list | None = None,
    component_type: str | list | None = None,
    model: str | list | None = None,
    sub_component: str | list | None = None,
    convert_to_meters: bool = True,
) -> dict:
    """Read all (or filtered) FACTS full-sample component files.

    Parameters
    ----------
    data_dir : str
        Path to the ``full_sample_components/`` directory.
    scenario : str or list, optional
        Filter by scenario(s), e.g. ``'ssp245'``.
    component_type : str or list, optional
        Filter by component(s), e.g. ``'icesheets'``.
    model : str or list, optional
        Filter by model(s), e.g. ``'ismipemuicesheet'``.
    sub_component : str or list, optional
        Filter by sub-component(s), e.g. ``'AIS'`` or ``['WAIS', 'EAIS']``.
    convert_to_meters : bool, default True
        If True, convert from native mm to meters.

    Returns
    -------
    dict
        Keyed by filename stem (without ``_globalsl.nc``).
        Each value is a ``pd.DataFrame`` from
        :func:`read_ar6_full_sample_component`.

    Notes
    -----
    This directory contains only glacier and ice-sheet components.
    For total GMSL (including ocean dynamics and land water storage),
    use :func:`read_ar6_workflow_totals` on the sibling
    ``full_sample_workflows/`` directory.
    """
    catalog = list_ar6_full_sample_components(data_dir)

    for col, val in [('scenario', scenario),
                     ('component_type', component_type),
                     ('model', model),
                     ('sub_component', sub_component)]:
        if val is not None:
            if isinstance(val, str):
                val = [val]
            catalog = catalog[catalog[col].isin(val)]

    result = {}
    for _, row in catalog.iterrows():
        key = row['filename'].replace('_globalsl.nc', '')
        result[key] = read_ar6_full_sample_component(
            row['filepath'], convert_to_meters=convert_to_meters,
        )

    return result


# ====================================================================
# Public API — workflow totals
# ====================================================================

def read_ar6_workflow_total(
    filepath: str,
    convert_to_meters: bool = True,
) -> pd.DataFrame:
    """Read a single ``total-workflow.nc`` file.

    Parameters
    ----------
    filepath : str
        Full path to a ``total-workflow.nc`` file inside a workflow
        scenario directory.
    convert_to_meters : bool, default True
        If True, convert from native int16 mm to float64 meters.

    Returns
    -------
    pd.DataFrame
        Shape ``(n_samples, n_years)``.  Same layout as
        :func:`read_ar6_full_sample_component`.

        ``.attrs`` includes ``dataset='ipcc_ar6_workflow_total'``,
        ``workflow``, ``confidence``, plus all base attributes.
    """
    df = _read_netcdf_samples(filepath, convert_to_meters)

    # Infer workflow name from path:
    #   .../full_sample_workflows/wf_1e/ssp245/total-workflow.nc
    parts = filepath.replace('\\', '/').split('/')
    workflow = None
    for i, p in enumerate(parts):
        if p.startswith('wf_'):
            workflow = p
            break

    wf_info = WORKFLOW_INFO.get(workflow, {})

    df.attrs.update({
        'dataset': 'ipcc_ar6_workflow_total',
        'workflow': workflow,
        'workflow_label': wf_info.get('label', workflow or 'unknown'),
        'confidence': wf_info.get('confidence', 'unknown'),
        'ice_ais_model': wf_info.get('ice_ais', ''),
        'ice_gis_model': wf_info.get('ice_gis', ''),
        'glaciers_model': wf_info.get('glaciers', ''),
        'reference': 'Fox-Kemper et al. (2021)',
        'doi': '10.1017/9781009157896.011',
        'data_doi': '10.5281/zenodo.5914709',
        'quantity': 'total_sea_level_change',
    })

    return df


def read_ar6_workflow_totals(
    data_dir: str,
    scenario: str | list | None = None,
    convert_to_meters: bool = True,
) -> dict:
    """Read ``total-workflow.nc`` from all FACTS workflow directories.

    Parameters
    ----------
    data_dir : str
        Path to the ``full_sample_workflows/`` directory (parent of
        ``wf_1e/``, ``wf_1f/``, …, ``wf_4/``).
    scenario : str or list, optional
        Filter by scenario(s), e.g. ``'ssp245'`` or
        ``['ssp245', 'ssp585']``.
    convert_to_meters : bool, default True
        If True, convert from native mm to meters.

    Returns
    -------
    dict
        Keyed by ``'{workflow}/{scenario}'`` (e.g. ``'wf_1e/ssp245'``).
        Each value is a ``pd.DataFrame`` from
        :func:`read_ar6_workflow_total`.

    Notes
    -----
    The seven AR6 workflows differ in their ice-sheet and glacier
    model choices.  Workflows ``wf_1e`` through ``wf_3f`` are
    medium-confidence; ``wf_4`` (Bamber SEJ) is low-confidence.
    See :data:`WORKFLOW_INFO` for model descriptions.

    All workflows share the same ocean dynamics (two-layer model)
    and land water storage (Kopp 2014) components.
    """
    if isinstance(scenario, str):
        scenario = [scenario]

    result = {}
    for wf_name in sorted(os.listdir(data_dir)):
        wf_path = os.path.join(data_dir, wf_name)
        if not os.path.isdir(wf_path) or not wf_name.startswith('wf_'):
            continue

        for scen_name in sorted(os.listdir(wf_path)):
            scen_path = os.path.join(wf_path, scen_name)
            if not os.path.isdir(scen_path):
                continue
            if scenario is not None and scen_name not in scenario:
                continue

            total_file = os.path.join(scen_path, 'total-workflow.nc')
            if not os.path.exists(total_file):
                continue

            key = f'{wf_name}/{scen_name}'
            result[key] = read_ar6_workflow_total(
                total_file, convert_to_meters=convert_to_meters,
            )

    return result


def read_ar6_workflow_components(
    data_dir: str,
    workflow: str,
    scenario: str,
    convert_to_meters: bool = True,
) -> dict:
    """Read all component files within a single workflow/scenario.

    Parameters
    ----------
    data_dir : str
        Path to the ``full_sample_workflows/`` directory.
    workflow : str
        Workflow name, e.g. ``'wf_1e'``.
    scenario : str
        Scenario name, e.g. ``'ssp245'``.
    convert_to_meters : bool, default True
        If True, convert from native mm to meters.

    Returns
    -------
    dict
        Keyed by filename stem.  Each value is a ``pd.DataFrame``.
        The ``'total-workflow'`` key holds the summed total.
    """
    scen_path = os.path.join(data_dir, workflow, scenario)
    if not os.path.isdir(scen_path):
        raise FileNotFoundError(f"Directory not found: {scen_path}")

    result = {}
    for f in sorted(os.listdir(scen_path)):
        if not f.endswith('.nc'):
            continue

        fp = os.path.join(scen_path, f)

        if f == 'total-workflow.nc':
            result['total-workflow'] = read_ar6_workflow_total(
                fp, convert_to_meters=convert_to_meters,
            )
        else:
            key = f.replace('_globalsl.nc', '').replace('.nc', '')
            result[key] = read_ar6_full_sample_component(
                fp, convert_to_meters=convert_to_meters,
            )

    return result
