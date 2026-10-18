"""
Component decomposition HDF5 I/O.

Save and load fitted parameters, projection ensembles, and observation data
from each per-component notebook so that downstream aggregation notebooks
can inherit results without re-running upstream fits.

File layout (single HDF5 file, one group per component)::

    component_results.h5
    ├── proj_years                (201,) — shared time axis
    ├── ocean_posteriors/         two-layer posterior draws (top level, written
    │                             by component_ocean before save_ocean_hybrid)
    ├── ocean/                    two-layer thermosteric model (save_ocean_hybrid)
    │   ├── observations/         full-depth calibration data
    │   ├── fit_diagnostic/       written by component_ocean after the save
    │   └── projections/{ssp}/    MC samples + percentiles
    ├── glacier/
    │   ...
    ├── greenland/
    │   ├── posteriors/discharge/ discharge delay-model posteriors
    │   ├── smb_sensitivity/      SMB model attributes (MAR emulator + drift state)
    │   ├── ocean_transfer/       surface→ocean T regression
    │   ├── observations/smb/     Mouginot SMB
    │   ├── observations/discharge/
    │   └── projections/{ssp}/    total, smb, discharge samples
    ├── eais/                     SMB (C_T) model, IMBIE-3 calibration
    ├── apeninsula/               linear-in-T level model, IMBIE-3 annual increments
    └── wais/
        ├── a4_scenarios/         scenario mixture params
        ...

Other top-level groups (blended*, wais_slow90, voi, hindcast_20c, ...) are
written by component_forecast, component_summation and later scripts.
The ``model_type`` strings of some groups predate the current models (see
the save_* docstrings).

All projection arrays are in meters relative to BASELINE_YEAR.
"""

import json
import os
import warnings
from datetime import datetime, timezone
from pathlib import Path

import h5py
import numpy as np

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
try:
    from slr_forecast.config import BASELINE_YEAR, N_SAMPLES, PROCESSED_DATA_DIR
except ImportError:
    BASELINE_YEAR = 2000.0
    N_SAMPLES = 2000
    PROCESSED_DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "processed"

DEFAULT_H5_PATH = PROCESSED_DATA_DIR / "component_results.h5"

PROJ_YEARS = np.arange(1950, 2151, dtype=float)
PROJ_SSPS = ["SSP1-2.6", "SSP2-4.5", "SSP3-7.0", "SSP5-8.5", "SSP1-1.9"]

# Standard percentile keys stored per SSP
_PCTILE_KEYS = ("median", "p5", "p17", "p83", "p95")


# =========================================================================
# Low-level helpers
# =========================================================================

def _require_file(path):
    """Open (or create) the HDF5 file and return the path.

    The root ``baseline_year`` attribute is refreshed on every save, so it
    describes the current pipeline rather than the run that created the
    file (MD-12). There is no root sample count: member counts differ by
    group (N_SAMPLES for most components, more for WAIS and the slow-WAIS
    mixture), so each group's count is the first dimension of its
    ``samples`` arrays.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        with h5py.File(path, "w") as f:
            f.attrs["created"] = datetime.now(timezone.utc).isoformat()
            f.create_dataset("proj_years", data=PROJ_YEARS)
    with h5py.File(path, "a") as f:
        f.attrs["baseline_year"] = BASELINE_YEAR
        if "n_samples" in f.attrs:
            del f.attrs["n_samples"]
    return path


def _write_scalar_attrs(grp, meta):
    """Write string and finite scalar metadata as group attributes.

    Anything else (None, NaN, lists, arrays) cannot be stored this way and
    is skipped with a warning naming the key, rather than dropped silently.
    """
    skipped = []
    for k, v in meta.items():
        if isinstance(v, str):
            grp.attrs[k] = v
        elif np.isscalar(v) and np.isfinite(v):
            grp.attrs[k] = float(v)
        else:
            skipped.append(k)
    if skipped:
        warnings.warn(f"{grp.name}: metadata not saved (not a string or a "
                      f"finite scalar): {skipped}")


def _save_projection_ssp(grp, ssp, proj_dict):
    """Write one SSP's projection data into *grp/{ssp}/*.

    Parameters
    ----------
    grp : h5py.Group
        The ``projections`` group for this component.
    ssp : str
    proj_dict : dict
        Must contain at least the percentile keys.  If ``'samples'``
        is present it is stored as well (chunked + compressed).
    """
    if ssp in grp:
        del grp[ssp]
    sg = grp.create_group(ssp)
    for key in _PCTILE_KEYS:
        if key in proj_dict:
            sg.create_dataset(key, data=np.asarray(proj_dict[key]))
    if "samples" in proj_dict and proj_dict["samples"] is not None:
        arr = np.asarray(proj_dict["samples"])
        sg.create_dataset(
            "samples", data=arr,
            chunks=(min(arr.shape[0], 500), arr.shape[1]),
            compression="gzip", compression_opts=4,
        )
    # Extra keys (e.g. smb_median, dyn_median, rate_median)
    for key in proj_dict:
        if key not in _PCTILE_KEYS and key != "samples" and key != "times":
            val = np.asarray(proj_dict[key])
            if val.ndim <= 2:
                sg.create_dataset(key, data=val)
            else:
                warnings.warn(f"{sg.name}: '{key}' not saved "
                              f"(ndim {val.ndim} > 2)")


def _load_projection_ssp(sg):
    """Read one SSP sub-group back into a dict."""
    out = {}
    for key in sg:
        ds = sg[key]
        if ds.shape == ():
            out[key] = ds[()]  # scalar dataset
        else:
            out[key] = ds[:]
    return out


# =========================================================================
# Generic save / load for observations
# =========================================================================

def _save_observations(comp_grp, years, H_obs, sigma, subgroup=None):
    """Save observation arrays under *comp_grp/observations[/subgroup]*."""
    path = "observations" if subgroup is None else f"observations/{subgroup}"
    if path in comp_grp:
        del comp_grp[path]
    og = comp_grp.create_group(path)
    og.create_dataset("years", data=np.asarray(years))
    og.create_dataset("H_obs", data=np.asarray(H_obs))
    og.create_dataset("sigma", data=np.asarray(sigma))


def _load_observations(comp_grp, subgroup=None):
    """Load observation arrays."""
    path = "observations" if subgroup is None else f"observations/{subgroup}"
    og = comp_grp[path]
    return {
        "years": og["years"][:],
        "H_obs": og["H_obs"][:],
        "sigma": og["sigma"][:],
    }


# =========================================================================
# Standard DOLS component (ocean, glacier, eais, apeninsula)
# =========================================================================

def save_dols_component(
    component_name,
    result,
    obs_years, obs_H, obs_sigma,
    proj_dict,
    model_order=2,
    extra_posteriors=None,
    extra_metadata=None,
    h5_path=None,
):
    """Save a standard Bayesian level-space (DOLS) component.

    Parameters
    ----------
    component_name : str
        Group name in the HDF5 file (e.g. ``'ocean'``, ``'glacier'``).
    result : BayesianLevelResult or BayesianPhysicalResult
        Fitted result object with ``posterior_samples`` and ``H0_posterior``.
    obs_years, obs_H, obs_sigma : array-like
        Observation data used for calibration (meters).
    proj_dict : dict
        ``{ssp: {'samples': ndarray, 'median': ndarray, ...}}``.
    model_order : int
        1 = linear, 2 = quadratic.
    extra_posteriors : dict or None
        Additional posterior arrays to save (e.g. ``{'tau_u': tau_array}``).
    extra_metadata : dict or None
        Scalar metadata (e.g. ``{'r2': 0.95, 'model_type': 'physical_1L'}``).
    h5_path : str or Path or None
        Output file.  Defaults to ``data/processed/component_results.h5``.
    """
    h5_path = _require_file(h5_path or DEFAULT_H5_PATH)

    with h5py.File(h5_path, "a") as f:
        # Clear old group
        if component_name in f:
            del f[component_name]
        cg = f.create_group(component_name)
        cg.attrs["model_order"] = model_order
        cg.attrs["saved_at"] = datetime.now(timezone.utc).isoformat()

        if extra_metadata:
            _write_scalar_attrs(cg, extra_metadata)

        # Posteriors
        pg = cg.create_group("posteriors")
        pg.create_dataset("posterior_samples",
                          data=np.asarray(result.posterior_samples))
        pg.create_dataset("H0_posterior",
                          data=np.asarray(result.H0_posterior))
        n_params = np.asarray(result.posterior_samples).shape[1]
        if n_params == 3:
            pg.attrs["param_names"] = json.dumps(["a", "b", "c"])
        elif n_params == 2:
            pg.attrs["param_names"] = json.dumps(["b", "c"])
        else:
            pg.attrs["param_names"] = json.dumps(
                [f"p{i}" for i in range(n_params)]
            )

        if extra_posteriors:
            for name, arr in extra_posteriors.items():
                pg.create_dataset(name, data=np.asarray(arr))

        # Observations
        _save_observations(cg, obs_years, obs_H, obs_sigma)

        # Projections
        prg = cg.create_group("projections")
        for ssp in proj_dict:
            _save_projection_ssp(prg, ssp, proj_dict[ssp])

    size_kb = os.path.getsize(h5_path) / 1024
    print(f"Saved {component_name} → {h5_path}  ({size_kb:.0f} KB total)")


# =========================================================================
# Ocean (thermosteric) — joint NOAA + EN4 calibration
# =========================================================================

def save_ocean(
    result_phys,
    obs_years, obs_H, obs_sigma,
    proj_dict,
    ocean_transfer=None,
    extra_metadata=None,
    h5_path=None,
):
    """Save thermosteric (ocean) component.

    ``result_phys`` should be the physical 1-layer result from the joint
    NOAA + EN4 calibration, with ``tau_u_posterior``, ``kappa_posterior``,
    and ``delta_posterior``.

    Parameters
    ----------
    ocean_transfer : dict or None
        Transfer function T_sub = kappa*S_u + delta, as extracted from
        ``result_phys``.  Scalar parameters are stored as HDF5 attrs;
        posterior arrays (kappa_posterior, delta_posterior) are stored
        as datasets.
    """
    extra_post = {}
    if hasattr(result_phys, "tau_u_posterior"):
        extra_post["tau_u_posterior"] = result_phys.tau_u_posterior
    if hasattr(result_phys, "tau_d_posterior") and result_phys.tau_d_posterior is not None:
        extra_post["tau_d_posterior"] = result_phys.tau_d_posterior
    if hasattr(result_phys, "kappa_posterior") and result_phys.kappa_posterior is not None:
        extra_post["kappa_posterior"] = result_phys.kappa_posterior
    if hasattr(result_phys, "delta_posterior") and result_phys.delta_posterior is not None:
        extra_post["delta_posterior"] = result_phys.delta_posterior
    if hasattr(result_phys, "sigma_ocean_posterior") and result_phys.sigma_ocean_posterior is not None:
        extra_post["sigma_ocean_posterior"] = result_phys.sigma_ocean_posterior

    meta = {"model_type": "physical_1layer_joint"}
    if hasattr(result_phys, "r2"):
        meta["r2"] = float(result_phys.r2)
    if hasattr(result_phys, "r2_ocean") and result_phys.r2_ocean is not None:
        meta["r2_ocean"] = float(result_phys.r2_ocean)
    if extra_metadata:
        meta.update(extra_metadata)

    save_dols_component(
        component_name="ocean",
        result=result_phys,
        obs_years=obs_years,
        obs_H=obs_H,
        obs_sigma=obs_sigma,
        proj_dict=proj_dict,
        model_order=2,
        extra_posteriors=extra_post or None,
        extra_metadata=meta,
        h5_path=h5_path,
    )

    # Save ocean transfer function as a separate sub-group
    if ocean_transfer is not None:
        h5_path = Path(h5_path or DEFAULT_H5_PATH)
        with h5py.File(h5_path, "a") as f:
            otg_path = "ocean/ocean_transfer"
            if otg_path in f:
                del f[otg_path]
            otg = f.create_group(otg_path)
            for key in ("kappa", "kappa_se", "delta", "delta_se",
                        "r2", "sigma_ocean", "source"):
                if key in ocean_transfer:
                    val = ocean_transfer[key]
                    if isinstance(val, str):
                        otg.attrs[key] = val
                    elif np.isscalar(val):
                        otg.attrs[key] = float(val)
            # Store posterior arrays if present
            for arr_key in ("kappa_posterior", "delta_posterior"):
                if arr_key in ocean_transfer and ocean_transfer[arr_key] is not None:
                    otg.create_dataset(arr_key,
                                       data=np.asarray(ocean_transfer[arr_key]))


def save_ocean_hybrid(
    obs_years, obs_H, obs_sigma,
    proj_dict,
    extra_metadata=None,
    h5_path=None,
):
    """Save thermosteric (ocean) component from hybrid NOAA + IPCC approach.

    Unlike :func:`save_ocean`, this does not require a fitted result object
    or posteriors. The hindcast and the projections come from the two-layer
    (Geoffroy) thermosteric model driven by Berkeley Earth GMST and the AR6
    GMST paths; its posterior draws are cached by component_ocean in the
    top-level ``ocean_posteriors`` group. The group is labelled
    ``model_type = 'twolayer_noaa'`` unless ``extra_metadata`` overrides it.
    No ``model_order`` is stored: the two-layer model has no rate-polynomial
    order, and no reader uses an ocean value (the name predates the
    two-layer model, when projections came from IPCC AR6 oceandynamics).

    Parameters
    ----------
    obs_years, obs_H, obs_sigma : array-like
        Full-depth observation data (meters, relative to BASELINE_YEAR).
    proj_dict : dict
        ``{ssp: {'samples': ndarray(N,T), 'median': ndarray(T), ...}}``.
    extra_metadata : dict or None
        Scalar metadata (e.g. correction rates, references).
    h5_path : str or Path or None
        Output file.  Defaults to ``data/processed/component_results.h5``.
    """
    h5_path = _require_file(h5_path or DEFAULT_H5_PATH)

    with h5py.File(h5_path, "a") as f:
        if "ocean" in f:
            del f["ocean"]
        cg = f.create_group("ocean")
        cg.attrs["model_type"] = "twolayer_noaa"
        cg.attrs["saved_at"] = datetime.now(timezone.utc).isoformat()

        if extra_metadata:
            _write_scalar_attrs(cg, extra_metadata)

        _save_observations(cg, obs_years, obs_H, obs_sigma)

        prg = cg.create_group("projections")
        for ssp in proj_dict:
            _save_projection_ssp(prg, ssp, proj_dict[ssp])

    size_kb = os.path.getsize(h5_path) / 1024
    print(f"Saved ocean (hybrid) → {h5_path}  ({size_kb:.0f} KB total)")


# =========================================================================
# Glacier
# =========================================================================

def _fit_extra_posteriors(result):
    """Nuisance posteriors of a fit_bayesian_level_annual_correlated result
    that the manuscript quotes (sigma_extra, T_eq0), so they can be read back
    and printed from the HDF5 (GL-2)."""
    out = {}
    sx = getattr(result, "sigma_extra_posterior", None)
    if sx is not None and np.any(np.asarray(sx) != 0):
        out["sigma_extra_posterior"] = np.asarray(sx)
    info = getattr(result, "design_info", None) or {}
    if info.get("T_eq0_samples") is not None:
        out["T_eq0_posterior"] = np.asarray(info["T_eq0_samples"])
    phi = getattr(result, "phi_posterior", None)
    if phi is not None:
        out["phi_posterior"] = np.asarray(phi)
    return out or None


def save_glacier(
    result_lin,
    obs_years, obs_H, obs_sigma,
    proj_dict,
    v_glacier_total_m=0.32,
    extra_metadata=None,
    h5_path=None,
):
    """Save glacier component (linear-in-T level model + volume cap).

    Default ``model_type`` 'linear_level_correlated' (the GlaMBIE fit with
    fit_bayesian_level_annual_correlated); component_glacier also passes
    its label through ``extra_metadata``.
    """
    meta = {"model_type": "linear_level_correlated",
            "v_glacier_total_m": v_glacier_total_m}
    if hasattr(result_lin, "r2"):
        meta["r2"] = float(result_lin.r2)
    if extra_metadata:
        meta.update(extra_metadata)

    save_dols_component(
        component_name="glacier",
        result=result_lin,
        obs_years=obs_years,
        obs_H=obs_H,
        obs_sigma=obs_sigma,
        proj_dict=proj_dict,
        model_order=1,
        extra_posteriors=_fit_extra_posteriors(result_lin),
        extra_metadata=meta,
        h5_path=h5_path,
    )


# =========================================================================
# EAIS
# =========================================================================

def save_eais(
    result_lin,
    obs_years, obs_H, obs_sigma,
    proj_dict,
    extra_metadata=None,
    h5_path=None,
):
    """Save East Antarctic Ice Sheet component.

    The stored ``model_type`` 'trend_only' is a legacy label: the current
    EAIS model is linear in T (SMB C_T model fitted to IMBIE-3). The label
    is kept because tests/test_eais.py checks it; pass ``model_type`` in
    ``extra_metadata`` to override it.
    """
    meta = {"model_type": "trend_only"}
    if hasattr(result_lin, "r2"):
        meta["r2"] = float(result_lin.r2)
    if extra_metadata:
        meta.update(extra_metadata)

    save_dols_component(
        component_name="eais",
        result=result_lin,
        obs_years=obs_years,
        obs_H=obs_H,
        obs_sigma=obs_sigma,
        proj_dict=proj_dict,
        model_order=1,
        extra_posteriors=_fit_extra_posteriors(result_lin),
        extra_metadata=meta,
        h5_path=h5_path,
    )


# =========================================================================
# Antarctic Peninsula
# =========================================================================

def save_apeninsula(
    result_lin,
    obs_years, obs_H, obs_sigma,
    proj_dict,
    extra_metadata=None,
    h5_path=None,
):
    """Save Antarctic Peninsula component.

    The stored ``model_type`` 'linear_dols' is a legacy label: the current
    model is the linear-in-T level fit to IMBIE-3 annual increments
    (fit_bayesian_level_annual_correlated). The label is kept because
    tests/test_apeninsula.py checks it; pass ``model_type`` in
    ``extra_metadata`` to override it.
    """
    meta = {"model_type": "linear_dols"}
    if hasattr(result_lin, "r2"):
        meta["r2"] = float(result_lin.r2)
    if extra_metadata:
        meta.update(extra_metadata)

    save_dols_component(
        component_name="apeninsula",
        result=result_lin,
        obs_years=obs_years,
        obs_H=obs_H,
        obs_sigma=obs_sigma,
        proj_dict=proj_dict,
        model_order=1,
        extra_posteriors=_fit_extra_posteriors(result_lin),
        extra_metadata=meta,
        h5_path=h5_path,
    )


# =========================================================================
# Greenland (joint SMB + discharge)
# =========================================================================

def save_greenland(
    result_discharge,
    smb_sensitivity,
    ocean_transfer,
    obs_smb_years, obs_smb_H, obs_smb_sigma,
    obs_dyn_years, obs_dyn_H, obs_dyn_sigma,
    greenland_proj,
    smb_projections=None,
    discharge_projections=None,
    extra_metadata=None,
    h5_path=None,
):
    """Save Greenland component (SMB emulator + drift state, discharge delay model).

    ``model_type`` is 'smb_emulator_drift_plus_discharge_delay': SMB comes
    from the MAR emulator with a drift state (``smb_sensitivity`` holds the
    emulator summary) and discharge from the analytic increment fit. (It
    was 'smb_literature_plus_discharge_delay' for the retired C_T model.)

    Parameters
    ----------
    result_discharge : SimpleNamespace
        Discharge delay-model fit result with gamma_posterior, r0_posterior,
        delta_posterior arrays.
    smb_sensitivity : object
        SMB model summary (the MAR emulator summary in production; the
        retired literature SMBSensitivity in older tests). Fields among
        C_T, C_T_sigma, ... and ``extra_attrs`` are stored as attributes.
    ocean_transfer : dict
        Surface→ocean temperature transfer function from
        ``fit_ocean_transfer_function``.
    obs_smb_years, obs_smb_H, obs_smb_sigma : array-like
        Mouginot SMB observations (meters).
    obs_dyn_years, obs_dyn_H, obs_dyn_sigma : array-like
        Mouginot discharge observations (meters).
    greenland_proj : dict
        ``{ssp: {'samples': ndarray, 'median': ..., 'smb_median': ...,
        'dyn_median': ...}}``.
    smb_projections : dict or None
        ``{ssp: {'samples': ndarray, ...}}`` for SMB sub-component.
    discharge_projections : dict or None
        ``{ssp: {'samples': ndarray, ...}}`` for discharge sub-component.
    extra_metadata : dict or None
    h5_path : str or Path or None
    """
    h5_path = _require_file(h5_path or DEFAULT_H5_PATH)

    with h5py.File(h5_path, "a") as f:
        if "greenland" in f:
            del f["greenland"]
        cg = f.create_group("greenland")
        cg.attrs["model_type"] = "smb_emulator_drift_plus_discharge_delay"
        cg.attrs["saved_at"] = datetime.now(timezone.utc).isoformat()
        if extra_metadata:
            _write_scalar_attrs(cg, extra_metadata)

        # ── Discharge posteriors (delay model) ──
        dg = cg.create_group("posteriors/discharge")
        for attr_name in ("gamma_posterior", "r0_posterior", "delta_posterior"):
            arr = getattr(result_discharge, attr_name, None)
            if arr is not None:
                dg.create_dataset(attr_name, data=np.asarray(arr))
        if hasattr(result_discharge, "r2_dyn"):
            dg.attrs["r2"] = float(result_discharge.r2_dyn)
        # Calibration demeaning constants — needed to reconstruct projections
        # without re-running the fit
        for cal_attr in ("H_mean_cal", "int_T_mean_cal", "t_mean_cal", "delta_best"):
            val = getattr(result_discharge, cal_attr, None)
            if val is not None:
                dg.attrs[cal_attr] = float(val)

        # ── SMB model attributes (literature C_T, or the Greenland emulator) ──
        sg = cg.create_group("smb_sensitivity")
        for field in ("C_T", "C_T_sigma", "C_T2", "C_T2_sigma",
                      "SMB_0", "reference", "temperature_frame", "AA_factor"):
            val = getattr(smb_sensitivity, field, None)
            if val is not None:
                sg.attrs[field] = val if isinstance(val, str) else float(val)
        for key, val in getattr(smb_sensitivity, 'extra_attrs', {}).items():
            sg.attrs[key] = val

        # ── Ocean transfer function ──
        otg = cg.create_group("ocean_transfer")
        for key in ("alpha", "beta", "alpha_se", "beta_se",
                     "r2", "residual_std", "lag_years", "n"):
            if key in ocean_transfer:
                otg.attrs[key] = float(ocean_transfer[key])

        # ── Observations ──
        _save_observations(cg, obs_smb_years, obs_smb_H, obs_smb_sigma,
                           subgroup="smb")
        _save_observations(cg, obs_dyn_years, obs_dyn_H, obs_dyn_sigma,
                           subgroup="discharge")

        # ── Total projections ──
        prg = cg.create_group("projections")
        for ssp in greenland_proj:
            _save_projection_ssp(prg, ssp, greenland_proj[ssp])

        # ── Sub-component projections (optional but recommended) ──
        if smb_projections is not None:
            smb_pg = cg.create_group("projections_smb")
            for ssp in smb_projections:
                _save_projection_ssp(smb_pg, ssp, smb_projections[ssp])

        if discharge_projections is not None:
            dyn_pg = cg.create_group("projections_discharge")
            for ssp in discharge_projections:
                _save_projection_ssp(dyn_pg, ssp, discharge_projections[ssp])

    size_kb = os.path.getsize(h5_path) / 1024
    print(f"Saved greenland → {h5_path}  ({size_kb:.0f} KB total)")


# =========================================================================
# WAIS (A4 deep-uncertainty framework)
# =========================================================================

def save_wais(
    a4_scenarios,
    obs_years, obs_H, obs_sigma,
    wais_proj,
    rheology_factor_median=1.0,
    rheology_factor_sigma=0.0,
    rheology_mode='B',
    wais_onset_year=None,
    extra_metadata=None,
    h5_path=None,
):
    """Save West Antarctic Ice Sheet component (A4 scenario framework).

    Parameters
    ----------
    a4_scenarios : dict
        ``{'S1_status_quo': {'P': ..., 'low_mm': ..., 'high_mm': ...,
        'alpha': ..., 'beta_loc': ..., 'beta_scale': ..., 'misi': ...}, ...}``
    obs_years, obs_H, obs_sigma : array-like
        IMBIE WAIS observations (meters).
    wais_proj : dict
        ``{ssp: {'samples': ndarray, ...}}``.
        WAIS is SSP-independent; same samples used for all SSPs.
    rheology_factor_median, rheology_factor_sigma : float
        Rheology correction parameters (n=3 → n=4). The correction is
        deprecated; the defaults (1.0, 0.0) record that none is applied.
    rheology_mode : str
        Which rheology correction mode was used ('A' or 'B'); default 'B',
        the production setting (no correction).
    wais_onset_year : float or None
        WAIS onset year for the power-law ramp.
    extra_metadata : dict or None
    h5_path : str or Path or None
    """
    h5_path = _require_file(h5_path or DEFAULT_H5_PATH)

    with h5py.File(h5_path, "a") as f:
        if "wais" in f:
            del f["wais"]
        cg = f.create_group("wais")
        cg.attrs["model_type"] = "a4_deep_uncertainty"
        cg.attrs["ssp_independent"] = True
        cg.attrs["saved_at"] = datetime.now(timezone.utc).isoformat()
        cg.attrs["rheology_factor_median"] = rheology_factor_median
        cg.attrs["rheology_factor_sigma"] = rheology_factor_sigma
        cg.attrs["rheology_mode"] = rheology_mode
        if wais_onset_year is not None:
            cg.attrs["wais_onset_year"] = float(wais_onset_year)
        if extra_metadata:
            _write_scalar_attrs(cg, extra_metadata)

        # ── A4 scenario parameters ──
        a4g = cg.create_group("a4_scenarios")
        for sname, sparams in a4_scenarios.items():
            sg = a4g.create_group(sname)
            sg.attrs["P"] = sparams["P"]
            # S1_status_quo has no low_mm/high_mm/alpha/beta -- it is
            # sampled directly from its quadratic-in-time posterior
            # (component_projections.get_s1_quadratic()) instead of
            # the skew-normal parametrization used for S2. NaN here means
            # "not applicable", not a missing value.
            sg.attrs["low_mm"] = sparams.get("low_mm", np.nan)
            sg.attrs["high_mm"] = sparams.get("high_mm", np.nan)
            sg.attrs["alpha"] = sparams.get("alpha", np.nan)
            sg.attrs["beta_loc"] = sparams.get("beta_loc", np.nan)
            sg.attrs["beta_scale"] = sparams.get("beta_scale", np.nan)
            sg.attrs["misi"] = sparams["misi"]

        # ── Observations ──
        _save_observations(cg, obs_years, obs_H, obs_sigma)

        # ── Projections ──
        prg = cg.create_group("projections")
        for ssp in wais_proj:
            _save_projection_ssp(prg, ssp, wais_proj[ssp])

    size_kb = os.path.getsize(h5_path) / 1024
    print(f"Saved wais → {h5_path}  ({size_kb:.0f} KB total)")


# =========================================================================
# Loading functions
# =========================================================================

def load_component(component_name, h5_path=None):
    """Load all saved data for one component.

    Returns
    -------
    dict with keys:
        'projections' : ``{ssp: {'samples': ndarray, 'median': ..., ...}}``
        'observations' : ``{'years': ..., 'H_obs': ..., 'sigma': ...}``
            or for Greenland: ``{'smb': {...}, 'discharge': {...}}``
        'posteriors' : dict of arrays (component-dependent)
        'metadata' : dict of scalar attributes
        'proj_years' : ndarray — shared time axis
    """
    h5_path = Path(h5_path or DEFAULT_H5_PATH)
    if not h5_path.exists():
        raise FileNotFoundError(h5_path)

    out = {}

    with h5py.File(h5_path, "r") as f:
        out["proj_years"] = f["proj_years"][:]

        if component_name not in f:
            raise KeyError(
                f"Component '{component_name}' not found. "
                f"Available: {list(f.keys())}"
            )
        cg = f[component_name]

        # ── Metadata (scalar attrs) ──
        out["metadata"] = dict(cg.attrs)

        # ── Projections ──
        projections = {}
        if "projections" in cg:
            for ssp in cg["projections"]:
                projections[ssp] = _load_projection_ssp(cg["projections"][ssp])

            # SSP-independent components (e.g. WAIS) store samples only
            # under the first SSP to avoid redundancy.  Propagate to the
            # others so downstream code sees a uniform structure.
            if cg.attrs.get("ssp_independent", False):
                ref_samples = None
                for ssp in projections:
                    if "samples" in projections[ssp]:
                        ref_samples = projections[ssp]["samples"]
                        break
                if ref_samples is not None:
                    for ssp in projections:
                        if "samples" not in projections[ssp]:
                            projections[ssp]["samples"] = ref_samples

        out["projections"] = projections

        # ── Sub-component projections (Greenland) ──
        for sub_key in ("projections_smb", "projections_discharge"):
            if sub_key in cg:
                sub_proj = {}
                for ssp in cg[sub_key]:
                    sub_proj[ssp] = _load_projection_ssp(cg[sub_key][ssp])
                out[sub_key] = sub_proj

        # ── Observations ──
        if "observations" in cg:
            obs_grp = cg["observations"]
            # Check if it has sub-groups (Greenland) or direct datasets
            if "years" in obs_grp:
                out["observations"] = {
                    "years": obs_grp["years"][:],
                    "H_obs": obs_grp["H_obs"][:],
                    "sigma": obs_grp["sigma"][:],
                }
            else:
                # Sub-grouped (e.g. Greenland smb/discharge)
                obs = {}
                for sub in obs_grp:
                    obs[sub] = {
                        "years": obs_grp[sub]["years"][:],
                        "H_obs": obs_grp[sub]["H_obs"][:],
                        "sigma": obs_grp[sub]["sigma"][:],
                    }
                out["observations"] = obs

        # ── Posteriors ──
        posteriors = {}
        if "posteriors" in cg:
            pg = cg["posteriors"]
            # Check for direct datasets (standard DOLS)
            if "posterior_samples" in pg:
                posteriors["posterior_samples"] = pg["posterior_samples"][:]
                posteriors["H0_posterior"] = pg["H0_posterior"][:]
                for extra in pg:
                    if extra not in ("posterior_samples", "H0_posterior"):
                        posteriors[extra] = pg[extra][:]
                if "param_names" in pg.attrs:
                    posteriors["param_names"] = json.loads(pg.attrs["param_names"])
            elif all(isinstance(pg[k], h5py.Group) for k in pg):
                # Sub-grouped (e.g. Greenland discharge/)
                for sub in pg:
                    posteriors[sub] = {}
                    for ds_name in pg[sub]:
                        posteriors[sub][ds_name] = pg[sub][ds_name][:]
                    posteriors[sub].update(dict(pg[sub].attrs))
            else:
                # Flat named posterior datasets, no posterior_samples key
                # (e.g. ocean two-layer model: a_posterior, b_u_posterior, ...)
                for ds_name in pg:
                    posteriors[ds_name] = pg[ds_name][:]
                posteriors.update(dict(pg.attrs))
        out["posteriors"] = posteriors

        # ── SMB sensitivity (Greenland) ──
        if "smb_sensitivity" in cg:
            out["smb_sensitivity"] = dict(cg["smb_sensitivity"].attrs)

        # ── Ocean transfer (Greenland) ──
        if "ocean_transfer" in cg:
            out["ocean_transfer"] = dict(cg["ocean_transfer"].attrs)

        # ── A4 scenarios (WAIS) ──
        if "a4_scenarios" in cg:
            a4 = {}
            for sname in cg["a4_scenarios"]:
                a4[sname] = dict(cg["a4_scenarios"][sname].attrs)
                # h5py stores bools as numpy bool_ — convert
                if "misi" in a4[sname]:
                    a4[sname]["misi"] = bool(a4[sname]["misi"])
            out["a4_scenarios"] = a4

    return out


def load_projections(component_name, h5_path=None):
    """Load only the projection dict for one component.

    Returns
    -------
    proj_years : ndarray
    projections : ``{ssp: {'samples': ndarray, 'median': ndarray, ...}}``
    """
    h5_path = Path(h5_path or DEFAULT_H5_PATH)
    with h5py.File(h5_path, "r") as f:
        proj_years = f["proj_years"][:]
        projections = {}
        for ssp in f[component_name]["projections"]:
            projections[ssp] = _load_projection_ssp(
                f[component_name]["projections"][ssp]
            )
    return proj_years, projections


def load_all_projections(h5_path=None, components=None):
    """Load projection dicts for all (or selected) components.

    Parameters
    ----------
    h5_path : str or Path or None
    components : list of str or None
        If None, loads all component groups found in the file.

    Returns
    -------
    proj_years : ndarray
    all_proj : ``{component: {ssp: {'samples': ..., 'median': ..., ...}}}``
    """
    h5_path = Path(h5_path or DEFAULT_H5_PATH)
    all_proj = {}

    with h5py.File(h5_path, "r") as f:
        proj_years = f["proj_years"][:]

        if components is None:
            components = [k for k in f.keys() if k != "proj_years"]

        for comp in components:
            if comp not in f:
                print(f"  Warning: '{comp}' not in {h5_path.name}, skipping")
                continue
            if "projections" not in f[comp]:
                continue
            cg = f[comp]
            all_proj[comp] = {}
            for ssp in cg["projections"]:
                all_proj[comp][ssp] = _load_projection_ssp(
                    cg["projections"][ssp]
                )

            # SSP-independent components (e.g. WAIS) store samples only
            # under the first SSP to avoid redundancy.  Propagate to the
            # others so downstream code sees a uniform structure.
            if cg.attrs.get("ssp_independent", False):
                ref_samples = None
                for ssp in all_proj[comp]:
                    if "samples" in all_proj[comp][ssp]:
                        ref_samples = all_proj[comp][ssp]["samples"]
                        break
                if ref_samples is not None:
                    for ssp in all_proj[comp]:
                        if "samples" not in all_proj[comp][ssp]:
                            all_proj[comp][ssp]["samples"] = ref_samples

    return proj_years, all_proj


def list_components(h5_path=None):
    """List components saved in the HDF5 file with summary info."""
    h5_path = Path(h5_path or DEFAULT_H5_PATH)
    if not h5_path.exists():
        print(f"File not found: {h5_path}")
        return []

    components = []
    with h5py.File(h5_path, "r") as f:
        print(f"File: {h5_path}")
        print(f"  Baseline year (stored): {f.attrs.get('baseline_year', '?')}"
              f"  (config: {BASELINE_YEAR})")
        print(f"  Proj years: {f['proj_years'][0]:.0f}–{f['proj_years'][-1]:.0f}")
        print()

        for key in sorted(f.keys()):
            if key == "proj_years":
                continue
            cg = f[key]
            components.append(key)
            model = cg.attrs.get("model_type", "unknown")
            saved = cg.attrs.get("saved_at", "?")
            ssps = list(cg["projections"].keys()) if "projections" in cg else []
            has_samples = False
            n_members = None
            if ssps and "samples" in cg[f"projections/{ssps[0]}"]:
                has_samples = True
                n_members = cg[f"projections/{ssps[0]}/samples"].shape[0]
            print(f"  {key}")
            print(f"    model: {model}")
            print(f"    SSPs: {', '.join(ssps)}")
            print(f"    full samples: {has_samples}"
                  + (f" ({n_members} members)" if has_samples else ""))
            print(f"    saved: {saved}")

    return components


# =========================================================================
# Provenance of groups derived from the blended forecast
# =========================================================================

DERIVED_FROM_BLENDED = ('blended_slow90', 'wais_slow90', 'voi',
                        'optimal_defense', 'risk_shapley')
"""Groups computed from `blended` (and the groups component_forecast writes
with it) by scripts that run after component_forecast."""


def stamp_derived(hf, group):
    """Record which `blended` run a derived group was computed from."""
    hf[group].attrs['source_saved_at'] = str(hf['blended'].attrs['saved_at'])
    hf[group].attrs['saved_at'] = datetime.now(timezone.utc).isoformat()


def check_derived(hf, groups=DERIVED_FROM_BLENDED):
    """Raise if a derived group is missing or was computed from an earlier
    `blended` run (e.g. after a partial rerun of component_forecast)."""
    current = str(hf['blended'].attrs['saved_at'])
    stale = [g for g in groups
             if g not in hf or hf[g].attrs.get('source_saved_at') != current]
    if stale:
        raise RuntimeError(
            f'Derived groups {stale} are missing or predate the current '
            f'blended forecast ({current}); rerun the pipeline steps after '
            f'`forecast` (slowwais, evpi, defense, shapley).')
