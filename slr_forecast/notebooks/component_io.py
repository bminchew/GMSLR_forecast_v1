"""
Component decomposition HDF5 I/O.

Save and load fitted parameters, projection ensembles, and observation data
from each per-component notebook so that downstream aggregation notebooks
can inherit results without re-running upstream fits.

File layout (single HDF5 file, one group per component)::

    component_results.h5
    ├── proj_years                (201,) — shared time axis
    ├── ocean/
    │   ├── posteriors/           fitted model params
    │   ├── observations/         calibration data
    │   └── projections/{ssp}/    MC samples + percentiles
    ├── glacier/
    │   ...
    ├── greenland/
    │   ├── posteriors/discharge/ discharge delay-model posteriors
    │   ├── smb_sensitivity/      literature C_T values
    │   ├── ocean_transfer/       surface→ocean T regression
    │   ├── observations/smb/     Mouginot SMB
    │   ├── observations/discharge/
    │   └── projections/{ssp}/    total, smb, discharge samples
    ├── eais/
    ├── apeninsula/
    └── wais/
        ├── a4_scenarios/         scenario mixture params
        ...

All projection arrays are in meters relative to BASELINE_YEAR.
"""

import json
import os
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
PROJ_SSPS = ["SSP1-2.6", "SSP2-4.5", "SSP3-7.0", "SSP5-8.5"]

# Standard percentile keys stored per SSP
_PCTILE_KEYS = ("median", "p5", "p17", "p83", "p95")


# =========================================================================
# Low-level helpers
# =========================================================================

def _require_file(path):
    """Open (or create) the HDF5 file and return the path."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        with h5py.File(path, "w") as f:
            f.attrs["created"] = datetime.now(timezone.utc).isoformat()
            f.attrs["baseline_year"] = BASELINE_YEAR
            f.attrs["n_samples"] = N_SAMPLES
            f.create_dataset("proj_years", data=PROJ_YEARS)
    return path


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
            for k, v in extra_metadata.items():
                if isinstance(v, str):
                    cg.attrs[k] = v
                elif np.isscalar(v) and np.isfinite(v):
                    cg.attrs[k] = float(v)

        # Posteriors
        pg = cg.create_group("posteriors")
        pg.create_dataset("posterior_samples",
                          data=np.asarray(result.posterior_samples))
        pg.create_dataset("H0_posterior",
                          data=np.asarray(result.H0_posterior))
        pg.attrs["param_names"] = json.dumps(["a", "b", "c"])

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
    or posteriors.  The hindcast is observational (NOAA + literature depth
    corrections) and projections come from IPCC AR6 oceandynamics.

    Parameters
    ----------
    obs_years, obs_H, obs_sigma : array-like
        Full-depth observation data (meters, relative to 2005).
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
        cg.attrs["model_type"] = "hybrid_noaa_ipcc"
        cg.attrs["model_order"] = 0
        cg.attrs["saved_at"] = datetime.now(timezone.utc).isoformat()

        if extra_metadata:
            for k, v in extra_metadata.items():
                if isinstance(v, str):
                    cg.attrs[k] = v
                elif np.isscalar(v) and np.isfinite(v):
                    cg.attrs[k] = float(v)

        _save_observations(cg, obs_years, obs_H, obs_sigma)

        prg = cg.create_group("projections")
        for ssp in proj_dict:
            _save_projection_ssp(prg, ssp, proj_dict[ssp])

    size_kb = os.path.getsize(h5_path) / 1024
    print(f"Saved ocean (hybrid) → {h5_path}  ({size_kb:.0f} KB total)")


# =========================================================================
# Glacier
# =========================================================================

def save_glacier(
    result_lin,
    obs_years, obs_H, obs_sigma,
    proj_dict,
    v_glacier_total_m=0.32,
    extra_metadata=None,
    h5_path=None,
):
    """Save glacier component (linear DOLS + volume cap)."""
    meta = {"model_type": "linear_dols", "v_glacier_total_m": v_glacier_total_m}
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
    """Save East Antarctic Ice Sheet component (trend-only)."""
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
    """Save Antarctic Peninsula component (linear DOLS)."""
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
    """Save Greenland component (SMB from literature + discharge delay model).

    Parameters
    ----------
    result_discharge : SimpleNamespace
        Discharge delay-model fit result with gamma_posterior, r0_posterior,
        delta_posterior arrays.
    smb_sensitivity : SMBSensitivity
        Literature-derived SMB sensitivity parameters.
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
        cg.attrs["model_type"] = "smb_literature_plus_discharge_delay"
        cg.attrs["saved_at"] = datetime.now(timezone.utc).isoformat()
        if extra_metadata:
            for k, v in extra_metadata.items():
                if isinstance(v, str):
                    cg.attrs[k] = v
                elif np.isscalar(v) and np.isfinite(v):
                    cg.attrs[k] = float(v)

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

        # ── SMB sensitivity (literature values, not posteriors) ──
        sg = cg.create_group("smb_sensitivity")
        for field in ("C_T", "C_T_sigma", "C_T2", "C_T2_sigma",
                      "SMB_0", "reference", "temperature_frame", "AA_factor"):
            val = getattr(smb_sensitivity, field, None)
            if val is not None:
                sg.attrs[field] = val if isinstance(val, str) else float(val)

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
    rheology_factor_median=1.28,
    rheology_factor_sigma=0.07,
    rheology_mode='A',
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
        A1 rheology correction parameters (n=3 → n=4).
    rheology_mode : str
        Which rheology correction mode was used ('A' or 'B').
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
            for k, v in extra_metadata.items():
                if isinstance(v, str):
                    cg.attrs[k] = v
                elif np.isscalar(v) and np.isfinite(v):
                    cg.attrs[k] = float(v)

        # ── A4 scenario parameters ──
        a4g = cg.create_group("a4_scenarios")
        for sname, sparams in a4_scenarios.items():
            sg = a4g.create_group(sname)
            sg.attrs["P"] = sparams["P"]
            sg.attrs["low_mm"] = sparams["low_mm"]
            sg.attrs["high_mm"] = sparams["high_mm"]
            sg.attrs["alpha"] = sparams.get("alpha", 0.0)
            sg.attrs["beta_loc"] = sparams.get("beta_loc", 0.0)
            sg.attrs["beta_scale"] = sparams.get("beta_scale", 0.0)
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
            else:
                # Sub-grouped (e.g. Greenland discharge/)
                for sub in pg:
                    posteriors[sub] = {}
                    for ds_name in pg[sub]:
                        posteriors[sub][ds_name] = pg[sub][ds_name][:]
                    posteriors[sub].update(dict(pg[sub].attrs))
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
        print(f"  Baseline year: {f.attrs.get('baseline_year', '?')}")
        print(f"  N samples: {f.attrs.get('n_samples', '?')}")
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
            if ssps and "samples" in cg[f"projections/{ssps[0]}"]:
                has_samples = True
            print(f"  {key}")
            print(f"    model: {model}")
            print(f"    SSPs: {', '.join(ssps)}")
            print(f"    full samples: {has_samples}")
            print(f"    saved: {saved}")

    return components
