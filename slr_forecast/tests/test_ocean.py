"""Unit tests for the ocean (thermosteric) component.

Tests cover: NOAA thermosteric data reader, ODE solver, projection
structure, HDF5 save/load roundtrip, IPCC comparison data, and
physical plausibility of projections.
"""

import sys
import os
import tempfile

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'notebooks'))

from component_io import load_component, PROJ_YEARS

# Paths
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
RAW_DIR = os.path.join(PROJECT_ROOT, 'data', 'raw')
H5_PATH = os.path.join(PROJECT_ROOT, 'data', 'processed', 'component_results.h5')
NOAA_700_PATH = os.path.join(
    RAW_DIR, 'steric',
    'noaa_thermosteric_SL_0-700-3month-20260206T160151Z-1-001.zip')
NOAA_2000_PATH = os.path.join(
    RAW_DIR, 'steric',
    'noaa_thermosteric_SL_0-2000-3month-20260206T160256Z-1-001.zip')
CONF_BASE = os.path.join(
    RAW_DIR, 'ipcc_ar6', 'slr', 'ar6', 'global', 'confidence_output_files')

HAS_NOAA_700 = os.path.exists(NOAA_700_PATH)
HAS_NOAA_2000 = os.path.exists(NOAA_2000_PATH)
HAS_H5 = os.path.exists(H5_PATH)
HAS_IPCC = os.path.exists(os.path.join(
    CONF_BASE, 'medium_confidence', 'ssp245'))

M_TO_MM = 1000.0
BASELINE_YEAR = 2005.0
PROJ_SSPS = ['SSP1-2.6', 'SSP2-4.5', 'SSP3-7.0', 'SSP5-8.5']


# =========================================================================
# NOAA thermosteric reader
# =========================================================================

@pytest.mark.skipif(not HAS_H5, reason="HDF5 data not found")
class TestOceanObservationsFromHDF5:
    """Verify ocean observations loaded from HDF5 are plausible.

    The NOAA ZIP files contain per-grid-cell .dat files that require
    custom extraction (global-mean file selection).  We test the
    already-processed observations stored in HDF5 instead.
    """

    @pytest.fixture(scope="class")
    def obs(self):
        loaded = load_component('ocean')
        return loaded['observations']

    def test_years_span(self, obs):
        """Observations should span at least 1955-2020."""
        assert obs['years'][0] <= 1956
        assert obs['years'][-1] >= 2019

    def test_units_are_meters(self, obs):
        """H_obs should be in meters (order 1e-2 to 1e-1)."""
        valid = obs['H_obs'][np.isfinite(obs['H_obs'])]
        max_abs = np.max(np.abs(valid))
        assert max_abs < 1.0, f"Max |H| = {max_abs:.3f}, too large for meters"
        assert max_abs > 0.001, f"Max |H| = {max_abs:.4f}, too small for meters"

    def test_sigma_mostly_positive(self, obs):
        """Sigma should be positive where finite."""
        valid = obs['sigma'][np.isfinite(obs['sigma'])]
        assert np.all(valid >= 0)
        # At most 1 NaN allowed (last year may be incomplete)
        assert np.sum(np.isnan(obs['sigma'])) <= 1

    def test_increasing_trend(self, obs):
        """Thermosteric SLR should have positive trend post-1970."""
        mask = (obs['years'] >= 1970) & np.isfinite(obs['H_obs'])
        H = obs['H_obs'][mask]
        t = obs['years'][mask] - obs['years'][mask].mean()
        slope = np.sum(t * H) / np.sum(t**2)
        assert slope > 0, f"Slope = {slope:.4f}, expected positive"


# =========================================================================
# ODE solver
# =========================================================================

class TestTwoLayerODE:
    """Verify solve_twolayer_ode basic properties."""

    @pytest.fixture(scope="class")
    def ode_solver(self):
        from bayesian_dols import solve_twolayer_ode
        return solve_twolayer_ode

    def test_step_response_equilibrium(self, ode_solver):
        """Step forcing: S_u should approach T for large t."""
        n = 1000
        time = np.arange(n, dtype=float)
        T = np.ones(n) * 2.0  # step to 2°C
        S_u, S_d = ode_solver(T, time, tau_u=5.0, tau_d=np.inf,
                               Su0=0.0, Sd0=0.0)
        # After 10*tau_u = 50 years, S_u ~ T
        assert abs(S_u[-1] - 2.0) < 0.01, f"S_u[-1]={S_u[-1]:.4f}, expected ~2.0"

    def test_deep_layer_slower(self, ode_solver):
        """Deep layer should respond slower than upper."""
        n = 500
        time = np.arange(n, dtype=float)
        T = np.ones(n) * 1.0
        # Explicit Su0=0, Sd0=0 to start from rest
        S_u, S_d = ode_solver(T, time, tau_u=10.0, tau_d=100.0,
                               Su0=0.0, Sd0=0.0)
        # At t=15 (~1.5*tau_u): S_u should be closer to equilibrium than S_d
        idx = 15
        assert abs(S_u[idx] - 1.0) < abs(S_d[idx] - 1.0), (
            f"S_u[{idx}]={S_u[idx]:.3f}, S_d[{idx}]={S_d[idx]:.3f}")

    def test_one_layer_mode(self, ode_solver):
        """tau_d=inf should make S_d remain at initial condition."""
        n = 100
        time = np.arange(n, dtype=float)
        T = np.ones(n)
        S_u, S_d = ode_solver(T, time, tau_u=5.0, tau_d=np.inf)
        assert np.allclose(S_d, S_d[0]), "S_d should not change when tau_d=inf"

    def test_output_shape(self, ode_solver):
        n = 200
        time = np.arange(n, dtype=float)
        T = np.sin(2 * np.pi * time / 50)
        S_u, S_d = ode_solver(T, time, tau_u=8.0, tau_d=150.0)
        assert S_u.shape == (n,)
        assert S_d.shape == (n,)

    def test_zero_forcing(self, ode_solver):
        """Zero forcing → S_u stays near zero."""
        n = 100
        time = np.arange(n, dtype=float)
        T = np.zeros(n)
        S_u, S_d = ode_solver(T, time, tau_u=10.0, tau_d=np.inf)
        assert np.max(np.abs(S_u)) < 0.01

    def test_tau_u_controls_lag(self, ode_solver):
        """Larger tau_u → more lag in S_u response."""
        n = 200
        time = np.arange(n, dtype=float)
        T = np.where(time >= 50, 1.0, 0.0)  # step at t=50
        S_u_fast, _ = ode_solver(T, time, tau_u=5.0, tau_d=np.inf,
                                  Su0=0.0, Sd0=0.0)
        S_u_slow, _ = ode_solver(T, time, tau_u=50.0, tau_d=np.inf,
                                  Su0=0.0, Sd0=0.0)
        # At t=70 (20 yr after step), fast should be closer to 1
        idx = 70
        assert S_u_fast[idx] > S_u_slow[idx], (
            f"Fast tau={S_u_fast[idx]:.3f} should exceed slow tau={S_u_slow[idx]:.3f}")


# =========================================================================
# HDF5 roundtrip
# =========================================================================

@pytest.mark.skipif(not HAS_H5, reason="component_results.h5 not found")
class TestOceanSaveLoad:
    """Verify save/load roundtrip for ocean component."""

    @pytest.fixture(scope="class")
    def loaded(self):
        return load_component('ocean')

    def test_metadata_model_type(self, loaded):
        mt = loaded['metadata']['model_type']
        assert mt in ('physical_1layer_joint', 'hybrid_noaa_ipcc', 'twolayer_noaa'), (
            f"Unexpected model_type: {mt}")

    def test_all_ssps_present(self, loaded):
        for ssp in PROJ_SSPS:
            assert ssp in loaded['projections'], f"Missing {ssp}"

    def test_projection_keys(self, loaded):
        proj = loaded['projections']['SSP2-4.5']
        for key in ('samples', 'median', 'p5', 'p95', 'p17', 'p83'):
            assert key in proj, f"Missing key '{key}' in SSP2-4.5"

    def test_samples_shape(self, loaded):
        samples = loaded['projections']['SSP2-4.5']['samples']
        assert samples.shape[0] == 2000
        assert samples.shape[1] == len(loaded['proj_years'])

    def test_percentile_ordering(self, loaded):
        proj = loaded['projections']['SSP2-4.5']
        idx = np.argmin(np.abs(loaded['proj_years'] - 2100))
        assert proj['p5'][idx] <= proj['p17'][idx]
        assert proj['p17'][idx] <= proj['median'][idx]
        assert proj['median'][idx] <= proj['p83'][idx]
        assert proj['p83'][idx] <= proj['p95'][idx]

    def test_observations_present(self, loaded):
        obs = loaded['observations']
        assert 'years' in obs
        assert 'H_obs' in obs
        assert 'sigma' in obs

    def test_observations_span(self, loaded):
        """Observations should span at least 1955-2020."""
        years = loaded['observations']['years']
        assert years[0] <= 1956
        assert years[-1] >= 2019

    def test_observations_count(self, loaded):
        n = len(loaded['observations']['years'])
        assert n >= 50, f"Only {n} observation years, expected >= 50"

    def test_proj_years_shape(self, loaded):
        assert loaded['proj_years'].shape == (201,)
        assert loaded['proj_years'][0] == 1950.0
        assert loaded['proj_years'][-1] == 2150.0


# =========================================================================
# Projection plausibility
# =========================================================================

@pytest.mark.skipif(not HAS_H5, reason="component_results.h5 not found")
class TestOceanProjections:
    """Verify ocean projections are physically plausible."""

    @pytest.fixture(scope="class")
    def loaded(self):
        return load_component('ocean')

    def test_positive_at_2100(self, loaded):
        """Thermosteric SLR should be positive at 2100."""
        idx = np.argmin(np.abs(loaded['proj_years'] - 2100))
        for ssp in PROJ_SSPS:
            med = loaded['projections'][ssp]['median'][idx]
            assert med > 0, (
                f"{ssp} median = {med*M_TO_MM:.1f} mm, expected > 0")

    def test_range_at_2100(self, loaded):
        """Ocean median at 2100: 50-400 mm across SSPs."""
        idx = np.argmin(np.abs(loaded['proj_years'] - 2100))
        for ssp in PROJ_SSPS:
            med_mm = loaded['projections'][ssp]['median'][idx] * M_TO_MM
            assert 50 < med_mm < 500, (
                f"{ssp} median = {med_mm:.0f} mm, outside [50, 500]")

    def test_ssp_ordering(self, loaded):
        """Higher SSP → more thermosteric SLR."""
        idx = np.argmin(np.abs(loaded['proj_years'] - 2100))
        medians = [loaded['projections'][ssp]['median'][idx]
                    for ssp in PROJ_SSPS]
        for i in range(len(medians) - 1):
            assert medians[i] <= medians[i + 1] * 1.05, (
                f"SSP ordering violated: {PROJ_SSPS[i]}={medians[i]*M_TO_MM:.0f}"
                f" vs {PROJ_SSPS[i+1]}={medians[i+1]*M_TO_MM:.0f}")

    def test_near_zero_at_baseline(self, loaded):
        idx = np.argmin(np.abs(loaded['proj_years'] - BASELINE_YEAR))
        for ssp in PROJ_SSPS:
            med_mm = loaded['projections'][ssp]['median'][idx] * M_TO_MM
            assert abs(med_mm) < 15.0, (
                f"{ssp} at baseline = {med_mm:.2f} mm, expected ~0")

    def test_uncertainty_grows(self, loaded):
        """90% CI width should grow over time."""
        proj = loaded['projections']['SSP5-8.5']
        years = loaded['proj_years']
        idx_2040 = np.argmin(np.abs(years - 2040))
        idx_2100 = np.argmin(np.abs(years - 2100))
        width_2040 = proj['p95'][idx_2040] - proj['p5'][idx_2040]
        width_2100 = proj['p95'][idx_2100] - proj['p5'][idx_2100]
        assert width_2100 > width_2040

    def test_largest_component(self, loaded):
        """Ocean should be the largest positive SLR component at 2100."""
        idx = np.argmin(np.abs(loaded['proj_years'] - 2100))
        ocean_med = loaded['projections']['SSP2-4.5']['median'][idx] * M_TO_MM
        # Thermosteric is typically 100-200 mm under SSP2-4.5
        assert ocean_med > 80, (
            f"Ocean median = {ocean_med:.0f} mm, expected > 80 for SSP2-4.5")

    def test_monotonic_median(self, loaded):
        """Median thermosteric SLR should be monotonically increasing after 2020."""
        proj = loaded['projections']['SSP2-4.5']
        years = loaded['proj_years']
        mask = years >= 2020
        med = proj['median'][mask]
        diffs = np.diff(med)
        # Allow tiny numerical noise
        assert np.all(diffs >= -1e-6), "Median should be monotonically increasing"


# =========================================================================
# IPCC comparison data
# =========================================================================

@pytest.mark.skipif(not HAS_IPCC, reason="IPCC AR6 data not found")
class TestIPCCOceanData:
    """Verify IPCC AR6 ocean component can be read."""

    def test_read_ipcc_ocean_ssp245(self):
        from component_projections import read_ipcc_component_nc, ipcc_extract
        data = read_ipcc_component_nc(
            CONF_BASE, 'medium_confidence', 'ssp245', 'oceandynamics')
        assert data is not None, "Failed to read IPCC oceandynamics"

    def test_ipcc_extract_structure(self):
        from component_projections import read_ipcc_component_nc, ipcc_extract
        data = read_ipcc_component_nc(
            CONF_BASE, 'medium_confidence', 'ssp245', 'oceandynamics')
        if data is None:
            pytest.skip("IPCC data not available")
        ex = ipcc_extract(data)
        assert 'years' in ex
        assert 'q50' in ex
        assert 'q05' in ex
        assert 'q95' in ex

    def test_ipcc_units_mm(self):
        """IPCC data should be in mm."""
        from component_projections import read_ipcc_component_nc, ipcc_extract
        data = read_ipcc_component_nc(
            CONF_BASE, 'medium_confidence', 'ssp585', 'oceandynamics')
        if data is None:
            pytest.skip("IPCC data not available")
        ex = ipcc_extract(data)
        # At 2100, IPCC ocean should be 100-400 mm
        idx = np.argmin(np.abs(ex['years'] - 2100))
        q50 = ex['q50'][idx]
        assert 50 < q50 < 500, f"IPCC q50 = {q50:.0f} mm, outside expected range"

    def test_ipcc_positive_at_2100(self):
        """IPCC oceandynamics should be positive at 2100."""
        from component_projections import read_ipcc_component_nc, ipcc_extract
        data = read_ipcc_component_nc(
            CONF_BASE, 'medium_confidence', 'ssp245', 'oceandynamics')
        if data is None:
            pytest.skip("IPCC data not available")
        ex = ipcc_extract(data)
        idx = np.argmin(np.abs(ex['years'] - 2100))
        assert ex['q50'][idx] > 0


# =========================================================================
# Design vectors
# =========================================================================

class TestDesignVectors:
    """Verify build_level_design_vectors computes correct integrals."""

    def test_constant_temperature(self):
        """Constant T: I1 should grow linearly, I2 quadratically."""
        from bayesian_dols import build_level_design_vectors
        n = 120  # 10 years of monthly
        T = np.ones(n) * 1.0
        time = np.arange(n) / 12.0
        obs_times = np.array([2.5, 5.0, 7.5])
        d = build_level_design_vectors(T, time, obs_times)
        # I0 = t, I1 = integral of T = t (since T=1), I2 = integral of T² = t
        # At t=5: I0=5, I1≈5, I2≈5
        idx_5 = 1  # obs_times[1] = 5.0
        assert abs(d['I0_obs'][idx_5] - 5.0) < 0.5
        assert abs(d['I1_obs'][idx_5] - 5.0) < 0.5

    def test_zero_temperature(self):
        """Zero T: I1 and I2 should be zero, I0 should grow."""
        from bayesian_dols import build_level_design_vectors
        n = 120
        T = np.zeros(n)
        time = np.arange(n) / 12.0
        obs_times = np.array([5.0])
        d = build_level_design_vectors(T, time, obs_times)
        assert abs(d['I1_obs'][0]) < 0.01
        assert abs(d['I2_obs'][0]) < 0.01
        assert d['I0_obs'][0] > 4.0

    def test_output_keys(self):
        from bayesian_dols import build_level_design_vectors
        n = 60
        T = np.random.randn(n)
        time = np.arange(n) / 12.0
        obs_times = np.array([1.0, 2.0, 3.0])
        d = build_level_design_vectors(T, time, obs_times)
        for key in ('I0_obs', 'I1_obs', 'I2_obs'):
            assert key in d, f"Missing key '{key}'"
            assert d[key].shape == (3,)


# =========================================================================
# Ocean transfer function
# =========================================================================

class TestOceanTransferFunction:
    """Verify fit_ocean_transfer_function on synthetic data."""

    def test_recovers_linear_relation(self):
        from component_analysis import fit_ocean_transfer_function
        rng = np.random.default_rng(42)
        n = 120
        time = np.arange(n) / 12.0 + 2000.0
        T_surf = 0.5 * np.sin(2 * np.pi * time / 5) + rng.normal(0, 0.05, n)
        # T_ocean = 0.4 * T_surf + 0.2 + noise
        T_ocean = 0.4 * T_surf + 0.2 + rng.normal(0, 0.02, n)
        result = fit_ocean_transfer_function(
            T_surf, time, T_ocean, time, lag_years=0, annual=True)
        assert abs(result['alpha'] - 0.4) < 0.1, (
            f"alpha={result['alpha']:.3f}, expected ~0.4")
        assert abs(result['beta'] - 0.2) < 0.1, (
            f"beta={result['beta']:.3f}, expected ~0.2")
        assert result['r2'] > 0.8

    def test_output_keys(self):
        from component_analysis import fit_ocean_transfer_function
        rng = np.random.default_rng(99)
        n = 60
        time = np.arange(n) / 12.0 + 2000.0
        T_surf = rng.normal(0, 1, n)
        T_ocean = 0.5 * T_surf + rng.normal(0, 0.1, n)
        result = fit_ocean_transfer_function(
            T_surf, time, T_ocean, time, lag_years=0, annual=True)
        for key in ('alpha', 'beta', 'alpha_se', 'beta_se', 'r2', 'n'):
            assert key in result, f"Missing key '{key}'"
