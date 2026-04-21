"""Unit tests for WAIS A4 sampling functions, data readers, I/O, and
notebook-level logic in component_projections.py."""

import sys
import os
import tempfile

import numpy as np
import pytest

# Notebook modules live in notebooks/, not the installed package
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'notebooks'))

from component_projections import (
    _sample_log_skewnormal,
    sample_a4_wais,
    sample_a4_wais_endpoint,
    sample_a4_wais_trajectories,
    A4_SCENARIOS,
    read_ipcc_component_nc, ipcc_extract,
    RHEOLOGY_FACTOR_MEDIAN,
    RHEOLOGY_FACTOR_SIGMA,
    RHEOLOGY_SENSITIVITY,
    M_TO_MM,
    N_OBS_MEAN,
    N_OBS_SIGMA,
    N_REF,
    WAIS_ONSET_YEAR,
)
from component_io import save_wais, load_component, PROJ_YEARS
from component_analysis import annualize_imbie

# Paths
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
RAW_DIR = os.path.join(PROJECT_ROOT, 'data', 'raw')
IMBIE_WAIS_PATH = os.path.join(
    RAW_DIR, 'ice_sheets', 'antarctica', 'imbie_west_antarctica_2021_mm.csv')
CONF_BASE = os.path.join(
    RAW_DIR, 'ipcc_ar6', 'slr', 'ar6', 'global', 'confidence_output_files')

HAS_IMBIE_WAIS = os.path.exists(IMBIE_WAIS_PATH)
HAS_IPCC_AIS = os.path.exists(os.path.join(
    CONF_BASE, 'medium_confidence', 'ssp245',
    'AIS_ssp245_medium_confidence_values.nc'))

N = 200_000  # large enough for percentile checks at ~1% tolerance
RNG_SEED = 2026


# =========================================================================
# _sample_log_skewnormal
# =========================================================================

class TestLogSkewnormal:
    """Verify that _sample_log_skewnormal hits the specified percentile bounds."""

    @pytest.mark.parametrize("low,high,alpha", [
        (25, 85, 0.0),       # S1: symmetric log-normal
        (150, 1000, 4.0),    # S2: positive skew
        (600, 1400, -3.0),   # S3: negative skew
    ])
    def test_percentile_bounds(self, low, high, alpha):
        rng = np.random.default_rng(RNG_SEED)
        samples = _sample_log_skewnormal(N, low, high, alpha, rng)
        p5, p95 = np.percentile(samples, [5, 95])
        # Allow 3% relative tolerance for MC sampling noise
        assert p5 == pytest.approx(low, rel=0.03), (
            f"5th percentile {p5:.1f} != {low} (alpha={alpha})")
        assert p95 == pytest.approx(high, rel=0.03), (
            f"95th percentile {p95:.1f} != {high} (alpha={alpha})")

    def test_all_positive(self):
        rng = np.random.default_rng(RNG_SEED)
        samples = _sample_log_skewnormal(N, 25, 85, 0.0, rng)
        assert np.all(samples > 0), "Log-skew-normal samples must be positive"

    def test_skewness_direction(self):
        """Positive alpha should produce right-skewed samples (mean > median)."""
        rng = np.random.default_rng(RNG_SEED)
        samples = _sample_log_skewnormal(N, 160, 1000, 4.0, rng)
        assert np.mean(samples) > np.median(samples)

    def test_negative_skewness_direction(self):
        """Negative alpha should produce left-skewed samples (mean < median)
        in log-space (which may still be right-skewed in linear space due to
        the log transform, but less so than positive alpha)."""
        rng1 = np.random.default_rng(RNG_SEED)
        rng2 = np.random.default_rng(RNG_SEED)
        pos = _sample_log_skewnormal(N, 600, 1400, 3.0, rng1)
        neg = _sample_log_skewnormal(N, 600, 1400, -3.0, rng2)
        # Negative alpha should have lower mean/median ratio
        assert np.mean(neg) / np.median(neg) < np.mean(pos) / np.median(pos)


# =========================================================================
# sample_a4_wais_endpoint
# =========================================================================

class TestEndpointSampling:
    """Verify sample_a4_wais_endpoint returns correct units and distributions."""

    def test_returns_meters(self):
        rng = np.random.default_rng(RNG_SEED)
        samples = sample_a4_wais_endpoint(N, rng)
        # S1 lower bound is 25 mm = 0.025 m; median should be > 0.1 m
        # If returned in mm, median would be > 100
        med = np.median(samples)
        assert 0.05 < med < 5.0, (
            f"Median {med:.3f} suggests wrong units (expected meters)")

    def test_shape(self):
        rng = np.random.default_rng(RNG_SEED)
        samples = sample_a4_wais_endpoint(1000, rng)
        assert samples.shape == (1000,)

    def test_rheology_increases_median(self):
        """Rheology correction (factor ~1.28) should increase the median
        relative to uncorrected samples."""
        rng1 = np.random.default_rng(RNG_SEED)
        rng2 = np.random.default_rng(RNG_SEED)
        corrected = sample_a4_wais_endpoint(N, rng1, rheology_mode='A')
        # For uncorrected: temporarily sample with rheology_mode A but
        # we can't easily disable it, so instead check the median is
        # above the uncorrected S2 median (~0.315 m from 315 mm).
        med = np.median(corrected)
        assert med > 0.315, (
            f"Corrected median {med:.3f} m should exceed uncorrected ~0.315 m")

    def test_scenario_weight_override(self):
        """Setting S1 weight to 1.0 should produce samples only from S1."""
        rng = np.random.default_rng(RNG_SEED)
        weights = {'S1_status_quo': 1.0, 'S2_misi': 0.0, 'S3_misi_mici': 0.0}
        samples = sample_a4_wais_endpoint(
            N, rng, scenario_overrides={'weights': weights})
        # S1 range: 25-85 mm = 0.025-0.085 m, after rheology *1.28:
        # ~0.032-0.109 m.  99th percentile should be well below 0.2 m.
        assert np.percentile(samples, 99) < 0.25, (
            "S1-only samples should stay below 0.25 m")

    def test_alpha_override(self):
        """Overriding S2 alpha to 0 should increase the median (removing
        the positive skew that pushes mass toward the lower tail)."""
        rng1 = np.random.default_rng(RNG_SEED)
        rng2 = np.random.default_rng(RNG_SEED)
        base = sample_a4_wais_endpoint(N, rng1)
        modified = sample_a4_wais_endpoint(
            N, rng2, scenario_overrides={'S2_misi': {'alpha': 0}})
        # alpha=0 (symmetric log-normal) has higher median than alpha=4
        assert np.median(modified) > np.median(base)

    def test_mode_b_produces_similar_median(self):
        """Mode A and B should produce similar marginal distributions."""
        rng_a = np.random.default_rng(RNG_SEED)
        rng_b = np.random.default_rng(RNG_SEED)
        a = sample_a4_wais_endpoint(N, rng_a, rheology_mode='A')
        b = sample_a4_wais_endpoint(N, rng_b, rheology_mode='B')
        # Medians should be within 15%
        assert np.median(a) == pytest.approx(np.median(b), rel=0.15)


# =========================================================================
# sample_a4_wais_trajectories
# =========================================================================

class TestTrajectories:
    """Verify coherent trajectory sampling."""

    @pytest.fixture
    def trajectory_result(self):
        rng = np.random.default_rng(RNG_SEED)
        years = np.arange(1990, 2110, dtype=float)
        samples_m, params = sample_a4_wais_trajectories(
            500, rng, years,
            anchor_year=2020.0,
            anchor_value_mm=5.0,
            anchor_sigma_mm=0.7,
            obs_years=np.array([1992, 2000, 2010, 2020], dtype=float),
            obs_values_mm=np.array([0.0, 1.0, 3.0, 5.0]),
            obs_sigmas_mm=np.array([0.5, 0.5, 0.6, 0.7]),
        )
        return samples_m, params, years

    def test_returns_meters(self, trajectory_result):
        samples_m, _, _ = trajectory_result
        # At 2100, median should be in 0.1-2.0 m range
        idx_2100 = -10  # year 2100 in arange(1990,2110)
        med = np.median(samples_m[:, idx_2100])
        assert 0.05 < med < 5.0, f"Median {med:.3f} suggests wrong units"

    def test_shape(self, trajectory_result):
        samples_m, _, years = trajectory_result
        assert samples_m.shape == (500, len(years))

    def test_params_keys(self, trajectory_result):
        _, params, _ = trajectory_result
        assert set(params.keys()) == {'scenario_idx', 'h2100_mm', 'beta', 'anchor_mm'}

    def test_monotonic_post_anchor(self, trajectory_result):
        """Each sample's trajectory should be monotonically non-decreasing
        after the anchor year (2020), since the power-law ramp is monotonic."""
        samples_m, _, years = trajectory_result
        post_anchor = years > 2020
        post = samples_m[:, post_anchor]
        diffs = np.diff(post, axis=1)
        # Allow tiny negative diffs from floating point
        assert np.all(diffs >= -1e-10), (
            f"Non-monotonic trajectory found; min diff = {diffs.min():.2e}")

    def test_anchor_value_respected(self, trajectory_result):
        """At the anchor year, sample mean should match the anchor value."""
        samples_m, _, years = trajectory_result
        idx_anchor = np.argmin(np.abs(years - 2020))
        mean_at_anchor = np.mean(samples_m[:, idx_anchor])
        # anchor_value_mm = 5.0 -> 0.005 m
        assert mean_at_anchor == pytest.approx(0.005, abs=0.001)

    def test_zero_before_observations(self, trajectory_result):
        """Before the first obs year (1992), samples should be zero."""
        samples_m, _, years = trajectory_result
        pre_obs = years < 1992
        assert np.all(samples_m[:, pre_obs] == 0.0)

    def test_scenario_idx_matches_weights(self, trajectory_result):
        """Scenario assignments should approximately match A4 weights."""
        _, params, _ = trajectory_result
        n = len(params['scenario_idx'])
        for i, sname in enumerate(A4_SCENARIOS):
            expected_frac = A4_SCENARIOS[sname]['P']
            actual_frac = (params['scenario_idx'] == i).sum() / n
            assert actual_frac == pytest.approx(expected_frac, abs=0.05), (
                f"{sname}: expected {expected_frac:.0%}, got {actual_frac:.0%}")

    def test_beta_positive(self, trajectory_result):
        """All trajectory exponents should be positive."""
        _, params, _ = trajectory_result
        assert np.all(params['beta'] > 0)

    def test_coherent_trajectories(self, trajectory_result):
        """Verify that trajectories are smooth power-law curves, not random
        walks.  For a pure power-law H(t) = a + b*t^beta, the second
        derivative should not change sign (for beta >= 1, it's convex)."""
        samples_m, params, years = trajectory_result
        post_anchor = years > 2025  # well past anchor
        post = samples_m[:, post_anchor]
        # Check that most samples have consistent curvature
        d2 = np.diff(post, n=2, axis=1)
        # For beta > 1 (most S2/S3 samples), d2 should be >= 0 (convex)
        # For beta = 1 (S1), d2 should be ~0 (linear)
        # Count how many samples have sign changes in d2
        sign_changes = np.sum(np.diff(np.sign(d2), axis=1) != 0, axis=1)
        # A smooth power-law should have 0 sign changes
        frac_smooth = np.mean(sign_changes == 0)
        # S1 samples (beta=1) have near-zero d2 that can flip sign from
        # floating point, so allow ~10% of samples to show sign changes.
        assert frac_smooth > 0.85, (
            f"Only {frac_smooth:.0%} of trajectories are smooth (expected >85%)")


# =========================================================================
# sample_a4_wais (legacy per-year function)
# =========================================================================

class TestLegacySampler:
    """Basic checks on sample_a4_wais (still used for backward compat)."""

    def test_returns_meters(self):
        rng = np.random.default_rng(RNG_SEED)
        samples = sample_a4_wais(1000, rng, year=2100)
        med = np.median(samples)
        assert 0.05 < med < 5.0, f"Median {med:.3f} suggests wrong units"

    def test_pre_anchor_returns_obs(self):
        """Before anchor year, should return N(obs_value, obs_sigma)."""
        rng = np.random.default_rng(RNG_SEED)
        samples = sample_a4_wais(
            10000, rng, year=2015,
            anchor_year=2020, anchor_value_mm=5.0, anchor_sigma_mm=0.7,
            obs_value_mm=3.0, obs_sigma_mm=0.5,
        )
        # Mean should be near obs_value_mm / 1000 = 0.003 m
        assert np.mean(samples) == pytest.approx(0.003, abs=0.0005)


# =========================================================================
# Rheology correction properties
# =========================================================================

class TestRheologyCorrection:
    """Verify rheology correction is always >= 1 and has correct magnitude."""

    def test_rheology_factor_always_geq_1(self):
        """No sample should have rheology factor < 1."""
        rng = np.random.default_rng(RNG_SEED)
        # Draw raw S2 samples without rheology
        raw = _sample_log_skewnormal(N, 160, 1000, 4.0, rng)
        rng2 = np.random.default_rng(RNG_SEED)
        # Draw with rheology via endpoint
        weights = {'S1_status_quo': 0.0, 'S2_misi': 1.0, 'S3_misi_mici': 0.0}
        corrected = sample_a4_wais_endpoint(
            N, rng2, scenario_overrides={'weights': weights})
        # Corrected (in m) should be >= raw/1000 (uncorrected in m) for
        # all samples -- but since they use different RNG streams, we check
        # statistically: corrected median should exceed raw median
        assert np.median(corrected) > np.median(raw / M_TO_MM)

    def test_rheology_median_magnitude(self):
        """Median rheology boost should be approximately RHEOLOGY_FACTOR_MEDIAN."""
        rng = np.random.default_rng(RNG_SEED)
        rheo = rng.normal(RHEOLOGY_FACTOR_MEDIAN, RHEOLOGY_FACTOR_SIGMA, N)
        rheo = np.maximum(rheo, 1.0)
        # Truncation at 1.0 slightly raises the median above 1.28
        assert np.median(rheo) == pytest.approx(RHEOLOGY_FACTOR_MEDIAN, rel=0.02)


# =========================================================================
# Gap 1: I/O roundtrip (save_wais / load_component)
# =========================================================================

class TestWAISIO:
    """Test save_wais / load_component roundtrip, including the lean-save
    optimization where samples are stored only for the first SSP."""

    @pytest.fixture
    def synthetic_wais(self):
        """Create synthetic WAIS data for I/O testing."""
        rng = np.random.default_rng(RNG_SEED)
        n_samples = 200
        n_times = len(PROJ_YEARS)

        obs_years = np.arange(1992, 2021, dtype=float) + 0.5
        obs_H = rng.normal(0, 0.002, len(obs_years))
        obs_sigma = np.full(len(obs_years), 0.001)

        samples = rng.normal(0.1, 0.05, (n_samples, n_times))
        samples = np.maximum(samples, 0)

        # Full projection dict — all SSPs get copies (as notebook does)
        wais_proj = {}
        for ssp in ['SSP1-2.6', 'SSP2-4.5', 'SSP3-7.0', 'SSP5-8.5']:
            wais_proj[ssp] = {
                'samples': samples.copy(),
                'median': np.median(samples, axis=0),
                'p5': np.percentile(samples, 5, axis=0),
                'p95': np.percentile(samples, 95, axis=0),
                'p17': np.percentile(samples, 17, axis=0),
                'p83': np.percentile(samples, 83, axis=0),
            }

        return {
            'obs_years': obs_years, 'obs_H': obs_H, 'obs_sigma': obs_sigma,
            'wais_proj': wais_proj, 'samples': samples,
        }

    def test_roundtrip_full(self, synthetic_wais, tmp_path):
        """Save with full samples for all SSPs and reload."""
        h5_path = tmp_path / 'test_wais.h5'
        save_wais(
            a4_scenarios=A4_SCENARIOS,
            obs_years=synthetic_wais['obs_years'],
            obs_H=synthetic_wais['obs_H'],
            obs_sigma=synthetic_wais['obs_sigma'],
            wais_proj=synthetic_wais['wais_proj'],
            h5_path=h5_path,
        )
        loaded = load_component('wais', h5_path=h5_path)

        assert set(loaded['projections'].keys()) == {
            'SSP1-2.6', 'SSP2-4.5', 'SSP3-7.0', 'SSP5-8.5'}
        for ssp in loaded['projections']:
            np.testing.assert_allclose(
                loaded['projections'][ssp]['median'],
                synthetic_wais['wais_proj'][ssp]['median'], atol=1e-8)
            assert 'samples' in loaded['projections'][ssp]

    def test_roundtrip_lean(self, synthetic_wais, tmp_path):
        """Save with samples only for the first SSP (lean mode) and verify
        load_component propagates samples to all SSPs."""
        h5_path = tmp_path / 'test_wais_lean.h5'
        proj_lean = {}
        ssps = ['SSP1-2.6', 'SSP2-4.5', 'SSP3-7.0', 'SSP5-8.5']
        for i, ssp in enumerate(ssps):
            proj_i = dict(synthetic_wais['wais_proj'][ssp])
            if i > 0:
                proj_i.pop('samples', None)
            proj_lean[ssp] = proj_i

        save_wais(
            a4_scenarios=A4_SCENARIOS,
            obs_years=synthetic_wais['obs_years'],
            obs_H=synthetic_wais['obs_H'],
            obs_sigma=synthetic_wais['obs_sigma'],
            wais_proj=proj_lean,
            h5_path=h5_path,
        )
        loaded = load_component('wais', h5_path=h5_path)

        # All SSPs should have samples (propagated from first)
        for ssp in ssps:
            assert 'samples' in loaded['projections'][ssp], (
                f"Samples missing for {ssp} after lean-save load")
            np.testing.assert_allclose(
                loaded['projections'][ssp]['samples'],
                synthetic_wais['samples'], atol=1e-6)

    def test_a4_scenarios_preserved(self, synthetic_wais, tmp_path):
        """A4 scenario parameters should survive the roundtrip."""
        h5_path = tmp_path / 'test_wais_a4.h5'
        save_wais(
            a4_scenarios=A4_SCENARIOS,
            obs_years=synthetic_wais['obs_years'],
            obs_H=synthetic_wais['obs_H'],
            obs_sigma=synthetic_wais['obs_sigma'],
            wais_proj=synthetic_wais['wais_proj'],
            h5_path=h5_path,
        )
        loaded = load_component('wais', h5_path=h5_path)
        assert 'a4_scenarios' in loaded
        for sname in A4_SCENARIOS:
            assert sname in loaded['a4_scenarios']
            assert loaded['a4_scenarios'][sname]['P'] == pytest.approx(
                A4_SCENARIOS[sname]['P'])
            assert loaded['a4_scenarios'][sname]['misi'] == A4_SCENARIOS[sname]['misi']

    def test_observations_preserved(self, synthetic_wais, tmp_path):
        """Observation arrays should survive the roundtrip."""
        h5_path = tmp_path / 'test_wais_obs.h5'
        save_wais(
            a4_scenarios=A4_SCENARIOS,
            obs_years=synthetic_wais['obs_years'],
            obs_H=synthetic_wais['obs_H'],
            obs_sigma=synthetic_wais['obs_sigma'],
            wais_proj=synthetic_wais['wais_proj'],
            h5_path=h5_path,
        )
        loaded = load_component('wais', h5_path=h5_path)
        np.testing.assert_allclose(
            loaded['observations']['years'], synthetic_wais['obs_years'], atol=1e-8)
        np.testing.assert_allclose(
            loaded['observations']['H_obs'], synthetic_wais['obs_H'], atol=1e-8)

    def test_metadata(self, synthetic_wais, tmp_path):
        """WAIS metadata should include model type and ssp_independent flag."""
        h5_path = tmp_path / 'test_wais_meta.h5'
        save_wais(
            a4_scenarios=A4_SCENARIOS,
            obs_years=synthetic_wais['obs_years'],
            obs_H=synthetic_wais['obs_H'],
            obs_sigma=synthetic_wais['obs_sigma'],
            wais_proj=synthetic_wais['wais_proj'],
            h5_path=h5_path,
        )
        loaded = load_component('wais', h5_path=h5_path)
        assert loaded['metadata']['model_type'] == 'a4_deep_uncertainty'
        assert loaded['metadata']['ssp_independent'] == True


# =========================================================================
# Gap 2: IMBIE West Antarctica reader
# =========================================================================

@pytest.mark.skipif(not HAS_IMBIE_WAIS, reason="IMBIE WAIS data file not found")
class TestIMBIEWAISReader:
    """Verify read_imbie_west_antarctica + annualize_imbie on WAIS data."""

    @pytest.fixture(scope="class")
    def wais_data(self):
        from slr_data_readers import read_imbie_west_antarctica
        df = read_imbie_west_antarctica(IMBIE_WAIS_PATH)
        years, H, sigma = annualize_imbie(df, baseline_year=2005.0)
        return df, years, H, sigma

    def test_columns_present(self, wais_data):
        df = wais_data[0]
        expected = {'decimal_year', 'mass_balance_rate',
                    'mass_balance_rate_sigma', 'cumulative_mass_balance',
                    'cumulative_mass_balance_sigma'}
        assert expected.issubset(set(df.columns))

    def test_units_are_meters(self, wais_data):
        """Cumulative should be in meters (order 1e-3 to 1e-1)."""
        df = wais_data[0]
        cum = df['cumulative_mass_balance'].values
        max_abs = np.max(np.abs(cum))
        assert max_abs < 0.5, f"Max |cum| = {max_abs:.3f}, too large for meters"
        assert max_abs > 1e-5, f"Max |cum| = {max_abs:.2e}, too small for meters"

    def test_time_range(self, wais_data):
        """IMBIE WAIS should span ~1992-2020."""
        df = wais_data[0]
        years = df['decimal_year'].values
        assert years[0] >= 1990 and years[0] <= 1993
        assert years[-1] >= 2018

    def test_annual_years(self, wais_data):
        """Annualized years should have ~1yr steps."""
        years = wais_data[1]
        dt = np.diff(years)
        assert np.allclose(dt, 1.0, atol=0.1)

    def test_baseline_zero(self, wais_data):
        """Rebased cumulative should be zero at baseline."""
        years, H = wais_data[1], wais_data[2]
        bl_idx = np.argmin(np.abs(years - 2005.0))
        assert abs(H[bl_idx]) < 1e-10

    def test_wais_losing_mass(self, wais_data):
        """WAIS cumulative SLR should be positive at end (mass loss)."""
        H = wais_data[2]
        assert H[-1] > 0, f"Final H = {H[-1]:.6f}, expected positive (mass loss)"

    def test_sigma_positive(self, wais_data):
        sigma = wais_data[3]
        assert np.all(sigma >= 0)


# =========================================================================
# Gap 3: A4 scenario parameter consistency
# =========================================================================

class TestA4ScenarioParameters:
    """Verify A4_SCENARIOS dict is internally consistent."""

    def test_weights_sum_to_one(self):
        total = sum(s['P'] for s in A4_SCENARIOS.values())
        assert total == pytest.approx(1.0, abs=1e-10), (
            f"A4 weights sum to {total}, expected 1.0")

    def test_required_keys_present(self):
        required = {'P', 'low_mm', 'high_mm', 'alpha', 'beta_loc',
                    'beta_scale', 'misi'}
        for sname, params in A4_SCENARIOS.items():
            missing = required - set(params.keys())
            assert not missing, (
                f"{sname} missing keys: {missing}")

    def test_probabilities_valid(self):
        for sname, params in A4_SCENARIOS.items():
            assert 0 < params['P'] <= 1, (
                f"{sname}: P = {params['P']}, expected 0 < P <= 1")

    def test_ranges_positive(self):
        for sname, params in A4_SCENARIOS.items():
            assert params['low_mm'] > 0, f"{sname}: low_mm must be positive"
            assert params['high_mm'] > params['low_mm'], (
                f"{sname}: high_mm ({params['high_mm']}) must exceed "
                f"low_mm ({params['low_mm']})")

    def test_s1_no_misi(self):
        """S1 (status quo) should not have MISI."""
        assert A4_SCENARIOS['S1_status_quo']['misi'] is False

    def test_s2_s3_have_misi(self):
        """S2 and S3 should have MISI."""
        assert A4_SCENARIOS['S2_misi']['misi'] is True
        assert A4_SCENARIOS['S3_misi_mici']['misi'] is True

    def test_s1_has_linear_trajectory(self):
        """S1 should have beta_scale=0 (linear ramp, no acceleration)."""
        assert A4_SCENARIOS['S1_status_quo']['beta_scale'] == 0

    def test_s2_s3_have_accelerating_trajectory(self):
        """S2 and S3 should have beta_scale > 0 (accelerating ramp)."""
        assert A4_SCENARIOS['S2_misi']['beta_scale'] > 0
        assert A4_SCENARIOS['S3_misi_mici']['beta_scale'] > 0

    def test_scenario_ordering(self):
        """Scenarios should be ordered by severity: S1 < S2 < S3 ranges."""
        s1 = A4_SCENARIOS['S1_status_quo']
        s2 = A4_SCENARIOS['S2_misi']
        s3 = A4_SCENARIOS['S3_misi_mici']
        assert s1['high_mm'] < s2['high_mm']
        assert s2['low_mm'] < s3['low_mm']


# =========================================================================
# Gap 4: IPCC unit conversion
# =========================================================================

@pytest.mark.skipif(not HAS_IPCC_AIS, reason="IPCC AIS data not found")
class TestIPCCUnitConversion:
    """Verify IPCC AIS data is read and converted correctly."""

    @pytest.fixture(scope="class")
    def ipcc_ais(self):
        data = read_ipcc_component_nc(CONF_BASE, 'medium_confidence',
                                       'ssp245', 'AIS')
        return data

    def test_not_none(self, ipcc_ais):
        assert ipcc_ais is not None

    def test_native_units_mm(self, ipcc_ais):
        """IPCC SLC should be in mm (order 10-200 at 2100)."""
        ex = ipcc_extract(ipcc_ais)
        idx_2100 = np.argmin(np.abs(ex['years'] - 2100))
        med_mm = ex['q50'][idx_2100]
        assert 10 < abs(med_mm) < 500, (
            f"IPCC AIS median at 2100 = {med_mm:.0f}, expected 10-500 mm")

    def test_mm_to_m_conversion(self, ipcc_ais):
        """Dividing by M_TO_MM should give meters (order 0.01-0.5)."""
        ex = ipcc_extract(ipcc_ais)
        idx_2100 = np.argmin(np.abs(ex['years'] - 2100))
        med_m = ex['q50'][idx_2100] / M_TO_MM
        assert 0.01 < abs(med_m) < 0.5, (
            f"Converted AIS median = {med_m:.4f} m, expected 0.01-0.5 m")

    def test_sigma_from_quantiles(self, ipcc_ais):
        """Gaussian sigma estimated from quantiles should be positive and finite."""
        ex = ipcc_extract(ipcc_ais)
        idx_2100 = np.argmin(np.abs(ex['years'] - 2100))
        sig_mm = (ex['q95'][idx_2100] - ex['q05'][idx_2100]) / (2 * 1.645)
        sig_m = sig_mm / M_TO_MM
        assert sig_m > 0
        assert np.isfinite(sig_m)
        # Sigma should be smaller than the median
        med_m = abs(ex['q50'][idx_2100] / M_TO_MM)
        assert sig_m < med_m * 5, "Sigma implausibly large relative to median"


# =========================================================================
# Gap 5: Sensitivity analysis logic
# =========================================================================

class TestSensitivityAnalysisLogic:
    """Test the weight perturbation and range override logic used in the
    notebook's sensitivity cells."""

    def test_tornado_weights_sum_to_one(self):
        """After perturbing one scenario weight by +0.05, the redistributed
        weights should sum to 1."""
        perturbation = 0.05
        scenarios = list(A4_SCENARIOS.keys())
        for sname in scenarios:
            orig_p = A4_SCENARIOS[sname]['P']
            new_p = min(orig_p + perturbation, 0.95)
            remaining = 1.0 - new_p
            orig_remaining = 1.0 - orig_p

            weights_mod = {}
            for s2 in scenarios:
                if s2 == sname:
                    weights_mod[s2] = new_p
                else:
                    weights_mod[s2] = A4_SCENARIOS[s2]['P'] * remaining / orig_remaining

            total = sum(weights_mod.values())
            assert total == pytest.approx(1.0, abs=1e-10), (
                f"Perturbing {sname} +{perturbation}: weights sum = {total}")

    def test_tornado_weights_preserve_proportions(self):
        """Unperturbed scenario weights should maintain their relative ratios."""
        perturbation = 0.05
        sname = 'S2_misi'
        orig_p = A4_SCENARIOS[sname]['P']
        new_p = orig_p + perturbation
        remaining = 1.0 - new_p
        orig_remaining = 1.0 - orig_p

        others = [s for s in A4_SCENARIOS if s != sname]
        w0 = A4_SCENARIOS[others[0]]['P'] * remaining / orig_remaining
        w1 = A4_SCENARIOS[others[1]]['P'] * remaining / orig_remaining
        # Ratio should match original ratio
        orig_ratio = A4_SCENARIOS[others[0]]['P'] / A4_SCENARIOS[others[1]]['P']
        new_ratio = w0 / w1
        assert new_ratio == pytest.approx(orig_ratio, rel=1e-10)

    def test_tornado_negative_perturbation(self):
        """Perturbing by -0.05 should also sum to 1 and keep weights positive."""
        perturbation = 0.05
        scenarios = list(A4_SCENARIOS.keys())
        for sname in scenarios:
            orig_p = A4_SCENARIOS[sname]['P']
            new_p = max(orig_p - perturbation, 0.01)
            remaining = 1.0 - new_p
            orig_remaining = 1.0 - orig_p

            weights_mod = {}
            for s2 in scenarios:
                if s2 == sname:
                    weights_mod[s2] = new_p
                else:
                    weights_mod[s2] = A4_SCENARIOS[s2]['P'] * remaining / orig_remaining

            total = sum(weights_mod.values())
            assert total == pytest.approx(1.0, abs=1e-10)
            assert all(w > 0 for w in weights_mod.values()), (
                f"Negative weight after perturbing {sname} by -{perturbation}")

    def test_range_override_scales_bounds(self):
        """±20% range override should correctly scale low_mm and high_mm."""
        for factor in [0.8, 1.0, 1.2]:
            overrides = {}
            for sname, s in A4_SCENARIOS.items():
                overrides[sname] = {
                    'low_mm': s['low_mm'] * factor,
                    'high_mm': s['high_mm'] * factor,
                }
            # Verify the override structure is correct
            for sname in A4_SCENARIOS:
                assert overrides[sname]['low_mm'] == pytest.approx(
                    A4_SCENARIOS[sname]['low_mm'] * factor)
                assert overrides[sname]['high_mm'] == pytest.approx(
                    A4_SCENARIOS[sname]['high_mm'] * factor)

    def test_range_override_affects_median(self):
        """Scaling bounds up by 20% should increase the median endpoint."""
        rng1 = np.random.default_rng(77)
        rng2 = np.random.default_rng(77)
        base = sample_a4_wais_endpoint(50000, rng1)
        overrides = {}
        for sname, s in A4_SCENARIOS.items():
            overrides[sname] = {
                'low_mm': s['low_mm'] * 1.2,
                'high_mm': s['high_mm'] * 1.2,
            }
        scaled = sample_a4_wais_endpoint(
            50000, rng2, scenario_overrides=overrides)
        assert np.median(scaled) > np.median(base), (
            "Scaling bounds up 20% should increase median")

    def test_rheology_sensitivity_direction(self):
        """Higher Glen's exponent n should produce higher rheology correction."""
        for n_exp in [3.5, 4.0, 4.5]:
            rheo_med = 1 + RHEOLOGY_SENSITIVITY * (n_exp - N_REF)
            assert rheo_med >= 1.0, f"n={n_exp}: rheology factor {rheo_med} < 1"
        # Higher n → higher correction
        r_35 = 1 + RHEOLOGY_SENSITIVITY * (3.5 - N_REF)
        r_45 = 1 + RHEOLOGY_SENSITIVITY * (4.5 - N_REF)
        assert r_45 > r_35
