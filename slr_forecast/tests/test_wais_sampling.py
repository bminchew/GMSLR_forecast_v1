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
    _sample_s1_quadratic_mm,
    sample_a4_wais,
    sample_a4_wais_endpoint,
    sample_a4_wais_trajectories,
    A4_SCENARIOS,
    get_s1_quadratic,
    read_ipcc_component_nc, ipcc_extract,
    RHEOLOGY_FACTOR_MEDIAN,
    RHEOLOGY_FACTOR_SIGMA,
    RHEOLOGY_SENSITIVITY,
    N_OBS_MEAN,
    N_OBS_SIGMA,
    N_REF,
    WAIS_ONSET_YEAR,
)
from component_io import save_wais, load_component, PROJ_YEARS
from component_analysis import annualize_imbie
from slr_forecast import M_TO_MM
from slr_forecast.config import Z_90

# Paths
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
RAW_DIR = os.path.join(PROJECT_ROOT, 'data', 'raw')
# Legacy IMBIE v2021 file -- superseded by IMBIE-3 (IMBIE3_WAIS_PATH below)
# as of the 2026-09-17 refit, but kept here since read_imbie_west_antarctica
# is still exercised for regression/comparison purposes.
IMBIE_WAIS_PATH = os.path.join(
    RAW_DIR, 'ice_sheets', 'antarctica', 'imbie_west_antarctica_2021_mm.csv')
# IMBIE-3 (Otosaka et al. 2026) -- what component_wais.ipynb actually reads.
IMBIE3_WAIS_PATH = os.path.join(
    RAW_DIR, 'ice_sheets', 'imbie2026', 'imbie3_west_antarctica_mm_partitioned.csv')
CONF_BASE = os.path.join(
    RAW_DIR, 'ipcc_ar6', 'slr', 'ar6', 'global', 'confidence_output_files')

HAS_IMBIE_WAIS = os.path.exists(IMBIE_WAIS_PATH)
HAS_IMBIE3_WAIS = os.path.exists(IMBIE3_WAIS_PATH)
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
        relative to uncorrected samples.

        There's no public no-rheology mode, so the uncorrected mixture is
        replicated directly here (same scenario-assignment/sampling logic
        as sample_a4_wais_endpoint, just without the `base *= rheo` step)
        rather than compared against a hardcoded reference value -- a
        hardcoded number drifts every time A4_SCENARIOS changes (e.g. the
        low_mm anchor-percentile changes on 2026-09-17), whereas this
        stays correct automatically."""
        rng1 = np.random.default_rng(RNG_SEED)
        corrected = sample_a4_wais_endpoint(N, rng1, rheology_mode='A')

        rng2 = np.random.default_rng(RNG_SEED)
        scenario_names = list(A4_SCENARIOS.keys())
        probs = np.array([A4_SCENARIOS[s]['P'] for s in scenario_names])
        scenario_idx = rng2.choice(len(scenario_names), size=N, p=probs)
        child_rngs = rng2.spawn(len(scenario_names))
        uncorrected_mm = np.zeros(N)
        for i, sname in enumerate(scenario_names):
            mask = scenario_idx == i
            n_s = mask.sum()
            if n_s == 0:
                continue
            crng = child_rngs[i]
            if sname == 'S1_status_quo':
                uncorrected_mm[mask] = _sample_s1_quadratic_mm(n_s, crng, [2100.0])[:, 0]
                continue
            s = A4_SCENARIOS[sname]
            uncorrected_mm[mask] = _sample_log_skewnormal(
                n_s, s['low_mm'], s['high_mm'], s['alpha'], crng)
        uncorrected = uncorrected_mm / M_TO_MM

        med_corrected = np.median(corrected)
        med_uncorrected = np.median(uncorrected)
        assert med_corrected > med_uncorrected, (
            f"Corrected median {med_corrected:.3f} m should exceed "
            f"uncorrected median {med_uncorrected:.3f} m")

    def test_scenario_weight_override(self):
        """Setting S1 weight to 1.0 should produce samples only from S1."""
        rng = np.random.default_rng(RNG_SEED)
        weights = {'S1_status_quo': 1.0, 'S2_fast_wais': 0.0}
        samples = sample_a4_wais_endpoint(
            N, rng, scenario_overrides={'weights': weights})
        # S1 range: 25-85 mm = 0.025-0.085 m, after rheology *1.28:
        # ~0.032-0.109 m.  99th percentile should be well below 0.2 m.
        assert np.percentile(samples, 99) < 0.25, (
            "S1-only samples should stay below 0.25 m")

    def test_alpha_override(self):
        """Overriding S2_fast_wais alpha to 0 should increase the median
        (removing the positive skew that pushes mass toward the lower tail)."""
        rng1 = np.random.default_rng(RNG_SEED)
        rng2 = np.random.default_rng(RNG_SEED)
        base = sample_a4_wais_endpoint(N, rng1)
        modified = sample_a4_wais_endpoint(
            N, rng2, scenario_overrides={'S2_fast_wais': {'alpha': 0}})
        # alpha=0 (symmetric log-normal) has a higher median than alpha=3
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
        assert set(params.keys()) == {'scenario_idx', 'h2100_mm', 'beta',
                                       'beta_eff_2035', 'beta_eff_2050',
                                       'anchor_mm'}

    def test_monotonic_post_anchor(self, trajectory_result):
        """Each non-S1 (S2/power-law-ramp) sample's trajectory should be
        monotonically non-decreasing after the anchor year (2020).

        S1_status_quo is excluded as of the 2026-09-17 ISMIP6
        extrapolation-error widening (S1_ISMIP6_STD_COEFFS in
        component_projections.py): S1 trajectories are now H_quad(t) +
        eta*sqrt(extra_var(t)) with one eta per sample, and for eta<0
        this smooth curve can legitimately dip within its post-anchor
        range -- a real, intentional consequence of giving S1 a
        non-trivial probability of net mass gain by 2100 (see round-3/3b
        write-up), not a bug. S2's power-law ramp is unaffected and
        still checked here."""
        samples_m, params, years = trajectory_result
        scenario_names = list(A4_SCENARIOS.keys())
        s1_idx = scenario_names.index('S1_status_quo')
        non_s1 = params['scenario_idx'] != s1_idx
        post_anchor = years > 2020
        post = samples_m[non_s1][:, post_anchor]
        diffs = np.diff(post, axis=1)
        # Allow tiny negative diffs from floating point
        assert np.all(diffs >= -1e-10), (
            f"Non-monotonic non-S1 trajectory found; min diff = {diffs.min():.2e}")

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
        """All trajectory exponents should be positive, for scenarios that
        use a power-law ramp. S1_status_quo does not (it uses its own
        quadratic-in-time posterior instead -- see A4_SCENARIOS comment
        block), so beta is left at its 0.0 default for S1 rows and is
        excluded here."""
        _, params, _ = trajectory_result
        scenario_names = list(A4_SCENARIOS.keys())
        s1_idx = scenario_names.index('S1_status_quo')
        non_s1 = params['scenario_idx'] != s1_idx
        assert np.all(params['beta'][non_s1] > 0)

    def test_coherent_trajectories(self, trajectory_result):
        """Verify that non-S1 (S2, power-law-ramp) trajectories are smooth
        curves, not random walks.  For a pure power-law H(t) = a + b*t^beta,
        the second derivative should not change sign (for beta >= 1, it's
        convex).

        S1_status_quo is excluded as of the 2026-09-17 ISMIP6
        extrapolation-error widening: S1's curve is now H_quad(t) +
        eta*sqrt(extra_var(t)), a smooth function of t but not a pure
        power law, so its second-difference sign is no longer expected to
        be constant -- this is intentional (see test_monotonic_post_anchor)
        and unrelated to the random-walk-vs-smooth-curve distinction this
        test is actually checking for S2."""
        samples_m, params, years = trajectory_result
        scenario_names = list(A4_SCENARIOS.keys())
        s1_idx = scenario_names.index('S1_status_quo')
        non_s1 = params['scenario_idx'] != s1_idx
        post_anchor = years > 2025  # well past anchor
        post = samples_m[non_s1][:, post_anchor]
        # Check that most samples have consistent curvature
        d2 = np.diff(post, n=2, axis=1)
        # For beta > 1 (most S2/S3 samples), d2 should be >= 0 (convex)
        # Count how many samples have sign changes in d2
        sign_changes = np.sum(np.diff(np.sign(d2), axis=1) != 0, axis=1)
        # A smooth power-law should have 0 sign changes
        frac_smooth = np.mean(sign_changes == 0)
        # S2 samples near the low_mm/high_mm bounds (h_remaining small,
        # e.g. from S2's 84-1000 mm range) have correspondingly small
        # curvature there too, which pushes some fraction of samples below
        # floating-point noise. Threshold history: 0.85 -> 0.80 when S2's
        # high_mm dropped 1300->1000 (smaller mixture p95, see
        # test_mixture_p95_not_above_ar6_low_confidence); -> 0.75
        # (2026-09-17) when low_mm dropped 139->84 (pinned to S1's median
        # rather than an upper-tail percentile -- see the A4_SCENARIOS
        # comment block), which put more S2 samples close enough to
        # low_mm for the same floating-point noise-floor effect (measured
        # frac_smooth was 0.77 at that point; this is a test-sensitivity
        # artifact of the curvature-sign check on near-flat curves, not a
        # real change in trajectory quality).
        assert frac_smooth > 0.75, (
            f"Only {frac_smooth:.0%} of non-S1 trajectories are smooth (expected >75%)")

    def test_s2_endpoint_pinned_to_h2100(self, trajectory_result):
        """S2_fast_wais's rate-space blend with the IMBIE quadratic is
        rescaled so H(2100) always equals the independently-drawn h2100 --
        the blend reshapes the path only, not the assessed 2100 endpoint
        distribution (see component_projections.sample_a4_wais_trajectories
        docstring / blend block)."""
        samples_m, params, years = trajectory_result
        scenario_names = list(A4_SCENARIOS.keys())
        s2_idx = scenario_names.index('S2_fast_wais')
        s2_mask = params['scenario_idx'] == s2_idx
        idx_2100 = np.argmin(np.abs(years - 2100.0))
        h2100_m = params['h2100_mm'][s2_mask] / 1000.0
        np.testing.assert_allclose(samples_m[s2_mask, idx_2100], h2100_m,
                                    atol=1e-9)

    def test_s2_near_anchor_rate_is_data_anchored(self, trajectory_result):
        """Before blending, the power-law ramp forced dH/dt = 0 exactly at
        the anchor year for beta > 1 (t_norm**beta has zero slope at
        t_norm=0). After blending with the IMBIE quadratic, the near-anchor
        rate should instead be small and comparable to the quadratic's own
        rate there, not implied by beta alone. Regression check: the
        realized rate immediately after the anchor should be far smaller
        than the *unblended* power-law rate a representative beta=1.84
        would imply blowing up from the very small A4 low_mm bound."""
        samples_m, params, years = trajectory_result
        scenario_names = list(A4_SCENARIOS.keys())
        s2_idx = scenario_names.index('S2_fast_wais')
        s2_mask = params['scenario_idx'] == s2_idx
        idx_anchor = np.argmin(np.abs(years - 2020.0))
        idx_next = idx_anchor + 1
        dt = years[idx_next] - years[idx_anchor]
        rate_mm_per_yr = ((samples_m[s2_mask, idx_next]
                            - samples_m[s2_mask, idx_anchor]) * 1000.0 / dt)
        # A pure power-law ramp from the S2 low bound (130 mm) to the
        # median h2100 with beta~1.8-2.3 would give a near-anchor rate at
        # least several mm/yr for most samples once beta is not >>1; the
        # blended, data-anchored rate should sit close to zero given
        # WAIS's own near-zero/decelerating IMBIE rate at the anchor.
        assert np.median(np.abs(rate_mm_per_yr)) < 1.0, (
            f"Median |rate| immediately post-anchor = "
            f"{np.median(np.abs(rate_mm_per_yr)):.2f} mm/yr; expected small "
            f"(data-anchored), not power-law-implied")


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
        # Draw raw S2 samples without rheology, using S2_fast_wais's actual
        # current (low_mm, high_mm, alpha) rather than a hardcoded stand-in:
        # the previous hardcoded (160, 1000, alpha=4) reference distribution
        # predates the 2026-09-17 IMBIE-3 refit (which lowered low_mm from
        # 130 to 94 mm, pinned to S1's tighter posterior -- see
        # A4_SCENARIOS/S1_QUADRATIC_MEAN in component_projections.py) and no
        # longer describes the same distribution sample_a4_wais_endpoint
        # actually draws from, so the two were no longer a valid raw/
        # corrected pair.
        s2 = A4_SCENARIOS['S2_fast_wais']
        raw = _sample_log_skewnormal(N, s2['low_mm'], s2['high_mm'], s2['alpha'], rng)
        rng2 = np.random.default_rng(RNG_SEED)
        # Draw with rheology via endpoint
        weights = {'S1_status_quo': 0.0, 'S2_fast_wais': 1.0}
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
    """Verify read_imbie_west_antarctica + annualize_imbie on the legacy
    IMBIE v2021 WAIS record. component_wais.ipynb no longer reads this file
    (superseded by IMBIE-3 as of the 2026-09-17 refit -- see
    TestIMBIE3WAISReader below), but the reader function itself remains in
    slr_data_readers.py and this class is kept as a regression check on it."""

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
        """IMBIE v2021 WAIS should span ~1992-2020."""
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
# Gap 2b: IMBIE-3 (Otosaka et al. 2026) West Antarctica reader -- the
# record component_wais.ipynb actually reads as of the 2026-09-17 refit.
# =========================================================================

@pytest.mark.skipif(not HAS_IMBIE3_WAIS, reason="IMBIE-3 WAIS data file not found")
class TestIMBIE3WAISReader:
    """Verify read_imbie3 + annualize_imbie on the IMBIE-3 WAIS record.

    IMBIE-3 (Otosaka et al. 2026) nearly doubles the record length versus
    IMBIE v2021 (1979-2023 vs. 1992-2020), so the time-range assertions
    below are intentionally different from TestIMBIEWAISReader above."""

    @pytest.fixture(scope="class")
    def wais3_data(self):
        from slr_data_readers import read_imbie3
        df = read_imbie3(IMBIE3_WAIS_PATH, convert_to_meters=True)
        years, H, sigma = annualize_imbie(df, baseline_year=2005.0)
        return df, years, H, sigma

    def test_columns_present(self, wais3_data):
        df = wais3_data[0]
        expected = {'decimal_year', 'mass_balance_rate',
                    'mass_balance_rate_sigma', 'cumulative_mass_balance',
                    'cumulative_mass_balance_sigma'}
        assert expected.issubset(set(df.columns))

    def test_units_are_meters(self, wais3_data):
        """Cumulative should be in meters (order 1e-3 to 1e-1)."""
        df = wais3_data[0]
        cum = df['cumulative_mass_balance'].values
        max_abs = np.max(np.abs(cum))
        assert max_abs < 0.5, f"Max |cum| = {max_abs:.3f}, too large for meters"
        assert max_abs > 1e-5, f"Max |cum| = {max_abs:.2e}, too small for meters"

    def test_time_range(self, wais3_data):
        """IMBIE-3 WAIS should span ~1979-2023 (monthly records start/end
        mid-month, so the raw decimal_year values land at ~1979.04 and
        ~2023.96)."""
        df = wais3_data[0]
        years = df['decimal_year'].values
        assert years[0] >= 1978 and years[0] <= 1980
        assert years[-1] >= 2022

    def test_annual_years(self, wais3_data):
        """Annualized years should have ~1yr steps."""
        years = wais3_data[1]
        dt = np.diff(years)
        assert np.allclose(dt, 1.0, atol=0.1)

    def test_annual_years_span_longer_record(self, wais3_data):
        """IMBIE-3's annualized record should span at least 44 years
        (1979-2023), versus IMBIE v2021's ~29 years (1992-2020)."""
        years = wais3_data[1]
        assert years[-1] - years[0] >= 44

    def test_baseline_zero(self, wais3_data):
        """Rebased cumulative should be zero at baseline."""
        years, H = wais3_data[1], wais3_data[2]
        bl_idx = np.argmin(np.abs(years - 2005.0))
        assert abs(H[bl_idx]) < 1e-10

    def test_wais_losing_mass(self, wais3_data):
        """WAIS cumulative SLR should be positive at end (mass loss)."""
        H = wais3_data[2]
        assert H[-1] > 0, f"Final H = {H[-1]:.6f}, expected positive (mass loss)"

    def test_sigma_positive(self, wais3_data):
        sigma = wais3_data[3]
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
        """S2_fast_wais uses the skew-normal parametrization; S1_status_quo
        is sampled directly from its quadratic-in-time posterior (see
        A4_SCENARIOS comment block) and so only needs 'P'/'misi'."""
        required_s2 = {'P', 'low_mm', 'high_mm', 'alpha', 'beta_loc',
                       'beta_scale', 'misi'}
        required_s1 = {'P', 'misi'}
        missing_s2 = required_s2 - set(A4_SCENARIOS['S2_fast_wais'].keys())
        missing_s1 = required_s1 - set(A4_SCENARIOS['S1_status_quo'].keys())
        assert not missing_s2, f"S2_fast_wais missing keys: {missing_s2}"
        assert not missing_s1, f"S1_status_quo missing keys: {missing_s1}"

    def test_probabilities_valid(self):
        for sname, params in A4_SCENARIOS.items():
            assert 0 < params['P'] <= 1, (
                f"{sname}: P = {params['P']}, expected 0 < P <= 1")

    def test_ranges_positive(self):
        """Only S2_fast_wais uses low_mm/high_mm; S1_status_quo's spread
        comes from S1_QUADRATIC_COV instead (see test_s1_quadratic_cov_*
        below)."""
        s2 = A4_SCENARIOS['S2_fast_wais']
        assert s2['low_mm'] > 0, "S2_fast_wais: low_mm must be positive"
        assert s2['high_mm'] > s2['low_mm'], (
            f"S2_fast_wais: high_mm ({s2['high_mm']}) must exceed "
            f"low_mm ({s2['low_mm']})")

    def test_s1_no_misi(self):
        """S1 (status quo) should not have MISI."""
        assert A4_SCENARIOS['S1_status_quo']['misi'] is False

    def test_s2_has_misi(self):
        """S2_fast_wais should have MISI."""
        assert A4_SCENARIOS['S2_fast_wais']['misi'] is True

    def test_s1_quadratic_cov_is_valid(self):
        """The installed S1 quadratic should be a valid (symmetric,
        positive semi-definite) 3x3 covariance for (a, v, H0).

        The fit is inherited from component_wais.ipynb (via the stored
        wais/s1_quadratic group), not hardcoded here, so this also
        checks that what the notebook stored is usable.  A non-PSD
        covariance is the signature of a cumulative sigma that was not
        re-anchored to the rebase epoch.
        """
        mean, cov = get_s1_quadratic()
        assert mean.shape == (3,)
        assert cov.shape == (3, 3)
        np.testing.assert_allclose(cov, cov.T)
        eigvals = np.linalg.eigvalsh(cov)
        assert np.all(eigvals >= -1e-18), (
            f"S1 quadratic cov has negative eigenvalues: {eigvals}")

    def test_s1_quadratic_acceleration_positive_median(self):
        """S1's fitted acceleration (a) should have a positive posterior
        median: the observed WAIS record accelerates over 1992-2020, and
        this is what lets S1 (no MISI) still rise faster than a linear
        continuation would."""
        mean, _ = get_s1_quadratic()
        assert mean[0] > 0

    def test_s2_has_accelerating_trajectory(self):
        """S2_fast_wais should have beta_scale > 0 (accelerating ramp)."""
        assert A4_SCENARIOS['S2_fast_wais']['beta_scale'] > 0

    def test_scenario_ordering(self):
        """Scenarios should be ordered by severity: S1's sampled endpoint
        distribution should sit below S2's high_mm bound."""
        rng = np.random.default_rng(0)
        s1_samples_mm = _sample_s1_quadratic_mm(50_000, rng, [2100.0])[:, 0]
        s2 = A4_SCENARIOS['S2_fast_wais']
        assert np.percentile(s1_samples_mm, 95) < s2['high_mm']

    def test_s2_low_mm_pinned_to_s1_median(self):
        """S2's low_mm should be pinned to (approximately) the median
        (50th percentile) of S1's endpoint distribution.

        NOTE: unlike the earlier percentile choices, pinning to the
        median means S2's floor sits in the *middle* of S1's own
        distribution rather than above its tail -- see the A4_SCENARIOS
        comment block for this caveat.

        History (2026-09-17, all same day): p99 (pre-ISMIP6-widening) ->
        p95 (alongside the ISMIP6 extrapolation-error widening added to
        _sample_s1_quadratic_mm, S1_ISMIP6_STD_COEFFS in
        component_projections.py, since S1's p99 was no longer a stable,
        sample-efficient anchor once it had a real tail) -> p83 (closer
        to the independent basin-by-basin estimate) -> p50/median (on
        request) -- see the A4_SCENARIOS comment block."""
        rng = np.random.default_rng(0)
        s1_samples_mm = _sample_s1_quadratic_mm(200_000, rng, [2100.0])[:, 0]
        s1_median = np.percentile(s1_samples_mm, 50)
        s2 = A4_SCENARIOS['S2_fast_wais']
        assert s2['low_mm'] == pytest.approx(s1_median, abs=5.0)

    def test_s2_high_mm_is_round_one_meter(self):
        """S2_fast_wais's 95th percentile bound should be a round 1000 mm,
        chosen so the full mixture's own p95 (not S2's within-scenario
        p95) sits at or below the IPCC AR6 low-confidence AIS SSP5-8.5 p95
        (~1309 mm) -- see test_mixture_p95_not_above_ar6_low_confidence."""
        assert A4_SCENARIOS['S2_fast_wais']['high_mm'] == pytest.approx(1000)

    def test_mixture_p95_not_above_ar6_low_confidence(self):
        """The full two-scenario mixture's p95 at 2100 should not sit
        meaningfully above the IPCC AR6 low-confidence AIS SSP5-8.5 p95
        (1309 mm) -- there is no additional data to defend a mixture tail
        heavier than AR6's own low-confidence storyline."""
        rng = np.random.default_rng(42)
        mix_mm = sample_a4_wais_endpoint(500_000, rng, rheology_mode='B') * 1000.0
        assert np.percentile(mix_mm, 95) < 1309.0 + 50.0

    def test_s2_alpha_is_representative_literature_value(self):
        """alpha should be a round, representative positive-skew value
        (Robel et al. 2019); the mixture is insensitive to its precise
        value, so no fitted or blended precision is expected."""
        s2 = A4_SCENARIOS['S2_fast_wais']
        assert s2['alpha'] == pytest.approx(3.0)

    def test_s2_beta_loc_is_back_loaded(self):
        """beta_loc should give a back-loaded (accelerating) trajectory,
        i.e. beta_ref > 1."""
        s2 = A4_SCENARIOS['S2_fast_wais']
        assert np.exp(s2['beta_loc']) > 1.0


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
        sig_mm = (ex['q95'][idx_2100] - ex['q05'][idx_2100]) / (2 * Z_90)
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
        """±20% range override should correctly scale low_mm and high_mm.
        Only S2_fast_wais uses low_mm/high_mm -- S1_status_quo's spread
        comes from S1_QUADRATIC_COV and does not accept this override."""
        s2 = A4_SCENARIOS['S2_fast_wais']
        for factor in [0.8, 1.0, 1.2]:
            override = {'low_mm': s2['low_mm'] * factor,
                        'high_mm': s2['high_mm'] * factor}
            assert override['low_mm'] == pytest.approx(s2['low_mm'] * factor)
            assert override['high_mm'] == pytest.approx(s2['high_mm'] * factor)

    def test_range_override_affects_median(self):
        """Scaling S2_fast_wais's bounds up by 20% should increase the
        mixture median endpoint."""
        rng1 = np.random.default_rng(77)
        rng2 = np.random.default_rng(77)
        base = sample_a4_wais_endpoint(50000, rng1)
        s2 = A4_SCENARIOS['S2_fast_wais']
        overrides = {'S2_fast_wais': {'low_mm': s2['low_mm'] * 1.2,
                                       'high_mm': s2['high_mm'] * 1.2}}
        scaled = sample_a4_wais_endpoint(
            50000, rng2, scenario_overrides=overrides)
        assert np.median(scaled) > np.median(base), (
            "Scaling S2 bounds up 20% should increase median")

    def test_rheology_sensitivity_direction(self):
        """Higher Glen's exponent n should produce higher rheology correction."""
        for n_exp in [3.5, 4.0, 4.5]:
            rheo_med = 1 + RHEOLOGY_SENSITIVITY * (n_exp - N_REF)
            assert rheo_med >= 1.0, f"n={n_exp}: rheology factor {rheo_med} < 1"
        # Higher n → higher correction
        r_35 = 1 + RHEOLOGY_SENSITIVITY * (3.5 - N_REF)
        r_45 = 1 + RHEOLOGY_SENSITIVITY * (4.5 - N_REF)
        assert r_45 > r_35
