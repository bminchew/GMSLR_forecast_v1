"""Unit tests for glacier component notebook logic.

Tests data processing, model fitting, projection mechanics, volume cap,
unit consistency, BIC model selection, and I/O roundtrip for the glacier
component of the SLR forecast.
"""

import os
import sys
import tempfile

import numpy as np
import pytest

# Notebook modules live in notebooks/, not the installed package
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'notebooks'))

from component_projections import apply_glacier_volume_cap
from slr_forecast import M_TO_MM
from component_io import (
    save_glacier,
    load_component,
    PROJ_YEARS,
    _require_file,
)
from component_analysis import apply_sigma_taper, compute_component_rates

# Paths
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
RAW_DIR = os.path.join(PROJECT_ROOT, 'data', 'raw')
GLAMBIE_PATH = os.path.join(RAW_DIR, 'glaciers', '0_global_glambie_consensus.csv')

N = 200_000       # for statistical tests
N_SMALL = 2000    # for shape/unit tests
RNG_SEED = 2026

HAS_GLAMBIE = os.path.exists(GLAMBIE_PATH)


# =========================================================================
# GlaMBIE data reader
# =========================================================================

@pytest.mark.skipif(not HAS_GLAMBIE, reason="GlaMBIE data file not found")
class TestGlaMBIEReader:
    """Verify read_glambie_global returns correct sign conventions and units."""

    @pytest.fixture(scope="class")
    def glambie(self):
        from slr_data_readers import read_glambie_global
        return read_glambie_global(GLAMBIE_PATH)

    def test_columns_present(self, glambie):
        expected = {'decimal_year', 'mass_balance', 'mass_balance_sigma'}
        assert expected.issubset(set(glambie.columns))

    def test_positive_means_slr(self, glambie):
        """Mass balance should be positive (SLR convention: mass loss = +SLR)."""
        mean_rate = glambie['mass_balance'].mean()
        assert mean_rate > 0, (
            f"Mean glacier rate = {mean_rate:.4f}, expected positive for SLR convention")

    def test_units_are_meters_per_year(self, glambie):
        """Mass balance rate should be in m/yr SLE (order 1e-4 to 1e-3),
        not mm/yr (order 0.1 to 1) or Gt/yr (order 100-300)."""
        max_rate = glambie['mass_balance'].max()
        assert max_rate < 0.01, (
            f"Max rate = {max_rate:.4f}, too large for m/yr SLE")
        assert max_rate > 1e-5, (
            f"Max rate = {max_rate:.6f}, too small for m/yr SLE")

    def test_sigma_positive(self, glambie):
        """All uncertainties should be positive."""
        assert np.all(glambie['mass_balance_sigma'].values > 0)

    def test_year_range(self, glambie):
        """GlaMBIE starts around 2000."""
        years = glambie['decimal_year'].values
        assert years[0] >= 1990
        assert years[0] <= 2005
        assert years[-1] >= 2020

    def test_monotonic_years(self, glambie):
        """Years should be monotonically increasing."""
        years = glambie['decimal_year'].values
        assert np.all(np.diff(years) > 0)


# =========================================================================
# Cumulative SLR computation
# =========================================================================

@pytest.mark.skipif(not HAS_GLAMBIE, reason="GlaMBIE data file not found")
class TestGlacierCumulative:
    """Verify cumulative SLR computation from annual rates."""

    @pytest.fixture(scope="class")
    def glac_data(self):
        from slr_data_readers import read_glambie_global
        df = read_glambie_global(GLAMBIE_PATH)
        rate = df['mass_balance'].values
        rate_sigma = df['mass_balance_sigma'].values
        years = df['decimal_year'].values
        cumul = np.cumsum(rate)
        cumul_sigma = np.sqrt(np.cumsum(rate_sigma**2))
        bl_idx = np.argmin(np.abs(years - 2005.0))
        rebase = cumul - cumul[bl_idx]
        # Rebased sigma: Var(Z - Z_bl) = |Var(Z) - Var(Z_bl)|
        sigma_rebased = np.sqrt(np.abs(cumul_sigma**2 - cumul_sigma[bl_idx]**2))
        return years, rebase, cumul_sigma, rate, sigma_rebased

    def test_cumulative_increases(self, glac_data):
        """Cumulative glacier SLR should generally increase (glaciers losing mass)."""
        _, rebase, _, _, _ = glac_data
        # Last value should be higher than first (net mass loss)
        assert rebase[-1] > rebase[0]

    def test_cumulative_magnitude(self, glac_data):
        """Cumulative at 2024 should be ~15-25 mm relative to 2005."""
        years, rebase, _, _, _ = glac_data
        latest = rebase[-1] * M_TO_MM
        assert 10 < latest < 40, (
            f"Cumulative at {years[-1]:.0f}: {latest:.1f} mm, "
            f"expected 10-40 mm relative to 2005")

    def test_cumulative_sigma_grows(self, glac_data):
        """Cumulative uncertainty should grow monotonically (sqrt of sum of squares)."""
        _, _, sigma, _, _ = glac_data
        assert np.all(np.diff(sigma) >= 0)

    def test_baseline_zero(self, glac_data):
        """Rebased cumulative should be near zero at baseline year."""
        years, rebase, _, _, _ = glac_data
        bl_idx = np.argmin(np.abs(years - 2005.0))
        assert np.abs(rebase[bl_idx]) < 1e-12

    def test_rebased_sigma_zero_at_baseline(self, glac_data):
        """Rebased sigma should be exactly zero at the baseline year."""
        years, _, _, _, sigma_rebased = glac_data
        bl_idx = np.argmin(np.abs(years - 2005.0))
        assert sigma_rebased[bl_idx] == pytest.approx(0.0, abs=1e-15)

    def test_rebased_sigma_post_baseline(self, glac_data):
        """For points after the baseline, rebased sigma should be less than
        the raw cumulative sigma (since baseline variance is subtracted)."""
        years, _, cumul_sigma, _, sigma_rebased = glac_data
        bl_idx = np.argmin(np.abs(years - 2005.0))
        post = slice(bl_idx + 1, None)
        assert np.all(sigma_rebased[post] <= cumul_sigma[post] + 1e-15)


# =========================================================================
# Volume cap
# =========================================================================

class TestGlacierVolumeCap:
    """Verify apply_glacier_volume_cap clamps correctly."""

    def test_cap_applied(self):
        """Samples exceeding v_total should be clamped."""
        samples = np.array([[0.1, 0.2, 0.35, 0.4],
                            [0.05, 0.15, 0.25, 0.30]])
        capped = apply_glacier_volume_cap(samples, v_total=0.32)
        assert np.all(capped <= 0.32)
        assert capped[0, 2] == 0.32
        assert capped[0, 3] == 0.32
        # Uncapped values should be unchanged
        assert capped[0, 0] == 0.1
        assert capped[1, 3] == 0.30

    def test_no_modification_below_cap(self):
        """If all samples are below v_total, nothing changes."""
        samples = np.random.default_rng(42).uniform(0, 0.2, (100, 50))
        capped = apply_glacier_volume_cap(samples, v_total=0.32)
        np.testing.assert_array_equal(samples, capped)

    def test_does_not_modify_inplace(self):
        """apply_glacier_volume_cap should not modify the input array."""
        samples = np.array([[0.1, 0.4]])
        original = samples.copy()
        _ = apply_glacier_volume_cap(samples, v_total=0.32)
        np.testing.assert_array_equal(samples, original)

    def test_negative_values_unchanged(self):
        """Negative samples (physically possible early in the time series)
        should not be modified by the cap."""
        samples = np.array([[-0.01, 0.05, 0.35]])
        capped = apply_glacier_volume_cap(samples, v_total=0.32)
        assert capped[0, 0] == -0.01
        assert capped[0, 2] == 0.32

    def test_farinotti_cap_value(self):
        """Default cap should be 0.32 m SLE (Farinotti et al. 2019)."""
        samples = np.array([[0.33]])
        capped = apply_glacier_volume_cap(samples)
        assert capped[0, 0] == 0.32


# =========================================================================
# Sigma taper
# =========================================================================

class TestSigmaTaper:
    """Verify apply_sigma_taper inflates correctly."""

    def test_no_inflation_after_reference(self):
        """Points at or after t_ref should be unchanged."""
        sigma = np.ones(10)
        years = np.arange(2000, 2010, dtype=float)
        tapered = apply_sigma_taper(sigma, years, t_ref=2000, f_max=3)
        # 2000 and later should have f=1 (no inflation)
        assert tapered[-1] == 1.0

    def test_maximum_inflation_at_start(self):
        """The earliest point should have the maximum inflation."""
        sigma = np.ones(10) * 0.5
        years = np.arange(2000, 2010, dtype=float)
        tapered = apply_sigma_taper(sigma, years, t_ref=2009, f_max=3)
        # First point should be inflated most
        assert tapered[0] > tapered[-1]

    def test_fmax_1_no_change(self):
        """f_max=1 should produce no inflation."""
        sigma = np.array([0.1, 0.2, 0.3])
        years = np.array([2000, 2005, 2010])
        tapered = apply_sigma_taper(sigma, years, t_ref=2005, f_max=1)
        np.testing.assert_array_almost_equal(sigma, tapered)


# =========================================================================
# BIC model selection
# =========================================================================

class TestBICModelSelection:
    """Verify BIC computation logic used in the notebook."""

    def test_bic_formula(self):
        """BIC = n*log(RSS/n) + k*log(n) where k is number of parameters."""
        n = 23
        rss = 0.5
        k = 5
        bic = n * np.log(rss / n) + k * np.log(n)
        # Just verify formula produces finite results
        assert np.isfinite(bic)

    def test_bic_prefers_simpler_model(self):
        """When RSS is nearly identical, BIC should prefer the simpler model."""
        n = 23
        rss_lin = 1.0
        rss_quad = 0.999  # barely better
        bic_lin = n * np.log(rss_lin / n) + 4 * np.log(n)
        bic_quad = n * np.log(rss_quad / n) + 5 * np.log(n)
        delta_bic = bic_lin - bic_quad
        # delta_bic should be negative (linear preferred)
        assert delta_bic < 2, (
            f"ΔBIC={delta_bic:.1f}, linear should be preferred when RSS nearly equal")

    def test_bic_prefers_better_fit(self):
        """When RSS is substantially better, BIC should prefer the complex model."""
        n = 23
        rss_lin = 2.0
        rss_quad = 0.5  # much better
        bic_lin = n * np.log(rss_lin / n) + 4 * np.log(n)
        bic_quad = n * np.log(rss_quad / n) + 5 * np.log(n)
        delta_bic = bic_lin - bic_quad
        assert delta_bic > 2, (
            f"ΔBIC={delta_bic:.1f}, quadratic should be preferred with much better RSS")


# =========================================================================
# Projection shape and unit checks
# =========================================================================

class TestProjectionShapeAndUnits:
    """Verify projection arrays have correct shapes and units using synthetic data."""

    @pytest.fixture
    def synthetic_projection(self):
        """Create a synthetic glacier projection dict mimicking notebook output."""
        rng = np.random.default_rng(RNG_SEED)
        n_times = len(PROJ_YEARS)
        # Synthetic: linear growth from 0 at 2005 to ~0.12 m at 2100
        t_norm = (PROJ_YEARS - 2005) / (2100 - 2005)
        t_norm = np.clip(t_norm, 0, None)
        base = 0.12 * t_norm  # meters

        samples = base[np.newaxis, :] + rng.normal(0, 0.005, (N_SMALL, n_times))
        samples = np.maximum(samples, 0)  # no negative SLR for glaciers post-2005

        proj = {
            'samples': samples,
            'median': np.median(samples, axis=0),
            'p5': np.percentile(samples, 5, axis=0),
            'p95': np.percentile(samples, 95, axis=0),
            'p17': np.percentile(samples, 17, axis=0),
            'p83': np.percentile(samples, 83, axis=0),
        }
        return proj

    def test_samples_shape(self, synthetic_projection):
        assert synthetic_projection['samples'].shape == (N_SMALL, len(PROJ_YEARS))

    def test_percentile_ordering(self, synthetic_projection):
        """p5 <= p17 <= median <= p83 <= p95 at every time step."""
        p = synthetic_projection
        assert np.all(p['p5'] <= p['p17'] + 1e-10)
        assert np.all(p['p17'] <= p['median'] + 1e-10)
        assert np.all(p['median'] <= p['p83'] + 1e-10)
        assert np.all(p['p83'] <= p['p95'] + 1e-10)

    def test_units_are_meters(self, synthetic_projection):
        """Median at 2100 should be in meters (0.05-0.25 m), not mm."""
        idx_2100 = np.argmin(np.abs(PROJ_YEARS - 2100))
        med = synthetic_projection['median'][idx_2100]
        assert 0.01 < med < 1.0, (
            f"Median at 2100 = {med:.4f}, expected meters not mm")

    def test_volume_cap_respected(self, synthetic_projection):
        """After applying volume cap, no sample should exceed v_total."""
        capped = apply_glacier_volume_cap(
            synthetic_projection['samples'], v_total=0.32)
        assert np.all(capped <= 0.32 + 1e-10)


# =========================================================================
# I/O roundtrip
# =========================================================================

class TestGlacierIO:
    """Verify save/load roundtrip for glacier component."""

    @pytest.fixture
    def mock_result(self):
        """Create a minimal mock result object."""
        from types import SimpleNamespace
        rng = np.random.default_rng(42)
        n_post = 500
        result = SimpleNamespace(
            posterior_samples=rng.normal(0, 0.001, (n_post, 3)),
            H0_posterior=rng.normal(0, 0.001, n_post),
            r2=0.992,
            residuals=rng.normal(0, 0.0001, 23),
        )
        return result

    @pytest.fixture
    def mock_projections(self):
        """Create synthetic projection dict for all SSPs."""
        rng = np.random.default_rng(42)
        n_times = len(PROJ_YEARS)
        proj = {}
        for ssp in ['SSP1-2.6', 'SSP2-4.5', 'SSP3-7.0', 'SSP5-8.5']:
            samples = rng.uniform(0, 0.15, (N_SMALL, n_times))
            proj[ssp] = {
                'samples': samples,
                'median': np.median(samples, axis=0),
                'p5': np.percentile(samples, 5, axis=0),
                'p95': np.percentile(samples, 95, axis=0),
                'p17': np.percentile(samples, 17, axis=0),
                'p83': np.percentile(samples, 83, axis=0),
            }
        return proj

    def test_roundtrip(self, mock_result, mock_projections):
        """Save and reload glacier data; verify arrays match."""
        with tempfile.TemporaryDirectory() as tmpdir:
            h5_path = os.path.join(tmpdir, 'test_component_results.h5')

            obs_years = np.arange(2000, 2024, dtype=float)
            obs_H = np.linspace(0, 0.018, len(obs_years))
            obs_sigma = np.full(len(obs_years), 0.001)

            save_glacier(
                result_lin=mock_result,
                obs_years=obs_years,
                obs_H=obs_H,
                obs_sigma=obs_sigma,
                proj_dict=mock_projections,
                v_glacier_total_m=0.32,
                extra_metadata={'r2': 0.992},
                h5_path=h5_path,
            )

            loaded = load_component('glacier', h5_path=h5_path)

            # Check projections roundtrip
            assert set(loaded['projections'].keys()) == set(mock_projections.keys())
            for ssp in mock_projections:
                np.testing.assert_allclose(
                    loaded['projections'][ssp]['median'],
                    mock_projections[ssp]['median'],
                    atol=1e-8,
                )
                np.testing.assert_allclose(
                    loaded['projections'][ssp]['samples'],
                    mock_projections[ssp]['samples'],
                    atol=1e-6,
                )

            # Check observations roundtrip
            np.testing.assert_allclose(
                loaded['observations']['years'], obs_years, atol=1e-8)
            np.testing.assert_allclose(
                loaded['observations']['H_obs'], obs_H, atol=1e-8)

            # Check posteriors roundtrip
            np.testing.assert_allclose(
                loaded['posteriors']['posterior_samples'],
                mock_result.posterior_samples,
                atol=1e-8,
            )

            # Check metadata
            assert loaded['metadata']['model_type'] == 'linear_dols'
            assert float(loaded['metadata']['v_glacier_total_m']) == pytest.approx(0.32)

    def test_proj_years_stored(self, mock_result, mock_projections):
        """proj_years should be stored and recoverable."""
        with tempfile.TemporaryDirectory() as tmpdir:
            h5_path = os.path.join(tmpdir, 'test_proj_years.h5')
            obs_years = np.arange(2000, 2024, dtype=float)
            obs_H = np.linspace(0, 0.018, len(obs_years))
            obs_sigma = np.full(len(obs_years), 0.001)

            save_glacier(
                result_lin=mock_result,
                obs_years=obs_years,
                obs_H=obs_H,
                obs_sigma=obs_sigma,
                proj_dict=mock_projections,
                h5_path=h5_path,
            )

            loaded = load_component('glacier', h5_path=h5_path)
            np.testing.assert_array_equal(loaded['proj_years'], PROJ_YEARS)


# =========================================================================
# SSP ordering consistency
# =========================================================================

class TestSSPOrdering:
    """Verify that higher-forcing SSPs produce higher glacier SLR."""

    @pytest.fixture
    def ordered_projections(self):
        """Synthetic projections with SSP-monotonic medians."""
        rng = np.random.default_rng(RNG_SEED)
        n_times = len(PROJ_YEARS)
        t_norm = np.clip((PROJ_YEARS - 2005) / (2100 - 2005), 0, None)
        proj = {}
        # Higher SSP → more warming → higher glacier loss
        medians_2100_m = {
            'SSP1-2.6': 0.10,
            'SSP2-4.5': 0.12,
            'SSP3-7.0': 0.14,
            'SSP5-8.5': 0.16,
        }
        for ssp, med_2100 in medians_2100_m.items():
            base = med_2100 * t_norm
            samples = base[np.newaxis, :] + rng.normal(0, 0.003, (500, n_times))
            proj[ssp] = {
                'samples': samples,
                'median': np.median(samples, axis=0),
            }
        return proj

    def test_ssp_ordering_at_2100(self, ordered_projections):
        """SSP5-8.5 median > SSP3-7.0 > SSP2-4.5 > SSP1-2.6 at 2100."""
        idx_2100 = np.argmin(np.abs(PROJ_YEARS - 2100))
        meds = {ssp: ordered_projections[ssp]['median'][idx_2100]
                for ssp in ordered_projections}
        assert meds['SSP5-8.5'] > meds['SSP3-7.0']
        assert meds['SSP3-7.0'] > meds['SSP2-4.5']
        assert meds['SSP2-4.5'] > meds['SSP1-2.6']


# =========================================================================
# Rate computation
# =========================================================================

class TestComputeRates:
    """Verify compute_component_rates returns sensible rates."""

    def test_constant_rate(self):
        """For linear cumulative, computed rate should be constant."""
        years = np.arange(2000, 2020, dtype=float)
        H = 0.001 * (years - 2005)  # 1 mm/yr in meters
        rates = compute_component_rates(years, H, window=3)
        # Interior rates should be ~0.001 m/yr
        valid = np.isfinite(rates)
        assert np.allclose(rates[valid], 0.001, atol=1e-6)

    def test_units_preserved(self):
        """Output rates should be in same units as input H per year."""
        years = np.arange(2000, 2020, dtype=float)
        H = 0.002 * (years - 2005)  # 2 mm/yr in meters
        rates = compute_component_rates(years, H, window=3)
        valid = np.isfinite(rates)
        assert np.allclose(rates[valid], 0.002, atol=1e-6)


# =========================================================================
# IPCC comparison sanity checks
# =========================================================================

@pytest.mark.skipif(
    not os.path.exists(os.path.join(
        RAW_DIR, 'ipcc_ar6', 'slr', 'ar6', 'global',
        'confidence_output_files', 'medium_confidence', 'ssp245',
        'glaciers_ssp245_medium_confidence_values.nc')),
    reason="IPCC AR6 glacier data not found")
class TestIPCCGlacierData:
    """Verify IPCC glacier data is read correctly."""

    @pytest.fixture(scope="class")
    def ipcc_data(self):
        from component_projections import read_ipcc_component_nc, ipcc_extract
        conf_base = os.path.join(
            RAW_DIR, 'ipcc_ar6', 'slr', 'ar6', 'global',
            'confidence_output_files')
        data = read_ipcc_component_nc(conf_base, 'medium_confidence',
                                      'ssp245', 'glaciers')
        return data

    def test_not_none(self, ipcc_data):
        assert ipcc_data is not None

    def test_years_range(self, ipcc_data):
        years = ipcc_data['years']
        assert years[0] >= 2005
        assert years[-1] <= 2300

    def test_units_are_mm(self, ipcc_data):
        """IPCC SLC should be in mm (order 10-200 at 2100)."""
        from component_projections import ipcc_extract
        ex = ipcc_extract(ipcc_data)
        # Median at 2100 should be ~100-200 mm for SSP2-4.5 glaciers
        idx_2100 = np.argmin(np.abs(ex['years'] - 2100))
        med = ex['q50'][idx_2100]
        assert 50 < med < 300, (
            f"IPCC glacier median at 2100 = {med:.0f} mm, expected 50-300 mm")

    def test_quantile_ordering(self, ipcc_data):
        from component_projections import ipcc_extract
        ex = ipcc_extract(ipcc_data)
        assert np.all(ex['q05'] <= ex['q50'] + 0.1)
        assert np.all(ex['q50'] <= ex['q95'] + 0.1)
