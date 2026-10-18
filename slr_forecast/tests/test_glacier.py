"""Unit tests for glacier component notebook logic.

Tests data processing (including the 95% -> 1-sigma conversion of the
GlaMBIE and Zemp readers), the production level-space fit (balance prior,
sigma_extra, BIC from m2lnL and k_fit), projection mechanics, volume cap,
unit consistency, and I/O roundtrip for the glacier component.
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
from slr_forecast.config import BASELINE_YEAR
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
ZEMP_PATH = os.path.join(RAW_DIR, 'glaciers', 'zemp2019', 'Zemp_etal_results_global.csv')
GT_PER_M_SLE = 362_500.0   # 362.5 Gt per mm SLE

N = 200_000       # for statistical tests
N_SMALL = 2000    # for shape/unit tests
RNG_SEED = 2026

HAS_GLAMBIE = os.path.exists(GLAMBIE_PATH)
HAS_ZEMP = os.path.exists(ZEMP_PATH)


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

    def test_sigma_is_one_sigma(self, glambie):
        """GlaMBIE reports 95% half-widths in Gt/yr; the reader must return
        1-sigma in m/yr SLE: |raw| / 1.96 / 362500."""
        import pandas as pd
        raw = pd.read_csv(GLAMBIE_PATH)
        np.testing.assert_allclose(
            glambie['mass_balance_sigma'].values,
            raw['combined_gt_errors'].abs().values / 1.96 / GT_PER_M_SLE, rtol=1e-12)
        np.testing.assert_allclose(
            glambie['mass_balance'].values,
            -raw['combined_gt'].values / GT_PER_M_SLE, rtol=1e-12)


@pytest.mark.skipif(not HAS_ZEMP, reason="Zemp et al. (2019) data file not found")
def test_zemp_reader_sigma_is_one_sigma():
    """Zemp et al. (2019) uncertainties are 95% half-widths in Gt/yr; the
    reader returns 1-sigma in m/yr SLE, with the geodetic term separate."""
    import pandas as pd
    from slr_data_readers import read_zemp2019_global
    z = read_zemp2019_global(ZEMP_PATH)
    raw = pd.read_csv(ZEMP_PATH, comment='#', skipinitialspace=True).set_index('Year')
    np.testing.assert_allclose(z['sigma_cor'].values,
                               raw['sig_Geod_Gt'].values / 1.96 / GT_PER_M_SLE, rtol=1e-12)
    ind = np.sqrt(raw['sig_Glac_Gt'] ** 2 + raw['sig_Int_Gt'] ** 2
                  + raw['sig_Area_Gt'] ** 2 + raw['sig_Crossed_Gt'] ** 2).values
    np.testing.assert_allclose(z['sigma_ind'].values, ind / 1.96 / GT_PER_M_SLE, rtol=1e-12)
    np.testing.assert_allclose(z['rate'].values, -raw['INT_Gt'].values / GT_PER_M_SLE, rtol=1e-12)


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
        bl_idx = np.argmin(np.abs(years - BASELINE_YEAR))
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
        """Cumulative 2001-2023 relative to the BASELINE_YEAR point should be
        about 18 mm (GlaMBIE mean rate ~0.75 mm/yr)."""
        years, rebase, _, _, _ = glac_data
        latest = rebase[-1] * M_TO_MM
        assert 14 < latest < 22, (
            f"Cumulative at {years[-1]:.1f}: {latest:.1f} mm, "
            f"expected 14-22 mm relative to {BASELINE_YEAR:.0f}")

    def test_cumulative_sigma_grows(self, glac_data):
        """Cumulative uncertainty should grow monotonically (sqrt of sum of squares)."""
        _, _, sigma, _, _ = glac_data
        assert np.all(np.diff(sigma) >= 0)

    def test_baseline_zero(self, glac_data):
        """Rebased cumulative should be zero at the first point, the one
        nearest BASELINE_YEAR (the notebook's anchor; GlaMBIE starts there)."""
        years, rebase, _, _, _ = glac_data
        bl_idx = np.argmin(np.abs(years - BASELINE_YEAR))
        assert bl_idx == 0
        assert np.abs(rebase[bl_idx]) < 1e-12

    def test_rebased_sigma_zero_at_baseline(self, glac_data):
        """Rebased sigma should be exactly zero at the baseline year."""
        years, _, _, _, sigma_rebased = glac_data
        bl_idx = np.argmin(np.abs(years - BASELINE_YEAR))
        assert sigma_rebased[bl_idx] == pytest.approx(0.0, abs=1e-15)

    def test_rebased_sigma_post_baseline(self, glac_data):
        """For points after the baseline, rebased sigma should be less than
        the raw cumulative sigma (since baseline variance is subtracted)."""
        years, _, cumul_sigma, _, sigma_rebased = glac_data
        bl_idx = np.argmin(np.abs(years - BASELINE_YEAR))
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

@pytest.mark.filterwarnings("ignore")
class TestLevelCorrelatedFit:
    """The production fit (fit_bayesian_level_annual_correlated) with the
    1850 balance prior, a fitted sigma_extra, and the BIC the notebook builds
    from its m2lnL and k_fit. Short chains on synthetic records: these test
    the bookkeeping, not convergence."""

    N_OBS = 24
    SIG = 1.5e-4                     # m/yr, 1-sigma per annual rate
    MCMC = dict(n_samples=600, n_walkers=24, n_burnin=400, thin=1, progress=False)
    PRIORS = dict(prior_scale_b=2e-3, prior_b_mean=0.61e-3, prior_c_mean=3e-4,
                  prior_c_sigma=1e-3, prior_H0_mean=0.0, prior_H0_sigma=2e-3,
                  prior_sigma_extra_scale=2e-4, fit_sigma_extra=True)

    @staticmethod
    def _record(rate):
        H = np.cumsum(rate)
        return H - H[0], 2000.5 + np.arange(len(rate))

    def _fit_pair(self, T, rate, prior_scale_a, balance):
        from bayesian_models import fit_bayesian_level_annual_correlated as fit
        H, t = self._record(rate)
        sig = np.full(len(rate), self.SIG)
        kw = dict(self.PRIORS, prior_scale_a=prior_scale_a, **self.MCMC)
        if balance is not None:
            kw['balance_T_eq0'] = balance
        lin = fit(H, rate, sig, T, t, order=1, seed=1, **kw)
        quad = fit(H, rate, sig, T, t, order=2, seed=2, **kw)
        n = len(rate)
        bic_l = lin.m2lnL + lin.k_fit * np.log(n)
        bic_q = quad.m2lnL + quad.k_fit * np.log(n)
        return lin, quad, bic_l - bic_q

    @pytest.fixture(scope="class")
    def linear_truth(self):
        """Linear rate b (T - T_eq0) with excess scatter, fitted with the
        balance prior and the notebook's tight prior on a."""
        rng = np.random.default_rng(1)
        T = np.linspace(0.1, 0.9, self.N_OBS) + rng.normal(0, 0.05, self.N_OBS)
        rate = 0.8e-3 * (T + 0.8) + rng.normal(0, np.hypot(self.SIG, 1.8e-4), self.N_OBS)
        return self._fit_pair(T, rate, prior_scale_a=0.04e-3, balance=(-0.86, 0.15))

    @pytest.fixture(scope="class")
    def quadratic_truth(self):
        """Strongly curved rate over a wide T range, wide prior on a, c free."""
        rng = np.random.default_rng(3)
        T = np.linspace(-0.5, 1.5, self.N_OBS)
        rate = 0.5e-3 * T + 3e-3 * T ** 2 + 0.3e-3 + rng.normal(0, self.SIG, self.N_OBS)
        return self._fit_pair(T, rate, prior_scale_a=10e-3, balance=None)

    def test_parameter_count(self, linear_truth):
        """k_fit counts b, T_eq0 (in c's place), H0 and sigma_extra, plus a
        for the quadratic (MD-1)."""
        lin, quad, _ = linear_truth
        assert lin.k_fit == 4
        assert quad.k_fit == 5
        assert np.isfinite(lin.m2lnL) and np.isfinite(quad.m2lnL)

    def test_balance_prior_ties_c_to_b(self, linear_truth):
        """With balance_T_eq0, c = -b T_eq0 in every posterior draw (order 1)."""
        lin, _, _ = linear_truth
        b = lin.posterior_samples[:, 1]
        c = lin.posterior_samples[:, 2]
        T_eq0 = lin.design_info['T_eq0_samples']
        np.testing.assert_allclose(c, -b * T_eq0, rtol=0, atol=1e-15)
        assert np.all(lin.posterior_samples[:, 0] == 0.0)

    def test_sigma_extra_fitted(self, linear_truth):
        """sigma_extra is sampled (positive draws, not a constant)."""
        lin, _, _ = linear_truth
        se = lin.sigma_extra_posterior
        assert np.all(se >= 0) and np.std(se) > 0
        assert 0.3e-4 < np.median(se) < 6e-4

    def test_bic_prefers_linear_for_linear_truth(self, linear_truth):
        """With no curvature in the data the fits are nearly identical, so
        Delta BIC (linear minus quadratic) is negative and close to -ln n."""
        _, _, delta_bic = linear_truth
        assert delta_bic < 0, f"Delta BIC = {delta_bic:.1f}, expected linear preferred"
        assert abs(delta_bic + np.log(self.N_OBS)) < 2.0

    def test_bic_prefers_quadratic_for_curved_truth(self, quadratic_truth):
        """Strong curvature must give Delta BIC well above 2."""
        _, _, delta_bic = quadratic_truth
        assert delta_bic > 10, f"Delta BIC = {delta_bic:.1f}, expected quadratic preferred"


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
        # Synthetic: linear growth from 0 at BASELINE_YEAR to ~0.12 m at 2100
        t_norm = (PROJ_YEARS - BASELINE_YEAR) / (2100 - BASELINE_YEAR)
        t_norm = np.clip(t_norm, 0, None)
        base = 0.12 * t_norm  # meters

        samples = base[np.newaxis, :] + rng.normal(0, 0.005, (N_SMALL, n_times))
        samples = np.maximum(samples, 0)  # no negative SLR for glaciers after the baseline

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
                extra_metadata={'r2': 0.992, 'model_type': 'linear_level_correlated'},
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
            # The notebook passes the production label through extra_metadata
            assert loaded['metadata']['model_type'] == 'linear_level_correlated'
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
        t_norm = np.clip((PROJ_YEARS - BASELINE_YEAR) / (2100 - BASELINE_YEAR), 0, None)
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
