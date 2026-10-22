"""Unit tests for Greenland ice sheet component.

Tests data readers, model functions, projection functions, unit consistency,
and I/O roundtrip for the Greenland component of the SLR forecast.
"""

import os
import sys
import tempfile

import numpy as np
import pytest

# Notebook modules live in notebooks/, not the installed package
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'notebooks'))

from component_analysis import (
    annualize_imbie, apply_sigma_taper, compute_component_rates,
    fit_discharge_delay_model,
)
from component_io import (
    save_greenland,
    load_component,
    PROJ_YEARS,
    _require_file,
)
from slr_forecast import M_TO_MM
from slr_forecast.config import Z_90
from slr_forecast.config import GT_TO_M_SLE
from smb_emulator import EmulatorSummary

# Paths
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
RAW_DIR = os.path.join(PROJECT_ROOT, 'data', 'raw')
IMBIE_GR_GT = os.path.join(RAW_DIR, 'ice_sheets', 'Gt', 'imbie_greenland_2021_Gt.csv')
MOUGINOT_PATH = os.path.join(RAW_DIR, 'ice_sheets', 'greenland', 'mouginot2019_data.xlsx')

N_SMALL = 2000   # for shape/unit tests
RNG_SEED = 2026


# =========================================================================
# Data reader availability checks
# =========================================================================

HAS_IMBIE_GR = os.path.exists(IMBIE_GR_GT)
HAS_MOUGINOT = os.path.exists(MOUGINOT_PATH)


# =========================================================================
# IMBIE Greenland reader
# =========================================================================

@pytest.mark.skipif(not HAS_IMBIE_GR, reason="IMBIE Greenland data file not found")
class TestIMBIEGreenlandReader:
    """Verify read_imbie_greenland returns correct sign conventions and units."""

    @pytest.fixture(scope="class")
    def imbie_gr(self):
        from slr_data_readers import read_imbie_greenland
        return read_imbie_greenland(IMBIE_GR_GT, convert_to_sle=True)

    def test_columns_present(self, imbie_gr):
        expected = {'decimal_year', 'mass_balance_rate',
                    'mass_balance_rate_sigma', 'cumulative_mass_balance',
                    'cumulative_mass_balance_sigma'}
        assert expected.issubset(set(imbie_gr.columns))

    def test_units_are_meters(self, imbie_gr):
        """Cumulative mass balance should be in meters (order 1e-3 to 1e-1),
        not mm (order 1 to 100) or Gt (order 1e3)."""
        cum = imbie_gr['cumulative_mass_balance'].values
        max_abs = np.max(np.abs(cum))
        # In meters, Greenland total loss ~0.01-0.05 m SLE
        assert max_abs < 1.0, (
            f"Max |cumulative_mb| = {max_abs:.3f}, too large for meters")
        assert max_abs > 1e-5, (
            f"Max |cumulative_mb| = {max_abs:.3e}, too small for meters")

    def test_sign_convention_slr(self, imbie_gr):
        """IMBIE Gt reader converts to SLR convention: mass loss is
        positive in cumulative_mass_balance. The late-period cumulative
        value should be positive (Greenland has been losing mass)."""
        # Check the last value — cumulative should be positive (SLR convention)
        cum_last = imbie_gr['cumulative_mass_balance'].values[-1]
        assert cum_last > 0, (
            f"Last cumulative_mb = {cum_last:.6f}, expected positive "
            f"(SLR convention: mass loss is positive)")

    def test_sigma_positive(self, imbie_gr):
        """Uncertainty should always be strictly positive."""
        sigma_rate = imbie_gr['mass_balance_rate_sigma'].values
        sigma_cum = imbie_gr['cumulative_mass_balance_sigma'].values
        assert np.all(sigma_rate > 0), "rate sigma should be positive"
        assert np.all(sigma_cum > 0), "cumulative sigma should be positive"

    def test_time_range(self, imbie_gr):
        """IMBIE Greenland should span roughly 1992-2020."""
        years = imbie_gr['decimal_year'].values
        assert years[0] >= 1990 and years[0] <= 1993
        assert years[-1] >= 2018 and years[-1] <= 2022

    def test_monthly_resolution(self, imbie_gr):
        """Should have roughly monthly time steps."""
        years = imbie_gr['decimal_year'].values
        dt = np.diff(years)
        median_dt = np.median(dt)
        assert 0.05 < median_dt < 0.15, (
            f"Median dt = {median_dt:.4f} yr, expected ~0.083 (monthly)")


# =========================================================================
# Mouginot reader
# =========================================================================

@pytest.mark.skipif(not HAS_MOUGINOT, reason="Mouginot data file not found")
class TestMouginotReader:
    """Verify read_mouginot2019_greenland returns correct SLR convention."""

    @pytest.fixture(scope="class")
    def mouginot(self):
        from slr_forecast.readers.ice_sheets import read_mouginot2019_greenland
        return read_mouginot2019_greenland(MOUGINOT_PATH)

    def test_columns_present(self, mouginot):
        expected = {'decimal_year', 'mb_rate', 'mb_rate_sigma',
                    'smb_rate', 'smb_rate_sigma',
                    'discharge_rate', 'discharge_rate_sigma',
                    'cumulative_mb', 'cumulative_mb_sigma'}
        assert expected.issubset(set(mouginot.columns))

    def test_units_are_meters(self, mouginot):
        """Rates should be in m/yr SLE (order 1e-4 to 1e-3)."""
        rate = mouginot['mb_rate'].values
        max_abs = np.max(np.abs(rate))
        assert max_abs < 0.1, (
            f"Max |mb_rate| = {max_abs:.6f}, too large for m/yr SLE")
        assert max_abs > 1e-6, (
            f"Max |mb_rate| = {max_abs:.6e}, too small for m/yr SLE")

    def test_sign_convention_slr(self, mouginot):
        """Mouginot reader should use SLR convention: positive = sea level
        rise. Total mass balance rate should be mostly positive (Greenland
        losing mass)."""
        mb = mouginot['mb_rate'].values
        # Post-2000, GrIS is consistently losing mass
        years = mouginot['decimal_year'].values
        post2000 = mb[years > 2000]
        frac_positive = np.mean(post2000 > 0)
        assert frac_positive > 0.8, (
            f"Only {frac_positive:.0%} of post-2000 mb_rate is positive "
            f"(expected >80% in SLR convention)")

    def test_discharge_positive(self, mouginot):
        """Discharge should always be positive (outflow = SLR)."""
        d = mouginot['discharge_rate'].values
        assert np.all(d > 0), "Discharge rate should always be positive (SLR convention)"

    def test_sigma_positive(self, mouginot):
        """All sigma columns should be non-negative."""
        for col in ['mb_rate_sigma', 'smb_rate_sigma',
                    'discharge_rate_sigma', 'cumulative_mb_sigma']:
            vals = mouginot[col].values
            assert np.all(vals >= 0), f"{col} has negative values"

    def test_time_range(self, mouginot):
        """Mouginot covers 1972-2018."""
        years = mouginot['decimal_year'].values
        assert years[0] >= 1972 and years[0] <= 1973
        assert years[-1] >= 2018 and years[-1] <= 2019

    def test_annual_resolution(self, mouginot):
        """Should be annual data."""
        years = mouginot['decimal_year'].values
        dt = np.diff(years)
        assert np.allclose(dt, 1.0, atol=0.1), (
            f"Expected annual steps, got dt range [{dt.min():.2f}, {dt.max():.2f}]")

    def test_cumulative_mb_sign(self, mouginot):
        """Cumulative mass balance should be positive at end (SLR convention:
        cumulative mass loss = positive SLR contribution)."""
        cum = mouginot['cumulative_mb'].values
        assert cum[-1] > 0, (
            f"Last cumulative_mb = {cum[-1]:.6f}, expected positive in SLR convention")


# =========================================================================
# Mouginot component preparation
# =========================================================================

@pytest.mark.skipif(not HAS_MOUGINOT, reason="Mouginot data file not found")
class TestMouginotComponentPrep:
    """Verify prepare_mouginot_components returns correct structure and units."""

    @pytest.fixture(scope="class")
    def mou_comp(self):
        from slr_forecast.readers.ice_sheets import read_mouginot2019_greenland
        from bayesian_models import prepare_mouginot_components
        df = read_mouginot2019_greenland(MOUGINOT_PATH)
        return prepare_mouginot_components(df, baseline_window=(1995, 2005))

    def test_keys_present(self, mou_comp):
        expected = {'time_smb', 'H_smb', 'sigma_smb',
                    'time_dyn', 'H_dyn', 'sigma_dyn',
                    'time_mb', 'H_mb', 'sigma_mb',
                    'sources', 'df'}
        assert expected.issubset(set(mou_comp.keys()))

    def test_units_meters(self, mou_comp):
        """Cumulative values should be in meters SLE."""
        for key in ('H_smb', 'H_dyn', 'H_mb'):
            vals = mou_comp[key]
            max_abs = np.max(np.abs(vals))
            assert max_abs < 0.5, (
                f"{key}: max |val| = {max_abs:.4f}, too large for meters")

    def test_baseline_near_zero(self, mou_comp):
        """Values near the baseline epoch (2000) should be near zero."""
        t = mou_comp['time_mb']
        bl_idx = np.argmin(np.abs(t - 2000.0))
        assert abs(mou_comp['H_mb'][bl_idx]) < 0.005, (
            f"H_mb at 2000 = {mou_comp['H_mb'][bl_idx]:.6f}, expected near zero")

    def test_consistent_lengths(self, mou_comp):
        """All time/value arrays should have the same length."""
        n = len(mou_comp['time_smb'])
        assert len(mou_comp['H_smb']) == n
        assert len(mou_comp['sigma_smb']) == n
        assert len(mou_comp['time_dyn']) == n
        assert len(mou_comp['H_dyn']) == n


# =========================================================================
# annualize_imbie
# =========================================================================

@pytest.mark.skipif(not HAS_IMBIE_GR, reason="IMBIE Greenland data file not found")
class TestAnnualizeIMBIE:
    """Test annualize_imbie helper function."""

    @pytest.fixture(scope="class")
    def annual(self):
        import pandas as pd
        from slr_data_readers import read_imbie_greenland
        df = read_imbie_greenland(IMBIE_GR_GT, convert_to_sle=True)
        years, H, sigma = annualize_imbie(df, baseline_year=2005.0)
        return years, H, sigma

    def test_annual_time_steps(self, annual):
        years, _, _ = annual
        dt = np.diff(years)
        assert np.allclose(dt, 1.0, atol=0.1)

    def test_rebased_to_baseline(self, annual):
        """Values should be near zero at the baseline year."""
        years, H, _ = annual
        bl_idx = np.argmin(np.abs(years - 2005.0))
        assert abs(H[bl_idx]) < 1e-10, (
            f"H at baseline = {H[bl_idx]:.2e}, expected exactly zero")

    def test_sigma_positive(self, annual):
        """Annualized sigma should be positive."""
        _, _, sigma = annual
        assert np.all(sigma >= 0)

    def test_units_meters(self, annual):
        """Annualized values should be in meters."""
        _, H, _ = annual
        max_abs = np.max(np.abs(H))
        assert max_abs < 0.5, f"Max |H| = {max_abs:.4f}, too large for meters"
        assert max_abs > 1e-5, f"Max |H| = {max_abs:.2e}, too small for meters"


# =========================================================================
# apply_sigma_taper
# =========================================================================

class TestSigmaTaper:
    """Test sigma inflation taper for early data."""

    def test_no_inflation_after_ref(self):
        years = np.arange(1970, 2020, dtype=float)
        sigma = np.ones(len(years)) * 0.001
        inflated = apply_sigma_taper(sigma, years, t_ref=2000, f_max=3.0)
        post_ref = years >= 2000
        np.testing.assert_array_equal(inflated[post_ref], sigma[post_ref])

    def test_max_inflation_at_start(self):
        years = np.arange(1970, 2020, dtype=float)
        sigma = np.ones(len(years)) * 0.001
        inflated = apply_sigma_taper(sigma, years, t_ref=2000, f_max=3.0)
        assert inflated[0] == pytest.approx(0.003, rel=0.01)

    def test_monotonic_decrease(self):
        years = np.arange(1970, 2020, dtype=float)
        sigma = np.ones(len(years)) * 0.001
        inflated = apply_sigma_taper(sigma, years, t_ref=2000, f_max=3.0)
        pre_ref = years < 2000
        diffs = np.diff(inflated[pre_ref])
        assert np.all(diffs <= 0), "Inflation should decrease toward t_ref"

    def test_fmax_one_no_change(self):
        years = np.arange(1970, 2020, dtype=float)
        sigma = np.ones(len(years)) * 0.001
        inflated = apply_sigma_taper(sigma, years, t_ref=2000, f_max=1.0)
        np.testing.assert_array_equal(inflated, sigma)


# =========================================================================
# compute_component_rates
# =========================================================================

class TestComputeRates:
    """Test rate computation via centred differences."""

    def test_linear_signal(self):
        """Rate of a linear signal should be constant."""
        years = np.arange(1990, 2020, dtype=float)
        H = 0.001 * (years - 2000)  # 1 mm/yr
        rates = compute_component_rates(years, H, window=3)
        valid = ~np.isnan(rates)
        assert np.allclose(rates[valid], 0.001, atol=1e-10)

    def test_edges_nan(self):
        """Edges should be NaN."""
        years = np.arange(1990, 2020, dtype=float)
        H = np.random.default_rng(42).normal(0, 0.01, len(years))
        rates = compute_component_rates(years, H, window=3)
        assert np.all(np.isnan(rates[:3]))
        assert np.all(np.isnan(rates[-3:]))


# =========================================================================
# Unit consistency: GT_TO_M_SLE chain
# =========================================================================

class TestUnitConsistency:
    """Verify conversion constants are self-consistent."""

    def test_gt_to_m_via_mm(self):
        """GT_TO_M_SLE = 1/362500 (1 Gt over 362.5e6 km² ocean → mm, then /1000 → m)."""
        expected = 1.0 / 362.5 / 1000.0
        assert GT_TO_M_SLE == pytest.approx(expected, rel=1e-10)

    def test_m_to_mm_roundtrip(self):
        """M_TO_MM * GT_TO_M_SLE should give mm per Gt."""
        mm_per_gt = M_TO_MM * GT_TO_M_SLE
        expected = 1.0 / 362.5
        assert mm_per_gt == pytest.approx(expected, rel=1e-10)


# =========================================================================
# I/O roundtrip
# =========================================================================

class TestGreenlandIO:
    """Test save_greenland / load_component roundtrip."""

    @pytest.fixture
    def greenland_data(self):
        """Create synthetic Greenland data for I/O testing."""
        rng = np.random.default_rng(RNG_SEED)
        n_obs = 47  # Mouginot years
        n_proj = len(PROJ_YEARS)

        obs_smb_years = np.arange(1972, 2019, dtype=float)
        obs_smb_H = rng.normal(0, 0.005, n_obs)
        obs_smb_sigma = np.full(n_obs, 0.001)
        obs_dyn_years = obs_smb_years.copy()
        obs_dyn_H = rng.normal(0.003, 0.002, n_obs)
        obs_dyn_sigma = np.full(n_obs, 0.0005)

        # Synthetic discharge posteriors
        n_post = 500
        from types import SimpleNamespace
        result_discharge = SimpleNamespace(
            gamma_posterior=rng.normal(0.0005, 0.0001, n_post),
            r0_posterior=rng.normal(0.001, 0.0002, n_post),
            delta_posterior=rng.normal(0.0, 0.001, n_post),
            r2_dyn=0.85,
        )

        # SMB model attributes as the notebook saves them: an EmulatorSummary
        # of the MAR emulator with the drift state (synthetic coefficients)
        gcms = ['GCM-A', 'GCM-B']
        fits = {g: {'beta': rng.normal([0.0, -60.0 - 20 * k, -40.0], 5.0, (n_post, 3)),
                    'segments': ['history', 'ssp585']}
                for k, g in enumerate(gcms)}
        M_draws = rng.normal(390.0, 60.0, n_post)
        drift = {'tau': rng.lognormal(np.log(12.0), 0.3, n_post),
                 'sd_d': rng.uniform(60.0, 180.0, n_post),
                 'sd_e': rng.uniform(10.0, 80.0, n_post)}
        smb_sens = EmulatorSummary({'fits': fits}, M_draws, np.std(M_draws),
                                   noise=False, feedback=True, drift=drift,
                                   M0_obs=374.0)

        ocean_transfer = {
            'alpha': 0.15, 'beta': -0.05,
            'alpha_se': 0.02, 'beta_se': 0.01,
            'r2': 0.72, 'residual_std': 0.08,
            'lag_years': 0.0, 'n': 30,
        }

        n_samples = 100
        greenland_proj = {}
        smb_projections = {}
        discharge_projections = {}
        for ssp in ['SSP2-4.5', 'SSP5-8.5']:
            samples = rng.normal(0.03, 0.01, (n_samples, n_proj))
            greenland_proj[ssp] = {
                'samples': samples,
                'median': np.median(samples, axis=0),
                'p5': np.percentile(samples, 5, axis=0),
                'p17': np.percentile(samples, 17, axis=0),
                'p83': np.percentile(samples, 83, axis=0),
                'p95': np.percentile(samples, 95, axis=0),
                'smb_median': np.median(samples * 0.4, axis=0),
                'dyn_median': np.median(samples * 0.6, axis=0),
            }
            smb_samples = samples * 0.4
            smb_projections[ssp] = {
                'samples': smb_samples,
                'median': np.median(smb_samples, axis=0),
                'p5': np.percentile(smb_samples, 5, axis=0),
                'p17': np.percentile(smb_samples, 17, axis=0),
                'p83': np.percentile(smb_samples, 83, axis=0),
                'p95': np.percentile(smb_samples, 95, axis=0),
            }
            dyn_samples = samples * 0.6
            discharge_projections[ssp] = {
                'samples': dyn_samples,
                'median': np.median(dyn_samples, axis=0),
                'p5': np.percentile(dyn_samples, 5, axis=0),
                'p17': np.percentile(dyn_samples, 17, axis=0),
                'p83': np.percentile(dyn_samples, 83, axis=0),
                'p95': np.percentile(dyn_samples, 95, axis=0),
            }

        return {
            'result_discharge': result_discharge,
            'smb_sensitivity': smb_sens,
            'ocean_transfer': ocean_transfer,
            'obs_smb_years': obs_smb_years,
            'obs_smb_H': obs_smb_H,
            'obs_smb_sigma': obs_smb_sigma,
            'obs_dyn_years': obs_dyn_years,
            'obs_dyn_H': obs_dyn_H,
            'obs_dyn_sigma': obs_dyn_sigma,
            'greenland_proj': greenland_proj,
            'smb_projections': smb_projections,
            'discharge_projections': discharge_projections,
        }

    def test_roundtrip(self, greenland_data, tmp_path):
        """Save and load Greenland data, verify all fields match."""
        h5_path = tmp_path / "test_component_results.h5"

        save_greenland(
            h5_path=h5_path,
            **greenland_data,
        )

        loaded = load_component('greenland', h5_path=h5_path)

        # Check projections
        assert set(loaded['projections'].keys()) == {'SSP2-4.5', 'SSP5-8.5'}
        for ssp in ['SSP2-4.5', 'SSP5-8.5']:
            orig = greenland_data['greenland_proj'][ssp]
            roundtrip = loaded['projections'][ssp]
            np.testing.assert_array_almost_equal(
                roundtrip['median'], orig['median'], decimal=10)
            np.testing.assert_array_almost_equal(
                roundtrip['samples'], orig['samples'], decimal=10)

        # Check sub-component projections
        assert 'projections_smb' in loaded
        assert 'projections_discharge' in loaded
        for ssp in ['SSP2-4.5', 'SSP5-8.5']:
            orig_smb = greenland_data['smb_projections'][ssp]
            roundtrip_smb = loaded['projections_smb'][ssp]
            np.testing.assert_array_almost_equal(
                roundtrip_smb['median'], orig_smb['median'], decimal=10)

        # Check observations (sub-grouped for Greenland)
        assert 'smb' in loaded['observations']
        assert 'discharge' in loaded['observations']
        np.testing.assert_array_almost_equal(
            loaded['observations']['smb']['H_obs'],
            greenland_data['obs_smb_H'], decimal=10)
        np.testing.assert_array_almost_equal(
            loaded['observations']['discharge']['years'],
            greenland_data['obs_dyn_years'], decimal=10)

        # Check posteriors (discharge sub-group)
        assert 'discharge' in loaded['posteriors']
        np.testing.assert_array_almost_equal(
            loaded['posteriors']['discharge']['gamma_posterior'],
            greenland_data['result_discharge'].gamma_posterior, decimal=10)

        # Check SMB model attributes (EmulatorSummary)
        assert 'smb_sensitivity' in loaded
        smb_attrs = loaded['smb_sensitivity']
        es = greenland_data['smb_sensitivity']
        assert smb_attrs['SMB_0'] == pytest.approx(es.SMB_0, rel=1e-12)
        assert smb_attrs['SMB_obs_1995_2005'] == pytest.approx(374.0)
        assert smb_attrs['gcms'] == 'GCM-A,GCM-B'
        assert smb_attrs['drift_tau_median'] == pytest.approx(
            es.extra_attrs['drift_tau_median'], rel=1e-12)
        assert smb_attrs['GCM-B_b1_median'] == pytest.approx(
            es.extra_attrs['GCM-B_b1_median'], rel=1e-12)
        assert 'C_T' not in smb_attrs          # the retired literature model

        # Check ocean transfer
        assert 'ocean_transfer' in loaded
        assert loaded['ocean_transfer']['alpha'] == pytest.approx(
            greenland_data['ocean_transfer']['alpha'], rel=1e-10)

        # Check metadata (the label is set in component_io.save_greenland)
        assert 'discharge_delay' in loaded['metadata']['model_type']

    def test_proj_years_preserved(self, greenland_data, tmp_path):
        """proj_years should match the standard grid."""
        h5_path = tmp_path / "test_component_results.h5"
        save_greenland(h5_path=h5_path, **greenland_data)
        loaded = load_component('greenland', h5_path=h5_path)
        np.testing.assert_array_equal(loaded['proj_years'], PROJ_YEARS)


# =========================================================================
# Discharge delay model fitting
# =========================================================================

class TestDischargeDelayModelWLS:
    """Legacy level estimator of fit_discharge_delay_model (sigma_rate=None).

    Fits H_dm = gamma * int_T_dm + r0 * t_dm by demeaned WLS with diagonal
    weights and the end-of-record rate row, BIC-weighted across candidate
    deltas.  Production has used the increment estimator (sigma_rate given,
    TestDischargeDelayModelIncrements) since 2026-10-17; the level estimator
    remains callable, so these tests keep its arithmetic checked.
    """

    @pytest.fixture
    def synthetic_discharge(self):
        """Generate synthetic discharge data from a known model:
        H(t) = gamma_true * int(T_ocean(t-delta_true)) + r0_true * t + const
        """
        rng = np.random.default_rng(42)

        # Ocean temperature: linear warming + sinusoidal variability
        T_ocean_years = np.arange(1960, 2025, dtype=float) + 0.5
        T_ocean_ann = 0.02 * (T_ocean_years - 1990) + 0.3 * np.sin(
            2 * np.pi * (T_ocean_years - 1960) / 20)

        # True parameters
        gamma_true = 0.0004   # m SLE / (deg-yr)
        r0_true = 0.0002      # m SLE / yr
        delta_true = 5.0      # years

        # Generate synthetic observations
        dyn_years = np.arange(1975, 2020, dtype=float) + 0.5
        T_shifted = np.interp(dyn_years, T_ocean_years + delta_true,
                              T_ocean_ann, left=np.nan, right=np.nan)
        valid = np.isfinite(T_shifted)
        dyn_years = dyn_years[valid]
        T_shifted = T_shifted[valid]

        dt = np.diff(dyn_years, prepend=dyn_years[0] - 1)
        int_T = np.cumsum(T_shifted * dt)

        H_true = gamma_true * int_T + r0_true * (dyn_years - dyn_years.mean())
        noise = rng.normal(0, 0.0005, len(dyn_years))
        H_obs = H_true + noise
        sigma = np.full(len(dyn_years), 0.001)

        return {
            'dyn_years': dyn_years, 'H_obs': H_obs, 'sigma': sigma,
            'T_ocean_ann': T_ocean_ann, 'T_ocean_years': T_ocean_years,
            'gamma_true': gamma_true, 'r0_true': r0_true,
            'delta_true': delta_true,
        }

    @pytest.fixture
    def fitted_result(self, synthetic_discharge):
        """Fit the delay model on synthetic data."""
        d = synthetic_discharge
        return fit_discharge_delay_model(
            dyn_years=d['dyn_years'], H_dyn=d['H_obs'],
            sigma_dyn=d['sigma'],
            T_ocean_ann=d['T_ocean_ann'],
            T_ocean_years=d['T_ocean_years'],
            delta_candidates=[3, 4, 5, 6, 7, 8],
            n_samples=N_SMALL, seed=42,
        )

    def test_recovers_gamma(self, synthetic_discharge, fitted_result):
        """WLS should recover the true gamma within uncertainty."""
        d = synthetic_discharge
        fr = fitted_result.fit_results[d['delta_true']]
        gamma_se = np.sqrt(fr['cov'][0, 0])
        assert fr['gamma'] == pytest.approx(d['gamma_true'], abs=3 * gamma_se), (
            f"gamma = {fr['gamma']:.6f}, expected {d['gamma_true']:.6f} "
            f"(+/- {3*gamma_se:.6f})")

    def test_recovers_r0(self, synthetic_discharge, fitted_result):
        """WLS should recover the true r0 within uncertainty."""
        d = synthetic_discharge
        fr = fitted_result.fit_results[d['delta_true']]
        r0_se = np.sqrt(fr['cov'][1, 1])
        assert fr['r0'] == pytest.approx(d['r0_true'], abs=3 * r0_se)

    def test_positive_r2_rate_at_correct_delta(self, synthetic_discharge, fitted_result):
        """Rate-space R² should be positive when using the correct delta."""
        d = synthetic_discharge
        r2 = fitted_result.fit_results[d['delta_true']]['r2_rate']
        assert r2 > 0.0, f"R²(rate) = {r2:.4f}, expected > 0"

    def test_wrong_delta_worse_r2_rate(self, synthetic_discharge, fitted_result):
        """Rate-space R² at the wrong delta should be lower than at the correct delta."""
        d = synthetic_discharge
        r2_correct = fitted_result.fit_results[d['delta_true']]['r2_rate']
        # delta_true + 3 = 8.0 is still in candidates
        r2_wrong = fitted_result.fit_results[d['delta_true'] + 3]['r2_rate']
        assert r2_correct > r2_wrong

    def test_bic_selects_correct_delta(self, synthetic_discharge, fitted_result):
        """BIC should prefer the correct delta over wrong ones."""
        assert fitted_result.delta_best == synthetic_discharge['delta_true'], (
            f"BIC selected delta={fitted_result.delta_best}, "
            f"expected {synthetic_discharge['delta_true']}")

    def test_covariance_positive_definite(self, synthetic_discharge, fitted_result):
        """WLS covariance should be positive definite."""
        d = synthetic_discharge
        cov = fitted_result.fit_results[d['delta_true']]['cov']
        eigvals = np.linalg.eigvalsh(cov)
        assert np.all(eigvals > 0), f"Covariance not positive definite: {eigvals}"

    def test_posterior_shape(self, fitted_result):
        """Posterior arrays should have n_samples elements."""
        assert len(fitted_result.gamma_posterior) == N_SMALL
        assert len(fitted_result.r0_posterior) == N_SMALL
        assert len(fitted_result.delta_posterior) == N_SMALL

    def test_rate_constraint_shifts_estimate(self, synthetic_discharge):
        """A strong rate constraint should shift the estimate toward
        the specified rate."""
        d = synthetic_discharge
        result_weak = fit_discharge_delay_model(
            d['dyn_years'], d['H_obs'], d['sigma'],
            d['T_ocean_ann'], d['T_ocean_years'],
            delta_candidates=[d['delta_true']],
            rate_constraint_weight=1, n_samples=100, seed=42)
        result_strong = fit_discharge_delay_model(
            d['dyn_years'], d['H_obs'], d['sigma'],
            d['T_ocean_ann'], d['T_ocean_years'],
            delta_candidates=[d['delta_true']],
            rate_constraint_weight=100, n_samples=100, seed=42)
        # Strong constraint should pull r0 closer to observed rate
        r_obs = result_weak.r_obs_dyn
        dist_weak = abs(result_weak.r0_best - r_obs)
        dist_strong = abs(result_strong.r0_best - r_obs)
        assert dist_strong <= dist_weak, (
            "Strong rate constraint did not pull r0 toward observed rate")

    def test_undemeaning_recovers_H(self, synthetic_discharge, fitted_result):
        """Un-demeaning should recover fitted values on calibration data."""
        d = synthetic_discharge
        delta = d['delta_true']
        fr = fitted_result.fit_results[delta]

        T_shifted = np.interp(d['dyn_years'], d['T_ocean_years'] + delta,
                              d['T_ocean_ann'], left=np.nan, right=np.nan)
        valid = np.isfinite(T_shifted)
        yrs_v = d['dyn_years'][valid]
        dt_v = np.diff(yrs_v, prepend=yrs_v[0] - 1)
        int_T = np.cumsum(T_shifted[valid] * dt_v)

        H_reconstructed = (fr['gamma'] * (int_T - fr['int_T_mean'])
                          + fr['r0'] * (yrs_v - fr['t_mean'])
                          + fr['H_mean'])

        H_obs_v = d['H_obs'][valid]
        rms = np.sqrt(np.mean((H_obs_v - H_reconstructed)**2))
        assert rms < 0.002, f"Un-demeaning RMS = {rms:.6f}, expected < 0.002"

    def test_xcorr_computed(self, fitted_result):
        """Cross-correlation should be computed with finite values."""
        assert len(fitted_result.xcorr_lags) == 13  # 0..12
        assert np.all(np.isfinite(fitted_result.xcorr_r))
        assert fitted_result.peak_r > 0  # should find some positive correlation


def _synthetic_increment_record(seed=7, scatter=1.5):
    """Synthetic discharge record built like the Mouginot one: annual rates
    gamma*T(t - delta) + r0 plus independent noise (scatter x the reported
    error), cumulated and rebased with bayesian_models._cumulate."""
    from bayesian_models import _cumulate
    rng = np.random.default_rng(seed)
    T_years = np.arange(1960, 2025, dtype=float) + 0.5
    T_ann = 0.02 * (T_years - 1990) + 0.3 * np.sin(2 * np.pi * (T_years - 1960) / 13)
    gamma, r0, delta = 0.0004, 0.0012, 6.0
    yrs = np.arange(1972, 2019, dtype=float) + 0.5
    sig_rate = np.full(len(yrs), 0.00003)
    rate = (gamma * np.interp(yrs - delta, T_years, T_ann) + r0
            + rng.normal(0, scatter * sig_rate))
    H, sig_H = _cumulate(rate, sig_rate, yrs, (1995, 2005))
    return dict(yrs=yrs, H=H, sig_H=sig_H, sig_rate=sig_rate, T_years=T_years,
                T_ann=T_ann, gamma=gamma, r0=r0, delta=delta)


class TestDischargeDelayModelIncrements:
    """Production estimator: fit_discharge_delay_model with sigma_rate.

    Synthetic record built like the Mouginot one: annual rates
    gamma*T(t - delta) + r0 plus independent noise, cumulated and rebased
    with bayesian_models._cumulate.
    """

    @pytest.fixture
    def synth(self):
        return _synthetic_increment_record()

    @pytest.fixture
    def fit(self, synth):
        d = synth
        return fit_discharge_delay_model(
            d['yrs'], d['H'], d['sig_H'], d['T_ann'], d['T_years'],
            delta_candidates=[4, 5, 6, 7, 8], n_samples=N_SMALL, seed=1,
            sigma_rate=d['sig_rate'])

    def test_recovers_parameters(self, synth, fit):
        fr = fit.fit_results[synth['delta']]
        se = np.sqrt(np.diag(fr['cov']))
        assert abs(fr['gamma'] - synth['gamma']) < 3 * se[0]
        assert abs(fr['r0'] - synth['r0']) < 3 * se[1]

    def test_selects_true_delta(self, synth, fit):
        assert fit.delta_best == synth['delta']

    def test_scale_absorbs_excess_scatter(self, synth, fit):
        fr = fit.fit_results[synth['delta']]
        assert fr['estimator'] == 'increments'
        assert fr['dof'] == fr['n_valid'] - 2
        assert 1.5 < fr['scale'] < 3.5            # true variance ratio 2.25

    def test_equals_gls_on_levels(self, synth, fit):
        """Increment WLS = GLS on the demeaned levels with the _cumulate covariance."""
        from component_greenland_robust_se import _increment_covariance
        d = synth
        fr = fit.fit_results[d['delta']]
        valid = fr['valid_mask']
        yrs_v = d['yrs'][valid]
        dt_v = np.diff(yrs_v, prepend=yrs_v[0] - 1)
        T_v = np.interp(yrs_v, d['T_years'] + d['delta'], d['T_ann'])
        int_T = np.cumsum(T_v * dt_v)
        Sigma, _ = _increment_covariance(yrs_v, d['sig_rate'][valid], dt_v, (1995, 2005))
        # Sigma is singular (the rebase anchor has zero variance) and the
        # level intercept is free, so GLS is taken on the differenced levels
        n = len(yrs_v)
        D = np.diff(np.eye(n), axis=0)
        Sd = D @ Sigma @ D.T
        Xd = D @ np.column_stack([int_T, yrs_v])
        yd = D @ d['H'][valid]
        b = np.linalg.solve(Xd.T @ np.linalg.solve(Sd, Xd), Xd.T @ np.linalg.solve(Sd, yd))
        assert np.allclose(b, [fr['gamma'], fr['r0']], rtol=1e-8)

    def test_rate_check_reported(self, synth, fit):
        fr = fit.fit_results[synth['delta']]
        assert abs(fr['r_model'] - fit.r_obs_dyn) < 5 * fit.sigma_r_obs * np.sqrt(fr['scale'])


class TestExtendWithArc:
    """Test the ARC cumulative discharge extension."""

    @staticmethod
    def _make_mou_comp(H_last=0.01, t_last=2018.5, sigma_last=0.001,
                       last_discharge_rate=0.00138):
        """Build a minimal mou_comp dict for testing."""
        import pandas as pd
        df = pd.DataFrame({
            'decimal_year': [t_last],
            'discharge_rate': [last_discharge_rate],
        })
        return {
            'time_dyn': np.array([t_last]),
            'H_dyn': np.array([H_last]),
            'sigma_dyn': np.array([sigma_last]),
            'df': df,
        }

    def test_single_point(self):
        """With one ARC year, cumulative should increase from H_last."""
        from bayesian_models import extend_with_arc
        from slr_forecast.units import GT_TO_M_SLE
        mou = self._make_mou_comp()
        arc_H, arc_sig, _, _ = extend_with_arc(
            mou, arc_years=np.array([2020.5]),
            arc_D_gtyr=np.array([500.0]),
            arc_D_sig_gtyr=np.array([50.0]),
            gt_to_m_sle=GT_TO_M_SLE,
        )
        assert len(arc_H) == 1
        assert arc_H[0] > 0.01  # should be above starting value
        assert arc_sig[0] > 0   # uncertainty should be positive

    def test_monotonic_cumulative(self):
        """With positive discharge rates, cumulative should increase."""
        from bayesian_models import extend_with_arc
        from slr_forecast.units import GT_TO_M_SLE
        mou = self._make_mou_comp()
        arc_H, _, _, _ = extend_with_arc(
            mou, arc_years=np.array([2020.5, 2023.5]),
            arc_D_gtyr=np.array([500.0, 510.0]),
            arc_D_sig_gtyr=np.array([50.0, 50.0]),
            gt_to_m_sle=GT_TO_M_SLE,
        )
        assert arc_H[1] > arc_H[0]

    def test_uncertainty_grows(self):
        """Cumulative uncertainty should grow with time."""
        from bayesian_models import extend_with_arc
        from slr_forecast.units import GT_TO_M_SLE
        mou = self._make_mou_comp()
        _, arc_sig, _, _ = extend_with_arc(
            mou, arc_years=np.array([2020.5, 2023.5]),
            arc_D_gtyr=np.array([500.0, 500.0]),
            arc_D_sig_gtyr=np.array([50.0, 50.0]),
            gt_to_m_sle=GT_TO_M_SLE,
        )
        assert arc_sig[1] > arc_sig[0]


class TestBICWeightedMixture:
    """BIC-weighted posterior mixing across candidate deltas, checked on the
    output of fit_discharge_delay_model (production increment estimator)."""

    @pytest.fixture(scope='class')
    def fit(self):
        # noisy record (scatter 3x reported): the weight splits about 1:2
        # between delta = 5 and 6, so two deltas carry many draws
        d = _synthetic_increment_record(seed=13, scatter=3.0)
        return fit_discharge_delay_model(
            d['yrs'], d['H'], d['sig_H'], d['T_ann'], d['T_years'],
            delta_candidates=[4, 5, 6, 7, 8], n_samples=N_SMALL, seed=3,
            sigma_rate=d['sig_rate'])

    def test_weights_are_the_bic_formula(self, fit):
        bics = np.array([fit.fit_results[d]['bic'] for d in fit.DELTA_CANDIDATES])
        w = np.exp(-0.5 * (bics - bics.min()))
        np.testing.assert_allclose(fit.bic_weights, w / w.sum(), rtol=1e-12)
        assert fit.bic_weights.sum() == pytest.approx(1.0, abs=1e-12)
        assert fit.DELTA_CANDIDATES[np.argmax(fit.bic_weights)] == fit.delta_best

    def test_delta_draws_follow_weights(self, fit):
        """Each delta gets round(N w) draws, the remainder going to the best."""
        counts = np.array([(fit.delta_posterior == d).sum() for d in fit.DELTA_CANDIDATES])
        assert counts.sum() == N_SMALL
        expected = np.round(N_SMALL * fit.bic_weights).astype(int)
        expected[np.argmax(fit.bic_weights)] += N_SMALL - expected.sum()
        np.testing.assert_array_equal(counts, expected)

    def test_draws_come_from_each_delta_fit(self, fit):
        """Within each delta the draws follow that delta's (gamma, r0) normal."""
        checked = 0
        for d in fit.DELTA_CANDIDATES:
            sel = fit.delta_posterior == d
            if sel.sum() < 200:
                continue
            checked += 1
            fr = fit.fit_results[d]
            se = np.sqrt(np.diag(fr['cov']))
            n = sel.sum()
            assert abs(fit.gamma_posterior[sel].mean() - fr['gamma']) < 5 * se[0] / np.sqrt(n)
            assert abs(fit.r0_posterior[sel].mean() - fr['r0']) < 5 * se[1] / np.sqrt(n)
            assert fit.gamma_posterior[sel].std() == pytest.approx(se[0], rel=0.15)
        assert checked >= 2


class TestCrossCorrelation:
    """The cross-correlation diagnostic of fit_discharge_delay_model
    (discharge rate vs ocean-T rate; not the model relation)."""

    def test_peak_at_true_lag(self):
        """If the cumulative discharge follows T_ocean(t - lag), the
        discharge rate follows the ocean-T rate at that lag."""
        rng = np.random.default_rng(42)
        true_lag = 5
        T_years = np.arange(1960, 2020, dtype=float) + 0.5
        T_ann = 0.02 * (T_years - 1990) + 0.3 * np.sin(2 * np.pi * T_years / 15)
        yrs = np.arange(1972, 2019, dtype=float) + 0.5
        H = 0.01 * np.interp(yrs - true_lag, T_years, T_ann) + rng.normal(0, 1e-6, len(yrs))
        sig = np.full(len(yrs), 1e-4)
        res = fit_discharge_delay_model(yrs, H, sig, T_ann, T_years,
                                        delta_candidates=[4, 5, 6], n_samples=100, seed=0)
        assert res.peak_lag == true_lag
        assert res.peak_r > 0.95

    def test_lag_direction(self):
        """Verify that lag > 0 means ocean T LEADS discharge:
        np.interp(t_dH, t_dT + lag, T) looks up T at physical time
        t_dH - lag, so an ocean impulse at t=2005.5 produces a
        discharge response at t=2005.5 + lag in the output."""
        t = np.arange(2000, 2020, dtype=float) + 0.5
        T = np.zeros(len(t))
        T[5] = 1.0  # impulse at 2005.5

        # np.interp(x, xp=t+3, fp=T) returns T where t+3 ≈ x,
        # i.e., T at physical time x-3.  The impulse at t[5]=2005.5
        # appears in the output at x=2005.5+3=2008.5.
        result = np.interp(t, t + 3, T, left=np.nan, right=np.nan)
        peak_idx = np.nanargmax(result)
        assert t[peak_idx] == pytest.approx(2008.5), (
            f"Impulse appeared at {t[peak_idx]}, expected 2008.5")


class TestOceanTransferFunction:
    """Test the ocean transfer function fitting."""

    def test_perfect_linear_relationship(self):
        """With T_ocean = alpha*T_surface + beta + noise, the fit should
        recover alpha and beta."""
        from component_analysis import fit_ocean_transfer_function

        rng = np.random.default_rng(42)
        n_months = 300
        t = np.arange(n_months) / 12.0 + 1990.0

        alpha_true = 0.15
        beta_true = -0.05
        T_surface = 0.5 * np.sin(2 * np.pi * t / 10) + 0.02 * (t - 2000)
        T_ocean = alpha_true * T_surface + beta_true + rng.normal(0, 0.02, n_months)

        result = fit_ocean_transfer_function(
            T_surface, t, T_ocean, t, lag_years=0, annual=True)

        assert result['alpha'] == pytest.approx(alpha_true, abs=0.05)
        assert result['beta'] == pytest.approx(beta_true, abs=0.05)
        assert result['r2'] > 0.8

    def test_returns_required_keys(self):
        """Result dict should contain all keys needed by the notebook."""
        from component_analysis import fit_ocean_transfer_function

        t = np.arange(120) / 12.0 + 1990.0
        T_s = np.sin(2 * np.pi * t / 5)
        T_o = 0.1 * T_s + 0.01

        result = fit_ocean_transfer_function(T_s, t, T_o, t, annual=True)

        for key in ('alpha', 'beta', 'alpha_se', 'beta_se', 'r2',
                    'residual_std', 'lag_years', 'n'):
            assert key in result, f"Missing key: {key}"
        assert result['alpha_se'] > 0
        assert result['beta_se'] > 0


class TestOceanLowpassTransfer:
    """Low-pass transfer T_ocean = g EMA_tau(T_surface) + c: the production
    form, which is not attenuated by surface variability the ocean ignores."""

    @staticmethod
    def _synthetic(g, tau, c, seed):
        from component_analysis import ema_filter
        rng = np.random.default_rng(seed)
        years = np.arange(1850, 2025)
        # forced trend plus interannual noise 4x the ocean's, as observed
        Ts = 0.004 * (years - 1850) + 0.02 * np.clip(years - 1980, 0, None)
        Ts = Ts + rng.normal(0, 0.8, len(years))
        To = g * ema_filter(Ts, tau) + c
        eps = np.zeros(len(years))
        for k in range(1, len(years)):
            eps[k] = 0.5 * eps[k - 1] + 0.1 * rng.standard_normal()
        To = To + eps
        t_mon = np.repeat(years, 12) + (np.tile(np.arange(12), len(years)) + 0.5) / 12
        oc = years >= 1970
        t_oc = np.repeat(years[oc], 12) + (np.tile(np.arange(12), oc.sum()) + 0.5) / 12
        return t_mon, np.repeat(Ts, 12), t_oc, np.repeat(To[oc], 12)

    def test_ema_filter(self):
        from component_analysis import ema_filter
        x = np.r_[np.zeros(5), np.ones(50)]
        np.testing.assert_array_equal(ema_filter(x, 0), x)
        E = ema_filter(x, 4.0)
        # step response 1 - exp(-k/tau) after the step
        np.testing.assert_allclose(E[5:], 1 - np.exp(-(np.arange(50) + 1) / 4.0))

    def test_recovers_gain_where_annual_ols_is_attenuated(self):
        from component_analysis import (fit_ocean_lowpass_transfer,
                                        fit_ocean_transfer_function)
        g_hat, ols = [], []
        for seed in range(8):
            t_s, Ts, t_o, To = self._synthetic(0.3, 4.0, -0.05, seed)
            tr = fit_ocean_lowpass_transfer(Ts, t_s, To, t_o)
            g_hat.append(tr['alpha'])
            ols.append(fit_ocean_transfer_function(Ts, t_s, To, t_o)['alpha'])
            assert np.isclose(tr['tau_weights'].sum(), 1.0)
        assert np.median(g_hat) == pytest.approx(0.3, abs=0.06)
        assert np.median(ols) < 0.15            # the attenuated annual fit

    def test_draws(self):
        from component_analysis import (fit_ocean_lowpass_transfer,
                                        draw_lowpass_transfer)
        t_s, Ts, t_o, To = self._synthetic(0.3, 4.0, -0.05, 0)
        tr = fit_ocean_lowpass_transfer(Ts, t_s, To, t_o, tau_grid=np.arange(1, 11))
        d = draw_lowpass_transfer(tr, 5000, np.random.default_rng(1))
        assert set(d) == {'tau', 'g', 'c', 'phi', 'sd'}
        assert all(len(v) == 5000 for v in d.values())
        assert set(np.unique(d['tau'])) <= set(tr['tau_grid'])
        # tau frequencies follow the weights
        freq = np.array([(d['tau'] == t).mean() for t in tr['tau_grid']])
        np.testing.assert_allclose(freq, tr['tau_weights'], atol=0.03)


class TestProjectDischargeForcingHold:
    """project_greenland_discharge holds the last annual ocean temperature
    flat after the forcing ends (GR-4), so the discharge rate stays at
    gamma * T_last + r0 instead of dropping to r0."""

    def test_rate_after_forcing_ends(self):
        from types import SimpleNamespace
        from greenland_discharge import project_greenland_discharge
        months = lambda y0, y1: np.arange(y0 * 12, y1 * 12) / 12.0 + 1 / 24
        t_obs = months(1990, 2001)                      # observed GMST, 1990-2000
        t_full = np.concatenate([t_obs, months(2001, 2100)])   # forcing ends Dec 2099
        T_full = np.where(t_full > 2001, 0.02 * (t_full - 2001), 0.0)
        ssp = 'SSPX'
        proj_years = np.arange(1990, 2131, dtype=float)
        gamma, r0, delta = 0.0004, 0.0012, 5.0
        rd = SimpleNamespace(gamma_posterior=np.array([gamma]), r0_posterior=np.array([r0]),
                             delta_posterior=np.array([delta]), int_T_mean_cal=0.0,
                             t_mean_cal=2000.0, H_mean_cal=0.0)
        noise_years = np.arange(2000, 2131)
        out = project_greenland_discharge(
            [ssp], proj_years, {ssp: T_full}, {ssp: t_full},
            {ssp: np.zeros((1, len(t_full)))}, np.zeros(len(t_obs)), t_obs,
            np.zeros(len(t_obs)), t_obs, rd,
            alpha_draws=np.array([1.0]), beta_draws=np.array([0.0]), AA_draws=np.array([1.0]),
            oc_noise=np.zeros((1, len(noise_years))), oc_noise_years=noise_years,
            oc_noise_pre=np.zeros((1, 1)), oc_noise_pre_years=np.array([1989]),
            baseline_year=2000.0, sample_warming_paths=False)[ssp][0]
        T_last = T_full[np.floor(t_full) == 2099].mean()          # last annual value
        rate = np.diff(out)
        after = proj_years[1:] >= 2099.5 + delta + 1               # delayed forcing past its end
        assert after.sum() > 10
        np.testing.assert_allclose(rate[after], gamma * T_last + r0, rtol=1e-10)
        assert np.all(rate[after] > r0 + 0.5 * gamma * T_last)
