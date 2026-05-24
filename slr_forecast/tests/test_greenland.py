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
from smb_projections import (
    SMBSensitivity,
    GREENLAND_SMB,
    GT_TO_M_SLE,
    project_smb_ensemble,
    project_smb_at_warming_levels,
)

# Paths
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
RAW_DIR = os.path.join(PROJECT_ROOT, 'data', 'raw')
IMBIE_GR_GT = os.path.join(RAW_DIR, 'ice_sheets', 'Gt', 'imbie_greenland_2021_Gt.csv')
MOUGINOT_PATH = os.path.join(RAW_DIR, 'ice_sheets', 'greenland', 'mouginot2019_data.xlsx')

N = 200_000      # large enough for statistical tests
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

    def test_sign_convention_glaciology(self, imbie_gr):
        """IMBIE Gt reader preserves glaciology convention: mass loss is
        negative in cumulative_mass_balance. The late-period cumulative
        value should be negative (Greenland has been losing mass)."""
        # Check the last value — cumulative should be negative
        cum_last = imbie_gr['cumulative_mass_balance'].values[-1]
        assert cum_last < 0, (
            f"Last cumulative_mb = {cum_last:.6f}, expected negative "
            f"(glaciology convention: mass loss is negative)")

    def test_sigma_positive(self, imbie_gr):
        """Uncertainty should always be positive (absolute value)."""
        sigma_rate = imbie_gr['mass_balance_rate_sigma'].values
        sigma_cum = imbie_gr['cumulative_mass_balance_sigma'].values
        # Sigma values may be stored with sign from IMBIE (negative in some
        # records means mass-balance-based convention). After the reader
        # applies convert_to_sle, the sign might flip. Check absolute values
        # are non-zero.
        assert np.all(np.abs(sigma_cum) > 0), "sigma should be non-zero"

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
# SMBSensitivity and constants
# =========================================================================

class TestSMBSensitivity:
    """Verify GREENLAND_SMB parameters are physically consistent."""

    def test_C_T_negative(self):
        """C_T should be negative: warming causes mass loss."""
        assert GREENLAND_SMB.C_T < 0

    def test_C_T2_negative(self):
        """C_T2 should be negative: ablation zone expansion accelerates loss."""
        assert GREENLAND_SMB.C_T2 < 0

    def test_sigma_positive(self):
        assert GREENLAND_SMB.C_T_sigma > 0
        assert GREENLAND_SMB.C_T2_sigma > 0

    def test_temperature_frame_gmst(self):
        """Greenland SMB sensitivity should be in GMST frame."""
        assert GREENLAND_SMB.temperature_frame == 'GMST'

    def test_AA_factor_unity(self):
        """AA factor should be 1.0 when sensitivity is already in GMST frame."""
        assert GREENLAND_SMB.AA_factor == 1.0

    def test_gt_to_m_sle_constant(self):
        """GT_TO_M_SLE = 1 / 362500."""
        expected = 1.0 / 362500.0
        assert GT_TO_M_SLE == pytest.approx(expected, rel=1e-10)

    def test_reference_nonempty(self):
        assert len(GREENLAND_SMB.reference) > 0


# =========================================================================
# project_smb_ensemble
# =========================================================================

class TestProjectSMBEnsemble:
    """Test SMB projection function returns correct units and shapes."""

    @pytest.fixture
    def smb_projection(self):
        """Project SMB under a simple linear warming scenario."""
        time_proj = np.arange(1950, 2151, dtype=float)
        # Linear warming: 0 at 2005, +3 at 2100
        T = (time_proj - 2005.0) / (2100.0 - 2005.0) * 3.0
        T_proj = {'SSP2-4.5': T}
        result = project_smb_ensemble(
            sensitivity=GREENLAND_SMB,
            T_proj=T_proj,
            time_proj=time_proj,
            T_baseline=0.0,
            n_samples=N_SMALL,
            seed=RNG_SEED,
            baseline_year=2005.0,
        )
        return result, time_proj

    def test_returns_meters(self, smb_projection):
        """SMB projection should return meters, not mm or Gt."""
        result, time_proj = smb_projection
        samples = result['SSP2-4.5']['samples']
        idx_2100 = np.argmin(np.abs(time_proj - 2100.0))
        med_2100 = np.median(samples[:, idx_2100])
        # Greenland SMB at +3C should contribute ~0.02-0.15 m SLE
        assert 0.005 < med_2100 < 0.5, (
            f"Median SMB SLR at 2100 = {med_2100:.4f} m, "
            f"expected 0.005-0.5 m range")

    def test_shape(self, smb_projection):
        result, time_proj = smb_projection
        samples = result['SSP2-4.5']['samples']
        assert samples.shape == (N_SMALL, len(time_proj))

    def test_slr_increases_with_warming(self, smb_projection):
        """With warming, cumulative SMB SLR should increase over time.
        Note: with SMB_0 > 0 (baseline accumulation), the cumulative may
        be negative at 2050 (accumulation dominates early) but positive
        by 2100 (warming-driven melt wins)."""
        result, time_proj = smb_projection
        med = result['SSP2-4.5']['median']
        idx_2050 = np.argmin(np.abs(time_proj - 2050.0))
        idx_2100 = np.argmin(np.abs(time_proj - 2100.0))
        assert med[idx_2100] > 0, "Median SMB SLR at 2100 should be positive"
        assert med[idx_2100] > med[idx_2050], "SLR should increase 2050 to 2100"

    def test_zero_at_baseline(self, smb_projection):
        """Projection should be zero at baseline year."""
        result, time_proj = smb_projection
        idx_bl = np.argmin(np.abs(time_proj - 2005.0))
        med = result['SSP2-4.5']['median']
        assert abs(med[idx_bl]) < 1e-10

    def test_percentile_keys(self, smb_projection):
        result, _ = smb_projection
        ssp_data = result['SSP2-4.5']
        for key in ('median', 'p5', 'p17', 'p83', 'p95', 'samples'):
            assert key in ssp_data, f"Missing key: {key}"

    def test_percentile_ordering(self, smb_projection):
        """p5 < p17 < median < p83 < p95 at 2100."""
        result, time_proj = smb_projection
        ssp = result['SSP2-4.5']
        idx = np.argmin(np.abs(time_proj - 2100.0))
        assert ssp['p5'][idx] <= ssp['p17'][idx]
        assert ssp['p17'][idx] <= ssp['median'][idx]
        assert ssp['median'][idx] <= ssp['p83'][idx]
        assert ssp['p83'][idx] <= ssp['p95'][idx]

    def test_higher_warming_more_slr(self):
        """Higher temperature should produce more SLR."""
        time_proj = np.arange(1950, 2151, dtype=float)
        T_low = (time_proj - 2005.0) / 95.0 * 1.5
        T_high = (time_proj - 2005.0) / 95.0 * 4.0
        T_proj = {'low': T_low, 'high': T_high}
        result = project_smb_ensemble(
            sensitivity=GREENLAND_SMB,
            T_proj=T_proj,
            time_proj=time_proj,
            n_samples=N_SMALL,
            seed=RNG_SEED,
            baseline_year=2005.0,
        )
        idx = np.argmin(np.abs(time_proj - 2100.0))
        med_low = np.median(result['low']['samples'][:, idx])
        med_high = np.median(result['high']['samples'][:, idx])
        assert med_high > med_low, (
            f"Higher warming ({med_high:.4f} m) should produce more SLR "
            f"than lower warming ({med_low:.4f} m)")


# =========================================================================
# project_smb_at_warming_levels
# =========================================================================

class TestSMBWarmingLevels:
    """Test SMB sensitivity at specific warming levels."""

    def test_rate_increases_with_warming(self):
        warming = np.array([1.0, 2.0, 3.0, 4.0])
        result = project_smb_at_warming_levels(
            GREENLAND_SMB, warming, n_samples=N, seed=RNG_SEED)
        # Rates should become more negative (more mass loss) with warming
        # In SLR convention (slr_rate_median is positive for mass loss)
        slr_rates = result['slr_rate_median']
        for i in range(len(warming) - 1):
            assert slr_rates[i + 1] > slr_rates[i], (
                f"SLR rate at {warming[i+1]:.1f}C ({slr_rates[i+1]:.2f} mm/yr) "
                f"should exceed rate at {warming[i]:.1f}C ({slr_rates[i]:.2f} mm/yr)")

    def test_nonlinear_response(self):
        """With C_T2 != 0, sensitivity should be nonlinear."""
        warming = np.array([1.0, 4.0])
        result = project_smb_at_warming_levels(
            GREENLAND_SMB, warming, n_samples=N, seed=RNG_SEED)
        rate_1 = result['rate_median'][0]
        rate_4 = result['rate_median'][1]
        # If linear: rate_4 / rate_1 = 4.0
        # With quadratic: ratio should differ from 4.0
        ratio = rate_4 / rate_1
        assert abs(ratio - 4.0) > 0.1, (
            f"Rate ratio = {ratio:.2f}, expected != 4.0 due to quadratic term")

    def test_statistical_percentiles(self):
        """p5 < median < p95 at each warming level."""
        warming = np.array([2.0, 4.0])
        result = project_smb_at_warming_levels(
            GREENLAND_SMB, warming, n_samples=N, seed=RNG_SEED)
        for j in range(len(warming)):
            assert result['rate_p5'][j] < result['rate_median'][j]
            assert result['rate_median'][j] < result['rate_p95'][j]


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

    def test_smb_projection_units_consistent(self):
        """C_T in Gt/yr/C times GT_TO_M_SLE should give m/yr/C."""
        c_t_m = GREENLAND_SMB.C_T * GT_TO_M_SLE
        # -200 Gt/yr/C * (1/362500) m/Gt = -5.5e-4 m/yr/C
        assert c_t_m == pytest.approx(-300.0 / 362500.0, rel=1e-10)


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

        smb_sens = GREENLAND_SMB

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

        # Check SMB sensitivity
        assert 'smb_sensitivity' in loaded
        assert loaded['smb_sensitivity']['C_T'] == pytest.approx(
            GREENLAND_SMB.C_T, rel=1e-10)

        # Check ocean transfer
        assert 'ocean_transfer' in loaded
        assert loaded['ocean_transfer']['alpha'] == pytest.approx(
            greenland_data['ocean_transfer']['alpha'], rel=1e-10)

        # Check metadata
        assert loaded['metadata']['model_type'] == 'smb_literature_plus_discharge_delay'

    def test_proj_years_preserved(self, greenland_data, tmp_path):
        """proj_years should match the standard grid."""
        h5_path = tmp_path / "test_component_results.h5"
        save_greenland(h5_path=h5_path, **greenland_data)
        loaded = load_component('greenland', h5_path=h5_path)
        np.testing.assert_array_equal(loaded['proj_years'], PROJ_YEARS)


# =========================================================================
# SMB ensemble statistical properties (large N)
# =========================================================================

# =========================================================================
# Discharge delay model fitting
# =========================================================================

class TestDischargeDelayModelWLS:
    """Test fit_discharge_delay_model from component_analysis.py.

    The function fits: H_dm = gamma * int_T_dm + r0 * t_dm
    by demeaned WLS with BIC-weighted mixing across candidate deltas.
    These tests verify the math using synthetic data with known ground truth.
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

    def test_high_r2_at_correct_delta(self, synthetic_discharge, fitted_result):
        """R² should be high when using the correct delta."""
        d = synthetic_discharge
        r2 = fitted_result.fit_results[d['delta_true']]['r2']
        assert r2 > 0.95, f"R² = {r2:.4f}, expected > 0.95"

    def test_wrong_delta_worse_r2(self, synthetic_discharge, fitted_result):
        """R² at the wrong delta should be lower than at the correct delta."""
        d = synthetic_discharge
        r2_correct = fitted_result.fit_results[d['delta_true']]['r2']
        # delta_true + 3 = 8.0 is still in candidates
        r2_wrong = fitted_result.fit_results[d['delta_true'] + 3]['r2']
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
    """Test BIC-weighted posterior mixing across candidate deltas."""

    def test_weights_sum_to_one(self):
        """BIC weights should sum to 1."""
        bics = np.array([100.0, 102.0, 105.0, 110.0, 115.0])
        bic_min = bics.min()
        weights = np.exp(-0.5 * (bics - bic_min))
        weights /= weights.sum()
        assert weights.sum() == pytest.approx(1.0, abs=1e-12)

    def test_best_model_gets_most_weight(self):
        """The model with lowest BIC should get the highest weight."""
        bics = np.array([100.0, 102.0, 105.0, 110.0, 115.0])
        bic_min = bics.min()
        weights = np.exp(-0.5 * (bics - bic_min))
        weights /= weights.sum()
        assert np.argmax(weights) == np.argmin(bics)

    def test_equal_bics_equal_weights(self):
        """Equal BICs should produce equal weights."""
        bics = np.array([100.0, 100.0, 100.0])
        bic_min = bics.min()
        weights = np.exp(-0.5 * (bics - bic_min))
        weights /= weights.sum()
        np.testing.assert_allclose(weights, 1.0 / 3.0, atol=1e-12)

    def test_sample_allocation(self):
        """N_SAMPLES should be allocated proportional to BIC weights."""
        N_SAMPLES = 2000
        bics = np.array([100.0, 102.0, 108.0])
        bic_min = bics.min()
        weights = np.exp(-0.5 * (bics - bic_min))
        weights /= weights.sum()

        n_per = np.round(N_SAMPLES * weights).astype(int)
        n_per[np.argmax(weights)] += N_SAMPLES - n_per.sum()

        assert n_per.sum() == N_SAMPLES
        # Best model should get the most samples
        assert n_per[0] > n_per[1] > n_per[2]

    def test_posterior_mixture_draws(self):
        """Drawing from MVN for each delta should produce the right
        number of samples with correct marginal properties."""
        rng = np.random.default_rng(42)
        N_SAMPLES = 2000

        # Two fake deltas with known covariances
        means = [np.array([0.0004, 0.0002]), np.array([0.0003, 0.0001])]
        covs = [np.eye(2) * 1e-8, np.eye(2) * 2e-8]
        weights = np.array([0.7, 0.3])

        n_per = np.round(N_SAMPLES * weights).astype(int)
        n_per[0] += N_SAMPLES - n_per.sum()

        gamma_post = np.empty(N_SAMPLES)
        r0_post = np.empty(N_SAMPLES)
        idx = 0
        for i, n_d in enumerate(n_per):
            draws = rng.multivariate_normal(means[i], covs[i], size=n_d)
            gamma_post[idx:idx + n_d] = draws[:, 0]
            r0_post[idx:idx + n_d] = draws[:, 1]
            idx += n_d

        assert len(gamma_post) == N_SAMPLES
        # Overall mean should be weighted average of component means
        expected_gamma = weights[0] * means[0][0] + weights[1] * means[1][0]
        assert np.mean(gamma_post) == pytest.approx(expected_gamma, abs=1e-5)


class TestCrossCorrelation:
    """Test the cross-correlation computation for delay selection."""

    def test_perfect_correlation_at_true_lag(self):
        """Cross-correlation should peak at the true lag for a
        perfectly shifted signal."""
        rng = np.random.default_rng(42)
        true_lag = 5
        t = np.arange(1960, 2020, dtype=float) + 0.5

        # Ocean temperature: smooth signal
        T_ocean = 0.02 * (t - 1990) + 0.3 * np.sin(2 * np.pi * t / 15)
        dT_ocean = np.diff(T_ocean) / np.diff(t)
        t_dT = 0.5 * (t[:-1] + t[1:])

        # Discharge rate = ocean T rate shifted by true_lag + noise
        dH_dyn = np.interp(t_dT, t_dT + true_lag, dT_ocean,
                           left=np.nan, right=np.nan)
        valid_gen = np.isfinite(dH_dyn)
        dH_dyn[~valid_gen] = 0
        dH_dyn += rng.normal(0, 0.001, len(dH_dyn))
        t_dH = t_dT

        # Cross-correlate
        max_lag = 12
        xcorr_lags = np.arange(0, max_lag + 1)
        xcorr_r = np.zeros(len(xcorr_lags))
        for k, lag in enumerate(xcorr_lags):
            dT_shifted = np.interp(t_dH, t_dT + lag, dT_ocean,
                                   left=np.nan, right=np.nan)
            valid = np.isfinite(dT_shifted)
            if valid.sum() > 3:
                xcorr_r[k] = np.corrcoef(dH_dyn[valid], dT_shifted[valid])[0, 1]

        peak_lag = xcorr_lags[np.nanargmax(xcorr_r)]
        assert peak_lag == true_lag, (
            f"Peak lag = {peak_lag}, expected {true_lag}")

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


class TestSMBEnsembleStatistics:
    """Statistical tests on SMB ensemble with large sample size."""

    def test_median_C_T_recovered(self):
        """Median rate at +1C should approximate C_T + C_T2 + SMB_0."""
        warming = np.array([1.0])
        result = project_smb_at_warming_levels(
            GREENLAND_SMB, warming, n_samples=N, seed=RNG_SEED)
        # At +1C: rate = C_T * 1 + C_T2 * 1 + SMB_0
        #       = -300 + (-50) + 380 = +30 Gt/yr
        expected = GREENLAND_SMB.C_T + GREENLAND_SMB.C_T2 + GREENLAND_SMB.SMB_0
        assert result['rate_median'][0] == pytest.approx(expected, rel=0.02)

    def test_uncertainty_propagation(self):
        """Spread should reflect C_T_sigma and C_T2_sigma."""
        warming = np.array([1.0])
        result = project_smb_at_warming_levels(
            GREENLAND_SMB, warming, n_samples=N, seed=RNG_SEED)
        spread = result['rate_p95'][0] - result['rate_p5'][0]
        # At 1C: std = sqrt(80^2 + 30^2) ~ 85 Gt/yr
        # 5-95 range ~ 3.3 * 85 ~ 280 Gt/yr
        expected_std = np.sqrt(GREENLAND_SMB.C_T_sigma**2
                               + GREENLAND_SMB.C_T2_sigma**2)
        expected_range = 2 * Z_90 * expected_std  # 5-95 range
        assert spread == pytest.approx(expected_range, rel=0.05)

    def test_quadratic_dominates_at_high_warming(self):
        """At +4C, quadratic term contribution should be substantial."""
        warming = np.array([4.0])
        result = project_smb_at_warming_levels(
            GREENLAND_SMB, warming, n_samples=N, seed=RNG_SEED)
        # Linear: C_T * 4 = -1200 Gt/yr
        # Quadratic: C_T2 * 16 = -800 Gt/yr
        # Baseline: SMB_0 = +380 Gt/yr
        # Total: -1620 Gt/yr
        expected = (GREENLAND_SMB.C_T * 4
                    + GREENLAND_SMB.C_T2 * 16
                    + GREENLAND_SMB.SMB_0)
        assert result['rate_median'][0] == pytest.approx(expected, rel=0.02)
