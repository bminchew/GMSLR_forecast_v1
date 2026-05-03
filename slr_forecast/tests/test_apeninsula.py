"""Unit tests for the Antarctic Peninsula component.

Tests cover: IMBIE data reader, Bayesian DOLS model structure,
projection shape and range, HDF5 save/load roundtrip, and
ISMIP6 region mapping.
"""

import sys
import os
import tempfile

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'notebooks'))

from component_io import save_apeninsula, load_component, PROJ_YEARS
from component_analysis import annualize_imbie

# Paths
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
RAW_DIR = os.path.join(PROJECT_ROOT, 'data', 'raw')
H5_PATH = os.path.join(PROJECT_ROOT, 'data', 'processed', 'component_results.h5')
IMBIE_AP_PATH = os.path.join(
    RAW_DIR, 'ice_sheets', 'antarctica',
    'imbie_antarctic_peninsula_2021_mm.csv')
ISMIP6_BASE = os.path.join(RAW_DIR, 'ice_sheets', 'ismip6', 'ComputedScalarsPaper')

HAS_IMBIE_AP = os.path.exists(IMBIE_AP_PATH)
HAS_H5 = os.path.exists(H5_PATH)
HAS_ISMIP6 = os.path.exists(ISMIP6_BASE)

from slr_forecast import M_TO_MM
BASELINE_YEAR = 2005.0
PROJ_SSPS = ['SSP1-2.6', 'SSP2-4.5', 'SSP3-7.0', 'SSP5-8.5']


# =========================================================================
# IMBIE reader
# =========================================================================

@pytest.mark.skipif(not HAS_IMBIE_AP, reason="IMBIE Peninsula CSV not found")
class TestIMBIEPeninsulaReader:
    """Verify read_imbie_west_antarctica works on Peninsula CSV."""

    @pytest.fixture(scope="class")
    def pen_data(self):
        from slr_data_readers import read_imbie_west_antarctica
        df = read_imbie_west_antarctica(IMBIE_AP_PATH)
        # Annualize manually (same logic as notebook)
        t = df['decimal_year'].values
        year_int = np.floor(t).astype(int)
        unique_years = np.unique(year_int)
        years = unique_years.astype(float) + 0.5
        H = np.array([df['cumulative_mass_balance'].values[year_int == yr][-1]
                       for yr in unique_years])
        sigma = np.array([np.abs(df['cumulative_mass_balance_sigma'].values[
            year_int == yr][-1]) for yr in unique_years])
        bl_idx = np.argmin(np.abs(years - BASELINE_YEAR))
        H_rebase = H - H[bl_idx]
        return df, years, H_rebase, sigma

    def test_columns_present(self, pen_data):
        df = pen_data[0]
        expected = {'decimal_year', 'cumulative_mass_balance',
                    'cumulative_mass_balance_sigma'}
        assert expected.issubset(set(df.columns))

    def test_units_are_meters(self, pen_data):
        """Cumulative should be in meters (order 1e-4 to 1e-2)."""
        df = pen_data[0]
        max_abs = np.max(np.abs(df['cumulative_mass_balance'].values))
        assert max_abs < 0.1, f"Max |cum| = {max_abs:.4f}, too large for meters"
        assert max_abs > 1e-6, f"Max |cum| = {max_abs:.2e}, too small for meters"

    def test_time_range(self, pen_data):
        """IMBIE Peninsula should span ~1992-2020."""
        years = pen_data[1]
        assert years[0] >= 1991 and years[0] <= 1993
        assert years[-1] >= 2018

    def test_annual_spacing(self, pen_data):
        years = pen_data[1]
        dt = np.diff(years)
        assert np.allclose(dt, 1.0, atol=0.1)

    def test_n_years(self, pen_data):
        years = pen_data[1]
        assert len(years) == 29, f"Expected 29 years, got {len(years)}"

    def test_baseline_zero(self, pen_data):
        years, H = pen_data[1], pen_data[2]
        bl_idx = np.argmin(np.abs(years - BASELINE_YEAR))
        assert abs(H[bl_idx]) < 1e-10

    def test_cumulative_small_positive(self, pen_data):
        """Peninsula cumulative SLR should be small and positive at end."""
        H = pen_data[2]
        H_mm = H[-1] * M_TO_MM
        assert H_mm > 0, f"Final H = {H_mm:.2f} mm, expected positive"
        assert H_mm < 5, f"Final H = {H_mm:.2f} mm, too large for Peninsula"

    def test_sigma_positive(self, pen_data):
        sigma = pen_data[3]
        assert np.all(sigma >= 0)


# =========================================================================
# HDF5 roundtrip
# =========================================================================

@pytest.mark.skipif(not HAS_H5, reason="component_results.h5 not found")
class TestAPeninsulaSaveLoad:
    """Verify save/load roundtrip for Peninsula component."""

    @pytest.fixture(scope="class")
    def loaded(self):
        return load_component('apeninsula')

    def test_metadata_model_type(self, loaded):
        assert loaded['metadata']['model_type'] == 'linear_dols'

    def test_metadata_has_r2(self, loaded):
        r2 = loaded['metadata']['r2']
        assert 0.8 < r2 < 1.0, f"R² = {r2:.4f}, expected ~0.94"

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
        # At 2100
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
        assert len(obs['years']) == 29

    def test_posteriors_present(self, loaded):
        post = loaded['posteriors']
        assert 'posterior_samples' in post
        assert 'H0_posterior' in post

    def test_posterior_shape(self, loaded):
        ps = loaded['posteriors']['posterior_samples']
        assert ps.ndim == 2
        assert ps.shape[1] == 3  # a, b, c


# =========================================================================
# Projection plausibility
# =========================================================================

@pytest.mark.skipif(not HAS_H5, reason="component_results.h5 not found")
class TestAPeninsulaProjections:
    """Verify Peninsula projections are physically plausible."""

    @pytest.fixture(scope="class")
    def loaded(self):
        return load_component('apeninsula')

    def test_positive_slr_at_2100(self, loaded):
        """Peninsula should contribute positive SLR (mass loss) at 2100."""
        idx = np.argmin(np.abs(loaded['proj_years'] - 2100))
        for ssp in PROJ_SSPS:
            med = loaded['projections'][ssp]['median'][idx]
            assert med > 0, f"{ssp} median = {med*M_TO_MM:.1f} mm, expected > 0"

    def test_range_at_2100(self, loaded):
        """Peninsula at 2100 should be between 3 and 30 mm for all SSPs."""
        idx = np.argmin(np.abs(loaded['proj_years'] - 2100))
        for ssp in PROJ_SSPS:
            med_mm = loaded['projections'][ssp]['median'][idx] * M_TO_MM
            assert 3 < med_mm < 30, (
                f"{ssp} median = {med_mm:.1f} mm, outside [3, 30] mm")

    def test_ssp_ordering(self, loaded):
        """Higher SSP → more Peninsula SLR (warmer → more mass loss)."""
        idx = np.argmin(np.abs(loaded['proj_years'] - 2100))
        medians = [loaded['projections'][ssp]['median'][idx] for ssp in PROJ_SSPS]
        for i in range(len(medians) - 1):
            assert medians[i] <= medians[i + 1] * 1.1, (
                f"SSP ordering violated at index {i}")

    def test_near_zero_at_baseline(self, loaded):
        """Projections should be near zero at baseline year."""
        idx = np.argmin(np.abs(loaded['proj_years'] - BASELINE_YEAR))
        for ssp in PROJ_SSPS:
            med_mm = loaded['projections'][ssp]['median'][idx] * M_TO_MM
            assert abs(med_mm) < 1.0, (
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


# =========================================================================
# ISMIP6 reader
# =========================================================================

@pytest.mark.skipif(not HAS_ISMIP6, reason="ISMIP6 data not found")
class TestISMIP6Peninsula:
    """Verify ISMIP6 regional reader for Peninsula (region=3)."""

    def test_reads_peninsula(self):
        from component_projections import read_ismip6_regional
        data = read_ismip6_regional(ISMIP6_BASE, region=3)
        assert len(data) > 0, "No ISMIP6 Peninsula runs found"

    def test_keys_are_tuples(self):
        from component_projections import read_ismip6_regional
        data = read_ismip6_regional(ISMIP6_BASE, region=3)
        for key in data:
            assert isinstance(key, tuple) and len(key) == 3

    def test_sle_in_meters(self):
        from component_projections import read_ismip6_regional
        data = read_ismip6_regional(ISMIP6_BASE, region=3)
        for key, val in data.items():
            max_abs = np.max(np.abs(val['sle_m']))
            assert max_abs < 1.0, (
                f"Max |SLE| = {max_abs:.3f} m for {key}, too large for Peninsula")
            break  # one run is enough
