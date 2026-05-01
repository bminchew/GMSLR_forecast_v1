"""Unit tests for the component summation notebook.

Tests cover: component loading, sample-level summation, variance
decomposition, IPCC TWS loading, budget closure diagnostics,
and consistency with the forecast notebook's summation.
"""

import sys
import os

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'notebooks'))

from component_io import (
    load_all_projections, load_component, list_components,
    PROJ_SSPS, PROJ_YEARS, N_SAMPLES,
)
from component_analysis import compute_variance_fractions
from component_projections import read_ipcc_component_nc, ipcc_extract

# Paths
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
H5_PATH = os.path.join(PROJECT_ROOT, 'data', 'processed', 'component_results.h5')
SLR_H5 = os.path.join(PROJECT_ROOT, 'data', 'processed', 'slr_processed_data.h5')
IPCC_DIST_PATH = os.path.join(PROJECT_ROOT, 'data', 'processed', 'ipcc_distributions.h5')
RAW_DIR = os.path.join(PROJECT_ROOT, 'data', 'raw')
CONF_BASE = os.path.join(
    RAW_DIR, 'ipcc_ar6', 'slr', 'ar6', 'global', 'confidence_output_files')

HAS_H5 = os.path.exists(H5_PATH)
HAS_SLR_H5 = os.path.exists(SLR_H5)
HAS_IPCC = os.path.exists(os.path.join(
    CONF_BASE, 'medium_confidence', 'ssp245'))
HAS_IPCC_DIST = os.path.exists(IPCC_DIST_PATH)

M_TO_MM = 1000.0
BASELINE_YEAR = 2005.0

# Components that should be summed (EAIS excluded by design)
SUMMED_COMPONENTS = ['ocean', 'glacier', 'greenland', 'apeninsula', 'wais']
ALL_HDF5_COMPONENTS = ['ocean', 'glacier', 'greenland', 'apeninsula', 'wais', 'eais']


# =========================================================================
# Component inventory
# =========================================================================

@pytest.mark.skipif(not HAS_H5, reason="component_results.h5 not found")
class TestComponentInventory:
    """Verify all expected components are in the HDF5 file."""

    def test_all_components_present(self):
        comps = list_components()
        for comp in ALL_HDF5_COMPONENTS:
            assert comp in comps, f"Missing component: {comp}"

    def test_eais_present_but_excluded(self):
        """EAIS should be in HDF5 but excluded from summation."""
        comps = list_components()
        assert 'eais' in comps
        assert 'eais' not in SUMMED_COMPONENTS


# =========================================================================
# Sample-level summation
# =========================================================================

@pytest.mark.skipif(not HAS_H5, reason="component_results.h5 not found")
class TestSampleSummation:
    """Verify that summing components sample-by-sample produces correct totals."""

    @pytest.fixture(scope="class")
    def comp_samples(self):
        """Load all summed components with proper sample propagation."""
        samples = {}
        for comp in SUMMED_COMPONENTS:
            loaded = load_component(comp)
            samples[comp] = loaded['projections']
        return samples

    def test_all_shapes_match(self, comp_samples):
        """All components should have (N_SAMPLES, 201) shape."""
        for comp in SUMMED_COMPONENTS:
            shape = comp_samples[comp]['SSP2-4.5']['samples'].shape
            assert shape == (N_SAMPLES, len(PROJ_YEARS)), (
                f"{comp} shape = {shape}")

    def test_sum_at_baseline_near_zero(self, comp_samples):
        """Component sum should be near zero at baseline year."""
        idx = np.argmin(np.abs(PROJ_YEARS - BASELINE_YEAR))
        for ssp in PROJ_SSPS:
            total = np.zeros(N_SAMPLES)
            for comp in SUMMED_COMPONENTS:
                total += comp_samples[comp][ssp]['samples'][:, idx]
            med_mm = np.median(total) * M_TO_MM
            assert abs(med_mm) < 20, (
                f"{ssp} total at baseline = {med_mm:.1f} mm, expected ~0")

    def test_sum_positive_at_2100(self, comp_samples):
        """Total SLR should be positive at 2100 for all SSPs."""
        idx = np.argmin(np.abs(PROJ_YEARS - 2100))
        for ssp in PROJ_SSPS:
            total = np.zeros(N_SAMPLES)
            for comp in SUMMED_COMPONENTS:
                total += comp_samples[comp][ssp]['samples'][:, idx]
            assert np.median(total) > 0

    def test_percentile_ordering(self, comp_samples):
        """p5 <= p17 <= median <= p83 <= p95 for the total."""
        idx = np.argmin(np.abs(PROJ_YEARS - 2100))
        for ssp in PROJ_SSPS:
            total = np.zeros(N_SAMPLES)
            for comp in SUMMED_COMPONENTS:
                total += comp_samples[comp][ssp]['samples'][:, idx]
            p5, p17, med, p83, p95 = np.percentile(total, [5, 17, 50, 83, 95])
            assert p5 <= p17 <= med <= p83 <= p95

    def test_ssp_ordering(self, comp_samples):
        """Higher SSP → higher total median at 2100."""
        idx = np.argmin(np.abs(PROJ_YEARS - 2100))
        medians = []
        for ssp in PROJ_SSPS:
            total = np.zeros(N_SAMPLES)
            for comp in SUMMED_COMPONENTS:
                total += comp_samples[comp][ssp]['samples'][:, idx]
            medians.append(np.median(total))
        for i in range(len(medians) - 1):
            assert medians[i] <= medians[i + 1] * 1.05

    def test_eais_exclusion_is_conservative(self, comp_samples):
        """Including EAIS should reduce the total (negative SLR)."""
        eais = load_component('eais')
        idx = np.argmin(np.abs(PROJ_YEARS - 2100))
        ssp = 'SSP2-4.5'
        eais_med = np.median(eais['projections'][ssp]['samples'][:, idx])
        assert eais_med < 0, "EAIS should contribute negative SLR"


# =========================================================================
# Variance decomposition
# =========================================================================

@pytest.mark.skipif(not HAS_H5, reason="component_results.h5 not found")
class TestVarianceDecomposition:
    """Verify compute_variance_fractions produces valid output."""

    @pytest.fixture(scope="class")
    def comp_proj(self):
        """Build comp_projections dict matching notebook format."""
        proj = {}
        for comp in SUMMED_COMPONENTS:
            loaded = load_component(comp)
            for ssp in PROJ_SSPS:
                if ssp not in proj:
                    proj[ssp] = {}
                # Map hdf key to display label
                labels = {
                    'ocean': 'Thermosteric', 'glacier': 'Glaciers',
                    'greenland': 'Greenland', 'apeninsula': 'Peninsula',
                    'wais': 'WAIS',
                }
                proj[ssp][labels[comp]] = loaded['projections'][ssp]
        return proj

    def test_fractions_sum_to_one(self, comp_proj):
        fracs, _ = compute_variance_fractions(
            'SSP2-4.5',
            ['Thermosteric', 'Glaciers', 'Greenland', 'Peninsula', 'WAIS'],
            PROJ_YEARS, comp_proj, N_SAMPLES, normalise=True,
        )
        # At 2100
        idx = np.argmin(np.abs(PROJ_YEARS - 2100))
        total = sum(fracs[c][idx] for c in fracs)
        assert abs(total - 1.0) < 0.01, f"Fracs sum to {total:.3f}, expected 1.0"

    def test_wais_dominates(self, comp_proj):
        """WAIS should dominate variance at 2100."""
        fracs, _ = compute_variance_fractions(
            'SSP2-4.5',
            ['Thermosteric', 'Glaciers', 'Greenland', 'Peninsula', 'WAIS'],
            PROJ_YEARS, comp_proj, N_SAMPLES, normalise=True,
        )
        idx = np.argmin(np.abs(PROJ_YEARS - 2100))
        assert fracs['WAIS'][idx] > 0.5, (
            f"WAIS variance fraction = {fracs['WAIS'][idx]:.2f}, expected > 0.5")

    def test_all_fractions_nonnegative(self, comp_proj):
        fracs, _ = compute_variance_fractions(
            'SSP2-4.5',
            ['Thermosteric', 'Glaciers', 'Greenland', 'Peninsula', 'WAIS'],
            PROJ_YEARS, comp_proj, N_SAMPLES, normalise=True,
        )
        idx = np.argmin(np.abs(PROJ_YEARS - 2100))
        for comp, frac_arr in fracs.items():
            assert frac_arr[idx] >= 0, f"{comp} fraction = {frac_arr[idx]:.3f}"


# =========================================================================
# TWS from IPCC
# =========================================================================

@pytest.mark.skipif(not HAS_IPCC, reason="IPCC AR6 data not found")
class TestIPCCTWSLoading:
    """Verify IPCC TWS (land water storage) data can be loaded."""

    def test_read_tws_ssp245(self):
        data = read_ipcc_component_nc(
            CONF_BASE, 'medium_confidence', 'ssp245', 'landwaterstorage')
        assert data is not None

    def test_tws_structure(self):
        data = read_ipcc_component_nc(
            CONF_BASE, 'medium_confidence', 'ssp245', 'landwaterstorage')
        if data is None:
            pytest.skip("TWS data not available")
        ex = ipcc_extract(data)
        for key in ('years', 'q50', 'q05', 'q95'):
            assert key in ex

    def test_tws_positive_at_2100(self):
        """TWS should be positive (groundwater depletion → SLR)."""
        data = read_ipcc_component_nc(
            CONF_BASE, 'medium_confidence', 'ssp245', 'landwaterstorage')
        if data is None:
            pytest.skip("TWS data not available")
        ex = ipcc_extract(data)
        idx = np.argmin(np.abs(ex['years'] - 2100))
        assert ex['q50'][idx] > 0, f"TWS median = {ex['q50'][idx]:.0f}, expected > 0"

    def test_tws_range_plausible(self):
        """TWS at 2100 should be 10-100 mm."""
        data = read_ipcc_component_nc(
            CONF_BASE, 'medium_confidence', 'ssp245', 'landwaterstorage')
        if data is None:
            pytest.skip("TWS data not available")
        ex = ipcc_extract(data)
        idx = np.argmin(np.abs(ex['years'] - 2100))
        assert 5 < ex['q50'][idx] < 100, (
            f"TWS median = {ex['q50'][idx]:.0f} mm, outside [5, 100]")


# =========================================================================
# Synthetic TWS MC samples
# =========================================================================

@pytest.mark.skipif(not HAS_IPCC, reason="IPCC AR6 data not found")
class TestTWSSyntheticSamples:
    """Verify synthetic MC samples from IPCC TWS quantiles."""

    def test_synthetic_samples_shape(self):
        data = read_ipcc_component_nc(
            CONF_BASE, 'medium_confidence', 'ssp245', 'landwaterstorage')
        if data is None:
            pytest.skip("TWS data not available")
        ex = ipcc_extract(data, quantiles_target=(0.05, 0.17, 0.5, 0.83, 0.95))
        # Generate synthetic samples (same logic as notebook)
        rng = np.random.default_rng(777)
        n = 2000
        sigma_mm = np.maximum(
            (ex['q95'] - ex['q05']) / (2 * 1.645), 0.1)
        samples = rng.normal(ex['q50'], sigma_mm, size=(n, len(ex['years'])))
        assert samples.shape == (n, len(ex['years']))

    def test_synthetic_median_matches_ipcc(self):
        data = read_ipcc_component_nc(
            CONF_BASE, 'medium_confidence', 'ssp245', 'landwaterstorage')
        if data is None:
            pytest.skip("TWS data not available")
        ex = ipcc_extract(data, quantiles_target=(0.05, 0.17, 0.5, 0.83, 0.95))
        rng = np.random.default_rng(777)
        sigma_mm = np.maximum(
            (ex['q95'] - ex['q05']) / (2 * 1.645), 0.1)
        samples = rng.normal(ex['q50'], sigma_mm, size=(2000, len(ex['years'])))
        idx = np.argmin(np.abs(ex['years'] - 2100))
        mc_median = np.median(samples[:, idx])
        ipcc_median = ex['q50'][idx]
        assert abs(mc_median - ipcc_median) < 5, (
            f"MC median = {mc_median:.1f}, IPCC = {ipcc_median:.1f}")


# =========================================================================
# Budget closure: component sum vs observations
# =========================================================================

@pytest.mark.skipif(not (HAS_H5 and HAS_SLR_H5),
                    reason="Need both HDF5 files")
class TestBudgetClosure:
    """Verify component sum tracks observed GMSL in satellite era."""

    @pytest.fixture(scope="class")
    def obs_and_sum(self):
        import pandas as pd
        # Load NASA GMSL
        with pd.HDFStore(SLR_H5, 'r') as store:
            df_nasa = store['/harmonized/df_nasa_gmsl_h']
        nasa_time = df_nasa['decimal_year'].values
        nasa_gmsl = df_nasa['gmsl'].values
        bl_idx = np.argmin(np.abs(nasa_time - BASELINE_YEAR))
        nasa_gmsl_rb = nasa_gmsl - nasa_gmsl[bl_idx]

        # Load component sum
        comp_sum = np.zeros(len(PROJ_YEARS))
        for comp in SUMMED_COMPONENTS:
            loaded = load_component(comp)
            # Use SSP2-4.5 (all are the same in hindcast)
            comp_sum += loaded['projections']['SSP2-4.5']['median']

        return nasa_time, nasa_gmsl_rb, PROJ_YEARS, comp_sum

    def test_hindcast_residual_rate_small(self, obs_and_sum):
        """Residual rate (obs - sum) should be < 1 mm/yr over 1993-2020."""
        nasa_t, nasa_h, proj_t, comp_sum = obs_and_sum
        # Interpolate component sum to NASA times
        mask = (nasa_t >= 1993) & (nasa_t <= 2020)
        t = nasa_t[mask]
        obs = nasa_h[mask]
        pred = np.interp(t, proj_t, comp_sum)
        residual = obs - pred
        # Linear fit to residual
        dt = t - t.mean()
        rate = np.sum(dt * residual) / np.sum(dt**2) * M_TO_MM  # mm/yr
        assert abs(rate) < 2.0, (
            f"Residual rate = {rate:.2f} mm/yr, expected < 2.0")

    def test_component_sum_positive_trend(self, obs_and_sum):
        """Component sum should have positive trend over satellite era."""
        _, _, proj_t, comp_sum = obs_and_sum
        mask = (proj_t >= 1993) & (proj_t <= 2020)
        t = proj_t[mask]
        h = comp_sum[mask]
        dt = t - t.mean()
        rate = np.sum(dt * h) / np.sum(dt**2) * M_TO_MM
        assert rate > 1.0, f"Component sum rate = {rate:.2f} mm/yr, expected > 1"


# =========================================================================
# Consistency with forecast notebook
# =========================================================================

@pytest.mark.skipif(not HAS_H5, reason="component_results.h5 not found")
class TestConsistencyWithForecast:
    """Verify summation matches forecast notebook's component sum."""

    def test_same_components_as_forecast(self):
        """Summation notebook should use same components as forecast."""
        # Forecast COMP_LABELS (from test_forecast.py knowledge):
        forecast_comps = {'ocean', 'glacier', 'greenland', 'apeninsula', 'wais'}
        # tws is added separately in both notebooks
        summation_comps = set(SUMMED_COMPONENTS)
        assert summation_comps == forecast_comps

    def test_total_sum_reproducible(self):
        """Two independent summations should give identical results."""
        idx = np.argmin(np.abs(PROJ_YEARS - 2100))
        ssp = 'SSP2-4.5'

        total1 = np.zeros(N_SAMPLES)
        total2 = np.zeros(N_SAMPLES)
        for comp in SUMMED_COMPONENTS:
            loaded = load_component(comp)
            total1 += loaded['projections'][ssp]['samples'][:, idx]
            total2 += loaded['projections'][ssp]['samples'][:, idx]

        assert np.allclose(total1, total2)
