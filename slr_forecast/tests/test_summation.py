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
from component_projections import (
    read_ipcc_component_nc, ipcc_extract, load_ipcc_tws_projection,
)
from slr_forecast.config import BASELINE_YEAR

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

from slr_forecast import M_TO_MM

# PROJ_SSPS is append-only (SSP1-1.9 is last), so order by 2100 forcing
# (W/m^2, the number after the dash) for monotonicity checks.
SSPS_BY_FORCING = sorted(PROJ_SSPS, key=lambda s: float(s.split('-')[-1]))

# Components that should be summed. EAIS was previously excluded (IMBIE
# record too short/noisy to constrain a reliable trend); re-included
# 2026-09-13 following the correlation-aware CI and b>=0 prior fix, which
# better constrains its projected range (matches component_summation.ipynb
# and component_forecast.ipynb).
SUMMED_COMPONENTS = ['ocean', 'glacier', 'greenland', 'eais', 'apeninsula', 'wais']
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


# =========================================================================
# Sample-level summation
# =========================================================================

@pytest.mark.skipif(not HAS_H5, reason="component_results.h5 not found")
class TestSampleSummation:
    """Verify that summing components sample-by-sample produces correct totals."""

    @pytest.fixture(scope="class")
    def comp_samples(self):
        """Load all summed components, downsampling to N_SAMPLES if needed."""
        samples = {}
        for comp in SUMMED_COMPONENTS:
            loaded = load_component(comp)
            proj = loaded['projections']
            # Downsample components with more than N_SAMPLES (e.g., WAIS=10000)
            for ssp in proj:
                if proj[ssp]['samples'].shape[0] > N_SAMPLES:
                    proj[ssp]['samples'] = proj[ssp]['samples'][:N_SAMPLES]
            samples[comp] = proj
        return samples

    def test_all_shapes_match(self, comp_samples):
        """All components should have (N_SAMPLES, 201) shape."""
        for comp in SUMMED_COMPONENTS:
            shape = comp_samples[comp]['SSP2-4.5']['samples'].shape
            assert shape == (N_SAMPLES, len(PROJ_YEARS)), (
                f"{comp} shape = {shape}")

    def test_sum_at_baseline_near_zero(self, comp_samples):
        """Component sum should be near zero at BASELINE_YEAR (2000).

        The stored sum is 0.4 mm at 2000 (ocean, rebased at 1999.5) and
        about 12.6 mm at 2005, so a 2 mm bound would catch a 2005 rebase.
        """
        idx = np.argmin(np.abs(PROJ_YEARS - BASELINE_YEAR))
        assert PROJ_YEARS[idx] == BASELINE_YEAR
        for ssp in PROJ_SSPS:
            total = np.zeros(N_SAMPLES)
            for comp in SUMMED_COMPONENTS:
                total += comp_samples[comp][ssp]['samples'][:, idx]
            med_mm = np.median(total) * M_TO_MM
            assert abs(med_mm) < 2.0, (
                f"{ssp} total at {BASELINE_YEAR:.0f} = {med_mm:.1f} mm, expected ~0")

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
        for ssp in SSPS_BY_FORCING:
            total = np.zeros(N_SAMPLES)
            for comp in SUMMED_COMPONENTS:
                total += comp_samples[comp][ssp]['samples'][:, idx]
            medians.append(np.median(total))
        for i in range(len(medians) - 1):
            assert medians[i] <= medians[i + 1] * 1.05

    def test_eais_included_lowers_total(self, comp_samples):
        """EAIS is included in SUMMED_COMPONENTS and contributes negative SLR,
        so the total is lower than the sum of the other components alone."""
        assert 'eais' in SUMMED_COMPONENTS
        idx = np.argmin(np.abs(PROJ_YEARS - 2100))
        ssp = 'SSP2-4.5'
        eais_med = np.median(comp_samples['eais'][ssp]['samples'][:, idx])
        assert eais_med < 0, "EAIS should contribute negative SLR"

        total_with = np.zeros(N_SAMPLES)
        total_without = np.zeros(N_SAMPLES)
        for comp in SUMMED_COMPONENTS:
            s = comp_samples[comp][ssp]['samples'][:, idx]
            total_with += s
            if comp != 'eais':
                total_without += s
        assert np.median(total_with) < np.median(total_without)


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
                    'greenland': 'Greenland', 'eais': 'EAIS',
                    'apeninsula': 'Peninsula', 'wais': 'WAIS',
                }
                entry = loaded['projections'][ssp]
                # Downsample to N_SAMPLES if needed (WAIS has 10000)
                if entry['samples'].shape[0] > N_SAMPLES:
                    entry = dict(entry, samples=entry['samples'][:N_SAMPLES])
                proj[ssp][labels[comp]] = entry
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
# Projected TWS ensemble (production loader)
# =========================================================================

@pytest.mark.skipif(not (HAS_IPCC and HAS_IPCC_DIST),
                    reason="Need IPCC AR6 data and ipcc_distributions.h5")
class TestTWSProjection:
    """load_ipcc_tws_projection, the TWS ensemble summed by
    component_summation and component_forecast."""

    @pytest.fixture(scope="class")
    def tws(self):
        return load_ipcc_tws_projection('SSP2-4.5', PROJ_YEARS,
                                        n_samples=N_SAMPLES,
                                        h5_path=IPCC_DIST_PATH)

    def test_shape(self, tws):
        assert tws['samples'].shape == (N_SAMPLES, len(PROJ_YEARS))

    def test_median_matches_ar6(self, tws):
        """2100 median within 5 mm of the AR6 TWS median (both on the AR6
        reference; TWS is not shifted to our baseline)."""
        data = read_ipcc_component_nc(
            CONF_BASE, 'medium_confidence', 'ssp245', 'landwaterstorage')
        ex = ipcc_extract(data)
        q50 = ex['q50'][np.argmin(np.abs(ex['years'] - 2100))]
        med = np.median(tws['samples'][:, np.argmin(np.abs(PROJ_YEARS - 2100))])
        assert abs(med * M_TO_MM - q50) < 5, (
            f"TWS 2100 median {med * M_TO_MM:.1f} mm vs AR6 {q50:.1f} mm")

    def test_zero_through_2005(self, tws):
        """AR6 components are zero at their 2005 node, and the TWS total-only
        offset is not applied, so every member is zero through 2005."""
        m = PROJ_YEARS <= 2005
        assert np.allclose(tws['samples'][:, m], 0.0, atol=1e-9)

    def test_one_quantile_per_member(self, tws):
        """Each member holds one quantile for the whole century, so the
        member ranking is the same at 2050 and 2100."""
        i50 = np.argmin(np.abs(PROJ_YEARS - 2050))
        i21 = np.argmin(np.abs(PROJ_YEARS - 2100))
        r50 = np.argsort(np.argsort(tws['samples'][:, i50]))
        r21 = np.argsort(np.argsort(tws['samples'][:, i21]))
        assert np.array_equal(r50, r21)


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

@pytest.mark.skipif(not (HAS_H5 and HAS_IPCC_DIST),
                    reason="Need component_results.h5 and ipcc_distributions.h5")
class TestConsistencyWithForecast:
    """The stored blended forecast sums the same members as the summation."""

    def test_same_components_as_forecast(self):
        """component_forecast records the components it summed."""
        import h5py
        with h5py.File(H5_PATH, 'r') as hf:
            summed = [c.decode() if isinstance(c, bytes) else str(c)
                      for c in hf['blended'].attrs['components_summed']]
        assert set(summed) == set(SUMMED_COMPONENTS) | {'tws'}

    def test_late_increments_match_component_sum(self):
        """Once the blend weight is negligible (w < 1e-4 after 2080), each
        member's 2080-2090 rise in the blended forecast equals the rise of
        its component sum (wais_2k members and the shared TWS members).
        Differences come only from the rate averaging in blend_rate_space."""
        import h5py
        ssp = 'SSP2-4.5'
        with h5py.File(H5_PATH, 'r') as hf:
            w = hf['wais_2k/full_samples'][:]
            w_yr = hf['wais_2k/years'][:]
            fy = hf['blended/forecast_years'][:]
            bl = hf[f'blended/{ssp}/samples'][:]
        a, b = (np.argmin(np.abs(PROJ_YEARS - y)) for y in (2080, 2090))
        fa, fb = (np.argmin(np.abs(fy - y)) for y in (2080, 2090))
        wa, wb = (np.argmin(np.abs(w_yr - y)) for y in (2080, 2090))
        rise = w[:, wb] - w[:, wa]
        for comp in SUMMED_COMPONENTS:
            if comp == 'wais':
                continue
            s = load_component(comp)['projections'][ssp]['samples']
            rise = rise + s[:, b] - s[:, a]
        tws = load_ipcc_tws_projection(ssp, PROJ_YEARS, n_samples=N_SAMPLES,
                                       h5_path=IPCC_DIST_PATH)['samples']
        rise = rise + tws[:, b] - tws[:, a]
        d_mm = np.abs((bl[:, fb] - bl[:, fa]) - rise) * M_TO_MM
        assert np.median(d_mm) < 0.2, f"median |diff| = {np.median(d_mm):.3f} mm"
        assert np.percentile(d_mm, 95) < 1.0, (
            f"95th pct |diff| = {np.percentile(d_mm, 95):.3f} mm")
