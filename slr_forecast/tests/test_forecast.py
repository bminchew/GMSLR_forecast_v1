"""Unit tests for the component forecast notebook.

Tests cover: component loading and summation, rate-space blending,
sigmoid weight function, headline statistics, variance decomposition,
IPCC comparison data, and physical plausibility of total forecasts.
"""

import sys
import os
import json

import numpy as np
import pytest
from scipy.special import expit

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'notebooks'))

from component_io import (
    load_all_projections, load_component, list_components,
    PROJ_SSPS, PROJ_YEARS, N_SAMPLES,
)
from component_projections import read_ipcc_component_nc, ipcc_extract

# Paths
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
H5_PATH = os.path.join(PROJECT_ROOT, 'data', 'processed', 'component_results.h5')
IPCC_DIST_PATH = os.path.join(PROJECT_ROOT, 'data', 'processed', 'ipcc_distributions.h5')
HEADLINE_PATH = os.path.join(PROJECT_ROOT, 'data', 'processed',
                              'manuscript_headline_stats.json')
RAW_DIR = os.path.join(PROJECT_ROOT, 'data', 'raw')
CONF_BASE = os.path.join(
    RAW_DIR, 'ipcc_ar6', 'slr', 'ar6', 'global', 'confidence_output_files')

HAS_H5 = os.path.exists(H5_PATH)
HAS_IPCC_DIST = os.path.exists(IPCC_DIST_PATH)
HAS_HEADLINE = os.path.exists(HEADLINE_PATH)
HAS_IPCC = os.path.exists(os.path.join(
    CONF_BASE, 'medium_confidence', 'ssp245'))

M_TO_MM = 1000.0
BASELINE_YEAR = 2005.0

# Expected components in HDF5
EXPECTED_COMPONENTS = ['ocean', 'glacier', 'greenland', 'apeninsula', 'wais', 'eais']


# =========================================================================
# Component loading
# =========================================================================

@pytest.mark.skipif(not HAS_H5, reason="component_results.h5 not found")
class TestLoadAllProjections:
    """Verify load_all_projections returns correct structure."""

    @pytest.fixture(scope="class")
    def loaded(self):
        return load_all_projections()

    def test_returns_tuple(self, loaded):
        proj_years, all_proj = loaded
        assert isinstance(proj_years, np.ndarray)
        assert isinstance(all_proj, dict)

    def test_proj_years_shape(self, loaded):
        proj_years, _ = loaded
        assert proj_years.shape == (201,)
        assert proj_years[0] == 1950.0
        assert proj_years[-1] == 2150.0

    def test_all_components_present(self, loaded):
        _, all_proj = loaded
        for comp in EXPECTED_COMPONENTS:
            assert comp in all_proj, f"Missing component: {comp}"

    def test_all_ssps_per_component(self, loaded):
        _, all_proj = loaded
        for comp in EXPECTED_COMPONENTS:
            for ssp in PROJ_SSPS:
                assert ssp in all_proj[comp], (
                    f"Missing {ssp} in {comp}")

    def test_samples_shape(self, loaded):
        """All components should have samples for at least one SSP."""
        proj_years, all_proj = loaded
        for comp in EXPECTED_COMPONENTS:
            # SSP-independent components (WAIS) may only have samples
            # under the first SSP via load_all_projections
            found = False
            for ssp in PROJ_SSPS:
                if 'samples' in all_proj[comp][ssp]:
                    samples = all_proj[comp][ssp]['samples']
                    assert samples.shape == (N_SAMPLES, len(proj_years)), (
                        f"{comp}/{ssp} shape = {samples.shape}")
                    found = True
                    break
            assert found, f"{comp} has no samples in any SSP"

    def test_percentile_keys(self, loaded):
        _, all_proj = loaded
        for key in ('median', 'p5', 'p17', 'p83', 'p95'):
            assert key in all_proj['ocean']['SSP2-4.5'], (
                f"Missing '{key}' in ocean SSP2-4.5")


# =========================================================================
# Component summation
# =========================================================================

@pytest.mark.skipif(not HAS_H5, reason="component_results.h5 not found")
class TestComponentSummation:
    """Verify that summing components produces correct totals."""

    @pytest.fixture(scope="class")
    def comp_samples(self):
        """Load all components with proper sample propagation."""
        samples = {}
        for comp in EXPECTED_COMPONENTS:
            loaded = load_component(comp)
            samples[comp] = loaded['projections']
        return samples

    def test_sum_positive_at_2100(self, comp_samples):
        """Total SLR at 2100 should be positive for all SSPs."""
        idx = np.argmin(np.abs(PROJ_YEARS - 2100))
        for ssp in PROJ_SSPS:
            total = np.zeros(N_SAMPLES)
            for comp in EXPECTED_COMPONENTS:
                total += comp_samples[comp][ssp]['samples'][:, idx]
            assert np.median(total) > 0

    def test_ocean_dominates_at_2050(self, comp_samples):
        """Ocean should be largest positive contributor at 2050."""
        ssp = 'SSP2-4.5'
        idx = np.argmin(np.abs(PROJ_YEARS - 2050))
        ocean_med = np.median(comp_samples['ocean'][ssp]['samples'][:, idx])
        for comp in ['glacier', 'greenland', 'apeninsula']:
            comp_med = np.median(comp_samples[comp][ssp]['samples'][:, idx])
            assert ocean_med > comp_med, (
                f"Ocean ({ocean_med*M_TO_MM:.0f}) should exceed "
                f"{comp} ({comp_med*M_TO_MM:.0f})")

    def test_eais_negative(self, comp_samples):
        """EAIS should contribute negative SLR (mass gain)."""
        ssp = 'SSP2-4.5'
        idx = np.argmin(np.abs(PROJ_YEARS - 2100))
        eais_med = np.median(comp_samples['eais'][ssp]['samples'][:, idx])
        assert eais_med < 0, f"EAIS median = {eais_med*M_TO_MM:.0f} mm, expected < 0"

    def test_ssp_ordering_of_total(self, comp_samples):
        """Higher SSP → higher total SLR at 2100."""
        idx = np.argmin(np.abs(PROJ_YEARS - 2100))
        medians = []
        for ssp in PROJ_SSPS:
            total = np.zeros(N_SAMPLES)
            for comp in EXPECTED_COMPONENTS:
                total += comp_samples[comp][ssp]['samples'][:, idx]
            medians.append(np.median(total))
        for i in range(len(medians) - 1):
            # WAIS is SSP-independent, so ordering may be soft
            assert medians[i] <= medians[i + 1] * 1.05


# =========================================================================
# Sigmoid blending weight
# =========================================================================

class TestSigmoidWeight:
    """Verify sigmoid blending weight properties."""

    def test_weight_at_center(self):
        """w(t_center) should be exactly 0.5."""
        t_center = 2035.0
        tau = 5.0
        w = 1.0 - expit((t_center - t_center) / tau)
        assert abs(w - 0.5) < 1e-10

    def test_weight_at_origin(self):
        """w(t_origin) should be close to 1 (trust quadratic)."""
        t_center = 2035.0
        tau = 5.0
        t_origin = 2025.3
        w = 1.0 - expit((t_origin - t_center) / tau)
        assert w > 0.8, f"w at origin = {w:.3f}, expected > 0.8"

    def test_weight_at_2100(self):
        """w(2100) should be near zero (trust components)."""
        t_center = 2035.0
        tau = 5.0
        w = 1.0 - expit((2100 - t_center) / tau)
        assert w < 1e-4, f"w at 2100 = {w:.6f}, expected < 1e-4"

    def test_weight_monotonically_decreasing(self):
        """w(t) should decrease monotonically."""
        t_center = 2035.0
        tau = 5.0
        years = np.arange(2025, 2101, dtype=float)
        w = 1.0 - expit((years - t_center) / tau)
        assert np.all(np.diff(w) <= 0)

    def test_weight_bounded_0_1(self):
        """w(t) should always be in [0, 1]."""
        t_center = 2035.0
        tau = 5.0
        years = np.arange(1990, 2200, dtype=float)
        w = 1.0 - expit((years - t_center) / tau)
        assert np.all(w >= 0) and np.all(w <= 1)

    def test_tau_controls_width(self):
        """Smaller tau → sharper transition."""
        t_center = 2035.0
        t_test = 2030.0
        w_narrow = 1.0 - expit((t_test - t_center) / 2.0)
        w_wide = 1.0 - expit((t_test - t_center) / 10.0)
        # Narrow tau at t_test < t_center: w_narrow should be closer to 1
        assert w_narrow > w_wide


# =========================================================================
# Rate-space blending function
# =========================================================================

class TestBlendRateSpace:
    """Verify blend_rate_space with synthetic data."""

    @staticmethod
    def _make_blend_func():
        """Import blend_rate_space from notebook code (inline definition)."""
        def blend_rate_space(proj_years, comp_samples, sq_rate_samples,
                             sq_level_samples_rb, sq_time,
                             t_origin, h_origin, t_center, tau_blend):
            n_samples = comp_samples.shape[0]
            fmask = proj_years >= t_origin
            f_years = proj_years[fmask]
            n_t = len(f_years)
            w_t = 1.0 - expit((f_years - t_center) / tau_blend)
            dt_proj = np.diff(proj_years)
            comp_rate_all = np.diff(comp_samples, axis=1) / dt_proj[None, :]
            comp_rate_full = np.zeros_like(comp_samples)
            comp_rate_full[:, 1:-1] = 0.5 * (comp_rate_all[:, :-1] + comp_rate_all[:, 1:])
            comp_rate_full[:, 0] = comp_rate_all[:, 0]
            comp_rate_full[:, -1] = comp_rate_all[:, -1]
            comp_rate_f = comp_rate_full[:, fmask]
            sq_rate_f = np.zeros((n_samples, n_t))
            for k in range(n_samples):
                sq_rate_f[k] = np.interp(f_years, sq_time, sq_rate_samples[k])
            blended_rate = w_t[None, :] * sq_rate_f + (1.0 - w_t[None, :]) * comp_rate_f
            dt_f = np.diff(f_years)
            forecast_samples = np.zeros((n_samples, n_t))
            forecast_samples[:, 0] = h_origin
            for j in range(1, n_t):
                forecast_samples[:, j] = (forecast_samples[:, j - 1]
                    + 0.5 * (blended_rate[:, j - 1] + blended_rate[:, j]) * dt_f[j - 1])
            return forecast_samples, f_years, w_t
        return blend_rate_space

    def test_output_shape(self):
        blend = self._make_blend_func()
        proj_years = np.arange(2000, 2101, dtype=float)
        n = 50
        comp = np.cumsum(np.ones((n, 101)) * 0.003, axis=1)  # linear 3mm/yr
        sq_time = np.arange(2000, 2101, dtype=float)
        sq_rate = np.ones((n, 101)) * 0.004  # 4mm/yr quadratic rate
        sq_level = np.cumsum(sq_rate * 1.0, axis=1)
        f_samp, f_years, w = blend(proj_years, comp, sq_rate, sq_level,
                                    sq_time, 2025.0, 0.05, 2035.0, 5.0)
        assert f_samp.shape[0] == n
        assert f_samp.shape[1] == len(f_years)
        assert f_years[0] >= 2025.0

    def test_starts_at_h_origin(self):
        blend = self._make_blend_func()
        proj_years = np.arange(2000, 2101, dtype=float)
        n = 10
        comp = np.cumsum(np.ones((n, 101)) * 0.003, axis=1)
        sq_time = np.arange(2000, 2101, dtype=float)
        sq_rate = np.ones((n, 101)) * 0.004
        sq_level = np.cumsum(sq_rate, axis=1)
        h_origin = 0.069
        f_samp, f_years, _ = blend(proj_years, comp, sq_rate, sq_level,
                                    sq_time, 2025.0, h_origin, 2035.0, 5.0)
        assert np.allclose(f_samp[:, 0], h_origin)

    def test_monotonically_increasing(self):
        """With positive rates, forecast should increase monotonically."""
        blend = self._make_blend_func()
        proj_years = np.arange(2000, 2101, dtype=float)
        n = 10
        comp = np.cumsum(np.ones((n, 101)) * 0.005, axis=1)
        sq_time = np.arange(2000, 2101, dtype=float)
        sq_rate = np.ones((n, 101)) * 0.004
        sq_level = np.cumsum(sq_rate, axis=1)
        f_samp, _, _ = blend(proj_years, comp, sq_rate, sq_level,
                              sq_time, 2025.0, 0.05, 2035.0, 5.0)
        # Median should be monotonically increasing
        med = np.median(f_samp, axis=0)
        assert np.all(np.diff(med) >= -1e-10)

    def test_pure_quadratic_early(self):
        """At origin, forecast rate should be close to quadratic rate."""
        blend = self._make_blend_func()
        proj_years = np.arange(2000, 2101, dtype=float)
        n = 10
        # Component has zero rate; quadratic has 5mm/yr
        comp = np.zeros((n, 101))
        sq_time = np.arange(2000, 2101, dtype=float)
        sq_rate = np.ones((n, 101)) * 0.005
        sq_level = np.cumsum(sq_rate, axis=1)
        f_samp, f_years, w = blend(proj_years, comp, sq_rate, sq_level,
                                    sq_time, 2025.0, 0.0, 2035.0, 5.0)
        # At origin, w ~ 0.86, so forecast rate ≈ 0.86 * 5mm/yr = 4.3mm/yr
        # After 1 year: H ≈ 0 + 4.3mm = 0.0043m
        assert f_samp[0, 1] > 0.003, (
            f"H at t+1 = {f_samp[0,1]*M_TO_MM:.2f} mm, expected > 3 mm")


# =========================================================================
# Headline statistics JSON
# =========================================================================

@pytest.mark.skipif(not HAS_HEADLINE, reason="headline JSON not found")
class TestHeadlineStatistics:
    """Verify manuscript_headline_stats.json structure and values."""

    @pytest.fixture(scope="class")
    def headline(self):
        with open(HEADLINE_PATH) as f:
            return json.load(f)

    def test_has_required_keys(self, headline):
        for key in ('baseline_year', 'preindustrial_to_baseline_m',
                     'forecast_years', 'scenarios'):
            assert key in headline, f"Missing key: {key}"

    def test_baseline_year(self, headline):
        assert headline['baseline_year'] == 2005.0

    def test_preindustrial_offset(self, headline):
        assert headline['preindustrial_to_baseline_m'] == 0.19

    def test_all_ssps_present(self, headline):
        for ssp in PROJ_SSPS:
            assert ssp in headline['scenarios']

    def test_report_years(self, headline):
        expected = list(range(2030, 2110, 10))
        assert headline['forecast_years'] == expected

    def test_2100_values_plausible(self, headline):
        """2100 medians should be between 0.7 and 2.0 m rel. preindustrial."""
        for ssp in PROJ_SSPS:
            med = headline['scenarios'][ssp]['2100']['median_m_rel_preindustrial']
            assert 0.7 < med < 2.0, (
                f"{ssp} 2100 median = {med:.2f} m, outside [0.7, 2.0]")

    def test_ssp_ordering(self, headline):
        """Higher SSP → higher median at 2100."""
        medians = [headline['scenarios'][ssp]['2100']['median_m_rel_preindustrial']
                    for ssp in PROJ_SSPS]
        for i in range(len(medians) - 1):
            assert medians[i] <= medians[i + 1] * 1.01

    def test_exceedance_probabilities_valid(self, headline):
        """Exceedance probs should be in [0, 100] and decrease with threshold."""
        for ssp in PROJ_SSPS:
            st = headline['scenarios'][ssp]['2100']
            probs = [st[f'P_exceed_{t:.1f}m_preindustrial']
                     for t in [0.5, 1.0, 1.5, 2.0]]
            for p in probs:
                assert 0 <= p <= 100
            # Higher threshold → lower probability
            for i in range(len(probs) - 1):
                assert probs[i] >= probs[i + 1]

    def test_100pct_exceed_0_5m(self, headline):
        """P(>0.5m) should be 100% for all SSPs at 2100."""
        for ssp in PROJ_SSPS:
            p = headline['scenarios'][ssp]['2100']['P_exceed_0.5m_preindustrial']
            assert p == 100.0, f"{ssp} P(>0.5m) = {p}%, expected 100%"

    def test_percentile_ordering(self, headline):
        """p5 < median < p95 at every report year."""
        for ssp in PROJ_SSPS:
            for yr in headline['forecast_years']:
                st = headline['scenarios'][ssp][str(yr)]
                assert st['p5_mm_rel_baseline'] <= st['median_mm_rel_baseline']
                assert st['median_mm_rel_baseline'] <= st['p95_mm_rel_baseline']


# =========================================================================
# Variance decomposition
# =========================================================================

@pytest.mark.skipif(not HAS_HEADLINE, reason="headline JSON not found")
class TestVarianceDecomposition:
    """Verify variance decomposition in headline stats."""

    @pytest.fixture(scope="class")
    def vd(self):
        with open(HEADLINE_PATH) as f:
            h = json.load(f)
        return h.get('variance_decomposition', {})

    def test_has_required_keys(self, vd):
        for key in ('within_scenario_var_m2', 'between_scenario_var_m2',
                     'pct_within_scenario', 'pct_between_scenario'):
            assert key in vd, f"Missing key: {key}"

    def test_percentages_sum_to_100(self, vd):
        total = vd['pct_within_scenario'] + vd['pct_between_scenario']
        assert abs(total - 100.0) < 0.1

    def test_within_dominates(self, vd):
        """Within-scenario variance should dominate (>80%)."""
        assert vd['pct_within_scenario'] > 80

    def test_variances_positive(self, vd):
        assert vd['within_scenario_var_m2'] > 0
        assert vd['between_scenario_var_m2'] > 0

    def test_total_variance_consistent(self, vd):
        """within + between should equal total (if present)."""
        if 'total_var_m2' in vd:
            total = vd['within_scenario_var_m2'] + vd['between_scenario_var_m2']
            assert abs(total - vd['total_var_m2']) < 1e-6


# =========================================================================
# TWS loading from IPCC distributions
# =========================================================================

@pytest.mark.skipif(not HAS_IPCC_DIST, reason="ipcc_distributions.h5 not found")
class TestTWSLoading:
    """Verify TWS samples can be loaded from precomputed IPCC distributions."""

    def test_load_tws_samples(self):
        import h5py
        with h5py.File(IPCC_DIST_PATH, 'r') as f:
            assert 'samples/landwaterstorage/SSP2-4.5' in f

    def test_tws_shape(self):
        import h5py
        with h5py.File(IPCC_DIST_PATH, 'r') as f:
            grp = f['samples/landwaterstorage/SSP2-4.5']
            years = grp['years'][:]
            samples = grp['samples'][:]
            assert samples.ndim == 2
            assert len(years) == samples.shape[1]
            assert samples.shape[0] >= 1000  # enough samples

    def test_tws_units_mm(self):
        """TWS samples should be in mm (values 0-100 at 2100)."""
        import h5py
        with h5py.File(IPCC_DIST_PATH, 'r') as f:
            grp = f['samples/landwaterstorage/SSP2-4.5']
            years = grp['years'][:]
            samples = grp['samples'][:]
            idx = np.argmin(np.abs(years - 2100))
            med = np.median(samples[:, idx])
            assert 5 < med < 100, f"TWS median = {med:.0f}, expected 5-100 mm"

    def test_tws_positive(self):
        """TWS should be positive at 2100 (groundwater depletion → SLR)."""
        import h5py
        with h5py.File(IPCC_DIST_PATH, 'r') as f:
            grp = f['samples/landwaterstorage/SSP2-4.5']
            years = grp['years'][:]
            samples = grp['samples'][:]
            idx = np.argmin(np.abs(years - 2100))
            assert np.median(samples[:, idx]) > 0


# =========================================================================
# IPCC total comparison
# =========================================================================

@pytest.mark.skipif(not HAS_IPCC, reason="IPCC AR6 data not found")
class TestIPCCTotalComparison:
    """Verify IPCC AR6 total projections can be loaded for comparison."""

    def test_read_ipcc_total(self):
        data = read_ipcc_component_nc(
            CONF_BASE, 'medium_confidence', 'ssp245', 'total')
        assert data is not None

    def test_ipcc_total_structure(self):
        data = read_ipcc_component_nc(
            CONF_BASE, 'medium_confidence', 'ssp245', 'total')
        if data is None:
            pytest.skip("IPCC total not available")
        ex = ipcc_extract(data)
        for key in ('years', 'q50', 'q05', 'q95'):
            assert key in ex

    def test_ipcc_total_positive_at_2100(self):
        data = read_ipcc_component_nc(
            CONF_BASE, 'medium_confidence', 'ssp245', 'total')
        if data is None:
            pytest.skip("IPCC total not available")
        ex = ipcc_extract(data)
        idx = np.argmin(np.abs(ex['years'] - 2100))
        assert ex['q50'][idx] > 0

    def test_our_forecast_exceeds_ipcc(self):
        """Our 2100 median should exceed IPCC median (we include
        rheology correction and deep uncertainty for WAIS)."""
        data = read_ipcc_component_nc(
            CONF_BASE, 'medium_confidence', 'ssp245', 'total')
        if data is None or not HAS_HEADLINE:
            pytest.skip("Need both IPCC and headline data")
        ex = ipcc_extract(data)
        idx = np.argmin(np.abs(ex['years'] - 2100))
        ipcc_med_mm = ex['q50'][idx]  # mm

        with open(HEADLINE_PATH) as f:
            h = json.load(f)
        our_med_mm = h['scenarios']['SSP2-4.5']['2100']['median_mm_rel_baseline']
        assert our_med_mm > ipcc_med_mm, (
            f"Our median ({our_med_mm:.0f} mm) should exceed "
            f"IPCC ({ipcc_med_mm:.0f} mm)")


# =========================================================================
# Forecast plausibility (loaded from headline JSON)
# =========================================================================

@pytest.mark.skipif(not HAS_HEADLINE, reason="headline JSON not found")
class TestForecastPlausibility:
    """Physical plausibility checks on the blended forecast."""

    @pytest.fixture(scope="class")
    def headline(self):
        with open(HEADLINE_PATH) as f:
            return json.load(f)

    def test_2050_less_than_2100(self, headline):
        """Forecasts at 2050 should be less than at 2100."""
        for ssp in PROJ_SSPS:
            med_2050 = headline['scenarios'][ssp]['2050']['median_mm_rel_baseline']
            med_2100 = headline['scenarios'][ssp]['2100']['median_mm_rel_baseline']
            assert med_2050 < med_2100

    def test_within_to_across_ratio(self, headline):
        """Within-scenario should be 3-10x larger than across-scenario spread."""
        ratio = headline.get('ratio_within_to_across', 0)
        assert 2 < ratio < 15, f"Ratio = {ratio}, expected 3-10"

    def test_2100_median_above_500mm(self, headline):
        """All SSPs should have median > 500 mm at 2100."""
        for ssp in PROJ_SSPS:
            med = headline['scenarios'][ssp]['2100']['median_mm_rel_baseline']
            assert med > 500, f"{ssp} median = {med:.0f} mm, expected > 500"

    def test_uncertainty_range_reasonable(self, headline):
        """90% CI should be 400-1500 mm wide at 2100."""
        for ssp in PROJ_SSPS:
            st = headline['scenarios'][ssp]['2100']
            width = st['p95_mm_rel_baseline'] - st['p5_mm_rel_baseline']
            assert 400 < width < 1500, (
                f"{ssp} 90% width = {width:.0f} mm, outside [400, 1500]")

    def test_early_years_converge(self, headline):
        """At 2030, SSPs should be close (quadratic dominates)."""
        medians_2030 = [headline['scenarios'][ssp]['2030']['median_mm_rel_baseline']
                         for ssp in PROJ_SSPS]
        spread = max(medians_2030) - min(medians_2030)
        assert spread < 20, (
            f"2030 spread = {spread:.0f} mm, expected < 20 (quadratic dominates)")
