"""Unit tests for the East Antarctic Ice Sheet (EAIS) component.

Tests cover: IMBIE data reader, SMB sensitivity parameters,
SMB projection logic, HDF5 save/load roundtrip, and ISMIP6
region mapping.
"""

import sys
import os
import tempfile

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'notebooks'))

from component_io import save_eais, load_component, PROJ_YEARS
from smb_projections import EAIS_SMB, project_smb_ensemble, GT_TO_M_SLE

# Paths
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
RAW_DIR = os.path.join(PROJECT_ROOT, 'data', 'raw')
H5_PATH = os.path.join(PROJECT_ROOT, 'data', 'processed', 'component_results.h5')
IMBIE_EAIS_PATH = os.path.join(
    RAW_DIR, 'ice_sheets', 'antarctica',
    'imbie_east_antarctica_2021_mm.csv')
ISMIP6_BASE = os.path.join(RAW_DIR, 'ice_sheets', 'ismip6', 'ComputedScalarsPaper')

HAS_IMBIE_EAIS = os.path.exists(IMBIE_EAIS_PATH)
HAS_H5 = os.path.exists(H5_PATH)
HAS_ISMIP6 = os.path.exists(ISMIP6_BASE)

M_TO_MM = 1000.0
BASELINE_YEAR = 2005.0
PROJ_SSPS = ['SSP1-2.6', 'SSP2-4.5', 'SSP3-7.0', 'SSP5-8.5']


# =========================================================================
# IMBIE reader
# =========================================================================

@pytest.mark.skipif(not HAS_IMBIE_EAIS, reason="IMBIE EAIS CSV not found")
class TestIMBIEEAISReader:
    """Verify read_imbie_west_antarctica works on EAIS CSV."""

    @pytest.fixture(scope="class")
    def eais_data(self):
        from slr_data_readers import read_imbie_west_antarctica
        df = read_imbie_west_antarctica(IMBIE_EAIS_PATH)
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

    def test_columns_present(self, eais_data):
        df = eais_data[0]
        expected = {'decimal_year', 'cumulative_mass_balance',
                    'cumulative_mass_balance_sigma'}
        assert expected.issubset(set(df.columns))

    def test_units_are_meters(self, eais_data):
        df = eais_data[0]
        max_abs = np.max(np.abs(df['cumulative_mass_balance'].values))
        assert max_abs < 0.1, f"Max |cum| = {max_abs:.4f}, too large for meters"

    def test_time_range(self, eais_data):
        years = eais_data[1]
        assert years[0] >= 1991 and years[0] <= 1993
        assert years[-1] >= 2018

    def test_n_years(self, eais_data):
        years = eais_data[1]
        assert len(years) == 29

    def test_annual_spacing(self, eais_data):
        years = eais_data[1]
        dt = np.diff(years)
        assert np.allclose(dt, 1.0, atol=0.1)

    def test_baseline_zero(self, eais_data):
        years, H = eais_data[1], eais_data[2]
        bl_idx = np.argmin(np.abs(years - BASELINE_YEAR))
        assert abs(H[bl_idx]) < 1e-10

    def test_near_mass_balance(self, eais_data):
        """EAIS is near mass balance — cumulative should be small."""
        H = eais_data[2]
        H_mm = H[-1] * M_TO_MM
        assert abs(H_mm) < 5, f"Final H = {H_mm:.2f} mm, too large for EAIS"

    def test_sigma_positive(self, eais_data):
        sigma = eais_data[3]
        assert np.all(sigma >= 0)


# =========================================================================
# SMB sensitivity parameters
# =========================================================================

class TestEAISSMBParameters:
    """Verify EAIS_SMB dataclass values match literature."""

    def test_C_T_positive(self):
        """EAIS C_T should be positive (warming → mass gain from snowfall)."""
        assert EAIS_SMB.C_T > 0

    def test_C_T_value(self):
        assert EAIS_SMB.C_T == 60.0

    def test_C_T_sigma_value(self):
        assert EAIS_SMB.C_T_sigma == 20.0

    def test_C_T2_zero(self):
        """EAIS has no quadratic term."""
        assert EAIS_SMB.C_T2 == 0.0
        assert EAIS_SMB.C_T2_sigma == 0.0

    def test_temperature_frame(self):
        assert EAIS_SMB.temperature_frame == 'GMST'

    def test_reference_not_empty(self):
        assert len(EAIS_SMB.reference) > 0

    def test_clausius_clapeyron_consistency(self):
        """C_T ~ 5%/°C of ~1200 Gt/yr accumulation → ~60 Gt/yr/°C."""
        cc_rate = 0.05  # 5%/°C
        accumulation = 1200  # Gt/yr
        expected = cc_rate * accumulation
        assert abs(EAIS_SMB.C_T - expected) < 20, (
            f"C_T={EAIS_SMB.C_T} vs CC estimate {expected}")


# =========================================================================
# SMB projection logic
# =========================================================================

class TestSMBProjection:
    """Verify project_smb_ensemble produces correct output structure."""

    @pytest.fixture(scope="class")
    def smb_result(self):
        time_proj = np.arange(1950, 2151, dtype=float)
        # Simple linear warming for testing
        T_proj = {}
        for ssp, rate in [('SSP1-2.6', 0.01), ('SSP5-8.5', 0.04)]:
            T = np.zeros(len(time_proj))
            T[time_proj >= 2005] = rate * (time_proj[time_proj >= 2005] - 2005)
            T_proj[ssp] = T
        return project_smb_ensemble(
            sensitivity=EAIS_SMB,
            T_proj=T_proj,
            time_proj=time_proj,
            T_baseline=0.0,
            n_samples=500,
            seed=42,
        )

    def test_returns_dict(self, smb_result):
        assert isinstance(smb_result, dict)
        assert 'SSP1-2.6' in smb_result
        assert 'SSP5-8.5' in smb_result

    def test_has_required_keys(self, smb_result):
        proj = smb_result['SSP5-8.5']
        for key in ('samples', 'median', 'p5', 'p95', 'p17', 'p83'):
            assert key in proj, f"Missing key '{key}'"

    def test_samples_shape(self, smb_result):
        samples = smb_result['SSP5-8.5']['samples']
        assert samples.shape == (500, 201)

    def test_eais_negative_slr(self, smb_result):
        """EAIS has positive C_T → warming → mass gain → negative SLR."""
        idx_2100 = 150  # 2100 - 1950
        for ssp in smb_result:
            med = smb_result[ssp]['median'][idx_2100]
            assert med < 0, (
                f"{ssp} EAIS median at 2100 = {med*M_TO_MM:.1f} mm, expected < 0")

    def test_higher_ssp_more_negative(self, smb_result):
        """SSP5-8.5 should have more negative SLR (more mass gain)."""
        idx_2100 = 150
        med_low = smb_result['SSP1-2.6']['median'][idx_2100]
        med_high = smb_result['SSP5-8.5']['median'][idx_2100]
        assert med_high < med_low, (
            f"SSP5-8.5 ({med_high*M_TO_MM:.1f}) should be more negative "
            f"than SSP1-2.6 ({med_low*M_TO_MM:.1f})")

    def test_percentile_ordering(self, smb_result):
        proj = smb_result['SSP5-8.5']
        idx = 150
        assert proj['p5'][idx] <= proj['p17'][idx]
        assert proj['p17'][idx] <= proj['median'][idx]
        assert proj['median'][idx] <= proj['p83'][idx]
        assert proj['p83'][idx] <= proj['p95'][idx]

    def test_zero_at_baseline(self, smb_result):
        """Projection should be near zero at baseline epoch."""
        # baseline is where dT ≈ 0, which is ~2005 = index 55
        idx_bl = 55
        for ssp in smb_result:
            med = smb_result[ssp]['median'][idx_bl]
            assert abs(med) < 0.001, (
                f"{ssp} at baseline = {med*M_TO_MM:.2f} mm, expected ~0")


# =========================================================================
# HDF5 roundtrip
# =========================================================================

@pytest.mark.skipif(not HAS_H5, reason="component_results.h5 not found")
class TestEAISSaveLoad:
    """Verify save/load roundtrip for EAIS component."""

    @pytest.fixture(scope="class")
    def loaded(self):
        return load_component('eais')

    def test_metadata_model_type(self, loaded):
        assert loaded['metadata']['model_type'] == 'trend_only'

    def test_metadata_has_r2(self, loaded):
        r2 = loaded['metadata']['r2']
        assert 0.3 < r2 < 0.8, f"R² = {r2:.4f}, expected ~0.57"

    def test_metadata_projection_method(self, loaded):
        assert loaded['metadata']['projection_method'] == 'smb_literature'

    def test_metadata_C_T(self, loaded):
        assert loaded['metadata']['C_T'] == 60.0

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


# =========================================================================
# Projection plausibility
# =========================================================================

@pytest.mark.skipif(not HAS_H5, reason="component_results.h5 not found")
class TestEAISProjections:
    """Verify EAIS projections are physically plausible."""

    @pytest.fixture(scope="class")
    def loaded(self):
        return load_component('eais')

    def test_negative_slr_at_2100(self, loaded):
        """EAIS should contribute negative SLR (mass gain) at 2100."""
        idx = np.argmin(np.abs(loaded['proj_years'] - 2100))
        for ssp in PROJ_SSPS:
            med = loaded['projections'][ssp]['median'][idx]
            assert med < 0, f"{ssp} median = {med*M_TO_MM:.1f} mm, expected < 0"

    def test_range_at_2100(self, loaded):
        """EAIS median at 2100 should be between -50 and 0 mm."""
        idx = np.argmin(np.abs(loaded['proj_years'] - 2100))
        for ssp in PROJ_SSPS:
            med_mm = loaded['projections'][ssp]['median'][idx] * M_TO_MM
            assert -50 < med_mm < 0, (
                f"{ssp} median = {med_mm:.1f} mm, outside [-50, 0]")

    def test_ssp_ordering(self, loaded):
        """Higher SSP → more negative EAIS (more mass gain from snowfall)."""
        idx = np.argmin(np.abs(loaded['proj_years'] - 2100))
        medians = [loaded['projections'][ssp]['median'][idx]
                    for ssp in PROJ_SSPS]
        for i in range(len(medians) - 1):
            assert medians[i] >= medians[i + 1] * 0.9, (
                f"SSP ordering violated: {PROJ_SSPS[i]}={medians[i]*M_TO_MM:.1f}"
                f" vs {PROJ_SSPS[i+1]}={medians[i+1]*M_TO_MM:.1f}")

    def test_near_zero_at_baseline(self, loaded):
        idx = np.argmin(np.abs(loaded['proj_years'] - BASELINE_YEAR))
        for ssp in PROJ_SSPS:
            med_mm = loaded['projections'][ssp]['median'][idx] * M_TO_MM
            assert abs(med_mm) < 2.0, (
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
class TestISMIP6EAIS:
    """Verify ISMIP6 regional reader for EAIS (region=2)."""

    def test_reads_eais(self):
        from component_projections import read_ismip6_regional
        data = read_ismip6_regional(ISMIP6_BASE, region=2)
        assert len(data) > 0, "No ISMIP6 EAIS runs found"

    def test_keys_are_tuples(self):
        from component_projections import read_ismip6_regional
        data = read_ismip6_regional(ISMIP6_BASE, region=2)
        for key in data:
            assert isinstance(key, tuple) and len(key) == 3

    def test_sle_in_meters(self):
        from component_projections import read_ismip6_regional
        data = read_ismip6_regional(ISMIP6_BASE, region=2)
        for key, val in data.items():
            max_abs = np.max(np.abs(val['sle_m']))
            assert max_abs < 0.5, (
                f"Max |SLE| = {max_abs:.3f} m for {key}, too large for EAIS")
            break
