"""
Tests for slr_forecast.readers — sign conventions, units, metadata.

These are regression tests for the known bugs documented in refactor.md §1:
- IMBIE Gt files use glaciology convention (negative = loss); must flip to SLR
- IMBIE mm files have negative sigma values; must apply np.abs()
- GlaMBIE files use glaciology convention; must flip to SLR
- All readers must tag DataFrames with sign_convention='slr'
"""

import numpy as np
import pandas as pd
import pytest

from slr_forecast.config import RAW_DATA_DIR


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ICE_GT = RAW_DATA_DIR / "ice_sheets" / "Gt"
ICE_ANT = RAW_DATA_DIR / "ice_sheets" / "antarctica"
GLACIERS = RAW_DATA_DIR / "glaciers"


# ---------------------------------------------------------------------------
# IMBIE Gt-format: sign convention must be SLR (positive = rise)
# ---------------------------------------------------------------------------

class TestIMBIEGtSignConvention:
    """IMBIE Gt readers must flip glaciology → SLR convention."""

    def _check_slr_convention(self, df, label):
        """Common checks for all IMBIE Gt readers."""
        # Metadata tag
        assert df.attrs.get("sign_convention") == "slr", (
            f"{label}: sign_convention not tagged as 'slr'"
        )

        # Cumulative: ice sheets have been losing mass → positive SLR
        # At end of record (~2020), cumulative should be positive
        cum = df["cumulative_mass_balance"]
        final = cum.iloc[-1]
        assert final > 0, (
            f"{label}: cumulative_mass_balance at end of record = {final:.6f} m, "
            "expected positive (SLR convention)"
        )

        # Sigma must be positive
        for col in ["mass_balance_rate_sigma", "cumulative_mass_balance_sigma"]:
            assert (df[col] >= 0).all(), (
                f"{label}: {col} has negative values"
            )

        # Units should be meters (values << 1)
        assert df.attrs.get("units", {}).get("cumulative_mass_balance") == "m"

    def test_greenland(self):
        from slr_forecast.readers import read_imbie_greenland
        df = read_imbie_greenland(str(ICE_GT / "imbie_greenland_2021_Gt.csv"))
        self._check_slr_convention(df, "Greenland")

    def test_east_antarctica(self):
        from slr_forecast.readers import read_imbie_east_antarctica
        df = read_imbie_east_antarctica(
            str(ICE_GT / "imbie_east_antarctica_2021_Gt.csv")
        )
        # EAIS is near mass balance — cumulative may be slightly negative
        # (slight mass gain). We check sign_convention tag and sigma only.
        assert df.attrs.get("sign_convention") == "slr"
        assert (df["mass_balance_rate_sigma"] >= 0).all()
        assert (df["cumulative_mass_balance_sigma"] >= 0).all()

    def test_antarctic_peninsula(self):
        from slr_forecast.readers import read_imbie_antarctic_peninsula
        df = read_imbie_antarctic_peninsula(
            str(ICE_GT / "imbie_antarctic_peninsula_2021_Gt.csv")
        )
        self._check_slr_convention(df, "Antarctic Peninsula")

    def test_antarctica_total(self):
        from slr_forecast.readers import read_imbie_antarctica
        df = read_imbie_antarctica(
            str(ICE_GT / "imbie_antarctica_2021_Gt.csv")
        )
        self._check_slr_convention(df, "Antarctica total")

    def test_all_ice_sheets(self):
        from slr_forecast.readers import read_imbie_all
        df = read_imbie_all(str(ICE_GT / "imbie_all_2021_Gt.csv"))
        self._check_slr_convention(df, "All ice sheets")


# ---------------------------------------------------------------------------
# IMBIE mm-format: sigma must be positive (known negative sigma quirk)
# ---------------------------------------------------------------------------

class TestIMBIEMmSigmaFix:
    """IMBIE mm readers must fix negative sigma values."""

    def test_west_antarctica_sigma_positive(self):
        from slr_forecast.readers import read_imbie_west_antarctica
        df = read_imbie_west_antarctica(
            str(ICE_ANT / "imbie_west_antarctica_2021_mm.csv")
        )
        assert df.attrs.get("sign_convention") == "slr"
        assert (df["mass_balance_rate_sigma"] >= 0).all(), (
            "WAIS mm: mass_balance_rate_sigma has negative values"
        )
        assert (df["cumulative_mass_balance_sigma"] >= 0).all(), (
            "WAIS mm: cumulative_mass_balance_sigma has negative values"
        )

    def test_west_antarctica_cumulative_positive(self):
        """WAIS has been losing mass → cumulative SLR contribution positive."""
        from slr_forecast.readers import read_imbie_west_antarctica
        df = read_imbie_west_antarctica(
            str(ICE_ANT / "imbie_west_antarctica_2021_mm.csv")
        )
        final = df["cumulative_mass_balance"].iloc[-1]
        assert final > 0, (
            f"WAIS cumulative_mass_balance at end = {final:.6f} m, "
            "expected positive"
        )

    def test_west_antarctica_units_meters(self):
        """Values should be in meters (not mm)."""
        from slr_forecast.readers import read_imbie_west_antarctica
        df = read_imbie_west_antarctica(
            str(ICE_ANT / "imbie_west_antarctica_2021_mm.csv")
        )
        # WAIS cumulative ~6.6 mm SLE = ~0.0066 m at 2020
        final = df["cumulative_mass_balance"].iloc[-1]
        assert final < 0.1, (
            f"WAIS cumulative = {final:.4f}, looks like mm not m"
        )


# ---------------------------------------------------------------------------
# GlaMBIE: sign convention must be SLR (positive = mass loss = rise)
# ---------------------------------------------------------------------------

class TestGlaMBIESignConvention:
    """GlaMBIE readers must flip negative mass loss → positive SLR."""

    def test_global_sign(self):
        from slr_forecast.readers import read_glambie_global
        df = read_glambie_global(
            str(GLACIERS / "0_global_glambie_consensus.csv")
        )
        assert df.attrs.get("sign_convention") == "slr"

        # Glaciers are losing mass globally → positive SLR contribution
        # Mean rate should be positive
        mean_rate = df["mass_balance"].mean()
        assert mean_rate > 0, (
            f"GlaMBIE global mean rate = {mean_rate:.8f}, "
            "expected positive (SLR convention)"
        )

    def test_global_sigma_positive(self):
        from slr_forecast.readers import read_glambie_global
        df = read_glambie_global(
            str(GLACIERS / "0_global_glambie_consensus.csv")
        )
        assert (df["mass_balance_sigma"] >= 0).all()

    def test_global_units_meters(self):
        from slr_forecast.readers import read_glambie_global
        df = read_glambie_global(
            str(GLACIERS / "0_global_glambie_consensus.csv")
        )
        # ~200 Gt/yr loss → ~0.55 mm/yr → ~5.5e-4 m/yr
        mean_rate = df["mass_balance"].mean()
        assert mean_rate < 0.01, (
            f"GlaMBIE mean rate = {mean_rate:.6f}, looks like Gt or mm not m"
        )


    def test_global_sigma_is_95pct_over_1p96(self):
        """GlaMBIE errors are 95% half-widths; both readers must convert
        them to 1-sigma by dividing by 1.96 (IN-10 regression)."""
        import sys
        from slr_forecast.config import PROJECT_ROOT
        from slr_forecast.readers import read_glambie_global
        sys.path.insert(0, str(PROJECT_ROOT / "notebooks"))
        from slr_data_readers import read_glambie_global as nb_read
        path = GLACIERS / "0_global_glambie_consensus.csv"
        raw = pd.read_csv(path)
        expected_gt = np.abs(raw["combined_gt_errors"].values) / 1.96
        df_pkg = read_glambie_global(str(path))
        df_nb = nb_read(str(path), convert_to_sle=False)
        np.testing.assert_allclose(df_nb["mass_balance_sigma"].values,
                                   expected_gt, rtol=1e-12)
        np.testing.assert_allclose(df_pkg["mass_balance_sigma"].values,
                                   expected_gt / 362.5 / 1000.0, rtol=1e-3)


# ---------------------------------------------------------------------------
# IPCC AR6 observed GMSL: sigma from the 17-83% range
# ---------------------------------------------------------------------------

class TestIPCCObservedSigma:
    """The 17-83% half-range of a normal distribution is 0.9542 sigma, so
    gmsl_sigma must be (upper - lower) / (2 * 0.9542) (MD-15)."""

    PATH = RAW_DATA_DIR / "gmslr" / "ipccar6_global_sea_level_observed.csv"

    def _check(self, df):
        from scipy.stats import norm
        half = (df["gmsl_upper"] - df["gmsl_lower"]).values / 2
        np.testing.assert_allclose(df["gmsl_sigma"].values * norm.ppf(0.83),
                                   half, rtol=1e-4)

    def test_package_reader(self):
        from slr_forecast.readers import read_ipcc_ar6_observed_gmsl
        self._check(read_ipcc_ar6_observed_gmsl(str(self.PATH)))

    def test_notebook_reader(self):
        import sys
        from slr_forecast.config import PROJECT_ROOT
        sys.path.insert(0, str(PROJECT_ROOT / "notebooks"))
        from slr_data_readers import read_ipcc_ar6_observed_gmsl
        self._check(read_ipcc_ar6_observed_gmsl(str(self.PATH)))


# ---------------------------------------------------------------------------
# Impact functions: basic sanity
# ---------------------------------------------------------------------------

class TestImpacts:
    def test_kulpstrauss_baseline(self):
        from slr_forecast.readers import people_displaced_kulpstrauss2019
        # At SLR=0, ~250–260 M people exposed
        val = people_displaced_kulpstrauss2019(0.0)
        assert 250 < val < 270

    def test_jevrejeva_zero(self):
        from slr_forecast.readers import slr_cost_jevrejeva2018
        assert slr_cost_jevrejeva2018(0.0) == 0.0

    def test_jevrejeva_positive(self):
        from slr_forecast.readers import slr_cost_jevrejeva2018
        assert slr_cost_jevrejeva2018(1.0) > 0
