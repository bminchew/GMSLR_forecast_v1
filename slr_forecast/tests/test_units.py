"""Tests for slr_forecast.units — conversion factors, tagging, assertions."""

import numpy as np
import pandas as pd
import pytest

from slr_forecast.units import (
    M_TO_MM,
    MM_TO_M,
    GT_TO_MM_SLE,
    GT_TO_M_SLE,
    OCEAN_AREA_KM2,
    OCEAN_AREA_M2,
    gt_to_m_sle,
    m_to_mm,
    mm_to_m,
    tag_units,
    tag_sign_convention,
    tag_baseline,
    assert_slr_convention,
    assert_units_meters,
    assert_sigma_positive,
)


# ---------------------------------------------------------------------------
# Conversion constants
# ---------------------------------------------------------------------------

def test_m_to_mm_roundtrip():
    assert M_TO_MM * MM_TO_M == pytest.approx(1.0)


def test_gt_to_mm_sle_value():
    """1 Gt over 362.5e6 km² ≈ 1/362.5 mm."""
    assert GT_TO_MM_SLE == pytest.approx(1.0 / 362.5)


def test_gt_to_m_sle_value():
    """GT_TO_M_SLE = GT_TO_MM_SLE / 1000."""
    assert GT_TO_M_SLE == pytest.approx(GT_TO_MM_SLE / 1000.0)


def test_ocean_area_consistency():
    assert OCEAN_AREA_M2 == pytest.approx(OCEAN_AREA_KM2 * 1e6)


# ---------------------------------------------------------------------------
# Conversion functions
# ---------------------------------------------------------------------------

def test_gt_to_m_sle_function():
    result = gt_to_m_sle(362.5)
    assert result == pytest.approx(1e-3)  # 362.5 Gt → 1 mm → 0.001 m


def test_m_to_mm_function():
    assert m_to_mm(0.5) == pytest.approx(500.0)


def test_mm_to_m_function():
    assert mm_to_m(500.0) == pytest.approx(0.5)


def test_conversions_vectorised():
    arr = np.array([0.0, 0.1, 0.5])
    result = m_to_mm(arr)
    np.testing.assert_allclose(result, [0.0, 100.0, 500.0])


# ---------------------------------------------------------------------------
# DataFrame tagging
# ---------------------------------------------------------------------------

def test_tag_units():
    df = pd.DataFrame({"gmsl": [0.1, 0.2]})
    tag_units(df, {"gmsl": "m"})
    assert df.attrs["units"] == {"gmsl": "m"}


def test_tag_sign_convention():
    df = pd.DataFrame({"x": [1]})
    tag_sign_convention(df, "slr")
    assert df.attrs["sign_convention"] == "slr"


def test_tag_sign_convention_invalid():
    df = pd.DataFrame({"x": [1]})
    with pytest.raises(ValueError, match="Unknown sign convention"):
        tag_sign_convention(df, "bad")


def test_tag_baseline():
    df = pd.DataFrame({"x": [1]})
    tag_baseline(df, 2005.0, (1995, 2005))
    assert df.attrs["baseline_year"] == 2005.0
    assert df.attrs["baseline_window"] == (1995, 2005)


# ---------------------------------------------------------------------------
# Assertion guards
# ---------------------------------------------------------------------------

def test_assert_slr_convention_passes():
    df = pd.DataFrame({"x": [1]})
    df.attrs["sign_convention"] = "slr"
    assert_slr_convention(df)  # should not raise


def test_assert_slr_convention_fails():
    df = pd.DataFrame({"x": [1]})
    with pytest.raises(ValueError, match="sign_convention='slr'"):
        assert_slr_convention(df)


def test_assert_units_meters_passes():
    assert_units_meters(np.array([0.0, 0.3, -0.1]))  # valid m


def test_assert_units_meters_fails_for_mm():
    with pytest.raises(ValueError, match="likely in mm"):
        assert_units_meters(np.array([0.0, 300.0, 500.0]))


def test_assert_sigma_positive_passes():
    assert_sigma_positive(np.array([0.01, 0.02, 0.05]))


def test_assert_sigma_positive_fails():
    with pytest.raises(ValueError, match="Negative sigma"):
        assert_sigma_positive(np.array([0.01, -0.02, 0.05]))
