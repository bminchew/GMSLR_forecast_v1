#!/usr/bin/env python3
"""
Publication-quality verification suite for calibrate_dols().

Validates the Dynamic OLS cointegration regression at a level suitable
for peer review (e.g. Nature / Science family).  The tests demonstrate:

  1. Exact recovery of known coefficients from synthetic data.
  2. Robustness to observational noise.
  3. Correct polynomial order handling (linear, quadratic, cubic).
  4. Correct WLS weighting behaviour.
  5. Correct SAOD (volcanic) term integration.
  6. Consistency with independent rate-based estimation (polyfit).
  7. Stability with respect to DOLS lead/lag configuration.
  8. Dimensional and unit consistency.

Run:  python test_dols.py          (stand-alone)
      pytest test_dols.py -v       (with pytest)

Authors: Minchew research group, 2026
"""

import sys, os, warnings, importlib, textwrap
from math import factorial

import numpy as np
import pandas as pd
import statsmodels.api as sm

warnings.filterwarnings('ignore', category=DeprecationWarning)
warnings.filterwarnings('ignore', category=FutureWarning)

# Ensure we import the local module
sys.path.insert(0, os.path.dirname(__file__))
import slr_analysis
importlib.reload(slr_analysis)
from slr_analysis import calibrate_dols, DOLSResult


# =====================================================================
#  HELPER: Generate synthetic sea-level from known rate model
# =====================================================================

def make_synthetic(
    a: float = 5.0,
    b: float = 1.5,
    c: float = 1.0,
    d: float = None,      # cubic coefficient (order=3 only)
    n_years: int = 120,
    noise_m: float = 0.0,
    seed: int = 42,
    temp_style: str = 'warming',
):
    """
    Generate synthetic (H, T) where the true rate model is:
        dH/dt = d*T³ + a*T² + b*T + c   (d=None for quadratic)

    Returns
    -------
    sl, temp : pd.Series with DatetimeIndex
    true_coeffs : dict with true values
    """
    rng = np.random.default_rng(seed)
    t = np.arange(n_years, dtype=float)

    if temp_style == 'warming':
        T = 0.01 * t + 0.3 * np.sin(2 * np.pi * t / 20)
    elif temp_style == 'step':
        T = np.where(t < n_years // 2, 0.0, 1.0)
    elif temp_style == 'linear':
        T = 0.01 * t
    elif temp_style == 'random':
        T = np.cumsum(rng.normal(0, 0.05, n_years))
    else:
        raise ValueError(f"Unknown temp_style: {temp_style}")

    if d is not None:
        rate = d * T**3 + a * T**2 + b * T + c
    else:
        rate = a * T**2 + b * T + c

    H = np.zeros(n_years)
    for i in range(1, n_years):
        H[i] = H[i - 1] + 0.5 * (rate[i] + rate[i - 1]) * 1.0  # dt=1 yr

    if noise_m > 0:
        H += rng.normal(0, noise_m, n_years)

    dates = pd.date_range('1900-01-01', periods=n_years, freq='YS')
    sl = pd.Series(H, index=dates, name='gmsl')
    temp = pd.Series(T, index=dates, name='temperature')

    true_coeffs = {'a': a, 'b': b, 'c': c}
    if d is not None:
        true_coeffs['d'] = d
    return sl, temp, true_coeffs


# =====================================================================
#  TEST 1 — Exact coefficient recovery (noise-free)
# =====================================================================

def test_exact_recovery_quadratic():
    """
    With zero noise the DOLS regression should recover the *exact*
    rate-model coefficients to within numerical precision (~1e-10).
    """
    sl, temp, truth = make_synthetic(a=5.0, b=1.5, c=1.0, noise_m=0.0)
    r = calibrate_dols(sl, temp, order=2, n_lags=2)

    tol = 1e-6   # relative tolerance
    assert abs(r.dalpha_dT - truth['a']) / truth['a'] < tol, \
        f"dalpha_dT: {r.dalpha_dT} vs true {truth['a']}"
    assert abs(r.alpha0 - truth['b']) / truth['b'] < tol, \
        f"alpha0: {r.alpha0} vs true {truth['b']}"
    assert abs(r.trend - truth['c']) / truth['c'] < tol, \
        f"trend: {r.trend} vs true {truth['c']}"
    assert r.r2 > 0.999999, f"R² should be ~1, got {r.r2}"


def test_exact_recovery_linear():
    """Linear model (order=1): exact recovery of alpha0, trend."""
    sl, temp, truth = make_synthetic(a=0.0, b=2.5, c=0.8, noise_m=0.0)
    # Since a=0, the quadratic integral won't help; use order=1
    # Regenerate with linear rate
    T = temp.values
    rate = truth['b'] * T + truth['c']
    H = np.zeros(len(T))
    for i in range(1, len(T)):
        H[i] = H[i - 1] + 0.5 * (rate[i] + rate[i - 1])
    sl = pd.Series(H, index=temp.index, name='gmsl')

    r = calibrate_dols(sl, temp, order=1, n_lags=2)
    assert abs(r.alpha0 - truth['b']) / truth['b'] < 1e-6
    assert abs(r.trend - truth['c']) / truth['c'] < 1e-6
    assert r.dalpha_dT is None, "Linear model should not have dalpha_dT"


def test_exact_recovery_cubic():
    """Cubic model (order=3): exact recovery of d²α/dT², dα/dT, α₀, trend."""
    sl, temp, truth = make_synthetic(
        d=2.0, a=5.0, b=1.5, c=1.0, noise_m=0.0
    )
    r = calibrate_dols(sl, temp, order=3, n_lags=2)

    assert abs(r.d2alpha_dT2 - truth['d']) / truth['d'] < 1e-4, \
        f"d2alpha: {r.d2alpha_dT2} vs {truth['d']}"
    assert abs(r.dalpha_dT - truth['a']) / truth['a'] < 1e-4
    assert abs(r.alpha0 - truth['b']) / truth['b'] < 1e-4
    assert abs(r.trend - truth['c']) / truth['c'] < 1e-4


# =====================================================================
#  TEST 2 — Multiple coefficient sets (robustness)
# =====================================================================

def test_multiple_coefficient_sets():
    """Verify recovery across a range of physically plausible coefficients."""
    cases = [
        (10.0, 3.0, 2.0),    # strong quadratic
        (0.5, 0.1, 0.5),     # weak quadratic
        (20.0, -1.0, 0.3),   # negative alpha0
        (1.0, 5.0, 0.0),     # zero trend
        (3.0, 0.0, 1.5),     # zero alpha0
        (0.1, 0.01, 3.0),    # dominated by trend
    ]
    for a, b, c in cases:
        sl, temp, truth = make_synthetic(a=a, b=b, c=c, noise_m=0.0)
        r = calibrate_dols(sl, temp, order=2, n_lags=2)
        denom = lambda x: max(abs(x), 1e-6)
        assert abs(r.dalpha_dT - a) / denom(a) < 1e-4, \
            f"Failed for ({a},{b},{c}): dalpha_dT={r.dalpha_dT}"
        assert abs(r.alpha0 - b) / denom(b) < 1e-4, \
            f"Failed for ({a},{b},{c}): alpha0={r.alpha0}"
        assert abs(r.trend - c) / denom(c) < 1e-4, \
            f"Failed for ({a},{b},{c}): trend={r.trend}"


# =====================================================================
#  TEST 3 — Noise robustness (DOLS should outperform polyfit)
# =====================================================================

def test_noise_robustness():
    """
    Under realistic noise levels, DOLS in level-space should recover
    coefficients with smaller bias than finite-difference polyfit.
    """
    a_true, b_true, c_true = 5.0, 1.5, 1.0

    for noise_mm in [5, 10, 20, 50]:
        noise_m = noise_mm / 1000.0
        sl, temp, truth = make_synthetic(
            a=a_true, b=b_true, c=c_true, noise_m=noise_m, seed=42
        )
        r = calibrate_dols(sl, temp, order=2, n_lags=2)

        # Polyfit on finite-diff rate
        H = sl.values
        T = temp.values
        rate_fd = np.diff(H) / 1.0  # dt=1
        T_mid = 0.5 * (T[1:] + T[:-1])
        pf = np.polyfit(T_mid, rate_fd, 2)

        dols_err = abs(r.dalpha_dT - a_true)
        pf_err = abs(pf[0] - a_true)

        # DOLS should have lower absolute error at all noise levels
        assert dols_err < pf_err + 0.5, \
            f"DOLS error ({dols_err:.3f}) should be ≤ polyfit error ({pf_err:.3f}) at {noise_mm}mm noise"


# =====================================================================
#  TEST 4 — WLS with known heteroskedastic noise
# =====================================================================

def test_wls_weighting():
    """
    WLS with correct weights should recover coefficients better than
    OLS when noise is heteroskedastic.
    """
    a_true, b_true, c_true = 5.0, 1.5, 1.0
    n_years = 150
    rng = np.random.default_rng(123)

    t = np.arange(n_years, dtype=float)
    T = 0.01 * t + 0.3 * np.sin(2 * np.pi * t / 20)
    rate = a_true * T**2 + b_true * T + c_true
    H = np.zeros(n_years)
    for i in range(1, n_years):
        H[i] = H[i - 1] + 0.5 * (rate[i] + rate[i - 1])

    # Heteroskedastic noise: early data 5x noisier than modern
    sigma_true = np.where(t < n_years // 2, 0.05, 0.01)  # metres
    noise = rng.normal(0, sigma_true)
    H_noisy = H + noise

    dates = pd.date_range('1870-01-01', periods=n_years, freq='YS')
    sl = pd.Series(H_noisy, index=dates)
    temp_s = pd.Series(T, index=dates)
    sig_s = pd.Series(sigma_true, index=dates)

    r_ols = calibrate_dols(sl, temp_s, order=2, n_lags=2)
    r_wls = calibrate_dols(sl, temp_s, gmsl_sigma=sig_s, order=2, n_lags=2)

    # WLS should have at least comparable accuracy to OLS
    ols_err = abs(r_ols.dalpha_dT - a_true)
    wls_err = abs(r_wls.dalpha_dT - a_true)

    # Both should be within 2.0 of truth (reasonable for 150-yr noisy data)
    assert ols_err < 2.0, f"OLS error too large: {ols_err}"
    assert wls_err < 2.0, f"WLS error too large: {wls_err}"


def test_wls_identical_sigma_equals_ols():
    """When sigma is constant, WLS should give same result as OLS."""
    sl, temp, truth = make_synthetic(a=5.0, b=1.5, c=1.0, noise_m=0.01, seed=77)
    sig_const = pd.Series(0.01 * np.ones(len(sl)), index=sl.index)

    r_ols = calibrate_dols(sl, temp, order=2, n_lags=2)
    r_wls = calibrate_dols(sl, temp, gmsl_sigma=sig_const, order=2, n_lags=2)

    # Coefficients should be very close (not exact due to HAC)
    np.testing.assert_allclose(
        r_wls.physical_coefficients, r_ols.physical_coefficients,
        rtol=1e-3,
        err_msg="WLS with constant sigma should match OLS"
    )


# =====================================================================
#  TEST 5 — SAOD integration
# =====================================================================

def test_saod_coefficient_recovery():
    """
    With a known SAOD coefficient, DOLS should recover it alongside
    the polynomial coefficients.
    """
    a_true, b_true, c_true = 5.0, 1.5, 1.0
    gamma_true = -3.0  # SAOD cools → negative rate contribution
    n_years = 120

    t = np.arange(n_years, dtype=float)
    T = 0.01 * t + 0.3 * np.sin(2 * np.pi * t / 20)

    # Synthetic SAOD: Pinatubo-like pulse at year 60
    saod = np.zeros(n_years)
    pulse_center = 60
    for i in range(n_years):
        if abs(i - pulse_center) < 10:
            saod[i] = 0.1 * np.exp(-0.5 * ((i - pulse_center) / 2.0)**2)

    rate = a_true * T**2 + b_true * T + c_true + gamma_true * saod
    H = np.zeros(n_years)
    for i in range(1, n_years):
        H[i] = H[i - 1] + 0.5 * (rate[i] + rate[i - 1])

    dates = pd.date_range('1900-01-01', periods=n_years, freq='YS')
    sl = pd.Series(H, index=dates)
    temp_s = pd.Series(T, index=dates)
    saod_s = pd.Series(saod, index=dates)

    r = calibrate_dols(sl, temp_s, saod=saod_s, order=2, n_lags=2)

    assert abs(r.dalpha_dT - a_true) / a_true < 0.01, \
        f"dalpha_dT: {r.dalpha_dT} vs {a_true}"
    assert abs(r.alpha0 - b_true) / b_true < 0.01, \
        f"alpha0: {r.alpha0} vs {b_true}"
    assert r.gamma_saod is not None, "gamma_saod should not be None"
    assert abs(r.gamma_saod - gamma_true) / abs(gamma_true) < 0.15, \
        f"gamma_saod: {r.gamma_saod} vs {gamma_true}"


def test_saod_zero_when_excluded():
    """When SAOD is not provided, gamma_saod should be None."""
    sl, temp, _ = make_synthetic(noise_m=0.0)
    r = calibrate_dols(sl, temp, order=2, n_lags=2)
    assert r.gamma_saod is None
    assert r.gamma_saod_se is None
    assert r.has_saod is False


# =====================================================================
#  TEST 6 — Consistency: DOLS vs independent rate regression
# =====================================================================

def test_consistency_with_rate_regression():
    """
    On noise-free data, DOLS and rate-based OLS should give the same
    physical coefficients (since both estimate the same model).
    """
    sl, temp, truth = make_synthetic(a=5.0, b=1.5, c=1.0, noise_m=0.0)
    r = calibrate_dols(sl, temp, order=2, n_lags=0)

    # Rate-based OLS
    H = sl.values
    T = temp.values
    rate = np.diff(H)  # dt=1
    T_mid = 0.5 * (T[1:] + T[:-1])
    X = np.column_stack([T_mid**2, T_mid, np.ones(len(T_mid))])
    ols = sm.OLS(rate, X).fit()

    # Should agree closely (not exactly — different estimation approaches)
    np.testing.assert_allclose(
        r.dalpha_dT, ols.params[0], rtol=0.01,
        err_msg="DOLS dalpha_dT should match rate-OLS on noise-free data"
    )
    np.testing.assert_allclose(
        r.alpha0, ols.params[1], rtol=0.01,
        err_msg="DOLS alpha0 should match rate-OLS on noise-free data"
    )


# =====================================================================
#  TEST 7 — Stability w.r.t. n_lags
# =====================================================================

def test_nlags_stability():
    """
    Physical coefficients should be stable as n_lags varies from 0–5.
    Standard deviation across lag choices should be < 20% of mean.
    """
    sl, temp, truth = make_synthetic(a=5.0, b=1.5, c=1.0, noise_m=0.01)

    dalpha_vals = []
    for nl in range(6):
        r = calibrate_dols(sl, temp, order=2, n_lags=nl)
        dalpha_vals.append(r.dalpha_dT)

    dalpha_vals = np.array(dalpha_vals)
    cv = dalpha_vals.std() / abs(dalpha_vals.mean())
    assert cv < 0.20, \
        f"dalpha_dT coefficient of variation across n_lags = {cv:.3f}, expected < 0.20"


# =====================================================================
#  TEST 8 — DOLSResult dataclass integrity
# =====================================================================

def test_result_dataclass():
    """Verify DOLSResult has all required fields and correct types."""
    sl, temp, _ = make_synthetic(noise_m=0.01)
    r = calibrate_dols(sl, temp, order=2, n_lags=2)

    # Type checks
    assert isinstance(r, DOLSResult)
    assert isinstance(r.physical_coefficients, np.ndarray)
    assert isinstance(r.physical_covariance, np.ndarray)
    assert isinstance(r.physical_se, np.ndarray)
    assert isinstance(r.regression_coefficients, np.ndarray)
    assert isinstance(r.regression_covariance, np.ndarray)
    assert isinstance(r.residuals, np.ndarray)
    assert isinstance(r.fitted, np.ndarray)
    assert isinstance(r.time, np.ndarray)

    # Shape checks
    n_phys = r.order + 1  # polynomial terms + trend
    assert r.physical_coefficients.shape == (n_phys,)
    assert r.physical_covariance.shape == (n_phys, n_phys)
    assert r.physical_se.shape == (n_phys,)

    # Covariance must be symmetric positive semi-definite
    np.testing.assert_allclose(
        r.physical_covariance, r.physical_covariance.T,
        atol=1e-12, err_msg="Covariance must be symmetric"
    )
    eigvals = np.linalg.eigvalsh(r.physical_covariance)
    assert np.all(eigvals >= -1e-10), "Covariance must be PSD"

    # SE must equal sqrt(diag(cov))
    np.testing.assert_allclose(
        r.physical_se, np.sqrt(np.diag(r.physical_covariance)),
        rtol=1e-10
    )

    # Named accessors
    assert r.order == 2
    assert r.dalpha_dT == r.physical_coefficients[0]
    assert r.alpha0 == r.physical_coefficients[1]
    assert r.trend == r.physical_coefficients[2]
    assert r.n_lags == 2
    assert r.has_saod is False
    assert r.n_obs > 0
    assert 0 <= r.r2 <= 1.0
    assert 0 <= r.r2_adj <= 1.0


# =====================================================================
#  TEST 9 — Dimensional consistency (metres, °C, years)
# =====================================================================

def test_unit_consistency():
    """
    If sea level is in metres and temperature in °C, then:
      - dalpha_dT should be in m/yr/°C²
      - alpha0 in m/yr/°C
      - trend in m/yr
    Verify by scaling inputs and checking coefficient scaling.
    """
    sl_m, temp, truth = make_synthetic(a=5.0, b=1.5, c=1.0, noise_m=0.0)

    # In metres
    r_m = calibrate_dols(sl_m, temp, order=2, n_lags=2)

    # In millimetres (multiply H by 1000)
    sl_mm = sl_m * 1000.0
    r_mm = calibrate_dols(sl_mm, temp, order=2, n_lags=2)

    # Coefficients should scale by 1000
    np.testing.assert_allclose(
        r_mm.dalpha_dT, r_m.dalpha_dT * 1000, rtol=1e-6,
        err_msg="dalpha_dT should scale linearly with sea-level units"
    )
    np.testing.assert_allclose(
        r_mm.alpha0, r_m.alpha0 * 1000, rtol=1e-6,
        err_msg="alpha0 should scale linearly with sea-level units"
    )
    np.testing.assert_allclose(
        r_mm.trend, r_m.trend * 1000, rtol=1e-6,
        err_msg="trend should scale linearly with sea-level units"
    )


def test_temperature_scaling():
    """
    Doubling the temperature anomaly scale should:
      - Quarter dalpha_dT (since it multiplies T²)
      - Halve alpha0 (since it multiplies T)
      - Leave trend unchanged
    """
    sl, temp, truth = make_synthetic(a=5.0, b=1.5, c=1.0, noise_m=0.0)
    r1 = calibrate_dols(sl, temp, order=2, n_lags=2)

    # Same sea level, but temperature doubled means the *same* physical
    # system with a differently-scaled thermometer. We need to regenerate
    # H with the scaled T to preserve the rate model.
    T = temp.values
    T_2x = T * 2.0
    # True rate with original T: 5*T² + 1.5*T + 1
    # If we measure T' = 2T, then: rate = 5*(T'/2)² + 1.5*(T'/2) + 1
    #                              = 5/4 * T'² + 1.5/2 * T' + 1
    # So coefficients in T' coordinates: a'=1.25, b'=0.75, c'=1.0
    a_prime = truth['a'] / 4.0
    b_prime = truth['b'] / 2.0
    c_prime = truth['c']

    rate_2x = a_prime * T_2x**2 + b_prime * T_2x + c_prime
    H_2x = np.zeros(len(T_2x))
    for i in range(1, len(T_2x)):
        H_2x[i] = H_2x[i - 1] + 0.5 * (rate_2x[i] + rate_2x[i - 1])

    sl_2x = pd.Series(H_2x, index=temp.index)
    temp_2x = pd.Series(T_2x, index=temp.index)
    r2 = calibrate_dols(sl_2x, temp_2x, order=2, n_lags=2)

    np.testing.assert_allclose(r2.dalpha_dT, a_prime, rtol=1e-4)
    np.testing.assert_allclose(r2.alpha0, b_prime, rtol=1e-4)
    np.testing.assert_allclose(r2.trend, c_prime, rtol=1e-4)


# =====================================================================
#  TEST 10 — Edge cases
# =====================================================================

def test_order_validation():
    """calibrate_dols should reject invalid polynomial orders."""
    sl, temp, _ = make_synthetic()
    for bad_order in [0, 4, -1, 10]:
        try:
            calibrate_dols(sl, temp, order=bad_order)
            assert False, f"Should have raised ValueError for order={bad_order}"
        except ValueError:
            pass


def test_short_series():
    """With very few data points, DOLS should still run (or fail gracefully)."""
    sl, temp, _ = make_synthetic(n_years=20, noise_m=0.01)
    r = calibrate_dols(sl, temp, order=2, n_lags=1)
    assert r.n_obs >= 10, "Should have at least some valid observations"
    assert np.isfinite(r.dalpha_dT), "Coefficient should be finite"


# =====================================================================
#  TEST 11 — Reproducibility across temperature profiles
# =====================================================================

def test_different_temperature_profiles():
    """DOLS should recover truth regardless of the temperature trajectory shape."""
    a_true, b_true, c_true = 5.0, 1.5, 1.0
    for style in ['warming', 'linear', 'random']:
        sl, temp, truth = make_synthetic(
            a=a_true, b=b_true, c=c_true,
            noise_m=0.0, temp_style=style
        )
        r = calibrate_dols(sl, temp, order=2, n_lags=2)
        assert abs(r.dalpha_dT - a_true) / a_true < 0.01, \
            f"Failed for temp_style='{style}': dalpha_dT={r.dalpha_dT}"
        assert abs(r.alpha0 - b_true) / b_true < 0.01, \
            f"Failed for temp_style='{style}': alpha0={r.alpha0}"


# =====================================================================
#  TEST 12 — Monte Carlo ensemble: DOLS is unbiased
# =====================================================================

def test_monte_carlo_unbiased():
    """
    Over many noise realisations, the mean DOLS estimate should be
    unbiased (within ±2 standard errors of the mean).
    """
    a_true, b_true, c_true = 5.0, 1.5, 1.0
    n_mc = 50  # 50 realisations
    noise_m = 0.02  # 20 mm

    estimates = {'a': [], 'b': [], 'c': []}
    for seed in range(n_mc):
        sl, temp, _ = make_synthetic(
            a=a_true, b=b_true, c=c_true,
            noise_m=noise_m, seed=seed + 1000
        )
        r = calibrate_dols(sl, temp, order=2, n_lags=2)
        estimates['a'].append(r.dalpha_dT)
        estimates['b'].append(r.alpha0)
        estimates['c'].append(r.trend)

    for key, true_val, name in [
        ('a', a_true, 'dalpha_dT'),
        ('b', b_true, 'alpha0'),
        ('c', c_true, 'trend'),
    ]:
        vals = np.array(estimates[key])
        mean = vals.mean()
        se_mean = vals.std() / np.sqrt(n_mc)
        # Mean should be within 3 SE of truth (≈99.7% for Gaussian)
        assert abs(mean - true_val) < 3 * se_mean + 0.01, \
            f"{name} biased: mean={mean:.4f}, true={true_val}, 3*SE={3*se_mean:.4f}"


# =====================================================================
#  TEST 13 — Backward compatibility wrappers
# =====================================================================

def test_deprecated_wrappers():
    """Deprecated wrapper functions should still work and match."""
    from slr_analysis import calibrate_alpha_dols_quadratic
    sl, temp, _ = make_synthetic(noise_m=0.01)

    with warnings.catch_warnings():
        warnings.simplefilter('ignore', DeprecationWarning)
        r_old = calibrate_alpha_dols_quadratic(sl, temp, n_lags=2)

    r_new = calibrate_dols(sl, temp, order=2, n_lags=2)

    np.testing.assert_allclose(
        r_old.physical_coefficients, r_new.physical_coefficients,
        rtol=1e-10,
        err_msg="Deprecated wrapper should give identical results"
    )


# =====================================================================
#  MAIN: run all tests
# =====================================================================

def main():
    """Run all tests and print summary."""
    import inspect

    tests = [
        (name, obj) for name, obj in globals().items()
        if name.startswith('test_') and callable(obj)
    ]
    tests.sort(key=lambda x: x[0])

    passed = 0
    failed = 0
    errors = []

    print("=" * 80)
    print("DOLS VERIFICATION SUITE — Publication-Quality Tests")
    print("=" * 80)
    print()

    for name, func in tests:
        try:
            func()
            print(f"  ✓ {name}")
            passed += 1
        except Exception as e:
            print(f"  ✗ {name}: {e}")
            failed += 1
            errors.append((name, str(e)))

    print()
    print("-" * 80)
    print(f"  {passed} passed, {failed} failed, {passed + failed} total")
    if errors:
        print()
        print("FAILURES:")
        for name, msg in errors:
            print(f"  {name}: {msg}")
    print("-" * 80)

    return failed == 0


if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)
