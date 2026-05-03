"""Unit tests for the Bayesian rate-and-state semi-empirical sea-level model.

Tests cover: solve_state_ode, build_state_level_design_vectors,
forward model H = a*I2 + b*I1 + c*I0 + d*IS + H0, fit_satellite_era_quadratic,
calibrate_exponential_prior, project_gmsl_state_ensemble, sensitivity
calculations, _rate_accel_prior_logp, and _state_level_log_prior bounds.
"""

import sys
import os

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'notebooks'))

from bayesian_models import (
    solve_state_ode,
    build_state_level_design_vectors,
    _state_level_log_prior,
    _state_level_log_prob,
    _rate_accel_prior_logp,
    fit_satellite_era_quadratic,
    calibrate_exponential_prior,
    SatelliteEraQuadraticResult,
)
from slr_projections import project_gmsl_state_ensemble

from slr_forecast import M_TO_MM


# =========================================================================
# Fixtures
# =========================================================================

@pytest.fixture
def constant_temperature():
    """Constant temperature = 1.0 C over 100 years, monthly cadence."""
    n = 1200  # 100 years * 12 months
    time = np.linspace(2000.0, 2100.0, n)
    T = np.ones(n) * 1.0
    return T, time


@pytest.fixture
def ramp_temperature():
    """Linearly increasing temperature from 0 to 2 C over 100 years."""
    n = 1200
    time = np.linspace(2000.0, 2100.0, n)
    T = np.linspace(0.0, 2.0, n)
    return T, time


@pytest.fixture
def prior_scales_default():
    """Default prior scales array (length 9) for _state_level_log_prior."""
    return np.array([
        0.002,   # [0] Exp mean for a
        0.010,   # [1] HN sigma for b
        0.002,   # [2] Normal mu for c
        0.005,   # [3] Normal sigma for c
        0.001,   # [4] Exp mean for d
        np.log(30.0),  # [5] LogNormal mu for log(tau)
        1.0,     # [6] LogNormal sigma for log(tau)
        0.001,   # [7] HC gamma for sigma_extra
        0.010,   # [8] Normal sigma for H0
    ])


@pytest.fixture
def satellite_era_prior():
    """A mock SatelliteEraQuadraticResult for rate/accel prior tests."""
    rate = 3.5e-3      # m/yr
    accel = 0.1e-3      # m/yr^2
    cov = np.array([
        [0.3e-3**2, 0.0],
        [0.0, 0.05e-3**2],
    ])
    return SatelliteEraQuadraticResult(
        rate=rate,
        accel=accel,
        rate_accel_cov=cov,
        rate_se=np.sqrt(cov[0, 0]),
        accel_se=np.sqrt(cov[1, 1]),
        eval_time=2020.0,
        coefficients=np.array([0.0, rate, accel / 2.0]),
        cov_params=np.eye(3) * 1e-8,
        t_start=1993.0,
        t_end=2020.0,
        n_obs=324,
        r2=0.99,
        fit_result=None,
        fit_method='WLS',
    )


# =========================================================================
# 1. solve_state_ode
# =========================================================================

class TestSolveStateODE:
    """Tests for the state-variable ODE solver."""

    def test_constant_temperature_convergence(self, constant_temperature):
        """With constant T, S should converge to T with timescale tau."""
        T, time = constant_temperature
        tau = 10.0
        S0 = 0.0  # start away from equilibrium
        S = solve_state_ode(T, time, tau, S0=S0)

        # Analytical solution: S(t) = T + (S0 - T)*exp(-(t - t0)/tau)
        T_val = T[0]
        expected = T_val + (S0 - T_val) * np.exp(-(time - time[0]) / tau)
        np.testing.assert_allclose(S, expected, rtol=1e-3)

    def test_constant_temperature_equilibrium(self, constant_temperature):
        """When S0=T[0], S should remain at T for all time."""
        T, time = constant_temperature
        tau = 50.0
        S = solve_state_ode(T, time, tau, S0=None)  # defaults to T[0]
        np.testing.assert_allclose(S, T, atol=1e-12)

    def test_small_tau_returns_temperature(self, constant_temperature):
        """For tau < 0.01, should return T.copy() (instantaneous limit)."""
        T, time = constant_temperature
        S = solve_state_ode(T, time, tau=0.005, S0=0.5)
        np.testing.assert_array_equal(S, T)

    def test_S0_defaults_to_T0(self, ramp_temperature):
        """When S0 is None, S[0] should equal T[0]."""
        T, time = ramp_temperature
        S = solve_state_ode(T, time, tau=20.0, S0=None)
        assert S[0] == T[0]

    def test_output_shape(self, constant_temperature):
        """Output shape should match input."""
        T, time = constant_temperature
        S = solve_state_ode(T, time, tau=10.0)
        assert S.shape == T.shape

    def test_large_tau_slow_response(self, ramp_temperature):
        """With large tau, S should lag far behind T."""
        T, time = ramp_temperature
        tau = 500.0
        S = solve_state_ode(T, time, tau, S0=0.0)
        # At end of record, S should be much less than T
        assert S[-1] < T[-1] * 0.5

    def test_small_tau_fast_response(self, ramp_temperature):
        """With small tau (but > 0.01), S should closely track T."""
        T, time = ramp_temperature
        tau = 0.05
        S = solve_state_ode(T, time, tau)
        # Should essentially equal T for tau -> 0
        np.testing.assert_allclose(S, T, atol=0.05)

    def test_exponential_decay_timescale(self):
        """Verify that 1/e decay occurs at t = tau."""
        n = 10000
        tau = 25.0
        time = np.linspace(0.0, 200.0, n)
        T = np.ones(n) * 1.0
        S0 = 0.0
        S = solve_state_ode(T, time, tau, S0=S0)

        # At t = tau, displacement should be ~1/e of initial
        idx_tau = np.argmin(np.abs(time - tau))
        displacement_ratio = (T[0] - S[idx_tau]) / (T[0] - S0)
        np.testing.assert_allclose(displacement_ratio, np.exp(-1.0), rtol=0.01)


# =========================================================================
# 2. build_state_level_design_vectors
# =========================================================================

class TestBuildStateDesignVectors:
    """Tests for design vector construction."""

    def test_constant_temperature_integrals(self):
        """With constant T, I1 = T*t, I2 = T^2*t, I0 = t."""
        n = 600  # 50 years monthly
        time = np.linspace(2000.0, 2050.0, n)
        T_val = 1.5
        T = np.ones(n) * T_val
        obs_times = np.array([2010.0, 2020.0, 2030.0, 2040.0, 2050.0])

        result = build_state_level_design_vectors(T, time, obs_times, tau=20.0)

        # I0 = t - t[0]
        for i, t_obs in enumerate(obs_times):
            idx = result['obs_idx'][i]
            elapsed = time[idx] - time[0]
            np.testing.assert_allclose(result['I0_obs'][i], elapsed, atol=0.1)

        # I1 = integral of T dt = T * elapsed (constant T)
        for i, t_obs in enumerate(obs_times):
            idx = result['obs_idx'][i]
            elapsed = time[idx] - time[0]
            np.testing.assert_allclose(
                result['I1_obs'][i], T_val * elapsed, rtol=0.01)

        # I2 = integral of T^2 dt = T^2 * elapsed (constant T)
        for i, t_obs in enumerate(obs_times):
            idx = result['obs_idx'][i]
            elapsed = time[idx] - time[0]
            np.testing.assert_allclose(
                result['I2_obs'][i], T_val**2 * elapsed, rtol=0.01)

    def test_constant_temp_zero_disequilibrium(self):
        """With constant T and S0=T[0], IS should be zero (no disequilibrium)."""
        n = 600
        time = np.linspace(2000.0, 2050.0, n)
        T = np.ones(n) * 1.0
        obs_times = np.array([2025.0])

        result = build_state_level_design_vectors(T, time, obs_times, tau=20.0)
        np.testing.assert_allclose(result['IS_obs'], 0.0, atol=1e-10)
        np.testing.assert_allclose(result['IS_full'], 0.0, atol=1e-10)

    def test_shape_consistency(self, ramp_temperature):
        """Output arrays should have consistent shapes."""
        T, time = ramp_temperature
        obs_times = np.linspace(2010.0, 2090.0, 9)
        result = build_state_level_design_vectors(T, time, obs_times, tau=30.0)

        n_monthly = len(T)
        n_obs = len(obs_times)

        assert result['I2_obs'].shape == (n_obs,)
        assert result['I1_obs'].shape == (n_obs,)
        assert result['I0_obs'].shape == (n_obs,)
        assert result['IS_obs'].shape == (n_obs,)
        assert result['I2_full'].shape == (n_monthly,)
        assert result['I1_full'].shape == (n_monthly,)
        assert result['I0_full'].shape == (n_monthly,)
        assert result['IS_full'].shape == (n_monthly,)
        assert result['S_full'].shape == (n_monthly,)
        assert result['obs_idx'].shape == (n_obs,)

    def test_integrals_start_at_zero(self, ramp_temperature):
        """All cumulative integrals should start at zero."""
        T, time = ramp_temperature
        obs_times = np.array([time[0]])
        result = build_state_level_design_vectors(T, time, obs_times, tau=10.0)

        assert result['I2_full'][0] == 0.0
        assert result['I1_full'][0] == 0.0
        assert result['I0_full'][0] == 0.0
        assert result['IS_full'][0] == 0.0

    def test_tau_stored_in_result(self, ramp_temperature):
        """The tau value should be stored in the output dict."""
        T, time = ramp_temperature
        obs_times = np.array([2050.0])
        result = build_state_level_design_vectors(T, time, obs_times, tau=42.0)
        assert result['tau'] == 42.0


# =========================================================================
# 3. Forward model consistency
# =========================================================================

class TestForwardModel:
    """Tests for H_model = a*I2 + b*I1 + c*I0 + d*IS + H0."""

    def test_constant_temp_equilibrium(self):
        """With constant T and S=T (equilibrium), H = (a*T^2+b*T+c)*t + H0."""
        n = 1200
        time = np.linspace(2000.0, 2100.0, n)
        T_val = 1.0
        T = np.ones(n) * T_val
        obs_times = np.linspace(2000.0, 2100.0, 101)

        a, b, c, d = 0.001, 0.003, 0.002, 0.0005
        H0 = 0.05  # meters

        result = build_state_level_design_vectors(T, time, obs_times, tau=20.0)

        H_model = (a * result['I2_obs'] + b * result['I1_obs']
                    + c * result['I0_obs'] + d * result['IS_obs'] + H0)

        # Expected: H = (a*T^2 + b*T + c)*elapsed + H0
        # because IS = 0 at equilibrium
        rate = a * T_val**2 + b * T_val + c
        elapsed = obs_times - obs_times[0]
        # Find the nearest monthly time for each obs_time
        obs_idx = result['obs_idx']
        elapsed_actual = time[obs_idx] - time[0]
        H_expected = rate * elapsed_actual + H0

        np.testing.assert_allclose(H_model, H_expected, rtol=1e-3)

    def test_units_are_meters(self):
        """Verify that with typical coefficient magnitudes, output is in meters."""
        n = 600
        time = np.linspace(2000.0, 2050.0, n)
        T = np.ones(n) * 1.0
        obs_times = np.array([2050.0])

        # Coefficients in m/yr (with per-C powers)
        a = 0.001      # m/yr/C^2
        b = 0.003      # m/yr/C
        c = 0.002      # m/yr
        d = 0.0005     # m/yr
        H0 = 0.0       # m

        result = build_state_level_design_vectors(T, time, obs_times, tau=20.0)
        H_model = (a * result['I2_obs'] + b * result['I1_obs']
                    + c * result['I0_obs'] + d * result['IS_obs'] + H0)

        # Rate ~ 0.006 m/yr, over 50 years ~ 0.3 m
        assert 0.1 < H_model[0] < 1.0, (
            f"H_model = {H_model[0]} m; expected O(0.1-1) m for 50-yr projection")


# =========================================================================
# 4. fit_satellite_era_quadratic
# =========================================================================

class TestFitSatelliteEraQuadratic:
    """Tests for quadratic fitting of GMSL records."""

    @pytest.fixture
    def synthetic_quadratic(self):
        """Synthetic data: H = c0 + c1*(t-t0) + c2*(t-t0)^2, no noise."""
        c0, c1, c2 = 0.0, 3.5e-3, 0.05e-3
        t_start = 1993.0
        time = np.linspace(t_start, 2020.0, 324)  # monthly
        dt = time - t_start
        gmsl = c0 + c1 * dt + c2 * dt**2
        return time, gmsl, c0, c1, c2, t_start

    def test_recover_coefficients(self, synthetic_quadratic):
        """Should recover known polynomial coefficients from noiseless data."""
        time, gmsl, c0, c1, c2, t_start = synthetic_quadratic
        result = fit_satellite_era_quadratic(
            time, gmsl, sigma=None, t_start=t_start)

        np.testing.assert_allclose(result.coefficients[0], c0, atol=1e-8)
        np.testing.assert_allclose(result.coefficients[1], c1, rtol=1e-6)
        np.testing.assert_allclose(result.coefficients[2], c2, rtol=1e-4)

    def test_rate_at_eval_time(self, synthetic_quadratic):
        """Rate at eval time should equal c1 + 2*c2*(t_eval - t0)."""
        time, gmsl, c0, c1, c2, t_start = synthetic_quadratic
        eval_time = 2020.0
        result = fit_satellite_era_quadratic(
            time, gmsl, sigma=None, t_start=t_start, eval_time=eval_time)

        expected_rate = c1 + 2.0 * c2 * (eval_time - t_start)
        np.testing.assert_allclose(result.rate, expected_rate, rtol=1e-5)

    def test_acceleration(self, synthetic_quadratic):
        """Acceleration should be 2*c2."""
        time, gmsl, c0, c1, c2, t_start = synthetic_quadratic
        result = fit_satellite_era_quadratic(
            time, gmsl, sigma=None, t_start=t_start)

        np.testing.assert_allclose(result.accel, 2.0 * c2, rtol=1e-4)

    def test_cov_params_shape(self, synthetic_quadratic):
        """Parameter covariance should be (3,3)."""
        time, gmsl, _, _, _, t_start = synthetic_quadratic
        result = fit_satellite_era_quadratic(
            time, gmsl, sigma=None, t_start=t_start)
        assert result.cov_params.shape == (3, 3)

    def test_cov_params_positive_semidefinite(self, synthetic_quadratic):
        """Parameter covariance should be positive semi-definite."""
        time, gmsl, _, _, _, t_start = synthetic_quadratic
        # Add small noise for a non-degenerate covariance
        rng = np.random.default_rng(42)
        gmsl_noisy = gmsl + rng.normal(0, 1e-4, len(gmsl))
        result = fit_satellite_era_quadratic(
            time, gmsl_noisy, sigma=None, t_start=t_start)
        eigvals = np.linalg.eigvalsh(result.cov_params)
        assert np.all(eigvals >= -1e-15), f"Negative eigenvalue: {eigvals.min()}"

    def test_rate_accel_cov_shape(self, synthetic_quadratic):
        """Rate/acceleration covariance should be (2,2)."""
        time, gmsl, _, _, _, t_start = synthetic_quadratic
        result = fit_satellite_era_quadratic(
            time, gmsl, sigma=None, t_start=t_start)
        assert result.rate_accel_cov.shape == (2, 2)

    def test_wls_mode(self, synthetic_quadratic):
        """WLS mode should run when sigma is provided."""
        time, gmsl, _, _, _, t_start = synthetic_quadratic
        sigma = np.ones_like(gmsl) * 1e-3
        result = fit_satellite_era_quadratic(
            time, gmsl, sigma=sigma, t_start=t_start)
        assert result.fit_method == 'WLS'
        assert result.n_obs == len(time)

    def test_few_observations_raises(self):
        """Should raise ValueError if fewer than 5 observations."""
        time = np.array([1993.0, 1994.0, 1995.0])
        gmsl = np.array([0.0, 0.001, 0.002])
        with pytest.raises(ValueError, match="at least 5"):
            fit_satellite_era_quadratic(time, gmsl)


# =========================================================================
# 5. calibrate_exponential_prior
# =========================================================================

class TestCalibrateExponentialPrior:
    """Tests for PC prior calibration."""

    def test_tail_probability(self):
        """P(X > threshold) should equal prob for the returned scale."""
        prob = 0.10
        threshold = 0.005
        mu = calibrate_exponential_prior(prob, threshold)
        # For Exponential(mu), P(X > u) = exp(-u/mu)
        actual_prob = np.exp(-threshold / mu)
        np.testing.assert_allclose(actual_prob, prob, rtol=1e-10)

    def test_different_parameters(self):
        """Verify with several (prob, threshold) combinations."""
        cases = [
            (0.05, 0.005),
            (0.10, 0.003),
            (0.20, 0.010),
            (0.01, 0.001),
        ]
        for prob, threshold in cases:
            mu = calibrate_exponential_prior(prob, threshold)
            actual = np.exp(-threshold / mu)
            np.testing.assert_allclose(actual, prob, rtol=1e-10,
                                       err_msg=f"Failed for prob={prob}, thr={threshold}")

    def test_larger_prob_gives_larger_scale(self):
        """Higher probability of exceeding threshold requires larger scale."""
        mu1 = calibrate_exponential_prior(0.05, 0.005)
        mu2 = calibrate_exponential_prior(0.20, 0.005)
        assert mu2 > mu1

    def test_invalid_prob_raises(self):
        """Probability outside (0,1) should raise."""
        with pytest.raises(ValueError):
            calibrate_exponential_prior(0.0, 0.005)
        with pytest.raises(ValueError):
            calibrate_exponential_prior(1.0, 0.005)

    def test_negative_threshold_raises(self):
        """Negative threshold should raise."""
        with pytest.raises(ValueError):
            calibrate_exponential_prior(0.10, -0.005)


# =========================================================================
# 6. project_gmsl_state_ensemble
# =========================================================================

class TestProjectGmslStateEnsemble:
    """Tests for ensemble projection with the rate-and-state model."""

    @pytest.fixture
    def constant_projection_inputs(self):
        """Inputs for a constant-temperature projection."""
        T_val = 1.0
        a, b, c, d = 0.001, 0.003, 0.002, 0.0001

        # Historical: 50 years monthly
        n_hist = 600
        hist_time = np.linspace(1950.0, 2000.0, n_hist)
        hist_T = np.ones(n_hist) * T_val

        # Projection: 100 years annual
        n_proj = 100
        proj_time = np.linspace(2000.0, 2100.0, n_proj)
        proj_df = pd.DataFrame({
            'temperature': np.ones(n_proj) * T_val,
            'decimal_year': proj_time,
        }, index=pd.date_range('2000-01-01', periods=n_proj, freq='YS'))

        coefficients = np.array([a, b, c, d])
        # Very small covariance -> nearly deterministic
        cov = np.eye(4) * 1e-20
        tau_samples = np.ones(200) * 30.0  # fixed tau

        return {
            'coefficients': coefficients,
            'coefficients_cov': cov,
            'tau_samples': tau_samples,
            'temperature_projections': {'const': proj_df},
            'historical_temperature': hist_T,
            'historical_time': hist_time,
            'baseline_year': 2000.0,
            'baseline_gmsl': 0.0,
            'n_samples': 50,
            'seed': 42,
        }

    def test_constant_temp_rate(self, constant_projection_inputs):
        """With constant T and near-zero cov, median rate ~ a*T^2 + b*T + c."""
        inputs = constant_projection_inputs
        a, b, c, d = inputs['coefficients']
        T_val = 1.0

        result = project_gmsl_state_ensemble(**inputs)
        df = result['scenarios']['const']

        # At equilibrium (S=T), rate = a*T^2 + b*T + c (d*(S-T)=0)
        expected_rate = a * T_val**2 + b * T_val + c
        # Check rates away from the start (where spin-up effects are small)
        median_rate = df['rate'].iloc[-20:].mean()
        np.testing.assert_allclose(median_rate, expected_rate, rtol=0.05)

    def test_output_shape(self, constant_projection_inputs):
        """Output median should have shape (n_time,)."""
        result = project_gmsl_state_ensemble(**constant_projection_inputs)
        df = result['scenarios']['const']
        n_proj = 100
        assert len(df) == n_proj

    def test_percentile_ordering(self, constant_projection_inputs):
        """p5 < median < p95 for GMSL, except possibly at baseline."""
        # Increase covariance so uncertainty is visible
        inputs = constant_projection_inputs.copy()
        inputs['coefficients_cov'] = np.diag([1e-8, 1e-8, 1e-8, 1e-8])
        inputs['n_samples'] = 200

        result = project_gmsl_state_ensemble(**inputs)
        df = result['scenarios']['const']

        # Check away from baseline where differences should be nonzero
        idx = df['decimal_year'] > 2050.0
        assert np.all(df.loc[idx, 'gmsl_lower'] <= df.loc[idx, 'gmsl'] + 1e-10)
        assert np.all(df.loc[idx, 'gmsl'] <= df.loc[idx, 'gmsl_upper'] + 1e-10)

    def test_gmsl_monotonic_with_positive_rate(self, constant_projection_inputs):
        """With all positive rates, GMSL should be non-decreasing after baseline."""
        result = project_gmsl_state_ensemble(**constant_projection_inputs)
        df = result['scenarios']['const']
        gmsl = df['gmsl'].values
        idx_base = np.argmin(np.abs(df['decimal_year'].values - 2000.0))
        diffs = np.diff(gmsl[idx_base:])
        assert np.all(diffs >= -1e-10), "GMSL should be non-decreasing"


# =========================================================================
# 7. Sensitivity calculations
# =========================================================================

class TestSensitivity:
    """Tests for rate-temperature sensitivity at equilibrium."""

    def test_equilibrium_sensitivity(self):
        """At equilibrium (S=T), dRate/dT = 2*a*T + b."""
        a = 0.001    # m/yr/C^2
        b = 0.003    # m/yr/C
        T = 1.0      # C

        # rate = a*T^2 + b*T + c + d*(S-T)
        # At equilibrium S=T, so dRate/dT = 2*a*T + b
        sensitivity = 2.0 * a * T + b
        expected = 0.005  # 2*0.001*1 + 0.003
        np.testing.assert_allclose(sensitivity, expected, atol=1e-12)

    def test_record_averaged_sensitivity(self):
        """Record-averaged sensitivity = a*(T1+T2) + b for linear T."""
        a = 0.001
        b = 0.003
        T1, T2 = 0.5, 1.5

        # Average of dRate/dT over [T1, T2]:
        # (1/(T2-T1)) * integral_{T1}^{T2} (2*a*T + b) dT
        #   = (1/(T2-T1)) * [a*T^2 + b*T]_{T1}^{T2}
        #   = a*(T1+T2) + b
        avg_sensitivity = a * (T1 + T2) + b
        expected = 0.001 * 2.0 + 0.003
        np.testing.assert_allclose(avg_sensitivity, expected, atol=1e-12)

    def test_sensitivity_units(self):
        """Sensitivity should be in m/yr/C."""
        a = 0.001    # m/yr/C^2
        b = 0.003    # m/yr/C
        T = 1.0

        # dRate/dT has units: d(m/yr)/d(C) = m/yr/C
        sensitivity = 2.0 * a * T + b
        # Convert to mm/yr/C for comparison
        sensitivity_mm = sensitivity * M_TO_MM
        assert 1.0 < sensitivity_mm < 20.0, (
            f"Sensitivity {sensitivity_mm} mm/yr/C outside plausible range")


# =========================================================================
# 8. _rate_accel_prior_logp
# =========================================================================

class TestRateAccelPrior:
    """Tests for the bivariate Gaussian rate/accel prior."""

    def test_zero_penalty_at_match(self, satellite_era_prior):
        """Returns 0 when model rate/accel exactly match the prior."""
        lp = _rate_accel_prior_logp(
            satellite_era_prior.rate,
            satellite_era_prior.accel,
            satellite_era_prior,
        )
        np.testing.assert_allclose(lp, 0.0, atol=1e-12)

    def test_negative_when_different(self, satellite_era_prior):
        """Returns negative value when model values differ from prior."""
        lp = _rate_accel_prior_logp(
            satellite_era_prior.rate + 1e-3,
            satellite_era_prior.accel + 0.5e-3,
            satellite_era_prior,
        )
        assert lp < 0.0

    def test_further_deviation_more_negative(self, satellite_era_prior):
        """Larger deviations should give more negative log-probability."""
        lp_small = _rate_accel_prior_logp(
            satellite_era_prior.rate + 0.1e-3,
            satellite_era_prior.accel,
            satellite_era_prior,
        )
        lp_large = _rate_accel_prior_logp(
            satellite_era_prior.rate + 1.0e-3,
            satellite_era_prior.accel,
            satellite_era_prior,
        )
        assert lp_large < lp_small

    def test_symmetric(self, satellite_era_prior):
        """Positive and negative deviations should give same log-probability."""
        delta = 0.5e-3
        lp_pos = _rate_accel_prior_logp(
            satellite_era_prior.rate + delta,
            satellite_era_prior.accel,
            satellite_era_prior,
        )
        lp_neg = _rate_accel_prior_logp(
            satellite_era_prior.rate - delta,
            satellite_era_prior.accel,
            satellite_era_prior,
        )
        np.testing.assert_allclose(lp_pos, lp_neg, rtol=1e-10)


# =========================================================================
# 9. Log-prior bounds (_state_level_log_prior)
# =========================================================================

class TestStateLevelLogPriorBounds:
    """Tests for hard boundary enforcement in the log-prior."""

    def _make_theta(self, a=0.001, b=0.003, c=0.002, d=0.0005,
                    log_tau=np.log(30.0), log_sigma_extra=np.log(1e-3),
                    H0=0.0):
        return np.array([a, b, c, d, log_tau, log_sigma_extra, H0])

    def test_valid_parameters_finite(self, prior_scales_default):
        """Valid parameters should give finite log-prior."""
        theta = self._make_theta()
        lp = _state_level_log_prior(theta, prior_scales_default, H0_prior_mean=0.0)
        assert np.isfinite(lp)

    def test_negative_a_returns_neginf(self, prior_scales_default):
        """a < 0 should return -inf."""
        theta = self._make_theta(a=-0.001)
        lp = _state_level_log_prior(theta, prior_scales_default, H0_prior_mean=0.0)
        assert lp == -np.inf

    def test_negative_b_returns_neginf(self, prior_scales_default):
        """b < 0 should return -inf."""
        theta = self._make_theta(b=-0.001)
        lp = _state_level_log_prior(theta, prior_scales_default, H0_prior_mean=0.0)
        assert lp == -np.inf

    def test_negative_d_returns_neginf(self, prior_scales_default):
        """d < 0 should return -inf."""
        theta = self._make_theta(d=-0.001)
        lp = _state_level_log_prior(theta, prior_scales_default, H0_prior_mean=0.0)
        assert lp == -np.inf

    def test_very_small_tau_returns_neginf(self, prior_scales_default):
        """tau < 0.1 (log_tau very negative) should return -inf."""
        theta = self._make_theta(log_tau=np.log(0.05))
        lp = _state_level_log_prior(theta, prior_scales_default, H0_prior_mean=0.0)
        assert lp == -np.inf

    def test_very_large_tau_returns_neginf(self, prior_scales_default):
        """tau > 5000 should return -inf."""
        theta = self._make_theta(log_tau=np.log(6000.0))
        lp = _state_level_log_prior(theta, prior_scales_default, H0_prior_mean=0.0)
        assert lp == -np.inf

    def test_tiny_sigma_extra_returns_neginf(self, prior_scales_default):
        """sigma_extra < 1e-12 should return -inf."""
        theta = self._make_theta(log_sigma_extra=np.log(1e-15))
        lp = _state_level_log_prior(theta, prior_scales_default, H0_prior_mean=0.0)
        assert lp == -np.inf

    def test_larger_a_less_probable(self, prior_scales_default):
        """Exponential prior: larger a should have lower log-prior."""
        theta_small = self._make_theta(a=0.0001)
        theta_large = self._make_theta(a=0.01)
        lp_small = _state_level_log_prior(
            theta_small, prior_scales_default, H0_prior_mean=0.0)
        lp_large = _state_level_log_prior(
            theta_large, prior_scales_default, H0_prior_mean=0.0)
        assert lp_small > lp_large

    def test_c_centered_on_prior_mean(self, prior_scales_default):
        """c at the prior mean should give higher log-prior than far away."""
        c_mean = prior_scales_default[2]
        theta_center = self._make_theta(c=c_mean)
        theta_far = self._make_theta(c=c_mean + 10.0 * prior_scales_default[3])
        lp_center = _state_level_log_prior(
            theta_center, prior_scales_default, H0_prior_mean=0.0)
        lp_far = _state_level_log_prior(
            theta_far, prior_scales_default, H0_prior_mean=0.0)
        assert lp_center > lp_far


# =========================================================================
# Integration: _state_level_log_prob
# =========================================================================

class TestStateLevelLogProb:
    """Integration tests for the full log-posterior."""

    @pytest.fixture
    def logprob_inputs(self, prior_scales_default):
        """Construct minimal inputs for _state_level_log_prob."""
        n = 600
        time = np.linspace(2000.0, 2050.0, n)
        T = np.ones(n) * 1.0
        obs_times = np.linspace(2001.0, 2049.0, 49)
        dv = build_state_level_design_vectors(T, time, obs_times, tau=30.0)

        # Synthetic observations: H = a*I2 + b*I1 + c*I0 + H0 (no disequilibrium)
        a, b, c, d = 0.001, 0.003, 0.002, 0.0005
        H0 = 0.05
        H_obs = (a * dv['I2_obs'] + b * dv['I1_obs']
                 + c * dv['I0_obs'] + d * dv['IS_obs'] + H0)
        sigma_obs = np.ones(len(obs_times)) * 0.001

        return {
            'T_monthly': T,
            'time_monthly': time,
            'obs_idx': dv['obs_idx'],
            'I2_obs': dv['I2_obs'],
            'I1_obs': dv['I1_obs'],
            'I0_obs': dv['I0_obs'],
            'H_obs': H_obs,
            'sigma_obs_fixed': sigma_obs,
            'prior_scales': prior_scales_default,
            'H0_prior_mean': H0,
        }

    def test_true_params_high_logprob(self, logprob_inputs):
        """True parameters should give a high (finite) log-posterior."""
        theta = np.array([0.001, 0.003, 0.002, 0.0005,
                          np.log(30.0), np.log(0.001), 0.05])
        lp = _state_level_log_prob(theta, **logprob_inputs)
        assert np.isfinite(lp)

    def test_wrong_params_lower_logprob(self, logprob_inputs):
        """Wrong parameters should give a lower log-posterior than true params."""
        theta_true = np.array([0.001, 0.003, 0.002, 0.0005,
                               np.log(30.0), np.log(0.001), 0.05])
        theta_wrong = np.array([0.005, 0.010, 0.010, 0.005,
                                np.log(30.0), np.log(0.001), 0.05])
        lp_true = _state_level_log_prob(theta_true, **logprob_inputs)
        lp_wrong = _state_level_log_prob(theta_wrong, **logprob_inputs)
        assert lp_true > lp_wrong

    def test_negative_a_returns_neginf(self, logprob_inputs):
        """Negative a should return -inf from the prior."""
        theta = np.array([-0.001, 0.003, 0.002, 0.0005,
                          np.log(30.0), np.log(0.001), 0.05])
        lp = _state_level_log_prob(theta, **logprob_inputs)
        assert lp == -np.inf
