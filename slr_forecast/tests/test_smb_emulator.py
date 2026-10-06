"""Tests for the Greenland SMB emulator (notebooks/smb_emulator.py)."""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'notebooks'))
import smb_emulator as S

HAVE_DATA = S.MAR_CSV.exists() and S.TEMP_CSV.exists()


class TestFitQuadraticAR1:
    def test_recovers_synthetic_coefficients(self):
        rng = np.random.default_rng(0)
        x = np.linspace(-2, 8, 150)
        eps = np.zeros_like(x)
        for k in range(1, len(x)):
            eps[k] = 0.3 * eps[k - 1] + rng.normal(0, 80)
        y = 20 - 90 * x - 12 * x**2 + eps
        r = S.fit_quadratic_ar1(x, y, 4000, np.random.default_rng(1))
        lo1, hi1 = np.percentile(r['beta'][:, 1], [0.5, 99.5])
        lo2, hi2 = np.percentile(r['beta'][:, 2], [0.5, 99.5])
        assert lo1 < -90 < hi1
        assert lo2 < -12 < hi2
        assert 0.0 < np.median(r['rho']) < 0.6

    def test_draw_shapes(self):
        x = np.linspace(0, 5, 40)
        r = S.fit_quadratic_ar1(x, -50 * x, 300, np.random.default_rng(2))
        assert r['beta'].shape == (300, 3)
        assert r['sigma'].shape == (300,) and np.all(r['sigma'] > 0)

    def test_recovers_coefficients_across_segments(self):
        # two AR(1) runs with one shared curve, as in the history + SSP fit
        rng = np.random.default_rng(3)
        xs = [np.linspace(-0.5, 0.5, 35), np.linspace(0.5, 5.0, 86)]
        ys = []
        for x in xs:
            eps = np.zeros_like(x)
            for k in range(1, len(x)):
                eps[k] = 0.3 * eps[k - 1] + rng.normal(0, 80)
            ys.append(30 - 50 * x - 40 * x**2 + eps)
        r = S.fit_quadratic_ar1(np.concatenate(xs), np.concatenate(ys), 4000,
                                np.random.default_rng(4), starts=(0, 35))
        lo1, hi1 = np.percentile(r['beta'][:, 1], [0.5, 99.5])
        lo2, hi2 = np.percentile(r['beta'][:, 2], [0.5, 99.5])
        assert lo1 < -50 < hi1
        assert lo2 < -40 < hi2

    def test_segment_breaks_change_the_whitening(self):
        Z = np.arange(6, dtype=float)
        one = S._prais_winsten(Z, 0.5, (0,))
        two = S._prais_winsten(Z, 0.5, (0, 3))
        assert np.allclose(one[:3], two[:3])
        assert two[3] == pytest.approx(np.sqrt(0.75) * 3.0)
        assert one[3] == pytest.approx(3.0 - 0.5 * 2.0)


class TestProjection:
    def _emu(self, n, b1=-100.0, b2=-15.0):
        return {'b1': np.full(n, b1), 'b2': np.full(n, b2), 'gcm': np.array(['X'] * n)}

    def test_zero_warming_gives_baseline_rate(self):
        n, years = 5, np.arange(1990, 2011, dtype=float)
        out = S.project_smb_emulator(self._emu(n), {'s': np.zeros_like(years)}, years,
                                     M0=380.0, baseline_year=2000)
        cum = out['s']['samples']
        assert np.allclose(cum[:, years == 2000], 0.0)
        assert np.allclose(np.diff(cum, axis=1), -380.0 * S.GT_TO_M_SLE)

    def test_matches_closed_form(self):
        n, years = 3, np.arange(2000, 2011, dtype=float)
        T = np.linspace(0, 3, len(years))
        b1 = np.array([-50.0, -80.0, -100.0])
        emu = {'b1': b1, 'b2': np.full(n, -15.0), 'gcm': np.array(['X'] * n)}
        out = S.project_smb_emulator(emu, {'s': T}, years, M0=0.0)
        x = np.tile(T, (n, 1))
        expect = np.cumsum((-b1[:, None] * x + 15 * x**2) * S.GT_TO_M_SLE, axis=1)
        assert np.allclose(out['s']['samples'], expect)

    def test_offsets_are_added_per_member(self):
        n, years = 2, np.arange(2000, 2006, dtype=float)
        off = np.array([np.zeros(len(years)), np.ones(len(years))])
        out = S.project_smb_emulator(self._emu(n, b2=0.0), {'s': np.zeros(len(years))}, years,
                                     M0=0.0, T_offsets={'s': off})
        assert np.allclose(out['s']['samples'][0], 0.0)
        assert np.all(np.diff(out['s']['samples'][1]) > 0)

    def test_per_member_anchor(self):
        n, years = 3, np.arange(2000, 2006, dtype=float)
        M0 = np.array([300.0, 374.0, 450.0])
        out = S.project_smb_emulator(self._emu(n), {'s': np.zeros(len(years))}, years, M0=M0)
        rate = np.diff(out['s']['samples'], axis=1)
        assert np.allclose(rate, -M0[:, None] * S.GT_TO_M_SLE)

    def test_smoothing_only_through_cutoff(self):
        years = np.arange(1990, 2031, dtype=float)
        T = np.where(years >= 2010, 1.0, 0.0)
        emu = {'b1': np.array([-1.0 / S.GT_TO_M_SLE]), 'b2': np.zeros(1), 'gcm': np.array(['X'])}
        out = S.project_smb_emulator(emu, {'s': T}, years, M0=0.0, smooth_through=2015)
        rate = np.diff(out['s']['samples'][0], prepend=0.0)   # = x, m SLE per K
        assert np.allclose(rate[years <= 2015], S._centred_mean(T)[years <= 2015])
        assert 0.0 < rate[years == 2008][0] < 1.0              # step smoothed before cutoff
        assert np.allclose(rate[years > 2015], 1.0)            # untouched after

    def test_noise_is_added_to_the_rate(self):
        n, years = 2, np.arange(2000, 2006, dtype=float)
        e = np.arange(n * len(years), dtype=float).reshape(n, -1)
        out = S.project_smb_emulator(self._emu(n), {'s': np.zeros(len(years))}, years,
                                     M0=0.0, noise=e)
        assert np.allclose(out['s']['samples'], np.cumsum(-e * S.GT_TO_M_SLE, axis=1))

    def test_ar1_noise_statistics(self):
        n = 4000
        e = S.ar1_noise(np.full(n, 80.0), np.full(n, 0.4), 50, np.random.default_rng(5))
        assert e[:, 0].std() == pytest.approx(80 / np.sqrt(1 - 0.16), rel=0.05)
        assert e[:, -1].std() == pytest.approx(80 / np.sqrt(1 - 0.16), rel=0.05)
        assert np.corrcoef(e[:, 20], e[:, 21])[0, 1] == pytest.approx(0.4, abs=0.05)

    def test_feedback_reproduces_fettweis_reference(self):
        # a constant anomaly whose 2000-2080 cumulative equals C_ref gives an
        # extra eps * A_ref at 2080
        years = np.arange(2000, 2101, dtype=float)
        a = S.FB_C_REF / 80.0
        F = S.elevation_feedback(np.full((1, len(years)), a), years, np.array([0.08]))
        assert F[0, years == 2000][0] == 0.0
        assert F[0, years == 2080][0] == pytest.approx(0.08 * S.FB_A_REF)

    def test_feedback_adds_loss_and_zero_eps_is_off(self):
        n, years = 2, np.arange(1990, 2101, dtype=float)
        T = np.clip(years - 2000, 0, None) / 25.0
        args = (self._emu(n), {'s': T}, years)
        base = S.project_smb_emulator(*args, M0=0.0)['s']['samples']
        off = S.project_smb_emulator(*args, M0=0.0, feedback_eps=np.zeros(n))['s']['samples']
        on = S.project_smb_emulator(*args, M0=0.0, feedback_eps=np.full(n, 0.08))['s']['samples']
        assert np.allclose(off, base)
        assert np.allclose(on[:, years <= 2000], base[:, years <= 2000])
        assert np.all(on[:, -1] > base[:, -1])

    def test_feedback_eps_truncated_at_zero(self):
        eps = S.draw_feedback_eps(20000, np.random.default_rng(6))
        assert eps.min() == 0.0
        assert np.median(eps) == pytest.approx(0.08, abs=0.005)

    def test_centred_mean_shortens_window_at_ends(self):
        T = np.arange(20, dtype=float)
        m = S._centred_mean(T)
        assert m[10] == pytest.approx(T[5:16].mean())
        assert m[0] == pytest.approx(T[:6].mean())
        assert m[-1] == pytest.approx(T[-6:].mean())


@pytest.mark.skipif(not HAVE_DATA, reason='emulator training data not present')
class TestFittedEmulator:
    @pytest.fixture(scope='class')
    def emu(self):
        return S.fit_smb_emulator(n_samples=2000, seed=600)

    def test_default_gcms_equal_weight(self, emu):
        assert set(emu['fits']) == set(S.GCMS)
        _, counts = np.unique(emu['gcm'], return_counts=True)
        assert counts.max() - counts.min() <= 1

    def test_each_gcm_fits_history_and_all_complete_ssps(self, emu):
        for g, f in emu['fits'].items():
            assert f['segments'][0] == 'history' and 'ssp585' in f['segments'], g

    def test_cesm2_matches_cross_gcm_script(self, emu):
        # scripts/cross_gcm_test_mar.py gives M 56.7, b1 -39.9, b2 -48.4 for CESM2
        m = np.median(emu['fits']['CESM2']['beta'], axis=0)
        assert m[0] == pytest.approx(56.7, abs=5)
        assert m[1] == pytest.approx(-39.9, abs=5)
        assert m[2] == pytest.approx(-48.4, abs=3)

    def test_coefficients_negative(self, emu):
        for g, f in emu['fits'].items():
            assert np.median(f['beta'][:, 1]) < 0 and np.median(f['beta'][:, 2]) < 0, g
