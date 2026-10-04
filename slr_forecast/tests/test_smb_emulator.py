"""Tests for the Greenland SMB emulator (notebooks/smb_emulator.py)."""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'notebooks'))
import smb_emulator as S

HAVE_DATA = S.GLAUDE_CSV.exists() and (S.NOEL_DIR / 'TGrIS_CESM2_Historical_1950-2014.dat').exists()


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


class TestProjection:
    def _emu(self, n, b1=-100.0, b2=-15.0):
        return {'b1': np.full(n, b1), 'b2': np.full(n, b2), 'rcm': np.array(['X'] * n)}

    def test_zero_warming_gives_baseline_rate(self):
        n, years = 5, np.arange(1990, 2011, dtype=float)
        out = S.project_smb_emulator(self._emu(n), {'s': np.zeros_like(years)}, years,
                                     np.full(n, 2.5), M0=380.0, baseline_year=2000)
        cum = out['s']['samples']
        assert np.allclose(cum[:, years == 2000], 0.0)
        assert np.allclose(np.diff(cum, axis=1), -380.0 * S.GT_TO_M_SLE)

    def test_matches_closed_form(self):
        n, years = 3, np.arange(2000, 2011, dtype=float)
        T = np.linspace(0, 1, len(years))
        aa = np.array([1.5, 2.5, 3.5])
        out = S.project_smb_emulator(self._emu(n), {'s': T}, years, aa, M0=0.0)
        x = aa[:, None] * T[None, :]
        expect = np.cumsum((100 * x + 15 * x**2) * S.GT_TO_M_SLE, axis=1)
        assert np.allclose(out['s']['samples'], expect)

    def test_offsets_are_added_per_member(self):
        n, years = 2, np.arange(2000, 2006, dtype=float)
        off = np.array([np.zeros(len(years)), np.ones(len(years))])
        out = S.project_smb_emulator(self._emu(n, b2=0.0), {'s': np.zeros(len(years))}, years,
                                     np.ones(n), M0=0.0, T_offsets={'s': off})
        assert np.allclose(out['s']['samples'][0], 0.0)
        assert np.all(np.diff(out['s']['samples'][1]) > 0)


@pytest.mark.skipif(not HAVE_DATA, reason='emulator training data not present')
class TestFittedEmulator:
    @pytest.fixture(scope='class')
    def emu(self):
        return S.fit_smb_emulator(n_samples=2000, seed=600)

    def test_default_models_exclude_hirham(self, emu):
        assert set(emu['fits']) == {'RACMO', 'MAR'}
        vals, counts = np.unique(emu['rcm'], return_counts=True)
        assert set(vals) == {'RACMO', 'MAR'} and counts[0] == counts[1]

    def test_racmo_curvature_matches_noel2021(self, emu):
        # Noel et al. (2021) Eq. 1 fitted the same RACMO run: b2 = -10.15
        lo, hi = np.percentile(emu['fits']['RACMO']['beta'][:, 2], [5, 95])
        assert lo < -10.15 < hi

    def test_mar_more_sensitive_than_racmo(self, emu):
        f = emu['fits']
        assert np.median(f['MAR']['beta'][:, 1]) < np.median(f['RACMO']['beta'][:, 1])
        assert np.median(f['MAR']['beta'][:, 2]) < np.median(f['RACMO']['beta'][:, 2])

    def test_coefficients_negative(self, emu):
        assert np.all(np.median(emu['b1']) < 0) and np.median(emu['b2']) < 0
