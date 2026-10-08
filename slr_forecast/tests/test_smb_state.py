"""Tests for the Greenland SMB drift state (notebooks/smb_state.py)."""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'notebooks'))
import smb_state as Z


@pytest.fixture
def record():
    rng = np.random.default_rng(1)
    T = 47
    r = 380.0 + rng.normal(0.0, 120.0, T)
    return r, np.full(T, 55.0)


def test_ffbs_matches_smoother(record):
    """Path draws reproduce the Rauch-Tung-Striebel mean and sd of delta."""
    r, s = record
    r = r.copy(); r[40:44] = np.nan
    phi, sd, se = np.exp(-1 / 23), 110.0, 55.0
    _, _, _, _, _, sm, ss = Z.kalman(r, s, [phi], [sd], [se], return_path=True)
    B = 20000
    p = Z.ffbs(np.tile(r, (B, 1)), s, np.full(B, phi), np.full(B, sd), np.full(B, se),
               np.random.default_rng(2))
    assert np.abs(p[:, :, 1].mean(0) - sm[0]).max() < 3.0
    assert np.abs(p[:, :, 1].std(0) - ss[0]).max() < 2.0


def test_level_is_constant_along_path(record):
    r, s = record
    B = 50
    p = Z.ffbs(np.tile(r, (B, 1)), s, np.full(B, 0.95), np.full(B, 100.0),
               np.full(B, 55.0), np.random.default_rng(3))
    assert np.allclose(p[:, :, 0], p[:, :1, 0], atol=1e-3)


def test_fit_drift_state_shapes_and_relaxation(record):
    r, s = record
    years = np.arange(1972, 2019, dtype=float)
    grid = np.arange(1900, 2151, dtype=float)
    n = 400
    b1, b2 = np.full(n, -50.0), np.full(n, -40.0)
    x = np.linspace(-0.3, 0.6, len(years))
    out = Z.fit_drift_state(r, s, x, b1, b2, years, grid, seed=4)
    assert out['delta'].shape == (n, len(grid))
    assert out['M'].shape == (n,)
    assert np.all(np.isfinite(out['delta']))
    # far from the record the state is the stationary AR(1): mean zero,
    # sd near sigma_d
    pos = out['sd_d'] > 0
    far = out['delta'][pos, -1]
    assert abs(np.mean(far)) < 3 * np.std(far) / np.sqrt(pos.sum())
    assert np.std(far / out['sd_d'][pos]) == pytest.approx(1.0, abs=0.15)
    # a member with sigma_d = 0 has no drift anywhere
    assert np.all(np.abs(out['delta'][~pos]) < 1e-3)   # Gt/yr; jitter only


def test_fit_drift_state_rejects_gaps(record):
    r, s = record
    years = np.arange(1972, 2019, dtype=float); years[5] += 0.5
    with pytest.raises(ValueError):
        Z.fit_drift_state(r, s, np.zeros(len(years)), np.zeros(3), np.zeros(3),
                          years, np.arange(1900, 2151, dtype=float), seed=0)
