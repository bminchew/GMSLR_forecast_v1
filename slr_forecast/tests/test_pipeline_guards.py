"""Guards for the N_SAMPLES = 10000 final run and for partial reruns.

- branch_row_index (scripts/build_slow_wais_p90.py) at any branch size
- check_derived / stamp_derived (component_io) provenance of derived groups
- fit_bayesian_level_annual_correlated reports convergence diagnostics
"""

import importlib.util
import os
import sys

import h5py
import numpy as np
import pytest

ROOT = os.path.join(os.path.dirname(__file__), '..')
sys.path.insert(0, os.path.join(ROOT, 'notebooks'))

from component_io import check_derived, stamp_derived  # noqa: E402

_spec = importlib.util.spec_from_file_location(
    'build_slow_wais_p90', os.path.join(ROOT, 'scripts', 'build_slow_wais_p90.py'))
bsw = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(bsw)


class TestBranchRowIndex:

    def test_whole_multiple_tiles(self):
        rows = bsw.branch_row_index(np.arange(2000.0), 18000)
        assert np.array_equal(np.bincount(rows), np.full(2000, 9))

    def test_duplicates_every_member_once_at_least(self):
        rng = np.random.default_rng(0)
        rows = bsw.branch_row_index(rng.normal(size=1805), 2000)
        assert len(rows) == 2000
        assert np.all(np.bincount(rows, minlength=1805) >= 1)

    @pytest.mark.parametrize('n, n_rows', [(10000, 18000), (9025, 2000)])
    def test_10k_sizes(self, n, n_rows):
        """10k members: S1 to 18000 rows (not a multiple), S2 thinned to 2000."""
        rng = np.random.default_rng(1)
        v = rng.lognormal(size=n)
        rows = bsw.branch_row_index(v, n_rows)
        assert len(rows) == n_rows
        assert rows.max() < n
        # rank-spaced picks keep the distribution: quantiles within a few spacings
        for q in (5, 50, 95):
            assert abs(np.percentile(v[rows], q) - np.percentile(v, q)) < 0.02 * np.ptp(v)

    def test_mixture_rows_shape_at_10k(self):
        rng = np.random.default_rng(2)
        r1, r2 = bsw.mixture_rows(rng.normal(size=10000), rng.normal(size=9025))
        assert len(r1) + len(r2) == bsw.N_OUT
        assert len(r1) == int(round(bsw.P_S1 * bsw.N_OUT))


class TestDerivedProvenance:

    def _file(self, tmp_path):
        f = h5py.File(tmp_path / 'r.h5', 'w')
        f.create_group('blended').attrs['saved_at'] = '2026-10-17 10:00:00+00:00'
        for g in ('blended_slow90', 'voi'):
            f.create_group(g)
        return f

    def test_current_passes(self, tmp_path):
        with self._file(tmp_path) as f:
            for g in ('blended_slow90', 'voi'):
                stamp_derived(f, g)
            check_derived(f, ('blended_slow90', 'voi'))

    def test_stale_raises(self, tmp_path):
        with self._file(tmp_path) as f:
            stamp_derived(f, 'blended_slow90')
            stamp_derived(f, 'voi')
            f['blended'].attrs['saved_at'] = '2026-10-18 10:00:00+00:00'   # forecast rerun
            with pytest.raises(RuntimeError, match='blended_slow90'):
                check_derived(f, ('blended_slow90', 'voi'))

    def test_missing_raises(self, tmp_path):
        with self._file(tmp_path) as f:
            with pytest.raises(RuntimeError):
                check_derived(f, ('blended_slow90',))


def test_level_fitter_reports_convergence():
    from bayesian_models import fit_bayesian_level_annual_correlated
    rng = np.random.default_rng(3)
    n = 25
    t = 2000.0 + np.arange(n)
    T = 0.5 + 0.02 * np.arange(n) + rng.normal(0, 0.1, n)
    sig = np.full(n, 0.0002)
    rate = 0.001 * T + 0.0002 + rng.normal(0, 0.0002, n)
    H = np.concatenate([[0.0], np.cumsum(rate[1:])])
    res = fit_bayesian_level_annual_correlated(
        H, rate, sig, T, t, n_samples=1000, n_walkers=16, n_burnin=500,
        progress=False, seed=0)
    d = res.sampler_diagnostics
    for k in ('autocorr_time_max', 'rhat_max', 'ess_bulk_min', 'converged'):
        assert k in d
    assert np.isfinite(d['autocorr_time_max']) and d['autocorr_time_max'] > 0
    assert res.trace.posterior.sizes['chain'] == 16


@pytest.mark.parametrize('region', ['east_antarctica', 'antarctic_peninsula', 'west_antarctica'])
def test_imbie_annual_sigma_reproduces_cumulative(region):
    """AN-3: the annual 1-sigma from imbie_annual_rates (RMS of the monthly
    sigmas), accumulated over independent years, reproduces IMBIE-3's own
    cumulative sigma at each year end."""
    from slr_data_readers import read_imbie3
    from component_analysis import imbie_annual_rates
    path = os.path.join(ROOT, 'data', 'raw', 'ice_sheets', 'imbie2026',
                        f'imbie3_{region}_mm_partitioned.csv')
    if not os.path.exists(path):
        pytest.skip('IMBIE-3 data not available')
    df = read_imbie3(path, convert_to_meters=True)
    yrs, _, sig = imbie_annual_rates(df)
    t = df['decimal_year'].values
    cum_sig = df['cumulative_mass_balance_sigma'].values
    full = np.array([np.sum(np.floor(t) == y) == 12 for y in yrs])
    end_sig = np.array([cum_sig[np.floor(t) == y][-1] for y in yrs])
    pred = np.sqrt(np.cumsum(sig ** 2))
    # compare over the leading run of complete years (record starts in January)
    k = np.argmax(~full) if not full.all() else len(full)
    assert k > 10
    assert np.allclose(pred[:k], end_sig[:k], rtol=1e-3)
