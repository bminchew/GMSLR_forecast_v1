"""Guards for the N_SAMPLES = 10000 final run and for partial reruns.

- branch_row_index (scripts/build_slow_wais_p90.py) at any branch size
- check_derived / stamp_derived (component_io) provenance of derived groups
- fit_bayesian_level_annual_correlated reports convergence diagnostics
- low-severity sweep guards: check_convergence tail ESS, component_io root
  attributes and skipped-metadata warnings, _anchor_covariance split,
  save_s1_quadratic skips identical rewrites
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


_PROCESSED = os.path.join(ROOT, 'data', 'processed', 'slr_processed_data.h5')


@pytest.mark.skipif(not os.path.exists(_PROCESSED), reason='slr_processed_data.h5 not built')
class TestProcessedDataCurrent:
    """slr_processed_data.h5 is built outside run_pipeline.sh, so stale tables can outlive
    reader fixes (MD-4, MD-24). Regenerate with scripts/refresh_processed_keys.py."""

    def test_berkeley_sigma_is_95pct_halfwidth_over_1p96(self):
        import pandas as pd
        for key in ('raw/df_berkeley', 'harmonized/df_berkeley_h'):
            df = pd.read_hdf(_PROCESSED, key)
            ratio = (df['temperature_unc'] / df['temperature_sigma']).dropna()
            np.testing.assert_allclose(ratio, 1.96, rtol=1e-9, err_msg=key)

    def test_no_negative_sigma_columns(self):
        import pandas as pd
        bad = []
        with pd.HDFStore(_PROCESSED, mode='r') as store:
            for key in store.keys():
                df = store[key]
                if not isinstance(df, pd.DataFrame):
                    continue
                for c in df.columns:
                    if str(c).endswith('_sigma') and (pd.to_numeric(df[c], errors='coerce') < 0).any():
                        bad.append(f'{key}:{c}')
        assert not bad, f'negative sigma columns: {bad}'

    def test_ipcc_observed_sigma_is_halfrange_over_0p9542(self):
        import pandas as pd
        for key in ('raw/df_ipcc_observed_gmsl', 'harmonized/df_ipcc_observed_gmsl_h'):
            df = pd.read_hdf(_PROCESSED, key)
            half = (df['gmsl_upper'] - df['gmsl_lower']) / 2
            np.testing.assert_allclose(half / df['gmsl_sigma'], 0.9542, rtol=1e-9, err_msg=key)


# ---------------------------------------------------------------------------
# Low-severity sweep (MD-11, MD-12, MD-13, MD-19, IN-23)
# ---------------------------------------------------------------------------


def test_check_convergence_requires_tail_ess(monkeypatch):
    """MD-11: `converged` checks tail ESS as well as R-hat and bulk ESS."""
    import arviz as az
    import bayesian_models
    from bayesian_models import check_convergence
    rng = np.random.default_rng(0)
    good = az.from_dict({'x': rng.normal(size=(8, 2000))})
    conv = check_convergence(good, quiet=True)
    assert conv['converged'] and min(conv['ess_tail'].values()) >= 100
    # Same chains with a tail ESS below 100 (bulk ESS and R-hat unchanged)
    real_ess = az.ess

    def fake_ess(trace, method='bulk', **kw):
        out = real_ess(trace, method=method, **kw)
        return out * 1e-3 if method == 'tail' else out
    monkeypatch.setattr(bayesian_models.az, 'ess', fake_ess)
    conv = check_convergence(good, quiet=True)
    assert min(conv['ess_bulk'].values()) >= 100
    assert min(conv['ess_tail'].values()) < 100
    assert not conv['converged']


class TestComponentIO:

    def test_root_baseline_refreshed_and_no_root_sample_count(self, tmp_path):
        """MD-12: a file created under an old baseline gets the current
        baseline_year on the next save, and the stale root n_samples
        attribute is dropped."""
        from component_io import _require_file, BASELINE_YEAR
        p = tmp_path / 'c.h5'
        with h5py.File(p, 'w') as f:
            f.attrs['baseline_year'] = 2005.0
            f.attrs['n_samples'] = 2000
            f.create_dataset('proj_years', data=np.arange(3.0))
        _require_file(p)
        with h5py.File(p, 'r') as f:
            assert f.attrs['baseline_year'] == BASELINE_YEAR
            assert 'n_samples' not in f.attrs

    def test_unsaveable_metadata_warns(self, tmp_path):
        """MD-13: metadata that cannot be stored is reported, not dropped
        silently; arrays with ndim > 2 in a projection dict likewise."""
        from component_io import _write_scalar_attrs, _save_projection_ssp
        with h5py.File(tmp_path / 'm.h5', 'w') as f:
            g = f.create_group('x')
            with pytest.warns(UserWarning, match='bad'):
                _write_scalar_attrs(g, {'ok': 1.0, 's': 'a', 'bad': None})
            assert g.attrs['ok'] == 1.0 and g.attrs['s'] == 'a'
            assert 'bad' not in g.attrs
            prg = f.create_group('projections')
            with pytest.warns(UserWarning, match='cube'):
                _save_projection_ssp(prg, 'SSP2-4.5', {
                    'median': np.zeros(3), 'cube': np.zeros((2, 2, 2))})
            assert 'cube' not in prg['SSP2-4.5']


def test_anchor_covariance_splits_on_baseline_year():
    """MD-19: with end-of-year labels and the zero-sigma 2000.0 row dropped,
    1999.0 and 2001.0 tie in argmin; 1999 must still be on the pre side."""
    from component_levelspace_robust_se import _anchor_covariance
    years = np.array([1997.0, 1998.0, 1999.0, 2001.0, 2002.0, 2003.0])
    sig = np.array([3.0, 2.0, 1.0, 1.0, 2.0, 3.0])
    S, _ = _anchor_covariance(years, sig, 2000.0)
    assert S[2, 3] == 0.0 and S[3, 2] == 0.0      # opposite sides
    assert S[1, 2] == 1.0                          # 1998, 1999 share 1999's history
    assert S[3, 4] == 1.0                          # 2001, 2002 share 2001's history
    assert np.linalg.eigvalsh(S).min() > -1e-12


def test_s1_quadratic_identical_refit_does_not_rewrite(tmp_path):
    """IN-23: an identical refit leaves the tracked JSON (and its
    fitted_at stamp) untouched; a changed fit rewrites it."""
    import json
    import component_projections as cp
    saved = cp._S1_QUADRATIC
    try:
        mean = np.array([1.2e-5, 2.5e-4, 0.0])
        cov = np.diag([1e-12, 1e-10, 0.0])
        p = tmp_path / 's1.json'
        cp.save_s1_quadratic(mean, cov, path=p, provenance='test')
        stamp = json.loads(p.read_text())['fitted_at']
        cp.save_s1_quadratic(mean, cov, path=p, provenance='test')
        assert json.loads(p.read_text())['fitted_at'] == stamp
        cp.save_s1_quadratic(mean * 1.01, cov, path=p, provenance='test')
        assert json.loads(p.read_text())['mean'][0] == pytest.approx(1.212e-5)
    finally:
        cp._S1_QUADRATIC = saved
