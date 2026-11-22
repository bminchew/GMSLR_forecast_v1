"""Guards that need no data files.

- slr_forecast.config._n_samples_from_env: parsing of SLR_N_SAMPLES
- notebooks/impact_costs.py: the adaptation-cost and discount-rate constants
  quoted in the paper, and the damage function on a small input
- default subcommands of scripts/thermosteric_record_sensitivity.py ('all')
  and scripts/cmip6_zostoga.py ('analyze'), the two run_pipeline.sh entries
  that run without arguments
- scripts/cmip6_zostoga.py extract-ar6 on a small synthetic file
"""

import argparse
import ast
import importlib.util
import os
import sys

import h5py
import numpy as np
import pytest

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
sys.path.insert(0, os.path.join(ROOT, 'notebooks'))

from slr_forecast.config import _n_samples_from_env  # noqa: E402


# =========================================================================
# SLR_N_SAMPLES
# =========================================================================

class TestNSamplesFromEnv:
    """_n_samples_from_env reads the variable at call time; monkeypatch
    restores the environment afterwards, and config is not reloaded."""

    def test_unset_gives_default(self, monkeypatch):
        monkeypatch.delenv('SLR_N_SAMPLES', raising=False)
        assert _n_samples_from_env(10000) == 10000

    @pytest.mark.parametrize('raw', ['', '   ', '\t\n'])
    def test_blank_gives_default(self, monkeypatch, raw):
        monkeypatch.setenv('SLR_N_SAMPLES', raw)
        assert _n_samples_from_env(10000) == 10000

    @pytest.mark.parametrize('raw, n', [('500', 500), (' 7 ', 7), ('10000', 10000),
                                        ('1', 1), ('007', 7)])
    def test_valid(self, monkeypatch, raw, n):
        monkeypatch.setenv('SLR_N_SAMPLES', raw)
        out = _n_samples_from_env(10000)
        assert out == n and type(out) is int

    @pytest.mark.parametrize('raw', ['0', '00', '-5', 'abc', '1e4', '10_000', '+7',
                                     '２０', '5.0', '1 000', '0x10'])
    def test_invalid(self, monkeypatch, raw):
        monkeypatch.setenv('SLR_N_SAMPLES', raw)
        with pytest.raises(ValueError, match='SLR_N_SAMPLES'):
            _n_samples_from_env(10000)


# =========================================================================
# impact_costs
# =========================================================================

class TestImpactCosts:

    def test_constants(self):
        import impact_costs as ic
        assert ic.DR == 0.03
        assert ic.HORIZON == 75
        assert ic.ADAPT_CAPITAL_PER_M == 4000
        # $4T/m annualized with the capital recovery factor at 3% over 75 yr
        assert ic.ADAPT_COST_PER_M == 134.6718535998775

    def test_damage_vec(self):
        import impact_costs as ic
        from slr_forecast.readers.impacts import (JEVREJEVA2018_COST_B,
                                                  JEVREJEVA2018_SLR_M)
        # At the tabulated points the function returns the tabulated costs
        np.testing.assert_allclose(ic.damage_vec(JEVREJEVA2018_SLR_M),
                                   JEVREJEVA2018_COST_B, rtol=0, atol=1e-9)
        x = np.array([-0.5, 0.0, 0.1, 0.5, 1.0, 2.0])
        y = ic.damage_vec(x)
        assert isinstance(y, np.ndarray) and y.shape == x.shape
        slope = (27000.0 - 14300.0) / (1.80 - 0.86)
        np.testing.assert_allclose(
            y, [0.0, 0.0, 1250.0, 2500.0 + (0.30 / 0.32) * 7700.0,
                14300.0 + (0.14 / 0.94) * 12700.0, 27000.0 + 0.2 * slope],
            rtol=1e-12)
        assert np.all(np.diff(y[1:]) > 0)
        s = ic.damage_vec(0.2)
        assert isinstance(s, float) and s == 2500.0


# =========================================================================
# Default subcommands of the scripts run_pipeline.sh calls without arguments
# =========================================================================

def _main_without_import(path, stubs, consts):
    """Return main() of *path*, compiled on its own, with the module's other
    names replaced by *stubs* and *consts*.

    thermosteric_record_sensitivity.py reads data at import time, so the
    function is taken from the source instead of importing the module."""
    with open(path, encoding='utf-8') as f:
        tree = ast.parse(f.read())
    fn = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'main']
    assert len(fn) == 1
    mod = ast.Module(body=fn, type_ignores=[])
    ns = {'argparse': argparse, 'sys': sys, '__doc__': ast.get_docstring(tree)}
    ns.update(consts)
    ns.update(stubs)
    exec(compile(mod, path, 'exec'), ns)
    return ns['main']


class TestThermorecDispatch:
    PATH = os.path.join(ROOT, 'scripts', 'thermosteric_record_sensitivity.py')

    def _run(self, monkeypatch, argv):
        calls = []
        stubs = {f'cmd_{c}': (lambda a, c=c: calls.append((c, a)))
                 for c in ('fit', 'all', 'collect')}
        main = _main_without_import(self.PATH, stubs, {'STRUCTURAL_FRAC': 0.14})
        monkeypatch.setattr(sys, 'argv', ['thermosteric_record_sensitivity.py'] + argv)
        main()
        assert len(calls) == 1
        return calls[0]

    def test_no_arguments_runs_all(self, monkeypatch):
        cmd, a = self._run(monkeypatch, [])
        assert cmd == 'all' and a.cmd == 'all'
        assert (a.n_burn, a.n_prod) == (6000, 30000)

    def test_explicit_subcommands(self, monkeypatch):
        cmd, a = self._run(monkeypatch, ['collect'])
        assert cmd == 'collect'
        cmd, a = self._run(monkeypatch, ['fit', '--record', 'en4'])
        assert cmd == 'fit' and a.record == 'en4' and a.scale == 0.14 + 1

    def test_fit_requires_record(self, monkeypatch):
        with pytest.raises(SystemExit):
            self._run(monkeypatch, ['fit'])


@pytest.fixture(scope='module')
def zostoga():
    spec = importlib.util.spec_from_file_location(
        'cmip6_zostoga_under_test', os.path.join(ROOT, 'scripts', 'cmip6_zostoga.py'))
    mod = importlib.util.module_from_spec(spec)
    saved = list(sys.path)
    try:
        spec.loader.exec_module(mod)     # constants and imports only; no I/O
    finally:
        sys.path[:] = saved
    return mod


class TestZostogaDispatch:

    def _run(self, monkeypatch, zostoga, argv, real_extract=False):
        calls = []
        for c in ('fetch', 'analyze') + (() if real_extract else ('extract_ar6',)):
            monkeypatch.setattr(zostoga, f'cmd_{c}',
                                lambda a, c=c: calls.append((c, a)))
        monkeypatch.setattr(sys, 'argv', ['cmip6_zostoga.py'] + argv)
        zostoga.main()
        return calls

    def test_no_arguments_runs_analyze(self, monkeypatch, zostoga):
        (cmd, a), = self._run(monkeypatch, zostoga, [])
        assert cmd == 'analyze' and a.cmd == 'analyze'
        assert a.out_csv == str(zostoga.DEF_CSV)
        assert a.download_dir == str(zostoga.DEF_DOWNLOAD_DIR)
        assert a.ar6_tlm == str(zostoga.DEF_AR6_TLM)
        assert a.summary_json == str(zostoga.DEF_SUMMARY)
        assert a.figure == str(zostoga.DEF_FIGURE)

    def test_extract_ar6_defaults(self, monkeypatch, zostoga):
        (cmd, a), = self._run(monkeypatch, zostoga, ['extract-ar6'])
        assert cmd == 'extract_ar6'
        assert a.source == str(zostoga.AR6_TLM_FULL)
        assert a.out == str(zostoga.DEF_AR6_TLM)

    def test_fetch_requires_paths(self, monkeypatch, zostoga):
        with pytest.raises(SystemExit):
            self._run(monkeypatch, zostoga, ['fetch'])

    def test_extract_ar6_copies_datasets(self, monkeypatch, zostoga, tmp_path):
        src = tmp_path / 'twolayer_SSPs.h5'
        rng = np.random.default_rng(0)
        data = {'ssp245/surface_temperature': rng.normal(size=(5, 7)).astype(np.float32),
                'ssp245/ECS': rng.normal(size=5),
                'ssp245/TCR': rng.normal(size=5).astype(np.float32)}
        assert set(data) == set(zostoga.AR6_TLM_DATASETS)
        with h5py.File(src, 'w') as f:
            for k, v in data.items():
                f.create_dataset(k, data=v)
            f.create_dataset('ssp585/ECS', data=np.zeros(3))   # not copied
        out = tmp_path / 'sub' / 'extract.h5'
        calls = self._run(monkeypatch, zostoga,
                          ['extract-ar6', '--source', str(src), '--out', str(out)],
                          real_extract=True)
        assert calls == []
        with h5py.File(out, 'r') as g:
            found = []
            g.visit(lambda n: found.append(n) if isinstance(g[n], h5py.Dataset) else None)
            assert sorted(found) == sorted(data)
            for k, v in data.items():
                assert g[k].dtype == v.dtype
                np.testing.assert_array_equal(g[k][:], v)
            assert 'source' in g.attrs and 'note' in g.attrs
