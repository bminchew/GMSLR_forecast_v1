"""Release-gating tests for the publication export.

Run from anywhere; with PYTHONDONTWRITEBYTECODE=1 and -p no:cacheprovider
nothing is written under the project:

    PYTHONDONTWRITEBYTECODE=1 /Users/minchew/opt/anaconda3/envs/python2025/bin/python \
        -m pytest -p no:cacheprovider publication/test_export.py

Two groups:

* Real tree: export the dev project (working tree) to a tmp dir and check the
  exact inventory, symlinks, denylist, import closure, run_pipeline.sh ENTRIES
  and that the shipped .gitignore keeps every shipped file. These tests fail
  with the exporter's own message while the dev tree is not exportable (for
  example untracked files in an allowlisted directory). Set
  TEST_EXPORT_ALLOWLIST=/path/to/allowlist to run them with an iteration-only
  override allowlist.
  Refusals against the real repo (alias spelling, ancestors, root) are also
  checked; they are rejected before anything is written.

* Fixture repo: a throwaway git repo in tmp (git root with a slr_forecast/
  subdirectory, a copy of the release tooling, a tiny allowlist) exercises
  --from-commit, --require-clean, --force into a fake clone, the refusal
  cases, and the checker's own detections.
"""
import hashlib
import importlib.util
import json
import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent          # P/publication
P = HERE.parent                                 # P (slr_forecast/)
EXPORTER = HERE / 'export_publication.sh'
CHECKER = HERE / 'check_export.py'
GITIGNORE_TEMPLATE = HERE / 'templates' / 'gitignore'

# Must match GENERATED_FILES in export_publication.sh.
GENERATED = [
    '.gitignore', 'PROVENANCE', 'data/data_manifest.md', 'data/raw/README.md',
    'data/processed/README.md', 'figures/.gitkeep', 'figures/diagnostics/.gitkeep',
    'notebooks/figures/.gitkeep',
]
BASH = '/bin/bash'


def _native_prefix():
    """Under Rosetta (x86_64 Python on Apple silicon), child processes inherit
    x86_64 and /usr/bin/git's xcrun shim fails; run git and bash as arm64."""
    if sys.platform != 'darwin' or platform.machine() != 'x86_64':
        return []
    r = subprocess.run(['sysctl', '-n', 'sysctl.proc_translated'], capture_output=True, text=True)
    return ['/usr/bin/arch', '-arm64'] if r.stdout.strip() == '1' else []


NATIVE = _native_prefix()


sys.dont_write_bytecode = True     # no __pycache__ next to check_export.py


def _load_checker():
    spec = importlib.util.spec_from_file_location('check_export', CHECKER)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


checker = _load_checker()


def run_export(args, exporter=EXPORTER, env_extra=None):
    env = dict(os.environ)
    env['PYTHON'] = sys.executable
    env.pop('EXPORT_ALLOWLIST', None)
    env.update(env_extra or {})
    return subprocess.run(NATIVE + [BASH, str(exporter)] + [str(a) for a in args],
                          capture_output=True, text=True, env=env)


def git(cwd, *args, check=True):
    r = subprocess.run(NATIVE + ['git', '-C', str(cwd)] + list(args), capture_output=True, text=True,
                       env=dict(os.environ, GIT_LITERAL_PATHSPECS='1'))
    if check and r.returncode != 0:
        raise AssertionError(f'git {args} failed: {r.stderr}')
    return r.stdout


def parse_allowlist(text):
    """(src, dst) pairs, parsed independently of the exporter."""
    out = []
    for line in text.splitlines():
        line = line.split('#', 1)[0].strip()
        if not line:
            continue
        if ' -> ' in line:
            src, dst = (s.strip().rstrip('/') for s in line.split(' -> ', 1))
        else:
            src = dst = line.rstrip('/')
        out.append((src, dst))
    return out


def expected_inventory(proj, allowlist_text):
    """Expanded allowlist (working tree) plus generated files."""
    inv = set()
    for src, dst in parse_allowlist(allowlist_text):
        if (proj / src).is_dir():
            inv.update(git(proj, 'ls-files', '--', src).splitlines())
        else:
            inv.add(dst)
    return inv | set(GENERATED)


def tree_files(root):
    files, _, links = checker.walk(root)
    assert not links, f'symlinks: {links}'
    return set(files)


def snapshot(root):
    """Content hash of every path under ROOT (including .git), for 'untouched' checks."""
    snap = {}
    for dirpath, dirnames, filenames in os.walk(root):
        for n in dirnames + filenames:
            p = Path(dirpath) / n
            rel = p.relative_to(root).as_posix()
            if p.is_symlink():
                snap[rel] = 'link:' + os.readlink(p)
            elif p.is_file():
                snap[rel] = hashlib.sha256(p.read_bytes()).hexdigest()
            else:
                snap[rel] = 'dir'
    return snap


def fail_msg(r):
    return f'exit {r.returncode}\n--- stdout ---\n{r.stdout[-4000:]}\n--- stderr ---\n{r.stderr[-4000:]}'


# =============================================================================
# Real tree
# =============================================================================
@pytest.fixture(scope='module')
def real_export(tmp_path_factory):
    out = tmp_path_factory.mktemp('real') / 'export'
    override = os.environ.get('TEST_EXPORT_ALLOWLIST')
    env = {'EXPORT_ALLOWLIST': override} if override else {}
    r = run_export([out], env_extra=env)
    if r.returncode != 0:
        pytest.fail('export of the dev working tree failed:\n' + fail_msg(r), pytrace=False)
    allow = Path(override).read_text() if override else (HERE / 'allowlist.txt').read_text()
    return out, allow


def test_real_inventory_exact(real_export):
    out, allow = real_export
    got = tree_files(out)
    want = expected_inventory(P, allow)
    assert got - want == set(), 'unexpected files'
    assert want - got == set(), 'missing files'


def test_real_no_symlinks(real_export):
    out, _ = real_export
    _, _, links = checker.walk(out)
    assert links == []


def test_real_no_excluded_patterns(real_export):
    out, _ = real_export
    files, dirs, _ = checker.walk(out)
    bad = [p for p in files + dirs if checker.excluded(p)]
    assert bad == []


def test_real_import_closure(real_export):
    out, _ = real_export
    r = subprocess.run([sys.executable, '-I', str(CHECKER), str(out), '--dev-root', str(P)],
                       capture_output=True, text=True, cwd=str(out.parent))
    assert r.returncode == 0, r.stdout[-6000:]
    for title in ('MISSING-LOCAL: 0', 'MISSING-MEMBER: 0', 'SYNTAX ERROR: 0', 'BAD SPEC PATH: 0'):
        assert title in r.stdout


def test_real_pipeline_entries_present(real_export):
    out, _ = real_export
    text = (P / 'run_pipeline.sh').read_text()
    m = checker.ENTRIES_BLOCK.search(text)
    assert m, 'no ENTRIES=( assignment in run_pipeline.sh'
    block = m.group(1)
    entries = [ln.strip().strip('"') for ln in block.splitlines() if ln.strip().startswith('"')]
    assert entries, 'no ENTRIES parsed from run_pipeline.sh'
    missing = []
    for e in entries:
        name, f, _timeout = e.split(':')
        rel = f if f.endswith('.py') else f'notebooks/{f}'
        if not (out / rel).is_file():
            missing.append(f'{name}: {rel}')
    assert missing == []


def test_real_gitignore_keeps_every_shipped_file(real_export, tmp_path):
    """git add -A in a copy must stage the whole inventory: nothing shipped is
    hidden by the shipped .gitignore."""
    out, _ = real_export
    clone = tmp_path / 'clone'
    shutil.copytree(out, clone, symlinks=True)
    git(clone, 'init', '-q')
    git(clone, 'add', '-A')
    staged = set(git(clone, 'ls-files').splitlines())
    assert staged == tree_files(clone)


def test_real_provenance_is_private(real_export):
    out, _ = real_export
    prov = (out / 'PROVENANCE').read_text()
    assert 'source_commit:' in prov and 'allowlist_sha256:' in prov and 'exporter:' in prov
    for word in ('/Users/', 'Dropbox', os.path.expanduser('~'), 'remote', 'branch', 'github.com'):
        assert word not in prov, word


def _real_alias(p):
    """P spelled through ~/Documents when it physically lives in ~/Dropbox/Documents."""
    home = os.path.expanduser('~')
    pre = os.path.join(home, 'Dropbox', 'Documents') + os.sep
    if str(p).startswith(pre):
        alias = Path(home, 'Documents', str(p)[len(pre):])
        if alias.exists() and os.path.samefile(alias, p):
            return alias
    return None


@pytest.mark.parametrize('case', ['physical', 'alias', 'git_root', 'home', 'root', 'dotdot'])
def test_real_refusals(case):
    """Containment refusals against the real repo. Never with --force; the
    probe paths would be created under publication/ only if the guard failed."""
    git_root = Path(git(P, 'rev-parse', '--show-toplevel').strip())
    probe = None
    if case == 'physical':
        target = probe = HERE / '_probe_target_physical'
        expect = 'inside the dev repo'
    elif case == 'alias':
        alias = _real_alias(P)
        if alias is None:
            pytest.skip('no ~/Documents alias of the dev repo on this machine')
        target = alias / 'publication' / '_probe_target_alias'
        probe = HERE / '_probe_target_alias'
        expect = 'inside the dev repo'
    elif case == 'git_root':
        target, expect = git_root, 'dev repo'
    elif case == 'home':
        target, expect = Path(os.path.expanduser('~')), 'contains the dev repo'
    elif case == 'root':
        target, expect = Path('/'), 'filesystem root'
    else:
        target, expect = f'{HERE}/../_probe', "'..'"
    r = run_export([target])
    assert r.returncode != 0
    assert expect in r.stderr, r.stderr
    if probe is not None:
        assert not probe.exists()


# =============================================================================
# Fixture repo
# =============================================================================
FIXTURE_ALLOWLIST = """\
# fixture allowlist
run_pipeline.sh
scripts/entry.py
notebooks/helper.py
src/pkg                       # directory entry
docs/readme_paper.md -> README.md
"""

FIXTURE_FILES = {
    'run_pipeline.sh': '#!/usr/bin/env bash\nENTRIES=(\n    "entry:scripts/entry.py:0"\n)\n',
    'scripts/entry.py': (
        'import sys\nfrom pathlib import Path\n'
        "ROOT = Path(__file__).resolve().parents[1]\n"
        "sys.path.insert(0, str(ROOT / 'notebooks'))\n"
        "sys.path.insert(0, str(ROOT / 'src'))\n"
        'from helper import f\nfrom pkg import mod\nfrom pkg.mod import g\n'
        'import importlib.util\n'
        "_s = importlib.util.spec_from_file_location('h', ROOT / 'notebooks' / 'helper.py')\n"
    ),
    'notebooks/helper.py': 'def f():\n    return 1\n',
    'src/pkg/__init__.py': 'from .mod import g\n',
    'src/pkg/mod.py': 'def g():\n    return 2\n',
    'docs/readme_paper.md': '# Paper repo\n',
    'archive/old.py': 'x = 1\n',          # dev-only: must never ship
    'notes_NOTES.md': 'private\n',        # dev-only: must never ship
}


@pytest.fixture
def dev(tmp_path):
    """A committed fixture dev repo. Returns (git_root, proj, exporter)."""
    git_root = tmp_path / 'dev'
    proj = git_root / 'slr_forecast'
    pub = proj / 'publication'
    (pub / 'templates').mkdir(parents=True)
    shutil.copy2(EXPORTER, pub / 'export_publication.sh')
    shutil.copy2(CHECKER, pub / 'check_export.py')
    shutil.copy2(GITIGNORE_TEMPLATE, pub / 'templates' / 'gitignore')
    (pub / 'allowlist.txt').write_text(FIXTURE_ALLOWLIST)
    (pub / 'data_manifest.md').write_text('# Data manifest (fixture)\n')
    for rel, text in FIXTURE_FILES.items():
        (proj / rel).parent.mkdir(parents=True, exist_ok=True)
        (proj / rel).write_text(text)
    os.chmod(proj / 'run_pipeline.sh', 0o755)      # the checker requires it executable
    (git_root / '.gitignore').write_text('*.log\n')
    git(git_root, 'init', '-q')
    git(git_root, 'add', '-A')
    git(git_root, '-c', 'user.name=fixture', '-c', 'user.email=fixture@example.invalid',
        'commit', '-q', '-m', 'fixture')
    return git_root, proj, pub / 'export_publication.sh'


def fixture_inventory():
    return {'run_pipeline.sh', 'scripts/entry.py', 'notebooks/helper.py', 'src/pkg/__init__.py',
            'src/pkg/mod.py', 'README.md'} | set(GENERATED)


def commit_all(git_root, msg='change'):
    git(git_root, 'add', '-A')
    git(git_root, '-c', 'user.name=fixture', '-c', 'user.email=fixture@example.invalid',
        'commit', '-q', '-m', msg)


def test_fixture_export_ok(dev, tmp_path):
    _, proj, exporter = dev
    out = tmp_path / 'out'
    r = run_export([out], exporter)
    assert r.returncode == 0, fail_msg(r)
    assert tree_files(out) == fixture_inventory()
    assert (out / 'README.md').read_text() == '# Paper repo\n'
    prov = (out / 'PROVENANCE').read_text()
    assert 'export_mode:          working-tree' in prov
    assert 'shipped_dirty:        no' in prov
    assert 'allowlist:            publication/allowlist.txt' in prov
    assert str(tmp_path) not in prov


def test_fixture_require_clean(dev, tmp_path):
    git_root, proj, exporter = dev
    r = run_export(['--require-clean', tmp_path / 'ok'], exporter)
    assert r.returncode == 0, fail_msg(r)
    # a modified shipped file
    (proj / 'notebooks/helper.py').write_text('def f():\n    return 3\n')
    r = run_export(['--require-clean', tmp_path / 'a'], exporter)
    assert r.returncode != 0 and 'notebooks/helper.py' in r.stderr
    assert not (tmp_path / 'a').exists()
    git(git_root, 'checkout', '-q', '--', '.')
    # modified release tooling
    with open(proj / 'publication/allowlist.txt', 'a') as fh:
        fh.write('# edit\n')
    r = run_export(['--require-clean', tmp_path / 'b'], exporter)
    assert r.returncode != 0 and 'publication/allowlist.txt' in r.stderr
    git(git_root, 'checkout', '-q', '--', '.')
    # an ignored file named as a file entry
    (proj / 'notebooks/extra.log').write_text('x\n')
    with open(proj / 'publication/allowlist.txt', 'a') as fh:
        fh.write('notebooks/extra.log\n')
    commit_all(git_root)          # commits the allowlist; extra.log stays ignored
    r = run_export(['--require-clean', tmp_path / 'c'], exporter)
    assert r.returncode != 0 and 'notebooks/extra.log' in r.stderr


def test_fixture_from_commit(dev, tmp_path):
    git_root, proj, exporter = dev
    head = git(git_root, 'rev-parse', 'HEAD').strip()
    # working-tree edits that must NOT reach a --from-commit export
    (proj / 'notebooks/helper.py').write_text('def f():\n    return 99\n')
    (proj / 'src/pkg/new.py').write_text('y = 2\n')                  # untracked in a dir entry
    with open(proj / 'publication/allowlist.txt', 'a') as fh:
        fh.write('archive/old.py\n')                                 # uncommitted allowlist edit
    out = tmp_path / 'out'
    r = run_export(['--from-commit', 'HEAD', out], exporter)
    assert r.returncode == 0, fail_msg(r)
    assert tree_files(out) == fixture_inventory()
    assert (out / 'notebooks/helper.py').read_text() == 'def f():\n    return 1\n'
    assert os.access(out / 'run_pipeline.sh', os.X_OK) is True    # committed as 100755
    assert os.access(out / 'scripts/entry.py', os.X_OK) is False  # committed as 100644
    prov = (out / 'PROVENANCE').read_text()
    assert f'source_commit:        {head}' in prov
    assert 'export_mode:          from-commit' in prov
    # a running exporter that differs from the commit is refused
    with open(exporter, 'a') as fh:
        fh.write('\n# local edit\n')
    r = run_export(['--from-commit', 'HEAD', tmp_path / 'out2'], exporter)
    assert r.returncode != 0 and 'differs from its copy' in r.stderr
    # EXPORT_ALLOWLIST is refused in release mode
    alt = tmp_path / 'alt.txt'
    alt.write_text('run_pipeline.sh\n')
    r = run_export(['--require-clean', tmp_path / 'out3'], exporter, {'EXPORT_ALLOWLIST': str(alt)})
    assert r.returncode != 0 and 'iteration only' in r.stderr


@pytest.mark.parametrize('args', [['--from-commit', ''], ['--from-commit=']])
def test_fixture_from_commit_empty_value(dev, tmp_path, args):
    """An empty revision must not fall back to a working-tree export."""
    _, _, exporter = dev
    out = tmp_path / 'out'
    r = run_export(args + [out], exporter)
    assert r.returncode != 0 and 'needs a revision' in r.stderr
    assert not out.exists()


def test_fixture_release_readme(dev, tmp_path):
    """Release mode requires a top-level README.md without draft markers."""
    git_root, proj, exporter = dev
    (proj / 'docs/readme_paper.md').write_text(
        '# Paper repo\n<!-- DRAFT for the paper repository -->\nCite [DOI pending].\n')
    commit_all(git_root)
    r = run_export(['--from-commit', 'HEAD', tmp_path / 'rel'], exporter)
    assert r.returncode != 0 and 'README NOT RELEASE-READY: 2 FAIL' in r.stderr, fail_msg(r)
    assert not (tmp_path / 'rel').exists()
    r = run_export([tmp_path / 'iter'], exporter)                # iteration: not checked
    assert r.returncode == 0, fail_msg(r)
    # no README at all
    allow = proj / 'publication/allowlist.txt'
    allow.write_text(allow.read_text().replace('docs/readme_paper.md -> README.md\n', ''))
    commit_all(git_root)
    for args in (['--from-commit', 'HEAD'], ['--require-clean']):
        out = tmp_path / ('rel_' + args[0].strip('-'))
        r = run_export(args + [out], exporter)
        assert r.returncode != 0 and 'README.md is missing' in r.stderr, fail_msg(r)
        assert not out.exists()


def test_fixture_runner_not_executable(dev, tmp_path):
    """run_pipeline.sh committed without the executable bit is refused in both modes."""
    git_root, proj, exporter = dev
    os.chmod(proj / 'run_pipeline.sh', 0o644)
    commit_all(git_root)
    assert '100644' in git(git_root, 'ls-files', '-s', 'slr_forecast/run_pipeline.sh')
    for args, out in ((['--from-commit', 'HEAD'], tmp_path / 'rel'), ([], tmp_path / 'iter')):
        r = run_export(args + [out], exporter)
        assert r.returncode != 0 and 'NOT EXECUTABLE: 1 FAIL' in r.stderr, fail_msg(r)
        assert 'run_pipeline.sh (mode 644)' in r.stderr
        assert not out.exists()


def test_fixture_untracked_in_directory_entry(dev, tmp_path):
    _, proj, exporter = dev
    (proj / 'src/pkg/ignored.log').write_text('ignored\n')   # ignored: silently not shipped
    r = run_export([tmp_path / 'ok'], exporter)
    assert r.returncode == 0, fail_msg(r)
    assert 'src/pkg/ignored.log' not in tree_files(tmp_path / 'ok')
    (proj / 'src/pkg/new.py').write_text('y = 2\n')           # untracked, not ignored: error
    r = run_export([tmp_path / 'bad'], exporter)
    assert r.returncode != 0
    assert 'UNTRACKED: src/pkg/new.py' in r.stderr
    assert not (tmp_path / 'bad').exists()


@pytest.mark.parametrize('kind', ['file', 'ancestor', 'inside_dir'])
def test_fixture_symlinked_source(dev, tmp_path, kind):
    """Symlinks are created in the tmp fixture, never in P."""
    _, proj, exporter = dev
    outside = tmp_path / 'outside'
    outside.mkdir()
    (outside / 'evil.py').write_text('secret = 1\n')
    allow = proj / 'publication/allowlist.txt'
    if kind == 'file':
        os.symlink(outside / 'evil.py', proj / 'notebooks/linked.py')
        allow.write_text(allow.read_text() + 'notebooks/linked.py\n')
    elif kind == 'ancestor':
        os.symlink(outside, proj / 'linkdir')
        allow.write_text(allow.read_text() + 'linkdir/evil.py\n')
    else:
        os.symlink(outside, proj / 'src/pkg/linked_dir')
    r = run_export([tmp_path / 'out'], exporter)
    assert r.returncode != 0, fail_msg(r)
    assert 'SYMLINK' in r.stderr
    assert not (tmp_path / 'out').exists()


@pytest.mark.parametrize('case', ['inside_project', 'inside_git_root', 'ancestor',
                                  'ancestor_force', 'dotdot', 'root', 'root_dot'])
def test_fixture_containment(dev, tmp_path, case):
    git_root, proj, exporter = dev
    before = snapshot(git_root)
    args = []
    if case == 'inside_project':
        target, expect = proj / 'out', 'inside the dev repo'
    elif case == 'inside_git_root':
        target, expect = git_root / 'nested' / 'out', 'inside the dev repo'
    elif case in ('ancestor', 'ancestor_force'):
        target, expect = tmp_path, 'contains the dev repo'
        if case == 'ancestor_force':
            args = ['--force']
            (tmp_path / 'PROVENANCE').write_text('x\n')   # would pass the --force marker test
    elif case == 'dotdot':
        target, expect = f'{tmp_path}/x/../out', "'..'"
    elif case == 'root':
        target, expect = '/', 'filesystem root'
    else:
        target, expect = '/.//', 'filesystem root'
    r = run_export(args + [target], exporter)
    assert r.returncode != 0
    assert expect in r.stderr, r.stderr
    assert snapshot(git_root) == before


def test_fixture_nonempty_target(dev, tmp_path):
    _, _, exporter = dev
    target = tmp_path / 'busy'
    target.mkdir()
    (target / 'keep.txt').write_text('mine\n')
    before = snapshot(target)
    r = run_export([target], exporter)
    assert r.returncode != 0 and 'not empty' in r.stderr
    # --force into a directory that is neither a git repo nor an export
    r = run_export(['--force', target], exporter)
    assert r.returncode != 0 and 'neither an empty git repository' in r.stderr
    # a PROVENANCE not written by this exporter does not qualify
    (target / 'PROVENANCE').write_text('Some other provenance\nsource_commit:        x\n')
    before = snapshot(target)
    r = run_export(['--force', target], exporter)
    assert r.returncode != 0 and 'neither an empty git repository' in r.stderr
    assert snapshot(target) == before


def test_fixture_force_refuses_unrelated_repo(dev, tmp_path):
    """--force must not replace the contents of an unrelated git repo."""
    _, _, exporter = dev
    repo = tmp_path / 'thesis'
    repo.mkdir()
    git(repo, 'init', '-q')
    (repo / 'thesis.tex').write_text('\\documentclass{article}\n')
    git(repo, 'add', '-A')
    git(repo, '-c', 'user.name=t', '-c', 'user.email=t@example.invalid', 'commit', '-q', '-m', 'v0')
    before = snapshot(repo)
    r = run_export(['--force', repo], exporter)
    assert r.returncode != 0 and 'neither an empty git repository' in r.stderr
    assert snapshot(repo) == before
    # a committed PROVENANCE that this exporter did not write does not qualify either
    (repo / 'PROVENANCE').write_text('made by hand\nsource_commit:        x\n')
    git(repo, 'add', '-A')
    git(repo, '-c', 'user.name=t', '-c', 'user.email=t@example.invalid', 'commit', '-q', '-m', 'v1')
    before = snapshot(repo)
    r = run_export(['--force', repo], exporter)
    assert r.returncode != 0 and 'neither an empty git repository' in r.stderr
    assert snapshot(repo) == before


def test_fixture_force_into_fake_clone(dev, tmp_path):
    _, _, exporter = dev
    clone = tmp_path / 'clone'
    clone.mkdir()
    git(clone, 'init', '-q')
    r = run_export(['--force', clone], exporter)                # no commits yet: accepted
    assert r.returncode == 0, fail_msg(r)
    (clone / 'stale.py').write_text('old\n')            # tracked, no longer shipped
    (clone / 'README.md').write_text('old readme\n')
    git(clone, 'add', '-A')
    git(clone, '-c', 'user.name=t', '-c', 'user.email=t@example.invalid', 'commit', '-q', '-m', 'v0')
    head = git(clone, 'rev-parse', 'HEAD').strip()

    # An untracked file in the clone would be deleted: refused, clone untouched.
    (clone / 'local_data.csv').write_text('downloaded\n')
    before = snapshot(clone)
    r = run_export(['--force', clone], exporter)
    assert r.returncode != 0 and 'local_data.csv' in r.stderr
    assert snapshot(clone) == before
    (clone / 'local_data.csv').unlink()

    r = run_export(['--force', clone], exporter)
    assert r.returncode == 0, fail_msg(r)
    assert git(clone, 'rev-parse', 'HEAD').strip() == head      # .git preserved
    assert not (clone / 'stale.py').exists()                     # stale file deleted
    assert (clone / 'README.md').read_text() == '# Paper repo\n'
    assert tree_files(clone) == fixture_inventory()
    # the clone now has uncommitted changes to tracked files: refused, untouched
    before = snapshot(clone)
    r = run_export(['--force', clone], exporter)
    assert r.returncode != 0 and 'modified tracked files' in r.stderr
    assert snapshot(clone) == before
    # once committed, re-export into the same clone is a no-op apart from PROVENANCE
    git(clone, 'add', '-A')
    git(clone, '-c', 'user.name=t', '-c', 'user.email=t@example.invalid', 'commit', '-q', '-m', 'v1')
    r = run_export(['--force', clone], exporter)
    assert r.returncode == 0, fail_msg(r)


def test_fixture_force_reexport_uncommitted_clone(dev, tmp_path):
    """Iteration loop: export into a fresh 'git init' dir, do not commit, drop an
    allowlist entry, re-export with --force: the dropped file disappears."""
    _, proj, exporter = dev
    clone = tmp_path / 'clone'
    clone.mkdir()
    git(clone, 'init', '-q')
    assert run_export(['--force', clone], exporter).returncode == 0
    assert (clone / 'notebooks/helper.py').exists()
    allow = proj / 'publication/allowlist.txt'
    allow.write_text(allow.read_text().replace('notebooks/helper.py\n', ''))
    (proj / 'scripts/entry.py').write_text('x = 1\n')     # no longer imports helper
    r = run_export(['--force', clone], exporter)
    assert r.returncode == 0, fail_msg(r)
    assert not (clone / 'notebooks/helper.py').exists()
    assert (clone / '.git').is_dir()
    assert tree_files(clone) == fixture_inventory() - {'notebooks/helper.py'}


NOTEBOOK_WITH_PATH = json.dumps({
    'cells': [{'cell_type': 'code', 'execution_count': 1, 'metadata': {},
               'outputs': [{'name': 'stdout', 'output_type': 'stream',
                            'text': ['/Users/someone/data/file.h5\n']}],
               'source': ['print(1)\n']}],
    'metadata': {}, 'nbformat': 4, 'nbformat_minor': 5}, indent=1) + '\n'


def test_fixture_release_mode_personal_paths(dev, tmp_path):
    git_root, proj, exporter = dev
    (proj / 'notebooks/nb.ipynb').write_text(NOTEBOOK_WITH_PATH)
    allow = proj / 'publication/allowlist.txt'
    allow.write_text(allow.read_text() + 'notebooks/nb.ipynb\n')
    commit_all(git_root)
    # iteration: warning only
    r = run_export([tmp_path / 'iter'], exporter)
    assert r.returncode == 0, fail_msg(r)
    assert 'FILES WITH PERSONAL PATHS: 1 (warning)' in r.stdout
    # release: fails, target not created
    r = run_export(['--require-clean', tmp_path / 'rel'], exporter)
    assert r.returncode != 0 and 'FILES WITH PERSONAL PATHS: 1 FAIL' in r.stderr
    assert not (tmp_path / 'rel').exists()
    r = run_export(['--from-commit', 'HEAD', tmp_path / 'rel2'], exporter)
    assert r.returncode != 0 and not (tmp_path / 'rel2').exists()
    # release with cleared outputs: passes, outputs gone in the export only
    r = run_export(['--from-commit', 'HEAD', '--clear-outputs', tmp_path / 'rel3'], exporter)
    assert r.returncode == 0, fail_msg(r)
    nb = json.loads((tmp_path / 'rel3/notebooks/nb.ipynb').read_text())
    assert nb['cells'][0]['outputs'] == [] and nb['cells'][0]['execution_count'] is None
    assert '/Users/' in (proj / 'notebooks/nb.ipynb').read_text()   # dev copy untouched


def test_fixture_clear_outputs_without_notebooks(dev, tmp_path):
    _, _, exporter = dev
    r = run_export(['--clear-outputs', tmp_path / 'out'], exporter)
    assert r.returncode == 0, fail_msg(r)
    assert tree_files(tmp_path / 'out') == fixture_inventory()


def test_fixture_failure_leaves_target_untouched(dev, tmp_path):
    git_root, proj, exporter = dev
    first = tmp_path / 'exp'
    assert run_export([first], exporter).returncode == 0
    before = snapshot(first)
    allow = proj / 'publication/allowlist.txt'
    allow.write_text(allow.read_text() + 'scripts/does_not_exist.py\n')
    r = run_export(['--force', first], exporter)
    assert r.returncode != 0 and 'MISSING: scripts/does_not_exist.py' in r.stderr
    assert snapshot(first) == before
    # a checker failure (missing local import) also leaves it untouched
    allow.write_text(FIXTURE_ALLOWLIST)
    (proj / 'scripts/entry.py').write_text('import not_shipped_module\n')
    (proj / 'notebooks/not_shipped_module.py').write_text('z = 0\n')
    r = run_export(['--force', first], exporter)
    assert r.returncode != 0 and 'not_shipped_module' in r.stderr
    assert snapshot(first) == before


# =============================================================================
# Checker unit checks
# =============================================================================
@pytest.mark.parametrize('path', ['archive/x.py', 'junk/y.py', 'notebooks/archive/z.py',
                                  'papers/a.md', 'manuscripts/v1/m.txt', 'notes/plan_v2.md',
                                  'figures/f.pdf', 'src/pkg/__pycache__/m.cpython-311.pyc',
                                  'publication/allowlist.txt', '.claude/settings.json',
                                  'notebooks/._helper.py'])
def test_checker_excludes(path):
    assert checker.excluded(path), path


@pytest.mark.parametrize('path', ['notebooks/helper.py', 'data/processed/README.md',
                                  'environment/Dockerfile', 'scripts/archive_reader.py'])
def test_checker_allows(path):
    assert checker.excluded(path) is None, path


def _write(root, files):
    for rel, text in files.items():
        (root / rel).parent.mkdir(parents=True, exist_ok=True)
        (root / rel).write_text(text)
        if rel.endswith('.sh'):
            os.chmod(root / rel, 0o755)


def _check(root, *extra):
    return subprocess.run([sys.executable, '-I', str(CHECKER), str(root)] + list(extra),
                          capture_output=True, text=True, cwd=str(root.parent))


def test_checker_detections(tmp_path):
    root = tmp_path / 'exp'
    _write(root, {
        'run_pipeline.sh': 'ENTRIES=(\n "a:scripts/a.py:0"\n)\n',
        'scripts/a.py': (
            'from pathlib import Path\nimport importlib.util\n'
            "ROOT = Path(__file__).resolve().parents[1]\n"
            "s1 = importlib.util.spec_from_file_location('m', ROOT / 'scripts' / 'b.py')\n"
            "s2 = importlib.util.spec_from_file_location('m', ROOT / 'notebooks' / 'helper.py')\n"
            "s3 = importlib.util.spec_from_file_location('m', make_path())\n"
            'from pkg import missing_name\nfrom pkg.sub import here\nfrom .sibling import q\n'
        ),
        'notebooks/helper.py': 'x = 1\n',
        'other/helper.py': 'x = 1\n',          # same basename as a missing target
        'src/pkg/__init__.py': 'present = 1\n',
        'src/pkg/sub.py': 'here = 1\n',
        'scripts/broken.py': 'def f(:\n',
    })
    r = _check(root)
    assert r.returncode == 1
    out = r.stdout
    assert 'scripts/b.py not in the export' in out            # exact path, not basename
    assert 'cannot evaluate path' in out
    assert 'from pkg import missing_name' in out
    assert 'scripts/a.py: .sibling' in out                     # relative import
    assert 'scripts/broken.py' in out and 'SYNTAX ERROR: 1 FAIL' in out
    assert 'notebooks/helper.py not in the export' not in out


def test_checker_inventory_and_release_paths(tmp_path):
    root = tmp_path / 'exp'
    _write(root, {'run_pipeline.sh': 'ENTRIES=(\n)\n', 'a.txt': 'path /Users/someone/x\n',
                  'junk/leak.py': 'x = 1\n', 'PROVENANCE': 'made in Dropbox\n'})
    exp = tmp_path / 'expected.txt'
    exp.write_text('run_pipeline.sh\na.txt\nPROVENANCE\nmissing.txt\n')
    r = _check(root, '--expected', str(exp))
    out = r.stdout
    assert r.returncode == 1
    assert 'UNEXPECTED FILE: 1 FAIL' in out and 'junk/leak.py' in out
    assert 'MISSING FILE: 1 FAIL' in out and 'missing.txt' in out
    assert 'junk/leak.py  [root dir' in out or "junk/leak.py  [dir junk/]" in out
    assert 'FILES WITH PERSONAL PATHS: 2 (warning)' in out          # iteration: warning
    r = _check(root, '--mode', 'release')
    assert 'FILES WITH PERSONAL PATHS: 2 FAIL' in r.stdout and 'PROVENANCE' in r.stdout


def _section(out, start, end):
    """Text of the checker output from the line starting with START to END."""
    return out.split(start, 1)[1].split(end, 1)[0]


def test_checker_entries_executables_and_dev_only(tmp_path):
    root = tmp_path / 'exp'
    nb = json.dumps({
        'cells': [{'cell_type': 'markdown', 'metadata': {}, 'source': ['See voi_table.\n']},
                  {'cell_type': 'code', 'execution_count': 1, 'metadata': {},
                   'outputs': [{'name': 'stdout', 'output_type': 'stream',
                                'text': ['compute_shapley_risk\n']}],
                   'source': ['x = 1\n']}],
        'metadata': {}, 'nbformat': 4, 'nbformat_minor': 5})
    _write(root, {
        # OTHER_ENTRIES=( before ENTRIES=( must not be taken for the ENTRIES block
        'run_pipeline.sh': 'OTHER_ENTRIES=(\n "x:missing_extra.py:0"\n)\nENTRIES=(\n "a:scripts/a.py:0"\n)\n',
        'scripts/a.py': '# avoid the void\nx = 1\n',
        'scripts/b.py': 'x = 2  # see compute_evpi\n',
        'notes.md': 'The VoI analysis.\n',
        'notebooks/n.ipynb': nb,
        'environment/Dockerfile': 'FROM scratch\n',
        'environment/entrypoint.sh': '#!/usr/bin/env bash\n',
    })
    os.chmod(root / 'environment/entrypoint.sh', 0o644)
    r = _check(root)
    out = r.stdout
    assert r.returncode == 1
    assert 'MISSING ENTRY: 0' in out
    assert 'NOT EXECUTABLE: 1 FAIL' in out and 'environment/entrypoint.sh (mode 644)' in out
    dev = _section(out, 'MENTIONS DEVELOPMENT-ONLY STEPS', '== 7.')
    assert dev.startswith(': 3 FAIL')
    assert 'notes.md: VoI' in dev and 'scripts/b.py: evpi' in dev
    assert 'notebooks/n.ipynb: voi' in dev                     # cell source, not the output
    assert 'scripts/a.py' not in dev and 'compute_shapley_risk' not in dev
    # without its Dockerfile, a missing entrypoint.sh is not required
    (root / 'environment/entrypoint.sh').unlink()
    (root / 'environment/Dockerfile').unlink()
    r = _check(root)
    assert 'NOT EXECUTABLE: 0' in r.stdout
