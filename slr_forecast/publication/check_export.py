"""Static checks of an exported publication tree (read-only).

This is the single validator of an export. export_publication.sh calls it on
its staging tree before installing anything; it can also be run by hand on an
export or on a clone of the paper repo.

Usage:
    python -I publication/check_export.py EXPORT_DIR
        [--expected LISTFILE] [--dev-root DEV_PROJECT_DIR]
        [--mode iteration|release] [--allow-unresolved]

Checks (every check marked FAIL makes the exit status 1):
  1. Inventory (with --expected): the files under EXPORT_DIR are exactly the
     paths in LISTFILE (one relative path per line); no other file and no
     directory that holds none of them.                                  FAIL
  2. No symlinks anywhere.                                               FAIL
  3. Denylist (second line of defense behind the exact inventory): no path
     matches EXCLUSION_POLICY below.                                     FAIL
  4. Imports in every .py file and notebook code cell:
       - absolute and relative imports of local modules resolve, on the same
         sys.path the code builds (the file's own directory, notebooks/,
         notebooks/arete_mpl/, scripts/, src/), with __init__.py in every
         parent package;
       - 'from M import name' with M local: name is a submodule of M or is
         bound at the top level of M (including inside if/try blocks). A
         module with 'import *' or a module-level __getattr__ is reported
         as undecidable, not failed;
       - non-local imports are importable by this interpreter (UNRESOLVED
         otherwise; --allow-unresolved downgrades that to a warning);
       - Python syntax errors.                                           FAIL
  5. importlib.util.spec_from_file_location paths: the path argument is
     evaluated from the AST (string literals, __file__, Path(...), .parent,
     .parents[n], .resolve(), '/', os.path.join/dirname/abspath, and
     module-level names bound to such expressions) and must name an exported
     file exactly. An argument that cannot be evaluated is a failure.    FAIL
  6. Every run_pipeline.sh ENTRIES file exists (the 'ENTRIES=(' assignment at
     the start of a line); paths that environment/Dockerfile (COPY/ADD) names
     exist; run_pipeline.sh and environment/entrypoint.sh (required when
     environment/Dockerfile ships) are executable.                       FAIL
  6b. No shipped file mentions the development-only steps (DEV_ONLY_PATTERN:
     VoI as a whole word, any case; evpi; compute_optimal_defense;
     compute_shapley_risk; results_figures_extra). Notebook outputs are not
     searched.                                                           FAIL
  7. Size and file count.
  8. Personal paths: '/Users/', 'Dropbox' and the home directory in every
     text file, PROVENANCE included (notebook source and outputs are counted
     separately). FAIL in --mode release, warning in --mode iteration.
     The bare username in USERNAME_MARKERS (case-sensitive) is reported
     separately, as a warning in both modes, because the name is legitimate
     in LICENSE and in literature citations.
  9. --mode release only: a top-level README.md exists and holds no draft
     marker (a bracketed placeholder ending in 'pending]', or the
     '<!-- DRAFT' comment of publication/templates/README_paper.md).     FAIL

The walk never enters the top-level .git directory and never follows symlinks.
Run with -I from outside the dev repo so the dev tree does not shadow the export.
"""
import argparse
import ast
import fnmatch
import importlib.util
import json
import os
import re
import sys
from pathlib import Path

# Directories put on sys.path by the pipeline code (relative to the root).
SEARCH_DIRS = ['notebooks', 'notebooks/arete_mpl', 'scripts', 'src']

# -----------------------------------------------------------------------------
# Exclusion policy: the one definition of what must never ship. Paths are
# normalized to POSIX form relative to the export root (no leading './').
#   'root_dirs'   first path component equals one of these
#   'any_dirs'    any directory component equals one of these
#   'name_globs'  any component (file or directory name) matches the glob
# -----------------------------------------------------------------------------
EXCLUSION_POLICY = {
    'root_dirs': ['papers', 'memory', 'manuscripts', 'publication', 'referee_reports'],
    'any_dirs': ['archive', 'junk', '__pycache__', '.ipynb_checkpoints', '.claude',
                 '.pytest_cache', '.git', '.vscode', '.idea'],
    'name_globs': [
        'run10k_*', 'save_figures_*', '*.egg-info', '.DS_Store', '._*', 'Icon\r',
        '*.pyc', 'settings.json', 'settings.local.json',
        'prototype_wais_*', 'playing_w_figures.ipynb', 'pipeline_bugs*.md',
        'decisions_group3.md', 'taxonomy.md', 'plan_*.md', '*NOTES*', '*litreview*',
        'wais_buttressing_model.*', 'wais_uq_causal_inference.*',
        'stats_v_process_description.*', 'hierarchical_slr_framework.*',
        'MEMORY.md', 'CLAUDE.md', 'TODO.md',
        # development-only pipeline steps and their runner (dev repo only)
        'compute_evpi.py', 'compute_optimal_defense.py', 'compute_shapley_risk.py',
        'results_figures_extra.ipynb', 'run_extras.sh',
        '*.tex', '*.h5', '*.nc', '*.pdf',
    ],
}

PERSONAL_MARKERS = ['/Users/', 'Dropbox']
# Case-sensitive; never fatal, because the name is legitimate in LICENSE and
# in literature citations.
USERNAME_MARKERS = ['minchew']

# Mentions of the development-only steps (VoI and the steps run_extras.sh runs
# in the dev repo). 'VoI' must be a whole word (not 'avoid'); any case.
DEV_ONLY_PATTERN = re.compile(
    r'(?<![a-z])voi(?![a-z])|evpi|compute_optimal_defense|compute_shapley_risk'
    r'|results_figures_extra', re.IGNORECASE)

# Files that must be executable in the export.
EXECUTABLES = ['run_pipeline.sh', 'environment/entrypoint.sh']

# Release mode: draft markers that must not be in the top-level README.md.
README_DRAFT_PATTERNS = [re.compile(r'\[[^\]\n]*pending\]', re.IGNORECASE),
                         re.compile(re.escape('<!-- DRAFT'))]

# The ENTRIES=( ... ) assignment of run_pipeline.sh (not OTHER_ENTRIES=( or an
# indented use): from 'ENTRIES=(' at the start of a line to the first line
# that starts with ')'.
ENTRIES_BLOCK = re.compile(r'^ENTRIES=\((.*?)^\)', re.MULTILINE | re.DOTALL)


def excluded(rel):
    """Return the rule that excludes the POSIX relative path REL, or None."""
    parts = [p for p in rel.split('/') if p not in ('', '.')]
    if not parts:
        return None
    if parts[0] in EXCLUSION_POLICY['root_dirs']:
        return f'root dir {parts[0]}/'
    for d in parts[:-1]:
        if d in EXCLUSION_POLICY['any_dirs']:
            return f'dir {d}/'
    if parts[-1] in EXCLUSION_POLICY['any_dirs']:
        return f'dir {parts[-1]}/'
    for comp in parts:
        for g in EXCLUSION_POLICY['name_globs']:
            if fnmatch.fnmatchcase(comp, g):
                return f'name {g!r}'
    return None


# -----------------------------------------------------------------------------
# Walking the tree
# -----------------------------------------------------------------------------
def walk(root):
    """Return (files, dirs, links) as sorted POSIX relative paths.
    The top-level .git is pruned; symlinks are never followed."""
    files, dirs, links = [], [], []
    for dirpath, dirnames, filenames in os.walk(root, followlinks=False):
        reldir = os.path.relpath(dirpath, root)
        reldir = '' if reldir == '.' else reldir.replace(os.sep, '/')
        if reldir == '':
            dirnames[:] = [d for d in dirnames if d != '.git']
            filenames = [f for f in filenames if f != '.git']   # worktree .git file
        keep = []
        for d in dirnames:
            rel = f'{reldir}/{d}' if reldir else d
            if os.path.islink(os.path.join(dirpath, d)):
                links.append(rel)
            else:
                dirs.append(rel)
                keep.append(d)
        dirnames[:] = keep
        for f in filenames:
            rel = f'{reldir}/{f}' if reldir else f
            if os.path.islink(os.path.join(dirpath, f)):
                links.append(rel)
            else:
                files.append(rel)
    return sorted(files), sorted(dirs), sorted(links)


# -----------------------------------------------------------------------------
# Notebook and module parsing
# -----------------------------------------------------------------------------
NON_PYTHON_CELL_MAGICS = ('%%bash', '%%sh', '%%html', '%%javascript', '%%js',
                          '%%latex', '%%markdown', '%%svg', '%%writefile', '%%script')


def notebook_parts(path):
    """Return (code_sources, outputs_json) of a notebook: one entry per Python
    code cell. IPython line magics and shell escapes are blanked; cells with a
    non-Python cell magic are skipped."""
    nb = json.loads(path.read_text(encoding='utf-8'))
    src, out = [], []
    for c in nb.get('cells', []):
        if c.get('cell_type') != 'code':
            continue
        s = c.get('source', [])
        s = ''.join(s) if isinstance(s, list) else s
        if s.lstrip().startswith(NON_PYTHON_CELL_MAGICS):
            continue
        s = '\n'.join('' if ln.lstrip().startswith(('%', '!')) else ln for ln in s.splitlines())
        src.append(s)
        out.append(json.dumps(c.get('outputs', [])))
    return src, out


_PARSE_CACHE = {}


def parse_module(path):
    if path not in _PARSE_CACHE:
        try:
            _PARSE_CACHE[path] = ast.parse(path.read_text(encoding='utf-8'), filename=str(path))
        except SyntaxError:
            _PARSE_CACHE[path] = None
    return _PARSE_CACHE[path]


def top_level_names(tree):
    """Names bound at module top level (descending into if/try/with/for/while).
    Returns (names, undecidable) where undecidable is True if the module has a
    star import or a module-level __getattr__."""
    names, undecidable = set(), False

    def bind_target(t):
        if isinstance(t, ast.Name):
            names.add(t.id)
        elif isinstance(t, (ast.Tuple, ast.List)):
            for e in t.elts:
                bind_target(e)
        elif isinstance(t, ast.Starred):
            bind_target(t.value)

    def visit(body):
        nonlocal undecidable
        for n in body:
            if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                names.add(n.name)
                if n.name == '__getattr__':
                    undecidable = True
            elif isinstance(n, ast.Assign):
                for t in n.targets:
                    bind_target(t)
            elif isinstance(n, (ast.AnnAssign, ast.AugAssign)):
                bind_target(n.target)
            elif isinstance(n, ast.Import):
                for a in n.names:
                    names.add(a.asname or a.name.split('.')[0])
            elif isinstance(n, ast.ImportFrom):
                for a in n.names:
                    if a.name == '*':
                        undecidable = True
                    else:
                        names.add(a.asname or a.name)
            elif isinstance(n, (ast.For, ast.AsyncFor)):
                bind_target(n.target)
                visit(n.body); visit(n.orelse)
            elif isinstance(n, (ast.With, ast.AsyncWith)):
                for it in n.items:
                    if it.optional_vars is not None:
                        bind_target(it.optional_vars)
                visit(n.body)
            elif isinstance(n, (ast.If, ast.While)):
                visit(n.body); visit(n.orelse)
            elif isinstance(n, ast.Try) or type(n).__name__ == 'TryStar':
                visit(n.body); visit(n.orelse); visit(n.finalbody)
                for h in n.handlers:
                    if h.name:
                        names.add(h.name)
                    visit(h.body)
        # walrus at top level is rare; ignored
    visit(tree.body)
    return names, undecidable


def find_module(base, parts):
    """Module file for dotted PARTS under directory BASE, or None."""
    b = base.joinpath(*parts) if parts else base
    if parts and b.with_suffix('.py').is_file():
        return b.with_suffix('.py')
    if (b / '__init__.py').is_file():
        return b / '__init__.py'
    return None


def parents_are_packages(base, parts):
    return all((base.joinpath(*parts[:i]) / '__init__.py').is_file() for i in range(1, len(parts)))


def resolve_absolute(mod, origin_dir, root):
    """Local module file for absolute import MOD, or None."""
    parts = mod.split('.')
    for d in [origin_dir] + [root / s for s in SEARCH_DIRS]:
        hit = find_module(d, parts)
        if hit is not None:
            return hit if parents_are_packages(d, parts) else None
    return None


def resolve_relative(level, mod, origin_file, root):
    """Module file for 'from .{mod} import ...' at LEVEL, or None."""
    base = origin_file.parent
    for _ in range(level - 1):
        base = base.parent
    try:
        base.relative_to(root)
    except ValueError:
        return None
    parts = mod.split('.') if mod else []
    return find_module(base, parts)


# -----------------------------------------------------------------------------
# spec_from_file_location path evaluation
# -----------------------------------------------------------------------------
class Unevaluable(Exception):
    pass


def module_bindings(tree):
    """Module-level 'NAME = expr' bindings (last one wins)."""
    env = {}
    for n in tree.body:
        if isinstance(n, ast.Assign) and len(n.targets) == 1 and isinstance(n.targets[0], ast.Name):
            env[n.targets[0].id] = n.value
    return env


def _dotted(node):
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        inner = _dotted(node.value)
        return f'{inner}.{node.attr}' if inner else None
    return None


def eval_path(node, env, file_path, depth=0):
    """Evaluate a path expression to an absolute normalized string."""
    if depth > 30:
        raise Unevaluable('recursion')
    ev = lambda n: eval_path(n, env, file_path, depth + 1)  # noqa: E731
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    if isinstance(node, ast.Name):
        if node.id == '__file__':
            if file_path is None:
                raise Unevaluable('__file__ in a notebook')
            return str(file_path)
        if node.id in env:
            return ev(env[node.id])
        raise Unevaluable(f'name {node.id}')
    if isinstance(node, ast.Attribute) and node.attr == 'parent':
        return os.path.dirname(os.path.normpath(ev(node.value)))
    if isinstance(node, ast.Subscript):
        val = node.value
        if isinstance(val, ast.Attribute) and val.attr == 'parents':
            idx = node.slice
            if isinstance(idx, ast.Constant) and isinstance(idx.value, int):
                p = os.path.normpath(ev(val.value))
                for _ in range(idx.value + 1):
                    p = os.path.dirname(p)
                return p
        raise Unevaluable('subscript')
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Div):
        return os.path.join(ev(node.left), ev(node.right))
    if isinstance(node, ast.Call):
        name = _dotted(node.func)
        args = node.args
        if name in ('Path', 'pathlib.Path', 'PurePath', 'str', 'os.fspath') and len(args) == 1:
            return ev(args[0])
        if name in ('Path', 'pathlib.Path') and len(args) > 1:
            return os.path.join(*[ev(a) for a in args])
        if name == 'os.path.join' and args:
            return os.path.join(*[ev(a) for a in args])
        if name == 'os.path.dirname' and len(args) == 1:
            return os.path.dirname(os.path.normpath(ev(args[0])))
        if name in ('os.path.abspath', 'os.path.realpath', 'os.path.normpath') and len(args) == 1:
            return os.path.normpath(ev(args[0]))
        if isinstance(node.func, ast.Attribute) and node.func.attr in ('resolve', 'absolute') and not args:
            return os.path.normpath(ev(node.func.value))
        if isinstance(node.func, ast.Attribute) and node.func.attr == 'joinpath':
            return os.path.join(ev(node.func.value), *[ev(a) for a in args])
    raise Unevaluable(ast.dump(node)[:60])


def spec_calls(tree):
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            name = _dotted(node.func) or ''
            if name.split('.')[-1] == 'spec_from_file_location':
                loc = node.args[1] if len(node.args) > 1 else None
                for kw in node.keywords:
                    if kw.arg == 'location':
                        loc = kw.value
                yield node, loc


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
def section(title, items, fatal=True):
    """Print a titled list; return the number of fatal problems."""
    uniq = sorted(set(items))
    tag = '' if not uniq else (' FAIL' if fatal else ' (warning)')
    print(f'  {title}: {len(uniq)}{tag}')
    for s in uniq:
        print(f'    {s}')
    return len(uniq) if fatal else 0


def dev_module_names(dev_root):
    names = set()
    if dev_root is None:
        return names
    for dirpath, dirnames, filenames in os.walk(dev_root, followlinks=False):
        dirnames[:] = [d for d in dirnames if d not in ('.git', 'data', '__pycache__')]
        for f in filenames:
            if f.endswith('.py'):
                names.add(f[:-3])
            if f == '__init__.py':
                names.add(os.path.basename(dirpath))
    return names


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('export_dir')
    ap.add_argument('--expected', default=None, help='file listing the exact expected inventory')
    ap.add_argument('--dev-root', default=None, help='dev project dir, to label missing local modules')
    ap.add_argument('--mode', choices=['iteration', 'release'], default='iteration')
    ap.add_argument('--allow-unresolved', action='store_true',
                    help='treat non-local imports this interpreter cannot find as warnings')
    a = ap.parse_args()
    root = Path(a.export_dir).resolve()
    if not root.is_dir():
        print(f'ERROR: not a directory: {root}')
        return 1
    dev_root = Path(a.dev_root).resolve() if a.dev_root else None
    dev_names = dev_module_names(dev_root)
    stdlib = set(sys.stdlib_module_names)
    home = os.path.expanduser('~')
    markers = PERSONAL_MARKERS + ([home] if home not in ('', '/') else [])
    problems = 0

    files, dirs, links = walk(root)
    print(f'Checking export ({a.mode} mode)')

    # ---- 1. inventory ------------------------------------------------------
    print('\n== 1. Inventory')
    if a.expected:
        expected = [ln.strip() for ln in Path(a.expected).read_text().splitlines() if ln.strip()]
        expected = [e[2:] if e.startswith('./') else e for e in expected]
        exp_set, got_set = set(expected), set(files)
        needed_dirs = {'/'.join(e.split('/')[:i]) for e in exp_set for i in range(1, e.count('/') + 1)}
        print(f'  expected {len(exp_set)} files, found {len(got_set)}')
        problems += section('UNEXPECTED FILE', got_set - exp_set)
        problems += section('MISSING FILE', exp_set - got_set)
        problems += section('UNEXPECTED DIRECTORY', set(dirs) - needed_dirs)
    else:
        print('  (no --expected list given; inventory not compared)')

    # ---- 2. symlinks -------------------------------------------------------
    print('\n== 2. Symlinks')
    problems += section('SYMLINK', links)

    # ---- 3. denylist -------------------------------------------------------
    print('\n== 3. Denylist (second line of defense)')
    leaks = []
    for rel in files + dirs + links:
        rule = excluded(rel)
        if rule:
            leaks.append(f'{rel}  [{rule}]')
    problems += section('EXCLUDED PATH', leaks)

    # ---- 4-5. imports and spec files ----------------------------------------
    py_files = [root / f for f in files if f.endswith('.py')]
    nb_files = [root / f for f in files if f.endswith('.ipynb')]
    third_party = {}
    missing_local, missing_member, undecidable, unresolved = [], [], [], []
    spec_bad, syntax = [], []
    for path in py_files + nb_files:
        rel = path.relative_to(root).as_posix()
        if path.suffix == '.ipynb':
            try:
                cells = notebook_parts(path)[0]
            except (ValueError, UnicodeDecodeError) as e:
                syntax.append(f'{rel}: notebook JSON: {e}')
                continue
            units = [(f'{rel} [cell {i}]', c, None) for i, c in enumerate(cells)]
        else:
            units = [(rel, path.read_text(encoding='utf-8'), path)]
        for label, code, file_path in units:
            try:
                tree = ast.parse(code)
            except SyntaxError as e:
                syntax.append(f'{label}: line {e.lineno}: {e.msg}')
                continue
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    targets = [(n.name, None, None) for n in node.names]
                elif isinstance(node, ast.ImportFrom):
                    targets = [(node.module or '', node.level, [n.name for n in node.names])]
                else:
                    continue
                for mod, level, members in targets:
                    if level:   # relative import: always local
                        hit = resolve_relative(level, mod, path, root)
                        shown = '.' * level + mod
                        if hit is None:
                            missing_local.append(f'{label}: {shown}')
                            continue
                    else:
                        shown = mod
                        hit = resolve_absolute(mod, path.parent, root)
                        if hit is None:
                            top = mod.split('.')[0]
                            if resolve_absolute(top, path.parent, root) is not None:
                                missing_local.append(f'{label}: {mod} (local package, submodule missing)')
                            elif top in stdlib:
                                pass
                            elif top in dev_names:
                                missing_local.append(f'{label}: {mod} (exists in the dev project)')
                            elif importlib.util.find_spec(top) is None:
                                unresolved.append(f'{label}: {mod}')
                            else:
                                third_party.setdefault(top, set()).add(rel)
                            continue
                    # local module found: check 'from M import name' members
                    if not members:
                        continue
                    mtree = parse_module(hit)
                    if mtree is None:
                        continue   # its own syntax error is reported when it is checked
                    names, undec = top_level_names(mtree)
                    for m in members:
                        if m == '*' or m in names:
                            continue
                        if hit.name == '__init__.py' and find_module(hit.parent, [m]) is not None:
                            continue
                        if undec:
                            undecidable.append(f'{label}: {shown}.{m}')
                        else:
                            missing_member.append(f'{label}: from {shown} import {m}')
            env = module_bindings(tree)
            for call, loc in spec_calls(tree):
                if loc is None:
                    spec_bad.append(f'{label}: line {call.lineno}: no location argument')
                    continue
                try:
                    target = os.path.normpath(eval_path(loc, env, file_path))
                except Unevaluable as e:
                    spec_bad.append(f'{label}: line {call.lineno}: cannot evaluate path ({e})')
                    continue
                if not os.path.isabs(target):
                    spec_bad.append(f'{label}: line {call.lineno}: relative path {target} (cwd-dependent)')
                    continue
                try:
                    trel = Path(target).relative_to(root).as_posix()
                except ValueError:
                    spec_bad.append(f'{label}: line {call.lineno}: {target} is outside the export')
                    continue
                if trel not in files:
                    spec_bad.append(f'{label}: line {call.lineno}: {trel} not in the export')

    print('\n== 4. Imports')
    print(f'  files checked: {len(py_files)} .py, {len(nb_files)} .ipynb')
    problems += section('SYNTAX ERROR', syntax)
    problems += section('MISSING-LOCAL', missing_local)
    problems += section('MISSING-MEMBER', missing_member)
    problems += section('UNRESOLVED (non-local, not importable here)', unresolved,
                        fatal=not a.allow_unresolved)
    section('UNDECIDABLE MEMBER (star import or __getattr__)', undecidable, fatal=False)
    print('  third-party top-level modules (resolved by this interpreter):')
    for t in sorted(third_party):
        print(f'    {t:14s} {len(third_party[t])} file(s)')
    print('\n== 5. spec_from_file_location paths')
    problems += section('BAD SPEC PATH', spec_bad)

    # ---- 6. run_pipeline.sh ENTRIES and environment/ references -------------
    print('\n== 6. Pipeline entries and environment references')
    missing_entries = []
    runner = root / 'run_pipeline.sh'
    if runner.is_file():
        text = runner.read_text()
        m = ENTRIES_BLOCK.search(text)
        if m is None:
            missing_entries.append('run_pipeline.sh has no ENTRIES=( block')
        else:
            block = m.group(1)
            for name, f in re.findall(r'"([^":]+):([^":]+):\d+"', block):
                rel = f if f.endswith('.py') else f'notebooks/{f}'
                if rel not in files:
                    missing_entries.append(f'{name}: {rel}')
    else:
        missing_entries.append('run_pipeline.sh itself')
    problems += section('MISSING ENTRY', missing_entries)

    missing_env = []
    dockerfile = root / 'environment' / 'Dockerfile'
    if dockerfile.is_file():
        for line in dockerfile.read_text().splitlines():
            m = re.match(r'\s*(?:COPY|ADD)\s+(.*)', line)
            if not m:
                continue
            args = [x for x in m.group(1).split() if not x.startswith('--')]
            for src in args[:-1]:   # last argument is the destination
                if not (root / src).exists():
                    missing_env.append(f'Dockerfile COPY {src}')
    problems += section('MISSING ENVIRONMENT REFERENCE', missing_env)

    not_exec = []
    for rel in EXECUTABLES:
        p = root / rel
        if not p.is_file():
            if rel != 'run_pipeline.sh' and dockerfile.is_file():
                not_exec.append(f'{rel} (missing; environment/Dockerfile runs it)')
            continue   # a missing run_pipeline.sh is reported above
        if not os.access(p, os.X_OK):
            not_exec.append(f'{rel} (mode {p.stat().st_mode & 0o777:o})')
    problems += section('NOT EXECUTABLE', not_exec)

    # ---- 6b. mentions of the development-only steps ---------------------------
    print('\n== 6b. Development-only steps (VoI, evpi, compute_optimal_defense, '
          'compute_shapley_risk, results_figures_extra)')
    dev_hits = []
    for f in files:
        raw = (root / f).read_bytes()
        if b'\0' in raw[:8192]:
            continue
        try:
            text = raw.decode('utf-8')
        except UnicodeDecodeError:
            continue
        if f.endswith('.ipynb'):
            try:   # every cell's source; outputs are not searched
                nb = json.loads(text)
                text = '\n'.join(''.join(c.get('source', [])) if isinstance(c.get('source'), list)
                                 else c.get('source', '') for c in nb.get('cells', []))
            except ValueError:
                pass
        found = sorted({m.group(0) for m in DEV_ONLY_PATTERN.finditer(text)})
        if found:
            dev_hits.append(f'{f}: ' + ', '.join(found))
    problems += section('MENTIONS DEVELOPMENT-ONLY STEPS', dev_hits)

    # ---- 7. size -------------------------------------------------------------
    sizes = {f: (root / f).stat().st_size for f in files}
    total = sum(sizes.values())
    print(f'\n== 7. Size: {len(files)} files, {total / 1e6:.2f} MB')
    by_top = {}
    for f, s in sizes.items():
        top = f.split('/')[0]
        by_top[top] = by_top.get(top, 0) + s
    for k, v in sorted(by_top.items(), key=lambda kv: -kv[1]):
        print(f'    {k:28s} {v / 1e6:8.2f} MB')

    # ---- 8. personal paths ---------------------------------------------------
    hits, name_hits = [], []
    for f in files:
        p = root / f
        raw = p.read_bytes()
        if b'\0' in raw[:8192]:
            continue   # binary (e.g. a wheel)
        try:
            text = raw.decode('utf-8')
        except UnicodeDecodeError:
            continue
        name_counts = {m: text.count(m) for m in USERNAME_MARKERS}
        if any(name_counts.values()):
            name_hits.append(f'{f}: ' + ', '.join(f"'{m}' x{n}" for m, n in name_counts.items() if n))
        counts = {m: text.count(m) for m in markers}
        if not any(counts.values()):
            continue
        desc = ', '.join(f"'{'~' if m == home else m}' x{n}" for m, n in counts.items() if n)
        if p.suffix == '.ipynb':
            # Where in the notebook: count the markers in code source and outputs.
            try:
                src, out = notebook_parts(p)
                ns = sum(s.count(m) for s in src for m in PERSONAL_MARKERS)
                no = sum(o.count(m) for o in out for m in PERSONAL_MARKERS)
                desc += f"  ['/Users/'+'Dropbox' in code source: {ns}, in outputs: {no}]"
            except ValueError:
                pass
        hits.append(f'{f}: {desc}')
    print('\n== 8. Personal paths (' + ', '.join(f"'{m}'" for m in PERSONAL_MARKERS) + ', home dir)')
    problems += section('FILES WITH PERSONAL PATHS', hits, fatal=(a.mode == 'release'))
    print('\n== 8b. Username (' + ', '.join(f"'{m}'" for m in USERNAME_MARKERS)
          + '; warning only)')
    section('FILES WITH THE USERNAME', name_hits, fatal=False)

    # ---- 9. README (release mode) ------------------------------------------
    if a.mode == 'release':
        print('\n== 9. Top-level README.md (release mode)')
        readme_bad = []
        readme = root / 'README.md'
        if not readme.is_file():
            readme_bad.append('README.md is missing (allowlist: publication/templates/README_paper.md -> README.md)')
        else:
            rtext = readme.read_text(encoding='utf-8', errors='replace')
            for pat in README_DRAFT_PATTERNS:
                for m in pat.finditer(rtext):
                    line = rtext.count('\n', 0, m.start()) + 1
                    readme_bad.append(f'README.md line {line}: draft marker {m.group(0)!r}')
        problems += section('README NOT RELEASE-READY', readme_bad)

    print(f'\n{"FAIL" if problems else "OK"}: {problems} problem(s)')
    return 1 if problems else 0


if __name__ == '__main__':
    sys.exit(main())
