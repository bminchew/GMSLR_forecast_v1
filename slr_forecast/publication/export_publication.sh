#!/usr/bin/env bash
#
# Export the curated publication repository from the development repository.
#
# Usage:
#   publication/export_publication.sh [options] TARGET_DIR
#
#   TARGET_DIR          Directory to install the export into. It must lie
#                       outside the dev git repo, must not contain it, and must
#                       not be '/'. A relative path is taken from the current
#                       directory; '..' components are refused.
#
#   --from-commit REV   Release path. Export the allowlisted files as they are
#                       in commit REV (git cat-file), not from the working tree.
#                       The allowlist, data manifest and templates are also
#                       read from REV, and the running exporter and checker
#                       must be identical to their copies in REV.
#   --require-clean     Release from the working tree at HEAD: every shipped
#                       file and every release-tooling file must be tracked,
#                       unmodified and not ignored.
#   --force             Allow a non-empty TARGET_DIR. It must be a git
#                       repository with no commits, or an earlier export (has
#                       a PROVENANCE written by this exporter); a git
#                       repository must have no modified tracked files. Its
#                       contents are replaced by the export with
#                       'rsync -a --delete --exclude /.git', so files no longer
#                       allowlisted are deleted and .git is preserved.
#   --clear-outputs     Clear code-cell outputs and execution counts in the
#                       exported notebooks (the dev copies are never touched).
#
#   Environment:
#     PYTHON            Interpreter for check_export.py and --clear-outputs
#                       (default: python3). Use the project environment, so
#                       third-party imports resolve.
#     EXPORT_ALLOWLIST  Iteration only: use this allowlist file instead of
#                       publication/allowlist.txt. Refused in release mode
#                       (--from-commit, --require-clean); recorded in PROVENANCE.
#
# Without --from-commit or --require-clean the export is an iteration export
# of the working tree: untracked or modified shipped files are allowed and
# listed in PROVENANCE, and home-directory paths only produce warnings.
#
# Steps:
#   1. Check the target (containment, emptiness) before anything is written.
#   2. Read the allowlist and expand every entry to files. A directory entry
#      expands to the files git tracks under it (git ls-files, or git ls-tree
#      for --from-commit); untracked, non-ignored files in it are an error.
#      Symlinks are refused everywhere.
#   3. Copy the files into a fresh staging directory, add the generated files
#      (placeholders, data/data_manifest.md, .gitignore, PROVENANCE), strip
#      extended attributes, optionally clear notebook outputs.
#   4. Validate the staging tree with check_export.py: exact inventory, no
#      symlinks, denylist, import closure, pipeline ENTRIES, executable bits,
#      development-only mentions, home paths, release README.
#   5. Only then install the staging tree into TARGET_DIR. On any failure the
#      target is left as it was and the staging directory is removed, except
#      when rsync itself fails partway through a --force install: the target
#      can then be partly updated (see publication/README.md).
#
# The export is one way: dev repo -> TARGET_DIR -> paper repo. Nothing is
# written to the dev repo. Portable to macOS /bin/bash 3.2 and Linux (no
# mapfile, no associative arrays, no sed -i, no GNU-only flags).

set -euo pipefail
umask 022
# git status must never write (refresh) the index of the dev repo or the target.
export GIT_OPTIONAL_LOCKS=0

die()  { echo "ERROR: $*" >&2; exit 1; }
warn() { echo "WARNING: $*" >&2; }
note() { echo "  $*"; }

# -----------------------------------------------------------------------------
# Locations
# -----------------------------------------------------------------------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
PROJ="$(cd "${SCRIPT_DIR}/.." && pwd -P)"                 # slr_forecast/ (P)
PYTHON="${PYTHON:-python3}"

# Release-tooling files, relative to P. In --require-clean mode they must be
# clean; in --from-commit mode they are read from (or compared with) REV.
T_EXPORTER="publication/export_publication.sh"
T_CHECKER="publication/check_export.py"
T_ALLOWLIST="publication/allowlist.txt"
T_MANIFEST="publication/data_manifest.md"
T_GITIGNORE="publication/templates/gitignore"

# First line of the PROVENANCE file this exporter writes.
PROVENANCE_HEADER="SPECTRUM publication export"

# Files the exporter generates in the export (in addition to the allowlist).
GENERATED_FILES=".gitignore
PROVENANCE
data/data_manifest.md
data/raw/README.md
data/processed/README.md
figures/.gitkeep
figures/diagnostics/.gitkeep
notebooks/figures/.gitkeep"

# -----------------------------------------------------------------------------
# Arguments
# -----------------------------------------------------------------------------
FORCE=0
CLEAR_OUTPUTS=0
REQUIRE_CLEAN=0
FROM_COMMIT=""
TARGET=""
while [ $# -gt 0 ]; do
    case "$1" in
        --force)          FORCE=1 ;;
        --clear-outputs)  CLEAR_OUTPUTS=1 ;;
        --require-clean)  REQUIRE_CLEAN=1 ;;
        --from-commit)    [ $# -ge 2 ] && [ -n "$2" ] || die "--from-commit needs a revision"
                          FROM_COMMIT="$2"; shift ;;
        --from-commit=*)  FROM_COMMIT="${1#--from-commit=}"
                          [ -n "$FROM_COMMIT" ] || die "--from-commit needs a revision" ;;
        -h|--help)        sed -n '2,/^set -euo/p' "$0" | sed '$d'; exit 0 ;;
        -*)               die "unknown option: $1" ;;
        *)                [ -z "$TARGET" ] || die "more than one TARGET_DIR given"
                          TARGET="$1" ;;
    esac
    shift
done
[ -n "$TARGET" ] || die "no TARGET_DIR given (see --help)"

RELEASE=0
MODE="working-tree"
if [ -n "$FROM_COMMIT" ]; then
    RELEASE=1; MODE="from-commit"
    [ "$REQUIRE_CLEAN" -eq 0 ] || die "--from-commit and --require-clean are exclusive (--from-commit is already clean)"
elif [ "$REQUIRE_CLEAN" -eq 1 ]; then
    RELEASE=1; MODE="require-clean"
fi
if [ -n "${EXPORT_ALLOWLIST:-}" ] && [ "$RELEASE" -eq 1 ]; then
    die "EXPORT_ALLOWLIST is for iteration only; unset it for a release export"
fi

command -v "$PYTHON" > /dev/null 2>&1 || die "python not found: $PYTHON (set PYTHON)"
command -v rsync > /dev/null 2>&1 || die "rsync not found"
if command -v shasum > /dev/null 2>&1; then SHA_CMD="shasum -a 256"; else SHA_CMD="sha256sum"; fi
sha256_of() { $SHA_CMD "$1" | cut -d' ' -f1; }

GIT_ROOT="$(git -C "$PROJ" rev-parse --show-toplevel 2>/dev/null)" \
    || die "$PROJ is not inside a git repository"
GIT_ROOT="$(cd "$GIT_ROOT" && pwd -P)"
case "$PROJ" in
    "$GIT_ROOT")   SUBDIR="" ;;
    "$GIT_ROOT"/*) SUBDIR="${PROJ#"$GIT_ROOT"/}" ;;
    *)             die "project $PROJ is not under its git root $GIT_ROOT" ;;
esac
# All git pathspecs below are literal (no glob magic in allowlist paths).
export GIT_LITERAL_PATHSPECS=1

# =============================================================================
# 1. Target containment (before anything is written anywhere)
# =============================================================================
# Identity tests use '[ a -ef b ]' (same device and inode), which is
# component-aware and independent of symlink spellings and of case-insensitive
# file systems. String forms are compared component-wise as a second check.

# True if the existing directory $1 is, or lies inside, the existing directory $2.
is_within() {
    local a="$1"
    while :; do
        [ "$a" -ef "$2" ] && return 0
        [ "$a" = "/" ] && return 1
        a="$(dirname "$a")"
    done
}

# True if path string $1 equals or lies inside path string $2 (component-wise).
str_within() {
    case "$1/" in "$2/"*) return 0 ;; esac
    return 1
}

# Lexical normalization of an absolute path: drop '.' and empty components.
# '..' never reaches here (refused beforehand).
lexical_norm() {   # always called in $(...), so 'set -f' stays local
    local out="" comp
    local IFS=/
    set -f            # no globbing of path components
    for comp in $1; do
        case "$comp" in ""|.) ;; *) out="$out/$comp" ;; esac
    done
    [ -n "$out" ] || out="/"
    echo "$out"
}

# Set TGT_LOGICAL, TGT_PHYS and TGT_EXISTS for the target, or die.
canonicalize_target() {
    local raw="$1" abs p rest="" comp
    case "/$raw/" in */../*) die "TARGET_DIR must not contain '..' components: $raw" ;; esac
    case "$raw" in /*) abs="$raw" ;; *) abs="$(pwd -L)/$raw" ;; esac
    TGT_LOGICAL="$(lexical_norm "$abs")"
    [ "$TGT_LOGICAL" != "/" ] || die "TARGET_DIR is the filesystem root"

    # Walk up to the nearest existing ancestor. Every non-existent component
    # must be a plain name; a dangling symlink there is refused.
    p="$TGT_LOGICAL"
    while [ ! -e "$p" ]; do
        [ ! -L "$p" ] || die "TARGET_DIR path has a dangling symlink: $p"
        rest="/$(basename "$p")${rest}"
        p="$(dirname "$p")"
    done
    if [ -n "$rest" ]; then
        [ -d "$p" ] || die "TARGET_DIR parent exists and is not a directory: $p"
        TGT_EXISTS=0
    else
        [ -d "$p" ] || die "TARGET_DIR exists and is not a directory: $p"
        TGT_EXISTS=1
    fi
    TGT_ANCHOR="$(cd "$p" && pwd -P)"       # physical, existing
    TGT_PHYS="$(lexical_norm "${TGT_ANCHOR}${rest}")"
    [ "$TGT_PHYS" != "/" ] || die "TARGET_DIR resolves to the filesystem root"
}

canonicalize_target "$TARGET"

# Spellings of the dev repo: physical, as invoked, and the ~/Documents alias.
DEV_SPELLINGS="$GIT_ROOT"
DEV_LOGICAL="$(lexical_norm "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -L)")"   # P as invoked
if [ -n "$SUBDIR" ]; then
    case "$DEV_LOGICAL" in */"$SUBDIR") DEV_LOGICAL="${DEV_LOGICAL%/"$SUBDIR"}" ;; esac
fi
if [ -d "$DEV_LOGICAL" ] && [ "$DEV_LOGICAL" -ef "$GIT_ROOT" ]; then
    DEV_SPELLINGS="$DEV_SPELLINGS
$DEV_LOGICAL"
fi
case "$GIT_ROOT" in
    "$HOME"/Dropbox/Documents/*)
        alias_root="$HOME/Documents/${GIT_ROOT#"$HOME"/Dropbox/Documents/}"
        if [ -d "$alias_root" ] && [ "$alias_root" -ef "$GIT_ROOT" ]; then
            DEV_SPELLINGS="$DEV_SPELLINGS
$alias_root"
        fi ;;
esac

# (a) Physical identity: the target's nearest existing ancestor must not be,
#     or lie inside, the dev repo; the dev repo must not lie inside the target.
if is_within "$TGT_ANCHOR" "$GIT_ROOT"; then
    die "TARGET_DIR $TGT_PHYS is inside the dev repo $GIT_ROOT"
fi
if [ "$TGT_EXISTS" -eq 1 ] && is_within "$GIT_ROOT" "$TGT_ANCHOR"; then
    die "TARGET_DIR $TGT_PHYS contains the dev repo $GIT_ROOT"
fi
# (b) String forms, physical and logical, against every dev-repo spelling.
#     This also catches a not-yet-existing target that names the dev repo.
while IFS= read -r dev; do
    [ -n "$dev" ] || continue
    for t in "$TGT_LOGICAL" "$TGT_PHYS"; do
        str_within "$t" "$dev" && die "TARGET_DIR $t is inside the dev repo ($dev)"
        str_within "$dev" "$t" && die "TARGET_DIR $t contains the dev repo ($dev)"
    done
done <<EOF
$DEV_SPELLINGS
EOF

# Emptiness and --force policy.
TGT_EMPTY=1
if [ "$TGT_EXISTS" -eq 1 ] && [ -n "$(ls -A "$TGT_PHYS")" ]; then
    TGT_EMPTY=0
    [ "$FORCE" -eq 1 ] || die "TARGET_DIR is not empty: $TGT_PHYS (use --force to replace its contents)"
    # A git repository at the target must be its own top level, and its
    # tracked files must be unmodified (the install would overwrite them).
    TGT_IS_GIT=0
    if [ -e "$TGT_PHYS/.git" ]; then
        tgt_top="$(git -C "$TGT_PHYS" rev-parse --show-toplevel 2> /dev/null)" \
            || die "--force refuses $TGT_PHYS: it has a .git that git does not accept"
        [ "$(cd "$tgt_top" && pwd -P)" = "$TGT_PHYS" ] \
            || die "--force refuses $TGT_PHYS: its .git is not the top level of a repository"
        TGT_IS_GIT=1
        tgt_status="$(git -C "$TGT_PHYS" status --porcelain --untracked-files=no)" \
            || die "git status failed in $TGT_PHYS"
        if [ -n "$tgt_status" ]; then
            printf '%s\n' "$tgt_status" >&2
            die "--force refuses $TGT_PHYS: it has modified tracked files (listed above); commit or restore them first"
        fi
    fi
    # Accepted: a git repository with no commits, or an earlier export.
    TGT_OK=0
    if [ "$TGT_IS_GIT" -eq 1 ] && ! git -C "$TGT_PHYS" rev-parse --verify --quiet HEAD > /dev/null; then
        TGT_OK=1
    fi
    if [ -f "$TGT_PHYS/PROVENANCE" ] && [ ! -L "$TGT_PHYS/PROVENANCE" ] \
       && [ "$(sed -n '1p' "$TGT_PHYS/PROVENANCE")" = "$PROVENANCE_HEADER" ] \
       && grep -q '^source_commit:  ' "$TGT_PHYS/PROVENANCE"; then
        TGT_OK=1
    fi
    [ "$TGT_OK" -eq 1 ] || die "--force refuses $TGT_PHYS: it is neither an empty git repository (no commits) nor an earlier export (a PROVENANCE written by this exporter)"
    # Never write through a symlink in the target (rsync would replace or follow it).
    TGT_LINKS="$(find "$TGT_PHYS" -path "$TGT_PHYS/.git" -prune -o -type l -print)"
    [ -z "$TGT_LINKS" ] || { echo "$TGT_LINKS" >&2; die "TARGET_DIR contains symlinks (listed above)"; }
fi

# =============================================================================
# 2. Work area (removed on exit; the target is only touched in step 5)
# =============================================================================
WORK="$(mktemp -d "${TMPDIR:-/tmp}/spectrum-export.XXXXXX")"
WORK="$(cd "$WORK" && pwd -P)"
INSTALL_TMP=""
cleanup() {
    rm -rf "$WORK"
    [ -z "$INSTALL_TMP" ] || rm -rf "$INSTALL_TMP"
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
is_within "$WORK" "$GIT_ROOT" && die "TMPDIR lies inside the dev repo; set TMPDIR elsewhere"
STAGE="$WORK/stage"
mkdir "$STAGE"

# =============================================================================
# 3. Source of the tooling inputs and of the shipped files
# =============================================================================
TREE=""            # git tree-ish for P in --from-commit mode
if [ -n "$FROM_COMMIT" ]; then
    COMMIT="$(git -C "$GIT_ROOT" rev-parse --verify --quiet "${FROM_COMMIT}^{commit}")" \
        || die "not a commit: $FROM_COMMIT"
    if [ -n "$SUBDIR" ]; then TREE="${COMMIT}:${SUBDIR}"; else TREE="${COMMIT}"; fi
    # blob_ref REL: object name of the file REL (relative to P) in the commit.
    blob_ref() { echo "${COMMIT}:${SUBDIR:+$SUBDIR/}$1"; }
    # Tooling inputs come from the commit.
    for f in "$T_ALLOWLIST" "$T_MANIFEST" "$T_GITIGNORE" "$T_EXPORTER" "$T_CHECKER"; do
        git -C "$GIT_ROOT" cat-file -e "$(blob_ref "$f")" 2> /dev/null \
            || die "$f is not in commit ${COMMIT:0:12}; commit the release tooling first"
    done
    # The running exporter and checker must be the committed ones.
    for f in "$T_EXPORTER" "$T_CHECKER"; do
        want="$(git -C "$GIT_ROOT" rev-parse "$(blob_ref "$f")")"
        have="$(git hash-object "$PROJ/$f")"
        [ "$want" = "$have" ] || die "$f differs from its copy in ${COMMIT:0:12}; run the committed version"
    done
    # cat-file blob: the raw blob, never a textconv or filter view.
    git -C "$GIT_ROOT" cat-file blob "$(blob_ref "$T_ALLOWLIST")" > "$WORK/allowlist.txt"
    git -C "$GIT_ROOT" cat-file blob "$(blob_ref "$T_MANIFEST")"  > "$WORK/data_manifest.md"
    git -C "$GIT_ROOT" cat-file blob "$(blob_ref "$T_GITIGNORE")" > "$WORK/gitignore"
    ALLOWLIST_LABEL="$T_ALLOWLIST (from commit)"
else
    COMMIT="$(git -C "$PROJ" rev-parse --verify HEAD)"
    if [ -n "${EXPORT_ALLOWLIST:-}" ]; then
        [ -f "$EXPORT_ALLOWLIST" ] || die "EXPORT_ALLOWLIST not found: $EXPORT_ALLOWLIST"
        cp "$EXPORT_ALLOWLIST" "$WORK/allowlist.txt"
        ALLOWLIST_LABEL="override via EXPORT_ALLOWLIST (iteration only; not a project file)"
    else
        cp "$PROJ/$T_ALLOWLIST" "$WORK/allowlist.txt"
        ALLOWLIST_LABEL="$T_ALLOWLIST"
    fi
    cp "$PROJ/$T_MANIFEST"  "$WORK/data_manifest.md"
    cp "$PROJ/$T_GITIGNORE" "$WORK/gitignore"
fi

# -----------------------------------------------------------------------------
# Parse the allowlist: one entry per line, '#' starts a comment.
#   PATH            ship PATH (file or directory) at the same relative path
#   SRC -> DEST     ship the file SRC as DEST (files only; e.g. a template
#                   installed as the top-level README.md)
# Output: $WORK/entries with lines "SRC<TAB>DEST".
# -----------------------------------------------------------------------------
check_rel() {   # check_rel PATH LABEL: relative, inside P, no '.'/'..', plain names
    case "$1" in
        ""|/*) die "allowlist $2 must be a relative path: '$1'" ;;
    esac
    case "/$1/" in
        */../*|*/./*|*//*) die "allowlist $2 must not contain '.', '..' or empty components: $1" ;;
        */.git/*) die "allowlist $2 must not name or pass through .git: $1" ;;
    esac
    case "$1" in
        *'"'*|*'\'*|*"	"*) die "allowlist $2 contains an unusual character: $1" ;;
    esac
}
TAB="$(printf '\t')"
: > "$WORK/entries"
while IFS= read -r line || [ -n "$line" ]; do
    line="${line%%#*}"
    # trim leading/trailing whitespace
    line="$(printf '%s' "$line" | sed -e 's/^[[:space:]]*//' -e 's/[[:space:]]*$//')"
    [ -n "$line" ] || continue
    case "$line" in
        *" -> "*)
            src="$(printf '%s' "${line%% -> *}" | sed -e 's/[[:space:]]*$//')"
            dst="$(printf '%s' "${line#* -> }" | sed -e 's/^[[:space:]]*//')" ;;
        *)  src="$line"; dst="$line" ;;
    esac
    src="${src%/}"; dst="${dst%/}"
    check_rel "$src" "path"; check_rel "$dst" "destination"
    printf '%s\t%s\n' "$src" "$dst" >> "$WORK/entries"
done < "$WORK/allowlist.txt"
N_ENTRIES="$(wc -l < "$WORK/entries" | tr -d ' ')"
[ "$N_ENTRIES" -gt 0 ] || die "allowlist has no active entries"

echo "Exporting $N_ENTRIES allowlist entries ($MODE)"
echo "  from: $PROJ${FROM_COMMIT:+ @ ${COMMIT:0:12}}"
echo "  to:   $TGT_PHYS"

# -----------------------------------------------------------------------------
# Expand entries to files. Output: $WORK/files with lines
# "SRC<TAB>DEST<TAB>MODE"; every SRC is a regular file relative to P, and MODE
# is the git file mode (from-commit) or '-' (working tree).
# -----------------------------------------------------------------------------
: > "$WORK/files"
ERRORS=0
err() { echo "  $*" >&2; ERRORS=$((ERRORS + 1)); }

# Refuse a working-tree path if it or any ancestor up to P is a symlink, or if
# it does not resolve to a location inside P.
check_no_symlink() {   # check_no_symlink REL
    local rel="$1" prefix="" comp rest="$1"
    while [ -n "$rest" ]; do
        comp="${rest%%/*}"
        case "$rest" in */*) rest="${rest#*/}" ;; *) rest="" ;; esac
        prefix="${prefix:+$prefix/}$comp"
        if [ -L "$PROJ/$prefix" ]; then
            err "SYMLINK: $prefix (allowlisted sources must not be or pass through symlinks)"
            return 1
        fi
    done
    local dir
    if [ -d "$PROJ/$rel" ]; then dir="$PROJ/$rel"; else dir="$(dirname "$PROJ/$rel")"; fi
    if ! is_within "$(cd "$dir" && pwd -P)" "$PROJ"; then
        err "OUTSIDE: $rel resolves outside $PROJ"
        return 1
    fi
    return 0
}

expand_worktree() {   # expand_worktree SRC DEST
    local src="$1" dst="$2" f mode
    if [ ! -e "$PROJ/$src" ] && [ ! -L "$PROJ/$src" ]; then
        err "MISSING: $src"; return 0
    fi
    check_no_symlink "$src" || return 0
    if [ -d "$PROJ/$src" ]; then
        [ "$src" = "$dst" ] || { err "RENAME: '$src -> $dst' is only allowed for files"; return; }
        # Symlinks anywhere inside (tracked or not) are refused.
        local links
        links="$(find "$PROJ/$src" -type l -print)"
        if [ -n "$links" ]; then
            printf '%s\n' "$links" | while IFS= read -r l; do echo "  SYMLINK: ${l#"$PROJ"/}" >&2; done
            err "SYMLINK: $src contains symlinks (listed above)"; return
        fi
        # Untracked, non-ignored files would silently not ship: refuse.
        local untracked
        untracked="$(git -C "$PROJ" ls-files --others --exclude-standard -- "$src")"
        if [ -n "$untracked" ]; then
            printf '%s\n' "$untracked" | while IFS= read -r u; do echo "  UNTRACKED: $u" >&2; done
            err "UNTRACKED: directory entry $src has untracked files (listed above); commit, ignore or remove them"
        fi
        # Tracked files only; modes 120000 (symlink) and 160000 (submodule) refused.
        git -C "$PROJ" ls-files -s -- "$src" > "$WORK/ls.tmp"
        [ -s "$WORK/ls.tmp" ] || { err "EMPTY: directory entry $src has no tracked files"; return; }
        while IFS= read -r line; do
            mode="${line%% *}"; f="${line#*"$TAB"}"
            case "$f" in '"'*) err "UNUSUAL NAME: $f"; continue ;; esac
            case "$mode" in 100644|100755) ;; *) err "MODE $mode: $f (symlink or submodule)"; continue ;; esac
            if [ ! -f "$PROJ/$f" ]; then err "DELETED: $f is tracked but missing in the working tree"; continue; fi
            printf '%s\t%s\t-\n' "$f" "$f" >> "$WORK/files"
        done < "$WORK/ls.tmp"
    elif [ -f "$PROJ/$src" ]; then
        printf '%s\t%s\t-\n' "$src" "$dst" >> "$WORK/files"
    else
        err "NOT A REGULAR FILE: $src"
    fi
}

expand_commit() {   # expand_commit SRC DEST
    local src="$1" dst="$2" line meta mode type f n=0
    git -C "$GIT_ROOT" ls-tree -r "$TREE" -- "$src" > "$WORK/ls.tmp"
    [ -s "$WORK/ls.tmp" ] || { err "MISSING: $src (not in commit ${COMMIT:0:12})"; return; }
    while IFS= read -r line; do
        meta="${line%%"$TAB"*}"; f="${line#*"$TAB"}"
        mode="${meta%% *}"; type="$(echo "$meta" | cut -d' ' -f2)"
        case "$f" in '"'*) err "UNUSUAL NAME: $f"; continue ;; esac
        if [ "$f" = "$src" ]; then
            :                                         # file entry
        else
            case "$f" in "$src"/*) ;; *) continue ;; esac
            [ "$src" = "$dst" ] || { err "RENAME: '$src -> $dst' is only allowed for files"; return; }
        fi
        case "$mode $type" in
            "100644 blob"|"100755 blob") ;;
            *) err "MODE $mode $type: $f (symlink or submodule in commit)"; continue ;;
        esac
        if [ "$f" = "$src" ]; then
            printf '%s\t%s\t%s\n' "$f" "$dst" "$mode" >> "$WORK/files"
        else
            printf '%s\t%s\t%s\n' "$f" "$f" "$mode" >> "$WORK/files"
        fi
        n=$((n + 1))
    done < "$WORK/ls.tmp"
    [ "$n" -gt 0 ] || err "MISSING: $src (not in commit ${COMMIT:0:12})"
}

while IFS="$TAB" read -r src dst; do
    if [ -n "$FROM_COMMIT" ]; then expand_commit "$src" "$dst"; else expand_worktree "$src" "$dst"; fi
done < "$WORK/entries"
[ "$ERRORS" -eq 0 ] || die "$ERRORS problem(s) expanding the allowlist (see above)"

# Duplicate destinations (a file named twice, or by a file and a directory entry).
LC_ALL=C sort -u "$WORK/files" > "$WORK/files.sorted"
cut -f2 "$WORK/files.sorted" | LC_ALL=C sort | uniq -d > "$WORK/dups"
[ ! -s "$WORK/dups" ] || { cat "$WORK/dups" >&2; die "destination path(s) shipped twice (listed above)"; }
# A generated file must not collide with a shipped one.
cut -f2 "$WORK/files.sorted" > "$WORK/shipped.dest"
printf '%s\n' "$GENERATED_FILES" > "$WORK/generated"
COLLIDE="$(LC_ALL=C sort "$WORK/shipped.dest" "$WORK/generated" | uniq -d)"
[ -z "$COLLIDE" ] || die "allowlisted path(s) collide with generated files: $COLLIDE"
N_SHIPPED="$(wc -l < "$WORK/files.sorted" | tr -d ' ')"

# -----------------------------------------------------------------------------
# git status of the shipped files and of the release tooling (working tree)
# -----------------------------------------------------------------------------
# Porcelain paths are relative to the git root; strip SUBDIR so that only
# P-relative paths of shipped files are recorded.
status_of() {   # status_of LISTFILE -> porcelain lines, P-relative
    tr '\n' '\0' < "$1" \
      | (cd "$PROJ" && xargs -0 git status --porcelain --ignored --untracked-files=all --) \
      | while IFS= read -r line; do
            if [ -n "$SUBDIR" ]; then
                printf '%s\n' "$line" | sed -e "s#\([ >]\)${SUBDIR}/#\1#g"
            else
                printf '%s\n' "$line"
            fi
        done
}
SHIPPED_STATUS=""
if [ -z "$FROM_COMMIT" ]; then
    cut -f1 "$WORK/files.sorted" > "$WORK/shipped.src"
    SHIPPED_STATUS="$(status_of "$WORK/shipped.src")"
    if [ "$REQUIRE_CLEAN" -eq 1 ]; then
        printf '%s\n' "$T_EXPORTER" "$T_CHECKER" "$T_ALLOWLIST" "$T_MANIFEST" "$T_GITIGNORE" > "$WORK/tooling"
        TOOL_STATUS="$(status_of "$WORK/tooling")"
        if [ -n "$SHIPPED_STATUS" ] || [ -n "$TOOL_STATUS" ]; then
            [ -z "$SHIPPED_STATUS" ] || { echo "Shipped files:" >&2; printf '%s\n' "$SHIPPED_STATUS" >&2; }
            [ -z "$TOOL_STATUS" ]    || { echo "Release tooling:" >&2; printf '%s\n' "$TOOL_STATUS" >&2; }
            die "--require-clean: the files above are modified, untracked or ignored"
        fi
    fi
fi

# =============================================================================
# 4. Stage
# =============================================================================
if [ -n "$FROM_COMMIT" ]; then
    # Exact blobs of the commit (git cat-file, so no export-subst or
    # export-ignore attribute can alter them), with the committed file mode.
    while IFS="$TAB" read -r src dst mode; do
        mkdir -p "$STAGE/$(dirname "$dst")"
        git -C "$GIT_ROOT" cat-file blob "$(blob_ref "$src")" > "$STAGE/$dst"
        if [ "$mode" = "100755" ]; then chmod 755 "$STAGE/$dst"; else chmod 644 "$STAGE/$dst"; fi
    done < "$WORK/files.sorted"
else
    while IFS="$TAB" read -r src dst mode; do
        mkdir -p "$STAGE/$(dirname "$dst")"
        cp -p "$PROJ/$src" "$STAGE/$dst"
    done < "$WORK/files.sorted"
fi

# Generated files ------------------------------------------------------------
mkdir -p "$STAGE/data/raw" "$STAGE/data/processed" \
         "$STAGE/figures/diagnostics" "$STAGE/notebooks/figures"
cp "$WORK/data_manifest.md" "$STAGE/data/data_manifest.md"
cp "$WORK/gitignore" "$STAGE/.gitignore"
cat > "$STAGE/data/raw/README.md" <<'EOF'
# data/raw

Raw input data are not stored in this repository. See `../data_manifest.md`
for every file the pipeline reads, its upstream source and citation, and how
to obtain it (download from source or the Zenodo processed bundle). Place the
files here with the relative paths listed there.
EOF
cat > "$STAGE/data/processed/README.md" <<'EOF'
# data/processed

Processed inputs and pipeline outputs live here. Apart from the small CSV
files committed in this directory, they are not stored in the repository.
See `../data_manifest.md` for which files the pipeline rebuilds, which come
from the preprocessing scripts, and where to get the processed bundle.
EOF
# Output directories the code writes into without creating them.
: > "$STAGE/figures/.gitkeep"
: > "$STAGE/figures/diagnostics/.gitkeep"
: > "$STAGE/notebooks/figures/.gitkeep"

# Optional: clear notebook outputs (staging copies only) ----------------------
if [ "$CLEAR_OUTPUTS" -eq 1 ]; then
    echo "Clearing notebook outputs in the export"
    { grep '\.ipynb$' "$WORK/shipped.dest" || true; } | sed "s#^#$STAGE/#" | tr '\n' '\0' \
      | xargs -0 "$PYTHON" -I -c '
import json, sys
for path in sys.argv[1:]:
    with open(path, encoding="utf-8") as f:
        nb = json.load(f)
    for cell in nb.get("cells", []):
        if cell.get("cell_type") == "code":
            cell["outputs"] = []
            cell["execution_count"] = None
    with open(path, "w", encoding="utf-8") as f:
        json.dump(nb, f, indent=1, ensure_ascii=False)
        f.write("\n")
'
fi

# Strip macOS extended attributes (quarantine, Dropbox, Finder info).
if command -v xattr > /dev/null 2>&1; then
    xattr -cr "$STAGE" 2> /dev/null || warn "xattr -cr failed on the staging tree"
fi

# PROVENANCE -------------------------------------------------------------------
# Records only: commit, mode, status of the shipped paths, date, hashes of the
# allowlist and release tooling, options, and sha256 of every exported file.
# No remote URL, branch, absolute path or path outside the shipped set.
if [ -n "$FROM_COMMIT" ]; then
    tool_hash() { git -C "$GIT_ROOT" cat-file blob "$(blob_ref "$1")" | $SHA_CMD | cut -d' ' -f1; }
else
    tool_hash() { sha256_of "$PROJ/$1"; }
fi
{
    echo "$PROVENANCE_HEADER"
    echo "==========================="
    echo "export_date_utc:      $(date -u +%Y-%m-%dT%H:%M:%SZ)"
    echo "export_mode:          $MODE"
    echo "source_commit:        $COMMIT"
    if [ -n "$FROM_COMMIT" ]; then
        echo "shipped_dirty:        no (exported from the commit snapshot, not the working tree)"
    elif [ -n "$SHIPPED_STATUS" ]; then
        echo "shipped_dirty:        yes ($(printf '%s\n' "$SHIPPED_STATUS" | wc -l | tr -d ' ') shipped path(s) differ from source_commit; listed below)"
    else
        echo "shipped_dirty:        no (every shipped file is tracked and matches source_commit)"
    fi
    echo "allowlist:            $ALLOWLIST_LABEL"
    echo "allowlist_sha256:     $(sha256_of "$WORK/allowlist.txt")"
    echo "data_manifest_sha256: $(sha256_of "$WORK/data_manifest.md")"
    echo "gitignore_sha256:     $(sha256_of "$WORK/gitignore")"
    echo "exporter:             $T_EXPORTER sha256 $(tool_hash "$T_EXPORTER")"
    echo "checker:              $T_CHECKER sha256 $(tool_hash "$T_CHECKER")"
    echo "options:              clear_outputs=$CLEAR_OUTPUTS"
    echo "files:                $N_SHIPPED shipped + $(printf '%s\n' "$GENERATED_FILES" | wc -l | tr -d ' ') generated (including this file)"
    if [ -z "$FROM_COMMIT" ]; then
        echo
        echo "## git status --porcelain --ignored of the shipped files (paths relative to the project root)"
        if [ -n "$SHIPPED_STATUS" ]; then printf '%s\n' "$SHIPPED_STATUS"; else echo "(clean)"; fi
    fi
    echo
    echo "## sha256 of exported files"
    LC_ALL=C sort "$WORK/shipped.dest" "$WORK/generated" | grep -vx 'PROVENANCE' \
      | while IFS= read -r f; do
            printf '%s  %s\n' "$(sha256_of "$STAGE/$f")" "$f"
        done
} > "$STAGE/PROVENANCE"

# =============================================================================
# 5. Validate the staging tree (check_export.py is the single validator)
# =============================================================================
LC_ALL=C sort "$WORK/shipped.dest" "$WORK/generated" > "$WORK/expected"

# Backstop: no symlinks at all.
LINKS="$(find "$STAGE" -type l -print)"
[ -z "$LINKS" ] || { echo "$LINKS" >&2; die "symlinks in the staging tree"; }

echo "Validating the staging tree"
if [ "$RELEASE" -eq 1 ]; then CHECK_MODE=release; else CHECK_MODE=iteration; fi
# -I: isolated mode, so neither the dev tree nor the cwd shadows the export.
if ! (cd "$WORK" && "$PYTHON" -I "$SCRIPT_DIR/check_export.py" "$STAGE" \
          --expected "$WORK/expected" --dev-root "$PROJ" --mode "$CHECK_MODE") \
        > "$WORK/check.log" 2>&1; then
    sed "s#$STAGE#<export>#g" "$WORK/check.log" >&2
    die "check_export.py rejected the staging tree; the target was not touched"
fi
sed "s#$STAGE#<export>#g" "$WORK/check.log" | sed 's/^/  | /'

# =============================================================================
# 6. Install into the target
# =============================================================================
RSYNC_OPTS="-a --delete --exclude /.git"
if [ "$TGT_EMPTY" -eq 1 ]; then
    # New or empty target: copy next to it, then rename into place.
    TGT_PARENT="$(dirname "$TGT_PHYS")"
    mkdir -p "$TGT_PARENT"
    INSTALL_TMP="$(mktemp -d "$TGT_PARENT/.spectrum-export-install.XXXXXX")"
    # shellcheck disable=SC2086
    rsync $RSYNC_OPTS "$STAGE/" "$INSTALL_TMP/"
    chmod 755 "$INSTALL_TMP"
    if [ "$TGT_EXISTS" -eq 1 ]; then rmdir "$TGT_PHYS"; fi
    if ! mv "$INSTALL_TMP" "$TGT_PHYS"; then
        [ "$TGT_EXISTS" -eq 0 ] || mkdir -p "$TGT_PHYS"
        die "could not move the export into $TGT_PHYS"
    fi
    INSTALL_TMP=""
else
    # Existing clone or earlier export: mirror the staging tree, keep .git.
    # rsync --delete may only remove files the target itself shipped: files
    # tracked in its git index, or listed in its previous PROVENANCE (an
    # earlier export, committed or not). Anything else (downloaded data, local
    # outputs, stray notes) stops the install before any change is made.
    : > "$WORK/deletable"
    if [ -e "$TGT_PHYS/.git" ]; then
        git -C "$TGT_PHYS" ls-files > "$WORK/deletable" \
            || die "git ls-files failed in $TGT_PHYS"
    fi
    if [ -f "$TGT_PHYS/PROVENANCE" ]; then
        sed -n 's/^[0-9a-f]\{64\}  //p' "$TGT_PHYS/PROVENANCE" >> "$WORK/deletable"
    fi
    echo "PROVENANCE" >> "$WORK/deletable"
    # shellcheck disable=SC2086
    rsync $RSYNC_OPTS -n -i "$STAGE/" "$TGT_PHYS/" > "$WORK/plan"
    { sed -n 's/^\*deleting[[:space:]]*//p' "$WORK/plan" | grep -v '/$' || true; } > "$WORK/deletions"
    LC_ALL=C sort -u "$WORK/deletable" > "$WORK/deletable.sorted"
    LC_ALL=C sort -u "$WORK/deletions" | LC_ALL=C comm -23 - "$WORK/deletable.sorted" > "$WORK/unsafe"
    if [ -s "$WORK/unsafe" ]; then
        sed 's/^/  NOT SHIPPED BY THE TARGET: /' "$WORK/unsafe" >&2
        die "--force would delete files the target did not ship (listed above); move them out first"
    fi
    if [ -s "$WORK/deletions" ]; then
        echo "Removing $(wc -l < "$WORK/deletions" | tr -d ' ') file(s) no longer shipped:"
        sed 's/^/    /' "$WORK/deletions"
    fi
    # shellcheck disable=SC2086
    rsync $RSYNC_OPTS "$STAGE/" "$TGT_PHYS/"
fi

# Verify: a dry run must find nothing left to change.
# shellcheck disable=SC2086
DRIFT="$(rsync $RSYNC_OPTS -n -i "$STAGE/" "$TGT_PHYS/")"
[ -z "$DRIFT" ] || { echo "$DRIFT" >&2; die "target differs from the staging tree after install"; }

N_FILES="$(wc -l < "$WORK/expected" | tr -d ' ')"
SIZE_KB="$(du -sk "$STAGE" | cut -f1)"
echo "Done: $N_FILES files ($N_SHIPPED shipped), ${SIZE_KB} KB, mode $MODE, commit ${COMMIT:0:12}$(
    [ -n "$SHIPPED_STATUS" ] && echo ' (shipped paths dirty; see PROVENANCE)')"
