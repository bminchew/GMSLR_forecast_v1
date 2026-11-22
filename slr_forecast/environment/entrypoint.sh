#!/usr/bin/env bash
#
# Container entrypoint for the SPECTRUM sea-level forecast pipeline.
#
#   docker run ... spectrum                    # full pipeline (all ENTRIES)
#   docker run ... spectrum ocean glacier      # selected entries
#   docker run ... spectrum --list             # list entries
#   docker run ... spectrum --help             # run_pipeline.sh help
#   docker run ... spectrum exec <cmd> [args]  # run any command in the env
#                                              # (e.g. exec pytest -rs tests)
#
# The micromamba base environment is already activated by the image's
# /usr/local/bin/_entrypoint.sh, which runs this script.
#
# Ensemble size: src/slr_forecast/config.py reads the environment variable
# SLR_N_SAMPLES (unset or empty means 10000, the production value).  Pass it
# with `docker run -e SLR_N_SAMPLES=500 ...`; as an environment variable it
# reaches every notebook kernel and script.  A reduced run overwrites
# data/processed/ in place.
set -euo pipefail
cd /work
# Output directories the scripts expect; recreated here because a host
# directory mounted at /work/figures hides the ones made in the image.
mkdir -p figures/diagnostics notebooks/figures 2>/dev/null || true

if [ "${1:-}" = "exec" ]; then
    shift
    if [ $# -eq 0 ]; then
        echo "ERROR: exec needs a command." >&2
        echo "Usage: docker run ... spectrum                    # full pipeline" >&2
        echo "       docker run ... spectrum ENTRY [ENTRY ...]  # selected entries" >&2
        echo "       docker run ... spectrum --list             # list entries" >&2
        echo "       docker run ... spectrum exec CMD [ARGS]    # e.g. exec bash" >&2
        exit 2
    fi
    exec "$@"
fi

_checks=1
case "${1:-}" in --list) _checks=0 ;; esac
for _arg in "$@"; do
    case "$_arg" in -h|--help) _checks=0 ;; esac
done
if [ "$_checks" = 1 ]; then
    if [ -z "$(ls -A /work/data/raw 2>/dev/null)" ]; then
        echo "WARNING: /work/data/raw is empty or missing; mount the raw data read-only" >&2
        echo "         (-v /path/to/data/raw:/work/data/raw:ro)." >&2
    fi
    if ! touch /work/data/processed/.write_test 2>/dev/null; then
        echo "ERROR: /work/data/processed is not writable; mount it read-write" >&2
        echo "       (-v /path/to/data/processed:/work/data/processed)." >&2
        exit 1
    fi
    rm -f /work/data/processed/.write_test
    # Results persist if /work/data/processed or one of its ancestors below /
    # (/work/data, or the whole repository at /work) is a mount; field 5 of
    # mountinfo is the mount point.
    if ! awk '$5 == "/work/data/processed" || $5 == "/work/data" || $5 == "/work" { found = 1 }
              END { exit !found }' /proc/self/mountinfo 2>/dev/null; then
        echo "WARNING: neither /work/data/processed nor /work/data nor /work is a mount;" >&2
        echo "         results stay inside the container and are lost when it is removed." >&2
    fi
    python - <<'EOF'
from slr_forecast import config
import threadpoolctl
pools = ", ".join(f"{d['internal_api']} {d.get('num_threads')}" for d in threadpoolctl.threadpool_info())
print(f"slr_forecast: PROJECT_ROOT={config.PROJECT_ROOT}  N_SAMPLES={config.N_SAMPLES}")
print(f"thread pools: {pools}")
EOF
fi

# Through bash, so a mounted repository need not keep the executable bit.
exec bash ./run_pipeline.sh "$@"
