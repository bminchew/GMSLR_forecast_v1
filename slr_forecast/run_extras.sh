#!/usr/bin/env bash
#
# Run the development-only steps (development repository only; not shipped).
#
# Usage:
#   ./run_extras.sh                 # run all four steps in order
#   ./run_extras.sh evpi figextra   # run just these two
#   ./run_extras.sh --list          # show the steps
#   ./run_extras.sh --help          # show this help
#
# Steps: value of information (evpi), optimal defense height (defense),
# Shapley risk attribution (shapley) and the unpublished figures of
# results_figures_extra.ipynb (figextra). They read
# data/processed/component_results.h5 and write their own groups to it, so
# run them after a completed ./run_pipeline.sh. Nothing in run_pipeline.sh
# reads what they write.
#
# The runner machinery (name checks, NB_TIMEOUT / NB_TIMEOUT_SCALE, PYTHON,
# interpreter report, --list) is that of run_pipeline.sh, which this script
# sources; see its header.

set -euo pipefail
. "$(cd "$(dirname "$0")" && pwd -P)/run_pipeline.sh"

# Development-only steps (name:filename:timeout_seconds), in run order
ENTRIES=(
    "evpi:notebooks/compute_evpi.py:0"
    "defense:notebooks/compute_optimal_defense.py:0"
    "shapley:notebooks/compute_shapley_risk.py:0"
    "figextra:results_figures_extra.ipynb:4800"
)

_needs_results=1
case "${1:-}" in --list) _needs_results=0 ;; esac
for _arg in "$@"; do
    case "$_arg" in -h|--help) _needs_results=0 ;; esac
done
if [ "$_needs_results" = 1 ] && [ ! -f data/processed/component_results.h5 ]; then
    echo "ERROR: data/processed/component_results.h5 not found; run ./run_pipeline.sh first"
    exit 1
fi

pipeline_main "$@"
