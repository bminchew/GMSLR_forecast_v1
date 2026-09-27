#!/usr/bin/env bash
#
# Run the SLR forecast pipeline (or individual notebooks).
#
# Usage:
#   ./run_pipeline.sh                     # run all notebooks in order
#   ./run_pipeline.sh summation forecast  # run just these two
#   ./run_pipeline.sh --list              # show available notebooks
#
# Each notebook is executed with a fresh kernel via jupyter nbconvert.
# Outputs are written in-place. Failures stop the pipeline.
# Entries whose file ends in .py are Python scripts, run from the repository
# root (path relative to it); they write their own groups to
# component_results.h5 and read the forecast written by the notebook before.

set -euo pipefail
cd "$(dirname "$0")"

NOTEBOOKS_DIR="notebooks"
DEFAULT_TIMEOUT=600  # 10 minutes

# Pipeline order and mapping (name:filename:timeout_seconds)
ENTRIES=(
    "ocean:component_ocean.ipynb:2400"
    "glacier:component_glacier.ipynb:1200"
    "greenland:component_greenland.ipynb:1200"
    "eais:component_eais.ipynb:1200"
    "peninsula:component_apeninsula.ipynb:1200"
    "wais:component_wais.ipynb:1200"
    "ratestate:bayesian_ratestate.ipynb:5400"
    "summation:component_summation.ipynb:1200"
    "forecast:component_forecast.ipynb:1200"
    "slowwais:scripts/build_slow_wais_p90.py:0"
    "evpi:notebooks/compute_evpi.py:0"
    "defense:notebooks/compute_optimal_defense.py:0"
    "shapley:notebooks/compute_shapley_risk.py:0"
    "figures:results_figures.ipynb:1200"
)

lookup_entry() {
    # Returns "filename:timeout" for a given name
    local target="$1"
    for entry in "${ENTRIES[@]}"; do
        local name="${entry%%:*}"
        local rest="${entry#*:}"
        if [ "$name" = "$target" ]; then
            echo "$rest"
            return 0
        fi
    done
    return 1
}

run_notebook() {
    local name="$1"
    local rest
    rest=$(lookup_entry "$name") || {
        echo "ERROR: unknown notebook '$name'. Use --list to see options."
        exit 1
    }
    local nbfile="${rest%%:*}"
    local timeout="${rest#*:}"

    if [[ "$nbfile" == *.py ]]; then
        if [ ! -f "$nbfile" ]; then
            echo "ERROR: $nbfile not found"
            return 1
        fi
        echo ""
        echo "========================================"
        echo "  Running script: $nbfile"
        echo "========================================"
        local start_time=$(date +%s)
        "${PYTHON:-python3}" "$nbfile" 2>&1
        local end_time=$(date +%s)
        echo "  Done: $nbfile ($(( end_time - start_time ))s)"
        return 0
    fi

    local path="${NOTEBOOKS_DIR}/${nbfile}"

    if [ ! -f "$path" ]; then
        echo "ERROR: $path not found"
        return 1
    fi

    echo ""
    echo "========================================"
    echo "  Running: $nbfile  (timeout: ${timeout}s)"
    echo "========================================"
    local start_time=$(date +%s)

    jupyter nbconvert \
        --to notebook \
        --execute \
        --inplace \
        --ExecutePreprocessor.timeout=$timeout \
        --ExecutePreprocessor.kernel_name=python3 \
        "$path" 2>&1

    local end_time=$(date +%s)
    local elapsed=$(( end_time - start_time ))
    echo "  Done: $nbfile (${elapsed}s)"
}

# --- Main ---

if [ "${1:-}" = "--list" ]; then
    echo "Available notebooks:"
    for entry in "${ENTRIES[@]}"; do
        _name="${entry%%:*}"
        _rest="${entry#*:}"
        _file="${_rest%%:*}"
        _timeout="${_rest#*:}"
        if [[ "$_file" == *.py ]]; then
            printf "  %-12s  %-35s  (script)\n" "$_name" "$_file"
        else
            printf "  %-12s  %-35s  (%sm timeout)\n" "$_name" "$_file" "$((_timeout / 60))"
        fi
    done
    exit 0
fi

if [ $# -gt 0 ]; then
    # Run specific notebooks
    for name in "$@"; do
        run_notebook "$name"
    done
else
    # Run full pipeline
    echo "Running full pipeline (${#ENTRIES[@]} notebooks)..."
    total_start=$(date +%s)

    for entry in "${ENTRIES[@]}"; do
        name="${entry%%:*}"
        run_notebook "$name"
    done

    total_end=$(date +%s)
    total_elapsed=$(( total_end - total_start ))
    echo ""
    echo "========================================"
    echo "  Pipeline complete (${total_elapsed}s total)"
    echo "========================================"
fi
