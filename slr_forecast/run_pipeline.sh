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

set -euo pipefail
cd "$(dirname "$0")"

NOTEBOOKS_DIR="notebooks"
DEFAULT_TIMEOUT=600  # 10 minutes

# Pipeline order and mapping (name:filename:timeout_seconds)
ENTRIES=(
    "ocean:component_ocean.ipynb:1200"
    "glacier:component_glacier.ipynb:600"
    "greenland:component_greenland.ipynb:600"
    "eais:component_eais.ipynb:600"
    "peninsula:component_apeninsula.ipynb:600"
    "wais:component_wais.ipynb:600"
    "ratestate:bayesian_ratestate.ipynb:2700"
    "summation:component_summation.ipynb:600"
    "forecast:component_forecast.ipynb:600"
    "figures:results_figures.ipynb:600"
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
        printf "  %-12s  %-35s  (%sm timeout)\n" "$_name" "$_file" "$((_timeout / 60))"
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
