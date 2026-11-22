#!/usr/bin/env bash
#
# Run the SLR forecast pipeline (or individual entries).
#
# Usage:
#   ./run_pipeline.sh                     # run all entries in order
#   ./run_pipeline.sh summation forecast  # run just these two
#   ./run_pipeline.sh --list              # show the entries
#   ./run_pipeline.sh --help              # show this help
#   SLR_N_SAMPLES=500 ./run_pipeline.sh   # reduced ensemble (default 10000; see
#                                         # src/slr_forecast/config.py); overwrites
#                                         # data/processed/ in place
#
# Each notebook is executed with a fresh kernel via jupyter nbconvert.
# Outputs are written in-place. Failures stop the pipeline. Every named entry
# is checked before the first one runs; an unknown name runs nothing.
# Entries whose file ends in .py are Python scripts, run from the repository
# root (path relative to it); they write their own groups to
# component_results.h5 and read the forecast written by the notebook before.
#
# The steps write into data/processed/component_results.h5, which they create
# only if it is absent: groups that an earlier run wrote and this run does not
# rewrite stay in the file.
#
# Notebook timeouts: each notebook entry's nbconvert timeout (per cell) was
# set for the macOS development machine (10 threads). A slower run (the
# container uses one thread) can
# hit them. NB_TIMEOUT_SCALE multiplies every notebook timeout (default 1,
# e.g. NB_TIMEOUT_SCALE=4); NB_TIMEOUT sets every notebook timeout in seconds
# and takes precedence (NB_TIMEOUT=-1: no limit).
#
# Interpreters: script entries run with ${PYTHON:-python3}; notebooks run in
# the Jupyter kernel named python3. Outside Docker these can be different
# interpreters, so the pipeline prints both before the first step.
#
# Sourcing this file defines ENTRIES and the functions below without running
# anything; pipeline_main "$@" then runs the entries in ENTRIES.

set -euo pipefail
# The invoked script (for --help), resolved before the cd below
SELF_SCRIPT="$(cd "$(dirname "$0")" && pwd -P)/$(basename "$0")"
cd "$(dirname "${BASH_SOURCE[0]}")"

NOTEBOOKS_DIR="notebooks"

# Pipeline order and mapping (name:filename:timeout_seconds)
ENTRIES=(
    "ocean:component_ocean.ipynb:14400"
    "oceanholdout:scripts/ocean_early_holdout.py:0"
    "glacier:component_glacier.ipynb:4800"
    "glacierdelay:scripts/glacier_delayed_response.py:0"
    "greenland:component_greenland.ipynb:4800"
    "smbsupp:scripts/plot_supp_greenland_smb_emulator.py:0"
    "smbrcm:scripts/plot_greenland_smb_mar_vs_emulator.py:0"
    "smbstate:scripts/smb_state_space_diagnostic.py:0"
    "smblag:scripts/joint_lag_fit_mar.py:0"
    "eais:component_eais.ipynb:4800"
    "peninsula:component_apeninsula.ipynb:4800"
    "pencurv:scripts/peninsula_curvature_check.py:0"
    "wais:component_wais.ipynb:4800"
    "ipccdist:notebooks/precompute_ipcc_distributions.py:0"
    "summation:component_summation.ipynb:4800"
    "forecast:component_forecast.ipynb:4800"
    "thermorec:scripts/thermosteric_record_sensitivity.py:0"
    "zostoga:scripts/cmip6_zostoga.py:0"
    "slowwais:scripts/build_slow_wais_p90.py:0"
    "grsens:scripts/greenland_sensitivity_forecast.py:0"
    "slownums:scripts/slow_wais_p90_numbers.py:0"
    "varshares:scripts/variance_shares.py:0"
    "numbers:scripts/manuscript_numbers.py:0"
    "figures:results_figures.ipynb:4800"
)

print_help() {
    # The header comment of the invoked script, without the leading '# '
    awk 'NR <= 2 { next } /^#/ { sub(/^# ?/, ""); print; next } { exit }' "$SELF_SCRIPT"
}

check_timeouts() {
    # Notebook timeout overrides (see the header)
    local re_timeout='^(-1|[1-9][0-9]*)$'
    if [ -n "${NB_TIMEOUT:-}" ] && ! [[ $NB_TIMEOUT =~ $re_timeout ]]; then
        echo "ERROR: NB_TIMEOUT must be -1 (no limit) or a positive number of seconds, got '${NB_TIMEOUT}'"
        exit 1
    fi
    NB_TIMEOUT_SCALE="${NB_TIMEOUT_SCALE:-1}"
    local re_scale='^([0-9]+([.][0-9]*)?|[.][0-9]+)$'
    if ! [[ $NB_TIMEOUT_SCALE =~ $re_scale ]] || ! awk -v s="$NB_TIMEOUT_SCALE" 'BEGIN { exit !(s > 0) }'; then
        echo "ERROR: NB_TIMEOUT_SCALE must be a positive number, got '${NB_TIMEOUT_SCALE}'"
        exit 1
    fi
}

lookup_entry() {
    # Returns "filename:timeout" for a given name
    local target="$1" entry
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

check_names() {
    # Every argument must name an entry; otherwise list the valid names and exit
    local name unknown=""
    for name in "$@"; do
        lookup_entry "$name" > /dev/null || unknown="$unknown '$name'"
    done
    if [ -n "$unknown" ]; then
        echo "ERROR: unknown entry name(s):$unknown. Nothing was run."
        echo "Valid names:"
        list_entries "${ENTRIES[@]}"
        exit 1
    fi
}

nb_timeout() {
    # Effective nbconvert timeout (s) for a notebook entry's timeout $1
    if [ -n "${NB_TIMEOUT:-}" ]; then
        echo "$NB_TIMEOUT"
    else
        awk -v b="$1" -v s="$NB_TIMEOUT_SCALE" 'BEGIN { t = int(b * s + 0.5); if (t < 1) t = 1; print t }'
    fi
}

report_interpreters() {
    # Scripts run with ${PYTHON:-python3}, notebooks in the python3 kernel
    local py="${PYTHON:-python3}"
    local spy kpy jup
    spy=$("$py" -c 'import sys; print(sys.executable)' 2>/dev/null) || spy="not runnable"
    echo "Scripts:   $py ($spy)"
    kpy=$(jupyter kernelspec list --json 2>/dev/null | "$py" -c '
import json, sys
print(json.load(sys.stdin)["kernelspecs"]["python3"]["spec"]["argv"][0])' 2>/dev/null) || kpy=""
    case "$kpy" in
        "")
            kpy="unknown: no python3 kernelspec found" ;;
        python|python3|python3.*)
            # jupyter_client starts a bare python with the interpreter running jupyter
            if jup=$(command -v jupyter 2>/dev/null); then
                kpy="$kpy, i.e. the interpreter of $jup: $(head -n 1 "$jup" | sed 's/^#! *//')"
            fi ;;
    esac
    echo "Notebooks: kernel python3 ($kpy)"
}

list_entries() {
    local entry _name _rest _file _timeout
    for entry in "$@"; do
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
}

run_notebook() {
    local name="$1"
    local rest
    rest=$(lookup_entry "$name") || {
        echo "ERROR: unknown entry '$name'. Use --list to see options."
        exit 1
    }
    local nbfile="${rest%%:*}"
    local timeout="${rest#*:}"
    if [[ "$nbfile" != *.py ]]; then
        timeout=$(nb_timeout "$timeout")
    fi

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
    if [ "$timeout" = "-1" ]; then
        echo "  Running: $nbfile  (no timeout)"
    else
        echo "  Running: $nbfile  (timeout: ${timeout}s)"
    fi
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

pipeline_main() {
    local arg name entry n_nb=0 n_py=0
    for arg in "$@"; do
        case "$arg" in
            -h|--help) print_help; exit 0 ;;
        esac
    done
    check_timeouts

    if [ "${1:-}" = "--list" ]; then
        echo "Available entries:"
        list_entries "${ENTRIES[@]}"
        echo ""
        echo "Notebook timeouts are for the macOS development machine (10 threads): scale them"
        echo "with NB_TIMEOUT_SCALE=<factor>, or set them with NB_TIMEOUT=<seconds> (-1: no limit)."
        exit 0
    fi

    if [ $# -gt 0 ]; then
        check_names "$@"
        report_interpreters
        # Run specific entries
        for name in "$@"; do
            run_notebook "$name"
        done
    else
        report_interpreters
        # Run all entries
        for entry in "${ENTRIES[@]}"; do
            case "$entry" in *.py:*) n_py=$((n_py + 1)) ;; *) n_nb=$((n_nb + 1)) ;; esac
        done
        echo "Running all ${#ENTRIES[@]} entries ($n_nb notebooks, $n_py scripts)..."
        local total_start total_end
        total_start=$(date +%s)

        for entry in "${ENTRIES[@]}"; do
            name="${entry%%:*}"
            run_notebook "$name"
        done

        total_end=$(date +%s)
        echo ""
        echo "========================================"
        echo "  Pipeline complete ($(( total_end - total_start ))s total)"
        echo "========================================"
    fi
}

# Run only when executed, not when sourced
[ "${BASH_SOURCE[0]}" = "$0" ] || return 0
pipeline_main "$@"
