#!/usr/bin/env bash
# Run the score-transformation + validation pipeline end-to-end.
#
# Default (no args) -- QUICKSTART mode:
#   Generates a synthetic master table (if not already present) and runs the
#   full pipeline against it, writing everything to ./quickstart-output/ so
#   nothing in the repo's real output/score-transformation/ is ever touched.
#   This lets anyone verify the pipeline runs correctly without needing the
#   real (non-public) ENIGMA-PD cohort data.
#
#   IMPORTANT: quickstart results are computed from randomly-generated fake
#   data. They do NOT reproduce the manuscript's reported results -- see
#   README "Data" section for that. Use this to check that the code runs and
#   produces sensibly-structured output, not to check specific statistics.
#
# --real -- REAL DATA mode:
#   Runs on your own input/score-transformation/master-table_subject-data.csv
#   (see README "Data" section) and writes to the real
#   output/score-transformation/ tree. Every script is run with its own default
#   paths, identical to running each command by hand as listed in the README.
#
# Usage:
#   ./run_all.sh          # quickstart: synthetic data, scratch output
#   ./run_all.sh --real   # real data (must already be in input/score-transformation/)
#
# Not covered here (see README "Running the pipeline" for the full list):
#   Step 0 (item-dimension mapping) is a rare maintenance step, only needed
#   when the harmonizer's output changes -- its output already ships in
#   input/score-transformation/, so it isn't part of routine reproduction.

set -euo pipefail

# This script lives in code/score-transformation/ but all paths below are
# relative to the project root, so cd two levels up before running anything.
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$PROJECT_ROOT"

MODE="quickstart"
if [[ "${1:-}" == "--real" ]]; then
    MODE="real"
elif [[ -n "${1:-}" ]]; then
    echo "Usage: $0 [--real]" >&2
    exit 1
fi

echo "======================================================================"
if [[ "$MODE" == "quickstart" ]]; then
    echo "QUICKSTART MODE -- synthetic data, scratch output (./quickstart-output/)"
    echo "Results will NOT match the manuscript -- see README 'Data' section."
else
    echo "REAL DATA MODE -- output/score-transformation/"
fi
echo "======================================================================"
echo

if [[ "$MODE" == "quickstart" ]]; then
    SYNTHETIC_TABLE="input/score-transformation/synthetic-data_master-table_subject-data.csv"
    OUTPUT_DIR="./quickstart-output"

    if [[ ! -f "$SYNTHETIC_TABLE" ]]; then
        echo "--- Generating synthetic master table ---"
        python3 code/score-transformation/generate_synthetic_master_table.py
        echo
    else
        echo "Using existing $SYNTHETIC_TABLE"
        echo "(delete it to regenerate, or run generate_synthetic_master_table.py directly)"
        echo
    fi

    echo "--- Step 1: score transformation ---"
    python3 code/score-transformation/run_pipeline.py \
        --master-table "$SYNTHETIC_TABLE" \
        --output-dir "$OUTPUT_DIR"
    echo

    echo "--- Step 2a: hypothesis testing (H1-H3) ---"
    python3 code/score-transformation/validation/hypothesis-testing.py \
        --master-table "$SYNTHETIC_TABLE" \
        --score-dir "$OUTPUT_DIR" \
        --output-dir "$OUTPUT_DIR/hypotheses-testing"
    echo

    echo "--- Step 2b: exploratory validation analyses ---"
    python3 code/score-transformation/validation/exploratory-validation-analyses.py \
        --score-dir "$OUTPUT_DIR" \
        --output-dir "$OUTPUT_DIR" \
        --out-dir "$OUTPUT_DIR/results_figures_tables"
    echo

    echo "======================================================================"
    echo "Done. Results in: $OUTPUT_DIR/"
    echo "These are from SYNTHETIC data and do not match the manuscript."
    echo "======================================================================"
else
    MASTER_TABLE="input/score-transformation/master-table_subject-data.csv"

    if [[ ! -f "$MASTER_TABLE" ]]; then
        echo "ERROR: $MASTER_TABLE not found." >&2
        echo "See README 'Data' section for the expected schema." >&2
        echo "Or run './run_all.sh' (no args) for the synthetic-data quickstart instead." >&2
        exit 1
    fi

    echo "--- Step 1: score transformation ---"
    python3 code/score-transformation/run_pipeline.py
    echo

    echo "--- Step 2a: hypothesis testing (H1-H3) ---"
    python3 code/score-transformation/validation/hypothesis-testing.py
    echo

    echo "--- Step 2b: exploratory validation analyses ---"
    python3 code/score-transformation/validation/exploratory-validation-analyses.py
    echo

    echo "======================================================================"
    echo "Done. Results in: output/score-transformation/"
    echo "======================================================================"
fi
