#!/usr/bin/env bash
# Plain (non-SLURM) equivalent of 03_hyperparameter-search_stage2.sh, for
# running the boost x percentile search on a single machine -- with or
# without a GPU -- instead of a SLURM cluster.
#
# Reads the same output/fine-tuning/hyperparameter-search/stage2_configs.csv
# as the SLURM version, one row per task id. Running every row sequentially
# on one machine will take a long time; pass a single task id to try just
# one combination first.
#
# Usage:
#   ./03_hyperparameter-search_stage2_local.sh          # run every row in stage2_configs.csv, one after another
#   ./03_hyperparameter-search_stage2_local.sh 7         # run just task 7 (0-indexed row, same numbering as the SLURM --array index)
#   DEVICE=cpu ./03_hyperparameter-search_stage2_local.sh 7   # no GPU available (default: cuda)
#
# Prerequisites:
#   1. Stage 1 completed (either hyperparameter-search_stage1.sh or its _local equivalent)
#   2. Run:
#        python3 code/fine-tuning/hyperparameter-search/02_parse-hyperparameter-search_stage1.py \
#            --results-dir output/fine-tuning/hyperparameter-search/stage1 \
#            --output output/fine-tuning/hyperparameter-search/stage2_configs.csv

set -euo pipefail

BASE="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$BASE"

DEVICE="${DEVICE:-cuda}"
CONFIG_FILE="output/fine-tuning/hyperparameter-search/stage2_configs.csv"

if [[ ! -f "$CONFIG_FILE" ]]; then
    echo "Error: $CONFIG_FILE not found!" >&2
    echo "Run 02_parse-hyperparameter-search_stage1.py first." >&2
    exit 1
fi

N_ROWS=$(($(wc -l < "$CONFIG_FILE") - 1))  # exclude header
MAX_TASK_ID=$((N_ROWS - 1))

run_task() {
    local TASK_ID=$1
    local LINE=$((TASK_ID + 2))
    local ROW
    ROW=$(sed -n "${LINE}p" "$CONFIG_FILE")

    if [[ -z "$ROW" ]]; then
        echo "Error: no config found for task $TASK_ID (line $LINE)" >&2
        exit 1
    fi

    # Parse CSV row: config_id,construct,epochs,learning_rate,margin,hard_negative_boost,hard_negative_percentile
    IFS=',' read -r CONFIG_ID CONSTRUCT EPOCHS LR MARGIN HN_BOOST HN_PERCENTILE <<< "$ROW"
    local OUTPUT_DIR="output/fine-tuning/hyperparameter-search/stage2/${CONSTRUCT}/boost${HN_BOOST}_pct${HN_PERCENTILE}"

    echo "=========================================="
    echo "Task $TASK_ID (config $CONFIG_ID): construct=$CONSTRUCT epochs=$EPOCHS lr=$LR margin=$MARGIN boost=$HN_BOOST pct=$HN_PERCENTILE device=$DEVICE"
    echo "Start Time: $(date)"
    echo "=========================================="

    python3 code/fine-tuning/semantic-learning_harmonize/semantic_learning.py \
        --data output/similarity-analysis/finetuning_dataset.csv \
        --construct "$CONSTRUCT" \
        --approach contrastive \
        --epochs "$EPOCHS" \
        --learning-rate "$LR" \
        --margin "$MARGIN" \
        --hard-negative-boost "$HN_BOOST" \
        --hard-negative-percentile "$HN_PERCENTILE" \
        --device "$DEVICE" \
        --cv \
        --output "$OUTPUT_DIR"

    echo "Finished task $TASK_ID: $(date)"
}

if [[ $# -eq 1 ]]; then
    run_task "$1"
else
    echo "Running all $N_ROWS configs sequentially on this machine (device=$DEVICE)."
    echo "This will take a long time -- pass a single task id (0-$MAX_TASK_ID) to run just one instead."
    for task_id in $(seq 0 "$MAX_TASK_ID"); do
        run_task "$task_id"
    done
fi
