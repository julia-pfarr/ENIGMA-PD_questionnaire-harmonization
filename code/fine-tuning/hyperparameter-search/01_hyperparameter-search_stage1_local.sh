#!/usr/bin/env bash
# Plain (non-SLURM) equivalent of 01_hyperparameter-search_stage1.sh, for
# running the same hyperparameter sweep on a single machine -- with or
# without a GPU -- instead of a SLURM cluster.
#
# This is the same 8 HP configs x 5 constructs = 40 jobs (task ids 0-39) as
# the SLURM --array version. Running all 40 sequentially on one machine will
# take a long time (each job is a full contrastive-learning training run);
# pass a single task id to try just one combination first.
#
# Usage:
#   ./01_hyperparameter-search_stage1_local.sh          # run all 40 configs, one after another
#   ./01_hyperparameter-search_stage1_local.sh 7         # run just task 7 (same numbering as the SLURM --array index)
#   DEVICE=cpu ./01_hyperparameter-search_stage1_local.sh 7   # no GPU available (default: cuda)
#
# After this completes, run 02_parse-hyperparameter-search_stage1.py to find
# the best epochs/lr/margin per construct, then stage 2.

set -euo pipefail

BASE="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$BASE"

DEVICE="${DEVICE:-cuda}"

# --- Constructs (5, excluding impulse_control which uses prototype) ---
CONSTRUCTS=(anxiety apathy depression psychosis sleep)

# --- HP configs (8) ---
EPOCHS_LIST=(3 3 3 3 5 5 5 5)
LR_LIST=(1e-5 1e-5 2e-5 2e-5 1e-5 1e-5 2e-5 2e-5)
MARGIN_LIST=(0.2 0.3 0.2 0.3 0.2 0.3 0.2 0.3)
N_CONFIGS=8
MAX_TASK_ID=39  # 0-39, matches the SLURM version's --array=0-39

# Fixed hard negative parameters for stage 1
HN_BOOST=3.0
HN_PERCENTILE=75

run_task() {
    local TASK_ID=$1
    local CONSTRUCT_IDX=$((TASK_ID / N_CONFIGS))
    local CONFIG_IDX=$((TASK_ID % N_CONFIGS))
    local CONSTRUCT=${CONSTRUCTS[$CONSTRUCT_IDX]}
    local EPOCHS=${EPOCHS_LIST[$CONFIG_IDX]}
    local LR=${LR_LIST[$CONFIG_IDX]}
    local MARGIN=${MARGIN_LIST[$CONFIG_IDX]}
    local OUTPUT_DIR="output/fine-tuning/hyperparameter-search/stage1/${CONSTRUCT}/e${EPOCHS}_lr${LR}_m${MARGIN}"

    echo "=========================================="
    echo "Task $TASK_ID: construct=$CONSTRUCT epochs=$EPOCHS lr=$LR margin=$MARGIN device=$DEVICE"
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
    echo "Running all $((MAX_TASK_ID + 1)) configs sequentially on this machine (device=$DEVICE)."
    echo "This will take a long time -- pass a single task id (0-$MAX_TASK_ID) to run just one instead."
    for task_id in $(seq 0 "$MAX_TASK_ID"); do
        run_task "$task_id"
    done
fi
