#!/bin/bash
#
# *** SLURM CLUSTER TEMPLATE -- not runnable as-is on its own. ***
# --account= and the environment `source` line below are specific to the
# author's cluster and must be filled in / adapted for yours before this
# will submit. If you don't have a SLURM cluster, use
# 01_hyperparameter-search_stage1_local.sh instead -- it runs the exact
# same hyperparameter sweep as a plain shell loop on a single machine
# (GPU or CPU).
#
#SBATCH --job-name=hp_s1_hn
#SBATCH --account=
#SBATCH --output=output/fine-tuning/logs/logs_hp_stage1_%a_%j.log
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gpus=a100:1
#SBATCH --array=0-39

# =============================================================================
# STAGE 1: Hyperparameter search with hard negative mining
# =============================================================================
# 8 HP configs × 5 constructs = 40 jobs (array 0-39)
# Hard negative parameters fixed at defaults (boost=3.0, percentile=75)
#
# After this completes, run parse_stage1_results.py to find the best
# epochs/lr/margin per construct, then submit stage 2.
# =============================================================================

echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Array Task ID: $SLURM_ARRAY_TASK_ID"
echo "Hostname: $(hostname)"
echo "Start Time: $(date)"
echo "=========================================="

# Activate environment -- EDIT THIS to point at your own venv/conda env
source "/path/to/your/venv/bin/activate"

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK

# --- Constructs (5, excluding impulse_control which uses prototype) ---
CONSTRUCTS=(anxiety apathy depression psychosis sleep)

# --- HP configs (8) ---
EPOCHS_LIST=(3 3 3 3 5 5 5 5)
LR_LIST=(1e-5 1e-5 2e-5 2e-5 1e-5 1e-5 2e-5 2e-5)
MARGIN_LIST=(0.2 0.3 0.2 0.3 0.2 0.3 0.2 0.3)

# --- Map array task ID to construct + config ---
N_CONFIGS=8
CONSTRUCT_IDX=$((SLURM_ARRAY_TASK_ID / N_CONFIGS))
CONFIG_IDX=$((SLURM_ARRAY_TASK_ID % N_CONFIGS))

CONSTRUCT=${CONSTRUCTS[$CONSTRUCT_IDX]}
EPOCHS=${EPOCHS_LIST[$CONFIG_IDX]}
LR=${LR_LIST[$CONFIG_IDX]}
MARGIN=${MARGIN_LIST[$CONFIG_IDX]}

# Fixed hard negative parameters for stage 1
HN_BOOST=3.0
HN_PERCENTILE=75

echo "Construct:  $CONSTRUCT"
echo "Config:     epochs=$EPOCHS, lr=$LR, margin=$MARGIN"
echo "Hard neg:   boost=$HN_BOOST, percentile=$HN_PERCENTILE"
echo ""

# Output directory: organized by construct and config
OUTPUT_DIR="output/fine-tuning/hyperparameter-search/stage1/${CONSTRUCT}/e${EPOCHS}_lr${LR}_m${MARGIN}"

python code/fine-tuning/semantic-learning_harmonize/semantic_learning.py \
    --data output/similarity-analysis/finetuning_dataset.csv \
    --construct $CONSTRUCT \
    --approach contrastive \
    --epochs $EPOCHS \
    --learning-rate $LR \
    --margin $MARGIN \
    --hard-negative-boost $HN_BOOST \
    --hard-negative-percentile $HN_PERCENTILE \
    --device cuda \
    --cv \
    --output "$OUTPUT_DIR"

EXIT_CODE=$?

echo ""
echo "=========================================="
echo "Finished: $(date)"
echo "Exit code: $EXIT_CODE"
echo "=========================================="

exit $EXIT_CODE