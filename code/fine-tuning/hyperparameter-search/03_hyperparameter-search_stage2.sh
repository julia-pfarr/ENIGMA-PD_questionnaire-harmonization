#!/bin/bash
#
# *** SLURM CLUSTER TEMPLATE -- not runnable as-is on its own. ***
# --account= and the environment `source` line below are specific to the
# author's cluster and must be filled in / adapted for yours before this
# will submit. If you don't have a SLURM cluster, use
# 03_hyperparameter-search_stage2_local.sh instead -- it runs the exact
# same search as a plain shell loop on a single machine (GPU or CPU).
#
#SBATCH --job-name=hp_s2_hn
#SBATCH --account=
#SBATCH --output=output/fine-tuning/logs/logshp_stage2_%a_%j.log
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gpus=a100:1
#SBATCH --array=0-11

# =============================================================================
# STAGE 2: Boost × Percentile search
# =============================================================================
# Uses the best epochs/lr/margin per construct from stage 1.
# Searches over:
#   boost:      [1.0, 2.0, 3.0, 5.0]     (1.0 = no HN mining, control)
#   percentile: [50, 75, 90]
#
# 5 constructs × 4 boost × 3 percentile = 60 jobs (array 0-59)
#
# IMPORTANT: Update --array above if parse_stage1_results.py reports
# a different total. The script prints the correct range.
#
# Prerequisites:
#   1. Stage 1 completed
#   2. Run: python parse_stage1_results.py \
#               --results-dir hp_search_hn/stage1 \
#               --output stage2_configs.csv
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

# --- Read config from CSV ---
CONFIG_FILE="output/fine-tuning/hyperparameter-search/stage2_configs.csv"
 
if [ ! -f "$CONFIG_FILE" ]; then
    echo "Error: $CONFIG_FILE not found!"
    echo "Run parse_stage1_results.py first."
    exit 1
fi
 
# Read the row for this array task (header is line 1, data starts line 2)
LINE=$((SLURM_ARRAY_TASK_ID + 2))
ROW=$(sed -n "${LINE}p" "$CONFIG_FILE")
 
if [ -z "$ROW" ]; then
    echo "Error: no config found for array task $SLURM_ARRAY_TASK_ID (line $LINE)"
    exit 1
fi
 
# Parse CSV row: config_id,construct,epochs,learning_rate,margin,hard_negative_boost,hard_negative_percentile
IFS=',' read -r CONFIG_ID CONSTRUCT EPOCHS LR MARGIN HN_BOOST HN_PERCENTILE <<< "$ROW"
 
echo "Config ID:    $CONFIG_ID"
echo "Construct:    $CONSTRUCT"
echo "Epochs:       $EPOCHS"
echo "LR:           $LR"
echo "Margin:       $MARGIN"
echo "HN Boost:     $HN_BOOST"
echo "HN Percentile: $HN_PERCENTILE"
echo ""
 
# Output directory
OUTPUT_DIR="output/fine-tuning/hyperparameter-search/stage2/${CONSTRUCT}/boost${HN_BOOST}_pct${HN_PERCENTILE}"
 
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