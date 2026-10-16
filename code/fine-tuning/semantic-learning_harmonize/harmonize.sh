#!/usr/bin/env bash
# Run the fine-tuned harmonizer models on the held-out test items for every construct.
# Activate your own Python environment before running this script.
set -euo pipefail

BASE="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$BASE/code/fine-tuning/semantic-learning_harmonize"

for construct in depression anxiety psychosis apathy sleep impulse_control; do
    python harmonizer.py harmonize \
      --items "$BASE/input/model-testing/validation_complete_round2.csv" \
      --construct "$construct" \
      --inventory "$BASE/output/fine-tuning/harmonized_inventory.csv" \
      --model-dir "$BASE/output/fine-tuning/models-final" \
      --output "$BASE/output/fine-tuning/testing/harmonized_${construct}.csv" \
      --no-auto-update
done
