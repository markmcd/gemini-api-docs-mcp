#!/bin/bash

# Configuration
models=(
  "gemini-3-flash-preview"
  "gemini-3-pro-preview"
  "gemini-2.5-flash"
  "gemini-2.5-pro"
  "gemini-2.5-flash-lite"
)

modes=("skill" "vanilla")
LOG_FILE="tests/full_analysis.log"

# Allow resuming a previous run by passing a batch ID
if [ -z "$1" ]; then
  BATCH_ID=$(date +%Y%m%d_%H%M%S)
  echo "Starting New Batch: $BATCH_ID"
else
  BATCH_ID="$1"
  echo "Resuming Batch: $BATCH_ID"
fi

echo "Results will be saved in tests/runs/${BATCH_ID}_..."

for model in "${models[@]}"; do
  for mode in "${modes[@]}"; do
    OUTPUT_DIR="tests/runs/${BATCH_ID}_${mode}_${model}"
    echo "================================================================" | tee -a "$LOG_FILE"
    echo "RUNNING: Model=$model, Mode=$mode" | tee -a "$LOG_FILE"
    echo "DIR: $OUTPUT_DIR" | tee -a "$LOG_FILE"
    echo "================================================================" | tee -a "$LOG_FILE"
    
    # Run the harness. We use || true to ensure the loop continues even if a run has failures.
    uv run tests/eval_harness.py --mode "$mode" --model "$model" --output-dir "$OUTPUT_DIR" >> "$LOG_FILE" 2>&1 || echo "Run finished with some failures (check result.json)" | tee -a "$LOG_FILE"
    
    echo "Finished: Model=$model, Mode=$mode at $(date)" | tee -a "$LOG_FILE"
    echo "" | tee -a "$LOG_FILE"
  done
done

echo "Full Analysis Complete: $(date)" | tee -a "$LOG_FILE"
