#!/bin/bash
while true; do
  clear
  echo "=== Evaluation Progress Monitor ==="
  echo "Time: $(date)"
  echo ""

  # 1. List Completed/Other Runs
  echo "--- Run History ---"
  # Find all run directories sorted by time (oldest first)
  ALL_RUNS=$(ls -d tests/runs/*/ 2>/dev/null | sort)
  
  if [ -z "$ALL_RUNS" ]; then
    echo "No runs found."
  else
    # Determine the latest run to distinguish it
    LATEST_RUN=$(ls -td tests/runs/*/ 2>/dev/null | head -n 1)
    
    for run in $ALL_RUNS; do
      DIR_NAME=$(basename "$run")
      
      # Check for result.json to see if it finished/has stats
      RESULT_FILE="$run/result.json"
      STATUS="In Progress/Partial"
      SUMMARY=""
      
      if [ -f "$RESULT_FILE" ]; then
        # Try to extract pass/total from result.json using grep/sed (simple parsing)
        TOTAL=$(grep -o '"total": [0-9]*' "$RESULT_FILE" | head -1 | cut -d' ' -f2)
        PASSED=$(grep -o '"passed": [0-9]*' "$RESULT_FILE" | head -1 | cut -d' ' -f2)
        if [ ! -z "$TOTAL" ]; then
           STATUS="Completed"
           SUMMARY="($PASSED/$TOTAL passed)"
        fi
      else
         # Fallback to file count
         COUNT=$(ls "$run/generated" 2>/dev/null | wc -l)
         SUMMARY="($COUNT generated files)"
      fi

      MARKER=" "
      if [ "$run" == "$LATEST_RUN" ]; then
        MARKER=">" # Mark current/latest
      fi
      
      printf "%s %-50s %-10s %s\n" "$MARKER" "$DIR_NAME" "$STATUS" "$SUMMARY"
    done
  fi

  echo ""
  
  # 2. Detailed Progress for Latest Run
  if [ ! -z "$LATEST_RUN" ]; then
    echo "--- Current Run Details ---"
    DIR_NAME=$(basename "$LATEST_RUN")
    COUNT=$(ls "$LATEST_RUN/generated" 2>/dev/null | wc -l)
    TOTAL=117
    PERCENT=$((COUNT * 100 / TOTAL))
    
    echo "Directory: $DIR_NAME"
    echo "Progress:  $COUNT / $TOTAL ($PERCENT%)"
    
    # Progress Bar
    BAR_LEN=$((PERCENT / 2))
    printf "["
    for ((i=0; i<BAR_LEN; i++)); do printf "="; done
    for ((i=BAR_LEN; i<50; i++)); do printf " "; done
    printf "]\n"

    # Show last 3 generated files with timestamps
    echo ""
    echo "Latest generated files:"
    find "$LATEST_RUN/generated" -maxdepth 1 -type f -printf "%T@ %p\n" | sort -nr | head -n 3 | while read -r time path; do
        printf "%s %s\n" "$(date -d "@$time" +"%H:%M:%S")" "$(basename "$path")"
    done
  fi
  
  sleep 5
done
