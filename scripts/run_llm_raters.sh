#!/bin/bash
# Run (or resume) the LLM raters detached from the terminal. Safe to re-run any time:
# finished items are cached in revision/clinician_kit/_cache_*.jsonl and skipped.
#   bash scripts/run_llm_raters.sh
#   tail -f revision/clinician_kit/_llm_gpt.log revision/clinician_kit/_llm_claude.log
cd "$(dirname "$0")/.." || exit 1
REPO=$(pwd)
PY=${PY:-$REPO/../.venv-rev/bin/python}
KIT=$REPO/revision/clinician_kit
export MEDBAYES_RESULTS_DIR=$REPO PYTHONUNBUFFERED=1

# rater processes already running (captured now, so the wait loop cannot match itself)
RUNNING=$(pgrep -f -- "-m medbayes_eval.revision.llm_raters" | tr '\n' ' ')

nohup bash -c "
  for p in $RUNNING; do while kill -0 \$p 2>/dev/null; do sleep 20; done; done
  '$PY' -m medbayes_eval.revision.llm_raters --raters gpt --workers 8 >> '$KIT/_llm_gpt.log' 2>&1 &
  '$PY' -m medbayes_eval.revision.llm_raters --raters claude --workers 6 >> '$KIT/_llm_claude.log' 2>&1 &
  wait
  echo \"[runner] all raters finished \$(date)\"
" >> "$KIT/_llm_runner.log" 2>&1 &
disown
echo "LLM raters queued in background (pid $!; waiting on: ${RUNNING:-none}). Logs: $KIT/_llm_*.log"
