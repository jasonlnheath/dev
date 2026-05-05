#!/bin/bash
STEPS=(
  "step0_repl:tests/step0_repl.mal:step0_repl.py"
  "step1_read_print:tests/step1_read_print.mal:step1_read_print.py"
  "step2_eval:tests/step2_eval.mal:step2_eval.py"
  "step3_env:tests/step3_env.mal:step3_env.py"
  "step4_if_fn_do:tests/step4_if_fn_do.mal:step4_if_fn_do.py"
  "step5_tco:tests/step5_tco.mal:step5_tco.py"
  "step6_file:tests/step6_file.mal:step6_file.py"
  "step7_quote:tests/step7_quote.mal:step7_quote.py"
  "step8_macros:tests/step8_macros.mal:step8_macros.py"
  "step9_try:tests/step9_try.mal:step9_try.py"
  "stepA_mal:tests/stepA_mal.mal:stepA_mal.py"
)

for entry in "${STEPS[@]}"; do
  IFS=':' read -r name test impl <<< "$entry"
  echo "=========================================="
  echo "Running step: $name ($impl)"
  echo "=========================================="
  python "C:\dev\ooda_driver.py" \
    --project . \
    --test "python run_test.py $test python $impl" \
    --task "Implement $name" \
    --primary-file "$impl" \
    --self-hint \
    --stuck-threshold 1 \
    --file-glob "*.py" \
    --timeout 60 \
    --step-label "$name"
  exit_code=$?
  echo "Step $name exit code: $exit_code"
  if [ $exit_code -ne 0 ]; then
    echo "STEP $name FAILED - stopping"
    break
  fi
  echo ""
done
echo "All steps complete"
