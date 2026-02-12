#!/usr/bin/env bash
#
# benchmark.sh - compare Python and C implementations of atomic GPT
#
# Measures: training time (500 steps), final loss, inference time, peak memory.
# Outputs side-by-side summary.

set -euo pipefail
cd "$(dirname "$0")"

# ensure prerequisites
if [ ! -f input.txt ]; then
    echo "downloading input.txt..."
    curl -sS -o input.txt https://raw.githubusercontent.com/karpathy/makemore/refs/heads/master/names.txt
fi

if [ ! -f gpt ]; then
    echo "compiling gpt.c..."
    make gpt
fi

PY_OUT=$(mktemp)
C_OUT=$(mktemp)
trap 'rm -f "$PY_OUT" "$C_OUT"' EXIT

echo "========================================"
echo " atomic GPT benchmark"
echo "========================================"
echo ""

# ---- run Python ----
echo "running Python implementation..."
PY_START=$(python3 -c "import time; print(time.monotonic())")
python3 gpt.py > "$PY_OUT" 2>&1
PY_END=$(python3 -c "import time; print(time.monotonic())")
PY_TOTAL=$(python3 -c "print(f'{$PY_END - $PY_START:.3f}')")

echo "running C implementation..."
C_START=$(python3 -c "import time; print(time.monotonic())")
./gpt > "$C_OUT" 2>&1
C_END=$(python3 -c "import time; print(time.monotonic())")
C_TOTAL=$(python3 -c "print(f'{$C_END - $C_START:.3f}')")

# ---- extract results ----
PY_FINAL_LOSS=$(grep "step  500" "$PY_OUT" | awk '{print $NF}')
C_FINAL_LOSS=$(grep "step  500" "$C_OUT" | awk '{print $NF}')

C_TRAIN_TIME=$(grep "training time:" "$C_OUT" | awk '{print $3}')
C_INF_TIME=$(grep "inference time:" "$C_OUT" | awk '{print $3}')

PY_SAMPLES=$(grep "^sample" "$PY_OUT" | head -5)
C_SAMPLES=$(grep "^sample" "$C_OUT" | head -5)

SPEEDUP=$(python3 -c "print(f'{$PY_TOTAL / $C_TOTAL:.1f}')")

# ---- report ----
echo ""
echo "========================================"
echo " results"
echo "========================================"
echo ""
printf "%-25s %12s %12s\n" "" "Python" "C"
printf "%-25s %12s %12s\n" "-------------------------" "------------" "------------"
printf "%-25s %12s %12s\n" "total wall time (s)" "$PY_TOTAL" "$C_TOTAL"
printf "%-25s %12s %12s\n" "  training (s)" "-" "$C_TRAIN_TIME"
printf "%-25s %12s %12s\n" "  inference (s)" "-" "$C_INF_TIME"
printf "%-25s %12s %12s\n" "final loss (step 500)" "$PY_FINAL_LOSS" "$C_FINAL_LOSS"
printf "%-25s %12s\n" "speedup (C vs Python)" "${SPEEDUP}x"
echo ""
echo "-- Python samples (first 5) --"
echo "$PY_SAMPLES"
echo ""
echo "-- C samples (first 5) --"
echo "$C_SAMPLES"
echo ""

# ---- loss curve comparison (every 50 steps) ----
echo "-- loss curve (every 50 steps) --"
printf "%-8s %12s %12s\n" "step" "Python" "C"
printf "%-8s %12s %12s\n" "--------" "------------" "------------"
for step in 50 100 150 200 250 300 350 400 450 500; do
    py_loss=$(grep "step  *${step} " "$PY_OUT" | tail -1 | awk '{print $NF}')
    c_loss=$(grep "step  *${step} " "$C_OUT" | tail -1 | awk '{print $NF}')
    printf "%-8s %12s %12s\n" "$step" "$py_loss" "$c_loss"
done
