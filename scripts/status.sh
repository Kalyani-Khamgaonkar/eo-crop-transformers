#!/bin/bash
# One-screen status: jobs, progress per experiment, running runs, failures.
cd ~/eo-crop-transformers
echo "=== JOBS ==="; squeue -u s3295303 -o "%.8i %.10j %.2t %.10M %.20R"
echo; echo "=== PROGRESS (done / claimed / total) ==="
for exp in fold1_sgd compat_cases nine_c1_c4 nine_rest; do
  [ -f experiments/$exp.yaml ] || continue
  total=$(grep -c "^  - {id" experiments/$exp.yaml)
  done=$(grep -l "\[done\]" logs/run_${exp}_[0-9]*.out 2>/dev/null | wc -l)
  claimed=$(ls claims/$exp 2>/dev/null | wc -l)
  printf "%-14s %3d / %3d / %3d\n" $exp $done $claimed $total
done
echo; echo "=== RUNNING NOW (last epoch line) ==="
for f in $(ls -t logs/run_*_[0-9]*.out 2>/dev/null | head -n 12); do
  grep -q "\[done\]" $f && continue
  last=$(grep "\[ep" $f | tail -n 1)
  [ -n "$last" ] && printf "%-34s %s\n" "$(basename $f .out)" "$last"
done
echo; echo "=== PROBLEMS ==="
p=$(grep -l -i "traceback\|error\|nan" logs/run_*_[0-9]*.out 2>/dev/null | grep -v smoke)
f=$(grep -h "FAILED" logs/eo-packed_*.out 2>/dev/null | tail -n 5)
[ -z "$p$f" ] && echo "none" || { echo "$p"; echo "$f"; }
