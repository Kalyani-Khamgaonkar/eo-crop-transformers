#!/bin/bash
# Progress of all runs: epoch, best val mIoU so far, sec per epoch, estimated finish time.
#   bash scripts/progress.sh                 # all known runs
#   bash scripts/progress.sh rlt20_0 apt20_0 # only these
# Separate jobs log to logs/<jobname>_<jobid>.out; packed jobs to logs/run_<config>_<index>.out.
E=${E:-200}
names=("$@")
if [ ${#names[@]} -eq 0 ]; then
  names=(flex20_0 flex20_1 flex20_2 flex20_3 flex20_4 flex20_5 flex20_6
         flexfull_0 flexfull_1 flexfull_2 flexfull_3 flexfull_4 flexfull_5 flexfull_6
         rlt20_0 rlt20_1 rltfull_0
         adaptive_20pct_0 adaptive_20pct_1 adaptive_20pct_2 adaptive_20pct_3
         apt20_0 apt20_1 aptfull_0 apt3d20_0 apt3d20_1 apt3dfull_0)
fi
printf "%-17s %-6s %-9s %-9s %-7s %s\n" run state epoch best_val s/epoch finish
for n in "${names[@]}"; do
  job=$n; [[ $n == adaptive_20pct_* ]] && job=adapt20                     # packed job name
  st=$(squeue -u "$USER" -h -n "$job" -o "%t" 2>/dev/null | sort | head -1); st=${st:-"-"}
  logs=$(ls logs/${n}_*.out logs/run_${n}.out 2>/dev/null)
  f=""
  [ -n "$logs" ] && f=$(grep -l "^\[ep\|^\[done\]" $logs 2>/dev/null | xargs -r ls -t | head -1)
  if [ -z "$f" ]; then
    [ "$st" != "-" ] && printf "%-17s %-6s %s\n" "$n" "$st" "waiting / starting"
    continue
  fi
  if grep -q "^\[done\]" "$f"; then printf "%-17s %-6s %s\n" "$n" "$st" "DONE: $(grep -h '^\[done\]' "$f" | cut -c8-40)"; continue; fi
  grep -h "^\[ep" $logs | awk -v n="$n" -v st="$st" -v E="$E" '
    { ep=$2; sub(/\]/,"",ep); ep+=0; s=$NF; sub(/s$/,"",s);
      split($6,m,"/"); if (m[2]+0>best) best=m[2]+0;
      if (ep>=last) last=ep; t[++k]=s }
    END { a=0; c=0; for(i=k;i>k-5&&i>0;i--){a+=t[i];c++}; a=a/c; left=(E-1-last)*a;
          cmd="date -d \"+" int(left) " seconds\" \"+%a %H:%M\""; cmd | getline fin; close(cmd);
          printf "%-17s %-6s %3d/%-5d %-9.3f %-7.0f %s\n", n, st, last+1, E, best, a, fin }'
done
