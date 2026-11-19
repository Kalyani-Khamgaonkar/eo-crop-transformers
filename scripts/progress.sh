#!/bin/bash
# Progress of the FlexiViT runs: epoch, best val mIoU so far, sec per epoch, estimated finish time.
#   bash scripts/progress.sh
E=${E:-200}
printf "%-11s %-6s %-9s %-9s %-7s %s\n" run state epoch best_val s/epoch finish
for n in flex20_0 flex20_1 flex20_2 flex20_3 flex20_4 flex20_5 flex20_6 flexfull_0 flexfull_1 flexfull_2 flexfull_3 flexfull_4 flexfull_5 flexfull_6; do
  st=$(squeue -u "$USER" -h -n "$n" -t R -o "%t" | head -1); st=${st:-"-"}
  f=$(grep -l "^\[ep" logs/${n}_*.out 2>/dev/null | xargs -r ls -t | head -1)
  if [ -z "$f" ]; then printf "%-11s %-6s %s\n" "$n" "$st" "not started"; continue; fi
  if grep -q "^\[done\]" "$f"; then printf "%-11s %-6s %s\n" "$n" "$st" "DONE: $(grep -h '^\[done\]' "$f" | cut -c8-40)"; continue; fi
  grep -h "^\[ep" logs/${n}_*.out | awk -v n="$n" -v st="$st" -v E="$E" '
    { ep=$2; sub(/\]/,"",ep); ep+=0; s=$NF; sub(/s$/,"",s);
      split($6,m,"/"); if (m[2]+0>best) best=m[2]+0;
      last=ep; t[++k]=s }
    END { a=0; c=0; for(i=k;i>k-5&&i>0;i--){a+=t[i];c++}; a=a/c; left=(E-1-last)*a;
          cmd="date -d \"+" int(left) " seconds\" \"+%a %H:%M\""; cmd | getline fin; close(cmd);
          printf "%-11s %-6s %3d/%-5d %-9.3f %-7.0f %s\n", n, st, last+1, E, best, a, fin }'
done
