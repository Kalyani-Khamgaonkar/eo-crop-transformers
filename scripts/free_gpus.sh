#!/bin/bash
# Free GPUs right now in itc-gpu and main-gpu, fastest type first (Turing skipped: no bf16, ~20x slower).
#   bash scripts/free_gpus.sh
# "Free" means not allocated at this moment; queued jobs of other users may still take them first.
sinfo -h -N -p itc-gpu,main-gpu -O "NodeList:20,Partition:12,StateCompact:12,Gres:45,GresUsed:60" 2>/dev/null | awk '
BEGIN { rank["blackwell"]=1; rank["hopper"]=2; rank["lovelace"]=3; rank["ampere"]=4 }
{
  node=$1; part=$2; st=$3; g=$4; u=$5
  if (st ~ /drain|down|maint|resv|fail/ || seen[node]++) next
  n = split(g, gs, ",")
  for (i = 1; i <= n; i++) {
    split(gs[i], a, ":"); type = a[2]; tot = a[3]; sub(/\(.*/, "", tot)
    if (!(type in rank)) continue
    used = 0
    m = split(u, us, ",")
    for (j = 1; j <= m; j++) { split(us[j], b, ":"); if (b[2] == type) { used = b[3]; sub(/\(.*/, "", used) } }
    free = tot - used
    if (free > 0) { printf "%d %-12s %-9s %-10s free %d of %d\n", rank[type], node, part, type, free, tot; total[type] += free }
  }
}
END { for (t in total) printf "total free %-10s %d\n", t, total[t] > "/dev/stderr" }' | sort -n | cut -d" " -f2-
