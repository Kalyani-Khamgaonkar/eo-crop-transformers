#!/bin/bash
# sbatch options for the K-th fastest GPU that is free right now (K = 1, 2, ...).
#   sbatch $(bash scripts/gpu_args.sh 1) --job-name=a slurm/train.sh ...
# Free GPUs are counted fastest first: Blackwell (itc-gpu) > Lovelace > Ampere (main-gpu).
# If fewer than K are free: any Lovelace or Ampere GPU (waits in the queue).
k=${1:-1}
pick=$(bash "$(dirname "$0")/free_gpus.sh" 2>/dev/null | awk '{ for (i = 0; i < $5; i++) print $3 }' | sed -n "${k}p")
case "$pick" in
  blackwell) echo "--partition=itc-gpu --gres=gpu:1" ;;
  lovelace)  echo "--partition=main-gpu --gres=gpu:lovelace:1" ;;
  ampere)    echo "--partition=main-gpu --gres=gpu:ampere:1" ;;
  *)         echo "--partition=itc-gpu,main-gpu --gres=gpu:1 --exclude=ctit084,ctit085,ctit088" ;;
esac
