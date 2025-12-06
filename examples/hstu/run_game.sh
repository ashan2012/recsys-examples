#!/bin/bash
set -euo pipefail

export CUDA_VISIBLE_DEVICES=4,5,6
export NCCL_TIMEOUT=3600
export NCCL_ASYNC_ERROR_HANDLING=1

LOG_DIR=./logs
mkdir -p "${LOG_DIR}"

start_ts=$(date +%s)
start_str=$(date '+%F %T')
echo "Training start: ${start_str}"

PYTHONPATH=${PYTHONPATH}:$(realpath ../) torchrun --nproc_per_node 3 --nnodes 1 --node_rank 0 --master_addr localhost --master_port 6000 ./training/pretrain_gr_retrieval.py --gin-config-file ./training/configs/gameid_retrieval.gin 

end_ts=$(date +%s)
end_str=$(date '+%F %T')
duration=$((end_ts - start_ts))
echo "Training end:   ${end_str}"
echo "Elapsed:        ${duration}s"

# If your training log prints 'tokens=XXXX' each step, you can summarize throughput:
total_tokens=$(grep -ho 'tokens=[0-9]\+' "${LOG_DIR}"/run_game.*.log | sed 's/tokens=//' | tail -1)
if [[ -n "${total_tokens}" ]]; then
  echo "Average tokens/sec: $(( total_tokens / duration ))"
else
  echo "No token count found in log; please add token reporting in training loop."
fi
