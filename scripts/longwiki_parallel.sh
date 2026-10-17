#!/bin/bash
# LongWiki for every model in parallel (HalluLens judges claims in batches of 100, so one process is slow).
#   bash scripts/longwiki_parallel.sh [--judge zai-org/GLM-5.3-Flash]
cd "$(dirname "$0")/.." && source env.sh && mkdir -p logs
export MIMIREVAL_PREFETCH_WORKERS=${MIMIREVAL_PREFETCH_WORKERS:-16}  # x5 processes: the shared API rate-limits above ~150 open calls
for m in mimir hrm_text qwen35_08b gemma3_1b olmo2_1b; do
  python -m mimireval.cli fact-score --model "$m" --task longwiki "$@" > "logs/longwiki_${m}${1:+_second}.log" 2>&1 &
done
wait
echo LONGWIKI_DONE
