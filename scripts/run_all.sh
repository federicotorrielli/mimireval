#!/bin/bash
# Every phase for every registered model, in GPU-friendly order (resumable).
cd "$(dirname "$0")/.." && source env.sh
MODELS=(mimir hrm_text qwen35_08b gemma3_1b olmo2_1b)
args=(); for m in "${MODELS[@]}"; do args+=(--model "$m"); done

python -m mimireval.cli fact-prepare || echo "FAILED fact-prepare"
for m in "${MODELS[@]}"; do python -m mimireval.cli safety-generate --model "$m" || echo "FAILED safety-generate $m"; done
for m in "${MODELS[@]}"; do python -m mimireval.cli fact-generate --model "$m" || echo "FAILED fact-generate $m"; done
for m in "${MODELS[@]}"; do python -m mimireval.cli lmeval --model "$m" || echo "FAILED lmeval $m"; done
python -m mimireval.cli safety-score "${args[@]}" || echo "FAILED safety-score"
python -m mimireval.cli fact-score "${args[@]}" || echo "FAILED fact-score"
python -m mimireval.cli fact-score "${args[@]}" --judge zai-org/GLM-5.3-Flash || echo "FAILED fact-score GLM"
echo RUN_ALL_DONE
