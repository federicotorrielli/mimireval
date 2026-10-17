#!/bin/bash
# Full mimireval run for one model (resumable: finished files are skipped).
#   bash scripts/evaluate_model.sh --model mimir
#   bash scripts/evaluate_model.sh --repo danish-foundation-models/DFM-Mimir-v2 --revision <sha> --key mimir_v2
set -euo pipefail
cd "$(dirname "$0")/.." && source env.sh

for step in safety-generate fact-generate lmeval safety-score fact-score; do
  python -m mimireval.cli $step "$@"
done
python -m mimireval.cli fact-score "$@" --judge zai-org/GLM-5.3-Flash  # second judge, for agreement
