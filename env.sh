# Source before every command: `source env.sh`
export PROJ=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
set -a; . "$PROJ/.env"; set +a
source "$PROJ/.venv/bin/activate"
export PYTHONUNBUFFERED=1
export OPENAI_API_KEY=${OPENAI_API_KEY:-unused}  # safety-eval builds an OpenAI client it never calls here
