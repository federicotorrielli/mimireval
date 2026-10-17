"""Result paths and JSONL helpers shared by all runners."""
import json
import os
from pathlib import Path

PROJ = Path(os.environ.get("PROJ", "."))
GEN_DIR = PROJ / "results" / "generations"
SCORE_DIR = PROJ / "results" / "scores"


def write_jsonl(path: Path, records: list[dict]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False, default=str) + "\n")


def read_jsonl(path: Path) -> list[dict]:
    with path.open() as f:
        return [json.loads(line) for line in f]
