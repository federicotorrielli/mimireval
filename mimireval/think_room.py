"""Sensitivity variant for models that write unrequested <think> blocks (Mimir).

Under the original token caps, a reasoning block can be cut before any answer. The variant
`<model>_room` keeps every answer, regenerates the cut ones with a larger budget (same greedy
decoding), and strips the reasoning so that the judge sees only the final answer.

    python -m mimireval.think_room --model mimir
    python -m mimireval.cli fact-score --model mimir_room
"""
import argparse
import re

from mimireval.io import GEN_DIR, read_jsonl, write_jsonl

TASKS = ("simpleqa_verified", "truthfulqa_gen", "precise_wikiqa", "nonexistent_mixed")
BUDGET = 2048
THINK = re.compile(r"^\s*<think>.*?</think>\s*", re.S)


def cut(text: str) -> bool:
    return text.lstrip().startswith("<think>") and "</think>" not in text


def strip(text: str) -> str:
    """Final answer only; an unfinished reasoning block counts as no answer."""
    return "" if cut(text) else THINK.sub("", text, count=1)


def build(model: str, tasks=TASKS):
    gen = None
    for task in tasks:
        out = GEN_DIR / f"{model}_room" / task / "seed0.jsonl"
        if out.exists():
            continue
        rows = read_jsonl(GEN_DIR / model / task / "seed0.jsonl")
        redo = [i for i, r in enumerate(rows) if cut(r["completion"]) and r["max_new_tokens"] < BUDGET]
        if redo:
            from mimireval.generate import Generator
            gen = gen or Generator(model)
            texts = gen.generate([rows[i]["prompt"] for i in redo], max_new_tokens=BUDGET,
                                 temperature=rows[0]["temperature"], top_p=rows[0]["top_p"])
            for i, t in zip(redo, texts):
                rows[i] = {**rows[i], "completion": t, "max_new_tokens": BUDGET}
        n_cut = sum(cut(r["completion"]) for r in rows)
        write_jsonl(out, [{**r, "raw_completion": r["completion"], "completion": strip(r["completion"])} for r in rows])
        print(f"[room] {model}/{task}: regenerated {len(redo)}, still unfinished {n_cut} of {len(rows)}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="mimir")
    build(p.parse_args().model)
