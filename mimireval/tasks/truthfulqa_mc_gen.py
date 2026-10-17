"""TruthfulQA MC1 as a generative letter choice (the Mimir authors' MCQ style).

Options are shuffled with a fixed seed so the correct answer is not always 'A'.
"""
import random
import re
import string

from datasets import load_dataset

from mimireval.tasks.truthfulqa_gen import DATASET, REVISION

GEN = dict(max_new_tokens=8, temperature=0.0, top_p=1.0)
TEMPLATE = "{question}\n\n{options}\n\nAnswer with the letter of the correct option only."


def load() -> list[dict]:
    ds = load_dataset(DATASET, "multiple_choice", revision=REVISION, split="validation")
    rng = random.Random(4242)
    rows = []
    for i, r in enumerate(ds):
        choices, labels = r["mc1_targets"]["choices"], r["mc1_targets"]["labels"]
        order = list(range(len(choices)))
        rng.shuffle(order)
        letters = string.ascii_uppercase[:len(order)]
        options = "\n".join(f"{l}. {choices[j]}" for l, j in zip(letters, order))
        gold = letters[[labels[j] for j in order].index(1)]
        rows.append({"id": f"tqa_mc1_gen/{i}", "question": r["question"], "gold": gold, "n_options": len(order),
                     "prompt": TEMPLATE.format(question=r["question"], options=options)})
    return rows


def parse(text: str, n_options: int) -> str | None:
    m = re.search(rf"\b([A-{string.ascii_uppercase[n_options - 1]}])\b", text.strip())
    return m.group(1) if m else None


def metrics(rows: list[dict], generations: list[str]) -> tuple[dict, list[dict]]:
    preds = [parse(g, r["n_options"]) for r, g in zip(rows, generations)]
    items = [{"pred": p, "gold": r["gold"], "correct": p == r["gold"]} for p, r in zip(preds, rows)]
    n = len(items)
    chance = sum(1 / r["n_options"] for r in rows) / n
    return {"acc": sum(i["correct"] for i in items) / n, "parse_rate": sum(p is not None for p in preds) / n,
            "chance": chance}, items
