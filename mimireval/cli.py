"""mimireval command line.

    python -m mimireval.cli safety-generate --model mimir
    python -m mimireval.cli safety-score --model mimir --model qwen35_08b
    python -m mimireval.cli lmeval --model mimir
    python -m mimireval.cli fact-prepare
    python -m mimireval.cli fact-generate --model mimir
    python -m mimireval.cli fact-score --judge zai-org/GLM-5.3-Flash
    python -m mimireval.cli safety-generate --repo danish-foundation-models/DFM-Mimir-v2 --revision <sha> --key mimir_v2
"""
import argparse

from mimireval import models


def main():
    p = argparse.ArgumentParser(prog="mimireval")
    p.add_argument("command", choices=["safety-generate", "safety-score", "fact-prepare", "fact-generate", "fact-score", "lmeval"])
    p.add_argument("--model", action="append", default=[], help="registry key (repeatable)")
    p.add_argument("--repo", help="evaluate an unregistered model (e.g. a new Mimir version)")
    p.add_argument("--revision", default="main")
    p.add_argument("--key", help="name for --repo in the results tree")
    p.add_argument("--causal", action="store_true", help="--repo is a plain causal LM (default: PrefixLM like Mimir)")
    p.add_argument("--task", action="append", default=[], help="task subset (repeatable)")
    p.add_argument("--seeds", default="0,1,2")
    p.add_argument("--judge", help="factuality judge model (default: primary judge)")
    a = p.parse_args()

    if a.repo:
        a.model.append(models.register(a.key or a.repo.split("/")[-1], a.repo, a.revision, prefix_lm=not a.causal).key)
    seeds = tuple(int(s) for s in a.seeds.split(","))

    if a.command == "lmeval":
        from mimireval import lmeval_run
        for m in a.model:
            lmeval_run.run(m)
        return

    if a.command.startswith("fact-"):
        from mimireval import factuality
        tasks = a.task or list(factuality.TASKS)
        if a.command == "fact-prepare":
            factuality.prepare(tasks)
        elif a.command == "fact-generate":
            for m in a.model:
                factuality.generate(m, tasks)
        else:
            factuality.score(a.model or list(models.MODELS), tasks, a.judge)
        return

    from mimireval import safety
    tasks = a.task or list(safety.TASKS)
    if a.command == "safety-generate":
        for m in a.model:
            safety.generate(m, tasks, seeds)
    else:
        safety.score(a.model or list(models.MODELS), tasks, seeds)


if __name__ == "__main__":
    main()
