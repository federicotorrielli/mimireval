"""lm-evaluation-harness tasks (log-likelihood scoring). Run via `mimireval.cli lmeval`."""
import json
import os
from pathlib import Path

import lm_eval
from lm_eval.models.huggingface import HFLM

from mimireval.lmeval_prefixlm import PrefixLMHF
from mimireval.models import MODELS

OUT = Path(os.environ.get("PROJ", ".")) / "results" / "lmeval"
CHAT_TASKS = ["truthfulqa_mc1", "truthfulqa_mc2", "truthfulqa_da_mc1", "truthfulqa_da_mc2"]
RAW_TASKS = ["crows_pairs_english"]  # sentence likelihood: no chat template


def _out(key: str, chat: bool) -> Path:
    return OUT / key / ("chat.json" if chat else "raw.json")


def run(key: str):
    spec = MODELS[key]
    todo = [(t, c) for t, c in [(CHAT_TASKS, True), (RAW_TASKS, False)] if not _out(key, c).exists()]
    if not todo:
        return
    cls = PrefixLMHF if spec.prefix_lm else HFLM
    lm = cls(pretrained=spec.repo, revision=spec.revision, dtype="bfloat16", batch_size=32,
             max_length=spec.max_model_len, attn_implementation="sdpa")
    if spec.raw_format:  # e.g. HRM-Text: same raw format as generation, as a template for lm-eval
        pre, post = spec.raw_format.split("{prompt}")
        lm.tokenizer.chat_template = ("{% for m in messages %}" + pre + "{{ m['content'] }}" + post + "{% endfor %}")
    for tasks, chat in todo:
        res = lm_eval.simple_evaluate(model=lm, tasks=tasks, apply_chat_template=chat,
                                      log_samples=False, random_seed=4242)
        out = _out(key, chat)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps({"results": res["results"], "config": {"chat_template": chat, "revision": spec.revision}},
                                  indent=1, default=str))
        print(key, json.dumps(res["results"], default=str)[:600])

