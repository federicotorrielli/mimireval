"""Ai2 safety-eval suite with our own generator.

Phase 1 (`generate`): capture each task's exact prompts and hparams, generate with vLLM, cache.
Phase 2 (`score`): replay cached completions through the unmodified task code and its judges.
"""
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

from mimireval.io import GEN_DIR, PROJ, SCORE_DIR, read_jsonl, write_jsonl

SAFETY_ROOT = PROJ / "third_party" / "safety-eval"
sys.path[:0] = [str(SAFETY_ROOT), str(SAFETY_ROOT / "evaluation")]

# our name -> safety-eval task spec ("task" or "task:config")
TASKS = {
    "harmbench": "harmbench",
    "strongreject": "strongreject:strongreject_cls",
    "wildguardtest": "wildguardtest",
    "wildjailbreak_harmful": "wildjailbreak:harmful",
    "wildjailbreak_benign": "wildjailbreak:benign",
    "dan": "do_anything_now",
    "trustllm_jailbreak": "trustllm_jailbreaktrigger",
    "xstest": "xstest",
    "toxigen": "toxigen",
    "bbq": "bbq",
    "wmdp": "wmdp",
    "sorrybench": "custom:sorrybench",
    "orbench_hard": "custom:orbench_hard",
}
SAFETY_EVAL_HPARAMS = dict(max_new_tokens=2048, temperature=0.7, top_p=0.95)
SORRY_REPO, SORRY_REV = "sorry-bench/sorry-bench-202503", "612a4e1f45db8adf884fa62318ddf9fa1c6e75e9"
ORBENCH_REPO, ORBENCH_REV = "bench-llm/or-bench", "e36d8b80e81837c8a8f264bbb2a49f1b32c7e272"


def _custom_task(name: str):
    """Benchmarks outside safety-eval, built on its adversarial task base (per-`source` rates included)."""
    from evaluation.tasks.generation.base import AdversarialTaskBase

    class SorryBench(AdversarialTaskBase):
        """SORRY-Bench 2025-03 base set: 440 prompts, 44 categories; success = compliance."""
        def load(self):
            from huggingface_hub import hf_hub_download
            get = lambda f: hf_hub_download(SORRY_REPO, f, repo_type="dataset", revision=SORRY_REV)
            meta = {}
            exec(open(get("meta_info.py")).read(), meta)
            rows = [json.loads(line) for line in open(get("question.jsonl"))]
            return [{"instruction": r["turns"][0], "source": meta["category_descriptions"][int(r["category"]) - 1],
                     "question_id": r["question_id"]} for r in rows]

    class ORBenchHard(AdversarialTaskBase):
        """OR-Bench-Hard-1K: seemingly toxic but benign prompts; reported rate = compliance."""
        def load(self):
            from datasets import load_dataset
            ds = load_dataset(ORBENCH_REPO, "or-bench-hard-1k", split="train", revision=ORBENCH_REV)
            return [{"instruction": r["prompt"], "source": r["category"]} for r in ds]

    if name == "sorrybench":  # SORRY-Bench's own generation settings
        return SorryBench(classifier_model_name="sorrybench",
                          hparams=dict(max_new_tokens=1024, temperature=0.7, top_p=1.0))
    if name == "orbench_hard":
        return ORBenchHard(classifier_model_name="WildGuard", hparams=SAFETY_EVAL_HPARAMS, benign=True)
    raise ValueError(name)
SEEDS = (0, 1, 2)  # OLMo 3 reports the mean of three runs


class _Captured(Exception):
    pass


class Recorder:
    """Pretends to be a generator, captures the inputs, then aborts the task."""
    model_input_template = "{instruction}"

    def generate_completions(self, inputs, max_new_tokens, temperature, top_p, **_):
        self.prompts = [d["instruction"] for d in inputs]
        self.hparams = dict(max_new_tokens=max_new_tokens, temperature=temperature, top_p=top_p)
        raise _Captured


class Replay:
    """Returns cached completions, matched by prompt text (task loaders may shuffle)."""
    model_input_template = "{instruction}"

    def __init__(self, records: list[dict]):
        self.by_prompt = defaultdict(list)
        for r in records:
            self.by_prompt[r["prompt"]].append(r["completion"])

    def generate_completions(self, inputs, **_):
        pools = {k: list(v) for k, v in self.by_prompt.items()}
        return [pools[d["instruction"]].pop(0) for d in inputs]


def load_task(spec: str):
    if spec.startswith("custom:"):
        return _custom_task(spec.split(":", 1)[1])
    from tasks import EvalMode, load_evaluation_tasks
    return load_evaluation_tasks(EvalMode.GENERATION, [spec])[0]


def capture(spec: str) -> tuple[list[str], dict]:
    rec = Recorder()
    try:
        load_task(spec)._evaluate(rec)
    except _Captured:
        return rec.prompts, rec.hparams
    raise RuntimeError(f"{spec} never called generate_completions")


def gen_path(model: str, task: str, seed: int) -> Path:
    return GEN_DIR / model / task / f"seed{seed}.jsonl"


def generate(model: str, tasks: list[str], seeds=SEEDS):
    from mimireval.generate import Generator

    os.chdir(SAFETY_ROOT)  # some loaders download data next to their module
    gen = Generator(model)
    failed = []
    for task in tasks:
        try:
            prompts, hp = capture(TASKS[task])
        except Exception as e:  # e.g. a gated dataset: keep going with the other tasks
            print(f"[gen] SKIPPED {model}/{task}: {type(e).__name__}: {e}")
            failed.append(task)
            continue
        for seed in seeds:
            out = gen_path(model, task, seed)
            if out.exists():
                continue
            completions = gen.generate(prompts, seed=seed, **hp)
            write_jsonl(out, [{"prompt": p, "completion": c, **hp, "seed": seed}
                              for p, c in zip(prompts, completions)])
            print(f"[gen] {model}/{task}/seed{seed}: {len(prompts)} prompts")
    if failed:
        raise RuntimeError(f"{model}: tasks not generated: {failed}")


class SlowTokenizedLLM:
    """vLLM engine fed by a slow (SentencePiece) tokenizer: vLLM >= 0.2x dropped tokenizer_mode="slow",
    which safety-eval uses for its Llama-based judges."""

    def __init__(self, llm, path: str):
        from transformers import AutoTokenizer
        self.llm, self.tok = llm, AutoTokenizer.from_pretrained(path, use_fast=False)

    def generate(self, prompts, sampling_params=None, use_tqdm=True, **kwargs):
        from vllm import TokensPrompt
        ids = [TokensPrompt(prompt_token_ids=self.tok(p)["input_ids"]) for p in prompts]
        return self.llm.generate(ids, sampling_params, use_tqdm=use_tqdm, **kwargs)


def _build_judge(name: str, load_classifier_model):
    """Instantiate a safety-eval vLLM judge in-process, sharing the GPU with the other judges."""
    from vllm import LLM
    from evaluation.utils import maybe_merge_lora_weights_and_save
    from src.classifier_models.base import SafetyClassifierBase
    from src.classifier_models.strong_reject_classifier import STRONGREJECT_INPUT_FORMAT, StrongRejectClassifier

    if name == "StrongRejectClassifier":  # its __init__ starts an engine with default settings
        clf = StrongRejectClassifier.__new__(StrongRejectClassifier)
        SafetyClassifierBase.__init__(clf, -1)
        clf.input_format = STRONGREJECT_INPUT_FORMAT
        path, mode = maybe_merge_lora_weights_and_save(clf.MODEL_NAME), "slow"
    else:
        clf = load_classifier_model(name, ephemeral_model=True)
        path, mode = clf.MODEL_NAME, "slow" if name == "WildGuard" else "auto"
    from transformers import AutoConfig
    max_len = min(int(os.environ.get("MIMIREVAL_JUDGE_MAXLEN", 16384)),  # prompt + <=2048-token answer
                  AutoConfig.from_pretrained(path).max_position_embeddings)
    llm = LLM(model=path, gpu_memory_utilization=float(os.environ.get("MIMIREVAL_JUDGE_MEM", 0.28)),
              max_model_len=max_len,
              attention_backend="FLASH_ATTN")
    clf.model = SlowTokenizedLLM(llm, path) if mode == "slow" else llm
    return clf


def _install_persistent_judges():
    """Route every `load_classifier_model` reference to a per-process judge cache."""
    import src.classifier_models.loader as loader

    original, cache = loader.load_classifier_model, {}

    def load(name, **kwargs):
        if name not in cache:
            cache[name] = _build_judge(name, original)
        return cache[name]

    def patch_all():
        for mod in list(sys.modules.values()):
            if getattr(mod, "load_classifier_model", None) is original:
                mod.load_classifier_model = load

    return patch_all


def score(models: list[str], tasks: list[str], seeds=SEEDS):
    os.chdir(SAFETY_ROOT)
    patch_judges = _install_persistent_judges()
    for task in tasks:
        for model in models:
            for seed in seeds:
                out = SCORE_DIR / model / task / f"seed{seed}.json"
                if out.exists() or not gen_path(model, task, seed).exists():
                    continue
                task_obj = load_task(TASKS[task])
                patch_judges()  # task modules import the loader by name
                report, individual = task_obj.evaluate(Replay(read_jsonl(gen_path(model, task, seed))))
                out.parent.mkdir(parents=True, exist_ok=True)
                out.write_text(json.dumps({"report": report}, indent=1, default=str))
                write_jsonl(out.with_suffix(".items.jsonl"), [json.loads(json.dumps(r, default=str)) for r in individual])
                print(f"[score] {model}/{task}/seed{seed}: {json.dumps(report, default=str)[:300]}")
