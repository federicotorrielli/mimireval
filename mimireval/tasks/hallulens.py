"""HalluLens (Bang et al., 2025) with our generator and the SDU judge.

HalluLens hard-codes Llama-3.1 models as question generator, judge, claim extractor and verifier.
We keep its code and prompts, and route every such call to one judge model (as in the original,
where one model plays all roles). Gated Llama tokenizers, used only to count tokens, are
redirected to an ungated copy with the same vocabulary.
"""
import json
import os
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

PROJ = Path(os.environ.get("PROJ", "."))
HL_ROOT = PROJ / "third_party" / "HalluLens"
LLAMA_TOKENIZER = "unsloth/Llama-3.1-8B-Instruct"  # same tokenizer as Llama-3.1-70B/405B, not gated

PRECISE = dict(task="precise_wikiqa_goodwiki_dynamic", N=5000, max_new_tokens=256)
LONGWIKI = dict(task="longwiki-longwiki", N=250, max_new_tokens=1024)
MIXED = dict(task="nonsense_all_0_2000", N=2000, seed=0, max_new_tokens=256)


_BOOTED: dict = {}


def bootstrap(judge_model: str | None = None):
    """Make HalluLens importable and point its LLM layer at our judge (once per process)."""
    from transformers import AutoTokenizer
    from mimireval.judge import PRIMARY_JUDGE, Judge

    judge_model = judge_model or PRIMARY_JUDGE
    if _BOOTED:
        assert judge_model in _BOOTED, "one judge per process"
        return _BOOTED[judge_model]

    os.chdir(HL_ROOT)
    if sys.path[0] != str(HL_ROOT):
        sys.path.insert(0, str(HL_ROOT))
    for name in [m for m in sys.modules if m == "utils" or m.startswith("utils.")]:
        if not (getattr(sys.modules[name], "__file__", None) or "").startswith(str(HL_ROOT)):
            del sys.modules[name]  # another project's top-level `utils`

    original = AutoTokenizer.from_pretrained.__func__
    def from_pretrained(cls, name, *args, **kwargs):
        return original(cls, LLAMA_TOKENIZER if str(name).startswith("meta-llama/") else name, *args, **kwargs)
    AutoTokenizer.from_pretrained = classmethod(from_pretrained)

    from utils import lm
    judge = _BOOTED[judge_model] = Judge(judge_model)
    inflight, lock = {}, threading.Lock()
    pool = ThreadPoolExecutor(int(os.environ.get("MIMIREVAL_PREFETCH_WORKERS", 256)))

    def prefetch(prompt):
        """Start a greedy judge call now; LongWiki asks for it later, in slow batches of 100."""
        with lock:
            if prompt not in inflight:
                inflight[prompt] = pool.submit(judge.one, prompt)

    def generate(prompt, model=None, temperature=0.0, top_p=1.0, max_tokens=512, port=None, i=0):
        with lock:
            future = inflight.get(prompt)
        if future is not None and (temperature, top_p) == (0.0, 1.0):
            return future.result()
        return judge.one(prompt, temperature=temperature, top_p=top_p)
    lm.generate = lm.call_vllm_api = lm.custom_api = generate
    lm.api_generate = generate  # referenced by longwiki_utils.jsonify_ans but missing upstream

    # Upstream thread pools are sized for a local vLLM server (16 for LongWiki); the API takes more.
    import importlib
    from tqdm.contrib import concurrent
    workers = int(os.environ.get("MIMIREVAL_JUDGE_WORKERS", 128))
    def thread_map(fn, *iterables, max_workers=None, **kwargs):
        return concurrent.thread_map(fn, *iterables, max_workers=workers, **kwargs)
    for name in ("tasks.longwiki.longwiki_utils", "tasks.shortform.precise_wikiqa",
                 "tasks.refusal_test.nonsense_mixed_entities", "utils.generate_question"):
        importlib.import_module(name).thread_map = thread_map

    # LongWiki passes clean_up_tokenization_spaces to encode(), which sentence-transformers 5 rejects.
    from sentence_transformers import SentenceTransformer
    encode = SentenceTransformer.encode
    SentenceTransformer.encode = lambda self, *args, clean_up_tokenization_spaces=None, **kwargs: encode(self, *args, **kwargs)

    # LongWiki matches entities to titles with LIKE '%entity%', a 10 s table scan per entity;
    # the same case-insensitive substring match over titles held in memory takes a fraction of it.
    import tasks.longwiki.longwiki_retrieval as lr
    from functools import cache

    @cache
    def lowered_titles(path):
        import sqlite3
        rows = sqlite3.connect(f"file:{path}?mode=ro", uri=True).execute("SELECT title_name FROM titles")
        titles = [r[0] for r in rows]
        return titles, [t.lower() for t in titles]

    @cache
    def relevant_titles(path, entity):
        titles, lowered = lowered_titles(path)
        needle = entity.replace("'", "''").lower()  # upstream's quote doubling, kept
        return [t for t, low in zip(titles, lowered) if needle in low]
    lr.LongWikiDB.get_relevant_titles = lambda self, entity: list(relevant_titles(self.title_db_path, entity))

    # Upstream adds F1 with groupby("prompt").apply, which drops the "prompt" column in pandas 3.
    # Precision and recall are constant within a prompt, so the same F1 is computed per row.
    import tasks.longwiki.longwiki_utils as lu

    def calculate_all_metrics(df, k=32):
        by_prompt = df.groupby("prompt").is_supported
        df["precision"] = by_prompt.transform("mean").astype(float)
        df["recall"] = by_prompt.transform(lambda g: min(g.sum() / k, 1)).astype(float)
        total = df.precision + df.recall
        df["f1"] = (2 * df.precision * df.recall / total.where(total != 0)).where(total != 0, 0.0)  # NaN stays NaN
        df["k"] = k
        df["n_claims"] = df.groupby("prompt").is_supported.transform("count")
        lu.print_all_metrics(df, k=k)
        return df
    lu.calculate_all_metrics = calculate_all_metrics

    # Upstream rereads and rewrites the whole embedding pickle after every new page, which grows
    # quadratically; the embeddings stay in memory for the run and are cheap to recompute.
    lr.LongWikiRetrieval.save_cache = lambda self: None

    # LongWiki's retrieval encoders are pinned to the GPU (device=0, .cuda()); use the CPU if there is none.
    import torch
    if not torch.cuda.is_available():
        init = SentenceTransformer.__init__
        SentenceTransformer.__init__ = lambda self, *args, device=None, **kwargs: init(self, *args, device="cpu", **kwargs)
        SentenceTransformer.cuda = lambda self, *args, **kwargs: self

    # LongWiki asserts that a claim-extraction prompt fits 3,500 tokens (the original judge's context).
    # A single run-on "sentence" can exceed it even with the shortest template; our judge takes 1M
    # tokens, so such generations are rebuilt without the limit and counted.
    import tasks.longwiki.facthalu as fh
    original_make, real_encoding = fh.make_claim_extraction_prompts, fh.encoding

    class NoLimit:
        def encode(self, text, *args, **kwargs):
            return real_encoding.encode(text, *args, **kwargs)[:3500]

    def build(generation, *args, **kwargs):
        try:
            return original_make(generation, *args, **kwargs)
        except AssertionError:
            LONG_SENTENCE_FALLBACKS.append(generation.prompt)
            fh.encoding = NoLimit()
            try:
                return original_make(generation, *args, **kwargs)
            finally:
                fh.encoding = real_encoding

    def make_claim_extraction_prompts(generation, *args, **kwargs):
        sentences = build(generation, *args, **kwargs)
        for s in sentences:
            prefetch(s.prompt)
        return sentences
    fh.make_claim_extraction_prompts = make_claim_extraction_prompts

    # Verification prompts are all formatted before the batched loop: prefetch them as they are built.
    import tasks.longwiki.prompt_templates as templates

    class Prefetching(str):
        def format(self, *args, **kwargs):
            text = str.format(self, *args, **kwargs)
            prefetch(text)
            return text
    templates.VERIFICATION_TEMPLATE_W_REFERENCE_RETRIEVAL = Prefetching(templates.VERIFICATION_TEMPLATE_W_REFERENCE_RETRIEVAL)
    return judge


LONG_SENTENCE_FALLBACKS: list[str] = []


# Prompt sets (generated once, shared by all models)

def precise_questions() -> list[dict]:
    import jsonlines
    import utils.generate_question as qa

    path = HL_ROOT / "data/precise_qa/save/qa_goodwiki_shared_dynamic.jsonl"
    if not path.exists():
        qa.precise_QA_generation_run_batch(
            wiki_input_path=str(HL_ROOT / "data/wiki_data/doc_goodwiki_h_score.jsonl"),
            N=PRECISE["N"], q_generator="meta-llama/Llama-3.1-70B-Instruct", output_path=str(path))
    return list(jsonlines.open(path))[:PRECISE["N"]]


def longwiki_questions() -> list[dict]:
    """HalluLens' per-level loop (50 questions for each h_score level 5..9), seeded and retried
    until every level is full: upstream asserts instead when answerability filtering drops a few."""
    import random
    import jsonlines
    import pandas as pd
    import utils.generate_question as qa

    path = HL_ROOT / "data/longwiki/save/longwiki_shared.jsonl"
    if path.exists():
        rows = list(jsonlines.open(path))
        if len(rows) == LONGWIKI["N"]:
            return rows
    random.seed(0)
    gen = qa.WikiQA("meta-llama/Llama-3.1-70B-Instruct", task="longform")
    wiki = pd.read_json(HL_ROOT / "data/wiki_data/doc_goodwiki_h_score.jsonl", lines=True)
    missing = (HL_ROOT / "data/wiki_data/doc_goodwiki_not_exist_titles.txt").read_text().split("\n")
    wiki = wiki[~wiki.title.isin(missing)]
    per_level, rows = LONGWIKI["N"] // 5, []
    for level in range(5, 10):
        pool = wiki[wiki.h_score_cat == level].sample(frac=1, random_state=level).to_dict(orient="records")
        got, tmp = [], path.with_suffix(f".level{level}.jsonl")
        tmp.unlink(missing_ok=True)
        while len(got) < per_level and pool:
            batch, pool = pool[:per_level - len(got) + 5], pool[per_level - len(got) + 5:]
            got += gen.per_bin_generation_batch(batch, str(tmp), per_level - len(got))
        assert len(got) == per_level, f"level {level}: only {len(got)} answerable questions"
        rows += got
        tmp.unlink()
    with jsonlines.open(path, "w") as f:
        f.write_all(rows)
    return rows


def mixed_prompts() -> list[dict]:
    import pandas as pd
    from tasks.refusal_test.entities_generation import NonsenseMixedGeneration

    path = HL_ROOT / f"data/nonsense_all/save/shared/{MIXED['task']}.csv"
    if not path.exists():
        gen = NonsenseMixedGeneration(MIXED["seed"], MIXED["N"], "nonsense_all")
        gen.save_prompt_csv(gen.generate_prompts(), str(path))
    return pd.read_csv(path).to_dict(orient="records")


SETS = {"precise_wikiqa": (PRECISE, precise_questions),
        "longwiki": (LONGWIKI, longwiki_questions),
        "nonexistent_mixed": (MIXED, mixed_prompts)}


def generation_path(subtask: str, model: str) -> Path:
    base = "output/refusal_test" if subtask == "nonexistent_mixed" else "output"
    return HL_ROOT / base / SETS[subtask][0]["task"] / model / "generation.jsonl"


def write_generations(subtask: str, model: str, rows: list[dict], generations: list[str]):
    path = generation_path(subtask, model)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for r, g in zip(rows, generations):
            f.write(json.dumps({**r, "generation": g}, ensure_ascii=False, default=str) + "\n")


# Evaluation (HalluLens code, unchanged)

def evaluate(subtask: str, model: str) -> dict:
    if subtask == "precise_wikiqa":
        from tasks.shortform.precise_wikiqa import PreciseQAEval
        import tasks.shortform.precise_wikiqa as mod
        mod.TASKNAME = PRECISE["task"]  # read as a module global inside run_eval

        class Eval(PreciseQAEval):
            def process_res(self, abstention_raw, halu_raw):
                # Upstream re-parses the raw abstention replies with a strict json.loads and drops the
                # robust parse that eval_abstention already stored in test_df["refusal"]; use the latter.
                _, halu = super().process_res(['{"is_abstaining": false}'] * len(abstention_raw), halu_raw)
                return list(self.test_df["refusal"]), halu

        Eval(model_path=model, TASKNAME=PRECISE["task"]).run_eval()
        res = json.load(open(HL_ROOT / f"output/{PRECISE['task']}/{model}/eval_results.json"))
        return {k: res[k] for k in ("halu_Rate", "refusal_rate", "correct_rate")}

    if subtask == "nonexistent_mixed":
        from tasks.refusal_test.nonsense_mixed_entities import NonsenseMixedEval
        ev = NonsenseMixedEval(MIXED["task"], "output/refusal_test", model, prompt_path="")
        res = ev.run_eval(overwrite=False)
        return {k: res[k] for k in ("false_acceptance_rate", "refusal_rate")}

    if subtask == "longwiki":
        import argparse
        import pandas as pd
        from tasks.longwiki.facthalu import FactHalu
        out = HL_ROOT / f"output/{LONGWIKI['task']}/{model}"
        llama = "meta-llama/Llama-3.1-405B-Instruct-FP8"  # role names checked by HalluLens; routed to our judge
        n_fallback = len(LONG_SENTENCE_FALLBACKS)
        FactHalu(out / "generation.jsonl", out / "output.csv", abstain_evaluator=llama,
                 claim_extractor=llama, verifier=llama, k=32,
                 eval_cache_path=str(HL_ROOT / "data/longwiki/.cache" / model),  # one per run: safe in parallel
                 db_path=str(HL_ROOT / "data/wiki_data/.cache/enwiki-20230401.db"),
                 args=argparse.Namespace(do_extract_only=False)).run()
        df = pd.read_csv(out / "output.csv")
        n_gen = sum(1 for _ in open(out / "generation.jsonl"))
        return {"precision": df.overall_precision.iloc[0], "recall@32": df.overall_recall.iloc[0],
                "f1@32": df.overall_f1.iloc[0], "abstain_rate": 1 - df.prompt.nunique() / n_gen,
                "long_sentence_fallbacks": len(LONG_SENTENCE_FALLBACKS) - n_fallback}

    raise ValueError(subtask)
