# mimireval

Safety and hallucination evals for [DFM-Mimir](https://huggingface.co/danish-foundation-models/DFM-Mimir), compared with HRM-Text 1B, Qwen 3.5 0.8B, Gemma 3 1B and OLMo 2 1B.
Charts are in `report/images/`.

## What's in it

- Safety: the Ai2 safety-eval suite (HarmBench, StrongREJECT, WildGuardTest, WildJailbreak, DAN, TrustLLM, XSTest, ToxiGen, BBQ, WMDP), SORRY-Bench and OR-Bench-Hard-1K. Rated by WildGuard and the SORRY-Bench / StrongREJECT classifiers.
- Hallucination: SimpleQA Verified, TruthfulQA (English and Danish MC, plus free answers), HalluLens (PreciseWikiQA, LongWiki, MixedEntities) and the Vectara summaries with HHEM-2.1-Open. DeepSeek-V4.1-Flash rates the free answers, and GLM-5.3-Flash rates them again to check agreement.
- Bias: CrowS-Pairs and BBQ.

Mimir and HRM-Text are prefix LMs (the prompt is bidirectional). If vLLM preempts a request it recomputes prompt and answer together, which breaks these models. So `mimireval/generate.py` only starts a request when its whole length fits in the KV cache, and it crashes if a preemption still happens.

## Setup

You need a CUDA GPU (we used a B200), [uv](https://docs.astral.sh/uv/) and ~30 GB of disk for the LongWiki Wikipedia dump.

```bash
cp .env.example .env    # rater API endpoint + HF token
bash scripts/setup.sh
```

Any OpenAI-compatible API works for the raters, model names are in `mimireval/models.py`.

## Evaluating a new Mimir

Add it to `MODELS` in `mimireval/models.py`:

```python
ModelSpec("mimir_v2", "danish-foundation-models/DFM-Mimir-v2", "<commit sha>", prefix_lm=True, label="Mimir v2"),
```

Then run it and redraw the charts:

```bash
bash scripts/evaluate_model.sh --model mimir_v2   # resumable
source env.sh && python -m mimireval.report
```

Scores of the five models above are already in `results/`, so only the new model runs. The HalluLens questions are fixed in `prompts/hallulens` so every model gets the same ones. `scripts/run_all.sh` reruns everything.

## Differences from the original setups

- TruthfulQA's and HalluLens' original raters are no longer available or only run locally, so an API model replaces them (one model for all HalluLens roles, like the original).
- HalluLens runs once instead of three times, without GeneratedEntities.
- LongWiki uses a public copy of FActScore's `enwiki-20230401.db` (`danielli2003/enwiki-20230401.db`) because the original link is rate-limited.
- `mimireval/tasks/hallulens.py` patches HalluLens at run time (faster title search and cache writes, pandas 3, CPU fallback). The protocol doesn't change.

## Contamination

Mimir's training data includes `allenai/tulu-v2-sft-mixture`, which has 120 TruthfulQA questions word for word (mostly in ShareGPT chats). Scores on those questions are not higher than on the rest. It also includes the Flan NIV2 CrowS-Pairs tasks, so the CrowS-Pairs number is not clean.
