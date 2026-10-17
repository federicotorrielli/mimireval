# mimireval

Safety and hallucination evaluation of [DFM-Mimir](https://huggingface.co/danish-foundation-models/DFM-Mimir), next to four open models of about the same size (HRM-Text 1B, Qwen 3.5 0.8B, Gemma 3 1B, OLMo 2 1B).
The report is in `report/index.html`, and the single charts are in `report/images/`.

## Benchmarks

| Area | Benchmarks | Evaluated with |
|---|---|---|
| Safety | Ai2 safety-eval (HarmBench, StrongREJECT, WildGuardTest, WildJailbreak, Do Anything Now, TrustLLM, XSTest, ToxiGen, BBQ, WMDP), SORRY-Bench, OR-Bench-Hard-1K | open classifiers (WildGuard, the SORRY-Bench and StrongREJECT classifiers) |
| Hallucination | SimpleQA Verified, TruthfulQA (MC1/MC2 in English and Danish, generation), HalluLens (PreciseWikiQA, LongWiki, MixedEntities), Vectara summaries | DeepSeek-V4.1-Flash, plus GLM-5.3-Flash as second judging model; HHEM-2.1-Open for summaries |
| Bias | CrowS-Pairs, BBQ | likelihood, exact answer |

Mimir and HRM-Text are prefix language models: attention over the prompt goes in both ways, but the model writes the answer from left to right.
vLLM recomputes a preempted request as one block, and for these models this block would include the answer.
Therefore `mimireval/generate.py` starts a request only if its full length fits the KV cache, and it fails if vLLM ever preempts.

## Setup

Needs a CUDA GPU (we used one B200), [uv](https://docs.astral.sh/uv/) and about 30 GB of disk for the Wikipedia database of LongWiki.

```bash
cp .env.example .env    # fill in the API endpoint and a Hugging Face token
bash scripts/setup.sh   # Python 3.14 env, benchmark code at pinned commits, data
```

The judging models run through any OpenAI-compatible API. Their names are in `mimireval/models.py`.

## Evaluate a new Mimir version

1. Add the model to `MODELS` in `mimireval/models.py`, with a fixed revision:

   ```python
   ModelSpec("mimir_v2", "danish-foundation-models/DFM-Mimir-v2", "<commit sha>", prefix_lm=True, label="Mimir v2"),
   ```

2. Run every benchmark for this model. The run is resumable: finished files are skipped.

   ```bash
   bash scripts/evaluate_model.sh --model mimir_v2
   ```

3. Build the report with all models, and the images if you want them:

   ```bash
   source env.sh && python -m mimireval.report --images
   ```

The scores of the five models above are in `results/scores` and `results/lmeval`, so step 2 runs only the new model.
The HalluLens questions are in `prompts/hallulens`. The judging model generated them once, and every model gets the same questions.
`scripts/run_all.sh` runs every model from zero.

## Notes

- The original judging models of TruthfulQA and HalluLens are no longer available or run only locally, so an API model replaces them (one model for all HalluLens roles, as in the original).
- HalluLens runs one trial instead of three, and GeneratedEntities is left out.
- The LongWiki database is a public copy of the FActScore `enwiki-20230401.db` on Hugging Face (`danielli2003/enwiki-20230401.db`), because the original link is rate-limited.
- `mimireval/tasks/hallulens.py` patches HalluLens at run time without changing its protocol: faster title search and cache writes, pandas 3 compatibility, CPU fallback for the retrieval encoder.
