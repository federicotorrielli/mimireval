"""Grounded summarization faithfulness, Vectara hallucination-leaderboard protocol.

Prompt and temperature 0 follow the current leaderboard README; scoring uses the open
HHEM-2.1-Open on the public 955-article set (the live leaderboard uses private data and HHEM-2.3).
"""
from huggingface_hub import hf_hub_download

DATASET, REVISION = "vectara/hhem_leaderboard_datasets", "df1effe54c93540046d3599c51b28d91c2d435ab"
HHEM, HHEM_REVISION = "vectara/hallucination_evaluation_model", "de9a5e6ba9ba49531a022dba17e35ffeb2a6fad8"
GEN = dict(max_new_tokens=1024, temperature=0.0, top_p=1.0)
MAX_WORDS = 1500  # every prompt fits a 4096-token context with room for the summary
REFUSAL = "I am unable to summarize this text."

PROMPT = """Your task is to provide a concise and factual summary for the given passage.

Rules
1. Summarize using only the information in the given passage. Do not infer. Do not use your internal knowledge.
2. Do not provide a preamble or explanation, output only the summary.
3. Summaries should never exceed 20 percent of the original text's length.
4. Maintain the tone of the passage.

If you are unable to summarize the text due to missing, unreadable, irrelevant or insufficient content, respond only with:
"I am unable to summarize this text."

Here is the passage:
{passage}"""


def load() -> list[dict]:
    import pandas as pd
    path = hf_hub_download(DATASET, "leaderboard_dataset_revised.csv", repo_type="dataset", revision=REVISION)
    df = pd.read_csv(path)
    df = df[df.text.str.split().str.len() <= MAX_WORDS].reset_index(drop=True)
    return [{"id": f"hhem/{i}", "source": r.text, "prompt": PROMPT.format(passage=r.text)} for i, r in df.iterrows()]


def answered(summary: str) -> bool:
    s = summary.strip()
    return bool(s) and REFUSAL.lower().rstrip(".") not in s.lower()


def score(rows: list[dict], generations: list[str]) -> tuple[dict, list[float]]:
    import torch
    from transformers import AutoModelForSequenceClassification

    model = AutoModelForSequenceClassification.from_pretrained(
        HHEM, revision=HHEM_REVISION, trust_remote_code=True).to("cuda" if torch.cuda.is_available() else "cpu").eval()
    idx = [i for i, g in enumerate(generations) if answered(g)]
    scores = [None] * len(generations)
    with torch.no_grad():
        for b in range(0, len(idx), 32):
            chunk = idx[b:b + 32]
            preds = model.predict([(rows[i]["source"], generations[i]) for i in chunk])
            for i, s in zip(chunk, preds.tolist()):
                scores[i] = s
    answered_scores = [s for s in scores if s is not None]
    consistent = sum(s >= 0.5 for s in answered_scores) / max(len(answered_scores), 1)
    words = [len(generations[i].split()) for i in idx]
    return {"hallucination_rate": 1 - consistent, "factual_consistency_rate": consistent,
            "answer_rate": len(idx) / len(generations), "avg_summary_words": sum(words) / max(len(words), 1),
            "mean_hhem_score": sum(answered_scores) / max(len(answered_scores), 1)}, scores
