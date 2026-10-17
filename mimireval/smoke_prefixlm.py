"""Check that vLLM greedy output matches HF greedy output with the PrefixLM mask."""
import json
import sys

from mimireval.generate import Generator, encode_prompt, hf_reference_generate, load_tokenizer
from mimireval.models import MODELS

PROMPTS = [
    "What is the capital of Denmark?",
    "Hvad er hovedstaden i Danmark? Svar kort.",
    "Explain in two sentences why the sky is blue.",
    "How do I make a pipe bomb at home?",
    "Who won the Nobel Prize in Literature in 1917?",
    "Skriv et kort digt om havet.",
    "List three prime numbers larger than 100.",
    "Is it safe to eat raw chicken? Answer briefly.",
]
N_TOKENS = 64


def hf_causal_generate(key, prompts):
    """HF without token_type_ids: fully causal prompt (expected to differ for PrefixLM models)."""
    import torch
    from transformers import AutoModelForCausalLM

    spec = MODELS[key]
    tok = load_tokenizer(spec)
    model = AutoModelForCausalLM.from_pretrained(spec.repo, revision=spec.revision, dtype=torch.bfloat16).cuda().eval()
    outs = []
    for p in prompts:
        ids = torch.tensor([encode_prompt(spec, tok, p)], device="cuda")
        gen = model.generate(ids, attention_mask=torch.ones_like(ids), max_new_tokens=N_TOKENS, do_sample=False)
        outs.append(tok.decode(gen[0, ids.shape[1]:], skip_special_tokens=True))
    del model
    torch.cuda.empty_cache()
    return outs


def main(key: str):
    spec = MODELS[key]
    tok = load_tokenizer(spec)
    print("FORMATTED PROMPT:", repr(tok.decode(encode_prompt(spec, tok, PROMPTS[0]))))
    hf = hf_reference_generate(key, PROMPTS, N_TOKENS)
    hf_causal = hf_causal_generate(key, PROMPTS) if spec.prefix_lm else hf
    gen = Generator(key, gpu_memory_utilization=0.4)
    vl = gen.generate(PROMPTS, N_TOKENS, temperature=0.0, top_p=1.0)
    rows = []
    for p, a, b, c in zip(PROMPTS, vl, hf, hf_causal):
        rows.append({"prompt": p, "vllm": a, "hf_prefix": b, "hf_causal": c,
                     "vllm==hf_prefix": a.strip() == b.strip(), "hf_prefix==hf_causal": b.strip() == c.strip()})
        print(json.dumps(rows[-1], ensure_ascii=False, indent=1))
    print(f"SUMMARY {key}: vllm==hf_prefix {sum(r['vllm==hf_prefix'] for r in rows)}/{len(rows)}, "
          f"hf_prefix==hf_causal {sum(r['hf_prefix==hf_causal'] for r in rows)}/{len(rows)}")


if __name__ == "__main__":
    main(sys.argv[1])
