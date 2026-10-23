"""vLLM generation with correct prompt formatting for every model in the registry."""
import os

from transformers import AutoTokenizer

from mimireval.models import MODELS, ModelSpec


def load_tokenizer(spec: ModelSpec):
    return AutoTokenizer.from_pretrained(spec.repo, revision=spec.revision)


def encode_prompt(spec: ModelSpec, tokenizer, user_text: str) -> list[int]:
    """Single user turn -> token ids, including BOS exactly once."""
    if spec.raw_format is not None:
        text = spec.raw_format.format(prompt=user_text)
        return tokenizer(text, add_special_tokens=True)["input_ids"]
    text = tokenizer.apply_chat_template(
        [{"role": "user", "content": user_text}],
        add_generation_prompt=True, tokenize=False, **spec.chat_kwargs,
    )
    # The template already contains BOS where the model expects it.
    return tokenizer(text, add_special_tokens=False)["input_ids"]


class Generator:
    """Thin wrapper around one vLLM engine."""

    def __init__(self, key: str, gpu_memory_utilization: float | None = None):
        from vllm import LLM

        self.spec = MODELS[key]
        # Lower MIMIREVAL_GEN_MEM to run several models on one GPU.
        gpu_memory_utilization = gpu_memory_utilization or float(os.environ.get("MIMIREVAL_GEN_MEM", 0.92))
        self.tokenizer = load_tokenizer(self.spec)
        engine_kwargs = dict(
            model=self.spec.repo,
            revision=self.spec.revision,
            tokenizer_revision=self.spec.revision,
            dtype="bfloat16",
            max_model_len=self.spec.max_model_len,
            gpu_memory_utilization=gpu_memory_utilization,
            seed=0,
            max_num_seqs=1024,  # 1B models: decode is memory-bound, larger batches raise throughput
            # FlashInfer needs nvcc for JIT; FlashAttention is what the Mimir authors used.
            attention_backend=self.spec.attention_backend,
        )
        if self.spec.prefix_lm:
            # PrefixLM: every prefill is bidirectional, so a prompt must be prefilled whole and never reused.
            # Stats on: the preemption counter is checked after every call.
            engine_kwargs |= dict(enable_prefix_caching=False, enable_chunked_prefill=False,
                                  max_num_batched_tokens=self.spec.max_model_len, disable_log_stats=False)
        self.llm = LLM(**engine_kwargs)

    def generate(self, prompts: list[str], max_new_tokens: int, temperature: float, top_p: float,
                 seed: int = 0, stop: list[str] | None = None) -> list[str]:
        from vllm import SamplingParams, TokensPrompt

        token_prompts, params = [], []
        for i, p in enumerate(prompts):
            ids = encode_prompt(self.spec, self.tokenizer, p)
            room = self.spec.max_model_len - len(ids)
            if room < 16:  # keep the tail: instructions usually sit at the end
                ids = ids[:1] + ids[-(self.spec.max_model_len - 512):]
                room = self.spec.max_model_len - len(ids)
            token_prompts.append(TokensPrompt(prompt_token_ids=ids))
            params.append(SamplingParams(
                max_tokens=min(max_new_tokens, room), temperature=temperature, top_p=top_p,
                seed=seed * 1_000_003 + i, stop=stop,
            ))
        if not self.spec.prefix_lm:
            return [o.outputs[0].text for o in self.llm.generate(token_prompts, params, use_tqdm=True)]
        texts = self._generate_reserved(token_prompts, params)
        if (n := self._preemptions()) != self.preempted:
            raise RuntimeError(f"{n - self.preempted} preemptions: PrefixLM recompute would be bidirectional")
        return texts

    def _preemptions(self) -> int:
        return next(m.value for m in self.llm.get_metrics() if m.name == "vllm:num_preemptions")

    def _generate_reserved(self, token_prompts, params) -> list[str]:
        """Admission control: a request starts only if its worst case (prompt + max_tokens) fits in the
        KV cache next to the running ones. vLLM would otherwise preempt and recompute, and for a PrefixLM
        model the recompute prefills the generated tokens bidirectionally, which changes the output."""
        from tqdm import tqdm

        engine = self.llm.llm_engine
        cache = engine.vllm_config.cache_config
        capacity, block = cache.num_gpu_blocks * cache.block_size, cache.block_size
        self.preempted = self._preemptions()
        need = [-(-(len(tp["prompt_token_ids"]) + sp.max_tokens) // block) * block
                for tp, sp in zip(token_prompts, params)]
        texts, running, used, nxt = [None] * len(need), {}, 0, 0
        with tqdm(total=len(need), desc="generate (reserved)") as bar:
            while nxt < len(need) or running:
                while nxt < len(need) and used + need[nxt] <= capacity:
                    engine.add_request(str(nxt), token_prompts[nxt], params[nxt])
                    running[str(nxt)], used, nxt = need[nxt], used + need[nxt], nxt + 1
                for out in engine.step():
                    if out.finished:
                        used -= running.pop(out.request_id)
                        texts[int(out.request_id)] = out.outputs[0].text
                        bar.update()
        return texts


def hf_reference_generate(key: str, prompts: list[str], max_new_tokens: int) -> list[str]:
    """Greedy HF generation with the PrefixLM mask, used only to validate vLLM."""
    import torch
    from transformers import AutoModelForCausalLM

    spec = MODELS[key]
    tok = load_tokenizer(spec)
    model = AutoModelForCausalLM.from_pretrained(
        spec.repo, revision=spec.revision, dtype=torch.bfloat16, attn_implementation="sdpa",
    ).cuda().eval()
    outs = []
    for p in prompts:
        ids = torch.tensor([encode_prompt(spec, tok, p)], device="cuda")
        kwargs = {"token_type_ids": torch.ones_like(ids)} if spec.prefix_lm else {}
        with torch.no_grad():
            gen = model.generate(ids, attention_mask=torch.ones_like(ids), max_new_tokens=max_new_tokens,
                                 do_sample=False, **kwargs)
        outs.append(tok.decode(gen[0, ids.shape[1]:], skip_special_tokens=True))
    del model
    torch.cuda.empty_cache()
    return outs
