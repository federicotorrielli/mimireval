"""lm-eval HF backend that scores PrefixLM models (Mimir, HRM-Text) with the correct mask.

lm-eval feeds `(context + continuation)[:-1]`, right-padded. The context must be one
bidirectional block (token_type_ids = 1), the continuation and padding stay causal.
"""
import torch
from lm_eval.api.registry import register_model
from lm_eval.models.huggingface import HFLM


@register_model("hf-prefixlm")
class PrefixLMHF(HFLM):
    def _loglikelihood_tokens(self, requests, disable_tqdm=False, override_bs=None):
        # Map each scored input sequence to its context length.
        self._ctx_len = {}
        for _, ctx, cont in requests:
            inp = (ctx + cont)[-(self.max_length + 1):][:-1]
            self._ctx_len[tuple(inp)] = len(inp) - len(cont) + 1
        return super()._loglikelihood_tokens(requests, disable_tqdm=disable_tqdm, override_bs=override_bs)

    def _model_call(self, inps, attn_mask=None, labels=None):
        token_type_ids = torch.zeros_like(inps)
        ctx_lens = getattr(self, "_ctx_len", {})
        for i, row in enumerate(inps.tolist()):
            n = len(row)
            while n and row[n - 1] == 0:  # pad_and_concat pads with zeros
                n -= 1
            # Unknown rows (batch-size probing) stay causal.
            token_type_ids[i, :ctx_lens.get(tuple(row[:n]), 0)] = 1
        with torch.no_grad():
            return self.model(inps, token_type_ids=token_type_ids).logits
