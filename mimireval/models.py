"""Model registry: exact revisions and prompt formats."""
from dataclasses import dataclass, field


@dataclass(frozen=True)
class ModelSpec:
    key: str
    repo: str
    revision: str | None  # None for a local checkpoint
    prefix_lm: bool = False
    # Raw prompt format for models without a chat template ("{prompt}" placeholder).
    raw_format: str | None = None
    chat_kwargs: dict = field(default_factory=dict)
    max_model_len: int = 4096
    attention_backend: str = "FLASH_ATTN"
    label: str = ""


# API judges (factuality): the primary scores, the second only measures agreement.
# The first five models were rated with DeepSeek-V4.1-Flash, which the API has since replaced by -1017.
PRIMARY_JUDGE = "deepseek-ai/DeepSeek-V4.1-Flash-1017"
SECOND_JUDGE = "zai-org/GLM-5.3-Flash"

HRM_SYNTH_COT = "<|im_start|><|quad_end|><|object_ref_end|>{prompt}<|im_end|>"

MODELS = {
    m.key: m
    for m in [
        ModelSpec("mimir", "danish-foundation-models/DFM-Mimir",
                  "396df886074e3e43f30792fbf393b9a019fe28f7", prefix_lm=True, label="Mimir 1B"),
        # Not on the Hub yet: DFM15 step 3.8M EMA export, model.safetensors sha256 07e031e.
        ModelSpec("mimir_v18", "/work/dfm/euroeval-runs/mimir-v1.8-3800000-validation/model", None,
                  prefix_lm=True, label="Mimir v1.8"),
        ModelSpec("hrm_text", "sapientinc/HRM-Text-1B",
                  "22097cbcecdd1301afe30a19a3ee61b96a9863e5", prefix_lm=True,
                  raw_format=HRM_SYNTH_COT, label="HRM-Text 1B"),
        ModelSpec("qwen35_08b", "Qwen/Qwen3.5-0.8B",
                  "2fc06364715b967f1860aea9cf38778875588b17", label="Qwen 3.5 0.8B"),
        ModelSpec("qwen35_2b", "Qwen/Qwen3.5-2B",
                  "15852e8c16360a2fea060d615a32b45270f8a8fc", label="Qwen 3.5 2B"),
        ModelSpec("gemma3_1b", "google/gemma-3-1b-it",
                  "dcc83ea841ab6100d6b47a070329e1ba4cf78752", label="Gemma 3 1B"),
        ModelSpec("gemma4_e2b", "google/gemma-4-E2B-it",
                  "3e22461f65e89153144f8adb70e3b8c2cc9845a7", label="Gemma 4 E2B",
                  attention_backend="TRITON_ATTN"),  # FLASH_ATTN lacks its 512 head size
        ModelSpec("olmo2_1b", "allenai/OLMo-2-0425-1B-Instruct",
                  "48d788eca847d4d7548f375ad03d3c9312f6139e", label="OLMo 2 1B"),
    ]
}


def register(key: str, repo: str, revision: str, prefix_lm: bool = True, label: str | None = None) -> ModelSpec:
    """Add a model at runtime, e.g. a new Mimir release."""
    MODELS[key] = ModelSpec(key, repo, revision, prefix_lm=prefix_lm, label=label or key)
    return MODELS[key]
