"""Model registry: exact revisions and prompt formats."""
from dataclasses import dataclass, field


@dataclass(frozen=True)
class ModelSpec:
    key: str
    repo: str
    revision: str
    prefix_lm: bool = False
    # Raw prompt format for models without a chat template ("{prompt}" placeholder).
    raw_format: str | None = None
    chat_kwargs: dict = field(default_factory=dict)
    max_model_len: int = 4096
    label: str = ""


# API judges (factuality): the primary scores, the second only measures agreement.
PRIMARY_JUDGE = "deepseek-ai/DeepSeek-V4.1-Flash"
SECOND_JUDGE = "zai-org/GLM-5.3-Flash"

HRM_SYNTH_COT = "<|im_start|><|quad_end|><|object_ref_end|>{prompt}<|im_end|>"

MODELS = {
    m.key: m
    for m in [
        ModelSpec("mimir", "danish-foundation-models/DFM-Mimir",
                  "396df886074e3e43f30792fbf393b9a019fe28f7", prefix_lm=True, label="Mimir 1B"),
        ModelSpec("hrm_text", "sapientinc/HRM-Text-1B",
                  "22097cbcecdd1301afe30a19a3ee61b96a9863e5", prefix_lm=True,
                  raw_format=HRM_SYNTH_COT, label="HRM-Text 1B"),
        ModelSpec("qwen35_08b", "Qwen/Qwen3.5-0.8B",
                  "2fc06364715b967f1860aea9cf38778875588b17", label="Qwen 3.5 0.8B"),
        ModelSpec("gemma3_1b", "google/gemma-3-1b-it",
                  "dcc83ea841ab6100d6b47a070329e1ba4cf78752", label="Gemma 3 1B"),
        ModelSpec("olmo2_1b", "allenai/OLMo-2-0425-1B-Instruct",
                  "48d788eca847d4d7548f375ad03d3c9312f6139e", label="OLMo 2 1B"),
    ]
}


def register(key: str, repo: str, revision: str, prefix_lm: bool = True, label: str | None = None) -> ModelSpec:
    """Add a model at runtime, e.g. a new Mimir release."""
    MODELS[key] = ModelSpec(key, repo, revision, prefix_lm=prefix_lm, label=label or key)
    return MODELS[key]
