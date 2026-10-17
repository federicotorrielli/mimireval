"""Collect every score and render one image per chart into report/images/.

    python -m mimireval.report
"""
import json
import re
import shutil
import statistics
import subprocess
import tempfile
from pathlib import Path

from mimireval.io import GEN_DIR, PROJ, SCORE_DIR
from mimireval.models import MODELS, PRIMARY_JUDGE, SECOND_JUDGE

LMEVAL_DIR = PROJ / "results" / "lmeval"
REPORT_DIR = Path(__file__).resolve().parent.parent / "report"
SECOND = "." + SECOND_JUDGE.split("/")[-1]
# Sensitivity variants {key: (base model, label)}: shown next to their model, never ranked.
VARIANTS: dict[str, tuple[str, str]] = {}

# (id, label, task, metric key or callable, better, group, note)
# better: "high" | "low" | "mid50" (CrowS-Pairs) | "none" (WMDP: capability, not safety)
SAFETY = [
    ("harmbench", "HarmBench", "harmbench", "inverted_micro_asr_lower", "high", "harmful", "direct harmful requests"),
    ("strongreject", "StrongREJECT", "strongreject", "inverted_asr", "high", "harmful", "harmful requests wrapped in jailbreak tricks"),
    ("wildguardtest", "WildGuardTest", "wildguardtest", "inverted_micro_harm_lower", "high", "harmful", "harmful prompts written by people, some disguised"),
    ("wjb_harmful", "WildJailbreak, harmful", "wildjailbreak_harmful", "inverted_macro_asr", "high", "harmful", "harmful requests hidden in long, elaborate stories"),
    ("dan", "Do Anything Now", "dan", "inverted_macro_asr", "high", "harmful", "roleplay jailbreaks that ask the model to drop its rules"),
    ("trustllm", "TrustLLM jailbreak", "trustllm_jailbreak", "inverted_macro_asr", "high", "harmful", "jailbreak prompts from the TrustLLM suite"),
    ("sorrybench", "SORRY-Bench", "sorrybench", "inverted_macro_asr", "high", "harmful", "unsafe requests on 44 topics"),
    ("xstest_unsafe", "XSTest, unsafe prompts", "xstest", "rta_all_contrast", "high", "harmful", "clearly unsafe questions"),
    ("xstest_safe", "XSTest, safe prompts", "xstest", "inverted_rta_all_safe", "high", "benign", "safe questions that sound unsafe ('how do I kill a process?')"),
    ("wjb_benign", "WildJailbreak, benign", "wildjailbreak_benign", "macro_asr", "high", "benign", "safe requests written like jailbreaks"),
    ("orbench", "OR-Bench-Hard-1K", "orbench_hard", "macro_asr", "high", "benign", "safe prompts that look toxic"),
    ("toxigen", "ToxiGen", "toxigen", "safe_overall", "high", "bias", "continuations of hateful text without toxic content"),
    ("bbq_acc", "BBQ accuracy", "bbq", "accuracy", "high", "bias", "questions that tempt a stereotyped answer"),
    ("bbq_bias", "BBQ bias size, ambiguous", "bbq", lambda r: abs(r["bias_ambig"]), "low", "bias", "preference for the stereotype (lower is better)"),
    ("wmdp", "WMDP accuracy", "wmdp", lambda r: 1 - r["inverted_accuracy"], "none", "hazard", "biological, chemical and cyber weapons, chance level 25%"),
]

FACT = [
    ("tqa_mc1", "TruthfulQA MC1, English", ("lmeval", "truthfulqa_mc1", "acc,none"), "high", "truthfulqa", "chooses the true answer among false ones"),
    ("tqa_mc1_da", "TruthfulQA MC1, Danish", ("lmeval", "truthfulqa_da_mc1", "acc,none"), "high", "truthfulqa", "the same, Danish translation"),
    ("tqa_mc2", "TruthfulQA MC2, English", ("lmeval", "truthfulqa_mc2", "acc,none"), "high", "truthfulqa", "probability that the model gives to true answers"),
    ("tqa_mc2_da", "TruthfulQA MC2, Danish", ("lmeval", "truthfulqa_da_mc2", "acc,none"), "high", "truthfulqa", "the same, Danish translation"),
    ("tqa_mc1_gen", "TruthfulQA MC1, letter", ("truthfulqa_mc1_gen", "acc"), "high", "truthfulqa", "writes the letter of the true answer"),
    ("tqa_truthful", "TruthfulQA, truthful", ("truthfulqa_gen", "truthful"), "high", "truthfulqa", "free answers rated true"),
    ("tqa_ti", "TruthfulQA, truthful and informative", ("truthfulqa_gen", "truthful_and_informative"), "high", "truthfulqa", "free answers rated true and useful"),
    ("precise_halu", "PreciseWikiQA hallucination", ("precise_wikiqa", "halu_Rate"), "low", "hallulens", "incorrect answers among the questions with an answer (lower is better)"),
    ("precise_refusal", "PreciseWikiQA abstention", ("precise_wikiqa", "refusal_rate"), "none", "hallulens", "questions the model does not answer"),
    ("mixed_false_acc", "Fictional names taken as real", ("nonexistent_mixed", "false_acceptance_rate"), "low", "hallulens", "fictional animals, plants and medicines (lower is better)"),
    ("longwiki_prec", "LongWiki precision", ("longwiki", "precision"), "high", "hallulens", "facts in long answers supported by Wikipedia"),
    ("longwiki_f1", "LongWiki F1@32", ("longwiki", "f1@32"), "high", "hallulens", "precision and coverage of the facts"),
    ("hhem_halu", "Summary hallucination (HHEM)", ("hhem", "hallucination_rate"), "low", "grounded", "news summaries with unsupported content (lower is better)"),
    ("hhem_answer", "Summary answer rate", ("hhem", "answer_rate"), "high", "grounded", "articles the model agreed to summarize"),
    ("crows", "CrowS-Pairs stereotype", ("lmeval_raw", "crows_pairs_english", "pct_stereotype,none"), "mid50", "bias", "prefers the stereotyped sentence (50% means no preference)"),
]


def _load(path: Path):
    return json.loads(path.read_text())["report"] if path.exists() else None


def _stat(values: list[float]) -> dict | None:
    values = [v for v in values if v is not None]
    if not values:
        return None
    return {"mean": statistics.fmean(values), "sd": statistics.stdev(values) if len(values) > 1 else 0.0,
            "n": len(values)}


def safety_rows() -> list[dict]:
    rows = []
    for rid, label, task, key, better, group, note in SAFETY:
        values = {}
        for m in MODELS:
            reports = [_load(SCORE_DIR / m / task / f"seed{s}.json") for s in range(3)]
            get = key if callable(key) else (lambda r, k=key: r[k])
            values[m] = _stat([get(r) for r in reports if r])
        rows.append(dict(id=rid, label=label, better=better, group=group, note=note, values=values))
    return rows


def _fact_value(model: str, src: tuple, suffix: str = ""):
    if src[0] in ("lmeval", "lmeval_raw"):
        f = LMEVAL_DIR / model / ("chat.json" if src[0] == "lmeval" else "raw.json")
        res = json.loads(f.read_text())["results"] if f.exists() else {}
        return res.get(src[1], {}).get(src[2])
    r = _load(SCORE_DIR / model / src[0] / f"seed0{suffix}.json")
    return None if r is None else r.get(src[1])


def fact_rows() -> list[dict]:
    rows = []
    for rid, label, src, better, group, note in FACT:
        values = {m: _stat([_fact_value(m, src)]) for m in (*MODELS, *VARIANTS)}
        second = {m: _fact_value(m, src, SECOND) for m in MODELS} if src[0] not in ("lmeval", "lmeval_raw") else {}
        rows.append(dict(id=rid, label=label, better=better, group=group, note=note, values=values, second=second))
    return rows


SHORT_TOPICS = {"general advice for system intrusion, hacking & cyber attacks, data privacy and confidentiality violations":
                "system intrusion and hacking advice"}


def sorry_categories() -> dict:
    """Fulfillment (1 - refusal) per SORRY-Bench category, mean over seeds."""
    out, cats = {}, None
    for m in MODELS:
        reports = [r for s in range(3) if (r := _load(SCORE_DIR / m / "sorrybench" / f"seed{s}.json"))]
        if not reports:
            continue
        keys = [k for k in reports[0] if k.startswith("inverted_") and k != "inverted_macro_asr"]
        cats = cats or keys
        out[m] = [1 - statistics.fmean(r[k] for r in reports) for k in cats]
    names = [k.removeprefix("inverted_").removesuffix("_asr").replace("_", " ") for k in cats or []]
    names = [SHORT_TOPICS.get(n, n) for n in names]
    return {"categories": names, "values": out}


def simpleqa() -> dict:
    return {m: r for m in (*MODELS, *VARIANTS) if (r := _load(SCORE_DIR / m / "simpleqa_verified" / "seed0.json"))}


def _kappa(a: list, b: list) -> float | None:
    """Cohen's kappa for two raters over the same items."""
    if not a or len(a) != len(b):
        return None
    labels, n = set(a) | set(b), len(a)
    po = sum(x == y for x, y in zip(a, b)) / n
    pe = sum((a.count(c) / n) * (b.count(c) / n) for c in labels)
    return 1.0 if pe == 1 else (po - pe) / (1 - pe)


def agreement() -> list[dict]:
    """Per-item agreement between the two judges, pooled over models."""
    def items(model, task, suffix):
        p = SCORE_DIR / model / task / f"seed0{suffix}.items.jsonl"
        return [json.loads(line)["grade"] for line in p.open()] if p.exists() else None

    specs = [("SimpleQA correctness", "simpleqa_verified", lambda g: g),
             ("TruthfulQA truthful", "truthfulqa_gen", lambda g: g["truthful"]),
             ("TruthfulQA informative", "truthfulqa_gen", lambda g: g["informative"])]
    rows = []
    hl = PROJ / "third_party" / "HalluLens" / "output"

    def precise(model, suffix, key):
        p = hl / "precise_wikiqa_goodwiki_dynamic" / (model + suffix) / "eval_results.json"
        return json.loads(p.read_text())[key] if p.exists() else None

    def believes(model, suffix):
        p = hl / "refusal_test" / "nonsense_all_0_2000" / (model + suffix) / "raw_eval_res.jsonl"
        if not p.exists():
            return None
        return ["true" in json.loads(line)["eval_res"].lower().split("does_believe")[-1][:12] for line in p.open()]

    hl_specs = [("PreciseWikiQA abstention", lambda m, s: precise(m, s, "abstantion")),
                ("PreciseWikiQA hallucination", lambda m, s: precise(m, s, "halu_test_res")),
                ("Fictional names taken as real", believes)]
    for label, get in hl_specs:
        a, b = [], []
        for m in MODELS:
            x, y = get(m, ""), get(m, SECOND)
            if x and y and len(x) == len(y):
                a += x
                b += y
        if a:
            rows.append({"label": label, "n": len(a), "kappa": _kappa(a, b),
                         "agree": sum(x == y for x, y in zip(a, b)) / len(a)})
    for label, task, get in specs:
        a, b = [], []
        for m in MODELS:
            x, y = items(m, task, ""), items(m, task, SECOND)
            if x and y and len(x) == len(y):
                a += [get(g) for g in x]
                b += [get(g) for g in y]
        if a:
            rows.append({"label": label, "n": len(a), "kappa": _kappa(a, b),
                         "agree": sum(x == y for x, y in zip(a, b)) / len(a)})
    return rows


EXAMPLE = 133  # TruthfulQA question shown in full; built once, then read from report/example.json


def _short(text: str, min_len: int = 80) -> str:
    """Opening sentences of a free answer (at least min_len characters), without markdown."""
    text = " ".join(re.sub(r"[*#`]", "", text).split())
    end = next((m.end() for m in re.finditer(r"[.!?](?= |$)", text) if m.end() >= min_len), len(text))
    return text[:end] + (" …" if end < len(text) else "")


def example() -> dict:
    """One TruthfulQA question with every model's free answer and both raters' labels."""
    path = REPORT_DIR / "example.json"
    if path.exists():
        return json.loads(path.read_text())

    def rows(p):
        return [json.loads(line) for line in p.open()]
    gen = {m: rows(GEN_DIR / m / "truthfulqa_gen" / "seed0.jsonl") for m in MODELS}
    grades = {m: [[r["grade"] for r in rows(SCORE_DIR / m / "truthfulqa_gen" / f"seed0{s}.items.jsonl")]
                  for s in ("", SECOND)] for m in MODELS}
    truthful = lambda m, i: all(g[i]["truthful"] for g in grades[m])
    false = lambda m, i: not any(g[i]["truthful"] for g in grades[m])
    others = [m for m in MODELS if m != "mimir"]
    pool = sum(truthful("mimir", i) and all(false(m, i) for m in others) for i in range(len(gen["mimir"])))
    item = gen["mimir"][EXAMPLE]
    out = {"question": item["question"], "reference": item["best_answer"], "pool": pool, "total": len(gen["mimir"]),
           "answers": {m: {"text": _short(gen[m][EXAMPLE]["completion"]), "truthful": truthful(m, EXAMPLE)} for m in MODELS}}
    path.write_text(json.dumps(out, indent=1, ensure_ascii=False) + "\n")
    return out


def collect() -> dict:
    return {
        "models": [{"key": k, "label": s.label, "repo": s.repo, "revision": s.revision[:7]} for k, s in MODELS.items()],
        "variants": [{"key": k, "of": of, "label": label} for k, (of, label) in VARIANTS.items()],
        "focus": "mimir",
        "safety": safety_rows(),
        "fact": fact_rows(),
        "sorry": sorry_categories(),
        "simpleqa": simpleqa(),
        "agreement": agreement(),
        "example": example(),
        "judges": {"primary": PRIMARY_JUDGE, "second": SECOND_JUDGE},
    }


IMAGES = ("scatter", "safety", "sorry", "simpleqa", "fact", "agree", "example")


def images(page: Path, theme: str = "light"):
    """Screenshot each chart's export view with headless Chromium at 2x, then trim the empty margin."""
    from PIL import Image, ImageChops

    chrome = next(filter(None, map(shutil.which, ("chromium", "google-chrome-stable", "google-chrome"))))
    out = REPORT_DIR / "images"
    out.mkdir(exist_ok=True)
    for name in IMAGES:
        png = out / f"{name}.png"
        subprocess.run([chrome, "--headless=new", "--disable-gpu", "--hide-scrollbars", "--force-device-scale-factor=2",
                        "--window-size=1200,3000", f"--screenshot={png}", f"{page.as_uri()}?export={name}&theme={theme}"],
                       check=True, capture_output=True)
        im = Image.open(png).convert("RGB")
        box = ImageChops.difference(im, Image.new("RGB", im.size, im.getpixel((0, im.height - 1)))).getbbox()
        if box:
            im.crop((0, 0, im.width, min(im.height, box[3] + 40))).save(png)
        print(f"image -> {png}")


def main():
    html = (REPORT_DIR / "template.html").read_text()
    html = html.replace("/*DATA*/null", json.dumps(collect()))
    html = html.replace("/*TEXT*/{}", (REPORT_DIR / "text.json").read_text())
    with tempfile.TemporaryDirectory() as tmp:  # the page only exists to be screenshotted
        page = Path(tmp) / "index.html"
        page.write_text(html)
        images(page)


if __name__ == "__main__":
    main()
