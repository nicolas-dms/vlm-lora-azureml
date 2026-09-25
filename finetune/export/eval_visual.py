"""Show, visually, what the fine-tuning actually changed.

The honest framing, and it matters more than the code below: on 500 samples and
256M parameters we did NOT teach the model to diagnose machinery. We taught it a
FORMAT and a VOCABULARY. So this script measures format compliance first and
field agreement second, and the chart says so out loud. Claiming "the model is
now a maintenance expert" would collapse under the first question from the room.

Two artifacts:
  eval_scoreboard.png - base vs fine-tuned on the test split, plus the
                        root_cause confusion matrix of the fine-tuned model
  eval_examples.png   - the same charts, side by side, prose vs JSON

Everything runs locally on CPU from the 10 MB adapter. No endpoint, no GPU, no
Azure call - which is also the point of brick 6.

    python eval_visual.py --adapter ../../tmp-artifacts/maint-vlm-adapter \
                          --dataset %TEMP%/vlm_ds_v1 --n 24
"""

import argparse
import json
import re
import textwrap
from pathlib import Path
from typing import Any, Dict, List, Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import torch  # noqa: E402
from PIL import Image  # noqa: E402
from peft import PeftModel  # noqa: E402
from transformers import AutoModelForVision2Seq, AutoProcessor  # noqa: E402

DEFAULT_BASE = "HuggingFaceTB/SmolVLM-256M-Instruct"

REQUIRED = ["device_id", "severity", "root_cause", "action_code", "eta_minutes"]
ENUMS = {
    "severity": ["info", "warning", "critical"],
    "root_cause": ["normal_operation", "cooling_degradation", "bearing_wear", "pressure_loss"],
    "action_code": ["NONE", "MONITOR", "CHK_COOLING", "CHK_BEARING", "CHK_SEALS", "STOP_MAINT"],
}
FIELDS = ["severity", "root_cause", "action_code", "eta_minutes"]

BASE_COLOR = "#c0392b"
TUNED_COLOR = "#1e8449"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--adapter", required=True, help="folder downloaded with az ml model download")
    parser.add_argument("--dataset", required=True, help="folder holding test.jsonl and images/")
    parser.add_argument("--split", default="test")
    parser.add_argument("--n", type=int, default=24, help="samples to score; 24 takes a few minutes on CPU")
    parser.add_argument("--examples", type=int, default=3, help="qualitative rows in the second figure")
    parser.add_argument("--base", default=None)
    parser.add_argument("--out", default="tmp-eval", help="output folder for the PNGs")
    parser.add_argument("--max_new_tokens", type=int, default=128)
    return parser.parse_args()


def extract_json(text: str) -> Optional[Dict[str, Any]]:
    """Pull the first balanced {...} out of free-form text.

    Deliberately generous, because the base model is the one being judged here.
    If it buries a valid object inside three sentences of prose, we still count
    it as valid JSON - otherwise the comparison is rigged and the audience is
    right to distrust it.
    """
    start = text.find("{")
    while start != -1:
        depth = 0
        for index in range(start, len(text)):
            if text[index] == "{":
                depth += 1
            elif text[index] == "}":
                depth -= 1
                if depth == 0:
                    try:
                        candidate = json.loads(text[start : index + 1])
                    except json.JSONDecodeError:
                        break
                    return candidate if isinstance(candidate, dict) else None
        start = text.find("{", start + 1)
    return None


def schema_ok(obj: Optional[Dict[str, Any]]) -> bool:
    if not obj or set(obj) != set(REQUIRED):
        return False
    if any(obj[field] not in values for field, values in ENUMS.items()):
        return False
    return isinstance(obj["eta_minutes"], int) and not isinstance(obj["eta_minutes"], bool)


def load_samples(dataset: Path, split: str, limit: int) -> List[Dict[str, Any]]:
    lines = (dataset / f"{split}.jsonl").read_text(encoding="utf-8").splitlines()
    samples = []
    for line in lines[:limit]:
        record = json.loads(line)
        samples.append(
            {
                "image": dataset / record["image"],
                "instruction": record["messages"][0]["content"][1]["text"],
                "truth": json.loads(record["messages"][1]["content"][0]["text"]),
            }
        )
    return samples


def generate(model, processor, image_path: Path, instruction: str, max_new_tokens: int) -> str:
    image = Image.open(image_path).convert("RGB")
    messages = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": instruction}]}]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=[text], images=[[image]], return_tensors="pt")
    with torch.no_grad():
        # Greedy: the demo has to give the same answer twice in a row on stage.
        out = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
    completion = out[0][inputs["input_ids"].shape[1] :]
    return processor.tokenizer.decode(completion, skip_special_tokens=True).strip()


def score(results: List[Dict[str, Any]], key: str, samples: List[Dict[str, Any]]) -> Dict[str, float]:
    total = len(results)
    parsed = [extract_json(r[key]) for r in results]
    metrics = {
        "valid JSON": sum(p is not None for p in parsed) / total,
        "schema OK": sum(schema_ok(p) for p in parsed) / total,
    }
    # An unparseable answer counts as wrong on every field. Any other denominator
    # would flatter whichever model refuses to answer in the expected shape.
    for field in FIELDS:
        metrics[field] = sum(
            bool(p) and p.get(field) == sample["truth"][field] for p, sample in zip(parsed, samples)
        ) / total
    return metrics


def plot_scoreboard(base_metrics, tuned_metrics, confusion, n, out_path: Path):
    labels = list(base_metrics)
    positions = range(len(labels))
    figure, (left, right) = plt.subplots(1, 2, figsize=(15, 5.5), gridspec_kw={"width_ratios": [1.25, 1]})

    height = 0.38
    left.barh([p + height / 2 for p in positions], [base_metrics[k] for k in labels], height,
              label="base model", color=BASE_COLOR)
    left.barh([p - height / 2 for p in positions], [tuned_metrics[k] for k in labels], height,
              label="+ LoRA adapter (10 MB)", color=TUNED_COLOR)
    for index, key in enumerate(labels):
        for value, offset in ((base_metrics[key], height / 2), (tuned_metrics[key], -height / 2)):
            left.text(value + 0.02, index + offset, f"{value:.0%}", va="center", fontsize=9)
    left.set_yticks(list(positions), labels)
    left.set_xlim(0, 1.12)
    left.set_xlabel(f"agreement with the ground truth, {n} unseen test charts")
    left.set_title("What the adapter bought: a format, then a vocabulary", loc="left", fontweight="bold")
    left.legend(loc="lower right")
    left.invert_yaxis()
    left.grid(axis="x", alpha=0.3)

    causes = ENUMS["root_cause"]
    right.imshow(confusion, cmap="Greens", vmin=0)
    right.set_xticks(range(len(causes)), [c.replace("_", "\n") for c in causes], fontsize=8)
    right.set_yticks(range(len(causes)), [c.replace("_", "\n") for c in causes], fontsize=8)
    right.set_xlabel("predicted")
    right.set_ylabel("actual")
    right.set_title("root_cause, fine-tuned model", loc="left", fontweight="bold")
    peak = max(max(row) for row in confusion) or 1
    for row in range(len(causes)):
        for column in range(len(causes)):
            value = confusion[row][column]
            right.text(column, row, value, ha="center", va="center", fontsize=10,
                       color="white" if value > peak * 0.6 else "black")

    figure.suptitle(
        "Format compliance is what a 500-sample LoRA teaches. Diagnostic accuracy is NOT the claim.",
        fontsize=10, style="italic", y=0.02,
    )
    figure.tight_layout(rect=(0, 0.04, 1, 1))
    figure.savefig(out_path, dpi=130)
    plt.close(figure)


def plot_examples(samples, results, rows: int, out_path: Path):
    rows = min(rows, len(samples))
    figure, axes = plt.subplots(rows, 3, figsize=(15, 3.6 * rows),
                                gridspec_kw={"width_ratios": [1.1, 1, 1]})
    axes = axes.reshape(rows, 3)

    for row in range(rows):
        sample, result = samples[row], results[row]
        axes[row][0].imshow(Image.open(sample["image"]))
        axes[row][0].axis("off")
        if row == 0:
            axes[row][0].set_title("the input: 30 min, 4 sensors", loc="left", fontweight="bold")

        panels = (
            ("base model", result["base"], BASE_COLOR),
            ("+ LoRA adapter", result["tuned"], TUNED_COLOR),
        )
        for column, (title, text, color) in enumerate(panels, start=1):
            axis = axes[row][column]
            axis.axis("off")
            body = "\n".join(textwrap.fill(line, 46) for line in text.splitlines())[:900]
            axis.text(0, 1, body, va="top", ha="left", fontsize=8, family="monospace",
                      transform=axis.transAxes)
            axis.set_title(title, loc="left", fontweight="bold", color=color)

        truth = json.dumps(sample["truth"], indent=1)
        axes[row][2].text(0, -0.02, f"ground truth\n{truth}", va="top", ha="left", fontsize=7,
                          family="monospace", color="#555555", transform=axes[row][2].transAxes)

    figure.tight_layout()
    figure.savefig(out_path, dpi=130)
    plt.close(figure)


def main():
    args = parse_args()
    adapter = Path(args.adapter)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    contract = adapter / "base_model.json"
    base, revision = args.base or DEFAULT_BASE, None
    if contract.exists() and args.base is None:
        info = json.loads(contract.read_text(encoding="utf-8"))
        base, revision = info["repo"], info.get("revision")

    kwargs = {"revision": revision} if revision else {}
    # Same preprocessing as training: 384 px, no image splitting. Change either and
    # the comparison silently stops measuring the adapter.
    processor = AutoProcessor.from_pretrained(base, do_image_splitting=False,
                                              size={"longest_edge": 384}, **kwargs)
    model = AutoModelForVision2Seq.from_pretrained(base, torch_dtype=torch.float32,
                                                   attn_implementation="eager", **kwargs)
    # ONE model, toggled. Loading two would double the RAM and, worse, leave the
    # room wondering whether both sides really got the same weights and the same
    # prompt. With disable_adapter() the adapter is the only variable.
    model = PeftModel.from_pretrained(model, str(adapter))
    model.eval()

    samples = load_samples(Path(args.dataset), args.split, args.n)
    print(f"base     : {base} @ {(revision or 'default')[:12]}")
    print(f"scoring  : {len(samples)} charts from {args.split}.jsonl, 2 generations each\n")

    results = []
    for index, sample in enumerate(samples, start=1):
        with model.disable_adapter():
            base_text = generate(model, processor, sample["image"], sample["instruction"], args.max_new_tokens)
        tuned_text = generate(model, processor, sample["image"], sample["instruction"], args.max_new_tokens)
        results.append({"base": base_text, "tuned": tuned_text})
        flag = "OK " if schema_ok(extract_json(tuned_text)) else "   "
        print(f"  [{index:3d}/{len(samples)}] {flag} {tuned_text[:70]!r}")

    base_metrics = score(results, "base", samples)
    tuned_metrics = score(results, "tuned", samples)

    causes = ENUMS["root_cause"]
    confusion = [[0] * len(causes) for _ in causes]
    for result, sample in zip(results, samples):
        predicted = extract_json(result["tuned"]) or {}
        if predicted.get("root_cause") in causes:
            confusion[causes.index(sample["truth"]["root_cause"])][causes.index(predicted["root_cause"])] += 1

    print(f"\n{'metric':14s} {'base':>8s} {'tuned':>8s}")
    for key in base_metrics:
        print(f"{key:14s} {base_metrics[key]:8.0%} {tuned_metrics[key]:8.0%}")

    scoreboard = out_dir / "eval_scoreboard.png"
    examples = out_dir / "eval_examples.png"
    plot_scoreboard(base_metrics, tuned_metrics, confusion, len(samples), scoreboard)
    plot_examples(samples, results, args.examples, examples)
    print(f"\n{scoreboard}\n{examples}")


if __name__ == "__main__":
    main()
