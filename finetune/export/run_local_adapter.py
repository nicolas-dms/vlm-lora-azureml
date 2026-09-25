"""Run the fine-tuned model OUTSIDE Azure, from the 10 MB adapter alone.

This is the closing argument of the demo: what you actually own after the
fine-tuning is a small, portable delta. Point it at any copy of the base weights
- a laptop, a CPU VM, a vLLM server, another cloud - and you get your model back.

    python run_local_adapter.py --adapter ./tmp-adapter --image ../data/sample_chart.png

--base defaults to the Hugging Face repo id, which is the honest default here:
offline, the machine running this script has no access to the AML asset. The SHA
recorded in the adapter's base_model.json is used automatically when present, so
"any copy of the base weights" means the RIGHT copy.
"""

import argparse
import json
from pathlib import Path

import torch
from PIL import Image
from peft import PeftModel
from transformers import AutoModelForVision2Seq, AutoProcessor

DEFAULT_BASE = "HuggingFaceTB/SmolVLM-256M-Instruct"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--adapter", required=True, help="folder downloaded with az ml model download")
    parser.add_argument("--image", required=True)
    parser.add_argument("--base", default=None, help="override the base model path or repo id")
    parser.add_argument("--instruction", default=None)
    parser.add_argument("--no_adapter", action="store_true", help="run the BASE model, for the before/after shot")
    return parser.parse_args()


def main():
    args = parse_args()
    adapter = Path(args.adapter)

    contract = adapter / "base_model.json"
    revision = None
    base = args.base or DEFAULT_BASE
    if contract.exists() and args.base is None:
        info = json.loads(contract.read_text(encoding="utf-8"))
        base, revision = info["repo"], info.get("revision")

    instruction = args.instruction
    if instruction is None and (adapter / "instruction.txt").exists():
        instruction = (adapter / "instruction.txt").read_text(encoding="utf-8")
    if not instruction:
        raise SystemExit("No instruction found; pass --instruction.")

    print(f"base     : {base} @ {revision or 'default'}")
    print(f"adapter  : {'(none - base model only)' if args.no_adapter else adapter}")

    kwargs = {"revision": revision} if revision else {}
    processor = AutoProcessor.from_pretrained(base, do_image_splitting=False, size={"longest_edge": 384}, **kwargs)
    model = AutoModelForVision2Seq.from_pretrained(
        base, torch_dtype=torch.float32, attn_implementation="eager", **kwargs
    )

    if not args.no_adapter:
        model = PeftModel.from_pretrained(model, str(adapter))
    model.eval()

    image = Image.open(args.image).convert("RGB")
    messages = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": instruction}]}]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=[text], images=[[image]], return_tensors="pt")

    with torch.no_grad():
        generated = model.generate(**inputs, max_new_tokens=128, do_sample=False)

    completion = generated[0][inputs["input_ids"].shape[1] :]
    print("-" * 60)
    print(processor.tokenizer.decode(completion, skip_special_tokens=True).strip())
    print("-" * 60)


if __name__ == "__main__":
    main()
