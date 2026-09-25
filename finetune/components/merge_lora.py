"""Step 2 - merge the LoRA adapter back into the base weights.

Why this is a SEPARATE step with a SEPARATE output rather than a flag on the
training job: the whole point of brick 4 is to put the two artifacts side by side
and show their size. ~10 MB of adapter against ~500 MB of merged weights is the
most legible slide in the demo, and you only get it if both exist as assets.

Runs on CPU. Merging is an addition, not a training run; renting a GPU for it
would be paying for nothing.
"""

import argparse
import shutil
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForVision2Seq, AutoProcessor


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_model", required=True)
    parser.add_argument("--adapter_input", required=True)
    parser.add_argument("--merged_output", required=True)
    return parser.parse_args()


def main():
    args = parse_args()
    merged_out = Path(args.merged_output)
    merged_out.mkdir(parents=True, exist_ok=True)

    # fp32 on the way in: merging in fp16 on CPU rounds the delta twice, and CPU
    # fp16 kernels are erratic. Cast to fp16 once, at save time.
    base = AutoModelForVision2Seq.from_pretrained(args.base_model, torch_dtype=torch.float32)
    model = PeftModel.from_pretrained(base, args.adapter_input)
    model = model.merge_and_unload()
    model = model.half()
    model.save_pretrained(str(merged_out), safe_serialization=True)

    # The processor must ship with the merged model: score.py loads both from
    # AZUREML_MODEL_DIR and has no access to the base asset at inference time.
    AutoProcessor.from_pretrained(args.base_model).save_pretrained(str(merged_out))

    for name in ("instruction.txt", "base_model.json"):
        candidate = Path(args.adapter_input) / name
        if candidate.exists():
            shutil.copy(candidate, merged_out / name)

    total = sum(p.stat().st_size for p in merged_out.rglob("*") if p.is_file())
    print(f"merged model written to {merged_out} ({total / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
