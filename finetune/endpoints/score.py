"""Scoring script for maint-vlm-ep.

Custom score.py rather than the MLflow no-code path, on purpose: the no-code
path stamps the asset as an mlflow_model and drags in its own serving stack
(pkg_resources, azureml-contrib-services), and it has nothing useful to say
about a VLM anyway.

Contract:
    in   {"image_b64": "<base64 PNG>", "instruction": "<optional override>"}
    out  {"work_order": {...}, "raw": "<model text>", "ok": true|false}
"""

import base64
import io
import json
import logging
import os

import torch
from PIL import Image
from transformers import AutoModelForVision2Seq, AutoProcessor

logger = logging.getLogger("maint-vlm")

_processor = None
_model = None
_instruction = ""


def _model_dir() -> str:
    """Find the folder that actually holds the weights.

    AZUREML_MODEL_DIR points at a wrapper directory whose layout depends on how
    the asset was registered; hardcoding a subfolder name works until the day it
    does not. Look for config.json instead.
    """
    root = os.environ["AZUREML_MODEL_DIR"]
    for dirpath, _, filenames in os.walk(root):
        if "config.json" in filenames:
            return dirpath
    raise RuntimeError(f"No config.json found under {root}")


def init():
    global _processor, _model, _instruction

    path = _model_dir()
    logger.info("loading model from %s", path)

    # Same processor settings as training. A VLM asked at 1536 px what it was
    # taught at 384 px does not degrade gracefully, it answers something else.
    _processor = AutoProcessor.from_pretrained(path, do_image_splitting=False, size={"longest_edge": 384})
    _model = AutoModelForVision2Seq.from_pretrained(path, torch_dtype=torch.float32, attn_implementation="eager")
    _model.eval()

    instruction_file = os.path.join(path, "instruction.txt")
    if os.path.exists(instruction_file):
        with open(instruction_file, encoding="utf-8") as handle:
            _instruction = handle.read()

    logger.info("ready")


def run(raw_data):
    payload = json.loads(raw_data)
    instruction = payload.get("instruction") or _instruction
    image = Image.open(io.BytesIO(base64.b64decode(payload["image_b64"]))).convert("RGB")

    messages = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": instruction}]}]
    text = _processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = _processor(text=[text], images=[[image]], return_tensors="pt")

    with torch.no_grad():
        generated = _model.generate(**inputs, max_new_tokens=128, do_sample=False)

    # Keep only the completion; the prompt is echoed back by generate().
    completion = generated[0][inputs["input_ids"].shape[1] :]
    answer = _processor.tokenizer.decode(completion, skip_special_tokens=True).strip()

    # A 256M model will sometimes emit almost-JSON. Report that honestly rather
    # than raising: a malformed answer is a demo talking point, not a 500.
    try:
        work_order = json.loads(answer)
        ok = True
    except json.JSONDecodeError:
        work_order = None
        ok = False

    return {"work_order": work_order, "raw": answer, "ok": ok}
