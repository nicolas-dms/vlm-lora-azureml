"""Step 1 - LoRA fine-tuning of a VLM on chart images -> JSON work orders.

Runs on a single T4 (Standard_NC4as_T4_v3, low priority). Every choice below that
looks arbitrary is a T4 constraint; the comments say which.

Scope reminder: this demo proves the chain runs end to end. It is NOT tuned for
output quality, and 2 epochs on ~500 samples of a 256M model will not make a
great model. That is the intended trade.
"""

import argparse
import importlib.util
import json
import shutil
from pathlib import Path

import torch
from PIL import Image
from peft import LoraConfig, get_peft_model
from torch.utils.data import Dataset
from transformers import (
    AutoModelForVision2Seq,
    AutoProcessor,
    Trainer,
    TrainerCallback,
    TrainingArguments,
)

# The projections we attach LoRA to. Note these names also exist inside the
# SigLIP vision tower, which is why the resolver below filters on the module
# path and not on the suffix alone - see text_lora_targets().
PROJECTION_SUFFIXES = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")


class LossToMlflow(TrainerCallback):
    """Log only the metrics we actually want, and nothing else.

    We deliberately do NOT use report_to=["mlflow"]. Transformers' own
    MLflowCallback dumps the entire TrainingArguments *and* the model config as
    MLflow params on_train_begin. For a VLM, model.config contains nested
    vision_config / text_config blocks whose string form runs to thousands of
    characters. Transformers filters values longer than mlflow's own limit
    (6000 chars in current releases); the AML tracking server enforces 500. The
    two disagree, so the very first log_batch is rejected:

        RestException: INVALID_PARAMETER_VALUE: No more than 500 characters per
        params Value. Request contains 2 of greater length.

    and the exception propagates out of trainer.train() - the run dies on
    telemetry, after the model is loaded and the data is ready. Logging the loss
    ourselves is three lines and has no such failure mode.
    """

    def on_log(self, args, state, control, logs=None, **kwargs):
        if not logs or not state.is_world_process_zero:
            return
        import mlflow

        for key in ("loss", "learning_rate", "grad_norm", "epoch"):
            if isinstance(logs.get(key), (int, float)):
                mlflow.log_metric(key, float(logs[key]), step=state.global_step)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_model", required=True, help="mounted base-smolvlm asset")
    parser.add_argument("--dataset", required=True, help="mounted maintenance_vlm_ds asset")
    parser.add_argument("--adapter_output", required=True)
    parser.add_argument("--epochs", type=float, default=2.0)
    parser.add_argument("--lora_r", type=int, default=8)
    parser.add_argument("--lora_alpha", type=int, default=16)
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    parser.add_argument("--learning_rate", type=float, default=2e-4)
    parser.add_argument("--grad_accum", type=int, default=8)
    parser.add_argument("--image_px", type=int, default=384)
    parser.add_argument("--max_samples", type=int, default=0, help="0 = use the whole train split")
    return parser.parse_args()


class ChartDataset(Dataset):
    """One jsonl line -> one (image, messages) pair. No preprocessing here.

    The processor is applied in the collator instead, because batching images
    and text together is exactly what an image processor needs to do at once.
    """

    def __init__(self, root: Path, split: str, max_samples: int = 0):
        self.root = root
        manifest = root / f"{split}.jsonl"
        if not manifest.is_file():
            # A mount-layout mistake otherwise surfaces as a bare FileNotFoundError,
            # 30 minutes and one GPU node after submission, naming a path that looks
            # perfectly plausible. Saying what IS at the mount point turns that into
            # a one-glance diagnosis.
            found = sorted(p.name + ("/" if p.is_dir() else "") for p in root.iterdir())
            raise FileNotFoundError(
                f"{manifest} not found. --dataset resolved to {root}, which contains: {found}"
            )
        lines = manifest.read_text(encoding="utf-8").splitlines()
        self.rows = [json.loads(line) for line in lines if line.strip()]
        if max_samples:
            self.rows = self.rows[:max_samples]

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        image = Image.open(self.root / row["image"]).convert("RGB")
        return {"image": image, "messages": row["messages"]}


class ChartCollator:
    """Applies the chat template AND pushes the images through the processor.

    Doing only one of the two is the classic VLM fine-tuning bug: the text then
    contains an <image> placeholder that never gets expanded into visual tokens,
    the model trains on a caption it cannot see, and the loss still goes down.
    """

    def __init__(self, processor):
        self.processor = processor
        self.pad_id = processor.tokenizer.pad_token_id
        self.image_id = processor.tokenizer.convert_tokens_to_ids("<image>")

    def __call__(self, examples):
        texts = [
            self.processor.apply_chat_template(e["messages"], tokenize=False, add_generation_prompt=False)
            for e in examples
        ]
        images = [[e["image"]] for e in examples]
        batch = self.processor(text=texts, images=images, return_tensors="pt", padding=True)

        labels = batch["input_ids"].clone()
        # Padding and image placeholders must not contribute to the loss.
        labels[labels == self.pad_id] = -100
        labels[labels == self.image_id] = -100
        batch["labels"] = labels
        return batch


def text_lora_targets(model) -> list[str]:
    """Return the fully-qualified names of the text-backbone projections only.

    Passing the bare suffixes to LoraConfig would also match the SigLIP vision
    tower's q/k/v_proj, i.e. it would silently train the image encoder - the one
    thing the plan says to freeze, and the fastest route to OOM on 16 GB.
    Resolving the names at runtime also means we do not depend on transformers
    keeping the exact module layout of Idefics3 stable.
    """
    targets = []
    for name, module in model.named_modules():
        if "vision" in name:
            continue
        if isinstance(module, torch.nn.Linear) and name.split(".")[-1] in PROJECTION_SUFFIXES:
            targets.append(name)
    if not targets:
        raise RuntimeError("No text-backbone projection matched; the model layout changed.")
    return targets


def main():
    args = parse_args()
    dataset_root = Path(args.dataset)
    adapter_out = Path(args.adapter_output)
    adapter_out.mkdir(parents=True, exist_ok=True)

    # do_image_splitting=False is not an optimisation, it is the memory budget.
    # SmolVLM tiles a large image into sub-images and each tile costs a full
    # block of visual tokens; on a T4 the 30-min charts fit only untiled.
    processor = AutoProcessor.from_pretrained(
        args.base_model,
        do_image_splitting=False,
        size={"longest_edge": args.image_px},
    )
    processor.tokenizer.padding_side = "right"

    # Load in fp32 and let the Trainer's AMP do fp16. Loading the weights in
    # fp16 AND setting fp16=True is the "Attempting to unscale FP16 gradients"
    # crash: the grad scaler needs fp32 master weights to unscale into.
    #
    # attn_implementation="eager": the T4 is Turing (sm75). No flash-attention-2,
    # and no bf16 either - bf16=True copied from an A100 tutorial is the single
    # most likely way to make this job fail.
    model = AutoModelForVision2Seq.from_pretrained(
        args.base_model,
        torch_dtype=torch.float32,
        attn_implementation="eager",
    )
    model.config.use_cache = False  # incompatible with gradient checkpointing

    targets = text_lora_targets(model)
    print(f"LoRA targets: {len(targets)} text modules, 0 vision modules")

    model = get_peft_model(
        model,
        LoraConfig(
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            target_modules=targets,
            bias="none",
            task_type="CAUSAL_LM",
        ),
    )
    model.print_trainable_parameters()

    train_ds = ChartDataset(dataset_root, "train", args.max_samples)
    print(f"train samples: {len(train_ds)}")

    training_args = TrainingArguments(
        output_dir="./outputs",
        num_train_epochs=args.epochs,
        # Batch 1 + accumulation 8: the memory goes into visual tokens, not
        # weights. A 256M model is tiny; its image sequence is not.
        per_device_train_batch_size=1,
        gradient_accumulation_steps=args.grad_accum,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        learning_rate=args.learning_rate,
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,
        fp16=True,
        bf16=False,
        logging_steps=5,
        save_strategy="no",
        # Mandatory with a custom Dataset that yields PIL images: otherwise the
        # Trainer drops every column it does not recognise and hands the
        # collator empty dicts.
        remove_unused_columns=False,
        dataloader_num_workers=2,
        # "none", not "mlflow" - see LossToMlflow above.
        report_to=[],
    )

    callbacks = [LossToMlflow()] if importlib.util.find_spec("mlflow") else []
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        data_collator=ChartCollator(processor),
        callbacks=callbacks,
    )
    result = trainer.train()
    print(f"final train loss: {result.training_loss:.4f}")

    # Adapter only: adapter_config.json + adapter_model.safetensors, ~10 MB
    # (2.44M trainable params in fp32).
    model.save_pretrained(str(adapter_out))

    # The prompt the model was trained against travels with the adapter. Without
    # it, whoever downloads these 10 MB has to guess the instruction, and a VLM
    # answers a different question when asked differently.
    manifest_path = dataset_root / "manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        (adapter_out / "instruction.txt").write_text(manifest["instruction"], encoding="utf-8")

    # The base model's identity IS the adapter's contract: 10 MB of LoRA weights
    # are meaningless without knowing which weights they are a delta of.
    source = Path(args.base_model) / "hf_source.json"
    if source.exists():
        shutil.copy(source, adapter_out / "base_model.json")

    print(f"adapter written to {adapter_out}")
    for item in sorted(adapter_out.iterdir()):
        print(f"  {item.name:40s} {item.stat().st_size / 1e6:8.2f} MB")


if __name__ == "__main__":
    main()
