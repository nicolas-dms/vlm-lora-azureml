"""Pull the base VLM from the Hugging Face Hub, once, pinned to a SHA.

This is brick 1 of the demo: a model that is NOT in the Azure AI model catalog
becomes a governed AML asset.

Two rules are load-bearing here and neither is cosmetic:

1. Download once, register as an asset, mount it in the job. The tempting
   alternative - calling from_pretrained("HuggingFaceTB/SmolVLM-256M-Instruct")
   inside the training script - makes every run depend on the Hub being up, on
   the training subnet having egress, and on the repo not having moved under us.
   AML compute has no guaranteed internet access; that single line is the most
   common reason a fine-tuning job dies at minute zero.

2. Pin the revision by commit SHA, never "main". "main" means the asset you
   registered on Monday and the asset you retrain from on Friday can be
   different weights under the same name, and nothing in the registry will say
   so. The SHA is what makes "base_model=base-smolvlm:1" an actual contract.

Usage:
    python fetch_base_model.py --out ./tmp-base-model
"""

import argparse
import json
from pathlib import Path

from huggingface_hub import HfApi, snapshot_download

DEFAULT_REPO = "HuggingFaceTB/SmolVLM-256M-Instruct"

# Take the safetensors weights, the processor and the configs. Skip the .bin
# duplicates (same weights, legacy pickle format) and the onnx/ folder: they
# would triple the asset size for nothing.
ALLOW_PATTERNS = ["*.safetensors", "*.json", "*.txt", "*.model", "*.jinja"]
IGNORE_PATTERNS = ["*.bin", "*.msgpack", "*.h5", "onnx/*", "runs/*"]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", default=DEFAULT_REPO)
    parser.add_argument("--out", required=True, help="local folder to download into")
    parser.add_argument(
        "--revision",
        default=None,
        help="commit SHA to pin. Omit to resolve the current head of main ONCE and pin that.",
    )
    return parser.parse_args()


def fetch(repo: str = DEFAULT_REPO, out: str = "./tmp-base-model", revision: str | None = None) -> dict:
    """Download the snapshot and return {repo, revision, path, size_mb}."""
    out_path = Path(out)

    # Resolve the branch to an immutable SHA before downloading anything, so the
    # thing we record and the thing we download cannot drift apart.
    revision = revision or HfApi().model_info(repo).sha

    local = Path(
        snapshot_download(
            repo_id=repo,
            revision=revision,
            local_dir=str(out_path),
            allow_patterns=ALLOW_PATTERNS,
            ignore_patterns=IGNORE_PATTERNS,
        )
    )

    size_mb = sum(p.stat().st_size for p in local.rglob("*") if p.is_file()) / 1e6
    info = {"repo": repo, "revision": revision, "path": str(local.resolve()), "size_mb": round(size_mb, 1)}

    # This file travels with the weights and, later, with the adapter. It is what
    # turns "a LoRA delta" into "a LoRA delta OF something identifiable".
    (local / "hf_source.json").write_text(json.dumps(info, indent=2), encoding="utf-8")
    return info


def main():
    args = parse_args()
    info = fetch(args.repo, args.out, args.revision)

    files = sorted(p.name for p in Path(info["path"]).rglob("*") if p.is_file())
    print(f"repo     : {info['repo']}")
    print(f"revision : {info['revision']}")
    print(f"files    : {len(files)} -> {files}")
    print(f"size     : {info['size_mb']:.1f} MB")
    # Machine-readable last line, so a shell wrapper can pick the SHA up without
    # parsing prose.
    print(f"REVISION={info['revision']}")


if __name__ == "__main__":
    main()
