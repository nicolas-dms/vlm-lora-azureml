"""Diagnose a failed AML pipeline job without going through the flaky `az` CLI.

Why this exists
---------------
``ml_client.jobs.stream`` raises a ``JobException`` whose message is the *pipeline*
error - a wrapper that almost never names the real cause. The useful text lives in
the failed **child** job's ``user_logs/std_log.txt`` (or ``system_logs`` when the
failure happened before user code started, e.g. image pull or identity errors).

This script walks pipeline -> children -> failed child -> logs, and prints the tail
of every log it finds, newest last.

Usage
-----
    python scripts/66_job_diag.py                      # latest job of the experiment
    python scripts/66_job_diag.py --job <pipeline-job-name>
    python scripts/66_job_diag.py --tail 200
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
from pathlib import Path

from azure.ai.ml import MLClient
from azure.identity import DefaultAzureCredential
from dotenv import load_dotenv

# Same single source of configuration as the notebook: .env at the repo root.
# Real environment variables win, so CI does not need the file.
load_dotenv(Path(__file__).resolve().parent.parent / ".env", override=False)

try:
    SUBSCRIPTION = os.environ["AZURE_SUBSCRIPTION_ID"]
    RESOURCE_GROUP = os.environ["AZURE_RESOURCE_GROUP"]
    WORKSPACE = os.environ["AZUREML_WORKSPACE_NAME"]
except KeyError as missing:
    raise SystemExit(f"Missing {missing}. Copy .env.example to .env and fill it in - see README.md.")

EXPERIMENT = "maint-vlm-lora"

INTERESTING = ("std_log", "stderr", "error", "traceback", "execution-wrapper", "lifecycler")


def banner(text: str) -> None:
    print(f"\n{'=' * 78}\n{text}\n{'=' * 78}")


def dump_logs(client: MLClient, job_name: str, workdir: Path, tail: int) -> None:
    target = workdir / job_name
    shutil.rmtree(target, ignore_errors=True)
    target.mkdir(parents=True, exist_ok=True)
    try:
        client.jobs.download(name=job_name, download_path=str(target), all=True)
    except Exception as exc:  # noqa: BLE001 - we want the reason, not a stack
        print(f"  (could not download artifacts: {exc})")
        return

    logs = sorted(p for p in target.rglob("*") if p.is_file() and p.suffix in {".txt", ".log", ""})
    if not logs:
        print("  (no log files found)")
        return

    for log in logs:
        if not any(key in log.name.lower() or key in log.parent.name.lower() for key in INTERESTING):
            continue
        try:
            lines = log.read_text(encoding="utf-8", errors="replace").splitlines()
        except OSError:
            continue
        if not lines:
            continue
        banner(f"{log.relative_to(target)}  ({len(lines)} lines, last {min(tail, len(lines))})")
        print("\n".join(lines[-tail:]))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--job", default=None, help="pipeline job name; default = latest in the experiment")
    parser.add_argument("--experiment", default=EXPERIMENT)
    parser.add_argument("--tail", type=int, default=120)
    args = parser.parse_args()

    client = MLClient(DefaultAzureCredential(), SUBSCRIPTION, RESOURCE_GROUP, WORKSPACE)

    job_name = args.job
    if job_name is None:
        jobs = list(client.jobs.list(max_results=20))
        candidates = [j for j in jobs if getattr(j, "experiment_name", None) == args.experiment]
        if not candidates:
            print(f"No job found in experiment {args.experiment}. Recent jobs:")
            for j in jobs[:10]:
                print(f"  {j.name}  {getattr(j, 'experiment_name', '?')}  {j.status}")
            return 1
        job_name = candidates[0].name

    pipeline = client.jobs.get(job_name)
    banner(f"pipeline {pipeline.name}  status={pipeline.status}")
    print(pipeline.studio_url)

    workdir = Path(__file__).resolve().parent.parent / "tmp-joblogs"

    children = [j for j in client.jobs.list(parent_job_name=job_name)]
    if not children:
        print("\nNo child jobs - the pipeline failed before scheduling any step.")
        dump_logs(client, job_name, workdir, args.tail)
        return 0

    print("\nchildren:")
    for child in children:
        print(f"  {child.display_name or child.name:20s} {child.status:12s} {child.name}")

    failed = [c for c in children if c.status in {"Failed", "Canceled"}]
    if not failed:
        print("\nNo failed child. The failure is at pipeline level.")
        dump_logs(client, job_name, workdir, args.tail)
        return 0

    for child in failed:
        banner(f"LOGS for failed step: {child.display_name or child.name} ({child.status})")
        dump_logs(client, child.name, workdir, args.tail)

    return 0


if __name__ == "__main__":
    sys.exit(main())
