# FineTuningAgent

Maintainer instructions for this repository. Read this before touching anything.
The companion file `.github/agents/run-demo.agent.md` is the *user-facing* runbook
(symptom -> cause -> command); this file is the *contributor* guardrail.

## Where this pushes

- Remote: **`https://github.com/nicolas-dms/vlm-lora-azureml`** (`origin`), branch **`main`**,
  **public**. Authenticated with the GitHub CLI (`gh auth status` to check).
- **The repo is public, so every push is irreversible.** Run the two checks under
  *Secrets and identifiers* BEFORE `git push`, every time - not once, at publication.
- Never `git push --force`, never amend a commit that is already on `origin/main`.

## What this repo is

A **self-contained demonstration** that you can take a vision-language model that is **absent from the
Azure AI model catalog**, fine-tune it with LoRA on Azure Machine Learning, and end up with a 10 MB
artifact that runs anywhere.

It was extracted from a larger private two-track demo. **Every reference to that origin has been
removed on purpose** (track A/B, phase IDs `F0`-`F5`, Fabric, Eventhouse, a specific subscription).
Do not reintroduce them.

## Scope discipline - the rule most easily broken

The subject is **the six bricks and their end-to-end integration. Nothing else.**

| # | Brick |
|---|---|
| 1 | A model absent from the catalog becomes a governed AML asset, pinned to a commit SHA, license in the tags |
| 2 | A versioned dataset, built from time-series telemetry |
| 3 | A LoRA pipeline on spot GPU |
| 4 | **Two** artifacts from one run - adapter and merged - shown side by side **with their size** |
| 5 | An AML endpoint that serves it |
| 6 | The adapter detached and run outside Azure |

**Explicitly out of scope:** fine-tuning quality, scalability, endpoint robustness, monitoring,
blue/green, autoscaling, load testing. Success means *"the chain runs end to end and every link is
visible"*, not *"the model is good"*.

If a proposed change does not serve one of those six bricks, it does not belong here. When in doubt,
choose the smaller model, the smaller dataset, the shorter path.

## Secrets and identifiers - non-negotiable

This repo is intended to be **public**. Before any commit:

- **No subscription ID, tenant ID, resource group, workspace name, storage account name, or local
  `C:\Users\...` path.** Configuration comes from **one gitignored `.env`** at the repo root
  (`AZURE_SUBSCRIPTION_ID` / `AZURE_RESOURCE_GROUP` / `AZUREML_WORKSPACE_NAME`, plus the optional
  `ENDPOINT_NAME` / `MAX_SAMPLES` / `AZURE_STORAGE_ACCOUNT`). `.env.example` is the committed
  template. Real environment variables always win, so CI needs no file.
- **Strip notebook outputs before committing.** This is where leaks actually happen: stored outputs
  contain blob URLs, the workspace GUID, Studio run links and local paths. Cleaning the source is not
  enough. Verify with a grep over the whole tree, not by eye.
- Never print or commit keys, connection strings or tokens. Prefer Managed Identity + RBAC.

### The two pre-push checks

Run both, every time, before `git push`. Matches inside `.env` and `.venv/` are expected - they are
gitignored. A match anywhere else stops the push.

```powershell
# 1. identifiers - fill the alternation from YOUR .env, plus your egress IP and username
Get-ChildItem -Recurse -File | Select-String -Pattern "<sub-id>|<tenant-id>|<rg>|<workspace>|<storage>|<your-ip>|<username>"

# 2. notebook outputs - every code cell must report 0 outputs and a null execution count
$j = Get-Content finetune\notebooks\demo_finetune.ipynb -Raw | ConvertFrom-Json
$j.cells | Where-Object { $_.cell_type -eq 'code' } | ForEach-Object { "{0} outputs exec={1}" -f $_.outputs.Count, $_.execution_count }
```

`scripts/strip_notebook_outputs.py` does the stripping when check 2 fails.

## Layout

```
finetune/
  data/         telemetry_sample.csv  <- 24 h of 4 machines, the committed input
                build_dataset.py, render_charts.py, labeling_rules.py, schema.json,
                build_dataset.kql (optional ADX source), sample_chart.png
  models/       fetch_base_model.py, base-model-asset.yml
  environments/ Dockerfile.train, Dockerfile.serve, vlm-lora-env.yml, vlm-serve-env.yml
  components/   train_lora.{py,yml}, merge_lora.{py,yml}
  pipelines/    finetune_pipeline.yml
  endpoints/    endpoint.yml, deployment.yml, score.py
  export/       run_local_adapter.py, eval_visual.py, README-portability.md
  notebooks/    demo_finetune.ipynb   <- the demo, runs bricks 1-6 in order
scripts/        60_gpu_cluster.ps1, 61_vlm_dataset.ps1, 66_job_diag.py
```

## Hard-won technical rules - do not relitigate

Each of these cost real time to discover. Changing one requires evidence, not a tutorial.

### Dataset

- The input is **`finetune/data/telemetry_sample.csv`** - 5,696 rows, 4 machines, 24 hours, six
  columns: `minute, device_id, temperature_c, humidity_pct, pressure_bar, vibration_mm_s`.
  CSV, not parquet, on purpose: a reader can open it, see what the columns contain, and edit a value
  to watch the label change. Parquet is smaller and completely opaque.
- **The 620 chart PNGs are not committed.** Windowing, labelling and index assignment are fully
  deterministic, so the images are a pure function of the CSV - `build_dataset.py` regenerates them
  byte-identically in under a minute. Committing them would add megabytes that can never drift out of
  sync with their source, only out of sync with the code that draws them.
- **Chronological** 70/10/20 split, windows straddling a boundary dropped. A random split would leak
  near-identical charts between train and test, because windows overlap (5-min stride, 30-min window),
  and the evaluation would be worthless.
- Fixed y-limits on every chart. The labelling rules are absolute ("temperature >= 40 degC"), so
  autoscaled axes would make the same visual height mean different values and the task unlearnable.

### Base model

- `HuggingFaceTB/SmolVLM-256M-Instruct`, Apache-2.0, **ungated**. Never pick a gated repo: the
  license click cannot be automated.
- **Never `from_pretrained("org/repo")` inside a training job.** AML compute has no guaranteed
  internet access. Download once, register as a `custom` model asset, mount it as a pipeline input.
- **Pin by commit SHA, never `main`.** Otherwise two assets with the same name can be different
  weights and nothing in the registry says so.

### Compute

- `Standard_NC4as_T4_v3`, **`--tier low_priority`**, `min_instances=0`, `max_instances=1`, idle 300 s.
- Dedicated GPU quota is commonly **0** on modern families. **Low-priority is a separate quota pool
  and it works.** A dedicated quota of 0 does not mean "no GPU".
- **Creating an AmlCompute proves nothing** - quota is enforced at *allocation*. The only decisive
  test is `--min-instances 1` then `az ml compute list-nodes`. Scale straight back to 0, it bills.
- The cluster **must** have `--identity-type SystemAssigned` plus `Storage Blob Data Contributor` on
  the workspace storage when the workspace uses `--system-datastores-auth-mode identity`. Without it
  every job dies in ~20 s with `UserError: Identity of the specified managed compute ... is not found`.

### Storage networking - the allow-list is scaffolding, not the destination

- An IP allow-list on the workspace storage (`defaultAction: Deny` + `ipRules: [<your IP>]`) unblocks
  *your laptop* and locks out **AML's own compute**. Cluster nodes are not in your VNet and they are
  **not** covered by `bypass: AzureServices`.
- The signature is brutal because there is nothing to read: the job fails after ~8 minutes and the
  image build leaves **one blob**, a `run_aggregate_log.txt` holding only its own run ID and a Studio
  link. **An empty log is itself the evidence** - a job that fails for a code reason writes a log.
  Confirm with `az storage blob list --container-name azureml --prefix "ExperimentRun/dcid.<imgbldrun>"
  --auth-mode login`.
- So the repair has **four** steps, not three, and the fourth is mandatory:
  `az storage account update ... --default-action Allow`. **Then drop the allow-list. It is
  scaffolding, not the destination.** This is safe here only because tenant policy already disables
  shared-key auth and anonymous blob access, so the data plane still demands an Entra token plus RBAC.

### Environments

- Build **on top of the curated images**, via a Dockerfile - never a conda file. A conda file creates
  a new environment inside the image and silently shadows the torch/CUDA build the ACPT image exists
  to provide.
- **Pin `numpy<2`.** Nothing in `transformers`/`peft` requires numpy 2 and nothing forbids it, so pip
  upgrades the image's numpy 1.x, and the image's `scipy` still does `from numpy import Inf` - an
  alias deleted in numpy 2.0. The job then dies inside an unrelated loss helper, 25 minutes and one
  GPU node later. **When you layer pip onto a curated image, pin what the image already depends on.**
- Environments are **immutable**. Re-registering the same version with a changed Dockerfile keeps
  serving the old image, and the fix looks like it did nothing. The YAMLs therefore declare **no
  version** (AML auto-increments) and components reference `@latest`.
- `num2words` is a hard, undeclared import of the Idefics3/SmolVLM processor.
- When the storage firewall is closed, AML abandons ACR Tasks and the build never starts - the only
  artifact is a three-line `20_image_build_log.txt`. Nominating `--image-build-compute cpu-cluster`
  makes the build run, **but nominating a build compute is a workaround for a state that must not
  exist**: fix the firewall instead. Note that AML **silently ignores** `--image-build-compute ""`
  (exit 0, value unchanged), so once set it stays set - harmless, only slower than ACR Tasks.

### Training - the T4 constraints are not negotiable

- **`fp16=True`, never `bf16`**, and `attn_implementation="eager"`. The T4 is Turing (sm75): no bf16,
  no flash-attention-2. *A `bf16=True` copied from an A100 tutorial is the single most likely cause
  of a failed job here.*
- Freeze the vision encoder. LoRA on the text backbone projections only - resolve target modules at
  runtime and **exclude anything under `vision`**, otherwise the `q_proj`/`k_proj`/`v_proj` suffixes
  also capture the SigLIP tower.
- Memory goes into **visual tokens, not weights**: `per_device_train_batch_size=1`,
  `gradient_accumulation_steps=8`, `gradient_checkpointing=True`, `do_image_splitting=False`,
  `size={"longest_edge": 384}`.
- `remove_unused_columns=False`, and `report_to=[]` (not `["mlflow"]`).
- `r=8`, `alpha=16`, `dropout=0.05`, `lr=2e-4` cosine, 2 epochs, ~500 samples. Sized for speed.

### Artifacts - always two, never one

- `maint-vlm-adapter` = **~9.8 MB** (2,442,240 params in fp32). `maint-vlm-merged` = **517.9 MB**.
- **The merged model is byte-for-byte the size of the base, and that is correct.**
  `merge_and_unload()` computes `W += (alpha/r)·B·A` - same shapes, same dtype, no parameter added.
  Failure signatures: **~1 GB** means the save stayed fp32; **~528 MB** means the LoRA layers were
  serialised too, i.e. the merge did not unload.
  **Corollary: identical size does NOT prove the merge happened.** Only the behavioural
  counter-example (`run_local_adapter.py --no_adapter`) proves it.
- Cross-tag both assets with `base_model`, `base_repo`, `base_revision`, `dataset`, `lora_r`,
  `lora_alpha`. An adapter without its base is meaningless - **that tag is the artifact's contract**.
- `train_lora.py` also writes `base_model.json` and `instruction.txt` into the adapter folder. Both
  are load-bearing: a different prompt at inference gets a different answer.

### Serving

- Model assets are type **`custom` with an explicit `score.py`**. Do **not** reach for the MLflow
  no-code path: it stamps the asset as `mlflow_model` and drags in its own serving stack
  (`pkg_resources`, `azureml-contrib-services`). That path was the source of every serving failure.
- One endpoint, one deployment, **CPU** (`Standard_DS3_v2`). At 256M the CPU is enough and it removes
  the 24/7 GPU cost question entirely.
- Probes must be generous (`initial_delay=600`): loading transformers on CPU exceeds the defaults, and
  a perfectly healthy deployment gets killed before it ever answers once.

## Cost

- A managed online endpoint **never scales to zero**. It is the only thing here that bills
  continuously. Create it late, delete it early.
- **Teardown deletes the ENDPOINT, never the deployment.** A deployment holding non-zero traffic
  refuses to be deleted, and the only useful sentence - *"Can't delete deployment with non-zero
  traffic weight"* - is buried three levels down in the error. Deleting the endpoint cascades.
- The teardown cell must be **idempotent** (list before deleting) and **self-contained** (redefine the
  endpoint name locally), because it is the cell people run alone, on a fresh kernel, after realising
  they forgot to switch things off. A teardown that raises when there is nothing to delete is broken.
- GPU cluster always `min_instances=0`.

## Tooling notes

- **There is no Azure ML MCP server.** Workspace, compute, datastore, data asset, pipeline job, model
  registry and online endpoint operations go through `az ml` CLI v2 or the `azure-ai-ml` SDK.
- `InvalidAuthenticationTokenTenant` from an Azure tool means the `az` CLI session drifted, not that
  the tool is broken. The same drift produces a bogus `ResourceGroupNotFound` on resources that exist.
  **Reflex: `az account show` before forming any other hypothesis.**
- PowerShell here is 5.1: `Select-String` has **no** `-Recurse`. Pipe `Get-ChildItem -Recurse` into it.
- Keep `.ps1` files ASCII-only.
- Notebook cell edits can be silently reverted by an unsaved editor buffer. **Verify after editing**,
  by grepping the file - do not assume the edit landed.

## Style

- Code, identifiers, comments and file content in **English**.
- Comments explain *why*, especially where the code looks odd. Most comments in this repo document a
  trap; deleting them re-arms it.
- Every script must be safely re-runnable. The demo gets replayed live.

## Open work

- [x] `README.md`, `LICENSE` (MIT), `.env.example`, `requirements.txt` / `requirements-local.txt`.
- [x] The notebook is self-sufficient: it creates both clusters, builds and registers the dataset on
      `workspaceblobstore`, and reads everything from `.env`. `scripts/` is optional.
- [x] `scripts/61_vlm_dataset.ps1` documented as the optional ADLS Gen2 variant; the common path no
      longer needs it.
- [x] The two evaluation PNGs live in `docs/` and are shown in the README. Notebook outputs are
      stripped, so without them the results are invisible to a reader. **They are the only build
      artifacts that are committed** - regenerate them with the `eval_visual.py` cell and re-copy
      them out of `tmp-eval/` when the numbers change.
- [x] Published at `nicolas-dms/vlm-lora-azureml`. The identifier grep and the notebook-output
      check are now **pre-push rituals**, not one-off tasks: run both before every push.

