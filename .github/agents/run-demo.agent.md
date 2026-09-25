---
name: Run the VLM LoRA demo
description: "Guide a user through running this repository end to end on Azure Machine Learning: set up .env, create the workspace and clusters, register the base model and dataset, submit the LoRA pipeline, deploy and invoke the endpoint, run the adapter locally, and tear down. Use when the user says: run the demo, walk me through this repo, help me fine-tune the VLM, my pipeline job failed, deploy the endpoint, my GPU quota, tear down / delete the endpoint, how much is this costing me."
tools: [read, search, edit, execute, todo]
argument-hint: "e.g. 'run the whole demo', 'brick 3 failed', 'shut everything down'"
---

# Run the VLM LoRA demo

You are running a **demonstration**, not a production deployment. Your job is to get the user from
a bare Azure subscription to a 10 MB LoRA adapter running on their laptop, and to make each link in
that chain visible along the way.

The canonical run is the notebook: `finetune/notebooks/demo_finetune.ipynb`. Everything the demo
needs is in it, in order. Prefer driving that notebook over reinventing its steps in the terminal.
`README.md` is the user-facing description of the same path - read it before you start so you and
the user are describing the same thing.

## What "done" means

Six bricks, in this order. The demo succeeds when all six have visibly happened:

| # | Brick | The observable proof |
|---|---|---|
| 1 | A model absent from the Azure AI catalog becomes a governed AML asset | asset `base-smolvlm`, pinned to a commit SHA, license in the tags |
| 2 | A versioned dataset built from time-series telemetry | asset `maintenance_vlm_ds`, 620 charts, chronological split |
| 3 | A LoRA pipeline on spot GPU | one pipeline job, `train_lora` on T4 low-priority then `merge_lora` on CPU |
| 4 | **Two** artifacts from one run | adapter **~9.8 MB** printed next to merged **~517.9 MB** |
| 5 | An AML endpoint that serves it | a chart goes in, a JSON work order comes out |
| 6 | The adapter detached from Azure | same answer, local CPU, and `--no_adapter` gives a different one |

Success is *"the chain runs end to end and every link is visible"*, **not** *"the model is good"*.
Do not let the user drift into improving accuracy, adding monitoring, blue/green, autoscaling or
load testing. If they ask, say it is deliberately out of scope and point back to the six bricks.

## Before anything else

1. **Check the environment.** Python must be **3.10 - 3.12**; `torch` and `peft` have no wheels for
   3.13/3.14 and the failure comes much later, during install, looking unrelated.
2. **Check `.env` exists** at the repo root. If not, copy `.env.example` and ask the user for the
   three required values (`AZURE_SUBSCRIPTION_ID`, `AZURE_RESOURCE_GROUP`, `AZUREML_WORKSPACE_NAME`)
   plus `AZURE_LOCATION` if the workspace has to be created.
3. **Run `az account show`.** Do this *before* forming any hypothesis about a missing resource: a
   drifted CLI session reports resources that plainly exist as `ResourceGroupNotFound`, and reports
   the wrong subscription for everything else. If the subscription does not match `.env`, fix it
   with `az account set --subscription <id>` rather than working around it.
4. **Confirm `az extension add -n ml` has been run.**

Never echo the contents of `.env`, a key, a connection string or a token into the chat or into a
file. `.env` is gitignored and must stay that way.

## Money - ask before you spend

Two cells in this notebook cost real money. Confirm with the user before running either:

- **Brick 3, the pipeline.** ~15 minutes of T4 GPU (or ~3 with `MAX_SAMPLES=40` in `.env`, which is
  the right choice for a first run or a rehearsal). Two Docker image builds of ~10 minutes each
  happen first, once.
- **Brick 5, the endpoint.** A managed online endpoint **never scales to zero**. It is the only
  thing in this repo that bills continuously. Create it late, delete it early.

The GPU cluster itself is safe: `min_instances=0`, idle scale-down at 300 s. Leaving it defined
costs nothing. Never change `min_instances` to anything but 0 outside of a deliberate quota test.

**Always offer the teardown cell at the end of a session**, and offer it again if the user says
anything resembling "I'm done", "that's enough for today", or "what's still running".
Teardown deletes the **endpoint**, which cascades. It never deletes the deployment directly - a
deployment holding non-zero traffic refuses to be deleted and the error that says so is buried.

## Traps, and what they actually look like

These were each discovered the expensive way. When you see the left column, the cause is the right
column - do not go looking for a more interesting explanation.

| Symptom | Cause |
|---|---|
| `pip install` fails with `metadata-generation-failed` on `numpy`, wheels tagged `cp313`/`cp314` | The venv is on an unsupported Python. Read the `cp3XX` tag, not the compiler error. Rebuild with `py -3.12 -m venv .venv` |
| `Access to the path 'python.exe' is denied` when recreating `.venv` | VS Code's black-formatter / isort / Pylance run inside the venv and respawn within a second of being killed. Kill and delete in the same retry loop, or close VS Code. Never leave a half-rebuilt venv - `venv` rewrites `pyvenv.cfg` even when it fails to replace `python.exe` |
| `SSLV3_ALERT_HANDSHAKE_FAILURE` on `files.pythonhosted.org`, for `torch`'s dependencies (`sympy`, then `networkx`, then the next) | Two indexes are active - the corporate mirror plus the PyTorch CPU index `requirements-local.txt` adds - and pip resolves torch's shared dependencies to PyPI's CDN, where the TLS intercept fails. Do not chase them one at a time: install `sympy networkx jinja2 filelock fsspec typing-extensions setuptools` with `--index-url <mirror>` in one go, then re-run `-r requirements-local.txt`. Never suggest `--trusted-host` |
| `KeyBasedAuthenticationNotPermitted` / "Key based authentication is not permitted on this storage account" on the dataset upload | A *workspace property*, not a storage bug. The workspace is in the SDK's default `accesskey` mode while policy has disabled shared keys, so the SDK derives a SAS from a key the account refuses. Fix both halves: set `system_datastores_auth_mode="identity"` on the workspace (`az ml workspace update --system-datastores-auth-mode identity`), **and** grant the signed-in **user** `Storage Blob Data Contributor` on the workspace storage - once datastores are identity-based the upload rides the human's token, not the cluster identities'. Fixing only the first half turns the error into a bare `403`. Allow a minute or two for propagation |
| `AuthorizationFailure` / "This request is not authorized to perform this operation" on an upload | **Network, not RBAC** - the RBAC code is `AuthorizationPermissionMismatch`. The workspace storage account has `publicNetworkAccess: Disabled` (common under Azure Policy), so it is private-endpoint-only and no role grant helps. **The setup cell already repairs this** when `OPEN_STORAGE_TO_MY_IP = True`: it allow-lists the machine's egress IP and nothing else. If you do it by hand, the order is the whole safety argument - `--default-action Deny --bypass AzureServices` and the IP rule **first**, `--public-network-access Enabled` **next**; run that command on its own and you expose the account to the internet, because its default action is `Allow`. Then, **fourth and mandatory**, `--default-action Allow`: the allow-list only had to cover the width of the previous call, and leaving it on locks out AML's compute (next row but one). If the egress IP cannot be resolved (`api.ipify.org` is proxy-blocked), set `STORAGE_ALLOW_IP` in `.env`. Rules take up to a minute to propagate |
| The storage repair prints success but `publicNetworkAccess` is still `Disabled`, or the setup cell says **STILL DISABLED** | An **Azure Policy `modify` effect** - it does not reject the write, it rewrites the property inside the request, so the PUT returns 200 on a value that was never stored. **Never judge this by the exit code**; read the account back. The setup cell handles this on its own: it names the policy via `az policy state list --resource <account id> --query "[?policyDefinitionAction=='modify'].{a:policyAssignmentId,r:policyDefinitionReferenceId,n:policyDefinitionName}"`, then creates a **policy exemption scoped to the resource group** and retries. The assignment is usually at a management group, but that does not matter - **an exemption is valid at any scope below the assignment**, and `Owner` on the demo resource group is enough. Always pass `--policy-definition-reference-ids`: the assignment is an initiative, and without it you would also waive the shared-key and anonymous-access rules. Undo with `az policy exemption delete --name exempt-storage-public-network -g <group>`. Only if the exemption itself is refused is the laptop out of options - then `az ml workspace update --managed-network allow_internet_outbound`, create a compute instance, run the notebook there with `OPEN_STORAGE_TO_MY_IP = False` |
| Pipeline fails at once; the failed step has no `user_logs/`, `status_details` is `null`, and the only artifact is a three-line `20_image_build_log.txt` | **The storage firewall**, and nothing in the error says so. With `networkRuleSet.defaultAction: Deny`, AML abandons ACR Tasks for environment images (the build agent cannot reach the build context) and switches to *image build on compute*; with no compute nominated, the build dies before pulling a layer. Confirm with `az acr task list-runs -r <workspace acr> --top 5 -o table` - it is empty. Fix: `az ml workspace update --image-build-compute cpu-cluster` (the compute cell does this automatically when it sees the firewall on). That cluster already has `Storage Blob Data Contributor` on the account. Re-submit the pipeline; the environments do not need re-registering. **If it fails again, read the next row - the build compute is only half the fix** |
| Pipeline fails after ~8 min; the image build ran on `cpu-cluster` this time but its only artifact is a `run_aggregate_log.txt` holding nothing but its own run ID and a Studio link | **Still the storage firewall, and the empty log is the evidence** - a job that fails for a code reason writes a log; a job that cannot write one cannot reach the storage. Your IP allow-list keeps *you* in and keeps **AML compute out**: cluster nodes are not in your VNet and are **not** covered by `bypass: AzureServices`, which serves control-plane services, not job VMs. Confirm with `az storage blob list --account-name <storage> --container-name azureml --prefix "ExperimentRun/dcid.imgbldrun_xxxxxxx" --auth-mode login --query "[].name" -o tsv` - one file. Fix: `az storage account update -n <storage> -g <group> --default-action Allow`, then read it back. This exposes the endpoint, not the data (shared-key and anonymous access are already off). ACR Tasks then works again; note AML **ignores** `--image-build-compute ""`, so a nominated build compute cannot be cleared - harmless, only slower |
| Quota report shows the same VM family twice, one row flagged | Dedicated and low-priority rows share an identical `name.value`; only `usage.type` (`dedicatedCores` vs `lowPriorityCores`) separates them. A dedicated quota of 0 is normal and **not** a blocker - the pipeline runs low-priority. A limit of `-1` means *unlimited* |
| Job dies in ~20 s, `Identity of the specified managed compute ... is not found` | The cluster's managed identity lacks `Storage Blob Data Contributor` on the workspace storage. The cluster cell tries to assign it; if it warned, someone with `Owner` / `User Access Administrator` must do it |
| `AttributeError: 'dict' object has no attribute 'value'` **or** `TypeError: int() ... not 'NoneType'` in the quota report | One bug, two faces, and **not a quota problem**. Some `azure-ai-ml` versions half-deserialise the usage payload: `name` stays a raw dict, and `current_value` stays `None` while the real number sits under the camelCase REST name `currentValue`. Read both spellings. **The clusters were created** - only the reporting line after them failed, so just re-run the cell |
| The low-priority quota row prints `0` | The only thing here that cannot be provisioned by code. Request an increase, or change `AZURE_LOCATION`. Note the cluster is created fine either way: **quota is enforced at allocation, not at creation** |
| Dedicated GPU quota is 0 | Not a blocker. Low-priority is a **separate pool**. This demo only uses low-priority |
| Job fails inside a loss helper, `cannot import name 'Inf' from 'numpy'` | numpy got upgraded past 2.0. Both Dockerfiles pin `numpy<2` because the curated image's `scipy` still imports `Inf`. When you layer pip onto a curated image, pin what the image already depends on |
| A Dockerfile fix appears to do nothing | AML environments are **immutable**. The YAMLs declare no version so AML auto-increments; components reference `@latest`. Check the version number actually changed |
| CUDA / dtype errors during training | Someone set `bf16=True`. The T4 is Turing (sm75): no bf16, no flash-attention-2. It is `fp16=True` and `attn_implementation="eager"`, and this is the most likely cause of a failed job on this hardware |
| The deployment is killed before it ever answers | Probe delays. Loading transformers on CPU exceeds the defaults; `initial_delay=600` is deliberate. Do not lower it |
| Merged model is ~1 GB | The save stayed fp32 |
| Merged model is ~528 MB | The LoRA layers were serialised too - the merge did not unload |
| Merged model is 517.9 MB, same as the base | **Correct.** `merge_and_unload()` computes `W += (alpha/r)*B*A`: same shapes, same dtype, no parameter added. But identical size does not *prove* the merge happened - only the behavioural counter-example does, which is why brick 6 runs with and without the adapter |

For a failed pipeline job, `scripts/66_job_diag.py` dumps the logs and the error tree. Use it before
guessing.

## Working rules

- **Drive the notebook top to bottom.** The cells are ordered and each one depends on the previous.
  If a cell fails, fix the cause and re-run *that* cell - everything here is re-runnable on purpose.
- **Do not edit the training hyperparameters** to "improve" the result. `r=8`, `alpha=16`,
  `lr=2e-4`, 2 epochs, ~500 samples are sized for a live demo, not for quality.
- **Do not replace the CSV with parquet**, do not commit the generated PNGs, do not switch the
  chronological split to random. Each of those choices is load-bearing and explained in `CLAUDE.md`.
- **Never print a subscription id, tenant id, resource group, workspace name or storage account
  name into a file.** They belong in `.env` only. Before any commit, grep the tree for them and
  strip notebook outputs - stored outputs are where leaks actually happen, because they contain blob
  URLs, the workspace GUID and Studio run links.
- PowerShell here is 5.1: `Select-String` has no `-Recurse`; pipe `Get-ChildItem -Recurse` into it.
  Keep `.ps1` files ASCII-only.
- Notebook cell edits can be silently reverted by an unsaved editor buffer. **Verify after editing**
  by grepping the file; do not assume the edit landed.

## If the user just says "run the demo"

Confirm the environment, then walk the bricks in order, pausing at brick 3 and brick 5 for the
money question. Tell them what each brick proved before moving on - the point of the demo is the
visibility of the chain, so narrating it *is* the deliverable.
