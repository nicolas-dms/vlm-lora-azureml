# Adapter portability

What the fine-tuning actually produces is not 500 MB of weights: it is **~10 MB of
delta**. The rest is the base model, which is public, versioned, and identical for
everyone.

## The contract

`maint-vlm-adapter` contains four things, and all four matter:

| file | role |
|---|---|
| `adapter_model.safetensors` | the LoRA weights |
| `adapter_config.json` | rank, alpha, target modules |
| `base_model.json` | **the repo and the SHA of the base model** |
| `instruction.txt` | the exact prompt it was trained on |

Without `base_model.json` the adapter is unusable: it is a delta, and a delta
without its starting point means nothing. Without `instruction.txt` you ask the
model a slightly different question from the one it was taught, and it answers
something else. Both files are written by `train_lora.py` - not left to
convention.

## Fetch the adapter

```powershell
az ml model download --name maint-vlm-adapter --version 1 `
  --download-path .\tmp-adapter `
  --resource-group $env:AZURE_RESOURCE_GROUP `
  --workspace-name $env:AZUREML_WORKSPACE_NAME

Get-ChildItem .\tmp-adapter -Recurse -File | Measure-Object Length -Sum
```

## Run it, outside Azure

```powershell
python finetune\export\run_local_adapter.py `
  --adapter .\tmp-adapter\maint-vlm-adapter `
  --image finetune\data\sample_chart.png
```

CPU, no GPU, no Azure subscription. Add `--no_adapter` to see what the base model
answers on the same image: chatty English prose instead of JSON. That is the
difference the fine-tuning bought.

## The other destinations

Not exercised in the demo, but all opened by the same two artifacts:

| target | what to ship | note |
|---|---|---|
| **vLLM** (anywhere) | the adapter | vLLM serves LoRA adapters hot, several per base model |
| **Ollama / llama.cpp** | the merged weights | GGUF conversion, then quantisation |
| **AI PC (NPU)** | the merged weights | ONNX export, local execution on Windows ML |
| **another AML registry** | either one | `az ml model create --registry-name ...` to share across workspaces |

The point is not that these four paths are equivalent. It is that none of them
requires redoing the fine-tuning.
