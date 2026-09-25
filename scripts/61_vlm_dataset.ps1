# OPTIONAL - the ADLS Gen2 variant of the dataset upload.
#
# You do not need this script to run the demo: the notebook builds the dataset and
# registers it on workspaceblobstore through the SDK, with no extra prerequisite.
# Use this one only when the dataset must live in an existing ADLS Gen2 filesystem
# that is already registered as an AML datastore. It then needs, in .env:
#   AZURE_STORAGE_ACCOUNT   the ADLS Gen2 account
# and -FileSystem / -Datastore matching your setup.
#
# The dataset is built locally (finetune/data/build_dataset.py) from the committed
# telemetry CSV, which costs nothing. This script is the only step that touches
# Azure: it pushes images/ + *.jsonl into the ADLS filesystem and declares the
# folder as a versioned AML data asset so the training pipeline can mount it.
#
# Idempotent: re-uploading overwrites the same paths; an existing asset version is
# reported and left alone (AML data asset versions are immutable by design - bump
# -Version to publish a new one).

[CmdletBinding()]
param(
    [string] $ResourceGroup,
    [string] $Workspace,
    [string] $StorageAccount,
    [string] $FileSystem     = 'snapshots',
    [string] $Datastore      = 'snapshots',
    [string] $LocalPath      = '',
    [int]    $Version        = 1,
    [string] $AssetName      = 'maintenance_vlm_ds'
)

$ErrorActionPreference = 'Stop'

# Parameter defaults are bound BEFORE the body runs, so "= $env:X" in the param
# block would read the variables as they were before .env was loaded.
. "$PSScriptRoot\load_dotenv.ps1"
if (-not $ResourceGroup) { $ResourceGroup = $env:AZURE_RESOURCE_GROUP }
if (-not $Workspace) { $Workspace = $env:AZUREML_WORKSPACE_NAME }
if (-not $StorageAccount) { $StorageAccount = $env:AZURE_STORAGE_ACCOUNT }
if (-not $LocalPath) { $LocalPath = Join-Path (Split-Path $PSScriptRoot -Parent) 'tmp-dataset' }

if (-not $ResourceGroup -or -not $Workspace -or -not $StorageAccount) {
    throw 'Fill in .env at the repo root (including AZURE_STORAGE_ACCOUNT), or pass the values explicitly.'
}

function Invoke-Az {
    param([string[]] $Arguments)
    Write-Host "az $($Arguments -join ' ')" -ForegroundColor DarkGray
    $output = & az @Arguments 2>&1
    if ($LASTEXITCODE -ne 0) {
        throw "az failed ($LASTEXITCODE): $($output | Out-String)"
    }
    return ($output | Out-String)
}

function Invoke-AzJson {
    param([string[]] $Arguments)
    $text = Invoke-Az -Arguments $Arguments
    if ([string]::IsNullOrWhiteSpace($text)) { return $null }
    return ($text | ConvertFrom-Json)
}

# --- 0. sanity --------------------------------------------------------------
if (-not (Test-Path $LocalPath)) {
    throw "Dataset folder not found: $LocalPath. Run finetune/data/build_dataset.py first."
}
foreach ($required in @('images', 'train.jsonl', 'val.jsonl', 'test.jsonl', 'manifest.json')) {
    if (-not (Test-Path (Join-Path $LocalPath $required))) {
        throw "Incomplete dataset: missing '$required' in $LocalPath."
    }
}
$imageCount = (Get-ChildItem (Join-Path $LocalPath 'images') -Filter *.png).Count
$sizeMb = [math]::Round(((Get-ChildItem $LocalPath -Recurse -File | Measure-Object Length -Sum).Sum / 1MB), 1)
Write-Host "Dataset: $imageCount images, $sizeMb MB, from $LocalPath" -ForegroundColor Cyan

$remotePath = "finetune/v$Version"

# --- 1. upload --------------------------------------------------------------
# The trailing '/*' is load-bearing. 'az storage fs directory upload --source DIR'
# uploads the directory ITSELF, i.e. it appends the source folder name to the
# destination: DIR\train.jsonl lands at <dest>/<basename DIR>/train.jsonl, one
# level deeper than intended. '--source DIR/*' uploads the CONTENTS. The data
# asset then points one level above the dataset root and the training job dies on
# a FileNotFoundError - 30 minutes of GPU later.
Write-Host "`n[1/2] Uploading to $StorageAccount/$FileSystem/$remotePath ..." -ForegroundColor Cyan
Invoke-Az -Arguments @(
    'storage', 'fs', 'directory', 'upload',
    '--account-name', $StorageAccount,
    '--file-system', $FileSystem,
    '--source', "$LocalPath/*",
    '--destination-path', $remotePath,
    '--recursive',
    '--auth-mode', 'login'
) | Out-Null

# Verify the LAYOUT, not just the count. The original version of this script
# checked 'length(@)' and got the expected 626 - the files were all there, just
# nested one level too deep. A count proves the upload happened; it says nothing
# about where. So assert the files the training job actually opens.
$remoteNames = Invoke-AzJson -Arguments @(
    'storage', 'fs', 'file', 'list',
    '--account-name', $StorageAccount,
    '--file-system', $FileSystem,
    '--path', $remotePath,
    '--recursive',
    '--auth-mode', 'login',
    '--query', '[].name',
    '-o', 'json'
)
Write-Host "  files in $remotePath : $($remoteNames.Count)" -ForegroundColor Green

foreach ($required in @('train.jsonl', 'val.jsonl', 'test.jsonl', 'manifest.json')) {
    if ($remoteNames -notcontains "$remotePath/$required") {
        throw "Layout check failed: '$remotePath/$required' not found. The dataset root is not where the asset will point."
    }
}
if (-not ($remoteNames | Where-Object { $_ -like "$remotePath/images/*.png" })) {
    throw "Layout check failed: no PNG directly under '$remotePath/images/'."
}
Write-Host "  layout OK: {train,val,test}.jsonl + images/ at the root of $remotePath" -ForegroundColor Green

# --- 2. register the data asset --------------------------------------------
Write-Host "`n[2/2] Registering data asset $AssetName`:$Version ..." -ForegroundColor Cyan
$assetUri = "azureml://datastores/$Datastore/paths/$remotePath"

$existing = & az ml data show --name $AssetName --version $Version `
    --resource-group $ResourceGroup --workspace-name $Workspace -o json 2>$null
if ($LASTEXITCODE -eq 0 -and -not [string]::IsNullOrWhiteSpace($existing)) {
    Write-Host "  already registered (versions are immutable, bump -Version to republish)" -ForegroundColor Yellow
} else {
    Invoke-Az -Arguments @(
        'ml', 'data', 'create',
        '--name', $AssetName,
        '--version', "$Version",
        '--type', 'uri_folder',
        '--path', $assetUri,
        '--description', 'Chart images + JSONL work orders rendered from telemetry (30-min, 4-sensor windows).',
        '--resource-group', $ResourceGroup,
        '--workspace-name', $Workspace
    ) | Out-Null
    Write-Host "  created" -ForegroundColor Green
}

Write-Host "`nData asset URI: $assetUri" -ForegroundColor Cyan
