<#
.SYNOPSIS
    Creates (or repairs) the low-priority GPU cluster used by the VLM LoRA pipeline,
    gives it a system-assigned identity, grants that identity data-plane access, and
    pins an aggressive idle scale-down so the cluster can never be left running.

.DESCRIPTION
    Three things must be true before any training job can run:

      1. The cluster exists with tier=low_priority. Dedicated GPU quota is often 0 on
         modern families; low-priority is a separate quota pool and it works.
      2. The cluster has its own managed identity. The workspace uses identity-based
         system datastores, so a cluster without an identity dies in ~20 s with
         "Identity of the specified managed compute ... is not found".
      3. min_instances = 0 AND idle_time_before_scale_down is short. This is the only
         real cost guarantee: the cluster releases its node by itself, with no human
         in the loop. Forgetting to stop it is not a failure mode here.

.PARAMETER IdleSeconds
    Seconds of inactivity before the node is released. Default 300 (5 min).
    Idle time is billed at the full node rate, so shorter is cheaper. Do not raise
    this to 1800/3600 "for safety": longer idle time costs more, it protects nothing.

.PARAMETER Stop
    Does not provision anything: forces the cluster back to min_instances = 0 and
    reports the nodes currently allocated. Use it when in doubt.

.NOTES
    ASCII only - PowerShell 5.1 reads .ps1 as ANSI.
    Uses the Invoke-Az wrapper: with $ErrorActionPreference = 'Stop', any native
    command writing to stderr raises a terminating NativeCommandError, and 2>$null
    does not prevent it.
#>

[CmdletBinding()]
param(
    [string] $ResourceGroup,
    [string] $Workspace,
    [string] $ComputeName = 'gpu-cluster-spot',
    [string] $VmSize = 'Standard_NC4as_T4_v3',
    [string] $ExtraDatastore = '',
    [int] $IdleSeconds = 300,
    [int] $MaxRetries = 10,
    [switch] $Stop
)

$ErrorActionPreference = 'Stop'

# Parameter defaults are bound BEFORE the body runs, so "= $env:X" in the param
# block would read the variables as they were before .env was loaded. Resolve here
# instead.
. "$PSScriptRoot\load_dotenv.ps1"
if (-not $ResourceGroup) { $ResourceGroup = $env:AZURE_RESOURCE_GROUP }
if (-not $Workspace) { $Workspace = $env:AZUREML_WORKSPACE_NAME }

if (-not $ResourceGroup -or -not $Workspace) {
    throw 'Fill in .env at the repo root (see .env.example), or pass -ResourceGroup / -Workspace.'
}
Set-Location (Split-Path $PSScriptRoot -Parent)

function Invoke-Az {
    param([Parameter(Mandatory)] [string[]] $Arguments)

    $errorFile = [System.IO.Path]::GetTempFileName()
    $previous = $ErrorActionPreference
    $ErrorActionPreference = 'Continue'
    try {
        $stdout = & az @Arguments 2> $errorFile
        $exitCode = $LASTEXITCODE
    }
    finally {
        $ErrorActionPreference = $previous
    }

    $stderr = ''
    if (Test-Path $errorFile) {
        $stderr = (Get-Content -Path $errorFile -Raw)
        Remove-Item -Path $errorFile -Force -ErrorAction SilentlyContinue
    }

    [pscustomobject]@{
        ExitCode = $exitCode
        StdOut   = ($stdout | Out-String)
        StdErr   = $stderr
    }
}

function Invoke-AzJson {
    param([Parameter(Mandatory)] [string[]] $Arguments)

    $result = Invoke-Az -Arguments $Arguments
    if ($result.ExitCode -ne 0) {
        throw ("az " + ($Arguments -join ' ') + " -> exit $($result.ExitCode)`n" + $result.StdErr)
    }
    return ($result.StdOut | ConvertFrom-Json)
}

function Get-Compute {
    $result = Invoke-Az -Arguments @('ml', 'compute', 'show', '-n', $ComputeName, '-g', $ResourceGroup, '-w', $Workspace, '-o', 'json')
    if ($result.ExitCode -ne 0) { return $null }
    return ($result.StdOut | ConvertFrom-Json)
}

# PowerShell 5.1 trap: @($null).Count is 1, not 0. ConvertFrom-Json on an empty
# JSON array can yield $null, which then counts as one phantom element - and that
# phantom prints with every field empty. Never wrap a parse result in @() directly.
function ConvertTo-Array {
    param([string] $Text)

    if (-not $Text -or -not $Text.Trim()) { return @() }
    $parsed = $Text | ConvertFrom-Json
    if ($null -eq $parsed) { return @() }
    return @($parsed | Where-Object { $null -ne $_ })
}

# The az ml v2 CLI emits snake_case (node_state), but some builds emit camelCase.
function Get-Prop {
    param($Object, [string[]] $Names)

    foreach ($name in $Names) {
        if ($Object.PSObject.Properties[$name]) { return $Object.$name }
    }
    return ''
}

function Show-Nodes {
    $result = Invoke-Az -Arguments @('ml', 'compute', 'list-nodes', '-n', $ComputeName, '-g', $ResourceGroup, '-w', $Workspace, '-o', 'json')
    if ($result.ExitCode -ne 0) {
        Write-Host "  could not list nodes: $($result.StdErr.Trim())" -ForegroundColor Yellow
        return
    }
    $nodes = ConvertTo-Array -Text $result.StdOut
    if ($nodes.Count -eq 0) {
        Write-Host "  0 node allocated - nothing is billing" -ForegroundColor Green
    }
    else {
        Write-Host "  $($nodes.Count) node(s) ALLOCATED - this is billing right now" -ForegroundColor Yellow
        foreach ($n in $nodes) {
            $id = Get-Prop -Object $n -Names @('node_id', 'nodeId')
            $state = Get-Prop -Object $n -Names @('node_state', 'nodeState')
            $ip = Get-Prop -Object $n -Names @('private_ip_address', 'privateIpAddress')
            Write-Host "    $id  state=$state  ip=$ip"
        }
    }
}

function Grant-Role {
    param(
        [Parameter(Mandatory)] [string] $PrincipalId,
        [Parameter(Mandatory)] [string] $Role,
        [Parameter(Mandatory)] [string] $Scope
    )

    for ($attempt = 1; $attempt -le $MaxRetries; $attempt++) {
        $result = Invoke-Az -Arguments @(
            'role', 'assignment', 'create',
            '--assignee-object-id', $PrincipalId,
            '--assignee-principal-type', 'ServicePrincipal',
            '--role', $Role,
            '--scope', $Scope,
            '-o', 'none'
        )

        if ($result.ExitCode -eq 0) { return 'granted' }
        if ($result.StdErr -match 'RoleAssignmentExists') { return 'already present' }
        if ($result.StdErr -match 'PrincipalNotFound|does not exist in the directory') {
            Write-Host "    identity not replicated yet, retry $attempt/$MaxRetries ..."
            Start-Sleep -Seconds 10
            continue
        }

        throw ("role assignment failed -> exit $($result.ExitCode)`n" + $result.StdErr)
    }

    throw "Identity $PrincipalId never became visible to Entra after $MaxRetries attempts."
}

Write-Host "=== GPU cluster ===" -ForegroundColor Cyan
Write-Host "cluster : $ComputeName"
Write-Host "size    : $VmSize (low_priority)"
Write-Host ""

# --- panic button -------------------------------------------------------------
if ($Stop) {
    $existing = Get-Compute
    if (-not $existing) {
        Write-Host "Cluster $ComputeName does not exist - nothing can be billing." -ForegroundColor Green
        return
    }

    Write-Host "Forcing min_instances = 0 ..."
    Invoke-AzJson -Arguments @(
        'ml', 'compute', 'update',
        '-n', $ComputeName, '-g', $ResourceGroup, '-w', $Workspace,
        '--min-instances', '0',
        '-o', 'json'
    ) | Out-Null
    Write-Host "  done"
    Write-Host ""
    Write-Host "Nodes still allocated (they are released after the idle delay):"
    Show-Nodes
    return
}

# --- 1. cluster ---------------------------------------------------------------
Write-Host "[1/4] Ensuring the cluster exists ..."
$compute = Get-Compute

if (-not $compute) {
    Write-Host "  creating (this takes a few seconds; nodes are allocated only on demand)"
    $compute = Invoke-AzJson -Arguments @(
        'ml', 'compute', 'create',
        '-n', $ComputeName, '-g', $ResourceGroup, '-w', $Workspace,
        '--type', 'amlcompute',
        '--size', $VmSize,
        '--tier', 'low_priority',
        '--min-instances', '0',
        '--max-instances', '1',
        '--idle-time-before-scale-down', "$IdleSeconds",
        '--identity-type', 'SystemAssigned',
        '-o', 'json'
    )
    Write-Host "  created"
}
else {
    Write-Host "  already present (state=$($compute.provisioning_state), tier=$($compute.tier))"
}

# --- 2. cost guarantee --------------------------------------------------------
Write-Host "`n[2/4] Enforcing scale-to-zero ..."
$currentMin = [int]$compute.min_instances
$currentIdle = $compute.idle_time_before_scale_down

Write-Host "  current: min_instances=$currentMin idle_time_before_scale_down=$currentIdle"
if ($currentMin -ne 0 -or "$currentIdle" -ne "$IdleSeconds") {
    $compute = Invoke-AzJson -Arguments @(
        'ml', 'compute', 'update',
        '-n', $ComputeName, '-g', $ResourceGroup, '-w', $Workspace,
        '--min-instances', '0',
        '--max-instances', '1',
        '--idle-time-before-scale-down', "$IdleSeconds",
        '-o', 'json'
    )
    Write-Host "  updated: min_instances=0 idle_time_before_scale_down=$IdleSeconds" -ForegroundColor Green
}
else {
    Write-Host "  already correct" -ForegroundColor Green
}

# --- 3. identity --------------------------------------------------------------
Write-Host "`n[3/4] Ensuring system-assigned identity ..."
$principalId = $null
if ($compute.identity -and $compute.identity.principal_id) {
    $principalId = $compute.identity.principal_id
}

if ($principalId) {
    Write-Host "  already present: $principalId"
}
else {
    $updated = Invoke-AzJson -Arguments @(
        'ml', 'compute', 'update',
        '-n', $ComputeName, '-g', $ResourceGroup, '-w', $Workspace,
        '--identity-type', 'SystemAssigned',
        '-o', 'json'
    )
    $principalId = $updated.identity.principal_id
    if (-not $principalId) {
        throw "The update returned no principal_id; check 'az ml compute show'."
    }
    Write-Host "  created: $principalId"
}

# --- 4. data-plane RBAC -------------------------------------------------------
Write-Host "`n[4/4] Granting data-plane access ..."

$ws = Invoke-AzJson -Arguments @('ml', 'workspace', 'show', '-n', $Workspace, '-g', $ResourceGroup, '-o', 'json')
$wsStorageId = $ws.storage_account

Write-Host "  $($wsStorageId.Split('/')[-1]) - Storage Blob Data Contributor"
Write-Host "    $(Grant-Role -PrincipalId $principalId -Role 'Storage Blob Data Contributor' -Scope $wsStorageId)"

# Optional: a second datastore backed by another storage account (an ADLS Gen2
# filesystem holding the dataset, for instance). Skipped by default - the notebook
# uploads to workspaceblobstore, which the grant above already covers.
if ($ExtraDatastore) {
    $ds = Invoke-AzJson -Arguments @('ml', 'datastore', 'show', '--name', $ExtraDatastore, '-g', $ResourceGroup, '-w', $Workspace, '-o', 'json')
    $extraStorageId = (Invoke-Az -Arguments @('storage', 'account', 'show', '-n', $ds.account_name, '--query', 'id', '-o', 'tsv')).StdOut.Trim()
    if (-not $extraStorageId) {
        throw "Could not resolve the resource id of storage account $($ds.account_name)."
    }
    Write-Host "  $($ds.account_name) - Storage Blob Data Reader"
    Write-Host "    $(Grant-Role -PrincipalId $principalId -Role 'Storage Blob Data Reader' -Scope $extraStorageId)"
}

Write-Host ""
Write-Host "Current allocation:"
Show-Nodes

Write-Host ""
Write-Host "Cost profile: 0 node at rest, 1 node only while a job runs," -ForegroundColor Green
Write-Host "released automatically after $IdleSeconds s of inactivity." -ForegroundColor Green
Write-Host ""
Write-Host "Role assignments can take a couple of minutes to take effect." -ForegroundColor Yellow
Write-Host "Check what is allocated at any time:" -ForegroundColor Cyan
Write-Host "  az ml compute list-nodes -n $ComputeName -g $ResourceGroup -w $Workspace -o table"
Write-Host "Force everything back to zero:" -ForegroundColor Cyan
Write-Host "  .\scripts\60_gpu_cluster.ps1 -Stop"
