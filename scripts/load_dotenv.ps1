# Dot-source this from any script in scripts/ to make the repo's single .env file
# the one place configuration lives:
#
#     . "$PSScriptRoot\load_dotenv.ps1"
#
# Existing environment variables win, so a shell where AZURE_RESOURCE_GROUP is
# already exported (CI, a devcontainer) is not silently overridden by a stale .env.
#
# ASCII only - PowerShell 5.1 reads .ps1 as ANSI.

$envFile = Join-Path (Split-Path $PSScriptRoot -Parent) '.env'
if (-not (Test-Path $envFile)) { return }

foreach ($line in Get-Content -Path $envFile) {
    $trimmed = $line.Trim()
    if (-not $trimmed -or $trimmed.StartsWith('#')) { continue }

    $split = $trimmed.IndexOf('=')
    if ($split -lt 1) { continue }

    $name = $trimmed.Substring(0, $split).Trim()
    $value = $trimmed.Substring($split + 1).Trim().Trim('"', "'")
    if (-not [Environment]::GetEnvironmentVariable($name)) {
        Set-Item -Path "env:$name" -Value $value
    }
}
