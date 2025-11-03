<#
.SYNOPSIS
Run Autorun.py with venv activation on Windows.

This wrapper will:
- Ensure the project `.venv` exists (otherwise call Autorun to create it)
- Dot-source the venv Activation script so the session uses the venv
- Forward any arguments to `python Autorun.py` (e.g. 'all', 'local', 'docker')

Usage:
    .\scripts\run-autorun.ps1 all
#>

param(
    [Parameter(ValueFromRemainingArguments=$true)]
    [string[]]$ForwardArgs
)

Set-StrictMode -Version Latest
Get-Location | Out-Null
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Definition
Set-Location $ScriptDir\..  # move to repo root

$venvActivate = Join-Path -Path (Join-Path -Path $PWD -ChildPath ".venv") -ChildPath "Scripts\Activate.ps1"

if (-Not (Test-Path $venvActivate)) {
    Write-Host "No virtualenv detected at .venv. Creating one (this may take a few seconds)..."
    python Autorun.py venv
    if (-Not (Test-Path $venvActivate)) {
        Write-Error "Failed to create venv at .venv. Aborting."
        exit 1
    }
}

try {
    Write-Host "Activating venv from: $venvActivate"
    . $venvActivate
} catch {
    Write-Warning "Activation failed: $_. Attempting to run python directly from venv."
}

$argsList = $ForwardArgs -join ' '
if ([string]::IsNullOrEmpty($argsList)) { $argsList = 'all' }

Write-Host "Running: python Autorun.py $argsList"
python Autorun.py @ForwardArgs
