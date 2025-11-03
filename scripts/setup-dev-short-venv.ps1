# Create a short-path virtualenv and install dev requirements to avoid Windows long-path issues
# Usage: Run in PowerShell (no admin required)
param(
    [string]$VenvPath = "C:\pf_venv"
)

Write-Host "Creating virtualenv at $VenvPath"
python -m venv $VenvPath
$activate = Join-Path $VenvPath 'Scripts\Activate.ps1'
if (Test-Path $activate) {
    Write-Host "Activating venv and installing dev-requirements..."
    & $activate; python -m pip install --upgrade pip setuptools wheel build
    pip install -r "$(Split-Path -Parent $MyInvocation.MyCommand.Definition)\..\dev-requirements.txt"
    Write-Host "Done. Activate with: & $activate"
} else {
    Write-Host "Failed to create virtualenv at $VenvPath. Check python installation and permissions."
}
