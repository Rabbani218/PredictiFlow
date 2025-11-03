<#
Start PredictiFlow development services locally in two separate PowerShell windows.

This script opens two new terminals (one for backend, one for frontend) and runs
the development servers. Requires PowerShell on Windows.
#>

$projectRoot = Split-Path -Parent $MyInvocation.MyCommand.Definition

Write-Host "Starting backend in new window..."
Start-Process powershell -ArgumentList "-NoExit", "-Command Set-Location '$projectRoot\backend'; python -m uvicorn app.main:app --reload --host 0.0.0.0 --port 8000"

Start-Sleep -Seconds 2
Write-Host "Starting frontend in new window..."
Start-Process powershell -ArgumentList "-NoExit", "-Command Set-Location '$projectRoot\frontend'; streamlit run app.py --server.port=8501"

Write-Host "Both processes started (check new windows)."
