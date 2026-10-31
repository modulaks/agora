# Agora: instalacja i start jednym poleceniem (Windows PowerShell).
# Uzycie:  powershell -ExecutionPolicy Bypass -File .\start.ps1
# Dodatkowe opcje ida dalej do programu, np.:  .\start.ps1 --dry-run --ticks 2

Set-Location $PSScriptRoot
$env:PYTHONUTF8 = "1"
$env:PYTHONIOENCODING = "utf-8"
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8

$py = Join-Path $PSScriptRoot ".venv\Scripts\python.exe"
if (-not (Test-Path $py)) {
    Write-Host "Tworze srodowisko .venv ..." -ForegroundColor Cyan
    python -m venv .venv
    if (-not (Test-Path $py)) { Write-Host "Nie udalo sie utworzyc .venv. Czy Python jest zainstalowany (python --version)?" -ForegroundColor Red; exit 1 }
}

Write-Host "Instaluje zaleznosci ..." -ForegroundColor Cyan
& $py -m pip install --disable-pip-version-check -q -r requirements.txt
if ($LASTEXITCODE -ne 0) { Write-Host "Instalacja zaleznosci nie powiodla sie (komunikat wyzej)." -ForegroundColor Red; exit 1 }

if (-not (Test-Path ".env")) { Copy-Item ".env.example" ".env" }

Write-Host "Start. Log zapisuje sie do community\agora-log.txt, podglad: http://localhost:8765" -ForegroundColor Cyan
if ($args.Count -gt 0) { & $py -m agora @args } else { & $py -m agora --viewer }
