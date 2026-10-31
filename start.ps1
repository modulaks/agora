# Agora: instalacja i start jednym poleceniem (Windows PowerShell).
# Uzycie:  powershell -ExecutionPolicy Bypass -File .\start.ps1
# Wybor modelu:
#   .\start.ps1                   (qwen, Ollama na serwerze GPU)
#   .\start.ps1 --model luna      (OpenAI, klucz OPENAI_API_KEY w .env)
#   .\start.ps1 --model sonnet    (Claude Sonnet 5.5, klucz ANTHROPIC_API_KEY w .env)
#   .\start.ps1 --dry-run --ticks 2

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

# Biblioteki instalujemy raz; ponownie tylko, gdy zmieni sie requirements.txt.
$stamp = Join-Path $PSScriptRoot ".venv\requirements.sha256"
$hash = (Get-FileHash "requirements.txt" -Algorithm SHA256).Hash
$installed = if (Test-Path $stamp) { (Get-Content $stamp -Raw).Trim() } else { "" }
if ($hash -ne $installed) {
    Write-Host "Instaluje zaleznosci (jednorazowo) ..." -ForegroundColor Cyan
    & $py -m pip install --disable-pip-version-check -q -r requirements.txt
    if ($LASTEXITCODE -ne 0) { Write-Host "Instalacja zaleznosci nie powiodla sie (komunikat wyzej)." -ForegroundColor Red; exit 1 }
    Set-Content -Path $stamp -Value $hash -Encoding ASCII
}

if (-not (Test-Path ".env")) { Copy-Item ".env.example" ".env" }

Write-Host "Start. Log zapisuje sie do community\agora-log.txt, podglad: http://localhost:8765" -ForegroundColor Cyan
& $py -m agora --viewer @args
