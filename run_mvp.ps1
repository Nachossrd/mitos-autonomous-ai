# ============================================================================
# Proyecto MITOS - Runner del MVP (PowerShell)
# ----------------------------------------------------------------------------
# Equivalente nativo de run_mvp.sh para Windows + PowerShell.
# ============================================================================

$ErrorActionPreference = "Stop"

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $ScriptDir

# ---- 1. Activar venv si no esta activo ------------------------------------
if (-not $env:VIRTUAL_ENV) {
    $activate = Join-Path $ScriptDir ".venv\Scripts\Activate.ps1"
    if (Test-Path $activate) {
        & $activate
    } else {
        Write-Host "ERROR: no encuentro .venv. Crealo con:" -ForegroundColor Red
        Write-Host "    python -m venv .venv"
        Write-Host "    .\.venv\Scripts\Activate.ps1"
        Write-Host "    pip install -r requirements.txt"
        exit 1
    }
}

# ---- 2. Validacion de dependencias del MVP --------------------------------
$pyCheck = @"
import importlib.util, sys
needed = ['torch', 'websockets', 'msgpack', 'nacl', 'cryptography', 'rich']
missing = [m for m in needed if importlib.util.find_spec(m) is None]
if missing:
    print(f'ERROR: faltan dependencias: {missing}')
    print('Ejecuta: pip install -r requirements.txt')
    sys.exit(1)
print('Dependencias OK')
"@
$pyCheck | python -
if ($LASTEXITCODE -ne 0) { exit 1 }

# ---- 3. Orquestador ------------------------------------------------------
Write-Host ""
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host " 1/2  Orquestador central" -ForegroundColor Cyan
Write-Host "============================================================" -ForegroundColor Cyan
python -m src.orchestrator.main

# ---- 4. Demo distribuida -------------------------------------------------
Write-Host ""
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host " 2/2  Modulo distribuido (P2P + DGC)" -ForegroundColor Cyan
Write-Host "      Esto tarda unos segundos en levantar los nodos..." -ForegroundColor DarkGray
Write-Host "============================================================" -ForegroundColor Cyan
python -m src.distributed.demo

Write-Host ""
Write-Host "MVP completado." -ForegroundColor Green
