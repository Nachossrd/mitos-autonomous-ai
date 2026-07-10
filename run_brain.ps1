# ============================================================================
# Proyecto MITOS - Runner del Cerebro autónomo (PowerShell)
# ----------------------------------------------------------------------------
# Equivalente nativo de run_brain.sh para Windows + PowerShell.
#
# Uso:
#     ./run_brain.ps1
#
# Si PowerShell rechaza ejecutarlo por política, descomenta una de estas:
#     # Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
#     # powershell -ExecutionPolicy Bypass -File .\run_brain.ps1
# ============================================================================

$ErrorActionPreference = "Stop"

# Movernos al directorio del script (raíz del proyecto), funcione desde donde funcione.
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $ScriptDir

$ModelDir  = "models"
# Fase 8: Qwen2.5-Coder-14B-Instruct (oficial). 14B parametros, ChatML,
# afinado a codigo. El IntelligenceRouter delega tareas creativas a
# Gemini/Groq por API. Llega partido en 2-3 splits ~9 GB.
$ModelFile = "qwen2.5-coder-14b-instruct-q4_k_m-00001-of-00002.gguf"
$ModelRepo = "Qwen/Qwen2.5-Coder-14B-Instruct-GGUF"
$ModelIncludeGlob = "qwen2.5-coder-14b-instruct-q4_k_m*.gguf"

# ---- 1. Activar venv (si no está ya activo) -------------------------------
if (-not $env:VIRTUAL_ENV) {
    $activate = Join-Path $ScriptDir ".venv\Scripts\Activate.ps1"
    if (Test-Path $activate) {
        Write-Host "Activando entorno virtual..." -ForegroundColor Yellow
        & $activate
    } else {
        Write-Host "ERROR: no encuentro .venv. Crealo con:" -ForegroundColor Red
        Write-Host "    python -m venv .venv"
        Write-Host "    .\.venv\Scripts\Activate.ps1"
        Write-Host "    pip install -r requirements.txt"
        exit 1
    }
} else {
    Write-Host "venv ya activo: $env:VIRTUAL_ENV" -ForegroundColor DarkGray
}

# ---- 2. Verificar / descargar modelo --------------------------------------
if (-not (Test-Path $ModelDir)) {
    New-Item -ItemType Directory -Path $ModelDir | Out-Null
}

$ggufs = Get-ChildItem -Path $ModelDir -Filter "*.gguf" -ErrorAction SilentlyContinue

# Chequeo especifico del modelo TARGET de Fase 8 (no "cualquier gguf").
# Antes el script veia el 7B viejo y daba por buena la descarga, dejando
# al operador colgado del modelo anterior sin avisar.
$hasTarget = $ggufs | Where-Object { $_.Name -match "qwen2.5-coder-14b" }

if (-not $hasTarget) {
    if ($ggufs -and $ggufs.Count -gt 0) {
        Write-Host "Detecte .gguf en $ModelDir/ pero NO el target Fase 8:" -ForegroundColor Yellow
        foreach ($g in $ggufs) { Write-Host "  - $($g.Name)" -ForegroundColor DarkGray }
        Write-Host "Bajando $ModelFile adicionalmente..." -ForegroundColor Yellow
    } else {
        Write-Host "No hay ningun .gguf en $ModelDir/." -ForegroundColor Yellow
        Write-Host "Descargando $ModelFile desde $ModelRepo ..." -ForegroundColor Yellow
    }

    # huggingface_hub >=0.30 renombro el CLI: `hf` reemplaza a `huggingface-cli`.
    # Preferimos `hf`; caemos al viejo solo si no esta disponible.
    $cli = $null
    if (Get-Command hf -ErrorAction SilentlyContinue) {
        $cli = "hf"
    } elseif (Get-Command huggingface-cli -ErrorAction SilentlyContinue) {
        $cli = "huggingface-cli"
    } else {
        Write-Host "ERROR: no encuentro `hf` ni `huggingface-cli`." -ForegroundColor Red
        Write-Host "Instala/actualiza las dependencias con:"
        Write-Host "    pip install --upgrade -r requirements.txt"
        exit 1
    }

    Write-Host "Usando CLI: $cli" -ForegroundColor DarkGray
    # Bajamos TODOS los splits del Q4_K_M con --include glob.
    & $cli download $ModelRepo --include $ModelIncludeGlob --local-dir $ModelDir
    if ($LASTEXITCODE -ne 0) {
        Write-Host "ERROR: la descarga del modelo fallo (exit $LASTEXITCODE)." -ForegroundColor Red
        Write-Host "Intenta manualmente:" -ForegroundColor Yellow
        Write-Host "    $cli download $ModelRepo --include $ModelIncludeGlob --local-dir $ModelDir"
        exit 1
    }
    Write-Host "Descarga completa." -ForegroundColor Green
} else {
    Write-Host "Modelo(s) detectado(s) en $ModelDir/:" -ForegroundColor Green
    foreach ($g in $ggufs) {
        Write-Host "  - $($g.Name)"
    }
}

# ---- 3. Validacion rapida de dependencias criticas ------------------------
$pyCheck = @"
import importlib.util, sys
needed = ['llama_cpp', 'chromadb', 'sentence_transformers', 'rich']
missing = [m for m in needed if importlib.util.find_spec(m) is None]
if missing:
    print(f'ERROR: faltan dependencias: {missing}')
    print('Ejecuta: pip install -r requirements.txt')
    sys.exit(1)
print('Dependencias OK')
"@

$pyCheck | python -
if ($LASTEXITCODE -ne 0) {
    exit 1
}

# ---- 4. Levantar la CLI ---------------------------------------------------
Write-Host ""
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host " Lanzando MITOS Brain (CLI interactiva)" -ForegroundColor Cyan
Write-Host "============================================================" -ForegroundColor Cyan
python -m src.brain.interactive
exit $LASTEXITCODE
