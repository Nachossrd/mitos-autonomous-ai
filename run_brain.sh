#!/usr/bin/env bash
# ============================================================================
# Proyecto MITOS - Runner del Cerebro autónomo
# ----------------------------------------------------------------------------
# 1. Activa el entorno virtual (.venv).
# 2. Verifica que exista al menos un .gguf bajo models/. Si no, descarga
#    Qwen2.5-7B-Instruct-abliterated-Q4_K_M.gguf desde Hugging Face.
# 3. Lanza la CLI interactiva: python -m src.brain.interactive
#
# Compatible con:
#   - Linux / macOS  ->  .venv/bin/activate
#   - Windows + Git Bash ->  .venv/Scripts/activate
# ============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

MODEL_DIR="models"
# Fase 8: upgrade a Qwen2.5-Coder-14B-Instruct (oficial).
# Razones:
#   - 14B parámetros (vs los 7B previos) → mejor razonamiento.
#   - Variante "Coder" → afinada a código, el 80% del trabajo de MITOS.
#   - ChatML idéntico al 7B previo → llm_engine.py no necesita cambios.
# El IntelligenceRouter (Fase 8) delega tareas complejas/visión a APIs
# externas (Gemini/Groq), así que el local solo necesita ser bueno en
# despacho y código local — exactamente lo que es Coder.
MODEL_FILE="qwen2.5-coder-14b-instruct-q4_k_m-00001-of-00002.gguf"
MODEL_REPO="Qwen/Qwen2.5-Coder-14B-Instruct-GGUF"
# Glob para que `hf download --include` baje todos los splits del
# mismo Q4_K_M en una sola pasada. Sin esto el script solo descarga
# el primer split y llama.cpp se queja al no encontrar el segundo.
MODEL_INCLUDE_GLOB="qwen2.5-coder-14b-instruct-q4_k_m*.gguf"

# ---- 1. Activar venv (auto-detect Linux/Mac vs Windows) -------------------
if [[ -f ".venv/bin/activate" ]]; then
    # shellcheck disable=SC1091
    source ".venv/bin/activate"
elif [[ -f ".venv/Scripts/activate" ]]; then
    # shellcheck disable=SC1091
    source ".venv/Scripts/activate"
else
    echo "ERROR: no encuentro .venv/. Crea el entorno con:"
    echo "    python -m venv .venv"
    echo "    .venv/Scripts/activate   # Windows"
    echo "    source .venv/bin/activate # Linux/Mac"
    echo "    pip install -r requirements.txt"
    exit 1
fi

# ---- 2. Verificar / descargar modelo --------------------------------------
mkdir -p "$MODEL_DIR"

# ¿Hay al menos un .gguf en models/?
shopt -s nullglob
ggufs=("$MODEL_DIR"/*.gguf)
shopt -u nullglob

if (( ${#ggufs[@]} == 0 )); then
    echo "No hay ningún .gguf en $MODEL_DIR/."
    echo "Descargando $MODEL_FILE desde $MODEL_REPO ..."

    # huggingface_hub >=0.30 renombró el CLI: `hf` reemplaza a `huggingface-cli`.
    # Preferimos `hf`; caemos al viejo solo si no está.
    if command -v hf >/dev/null 2>&1; then
        HF_CLI="hf"
    elif command -v huggingface-cli >/dev/null 2>&1; then
        HF_CLI="huggingface-cli"
    else
        echo "ERROR: no encuentro 'hf' ni 'huggingface-cli'."
        echo "Instala/actualiza con: pip install --upgrade -r requirements.txt"
        exit 1
    fi
    echo "Usando CLI: $HF_CLI"
    "$HF_CLI" download "$MODEL_REPO" --include "$MODEL_INCLUDE_GLOB" --local-dir "$MODEL_DIR"
    echo "Descarga completa."
else
    echo "Modelo(s) detectado(s) en $MODEL_DIR/:"
    for g in "${ggufs[@]}"; do
        echo "  - $(basename "$g")"
    done
fi

# ---- 3. Validación rápida de dependencias críticas ------------------------
python - <<'PY'
import importlib.util, sys
needed = ["llama_cpp", "chromadb", "sentence_transformers", "rich"]
missing = [m for m in needed if importlib.util.find_spec(m) is None]
if missing:
    print(f"ERROR: faltan dependencias: {missing}")
    print("Ejecuta: pip install -r requirements.txt")
    sys.exit(1)
print("Dependencias OK")
PY

# ---- 4. Levantar la CLI ---------------------------------------------------
echo ""
echo "============================================================"
echo " Lanzando MITOS Brain (CLI interactiva)"
echo "============================================================"
python -m src.brain.interactive
