#!/usr/bin/env bash
# ============================================================================
# Proyecto MITOS - Runner del MVP
# ----------------------------------------------------------------------------
# Activa el entorno virtual, ejecuta la demo integrada del orquestador y luego
# la demo del módulo distribuido (P2P + DGC + Elastic Averaging).
#
# Compatible con:
#   - Linux / macOS  ->  .venv/bin/activate
#   - Windows + Git Bash ->  .venv/Scripts/activate
# ============================================================================

set -euo pipefail

# Movernos al directorio del script (raíz del proyecto), funcione desde donde funcione.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# ---- Activar venv (auto-detect Linux/Mac vs Windows) -----------------------
if [[ -f ".venv/bin/activate" ]]; then
    # shellcheck disable=SC1091
    source ".venv/bin/activate"
elif [[ -f ".venv/Scripts/activate" ]]; then
    # shellcheck disable=SC1091
    source ".venv/Scripts/activate"
else
    echo "ERROR: no encuentro .venv/. Crea el entorno con:"
    echo "    python -m venv .venv"
    echo "    .venv/Scripts/activate    # Windows"
    echo "    source .venv/bin/activate # Linux/Mac"
    echo "    pip install -r requirements.txt"
    exit 1
fi

# ---- Validación rápida de dependencias -------------------------------------
python - <<'PY'
import importlib.util, sys
needed = ["torch", "websockets", "msgpack", "nacl", "cryptography", "rich"]
missing = [m for m in needed if importlib.util.find_spec(m) is None]
if missing:
    print(f"ERROR: faltan dependencias: {missing}")
    print("Ejecuta: pip install -r requirements.txt")
    sys.exit(1)
PY

# ---- 1. Demo integrada (Evolución + Seguridad + Filtrado) ------------------
echo ""
echo "============================================================"
echo " 1/2  Orquestador central"
echo "============================================================"
python -m src.orchestrator.main

# ---- 2. Demo del módulo distribuido ---------------------------------------
echo ""
echo "============================================================"
echo " 2/2  Módulo distribuido (P2P + DGC)"
echo "      Esto tarda unos segundos en levantar los nodos..."
echo "============================================================"
python -m src.distributed.demo

echo ""
echo "MVP completado."
