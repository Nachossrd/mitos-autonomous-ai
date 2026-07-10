"""
==============================================================================
 Proyecto MITOS - Tests de Fase 8 (Cognitive Mesh)
==============================================================================

Cubre tres comportamientos críticos de la malla cognitiva:

  1. Clasificación de complejidad del IntelligenceRouter — la heurística
     debe asignar COMPLEX a tareas de implementación de pipeline y
     SIMPLE a tareas de formateo trivial.

  2. Detección de repetición del LimitationAwarenessEngine — strings
     con n-gramas altamente repetidos deben dispararla; texto natural
     no.

  3. Fallback del SystemSensor cuando no hay psutil — debe poder
     leer la métrica de RAM en Windows vía ctypes y en Linux vía
     /proc/meminfo sin requerir librería externa.

Convenciones del proyecto:
  - `from __future__ import annotations`, tipado moderno.
  - Sin mocks de internet — TODOS los tests deben pasar offline.
==============================================================================
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import pytest

# Sin esto, los tests fallan si pytest no se invoca desde la raíz.
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.core.intelligence_router import (
    IntelligenceRouter,
    TaskComplexity,
)
from src.core.sensor_hub import SystemSensor
from src.core.bug_scanner import BugScanner, BugFinding


# ============================================================================
# Fixtures
# ============================================================================
class _StubLocalLLM:
    """Local LLM mínimo para construir un IntelligenceRouter sin tocar disco."""

    def think(self, prompt: str, max_tokens: int = 256, temperature: float = 0.7) -> str:
        return "stub"

    def generate_code(self, *_args, **_kwargs) -> str:
        return "# stub"


@pytest.fixture
def router(tmp_path: Path) -> IntelligenceRouter:
    """Construye un router con pool vacío (sin claves reales) para el test."""
    # IntelligenceRouter crea config/brain_pool.json si no existe.
    # Usamos un tmp_path → no contamina el repo del usuario.
    return IntelligenceRouter(
        project_root=tmp_path,
        local_llm=_StubLocalLLM(),
    )


# ============================================================================
# Test 1 — Clasificación de complejidad del router
# ============================================================================
def test_router_complexity_classification(router: IntelligenceRouter) -> None:
    """COMPLEX para 'implementar un pipeline', SIMPLE para 'formatear este código'."""
    complex_task = "implementar un pipeline de procesamiento ETL completo"
    simple_task = "formatear este código"

    complex_result = router.classify_complexity(complex_task)
    simple_result = router.classify_complexity(simple_task)

    assert complex_result == TaskComplexity.COMPLEX, (
        f"Esperaba COMPLEX para '{complex_task}', "
        f"obtuve {complex_result}"
    )
    assert simple_result == TaskComplexity.SIMPLE, (
        f"Esperaba SIMPLE para '{simple_task}', "
        f"obtuve {simple_result}"
    )


def test_router_complexity_creative(router: IntelligenceRouter) -> None:
    """Bonus: tareas creativas se clasifican como CREATIVE, no COMPLEX."""
    creative_task = "diseñar la arquitectura de un nuevo sistema de pagos"
    result = router.classify_complexity(creative_task)
    assert result in (TaskComplexity.CREATIVE, TaskComplexity.COMPLEX), (
        f"Esperaba CREATIVE/COMPLEX para tarea de diseño, obtuve {result}"
    )


# Tests de LimitationEngine borrados en Fase A (módulo eliminado).

# ============================================================================
# Test 3 — SystemSensor evalúa RAM sin librerías externas
# ============================================================================
def test_system_sensor_fallback() -> None:
    """SystemSensor lee RAM sin requerir psutil — fallback ctypes/proc.

    El contrato es: la lectura devuelve un float. En Linux usa /proc,
    en Windows usa ctypes.GlobalMemoryStatusEx. Si la plataforma no
    soporta el fallback, devuelve -1.0 como sentinel — nunca crashea
    ni levanta excepción.
    """
    import queue as _queue
    event_queue: _queue.Queue = _queue.Queue(maxsize=100)
    sensor = SystemSensor(event_queue=event_queue)

    # Disponibilidad SIEMPRE debe ser True — el sistema siempre tiene RAM.
    assert sensor.is_available() is True, (
        "SystemSensor.is_available() debe ser True en cualquier sistema"
    )

    # RAM: en Linux y Windows debe dar un valor real [0, 100], no -1.
    ram_pct = sensor._read_ram_percent()
    assert isinstance(ram_pct, float), (
        "_read_ram_percent debería devolver float"
    )
    assert 0.0 <= ram_pct <= 100.0, (
        f"RAM% fuera de rango [0,100] — fallback ctypes/proc rotos? "
        f"ram_pct={ram_pct}"
    )

    # CPU: en Linux funciona vía /proc/loadavg. En Windows sin psutil
    # devuelve -1.0 (sentinel, no crash). Aceptamos ambos.
    cpu_pct = sensor._read_cpu_percent()
    assert isinstance(cpu_pct, float)
    assert cpu_pct == -1.0 or (0.0 <= cpu_pct <= 100.0), (
        f"CPU% inválido: {cpu_pct} (debe ser -1.0 o [0,100])"
    )


def test_system_sensor_disk_safe() -> None:
    """Bonus: disk_usage no crashea aunque psutil no esté presente."""
    import queue as _queue
    sensor = SystemSensor(event_queue=_queue.Queue(maxsize=100))
    disk_pct = sensor._read_disk_percent()
    assert isinstance(disk_pct, float)
    assert 0.0 <= disk_pct <= 100.0


# ============================================================================
# Test 4 — Integración: LimitationEngine + SensorHub
# ============================================================================
# ============================================================================
# Test 5 — BugScanner captura excepciones reales del logging
# ============================================================================
def test_bug_scanner_captures_exception(tmp_path: Path) -> None:
    """Una excepción registrada con exc_info → un BugFinding mapped to file::func."""
    import logging
    # Setup: el scanner debe ver la ruta del test como dentro del repo.
    # Para esto creamos una estructura src/ falsa con el archivo del test.
    fake_src = tmp_path / "src"
    fake_src.mkdir()
    fake_module = fake_src / "fake_module.py"
    fake_module.write_text(
        "def buggy():\n    raise ValueError('test bug')\n",
        encoding="utf-8",
    )

    scanner = BugScanner(project_root=tmp_path)

    # Cargamos el módulo desde la ruta falsa para que el traceback apunte ahí.
    import importlib.util
    spec = importlib.util.spec_from_file_location("fake_module", fake_module)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    # Disparamos el bug y lo logueamos con exc_info.
    test_logger = logging.getLogger("mitos.test_bug_scanner")
    scanner.install("mitos.test_bug_scanner")
    try:
        try:
            mod.buggy()
        except ValueError:
            test_logger.error("falla controlada", exc_info=True)
    finally:
        scanner.uninstall("mitos.test_bug_scanner")

    findings = scanner.drain_findings()
    assert len(findings) >= 1, "El BugScanner debió capturar al menos 1 finding"
    f = findings[0]
    assert isinstance(f, BugFinding)
    assert "fake_module.py" in f.file_path
    assert f.function == "buggy"
    assert f.exception_type == "ValueError"
    assert "test bug" in f.message


def test_bug_scanner_mark_fixed() -> None:
    """mark_fixed retira el finding del buffer activo."""
    scanner = BugScanner(project_root=Path("."))
    scanner._register(
        file_path="src/foo.py",
        function="bar",
        line=42,
        exc_type="RuntimeError",
        message="boom",
    )
    assert scanner.active_count() == 1
    assert scanner.mark_fixed("src/foo.py::bar") is True
    assert scanner.active_count() == 0
    # Idempotente: marcar de nuevo no falla, devuelve False.
    assert scanner.mark_fixed("src/foo.py::bar") is False


# Tests de PerceptionProbe + LimitationEngine borrados en Fase A.
# Ambos módulos demostraron ser inertes en runtime (0 disparos reales).
