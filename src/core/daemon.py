"""
==============================================================================
 Proyecto MITOS - MinimalDaemon (Fase B refactor — 3115 → ~250 LOC)
==============================================================================

Reescritura completa. El daemon clásico (Fases 2-7) tenía 3115 líneas con
un ciclo cognitivo, decision_engine, personality, goal_tree, strategic_planner
y self-modification pipeline. En el modo `--voice` (que es el que se usa)
NADA de eso se ejecutaba — solo se usaban 5 atributos:

  - daemon.llm                    (LLMEngine)
  - daemon.memory                 (MemorySystem)
  - daemon.introspector           (find_weaknesses para self_improvement)
  - daemon._extra_weaknesses      (buffer que BugScanner inyecta)
  - daemon._alive                 (flag de shutdown)

Esta versión expone solo eso, con un constructor compatible con `src/main.py`.
Los 18 módulos del ciclo cognitivo viejo (decision_engine, personality,
goal_tree, mcp_factory, etc.) se borraron en Fase B porque NUNCA se
ejecutaron en producción.

Si en el futuro quieres recuperar el ciclo autónomo clásico, el backup
está en `src/core/daemon.py.backup`.

Convenciones:
  - Logger jerárquico `mitos.core.daemon`.
  - `from __future__ import annotations`, tipado moderno.
==============================================================================
"""

from __future__ import annotations

import logging
import signal
import threading
from pathlib import Path
from typing import Any

from src.brain.llm_engine import LLMEngine
from src.brain.memory import MemorySystem

log = logging.getLogger("mitos.core.daemon")


# ============================================================================
# Stubs ligeros para los pocos atributos que el modo --voice consulta.
# ============================================================================

class _MinimalIntrospector:
    """Reemplazo mínimo de src.self_mod.introspector.Introspector.

    Solo expone `find_weaknesses()` que SelfImprovementLoop usa para
    elegir un target estructural. Sin esto el `_try_self_modify_structural`
    siempre devolvería None y el loop nunca propondría mejoras.

    Implementación: walk del filesystem buscando funciones largas y
    sin docstring. Sin AST complejo — basta para que el loop tenga
    targets reales que sugerir.
    """

    def __init__(self, project_root: Path) -> None:
        self.root = project_root
        self._cache: list[str] = []
        self._cache_mtime: float = 0.0

    def find_weaknesses(self) -> list[str]:
        """Devuelve lista de strings 'src/file.py:func - razón'."""
        import time as _time
        # Cacheo de 5 min para no escanear en cada ciclo del self_improvement.
        if self._cache and (_time.time() - self._cache_mtime) < 300:
            return list(self._cache)
        import ast
        weaknesses: list[str] = []
        for path in (self.root / "src").rglob("*.py"):
            if "__pycache__" in str(path):
                continue
            try:
                tree = ast.parse(path.read_text(encoding="utf-8"))
            except (SyntaxError, OSError):
                continue
            rel = str(path.relative_to(self.root)).replace("\\", "/")
            for node in ast.walk(tree):
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    end = getattr(node, "end_lineno", node.lineno)
                    n_lines = (end or node.lineno) - node.lineno + 1
                    if n_lines > 50:
                        weaknesses.append(
                            f"{rel}:{node.name} - demasiado larga "
                            f"({n_lines} líneas, umbral 50)"
                        )
                    if not ast.get_docstring(node):
                        weaknesses.append(
                            f"{rel}:{node.name} - sin docstring"
                        )
        self._cache = weaknesses[:200]
        self._cache_mtime = _time.time()
        log.info("find_weaknesses: %d debilidades detectadas", len(weaknesses))
        return list(self._cache)


# ============================================================================
# MitosDaemon — versión minimal para modo --voice
# ============================================================================

class MitosDaemon:
    """Daemon mínimo: solo lo que `--voice` y SelfImprovementLoop necesitan.

    Compatible con el constructor anterior — `meta_objective`, `dry_run`,
    `max_cycles` se aceptan pero solo `project_root` y los subsistemas
    cargados al __init__ son funcionales.
    """

    def __init__(
        self,
        project_root: str | Path,
        meta_objective: str = "",
        dry_run: bool = False,
        max_cycles: int | None = None,
        transcript_path: str | Path | None = None,
    ) -> None:
        self.root: Path = Path(project_root).resolve()
        self.meta_objective: str = meta_objective
        self.dry_run: bool = bool(dry_run)
        self.max_cycles: int | None = max_cycles
        self._alive: bool = False

        log.info("inicializando daemon en %s", self.root)

        # Subsistemas que el modo voz consulta.
        self.llm: LLMEngine = LLMEngine()
        self.memory: MemorySystem = MemorySystem()
        self.introspector: _MinimalIntrospector = _MinimalIntrospector(self.root)

        # Buffer de weaknesses que BugScanner / SelfImprovement inyectan
        # antes de que la próxima respuesta de MITOS las consume.
        self._extra_weaknesses: list[str] = []

        log.info(
            "daemon inicializado. meta=%r",
            (meta_objective[:80] if meta_objective else ""),
        )

    # ==================================================================
    #             API que CognitiveEngine y DialogLoop esperan
    # ==================================================================
    def _install_signal_handlers(self) -> None:
        """Stub — los signals los maneja el dashboard/dialog en su loop."""
        # En el daemon clásico esto registraba SIGINT/SIGTERM. En --voice
        # el DialogLoop ya captura KeyboardInterrupt en su run(), así que
        # no necesitamos handlers redundantes.
        try:
            signal.signal(signal.SIGINT, signal.default_int_handler)
        except (ValueError, OSError):
            pass

    def _bootstrap(self) -> None:
        """Stub — scan inicial opcional.

        Si quieres pre-poblar weaknesses al arrancar para que el primer
        ciclo del SelfImprovementLoop tenga algo que sugerir, descoméntalo.
        """
        # No-op intencional. El introspector cachea on-demand cuando
        # SelfImprovement lo invoca.
        return

    def _cognitive_cycle(self) -> Any:
        """Stub — el ciclo cognitivo viejo. En --voice no se usa.

        Si CognitiveEngine.run_cycle() lo invoca (modo dashboard), devuelve
        un resultado vacío que no rompe el flow.
        """
        from dataclasses import dataclass, field
        @dataclass
        class _NoopCycleResult:
            cycle_id: int = 0
            action_taken: str = "noop"
            goal_pursued: str = ""
            success: bool = True
            duration_s: float = 0.0
            self_modified: bool = False
            outcome: str = ""
            decision: Any = None
            drive_chosen: str = "none"
        return _NoopCycleResult()

    def _post_cycle(self, result: Any) -> None:
        """Stub — el daemon clásico actualizaba personality/strategic aquí.

        En --voice no se usa.
        """
        return

    def start(self) -> None:
        """Stub para compatibilidad con run_daemon.py viejo.

        En --voice el loop lo lleva el DialogLoop. Si alguien llama a
        start() (modo headless legacy), simplemente esperamos a que se
        cancele.
        """
        log.warning(
            "MitosDaemon.start() llamado: el ciclo cognitivo clásico fue "
            "removido en Fase B. Usa `python -m src.main --voice` o "
            "modo dashboard. Saliendo."
        )
        return
