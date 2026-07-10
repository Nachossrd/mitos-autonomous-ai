"""
==============================================================================
 Proyecto MITOS - SelfPatcher (Fase C — auto-modificación REAL del propio código)
==============================================================================

El operador autorizó "luz verde" para que MITOS modifique su propio
código. Hasta ahora `_try_self_modify_structural` solo añadía la
weakness a una cola que nadie consumía — era teatro.

Este módulo cierra el bucle de verdad:

  1. Toma una weakness ("src/X.py:func - demasiado larga")
  2. Lee el archivo, extrae la función con AST
  3. Pide a Gemini Coder que la refactorice arreglando el problema
  4. Valida que el código nuevo COMPILE (ast.parse)
  5. Hace backup del archivo entero en .mitos_backups/
  6. Reemplaza la función en el archivo
  7. Corre pytest sobre tests/
  8. Si tests pasan → mantiene el cambio + reporta + persiste en
     `data/self_patches.json`
  9. Si tests fallan → ROLLBACK desde backup + reporta

GUARDRAILS DUROS (no negociables):
  - Whitelist de paths permitidos. Files CRÍTICOS (daemon, main,
    dialog_loop, voice_engine, intelligence_router) NUNCA se tocan.
  - Cap: 3 patches exitosos por sesión.
  - Cap: 1 intento por (archivo, función) por sesión.
  - AST parse antes Y después.
  - pytest --timeout=30 obligatorio.

Convenciones:
  - Logger jerárquico `mitos.self_patcher`.
==============================================================================
"""

from __future__ import annotations

import ast
import json
import logging
import re
import shutil
import subprocess
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

log = logging.getLogger("mitos.self_patcher")


# ============================================================================
# Constantes de seguridad
# ============================================================================

# Archivos que MITOS puede auto-mutar. Todo lo que NO esté aquí está
# protegido. Por defecto, archivos legacy del CLI brain + utilities.
_SAFE_PATHS_PREFIX: tuple[str, ...] = (
    "src/brain/agent.py",
    "src/brain/response_filter.py",
    "src/brain/interactive.py",  # CLI legacy — no afecta --voice
    "src/scripts/",
    "src/tools/",
    "src/core/operator_preferences.py",
    "src/core/user_profile.py",
    "src/core/emotion_detector.py",
    "src/core/capability_gaps.py",
    "src/core/perception_probe.py",   # ya borrado pero por si vuelve
    "src/core/filesystem_indexer.py",
    "src/core/face_recognizer.py",
    "src/core/vision_glance.py",
    "src/core/auto_bootstrap.py",
    "src/core/action_executor.py",
    "src/core/tools.py",
    "src/core/tool_builder.py",
    "src/core/bug_scanner.py",
    "src/core/self_improvement.py",
    "src/core/self_patcher.py",       # MITOS puede mejorarme a mí también
    "src/core/cognitive_engine.py",
)

# Archivos PROHIBIDOS — romperlos mata la sesión --voice.
_PROTECTED_PATHS: frozenset[str] = frozenset({
    "src/main.py",
    "src/core/daemon.py",
    "src/core/dialog_loop.py",
    "src/core/voice_engine.py",
    "src/core/intelligence_router.py",
    "src/core/sensor_hub.py",
    "src/brain/llm_engine.py",
    "src/brain/memory.py",
    "src/brain/identity.py",
})

# Caps por sesión.
_MAX_PATCHES_PER_SESSION: int = 3
_MAX_ATTEMPTS_PER_TARGET: int = 1
_PYTEST_TIMEOUT_S: int = 60


@dataclass
class PatchAttempt:
    """Registro de un intento de auto-patch."""

    target_file: str
    target_function: str
    weakness: str
    timestamp: float = field(default_factory=time.time)
    success: bool = False
    error: str = ""
    backup_path: str = ""
    pytest_passed: bool = False
    bytes_before: int = 0
    bytes_after: int = 0


# ============================================================================
# SelfPatcher
# ============================================================================

class SelfPatcher:
    """Auto-modificación de código MITOS con validación + rollback."""

    def __init__(
        self,
        project_root: str | Path,
        router: Any,
        memory: Any = None,
    ) -> None:
        self.root: Path = Path(project_root).resolve()
        self.router: Any = router
        self.memory: Any = memory
        self.backup_dir: Path = self.root / ".mitos_backups"
        self.registry_path: Path = self.root / "data" / "self_patches.json"
        self.backup_dir.mkdir(parents=True, exist_ok=True)
        self.registry_path.parent.mkdir(parents=True, exist_ok=True)

        # Estado de la sesión.
        self._patches_this_session: int = 0
        self._attempted_targets: set[str] = set()
        self._history: list[PatchAttempt] = self._load_history()

        log.info(
            "SelfPatcher listo: %d patches históricos, %d safe paths",
            len(self._history), len(_SAFE_PATHS_PREFIX),
        )

    # ==================================================================
    #                       PUERTA DE ENTRADA
    # ==================================================================
    def try_patch(self, weakness: str) -> PatchAttempt | None:
        """Intenta resolver una weakness mutando código. None si no aplica.

        Args:
            weakness: string del Introspector, e.g.
                "src/brain/agent.py:__init__ - demasiado larga (65 líneas...)"

        Returns:
            PatchAttempt con success=True si todo bien, False si rollbacked,
            None si la weakness no cumple los guardrails.
        """
        if self._patches_this_session >= _MAX_PATCHES_PER_SESSION:
            log.debug("Cap de patches alcanzado (%d)", _MAX_PATCHES_PER_SESSION)
            return None

        parsed = self._parse_weakness(weakness)
        if parsed is None:
            return None
        target_file, target_func, reason = parsed

        # GUARDRAIL 1: whitelist.
        if not self._is_safe_to_patch(target_file):
            log.debug("Path %r no está en whitelist — skip", target_file)
            return None

        # GUARDRAIL 2: no reintentar mismo target.
        target_key = f"{target_file}::{target_func}"
        if target_key in self._attempted_targets:
            return None
        self._attempted_targets.add(target_key)

        attempt = PatchAttempt(
            target_file=target_file,
            target_function=target_func,
            weakness=weakness,
        )

        # ---- Pipeline ----
        try:
            file_path = self.root / target_file
            if not file_path.is_file():
                attempt.error = f"file not found: {file_path}"
                self._record(attempt)
                return attempt

            original_source = file_path.read_text(encoding="utf-8")
            attempt.bytes_before = len(original_source)

            # Validar que el original compila (sanity check).
            try:
                ast.parse(original_source)
            except SyntaxError as e:
                attempt.error = f"original file ya no compila: {e}"
                self._record(attempt)
                return attempt

            # Extraer función con AST.
            func_source = self._extract_function_source(
                original_source, target_func,
            )
            if not func_source:
                attempt.error = f"no encontré función {target_func} en AST"
                self._record(attempt)
                return attempt

            # Delegar refactor a Gemini.
            new_func_source = self._delegate_refactor(
                func_source, reason, target_file, target_func,
            )
            if not new_func_source:
                attempt.error = "Gemini no devolvió código válido"
                self._record(attempt)
                return attempt

            # Validar que el NUEVO código compila.
            try:
                ast.parse(new_func_source)
            except SyntaxError as e:
                attempt.error = f"código nuevo NO compila: {e}"
                log.warning("Patch rechazado por SyntaxError: %s", e)
                self._record(attempt)
                return attempt

            # Backup + escribir nueva versión.
            backup_path = self._backup_file(file_path)
            attempt.backup_path = str(backup_path)

            new_source = self._replace_function_in_source(
                original_source, target_func, new_func_source,
            )
            if not new_source or new_source == original_source:
                attempt.error = "no se pudo aplicar el reemplazo"
                self._record(attempt)
                return attempt

            # Validar archivo entero antes de escribir.
            try:
                ast.parse(new_source)
            except SyntaxError as e:
                attempt.error = (
                    f"archivo completo no compila tras reemplazo: {e}"
                )
                self._record(attempt)
                return attempt

            file_path.write_text(new_source, encoding="utf-8")
            attempt.bytes_after = len(new_source)

            # Test suite — el guardrail más importante.
            attempt.pytest_passed = self._run_pytest()
            if not attempt.pytest_passed:
                # ROLLBACK.
                log.warning(
                    "pytest falló tras patch de %s — restaurando backup",
                    target_file,
                )
                shutil.copy2(backup_path, file_path)
                attempt.error = "pytest falló — rolled back"
                self._record(attempt)
                return attempt

            # Éxito.
            attempt.success = True
            self._patches_this_session += 1
            self._record(attempt)
            log.info(
                "✓ Patch exitoso #%d/%d: %s::%s (%d→%d bytes)",
                self._patches_this_session, _MAX_PATCHES_PER_SESSION,
                target_file, target_func,
                attempt.bytes_before, attempt.bytes_after,
            )
            return attempt
        except Exception as e:  # noqa: BLE001
            attempt.error = f"excepción no manejada: {e}"
            log.warning("try_patch crasheó: %s", e)
            self._record(attempt)
            return attempt

    # ==================================================================
    #                       PARSING DE WEAKNESS
    # ==================================================================
    @staticmethod
    def _parse_weakness(weakness: str) -> tuple[str, str, str] | None:
        """Devuelve (path, function, reason) o None si no se puede parsear.

        Formato esperado del Introspector:
            "src/brain/agent.py:__init__ - demasiado larga (65 líneas...)"
        """
        if not weakness:
            return None
        m = re.match(
            r"(?:\[BUG\]\s+)?([\w/.-]+\.py):([\w_]+)\s+-\s+(.+)",
            weakness.strip(),
        )
        if not m:
            return None
        return (m.group(1), m.group(2), m.group(3))

    @staticmethod
    def _is_safe_to_patch(rel_path: str) -> bool:
        """True si el path está permitido."""
        normalized = rel_path.replace("\\", "/")
        if normalized in _PROTECTED_PATHS:
            return False
        return any(
            normalized.startswith(p) or normalized == p
            for p in _SAFE_PATHS_PREFIX
        )

    # ==================================================================
    #                       AST: extracción y reemplazo
    # ==================================================================
    @staticmethod
    def _extract_function_source(file_source: str, func_name: str) -> str:
        """Devuelve el texto fuente de la función `func_name` (o vacío)."""
        try:
            tree = ast.parse(file_source)
        except SyntaxError:
            return ""
        lines = file_source.splitlines(keepends=True)
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                if node.name == func_name:
                    start = node.lineno - 1
                    end = getattr(node, "end_lineno", node.lineno)
                    return "".join(lines[start:end])
        return ""

    @staticmethod
    def _replace_function_in_source(
        file_source: str, func_name: str, new_func_source: str,
    ) -> str:
        """Reemplaza la función `func_name` en el source. '' si falla."""
        try:
            tree = ast.parse(file_source)
        except SyntaxError:
            return ""
        lines = file_source.splitlines(keepends=True)
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                if node.name == func_name:
                    start = node.lineno - 1
                    end = getattr(node, "end_lineno", node.lineno)
                    # Preservar la indentación del def original.
                    original_indent = len(lines[start]) - len(lines[start].lstrip())
                    new_lines = new_func_source.splitlines(keepends=True)
                    if new_lines:
                        new_first_indent = (
                            len(new_lines[0]) - len(new_lines[0].lstrip())
                        )
                        delta = original_indent - new_first_indent
                        if delta > 0:
                            new_lines = [" " * delta + l for l in new_lines]
                        elif delta < 0:
                            new_lines = [
                                l[(-delta):] if l.strip() else l
                                for l in new_lines
                            ]
                    # Asegurar newline final si falta.
                    if new_lines and not new_lines[-1].endswith("\n"):
                        new_lines[-1] += "\n"
                    return "".join(lines[:start] + new_lines + lines[end:])
        return ""

    # ==================================================================
    #                       DELEGACIÓN A GEMINI
    # ==================================================================
    def _delegate_refactor(
        self, func_source: str, reason: str,
        target_file: str, target_func: str,
    ) -> str:
        """Pide a Gemini que refactorice la función. '' si falla."""
        if self.router is None:
            return ""
        if not getattr(self.router, "_has_internet", False):
            return ""
        if not self.router.external_models:
            return ""

        from src.core.intelligence_router import RoutingDecision
        best = min(self.router.external_models, key=lambda m: m.priority)
        decision = RoutingDecision(
            provider=best.provider,
            model_name=best.model_name,
            reason="self-patch: refactor de función",
            estimated_tokens=2000,
        )
        prompt = (
            f"Eres un programador Python senior. La siguiente función "
            f"tiene este problema: '{reason}'.\n\n"
            f"Archivo: {target_file}\n"
            f"Función: {target_func}\n\n"
            f"CÓDIGO ORIGINAL:\n"
            f"```python\n{func_source}\n```\n\n"
            "REGLAS ABSOLUTAS:\n"
            "- PRESERVA la signature exacta (nombre, parámetros, types, "
            "  decoradores si los hay).\n"
            "- PRESERVA el comportamiento observable — el código nuevo "
            "  debe ser DROP-IN replacement.\n"
            "- Si la función llama a `self.X`, mantén esas llamadas.\n"
            "- Si necesitas funciones auxiliares, INCLÚYELAS antes "
            "  de la función principal en el mismo bloque.\n"
            "- Sintaxis Python 3.11+. Tipado moderno.\n"
            "- Sin imports nuevos (usa solo los que ya están en el archivo "
            "  o stdlib obvios como `re`, `time`).\n\n"
            "Devuelve SOLO el código refactorizado, en un único bloque "
            "```python ... ```. Sin explicación previa ni posterior."
        )
        try:
            answer = self.router.execute_sync(
                task=prompt, routing=decision, system_prompt="",
            )
        except Exception as e:  # noqa: BLE001
            log.warning("router para self_patch: %s", e)
            return ""
        if not answer:
            return ""
        # Extraer bloque ```python ... ```.
        m = re.search(r"```(?:python|py)?\s*\n(.*?)\n```", answer, re.DOTALL)
        if m:
            return m.group(1).strip()
        # Fallback: si arranca con def, asumimos que es todo código.
        if re.match(r"^\s*(?:def|async def|@)", answer.strip()):
            return answer.strip()
        return ""

    # ==================================================================
    #                       BACKUP + ROLLBACK
    # ==================================================================
    def _backup_file(self, file_path: Path) -> Path:
        """Copia el archivo a .mitos_backups/{nombre}_{ts}.bak."""
        ts = int(time.time())
        backup_name = (
            str(file_path.relative_to(self.root)).replace("\\", "_")
            .replace("/", "_")
            + f"_{ts}.bak"
        )
        backup_path = self.backup_dir / backup_name
        shutil.copy2(file_path, backup_path)
        log.info("Backup: %s → %s", file_path.name, backup_path.name)
        return backup_path

    # ==================================================================
    #                       PYTEST
    # ==================================================================
    def _run_pytest(self) -> bool:
        """True si la suite de tests pasa."""
        try:
            completed = subprocess.run(
                [sys.executable, "-m", "pytest", "tests/", "-q", "--tb=short"],
                capture_output=True, text=True,
                timeout=_PYTEST_TIMEOUT_S,
                cwd=str(self.root),
            )
        except subprocess.TimeoutExpired:
            log.warning("pytest timeout (%ds) — fail", _PYTEST_TIMEOUT_S)
            return False
        except Exception as e:  # noqa: BLE001
            log.warning("pytest no se pudo ejecutar: %s", e)
            return False
        ok = completed.returncode == 0
        if not ok:
            log.warning(
                "pytest rc=%d. stdout tail:\n%s",
                completed.returncode,
                (completed.stdout or "")[-500:],
            )
        return ok

    # ==================================================================
    #                       PERSISTENCIA
    # ==================================================================
    def _load_history(self) -> list[PatchAttempt]:
        if not self.registry_path.is_file():
            return []
        try:
            raw = json.loads(self.registry_path.read_text(encoding="utf-8"))
            return [PatchAttempt(**a) for a in raw.get("attempts", [])]
        except (OSError, ValueError, TypeError) as e:
            log.debug("load patch history: %s", e)
            return []

    def _record(self, attempt: PatchAttempt) -> None:
        self._history.append(attempt)
        try:
            self.registry_path.write_text(
                json.dumps(
                    {"attempts": [asdict(a) for a in self._history]},
                    indent=2, ensure_ascii=False,
                ),
                encoding="utf-8",
            )
        except OSError as e:
            log.debug("persist history: %s", e)
        # Guardar en memoria semántica también.
        if self.memory is not None and attempt.success:
            try:
                self.memory.store_knowledge(
                    content=(
                        f"[SELF-PATCH EXITOSO] {attempt.target_file}::"
                        f"{attempt.target_function}\n"
                        f"Razón: {attempt.weakness}\n"
                        f"Bytes: {attempt.bytes_before} → {attempt.bytes_after}\n"
                        f"Backup: {attempt.backup_path}"
                    ),
                    importance=0.95,
                    source="self_patcher",
                )
            except Exception as e:  # noqa: BLE001
                log.debug("memory.store self_patch: %s", e)

    def stats(self) -> dict[str, int]:
        return {
            "patches_this_session": self._patches_this_session,
            "history_total": len(self._history),
            "history_success": sum(1 for a in self._history if a.success),
        }
