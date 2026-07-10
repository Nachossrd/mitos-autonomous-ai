"""
==============================================================================
 Proyecto MITOS - BugScanner (Fase 8+ — autodetección de fallos)
==============================================================================

Cierra el bucle "MITOS debería darse cuenta solo cuando su código
falla y repararlo". Sin esto, las excepciones se loguean y mueren
ahí — el `Archaeologist` solo encuentra debilidades estructurales
(docstrings, type hints, funciones largas), nunca BUGS REALES.

Implementación: un `logging.Handler` enganchado al logger raíz
captura todo record `>= ERROR` o con `exc_info`. Para cada uno
extrae la primera frame de stack dentro de `src/` y produce una
`BugFinding` con `file::function`. Después de cada ciclo, el
`CognitiveEngine` llama `drain_findings()` y los inyecta como
weaknesses urgentes (prefijo `[BUG]`) en el siguiente ciclo.

El `decision_engine` y `_plan_action` ya saben elegir
`self_modify` sobre weaknesses — solo necesitan recibirlas.

Convenciones:
  - Logger jerárquico `mitos.bug_scanner`.
  - Sin overhead de I/O: no parsea archivos de log, captura en RAM.
==============================================================================
"""

from __future__ import annotations

import logging
import re
import time
import traceback
from dataclasses import dataclass
from pathlib import Path

log = logging.getLogger("mitos.bug_scanner")


# Tamaño del buffer circular de findings. Más que esto y empezamos a
# reportar bugs antiguos que ya no son relevantes.
_MAX_FINDINGS: int = 50

# TTL antes de auto-purgar un finding sin re-emisión.
_FINDING_TTL_S: float = 600.0  # 10 min

# Regex para extraer `archivo.py", line N, in funcion`.
_FRAME_RE: re.Pattern[str] = re.compile(
    r'File "([^"]+)", line (\d+), in ([^\s]+)'
)


@dataclass(frozen=True)
class BugFinding:
    """Un fallo detectado en el código fuente del propio MITOS."""

    file_path: str        # ruta relativa al repo, e.g. "src/brain/agent.py"
    function: str         # nombre de la función afectada
    line: int
    exception_type: str   # "ValueError", "AttributeError", ...
    message: str          # primeros 200 chars del mensaje
    detected_at: float

    @property
    def target_id(self) -> str:
        """Formato compatible con weakness: `src/foo.py::func`."""
        return f"{self.file_path}::{self.function}"

    def as_weakness(self) -> str:
        """Render para inyectar en `world_state['weaknesses']`."""
        return (
            f"[BUG] {self.file_path}:{self.function} (L{self.line}) — "
            f"{self.exception_type}: {self.message[:120]}"
        )


class BugScanner(logging.Handler):
    """Captura excepciones en tiempo real → produce weaknesses urgentes."""

    def __init__(self, project_root: str | Path) -> None:
        super().__init__(level=logging.WARNING)
        self.root: Path = Path(project_root).resolve()
        # Dedupe por (file, function) — un bug repetido cuenta una vez.
        self._findings: dict[tuple[str, str], BugFinding] = {}
        log.debug("BugScanner enganchado a root=%s", self.root)

    # ==================================================================
    #                       INTERCEPTAR LOGS
    # ==================================================================
    def emit(self, record: logging.LogRecord) -> None:
        """Llamado por logging para cada record. NO debe lanzar nunca."""
        try:
            if record.exc_info:
                self._handle_exception(record)
            elif record.levelno >= logging.ERROR:
                self._handle_error_message(record)
        except Exception:  # noqa: BLE001
            # Un handler que crashea bloquea TODO el logging.
            # Silencio total y nunca volvemos a tocar el record.
            pass

    def _handle_exception(self, record: logging.LogRecord) -> None:
        """Procesa un record con `exc_info` adjunto (stack trace)."""
        exc_type, exc_value, exc_tb = record.exc_info or (None, None, None)
        if exc_type is None or exc_tb is None:
            return

        # Buscamos la PRIMERA frame dentro de `src/` — esa es nuestra culpa,
        # no de stdlib o terceros.
        own_frame = self._first_own_frame(exc_tb)
        if own_frame is None:
            return

        file_rel, line, func = own_frame
        self._register(
            file_path=file_rel,
            function=func,
            line=line,
            exc_type=getattr(exc_type, "__name__", str(exc_type)),
            message=str(exc_value)[:200] if exc_value else "",
        )

    def _handle_error_message(self, record: logging.LogRecord) -> None:
        """Record nivel ERROR sin exc_info: usamos pathname:lineno:funcName."""
        try:
            file_abs = Path(record.pathname).resolve()
            file_rel = self._relativize(file_abs)
        except (OSError, ValueError):
            return
        if file_rel is None:
            return
        self._register(
            file_path=file_rel,
            function=str(record.funcName or "?"),
            line=int(record.lineno or 0),
            exc_type="ErrorLog",
            message=record.getMessage()[:200],
        )

    # ==================================================================
    #                       MAPEO Y REGISTRO
    # ==================================================================
    def _first_own_frame(self, tb: object) -> tuple[str, int, str] | None:
        """Recorre el traceback y devuelve el primer frame dentro de src/."""
        try:
            formatted = "".join(traceback.format_tb(tb))  # type: ignore[arg-type]
        except Exception:  # noqa: BLE001
            return None

        for match in _FRAME_RE.finditer(formatted):
            file_abs_str, line_str, func = match.groups()
            try:
                file_rel = self._relativize(Path(file_abs_str).resolve())
            except (OSError, ValueError):
                continue
            if file_rel is not None:
                return (file_rel, int(line_str), func)
        return None

    def _relativize(self, path: Path) -> str | None:
        """Devuelve path relativo al repo, o None si está fuera."""
        try:
            rel = path.relative_to(self.root)
        except ValueError:
            return None
        rel_str = str(rel).replace("\\", "/")
        # Solo nos interesan los .py de nuestro src/ y tests/, no de venv.
        if not (rel_str.startswith("src/") or rel_str.startswith("tests/")):
            return None
        return rel_str

    def _register(
        self,
        file_path: str,
        function: str,
        line: int,
        exc_type: str,
        message: str,
    ) -> None:
        key = (file_path, function)
        # Si ya existía, refrescamos timestamp pero NO duplicamos.
        existing = self._findings.get(key)
        if existing is not None:
            self._findings[key] = BugFinding(
                file_path=existing.file_path,
                function=existing.function,
                line=line or existing.line,
                exception_type=exc_type,
                message=message,
                detected_at=time.time(),
            )
            return

        if len(self._findings) >= _MAX_FINDINGS:
            # Tiramos el más antiguo.
            oldest_key = min(
                self._findings, key=lambda k: self._findings[k].detected_at
            )
            self._findings.pop(oldest_key, None)

        self._findings[key] = BugFinding(
            file_path=file_path,
            function=function,
            line=line,
            exception_type=exc_type,
            message=message,
            detected_at=time.time(),
        )

    # ==================================================================
    #                       API PARA EL ENGINE
    # ==================================================================
    def drain_findings(self) -> list[BugFinding]:
        """Devuelve findings vigentes y purga los expirados.

        NO limpia los activos — los mismos bugs siguen siendo weaknesses
        hasta que `mark_fixed()` los retire o el TTL los expire.
        """
        now = time.time()
        # Purgar expirados.
        expired = [
            k for k, f in self._findings.items()
            if (now - f.detected_at) > _FINDING_TTL_S
        ]
        for k in expired:
            self._findings.pop(k, None)
        return list(self._findings.values())

    def mark_fixed(self, target_id: str) -> bool:
        """El CognitiveEngine llama esto cuando un self_modify exitoso
        toca el target del bug — retiramos el finding."""
        for key, finding in list(self._findings.items()):
            if finding.target_id == target_id:
                del self._findings[key]
                log.info("Bug %s marcado como reparado", target_id)
                return True
        return False

    def active_count(self) -> int:
        """Para el dashboard / métricas."""
        return len(self._findings)

    # ==================================================================
    #                       INSTALACIÓN / DESINSTALACIÓN
    # ==================================================================
    def install(self, logger_name: str = "mitos") -> None:
        """Engancha el handler al árbol de loggers `mitos.*`."""
        root = logging.getLogger(logger_name)
        if self not in root.handlers:
            root.addHandler(self)
            log.info("BugScanner instalado en logger '%s'", logger_name)

    def uninstall(self, logger_name: str = "mitos") -> None:
        root = logging.getLogger(logger_name)
        if self in root.handlers:
            root.removeHandler(self)
