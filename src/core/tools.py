"""
==============================================================================
 Proyecto MITOS - Tools reales (Fase 8+ — fin del fingir)
==============================================================================

Antes de este módulo, MITOS decía "Sí, logré usarlo" cuando le pedías
acceder a un archivo. Mentía sin querer: el LLM razonaba sobre la
operación pero no había ningún I/O detrás. Esto resolvía cero
problemas reales y rompía la confianza del operador.

Este módulo le da a MITOS la capacidad de TOCAR DE VERDAD el disco
del operador. Cuando el operador menciona una ruta, el CLI brain
llama a `FilesystemTool.list_dir()` AUTOMÁTICAMENTE y le inyecta el
contenido REAL al contexto del LLM. Si el LLM dice "leí el archivo X",
es porque el archivo X salió en el contexto inyectado.

Diseño:
  - Stateless: cada llamada es independiente. Sin caché interno.
  - Safe: rutas resueltas con `Path.resolve()`. Sin traversal raro.
  - Tamaño limitado: leer un archivo entero de 100 MB rompe el contexto
    del LLM. Truncamos a `_MAX_READ_BYTES`.
  - Sin "out-of-the-box dangerous": no expone exec() ni eval(). Para
    ejecutar código existe `CodeRunner` aparte (subprocess controlado).

Convenciones:
  - Logger jerárquico `mitos.tools`.
==============================================================================
"""

from __future__ import annotations

import logging
import os
import re
from dataclasses import dataclass
from pathlib import Path

log = logging.getLogger("mitos.tools")


_MAX_READ_BYTES: int = 32_000          # ~8k tokens approximados
_MAX_DIR_ENTRIES: int = 200            # antes de truncar listing
_PATH_HINT_RE: re.Pattern[str] = re.compile(
    # Detecta rutas estilo Windows (C:\X\Y) o estilo Unix (/x/y).
    # Quoted con `, ', " también. Se usa para auto-exploración.
    r'(?:^|[\s`\'"])([A-Za-z]:\\[^\s`\'"]+|/[^\s`\'"]+)'
)


@dataclass(frozen=True)
class ListEntry:
    name: str
    is_dir: bool
    size_bytes: int  # 0 para directorios


@dataclass(frozen=True)
class ReadResult:
    path: str
    truncated: bool
    bytes_read: int
    content: str
    error: str = ""


class FilesystemTool:
    """Acceso real al disco — sin sandbox, con guardas mínimas."""

    # ==================================================================
    #                       LISTAR DIRECTORIO
    # ==================================================================
    @staticmethod
    def list_dir(path: str | Path) -> list[ListEntry]:
        """Lista entradas. Vacía si el path no existe o no es dir."""
        try:
            p = Path(path).expanduser().resolve()
        except (OSError, RuntimeError) as e:
            log.debug("list_dir resolve(%s): %s", path, e)
            return []
        if not p.is_dir():
            return []

        try:
            raw_entries = list(p.iterdir())
        except (OSError, PermissionError) as e:
            log.warning("list_dir(%s) sin permiso: %s", p, e)
            return []

        # Orden estable: dirs primero, luego archivos por nombre.
        raw_entries.sort(key=lambda e: (not e.is_dir(), e.name.lower()))

        entries: list[ListEntry] = []
        for e in raw_entries[:_MAX_DIR_ENTRIES]:
            try:
                size = 0 if e.is_dir() else e.stat().st_size
            except OSError:
                size = 0
            entries.append(ListEntry(
                name=e.name,
                is_dir=e.is_dir(),
                size_bytes=int(size),
            ))
        return entries

    # ==================================================================
    #                       LEER ARCHIVO
    # ==================================================================
    @staticmethod
    def read_file(path: str | Path) -> ReadResult:
        """Lee un archivo de texto. Trunca si excede `_MAX_READ_BYTES`."""
        try:
            p = Path(path).expanduser().resolve()
        except (OSError, RuntimeError) as e:
            return ReadResult(
                path=str(path), truncated=False, bytes_read=0,
                content="", error=f"path inválido: {e}",
            )
        if not p.is_file():
            return ReadResult(
                path=str(p), truncated=False, bytes_read=0,
                content="", error="no es un archivo o no existe",
            )

        try:
            data = p.read_bytes()
        except (OSError, PermissionError) as e:
            return ReadResult(
                path=str(p), truncated=False, bytes_read=0,
                content="", error=f"sin permiso: {e}",
            )

        truncated = len(data) > _MAX_READ_BYTES
        if truncated:
            data = data[:_MAX_READ_BYTES]

        try:
            text = data.decode("utf-8")
        except UnicodeDecodeError:
            try:
                text = data.decode("latin-1")
            except UnicodeDecodeError:
                return ReadResult(
                    path=str(p), truncated=truncated,
                    bytes_read=len(data), content="",
                    error="archivo binario (no decodificable como texto)",
                )

        return ReadResult(
            path=str(p),
            truncated=truncated,
            bytes_read=len(data),
            content=text,
        )

    # ==================================================================
    #                       GLOB
    # ==================================================================
    @staticmethod
    def glob(base: str | Path, pattern: str) -> list[str]:
        """Devuelve rutas que matchean `pattern` bajo `base`."""
        try:
            p = Path(base).expanduser().resolve()
        except (OSError, RuntimeError):
            return []
        if not p.is_dir():
            return []
        try:
            return [str(m) for m in p.glob(pattern)][:_MAX_DIR_ENTRIES]
        except (OSError, ValueError) as e:
            log.debug("glob(%s, %s): %s", base, pattern, e)
            return []

    # ==================================================================
    #                       EXISTE
    # ==================================================================
    @staticmethod
    def exists(path: str | Path) -> bool:
        try:
            return Path(path).expanduser().resolve().exists()
        except (OSError, RuntimeError):
            return False

    # ==================================================================
    #                       EXTRAER RUTAS DEL TEXTO
    # ==================================================================
    @staticmethod
    def extract_path_hints(text: str) -> list[str]:
        """Encuentra rutas Windows/Unix mencionadas en `text`.

        Útil para que el CLI brain explore AUTOMÁTICAMENTE las rutas que
        el operador menciona, antes de que el LLM responda — así el LLM
        ya tiene el contenido real en el contexto y no inventa.
        """
        if not text:
            return []
        hits = _PATH_HINT_RE.findall(text)
        # Limpieza: quitar separadores finales que el LLM a veces sufija.
        cleaned: list[str] = []
        seen: set[str] = set()
        for h in hits:
            h = h.rstrip(".,;:!?)]}'\"`")
            if h and h not in seen:
                cleaned.append(h)
                seen.add(h)
        return cleaned


class CodeRunner:
    """Ejecuta código Python en subprocess. Aislado del intérprete actual.

    Tras Python 3.10+ usamos `subprocess.run` con timeout y captura. NO
    pasamos por `shell=True` jamás — los argumentos van como list[str].
    """

    @staticmethod
    def run_python(
        code: str,
        timeout_s: float = 10.0,
        cwd: str | Path | None = None,
    ) -> dict[str, str | int]:
        """Ejecuta `code` con `python -c`. Devuelve dict con stdout/stderr/rc."""
        import subprocess
        import sys
        try:
            completed = subprocess.run(
                [sys.executable, "-c", code],
                capture_output=True,
                text=True,
                timeout=float(timeout_s),
                cwd=str(cwd) if cwd else None,
                shell=False,
                env={**os.environ, "PYTHONIOENCODING": "utf-8"},
            )
            return {
                "stdout": completed.stdout[-4000:],
                "stderr": completed.stderr[-4000:],
                "returncode": int(completed.returncode),
            }
        except subprocess.TimeoutExpired:
            return {
                "stdout": "",
                "stderr": f"Timeout tras {timeout_s}s",
                "returncode": -1,
            }
        except (OSError, ValueError) as e:
            return {
                "stdout": "",
                "stderr": f"Error ejecutando subprocess: {e}",
                "returncode": -1,
            }
