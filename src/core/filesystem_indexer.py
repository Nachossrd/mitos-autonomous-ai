"""
==============================================================================
 Proyecto MITOS - FilesystemIndexer (Fase 8+ — exploración proactiva)
==============================================================================

Antes de este módulo, MITOS no sabía qué había en el sistema del
operador. Cuando el operador decía "en C:\\Downloads\\Radar hay un
modelo", MITOS tenía que pedir la ruta o adivinarla. Eso choca con
la autonomía que el operador quiere.

Este indexador corre en un thread daemon al arrancar el sistema y:

  1. Escanea las rutas típicas del operador (Downloads, Documents,
     Desktop, OneDrive si existe).
  2. Construye un mapa nombre_proyecto → ruta_absoluta.
  3. Persiste en `data/fs_index.json` para que sobreviva reinicios.
  4. Se refresca cada `_REFRESH_INTERVAL_S` segundos en background.
  5. Expone `find_project(nombre)` para que el dialog_loop traduzca
     "Radar" → `C:\\Users\\X\\Downloads\\Radar` automáticamente.

Diseño:
  - Profundidad limitada a 3 niveles desde cada ruta raíz — no
    exploramos un repo node_modules entero.
  - Skip de carpetas pesadas (.git, node_modules, .venv, __pycache__).
  - Sin lectura de contenido — solo nombres + tamaños.

Convenciones:
  - Logger `mitos.fs_indexer`.
==============================================================================
"""

from __future__ import annotations

import json
import logging
import os
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path

log = logging.getLogger("mitos.fs_indexer")


_REFRESH_INTERVAL_S: float = 600.0  # cada 10 min re-indexa
_MAX_DEPTH: int = 3
_SKIP_NAMES: frozenset[str] = frozenset({
    ".git", ".venv", "venv", "node_modules", "__pycache__",
    ".idea", ".vscode", "dist", "build", ".pytest_cache",
    ".mypy_cache", "AppData", "Local Settings",
})
_MAX_PROJECTS: int = 500


@dataclass
class ProjectEntry:
    """Una carpeta candidata a proyecto encontrada en disco."""

    name: str
    path: str
    depth: int
    has_code: bool       # True si dentro hay .py, .js, .ts, .go, .rs
    has_models: bool     # True si dentro hay .gguf, .pt, .onnx, .safetensors
    size_files: int      # cantidad de archivos en el primer nivel


@dataclass
class FilesystemIndex:
    projects: dict[str, ProjectEntry] = field(default_factory=dict)
    indexed_at: float = 0.0
    roots: list[str] = field(default_factory=list)


class FilesystemIndexer:
    """Escanea + indexa filesystem en background."""

    def __init__(self, project_root: str | Path) -> None:
        self.root: Path = Path(project_root).resolve()
        self.index_path: Path = self.root / "data" / "fs_index.json"
        self.index: FilesystemIndex = FilesystemIndex()
        self._thread: threading.Thread | None = None
        self._running: bool = False
        self._lock: threading.Lock = threading.Lock()

    # ==================================================================
    #                       LIFECYCLE
    # ==================================================================
    def start(self) -> None:
        """Lanza el thread de indexación. Idempotente."""
        if self._thread is not None and self._thread.is_alive():
            return
        # Cargamos índice persistido para tener algo desde el primer ciclo.
        self._load_persisted()
        self._running = True
        self._thread = threading.Thread(
            target=self._loop, name="fs-indexer", daemon=True,
        )
        self._thread.start()
        log.info("FilesystemIndexer arrancado")

    def stop(self) -> None:
        self._running = False

    # ==================================================================
    #                       API DE CONSULTA (thread-safe)
    # ==================================================================
    def find_project(self, name_hint: str) -> ProjectEntry | None:
        """Busca un proyecto por nombre parcial (case-insensitive)."""
        if not name_hint:
            return None
        needle = name_hint.lower()
        with self._lock:
            # Coincidencia exacta primero.
            for p in self.index.projects.values():
                if p.name.lower() == needle:
                    return p
            # Sustring.
            for p in self.index.projects.values():
                if needle in p.name.lower():
                    return p
        return None

    def list_projects(self) -> list[ProjectEntry]:
        with self._lock:
            return list(self.index.projects.values())

    def summary(self) -> str:
        with self._lock:
            n = len(self.index.projects)
            with_code = sum(1 for p in self.index.projects.values() if p.has_code)
            with_models = sum(1 for p in self.index.projects.values() if p.has_models)
        return f"{n} proyectos indexados ({with_code} con código, {with_models} con modelos)"

    # ==================================================================
    #                       BUCLE DE INDEXACIÓN
    # ==================================================================
    def _loop(self) -> None:
        while self._running:
            try:
                self._reindex()
                self._persist()
            except Exception as e:  # noqa: BLE001
                log.warning("indexación crasheó: %s", e)
            # Sleep en pasos cortos para reaccionar a stop() rápido.
            slept = 0.0
            while self._running and slept < _REFRESH_INTERVAL_S:
                time.sleep(5.0)
                slept += 5.0

    def _reindex(self) -> None:
        roots = self._candidate_roots()
        projects: dict[str, ProjectEntry] = {}

        for r in roots:
            if not r.is_dir():
                continue
            self._walk(r, depth=0, projects=projects)
            if len(projects) >= _MAX_PROJECTS:
                break

        with self._lock:
            self.index = FilesystemIndex(
                projects=projects,
                indexed_at=time.time(),
                roots=[str(r) for r in roots],
            )
        log.info("Reindex completo: %d proyectos", len(projects))

    def _walk(
        self,
        path: Path,
        depth: int,
        projects: dict[str, ProjectEntry],
    ) -> None:
        if depth > _MAX_DEPTH or len(projects) >= _MAX_PROJECTS:
            return
        try:
            entries = list(path.iterdir())
        except (OSError, PermissionError):
            return

        has_code = False
        has_models = False
        n_files = 0
        sub_dirs: list[Path] = []

        for e in entries:
            if e.name in _SKIP_NAMES or e.name.startswith("."):
                continue
            try:
                if e.is_dir():
                    sub_dirs.append(e)
                else:
                    n_files += 1
                    name_l = e.name.lower()
                    if name_l.endswith((".py", ".js", ".ts", ".go", ".rs", ".java")):
                        has_code = True
                    elif name_l.endswith((".gguf", ".pt", ".onnx", ".safetensors", ".pth")):
                        has_models = True
            except OSError:
                continue

        # Registramos esta carpeta como proyecto si tiene contenido interesante.
        if depth > 0 and (has_code or has_models or n_files >= 3):
            key = path.name
            # Resolución de colisiones: si ya existe ese nombre, sufijo con depth.
            if key in projects:
                key = f"{path.name}@{path.parent.name}"
            projects[key] = ProjectEntry(
                name=path.name,
                path=str(path),
                depth=depth,
                has_code=has_code,
                has_models=has_models,
                size_files=n_files,
            )

        # Recursión solo si no encontramos archivos clave en este nivel y
        # estamos cerca de la raíz — evita explorar `node_modules` aunque no
        # esté en SKIP_NAMES.
        if depth < _MAX_DEPTH:
            for sd in sub_dirs[:30]:  # cap por nivel
                self._walk(sd, depth=depth + 1, projects=projects)

    # ==================================================================
    #                       RAÍCES A ESCANEAR
    # ==================================================================
    @staticmethod
    def _candidate_roots() -> list[Path]:
        """Devuelve rutas típicas del operador (cross-platform)."""
        home = Path.home()
        candidates = [
            home / "Downloads",
            home / "Documents",
            home / "Desktop",
            home / "Documentos",
            home / "Descargas",
            home / "Escritorio",
            home / "Projects",
            home / "code",
            home / "src",
        ]
        # En Windows, OneDrive a veces redirige carpetas estándar.
        onedrive = os.environ.get("OneDrive")
        if onedrive:
            od = Path(onedrive)
            candidates += [od / "Downloads", od / "Documents", od / "Desktop"]
        return [c.resolve() for c in candidates if c.is_dir()]

    # ==================================================================
    #                       PERSISTENCIA
    # ==================================================================
    def _persist(self) -> None:
        try:
            self.index_path.parent.mkdir(parents=True, exist_ok=True)
            with self._lock:
                payload = {
                    "indexed_at": self.index.indexed_at,
                    "roots": self.index.roots,
                    "projects": {
                        k: {
                            "name": v.name,
                            "path": v.path,
                            "depth": v.depth,
                            "has_code": v.has_code,
                            "has_models": v.has_models,
                            "size_files": v.size_files,
                        } for k, v in self.index.projects.items()
                    },
                }
            self.index_path.write_text(
                json.dumps(payload, indent=2), encoding="utf-8",
            )
        except OSError as e:
            log.debug("persist fs_index: %s", e)

    def _load_persisted(self) -> None:
        if not self.index_path.is_file():
            return
        try:
            data = json.loads(self.index_path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as e:
            log.debug("load fs_index: %s", e)
            return
        projects = {
            k: ProjectEntry(
                name=v["name"],
                path=v["path"],
                depth=int(v.get("depth", 0)),
                has_code=bool(v.get("has_code", False)),
                has_models=bool(v.get("has_models", False)),
                size_files=int(v.get("size_files", 0)),
            ) for k, v in (data.get("projects") or {}).items()
        }
        with self._lock:
            self.index = FilesystemIndex(
                projects=projects,
                indexed_at=float(data.get("indexed_at", 0.0)),
                roots=list(data.get("roots", [])),
            )
        log.info("FS index cargado de disco: %d proyectos", len(projects))
