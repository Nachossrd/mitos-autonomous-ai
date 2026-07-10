"""
==============================================================================
 Proyecto MITOS - Introspector (auto-conciencia estructural)
==============================================================================

Este módulo permite al daemon LEER Y ENTENDER su propio código fuente. Es
la base de cualquier capacidad de auto-modificación: no se puede mejorar
lo que no se puede mirar.

El Introspector:
  - Escanea recursivamente `src/**/*.py` (saltando `__pycache__`).
  - Cachea el texto y el AST de cada archivo en RAM, para que las
    consultas posteriores sean O(1) por archivo y O(n) por nodos.
  - Resuelve consultas en lenguaje natural a funciones concretas
    (matching por nombre + docstring + heurísticas léxicas).
  - Detecta debilidades estructurales del propio código (sin docstring,
    funciones demasiado largas, `bare except`, falta de type hints).

Restricciones de diseño:
  - Solo `ast` estándar (sin libcst). Esto pierde comentarios al
    re-emitir, pero es suficiente para introspección.
  - El cache es estrictamente en memoria. Si el código fuente cambia
    en disco, hay que volver a llamar `scan()` para refrescarlo.
  - No mutar nada del proyecto: este módulo es READ-ONLY.

Convenciones (per INFORME_FORENSE §11):
  - Logger jerárquico bajo `mitos.self_mod.introspector`.
  - Tipado moderno (`list[str]`, `dict[str, Any]`, `X | None`).
  - `from __future__ import annotations` para tipos perezosos.
==============================================================================
"""

from __future__ import annotations

import ast
import logging
import re
from pathlib import Path
from typing import Any

log = logging.getLogger("mitos.self_mod.introspector")


# ============================================================================
# Constantes de heurística
# ============================================================================

# Umbral por encima del cual una función se considera "muy larga".
# 50 líneas es la frontera que usa Sonarqube por defecto para "code smell".
_WEAKNESS_MAX_FUNCTION_LINES: int = 50

# Score mínimo para considerar una función como "match" de una consulta.
# Por debajo de esto, `find_function_or_file` devuelve None.
_MATCH_THRESHOLD: float = 0.3

# Pesos del scoring léxico en `_match_score`.
_W_NAME_FULL: float = 0.5      # query completa dentro del nombre
_W_NAME_PARTIAL: float = 0.3   # alguna palabra de la query dentro del nombre
_W_DOCSTRING: float = 0.4      # query dentro del docstring
_W_NAME_IN_QUERY: float = 0.8  # el nombre exacto aparece como token en el query

# Regex pre-compiladas para extraer el nombre de función de un goal-string.
#
# Cobertura:
#   "src/brain/agent.py:_emit_banner"             -> "_emit_banner"
#   "Fix: src\\brain\\agent.py:_emit_banner - X"  -> "_emit_banner"
#   "improve the _learn method"                    -> "_learn"
#   "memoize fibonacci"                            -> "fibonacci" (vía fallback)
_RE_PATHLIKE_FUNC: re.Pattern[str] = re.compile(
    r"[:/\\]([_a-zA-Z][_a-zA-Z0-9]+)(?=\s|$|[\-,.])"
)
_RE_DUNDER_OR_PRIVATE: re.Pattern[str] = re.compile(
    r"\b(_{1,2}[a-zA-Z][a-zA-Z0-9_]*)\b"
)


# ============================================================================
# Introspector
# ============================================================================
class Introspector:
    """
    Lector y analizador del código fuente del proyecto.

    Atributos:
        root:        ruta absoluta del directorio raíz del proyecto.
        _file_cache: mapping `path relativo` -> texto fuente.
        _ast_cache:  mapping `path relativo` -> `ast.Module`.
    """

    def __init__(self, project_root: str | Path = ".") -> None:
        """
        Args:
            project_root: directorio raíz del proyecto. Todos los paths se
                          devolverán RELATIVOS a esta carpeta para que sean
                          portables entre máquinas.
        """
        self.root: Path = Path(project_root).resolve()
        self._file_cache: dict[str, str] = {}
        self._ast_cache: dict[str, ast.Module] = {}
        log.debug("Introspector inicializado en %s", self.root)

    # ==================================================================
    #                          ESCANEO
    # ==================================================================
    def scan(self) -> dict[str, int]:
        """
        Escanea `src/**/*.py` y puebla los cachés.

        Cada archivo que no parsea se ignora silenciosamente (con WARN en
        log): el daemon no debe morir porque un archivo del operador esté
        roto en disco.

        Returns:
            Conteo agregado: `{files, functions, classes, lines}`.
        """
        stats: dict[str, int] = {
            "files": 0,
            "functions": 0,
            "classes": 0,
            "lines": 0,
        }

        # Reset de cachés: una nueva llamada a scan() es la única forma
        # de refrescar el estado tras cambios en disco. Esto mantiene la
        # semántica predecible (no hay invalidaciones implícitas).
        self._file_cache.clear()
        self._ast_cache.clear()

        for py_file in self.root.rglob("src/**/*.py"):
            # Saltamos cachés y artefactos de build.
            if "__pycache__" in py_file.parts:
                continue

            try:
                content = py_file.read_text(encoding="utf-8", errors="ignore")
            except OSError as e:
                log.warning("No pude leer %s: %s", py_file, e)
                continue

            # Path RELATIVO al root para que los IDs sean estables y
            # portables entre máquinas (no expone la home del usuario).
            # FORWARD SLASHES siempre: las consultas del LLM y los goals
            # usan slashes; mantenerlos homogéneos evita bugs de matching
            # ("src\\brain\\..." vs "src/brain/...").
            try:
                rel_path = py_file.relative_to(self.root).as_posix()
            except ValueError:
                # py_file no está bajo self.root: muy raro con rglob,
                # pero por seguridad lo saltamos.
                continue

            self._file_cache[rel_path] = content

            try:
                tree = ast.parse(content)
            except SyntaxError as e:
                log.warning("SyntaxError parseando %s: %s", rel_path, e)
                continue
            except (ValueError, RecursionError, MemoryError) as e:
                log.warning("Parser falló en %s: %s", rel_path, e)
                continue

            self._ast_cache[rel_path] = tree

            stats["files"] += 1
            stats["lines"] += len(content.splitlines())

            # Conteo de FunctionDef + AsyncFunctionDef juntos: ambas son
            # "funciones" desde el punto de vista del operador.
            for node in ast.walk(tree):
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    stats["functions"] += 1
                elif isinstance(node, ast.ClassDef):
                    stats["classes"] += 1

        log.info(
            "scan completo: %d archivos / %d funciones / %d clases / %d líneas",
            stats["files"], stats["functions"], stats["classes"], stats["lines"],
        )
        return stats

    # ==================================================================
    #                    BÚSQUEDA DE CÓDIGO RELEVANTE
    # ==================================================================
    def find_function_or_file(self, description: str) -> dict[str, Any] | None:
        """
        Encuentra la función cuyo nombre/docstring más se acerque a `description`.

        Pipeline:
            (1) Extraer un "hint" de nombre de función de la descripción
                (cubre formatos típicos de goals como
                 ``Fix: src/brain/agent.py:_emit_banner - sin docstring``).
                Si hay un nombre EXACTO en el AST, lo devuelve sin
                fuzzy match.
            (2) Si no hay hint o no encuentra match exacto, cae al
                scoring léxico clásico (nombre + docstring + token-match)
                y devuelve el mejor por encima de `_MATCH_THRESHOLD`.

        Args:
            description: descripción libre del objetivo.

        Returns:
            Dict con ``{file, function, code, lineno, score}`` o None si
            no se encontró nada por encima del umbral.
        """
        if not description:
            return None
        if not self._ast_cache:
            log.debug("find_function_or_file llamado sin scan() previo")
            return None

        # Normalizar separadores: el goal puede traer Windows-style
        # backslashes desde un sistema antiguo. Los unificamos a "/".
        normalized = description.replace("\\", "/")
        query = normalized.lower()

        # ---- (1) Resolución por hint exacto -------------------------
        hint = self._extract_function_name_hint(normalized)
        if hint is not None:
            for rel_path, tree in self._ast_cache.items():
                for node in ast.walk(tree):
                    if not isinstance(
                        node, (ast.FunctionDef, ast.AsyncFunctionDef)
                    ):
                        continue
                    if node.name != hint:
                        continue
                    record = self._make_match_record(
                        rel_path=rel_path, node=node, score=1.0
                    )
                    if record is not None:
                        log.debug(
                            "find_function_or_file: hit por hint %r -> %s:%s",
                            hint, rel_path, node.name,
                        )
                        return record
            # Si el hint no machea ninguna función, seguimos al fuzzy
            # (el hint puede ser correcto pero el AST haber cambiado).

        # ---- (2) Fuzzy match clásico --------------------------------
        best_match: dict[str, Any] | None = None
        best_score: float = 0.0

        for rel_path, tree in self._ast_cache.items():
            for node in ast.walk(tree):
                if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    continue

                score = self._match_score(node, query)
                if score <= best_score:
                    continue

                record = self._make_match_record(
                    rel_path=rel_path, node=node, score=score
                )
                if record is None:
                    continue
                best_score = score
                best_match = record

        if best_score < _MATCH_THRESHOLD:
            return None
        return best_match

    # ------------------------------------------------------------------
    def _make_match_record(
        self,
        rel_path: str,
        node: ast.FunctionDef | ast.AsyncFunctionDef,
        score: float,
    ) -> dict[str, Any] | None:
        """Construye el dict de retorno y extrae el fragmento de código."""
        source = self._file_cache.get(rel_path, "")
        if not source:
            return None
        lines = source.splitlines()
        start = max(0, node.lineno - 1)
        end_lineno = getattr(node, "end_lineno", None)
        end = end_lineno if end_lineno is not None else start + 20
        func_code = "\n".join(lines[start:end])
        return {
            "file": rel_path,
            "function": node.name,
            "code": func_code,
            "lineno": node.lineno,
            "score": score,
        }

    # ------------------------------------------------------------------
    @staticmethod
    def _extract_function_name_hint(query: str) -> str | None:
        """
        Intenta sacar un nombre de función del query.

        Heurística en dos pasadas:
          (a) Patrón path-like:  ``path/file.py:_funcname`` -> ``_funcname``.
              Cubre los goals que crea ``find_weaknesses`` y los que el
              LLM redacta basándose en ellos.
          (b) Patrón private/dunder: cualquier identificador que empiece
              por ``_`` (``_learn``, ``__init__``). Pesa más que un
              identificador cualquiera porque el operador casi nunca
              llamará a los privados por accidente.

        Devuelve None si ningún patrón da resultado: el caller cae al
        fuzzy match. Esto NO es un fallo, es un caso normal.
        """
        # (a) path:funcname
        m = _RE_PATHLIKE_FUNC.search(query)
        if m:
            candidate = m.group(1)
            # Evitar capturar el sufijo de archivo ("py") o componentes
            # del path que casualmente acaban en identificador.
            if candidate not in {"py", "src", "tests"}:
                return candidate

        # (b) identificador privado o dunder en cualquier sitio del query.
        m = _RE_DUNDER_OR_PRIVATE.search(query)
        if m:
            return m.group(1)

        return None

    # ==================================================================
    #                  DETECCIÓN DE DEBILIDADES
    # ==================================================================
    def find_weaknesses(self) -> list[str]:
        """
        Analiza el código cacheado y reporta debilidades estructurales.

        Categorías detectadas:
          1. Función sin docstring.
          2. Función con más de `_WEAKNESS_MAX_FUNCTION_LINES` líneas.
          3. Función sin type hint de retorno (excluyendo `__init__`).
          4. Bloque `except:` sin tipo (bare except, mala práctica).

        Returns:
            Lista de strings legibles. Cada string identifica el archivo,
            la función y la categoría de la debilidad.
        """
        weaknesses: list[str] = []

        if not self._ast_cache:
            log.debug("find_weaknesses llamado sin scan() previo")
            return weaknesses

        for rel_path, tree in self._ast_cache.items():
            for node in ast.walk(tree):
                if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    continue

                fq = f"{rel_path}:{node.name}"

                # (1) Falta docstring.
                if not ast.get_docstring(node):
                    weaknesses.append(f"{fq} - sin docstring")

                # (2) Función demasiado larga.
                end_lineno = getattr(node, "end_lineno", None)
                if end_lineno is not None:
                    length = end_lineno - node.lineno
                    if length > _WEAKNESS_MAX_FUNCTION_LINES:
                        weaknesses.append(
                            f"{fq} - demasiado larga ({length} líneas, "
                            f"umbral {_WEAKNESS_MAX_FUNCTION_LINES})"
                        )

                # (3) Sin type hint de retorno (saltando dunders triviales).
                if node.returns is None and node.name != "__init__":
                    weaknesses.append(f"{fq} - sin return type hint")

                # (4) Bare except dentro del cuerpo de la función.
                for child in ast.walk(node):
                    if isinstance(child, ast.ExceptHandler) and child.type is None:
                        weaknesses.append(
                            f"{fq} - bare except (mala práctica)"
                        )
                        # Solo reportar una vez por función aunque haya
                        # varios bare except, para no saturar.
                        break

        log.info("find_weaknesses: %d debilidades detectadas", len(weaknesses))
        return weaknesses

    # ==================================================================
    #                      WRAPPERS DE CONVENIENCIA
    # ==================================================================
    def get_relevant_code(self, topic: str, max_chars: int = 1000) -> str:
        """
        Devuelve el código más relevante a `topic` como un string compacto.

        Útil para inyectar contexto en prompts del LLM sin tener que
        construir el formato manualmente. Si no hay match, devuelve "".

        Args:
            topic:     descripción libre del objetivo.
            max_chars: tope de longitud (truncamiento por la derecha).

        Returns:
            String con encabezado `File: ... Function: ...` seguido del
            código, o "" si no hubo match suficiente.
        """
        match = self.find_function_or_file(topic)
        if match is None:
            return ""
        code = match["code"]
        if len(code) > max_chars:
            code = code[:max_chars]
        return f"File: {match['file']}, Function: {match['function']}\n{code}"

    # ==================================================================
    #                          UTILIDADES
    # ==================================================================
    @staticmethod
    def _match_score(
        node: ast.FunctionDef | ast.AsyncFunctionDef, query: str
    ) -> float:
        """
        Puntúa cuán bien una función casa con `query` (en minúsculas).

        Heurística (suma capada a 1.0):
            +0.8 si el nombre EXACTO de la función aparece como token
                 dentro del query (separado por non-word chars).
                 Esto cubre el caso típico: el goal incluye el nombre
                 textual de la función a tocar.
            +0.5 si el query está contenido en el nombre.
            +0.3 si lo anterior no se cumple pero alguna palabra de la
                 query aparece dentro del nombre.
            +0.4 si el query aparece en el docstring.
        """
        score: float = 0.0
        name_lower = node.name.lower()

        # (a) Match: el nombre completo aparece como token en el query.
        # Usamos re.search con \b para limitar a tokens, no a substrings
        # casuales (ej. "_emit" no debería matchear con "_emit_banner").
        if re.search(rf"(?<![A-Za-z0-9_]){re.escape(name_lower)}(?![A-Za-z0-9_])", query):
            score += _W_NAME_IN_QUERY

        # (b) Match clásico: query dentro de name (full > partial).
        if query in name_lower:
            score += _W_NAME_FULL
        elif any(word and word in name_lower for word in query.split()):
            score += _W_NAME_PARTIAL

        # (c) Match por docstring (acumulativo).
        docstring = ast.get_docstring(node)
        if docstring and query in docstring.lower():
            score += _W_DOCSTRING

        return min(1.0, score)

    # ==================================================================
    #                       INTROSPECCIÓN PÚBLICA
    # ==================================================================
    @property
    def cached_files(self) -> list[str]:
        """Rutas relativas de los archivos actualmente en caché."""
        return list(self._file_cache.keys())

    @property
    def is_scanned(self) -> bool:
        """True si ya hay datos cacheados (al menos un scan exitoso)."""
        return bool(self._ast_cache)
