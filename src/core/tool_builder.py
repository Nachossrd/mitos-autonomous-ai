"""
==============================================================================
 Proyecto MITOS - ToolBuilder (Fase B — desarrollo real de herramientas)
==============================================================================

El operador pidió varias veces "desarrolla la herramienta para X" y MITOS
respondía "no puedo programar herramientas de forma autónoma". MENTIRA:
tiene IntelligenceRouter cableado a Gemini Coder. La capacidad estaba
inerte porque nadie había construido el puente.

Este módulo cierra esa brecha. Cuando el operador dice:

  "Desarrolla la herramienta de análisis automático para YouTube Music"

MITOS:
  1. Detecta el intent "build tool"
  2. Extrae el propósito ("análisis automático YouTube Music")
  3. Pide a Gemini que genere un módulo Python COMPLETO con:
       - imports
       - función principal documentada
       - manejo de errores
       - ejemplo de uso en if __name__ == "__main__"
  4. Valida que el código compile (ast.parse)
  5. Lo guarda en `src/tools/<slug>.py`
  6. Lo registra en `data/built_tools.json`
  7. Reporta por TTS al operador

NO ejecuta el código autogenerado automáticamente (riesgo) — el operador
decide cuándo invocarlo con `python -m src.tools.<slug>`.

Convenciones:
  - Logger `mitos.tool_builder`.
==============================================================================
"""

from __future__ import annotations

import ast
import json
import logging
import re
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

log = logging.getLogger("mitos.tool_builder")


# Triggers detectados en utterances del operador.
# Las variantes whisper más comunes para "desarrolla" / "desarrollar":
# "desarrolla", "desarriba", "desaroya", "desa robla", "desarroya",
# "desarollo", "desarólla", etc.
# Cubrimos con regex laxa: `desa[a-z]{3,9}` + sustantivo de tool.
_NOUN = r"(?:herramienta|capacidad|script|programa|m[oó]dulo|c[oó]digo)"
_ARTICLE = r"(?:la|una|el|un|los|las)?"
_BUILD_PATTERNS: tuple[str, ...] = (
    rf"desa[a-z]{{3,9}}\s+{_ARTICLE}\s*{_NOUN}",
    rf"constru[yi]e?\s+{_ARTICLE}\s*{_NOUN}",
    rf"programa\s+{_ARTICLE}\s*{_NOUN}",
    rf"crea\s+{_ARTICLE}\s*{_NOUN}",
    rf"hazme?\s+{_ARTICLE}\s*{_NOUN}",
    rf"escribe\s+{_ARTICLE}\s*{_NOUN}",
    rf"genera\s+{_ARTICLE}\s*{_NOUN}",
)


@dataclass
class BuiltTool:
    """Registro de una tool generada por delegación."""

    slug: str
    description: str
    file_path: str
    requested_at: float
    bytes_generated: int
    provider: str               # "gemini" | "groq" | ...
    is_valid_python: bool
    error: str = ""


class ToolBuilder:
    """Genera código Python on-demand vía router + lo guarda."""

    def __init__(
        self,
        project_root: str | Path,
        router: Any = None,
        memory: Any = None,
    ) -> None:
        """
        Args:
            project_root: raíz del proyecto MITOS.
            router:       IntelligenceRouter para delegar a Gemini.
            memory:       MemorySystem opcional para guardar la transcripción.
        """
        self.root: Path = Path(project_root).resolve()
        self.router: Any = router
        self.memory: Any = memory
        self.tools_dir: Path = self.root / "src" / "tools"
        self.registry_path: Path = self.root / "data" / "built_tools.json"
        self.tools_dir.mkdir(parents=True, exist_ok=True)
        # Asegurar __init__.py para que sea importable como módulo.
        init_file = self.tools_dir / "__init__.py"
        if not init_file.is_file():
            init_file.write_text(
                '"""Herramientas auto-generadas por MITOS ToolBuilder."""\n',
                encoding="utf-8",
            )
        self._registry: list[BuiltTool] = self._load_registry()
        log.info(
            "ToolBuilder listo (%d tools auto-generadas previas)",
            len(self._registry),
        )

    # ==================================================================
    #                       DETECCIÓN DE INTENT
    # ==================================================================
    @staticmethod
    def is_build_request(text: str) -> bool:
        """True si la utterance suena a 'construye herramienta para X'."""
        if not text:
            return False
        low = text.lower()
        for pat in _BUILD_PATTERNS:
            if re.search(pat, low):
                return True
        return False

    @staticmethod
    def extract_purpose(text: str) -> str:
        """Saca el 'qué quiero que haga' de la frase del operador.

        'desarrolla la herramienta de análisis automático para YouTube Music'
            → 'análisis automático para YouTube Music'
        """
        low = text.lower().strip()
        # Quita el verbo + sustantivo + artículos (laxo con whisper-errors).
        cleaned = re.sub(
            rf"^(?:desa[a-z]{{3,9}}|constru[yi]e?|construir|programa|"
            rf"programar|crea|crear|haz|hazme|escribe|escribir|genera)\s+"
            rf"{_ARTICLE}\s*"
            rf"{_NOUN}\s+"
            rf"(?:de|para|que|a)\s+",
            "", low,
        ).strip(".,;:!?\"'`")
        return cleaned or low

    # ==================================================================
    #                       CONSTRUCCIÓN
    # ==================================================================
    def build(self, purpose: str) -> BuiltTool | None:
        """Genera código para `purpose` y lo guarda. None si falla."""
        if not purpose:
            return None
        if self.router is None:
            log.warning("build: router no disponible — no puedo delegar")
            return None
        if not getattr(self.router, "_has_internet", False):
            log.warning("build: sin internet — no puedo pedir código a Gemini")
            return None
        if not self.router.external_models:
            log.warning("build: pool de modelos externos vacío")
            return None

        slug = self._slug_for(purpose)
        file_path = self.tools_dir / f"{slug}.py"
        if file_path.is_file():
            # Sufijo numérico para no sobreescribir.
            n = 2
            while (self.tools_dir / f"{slug}_v{n}.py").is_file():
                n += 1
            slug = f"{slug}_v{n}"
            file_path = self.tools_dir / f"{slug}.py"

        from src.core.intelligence_router import RoutingDecision
        best = min(self.router.external_models, key=lambda m: m.priority)
        decision = RoutingDecision(
            provider=best.provider,
            model_name=best.model_name,
            reason=f"tool_builder: generar {slug}",
            estimated_tokens=1500,
        )

        prompt = (
            f"Eres un programador Python senior. Genera un módulo Python "
            f"COMPLETO y AUTOCONTENIDO para esta tarea:\n\n"
            f"  PROPÓSITO: {purpose}\n\n"
            "REQUISITOS:\n"
            "- Sintaxis Python 3.11+ válida.\n"
            "- Tipado moderno (list[str], dict[str, Any], etc.).\n"
            "- Función principal documentada con docstring claro.\n"
            "- Manejo defensivo de errores (try/except concretos).\n"
            "- Si necesita dependencias, las importa al principio y "
            "  añade un comentario `# pip install X` arriba.\n"
            "- Termina con `if __name__ == \"__main__\":` con ejemplo "
            "  ejecutable corto.\n"
            "- Sin código malicioso, sin os.system, sin eval/exec.\n\n"
            "DEVUELVE SOLO EL CÓDIGO PYTHON, dentro de un único bloque "
            "```python ... ```. Nada de explicación previa ni posterior. "
            "Si la tarea es ambigua, elige la interpretación más útil "
            "y comenta tus suposiciones DENTRO del código."
        )

        log.info("Delegando a %s la generación de %s", best.provider.value, slug)
        try:
            answer = self.router.execute_sync(
                task=prompt, routing=decision, system_prompt="",
            )
        except Exception as e:  # noqa: BLE001
            log.warning("router.execute_sync para tool_builder: %s", e)
            return None

        code = self._extract_code_block(answer)
        if not code:
            log.warning(
                "build: %s no devolvió bloque de código reconocible",
                best.provider.value,
            )
            return None

        # Validación: ¿compila?
        is_valid = True
        err = ""
        try:
            ast.parse(code)
        except SyntaxError as e:
            is_valid = False
            err = f"SyntaxError en línea {e.lineno}: {e.msg}"
            log.warning("build: código NO compila: %s", err)
            # Lo guardamos igual con sufijo _BROKEN para inspección.
            file_path = self.tools_dir / f"{slug}_BROKEN.py"

        # Header con metadata.
        header = (
            f'"""Auto-generado por MITOS ToolBuilder.\n\n'
            f"  Operador pidió: {purpose}\n"
            f"  Proveedor: {best.provider.value} ({best.model_name})\n"
            f"  Generado: {time.strftime('%Y-%m-%d %H:%M:%S')}\n"
            f"  Válido Python: {is_valid}\n"
            f'"""\n\n'
        )
        try:
            file_path.write_text(header + code, encoding="utf-8")
        except OSError as e:
            log.warning("build: no pude escribir %s: %s", file_path, e)
            return None

        tool = BuiltTool(
            slug=slug,
            description=purpose,
            file_path=str(file_path),
            requested_at=time.time(),
            bytes_generated=len(code),
            provider=best.provider.value,
            is_valid_python=is_valid,
            error=err,
        )
        self._registry.append(tool)
        self._persist_registry()

        # Guardar en knowledge para que el RAG pueda encontrarla.
        if self.memory is not None:
            try:
                self.memory.store_knowledge(
                    content=(
                        f"[TOOL AUTO-GENERADA por delegación a {best.provider.value}]\n"
                        f"Propósito: {purpose}\n"
                        f"Archivo: {file_path}\n"
                        f"Válido: {is_valid}\n"
                        f"Para usar: python -m src.tools.{slug}"
                    ),
                    importance=0.9,
                    source=f"tool_builder_{best.provider.value}",
                )
            except Exception as e:  # noqa: BLE001
                log.debug("memory.store_knowledge tool: %s", e)

        log.info(
            "Tool generada: %s (%d bytes, válido=%s)",
            file_path.name, len(code), is_valid,
        )
        return tool

    # ==================================================================
    #                       HELPERS
    # ==================================================================
    @staticmethod
    def _slug_for(purpose: str) -> str:
        """Slug ASCII corto para nombre de archivo."""
        import unicodedata
        s = unicodedata.normalize("NFKD", purpose)
        s = "".join(c for c in s if not unicodedata.combining(c))
        s = re.sub(r"[^a-z0-9]+", "_", s.lower()).strip("_")
        # Limitamos longitud y quitamos stopwords genéricas.
        parts = [p for p in s.split("_") if p and p not in (
            "para", "de", "que", "una", "un", "el", "la",
            "automatico", "automatica",
        )]
        slug = "_".join(parts[:5])[:40]
        return slug or "tool"

    @staticmethod
    def _extract_code_block(text: str) -> str:
        """Extrae el primer bloque ```python ... ``` del texto."""
        if not text:
            return ""
        # Patrón estándar con lenguaje.
        m = re.search(
            r"```(?:python|py)?\s*\n(.*?)\n```",
            text, re.DOTALL,
        )
        if m:
            return m.group(1).strip()
        # Sin fence: asumimos que TODO es código si arranca con import/def/class.
        if re.match(r"^\s*(?:import|from|def|class|#|\"\"\")", text.strip()):
            return text.strip()
        return ""

    def _load_registry(self) -> list[BuiltTool]:
        if not self.registry_path.is_file():
            return []
        try:
            raw = json.loads(self.registry_path.read_text(encoding="utf-8"))
            return [BuiltTool(**t) for t in raw.get("tools", [])]
        except (OSError, ValueError, TypeError) as e:
            log.debug("load registry: %s", e)
            return []

    def _persist_registry(self) -> None:
        try:
            self.registry_path.parent.mkdir(parents=True, exist_ok=True)
            self.registry_path.write_text(
                json.dumps(
                    {"tools": [asdict(t) for t in self._registry]},
                    indent=2, ensure_ascii=False,
                ),
                encoding="utf-8",
            )
        except OSError as e:
            log.debug("persist registry: %s", e)

    def list_tools(self) -> list[BuiltTool]:
        return list(self._registry)
