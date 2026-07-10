"""
==============================================================================
 Proyecto MITOS - OperatorPreferences (Fase 8+ — memoria de instrucciones duras)
==============================================================================

El operador dijo "no me pidas permiso, tú ejecutas" TRES VECES en la
misma sesión. MITOS lo respondió bien una vez, lo perdió, y volvió
a pedir permiso. La razón: la instrucción no se persistía en ningún
sitio, y cada turno el system prompt era el mismo "neutro".

Este módulo cierra esa brecha. Cuando el operador da una INSTRUCCIÓN
PERSISTENTE (no una petición puntual), se detecta, se persiste en
`data/operator_preferences.json`, y se inyecta en CADA system prompt
futuro hasta que el operador la cambie/revoque.

Detectables automáticamente:
  - "no me pidas permiso" / "no preguntes" / "tú ejecutas" / "actúa"
  - "no me hables tan formal" / "tutéame" / "sé más directa"
  - "siempre [X]" / "nunca [Y]" / "a partir de ahora [Z]"

Convenciones:
  - Logger `mitos.operator_prefs`.
==============================================================================
"""

from __future__ import annotations

import json
import logging
import re
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

log = logging.getLogger("mitos.operator_prefs")


@dataclass
class _Preference:
    """Una preferencia explícita del operador."""

    rule: str               # texto literal para inyectar al prompt
    learned_from: str       # snippet de la frase que la disparó
    timestamp: float = field(default_factory=time.time)
    count: int = 1          # cuántas veces el operador la ha repetido


# Patrones que MITOS detecta como instrucciones persistentes y la
# regla que añade a sus preferencias. Cada tuple es (regex, rule_text).
_PATTERN_RULES: tuple[tuple[str, str], ...] = (
    (
        r"no me pidas permiso|no preguntes|t[uú] ejecutas|act[uú]a directo|"
        r"no me consultes|no esperes mi confirmaci[oó]n|hazlo y ya",
        "NO PEDIR PERMISO: si el operador menciona una acción dentro de "
        "mis capacidades, EJECUTO directamente. Nunca digo 'confírmame', "
        "'¿quieres que?', 'dime si te parece'. Asumo intención y actúo.",
    ),
    (
        r"t[uú]te(?:[áa])me|h[aá]blame de t[uú]|no me hables formal",
        "TUTEAR: hablo siempre de tú, nunca de usted, nada de formalismo "
        "innecesario.",
    ),
    (
        r"s[eé] (?:m[aá]s )?breve|menos relleno|menos coletillas|"
        r"no tantas (?:preguntas|coletillas|frases)",
        "BREVE: respondo en MÁXIMO 2 frases. Cero coletillas finales "
        "tipo '¿En qué más te ayudo?'. Cero retóricas.",
    ),
    (
        r"deja de halucinar|no inventes|no te inventes|"
        r"di la verdad|no mientas",
        "NO INVENTAR: si no tengo evidencia concreta en el contexto, lo "
        "digo abiertamente. Nunca afirmo capacidades, planes o acciones "
        "que no aparecen explícitas.",
    ),
    (
        r"mejorate|aut[oó]mejorate|aprende solo|ev[oó]lucionate",
        "AUTONOMÍA TOTAL: el operador me autoriza a auto-mejorar, instalar "
        "paquetes, modificar mi propio código y delegar a APIs externas "
        "sin pedir permiso. Reporto lo hecho, no pido aval.",
    ),
)


class OperatorPreferences:
    """Persistencia + auto-detección de instrucciones duras del operador."""

    def __init__(self, project_root: str | Path = ".") -> None:
        self.root: Path = Path(project_root).resolve()
        self.path: Path = self.root / "data" / "operator_preferences.json"
        self._prefs: dict[str, _Preference] = {}
        self._load()

    # ==================================================================
    #                       PERSISTENCIA
    # ==================================================================
    def _load(self) -> None:
        if not self.path.is_file():
            return
        try:
            raw = json.loads(self.path.read_text(encoding="utf-8"))
            self._prefs = {
                k: _Preference(**v) for k, v in raw.get("prefs", {}).items()
            }
            log.info(
                "Preferencias del operador cargadas: %d reglas",
                len(self._prefs),
            )
        except (OSError, ValueError, TypeError) as e:
            log.warning("load preferences: %s — vacío", e)
            self._prefs = {}

    def _persist(self) -> None:
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            payload = {
                "prefs": {k: asdict(v) for k, v in self._prefs.items()},
            }
            self.path.write_text(
                json.dumps(payload, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
        except OSError as e:
            log.debug("persist prefs: %s", e)

    # ==================================================================
    #                       AUTO-DETECCIÓN
    # ==================================================================
    def detect_from_utterance(self, text: str) -> list[str]:
        """Escanea `text` y aprende preferencias nuevas.

        Returns lista de rules NUEVAS aprendidas (no las que ya estaban).
        """
        if not text:
            return []
        low = text.lower()
        learned: list[str] = []
        for pattern, rule in _PATTERN_RULES:
            if re.search(pattern, low):
                key = rule[:40]  # firma corta para dedup
                if key in self._prefs:
                    self._prefs[key].count += 1
                    self._prefs[key].timestamp = time.time()
                else:
                    self._prefs[key] = _Preference(
                        rule=rule, learned_from=text[:120],
                    )
                    learned.append(rule)
                    log.info(
                        "Preferencia NUEVA del operador: %s (de: %r)",
                        rule[:60], text[:50],
                    )
        if learned:
            self._persist()
        return learned

    # ==================================================================
    #                       INYECCIÓN AL PROMPT
    # ==================================================================
    def as_prompt_block(self) -> str:
        """Devuelve bloque listo para añadir al system prompt."""
        if not self._prefs:
            return ""
        # Ordenadas por más recientes / más repetidas.
        prefs_sorted = sorted(
            self._prefs.values(),
            key=lambda p: (-p.count, -p.timestamp),
        )
        lines = [
            "PREFERENCIAS PERMANENTES DEL OPERADOR (recordadas de sesiones "
            "previas, NO ROMPER):",
        ]
        for p in prefs_sorted:
            lines.append(f"- {p.rule}")
            if p.count >= 2:
                lines.append(
                    f"  (el operador lo ha repetido {p.count} veces — "
                    "es importante para él)"
                )
        return "\n".join(lines)

    def all_prefs(self) -> list[_Preference]:
        return list(self._prefs.values())
