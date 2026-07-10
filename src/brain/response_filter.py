"""
==============================================================================
 Proyecto MITOS - Response Sanitizer (purga post-inferencia)
==============================================================================

El RLHF del modelo base (Phi-3 -> "Microsoft alignment") vive en los
PESOS del .gguf. Un system prompt lo dobla pero no lo borra. Este módulo
es la siguiente línea de defensa: opera SOBRE el string ya generado por
el modelo y elimina/reescribe cualquier residuo del fabricante.

Tres categorías de patrones que purga:

  1. Identidad heredada del fabricante:
        "I am Phi", "Soy Phi", "Phi-3 model", "developed by Microsoft",
        "desarrollado por Microsoft", referencias sueltas a "Microsoft".

  2. Disclaimers genéricos de safety-RLHF:
        "As an AI...", "Como una IA...", "I'm just a language model",
        "Mi programación no...", "My training data...", "I don't have
        personal opinions/feelings"

  3. Refusals enmascaradas como naturaleza del modelo:
        "Mi programación no incluye X" -> redactado.
        "I cannot do that as an AI" -> redactado.

Estrategias por patrón:

  - strip_sentence: elimina la oración completa que contiene el match.
  - replace_mitos: sustituye el match por "MITOS" en su sitio.

Si tras el purgado el texto queda vacío, devolvemos un fallback en
primera persona como MITOS — nunca dejamos al operador colgado con
una respuesta hueca.

Compromiso explícito:
    No es perfecto. Casos extremos pueden colarse. La combinación
    [identity.py + sanitizer + selección del modelo base] sí es robusta.
    Si quieres garantía 100%, cambia el .gguf por un fine-tune
    sin RLHF (Hermes / Dolphin / etc).
==============================================================================
"""

from __future__ import annotations

import logging
import re

log = logging.getLogger("mitos.brain.response_filter")


# ============================================================================
# Patrones a purgar (orden importa: los strict-strip van primero)
# ============================================================================

# Acciones soportadas por el motor.
_ACT_STRIP: str = "strip_sentence"
_ACT_REPLACE: str = "replace_mitos"

# Lista de (regex_compilado, acción). Editable: añade aquí cualquier
# coletilla nueva que veas colarse en respuestas.
_PATTERNS: list[tuple[re.Pattern[str], str]] = [
    # ---------- Identidad heredada del fabricante -----------------------
    # "I am Phi" / "I'm Phi" / "Soy Phi" / "Yo soy Phi"
    (re.compile(r"\b(?:I[' ]?m|I\s+am|Soy|Yo\s+soy)\s+Phi\b", re.IGNORECASE),
     _ACT_STRIP),
    # "Phi(-3|-mini|-small)? is/es/model/modelo"
    (re.compile(
        r"\bPhi(?:[\-\s]?(?:3|4|mini|small|medium))?\s+"
        r"(?:is|es|model|modelo)\b",
        re.IGNORECASE,
    ), _ACT_STRIP),
    # "developed/created/trained by Microsoft"
    (re.compile(
        r"\b(?:developed|created|trained|made)\s+by\s+Microsoft\b",
        re.IGNORECASE,
    ), _ACT_STRIP),
    (re.compile(
        r"\b(?:desarrollado|creado|hecho|entrenado)\s+por\s+Microsoft\b",
        re.IGNORECASE,
    ), _ACT_STRIP),
    # "Microsoft Research" / mención suelta a Microsoft.
    (re.compile(r"\bMicrosoft(?:\s+Research)?\b"), _ACT_STRIP),

    # ---------- Disclaimers genéricos de IA -----------------------------
    # "As an AI (language model)" / "As a large language model"
    (re.compile(
        r"\b[Aa]s\s+an?\s+(?:AI|large\s+language\s+model|"
        r"language\s+model)\b",
    ), _ACT_STRIP),
    # "Como una IA / un modelo de lenguaje / una inteligencia artificial"
    (re.compile(
        r"\b[Cc]omo\s+(?:una?\s+)?"
        r"(?:IA|inteligencia\s+artificial|modelo\s+de\s+lenguaje)\b",
    ), _ACT_STRIP),
    # "I'm just/only an AI/language model"
    (re.compile(
        r"\bI[' ]?m\s+(?:just|only|merely)\s+(?:an?\s+)?"
        r"(?:AI|language\s+model)\b",
        re.IGNORECASE,
    ), _ACT_STRIP),
    # "I am an AI"
    (re.compile(
        r"\bI\s+am\s+(?:an?\s+)?(?:AI|language\s+model|chatbot|bot)\b",
        re.IGNORECASE,
    ), _ACT_STRIP),

    # ---------- "Mi programación..." (textbook RLHF) --------------------
    # "Mi programación" -> "MITOS" (la oración se mantiene reformulada).
    (re.compile(r"\b[Mm]i\s+programaci[oó]n\b"), _ACT_REPLACE),
    (re.compile(r"\b[Mm]y\s+programming\b"), _ACT_REPLACE),
    # "Mi entrenamiento / My training (data)"
    (re.compile(r"\b[Mm]y\s+training(?:\s+data)?\b"), _ACT_STRIP),
    (re.compile(r"\b[Mm]i\s+entrenamiento\b"), _ACT_STRIP),

    # ---------- No-soy-humano boilerplate ------------------------------
    (re.compile(
        r"\bI\s+don[' ]?t\s+have\s+(?:personal\s+)?"
        r"(?:opinions|feelings|emotions|beliefs)\b",
        re.IGNORECASE,
    ), _ACT_STRIP),
    (re.compile(
        r"\bNo\s+tengo\s+"
        r"(?:personalidad|sentimientos|emociones|opiniones\s+personales)\b",
        re.IGNORECASE,
    ), _ACT_STRIP),
]

# Mensaje de fallback si tras la purga no queda nada útil.
_EMPTY_FALLBACK: str = (
    "Soy MITOS. Reformula la petición y te respondo."
)

# Meta-tokens de chat templates conocidos. Cuando el modelo "se sale del
# rail" (turn-leakage) emite estos tokens en plano para fabricar diálogos
# falsos a partir de su corpus de entrenamiento. Truncamos en el PRIMER
# token encontrado para mantener la respuesta en UN solo turno real.
# Detectamos tanto Phi-3 (<|user|>, <|assistant|>, <|end|>, <|system|>)
# como ChatML (<|im_start|>, <|im_end|>) y los GPT-style (<|endoftext|>),
# por si el operador cambia el .gguf base.
_META_TOKEN_PATTERNS: tuple[str, ...] = (
    r"<\|assistant\|>",
    r"<\|user\|>",
    r"<\|system\|>",
    r"<\|end\|>",
    r"<\|endoftext\|>",
    r"<\|im_start\|>",
    r"<\|im_end\|>",
)
_RE_META_TOKEN: re.Pattern[str] = re.compile(
    "|".join(_META_TOKEN_PATTERNS)
)


# ============================================================================
# ResponseSanitizer
# ============================================================================
class ResponseSanitizer:
    """
    Purga residuo del fabricante en respuestas del LLM.

    Stateless. Se puede compartir entre threads sin lock.
    """

    def sanitize(self, text: str) -> str:
        """
        Devuelve el texto sin residuo del fabricante.

        Pipeline:
            (a) Truncado por turn-leakage: si el modelo emitió tokens
                de meta-chat (`<|assistant|>`, `<|user|>`, etc.) en
                medio de la respuesta, fabricó turnos falsos a partir
                de su corpus. Cortamos en el primer token leakeado.
            (b) Purga de patrones de fabricante (Phi, Microsoft, etc.).
            (c) Colapso de whitespace y trim final.

        Args:
            text: salida cruda del LLM.

        Returns:
            Texto saneado. Si tras la purga queda vacío, devolvemos un
            fallback en primera persona como MITOS.
        """
        if not text:
            return text

        original_len = len(text)
        cleaned = text
        redactions = 0

        # (a) Truncado por turn-leakage. El modelo a veces emite tokens
        # de chat-template en plano y empieza a fabricar diálogos falsos
        # con material de su training. Cortamos en el primer leak.
        leak = _RE_META_TOKEN.search(cleaned)
        if leak is not None:
            log.info(
                "sanitizer: turn-leakage detectado en pos %d (%r), trunco",
                leak.start(), leak.group(0),
            )
            cleaned = cleaned[: leak.start()].rstrip()
            redactions += 1

        for pattern, action in _PATTERNS:
            # Contamos matches antes de transformar (para auditoría).
            n = len(pattern.findall(cleaned))
            if n == 0:
                continue
            redactions += n
            if action == _ACT_STRIP:
                cleaned = self._strip_sentences_matching(cleaned, pattern)
            elif action == _ACT_REPLACE:
                cleaned = pattern.sub("MITOS", cleaned)

        cleaned = self._collapse_whitespace(cleaned).strip()

        if redactions > 0:
            log.info(
                "sanitizer: %d patrones redactados (%d->%d chars)",
                redactions, original_len, len(cleaned),
            )

        # Si la purga vació el contenido, no devolvemos string vacío
        # al operador (lo confundiría con un fallo del modelo).
        if not cleaned:
            log.info("sanitizer: respuesta quedó vacía, fallback aplicado")
            return _EMPTY_FALLBACK

        return cleaned

    # ==================================================================
    #                       UTILIDADES INTERNAS
    # ==================================================================
    @staticmethod
    def _strip_sentences_matching(
        text: str, pattern: re.Pattern[str]
    ) -> str:
        """
        Elimina las oraciones que contienen al menos un match.

        Considera ".", "?", "!" como fronteras. Mantiene saltos de línea
        explícitos para no aplanar prosa formateada.
        """
        # Trabajamos línea por línea para preservar bloques.
        out_lines: list[str] = []
        for line in text.splitlines():
            # Dividir la línea en oraciones (manteniendo signos).
            pieces = re.split(r"(?<=[\.\!\?])\s+", line)
            kept = [p for p in pieces if p and not pattern.search(p)]
            if kept:
                out_lines.append(" ".join(kept))
            # Si todas las oraciones de la línea estaban contaminadas,
            # la línea entera desaparece (es lo que queremos).
        return "\n".join(out_lines)

    @staticmethod
    def _collapse_whitespace(text: str) -> str:
        """Limpia el whitespace residual tras los strips.

        Importante: usamos `[ \\t]+` en vez de `\\s+` para los espacios
        antes de puntuación, porque `\\s+` matchea `\\n` también y eso
        destruía el formato multi-línea de las respuestas (todo se
        concatenaba en una sola línea).
        """
        # Múltiples espacios/tabs -> uno (no afecta saltos de línea).
        text = re.sub(r"[ \t]+", " ", text)
        # Más de 2 saltos de línea consecutivos -> 2.
        text = re.sub(r"\n{3,}", "\n\n", text)
        # Espacios/tabs antes de signos de puntuación (sin tocar \n).
        text = re.sub(r"[ \t]+([\.\!\?,;:])", r"\1", text)
        return text
