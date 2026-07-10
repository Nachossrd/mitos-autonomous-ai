"""
==============================================================================
 Proyecto MITOS - InnerThought (Fase 7 — Monólogo interno)
==============================================================================

El daemon de Fases 3-6 actúa pero no PIENSA sobre sí mismo. Tras cada
ciclo "razona" delegando al LLM ("debería elegir self_modify o reflect?"),
pero nunca evalúa su estado emocional operativo:

  - ¿Estoy frustrado por fallos recientes?
  - ¿Tengo curiosidad acerca de capacidades nuevas?
  - ¿Qué preguntas me persiguen sin respuesta?
  - ¿Debería cambiar a modo exploración porque mi enfoque actual no funciona?

Este módulo añade ese monólogo interno. Cada ciclo:

  1. `think(cycle_result, environment, delta) -> str`:
     - Si hay `cycle_result` previo → `_process_result()` lo digiere
       emocionalmente (confidence ±, frustraciones).
     - Cada 3 ciclos → `_deep_think()` consulta al LLM con un prompt
       compacto pidiendo "preocupaciones / oportunidades / frenos".
     - SIEMPRE devuelve el `_get_compressed_thought()` que el daemon
       inyecta en el prompt de `_plan_action` como bloque
       "PENSAMIENTO INTERNO".

  2. `should_be_proactive() -> dict | None`:
     Si la confidence cayó debajo de 0.3 (exploration_mode=True) y el
     entorno indica capacidades fácilmente adquiribles (instalar
     opencv, pytest, ampliar contexto), devuelve un plan
     `{"action": "acquire_capability", "target": ..., "reason": ...}`
     que el daemon ejecuta SALTÁNDOSE el flujo normal de drives.

Diseño:

  - `MentalState` es MUTABLE (cambia cada ciclo).
  - `Thought` es inmutable (registro histórico de un pensamiento).
  - El LLM se invoca solo cada 3 ciclos para no saturar la inferencia.

Convenciones (per INFORME_FORENSE §11):
  - Logger jerárquico `mitos.inner_thought`.
  - `from __future__ import annotations`, tipado moderno.
==============================================================================
"""

from __future__ import annotations

import logging
import re
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Deque

log = logging.getLogger("mitos.inner_thought")


# ============================================================================
# Constantes
# ============================================================================

# Cada cuántos ciclos cognitivos invocar `_deep_think` (consulta al LLM).
# Tres es suficiente para no saturar la inferencia, pero suficientemente
# frecuente para que el monólogo refleje el estado real.
_DEEP_THINK_EVERY: int = 3

# Ajustes de confidence por outcome del ciclo.
_CONFIDENCE_UP: float = 0.05
_CONFIDENCE_DOWN: float = 0.10

# Umbral bajo el cual MITOS activa exploration_mode.
_EXPLORATION_THRESHOLD: float = 0.3
# Umbral sobre el cual MITOS desactiva exploration_mode automáticamente.
_RECOVERY_THRESHOLD: float = 0.55

# Capacidad máxima de los buffers (deque). Sin esto crecen sin límite
# entre sesiones largas.
_MAX_INSIGHTS: int = 10
_MAX_FRUSTRATIONS: int = 10
_MAX_QUESTIONS: int = 8
_MAX_THOUGHTS_LOG: int = 50

# Tokens y temperatura del prompt de `_deep_think`.
_DEEP_THINK_TOKENS: int = 120
_DEEP_THINK_TEMP: float = 0.6


# ============================================================================
# Dataclasses
# ============================================================================
@dataclass(frozen=True)
class Thought:
    """Registro inmutable de un pensamiento del daemon."""

    content: str
    category: str  # "insight" | "frustration" | "question" | "state"
    timestamp: float
    triggered_by: str  # "deep_think" | "cycle_result" | "anomaly" | "operator"


@dataclass
class MentalState:
    """Estado emocional/cognitivo del daemon. Mutable por diseño."""

    current_focus: str = ""
    open_questions: Deque[str] = field(
        default_factory=lambda: deque(maxlen=_MAX_QUESTIONS)
    )
    recent_insights: Deque[str] = field(
        default_factory=lambda: deque(maxlen=_MAX_INSIGHTS)
    )
    frustrations: Deque[str] = field(
        default_factory=lambda: deque(maxlen=_MAX_FRUSTRATIONS)
    )
    confidence_level: float = 0.5  # [0, 1]
    exploration_mode: bool = False


# ============================================================================
# InnerThought
# ============================================================================
class InnerThought:
    """Monólogo interno: estado mental + razonamiento periódico al LLM."""

    def __init__(
        self,
        llm_engine: Any,
        perception: Any,
        memory_system: Any | None = None,
    ) -> None:
        """
        Args:
            llm_engine:    instancia con `.think(prompt, max_tokens,
                           temperature) -> str`.
            perception:    instancia de `EnvironmentPerception`. Se usa
                           tanto en `_deep_think` como en
                           `should_be_proactive`.
            memory_system: opcional. Si está disponible, los insights
                           se persisten como `reflections` con importance
                           alta para que el recall semántico los traiga
                           en ciclos futuros.
        """
        self.llm = llm_engine
        self.perception = perception
        self.memory = memory_system

        self.mental_state: MentalState = MentalState()
        self._thoughts_log: Deque[Thought] = deque(maxlen=_MAX_THOUGHTS_LOG)
        self._cycle_count: int = 0
        # Cache del último deep_think para que `_get_compressed_thought`
        # tenga contenido reciente entre invocaciones del LLM.
        self._last_deep_thought: str = ""
        log.info("InnerThought listo")

    # ==================================================================
    #                       API PRINCIPAL
    # ==================================================================
    def think(
        self,
        cycle_result: Any | None,
        environment: Any | None,
        delta: Any | None,
    ) -> str:
        """Procesa el ciclo recién terminado y devuelve el pensamiento comprimido.

        Args:
            cycle_result: `CycleResult` del ciclo anterior (puede ser None
                          en el primer ciclo).
            environment:  `EnvironmentSnapshot` actual (perception).
            delta:        `PerceptionDelta` con cambios desde el ciclo
                          previo.

        Returns:
            String ultra-comprimido (1-3 líneas) listo para inyectar en
            el bloque "PENSAMIENTO INTERNO" del prompt del daemon.
        """
        self._cycle_count += 1

        # --- (1) Digerir el ciclo recién pasado ---
        if cycle_result is not None:
            self._process_result(cycle_result)

        # --- (2) Procesar urgencias/anomalías del delta ---
        if delta is not None:
            self._absorb_delta(delta)

        # --- (3) Cada N ciclos, llamar al LLM para razonar más profundo ---
        if (
            environment is not None
            and self._cycle_count % _DEEP_THINK_EVERY == 0
        ):
            try:
                self._deep_think(environment, delta)
            except Exception as e:  # noqa: BLE001
                # Nunca dejamos que el monólogo tumbe el ciclo.
                log.warning("_deep_think falló (degradado): %s", e)

        # --- (4) Reglas automáticas de cambio de modo ---
        self._update_exploration_mode()

        return self._get_compressed_thought()

    # ==================================================================
    #                       (1) PROCESAR RESULTADO
    # ==================================================================
    def _process_result(self, cycle_result: Any) -> None:
        """Ajusta confidence y frustraciones según el outcome del ciclo."""
        success = bool(getattr(cycle_result, "success", False))
        action = str(getattr(cycle_result, "action_taken", "")).strip()
        outcome = str(getattr(cycle_result, "outcome", "")).strip()
        self_modified = bool(getattr(cycle_result, "self_modified", False))

        if success:
            self.mental_state.confidence_level = min(
                1.0, self.mental_state.confidence_level + _CONFIDENCE_UP
            )
            # Las mutaciones reales generan más confianza que el reflect.
            if self_modified:
                self.mental_state.confidence_level = min(
                    1.0, self.mental_state.confidence_level + _CONFIDENCE_UP
                )
                insight = (
                    f"Conseguí mutar exitosamente vía '{action}': "
                    f"{outcome[:80]}"
                )
                self.mental_state.recent_insights.append(insight)
                self._log_thought(insight, "insight", "cycle_result")
        else:
            self.mental_state.confidence_level = max(
                0.0, self.mental_state.confidence_level - _CONFIDENCE_DOWN
            )
            frustration = (
                f"Falló '{action}': {outcome[:120]}"
                if outcome
                else f"Falló '{action}' sin razón clara"
            )
            self.mental_state.frustrations.append(frustration)
            self._log_thought(frustration, "frustration", "cycle_result")

        # current_focus = la acción del ciclo, para tener foco vigente.
        if action:
            self.mental_state.current_focus = action

    def _absorb_delta(self, delta: Any) -> None:
        """Mueve urgent/anomalies del entorno al estado mental."""
        urgent = list(getattr(delta, "urgent", []) or [])
        anomalies = list(getattr(delta, "anomalies", []) or [])
        new_events = list(getattr(delta, "new_events", []) or [])

        # Urgent → frustración (algo del entorno está mal).
        for u in urgent[:3]:
            self.mental_state.frustrations.append(f"[entorno] {u}")
            self._log_thought(u, "frustration", "anomaly")

        # Anomalies sin solapamiento con urgent → preguntas abiertas.
        for a in anomalies[:3]:
            question = f"¿Cómo manejar: {a[:100]}?"
            if question not in self.mental_state.open_questions:
                self.mental_state.open_questions.append(question)
                self._log_thought(question, "question", "anomaly")

        # New events → posibles insights.
        for ev in new_events[:3]:
            if any(kw in ev.lower() for kw in (
                "tool", "mcp", "pip", "recuperado", "construida",
            )):
                self.mental_state.recent_insights.append(f"[evento] {ev}")
                self._log_thought(ev, "insight", "cycle_result")

    # ==================================================================
    #                       (2) DEEP THINK (LLM)
    # ==================================================================
    def _deep_think(self, environment: Any, delta: Any | None) -> None:
        """Consulta al LLM para razonar sobre el estado interno."""
        state_summary = self._build_state_summary(environment, delta)

        prompt = (
            "Eres MITOS y estás pensando en voz interna (NO le hablas al "
            "operador).\n"
            "Tu estado actual es:\n"
            f"{state_summary}\n\n"
            "Escribe un párrafo CORTO (máximo 4 líneas) reflexionando sobre:\n"
            "  - Una preocupación concreta y por qué te frena\n"
            "  - Una oportunidad que detectas y cómo aprovecharla\n"
            "  - Una pregunta abierta que te persiga (formato '¿...?')\n"
            "Sé directo y específico. Sin saludos. Sin disclaimers.\n"
            "Pensamiento interno:\n"
        )

        try:
            response = self.llm.think(
                prompt,
                max_tokens=_DEEP_THINK_TOKENS,
                temperature=_DEEP_THINK_TEMP,
            )
        except Exception as e:  # noqa: BLE001
            log.debug("llm.think en _deep_think falló: %s", e)
            return

        response = (response or "").strip()
        if not response:
            return

        self._last_deep_thought = response
        self._log_thought(response, "state", "deep_think")

        # Extracción de preguntas abiertas (formato '¿...?').
        questions = re.findall(r"¿[^?\n]{4,200}\?", response)
        for q in questions[:3]:
            q_clean = q.strip()
            if q_clean and q_clean not in self.mental_state.open_questions:
                self.mental_state.open_questions.append(q_clean)
                self._log_thought(q_clean, "question", "deep_think")

        # Persistir como reflection en memoria para recall posterior.
        if self.memory is not None:
            try:
                self.memory.store_reflection(
                    content=f"[InnerThought] {response[:400]}",
                    importance=0.7,
                    source="inner_thought",
                )
            except Exception as e:  # noqa: BLE001
                log.debug("memory.store_reflection falló: %s", e)

        # Detector simple de "estoy bloqueado" para activar exploración.
        lowered = response.lower()
        if any(kw in lowered for kw in (
            "bloquead", "estancad", "no puedo", "no avanzo", "no se",
            "no sé", "incapaz",
        )):
            self.mental_state.exploration_mode = True
            log.info(
                "InnerThought: exploration_mode activado por señales en "
                "deep_think"
            )

    def _build_state_summary(self, environment: Any, delta: Any | None) -> str:
        """Resumen compacto del estado para inyectar en el prompt del LLM."""
        ms = self.mental_state
        parts: list[str] = []

        parts.append(
            f"- confianza={ms.confidence_level:.2f}, "
            f"modo={'exploración' if ms.exploration_mode else 'normal'}, "
            f"foco='{ms.current_focus[:40]}'"
        )

        if ms.frustrations:
            last_frustration = list(ms.frustrations)[-1]
            parts.append(f"- última frustración: {last_frustration[:120]}")
        if ms.recent_insights:
            last_insight = list(ms.recent_insights)[-1]
            parts.append(f"- último insight: {last_insight[:120]}")
        if ms.open_questions:
            last_q = list(ms.open_questions)[-1]
            parts.append(f"- pregunta abierta: {last_q[:120]}")

        # Limitaciones del entorno (perception las trae ya digeridas).
        missing = list(getattr(environment, "missing_capabilities", []) or [])
        if missing:
            parts.append(f"- limitaciones: {'; '.join(missing[:3])}")

        if delta is not None:
            urgent = list(getattr(delta, "urgent", []) or [])
            if urgent:
                parts.append(f"- alertas: {'; '.join(urgent[:2])}")

        return "\n".join(parts)

    # ==================================================================
    #                       (3) PENSAMIENTO COMPRIMIDO
    # ==================================================================
    def _get_compressed_thought(self) -> str:
        """Renderiza el estado mental en 1-3 líneas para el prompt principal."""
        ms = self.mental_state
        mood = self._classify_mood(ms.confidence_level)
        mode = "exploración" if ms.exploration_mode else "normal"

        lines: list[str] = []
        lines.append(
            f"estado={mood} (conf={ms.confidence_level:.2f}) | modo={mode}"
        )

        # Último insight si lo hay.
        if ms.recent_insights:
            last_insight = list(ms.recent_insights)[-1]
            lines.append(f"insight: {last_insight[:120]}")
        elif self._last_deep_thought:
            # Si no hay insight reciente pero sí deep_thought, usar la
            # primera línea de éste como pista.
            first_line = self._last_deep_thought.splitlines()[0]
            lines.append(f"reflexión: {first_line[:120]}")

        # Pregunta abierta más urgente (la más reciente).
        if ms.open_questions:
            last_q = list(ms.open_questions)[-1]
            lines.append(f"pendiente: {last_q[:120]}")

        return "\n".join(lines)

    @staticmethod
    def _classify_mood(confidence: float) -> str:
        """Etiqueta humana del nivel de confidence."""
        if confidence >= 0.85:
            return "confiado"
        if confidence >= 0.60:
            return "estable"
        if confidence >= 0.35:
            return "incierto"
        if confidence >= 0.20:
            return "frustrado"
        return "agobiado"

    def _update_exploration_mode(self) -> None:
        """Activa/desactiva exploration_mode según confidence."""
        if self.mental_state.confidence_level < _EXPLORATION_THRESHOLD:
            if not self.mental_state.exploration_mode:
                self.mental_state.exploration_mode = True
                log.info(
                    "InnerThought: exploration_mode ON (conf=%.2f < %.2f)",
                    self.mental_state.confidence_level,
                    _EXPLORATION_THRESHOLD,
                )
        elif self.mental_state.confidence_level > _RECOVERY_THRESHOLD:
            if self.mental_state.exploration_mode:
                self.mental_state.exploration_mode = False
                log.info(
                    "InnerThought: exploration_mode OFF (conf=%.2f > %.2f)",
                    self.mental_state.confidence_level,
                    _RECOVERY_THRESHOLD,
                )

    # ==================================================================
    #                       (4) PROACTIVIDAD
    # ==================================================================
    def should_be_proactive(self) -> dict[str, str] | None:
        """Decide si el daemon debe BYPASEAR el flujo normal de drives.

        Solo entra en modo proactivo si:
          - `exploration_mode` está activo (frustración acumulada).
          - Detecta una capacidad ausente fácilmente adquirible.

        Returns:
            Dict con `action`, `target`, `reason` listo para que
            `_execute_plan` lo ejecute, o `None` si no procede.
        """
        if not self.mental_state.exploration_mode:
            return None

        # Necesitamos el último snapshot del entorno para decidir.
        snapshot = getattr(self.perception, "_last_snapshot", None)
        if snapshot is None:
            return None

        installed: set[str] = set(
            getattr(snapshot, "installed_packages", []) or []
        )

        # (a) Falta opencv (visión imposible).
        if not installed.intersection({"opencv-python", "opencv-contrib-python"}):
            return {
                "action": "acquire_capability",
                "target": "instalar opencv-python para detectar la cámara",
                "reason": (
                    "Sin opencv-python no puedo procesar imágenes ni "
                    "abrir la cámara — capacidad latente bloqueada"
                ),
            }

        # (b) Falta pytest (mis behavior_tests pierden eficacia).
        if "pytest" not in installed:
            return {
                "action": "acquire_capability",
                "target": "instalar pytest para fortalecer behavior_tests",
                "reason": (
                    "Sin pytest, las regresiones de Fase 5 se descubren "
                    "tarde — instalar fortalece el pipeline"
                ),
            }

        # (c) Contexto reducido — construir compresor.
        ctx_tokens = int(getattr(snapshot, "context_window_tokens", 4096))
        if ctx_tokens < 8192:
            return {
                "action": "acquire_capability",
                "target": (
                    "contexto reducido — construir context_compressor "
                    "como tool propia"
                ),
                "reason": (
                    f"Mi ventana de contexto es {ctx_tokens} tokens; "
                    "comprimir histórico me permitirá razonar más profundo"
                ),
            }

        # (d) Sin internet → muchos drives están degradados.
        if not getattr(snapshot, "has_internet", True):
            return None  # No podemos pip install sin internet — esperar.

        # En exploration_mode pero sin capacidades obvias que adquirir:
        # devolvemos None y dejamos que el flujo normal elija reflect.
        return None

    # ==================================================================
    #                       LOGGING DEL HISTORIAL
    # ==================================================================
    def _log_thought(
        self, content: str, category: str, triggered_by: str
    ) -> None:
        """Registra un Thought en el histórico circular."""
        self._thoughts_log.append(
            Thought(
                content=content,
                category=category,
                timestamp=time.time(),
                triggered_by=triggered_by,
            )
        )
