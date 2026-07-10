"""
==============================================================================
 Proyecto MITOS - ContextManager (Fase 7 — Gestión de presupuesto de tokens)
==============================================================================

Qwen 2.5-7B (ChatML, n_ctx=4096) tiene una ventana de contexto pequeña
para lo que MITOS le mete:

  - identidad + reglas + reflexión               ~ 1500 tokens
  - inner thought (cadena de razonamiento)       ~ 400 tokens
  - estrategia (objetivos + plan de sesión)      ~ 300 tokens
  - anti-patterns                                ~ 200 tokens
  - código original a modificar                  ~ 600 tokens
  - percepción del entorno                       ~ 150 tokens
  ─────────────────────────────────────────────
  TOTAL                                          ~ 3150 tokens
  + espacio para la respuesta                    ~  300-600 tokens
  ─────────────────────────────────────────────
  GRAN TOTAL                                     ~ 3450-3750 tokens

Cuando saturamos los 4096 tokens, Qwen empieza a:

  - Ignorar el inicio del prompt (system + estrategia).
  - Producir respuestas truncadas (queda sin espacio para escribir).
  - O peor, generar basura porque la KV-cache se corrompe.

Este `ContextManager` reparte el presupuesto SEGÚN LA ACCIÓN:

  - `self_modify` → más `task_context` y `anti_patterns`, menos `perception`.
  - `reflect` / `plan` → más `perception` + `strategy`, cero `anti_patterns`.
  - `build_tool` → más `task_context` + `response_space`.

`build_prompt()` aplica los presupuestos truncando cada sección con
`_truncate()` antes de ensamblar el prompt final.

Aproximación tokens ≈ chars / 4 (válida para mezcla ES/EN con Qwen
BPE tokenizer; sobrestima un poco para acentos, lo cual es deseable —
deja margen).

Convenciones (per INFORME_FORENSE §11):
  - Logger jerárquico `mitos.context_manager`.
  - `from __future__ import annotations`, tipado moderno.
  - `dataclass(frozen=True)` para snapshots inmutables.
==============================================================================
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

log = logging.getLogger("mitos.context_manager")


# ============================================================================
# Constantes
# ============================================================================

# Aproximación de tokenización: 1 token ≈ 4 chars para mezcla ES/EN
# con tokenizers BPE modernos (Qwen, Llama, Phi). Sobrestima un poco
# para acentos — preferible a infraestimar (overflow del contexto).
_CHARS_PER_TOKEN: int = 4

# Sufijo que añadimos cuando truncamos. Lo contamos contra el budget
# para no excedernos.
_TRUNC_SUFFIX: str = "\n[...truncado...]"
_TRUNC_SUFFIX_CHARS: int = len(_TRUNC_SUFFIX)


# ============================================================================
# Dataclasses
# ============================================================================
@dataclass(frozen=True)
class ContextBudget:
    """Reparto inmutable de tokens entre las 7 secciones del prompt."""

    system_prompt: int
    perception: int
    inner_thought: int
    strategy: int
    task_context: int
    anti_patterns: int
    response_space: int
    total: int


# ============================================================================
# ContextManager
# ============================================================================
class ContextManager:
    """Reparte tokens entre secciones del prompt según el tipo de acción."""

    # --- Presupuestos mínimos protegidos ---
    # Estos NO se sacrifican ni siquiera bajo presión: por debajo de
    # ellos el modelo dejaría de responder coherentemente.
    _min_response_space: int = 300
    _system_overhead: int = 200

    def __init__(self, max_context_tokens: int = 4096) -> None:
        """
        Args:
            max_context_tokens: ventana de contexto del modelo cargado.
                                Para Qwen2.5-7B-Instruct con `n_ctx=4096`
                                pasar 4096. Para modelos con 8k o 32k,
                                ajustar para aprovechar el espacio.
        """
        if max_context_tokens < 1024:
            raise ValueError(
                f"max_context_tokens demasiado pequeño: {max_context_tokens}"
            )
        self.max_context_tokens: int = int(max_context_tokens)
        log.info(
            "ContextManager listo (max_context=%d tokens, min_response=%d)",
            self.max_context_tokens,
            self._min_response_space,
        )

    # ==================================================================
    #                       API — REPARTO
    # ==================================================================
    def allocate(
        self,
        action_type: str,
        urgency_level: float = 0.0,
    ) -> ContextBudget:
        """Calcula el reparto óptimo de tokens para `action_type`.

        Args:
            action_type:   nombre de la acción del daemon. Reconocidos:
                           ``"self_modify"``, ``"reflect"``, ``"plan"``,
                           ``"build_tool"``. Cualquier otro cae al
                           reparto por defecto ("balanced").
            urgency_level: [0, 1]. Modula el reparto:
                             - urgency 1.0 → +50% a `anti_patterns`,
                               −20% a `perception`.
                             - urgency 0.0 → reparto nominal.

        Returns:
            `ContextBudget` con la suma de campos ≤ `max_context_tokens`.
        """
        urgency = max(0.0, min(1.0, float(urgency_level)))
        action = (action_type or "").strip().lower()

        # --- Repartos nominales por acción ---
        if action == "self_modify":
            sys_p = 400
            perception = 60
            thought = 200
            strategy = 100
            task = 600
            anti = 200
            response = self._min_response_space
        elif action in ("reflect", "plan"):
            sys_p = 400
            perception = 150
            thought = 120
            strategy = 150
            task = 150
            anti = 0
            response = 400
        elif action in ("build_tool", "create_tool", "mcp", "create_mcp"):
            sys_p = 400
            perception = 100
            thought = 150
            strategy = 200
            task = 400
            anti = 100
            response = self._min_response_space + 300
        else:
            # Balanced default (learn_github, explore_web, etc).
            sys_p = 400
            perception = 100
            thought = 150
            strategy = 150
            task = 300
            anti = 100
            response = self._min_response_space

        # --- Modulación por urgencia ---
        if urgency > 0.0:
            # Más espacio para "no volver a equivocarse": +50% a anti.
            anti = int(anti * (1.0 + 0.5 * urgency))
            # Menos espacio dedicado a contemplar el entorno: -20%.
            perception = int(perception * (1.0 - 0.2 * urgency))

        # --- Asegurar que el total cuadra ---
        # Si nos pasamos, reducimos proporcionalmente las secciones
        # "sacrificables" (task, anti, perception, thought, strategy)
        # MANTENIENDO sys_p y response intactos.
        budget = self._fit_to_budget(
            sys_p=sys_p,
            perception=perception,
            thought=thought,
            strategy=strategy,
            task=task,
            anti=anti,
            response=response,
        )

        log.debug(
            "allocate(%r, urgency=%.2f) → sys=%d perc=%d thought=%d "
            "strat=%d task=%d anti=%d resp=%d total=%d",
            action_type, urgency,
            budget.system_prompt, budget.perception, budget.inner_thought,
            budget.strategy, budget.task_context, budget.anti_patterns,
            budget.response_space, budget.total,
        )
        return budget

    # ==================================================================
    #                       API — ENSAMBLAJE
    # ==================================================================
    def build_prompt(
        self,
        budget: ContextBudget,
        perception_text: str,
        thought_text: str,
        strategy_text: str,
        task_text: str,
        anti_pattern_text: str,
        action_instruction: str,
    ) -> str:
        """Ensambla el prompt final truncando cada sección a su budget.

        Orden canónico:
          1. PERCEPCIÓN  (qué veo del entorno AHORA)
          2. PENSAMIENTO INTERNO (cadena de razonamiento)
          3. ESTRATEGIA (objetivos + plan de sesión)
          4. CONTEXTO DE LA TAREA (código, goal, memoria semántica)
          5. ANTI-PATTERNS (errores pasados a evitar)
          6. INSTRUCCIÓN DE ACCIÓN (la pregunta concreta al LLM)

        El system prompt + identidad va FUERA de aquí (lo inyecta
        `LLMEngine._build_prompt` con su chat template y el priming
        turn). Aquí solo construimos el contenido del último turno
        de usuario.

        Si una sección viene vacía, se omite completa (no inyectamos
        cabeceras huérfanas).

        Args:
            budget:             reparto calculado por `allocate()`.
            perception_text:    output de `EnvironmentPerception.get_compressed_context()`.
            thought_text:       cadena interna del daemon (puede venir vacía).
            strategy_text:      output de `StrategicPlanner.get_strategic_context()`.
            task_text:          código a modificar / spec / memoria relevante.
            anti_pattern_text:  bloque del `AntiPatternInjector`.
            action_instruction: pregunta concreta al LLM con el formato
                                de respuesta. NO se trunca.

        Returns:
            Prompt final listo para `llm.think(...)`.
        """
        sections: list[str] = []

        if perception_text:
            chunk = self._truncate(perception_text, budget.perception)
            if chunk:
                sections.append(f"=== PERCEPCIÓN DEL ENTORNO ===\n{chunk}")

        if thought_text:
            chunk = self._truncate(thought_text, budget.inner_thought)
            if chunk:
                sections.append(f"=== PENSAMIENTO INTERNO ===\n{chunk}")

        if strategy_text:
            chunk = self._truncate(strategy_text, budget.strategy)
            if chunk:
                sections.append(f"=== ESTRATEGIA DE LARGO PLAZO ===\n{chunk}")

        if task_text:
            chunk = self._truncate(task_text, budget.task_context)
            if chunk:
                sections.append(f"=== CONTEXTO DE LA TAREA ===\n{chunk}")

        if anti_pattern_text and budget.anti_patterns > 0:
            chunk = self._truncate(anti_pattern_text, budget.anti_patterns)
            if chunk:
                sections.append(f"=== ANTI-PATTERNS (NO REPETIR) ===\n{chunk}")

        if action_instruction:
            # No truncamos la instrucción de acción: si no entra entera,
            # el modelo no sabrá qué se le pide.
            sections.append(action_instruction.strip())

        return "\n\n".join(sections)

    # ==================================================================
    #                       UTILIDADES — TRUNCADO
    # ==================================================================
    def _truncate(self, text: str, token_budget: int) -> str:
        """Trunca `text` a `token_budget` tokens (aproximación 1tok=4chars).

        Estrategia:
          - Si el texto cabe → se devuelve tal cual.
          - Si no:
              1. Corta al char_budget duro.
              2. Busca el último salto de línea dentro del recorte.
              3. Si ese salto está en la mitad inferior del recorte
                 (preserva ≥50% del contenido), corta ahí — así no
                 se parte una línea por la mitad.
              4. Añade el sufijo `\\n[...truncado...]`.
        """
        if not text:
            return ""
        if token_budget <= 0:
            return ""

        # Reservamos espacio para el sufijo de truncado dentro del budget.
        char_budget = max(1, token_budget * _CHARS_PER_TOKEN)
        if len(text) <= char_budget:
            return text

        usable_budget = max(1, char_budget - _TRUNC_SUFFIX_CHARS)
        hard_cut = text[:usable_budget]

        # Intentamos cortar limpiamente en el último salto de línea
        # dentro del segmento útil. Solo aceptamos el corte si retiene
        # al menos la mitad del contenido — si el último \n está muy
        # cerca del inicio, mejor un corte duro que un trozo vacío.
        last_newline = hard_cut.rfind("\n")
        if last_newline > usable_budget // 2:
            return hard_cut[:last_newline].rstrip() + _TRUNC_SUFFIX
        return hard_cut.rstrip() + _TRUNC_SUFFIX

    # ==================================================================
    #                       UTILIDADES — REPARTO
    # ==================================================================
    def _fit_to_budget(
        self,
        sys_p: int,
        perception: int,
        thought: int,
        strategy: int,
        task: int,
        anti: int,
        response: int,
    ) -> ContextBudget:
        """Asegura que la suma de campos ≤ `max_context_tokens`.

        Si los nominales superan el budget, recorta proporcionalmente
        las secciones SACRIFICABLES (task, anti, perception, thought,
        strategy) preservando `sys_p` y `response`.
        """
        # Saneamos negativos (urgencia=1.0 puede tirar perception a 0).
        perception = max(0, perception)
        thought = max(0, thought)
        strategy = max(0, strategy)
        task = max(0, task)
        anti = max(0, anti)
        response = max(self._min_response_space, response)
        sys_p = max(self._system_overhead, sys_p)

        total = sys_p + perception + thought + strategy + task + anti + response
        if total <= self.max_context_tokens:
            return ContextBudget(
                system_prompt=sys_p,
                perception=perception,
                inner_thought=thought,
                strategy=strategy,
                task_context=task,
                anti_patterns=anti,
                response_space=response,
                total=total,
            )

        # Pasamos del budget: recortamos las sacrificables.
        excess = total - self.max_context_tokens
        sacrificable = perception + thought + strategy + task + anti
        if sacrificable <= 0:
            # Caso degenerado: sys_p + response ya supera el budget.
            # Sacrificamos `response` hasta el mínimo y devolvemos.
            response = max(
                self._min_response_space,
                self.max_context_tokens - sys_p - 1,
            )
            total = sys_p + response
            return ContextBudget(
                system_prompt=sys_p,
                perception=0,
                inner_thought=0,
                strategy=0,
                task_context=0,
                anti_patterns=0,
                response_space=response,
                total=total,
            )

        # Factor de escala lineal (cap a [0, 1]).
        factor = max(0.0, (sacrificable - excess) / sacrificable)
        perception = int(perception * factor)
        thought = int(thought * factor)
        strategy = int(strategy * factor)
        task = int(task * factor)
        anti = int(anti * factor)

        total = sys_p + perception + thought + strategy + task + anti + response
        return ContextBudget(
            system_prompt=sys_p,
            perception=perception,
            inner_thought=thought,
            strategy=strategy,
            task_context=task,
            anti_patterns=anti,
            response_space=response,
            total=total,
        )
