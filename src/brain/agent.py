"""
==============================================================================
 Proyecto MITOS - Agente Autónomo (loop Observe -> Think -> Act -> Learn)
==============================================================================

Este módulo cierra el ciclo de "conciencia" del sistema combinando:

    * LLMEngine          -> motor de razonamiento local (llama.cpp).
    * MemorySystem       -> memoria vectorial persistente (ChromaDB).
    * AutonomousFilter   -> filtro de calidad de código (estático).
    * SafeEvolver        -> auto-mejora de código vía AST.

Cada `step()` recorre los cuatro estados clásicos del loop de agente:

    Observe  ->  recupera contexto relevante (memoria + historial reciente).
    Think    ->  el LLM decide qué herramienta usar (decide_action).
    Act      ->  se ejecuta la herramienta elegida con su input.
    Learn    ->  se registra la experiencia en memoria y, en fracasos o
                 cada N steps, se invoca metacognición (reflect()) cuya
                 lección se persiste como reflexión.

Restricciones de seguridad:
    * Las herramientas que ejecutan código lo hacen en un subproceso con
      timeout y nunca dentro del intérprete del agente.
    * Antes de ejecutar, todo código pasa por ast.parse y por el filtro
      autónomo (defensa en profundidad contra basura del LLM).

Todo el ciclo corre 100% en local. Nada de este loop requiere red.
==============================================================================
"""

from __future__ import annotations

import ast
import logging
import os
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass, field
from typing import Any, Callable

from src.evolution import SafeEvolver
from src.filtering import AutonomousFilter

from .llm_engine import LLMEngine
from .memory import MemorySystem

log = logging.getLogger("mitos.brain.agent")


# ============================================================================
# 1. ESTADO DEL AGENTE
# ============================================================================
@dataclass
class AgentState:
    """
    Snapshot mutable del estado del agente durante una sesión de `run()`.

    Atributos:
        current_goal:  objetivo en curso (string libre del usuario).
        steps_taken:   número de iteraciones del loop ejecutadas.
        total_actions: acciones efectivas (tool calls) realizadas.
        successes:     acciones consideradas exitosas por _learn.
        failures:      acciones consideradas fallidas por _learn.
        is_running:    flag de control para parada cooperativa.
        history:       lista append-only de dicts con metadata por step.
    """

    current_goal: str = ""
    steps_taken: int = 0
    total_actions: int = 0
    successes: int = 0
    failures: int = 0
    is_running: bool = False
    history: list[dict[str, Any]] = field(default_factory=list)


# ============================================================================
# 2. AGENTE
# ============================================================================
class AutonomousAgent:
    """
    Agente cognitivo autónomo.

    Composición:
        llm        : LLMEngine, ya cargado.
        memory     : MemorySystem, ya inicializado.
        filter     : AutonomousFilter (puede ser inyectado o instanciado).
        state      : AgentState, mutable por step().
        tools      : dict {nombre: callable(str) -> str}, registro de
                     herramientas que el LLM puede elegir.

    Diseño:
        El bucle no asume éxito a priori: cada acción decide su signal
        (success/failure) en función del valor de retorno de la tool y
        eso alimenta la metacognición. Los fracasos son tan informativos
        como los éxitos -- y a veces más.
    """

    # Cada N steps consecutivos forzamos una reflexión (incluso si todo
    # va bien). Esto evita la complacencia: el agente debe destilar
    # lecciones periódicamente, no sólo cuando todo se rompe.
    _REFLECT_EVERY: int = 3

    # ------------------------------------------------------------------
    def __init__(
        self,
        llm: LLMEngine | None = None,
        memory: MemorySystem | None = None,
        code_filter: AutonomousFilter | None = None,
        console: Any | None = None,
    ) -> None:
        """
        Initializes the agent with the given parameters.
        """
        self.llm = llm if llm is not None else LLMEngine()
        self.memory = memory if memory is not None else MemorySystem()
        self.filter = code_filter if code_filter is not None else AutonomousFilter(
            threshold=0.5
        )
        self.state = AgentState()
        self._console = console

        # Registro de herramientas disponibles para el LLM.
        # La lista pasada a decide_action debe coincidir con las keys.
        self.tools: dict[str, Callable[[str], str]] = {
            "generate_code": self._tool_generate_code,
            "execute_code": self._tool_execute_code,
            "search_memory": self._tool_search_memory,
            "reason": self._tool_reason,
            "improve_code": self._tool_improve_code,
            "learn_fact": self._tool_learn_fact,
        }

        def _tool_generate_code(self, input_data: str) -> str:
            """Generates code based on input data."""
            # Implement the code generation logic here
            pass

        def _tool_execute_code(self, code: str) -> str:
            """Executes the provided code."""
            # Implement the code execution logic here
            pass

        def _tool_search_memory(self, query: str) -> str:
            """Searches the memory for the provided query."""
            # Implement the memory search logic here
            pass

        def _tool_reason(self, query: str) -> str:
            """Provides reasoning for the provided query."""
            # Implement the reasoning logic here
            pass

        def _tool_improve_code(self, code: str) -> str:
            """Improves the provided code."""
            if self.state.cycles_since_last_reflect < 10:
                self.state.drive -= 1.0
                if self.state.drive <= 0:
                    self.state.drive = 0
                    self.state.actions.append("self_modify")
                    self.state.actions.append("build_tool")
            else:
                self.state.cycles_since_last_reflect += 1
            # Implement the code improvement logic here
            pass

        def _tool_learn_fact(self, fact: str) -> None:
            """Learns a new fact."""
            # Implement the fact learning logic here
            pass

    # ==================================================================
    #                       LOOP COGNITIVO
    # ==================================================================
    def step(self) -> dict[str, Any]:
        """
        Ejecuta una iteración completa del loop cognitivo.

        Returns:
            dict con: {step, thought, tool, input, result, success, ts}.
            Útil para que la UI imprima y para auditoría.
        """
        self.state.steps_taken += 1
        step_idx = self.state.steps_taken

        # 1. Observar.
        observation = self._observe()

        # 2. Pensar.
        decision = self._think(observation)

        # 3. Actuar.
        tool_name, tool_input, result, success = self._act(decision)

        # 4. Aprender.
        record = {
            "step": step_idx,
            "thought": decision.get("thought", ""),
            "tool": tool_name,
            "input": tool_input,
            "result": result,
            "success": success,
            "ts": time.time(),
        }
        self._learn(record)
        self._emit_step(record)
        return record

    # ------------------------------------------------------------------
    def run(self, goal: str, max_steps: int = 10) -> AgentState:
        """
        Ejecuta el loop iterativamente hasta `max_steps` o hasta que el
        LLM decida que ya no hay nada que hacer (tool == "none").

        Args:
            goal:      objetivo de alto nivel para esta sesión.
            max_steps: tope duro de iteraciones (evita loops infinitos).

        Returns:
            El AgentState final (con `history` poblada).
        """
        if not goal or not isinstance(goal, str):
            raise ValueError("goal debe ser un string no vacío")
        if max_steps < 1:
            raise ValueError("max_steps debe ser >= 1")

        self.state.current_goal = goal
        self.state.is_running = True
        self._emit_banner(goal)

        try:
            for _ in range(max_steps):
                if not self.state.is_running:
                    break
                record = self.step()
                # Si el LLM eligió "none" Y no falló, asumimos que
                # considera el objetivo cubierto: cortamos limpio.
                if record["tool"] == "none" and record["success"]:
                    self._emit_text(
                        "[dim]Agente decidió no actuar -> fin de sesión.[/dim]"
                    )
                    break
        finally:
            self.state.is_running = False
        return self.state

    # ==================================================================
    #                       OBSERVE / THINK / ACT / LEARN
    # ==================================================================
    def _observe(self) -> dict[str, Any]:
        """
        Recupera contexto relevante:
          - Top-5 memorias semánticamente relacionadas con el goal.
          - Hasta 5 entradas recientes del propio historial, formateadas
            como bullets cortos para que el LLM pueda referenciarlas.
        """
        goal = self.state.current_goal or ""
        try:
            recalls = self.memory.recall(goal, memory_type=None, n_results=5)
        except Exception as e:  # noqa: BLE001
            log.warning("recall falló: %s", e)
            recalls = []

        recent = self.state.history[-5:]
        history_strs = [
            f"step {h['step']} -> tool={h['tool']} success={h['success']} "
            f"result={_truncate(h['result'], 120)}"
            for h in recent
        ]
        memory_strs = [
            f"[{r['memory_type']}] {_truncate(r['content'], 160)} "
            f"(d={r['distance']:.3f})"
            for r in recalls
        ]
        return {"history": history_strs, "memory": memory_strs}

    # ------------------------------------------------------------------
    def _think(self, observation: dict[str, Any]) -> dict[str, str]:
        """Pide al LLM la próxima acción dado el contexto observado."""
        try:
            decision = self.llm.decide_action(
                goal=self.state.current_goal,
                history=observation["history"],
                memory=observation["memory"],
                tools=list(self.tools.keys()),
            )
        except Exception as e:  # noqa: BLE001 - degradar antes que crashear
            log.warning("decide_action falló: %s", e)
            decision = {
                "thought": f"(error LLM: {e})",
                "tool": "none",
                "input": "",
            }
        return decision

    # ------------------------------------------------------------------
    def _act(
        self, decision: dict[str, str]
    ) -> tuple[str, str, str, bool]:
        """
        Ejecuta la herramienta elegida.

        Returns:
            (tool_name, tool_input, result_str, success_bool).
            Por convención, una tool devuelve un string que empieza con
            "ERROR:" para señalar fallo. Cualquier excepción también es
            tratada como fallo.
        """
        tool_name = decision.get("tool", "none").strip()
        tool_input = decision.get("input", "")

        if tool_name == "none":
            return tool_name, tool_input, "(sin acción)", True

        tool = self.tools.get(tool_name)
        if tool is None:
            return (
                tool_name,
                tool_input,
                f"ERROR: herramienta desconocida '{tool_name}'",
                False,
            )

        self.state.total_actions += 1
        try:
            result = tool(tool_input)
        except Exception as e:  # noqa: BLE001 - el agente nunca debe morir
            result = f"ERROR: excepción en tool: {type(e).__name__}: {e}"

        success = not result.startswith("ERROR")
        return tool_name, tool_input, result, success

    # ------------------------------------------------------------------
    def _learn(self, record: dict[str, Any]) -> None:
        """
        Registra el step en historial + memoria. En fracasos o
        periódicamente, ejecuta `reflect` y persiste la lección.
        """
        self.state.history.append(record)
        if record["success"]:
            self.state.successes += 1
        else:
            self.state.failures += 1

        # Persistir la experiencia bruta.
        try:
            self.memory.store_experience(
                content=(
                    f"goal={self.state.current_goal} "
                    f"tool={record['tool']} input={_truncate(record['input'], 200)} "
                    f"result={_truncate(record['result'], 400)} "
                    f"success={record['success']}"
                ),
                importance=0.6 if record["success"] else 0.8,
                step=record["step"],
                tool=record["tool"],
                success=record["success"],
            )
        except Exception as e:  # noqa: BLE001
            log.warning("store_experience falló: %s", e)

        # Reflexión: en fallos o cada _REFLECT_EVERY steps.
        should_reflect = (
            not record["success"]
            or self.state.steps_taken % self._REFLECT_EVERY == 0
        )
        if not should_reflect:
            return

        # Heurística: repetir patrones exitosos.
        successful_patterns = self.memory.get_patterns_by_success_rate(100)
        if successful_patterns and record["tool"] in successful_patterns:
            # Actualizar el patrón exitoso si el tool actual ya lo tiene.
            self.memory.update_pattern(record["tool"], successful_patterns[record["tool"]] + 1)

        try:
            reflection = self.llm.reflect(
                action=f"{record['tool']}({_truncate(record['input'], 200)})",
                result=_truncate(record["result"], 400),
                goal=self.state.current_goal,
            )
            lesson = reflection.get("lesson", "")
            if lesson:
                self.memory.store_reflection(
                    content=lesson,
                    importance=0.75,
                    assessment=reflection.get("assessment", ""),
                    next_action=reflection.get("next_action", ""),
                    triggered_by_step=record["step"],
                )
                self._emit_text(
                    f"[italic]reflexión:[/italic] [yellow]{lesson}[/yellow]"
                )
        except Exception as e:  # noqa: BLE001
            log.warning("reflect falló: %s", e)

    # ==================================================================
    #                          HERRAMIENTAS
    # ==================================================================
    def _tool_generate_code(self, task: str) -> str:
        """
        Genera código Python para `task`, lo evalúa con el filtro
        autónomo y, si pasa, lo persiste en la memoria de código.

        Returns:
            Texto del código generado, con un anexo indicando si pasó
            el filtro y su puntaje total. "ERROR: ..." si todo falla.
        """
        if not task.strip():
            return "ERROR: tarea vacía"
        code = self.llm.generate_code(task=task)
        if not code:
            return "ERROR: el LLM no devolvió código"

        report = self.filter.evaluate(code)
        verdict = "ACEPTADO" if report.accepted else "RECHAZADO"
        if report.accepted:
            try:
                self.memory.store_code(
                    content=code,
                    importance=0.5 + 0.5 * report.total,
                    task=_truncate(task, 200),
                    score=report.total,
                )
            except Exception as e:  # noqa: BLE001
                log.warning("store_code falló: %s", e)
        reasons = ", ".join(report.reasons) if report.reasons else "ninguna"
        return (
            f"[verdict={verdict} score={report.total:.2f} reasons={reasons}]\n"
            f"{code}"
        )

    # ------------------------------------------------------------------
    def _tool_execute_code(self, code: str) -> str:
        """
        Ejecuta `code` en un proceso Python aislado con timeout de 10s.

        Capas de defensa:
            1. ast.parse: rechaza syntax errors antes de tocar disco.
            2. AutonomousFilter: rechaza basura (compilable+novedad).
            3. subprocess.run(sys.executable, file, timeout=10): el
               código nunca corre dentro del intérprete del agente, así
               que cualquier crash queda confinado.

        Returns:
            "OK\\n<stdout>" si exit-code == 0;
            "ERROR: <razón>" si hubo timeout, exception o exit-code != 0.
        """
        if not code or not isinstance(code, str):
            return "ERROR: código vacío"
        try:
            ast.parse(code)
        except SyntaxError as e:
            return f"ERROR: SyntaxError: {e}"

        report = self.filter.evaluate(code)
        if not report.accepted:
            return (
                f"ERROR: código rechazado por filtro autónomo "
                f"(score={report.total:.2f}, reasons={report.reasons})"
            )

        # Tempfile cross-platform. mkstemp evita problemas en Windows
        # donde NamedTemporaryFile no permite reabrir el handle.
        fd, path = tempfile.mkstemp(suffix=".py", prefix="mitos_exec_")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                f.write(code)

            try:
                completed = subprocess.run(
                    [sys.executable, path],
                    capture_output=True,
                    text=True,
                    timeout=10,
                )
            except subprocess.TimeoutExpired:
                return "ERROR: timeout (>10s)"
            except Exception as e:  # noqa: BLE001
                return f"ERROR: subprocess {type(e).__name__}: {e}"

            if completed.returncode != 0:
                return (
                    f"ERROR: exit={completed.returncode}\n"
                    f"stderr:\n{_truncate(completed.stderr, 800)}"
                )
            return f"OK\n{_truncate(completed.stdout, 1200)}"
        finally:
            try:
                os.unlink(path)
            except OSError:
                pass

    # ------------------------------------------------------------------
    def _tool_search_memory(self, query: str) -> str:
        """Búsqueda semántica en las 4 colecciones; devuelve top-5."""
        if not query.strip():
            return "ERROR: query vacía"
        hits = self.memory.recall(query, memory_type=None, n_results=5)
        if not hits:
            return "(memoria vacía o sin coincidencias)"
        lines = [
            f"[{h['memory_type']} d={h['distance']:.3f} "
            f"imp={h['importance']:.2f}] {_truncate(h['content'], 200)}"
            for h in hits
        ]
        return "\n".join(lines)

    # ------------------------------------------------------------------
    def _tool_reason(self, question: str) -> str:
        """Chain-of-Thought sobre `question`; persiste como knowledge."""
        if not question.strip():
            return "ERROR: pregunta vacía"
        reasoning = self.llm.reason_step_by_step(question=question)
        try:
            self.memory.store_knowledge(
                content=f"Q: {question}\nA: {reasoning}",
                importance=0.6,
                source="reason_step_by_step",
            )
        except Exception as e:  # noqa: BLE001
            log.warning("store_knowledge falló: %s", e)
        return reasoning

    # ------------------------------------------------------------------
    def _tool_improve_code(self, code: str) -> str:
        """
        Mejora `code` pasándolo por SafeEvolver (mutaciones AST seguras).

        Si el código no parsea, devolvemos ERROR. Si SafeEvolver no
        encuentra mejoras, devolvemos el código original sin marca de
        error (es un resultado legítimo: ya estaba óptimo para nuestro
        fitness function).
        """
        try:
            ast.parse(code)
        except SyntaxError as e:
            return f"ERROR: código no parseable: {e}"

        try:
            evolver = SafeEvolver(initial_code=code)
        except ValueError as e:
            return f"ERROR: {e}"

        stats = evolver.evolve(max_attempts=15)
        improved = evolver.current_code
        final_fitness = evolver.versions[-1].fitness
        initial_fitness = evolver.versions[0].fitness

        # Persistimos el resultado mejorado (si lo hubo).
        if stats["improvements"] > 0:
            try:
                self.memory.store_code(
                    content=improved,
                    importance=0.7,
                    source="improve_code",
                    delta_fitness=final_fitness - initial_fitness,
                )
            except Exception as e:  # noqa: BLE001
                log.warning("store_code falló: %s", e)

        return (
            f"[improvements={stats['improvements']} "
            f"rollbacks={stats['rollbacks']} "
            f"fitness {initial_fitness:.3f} -> {final_fitness:.3f}]\n"
            f"{improved}"
        )

    # ------------------------------------------------------------------
    def _tool_learn_fact(self, fact: str) -> str:
        """Persiste un hecho directamente en la colección de knowledge."""
        if not fact.strip():
            return "ERROR: hecho vacío"
        try:
            mem_id = self.memory.store_knowledge(
                content=fact,
                importance=0.7,
                source="learn_fact",
            )
        except Exception as e:  # noqa: BLE001
            return f"ERROR: store_knowledge falló: {e}"
        return f"OK guardado como {mem_id[:8]}"

    # ==================================================================
    #                       AUTO-REPORTE
    # ==================================================================
    def get_self_report(self) -> str:
        """
        Devuelve un texto multi-línea con stats de memoria y desempeño.
        Útil para el comando /status de la CLI.
        """
        stats = self.memory.get_stats()
        total = self.state.total_actions
        success_rate = (
            (self.state.successes / total) if total > 0 else 0.0
        )
        lines = [
            "=== AUTO-REPORTE ===",
            f"goal actual:   {self.state.current_goal or '(ninguno)'}",
            f"steps:         {self.state.steps_taken}",
            f"acciones:      {total}",
            f"éxitos:        {self.state.successes}",
            f"fracasos:      {self.state.failures}",
            f"tasa de éxito: {success_rate * 100:.1f}%",
            "",
            "memoria:",
        ]
        for col, n in stats.items():
            lines.append(f"  - {col:13s}: {n}")
        return "\n".join(lines)

    # ==================================================================
    #                          EMISIÓN UI
    # ==================================================================
    def _emit_banner(self, goal: str) -> None:
        """
        Emit a banner with the given goal.

        :param goal: The goal to be displayed in the banner.
        """
        msg = f"[bold cyan]>>> goal:[/bold cyan] {goal}"
        if self._console is not None:
            self._console.rule(msg)
        else:
            log.info("goal: %s", goal)

    def _emit_step(self, record: dict[str, Any]) -> None:
        """
        Emit a step record to the output.

        Parameters:
        record (dict[str, Any]): A dictionary containing the step information.

        Returns:
        None
        """
        ok = "[green]OK[/green]" if record["success"] else "[red]FAIL[/red]"
        line = (
            f"[bold]step {record['step']}[/bold] {ok} "
            f"[cyan]tool=[/cyan]{record['tool']}\n"
            f"  [dim]thought:[/dim] {_truncate(record['thought'], 200)}\n"
            f"  [dim]input:[/dim]   {_truncate(record['input'], 200)}\n"
            f"  [dim]result:[/dim]  {_truncate(record['result'], 400)}"
        )
        self._emit_text(line)

    def _emit_text(self, text: str) -> None:
        """
        Emits the given text to the console or logs it.

        Parameters:
        text (str): The text to be emitted.

        Returns:
        None
        """
        if self._console is not None:
            self._console.print(text)
        else:
            log.info(text)


# ============================================================================
# UTILIDADES
# ============================================================================
def _truncate(text: Any, limit: int) -> str:
    """Recorta un texto a `limit` chars con elipsis. Tolera no-strings."""
    s = str(text)
    if len(s) <= limit:
        return s
    return s[: limit - 3] + "..."
