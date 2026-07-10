"""
==============================================================================
 Proyecto MITOS - CognitiveDashboard (Fase 8 — Observabilidad en tiempo real)
==============================================================================

Reemplazo del `while True: print(...)` que tenía el daemon en consola.
Renderiza un layout 2x2 con `rich.live` que se refresca dos veces por
segundo y NO bloquea la inferencia (los paneles consultan estado en
memoria, no piden trabajo al daemon).

Layout:

    ┌─────────────────────────┬─────────────────────────┐
    │  PERCEPCIÓN SENSORIAL   │  ENRUTAMIENTO (BRAIN)   │
    │  cámara / mic / sistema │  local vs Gemini/Groq   │
    ├─────────────────────────┼─────────────────────────┤
    │  AUTOCONCIENCIA &       │  ESTADO DEL CICLO       │
    │  LIMITACIONES           │  últimas 3 acciones     │
    └─────────────────────────┴─────────────────────────┘

Cada panel tiene una función `_render_*_panel()` que devuelve un
`Panel` listo. `_generate_layout()` los compone. `run_monitor()` hace
loop con `Live` y llama a `self.engine.run_cycle()`.

Convenciones:
  - Logger jerárquico `mitos.dashboard`.
  - `from __future__ import annotations`, tipado moderno.
  - Si `rich` no está instalado, mensaje claro al operador en lugar
    de un ImportError críptico — instalación opcional.
==============================================================================
"""

from __future__ import annotations

import logging
import time
from typing import Any

log = logging.getLogger("mitos.dashboard")


# ============================================================================
# Import perezoso de Rich para que el sistema arranque sin él si falta
# ============================================================================
try:
    from rich.console import Console, Group
    from rich.layout import Layout
    from rich.live import Live
    from rich.panel import Panel
    from rich.table import Table
    from rich.text import Text
    _RICH_AVAILABLE = True
except ImportError:
    _RICH_AVAILABLE = False
    Console = Group = Layout = Live = Panel = Table = Text = None  # type: ignore[misc]


# ============================================================================
# CognitiveDashboard
# ============================================================================
class CognitiveDashboard:
    """Monitor visual 2x2 que vive durante toda la sesión cognitiva."""

    def __init__(self, engine: Any) -> None:
        """
        Args:
            engine: instancia de `CognitiveEngine` con `sensors`, `router`,
                    `limitations`, `state` y `run_cycle()` disponibles.
        """
        if not _RICH_AVAILABLE:
            raise RuntimeError(
                "El dashboard requiere `rich`. Instala con:\n"
                "    pip install rich\n"
                "Sin él, ejecuta el daemon en modo headless con run_daemon.py"
            )
        self.engine = engine
        self.console: Console = Console()
        # Cache del último render por panel — útil si una llamada falla
        # podemos seguir mostrando el último estado bueno.
        self._last_panels: dict[str, Panel] = {}
        log.info("CognitiveDashboard inicializado")

    # ==================================================================
    #                       PANEL 1 — PERCEPCIÓN SENSORIAL
    # ==================================================================
    def _render_perception_panel(self) -> Panel:
        try:
            summary = self.engine.sensors.get_situation_summary()
        except Exception as e:  # noqa: BLE001
            log.debug("sensors.get_situation_summary falló: %s", e)
            summary = f"(sensor_hub no disponible: {e})"

        try:
            caps = dict(self.engine.sensors.get_capabilities())
        except Exception as e:  # noqa: BLE001
            log.debug("sensors.get_capabilities falló: %s", e)
            caps = {}

        table = Table.grid(padding=(0, 1))
        table.add_column(justify="left", style="bold cyan", no_wrap=True)
        table.add_column(justify="left")
        for cap_name, enabled in sorted(caps.items()):
            mark = "[green]●[/]" if enabled else "[dim]○[/]"
            table.add_row(f"{mark} {cap_name}", "")
        if not caps:
            table.add_row("[yellow]Sin sensores activos[/]", "")

        body = Text.from_markup(
            "\n".join([
                summary,
                "",
                "[dim]hardware:[/]",
            ])
        )

        # Componemos texto + tabla en el contenido. Como `Panel`
        # admite un solo renderable, agrupamos con `Group`.
        group = Group(body, table)
        panel = Panel(
            group,
            title="[bold]🎥 PERCEPCIÓN SENSORIAL[/]",
            border_style="cyan",
        )
        self._last_panels["perception"] = panel
        return panel

    # ==================================================================
    #                       PANEL 2 — ENRUTAMIENTO (BRAIN POOL)
    # ==================================================================
    def _render_routing_panel(self) -> Panel:
        try:
            summary = self.engine.router.get_usage_summary()
        except Exception as e:  # noqa: BLE001
            log.debug("router.get_usage_summary falló: %s", e)
            summary = f"(router no disponible: {e})"

        # Top: el resumen textual del router.
        # Bottom: contadores que vienen del CognitiveState.
        try:
            local = int(self.engine.state.total_local_calls)
            external = int(self.engine.state.total_external_calls)
            total = local + external
            local_pct = (100.0 * local / total) if total else 0.0
            external_pct = (100.0 * external / total) if total else 0.0
        except Exception as e:  # noqa: BLE001
            log.debug("state counters: %s", e)
            local, external, local_pct, external_pct = 0, 0, 0.0, 0.0

        table = Table.grid(padding=(0, 1))
        table.add_column(style="bold", no_wrap=True)
        table.add_column(justify="right")
        table.add_row(
            "🏠 Local (Qwen Coder 14B)",
            f"[cyan]{local}[/] ({local_pct:.0f}%)",
        )
        table.add_row(
            "🌍 Externo (API pool)",
            f"[magenta]{external}[/] ({external_pct:.0f}%)",
        )

        body = Text(summary, style="white")
        group = Group(body, Text(""), table)
        panel = Panel(
            group,
            title="[bold]🧠 ENRUTAMIENTO (BRAIN POOL)[/]",
            border_style="magenta",
        )
        self._last_panels["routing"] = panel
        return panel

    # ==================================================================
    #                       PANEL 3 — AUTOCONCIENCIA / LIMITACIONES
    # ==================================================================
    def _render_limitations_panel(self) -> Panel:
        try:
            summary = self.engine.limitations.get_awareness_summary()
        except Exception as e:  # noqa: BLE001
            log.debug("limitations.get_awareness_summary falló: %s", e)
            summary = f"(limitation_engine no disponible: {e})"

        try:
            active = [
                l for l in self.engine.limitations.active_limitations
                if not l.resolved
            ]
        except Exception:  # noqa: BLE001
            active = []

        from rich.console import Group

        # Color del border según severidad máxima.
        max_sev = max((l.severity for l in active), default=0.0)
        if max_sev > 0.7:
            border = "red"
            icon = "🔴"
        elif max_sev > 0.4:
            border = "yellow"
            icon = "🟡"
        else:
            border = "green"
            icon = "🟢"

        # Si hay limitaciones, mostramos también su plan inmediato.
        plan_text = ""
        if active:
            try:
                plan = self.engine.limitations.get_solution_plan()
                if plan and plan.get("action"):
                    plan_text = (
                        f"\n[dim]Plan próximo:[/] [italic]{plan['action']}[/]"
                    )
            except Exception:  # noqa: BLE001
                plan_text = ""

        # Bugs + probes activos del CognitiveState (Fase 8+).
        bugs = int(getattr(self.engine.state, "bugs_pending", 0))
        probes = int(getattr(self.engine.state, "probes_taken", 0))
        if bugs:
            plan_text += (
                f"\n[bold red]🐞 {bugs} bug(s) pendiente(s) "
                "inyectados como weakness[/]"
            )
        if probes:
            plan_text += (
                f"\n[dim]👁  {probes} probe(s) sensoriales por curiosidad[/]"
            )

        body = Text.from_markup(f"{icon} {summary}{plan_text}")
        panel = Panel(
            body,
            title="[bold]🔬 AUTOCONCIENCIA & LIMITACIONES[/]",
            border_style=border,
        )
        self._last_panels["limitations"] = panel
        return panel

    # ==================================================================
    #                       PANEL 4 — ESTADO DEL CICLO
    # ==================================================================
    def _render_cycle_panel(self) -> Panel:
        try:
            cycle = int(self.engine.state.cycle_number)
            mode = str(self.engine.state.mode)
            consec_fail = int(self.engine.state.consecutive_failures)
        except Exception as e:  # noqa: BLE001
            log.debug("state lookup: %s", e)
            cycle, mode, consec_fail = 0, "unknown", 0

        # Color del modo.
        mode_color = {
            "operational": "green",
            "exploration": "cyan",
            "degraded": "yellow",
            "delegating": "magenta",
            "training": "blue",
        }.get(mode, "white")

        # Tabla con las últimas 3 acciones.
        actions_table = Table(
            show_header=True,
            header_style="bold",
            expand=True,
        )
        actions_table.add_column("Ciclo", justify="right", width=6)
        actions_table.add_column("Acción", style="white")
        actions_table.add_column("Target", style="dim")
        actions_table.add_column("Routing", justify="center", width=10)
        actions_table.add_column("Result", justify="center", width=7)

        try:
            recent = list(self.engine.state.action_history)[-3:]
        except Exception:  # noqa: BLE001
            recent = []

        if not recent:
            actions_table.add_row(
                "—", "[dim]sin historia[/]", "—", "—", "—"
            )
        else:
            for rec in recent:
                result_marker = (
                    "[green]✔[/]" if rec.success else "[red]✘[/]"
                )
                routing_color = (
                    "cyan" if rec.routing_provider == "local" else "magenta"
                )
                actions_table.add_row(
                    str(rec.cycle),
                    rec.action[:20],
                    (rec.target or "—")[:30],
                    f"[{routing_color}]{rec.routing_provider}[/]",
                    result_marker,
                )

        bugs = int(getattr(self.engine.state, "bugs_pending", 0))
        probes = int(getattr(self.engine.state, "probes_taken", 0))
        extras = ""
        if bugs:
            extras += f"  [bold red]🐞 bugs:[/] {bugs}"
        if probes:
            extras += f"  [dim]👁 probes:[/] {probes}"
        header = Text.from_markup(
            f"[bold]Ciclo:[/] {cycle}  "
            f"[bold]Modo:[/] [{mode_color}]{mode}[/]  "
            f"[bold]Fallos seguidos:[/] {consec_fail}{extras}"
        )
        group = Group(header, Text(""), actions_table)
        panel = Panel(
            group,
            title="[bold]🔄 ESTADO DEL CICLO[/]",
            border_style="blue",
        )
        self._last_panels["cycle"] = panel
        return panel

    # ==================================================================
    #                       LAYOUT MAESTRO
    # ==================================================================
    def _generate_layout(self) -> Layout:
        """Compone el layout 2x2 con los 4 paneles renderizados."""
        layout = Layout(name="root")
        layout.split_column(
            Layout(name="top", ratio=1),
            Layout(name="bottom", ratio=1),
        )
        layout["top"].split_row(
            Layout(name="perception", ratio=1),
            Layout(name="routing", ratio=1),
        )
        layout["bottom"].split_row(
            Layout(name="limitations", ratio=1),
            Layout(name="cycle", ratio=1),
        )

        # Render seguro: si un panel falla, usamos el último cacheado.
        try:
            layout["perception"].update(self._render_perception_panel())
        except Exception as e:  # noqa: BLE001
            log.warning("render perception falló: %s", e)
            layout["perception"].update(
                self._last_panels.get("perception")
                or Panel("(error renderizando)", border_style="red")
            )

        try:
            layout["routing"].update(self._render_routing_panel())
        except Exception as e:  # noqa: BLE001
            log.warning("render routing falló: %s", e)
            layout["routing"].update(
                self._last_panels.get("routing")
                or Panel("(error renderizando)", border_style="red")
            )

        try:
            layout["limitations"].update(self._render_limitations_panel())
        except Exception as e:  # noqa: BLE001
            log.warning("render limitations falló: %s", e)
            layout["limitations"].update(
                self._last_panels.get("limitations")
                or Panel("(error renderizando)", border_style="red")
            )

        try:
            layout["cycle"].update(self._render_cycle_panel())
        except Exception as e:  # noqa: BLE001
            log.warning("render cycle falló: %s", e)
            layout["cycle"].update(
                self._last_panels.get("cycle")
                or Panel("(error renderizando)", border_style="red")
            )

        return layout

    # ==================================================================
    #                       LOOP PRINCIPAL
    # ==================================================================
    def run_monitor(self, cycle_interval_s: float = 15.0) -> None:
        """Loop infinito: corre ciclos + refresca el dashboard.

        Args:
            cycle_interval_s: cuántos segundos esperar entre ciclos.
                              El refresh visual del Live es independiente
                              (2 Hz por defecto) y NO bloquea el ciclo.
        """
        log.info(
            "Dashboard iniciado — cycle_interval=%.1fs, refresh=2Hz",
            cycle_interval_s,
        )

        with Live(
            self._generate_layout(),
            refresh_per_second=2,
            console=self.console,
            screen=False,  # mantiene scrollback del terminal
        ) as live:
            try:
                while True:
                    cycle_start = time.time()

                    # 1. Ejecuta UN ciclo cognitivo.
                    try:
                        result = self.engine.run_cycle()
                        log.debug(
                            "Ciclo completado: %s",
                            result.get("action") if isinstance(result, dict) else result,
                        )
                    except Exception as e:  # noqa: BLE001
                        log.error("run_cycle crasheó: %s", e)
                        # No abortamos el monitor por un fallo de un ciclo.
                        # Mostraremos la situación en el siguiente refresh
                        # y reintentaremos.

                    # 2. Refresh inmediato tras el ciclo.
                    live.update(self._generate_layout())

                    # 3. Sleep adaptativo. Si el ciclo tardó MÁS que el
                    # intervalo (LLM lento), arrancamos el siguiente ya.
                    # Mientras tanto, refrescamos cada 500ms para que
                    # los contadores no se queden frozen.
                    elapsed = time.time() - cycle_start
                    remaining = max(0.0, cycle_interval_s - elapsed)
                    deadline = time.time() + remaining
                    while time.time() < deadline:
                        time.sleep(0.5)
                        live.update(self._generate_layout())

            except KeyboardInterrupt:
                log.info("Dashboard interrumpido por usuario (Ctrl+C)")
                self.console.print(
                    "\n[bold yellow]Dashboard apagado por el operador.[/]"
                )
            except Exception as e:  # noqa: BLE001
                log.error("Error fatal en run_monitor: %s", e, exc_info=True)
                self.console.print(
                    f"\n[bold red]ERROR FATAL:[/] {type(e).__name__}: {e}"
                )
            finally:
                try:
                    self.engine.shutdown()
                except Exception as e:  # noqa: BLE001
                    log.debug("shutdown durante finally: %s", e)
