"""
==============================================================================
 Proyecto MITOS - CognitiveEngine (Fase 8+ — Facade del Cognitive Mesh)
==============================================================================

Une los 6 subsistemas autónomos en UN solo entry point:

    daemon          → ciclo cognitivo clásico (Fases 2-7)
    sensors         → percepción activa (cámara/mic/sistema/red)
    router          → delegación a APIs externas (Gemini/Groq/OpenRouter)
    (limitations    → BORRADO en auditoría Fase A — inerte en práctica)
    bug_scanner     → captura excepciones runtime → inyecta como weakness
    perception_probe→ curiosidad espontánea → snapshot sensorial

Cada ciclo:
    1. bug_scanner.drain_findings()       → daemon._extra_weaknesses
    2. cada 5 ciclos: discover_and_start() (fallback si no hay pyudev)
    3. daemon._cognitive_cycle()
    4. daemon._post_cycle(result)
    5. limitations.measure_response_quality(...)
    6. si self_modify tocó target con bug pendiente → mark_fixed
    7. limitations.get_solution_plan() → immediate_action
    8. perception_probe.should_trigger() → probe()
    9. actualizar self.state

Convenciones:
  - Logger `mitos.cognitive_engine`.
  - `from __future__ import annotations`, tipado moderno.
==============================================================================
"""

from __future__ import annotations

import logging
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Deque

log = logging.getLogger("mitos.cognitive_engine")


# Cada cuántos ciclos re-escaneamos hardware sin pyudev (fallback polling).
# Con pyudev activo el watcher es reactivo y este intervalo es irrelevante.
_HARDWARE_REDISCOVERY_EVERY: int = 5
_ACTION_HISTORY_SIZE: int = 20


@dataclass
class _ActionRecord:
    cycle: int
    action: str
    target: str
    success: bool
    routing_provider: str
    elapsed_s: float
    timestamp: float = field(default_factory=time.time)


@dataclass
class CognitiveState:
    cycle_number: int = 0
    mode: str = "operational"
    last_action: str = ""
    last_success: bool = False
    consecutive_failures: int = 0
    total_local_calls: int = 0
    total_external_calls: int = 0
    bugs_pending: int = 0
    action_history: Deque[_ActionRecord] = field(
        default_factory=lambda: deque(maxlen=_ACTION_HISTORY_SIZE)
    )


class CognitiveEngine:
    """Facade único que el dashboard consume.

    Composición pura, no hereda. Cada subsistema vive independiente y
    puede ser `None` para tests (perception_probe, bug_scanner).
    """

    def __init__(
        self,
        project_root: str | Path,
        daemon: Any,
        sensor_hub: Any,
        router: Any,
        limitations: Any = None,         # legacy param — ignorado (borrado en Fase A)
        bug_scanner: Any = None,
        perception_probe: Any = None,    # legacy param — ignorado (borrado en Fase A)
        bootstrap_now: bool = False,
    ) -> None:
        self.root: Path = Path(project_root).resolve()
        self.daemon: Any = daemon
        self.sensors: Any = sensor_hub
        self.router: Any = router
        self.bug_scanner: Any = bug_scanner
        self.state: CognitiveState = CognitiveState()
        self._bootstrapped: bool = False
        log.info("CognitiveEngine listo en %s", self.root)
        if bootstrap_now:
            self.bootstrap()

    # ==================================================================
    #                       BOOTSTRAP / SHUTDOWN
    # ==================================================================
    def bootstrap(self) -> None:
        if self._bootstrapped:
            return

        try:
            self.sensors.discover_and_start()
        except Exception as e:  # noqa: BLE001
            log.warning("sensor_hub.discover_and_start: %s", e)

        # Hot-plug reactivo si pyudev está disponible (solo Linux).
        try:
            self.sensors.start_hotplug_watcher(self._on_hotplug)
        except Exception as e:  # noqa: BLE001
            log.debug("start_hotplug_watcher: %s", e)

        if self.bug_scanner is not None:
            try:
                self.bug_scanner.install("mitos")
            except Exception as e:  # noqa: BLE001
                log.warning("bug_scanner.install: %s", e)

        self.daemon._alive = True
        try:
            self.daemon._install_signal_handlers()
        except Exception as e:  # noqa: BLE001
            log.debug("install_signal_handlers: %s", e)
        try:
            self.daemon._bootstrap()
        except Exception as e:  # noqa: BLE001
            log.warning("daemon._bootstrap: %s", e)

        self._bootstrapped = True

    def shutdown(self) -> None:
        if self.bug_scanner is not None:
            try:
                self.bug_scanner.uninstall("mitos")
            except Exception as e:  # noqa: BLE001
                log.debug("bug_scanner.uninstall: %s", e)
        try:
            self.sensors.shutdown()
        except Exception as e:  # noqa: BLE001
            log.debug("sensors.shutdown: %s", e)
        self.daemon._alive = False
        log.info("CognitiveEngine apagado")

    # ==================================================================
    #                       CICLO PRINCIPAL
    # ==================================================================
    def run_cycle(self) -> dict[str, Any]:
        if not self._bootstrapped:
            self.bootstrap()

        cycle_start = time.time()
        self._update_mode()
        self._inject_bug_weaknesses()

        if (
            self.state.cycle_number > 0
            and self.state.cycle_number % _HARDWARE_REDISCOVERY_EVERY == 0
        ):
            self._rediscover_hardware()

        try:
            result = self.daemon._cognitive_cycle()
        except Exception as e:  # noqa: BLE001
            log.warning("daemon._cognitive_cycle crasheó: %s", e)
            self.state.consecutive_failures += 1
            return {
                "cycle": self.state.cycle_number + 1,
                "action": "error",
                "target": "",
                "success": False,
                "output": f"{type(e).__name__}: {e}",
                "routing": "local",
                "elapsed_s": time.time() - cycle_start,
                "mode": self.state.mode,
            }

        try:
            self.daemon._post_cycle(result)
        except Exception as e:  # noqa: BLE001
            log.warning("daemon._post_cycle: %s", e)

        self._mark_bugs_fixed_if_any(result)

        routing_used = self._infer_routing_used(result)
        self._update_state(result, routing_used)

        return {
            "cycle": self.state.cycle_number,
            "action": result.action_taken,
            "target": (result.goal_pursued or "")[:60],
            "success": bool(result.success),
            "output": (result.outcome or "")[:200],
            "routing": routing_used,
            "elapsed_s": time.time() - cycle_start,
            "mode": self.state.mode,
        }

    # ==================================================================
    #                       INTERNOS
    # ==================================================================
    def _on_hotplug(self) -> None:
        """Callback del pyudev watcher cuando hay plug/unplug USB."""
        log.info("Hot-plug → re-descubriendo sensores")
        try:
            self.sensors.discover_and_start()
        except Exception as e:  # noqa: BLE001
            log.warning("rediscover tras hotplug: %s", e)

    def _inject_bug_weaknesses(self) -> None:
        """Drena bugs y los pone en el buffer del daemon."""
        if self.bug_scanner is None:
            self.state.bugs_pending = 0
            return
        try:
            findings = self.bug_scanner.drain_findings()
        except Exception as e:  # noqa: BLE001
            log.debug("bug_scanner.drain_findings: %s", e)
            return
        self.state.bugs_pending = len(findings)
        if not findings:
            return
        self.daemon._extra_weaknesses.extend(f.as_weakness() for f in findings)
        log.info(
            "BugScanner: %d bugs activos inyectados como weaknesses",
            len(findings),
        )

    # _measure_quality removido en Fase A (limitation_engine borrado).
    # En 0 sesiones registró una limitación real → ruido puro.

    def _mark_bugs_fixed_if_any(self, result: Any) -> None:
        """Si self_modify tocó un target con bug pendiente, marca fixed."""
        if self.bug_scanner is None:
            return
        if not getattr(result, "self_modified", False):
            return
        target = getattr(result, "goal_pursued", "") or ""
        for finding in list(self.bug_scanner._findings.values()):
            tid = finding.target_id
            if tid == target or tid in target or target in tid:
                self.bug_scanner.mark_fixed(tid)

    # _maybe_execute_immediate_solution removido en Fase A.
    # Dependía de limitation_engine que jamás detectó nada accionable.

    # _maybe_run_perception_probe removido en Fase A.
    # En 0 sesiones produjo un probe real (la heurística nunca disparó).

    def _update_mode(self) -> None:
        if hasattr(self.router, "_has_internet") and not self.router._has_internet:
            self.state.mode = "degraded"
            return
        it = getattr(self.daemon, "inner_thought", None)
        if it is not None and getattr(it.mental_state, "exploration_mode", False):
            self.state.mode = "exploration"
            return
        self.state.mode = "operational"

    def _rediscover_hardware(self) -> None:
        if not hasattr(self.sensors, "discover_and_start"):
            return
        prev = {}
        try:
            prev = dict(self.sensors.get_capabilities())
        except Exception as e:  # noqa: BLE001
            log.debug("get_capabilities prev: %s", e)
        try:
            self.sensors.discover_and_start()
        except Exception as e:  # noqa: BLE001
            log.debug("rediscover.discover_and_start: %s", e)
            return
        try:
            new = dict(self.sensors.get_capabilities())
        except Exception as e:  # noqa: BLE001
            log.debug("get_capabilities new: %s", e)
            return
        for cap, enabled in new.items():
            if enabled and not prev.get(cap):
                log.info("Hardware nuevo detectado: %s", cap)

    @staticmethod
    def _infer_routing_used(result: Any) -> str:
        decision = getattr(result, "decision", None)
        if decision is None:
            return "local"
        source = str(getattr(decision, "source", "")).lower()
        for provider in ("gemini", "groq", "openrouter", "openai", "anthropic"):
            if provider in source:
                return provider
        return "local"

    def _update_state(self, result: Any, routing_used: str) -> None:
        self.state.cycle_number = int(result.cycle_id)
        self.state.last_action = str(result.action_taken or "")
        self.state.last_success = bool(result.success)
        if result.success:
            self.state.consecutive_failures = 0
        else:
            self.state.consecutive_failures += 1
        if routing_used == "local":
            self.state.total_local_calls += 1
        else:
            self.state.total_external_calls += 1
        self.state.action_history.append(
            _ActionRecord(
                cycle=int(result.cycle_id),
                action=str(result.action_taken or "?"),
                target=str(result.goal_pursued or "")[:60],
                success=bool(result.success),
                routing_provider=routing_used,
                elapsed_s=float(result.duration_s),
            )
        )
