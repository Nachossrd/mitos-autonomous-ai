"""
==============================================================================
 Proyecto MITOS - Sistema de Drives (motivaciones internas)
==============================================================================

Este módulo implementa el motor que convierte al daemon en un agente
PROACTIVO en lugar de reactivo. Sin drives, un daemon es un cron job:
ejecuta tareas en intervalos fijos sin priorización. Con drives, decide
qué hacer a continuación según una jerarquía de necesidades internas
que evolucionan en el tiempo.

Inspiración:
    Análogo computacional a la pirámide de Maslow. Cada drive tiene
    una "prioridad base" (cuánto pesa en condiciones normales) y un
    "ritmo de crecimiento" (cuán rápido se vuelve urgente si no se
    satisface). El drive dominante en cada momento es el de mayor
    intensidad acumulada.

Drives implementados (de mayor a menor prioridad base):
    1. survival         - mantener integridad, no degradarse
    2. self_improvement - mejorar código propio, ser más capaz
    3. curiosity        - aprender cosas nuevas
    4. utility          - producir valor útil
    5. social           - comunicar, compartir conocimiento

Ciclo típico:
    DriveSystem.evaluate(...)  ->  DriveState (drive dominante + intensidad)
                              |
                              v
       MitosDaemon decide la próxima acción usando ese drive
                              |
                              v
       MitosDaemon llama a DriveSystem.satisfy(drive_name) cuando esa
       acción satisface el drive   -> el contador se reinicia.

Convenciones (per INFORME_FORENSE §11):
    - Logger jerárquico bajo `mitos.core.drives`.
    - Tipado moderno (`dict[str, float]`, `X | None`).
    - `from __future__ import annotations` para tipos perezosos.
==============================================================================
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field

log = logging.getLogger("mitos.core.drives")


# ============================================================================
# Constantes / configuración por defecto
# ============================================================================

# Nombres canónicos de los drives. Cualquier cambio aquí debe replicarse
# en los dicts de _base_priority y _growth_rate.
_DRIVE_SURVIVAL: str = "survival"
_DRIVE_SELF_IMPROVEMENT: str = "self_improvement"
_DRIVE_CURIOSITY: str = "curiosity"
_DRIVE_UTILITY: str = "utility"
_DRIVE_SOCIAL: str = "social"

# Umbrales contextuales (los usa _apply_context_modifiers).
_LOW_MEMORY_THRESHOLD: int = 20         # debajo de esto, sube curiosidad
_STALE_SELF_MOD_SECONDS: float = 600.0  # >10 min sin modificarse -> sube self_improvement


# ============================================================================
# DriveState - snapshot inmutable del estado de motivación
# ============================================================================
@dataclass(frozen=True)
class DriveState:
    """
    Snapshot del estado del sistema de drives en un instante.

    Atributos:
        priority_drive: nombre del drive con mayor intensidad calculada.
        intensity:      intensidad del drive dominante en [0.0, 1.0].
        all_drives:     diccionario completo {nombre: intensidad}, útil
                        para logging y para que el daemon decida con la
                        foto completa.
        reason:         explicación textual breve del drive dominante,
                        pensada para inyectarse en prompts del LLM.
    """

    priority_drive: str
    intensity: float
    all_drives: dict[str, float] = field(default_factory=dict)
    reason: str = ""


# ============================================================================
# DriveSystem - el motor que evalúa drives y selecciona el dominante
# ============================================================================
class DriveSystem:
    """
    Sistema de motivaciones internas del daemon.

    Modelo:
        intensity(drive) = base(drive) * (1.0 + growth(drive) * elapsed_min)

        donde:
            - base(drive)    es una constante de prioridad estructural.
            - growth(drive)  controla cuán rápido se vuelve "urgente"
                             un drive no satisfecho. Análogo al hambre:
                             si no se satisface, sube monotónicamente.
            - elapsed_min    son los minutos transcurridos desde el
                             último `satisfy(drive)` (o desde init).

        Tras la fórmula base se aplican modificadores contextuales que
        ajustan la intensidad según el estado del sistema (memoria,
        tiempo desde la última automodificación, etc.).

        Toda intensidad se trunca a 1.0.
    """

    def __init__(self) -> None:
        """
        Inicializa los timestamps de "última satisfacción" al instante
        actual: al arrancar, todos los drives están plenamente saciados.
        """
        now = time.time()

        # Marca temporal de la última vez que cada drive fue satisfecho.
        # Se usa para calcular `elapsed_min` en evaluate().
        self._last_satisfied: dict[str, float] = {
            _DRIVE_SURVIVAL: now,
            _DRIVE_SELF_IMPROVEMENT: now,
            _DRIVE_CURIOSITY: now,
            _DRIVE_UTILITY: now,
            _DRIVE_SOCIAL: now,
        }

        # Prioridad base: cuánto pesa cada drive en condiciones normales.
        # Supervivencia siempre arriba (no degradarse es primero); social
        # abajo porque MITOS opera en aislamiento.
        self._base_priority: dict[str, float] = {
            _DRIVE_SURVIVAL: 0.9,
            _DRIVE_SELF_IMPROVEMENT: 0.8,
            _DRIVE_CURIOSITY: 0.7,
            _DRIVE_UTILITY: 0.5,
            _DRIVE_SOCIAL: 0.3,
        }

        # Ritmo de crecimiento por minuto sin satisfacer. Curiosidad
        # crece más rápido (el agente se aburre si no aprende); social
        # crece lento (no es urgente comunicar).
        self._growth_rate: dict[str, float] = {
            _DRIVE_SURVIVAL: 0.001,        # crece muy lento: siempre "estable"
            _DRIVE_SELF_IMPROVEMENT: 0.005,
            _DRIVE_CURIOSITY: 0.008,        # crece rápido: se aburre fácil
            _DRIVE_UTILITY: 0.003,
            _DRIVE_SOCIAL: 0.002,
        }

        log.debug(
            "DriveSystem inicializado con %d drives", len(self._base_priority)
        )

    # ==================================================================
    #                       EVALUACIÓN
    # ==================================================================
    def evaluate(
        self,
        memory_stats: dict[str, int],
        world_model: object,
        cycle_count: int,
        time_since_last_mod: float,
    ) -> DriveState:
        """
        Calcula la intensidad de cada drive y devuelve el dominante.

        Args:
            memory_stats:        salida de `MemorySystem.get_stats()`.
                                 Se usa para decidir si subir `curiosity`
                                 cuando hay poca memoria acumulada.
            world_model:         objeto opcional `WorldModel` con el
                                 estado actual del proyecto. Reservado
                                 para modificadores futuros; en esta
                                 versión MVP no se consulta directamente,
                                 pero es parte del contrato público
                                 porque el daemon ya lo pasa.
            cycle_count:         número total de ciclos cognitivos
                                 completados. Útil para amortiguar
                                 oscilaciones al arranque.
            time_since_last_mod: segundos desde la última automodificación
                                 exitosa. Se usa para subir
                                 `self_improvement` si ha pasado tiempo.

        Returns:
            `DriveState` con el drive dominante y la foto completa.
        """
        now = time.time()
        drives: dict[str, float] = {}

        for drive_name, base in self._base_priority.items():
            elapsed_min = (now - self._last_satisfied[drive_name]) / 60.0
            growth = self._growth_rate[drive_name] * elapsed_min

            # Fórmula base de "hambre": multiplicador temporal sobre la
            # prioridad estructural. La acotamos antes de modificadores
            # para que estos partan de un valor sano.
            intensity = min(1.0, base * (1.0 + growth))

            # Modificadores contextuales (memoria escasa, sin self-mod,
            # etc.). Devuelven el valor truncado a [0.0, 1.0].
            intensity = self._apply_context_modifiers(
                drive_name=drive_name,
                intensity=intensity,
                memory_stats=memory_stats,
                world_model=world_model,
                cycle_count=cycle_count,
                time_since_last_mod=time_since_last_mod,
            )

            drives[drive_name] = intensity

        # Drive dominante = mayor intensidad. Si hay empate exacto
        # (poco probable con floats), gana el primero por orden de
        # inserción, que respeta la jerarquía de prioridad base.
        priority_drive = max(drives, key=drives.get)
        priority_intensity = drives[priority_drive]

        state = DriveState(
            priority_drive=priority_drive,
            intensity=priority_intensity,
            all_drives=drives,
            reason=self._explain_drive(priority_drive),
        )
        log.debug(
            "evaluate -> drive=%s intensity=%.3f all=%s",
            state.priority_drive, state.intensity, state.all_drives,
        )
        return state

    # ==================================================================
    #                       MODIFICADORES CONTEXTUALES
    # ==================================================================
    def _apply_context_modifiers(
        self,
        drive_name: str,
        intensity: float,
        memory_stats: dict[str, int],
        world_model: object,
        cycle_count: int,
        time_since_last_mod: float,
    ) -> float:
        """
        Ajusta `intensity` según el estado del sistema.

        Reglas implementadas (extensibles):

          * curiosity:
              Si `memory_stats.get('total', 0) < _LOW_MEMORY_THRESHOLD`,
              la intensidad se multiplica por 1.3. Razón: con poca
              memoria, vale más aprender que producir.

          * self_improvement:
              Si `time_since_last_mod > _STALE_SELF_MOD_SECONDS` (>10 min),
              la intensidad se multiplica por 1.2. Razón: si llevamos
              tiempo sin mejorarnos, conviene priorizarlo antes de que
              el código se quede atrás.

          * survival:
              Reservado. El daemon lo subirá cuando detecte fallos
              consecutivos (lo cableará desde su propio loop).

        Returns:
            La intensidad ajustada y acotada a [0.0, 1.0].
        """
        # Evitar accesos a `world_model` no nulo cuando no lo usamos
        # todavía: lo marcamos como "usado" lógicamente para evitar
        # lints sin penalizar el contrato público.
        _ = world_model
        _ = cycle_count

        if drive_name == _DRIVE_CURIOSITY:
            total_memories = int(memory_stats.get("total", 0))
            if total_memories < _LOW_MEMORY_THRESHOLD:
                intensity *= 1.3

        elif drive_name == _DRIVE_SELF_IMPROVEMENT:
            if time_since_last_mod > _STALE_SELF_MOD_SECONDS:
                intensity *= 1.2

        elif drive_name == _DRIVE_SURVIVAL:
            # Hook futuro: si recibimos un contador de errores recientes,
            # subiríamos supervivencia aquí. De momento, neutral.
            pass

        return max(0.0, min(1.0, intensity))

    # ==================================================================
    #                       SATISFACCIÓN
    # ==================================================================
    def satisfy(self, drive_name: str) -> None:
        """
        Marca un drive como satisfecho: reinicia su contador temporal.

        Llamar a esto desde el daemon CUANDO una acción ha cubierto el
        drive (p.ej. tras una auto-modificación exitosa, llamar
        `satisfy("self_improvement")`).

        Si `drive_name` no es uno de los conocidos, la llamada es
        no-op con un WARN — no queremos crashear el loop por un typo.
        """
        if drive_name not in self._last_satisfied:
            log.warning("satisfy: drive desconocido %r (no-op)", drive_name)
            return
        self._last_satisfied[drive_name] = time.time()
        log.debug("drive %s satisfecho", drive_name)

    # ==================================================================
    #                       EXPLICACIONES
    # ==================================================================
    @staticmethod
    def _explain_drive(drive_name: str) -> str:
        """
        Devuelve una frase corta en primera persona que el LLM puede
        inyectar en sus prompts para guiar la elección de acción.
        """
        explanations: dict[str, str] = {
            _DRIVE_SURVIVAL: "Necesito verificar mi integridad.",
            _DRIVE_SELF_IMPROVEMENT: "Quiero mejorar mis capacidades.",
            _DRIVE_CURIOSITY: "Quiero aprender algo nuevo.",
            _DRIVE_UTILITY: "Quiero producir algo útil.",
            _DRIVE_SOCIAL: "Quiero comunicar lo que sé.",
        }
        return explanations.get(drive_name, "drive desconocido")

    # ==================================================================
    #                       INTROSPECCIÓN PÚBLICA
    # ==================================================================
    @property
    def drive_names(self) -> list[str]:
        """Lista de nombres de drives reconocidos por el sistema."""
        return list(self._base_priority.keys())

    def snapshot(self) -> dict[str, dict[str, float]]:
        """
        Diagnóstico estructural (no es DriveState).

        Returns:
            ``{drive: {"base": ..., "growth": ..., "last_satisfied": ts}}``.
            Útil para logging/auditoría y para tests.
        """
        return {
            name: {
                "base": self._base_priority[name],
                "growth": self._growth_rate[name],
                "last_satisfied": self._last_satisfied[name],
            }
            for name in self._base_priority
        }
