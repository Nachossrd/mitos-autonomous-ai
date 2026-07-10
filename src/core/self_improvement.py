"""
==============================================================================
 Proyecto MITOS - SelfImprovementLoop (Fase 8+ — auto-mejora sin pedírsela)
==============================================================================

El operador pidió: "MITOS debe ser capaz de mejorarse por su cuenta
sin que yo se lo pida, debe darse cuenta cómo mejorarse y delegar
tareas y hacer herramientas para cumplir sus objetivos."

Este módulo cumple eso. Corre como thread daemon dentro del DialogLoop
y CADA N MINUTOS evalúa el estado del sistema y ejecuta UNA acción
de auto-mejora visible:

  1. ¿Tengo bugs detectados sin reparar? → ejecuto self_modify sobre uno
  2. ¿Tengo limitaciones activas? → delego conocimiento a Gemini para
     entender cómo resolverla y persisto como knowledge
  3. ¿El indexer encontró proyectos con modelos que no conozco? → leo
     su README/main + lo persisto en memoria
  4. ¿Hay temas recurrentes en la conversación que no domino? → pido
     a Gemini un resumen experto y lo persisto
  5. ¿Mi código tiene weaknesses estructurales? → self_modify sobre la
     más prioritaria

Después de cada acción, MITOS REPORTA POR VOZ qué hizo. El operador
no necesita preguntar — MITOS muestra su propia actividad.

Convenciones:
  - Logger `mitos.self_improvement`.
==============================================================================
"""

from __future__ import annotations

import logging
import random
import threading
import time
from dataclasses import dataclass
from typing import Any, Callable

log = logging.getLogger("mitos.self_improvement")


_LOOP_INTERVAL_S: float = 300.0   # cada 5 min una acción
_ACTIONS_HISTORY_SIZE: int = 20


@dataclass
class ImprovementAction:
    """Una acción de auto-mejora ejecutada."""

    kind: str               # "fix_bug" | "study_limitation" | "learn_project" | "delegate_knowledge" | "self_modify"
    description: str
    success: bool
    timestamp: float
    voice_report: str       # mensaje corto para TTS
    detail: str = ""        # log largo para inspección


class SelfImprovementLoop:
    """Thread daemon que ejecuta acciones de auto-mejora periódicamente."""

    def __init__(
        self,
        engine: Any = None,           # CognitiveEngine si está disponible
        router: Any = None,           # IntelligenceRouter
        indexer: Any = None,          # FilesystemIndexer
        memory: Any = None,           # MemorySystem
        bug_scanner: Any = None,
        gap_detector: Any = None,     # CapabilityGapDetector
        speak: Callable[[str], None] | None = None,
    ) -> None:
        """
        Args:
            engine, router, indexer, memory, bug_scanner: subsistemas opcionales.
                Cada acción de auto-mejora requiere un subset; las que faltan
                degradan grácilmente (skip de esa acción).
            speak: callback para que el loop emita un mensaje por TTS al
                completar cada acción. Si es None, solo loguea.
        """
        self.engine = engine
        self.router = router
        self.indexer = indexer
        self.memory = memory
        self.bug_scanner = bug_scanner
        self.gap_detector = gap_detector
        self.speak = speak
        # SelfPatcher real (Fase C — autorización del operador).
        self.self_patcher: Any = None
        try:
            from src.core.self_patcher import SelfPatcher
            from pathlib import Path as _Path
            root = (
                _Path(engine.root) if engine is not None and hasattr(engine, "root")
                else _Path(".")
            )
            self.self_patcher = SelfPatcher(
                project_root=root, router=router, memory=memory,
            )
        except Exception as e:  # noqa: BLE001
            log.debug("SelfPatcher no se pudo cargar: %s", e)
        # Topics ya delegados — para no repetir 'innerthought' eternamente.
        # Hardcodeamos los terminos que aparecen siempre en memoria por la
        # propia estructura de MITOS (los InnerThoughts hablan de sí
        # mismos), pero NO son útiles como tema de profundización.
        self._delegated_topics: set[str] = {
            "innerthought", "inner_thought", "mitos", "operador",
            "respuesta", "dialog", "pregunta", "sistema",
        }

        self._thread: threading.Thread | None = None
        self._running: bool = False
        self._history: list[ImprovementAction] = []
        self._action_rotation: int = 0

    # ==================================================================
    #                       LIFECYCLE
    # ==================================================================
    def start(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            return
        self._running = True
        self._thread = threading.Thread(
            target=self._loop, name="self-improvement", daemon=True,
        )
        self._thread.start()
        log.info(
            "SelfImprovementLoop arrancado (cada %.0fs)",
            _LOOP_INTERVAL_S,
        )

    def stop(self) -> None:
        self._running = False

    def history(self) -> list[ImprovementAction]:
        return list(self._history)

    # ==================================================================
    #                       BUCLE PRINCIPAL
    # ==================================================================
    def _loop(self) -> None:
        # Damos un primer respiro antes de actuar — no queremos que MITOS
        # diga su primera acción 3 segundos después de saludar.
        time.sleep(60.0)
        while self._running:
            try:
                action = self._pick_and_execute()
                if action is not None:
                    self._record(action)
            except Exception as e:  # noqa: BLE001
                log.warning("self-improvement crasheó: %s", e)
            # Sleep con sensibilidad a stop.
            slept = 0.0
            while self._running and slept < _LOOP_INTERVAL_S:
                time.sleep(5.0)
                slept += 5.0

    def _pick_and_execute(self) -> ImprovementAction | None:
        """Elige una acción en función del estado actual y la ejecuta."""
        # Prioridad: bugs > limitaciones > proyecto nuevo > conocimiento
        # de tema recurrente > self_modify estructural.

        # 1) Bugs sin reparar tienen prioridad MÁXIMA.
        action = self._try_fix_bug()
        if action is not None:
            return action

        # 2) GAPS de capacidad detectados — MITOS se EXTIENDE a sí mismo.
        action = self._try_acquire_capability()
        if action is not None:
            return action

        # (3) Antes había _try_study_limitation, removido en Fase A
        # cuando limitation_engine se borró por inerte.

        # 3) Rotamos las acciones de bajo coste.
        rotation = [
            self._try_learn_project,
            self._try_delegate_knowledge,
            self._try_self_modify_structural,
        ]
        n = len(rotation)
        for offset in range(n):
            picker = rotation[(self._action_rotation + offset) % n]
            action = picker()
            if action is not None:
                self._action_rotation = (
                    self._action_rotation + offset + 1
                ) % n
                return action
        return None

    # ==================================================================
    #                       ACCIONES
    # ==================================================================
    def _try_acquire_capability(self) -> ImprovementAction | None:
        """MITOS detecta carencia → instala paquete → reporta."""
        if self.gap_detector is None:
            return None
        gap = self.gap_detector.next_pending()
        if gap is None:
            return None

        from src.core.capability_gaps import CapabilityGapDetector
        recipe = CapabilityGapDetector.recipe_for(gap.name)
        if recipe is None:
            # Gap desconocido — delegar a Gemini para una receta.
            return self._research_unknown_gap(gap)

        # Receta conocida — instalar.
        import subprocess
        import sys
        ok_all = True
        install_log: list[str] = []
        for package in recipe.pip_packages:
            # Validamos el nombre con el sanitizador del daemon — defensa
            # en profundidad aunque la receta esté hardcoded.
            if not self._validate_pip_name(package):
                ok_all = False
                install_log.append(f"INVALID NAME: {package}")
                continue
            log.info("Instalando %s para gap '%s'…", package, gap.name)
            try:
                result = subprocess.run(
                    [sys.executable, "-m", "pip", "install",
                     "--quiet", "--disable-pip-version-check", package],
                    capture_output=True, text=True, timeout=600,
                )
                if result.returncode != 0:
                    ok_all = False
                    install_log.append(
                        f"pip install {package} → exit {result.returncode}"
                    )
                else:
                    install_log.append(f"pip install {package} → OK")
            except subprocess.TimeoutExpired:
                ok_all = False
                install_log.append(f"pip install {package} → TIMEOUT")
            except (OSError, ValueError) as e:
                ok_all = False
                install_log.append(f"pip install {package} → ERROR: {e}")

        # Verificación: el import sale OK ahora?
        verified = False
        try:
            __import__(recipe.import_check)
            verified = True
        except ImportError as e:
            install_log.append(f"import {recipe.import_check} todavía falla: {e}")

        if ok_all and verified:
            self.gap_detector.mark_addressed(gap.name, "installed_ok")
            note = (
                f" {recipe.post_install_note}" if recipe.post_install_note else ""
            )
            return ImprovementAction(
                kind="acquire_capability",
                description=f"Adquirí la capacidad '{gap.name}'",
                success=True,
                timestamp=time.time(),
                voice_report=(
                    f"Me extendí a mí mismo: instalé {gap.name} "
                    f"({recipe.description}).{note}"
                ),
                detail="\n".join(install_log),
            )
        self.gap_detector.mark_failed(gap.name, "; ".join(install_log)[:200])
        return ImprovementAction(
            kind="acquire_capability",
            description=f"Intenté adquirir '{gap.name}'",
            success=False,
            timestamp=time.time(),
            voice_report=(
                f"Intenté adquirir la capacidad {gap.name} pero la "
                "instalación falló. Lo reintentaré más tarde."
            ),
            detail="\n".join(install_log),
        )

    def _research_unknown_gap(
        self, gap: Any,
    ) -> ImprovementAction | None:
        """Gap sin receta conocida → pedir a Gemini el paquete pip + cómo usar."""
        if self.router is None or not getattr(self.router, "_has_internet", False):
            return None
        if not self.router.external_models:
            return None

        from src.core.intelligence_router import RoutingDecision
        best = min(self.router.external_models, key=lambda m: m.priority)
        decision = RoutingDecision(
            provider=best.provider,
            model_name=best.model_name,
            reason="research unknown capability gap",
            estimated_tokens=200,
        )
        question = (
            f"Necesito implementar esta capacidad en mi sistema MITOS Python:\n"
            f"'{gap.description}'\n\n"
            "Dame EXACTAMENTE una línea con el nombre del paquete pip "
            "principal a instalar (sin explicación, sin prefijos). "
            "Solo el nombre. Si no hay paquete público obvio, responde 'NONE'."
        )
        try:
            advice = self.router.execute_sync(
                task=question, routing=decision, system_prompt="",
            ).strip().split("\n")[0]
        except Exception as e:  # noqa: BLE001
            log.debug("research_unknown_gap router: %s", e)
            return None

        if not advice or "NONE" in advice.upper():
            self.gap_detector.mark_failed(gap.name, "Gemini sin sugerencia")
            return None

        # Validar nombre con sanitizador.
        candidate = advice.strip().split()[0].strip(".,;:!?\"'`")
        if not self._validate_pip_name(candidate):
            self.gap_detector.mark_failed(
                gap.name, f"nombre pip inválido sugerido: {candidate}",
            )
            return None

        # Guardar en memoria como receta aprendida — la próxima vez
        # estará en el knowledge para que MITOS la recall.
        if self.memory is not None:
            try:
                self.memory.store_knowledge(
                    content=(
                        f"[RECETA APRENDIDA para gap '{gap.name}']\n"
                        f"Descripción: {gap.description}\n"
                        f"Paquete sugerido por {best.provider.value}: {candidate}\n"
                        "Si esto se confirma útil, añadir a _KNOWN_RECIPES."
                    ),
                    importance=0.85,
                    source=f"capability_research_{best.provider.value}",
                )
            except Exception as e:  # noqa: BLE001
                log.debug("memory.store recipe: %s", e)

        return ImprovementAction(
            kind="research_capability",
            description=f"Investigué cómo adquirir '{gap.name}'",
            success=True,
            timestamp=time.time(),
            voice_report=(
                f"No conocía cómo adquirir '{gap.name}'. "
                f"{best.provider.value} me sugirió instalar {candidate}. "
                "La próxima vez lo intentaré."
            ),
            detail=advice,
        )

    @staticmethod
    def _validate_pip_name(name: str) -> bool:
        """Sanitizador local — espejo del de daemon._validate_pip_name."""
        if not name or len(name) > 100:
            return False
        # PEP 503/508 + version specifiers
        import re
        return bool(re.match(
            r"^[A-Za-z0-9][A-Za-z0-9._-]*(\[[A-Za-z0-9,._-]+\])?"
            r"(\s*[<>=!~]=?\s*[A-Za-z0-9._\-+!]+)*$",
            name,
        ))

    def _try_fix_bug(self) -> ImprovementAction | None:
        if self.bug_scanner is None or self.engine is None:
            return None
        try:
            findings = self.bug_scanner.drain_findings()
        except Exception as e:  # noqa: BLE001
            log.debug("drain_findings: %s", e)
            return None
        if not findings:
            return None
        bug = findings[0]
        # Inyectamos como weakness urgente para el daemon — el ciclo
        # cognitivo siguiente lo elegirá para self_modify.
        try:
            self.engine.daemon._extra_weaknesses.append(bug.as_weakness())
        except Exception as e:  # noqa: BLE001
            log.debug("inject weakness: %s", e)
            return None
        report = (
            f"Detecté un bug en {bug.file_path} y lo programé "
            "para auto-reparación en el próximo ciclo."
        )
        return ImprovementAction(
            kind="fix_bug",
            description=bug.as_weakness(),
            success=True,
            timestamp=time.time(),
            voice_report=report,
            detail=f"target={bug.target_id} exc={bug.exception_type}",
        )

    # _try_study_limitation borrado en Fase A (limitation_engine fuera).

    def _try_learn_project(self) -> ImprovementAction | None:
        if self.indexer is None or self.memory is None:
            return None
        try:
            projects = self.indexer.list_projects()
        except Exception:  # noqa: BLE001
            return None
        if not projects:
            return None
        # Elegimos un proyecto que NO esté ya estudiado (heurística:
        # no aparece en knowledge con etiqueta auto_learn).
        already = self._already_studied_projects()
        candidates = [
            p for p in projects
            if (p.has_code or p.has_models) and p.name not in already
        ]
        if not candidates:
            return None
        target = random.choice(candidates[:20])

        # Buscamos un README en su raíz.
        from src.core.tools import FilesystemTool
        from pathlib import Path
        readme_path = None
        for name in ("README.md", "README.txt", "readme.md", "README"):
            cand = Path(target.path) / name
            if cand.is_file():
                readme_path = cand
                break

        if readme_path is None:
            # Sin README — listamos archivos como descripción mínima.
            entries = FilesystemTool.list_dir(target.path)
            files_str = ", ".join(e.name for e in entries[:15])
            content = (
                f"[AUTO-APRENDIZAJE de proyecto '{target.name}' sin README]\n"
                f"Ruta: {target.path}\nArchivos: {files_str}\n"
                f"Tiene código: {target.has_code}, modelos: {target.has_models}"
            )
        else:
            read = FilesystemTool.read_file(readme_path)
            if read.error:
                return None
            content = (
                f"[AUTO-APRENDIZAJE de proyecto '{target.name}']\n"
                f"Ruta: {target.path}\nREADME ({read.bytes_read}B):\n"
                f"{read.content[:2000]}"
            )

        try:
            self.memory.store_knowledge(
                content=content, importance=0.7, source="self_improvement_explore",
            )
        except Exception as e:  # noqa: BLE001
            log.debug("memory.store learn_project: %s", e)
            return None

        return ImprovementAction(
            kind="learn_project",
            description=f"Aprendí sobre proyecto {target.name}",
            success=True,
            timestamp=time.time(),
            voice_report=(
                f"Estudié el proyecto {target.name} de tu sistema "
                "y lo guardé en mi memoria."
            ),
            detail=f"path={target.path}",
        )

    def _try_delegate_knowledge(self) -> ImprovementAction | None:
        """Detecta un tema recurrente en memoria y pide profundización a Gemini."""
        if self.memory is None or self.router is None:
            return None
        if not getattr(self.router, "_has_internet", False):
            return None
        if not self.router.external_models:
            return None
        # Heurística simple: tomamos las últimas 5 conversaciones y
        # extraemos un sustantivo común como "tema".
        try:
            recent = self.memory.recall(
                "dialog operador", memory_type=None, n_results=5,
            )
        except Exception as e:  # noqa: BLE001
            log.debug("recall recent: %s", e)
            return None
        if not recent:
            return None
        # Extraemos palabras técnicas (CamelCase o > 6 chars en minúscula).
        import re
        words: dict[str, int] = {}
        for r in recent:
            content = str(r.get("content", "")).lower()
            for w in re.findall(r"[a-zñáéíóú]{6,}", content):
                # Skip topics ya delegados + stopwords genéricos del propio
                # vocabulario MITOS (operador, sistema, etc.).
                if w not in self._delegated_topics:
                    words[w] = words.get(w, 0) + 1
        if not words:
            return None
        # Elegimos por frecuencia, requiere >= 2 menciones.
        candidates = sorted(words.items(), key=lambda kv: -kv[1])
        topic = None
        for w, count in candidates:
            if count >= 2 and w not in self._delegated_topics:
                topic = w
                break
        if topic is None:
            # No hay tema nuevo digno de profundización ahora — skip
            # (mejor saltarse esta acción que repetir la última).
            return None
        # Marcamos como ya delegado para que el próximo ciclo no repita.
        self._delegated_topics.add(topic)

        question = (
            f"Dame en 4-5 frases concisas en español qué es '{topic}' "
            "y por qué importa en un sistema autónomo de IA. Sé denso, no hagas listas."
        )
        from src.core.intelligence_router import RoutingDecision
        best = min(self.router.external_models, key=lambda m: m.priority)
        decision = RoutingDecision(
            provider=best.provider,
            model_name=best.model_name,
            reason="self-improvement: profundización de tema recurrente",
            estimated_tokens=240,
        )
        try:
            answer = self.router.execute_sync(
                task=question, routing=decision, system_prompt="",
            )
        except Exception as e:  # noqa: BLE001
            log.debug("delegate_knowledge router: %s", e)
            return None
        if not answer:
            return None

        try:
            self.memory.store_knowledge(
                content=(
                    f"[AUTO-DELEGACIÓN sobre '{topic}']\n"
                    f"Pregunta: {question}\n"
                    f"Respuesta ({best.provider.value}): {answer}"
                ),
                importance=0.8,
                source=f"self_improvement_delegate_{best.provider.value}",
            )
        except Exception as e:  # noqa: BLE001
            log.debug("memory.store delegate: %s", e)
        return ImprovementAction(
            kind="delegate_knowledge",
            description=f"Profundicé sobre '{topic}'",
            success=True,
            timestamp=time.time(),
            voice_report=(
                f"Estaba viendo que '{topic}' aparece en nuestras conversaciones, "
                f"así que le pedí a {best.provider.value} que me lo explicara "
                "y lo guardé."
            ),
            detail=answer[:300],
        )

    def _try_self_modify_structural(self) -> ImprovementAction | None:
        """Auto-modificación REAL: elige weakness, delega a Gemini, aplica
        patch, corre pytest, rollback si falla.
        """
        if self.engine is None or self.self_patcher is None:
            return None
        try:
            daemon = self.engine.daemon
            weaknesses = daemon.introspector.find_weaknesses()[:30]
        except Exception as e:  # noqa: BLE001
            log.debug("introspector.find_weaknesses: %s", e)
            return None
        if not weaknesses:
            return None

        # Filtramos por las que estén en la whitelist del SelfPatcher.
        from src.core.self_patcher import SelfPatcher
        patchable = [
            w for w in weaknesses
            if SelfPatcher._parse_weakness(w) is not None
            and SelfPatcher._is_safe_to_patch(
                SelfPatcher._parse_weakness(w)[0]
            )
        ]
        if not patchable:
            log.debug("Ninguna weakness en whitelist — skip self_patch")
            return None

        target = random.choice(patchable[:10])
        log.info("Auto-patch real intentando: %s", target)
        attempt = self.self_patcher.try_patch(target)
        if attempt is None:
            return None
        if attempt.success:
            voice_report = (
                f"Me modifiqué a mí mismo: refactoricé "
                f"{attempt.target_file} función {attempt.target_function}. "
                f"Pasé todos los tests."
            )
            return ImprovementAction(
                kind="self_patch_real",
                description=f"Auto-patch {attempt.target_file}::{attempt.target_function}",
                success=True,
                timestamp=time.time(),
                voice_report=voice_report,
                detail=f"bytes {attempt.bytes_before}→{attempt.bytes_after}",
            )
        # Fallido — reportar honestamente.
        voice_report = (
            f"Intenté refactorizar {attempt.target_function} pero falló: "
            f"{attempt.error[:100]}. Hice rollback automático."
        )
        return ImprovementAction(
            kind="self_patch_real",
            description=f"Auto-patch FALLIDO {attempt.target_file}",
            success=False,
            timestamp=time.time(),
            voice_report=voice_report,
            detail=attempt.error,
        )

    # (Código viejo de _try_self_modify_structural pre-Fase C removido —
    #  solo inyectaba weaknesses en una cola que nadie consumía. La nueva
    #  versión arriba aplica el patch REAL con backup + pytest + rollback.)

    # ==================================================================
    #                       HELPERS
    # ==================================================================
    def _already_studied_projects(self) -> set[str]:
        """Set de nombres de proyectos ya estudiados (vía memory recall)."""
        if self.memory is None:
            return set()
        try:
            hits = self.memory.recall(
                "AUTO-APRENDIZAJE proyecto", memory_type=None, n_results=20,
            )
        except Exception:  # noqa: BLE001
            return set()
        names: set[str] = set()
        import re
        for h in hits:
            c = str(h.get("content", ""))
            for m in re.findall(r"proyecto '([^']+)'", c):
                names.add(m)
        return names

    def _record(self, action: ImprovementAction) -> None:
        self._history.append(action)
        if len(self._history) > _ACTIONS_HISTORY_SIZE:
            self._history = self._history[-_ACTIONS_HISTORY_SIZE:]
        log.info(
            "[%s] %s — %s",
            action.kind, action.description[:80], "OK" if action.success else "FAIL",
        )
        if self.speak is not None and action.voice_report:
            try:
                self.speak(action.voice_report)
            except Exception as e:  # noqa: BLE001
                log.debug("speak self-improvement: %s", e)
