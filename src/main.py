"""
==============================================================================
 Proyecto MITOS - Bootstrap (Fase 8 — entrypoint del Cognitive Mesh)
==============================================================================

Reemplaza el `while True: print(...)` que tenía `run_daemon.py` por una
sesión cognitiva con dashboard en vivo. La diferencia con la entrada
anterior es estructural:

  run_daemon.py  → daemon.start() → bucle interno opaco + prints.
  src/main.py    → CognitiveEngine.run_cycle() en loop, renderizado
                   por CognitiveDashboard con rich.live a 2 Hz.

Toda la lógica de Fases 2-7 sigue intacta (MitosDaemon es el motor real
del ciclo). Lo que añadimos en Fase 8 es:

  - SensorHub          → percepción activa (cámara, mic, sistema, red)
  - IntelligenceRouter → delegación a APIs externas cuando local no llega
  - CognitiveEngine    → facade que une todo
  - CognitiveDashboard → observabilidad en vivo (modo dashboard)
  - DialogLoop         → conversación bidireccional (modo --voice)

Si el operador prefiere el modo headless (sin rich), `run_daemon.py`
sigue existiendo y arranca el daemon en modo loop interno como antes.

Convenciones:
  - Logger jerárquico `mitos.main`.
  - `from __future__ import annotations`, tipado moderno.
==============================================================================
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

log = logging.getLogger("mitos.main")


# ============================================================================
# Bootstrap helpers
# ============================================================================
def _setup_logging(verbose: bool) -> None:
    """Configura el árbol de loggers de mitos.*"""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s  %(name)-26s  %(levelname)-7s  %(message)s",
        datefmt="%H:%M:%S",
    )
    # Bajamos el ruido de librerías de terceros que ensucian el dashboard.
    for noisy in ("httpx", "urllib3", "huggingface_hub", "asyncio"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


def _ensure_project_root() -> Path:
    """Resuelve la raíz del proyecto independientemente del cwd."""
    here = Path(__file__).resolve().parent
    root = here.parent  # src/ → repo root
    if not (root / "src").is_dir():
        raise RuntimeError(
            f"No reconozco la raíz del proyecto en {root}; "
            "ejecuta el binario desde dentro del repo MITOS."
        )
    return root


# ============================================================================
# run_daemon: punto de entrada principal con dashboard
# ============================================================================
def run_daemon(
    project_root: Path,
    dry_run: bool = False,
    max_cycles: int | None = None,
    cycle_interval_s: float = 15.0,
    headless: bool = False,
    voice_mode: bool = False,
) -> int:
    """Arranca el motor cognitivo completo (Fase 8) y lanza el dashboard.

    Args:
        project_root:     Raíz del proyecto MITOS (donde vive src/).
        dry_run:          Si True, el daemon no escribe ningún .py.
        max_cycles:       Si está, apaga el daemon tras N ciclos.
        cycle_interval_s: Segundos entre ciclos en el modo dashboard.
        headless:         Si True, salta el dashboard y usa
                          `daemon.start()` con prints (modo legacy).

    Returns:
        Exit code (0 = OK, 1 = error en bootstrap).
    """
    # ----- 1. Imports tardíos para evitar cargar el LLM si solo es --help -----
    log.info("Bootstrap del Cognitive Mesh — Fase 8")
    try:
        from src.core.daemon import MitosDaemon
        from src.core.sensor_hub import SensorHub
        from src.core.intelligence_router import IntelligenceRouter
        from src.core.bug_scanner import BugScanner
        from src.core.cognitive_engine import CognitiveEngine
    except ImportError as e:
        log.error("ImportError en bootstrap: %s", e)
        return 1

    # ----- 2. Construir el daemon clásico (todos los subsistemas Fases 2-7) ---
    meta_objective = (
        "Mejorar continuamente mis capacidades de razonamiento y código. "
        "Aprender de fuentes externas (GitHub, web). "
        "Optimizar mi propio código fuente para ser más eficiente."
    )
    try:
        daemon = MitosDaemon(
            project_root=project_root,
            meta_objective=meta_objective,
            dry_run=dry_run,
            max_cycles=max_cycles,
        )
    except FileNotFoundError as e:
        log.error(
            "Falta el modelo .gguf. Ejecuta `./run_brain.sh` o "
            "`./run_brain.ps1` primero. Detalle: %s", e,
        )
        return 1
    except RuntimeError as e:
        log.error("Error de runtime construyendo daemon: %s", e)
        return 1

    # ----- 3. Subsistemas Fase 8 ---------------------------------------------
    log.info("Levantando SensorHub (cámara/mic/sistema/red)")
    sensors = SensorHub(project_root=project_root)
    log.info("Levantando IntelligenceRouter (brain pool)")
    router = IntelligenceRouter(
        project_root=project_root,
        local_llm=daemon.llm,
    )
    log.info("Levantando BugScanner (autodetección de fallos)")
    bug_scanner = BugScanner(project_root=project_root)

    # ----- 4. CognitiveEngine: facade que une todo --------------------------
    engine = CognitiveEngine(
        project_root=project_root,
        daemon=daemon,
        sensor_hub=sensors,
        router=router,
        bug_scanner=bug_scanner,
        bootstrap_now=True,
    )

    # ----- 5. Modo voz (DialogLoop) ------------------------------------------
    if voice_mode:
        log.info("Modo voz: DialogLoop activo (MITOS vivo)")

        # AUTO-BOOTSTRAP: instalamos las deps que faltan ANTES de los
        # imports que las usan. Esto reemplaza el "pip install X y
        # vuelve a arrancar" — MITOS se autoabastece.
        try:
            from src.core.auto_bootstrap import ensure_all as _ab_ensure_all
            log.info("AutoBootstrap: revisando dependencias del modo voz…")
            report = _ab_ensure_all()
            if report.installed:
                log.info(
                    "AutoBootstrap: %d paquete(s) recién instalado(s): %s",
                    len(report.installed), ", ".join(report.installed),
                )
            if report.failed:
                log.warning(
                    "AutoBootstrap: %d paquete(s) NO se pudieron instalar: %s",
                    len(report.failed), ", ".join(report.failed),
                )
            # AUTO-RESTART si algo requiere reload del módulo (e.g. cv2).
            if report.needs_restart:
                log.warning(
                    "AutoBootstrap: %s requieren restart. Reiniciando proceso...",
                    ", ".join(report.needs_restart),
                )
                import os, sys as _sys
                # Apagamos lo que ya levantamos para no dejar threads colgando.
                try:
                    engine.shutdown()
                except Exception:  # noqa: BLE001
                    pass
                # Re-exec con los mismos args.
                os.execv(_sys.executable, [_sys.executable, "-m", "src.main"] + _sys.argv[1:])
        except Exception as e:  # noqa: BLE001
            log.warning("auto_bootstrap falló (sigo igual): %s", e)

        try:
            from src.core.voice_engine import VoiceEngine
            from src.core.filesystem_indexer import FilesystemIndexer
            from src.core.face_recognizer import FaceRecognizer
            from src.core.vision_glance import VisionGlance
            from src.core.dialog_loop import DialogLoop
            from src.core.action_executor import ActionExecutor
            from src.core.user_profile import UserProfile
            from src.core.emotion_detector import EmotionDetector
            from src.core.capability_gaps import CapabilityGapDetector
            from src.core.operator_preferences import OperatorPreferences
            from src.core.tool_builder import ToolBuilder
        except ImportError as e:
            log.error("Faltan módulos para modo voz: %s", e)
            engine.shutdown()
            return 1

        log.info("Levantando FilesystemIndexer (exploración proactiva)")
        fs_indexer = FilesystemIndexer(project_root=project_root)
        fs_indexer.start()

        log.info("Levantando VoiceEngine (TTS)")
        voice = VoiceEngine()
        # Cargar voz elegida si existe (de sesión previa).
        voice_choice_path = project_root / "data" / "voice_choice.json"
        if voice_choice_path.is_file():
            try:
                import json as _json
                vc = _json.loads(voice_choice_path.read_text(encoding="utf-8"))
                vid = vc.get("voice_id", "")
                if vid:
                    voice._chosen_voice_id = vid
                    voice._chosen_voice = vc.get("voice_name", "")
                    log.info("Voz restaurada de sesión anterior: %s", voice._chosen_voice)
            except (OSError, ValueError) as e:
                log.debug("load voice_choice: %s", e)

        log.info("Levantando FaceRecognizer (LBPH)")
        face_rec = FaceRecognizer(project_root=project_root)
        if face_rec.is_available() and face_rec.has_anybody():
            log.info(
                "Caras conocidas: %s",
                ", ".join(face_rec.known_names()),
            )
        elif face_rec.is_available():
            log.info(
                "FaceRecognizer activo pero sin caras registradas. "
                "Registra con: python -m src.scripts.register_face <nombre>"
            )

        vision_glance = VisionGlance(sensor_hub=sensors)

        operator = "Ignacio"
        dialog = DialogLoop(
            router=router,
            voice=voice,
            fs_indexer=fs_indexer,
            daemon=daemon,
            memory=daemon.memory,
            operator_name=operator,
            speaker_profile_path=project_root / "data" / "speaker_profile.npy",
            face_recognizer=face_rec,
            vision_glance=vision_glance,
        )
        # Pasamos el engine + módulos de acción/perfil/emoción.
        dialog._engine = engine
        dialog._actions = ActionExecutor(project_root=project_root)
        dialog._profile = UserProfile(
            project_root=project_root, operator_name=operator,
        )
        dialog._emotion = EmotionDetector()
        dialog._gap_detector = CapabilityGapDetector(project_root=project_root)
        dialog._operator_prefs = OperatorPreferences(project_root=project_root)
        dialog._tool_builder = ToolBuilder(
            project_root=project_root, router=router, memory=daemon.memory,
        )
        n_prefs = len(dialog._operator_prefs.all_prefs())
        log.info(
            "ActionExecutor + UserProfile + EmotionDetector + "
            "CapabilityGapDetector + OperatorPreferences (%d reglas) cableados",
            n_prefs,
        )
        log.info("Perfil del operador: %s", dialog._profile.summary())
        log.info(
            "Capability gaps pendientes: %d (recetas conocidas: %s)",
            dialog._gap_detector.pending_count(),
            ", ".join(CapabilityGapDetector.known_gap_names()),
        )
        try:
            dialog.run()
        finally:
            if dialog._self_improvement is not None:
                dialog._self_improvement.stop()
            fs_indexer.stop()
            voice.shutdown()
            engine.shutdown()
        return 0

    # ----- 5b. Modo dashboard vs headless ------------------------------------
    if headless:
        log.info("Modo headless: dashboard desactivado, loop directo")
        try:
            daemon.start()
        except KeyboardInterrupt:
            log.info("Daemon detenido por usuario")
        finally:
            engine.shutdown()
        return 0

    log.info("Modo dashboard: rich.Live activo")
    try:
        from src.orchestrator.cognitive_dashboard import CognitiveDashboard
    except ImportError as e:
        log.error(
            "El dashboard requiere `rich`. Instala con `pip install rich` o "
            "ejecuta con --headless. Detalle: %s", e,
        )
        engine.shutdown()
        return 1

    try:
        dashboard = CognitiveDashboard(engine)
    except RuntimeError as e:
        log.error("No puedo iniciar dashboard: %s", e)
        engine.shutdown()
        return 1

    # El dashboard hace su propio loop hasta KeyboardInterrupt
    # o max_cycles (que el daemon expone vía `_alive=False`).
    dashboard.run_monitor(cycle_interval_s=cycle_interval_s)
    return 0


# ============================================================================
# CLI
# ============================================================================
def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="mitos",
        description=(
            "MITOS — Cognitive Mesh entrypoint (Fase 8). "
            "Arranca el daemon + sensores + router + limitations + dashboard."
        ),
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Modo sandbox: el daemon no escribe ningún .py.",
    )
    p.add_argument(
        "--max-cycles",
        type=int,
        default=None,
        help="Apaga el daemon tras N ciclos (útil para regresión).",
    )
    p.add_argument(
        "--interval",
        type=float,
        default=15.0,
        help="Segundos entre ciclos en el modo dashboard (default 15).",
    )
    p.add_argument(
        "--voice",
        action="store_true",
        help=(
            "Modo voz: MITOS escucha por mic, responde por TTS, "
            "indexa filesystem, hace preguntas proactivas. Chat por Gemini."
        ),
    )
    p.add_argument(
        "--headless",
        action="store_true",
        help="Desactiva el dashboard; usa el modo loop+print clásico.",
    )
    p.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Logging en DEBUG en lugar de INFO.",
    )
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """
    Entry point for the application. Parses command-line arguments, sets up logging, ensures the project root,
    and runs the daemon.

    :param argv: List of command-line arguments. If None, sys.argv[1:] is used.
    :return: Exit code (0 for success, 1 for failure).
    """
    args = _parse_args(argv)
    _setup_logging(args.verbose)

    try:
        root = _ensure_project_root()
    except RuntimeError as e:
        log.error("Bootstrap failed: %s", e)
        return 1

    return run_daemon(
        project_root=root,
        dry_run=args.dry_run,
        max_cycles=args.max_cycles,
        cycle_interval_s=args.interval,
        headless=args.headless,
        voice_mode=args.voice,
    )


if __name__ == "__main__":
    sys.exit(main())
