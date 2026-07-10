#!/usr/bin/env python
"""
==============================================================================
 Proyecto MITOS - Entry point del daemon autónomo + consola del operador
==============================================================================

Lanza el daemon cognitivo en un thread y abre una consola interactiva en
el thread principal. Así puedes:

  - Verlo trabajar (logs del daemon en stderr).
  - Hablar con él (/ask <pregunta>).
  - Inyectarle objetivos prioritarios (/say <texto>).
  - Pausarlo / reanudarlo / apagarlo limpio.

Concurrencia:
  El LLMEngine tiene un threading.RLock interno. Cuando tú haces /ask,
  esperas (a lo sumo) a que el ciclo cognitivo en curso termine su
  llamada al LLM, y entonces respondes tú. Para respuesta inmediata,
  usa primero `/pause`.

Uso:
    .venv/Scripts/python run_daemon.py        (Windows)
    .venv/bin/python    run_daemon.py         (Linux/Mac)
==============================================================================
"""

from __future__ import annotations

import argparse
import logging
import sys
import threading
import time
from pathlib import Path
from typing import Any

from rich.console import Console
from rich.tree import Tree

# Consola rich específica para output con formato (status, etc.). Usa
# stderr para no mezclarse con stdout (que el operador puede redirigir).
_rich_console: Console = Console(stderr=True)

# ============================================================================
# Configuración de logging
# ============================================================================
# El daemon emite mucha información: queremos timestamps y nombre de
# logger para auditar qué módulo dijo qué.
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    datefmt="%H:%M:%S",
    stream=sys.stderr,
)

# Silenciamos librerías ruidosas; nos quedamos solo con sus avisos.
for noisy in (
    "httpx", "httpcore", "chromadb", "urllib3",
    "sentence_transformers", "huggingface_hub",
):
    logging.getLogger(noisy).setLevel(logging.WARNING)

# Import después del basicConfig para que el logger del daemon ya esté
# bajo nuestro formato.
from src.core.daemon import MitosDaemon  # noqa: E402


# ============================================================================
# Meta-objetivo del operador (editable)
# ============================================================================
META_OBJECTIVE: str = (
    "Mejorar continuamente mis capacidades de razonamiento y código. "
    "Aprender de fuentes externas (GitHub, web). "
    "Optimizar mi propio código fuente para ser más eficiente."
)


# ============================================================================
# Banner
# ============================================================================
_BANNER: str = r"""
+-----------------------------------------------------------+
|                                                           |
|           MITOS  -  DAEMON AUTONOMO  (Fase 3)             |
|                                                           |
|   Loop cognitivo en background + consola del operador.    |
|   /help para ver comandos. /quit para salir.              |
|                                                           |
+-----------------------------------------------------------+
"""

_HELP: str = """
Comandos:
  /status         resumen del estado del daemon (ciclos, goals, memoria)
  /goals          lista los objetivos pendientes
  /memory         estadísticas por colección de memoria
  /pause          pausa el loop cognitivo (deja libre el LLM)
  /resume         reanuda el loop
  /say <texto>    inyecta un objetivo de alta prioridad
  /ask <pregunta> conversa directamente con MITOS (auto-pause + auto-resume)
  /undo           rollback de la última automodificación (auto-pausa)
  /help           muestra esta ayuda
  /quit           apaga el daemon y sale (limpio)
"""


# ============================================================================
# Consola del operador (thread principal)
# ============================================================================
def operator_console(daemon: MitosDaemon) -> None:
    """Lee comandos del operador y los enruta hacia el daemon."""
    print(_HELP, file=sys.stderr)

    while daemon.is_alive:
        try:
            line = input("operator> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nSeñal de salida recibida.", file=sys.stderr)
            break

        if not line:
            continue
        if line in ("/quit", "/exit"):
            break
        if line == "/help":
            print(_HELP, file=sys.stderr)
            continue
        if line == "/status":
            _print_status(daemon)
            continue
        if line == "/goals":
            _print_goals(daemon)
            continue
        if line == "/memory":
            _print_memory(daemon)
            continue
        if line == "/pause":
            daemon.pause()
            print("[OK] daemon pausado.", file=sys.stderr)
            continue
        if line == "/resume":
            daemon.resume()
            print("[OK] daemon reanudado.", file=sys.stderr)
            continue
        if line == "/undo":
            # Pausa para que un ciclo en curso no se atraviese; restaura;
            # reanuda solo si no estaba pausado de antes.
            was_paused = daemon.is_paused
            if not was_paused:
                daemon.pause()
            ok = daemon.undo_last_self_mod()
            if not was_paused:
                daemon.resume()
            if ok:
                print(
                    "[OK] última automodificación revertida desde backup.",
                    file=sys.stderr,
                )
            else:
                print(
                    "[NO] sin historial de modificaciones o backup ausente.",
                    file=sys.stderr,
                )
            continue
        if line.startswith("/say "):
            ok = daemon.inject_goal(line[len("/say "):])
            msg = "objetivo añadido" if ok else "duplicado o vacío, no añadido"
            print(f"[{'OK' if ok else 'NO'}] {msg}.", file=sys.stderr)
            continue
        if line.startswith("/ask "):
            _ask(daemon, line[len("/ask "):])
            continue
        if line.startswith("/"):
            print(f"comando desconocido: {line}", file=sys.stderr)
            print("Usa /help para ver los disponibles.", file=sys.stderr)
            continue

        # Sin prefijo: interpretamos como /ask para ser tolerantes.
        _ask(daemon, line)


# ============================================================================
# Helpers de presentación
# ============================================================================
def _print_status(daemon: MitosDaemon) -> None:
    """Renderiza el `status_snapshot` como árbol rich (lectura cómoda)."""
    snap = daemon.status_snapshot()
    tree = Tree("[bold cyan]STATUS[/bold cyan]")
    _populate_tree(tree, snap)
    _rich_console.print(tree)


def _populate_tree(node: Tree, data: Any) -> None:
    """Inserta recursivamente un dict/list arbitrario como ramas del árbol.

    Sólo formatea visualmente: no muta `data`. Strings, ints, floats y
    bools se imprimen tal cual; dicts se abren en sub-ramas; listas se
    cortan a primeros 8 items para no saturar la consola.
    """
    if isinstance(data, dict):
        for key, value in data.items():
            if isinstance(value, (dict, list, tuple)) and value:
                sub = node.add(f"[bold]{key}[/bold]")
                _populate_tree(sub, value)
            else:
                node.add(f"[bold]{key}[/bold]: {_fmt_scalar(value)}")
    elif isinstance(data, (list, tuple)):
        for i, item in enumerate(list(data)[:8]):
            if isinstance(item, (dict, list, tuple)):
                sub = node.add(f"[{i}]")
                _populate_tree(sub, item)
            else:
                node.add(f"[{i}] {_fmt_scalar(item)}")
        if len(data) > 8:
            node.add(f"[dim]... +{len(data) - 8} más[/dim]")
    else:
        node.add(_fmt_scalar(data))


def _fmt_scalar(value: Any) -> str:
    """Formato compacto para escalares (None, bool, num, str)."""
    if value is None:
        return "[dim]None[/dim]"
    if isinstance(value, bool):
        return "[green]True[/green]" if value else "[red]False[/red]"
    if isinstance(value, float):
        return f"{value:.2f}"
    return str(value)


def _print_goals(daemon: MitosDaemon) -> None:
    goals = daemon.goals.all_goals()
    if not goals:
        print("(sin objetivos)", file=sys.stderr)
        return
    print(f"\n--- GOALS ({len(goals)}) ---", file=sys.stderr)
    for g in goals:
        marker = {
            "pending": "·",
            "completed": "+",
            "failed": "X",
        }.get(g.status, "?")
        print(
            f"  [{marker}] [{g.id}] p={g.priority:.2f} src={g.source} "
            f"-> {g.description[:80]}",
            file=sys.stderr,
        )
    print("", file=sys.stderr)


def _print_memory(daemon: MitosDaemon) -> None:
    stats = daemon.memory.get_stats()
    print("\n--- MEMORY ---", file=sys.stderr)
    for k, v in stats.items():
        print(f"  {k}: {v}", file=sys.stderr)
    print("", file=sys.stderr)


def _ask(daemon: MitosDaemon, question: str) -> None:
    """Auto-pausa + ask + auto-resume para respuesta más rápida."""
    if not question.strip():
        print("(pregunta vacía)", file=sys.stderr)
        return
    was_paused = daemon.is_paused
    if not was_paused:
        daemon.pause()
    print("[pensando...]", file=sys.stderr)
    t0 = time.time()
    try:
        answer = daemon.ask_operator(question)
    except Exception as e:  # noqa: BLE001
        print(f"[ERROR] {type(e).__name__}: {e}", file=sys.stderr)
        answer = ""
    elapsed = time.time() - t0
    if not was_paused:
        daemon.resume()
    if answer:
        print(f"\n--- RESPUESTA ({elapsed:.1f}s) ---", file=sys.stderr)
        print(answer, file=sys.stderr)
        print("", file=sys.stderr)


# ============================================================================
# Main
# ============================================================================
def _ensure_abliterated_model(project_root: Path) -> None:
    """
    Verifica que el .gguf por defecto (Fase 5:
    Qwen2.5-7B-Instruct-abliterated) esté presente en `models/`. Si no,
    lo descarga con `hf download` o, como fallback, `huggingface-cli`.

    Importante: comprueba el archivo ESPECÍFICO, no "cualquier .gguf".
    Si el operador tiene un modelo distinto en `models/` (p.ej. un
    Phi-3 viejo), seguimos descargando Qwen — el chat template del
    engine es ChatML y mezclarlo con Phi-3 produce turn-leakage. Sin
    este chequeo estricto, una migración deja el sistema cargando un
    .gguf incompatible con el template configurado.

    No abortamos si la descarga falla: dejamos que `LLMEngine` reporte
    el error con su mensaje específico, que incluye instrucciones para
    descarga manual.
    """
    import shutil
    import subprocess

    # Fase 8: Qwen2.5-Coder-14B-Instruct GGUF oficial (Q4_K_M,
    # ~9 GB partido en 2-3 splits). ChatML template — el llm_engine ya
    # está cableado para hablar ChatML, así que mezclar este .gguf con
    # otro de template distinto (Phi-3, Llama) daría turn-leakage.
    #
    # Razón del swap respecto a Fase 7 (Qwen 7B base):
    #   - 14B parámetros vs 7B → más capacidad de razonamiento.
    #   - Variante "Coder" → entrenada específicamente sobre código,
    #     que es el 80% del workload de MITOS (_do_self_modify, etc.).
    #   - El `IntelligenceRouter` de Fase 8 delega tareas creativas y
    #     visión a Gemini 3.5 Flash, así que el local solo necesita
    #     ser bueno en código y dispatch — exactamente lo que es Coder.
    #
    # Nota sobre splits: llama.cpp resuelve automáticamente los
    # `-00002-of-N.gguf` cuando se le pasa el `-00001-of-N.gguf`,
    # así que solo chequeamos la presencia del primer split.
    repo = "Qwen/Qwen2.5-Coder-14B-Instruct-GGUF"
    file_glob = "qwen2.5-coder-14b-instruct-q4_k_m*.gguf"
    file_pattern = "qwen2.5-coder-14b-instruct-q4_k_m"

    models_dir = project_root / "models"
    models_dir.mkdir(exist_ok=True)

    # Chequeo por patrón: cualquier .gguf cuyo nombre encaje con
    # `file_pattern` cuenta como modelo presente. Esto cubre tanto
    # el split en 2 partes como un eventual single-file repack.
    if list(models_dir.glob(file_glob)):
        return

    # Aviso si hay otros .gguf sueltos no compatibles con ChatML
    # (típicamente residuos de migraciones previas — Phi-3, Llama 2).
    other_ggufs = [
        p.name for p in models_dir.glob("*.gguf")
        if file_pattern not in p.name.lower()
    ]
    if other_ggufs:
        print(
            f"\n[Fase 5] AVISO: hay otro(s) .gguf en {models_dir} que NO\n"
            f"  son compatibles con ChatML (probable residuo de modelos\n"
            f"  anteriores):\n"
            + "\n".join(f"  - {n}" for n in other_ggufs)
            + "\n  Puedes borrarlos cuando quieras para liberar disco.\n",
            file=sys.stderr,
        )

    print(
        f"\n[Fase 8] No encuentro Qwen2.5-Coder-14B (q4_k_m) en {models_dir}.\n"
        f"Descargando ({repo} :: {file_glob}, ~9 GB en 2-3 splits).\n"
        f"Esto va a tardar bastante mas que la sesion anterior.\n",
        file=sys.stderr,
    )

    # Preferimos `hf` (nuevo); caemos a `huggingface-cli` si no está.
    cli = shutil.which("hf") or shutil.which("huggingface-cli")
    if cli is None:
        print(
            "ERROR: no hay 'hf' ni 'huggingface-cli' en PATH. "
            "Instala con `pip install -r requirements.txt` y descarga manual:",
            file=sys.stderr,
        )
        print(
            f"  huggingface-cli download {repo} --include "
            f"\"{file_glob}\" --local-dir {models_dir}",
            file=sys.stderr,
        )
        return

    # Para repos con splits, `--include "<glob>"` baja todos los
    # archivos del split en una sola llamada.
    try:
        subprocess.run(
            [cli, "download", repo, "--include", file_glob,
             "--local-dir", str(models_dir)],
            check=True,
        )
    except subprocess.CalledProcessError as e:
        print(
            f"ERROR: descarga falló (exit {e.returncode}). "
            f"Intenta manualmente:\n"
            f"  {cli} download {repo} --include \"{file_glob}\" "
            f"--local-dir {models_dir}",
            file=sys.stderr,
        )


def _build_arg_parser() -> argparse.ArgumentParser:
    """CLI de Fase 5: modo sandbox y límite de ciclos."""
    parser = argparse.ArgumentParser(
        prog="run_daemon",
        description=(
            "MITOS - Daemon autónomo. Sin flags arranca normalmente. "
            "Con --dry-run entra en modo sandbox: ningún archivo .py "
            "se modifica; cada decisión queda en un transcript Markdown."
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help=(
            "Sandbox: el daemon corre el ciclo cognitivo completo "
            "(drives, goals, LLM, behavior_tester, regression_detector) "
            "pero NUNCA llama a rewriter.apply_change ni add_new_tool. "
            "Genera un transcript Markdown auditable."
        ),
    )
    parser.add_argument(
        "--max-cycles",
        type=int,
        default=None,
        metavar="N",
        help=(
            "Apaga el daemon limpiamente tras completar N ciclos "
            "cognitivos. Útil con --dry-run para experimentos cortos. "
            "Default: ilimitado."
        ),
    )
    parser.add_argument(
        "--transcript",
        type=str,
        default=None,
        metavar="PATH",
        help=(
            "Archivo Markdown donde escribir el transcript. Si se omite "
            "y --dry-run está activo, se genera "
            "`dry_run_<timestamp>.md` en el directorio del proyecto."
        ),
    )
    parser.add_argument(
        "--no-console",
        action="store_true",
        help=(
            "Arranca el daemon sin la consola interactiva del operador. "
            "Útil con --max-cycles para correr el sandbox de forma "
            "completamente desatendida."
        ),
    )
    return parser


def main() -> int:
    args = _build_arg_parser().parse_args()
    project_root = Path(__file__).resolve().parent

    print(_BANNER, file=sys.stderr)
    if args.dry_run:
        print(
            "[MODO SANDBOX] --dry-run activo. Ningún archivo .py será "
            "modificado durante esta sesión.",
            file=sys.stderr,
        )
    if args.max_cycles is not None:
        print(
            f"[LÍMITE] max_cycles={args.max_cycles} "
            f"(apagado automático).",
            file=sys.stderr,
        )

    # Fase 4: asegura modelo abliterado antes de instanciar el daemon.
    _ensure_abliterated_model(project_root)

    try:
        daemon = MitosDaemon(
            project_root=project_root,
            meta_objective=META_OBJECTIVE,
            dry_run=args.dry_run,
            max_cycles=args.max_cycles,
            transcript_path=args.transcript,
        )
    except FileNotFoundError as e:
        print(f"\nERROR: {e}\n", file=sys.stderr)
        print(
            "Sugerencia: ejecuta primero `./run_brain.ps1` para descargar "
            "el modelo, o copia tu propio .gguf en models/.",
            file=sys.stderr,
        )
        return 1
    except RuntimeError as e:
        print(f"\nERROR de runtime: {e}\n", file=sys.stderr)
        return 1

    # Arrancar el daemon en background.
    daemon_thread = threading.Thread(
        target=daemon.start,
        name="MITOS-daemon",
        daemon=True,
    )
    daemon_thread.start()

    # Pequeño respiro para que el bootstrap termine antes de mostrar la
    # consola (estética; los logs salen igual aunque sigas tecleando).
    time.sleep(1.0)

    if args.no_console:
        # Modo desatendido: esperamos al daemon (Ctrl+C lo apaga limpio).
        print(
            "[--no-console] esperando al daemon "
            "(Ctrl+C para apagar)...",
            file=sys.stderr,
        )
        try:
            while daemon.is_alive:
                daemon_thread.join(timeout=1.0)
        except KeyboardInterrupt:
            print("\nInterrumpido por el usuario.", file=sys.stderr)
    else:
        try:
            operator_console(daemon)
        except KeyboardInterrupt:
            print("\nInterrumpido por el usuario.", file=sys.stderr)

    # Apagado cooperativo.
    print("\nApagando daemon (puede tardar unos segundos)...", file=sys.stderr)
    daemon.stop_async()
    daemon_thread.join(timeout=15.0)
    if daemon_thread.is_alive():
        print(
            "(el daemon no terminó en 15s; el proceso saldrá igual)",
            file=sys.stderr,
        )

    # Si hubo transcript, recordamos la ruta al operador.
    if args.dry_run or args.transcript:
        # El daemon ya logueó la ruta; aquí lo repetimos en stdout normal
        # para que sea fácil pipear / abrir desde el shell.
        tp = getattr(daemon, "_transcript_path", None)
        if tp is not None:
            print(f"\nTranscript: {tp}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
