"""
==============================================================================
 Proyecto MITOS - CLI Interactiva del Cerebro
==============================================================================

Interfaz de terminal sobre AutonomousAgent. Soporta:

  /status            -> agent.get_self_report()
  /memory            -> tabla con stats de memoria
  /auto <objetivo>   -> ejecuta el loop autónomo durante N steps
  /code <tarea>      -> genera código (sin loop) y muestra veredicto del filtro
  /quit              -> salir
  <cualquier otro>   -> recall de memoria + reason_step_by_step + persistencia

Todo se ejecuta 100% local: la CLI no abre sockets ni hace llamadas remotas.
==============================================================================
"""

from __future__ import annotations

import logging
import sys
import time

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from .agent import AutonomousAgent
from .llm_engine import LLMEngine
from .memory import MemorySystem


console = Console()


# ============================================================================
# SUBSISTEMAS LAZY (voz, visión, router) — instanciados solo cuando se usan
# ============================================================================
_voice_engine = None
_vision_glance = None
_router = None


def _get_voice():
    """Inicializa VoiceEngine la primera vez que se llama."""
    global _voice_engine
    if _voice_engine is not None:
        return _voice_engine
    try:
        from src.core.voice_engine import VoiceEngine
        _voice_engine = VoiceEngine()
        if _voice_engine.is_available():
            console.print(
                "[dim]🔊 Voz activada (pyttsx3). "
                "Mitos hablará sus respuestas en voz alta.[/dim]"
            )
        else:
            console.print(
                "[dim]🔇 Voz no disponible (instala pyttsx3 para activarla).[/dim]"
            )
    except Exception as e:  # noqa: BLE001
        console.print(f"[dim]🔇 VoiceEngine no se pudo cargar: {e}[/dim]")
        _voice_engine = None
    return _voice_engine


def _get_vision():
    """Inicializa VisionGlance la primera vez."""
    global _vision_glance
    if _vision_glance is not None:
        return _vision_glance
    try:
        from src.core.vision_glance import VisionGlance
        _vision_glance = VisionGlance()
    except Exception as e:  # noqa: BLE001
        console.print(f"[dim]👁  VisionGlance no se pudo cargar: {e}[/dim]")
        _vision_glance = None
    return _vision_glance


def _get_router(local_llm):
    """Inicializa IntelligenceRouter la primera vez (necesita local_llm)."""
    global _router
    if _router is not None:
        return _router
    try:
        from pathlib import Path
        from src.core.intelligence_router import IntelligenceRouter
        project_root = Path(__file__).resolve().parent.parent.parent
        _router = IntelligenceRouter(
            project_root=project_root,
            local_llm=local_llm,
        )
    except Exception as e:  # noqa: BLE001
        console.print(f"[dim]🌐 Router no se pudo cargar: {e}[/dim]")
        _router = None
    return _router


# Patrones que indican "puedes verme/sentirme/oírme/percibirme" — disparan
# el modo perceptivo en vez del razonamiento normal.
_VISION_INTENT_PATTERNS: tuple[str, ...] = (
    "puedes verme", "puede verme", "podes verme", "me ves",
    "qué ves", "que ves", "describe lo que ves",
    "qué hay frente a ti", "que hay frente a ti", "mírame", "mirame",
    "toma una foto", "sácame una foto", "sacame una foto",
    "puede verme?", "puedes verme?",  # con interrogación literal por si rompe match
)
_AUDIO_INTENT_PATTERNS: tuple[str, ...] = (
    "puedo oírte", "puedo oirte", "podré oírte", "podre oirte",
    "podré oirlo", "podre oirlo", "podré oírlo", "podre oirlo",
    "puedes hablar", "habla en voz alta", "di algo en voz alta",
    "puedes hablarme", "puede hablarme",
)


def _matches_any(prompt: str, patterns: tuple[str, ...]) -> bool:
    """Determine if any of the specified patterns exist within the lowercase prompt.

    Args:
        prompt: The input string to search within.
        patterns: A tuple of substring patterns to check against the prompt.

    Returns:
        True if at least one pattern is found in the lowercase prompt, False otherwise.
    """
    p = prompt.lower()
    return any(pat in p for pat in patterns)


# Cuántas rutas auto-exploramos por mensaje (límite por seguridad — un
# operador podría pegar 50 rutas y reventaríamos el contexto del LLM).
_MAX_AUTO_EXPLORE_PATHS: int = 3
_MAX_AUTO_EXPLORE_ENTRIES_PER_DIR: int = 30


def _auto_explore_paths(prompt: str) -> str:
    """Detecta rutas mencionadas en `prompt` y devuelve su contenido REAL.

    Para directorios: lista las primeras N entradas.
    Para archivos: lee los primeros KB.
    Para rutas inexistentes: dice claramente "no existe" — esto evita
    que el LLM finja que sí.

    Devuelve el bloque listo para prependerse al prompt del LLM. Vacío
    si no hay rutas detectadas.
    """
    from src.core.tools import FilesystemTool
    from pathlib import Path

    hints = FilesystemTool.extract_path_hints(prompt)
    if not hints:
        return ""

    blocks: list[str] = []
    for path_str in hints[:_MAX_AUTO_EXPLORE_PATHS]:
        if not FilesystemTool.exists(path_str):
            blocks.append(f"[FS] {path_str} → NO EXISTE")
            continue
        p = Path(path_str).expanduser().resolve()
        if p.is_dir():
            entries = FilesystemTool.list_dir(p)
            lines = [f"[FS] listado de {p} (n={len(entries)}):"]
            for e in entries[:_MAX_AUTO_EXPLORE_ENTRIES_PER_DIR]:
                kind = "DIR" if e.is_dir else f"{e.size_bytes}B"
                lines.append(f"  - {e.name} [{kind}]")
            if len(entries) > _MAX_AUTO_EXPLORE_ENTRIES_PER_DIR:
                lines.append(
                    f"  ... y {len(entries) - _MAX_AUTO_EXPLORE_ENTRIES_PER_DIR} más"
                )
            blocks.append("\n".join(lines))
        else:
            result = FilesystemTool.read_file(p)
            if result.error:
                blocks.append(f"[FS] read {p} → ERROR: {result.error}")
            else:
                preview = result.content[:1500]
                truncado = " (truncado)" if result.truncated else ""
                blocks.append(
                    f"[FS] read {p} ({result.bytes_read}B{truncado}):\n"
                    f"```\n{preview}\n```"
                )
    return "\n\n".join(blocks)


def cmd_explore(path: str) -> None:
    """Lista o lee la ruta indicada — sin invocar al LLM."""
    from src.core.tools import FilesystemTool
    from pathlib import Path

    path = path.strip()
    if not path:
        console.print("[red]Uso: /explore <ruta>[/red]")
        return
    if not FilesystemTool.exists(path):
        console.print(f"[red]No existe:[/red] {path}")
        return
    p = Path(path).expanduser().resolve()
    if p.is_dir():
        entries = FilesystemTool.list_dir(p)
        table = Table(title=f"{p} (n={len(entries)})")
        table.add_column("nombre")
        table.add_column("tipo", justify="center")
        table.add_column("tamaño", justify="right")
        for e in entries[:80]:
            kind = "[blue]DIR[/]" if e.is_dir else "file"
            size = "—" if e.is_dir else f"{e.size_bytes:,} B"
            table.add_row(e.name, kind, size)
        if len(entries) > 80:
            table.add_row(f"... y {len(entries) - 80} más", "", "")
        console.print(table)
    else:
        result = FilesystemTool.read_file(p)
        if result.error:
            console.print(f"[red]{result.error}[/red]")
            return
        title = f"{p} ({result.bytes_read:,}B"
        if result.truncated:
            title += " — TRUNCADO"
        title += ")"
        console.print(Panel(result.content, title=title, border_style="green"))


def cmd_listen(agent: AutonomousAgent) -> None:
    """Escucha una utterance por el mic, la transcribe, la procesa como prompt.

    Usa MicrophoneSensor del SensorHub para STT con whisper. Si whisper
    no está instalado, sugiere `pip install openai-whisper sounddevice`.
    """
    try:
        from src.core.sensor_hub import MicrophoneSensor
        import queue as _queue
    except ImportError as e:
        console.print(f"[red]Sensor hub no importable: {e}[/red]")
        return

    sensor = MicrophoneSensor(event_queue=_queue.Queue(maxsize=10))
    if not sensor.is_available():
        if not sensor.ensure_dependency() or not sensor.is_available():
            console.print(
                "[yellow]Sin micrófono. Instala con "
                "`pip install sounddevice openai-whisper` y reintenta.[/yellow]"
            )
            return

    console.print(
        "[bold cyan]🎙  Escuchando 5s... Habla AHORA.[/bold cyan]"
    )
    try:
        text = sensor.record_and_transcribe(seconds=5)
    except Exception as e:  # noqa: BLE001
        console.print(f"[red]STT falló: {e}[/red]")
        return

    if not text:
        console.print("[yellow]No transcribí nada (silencio o whisper sin modelo).[/yellow]")
        return

    console.print(Panel(text, title="🎙  transcripción", border_style="cyan"))
    # Procesamos como si lo hubieras tecleado.
    cmd_freeform(agent, text)


# ============================================================================
# COMANDOS
# ============================================================================
def cmd_status(agent: AutonomousAgent) -> None:
    """Imprime el auto-reporte del agente dentro de un panel."""
    console.print(
        Panel(agent.get_self_report(), title="status", border_style="cyan")
    )


def cmd_memory(agent: AutonomousAgent) -> None:
    """Tabla con conteo por colección de la memoria vectorial."""
    stats = agent.memory.get_stats()
    table = Table(title="Memoria vectorial (ChromaDB local)")
    table.add_column("Colección", style="cyan")
    table.add_column("Documentos", justify="right", style="bold")
    for name, count in stats.items():
        table.add_row(name, str(count))
    console.print(table)


def cmd_auto(agent: AutonomousAgent, goal: str) -> None:
    """Ejecuta el bucle autónomo del agente y reporta progreso."""
    goal = goal.strip()
    if not goal:
        console.print("[red]Uso: /auto <objetivo>[/red]")
        return

    t0 = time.time()
    state = agent.run(goal=goal, max_steps=10)
    elapsed = time.time() - t0

    summary = Table(title=f"Resumen sesión autónoma ({elapsed:.1f}s)")
    summary.add_column("Métrica", style="cyan")
    summary.add_column("Valor", justify="right")
    summary.add_row("steps", str(state.steps_taken))
    summary.add_row("acciones", str(state.total_actions))
    summary.add_row("éxitos", f"[green]{state.successes}[/green]")
    summary.add_row("fracasos", f"[red]{state.failures}[/red]")
    if state.total_actions > 0:
        rate = state.successes / state.total_actions * 100
        summary.add_row("tasa éxito", f"{rate:.1f}%")
    console.print(summary)


def cmd_code(agent: AutonomousAgent, task: str) -> None:
    """
    Genera código para `task` sin pasar por el loop completo. Muestra el
    código en un panel y el veredicto del filtro autónomo en otro.
    """
    task = task.strip()
    if not task:
        console.print("[red]Uso: /code <descripción de la tarea>[/red]")
        return

    code = agent.llm.generate_code(task=task)
    if not code:
        console.print("[red]El LLM no devolvió código.[/red]")
        return

    report = agent.filter.evaluate(code)
    color = "green" if report.accepted else "red"
    verdict = "ACEPTADO" if report.accepted else "RECHAZADO"

    console.print(
        Panel(code, title="código generado", border_style=color)
    )

    detail = Table(title=f"Veredicto del filtro: {verdict}")
    detail.add_column("Métrica", style="cyan")
    detail.add_column("Valor", justify="right")
    detail.add_row("compilable", f"{report.compilable:.2f}")
    detail.add_row("complexity", f"{report.complexity:.2f}")
    detail.add_row("coherence", f"{report.coherence:.2f}")
    detail.add_row("novelty", f"{report.novelty:.2f}")
    detail.add_row("TOTAL", f"[bold]{report.total:.2f}[/bold]")
    if report.reasons:
        detail.add_row("razones", ", ".join(report.reasons))
    console.print(detail)

    if report.accepted:
        try:
            agent.memory.store_code(
                content=code,
                importance=0.5 + 0.5 * report.total,
                task=task,
                score=report.total,
                origin="/code",
            )
            console.print("[dim]código persistido en memoria.[/dim]")
        except Exception as e:  # noqa: BLE001
            console.print(f"[yellow]no pude guardar en memoria: {e}[/yellow]")


# Patrones de respuestas viejas estilo bot ("Step 1: ... Step 2: ...") que
# autoenvenenan el contexto si se recuperan. Si un hit del recall empata
# con cualquiera, lo descartamos.
_LOW_QUALITY_RECALL_PATTERNS: tuple[str, ...] = (
    "step 1:", "step 2:", "step 3:",
    "1.\n", "ranking: 1/1",
    "como una ia", "como un modelo de lenguaje",
    "no tengo conciencia", "no tengo sentimientos",
)


def _is_low_quality_recall(content: str) -> bool:
    """True si la entrada huele a respuesta robotica vieja."""
    low = content.lower()
    return any(p in low for p in _LOW_QUALITY_RECALL_PATTERNS)


def _is_conversational(prompt: str) -> bool:
    """¿Es una pregunta conversacional vs una orden técnica?

    Conversacional → `think()` directo, breve, sin forzar Chain-of-Thought.
    Técnica/imperativa → `reason_step_by_step()`.
    """
    p = prompt.lower().strip()
    # Palabras imperativas técnicas: si empieza así → razonamiento.
    technical_starts = (
        "implementa", "implement", "refactoriza", "refactor", "diseña",
        "design", "crea ", "create ", "genera ", "generate ", "escribe ",
        "write ", "arregla", "fix ", "depura", "debug", "explica",
        "explain", "analiza", "analyze", "optimiza", "optimize",
    )
    if any(p.startswith(s) for s in technical_starts):
        return False
    # Conversacional clara.
    conversational_starts = (
        "hola", "buenas", "qué tal", "que tal", "cómo estás", "como estas",
        "sabes", "te acuerdas", "recuerdas", "dime", "cuéntame", "cuentame",
        "qué piensas", "que piensas", "tu opinión", "tu opinion", "crees",
    )
    if any(p.startswith(s) for s in conversational_starts):
        return True
    # Pregunta corta y única → conversacional. Pregunta larga sin verbo
    # imperativo arriba → mejor razonar (puede ser análisis).
    if p.endswith("?") and len(p) < 100:
        return True
    return False


def cmd_freeform(agent: AutonomousAgent, prompt: str) -> None:
    """
    Default para input que no es comando:
        0. Si pregunta de visión/audio → toma foto + TTS.
        1. Recall de memoria (filtrado anti-veneno).
        2. Inferencia (CoT solo si la pregunta lo amerita).
        3. Panel con la respuesta.
        4. Lee la respuesta en voz alta (si VoiceEngine disponible).
        5. Persiste solo si la calidad supera un mínimo.
    """
    prompt = prompt.strip()
    if not prompt:
        return

    # 0. Intent de visión: el usuario pregunta "¿puedes verme?".
    if _matches_any(prompt, _VISION_INTENT_PATTERNS):
        response = _handle_vision_intent(agent, prompt)
        if response:
            return  # ya manejado, salimos.

    # 0b. Intent de audio: el usuario pregunta "¿puedes hablar?".
    if _matches_any(prompt, _AUDIO_INTENT_PATTERNS):
        _handle_audio_intent()
        # No salimos — además del demo de voz, seguimos con la respuesta normal.

    # 0c. AUTO-EXPLORACIÓN: si el operador mencionó rutas, las leemos
    # de verdad antes de invocar al LLM. Esto cierra el bucle de
    # "MITOS finge haber leído el archivo" — solo dice que leyó si el
    # contenido REAL aparece inyectado en el contexto.
    fs_context = _auto_explore_paths(prompt)
    if fs_context:
        console.print(
            f"[dim]📂 auto-exploré {fs_context.count('[FS]')} ruta(s) "
            "mencionada(s) en tu mensaje[/dim]"
        )
        # El LLM recibirá el contexto FS prepended al prompt — verá
        # contenido real, no podrá inventar.
        prompt = f"{fs_context}\n\nMensaje del operador:\n{prompt}"

    # 1. Contexto desde memoria — pedimos más para poder filtrar.
    raw_hits = agent.memory.recall(prompt, memory_type=None, n_results=8)
    hits = [h for h in raw_hits if not _is_low_quality_recall(h["content"])][:3]
    if hits:
        ctx_table = Table(title="contexto desde memoria (top-3 filtrado)")
        ctx_table.add_column("tipo", style="cyan")
        ctx_table.add_column("d", justify="right")
        ctx_table.add_column("contenido", overflow="fold")
        for h in hits:
            ctx_table.add_row(
                h["memory_type"],
                f"{h['distance']:.3f}",
                (h["content"][:200] + "...") if len(h["content"]) > 200 else h["content"],
            )
        console.print(ctx_table)
    elif raw_hits:
        console.print(
            f"[dim]memoria: {len(raw_hits)} hits encontrados pero todos "
            "filtrados por baja calidad (respuestas robóticas viejas).[/dim]"
        )

    # 2. Inferencia: think() para conversacional, CoT solo si es técnico.
    # Fase 8+: max_tokens BAJOS por defecto para latencia razonable en
    # 14B sobre CPU. Conversacional 160 tokens ~= 3-4 frases naturales,
    # tarda ~40-60s en lugar de los 10-20 min que provocaba 320 tokens
    # con razonamiento explícito.
    import time as _time
    t0 = _time.time()
    with console.status("[yellow]pensando...[/yellow]", spinner="dots"):
        if _is_conversational(prompt):
            response = agent.llm.think(
                prompt, max_tokens=160, temperature=0.7,
            )
            mode = "directo"
        else:
            response = agent.llm.reason_step_by_step(
                question=prompt, max_tokens=220, temperature=0.5,
            )
            mode = "razonamiento"
    elapsed = _time.time() - t0
    if elapsed > 60.0:
        console.print(
            f"[dim]respuesta en {elapsed:.0f}s. Si es muy lento, prueba "
            "/explore <ruta> en lugar de mensaje libre, o cambia al 7B abliterated.[/dim]"
        )

    # 3. Panel con la respuesta.
    console.print(
        Panel(
            response,
            title=f"{mode}: {prompt[:60]}",
            border_style="magenta",
        )
    )

    # 4. Voz: leemos la respuesta en altavoz si VoiceEngine está disponible.
    voice = _get_voice()
    if voice is not None and voice.is_available():
        # async para que el panel se vea ya impreso y el operador empiece
        # a oírlo sin tener que esperar a que termine de hablar.
        voice.say_async(response)

    # 5. Persistencia — pero NO si la respuesta es baja calidad
    # (no envenenamos el recall futuro).
    if _is_low_quality_recall(response):
        console.print(
            "[dim]respuesta no persistida (detectada baja calidad — "
            "evitamos envenenar el RAG futuro).[/dim]"
        )
        return
    try:
        agent.memory.store_knowledge(
            content=f"Q: {prompt}\nA: {response}",
            importance=0.6,
            source="cli_freeform",
        )
    except Exception as e:  # noqa: BLE001
        console.print(f"[yellow]no pude guardar Q&A: {e}[/yellow]")


# ============================================================================
# HANDLERS DE INTENT
# ============================================================================
def _capture_image() -> str | None:
    """Captura una imagen de la cámara y la devuelve en base64."""
    glance = _get_vision()
    if glance is None or not glance.is_available():
        console.print(
            "[yellow]Quiero verte pero no tengo cámara disponible "
            "(instala `pip install opencv-python` y conecta una webcam).[/yellow]"
        )
        return None

    console.print("[dim]📸 capturando frame...[/dim]")
    image_b64 = glance.snapshot_b64()
    if not image_b64:
        console.print("[yellow]No pude obtener una foto de la cámara.[/yellow]")
        return None
    return image_b64


def _describe_captured_image(agent: AutonomousAgent, image_b64: str) -> str | None:
    """Envía la imagen a Gemini para obtener una descripción."""
    router = _get_router(agent.llm)
    if router is None:
        console.print(
            "[yellow]Foto capturada, pero el IntelligenceRouter no está "
            "para mandársela a Gemini.[/yellow]"
        )
        return None

    console.print("[dim]🌐 delegando descripción visual a Gemini...[/dim]")
    description = router.describe_image(
        image_b64=image_b64,
        question=(
            "Describe en español, breve y directo (3-4 frases), lo que "
            "ves en esta imagen. Si hay una persona, descríbela. Si hay "
            "un entorno físico, descríbelo. Habla en primera persona "
            "como si fueras MITOS viendo a través de mi cámara."
        ),
    )
    if not description:
        console.print(
            "[yellow]Gemini no devolvió descripción (sin internet, "
            "rate-limit, o sin clave válida).[/yellow]"
        )
        return None
    return description


def _process_vision_result(agent: AutonomousAgent, prompt: str, description: str) -> None:
    """Muestra la descripción, la reproduce por voz y la guarda en memoria."""
    console.print(
        Panel(
            description,
            title="lo que veo a través de la cámara",
            border_style="green",
        )
    )

    voice = _get_voice()
    if voice is not None and voice.is_available():
        voice.say_async(description)

    try:
        agent.memory.store_knowledge(
            content=f"VISION Q: {prompt}\nA: {description}",
            importance=0.7,
            source="cli_vision_glance",
        )
    except Exception as e:  # noqa: BLE001
        console.print(f"[yellow]no pude guardar Q&A visión: {e}[/yellow]")


def _handle_vision_intent(agent: AutonomousAgent, prompt: str) -> str:
    """Toma una foto, la manda a Gemini, devuelve la descripción.

    Devuelve la descripción (string) si todo fue OK, o "" si algo falló
    y queremos que el caller siga con el flujo normal.
    """
    image_b64 = _capture_image()
    if not image_b64:
        return ""

    description = _describe_captured_image(agent, image_b64)
    if not description:
        return ""

    _process_vision_result(agent, prompt, description)
    return description


def _handle_audio_intent() -> None:
    """Demo de voz: dice una frase de prueba para que el operador confirme."""
    voice = _get_voice()
    if voice is None or not voice.is_available():
        console.print(
            "[yellow]No tengo voz cargada. Instala con "
            "`pip install pyttsx3` y reinicia el brain.[/yellow]"
        )
        return
    frase = (
        f"Hola, soy MITOS, hablando con la voz {voice._chosen_voice or 'por defecto'} "
        "de tu sistema. Si me oyes, el canal de audio funciona."
    )
    console.print(f"[dim]🔊 hablando: {frase[:60]}...[/dim]")
    voice.say(frase)  # BLOCANTE — esperamos a que termine el demo


# ============================================================================
# DISPATCH
# ============================================================================
def dispatch(agent: AutonomousAgent, raw: str) -> bool:
    """
    Despacha una línea de entrada al comando adecuado.

    Returns:
        True para seguir en el loop, False para salir.
    """
    raw = raw.strip()
    if not raw:
        return True

    if raw == "/quit" or raw == "/exit":
        return False
    if raw == "/status":
        cmd_status(agent)
        return True
    if raw == "/memory":
        cmd_memory(agent)
        return True
    if raw.startswith("/auto "):
        cmd_auto(agent, raw[len("/auto "):])
        return True
    if raw.startswith("/code "):
        cmd_code(agent, raw[len("/code "):])
        return True
    if raw.startswith("/explore "):
        cmd_explore(raw[len("/explore "):])
        return True
    if raw == "/listen":
        cmd_listen(agent)
        return True
    if raw.startswith("/"):
        console.print(f"[red]comando desconocido:[/red] {raw}")
        console.print(
            "[dim]disponibles: /status /memory /auto <goal> /code <task> "
            "/explore <path> /listen /quit[/dim]"
        )
        return True

    # Default: razonamiento libre.
    cmd_freeform(agent, raw)
    return True


# ============================================================================
# MAIN
# ============================================================================
def _welcome() -> None:
    """
    Displays a welcome message with available commands for the MITOS Brain CLI.
    """
    console.print(
        Panel.fit(
            "[bold cyan]MITOS Brain[/bold cyan] - CLI interactiva (100% local)\n"
            "[dim]Comandos: [/dim]"
            "[cyan]/status[/cyan]  "
            "[cyan]/memory[/cyan]  "
            "[cyan]/auto <goal>[/cyan]  "
            "[cyan]/code <task>[/cyan]  "
            "[cyan]/quit[/cyan]\n"
            "[dim]Cualquier otro texto se interpreta como pregunta libre "
            "(Chain-of-Thought + recall + persistencia).[/dim]",
            border_style="cyan",
        )
    )


def _init_agent() -> AutonomousAgent | None:
    """
    Inicializa LLM + memoria + agente con mensajes claros si algo falla.
    Devuelve None si no se pudo arrancar (la CLI termina con código 1).
    """
    try:
        with console.status(
            "[yellow]cargando modelo local...[/yellow]", spinner="dots"
        ):
            llm = LLMEngine()
    except FileNotFoundError as e:
        console.print(Panel(str(e), title="modelo no encontrado",
                            border_style="red"))
        return None
    except RuntimeError as e:
        console.print(Panel(str(e), title="error de runtime",
                            border_style="red"))
        return None

    try:
        memory = MemorySystem()
    except Exception as e:  # noqa: BLE001
        console.print(Panel(str(e), title="error inicializando memoria",
                            border_style="red"))
        return None

    return AutonomousAgent(llm=llm, memory=memory, console=console)


def main() -> int:
    """
    Main function that initializes logging, runs the agent, and handles user input.
    """
    logging.basicConfig(
        level=logging.INFO,
        format="[%(name)s] %(message)s",
        stream=sys.stderr,
    )

    _welcome()

    agent = _init_agent()
    if agent is None:
        return 1

    console.print("[green]listo.[/green]")

    while True:
        try:
            line = console.input("[bold cyan]mitos>[/bold cyan] ")
        except (EOFError, KeyboardInterrupt):
            console.print("\n[dim]saliendo...[/dim]")
            break
        try:
            cont = dispatch(agent, line)
        except KeyboardInterrupt:
            console.print("\n[yellow]comando interrumpido.[/yellow]")
            continue
        if not cont:
            console.print("[dim]adios.[/dim]")
            break

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
