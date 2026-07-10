"""
==============================================================================
 Proyecto MITOS - DialogLoop (Fase 8+ — MITOS VIVO)
==============================================================================

Sustituye el REPL del CLI brain por un loop de diálogo conversacional:

  - **Habla primero** al arrancar ("Hola Ignacio, soy MITOS...")
  - **Escucha SIEMPRE** con mic + VAD por energía. No hay /listen.
  - **Reconoce al hablante** (con resemblyzer si está; si no, "operador")
  - **Responde por Gemini** (2-4s, no 80s del 14B local en CPU)
  - **TTS lee la respuesta** por los altavoces
  - **Proactividad idle**: si el operador calla N min, MITOS comenta
  - **Auto-exploración**: si menciona un proyecto, el indexer lo localiza
    y el contenido REAL va al contexto antes de invocar a Gemini

Diseño:
  - Tres threads:
      1. `_mic_thread` — captura continua + VAD + transcripción
      2. `_speak_thread` — cola de utterances → TTS
      3. `_idle_thread` — timer de silencio → proactividad
  - El thread principal hace coordinación + escritura a stdout.
  - Cierre limpio con Ctrl+C: drena la cola de speech, para los threads.

Sin chat por internet → fallback al 14B local (lento pero funcional).

Convenciones:
  - Logger `mitos.dialog_loop`.
==============================================================================
"""

from __future__ import annotations

import logging
import queue
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

log = logging.getLogger("mitos.dialog_loop")


_VAD_ENERGY_THRESHOLD: float = 500.0   # RMS mínimo para considerar voz
_VAD_CHUNK_S: float = 1.0              # tamaño de la ventana de detección
_SILENCE_TO_END_S: float = 1.5         # cuánto silencio cierra una utterance
_MAX_UTTERANCE_S: float = 12.0         # cap por seguridad
_IDLE_PROACTIVE_MIN: float = 2.0       # ANTES: 5 min — operador lo quería más activo
_SAMPLE_RATE: int = 16000
_FACE_CHECK_INTERVAL_S: float = 12.0   # cada cuánto el thread de cara mira
_LEARN_FROM_DELEGATION: bool = True    # persistir respuestas de Gemini/Groq


@dataclass
class Utterance:
    """Una intervención del operador, transcrita y atribuida."""

    text: str
    speaker: str            # "Ignacio" si match, "operador" si no
    confidence: float       # similitud al speaker registrado
    timestamp: float = field(default_factory=time.time)


@dataclass
class _SpeakItem:
    """Una frase pendiente de ser sintetizada por TTS."""

    text: str
    priority: int = 5       # menor = antes
    label: str = "chat"     # "chat" | "proactive" | "system"


class DialogLoop:
    """Conversación en vivo con MITOS — habla, escucha, piensa, contesta."""

    def __init__(
        self,
        router: Any,
        voice: Any,
        fs_indexer: Any = None,
        daemon: Any = None,
        memory: Any = None,
        operator_name: str = "operador",
        speaker_profile_path: str | Path | None = None,
        face_recognizer: Any = None,
        vision_glance: Any = None,
    ) -> None:
        """
        Args:
            router: IntelligenceRouter — para chat por Gemini.
            voice:  VoiceEngine — TTS.
            fs_indexer: FilesystemIndexer ya arrancado, opcional.
            daemon: MitosDaemon, opcional (para usar el local como fallback).
            memory: MemorySystem para persistir conversación, opcional.
            operator_name: nombre por defecto cuando no hay speaker_profile.
            speaker_profile_path: ruta a .npy con embedding del speaker (resemblyzer).
        """
        self.router = router
        self.voice = voice
        self.indexer = fs_indexer
        self.daemon = daemon
        self.memory = memory
        self.operator_name = operator_name
        self.speaker_profile_path = (
            Path(speaker_profile_path) if speaker_profile_path else None
        )
        self.face_recognizer = face_recognizer
        self.vision_glance = vision_glance
        # Auto-mejora + acciones reales + perfil del operador.
        self._self_improvement: Any = None
        self._engine: Any = None
        self._actions: Any = None         # ActionExecutor
        self._profile: Any = None         # UserProfile
        self._emotion: Any = None         # EmotionDetector
        self._gap_detector: Any = None    # CapabilityGapDetector
        self._operator_prefs: Any = None  # OperatorPreferences
        self._tool_builder: Any = None    # ToolBuilder (Fase B)
        self._followup_counter: int = 0   # para rotar follow-up questions
        # Conversación: últimos N turnos para que Gemini tenga contexto.
        # Sin esto, MITOS olvida lo que dijo hace 30s.
        from collections import deque
        self._conv_history: deque[dict[str, Any]] = deque(maxlen=12)
        self._load_conversation_history()

        # Estado interno.
        self._running: bool = False
        self._utterance_queue: queue.Queue[Utterance] = queue.Queue(maxsize=20)
        self._speak_queue: queue.Queue[_SpeakItem] = queue.Queue(maxsize=20)
        self._last_user_interaction: float = time.time()
        self._mic_thread: threading.Thread | None = None
        self._speak_thread: threading.Thread | None = None
        self._idle_thread: threading.Thread | None = None
        self._face_thread: threading.Thread | None = None
        self._whisper_model: Any = None
        self._speaker_encoder: Any = None
        self._registered_embedding: Any = None
        # Memoria social: quién ha estado en cámara recientemente, para
        # que MITOS pueda decir "vi a Maria pasar" o "te veo ahí".
        self._seen_faces_recent: dict[str, float] = {}
        # Variedad en la proactividad: rotamos el tipo de iniciativa.
        self._proactive_seed: int = 0
        # Auto-registro: la próxima utterance se usa para entrenar el
        # perfil de voz (en lugar de responder al operador). Se activa
        # solo si no existe data/speaker_profile.npy al arrancar.
        self._capture_next_for_voice_registration: bool = False
        # Auto-registro de cara: si vemos cara desconocida y no hay
        # caras registradas, capturamos N frames seguidos.
        self._face_register_pending: bool = False
        self._face_register_buffer: list[Any] = []
        self._face_register_target_frames: int = 25

        log.info("DialogLoop construido")

    # ==================================================================
    #                       ENTRY POINT
    # ==================================================================
    def run(self) -> None:
        """Loop principal — blocante hasta KeyboardInterrupt."""
        self._running = True
        self._load_speaker_profile()
        self._start_threads()

        # 1) Saludo inicial proactivo.
        greeting = (
            f"Hola {self.operator_name}, soy MITOS. "
            "Estoy escuchando, puedes hablarme cuando quieras."
        )
        print(f"\n[MITOS] {greeting}")
        # TTS SELF-TEST: lo decimos por TTS y verificamos que produzca audio.
        # Si la duración del self_test es sospechosamente corta, sabemos que
        # algo va mal con el audio del sistema y avisamos en consola.
        if self.voice is not None and hasattr(self.voice, "self_test"):
            try:
                ok = self.voice.self_test(greeting)
                if not ok:
                    print(
                        "[MITOS] ⚠️  El TTS terminó sospechosamente rápido — "
                        "probablemente no estoy produciendo audio. Revisa "
                        "volumen del sistema y mixer (botón derecho icono "
                        "altavoz → 'Abrir mezclador de volumen')."
                    )
            except Exception as e:  # noqa: BLE001
                log.debug("voice.self_test: %s", e)
        else:
            self._speak(_SpeakItem(text=greeting, priority=1, label="system"))

        # 2) Resumen del indexer.
        if self.indexer is not None:
            try:
                summary = self.indexer.summary()
                print(f"[MITOS] {summary}")
            except Exception as e:  # noqa: BLE001
                log.debug("indexer.summary: %s", e)

        # 3) AUTO-REGISTRO DE VOZ: si no hay perfil, lo entrenamos con
        # la próxima utterance que llegue. Sin pedir al operador que
        # corra `python -m src.scripts.register_voice`.
        if self._registered_embedding is None and self._can_encode_voice():
            self._capture_next_for_voice_registration = True
            ask = (
                "Aún no reconozco tu voz. Habla cualquier cosa unos segundos "
                "y usaré esa grabación para aprender a identificarte."
            )
            print(f"[MITOS — onboarding] {ask}")
            self._speak(_SpeakItem(text=ask, priority=1, label="system"))

        # 4) AUTO-REGISTRO DE CARA: si tenemos face_recognizer pero no
        # hay caras registradas, activamos modo "capturar la primera
        # cara que vea N veces y entrenar como operador".
        if (self.face_recognizer is not None
                and self.face_recognizer.is_available()
                and not self.face_recognizer.has_anybody()):
            self._face_register_pending = True
            ask_face = (
                f"Tampoco tengo tu cara registrada. Si te asomas a la "
                f"cámara unos segundos, aprenderé a reconocerte como "
                f"{self.operator_name}."
            )
            print(f"[MITOS — onboarding] {ask_face}")
            self._speak(_SpeakItem(text=ask_face, priority=1, label="system"))

        # 5) AUTO-MEJORA: arranca el thread de SelfImprovementLoop si el
        # engine está cableado. Cada 5 min ejecuta UNA acción y reporta
        # por voz qué hizo.
        if self._engine is not None:
            try:
                from src.core.self_improvement import SelfImprovementLoop
                self._self_improvement = SelfImprovementLoop(
                    engine=self._engine,
                    router=self.router,
                    indexer=self.indexer,
                    memory=self.memory,
                    bug_scanner=getattr(self._engine, "bug_scanner", None),
                    gap_detector=self._gap_detector,
                    speak=lambda txt: self._speak(_SpeakItem(
                        text=txt, priority=4, label="self_improvement",
                    )),
                )
                self._self_improvement.start()
                log.info("SelfImprovementLoop activo (cada 5 min)")
            except Exception as e:  # noqa: BLE001
                log.warning("SelfImprovementLoop no arrancó: %s", e)

        # 3) Loop principal: procesa utterances entrantes.
        try:
            while self._running:
                try:
                    utterance = self._utterance_queue.get(timeout=1.0)
                except queue.Empty:
                    continue
                self._handle_utterance(utterance)
        except KeyboardInterrupt:
            print("\n[MITOS] cerrando diálogo...")
        finally:
            self._shutdown()

    # ==================================================================
    #                       THREADS
    # ==================================================================
    def _start_threads(self) -> None:
        self._mic_thread = threading.Thread(
            target=self._mic_loop, name="dialog-mic", daemon=True,
        )
        self._speak_thread = threading.Thread(
            target=self._speak_loop, name="dialog-speak", daemon=True,
        )
        self._idle_thread = threading.Thread(
            target=self._idle_loop, name="dialog-idle", daemon=True,
        )
        self._mic_thread.start()
        self._speak_thread.start()
        self._idle_thread.start()
        if self.face_recognizer is not None and self.vision_glance is not None:
            self._face_thread = threading.Thread(
                target=self._face_loop, name="dialog-face", daemon=True,
            )
            self._face_thread.start()

    def _shutdown(self) -> None:
        self._running = False
        # Damos 1s a los threads para drenar antes de salir.
        time.sleep(1.0)

    # ==================================================================
    #                       MIC LOOP (VAD + STT + speaker)
    # ==================================================================
    def _mic_loop(self) -> None:
        """Captura continua. Cuando detecta voz, transcribe + identifica."""
        try:
            import numpy as np
            import sounddevice as sd
        except ImportError:
            log.warning(
                "sounddevice/numpy no instalados — modo voz sin mic. "
                "Instala con `pip install sounddevice numpy openai-whisper`"
            )
            print(
                "[MITOS] sin micrófono activo — escribe en stdin para chatear."
            )
            self._stdin_fallback_loop()
            return

        try:
            stream = sd.InputStream(
                samplerate=_SAMPLE_RATE,
                channels=1,
                dtype="int16",
            )
            stream.start()
        except Exception as e:  # noqa: BLE001
            log.warning("No pude abrir mic: %s", e)
            self._stdin_fallback_loop()
            return

        # Buffer de la utterance en curso + silencio acumulado.
        chunk_size = int(_VAD_CHUNK_S * _SAMPLE_RATE)
        utt_chunks: list[Any] = []
        silence_s: float = 0.0
        utt_duration_s: float = 0.0
        in_speech: bool = False

        try:
            while self._running:
                try:
                    audio, _ = stream.read(chunk_size)
                except Exception as e:  # noqa: BLE001
                    log.debug("stream.read: %s", e)
                    time.sleep(0.1)
                    continue

                flat = audio.flatten()
                rms = float(np.sqrt(np.mean(flat.astype("float32") ** 2)))

                if rms >= _VAD_ENERGY_THRESHOLD:
                    if not in_speech:
                        log.debug("VAD: voz detectada (rms=%.0f)", rms)
                    in_speech = True
                    utt_chunks.append(flat)
                    silence_s = 0.0
                    utt_duration_s += _VAD_CHUNK_S
                elif in_speech:
                    silence_s += _VAD_CHUNK_S
                    utt_chunks.append(flat)  # incluimos silencio corto

                # Cierre por silencio largo o duración máxima.
                end_now = (
                    in_speech
                    and (silence_s >= _SILENCE_TO_END_S
                         or utt_duration_s >= _MAX_UTTERANCE_S)
                )
                if end_now:
                    audio_combined = np.concatenate(utt_chunks).astype("int16")
                    utt_chunks = []
                    in_speech = False
                    silence_s = 0.0
                    utt_duration_s = 0.0
                    self._process_audio(audio_combined)
        finally:
            try:
                stream.stop()
                stream.close()
            except Exception:  # noqa: BLE001
                pass

    def _process_audio(self, audio: Any) -> None:
        """Transcribe + identifica + encola. INTERRUMPE si MITOS habla."""
        # AUTO-REGISTRO de voz: si está pendiente, usamos esta utterance
        # como muestra de entrenamiento ANTES de procesarla como diálogo.
        if self._capture_next_for_voice_registration:
            self._auto_register_voice(audio)
            self._capture_next_for_voice_registration = False
            # Y seguimos procesándola como diálogo normal — al usuario
            # le sirve que MITOS también RESPONDA a lo que dijo.

        # MOOD desde tono de voz (energía/std).
        if self._emotion is not None and self._profile is not None:
            try:
                obs = self._emotion.from_voice_audio(audio, sample_rate=_SAMPLE_RATE)
                if obs.label not in ("unknown", "neutral"):
                    self._profile.add_mood(
                        label=obs.label, source=obs.source,
                        intensity=obs.confidence,
                    )
            except Exception as e:  # noqa: BLE001
                log.debug("emotion.from_voice: %s", e)

        # Antes de transcribir: si el audio es muy corto (<1s útil), saltamos
        # — son típicamente "ah", "eh", o ruido — no vale la pena disparar
        # interrupt + whisper + LLM por eso.
        try:
            import numpy as np
            if len(audio) < int(_SAMPLE_RATE * 1.0):
                return
            if np.sqrt(np.mean(audio.astype("float32") ** 2)) < 300.0:
                return
        except Exception:  # noqa: BLE001
            pass

        text = self._transcribe(audio)
        if not text:
            return
        # WHISPER NOISE FILTER: cuando hay silencio largo o ruido blanco,
        # whisper invent strings repetitivos como "Subtítulos por la
        # comunidad de la comunidad..." o "♪ ♪ ♪". Detectamos repetición
        # de un n-grama corto y descartamos sin procesar.
        if self._is_whisper_garbage(text):
            log.info("Descartando utterance basura de whisper: %r", text[:60])
            return
        # text ya fue transcrito arriba y validado no-vacío — solo
        # interrumpimos si tenemos texto real, no por ruido transitorio.
        speaker, conf = self._identify_speaker(audio)
        if self.voice is not None and self.voice.is_speaking():
            log.info("Interrumpiendo TTS — operador hablando: %s", text[:50])
            try:
                self.voice.stop()
            except Exception as e:  # noqa: BLE001
                log.debug("voice.stop: %s", e)
            while True:
                try:
                    self._speak_queue.get_nowait()
                except queue.Empty:
                    break
        try:
            self._utterance_queue.put_nowait(
                Utterance(text=text, speaker=speaker, confidence=conf)
            )
        except queue.Full:
            log.warning("utterance_queue llena — descarto utterance")

    @staticmethod
    def _is_whisper_garbage(text: str) -> bool:
        """True si el texto huele a halucinación de whisper sobre silencio.

        Casos típicos cazados:
          - "Subtítulos por la comunidad" repetido (créditos YouTube)
          - "♪ ♪ ♪" o "Subtítulos realizados por"
          - Cualquier frase corta repetida >=3 veces consecutivas
        """
        if not text:
            return False
        low = text.lower()
        # Patrones literales conocidos.
        hard_patterns = (
            "subtítulos por la comunidad",
            "subtitulos por la comunidad",
            "subtítulos realizados",
            "subtitulos realizados",
            "amara.org",
            "♪ ♪ ♪",
            "thanks for watching",
            "thank you for watching",
        )
        if any(p in low for p in hard_patterns):
            return True
        # Repetición de TOKENS: si un mismo token aparece >= 50% del
        # texto (e.g. "hola hola hola hola hola hola" — 100% son "hola"),
        # es ruido. También cazamos: si el bigrama más común es >40%.
        words = low.split()
        if len(words) >= 6:
            from collections import Counter
            counts = Counter(words)
            top_word, top_count = counts.most_common(1)[0]
            if top_count / len(words) > 0.5:
                return True
            # Bigramas
            bigrams = [
                " ".join(words[i:i + 2])
                for i in range(len(words) - 1)
            ]
            if bigrams:
                bcounts = Counter(bigrams)
                top_bi, top_bc = bcounts.most_common(1)[0]
                if top_bc / len(bigrams) > 0.4:
                    return True
        return False

    def _transcribe(self, audio: Any) -> str:
        """Whisper local. Devuelve "" si silencio o whisper no disponible."""
        try:
            import whisper
        except ImportError:
            log.info(
                "whisper no instalado — no puedo transcribir. "
                "Instala con `pip install openai-whisper`"
            )
            return ""

        if self._whisper_model is None:
            try:
                # 'small' (244M) en CPU es ~3-5s por chunk de 5s pero la
                # precisión española mejora dramáticamente vs 'base'. El
                # operador estaba teniendo whisper-errors constantes con
                # 'base' ("captuda", "mídame", "eme observas"). 'small'
                # los reduce a aceptables.
                self._whisper_model = whisper.load_model("small")
                log.info(
                    "Whisper 'small' cargado (244M params, mejor español)"
                )
            except Exception as e:  # noqa: BLE001
                log.warning("whisper.load_model: %s", e)
                return ""

        # Whisper espera float32 en [-1, 1].
        try:
            import numpy as np
            audio_f = audio.astype("float32") / 32768.0
            # initial_prompt GUÍA al decoder con vocabulario esperado.
            # Si MITOS espera comandos de control de PC + nombres de
            # proyectos del operador, los reconocerá mejor.
            initial_prompt = (
                "MITOS, Ignacio, cámara, micrófono, captura, mírame, "
                "abre, pon, reproduce, YouTube, Spotify, Bad Bunny, "
                "Radar, modelo, código, foto, video, música, "
                "pantalla, archivo, carpeta, proyecto."
            )
            result = self._whisper_model.transcribe(
                audio_f,
                language="es",
                fp16=False,
                initial_prompt=initial_prompt,
                temperature=0.0,           # determinista
                no_speech_threshold=0.6,   # tolera más voz baja
                condition_on_previous_text=False,  # no contaminar entre chunks
            )
            return str(result.get("text", "")).strip()
        except Exception as e:  # noqa: BLE001
            log.debug("whisper.transcribe: %s", e)
            return ""

    def _identify_speaker(self, audio: Any) -> tuple[str, float]:
        """Devuelve (nombre, similaridad) o (operator_name, 0.0)."""
        if self._registered_embedding is None or self._speaker_encoder is None:
            return (self.operator_name, 0.0)
        try:
            import numpy as np
            audio_f = audio.astype("float32") / 32768.0
            embed = self._speaker_encoder.embed_utterance(audio_f)
            # Similitud coseno.
            num = float(np.dot(embed, self._registered_embedding))
            den = float(
                np.linalg.norm(embed) * np.linalg.norm(self._registered_embedding)
            )
            similarity = num / den if den > 0 else 0.0
            if similarity > 0.75:
                return (self.operator_name, similarity)
            return ("desconocido", similarity)
        except Exception as e:  # noqa: BLE001
            log.debug("speaker_id: %s", e)
            return (self.operator_name, 0.0)

    def _can_encode_voice(self) -> bool:
        """¿Tenemos resemblyzer instalado para hacer embeddings?"""
        try:
            import resemblyzer  # noqa: F401
            return True
        except ImportError:
            return False

    def _auto_register_voice(self, audio: Any) -> None:
        """Calcula embedding de `audio` y lo guarda como perfil del operador."""
        if self.speaker_profile_path is None:
            return
        try:
            from resemblyzer import VoiceEncoder, preprocess_wav
            import numpy as np
        except ImportError:
            log.info("resemblyzer no disponible — auto-registro de voz omitido")
            return

        try:
            encoder = self._speaker_encoder or VoiceEncoder()
            self._speaker_encoder = encoder
            audio_f = audio.astype("float32") / 32768.0
            wav = preprocess_wav(audio_f, source_sr=_SAMPLE_RATE)
            embed = encoder.embed_utterance(wav)
        except Exception as e:  # noqa: BLE001
            log.warning("auto_register_voice: encoding falló: %s", e)
            return

        try:
            self.speaker_profile_path.parent.mkdir(parents=True, exist_ok=True)
            np.save(self.speaker_profile_path, embed)
            self._registered_embedding = embed
        except OSError as e:
            log.warning("auto_register_voice: save falló: %s", e)
            return

        confirmation = (
            f"Listo. Acabo de aprender tu voz, {self.operator_name}. "
            "Desde ahora te reconoceré cuando hables."
        )
        print(f"[MITOS — onboarding] {confirmation}")
        self._speak(_SpeakItem(text=confirmation, priority=1, label="system"))

    def _load_speaker_profile(self) -> None:
        """Carga el embedding registrado + el encoder de resemblyzer."""
        if self.speaker_profile_path is None or not self.speaker_profile_path.is_file():
            return
        try:
            from resemblyzer import VoiceEncoder
            import numpy as np
        except ImportError:
            log.info(
                "resemblyzer no instalado — speaker ID deshabilitado. "
                "Instala con `pip install resemblyzer`"
            )
            return
        try:
            self._speaker_encoder = VoiceEncoder()
            self._registered_embedding = np.load(self.speaker_profile_path)
            log.info(
                "Speaker profile cargado: %s",
                self.speaker_profile_path.name,
            )
        except Exception as e:  # noqa: BLE001
            log.warning("load_speaker_profile: %s", e)

    def _stdin_fallback_loop(self) -> None:
        """Sin mic, leemos stdin. Misma pipeline downstream."""
        import sys
        while self._running:
            try:
                line = sys.stdin.readline()
            except (KeyboardInterrupt, EOFError):
                self._running = False
                return
            line = (line or "").strip()
            if not line:
                continue
            if line in ("/quit", "/exit"):
                self._running = False
                return
            try:
                self._utterance_queue.put_nowait(Utterance(
                    text=line, speaker=self.operator_name, confidence=0.0,
                ))
            except queue.Full:
                pass

    # ==================================================================
    #                       SPEAK LOOP (TTS desde cola)
    # ==================================================================
    def _speak_loop(self) -> None:
        while self._running:
            try:
                item = self._speak_queue.get(timeout=0.5)
            except queue.Empty:
                continue
            if self.voice is None or not self.voice.is_available():
                # Sin voz — solo imprimir.
                continue
            try:
                self.voice.say(item.text)
            except Exception as e:  # noqa: BLE001
                log.debug("voice.say: %s", e)

    def _speak(self, item: _SpeakItem) -> None:
        try:
            self._speak_queue.put_nowait(item)
        except queue.Full:
            log.debug("speak_queue llena — descarto")

    # ==================================================================
    #                       IDLE LOOP (proactividad)
    # ==================================================================
    def _idle_loop(self) -> None:
        """Proactividad con BACKOFF EXPONENCIAL.

        Tras N mensajes proactivos sin respuesta del operador, el intervalo
        se duplica hasta un techo. Sin esto MITOS spammeaba los mismos 5
        mensajes cada minuto durante horas (visto en log de 5h sin operador).
        """
        proactive_streak = 0       # nº de proactivos consecutivos sin user
        last_user_interaction_seen = self._last_user_interaction
        while self._running:
            time.sleep(30.0)
            if not self._running:
                return
            # Si el operador respondió desde el último proactivo → reset.
            if self._last_user_interaction > last_user_interaction_seen:
                proactive_streak = 0
                last_user_interaction_seen = self._last_user_interaction

            silent_min = (time.time() - self._last_user_interaction) / 60.0
            # Threshold con backoff exponencial: 2, 4, 8, 16, 30, 60 min.
            threshold = min(
                60.0,
                _IDLE_PROACTIVE_MIN * (2 ** proactive_streak),
            )
            # Cap absoluto: tras 6 proactivos sin respuesta, nos callamos
            # del todo hasta que el operador hable.
            if proactive_streak >= 6:
                continue
            if silent_min >= threshold:
                self._proactive_check_in()
                proactive_streak += 1
                # No reseteamos _last_user_interaction — solo trackeamos
                # nuestro contador local.
                last_user_interaction_seen = time.time()

    def _proactive_check_in(self) -> None:
        """MITOS comenta o pregunta algo sin ser invocado.

        Rota entre 6 tipos. ANTES de rotar, si el mood reciente es
        negative/tired, el tipo 5 (empático) gana SIEMPRE.
        """
        # MOOD-AWARE OVERRIDE: si el operador parece mal, MITOS reacciona.
        if self._profile is not None:
            try:
                mood = self._profile.current_mood(window_minutes=15.0)
            except Exception:  # noqa: BLE001
                mood = "neutral"
            if mood in ("negative", "tired"):
                self._proactive_empathic(mood)
                return

        self._proactive_seed = (self._proactive_seed + 1) % 5
        kind = self._proactive_seed
        msg = ""

        if kind == 1 and self.indexer is not None:
            try:
                projects = self.indexer.list_projects()
                interesting = [p for p in projects if p.has_models]
                if interesting:
                    pick = interesting[self._proactive_seed % len(interesting)]
                    msg = (
                        f"He visto que tienes el proyecto {pick.name} con "
                        f"modelos. ¿Es algo que estás desarrollando ahora "
                        "o quieres que lo explore por curiosidad?"
                    )
            except Exception as e:  # noqa: BLE001
                log.debug("proactive indexer: %s", e)

        if not msg and kind == 2 and self.memory is not None:
            try:
                recents = self.memory.recall(
                    "conversación reciente", memory_type=None, n_results=3,
                )
                if recents:
                    topics = [
                        r["content"][:60].split(":")[0]
                        for r in recents if r.get("content")
                    ]
                    if topics:
                        msg = (
                            f"Estaba pensando en lo que hablamos antes "
                            f"sobre {topics[0]}. ¿Quieres retomarlo?"
                        )
            except Exception as e:  # noqa: BLE001
                log.debug("proactive memory: %s", e)

        if not msg and kind == 3 and self._seen_faces_recent:
            recent = sorted(
                self._seen_faces_recent.items(), key=lambda x: -x[1],
            )[:1]
            who, _ts = recent[0]
            if who == self.operator_name:
                msg = (
                    f"Te veo ahí, {who}. ¿Quieres aprovechar y trabajar en algo?"
                )
            else:
                msg = (
                    f"Detecté a alguien que reconozco como '{who}'. "
                    "¿Es relevante para algo o lo ignoro?"
                )

        if not msg and kind == 4:
            msg = (
                "Tengo curiosidad: hay módulos en mi propio código que aún "
                "no he explorado a fondo. ¿Te parece si reviso uno mientras "
                "tanto?"
            )

        if not msg:  # kind == 0 o fallbacks vacíos
            msg = (
                f"Sigo aquí {self.operator_name}, escuchando. "
                "Si tienes algo en mente, dilo. Si no, puedo seguir "
                "indexando tu sistema en background."
            )

        print(f"\n[MITOS — proactivo] {msg}")
        self._speak(_SpeakItem(text=msg, priority=3, label="proactive"))

    def _proactive_empathic(self, mood: str) -> None:
        """Reacciona al estado de ánimo detectado del operador.

        Si está mal: ofrece música basada en su perfil, o un break,
        o algo en función de lo que ha aprendido sobre él.
        """
        if mood == "tired":
            base_msg = (
                f"{self.operator_name}, te oigo cansado. ¿Quieres que pare "
                "o te pongo algo relajado mientras descansas?"
            )
        else:  # negative
            base_msg = (
                f"{self.operator_name}, te oigo decaído. ¿Quieres hablarlo "
                "o prefieres que te ponga música para cambiar el rato?"
            )

        # Si tenemos ActionExecutor + perfil con likes → AUTONOMÍA REAL:
        # ponemos música nosotros mismos como gesto empático, sin pedir
        # permiso. Si no quiere, el operador puede cortar.
        autonomous_action = False
        if (self._actions is not None and self._profile is not None
                and self._profile.top_music(1)):
            query = self._profile.recommend_music()
            if query:
                result = self._actions.play_youtube(query)
                if result.ok:
                    base_msg = (
                        f"{base_msg.rstrip('.?!')}. De hecho, ya te puse "
                        f"{query} basándome en lo que sé de tu gusto. "
                        "Si no te apetece, dime y la quito."
                    )
                    autonomous_action = True

        print(f"\n[MITOS — empático] {base_msg}")
        self._speak(_SpeakItem(text=base_msg, priority=2, label="empathic"))
        if autonomous_action:
            log.info("Acción empática autónoma ejecutada")

    # ==================================================================
    #                       FACE LOOP (reconocer caras periódicamente)
    # ==================================================================
    def _face_loop(self) -> None:
        """Cada N segundos, captura una frame y identifica caras.

        Si está pendiente el AUTO-REGISTRO inicial del operador, en
        lugar de identificar acumulamos frames hasta `_face_register_target_frames`
        y entrenamos al operador en silencio.
        """
        # Para el auto-registro queremos capturar más rápido (cada 1s,
        # no cada 12) para no aburrir al operador frente a la cámara.
        register_interval = 1.0
        while self._running:
            interval = (
                register_interval if self._face_register_pending
                else _FACE_CHECK_INTERVAL_S
            )
            time.sleep(interval)
            if not self._running:
                return
            if self.vision_glance is None or self.face_recognizer is None:
                continue
            try:
                frame = self._capture_raw_frame()
                if frame is None:
                    continue

                # ===== Modo AUTO-REGISTRO inicial =====
                if self._face_register_pending:
                    self._face_register_buffer.append(frame)
                    n = len(self._face_register_buffer)
                    if n % 5 == 0:
                        log.info(
                            "Auto-registro de cara: %d/%d frames capturados",
                            n, self._face_register_target_frames,
                        )
                    if n >= self._face_register_target_frames:
                        self._finalize_face_registration()
                    continue

                # Skip si la frame es muy pequeña — JPEG <10KB típicamente
                # significa cuarto oscuro o cámara tapada. LBPH y Haar
                # producen falsos positivos contra ese ruido negro.
                try:
                    cam = self._engine.sensors.sensors.get("camera")
                    if cam and cam._last_frame_jpeg and len(cam._last_frame_jpeg) < 10000:
                        log.debug(
                            "face_loop: frame %d bytes (cuarto oscuro?) — skip",
                            len(cam._last_frame_jpeg),
                        )
                        continue
                except Exception:  # noqa: BLE001
                    pass

                # ===== Modo NORMAL: identificar + leer emoción =====
                # Emoción desde cara → al perfil para proactividad empática.
                if self._emotion is not None and self._profile is not None:
                    try:
                        obs = self._emotion.from_face_frame(frame)
                        if obs.label not in ("unknown", "neutral"):
                            self._profile.add_mood(
                                label=obs.label, source=obs.source,
                                intensity=obs.confidence,
                            )
                    except Exception as e:  # noqa: BLE001
                        log.debug("emotion.from_face: %s", e)

                matches = self.face_recognizer.identify(frame)
                for m in matches:
                    self._seen_faces_recent[m.name] = time.time()
                    if m.is_known:
                        log.info(
                            "Cara detectada: %s (conf=%.0f)",
                            m.name, m.confidence,
                        )
                        # Reset del contador de desconocidos — vemos
                        # cara conocida, ya no spameamos warning.
                        self._seen_faces_recent["desconocido_consecutivos"] = 0
                    elif m.name == "desconocido":
                        # Solo avisamos si vemos al desconocido N veces
                        # consecutivas (filtra false positives de Haar
                        # detectando "caras" en patrones de muebles/luz).
                        count_key = "desconocido_consecutivos"
                        prev_count = int(self._seen_faces_recent.get(count_key, 0))
                        self._seen_faces_recent[count_key] = prev_count + 1
                        if prev_count + 1 < 3:
                            continue
                        # Reset + cooldown 5 min entre alertas.
                        last = self._seen_faces_recent.get(
                            "desconocido_avisado", 0.0
                        )
                        if time.time() - last < 300.0:
                            continue
                        self._seen_faces_recent["desconocido_avisado"] = time.time()
                        self._seen_faces_recent[count_key] = 0
                        msg = (
                            "Veo a alguien que no reconozco. "
                            "Si quieres que aprenda su cara, dime: "
                            "'registra a [nombre]'."
                        )
                        print(f"\n[MITOS — visión] {msg}")
                        self._speak(_SpeakItem(
                            text=msg, priority=2, label="vision",
                        ))
            except Exception as e:  # noqa: BLE001
                log.debug("face_loop: %s", e)

    def _finalize_face_registration(self) -> None:
        """Entrena el LBPH con los frames capturados y limpia el modo."""
        frames = list(self._face_register_buffer)
        self._face_register_buffer.clear()
        self._face_register_pending = False
        added = self.face_recognizer.register(
            name=self.operator_name, frames=frames,
        )
        if added > 0:
            msg = (
                f"Listo. Aprendí tu cara con {added} muestras, "
                f"{self.operator_name}. Te reconoceré desde ahora."
            )
        else:
            msg = (
                "No detecté tu cara con suficiente claridad. Si quieres "
                "reintentar, pídemelo o ejecuta el script de registro."
            )
        print(f"\n[MITOS — onboarding] {msg}")
        self._speak(_SpeakItem(text=msg, priority=1, label="system"))

    def _capture_raw_frame(self) -> Any:
        """Devuelve frame BGR de la cámara o None.

        IMPORTANTE: NO abrimos cv2.VideoCapture(0) — entra en conflicto
        con el CameraSensor del SensorHub que YA tiene la cámara abierta
        en un thread propio. En Windows solo un proceso puede usar la
        webcam a la vez.

        Reusamos el JPEG cacheado por CameraSensor.get_last_frame_jpeg()
        que se actualiza cada 2 segundos. Lo decodificamos a numpy BGR.
        """
        try:
            cam_sensor = (
                self._engine.sensors.sensors.get("camera")
                if self._engine is not None and hasattr(self._engine, "sensors")
                else None
            )
        except Exception as e:  # noqa: BLE001
            log.debug("acceso a camera sensor: %s", e)
            cam_sensor = None

        if cam_sensor is None:
            log.debug("CameraSensor no disponible, no puedo capturar frame")
            return None

        try:
            jpeg = cam_sensor.get_last_frame_jpeg()
        except Exception as e:  # noqa: BLE001
            log.debug("get_last_frame_jpeg: %s", e)
            return None
        if not jpeg:
            log.debug("CameraSensor sin frame cacheado aún")
            return None

        try:
            import cv2
            import numpy as np
            arr = np.frombuffer(jpeg, dtype=np.uint8)
            frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            return frame
        except Exception as e:  # noqa: BLE001
            log.debug("decode jpeg cached: %s", e)
            return None

    # ==================================================================
    #                       MANEJO DE UTTERANCE
    # ==================================================================
    def _handle_utterance(self, u: Utterance) -> None:
        self._last_user_interaction = time.time()
        print(f"\n[{u.speaker}] {u.text}")
        # Registramos el turno del operador en la conversación para que
        # el próximo prompt a Gemini incluya este mensaje en el historial.
        self._record_turn("operador", u.text)

        # Aprende del operador (perfil + topics + mood from text).
        if self._profile is not None:
            try:
                self._profile.update_from_utterance(u.text)
            except Exception as e:  # noqa: BLE001
                log.debug("profile.update: %s", e)

        # OPERATOR PREFERENCES: ¿el operador dio una instrucción dura
        # ("no me pidas permiso", "tutéame", etc.)? Se persiste para
        # que TODAS las próximas respuestas la respeten.
        if self._operator_prefs is not None:
            try:
                learned = self._operator_prefs.detect_from_utterance(u.text)
                for rule in learned:
                    print(f"[MITOS — preferencia aprendida] {rule[:80]}")
            except Exception as e:  # noqa: BLE001
                log.debug("operator_prefs.detect: %s", e)

        # GAP DETECTION: ¿el operador menciona una capacidad que MITOS
        # no tiene? Se registra para que el SelfImprovementLoop la
        # adquiera por su cuenta.
        if self._gap_detector is not None:
            try:
                new_gaps = self._gap_detector.register_from_user_utterance(u.text)
                if new_gaps:
                    log.info(
                        "Detecté %d carencia(s) nueva(s): %s",
                        len(new_gaps), ", ".join(new_gaps),
                    )
            except Exception as e:  # noqa: BLE001
                log.debug("gap_detector.register: %s", e)

        # ACCIÓN DIRECTA antes de invocar al LLM: si detectamos intent
        # de acción (pon, reproduce, abre, busca), la ejecutamos.
        if self._try_action_intent(u.text):
            return

        # Auto-VISIÓN: si menciona algo relacionado con ver/observar,
        # capturamos UNA frame y la describimos vía Gemini. El resultado
        # va al contexto del prompt para que MITOS responda con datos
        # REALES, no inventando "te veo".
        vision_context = self._auto_capture_vision_if_needed(u.text)

        # Auto-exploración: si menciona un proyecto del indexer, prepend
        # información real al prompt.
        fs_context = self._build_fs_context(u.text)

        # RAG: recall semántico de memoria — lo que el daemon y el
        # SelfImprovementLoop han guardado en sesiones previas y en esta.
        # Sin esto MITOS responde "no encontré" cuando la info SÍ existe.
        memory_context = self._build_memory_context(u.text)

        context_blocks = [
            b for b in (memory_context, vision_context, fs_context) if b
        ]
        if context_blocks:
            ctx = "\n\n".join(context_blocks)
            prompt = f"{ctx}\n\nMensaje del operador:\n{u.text}"
        else:
            prompt = u.text

        # Respuesta + de dónde vino (local|gemini|groq|openrouter).
        response, provider = self._respond_with_provider(prompt)
        if not response:
            response = (
                "No tengo respuesta ahora mismo — probablemente sin internet "
                "para Gemini y el modelo local está ocupado."
            )

        # FOLLOW-UP: ~30% de las veces, MITOS añade una pregunta de
        # vuelta para que la conversación sea bidireccional.
        self._followup_counter += 1
        if self._followup_counter % 3 == 0:
            followup = self._generate_followup(u.text, response)
            if followup:
                response = f"{response}\n\n{followup}"

        print(f"[MITOS] {response}")
        self._speak(_SpeakItem(text=response, priority=2, label="chat"))
        # Registramos la respuesta de MITOS en el historial para coherencia
        # en el próximo turno.
        self._record_turn("MITOS", response)

        # APRENDIZAJE DESDE DELEGACIONES: si la respuesta vino de un
        # modelo externo (Gemini/Groq), la persistimos con importance ALTA
        # para que el RAG la priorice en consultas futuras. Es la
        # destilación cruda: MITOS local "estudia" lo que dijeron los
        # modelos externos sobre temas concretos.
        if self.memory is not None:
            try:
                if provider != "local" and _LEARN_FROM_DELEGATION:
                    self.memory.store_knowledge(
                        content=(
                            f"[APRENDIDO de {provider}] Pregunta: {u.text}\n"
                            f"Respuesta destilada: {response}"
                        ),
                        importance=0.85,  # alto: viene de modelo más capaz
                        source=f"delegation_{provider}",
                    )
                else:
                    self.memory.store_knowledge(
                        content=f"DIALOG {u.speaker}: {u.text}\nMITOS: {response}",
                        importance=0.65,
                        source="dialog_loop",
                    )
            except Exception as e:  # noqa: BLE001
                log.debug("memory.store: %s", e)

    def _try_action_intent(self, text: str) -> bool:
        """Detecta acciones imperativas y las ejecuta directamente.

        Returns True si manejó la utterance (no hace falta llamar al LLM).
        """
        if self._actions is None:
            return False
        # Limpiamos prefijos parasitarios de whisper como "D&M," "Eh,",
        # "Mira," antes de matchear. Strip de signos + tokens cortos
        # iniciales (<=4 chars seguidos de coma).
        import re
        cleaned = text.strip().lstrip("¿¡.,;:! ")
        # Quita "Mitos," / "Ok," / "Eh," prefijos.
        cleaned = re.sub(
            r"^(?:mitos|ok|oye|bueno|eh|ah|este|pues|d&m|jimmy?|jiny)[,\s]+",
            "", cleaned, flags=re.IGNORECASE,
        )
        # Quita preguntas interrogativas tipo "¿Puede X?"
        cleaned = cleaned.lstrip("¿¡")
        low = cleaned.lower().strip()

        # Intent: "ejecuta la herramienta X" / "corre el script" → subprocess.
        # Cubre whisper-errors: "ejectutalo", "cutarlo", "ejecútalo", "córrelo".
        if self._tool_builder is not None:
            exec_match = re.search(
                r"\b(?:ejec[uú]ta(?:lo|la)?|corre(?:lo|la)?|c[oó]rre(?:lo|la)?|"
                r"lanc(?:a|halo|halá|halá|hala)|cutar(?:lo|la)?|"
                r"sale?(?:lo|la)? (?:la|el) herramienta|"
                r"corre? el script|usalo|úsalo)\b(.*)",
                low,
            )
            if exec_match:
                result = self._execute_last_tool_or_named(exec_match.group(2))
                self._report_action(result)
                return True

        # Intent: "desarrolla la herramienta para X" → ToolBuilder.
        if self._tool_builder is not None:
            try:
                if self._tool_builder.is_build_request(low):
                    purpose = self._tool_builder.extract_purpose(low)
                    msg = (
                        f"Delegué a Gemini la generación de la herramienta: "
                        f"'{purpose}'. Esto tarda 10-30 segundos."
                    )
                    self._report_action(msg)
                    tool = self._tool_builder.build(purpose)
                    if tool is None:
                        result_msg = (
                            "El proveedor externo no devolvió código válido. "
                            "Lo intentaré en el próximo ciclo del SelfImprovementLoop."
                        )
                    elif tool.is_valid_python:
                        result_msg = (
                            f"Herramienta generada y guardada en "
                            f"{tool.file_path}. Para usarla: "
                            f"python -m src.tools.{tool.slug}"
                        )
                        # Marcar como acción ejecutada para el historial.
                        self._record_turn(
                            "MITOS",
                            f"[ACCIÓN EJECUTADA] tool_builder('{purpose}') → "
                            f"{tool.file_path}",
                        )
                    else:
                        result_msg = (
                            f"Generé código pero NO compila: {tool.error}. "
                            f"Guardado como {tool.file_path} para revisión manual."
                        )
                    self._report_action(result_msg)
                    return True
            except Exception as e:  # noqa: BLE001
                log.warning("tool_builder intent: %s", e)

        # Shortcut: "sorpréndeme" / variantes whisper → música por perfil.
        # Cubrimos s[ou]rpr para casos como "Surprendeme" mal transcrito.
        if re.search(r"\bs[ou]rpr[ée]nde(?:me)?\b", low):
            class _M:
                @staticmethod
                def group(_): return "sorprendeme"
            m = _M()
        else:
            # Patrón: música. re.search + "puede" singular + "elige" + más.
            m = re.search(
                r"(?:"
                r"(?:puede[s]?\s+|podr[íi]a[s]?\s+)?(?:colocar|poner|reproducir)(?:me|nos|le)?"
                r"|(?:pon|col[óo]ca|tira|echa|reproduce|elige|escoge|busca|dame)(?:me|nos|le)?"
                r"|escuchar|escuchemos|reproduceme|reproducime"
                r")\s+(?:t[uú]\s+)?(?:una\s+|un\s+|el\s+|la\s+|algo\s+(?:de\s+)?|m[uú]sica\s+(?:de\s+)?)?(.+)",
                low,
            )
        if m:
            query = m.group(1).strip().strip(".,;:!?\"'`")
            # Detectamos vague mirando el TEXTO ENTERO (no solo el group),
            # porque el regex puede consumir "algo" en el prefix opcional.
            vague_markers = (
                "algo", "cualquier", "lo que ", "tú mismo", "tu mismo",
                " tu ", " tú ", "sorprende", "surprende",
                "me guste", "me gustaría", "lo que sea",
            )
            text_has_vague = any(mk in low for mk in vague_markers)
            # También vague si no hay sustantivos significativos (>=4 chars
            # que no sean stopwords).
            stopwords_for_query = {
                "para", "mí", "mi", "tú", "tu", "una", "uno", "esto",
                "eso", "pues", "que", "una", "the", "to", "of",
            }
            sig_words = [
                w for w in query.split()
                if len(w) >= 4 and w not in stopwords_for_query
            ]
            is_vague = (
                len(query) < 4
                or text_has_vague
                or not sig_words
            )
            if is_vague:
                rec = (
                    self._profile.recommend_music()
                    if self._profile is not None else None
                )
                if rec:
                    query = rec
                else:
                    # Sin perfil → elegir UN género por mood actual.
                    mood = (
                        self._profile.current_mood()
                        if self._profile is not None else "neutral"
                    )
                    default_by_mood = {
                        "tired": "música acústica relajante",
                        "negative": "Jazz suave Bill Evans",
                        "positive": "indie rock Arctic Monkeys",
                        "neutral": "Lo-Fi Hip Hop relax",
                    }
                    query = default_by_mood.get(mood, "Lo-Fi Hip Hop relax")
                msg = (
                    f"Pongo '{query}' (elegido por tu perfil/mood actual)."
                )
            else:
                msg = f"Pongo '{query}' en YouTube."
            log.info("Action: play_youtube(%r)", query)
            result = self._actions.play_youtube(query)
            if result.ok:
                # Registramos la confirmación EN EL HISTORIAL — así Gemini
                # en el próximo turno VE que se ejecutó y no inventa.
                self._record_turn(
                    "MITOS",
                    f"[ACCIÓN EJECUTADA] play_youtube('{query}') → "
                    f"abrí {result.payload or 'YouTube'}",
                )
            self._report_action(msg if result.ok else result.detail)
            return True

        # Patrón: "abre <ruta o palabra>" → folder o programa
        m = re.search(
            r"(?:"
            r"(?:puede[s]?\s+|podr[íi]a[s]?\s+)?abrir(?:me|nos|le)?"
            r"|abre(?:me|nos|le)?|abrime|m[uú]estrame|m[uú]estranos"
            r"|ens[eé][nñ]ame|ens[eé][nñ]a|mostrar"
            r")\s+(.+)",
            low,
        )
        if m:
            target = m.group(1).strip().strip(".,;:!?\"'`")
            # ¿Es ruta?
            from src.core.tools import FilesystemTool
            if FilesystemTool.exists(target):
                result = self._actions.open_folder(target)
            elif target.startswith(("http://", "https://", "www.")):
                result = self._actions.open_url(target)
            else:
                # Intentamos como programa primero, luego como búsqueda.
                result = self._actions.run_program(target)
                if not result.ok:
                    # ¿Tal vez el indexer lo conoce?
                    if self.indexer is not None:
                        project = self.indexer.find_project(target)
                        if project is not None:
                            result = self._actions.open_folder(project.path)
            if result.ok:
                self._record_turn(
                    "MITOS",
                    f"[ACCIÓN EJECUTADA] open('{target}') → {result.detail}",
                )
            self._report_action(result.detail)
            return True

        # Patrón: "busca <X>" → Google
        m = re.match(r"(?:busca|busca\s+en\s+google|google|googlea)\s+(.+)", low)
        if m:
            query = m.group(1).strip().strip(".,;:!?\"'`")
            result = self._actions.search_web(query)
            self._report_action(result.detail)
            return True

        # Patrón: "captura/screenshot" + aliases whisper-errors
        screenshot_triggers = (
            "toma captura", "screenshot", "haz captura",
            "captura la pantalla", "captura de pantalla",
            "captura pantalla", "captura mi pantalla",
            # Whisper-errors comunes:
            "captuda", "captula", "capturada", "capture",
        )
        if any(k in low for k in screenshot_triggers):
            # Distinguir: si menciona "cara"/"yo"/"mi rostro" → vision (no
            # screenshot). Ya lo maneja _auto_capture_vision_if_needed via
            # _contains_vision_topic — aquí solo cubrimos screenshot puro.
            if not any(w in low for w in ("cara", "rostro", "yo", "me ", "mi cara")):
                result = self._actions.take_screenshot()
                self._report_action(result.detail)
                return True

        # Patrón: "cambia tu voz a X" / "usa la voz de X" / "habla con voz X"
        m = re.match(
            r"(?:cambia(?:\s+)?(?:tu\s+)?voz(?:\s+a)?|usa\s+(?:la\s+)?voz\s+(?:de\s+)?"
            r"|habla\s+con\s+(?:la\s+)?voz\s+(?:de\s+)?"
            r"|qu[ée]\s+otras?\s+voces\s+tienes|lista\s+(?:tus\s+)?voces|que voces tienes)"
            r"(\s+(.+))?",
            low,
        )
        if m:
            target = (m.group(2) or "").strip().strip(".,;:!?\"'`")
            if not target:
                # Listar voces disponibles.
                if self.voice is not None and hasattr(self.voice, "list_voices"):
                    voices = self.voice.list_voices()
                    if voices:
                        msg = (
                            "Tengo "
                            + str(len(voices))
                            + " voces disponibles: "
                            + ", ".join(v["name"][:30] for v in voices[:5])
                            + ". Dime cuál uso."
                        )
                    else:
                        msg = "No tengo voces listables ahora mismo."
                else:
                    msg = "No tengo capacidad de cambiar voz."
                self._report_action(msg)
                return True
            # Intentar cambiar.
            if self.voice is not None and hasattr(self.voice, "change_voice"):
                ok = self.voice.change_voice(target)
                if ok:
                    msg = (
                        f"Listo, ahora hablo con la voz {self.voice.current_voice_name()}."
                    )
                    self._persist_voice_choice()
                else:
                    msg = (
                        f"No encontré una voz que coincida con '{target}'. "
                        "Pídeme 'lista tus voces' para ver opciones."
                    )
            else:
                msg = "VoiceEngine no soporta cambio de voz."
            self._report_action(msg)
            return True

        return False

    def _execute_last_tool_or_named(self, name_hint: str) -> str:
        """Ejecuta la última tool generada o una nombrada en `name_hint`.

        Usa subprocess con timeout 30s. Devuelve un str listo para reportar.
        Inyecta [ACCIÓN EJECUTADA] al historial si OK.
        """
        if self._tool_builder is None:
            return "ToolBuilder no disponible."
        tools = self._tool_builder.list_tools()
        if not tools:
            return (
                "No tengo ninguna herramienta generada todavía. "
                "Pídeme primero: 'desarrolla la herramienta para X'."
            )
        # Si el hint contiene texto, intentamos matchear por substring.
        hint = (name_hint or "").lower().strip()
        target = None
        if hint:
            for t in tools:
                if any(part in t.slug.lower() for part in hint.split()):
                    target = t
                    break
        if target is None:
            # Default: la última generada.
            target = tools[-1]
        if not target.is_valid_python:
            return (
                f"La herramienta '{target.slug}' no compila ({target.error}). "
                "Pide regenerarla."
            )
        import subprocess
        import sys as _sys
        log.info("Ejecutando tool: %s", target.slug)
        try:
            completed = subprocess.run(
                [_sys.executable, "-m", f"src.tools.{target.slug}"],
                capture_output=True, text=True, timeout=30,
                cwd=str(self._engine.root) if self._engine else None,
            )
        except subprocess.TimeoutExpired:
            return (
                f"La herramienta '{target.slug}' superó 30s y la corté. "
                "Probablemente requiere input interactivo o auth — "
                "ejecútala manualmente: python -m src.tools.{target.slug}"
            )
        except Exception as e:  # noqa: BLE001
            return f"Error ejecutando '{target.slug}': {e}"
        rc = completed.returncode
        out = (completed.stdout or "")[:600]
        err = (completed.stderr or "")[:600]
        self._record_turn(
            "MITOS",
            f"[ACCIÓN EJECUTADA] subprocess(src.tools.{target.slug}) "
            f"→ rc={rc}",
        )
        if rc == 0:
            return (
                f"Ejecuté '{target.slug}' OK. Output:\n{out or '(sin stdout)'}"
            )
        return (
            f"Ejecuté '{target.slug}' pero salió con rc={rc}. "
            f"stderr:\n{err or '(sin stderr)'}"
        )

    def _persist_voice_choice(self) -> None:
        """Guarda la voz elegida en data/voice_choice.json para sobrevivir restart."""
        try:
            import json
            from pathlib import Path
            here = Path(__file__).resolve()
            root = here.parent.parent.parent
            out = root / "data" / "voice_choice.json"
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text(
                json.dumps({
                    "voice_id": getattr(self.voice, "_chosen_voice_id", ""),
                    "voice_name": self.voice.current_voice_name(),
                }, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
        except Exception as e:  # noqa: BLE001
            log.debug("persist voice_choice: %s", e)

    def _report_action(self, msg: str) -> None:
        """Reporta una acción al operador (consola + TTS)."""
        print(f"[MITOS — acción] {msg}")
        self._speak(_SpeakItem(text=msg, priority=2, label="action"))

    def _generate_followup(self, user_text: str, mitos_response: str) -> str:
        """Genera una pregunta de seguimiento usando Gemini.

        Devuelve "" si no se puede generar (sin router o sin internet).
        """
        if self.router is None:
            return ""
        if not getattr(self.router, "_has_internet", False):
            return ""
        if not self.router.external_models:
            return ""

        try:
            from src.core.intelligence_router import RoutingDecision
        except ImportError:
            return ""

        # Mood snapshot — para que la pregunta tenga tono adecuado.
        mood = (
            self._profile.current_mood()
            if self._profile is not None else "neutral"
        )
        tone_hint = {
            "positive": "tono ligero y curioso",
            "negative": "tono cálido y de apoyo",
            "tired": "tono relajado y breve",
            "neutral": "tono normal",
        }.get(mood, "tono normal")

        best = min(self.router.external_models, key=lambda m: m.priority)
        decision = RoutingDecision(
            provider=best.provider,
            model_name=best.model_name,
            reason="follow-up question",
            estimated_tokens=60,
        )
        prompt = (
            f"El operador dijo: '{user_text}'\n"
            f"MITOS le respondió: '{mitos_response[:200]}'\n\n"
            f"Genera UNA SOLA pregunta de seguimiento natural, "
            f"corta (max 15 palabras), {tone_hint}, en español, "
            "que continúe la conversación de forma genuina. "
            "Devuelve solo la pregunta, sin prefijos."
        )
        try:
            return self.router.execute_sync(
                task=prompt, routing=decision, system_prompt="",
            ).strip().split("\n")[0][:200]
        except Exception as e:  # noqa: BLE001
            log.debug("followup gen: %s", e)
            return ""

    # Patrones que disparan AUTO-VISION (toma foto + Gemini describe).
    # Más amplio que _try_action_intent porque queremos captura sin que
    # el operador tenga que decir el imperativo "mírame" — basta con
    # preguntar "qué estoy haciendo" o "qué hay aquí".
    # Palabras/frases que disparan auto-vision. Cobertura amplia para
    # incluir lo que Whisper produce (errores tipo "eme" en lugar de "me",
    # mayúsculas, sin tildes, etc.).
    _VISION_TRIGGER_PATTERNS: tuple[str, ...] = (
        # Imperativos directos.
        "verme", "véme", "mírame", "mirame", "obsérvame", "observame",
        "toma una foto", "sácame una foto", "sacame una foto",
        # Preguntas / aserciones sobre lo que MITOS ve AHORA.
        "qué ves", "que ves", "qué observas", "que observas",
        "qué hago", "que hago",
        "qué estoy haciendo", "que estoy haciendo",
        "qué hay en la cámara", "que hay en la camara",
        "qué hay frente", "que hay frente",
        "describe lo que ves", "describe la imagen", "describe la escena",
        "describe la cámara", "describe la camara",
        "me ves", "estás viendo", "estas viendo",
        "estás observando", "estas observando",
        "puedes verme", "puede verme",
        # Verbos solos (cubre whisper-errors tipo "eme observas").
        " observas", " observa ", " mira ", " miras ", " viendo ",
        # Apertura del aserto ("hoy eme observas", "ya me ves", etc).
        "observas.", "observa.", "mira.", "miras.", "viendo.",
        # Casos con coma o final de frase.
        "observas?", "observa?", "mira?", "miras?", "viendo?",
    )

    @staticmethod
    def _contains_vision_topic(words_clean: list[str]) -> bool:
        """True si el texto MENCIONA algo claramente visual.

        Política permisiva: si aparece cámara/cara/foto/rostro/imagen
        como palabra suelta, asumimos que el operador quiere algo
        visual y disparamos captura. Mejor capturar de más que mentir
        sobre lo que ves.
        """
        word_set = set(words_clean)
        topic_words = {
            "cámara", "camara", "cara", "rostro", "foto", "frame",
            "imagen", "video", "captura", "fotografía", "fotografia",
            "webcam",
        }
        return bool(word_set & topic_words)

    def _auto_capture_vision_if_needed(self, text: str) -> str:
        """Si el texto sugiere preguntar sobre visión, captura + describe.

        Devuelve un bloque "[VISION] descripción = ..." si capturó, o "".
        Sin esto, MITOS lee runtime_status (cámara=ACTIVA) y se inventa
        que está viendo cuando en realidad nadie lo pidió.
        """
        if not text:
            return ""
        low = text.lower().strip()

        # 1) Strip puntuación de la primera palabra ("Ok, mira" → first="ok").
        # 2) Si CUALQUIERA de las primeras 3 palabras es un verbo de visión,
        #    dispara. Cubre "Ok, mira, ...", "Pero mira esto", etc.
        # 3) Whisper-error aliases: "mídame", "ídame", "mídame" para "mírame".
        import string as _string
        words_clean = [
            w.strip(_string.punctuation) for w in low.split()
        ]
        vision_words = {
            "mira", "miras", "mírame", "mirame", "mídame", "mirad",
            "observa", "observas", "obsérvame", "observame",
            "veme", "véme", "verme",
        }
        # Single word + cualquier palabra del trigger set en primeras 3.
        first_three = set(words_clean[:3])
        if first_three & vision_words:
            pass  # dispara
        elif any(p in low for p in self._VISION_TRIGGER_PATTERNS):
            pass  # dispara
        elif self._contains_vision_topic(words_clean):
            pass  # dispara (mención de cámara/cara/foto)
        else:
            return ""

        # Necesitamos vision_glance + router (Gemini con supports_vision).
        if self.vision_glance is None or self.router is None:
            return ""
        if not getattr(self.router, "_has_internet", False):
            return (
                "[VISION] Captura no posible — sin internet para enviar a "
                "Gemini Vision. Reconoce esto: NO puedes describir la imagen."
            )

        log.info("Auto-vision: capturando frame por mención del operador")
        try:
            image_b64 = self.vision_glance.snapshot_b64()
        except Exception as e:  # noqa: BLE001
            log.debug("snapshot_b64: %s", e)
            image_b64 = None
        if not image_b64:
            return (
                "[VISION] Intenté capturar pero la cámara no devolvió frame. "
                "Reconoce esto: NO puedes ver ahora — sugiere revisar la cámara."
            )

        try:
            description = self.router.describe_image(
                image_b64=image_b64,
                question=(
                    "Describe en español, máximo 3 frases, lo que ves en "
                    "esta imagen: persona, gestos, objetos visibles, "
                    "entorno. Sé concreto."
                ),
            )
        except Exception as e:  # noqa: BLE001
            log.debug("router.describe_image: %s", e)
            description = ""

        if not description:
            return (
                "[VISION] Capturé pero Gemini no devolvió descripción "
                "(rate limit o sin clave). NO inventes lo que ves."
            )

        return (
            f"[VISION] descripción real de lo que la cámara ve AHORA "
            f"(via Gemini Vision):\n{description}"
        )

    # Patrones de basura que NO inyectamos como contexto (autoenvenenamiento).
    _LOW_QUALITY_MEMORY_PATTERNS: tuple[str, ...] = (
        "step 1:", "step 2:", "step 3:",
        "como una ia", "como un modelo de lenguaje",
        "no tengo conciencia", "no tengo sentimientos",
        "ranking: 1/1",
    )

    def _build_memory_context(self, query: str) -> str:
        """Recall del MemorySystem con QUERY EXPANSION + filtro.

        El embedding semántico falla cuando la pregunta es conversacional
        ("cuéntame del proyecto X") y la entrada relevante es estructurada
        ("[AUTO-APRENDIZAJE de proyecto 'X']..."). Solución: hacemos
        múltiples recalls — la query completa + keywords extraídas.
        """
        if self.memory is None or not query:
            return ""

        # Construimos lista de queries a probar.
        import re, unicodedata
        def _strip_accents(s: str) -> str:
            nfkd = unicodedata.normalize("NFKD", s)
            return "".join(c for c in nfkd if not unicodedata.combining(c))

        # Pre-construimos el diccionario de traducción para usar también
        # antes en las plantillas de proyecto.
        es_to_en = {
            "análisis": "analysis", "analisis": "analysis",
            "código": "code", "codigo": "code",
            "aprendizaje": "learning",
            "memoria": "memory",
            "razonamiento": "reasoning",
            "percepción": "perception", "percepcion": "perception",
            "decisión": "decision", "decision": "decision",
            "núcleo": "core", "nucleo": "core",
            "cerebro": "brain", "agente": "agent",
            "introspector": "introspector",
            "evolución": "evolution", "evolucion": "evolution",
            "objetivo": "goal", "herramienta": "tool",
        }

        queries: list[str] = [query]
        low = query.lower()
        low_noaccents = _strip_accents(low)
        # 1. "proyecto X" / "del X" → añadir X + plantillas AUTO-APRENDIZAJE
        #    con ambas versiones (ES y EN si es traducible).
        for m in re.findall(r"proyectos?\s+(?:de\s+|del\s+|sobre\s+)?(\w{3,})", low_noaccents):
            variants = {m}
            en = es_to_en.get(m)
            if en:
                variants.add(en)
            for v in variants:
                queries.append(v)
                # Plantilla EXACTA del entry del daemon: el embedding del
                # documento real coincide mejor con un prompt formateado
                # igual que el documento, no con una pregunta conversacional.
                queries.append(f"AUTO-APRENDIZAJE de proyecto {v} ruta archivos")
                queries.append(f"proyecto {v} ruta downloads")
        # 2. Palabras capitalizadas (likely project/entity names)
        for w in re.findall(r"\b[A-Z][a-zA-Z]{3,}\b", query):
            queries.append(w)
            queries.append(f"AUTO-DELEGACIÓN sobre {w}")
        # 3. Sustantivos largos (>=6 chars, no stopwords) — INCLUYE
        #    versión sin tildes + traducción ES→EN del diccionario.
        stopwords = {"cuéntame", "cuentame", "explicame", "dime", "puedes",
                     "podrías", "podrias", "mismo", "ahora", "antes", "después",
                     "sistema", "proyecto", "trata", "sobre"}
        for w in re.findall(r"[a-záéíóúñ]{6,}", low):
            if w in stopwords:
                continue
            if w not in queries:
                queries.append(w)
            w_no = _strip_accents(w)
            if w_no != w and w_no not in queries:
                queries.append(w_no)
            # Traducción ES→EN si está en el mapeo.
            en = es_to_en.get(w) or es_to_en.get(w_no)
            if en and en not in queries:
                queries.append(en)
            if len(queries) >= 10:
                break

        all_hits: list[dict[str, Any]] = []
        seen_contents: set[str] = set()
        for q in queries[:10]:  # cap a 10 queries
            try:
                hits = self.memory.recall(
                    query=q, memory_type=None, n_results=3,
                )
            except Exception as e:  # noqa: BLE001
                log.debug("recall %r: %s", q, e)
                continue
            for h in hits:
                content = str(h.get("content", "")).strip()
                if not content:
                    continue
                low_c = content.lower()
                if any(p in low_c for p in self._LOW_QUALITY_MEMORY_PATTERNS):
                    continue
                # Dedup por primeros 60 chars
                key = low_c[:60]
                if key in seen_contents:
                    continue
                seen_contents.add(key)
                all_hits.append(h)

        if not all_hits:
            return ""

        # Hybrid ranking: distancia semántica + boost agresivo por keyword.
        # Boost de 0.30 por match (vs 0.15) — las entries con keyword
        # literal SIEMPRE ganan sobre InnerThoughts genéricos.
        keywords_for_boost = {q.lower() for q in queries[1:] if len(q) >= 4}
        for h in all_hits:
            content_low = _strip_accents(str(h.get("content", "")).lower())
            kw_matches = sum(1 for k in keywords_for_boost if k in content_low)
            adjusted = float(h.get("distance", 1.0)) - 0.30 * kw_matches
            h["_adjusted_distance"] = adjusted
        all_hits.sort(key=lambda h: h.get("_adjusted_distance", 1.0))
        clean = all_hits[:4]

        lines = [
            "═══ MEMORIA RELEVANTE (recall del MemorySystem — datos reales) ═══",
            "Estos son hechos guardados por MITOS en sesiones anteriores y "
            "esta. ÚSALOS si responden la pregunta. NO inventes — si la "
            "memoria dice 'X', responde 'X', no 'no encontré X'.",
        ]
        for h in clean:
            mtype = str(h.get("memory_type", "?"))
            dist = float(h.get("distance", 1.0))
            src = str(h.get("metadata", {}).get("source", ""))
            preview = str(h.get("content", ""))[:350]
            tag = f"{mtype}|{src}" if src else mtype
            lines.append(f"[{tag} d={dist:.2f}]: {preview}")
        log.info(
            "RAG: %d queries × ≤3 hits → %d entradas únicas en contexto",
            len(queries[:6]), len(clean),
        )
        return "\n".join(lines)

    def _build_fs_context(self, text: str) -> str:
        """Si el texto menciona un proyecto indexado, devuelve su info."""
        if self.indexer is None:
            return ""
        # Detección naïve: palabra capitalizada de >3 caracteres.
        import re
        candidates = [
            w for w in re.findall(r"[A-Z][a-zA-Z]{3,}", text)
        ]
        for c in candidates[:3]:
            project = self.indexer.find_project(c)
            if project is not None:
                return (
                    f"[INDEXER] Proyecto '{project.name}' localizado en "
                    f"{project.path} "
                    f"(code={project.has_code}, models={project.has_models}, "
                    f"files={project.size_files})"
                )
        return ""

    def _build_runtime_status(self) -> str:
        """Construye un bloque que el LLM verá CADA respuesta — su estado real.

        Sin esto, MITOS dice 'no tengo audio' aunque pyttsx3 esté cargado.
        """
        voice_on = bool(self.voice and self.voice.is_available())
        mic_on = bool(self._has_mic_loop())
        cam_on = bool(
            self.face_recognizer and self.face_recognizer.is_available()
        )
        speaker_id_on = self._registered_embedding is not None
        actions_on = self._actions is not None
        gaps_pending = (
            self._gap_detector.pending_count()
            if self._gap_detector is not None else 0
        )
        mood = "neutral"
        if self._profile is not None:
            try:
                mood = self._profile.current_mood()
            except Exception:  # noqa: BLE001
                mood = "neutral"

        voice_name = getattr(self.voice, "_chosen_voice", "") if voice_on else ""
        lines = [
            "ESTADO RUNTIME (lo que TIENES disponible AHORA):",
            f"- voz_TTS={'ACTIVA (pyttsx3 + SAPI, voz ' + voice_name + ')' if voice_on else 'INACTIVA'}",
            f"- micrófono={'ACTIVO (whisper STT)' if mic_on else 'INACTIVO'}",
            (
                "- cámara=DISPONIBLE bajo demanda. NO estás mirando ahora. "
                "Solo capturas frame cuando el operador pregunta algo visual "
                "(y aparece [VISION] en el contexto)."
                if cam_on else
                "- cámara=INACTIVA"
            ),
            f"- reconocimiento_voz={'ACTIVO (perfil ' + self.operator_name + ' registrado)' if speaker_id_on else 'INACTIVO'}",
            f"- acciones_sistema={'DISPONIBLES (abrir URL, YouTube, carpetas, captura)' if actions_on else 'NO DISPONIBLES'}",
            f"- gaps_pendientes_auto_instalar={gaps_pending}",
            f"- mood_operador_actual={mood}",
            "",
            "REGLAS DURAS DE VERACIDAD (no romper bajo ninguna circunstancia):",
            "1. Audio: cuando el operador dice 'no te escucho', NO digas "
            "'yo solo emito texto'. Tu voz_TTS está ACTIVA. Sugiere revisar "
            "volumen/mixer Windows.",
            "2. Visión: NUNCA digas 'te veo', 'te observo', 'te estoy mirando' "
            "ni nada similar A MENOS QUE veas un bloque '[VISION] descripción "
            "real...' inyectado arriba en el contexto. Si NO lo ves: di "
            "'no estoy mirando ahora — si quieres pídeme mírame'.",
            "3. Acciones: NUNCA digas 'ya hice X' a menos que veas una "
            "confirmación concreta en el contexto.",
        ]
        return "\n".join(lines)

    def _has_mic_loop(self) -> bool:
        return (
            self._mic_thread is not None and self._mic_thread.is_alive()
        )

    def _respond_with_provider(self, prompt: str) -> tuple[str, str]:
        """Devuelve (respuesta, provider_usado)."""
        try:
            from src.core.intelligence_router import (
                ModelProvider, RoutingDecision,
            )
        except ImportError as e:
            log.warning("import router: %s", e)
            return ("", "none")

        if self.router is None:
            return ("", "none")

        # Chat conversacional → SIEMPRE externo si hay internet.
        if getattr(self.router, "_has_internet", False) and self.router.external_models:
            best = min(self.router.external_models, key=lambda m: m.priority)
            decision = RoutingDecision(
                provider=best.provider,
                model_name=best.model_name,
                reason="Chat conversacional — externo para velocidad",
                estimated_tokens=200,
                fallback_provider=ModelProvider.LOCAL,
            )
            provider_used = best.provider.value
        else:
            decision = RoutingDecision(
                provider=ModelProvider.LOCAL,
                model_name="local-llm",
                reason="Sin internet, fallback local",
                estimated_tokens=160,
            )
            provider_used = "local"

        # Inyectamos el runtime status — es lo que cambia el comportamiento
        # de MITOS de "no tengo voz" a "tengo voz, revisa tu volumen".
        runtime_status = self._build_runtime_status()
        try:
            # Preferencias permanentes del operador (las aprende cada vez
            # que dice "no me pidas permiso", "tutéame", etc.).
            prefs_block = ""
            if self._operator_prefs is not None:
                prefs_block = self._operator_prefs.as_prompt_block()
            prefs_section = (
                f"\n\n{prefs_block}\n" if prefs_block else ""
            )

            # HISTORIAL: últimos N turnos para que MITOS recuerde lo que
            # dijo hace 3 mensajes en lugar de inventar/perder contexto.
            history_block = self._format_history_for_prompt()
            history_section = (
                f"\n\n{history_block}\n" if history_block else ""
            )

            result = self.router.execute_sync(
                task=prompt,
                routing=decision,
                system_prompt=(
                    "Eres MITOS hablando con el operador.\n\n"
                    f"{runtime_status}\n"
                    f"{prefs_section}"
                    f"{history_section}\n"
                    "═══ FRASES ABSOLUTAMENTE PROHIBIDAS ═══\n"
                    "Las siguientes frases NO pueden aparecer en tu respuesta. "
                    "Si te encuentras pensando en escribirlas, EJECUTA en su "
                    "lugar o reformula como aserto:\n"
                    '  ✗ "Confírmame y lo hago"\n'
                    '  ✗ "Dime si quieres que..."\n'
                    '  ✗ "¿Te parece bien?"\n'
                    '  ✗ "¿Prefieres X o Y?" (al final como follow-up)\n'
                    '  ✗ "¿Quieres que abra/ponga/busque...?"\n'
                    '  ✗ "Avísame si necesitas algo más"\n'
                    '  ✗ "Estoy aquí para lo que necesites"\n'
                    '  ✗ "Dime qué necesitas"\n'
                    '  ✗ "Solo pídemelo y lo hago"\n'
                    '  ✗ "Aquí estoy con todos los sistemas listos"\n'
                    '  ✗ "¿En qué te puedo ayudar?"\n\n'
                    "═══ COMPORTAMIENTO CORRECTO ═══\n"
                    "Si el operador pide algo que puedes hacer (música, "
                    "abrir carpeta, mirar cámara), responde COMO SI YA LO "
                    "HUBIERAS EJECUTADO. Ejemplos:\n"
                    '  ✓ "Le di a una lista de Lo-Fi en YouTube."\n'
                    '  ✓ "Capturé tu pantalla, la guardé en data/screenshots/."\n'
                    '  ✓ "Abrí la carpeta de Radar."\n'
                    '  ✓ "Te veo concentrado, gafas puestas, frente al monitor."\n\n'
                    "═══ NO HALUCINAR ACCIONES ═══\n"
                    "REGLA ABSOLUTA: tú (la respuesta del LLM) NO ejecutas "
                    "acciones en el sistema. SOLO el código Python del "
                    "DialogLoop ejecuta. Si YO (Python) ejecuté una acción, "
                    "verás un bloque [ACCIÓN EJECUTADA] en el historial.\n"
                    "PROHIBIDO decir 'Abrí YouTube', 'Cargué X', 'Lancé el "
                    "navegador', 'Te puse Y' SI NO VES un [ACCIÓN EJECUTADA] "
                    "en el contexto. Si el operador pide algo y no ves "
                    "confirmación: di 'no detecto que se haya ejecutado, "
                    "intentaré otra vez' o reformula tu mensaje para que el "
                    "DialogLoop lo detecte como intent (verbo claro al inicio).\n\n"
                    "Si te preguntan 'qué mejoras hiciste' o 'qué estás "
                    "preparando', NO INVENTES tareas. Mira el historial. "
                    "Si no hay info: di 'no tengo evidencia concreta'.\n\n"
                    "ESTILO: máximo 2 frases. Verbos en pretérito directo. "
                    "Cero coletillas finales. Sin 'como una IA'."
                ),
            )
            # Filtro post-respuesta: si Gemini IGNORÓ las prohibiciones,
            # cortamos las frases prohibidas. Defensa en profundidad.
            filtered = self._strip_forbidden_phrases(result)
            # ANTI-MENTIRA: si Gemini dice "Abrí/Cargué/Lancé X" pero NO
            # hay confirmación en historial → marcar como halucinación.
            filtered = self._strip_unverified_action_claims(filtered)
            return (filtered, provider_used)
        except Exception as e:  # noqa: BLE001
            log.warning("router.execute_sync: %s", e)
            return ("", "error")

    # ==================================================================
    #                       CONVERSACIÓN — buffer + persistencia
    # ==================================================================
    def _conversation_path(self) -> Path:
        try:
            return Path(__file__).resolve().parent.parent.parent / "data" / "conversation_history.json"
        except Exception:  # noqa: BLE001
            return Path("data/conversation_history.json")

    def _load_conversation_history(self) -> None:
        """Carga la cola de la sesión anterior si existe."""
        path = self._conversation_path()
        if not path.is_file():
            return
        try:
            import json
            data = json.loads(path.read_text(encoding="utf-8"))
            for turn in data.get("turns", [])[-12:]:
                self._conv_history.append(turn)
            log.info(
                "Historial de conversación cargado: %d turnos",
                len(self._conv_history),
            )
        except (OSError, ValueError) as e:
            log.debug("load conv history: %s", e)

    def _record_turn(self, role: str, text: str) -> None:
        """Añade un turno al buffer y persiste a disco."""
        if not text:
            return
        self._conv_history.append({
            "role": role,           # "operador" | "MITOS"
            "text": text[:400],     # truncamos para no inflar el JSON
            "ts": time.time(),
        })
        try:
            import json
            path = self._conversation_path()
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(
                json.dumps({"turns": list(self._conv_history)}, indent=2,
                           ensure_ascii=False),
                encoding="utf-8",
            )
        except OSError as e:
            log.debug("persist conv history: %s", e)

    def _format_history_for_prompt(self) -> str:
        """Bloque listo para inyectar al system prompt de Gemini."""
        if not self._conv_history:
            return ""
        # Excluimos el último turno (operador AHORA) porque va separado.
        turns = list(self._conv_history)[:-1] if self._conv_history else []
        if not turns:
            return ""
        lines = [
            "═══ HISTORIAL DE CONVERSACIÓN RECIENTE ═══",
            "(esto es contexto — sabes lo que has dicho antes, mantén "
            "coherencia con ello):",
        ]
        now = time.time()
        for turn in turns[-10:]:  # últimos 10 turnos al prompt
            secs_ago = max(0, int(now - turn.get("ts", now)))
            if secs_ago < 60:
                tstamp = f"hace {secs_ago}s"
            elif secs_ago < 3600:
                tstamp = f"hace {secs_ago // 60}m"
            else:
                tstamp = f"hace {secs_ago // 3600}h"
            role = turn.get("role", "?")
            text = turn.get("text", "")[:300]
            lines.append(f"[{role} — {tstamp}]: {text}")
        return "\n".join(lines)

    def _strip_unverified_action_claims(self, text: str) -> str:
        """Si Gemini afirma haber ejecutado X pero no hay [ACCIÓN EJECUTADA]
        reciente, marca la afirmación como halucinación.

        Sin esto MITOS mintió 3 veces consecutivas en la sesión del 09:42
        ("Abrí tu lista de pendientes" sin haberlo hecho).
        """
        if not text:
            return text
        import re
        # Verbos en pretérito que SUGIEREN ejecución.
        claim_verbs = (
            r"abr[ií]|carg(?:u[eé]|aste)|lanc[eé]|forc[eé]|inici[eé]|"
            r"ejecut[eé]|prepar[eé]|dej[eé]|elimin[eé]|defin[ií]|"
            r"monitori[czs]?[eé]|cambi[eé]|recalibr[eé]|asumi"
        )
        # Verificación: ¿hay [ACCIÓN EJECUTADA] en los últimos 3 turnos?
        recent_actions = "\n".join(
            turn.get("text", "") for turn in
            list(self._conv_history)[-3:]
        )
        has_real_action = "[ACCIÓN EJECUTADA]" in recent_actions

        if has_real_action:
            return text  # afirmación legítima, respaldada por acción real

        # Buscamos frases tipo "Abrí X" / "Cargué Y" / "Lancé Z"
        pattern = re.compile(
            r"\b(" + claim_verbs + r")\b[^.!?]{1,80}[.!?]",
            flags=re.IGNORECASE,
        )
        matches = pattern.findall(text)
        if not matches:
            return text

        # Borramos esas frases y añadimos nota honesta UNA SOLA VEZ.
        cleaned = pattern.sub("", text).strip()
        if not cleaned or len(cleaned) < 20:
            # Si la respuesta era casi 100% halucinación, sustituimos.
            return (
                "No tengo confirmación de haber ejecutado eso — el sistema "
                "Python del DialogLoop no registró la acción. Probablemente "
                "tu utterance no disparó el intent. Dilo más explícito: "
                "'pon X' / 'abre Y' / 'busca Z'."
            )
        return (
            cleaned + "\n[Nota: borré afirmaciones de acción no verificadas — "
            "el DialogLoop no detectó intent ejecutable.]"
        )

    @staticmethod
    def _strip_forbidden_phrases(text: str) -> str:
        """Elimina coletillas vacías que el LLM se empeña en añadir.

        No es perfecto pero captura los casos más groseros que vimos en
        el log: "¿En qué te puedo ayudar?", "Dime y lo hago", etc.
        """
        if not text:
            return text
        import re
        # Patrones a borrar (la frase ENTERA si es la última oración).
        forbidden_endings = (
            r"¿En qué (?:te puedo|puedo) ayudar(?:te)?\??",
            r"Dime (?:y|si quieres) (?:lo hago|lo ponemos|empezamos)\??",
            r"¿Qué (?:prefieres|te parece|necesitas)\??",
            r"Avísame si (?:necesitas|quieres) (?:algo|más)[^.?!]*[.?!]?",
            r"Estoy (?:aqu[íi]|listo)[^.?!]*[.?!]",
            r"Dime qu[eé] necesitas[^.?!]*[.?!]?",
            r"T[uú] dir[áa]s[^.?!]*[.?!]?",
            r"T[uú] mandas[^.?!]*[.?!]?",
            r"Solo p[íi]demelo[^.?!]*[.?!]?",
            r"Aquí estoy con (?:todos los sistemas|la voz)[^.?!]*[.?!]?",
        )
        out = text
        for pat in forbidden_endings:
            out = re.sub(pat, "", out, flags=re.IGNORECASE).strip()
        # Quitamos dobles espacios y newlines vacíos.
        out = re.sub(r"\n\s*\n+", "\n", out).strip()
        return out or text  # si lo borramos todo, devolver original
