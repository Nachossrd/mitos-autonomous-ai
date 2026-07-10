"""
==============================================================================
 Proyecto MITOS - VoiceEngine (Fase 8+ — TTS local)
==============================================================================

Le da voz a MITOS por los altavoces del sistema. Usa pyttsx3 (SAPI5 en
Windows, NSSpeechSynthesizer en macOS, eSpeak en Linux). Si pyttsx3
falla u opera en silencio, intenta fallback nativo:

  - Windows: subprocess PowerShell → System.Speech.Synthesis
  - macOS:   subprocess `say` (binario nativo)
  - Linux:   subprocess `espeak` o `spd-say`

Diseño:
  - El engine de pyttsx3 se crea EN EL THREAD QUE LO USA — SAPI5 es COM,
    y COM en Windows requiere `pythoncom.CoInitialize()` por thread.
    Sin esto, llamar `runAndWait` desde un thread daemon devuelve
    silenciosamente (no falla, simplemente no produce audio). Esa fue
    la causa de "no te escucho" en la sesión 21:19.
  - `say()` es BLOCANTE. Si quieres no bloquear, usa una cola externa.
  - `self_test()` produce una frase conocida y reporta si el engine
    completó sin error — útil para diagnosticar al inicio.

Convenciones:
  - Logger `mitos.voice_engine`.
==============================================================================
"""

from __future__ import annotations

import logging
import platform
import subprocess
import threading
import time
from typing import Any

log = logging.getLogger("mitos.voice_engine")


_SPANISH_VOICE_HINTS: tuple[str, ...] = (
    "spanish", "español", "espanol", "es-es", "es_es", "es-mx", "es_mx",
    "helena", "sabina", "pablo", "monica", "paulina", "jorge",
)

_DEFAULT_RATE: int = 175
_DEFAULT_VOLUME: float = 0.95


class VoiceEngine:
    """TTS local con fallback nativo cuando pyttsx3 falla."""

    def __init__(
        self,
        rate: int = _DEFAULT_RATE,
        volume: float = _DEFAULT_VOLUME,
    ) -> None:
        self._rate: int = int(rate)
        self._volume: float = float(volume)
        self._system: str = platform.system()
        self._available_pyttsx3: bool = False
        self._available_native: bool = False
        self._chosen_voice: str = ""
        self._lock: threading.Lock = threading.Lock()
        # Engine por-thread — clave es threading.get_ident().
        # Crítico para Windows + SAPI5 + COM.
        self._engines_by_thread: dict[int, Any] = {}
        self._engines_lock: threading.Lock = threading.Lock()
        self._com_initialized: set[int] = set()

        self._probe_capabilities()

    # ==================================================================
    #                       DISCOVERY
    # ==================================================================
    def _probe_capabilities(self) -> None:
        """Detecta qué backends TTS están disponibles."""
        # pyttsx3.
        try:
            import pyttsx3
            self._available_pyttsx3 = True
            # Probamos init aquí solo para listar voces y elegir una.
            try:
                tmp = pyttsx3.init()
                voices = tmp.getProperty("voices") or []
                chosen = self._pick_spanish_voice(voices)
                if chosen is not None:
                    self._chosen_voice = getattr(chosen, "name", chosen.id)
                    self._chosen_voice_id: str = chosen.id
                    log.info("Voz seleccionada: %s", self._chosen_voice)
                else:
                    log.info(
                        "No encontré voz española; usaré la voz por defecto"
                    )
                    self._chosen_voice_id = ""
                # Soltamos el engine temporal — los reales se crean por thread.
                tmp.stop()
            except Exception as e:  # noqa: BLE001
                log.debug("probe init pyttsx3: %s", e)
                self._chosen_voice_id = ""
        except ImportError:
            self._chosen_voice_id = ""
            log.info("pyttsx3 no instalado")

        # Native fallback.
        if self._system == "Windows":
            # PowerShell + System.Speech siempre disponible en Win10+.
            self._available_native = True
        elif self._system == "Darwin":
            try:
                subprocess.run(
                    ["say", "-v", "?"], capture_output=True, timeout=2,
                )
                self._available_native = True
            except (OSError, subprocess.TimeoutExpired):
                pass
        else:  # Linux
            for binary in ("espeak", "spd-say"):
                try:
                    subprocess.run(
                        [binary, "--version"], capture_output=True, timeout=2,
                    )
                    self._available_native = True
                    self._linux_tts_binary: str = binary
                    break
                except (OSError, subprocess.TimeoutExpired):
                    continue

    @staticmethod
    def _pick_spanish_voice(voices: list[Any]) -> Any | None:
        for v in voices:
            attrs = " ".join(
                str(getattr(v, k, "")).lower()
                for k in ("id", "name", "languages")
            )
            if any(hint in attrs for hint in _SPANISH_VOICE_HINTS):
                return v
        return None

    # ==================================================================
    #                       ENGINE POR THREAD (Windows + COM)
    # ==================================================================
    def _get_engine_for_current_thread(self) -> Any:
        """Devuelve un pyttsx3 engine VIVO para el thread que llama.

        El truco: en Windows, SAPI5 es COM. CoInitialize hay que llamarlo
        UNA VEZ por thread antes de tocar SAPI. Sin esto, runAndWait()
        retorna inmediatamente sin reproducir nada.
        """
        if not self._available_pyttsx3:
            return None
        tid = threading.get_ident()

        # CoInitialize por-thread en Windows.
        if self._system == "Windows" and tid not in self._com_initialized:
            try:
                import pythoncom
                pythoncom.CoInitialize()
                self._com_initialized.add(tid)
            except Exception as e:  # noqa: BLE001
                log.debug("pythoncom.CoInitialize: %s", e)

        with self._engines_lock:
            engine = self._engines_by_thread.get(tid)
            if engine is not None:
                return engine
            try:
                import pyttsx3
                engine = pyttsx3.init()
                engine.setProperty("rate", self._rate)
                engine.setProperty("volume", self._volume)
                if self._chosen_voice_id:
                    try:
                        engine.setProperty("voice", self._chosen_voice_id)
                    except Exception as e:  # noqa: BLE001
                        log.debug("setProperty voice: %s", e)
                self._engines_by_thread[tid] = engine
                return engine
            except Exception as e:  # noqa: BLE001
                log.warning("pyttsx3.init en thread %d: %s", tid, e)
                return None

    # ==================================================================
    #                       API PÚBLICA
    # ==================================================================
    def is_available(self) -> bool:
        return self._available_pyttsx3 or self._available_native

    def say(self, text: str) -> None:
        """Sintetiza `text`. Blocante. No-op si no disponible.

        IMPORTANTE: pyttsx3 en Windows depende de SAPI5 + COM + Windows
        message pump. En el MAIN thread funciona. En threads daemon
        (como `dialog-speak`) `runAndWait` retorna OK sin error pero
        sin producir audio (cazado empíricamente 22:05). Para esos
        threads, forzamos el fallback nativo subprocess que es 100%
        independiente del message pump del proceso.
        """
        if not text:
            return
        with self._lock:
            is_main = threading.current_thread() is threading.main_thread()

            # MAIN thread: pyttsx3 funciona bien aquí.
            if is_main and self._available_pyttsx3:
                engine = self._get_engine_for_current_thread()
                if engine is not None:
                    try:
                        engine.say(text)
                        engine.runAndWait()
                        return
                    except RuntimeError as e:
                        log.debug("pyttsx3 say RuntimeError: %s", e)
                    except Exception as e:  # noqa: BLE001
                        log.warning("pyttsx3 say falló: %s — fallback", e)

            # Cualquier OTRO thread, o pyttsx3 no disponible: subprocess.
            if self._available_native:
                self._say_native(text)
                return

            log.warning("Ningún backend TTS disponible — text=%r", text[:60])

    def say_async(self, text: str) -> None:
        if not text:
            return
        thread = threading.Thread(
            target=self.say, args=(text,), daemon=True,
        )
        thread.start()

    def stop(self) -> None:
        """Corta lo que esté hablando AHORA."""
        if not self._available_pyttsx3:
            return
        with self._engines_lock:
            for engine in self._engines_by_thread.values():
                try:
                    engine.stop()
                except Exception as e:  # noqa: BLE001
                    log.debug("engine.stop: %s", e)

    def is_speaking(self) -> bool:
        if not self._available_pyttsx3:
            return False
        with self._engines_lock:
            for engine in self._engines_by_thread.values():
                try:
                    if engine.isBusy():
                        return True
                except Exception:  # noqa: BLE001
                    continue
        return False

    def shutdown(self) -> None:
        with self._engines_lock:
            for engine in self._engines_by_thread.values():
                try:
                    engine.stop()
                except Exception as e:  # noqa: BLE001
                    log.debug("shutdown engine.stop: %s", e)
            self._engines_by_thread.clear()
        # CoUninitialize si hicimos CoInitialize.
        if self._system == "Windows":
            try:
                import pythoncom
                for _tid in self._com_initialized:
                    try:
                        pythoncom.CoUninitialize()
                    except Exception:  # noqa: BLE001
                        pass
                self._com_initialized.clear()
            except ImportError:
                pass

    # ==================================================================
    #                       FALLBACK NATIVO
    # ==================================================================
    def _say_native(self, text: str) -> None:
        """Reproduce vía subprocess. Usa COM SAPI.SpVoice en Windows.

        SAPI.SpVoice es MÁS confiable que System.Speech porque va directo
        al engine COM de Windows y no necesita .NET Framework (que algunos
        Windows tienen mutilado).
        """
        safe = text.replace('"', "'").replace("\n", " ").replace("`", "'")
        try:
            if self._system == "Windows":
                # COM SAPI directo. Selecciona voz española si existe.
                voice_id = self._chosen_voice_id or ""
                # Construimos PS con COM nativo de SAPI.
                ps = (
                    "$v = New-Object -ComObject SAPI.SpVoice; "
                )
                if voice_id:
                    # Buscamos por nombre/id en GetVoices.
                    voice_hint = self._chosen_voice or "Spanish"
                    ps += (
                        "$voices = $v.GetVoices(); "
                        "foreach ($vc in $voices) { "
                        f"  if ($vc.GetDescription() -match '{voice_hint[:30]}') {{ "
                        "    $v.Voice = $vc; break "
                        "  } "
                        "} "
                    )
                # Rate de SAPI: -10 a 10. ~0 = 175 wpm. Volume 0-100.
                ps += (
                    "$v.Rate = 0; "
                    "$v.Volume = 100; "
                    f'$v.Speak("{safe}") | Out-Null'
                )
                result = subprocess.run(
                    ["powershell", "-NoProfile", "-Command", ps],
                    capture_output=True, text=True, timeout=120,
                )
                if result.returncode != 0:
                    log.warning(
                        "SAPI SpVoice falló (rc=%d): %s",
                        result.returncode, (result.stderr or "")[-200:],
                    )
            elif self._system == "Darwin":
                subprocess.run(["say", safe], capture_output=True, timeout=120)
            else:
                binary = getattr(self, "_linux_tts_binary", "espeak")
                subprocess.run([binary, safe], capture_output=True, timeout=120)
        except (OSError, subprocess.TimeoutExpired) as e:
            log.warning("TTS nativo falló: %s", e)

    # ==================================================================
    #                       SELF-TEST AL ARRANQUE
    # ==================================================================
    def self_test(self, phrase: str | None = None) -> bool:
        """Reproduce una frase de prueba y reporta si el engine terminó OK."""
        if not self.is_available():
            log.warning("self_test: TTS no disponible")
            return False
        test = phrase or (
            "Prueba de audio. Si escuchas esta voz, el TTS funciona."
        )
        t0 = time.time()
        try:
            self.say(test)
        except Exception as e:  # noqa: BLE001
            log.warning("self_test crasheó: %s", e)
            return False
        elapsed = time.time() - t0
        # SAPI a rate 0 ≈ 180 palabras/min ≈ 3 chars/seg. Una frase de
        # 74 chars debería tardar al menos 8s. Si terminó en <3s con 50+
        # chars, casi seguro NO sonó nada — el subprocess retornó sin
        # esperar al engine, o el engine fall-throughed silenciosamente.
        min_expected_s = max(1.5, len(test) / 25.0)
        if elapsed < min_expected_s:
            log.warning(
                "self_test SOSPECHOSO: TTS terminó en %.2fs (esperado >=%.1fs "
                "para %d chars). Probablemente NO produjo audio. "
                "Verifica: 1) volumen del sistema, 2) mixer de aplicaciones "
                "Windows (icono altavoz → Mezclador de volumen), "
                "3) que la app python.exe no esté muteada.",
                elapsed, min_expected_s, len(test),
            )
            return False
        log.info(
            "self_test OK: TTS reprodujo %d chars en %.1fs (voz=%s)",
            len(test), elapsed, self._chosen_voice or "default",
        )
        return True

    # ==================================================================
    #                       AGENCIA SOBRE LA VOZ PROPIA
    # ==================================================================
    def list_voices(self) -> list[dict[str, str]]:
        """Devuelve voces disponibles del sistema con metadata mínima."""
        if not self._available_pyttsx3:
            return []
        try:
            import pyttsx3
            engine = pyttsx3.init()
            voices = engine.getProperty("voices") or []
            result = [
                {
                    "id": str(getattr(v, "id", "")),
                    "name": str(getattr(v, "name", "")),
                    "languages": str(getattr(v, "languages", "")),
                    "gender": str(getattr(v, "gender", "")),
                }
                for v in voices
            ]
            engine.stop()
            return result
        except Exception as e:  # noqa: BLE001
            log.debug("list_voices: %s", e)
            return []

    def change_voice(self, voice_hint: str) -> bool:
        """Cambia a la primera voz cuyo nombre/id matchee `voice_hint`.

        Devuelve True si se cambió, False si no se encontró. La nueva
        voz se aplica a TODAS las síntesis siguientes (descarta engines
        cacheados por thread para que tomen la voz nueva).
        """
        if not self._available_pyttsx3 or not voice_hint:
            return False
        hint = voice_hint.lower()
        for v in self.list_voices():
            attrs = " ".join((v["id"], v["name"], v["languages"])).lower()
            if hint in attrs:
                self._chosen_voice_id = v["id"]
                self._chosen_voice = v["name"]
                # Invalidamos engines cacheados — los próximos say()
                # crearán nuevos con la voz nueva.
                with self._engines_lock:
                    for engine in self._engines_by_thread.values():
                        try:
                            engine.stop()
                        except Exception:  # noqa: BLE001
                            pass
                    self._engines_by_thread.clear()
                log.info("Voz cambiada a: %s", self._chosen_voice)
                return True
        log.info("change_voice('%s'): no encontré ninguna que matchee", voice_hint)
        return False

    def current_voice_name(self) -> str:
        return self._chosen_voice or "default"
