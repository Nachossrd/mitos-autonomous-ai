"""
==============================================================================
 Proyecto MITOS - SensorHub (Fase 8 — Sensores físicos ACTIVOS)
==============================================================================

Toda la maquinaria de Fase 7 (`EnvironmentPerception`) solo DETECTABA
hardware — preguntaba "¿hay cámara?" pero no la usaba. Fase 8 cambia
ese paradigma: cada sensor es un THREAD daemon que captura datos en
tiempo real y los emite a una cola compartida.

Lo que esto desbloquea:

  - **CameraSensor**: hace `cv2.VideoCapture`, detecta movimiento por
    `absdiff` y caras por Haar cascade. Emite eventos `motion` y
    `face_detected` que el `CognitiveEngine` puede usar para reaccionar.
  - **MicrophoneSensor**: graba chunks de 5s con `sounddevice`. Si el
    RMS supera el umbral, emite `audio_activity`. Si Whisper está
    instalado, transcribe localmente y emite `speech`.
  - **SystemSensor**: vigila CPU/RAM/disco. Emite `high_cpu`, `high_ram`,
    `disk_full` cuando se cruzan umbrales.
  - **NetworkSensor**: ping a 8.8.8.8 cada 60s. Emite `internet_lost` y
    `internet_recovered`.

El `SensorHub` orquesta todo: `discover_and_start()` intenta arrancar
cada sensor (incluyendo `ensure_dependency()` para instalar el paquete
que falte), pone callbacks y mantiene un buffer de eventos recientes
que el ciclo cognitivo consume vía `get_situation_summary()`.

Convenciones (per INFORME_FORENSE §11):
  - Logger jerárquico `mitos.sensor_hub` y subloggers por sensor.
  - `from __future__ import annotations`, tipado moderno.
  - Threads daemon=True: el proceso principal NO se queda colgado por
    un sensor mal cerrado.
  - Métodos abstractos lanzan `NotImplementedError` con mensaje (no
    `pass` ni `...`).
==============================================================================
"""

from __future__ import annotations

import logging
import platform
import queue
import shutil
import socket
import subprocess
import sys
import threading
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

log = logging.getLogger("mitos.sensor_hub")


# ============================================================================
# Constantes
# ============================================================================
_QUEUE_MAXSIZE: int = 100
_MAX_RECENT_EVENTS: int = 20

# Umbrales del SystemSensor — alineados con los de `perception.py` Fase 7.
_CPU_HIGH_PCT: float = 85.0
_RAM_HIGH_PCT: float = 90.0
_DISK_HIGH_PCT: float = 95.0

# Subprocess pip install timeout: una dep tarda 30-90s normalmente.
_PIP_INSTALL_TIMEOUT_S: float = 180.0


# ============================================================================
# SensorEvent
# ============================================================================
@dataclass
class SensorEvent:
    """Evento generado por un sensor activo."""

    sensor_name: str
    event_type: str  # "motion"|"face_detected"|"speech"|"audio_activity"|...
    data: Any
    timestamp: float = field(default_factory=time.time)
    metadata: dict[str, Any] = field(default_factory=dict)


# ============================================================================
# BaseSensor — clase abstracta
# ============================================================================
class BaseSensor(ABC):
    """Sensor base. Cada subclase implementa `_loop()` e `is_available()`."""

    def __init__(self, name: str, event_queue: queue.Queue[SensorEvent]) -> None:
        self.name: str = name
        self._queue: queue.Queue[SensorEvent] = event_queue
        self._running: bool = False
        self._thread: threading.Thread | None = None

    # ==================================================================
    #                       CICLO DE VIDA DEL THREAD
    # ==================================================================
    def start(self) -> None:
        """Arranca el thread del sensor. Idempotente."""
        if self._running:
            log.debug("Sensor '%s' ya estaba activo", self.name)
            return
        self._running = True
        self._thread = threading.Thread(
            target=self._loop_wrapper,
            name=f"sensor-{self.name}",
            daemon=True,
        )
        self._thread.start()
        log.info("Sensor '%s' iniciado", self.name)

    def stop(self) -> None:
        """Señala parada y espera a que el thread termine."""
        if not self._running:
            return
        self._running = False
        if self._thread is not None and self._thread.is_alive():
            self._thread.join(timeout=5.0)
            if self._thread.is_alive():
                log.warning(
                    "Sensor '%s': thread no terminó en 5s — abandonando",
                    self.name,
                )
        log.info("Sensor '%s' detenido", self.name)

    def _loop_wrapper(self) -> None:
        """Wrapper alrededor de `_loop` que captura excepciones top-level.

        Sin esto, una excepción no manejada en el loop dejaría el thread
        muerto y `_running` en True — ningún operador podría reiniciar.
        """
        try:
            self._loop()
        except Exception as e:  # noqa: BLE001
            log.warning(
                "Sensor '%s': _loop crasheó: %s — apagando thread",
                self.name, e,
            )
        finally:
            self._running = False

    # ==================================================================
    #                       EMISIÓN DE EVENTOS
    # ==================================================================
    def _emit(
        self,
        event_type: str,
        data: Any,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Encola un evento. Si la cola está llena, descarta el más viejo."""
        event = SensorEvent(
            sensor_name=self.name,
            event_type=event_type,
            data=data,
            metadata=dict(metadata) if metadata else {},
        )
        try:
            self._queue.put_nowait(event)
        except queue.Full:
            # Cola saturada: descartamos el más viejo y reintentamos
            # una vez. Sin esto el sensor se bloquea.
            try:
                self._queue.get_nowait()
            except queue.Empty:
                return
            try:
                self._queue.put_nowait(event)
            except queue.Full:
                log.debug(
                    "Sensor '%s': cola saturada, evento descartado",
                    self.name,
                )

    # ==================================================================
    #                       MÉTODOS ABSTRACTOS
    # ==================================================================
    @abstractmethod
    def _loop(self) -> None:
        """Bucle de captura. Mientras `self._running`, capturar y emitir."""
        raise NotImplementedError(
            f"{type(self).__name__} debe implementar _loop()"
        )

    @abstractmethod
    def is_available(self) -> bool:
        """True si el hardware/dep está presente. Llamar antes de start()."""
        raise NotImplementedError(
            f"{type(self).__name__} debe implementar is_available()"
        )


# ============================================================================
# CameraSensor
# ============================================================================
class CameraSensor(BaseSensor):
    """Sensor de cámara: movimiento + detección de caras vía OpenCV."""

    _FRAME_INTERVAL_S: float = 2.0  # capturar cada 2s, no saturar CPU
    _MOTION_THRESHOLD: float = 30.0  # mean(absdiff) > este → movimiento
    _FACE_DETECT_EVERY_N_FRAMES: int = 3  # cada cuántos frames buscar caras

    def __init__(
        self,
        event_queue: queue.Queue[SensorEvent],
        camera_index: int = 0,
    ) -> None:
        super().__init__("camera", event_queue)
        self._cam_index: int = camera_index
        self._cv2: Any = None
        self._last_frame_jpeg: bytes | None = None

    # ==================================================================
    #                       DISPONIBILIDAD + DEPS
    # ==================================================================
    def is_available(self) -> bool:
        """True si cv2 está importable Y un device responde."""
        try:
            import cv2  # type: ignore[import-not-found]
        except ImportError:
            return False
        try:
            cap = cv2.VideoCapture(self._cam_index)
            opened = bool(cap.isOpened())
            cap.release()
            return opened
        except Exception as e:  # noqa: BLE001
            log.debug("is_available(camera): %s", e)
            return False

    def ensure_dependency(self) -> bool:
        """Importa cv2 instalando opencv-python si hace falta."""
        try:
            import cv2  # type: ignore[import-not-found]
            self._cv2 = cv2
            return True
        except ImportError:
            log.info("Instalando opencv-python…")
        try:
            completed = subprocess.run(
                [sys.executable, "-m", "pip", "install", "opencv-python"],
                capture_output=True,
                text=True,
                timeout=_PIP_INSTALL_TIMEOUT_S,
            )
        except (OSError, subprocess.TimeoutExpired) as e:
            log.warning("pip install opencv-python falló: %s", e)
            return False
        if completed.returncode != 0:
            log.warning(
                "pip install opencv-python exit=%d: %s",
                completed.returncode,
                (completed.stderr or "")[-200:],
            )
            return False
        try:
            import cv2  # type: ignore[import-not-found]
            self._cv2 = cv2
            return True
        except ImportError:
            return False

    # ==================================================================
    #                       BUCLE DE CAPTURA
    # ==================================================================
    def _loop(self) -> None:
        if not self.ensure_dependency():
            log.error("CameraSensor: sin opencv — sensor desactivado")
            return

        cv2 = self._cv2
        cap = cv2.VideoCapture(self._cam_index)
        if not cap.isOpened():
            log.error("CameraSensor: no pude abrir cámara %d", self._cam_index)
            return

        # Cargamos el clasificador UNA SOLA VEZ. Hacerlo cada frame era
        # ~50ms de I/O malgastado.
        face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        )
        if face_cascade.empty():
            log.warning(
                "CameraSensor: Haar cascade no cargó — sigo solo con movimiento"
            )
            face_cascade = None

        log.info(
            "CameraSensor: cámara %d abierta, capturando cada %.1fs",
            self._cam_index, self._FRAME_INTERVAL_S,
        )

        prev_gray = None
        frame_count = 0

        try:
            while self._running:
                ret, frame = cap.read()
                if not ret or frame is None:
                    time.sleep(1.0)
                    continue

                frame_count += 1
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                gray_small = cv2.resize(gray, (320, 240))

                # --- Detección de movimiento ---
                if prev_gray is not None:
                    diff = cv2.absdiff(prev_gray, gray_small)
                    motion_score = float(diff.mean())
                    if motion_score > self._MOTION_THRESHOLD:
                        self._emit(
                            "motion",
                            {
                                "score": motion_score,
                                "frame_shape": list(frame.shape),
                            },
                            metadata={
                                "description": (
                                    f"Movimiento detectado "
                                    f"(score={motion_score:.1f})"
                                ),
                            },
                        )

                # --- Detección de caras (cada N frames) ---
                if (
                    face_cascade is not None
                    and frame_count % self._FACE_DETECT_EVERY_N_FRAMES == 0
                ):
                    faces = face_cascade.detectMultiScale(
                        gray_small, scaleFactor=1.3, minNeighbors=5
                    )
                    if len(faces) > 0:
                        self._emit(
                            "face_detected",
                            {
                                "count": int(len(faces)),
                                "locations": [
                                    [int(v) for v in box] for box in faces
                                ],
                            },
                            metadata={
                                "description": (
                                    f"{len(faces)} cara(s) en frame"
                                ),
                            },
                        )

                # --- Cache del último frame (JPEG comprimido) ---
                ok, buffer = cv2.imencode(
                    ".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 50]
                )
                if ok:
                    self._last_frame_jpeg = buffer.tobytes()
                    # Log INFO al primer cache + cada 30 frames (cada minuto)
                    # para confirmar que la cache está viva.
                    if frame_count == 1 or frame_count % 30 == 0:
                        log.info(
                            "CameraSensor: frame %d cacheado (%d bytes JPEG)",
                            frame_count, len(self._last_frame_jpeg),
                        )
                else:
                    if frame_count <= 3:
                        log.warning(
                            "CameraSensor: imencode falló en frame %d", frame_count,
                        )

                prev_gray = gray_small
                time.sleep(self._FRAME_INTERVAL_S)
        finally:
            cap.release()
            log.info("CameraSensor: VideoCapture liberada")

    def get_last_frame_jpeg(self) -> bytes | None:
        """Bytes del último frame en JPEG (para tools de visión vía API)."""
        return self._last_frame_jpeg


# ============================================================================
# MicrophoneSensor
# ============================================================================
class MicrophoneSensor(BaseSensor):
    """Sensor de micrófono: actividad de audio + transcripción Whisper."""

    _SAMPLE_RATE: int = 16_000
    _CHUNK_DURATION_S: int = 5
    _SILENCE_THRESHOLD: float = 500.0  # RMS bajo este = silencio

    def __init__(self, event_queue: queue.Queue[SensorEvent]) -> None:
        super().__init__("microphone", event_queue)
        self._whisper_model: Any = None  # cache para no recargar cada chunk

    def is_available(self) -> bool:
        """True si hay al menos un device con channels de entrada."""
        try:
            import sounddevice as sd  # type: ignore[import-not-found]
        except ImportError:
            return False
        try:
            devices = sd.query_devices()
        except Exception as e:  # noqa: BLE001
            log.debug("sounddevice.query_devices: %s", e)
            return False
        for dev in devices:
            if isinstance(dev, dict) and int(dev.get("max_input_channels", 0)) > 0:
                return True
        return False

    def ensure_dependency(self) -> bool:
        """Importa sounddevice y numpy, instalándolos si hace falta."""
        try:
            import sounddevice  # type: ignore[import-not-found]
            import numpy  # type: ignore[import-not-found]
            return True
        except ImportError:
            log.info("Instalando sounddevice + numpy…")
        try:
            completed = subprocess.run(
                [
                    sys.executable, "-m", "pip", "install",
                    "sounddevice", "numpy",
                ],
                capture_output=True,
                text=True,
                timeout=_PIP_INSTALL_TIMEOUT_S,
            )
        except (OSError, subprocess.TimeoutExpired) as e:
            log.warning("pip install sounddevice+numpy falló: %s", e)
            return False
        if completed.returncode != 0:
            log.warning(
                "pip install sounddevice/numpy exit=%d: %s",
                completed.returncode,
                (completed.stderr or "")[-200:],
            )
            return False
        try:
            import sounddevice  # type: ignore[import-not-found]  # noqa: F401
            import numpy  # type: ignore[import-not-found]  # noqa: F401
            return True
        except ImportError:
            return False

    # ==================================================================
    #                       BUCLE DE CAPTURA
    # ==================================================================
    def _loop(self) -> None:
        if not self.ensure_dependency():
            log.error("MicrophoneSensor: sin sounddevice — desactivado")
            return

        import numpy as np
        import sounddevice as sd

        log.info(
            "MicrophoneSensor activo (chunks de %ds @ %d Hz)",
            self._CHUNK_DURATION_S, self._SAMPLE_RATE,
        )

        while self._running:
            try:
                audio = sd.rec(
                    int(self._CHUNK_DURATION_S * self._SAMPLE_RATE),
                    samplerate=self._SAMPLE_RATE,
                    channels=1,
                    dtype="int16",
                )
                sd.wait()
            except Exception as e:  # noqa: BLE001
                log.warning("MicrophoneSensor: rec falló: %s", e)
                time.sleep(2.0)
                continue

            try:
                rms = float(np.sqrt(np.mean(audio.astype(np.float64) ** 2)))
            except Exception as e:  # noqa: BLE001
                log.debug("RMS calc falló: %s", e)
                continue

            if rms <= self._SILENCE_THRESHOLD:
                continue

            self._emit(
                "audio_activity",
                {
                    "rms": rms,
                    "duration_s": self._CHUNK_DURATION_S,
                    "samples": int(len(audio)),
                },
                metadata={"description": f"Audio detectado (RMS={rms:.0f})"},
            )

            transcript = self._try_transcribe(audio)
            if transcript:
                self._emit(
                    "speech",
                    {"text": transcript, "confidence": 0.8},
                    metadata={
                        "description": f"Escuché: '{transcript[:80]}'"
                    },
                )

    # ==================================================================
    #                       TRANSCRIPCIÓN (opcional, Whisper)
    # ==================================================================
    def _try_transcribe(self, audio: Any) -> str | None:
        """Transcribe con Whisper local si está disponible. None si no."""
        try:
            import whisper  # type: ignore[import-not-found]
        except ImportError:
            return None

        if self._whisper_model is None:
            try:
                self._whisper_model = whisper.load_model("small")
            except Exception as e:  # noqa: BLE001
                log.debug("whisper.load_model falló: %s", e)
                return None

        import tempfile
        import wave

        try:
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as fh:
                tmp_path = fh.name
            wf = wave.open(tmp_path, "wb")
            try:
                wf.setnchannels(1)
                wf.setsampwidth(2)
                wf.setframerate(self._SAMPLE_RATE)
                wf.writeframes(audio.tobytes())
            finally:
                wf.close()

            result = self._whisper_model.transcribe(tmp_path, language="es")
        except Exception as e:  # noqa: BLE001
            log.debug("whisper.transcribe falló: %s", e)
            return None
        finally:
            try:
                Path(tmp_path).unlink(missing_ok=True)
            except OSError:
                log.debug("no pude borrar tmp wav %s", tmp_path)

        text = str(result.get("text", "")).strip()
        return text or None

    # ==================================================================
    #                       API DE USO PUNTUAL (CLI brain)
    # ==================================================================
    def record_and_transcribe(self, seconds: float = 5.0) -> str:
        """Graba N segundos de audio y devuelve la transcripción.

        Esto es para el `/listen` del CLI brain: no usa el thread del
        sensor, abre el mic solo el tiempo necesario y lo cierra.
        Devuelve "" si no pudo grabar ni transcribir.
        """
        if not self.ensure_dependency():
            return ""

        import numpy as np
        import sounddevice as sd

        try:
            audio = sd.rec(
                int(seconds * self._SAMPLE_RATE),
                samplerate=self._SAMPLE_RATE,
                channels=1,
                dtype="int16",
            )
            sd.wait()
        except Exception as e:  # noqa: BLE001
            log.warning("record_and_transcribe rec/wait: %s", e)
            return ""

        audio_flat = audio.flatten()
        rms = float(np.sqrt(np.mean(audio_flat.astype("float32") ** 2)))
        if rms < self._SILENCE_THRESHOLD:
            log.info("Silencio detectado (RMS=%.0f); no transcribo", rms)
            return ""

        transcript = self._try_transcribe(audio_flat)
        return transcript or ""


# ============================================================================
# SystemSensor
# ============================================================================
class SystemSensor(BaseSensor):
    """Vigila CPU/RAM/Disco usando solo stdlib."""

    _CHECK_INTERVAL_S: float = 30.0

    def __init__(self, event_queue: queue.Queue[SensorEvent]) -> None:
        super().__init__("system", event_queue)

    def is_available(self) -> bool:
        """Siempre disponible: lee /proc o ctypes."""
        return True

    # ==================================================================
    #                       BUCLE
    # ==================================================================
    def _loop(self) -> None:
        log.info(
            "SystemSensor activo (chequeo cada %.0fs)",
            self._CHECK_INTERVAL_S,
        )
        while self._running:
            cpu_pct = self._read_cpu_percent()
            ram_pct = self._read_ram_percent()
            disk_pct = self._read_disk_percent()

            if cpu_pct >= _CPU_HIGH_PCT:
                self._emit(
                    "high_cpu",
                    {"cpu_percent": cpu_pct},
                    metadata={
                        "description": (
                            f"CPU al {cpu_pct:.0f}% — proceso pesado en bg"
                        ),
                    },
                )
            if ram_pct >= _RAM_HIGH_PCT:
                self._emit(
                    "high_ram",
                    {"ram_percent": ram_pct},
                    metadata={
                        "description": (
                            f"RAM al {ram_pct:.0f}% — riesgo de OOM"
                        ),
                    },
                )
            if disk_pct >= _DISK_HIGH_PCT:
                self._emit(
                    "disk_full",
                    {"disk_percent": disk_pct},
                    metadata={
                        "description": f"Disco al {disk_pct:.0f}%",
                    },
                )

            time.sleep(self._CHECK_INTERVAL_S)

    # ==================================================================
    #                       LECTURAS
    # ==================================================================
    @staticmethod
    def _read_cpu_percent() -> float:
        """CPU agregada en %. -1 si no se puede medir."""
        system = platform.system()
        if system == "Linux":
            try:
                with open("/proc/loadavg", encoding="utf-8") as fh:
                    load = float(fh.read().split()[0])
                import os
                cores = os.cpu_count() or 1
                return min(100.0, load * 100.0 / cores)
            except (OSError, ValueError, IndexError) as e:
                log.debug("read /proc/loadavg: %s", e)
                return -1.0
        # Windows / Mac: sin psutil no hay manera barata → -1 (no alerta)
        return -1.0

    @staticmethod
    def _read_ram_percent() -> float:
        """RAM usada en %. -1 si no se puede medir."""
        system = platform.system()
        if system == "Linux":
            try:
                total_kb = 0
                avail_kb = 0
                with open("/proc/meminfo", encoding="utf-8") as fh:
                    for line in fh:
                        if line.startswith("MemTotal:"):
                            total_kb = int(line.split()[1])
                        elif line.startswith("MemAvailable:"):
                            avail_kb = int(line.split()[1])
                        if total_kb and avail_kb:
                            break
                if total_kb <= 0:
                    return -1.0
                return (1.0 - avail_kb / total_kb) * 100.0
            except (OSError, ValueError, IndexError) as e:
                log.debug("read /proc/meminfo: %s", e)
                return -1.0
        if system == "Windows":
            return SystemSensor._read_ram_windows_ctypes()
        return -1.0

    @staticmethod
    def _read_ram_windows_ctypes() -> float:
        """RAM% en Windows vía GlobalMemoryStatusEx."""
        try:
            import ctypes

            class MemoryStatusEx(ctypes.Structure):
                _fields_ = [
                    ("dwLength", ctypes.c_ulong),
                    ("dwMemoryLoad", ctypes.c_ulong),
                    ("ullTotalPhys", ctypes.c_ulonglong),
                    ("ullAvailPhys", ctypes.c_ulonglong),
                    ("ullTotalPageFile", ctypes.c_ulonglong),
                    ("ullAvailPageFile", ctypes.c_ulonglong),
                    ("ullTotalVirtual", ctypes.c_ulonglong),
                    ("ullAvailVirtual", ctypes.c_ulonglong),
                    ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
                ]

            stat = MemoryStatusEx()
            stat.dwLength = ctypes.sizeof(MemoryStatusEx)
            ok = ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(stat))
            if not ok:
                return -1.0
            return float(stat.dwMemoryLoad)
        except (OSError, AttributeError) as e:
            log.debug("RAM ctypes: %s", e)
            return -1.0

    @staticmethod
    def _read_disk_percent() -> float:
        """Disco usado del volumen raíz, en %."""
        try:
            target = "/" if platform.system() != "Windows" else "C:\\"
            usage = shutil.disk_usage(target)
            if usage.total <= 0:
                return -1.0
            return usage.used / usage.total * 100.0
        except OSError as e:
            log.debug("disk_usage: %s", e)
            return -1.0


# ============================================================================
# NetworkSensor
# ============================================================================
class NetworkSensor(BaseSensor):
    """Vigila la conectividad: avisa cuando se pierde o se recupera."""

    _CHECK_INTERVAL_S: float = 60.0
    _PROBES: tuple[tuple[str, int], ...] = (
        ("8.8.8.8", 53),
        ("1.1.1.1", 53),
    )
    _PROBE_TIMEOUT_S: float = 3.0

    def __init__(self, event_queue: queue.Queue[SensorEvent]) -> None:
        super().__init__("network", event_queue)
        self._was_online: bool | None = None

    def is_available(self) -> bool:
        """Siempre disponible: TCP/DNS funciona en cualquier OS soportado."""
        return True

    def _loop(self) -> None:
        log.info(
            "NetworkSensor activo (chequeo cada %.0fs)",
            self._CHECK_INTERVAL_S,
        )
        while self._running:
            is_online = self._probe_internet()

            if self._was_online is not None and is_online != self._was_online:
                if is_online:
                    self._emit(
                        "internet_recovered",
                        {},
                        metadata={
                            "description": (
                                "Internet recuperado — APIs externas disponibles"
                            ),
                        },
                    )
                else:
                    self._emit(
                        "internet_lost",
                        {},
                        metadata={
                            "description": (
                                "Internet perdido — solo LLM local"
                            ),
                        },
                    )
            self._was_online = is_online
            time.sleep(self._CHECK_INTERVAL_S)

    @classmethod
    def _probe_internet(cls) -> bool:
        """True si abrimos TCP a algún DNS público en <`_PROBE_TIMEOUT_S`."""
        for host, port in cls._PROBES:
            try:
                with socket.create_connection(
                    (host, port), timeout=cls._PROBE_TIMEOUT_S
                ):
                    return True
            except (OSError, socket.timeout):
                continue
        return False


# ============================================================================
# SensorHub — orquestador
# ============================================================================
class SensorHub:
    """Gestiona todos los sensores y centraliza eventos en una cola."""

    def __init__(self, project_root: str | Path = ".") -> None:
        self.root: Path = Path(project_root).resolve()
        self.event_queue: queue.Queue[SensorEvent] = queue.Queue(
            maxsize=_QUEUE_MAXSIZE
        )
        self.sensors: dict[str, BaseSensor] = {}
        self._recent_events: list[SensorEvent] = []
        self._recent_lock: threading.Lock = threading.Lock()
        self._callbacks: list[Callable[[SensorEvent], None]] = []
        self._processor_thread: threading.Thread | None = None
        self._running: bool = False
        # Fase 8+: hot-plug watcher opcional (pyudev en Linux).
        self._hotplug_thread: threading.Thread | None = None
        self._on_hotplug: Callable[[], None] | None = None
        log.info("SensorHub inicializado en %s", self.root)

    # ==================================================================
    #                       DESCUBRIMIENTO E INICIO
    # ==================================================================
    def discover_and_start(self) -> None:
        """Descubre hardware, instala deps faltantes, arranca sensores.

        Es idempotente: llamarlo dos veces NO duplica el processor_thread
        ni reinicia sensores ya vivos. Cada llamada puede AÑADIR sensores
        nuevos (USB enchufado en caliente).
        """
        log.info("=== DESCUBRIENDO HARDWARE ===")

        # Sensores SIEMPRE disponibles. _try_register es no-op si ya viven.
        self._try_register(SystemSensor(self.event_queue))
        self._try_register(NetworkSensor(self.event_queue))

        cam = CameraSensor(self.event_queue)
        if cam.is_available() or (cam.ensure_dependency() and cam.is_available()):
            self._try_register(cam)

        mic = MicrophoneSensor(self.event_queue)
        if mic.is_available() or (mic.ensure_dependency() and mic.is_available()):
            self._try_register(mic)

        # Processor: solo arrancamos UNO en toda la vida del SensorHub.
        if self._processor_thread is None or not self._processor_thread.is_alive():
            self._running = True
            self._processor_thread = threading.Thread(
                target=self._process_events,
                name="sensor-hub-processor",
                daemon=True,
            )
            self._processor_thread.start()
        log.info("=== %d SENSORES ACTIVOS ===", len(self.sensors))

    def _try_register(self, sensor: BaseSensor) -> None:
        if sensor.name in self.sensors:
            return  # ya vivo — idempotente
        try:
            sensor.start()
        except Exception as e:  # noqa: BLE001
            log.warning("No pude iniciar sensor '%s': %s", sensor.name, e)
            return
        self.sensors[sensor.name] = sensor
        log.info("Sensor '%s' activado", sensor.name)

    # ==================================================================
    #                       HOT-PLUG WATCHER (Linux, opcional)
    # ==================================================================
    def start_hotplug_watcher(
        self, on_hotplug: Callable[[], None]
    ) -> bool:
        """Si pyudev está instalado, lanza thread reactivo a USB plug/unplug.

        En Windows/Mac sin pyudev, devuelve False y el CognitiveEngine
        cae a su modo polling (re-discover cada N ciclos).

        Args:
            on_hotplug: callback que será llamado SIN argumentos cada vez
                        que el sistema reporta un cambio de dispositivo.
                        Típicamente: lambda: hub.discover_and_start().
        """
        if self._hotplug_thread is not None and self._hotplug_thread.is_alive():
            return True  # idempotente

        try:
            import pyudev
        except ImportError:
            log.info(
                "pyudev no disponible — hot-plug reactivo deshabilitado. "
                "Instala con `pip install pyudev` (solo Linux)."
            )
            return False

        self._on_hotplug = on_hotplug

        def _watch() -> None:
            try:
                context = pyudev.Context()
                monitor = pyudev.Monitor.from_netlink(context)
                monitor.filter_by(subsystem="usb")
                for device in iter(monitor.poll, None):
                    if not self._running:
                        return
                    action = getattr(device, "action", None)
                    if action in ("add", "remove"):
                        log.info(
                            "Hot-plug USB detectado: action=%s device=%s",
                            action, getattr(device, "device_path", "?"),
                        )
                        if self._on_hotplug is not None:
                            try:
                                self._on_hotplug()
                            except Exception as e:  # noqa: BLE001
                                log.warning("on_hotplug callback falló: %s", e)
            except Exception as e:  # noqa: BLE001
                log.warning("pyudev watcher murió: %s", e)

        self._hotplug_thread = threading.Thread(
            target=_watch, name="sensor-hub-hotplug", daemon=True,
        )
        self._hotplug_thread.start()
        log.info("Hot-plug watcher (pyudev) activo")
        return True

    # ==================================================================
    #                       PROCESADOR DE EVENTOS
    # ==================================================================
    def _process_events(self) -> None:
        """Consume la cola, mantiene buffer reciente y dispara callbacks."""
        while self._running:
            try:
                event = self.event_queue.get(timeout=1.0)
            except queue.Empty:
                continue

            with self._recent_lock:
                self._recent_events.append(event)
                if len(self._recent_events) > _MAX_RECENT_EVENTS:
                    self._recent_events = self._recent_events[
                        -_MAX_RECENT_EVENTS:
                    ]

            for callback in list(self._callbacks):
                try:
                    callback(event)
                except Exception as e:  # noqa: BLE001
                    log.debug("callback de SensorHub falló: %s", e)

    # ==================================================================
    #                       API PÚBLICA
    # ==================================================================
    def on_event(self, callback: Callable[[SensorEvent], None]) -> None:
        """Registra un callback que será invocado con cada evento."""
        self._callbacks.append(callback)

    def get_recent_events(self, max_events: int = 5) -> list[SensorEvent]:
        """Devuelve copia de los últimos N eventos (orden cronológico)."""
        with self._recent_lock:
            return list(self._recent_events[-max_events:])

    def get_situation_summary(self) -> str:
        """Resumen comprimido para inyectar al prompt del dispatcher."""
        with self._recent_lock:
            recent_copy = list(self._recent_events[-3:])

        if not recent_copy:
            active = sorted(self.sensors.keys())
            if not active:
                return "Sin sensores activos"
            return f"Sin actividad sensorial. Sensores activos: {', '.join(active)}"

        lines: list[str] = []
        for event in recent_copy:
            desc = str(
                event.metadata.get("description")
                or f"{event.event_type}"
            )
            lines.append(f"• {desc}")
        active = sorted(self.sensors.keys())
        if active:
            lines.append(f"[Sensores activos: {', '.join(active)}]")
        return "\n".join(lines)

    def get_capabilities(self) -> dict[str, bool]:
        """Qué puede percibir MITOS ahora mismo."""
        return {
            "vision": "camera" in self.sensors,
            "hearing": "microphone" in self.sensors,
            "system_monitoring": "system" in self.sensors,
            "network_awareness": "network" in self.sensors,
        }

    def shutdown(self) -> None:
        """Apaga todos los sensores y el procesador de eventos."""
        self._running = False
        for sensor in list(self.sensors.values()):
            try:
                sensor.stop()
            except Exception as e:  # noqa: BLE001
                log.debug("sensor.stop(%s): %s", sensor.name, e)
        if self._processor_thread is not None and self._processor_thread.is_alive():
            self._processor_thread.join(timeout=3.0)
        log.info("SensorHub apagado")
