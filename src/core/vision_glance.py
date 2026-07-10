"""
==============================================================================
 Proyecto MITOS - VisionGlance (Fase 8+ — visión bajo demanda)
==============================================================================

Le da ojos puntuales a MITOS, distintos del `CameraSensor` del SensorHub:

  - CameraSensor (sensor_hub.py): vive en un thread propio, captura cada
    2 segundos, busca movimiento + caras. Es pasivo.

  - VisionGlance (este módulo): SIN thread. Cuando el operador dice
    "puedes verme?", abre la cámara, captura UNA frame, la codifica
    a JPEG base64, la cierra. Listo para mandar al modelo de visión
    (Gemini Flash con supports_vision=True) y obtener una descripción.

Se separa del SensorHub porque su contrato es distinto: el sensor
necesita exclusividad sobre la cámara (no puede haber dos VideoCapture
sobre el mismo índice). El glance puntual solo se usa cuando el sensor
NO está activo, o se reusa la última frame que cacheó el sensor.

Convenciones:
  - Logger jerárquico `mitos.vision_glance`.
==============================================================================
"""

from __future__ import annotations

import base64
import logging
import time
from typing import Any

log = logging.getLogger("mitos.vision_glance")


# Cuántos intentos hacemos para abrir la cámara. cv2 a veces falla en
# el primer warmup (sobre todo en webcam USB en Windows).
_MAX_OPEN_ATTEMPTS: int = 3
_WARMUP_FRAMES: int = 5  # frames a descartar antes de capturar (auto-expo)


class VisionGlance:
    """Captura puntual de una frame de cámara + codificación base64."""

    def __init__(
        self,
        camera_index: int = 0,
        sensor_hub: Any = None,
    ) -> None:
        """
        Args:
            camera_index: índice de webcam (0 = primera). Usable solo si
                          NO hay un CameraSensor activo sobre el mismo.
            sensor_hub:   opcional. Si existe Y tiene CameraSensor activo,
                          intentaremos reusar la última frame cacheada
                          en lugar de abrir un VideoCapture nuevo (lo que
                          chocaría con el sensor).
        """
        self.camera_index: int = int(camera_index)
        self.sensors: Any = sensor_hub
        log.debug("VisionGlance listo (camera_index=%d)", self.camera_index)

    # ==================================================================
    #                       DISPONIBILIDAD
    # ==================================================================
    def is_available(self) -> bool:
        """Hay forma de tomar una foto AHORA?"""
        if self._has_sensor_cached_frame():
            return True
        try:
            import cv2  # noqa: F401
        except ImportError:
            return False
        return True

    def _has_sensor_cached_frame(self) -> bool:
        if self.sensors is None:
            log.warning("vision_glance: self.sensors=None — no hay SensorHub")
            return False
        sensors_dict = getattr(self.sensors, "sensors", {})
        cam = sensors_dict.get("camera")
        if cam is None:
            log.warning(
                "vision_glance: no hay sensor 'camera' en SensorHub "
                "(sensores activos: %s)",
                list(sensors_dict.keys()),
            )
            return False
        jpeg = getattr(cam, "_last_frame_jpeg", None)
        if jpeg is None:
            log.warning(
                "vision_glance: camera sensor existe pero _last_frame_jpeg=None "
                "(aún no capturó frame o crasheó el loop interno)"
            )
            return False
        log.info(
            "vision_glance: usando frame cacheado de CameraSensor (%d bytes)",
            len(jpeg),
        )
        return True

    # ==================================================================
    #                       CAPTURA
    # ==================================================================
    def snapshot_b64(self) -> str | None:
        """Devuelve JPEG en base64 listo para API multimodal, o None."""
        # 1) Reuso de la frame que el SensorHub ya tiene.
        if self._has_sensor_cached_frame():
            try:
                jpeg = self.sensors.sensors["camera"]._last_frame_jpeg
                return base64.b64encode(jpeg).decode("ascii")
            except Exception as e:  # noqa: BLE001
                log.debug("reuso de frame cacheada falló: %s", e)

        # 2) Captura nueva con cv2.
        try:
            import cv2
        except ImportError:
            log.info(
                "opencv-python no instalado — visión bajo demanda no disponible. "
                "Instala con `pip install opencv-python`."
            )
            return None

        cap = None
        try:
            for attempt in range(_MAX_OPEN_ATTEMPTS):
                cap = cv2.VideoCapture(self.camera_index, cv2.CAP_DSHOW)
                if cap.isOpened():
                    break
                cap.release()
                cap = None
                time.sleep(0.3)
            if cap is None or not cap.isOpened():
                log.info("No pude abrir la cámara índice %d", self.camera_index)
                return None

            # Warmup: descartamos las primeras frames porque suelen ser negras
            # mientras la cámara ajusta exposición.
            for _ in range(_WARMUP_FRAMES):
                cap.read()

            ok, frame = cap.read()
            if not ok or frame is None:
                log.warning("Lectura de frame falló")
                return None

            ok_enc, jpeg = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
            if not ok_enc:
                log.warning("Encoding JPEG falló")
                return None

            return base64.b64encode(jpeg.tobytes()).decode("ascii")
        except Exception as e:  # noqa: BLE001
            log.warning("snapshot_b64 crasheó: %s", e)
            return None
        finally:
            if cap is not None:
                cap.release()
