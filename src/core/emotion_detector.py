"""
==============================================================================
 Proyecto MITOS - EmotionDetector (Fase 8+ — leer la cara + el tono)
==============================================================================

MITOS necesita empatía para recomendar canciones / cambiar tono /
ofrecer ayuda según cómo te ve y cómo te oye.

Dos canales:
  1) Cara: smile detection con cv2 Haar (sin extra-deps). Detecta una
     sonrisa dentro del bounding box de cara — proxy crudo pero útil.
     Si DeepFace o FER están disponibles, los usaríamos para 7 emociones.
  2) Voz: RMS energy + variación. Energía baja + variación baja → triste
     o cansado. Energía alta + variación alta → animado.

Las observaciones se publican al `UserProfile.add_mood()`. El
DialogLoop las consulta para proactividad empática.

Convenciones:
  - Logger `mitos.emotion`.
==============================================================================
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

log = logging.getLogger("mitos.emotion")


@dataclass(frozen=True)
class EmotionObservation:
    """Lo que MITOS ve/oye en un instante."""

    label: str            # "positive" | "neutral" | "tired" | "negative" | "unknown"
    confidence: float     # [0, 1]
    source: str           # "face_smile" | "voice_energy"


class EmotionDetector:
    """Detector ligero de emoción desde frames y audio."""

    def __init__(self) -> None:
        self._face_cascade: Any = None
        self._smile_cascade: Any = None
        self._available_face: bool = self._init_haar()

    # ==================================================================
    #                       INIT (Haar cascades)
    # ==================================================================
    def _init_haar(self) -> bool:
        try:
            import cv2
            face_path = (
                Path(cv2.data.haarcascades)
                / "haarcascade_frontalface_default.xml"
            )
            smile_path = (
                Path(cv2.data.haarcascades) / "haarcascade_smile.xml"
            )
            if not face_path.is_file() or not smile_path.is_file():
                log.warning("Haar cascades no encontradas en cv2.data")
                return False
            self._face_cascade = cv2.CascadeClassifier(str(face_path))
            self._smile_cascade = cv2.CascadeClassifier(str(smile_path))
            return True
        except ImportError:
            log.info(
                "cv2 no disponible — detección facial deshabilitada"
            )
            return False
        except Exception as e:  # noqa: BLE001
            log.warning("Haar init: %s", e)
            return False

    # ==================================================================
    #                       CARA (sonrisa)
    # ==================================================================
    def from_face_frame(self, frame: Any) -> EmotionObservation:
        """Analiza una frame BGR y devuelve emoción inferida."""
        if not self._available_face or frame is None:
            return EmotionObservation("unknown", 0.0, "face_smile")
        try:
            import cv2
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = self._face_cascade.detectMultiScale(
                gray, scaleFactor=1.1, minNeighbors=5, minSize=(80, 80),
            )
            if len(faces) == 0:
                return EmotionObservation("unknown", 0.0, "face_smile")
            # Tomamos la cara más grande.
            faces = sorted(
                faces, key=lambda r: r[2] * r[3], reverse=True,
            )
            x, y, w, h = faces[0]
            face_roi = gray[y:y + h, x:x + w]
            smiles = self._smile_cascade.detectMultiScale(
                face_roi, scaleFactor=1.7, minNeighbors=22,
                minSize=(int(w * 0.2), int(h * 0.1)),
            )
            if len(smiles) > 0:
                # Mientras más rectángulos de sonrisa, más confianza.
                conf = min(1.0, 0.5 + 0.1 * len(smiles))
                return EmotionObservation("positive", conf, "face_smile")
            return EmotionObservation("neutral", 0.6, "face_smile")
        except Exception as e:  # noqa: BLE001
            log.debug("from_face_frame: %s", e)
            return EmotionObservation("unknown", 0.0, "face_smile")

    # ==================================================================
    #                       VOZ (energía / variabilidad)
    # ==================================================================
    @staticmethod
    def from_voice_audio(audio: Any, sample_rate: int = 16000) -> EmotionObservation:
        """Analiza audio int16 mono y devuelve emoción inferida del tono.

        Heurística cruda:
          - RMS muy bajo + std bajo → tired
          - RMS alto + std alto → positive
          - El resto → neutral
        """
        if audio is None or len(audio) == 0:
            return EmotionObservation("unknown", 0.0, "voice_energy")
        try:
            import numpy as np
            arr = np.asarray(audio, dtype="float32")
            rms = float(np.sqrt(np.mean(arr ** 2)))
            std = float(np.std(arr))
            # Normalización tosca contra int16 max (32768).
            rms_n = rms / 32768.0
            std_n = std / 32768.0
            if rms_n < 0.005 and std_n < 0.01:
                return EmotionObservation("tired", 0.6, "voice_energy")
            if rms_n > 0.03 and std_n > 0.06:
                return EmotionObservation("positive", 0.65, "voice_energy")
            if rms_n < 0.01:
                return EmotionObservation("tired", 0.5, "voice_energy")
            return EmotionObservation("neutral", 0.5, "voice_energy")
        except Exception as e:  # noqa: BLE001
            log.debug("from_voice_audio: %s", e)
            return EmotionObservation("unknown", 0.0, "voice_energy")
