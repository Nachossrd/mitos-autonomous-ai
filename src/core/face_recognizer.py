"""
==============================================================================
 Proyecto MITOS - FaceRecognizer (Fase 8+ — visión que SABE quién es quién)
==============================================================================

Le da a MITOS la capacidad de DISTINGUIR caras, no solo detectarlas:

  - CameraSensor: detecta que HAY una cara (Haar cascade).
  - FaceRecognizer (este): dice "es Ignacio" o "es desconocido nuevo, ¿lo
    registro?" — y guarda la identificación en la memoria del proyecto.

Implementación: LBPH (Local Binary Patterns Histograms) de
`opencv-contrib-python`. Sin deep learning — entrena con 20-30 frames
en segundos y predice en milisegundos. Menos preciso que face_recognition
(dlib) pero infinitamente más fácil de instalar en Windows.

Persistencia:
  - data/faces/lbph_model.yml      ← el modelo LBPH entrenado
  - data/faces/names.json          ← {label_id: nombre}

API de uso:
  - register(name, frames)         ← añade caras al modelo
  - identify(frame) -> list[Match] ← quién ve en la frame
  - known_names() -> list[str]

Convenciones:
  - Logger `mitos.face_recognizer`.
==============================================================================
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

log = logging.getLogger("mitos.face_recognizer")


# LBPH devuelve "confidence" como DISTANCIA (menor = más parecido).
# Empíricamente: <50 = match seguro, 50-80 = dudoso, >80 = desconocido.
# LBPH: confidence MENOR = mejor match. Empíricamente con 20-30 muestras
# de entrenamiento, valores >75 son falsos positivos casi siempre. Pero
# entre 65-75 hay zona gris. Subimos a 85 para reducir flicker de
# "te veo / no te veo" cuando un mismo rostro registrado oscila.
_LBPH_CONFIDENCE_THRESHOLD: float = 85.0
_FACE_INPUT_SIZE: tuple[int, int] = (160, 160)


@dataclass(frozen=True)
class FaceMatch:
    """Una cara detectada + identificada en una frame."""

    name: str                # "Ignacio", "desconocido", "?"
    confidence: float        # menor = mejor match (LBPH semántica)
    bbox: tuple[int, int, int, int]  # x, y, w, h en pixels
    is_known: bool


class FaceRecognizer:
    """Reconocimiento facial con LBPH. Aprende online."""

    def __init__(self, project_root: str | Path) -> None:
        self.root: Path = Path(project_root).resolve()
        self.faces_dir: Path = self.root / "data" / "faces"
        self.model_path: Path = self.faces_dir / "lbph_model.yml"
        self.names_path: Path = self.faces_dir / "names.json"

        self._recognizer: Any = None
        self._face_cascade: Any = None
        self._names: dict[int, str] = {}
        self._available: bool = False

        self._init()

    # ==================================================================
    #                       INIT
    # ==================================================================
    def _init(self) -> None:
        try:
            import cv2
        except ImportError:
            log.info(
                "opencv-python no instalado — face recognition deshabilitado. "
                "Instala con `pip install opencv-contrib-python`"
            )
            return

        # cv2.face vive en opencv-contrib-python (NO en opencv-python pelado).
        if not hasattr(cv2, "face"):
            log.warning(
                "opencv-python pelado no tiene cv2.face. Instala el contrib:\n"
                "  pip uninstall opencv-python -y && pip install opencv-contrib-python"
            )
            return

        try:
            self._recognizer = cv2.face.LBPHFaceRecognizer_create()
        except Exception as e:  # noqa: BLE001
            log.warning("LBPHFaceRecognizer_create falló: %s", e)
            return

        # Haar cascade preempaquetado en cv2.
        cascade_path = (
            Path(cv2.data.haarcascades) / "haarcascade_frontalface_default.xml"
        )
        if cascade_path.is_file():
            self._face_cascade = cv2.CascadeClassifier(str(cascade_path))

        # Cargar modelo persistido si existe.
        if self.model_path.is_file():
            try:
                self._recognizer.read(str(self.model_path))
                log.info("Modelo LBPH cargado de %s", self.model_path)
            except Exception as e:  # noqa: BLE001
                log.warning("LBPH read falló: %s", e)

        if self.names_path.is_file():
            try:
                self._names = {
                    int(k): str(v)
                    for k, v in json.loads(
                        self.names_path.read_text(encoding="utf-8")
                    ).items()
                }
                log.info("Caras registradas: %s", list(self._names.values()))
            except (OSError, ValueError) as e:
                log.debug("load names.json: %s", e)

        self._available = True

    # ==================================================================
    #                       API
    # ==================================================================
    def is_available(self) -> bool:
        return self._available

    def known_names(self) -> list[str]:
        return list(self._names.values())

    def has_anybody(self) -> bool:
        return bool(self._names)

    # ==================================================================
    #                       DETECCIÓN INTERNA
    # ==================================================================
    def _detect_faces(self, frame: Any) -> list[Any]:
        """Devuelve lista de regiones (faces, bboxes) en grayscale."""
        if self._face_cascade is None:
            return []
        try:
            import cv2
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            rects = self._face_cascade.detectMultiScale(
                gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60),
            )
            return [(gray[y:y + h, x:x + w], (int(x), int(y), int(w), int(h)))
                    for (x, y, w, h) in rects]
        except Exception as e:  # noqa: BLE001
            log.debug("_detect_faces: %s", e)
            return []

    # ==================================================================
    #                       IDENTIFICAR
    # ==================================================================
    def identify(self, frame: Any) -> list[FaceMatch]:
        """Devuelve lista de FaceMatch para cada cara en la frame."""
        if not self._available or not self._names:
            return []
        try:
            import cv2  # noqa: F401
        except ImportError:
            return []

        detected = self._detect_faces(frame)
        results: list[FaceMatch] = []
        for face_img, bbox in detected:
            try:
                import cv2
                resized = cv2.resize(face_img, _FACE_INPUT_SIZE)
                label, confidence = self._recognizer.predict(resized)
                is_known = (
                    confidence < _LBPH_CONFIDENCE_THRESHOLD
                    and int(label) in self._names
                )
                name = (
                    self._names[int(label)] if is_known else "desconocido"
                )
                results.append(FaceMatch(
                    name=name,
                    confidence=float(confidence),
                    bbox=bbox,
                    is_known=is_known,
                ))
            except Exception as e:  # noqa: BLE001
                log.debug("predict: %s", e)
        return results

    # ==================================================================
    #                       REGISTRAR
    # ==================================================================
    def register(self, name: str, frames: list[Any]) -> int:
        """Añade caras al modelo. Devuelve N caras efectivamente añadidas."""
        if not self._available:
            return 0
        try:
            import cv2  # noqa: F401
            import numpy as np
        except ImportError:
            return 0

        # Asignamos label_id estable por nombre — si ya existe, reusamos
        # para que el operador pueda hacer registros incrementales.
        label_id = self._label_for(name)

        samples: list[Any] = []
        labels_list: list[int] = []
        for frame in frames:
            detected = self._detect_faces(frame)
            if not detected:
                continue
            # Tomamos la cara más grande (asumimos que el operador es la
            # más cercana a la cámara durante el registro).
            detected.sort(key=lambda d: d[1][2] * d[1][3], reverse=True)
            face_img, _ = detected[0]
            try:
                import cv2
                resized = cv2.resize(face_img, _FACE_INPUT_SIZE)
                samples.append(resized)
                labels_list.append(label_id)
            except Exception as e:  # noqa: BLE001
                log.debug("resize: %s", e)

        if not samples:
            log.warning("register(%s): 0 caras detectadas en %d frames",
                        name, len(frames))
            return 0

        try:
            arr = np.array(labels_list)
            if self._recognizer.empty() if hasattr(self._recognizer, "empty") else not self._names:
                self._recognizer.train(samples, arr)
            else:
                # update() en LBPH añade sin olvidar lo previo.
                self._recognizer.update(samples, arr)
        except Exception as e:  # noqa: BLE001
            log.warning("LBPH train/update falló: %s", e)
            return 0

        self._names[label_id] = name
        self._persist()
        log.info(
            "Cara registrada: %s (label=%d, %d samples)",
            name, label_id, len(samples),
        )
        return len(samples)

    # ==================================================================
    #                       PERSISTENCIA
    # ==================================================================
    def _persist(self) -> None:
        try:
            self.faces_dir.mkdir(parents=True, exist_ok=True)
            self._recognizer.write(str(self.model_path))
            self.names_path.write_text(
                json.dumps(
                    {str(k): v for k, v in self._names.items()},
                    indent=2,
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )
        except OSError as e:
            log.warning("persist face model: %s", e)

    def _label_for(self, name: str) -> int:
        """Devuelve label_id para `name`, asignando uno nuevo si no existe."""
        for lid, nm in self._names.items():
            if nm.lower() == name.lower():
                return lid
        return max(self._names.keys(), default=-1) + 1
