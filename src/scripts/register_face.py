"""
==============================================================================
 Proyecto MITOS - Script de registro de cara
==============================================================================

Uso:
    python -m src.scripts.register_face <nombre> [n_frames]

Captura `n_frames` (default 30) de la cámara, detecta caras, las usa
para entrenar / actualizar el modelo LBPH en `data/faces/`. Después
el DialogLoop podrá distinguir esta cara cuando aparezca en la cámara.

Requisitos:
    pip install opencv-contrib-python

Ejemplos:
    # Registrarte
    python -m src.scripts.register_face Ignacio

    # Registrar a otra persona con más frames
    python -m src.scripts.register_face Maria 50
==============================================================================
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

_DEFAULT_FRAMES: int = 30
_CAPTURE_INTERVAL_S: float = 0.3  # entre frames


def main(argv: list[str] | None = None) -> int:
    argv = argv or sys.argv[1:]
    if not argv:
        print(
            "Uso: python -m src.scripts.register_face <nombre> [n_frames]",
            file=sys.stderr,
        )
        return 1
    name = argv[0]
    n_frames = int(argv[1]) if len(argv) > 1 else _DEFAULT_FRAMES

    try:
        import cv2
    except ImportError:
        print(
            "ERROR: opencv no instalado.\n"
            "  pip install opencv-contrib-python",
            file=sys.stderr,
        )
        return 1

    # Resolver raíz del proyecto.
    here = Path(__file__).resolve()
    project_root = here.parent.parent.parent  # src/scripts → repo

    from src.core.face_recognizer import FaceRecognizer
    fr = FaceRecognizer(project_root=project_root)
    if not fr.is_available():
        print(
            "ERROR: FaceRecognizer no disponible. "
            "Asegúrate de tener opencv-contrib-python (no opencv-python pelado).",
            file=sys.stderr,
        )
        return 1

    print(f"Voy a capturar {n_frames} frames de tu cara para registrar '{name}'.")
    print("Posiciónate frente a la cámara, mueve un poco la cabeza para variedad.")
    print("Empezamos en...")
    for n in (3, 2, 1):
        print(f"  {n}...", end=" ", flush=True)
        time.sleep(1.0)
    print("CAPTURANDO")

    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    if not cap.isOpened():
        print("ERROR: no pude abrir la cámara índice 0", file=sys.stderr)
        return 1

    frames: list = []
    captured_with_face = 0
    try:
        for i in range(n_frames * 2):  # tomamos el doble por si fallan
            ok, frame = cap.read()
            if not ok:
                continue
            frames.append(frame)
            faces_in_frame = fr._detect_faces(frame)
            if faces_in_frame:
                captured_with_face += 1
            print(
                f"  frame {i + 1}: {'cara' if faces_in_frame else 'sin cara'} "
                f"  (con cara útiles: {captured_with_face}/{n_frames})"
            )
            if captured_with_face >= n_frames:
                break
            time.sleep(_CAPTURE_INTERVAL_S)
    finally:
        cap.release()

    if captured_with_face == 0:
        print(
            "\nNo detecté ninguna cara en las capturas. "
            "Verifica iluminación + que tu cara esté centrada.",
            file=sys.stderr,
        )
        return 1

    print(f"\nEntrenando LBPH con {captured_with_face} caras...")
    added = fr.register(name=name, frames=frames)
    if added == 0:
        print("ERROR: el entrenamiento no añadió ninguna cara.", file=sys.stderr)
        return 1

    print(f"\n✓ '{name}' registrado con {added} muestras.")
    print(f"  Modelo en:   {fr.model_path}")
    print(f"  Nombres en:  {fr.names_path}")
    print(f"  Conocidos:   {fr.known_names()}")
    print(
        "\nCuando arranques `python -m src.main --voice`, MITOS reconocerá "
        "esta cara cuando aparezca en la cámara."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
