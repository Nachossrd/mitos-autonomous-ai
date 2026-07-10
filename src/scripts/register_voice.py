"""
==============================================================================
 Proyecto MITOS - Script de registro de voz del operador
==============================================================================

Uso:
    python -m src.scripts.register_voice [duracion_segundos]

Captura `duracion` segundos de audio del micrófono, calcula el
embedding de voz con resemblyzer (256-dim) y lo persiste en
`data/speaker_profile.npy`. Después, el DialogLoop comparará cada
utterance entrante con este embedding y reconocerá al operador por
similitud cosine (> 0.75 = match).

Requisitos:
    pip install resemblyzer sounddevice numpy

Si quieres re-registrar (cambiaste de micrófono, te resfriaste, etc.),
simplemente vuelve a ejecutar el script — sobreescribe el .npy.
==============================================================================
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

_DEFAULT_SECONDS: float = 10.0
_SAMPLE_RATE: int = 16000


def main(argv: list[str] | None = None) -> int:
    argv = argv or sys.argv[1:]
    seconds = float(argv[0]) if argv else _DEFAULT_SECONDS

    try:
        import numpy as np
        import sounddevice as sd
    except ImportError:
        print(
            "ERROR: faltan deps. Instala con:\n"
            "  pip install sounddevice numpy",
            file=sys.stderr,
        )
        return 1

    try:
        from resemblyzer import VoiceEncoder, preprocess_wav
    except ImportError:
        print(
            "ERROR: resemblyzer no instalado.\n"
            "  pip install resemblyzer",
            file=sys.stderr,
        )
        return 1

    # Resolver raíz del proyecto y crear data/ si no existe.
    here = Path(__file__).resolve()
    project_root = here.parent.parent.parent  # src/scripts/ → repo
    output_path = project_root / "data" / "speaker_profile.npy"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Voy a grabar {seconds:.0f} segundos de tu voz.")
    print(
        "Habla NATURAL — di varias frases distintas (puede ser cualquier cosa: "
        "lee un párrafo, cuenta tu día, recita...). Cuanto más variado, mejor "
        "es el embedding.\n"
    )
    for n in (3, 2, 1):
        print(f"  {n}...", end=" ", flush=True)
        time.sleep(1.0)
    print("HABLA!")

    try:
        audio = sd.rec(
            int(seconds * _SAMPLE_RATE),
            samplerate=_SAMPLE_RATE,
            channels=1,
            dtype="float32",
        )
        sd.wait()
    except Exception as e:  # noqa: BLE001
        print(f"\nERROR grabando audio: {e}", file=sys.stderr)
        return 1

    print("\nGrabación completa. Calculando embedding...")

    try:
        encoder = VoiceEncoder()
    except Exception as e:  # noqa: BLE001
        print(f"ERROR cargando VoiceEncoder: {e}", file=sys.stderr)
        return 1

    audio_flat = audio.flatten()
    # resemblyzer's preprocess_wav espera 16kHz mono.
    try:
        wav = preprocess_wav(audio_flat, source_sr=_SAMPLE_RATE)
        embed = encoder.embed_utterance(wav)
    except Exception as e:  # noqa: BLE001
        print(f"ERROR calculando embedding: {e}", file=sys.stderr)
        return 1

    np.save(output_path, embed)
    print(f"\n✓ Embedding guardado en {output_path}")
    print(
        "\nLa próxima vez que arranques `python -m src.main --voice`, "
        "MITOS reconocerá tu voz por similitud cosine."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
