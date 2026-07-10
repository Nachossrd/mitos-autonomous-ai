"""
==============================================================================
 Proyecto MITOS - Importador de voz desde archivo
==============================================================================

Toma un archivo de audio (.ogg, .wav, .mp3, .m4a, .opus, etc.) y lo
convierte en `data/speaker_profile.npy` para que MITOS reconozca al
hablante en --voice.

Uso:
    python -m src.scripts.import_voice_file <ruta_audio> [--name Ignacio]

Pipeline:
  1. Decodifica con soundfile (preferido), librosa (fallback) o
     ffmpeg+subprocess (último recurso).
  2. Resamplea a 16 kHz mono float32 (lo que resemblyzer espera).
  3. Calcula embedding 256-dim con resemblyzer.
  4. Guarda en data/speaker_profile.npy.

Convenciones:
  - Logger `mitos.import_voice`.
==============================================================================
"""

from __future__ import annotations

import argparse
import logging
import sys
import tempfile
from pathlib import Path
from typing import Any

log = logging.getLogger("mitos.import_voice")

_TARGET_SR: int = 16000


def _decode_audio(path: Path) -> tuple[Any, int]:
    """Devuelve (samples_float32_mono, sample_rate). Levanta si no puede."""
    # 1) soundfile — más rápido, sin deps externas si libsndfile soporta el formato.
    try:
        import soundfile as sf
        data, sr = sf.read(str(path), always_2d=False, dtype="float32")
        # Mono.
        if data.ndim == 2:
            data = data.mean(axis=1).astype("float32")
        return data, int(sr)
    except Exception as e:  # noqa: BLE001
        log.debug("soundfile.read falló: %s", e)

    # 2) librosa — maneja casi cualquier formato vía audioread/ffmpeg.
    try:
        import librosa
        data, sr = librosa.load(str(path), sr=None, mono=True)
        return data.astype("float32"), int(sr)
    except Exception as e:  # noqa: BLE001
        log.debug("librosa.load falló: %s", e)

    # 3) ffmpeg → wav temporal → soundfile.
    import shutil
    import subprocess
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        raise RuntimeError(
            "No pude decodificar el audio. Instala uno de: "
            "`pip install soundfile librosa` o ffmpeg en PATH."
        )
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        tmp_path = Path(tmp.name)
    try:
        result = subprocess.run(
            [
                ffmpeg, "-y", "-i", str(path),
                "-ac", "1", "-ar", str(_TARGET_SR),
                "-acodec", "pcm_s16le", str(tmp_path),
            ],
            capture_output=True, text=True, timeout=120,
        )
        if result.returncode != 0:
            raise RuntimeError(
                f"ffmpeg falló: {(result.stderr or '')[-200:]}"
            )
        import soundfile as sf
        data, sr = sf.read(str(tmp_path), always_2d=False, dtype="float32")
        if data.ndim == 2:
            data = data.mean(axis=1).astype("float32")
        return data, int(sr)
    finally:
        try:
            tmp_path.unlink(missing_ok=True)
        except OSError:
            pass


def _resample_to(audio: Any, source_sr: int, target_sr: int) -> Any:
    """Resampleo simple (librosa.resample si está; sino numpy decimación)."""
    if source_sr == target_sr:
        return audio
    try:
        import librosa
        return librosa.resample(audio, orig_sr=source_sr, target_sr=target_sr)
    except ImportError:
        pass
    # Fallback aritmético — funciona para downsample, peor calidad.
    import numpy as np
    ratio = target_sr / source_sr
    n_new = int(len(audio) * ratio)
    idx = (np.arange(n_new) / ratio).astype("int64")
    idx = np.clip(idx, 0, len(audio) - 1)
    return audio[idx]


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)-7s  %(message)s",
    )
    parser = argparse.ArgumentParser(
        prog="import_voice_file",
        description="Importa un archivo de audio como perfil de voz del operador.",
    )
    parser.add_argument("audio_path", help="Ruta al archivo (.ogg/.wav/.mp3/...)")
    parser.add_argument(
        "--name", default="Ignacio",
        help="Nombre del operador (informativo, default Ignacio)",
    )
    parser.add_argument(
        "--output", default=None,
        help="Ruta de salida del .npy (default data/speaker_profile.npy)",
    )
    args = parser.parse_args(argv)

    audio_path = Path(args.audio_path).expanduser().resolve()
    if not audio_path.is_file():
        log.error("No existe: %s", audio_path)
        return 1

    # Output por defecto: <repo>/data/speaker_profile.npy
    here = Path(__file__).resolve()
    project_root = here.parent.parent.parent
    output_path = (
        Path(args.output) if args.output
        else project_root / "data" / "speaker_profile.npy"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)

    log.info("Decodificando %s …", audio_path.name)
    try:
        audio, sr = _decode_audio(audio_path)
    except RuntimeError as e:
        log.error("%s", e)
        return 1
    log.info("  → %d muestras @ %d Hz (%.1fs)", len(audio), sr, len(audio) / sr)

    log.info("Resampleando a %d Hz …", _TARGET_SR)
    audio = _resample_to(audio, sr, _TARGET_SR)
    log.info("  → %d muestras", len(audio))

    try:
        from resemblyzer import VoiceEncoder, preprocess_wav
    except ImportError:
        log.error(
            "resemblyzer no instalado. Instala con:\n"
            "  pip install resemblyzer"
        )
        return 1

    log.info("Cargando VoiceEncoder de resemblyzer…")
    try:
        encoder = VoiceEncoder()
    except Exception as e:  # noqa: BLE001
        log.error("VoiceEncoder falló: %s", e)
        return 1

    try:
        wav = preprocess_wav(audio, source_sr=_TARGET_SR)
        log.info("Calculando embedding (256-dim)…")
        embed = encoder.embed_utterance(wav)
    except Exception as e:  # noqa: BLE001
        log.error("Encoding falló: %s", e)
        return 1

    import numpy as np
    np.save(output_path, embed)
    log.info("✓ Embedding guardado en %s", output_path)
    log.info("  shape=%s  norm=%.3f", embed.shape, float(np.linalg.norm(embed)))
    log.info(
        "Operador registrado como '%s'. Al arrancar --voice, MITOS te "
        "reconocerá por similitud cosine > 0.75.",
        args.name,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
