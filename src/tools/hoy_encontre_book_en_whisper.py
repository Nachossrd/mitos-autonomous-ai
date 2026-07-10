"""Auto-generado por MITOS ToolBuilder.

  Operador pidió: hoy encontré un book en el whisper de transcripción. cuando te dije desarrollar una herramienta, no la desarrollaste
  Proveedor: gemini (gemini-3.5-flash)
  Generado: 2026-06-21 10:32:07
  Válido Python: True
"""

# pip install openai-whisper torch pydub

import logging
import pathlib
import tempfile
from typing import Any, Generator
import whisper
import torch
from pydub import AudioSegment

# Configuración de logging para producción
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("WhisperTranscriber")


class WhisperTranscriptionError(Exception):
    """Excepción personalizada para errores del módulo de transcripción."""
    pass


class RobustWhisperTranscriber:
    """
    Clase de transcripción robusta utilizando OpenAI Whisper local.
    Resuelve "bugs" comunes de Whisper como:
    1. Bucles de repetición infinita (hallucination loops).
    2. Alucinaciones en silencios prolongados.
    3. Consumo descontrolado de VRAM/RAM.
    4. Falta de soporte para formatos de audio no estándar.
    """

    def __init__(self, model_name: str = "base", device: str | None = None) -> None:
        """
        Inicializa el modelo de Whisper de forma segura.
        
        :param model_name: Nombre del modelo ('tiny', 'base', 'small', 'medium', 'large').
        :param device: Dispositivo de ejecución ('cuda', 'cpu', 'mps'). Si es None, se autodetecta.
        """
        self.model_name = model_name
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
            
        logger.info(f"Inicializando modelo Whisper '{self.model_name}' en dispositivo '{self.device}'...")
        try:
            self.model = whisper.load_model(self.model_name, device=self.device)
            logger.info("Modelo cargado exitosamente.")
        except Exception as e:
            logger.error(f"Error al cargar el modelo Whisper: {e}")
            raise WhisperTranscriptionError(f"No se pudo inicializar el modelo: {e}") from e

    def preprocess_audio(self, input_path: pathlib.Path) -> pathlib.Path:
        """
        Normaliza el audio de entrada a un formato óptimo para Whisper (WAV, 16000Hz, Mono).
        Esto previene fallos de decodificación y reduce el uso de memoria.
        """
        if not input_path.exists():
            raise FileNotFoundError(f"El archivo de audio no existe: {input_path}")

        try:
            logger.info(f"Preprocesando audio: {input_path}")
            audio = AudioSegment.from_file(str(input_path))
            
            # Convertir a mono y 16000Hz (estándar de Whisper)
            audio = audio.set_frame_rate(16000).set_channels(1)
            
            # Guardar en un archivo temporal seguro
            temp_file = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
            temp_path = pathlib.Path(temp_file.name)
            temp_file.close()
            
            audio.export(temp_path, format="wav")
            logger.info(f"Audio normalizado guardado temporalmente en: {temp_path}")
            return temp_path
        except Exception as e:
            logger.error(f"Error durante el preprocesamiento de audio: {e}")
            raise WhisperTranscriptionError(f"Error al procesar el archivo de audio: {e}") from e

    def _clean_hallucinations(self, segments: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """
        Filtra alucinaciones comunes de Whisper (repeticiones infinitas en silencios
        y frases fantasma como 'Subtítulos por la comunidad...').
        """
        cleaned_segments = []
        consecutive_repeats = 0
        last_text = ""

        # Frases típicas que Whisper alucina cuando hay silencio
        hallucination_blacklist = {
            "subtítulos por la comunidad",
            "gracias por ver",
            "thank you for watching",
            "thank you",
            "gracias",
            "reproducido por",
            "asociación de",
            "subtitles by",
            "y"
        }

        for seg in segments:
            text = seg.get("text", "").strip()
            text_lower = text.lower().strip(".,!? ")
            duration = seg.get("end", 0) - seg.get("start", 0)

            # 1. Filtrar frases vacías o extremadamente cortas sin contenido real
            if not text_lower:
                continue

            # 2. Filtrar frases de la blacklist si ocurren en segmentos sospechosamente largos/cortos
            if any(phrase in text_lower for phrase in hallucination_blacklist) and duration > 3.0:
                logger.warning(f"Alucinación detectada y eliminada: '{text}'")
                continue

            # 3. Detectar bucles de repetición (Whisper Loop Bug)
            if text_lower == last_text:
                consecutive_repeats += 1
            else:
                consecutive_repeats = 0
                last_text = text_lower

            if consecutive_repeats > 2:
                logger.warning(f"Bucle de repetición detectado. Omitiendo segmento: '{text}'")
                continue

            cleaned_segments.append(seg)

        return cleaned_segments

    def transcribe(
        self, 
        audio_path: str | pathlib.Path, 
        language: str | None = None,
        beam_size: int = 5
    ) -> dict[str, Any]:
        """
        Transcribe un archivo de audio de manera robusta.
        
        :param audio_path: Ruta al archivo de audio.
        :param language: Código de idioma (ej. 'es', 'en'). Si es None, se autodetecta.
        :param beam_size: Tamaño del beam search para mejorar precisión y reducir bucles.
        :return: Diccionario con el texto completo y los segmentos limpios.
        """
        path = pathlib.Path(audio_path)
        temp_audio_path = None
        
        try:
            # 1. Preprocesar audio para evitar fallos de formato
            temp_audio_path = self.preprocess_audio(path)
            
            # 2. Configurar parámetros de transcripción defensivos
            options = {
                "language": language,
                "beam_size": beam_size,
                "best_of": 5,
                "temperature": 0.0,  # 0.0 reduce la aleatoriedad y previene alucinaciones
                "condition_on_previous_text": False,  # Evita que un error/bucle se propague al siguiente segmento
            }
            
            logger.info("Iniciando transcripción con Whisper...")
            result = self.model.transcribe(str(temp_audio_path), **options)
            
            # 3. Post-procesar segmentos para limpiar bugs de alucinación
            raw_segments = result.get("segments", [])
            cleaned_segments = self._clean_hallucinations(raw_segments)
            
            # Reconstruir el texto completo basado en los segmentos limpios
            full_text = " ".join([seg["text"].strip() for seg in cleaned_segments])
            
            return {
                "text": full_text,
                "segments": cleaned_segments,
                "language": result.get("language", "unknown")
            }
            
        except Exception as e:
            logger.error(f"Fallo catastrófico en la transcripción: {e}")
            raise WhisperTranscriptionError(f"Error durante la transcripción: {e}") from e
            
        finally:
            # Limpieza defensiva del archivo temporal
            if temp_audio_path and temp_audio_path.exists():
                try:
                    temp_audio_path.unlink()
                    logger.info("Archivo temporal de audio eliminado.")
                except Exception as e:
                    logger.warning(f"No se pudo eliminar el archivo temporal {temp_audio_path}: {e}")


def generate_srt(segments: list[dict[str, Any]]) -> str:
    """Genera un string en formato SRT a partir de los segmentos de Whisper."""
    
    def format_time(seconds: float) -> str:
        hrs = int(seconds // 3600)
        mins = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        msecs = int((seconds - int(seconds)) * 1000)
        return f"{hrs:02d}:{mins:02d}:{secs:02d},{msecs:03d}"

    srt_lines = []
    for i, seg in enumerate(segments, start=1):
        start = format_time(seg["start"])
        end = format_time(seg["end"])
        text = seg["text"].strip()
        srt_lines.append(f"{i}\n{start} --> {end}\n{text}\n")
        
    return "\n".join(srt_lines)


if __name__ == "__main__":
    print("--- Iniciando Demostración del Transcriptor Robusto ---")
    
    # Creamos un archivo de audio de prueba silencioso usando pydub para que el script sea autocontenido
    temp_test_audio = pathlib.Path("test_silence_temp.wav")
    try:
        print("Generando audio de prueba (3 segundos de silencio)...")
        silence = AudioSegment.silent(duration=3000)  # 3 segundos
        silence.export(temp_test_audio, format="wav")
        
        # Inicializar transcriptor (usamos el modelo 'tiny' para que sea rápido en el test)
        transcriber = RobustWhisperTranscriber(model_name="tiny")
        
        print(f"Transcribiendo archivo de prueba: {temp_test_audio}...")
        resultado = transcriber.transcribe(temp_test_audio, language="es")
        
        print("\n--- RESULTADOS ---")
        print(f"Idioma detectado: {resultado['language']}")
        print(f"Texto transcrito (Debería estar vacío o casi vacío por ser silencio): '{resultado['text']}'")
        print(f"Número de segmentos válidos: {len(resultado['segments'])}")
        
        # Generar SRT de ejemplo
        srt_content = generate_srt(resultado["segments"])
        print("\n--- FORMATO SRT GENERADO ---")
        print(srt_content if srt_content else "[Sin segmentos para mostrar]")
        
    except Exception as err:
        print(f"Ocurrió un error en la ejecución: {err}")
        
    finally:
        # Limpieza del archivo de prueba
        if temp_test_audio.exists():
            temp_test_audio.unlink()
            print("Archivo de prueba temporal eliminado.")