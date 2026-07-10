"""
==============================================================================
 Proyecto MITOS - CapabilityGapDetector (Fase 8+ — MITOS detecta lo que le falta)
==============================================================================

El operador dijo: "se supone que MITOS debe darse cuenta de lo que
queda fuera y hacerlo por su cuenta". Razón total.

Este módulo cierra esa brecha de tres formas:

  1) DETECCIÓN PASIVA en conversaciones: cuando el operador menciona
     una capacidad que MITOS no tiene ("hey MITOS como wake word",
     "controla Spotify directo", "lee emociones reales"), se registra
     como gap automáticamente.

  2) DETECCIÓN ACTIVA desde el sistema: limitaciones del
     LimitationEngine, fallos repetidos del IntelligenceRouter,
     dependencias que no están instaladas.

  3) RECETARIO ENCODE de adquisición: para gaps conocidos hay receta
     (pip install + cuál módulo crear + cómo cablear). Para gaps
     desconocidos, MITOS pide la receta a Gemini, la valida con su
     sanitizador de pip y la persiste para próxima vez.

El SelfImprovementLoop procesa UN gap por ciclo: instala paquete,
verifica que importa, reporta por TTS, persiste en memoria.

Convenciones:
  - Logger `mitos.capability_gaps`.
==============================================================================
"""

from __future__ import annotations

import json
import logging
import re
import threading
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

log = logging.getLogger("mitos.capability_gaps")


_MAX_GAPS_PERSISTED: int = 100


# ============================================================================
# Recetas conocidas — gap_name → cómo adquirirlo
# ============================================================================
@dataclass(frozen=True)
class _Recipe:
    """Cómo MITOS adquiere una capacidad concreta."""

    gap_name: str
    description: str
    pip_packages: tuple[str, ...]
    import_check: str  # nombre de import que verifica que ya hay capacidad
    post_install_note: str = ""  # qué decirle al operador tras instalar


_KNOWN_RECIPES: dict[str, _Recipe] = {
    "wake_word": _Recipe(
        gap_name="wake_word",
        description="Detección de 'Hey MITOS' como wake word offline",
        pip_packages=("pvporcupine",),
        import_check="pvporcupine",
        post_install_note=(
            "Necesitas un PV_ACCESS_KEY gratuito de "
            "https://console.picovoice.ai para que pvporcupine funcione."
        ),
    ),
    "deep_emotions": _Recipe(
        gap_name="deep_emotions",
        description="Detección de 7 emociones faciales (FER pre-trained)",
        pip_packages=("fer",),
        import_check="fer",
        post_install_note=(
            "FER pesa ~500MB con TensorFlow. La primera detección será lenta."
        ),
    ),
    "spotify_control": _Recipe(
        gap_name="spotify_control",
        description="Control directo de Spotify (no solo abrir YouTube)",
        pip_packages=("spotipy",),
        import_check="spotipy",
        post_install_note=(
            "spotipy requiere SPOTIPY_CLIENT_ID + SPOTIPY_CLIENT_SECRET "
            "(cuenta dev gratuita en developer.spotify.com)."
        ),
    ),
    "audio_playback": _Recipe(
        gap_name="audio_playback",
        description="Reproducción local de archivos de audio sin navegador",
        pip_packages=("playsound==1.2.2",),  # 1.3 está roto en Windows
        import_check="playsound",
    ),
    "screen_control": _Recipe(
        gap_name="screen_control",
        description="Control de teclado/ratón programático",
        pip_packages=("pyautogui",),
        import_check="pyautogui",
    ),
    "web_scraping": _Recipe(
        gap_name="web_scraping",
        description="Lectura de páginas web (más allá de abrirlas)",
        pip_packages=("requests", "beautifulsoup4"),
        import_check="bs4",
    ),
    "ocr": _Recipe(
        gap_name="ocr",
        description="OCR — leer texto de imágenes y pantalla",
        pip_packages=("pytesseract",),
        import_check="pytesseract",
        post_install_note=(
            "pytesseract requiere también el binario Tesseract instalado: "
            "https://github.com/UB-Mannheim/tesseract/wiki"
        ),
    ),
}


# ============================================================================
# Patrones de detección de gap desde la conversación
# ============================================================================
_UTTERANCE_PATTERNS: tuple[tuple[str, str], ...] = (
    # (regex, gap_name)
    (r"\b(wake\s*word|hey\s*mitos|palabra\s*clave|palabra\s*despertar)\b", "wake_word"),
    (r"\b(emociones?\s+(reales|profundas|complejas|m[áa]s)|fer|deepface)\b", "deep_emotions"),
    (r"\b(spotify|reproductor\s+real|controla(r)?\s+spotify)\b", "spotify_control"),
    (r"\b(reproduce\s+(este\s+)?archivo|toca\s+este\s+\.mp3|playsound)\b", "audio_playback"),
    (r"\b(controla(r)?\s+(el\s+)?(teclado|rat[óo]n|mouse)|automatiza\s+clicks?)\b", "screen_control"),
    (r"\b(scrape(a|ar|o)?\s+web|lee\s+esta?\s+web|beautifulsoup)\b", "web_scraping"),
    (r"\b(ocr|leer?\s+texto\s+de\s+(la\s+)?(imagen|pantalla|foto)|tesseract)\b", "ocr"),
)


# ============================================================================
# Dataclasses
# ============================================================================
@dataclass
class CapabilityGap:
    """Una capacidad que MITOS detectó que NO tiene."""

    name: str                  # "wake_word" | "deep_emotions" | ...
    description: str           # qué hace
    source: str                # "user_utterance" | "limitation" | "import_check" | "manual"
    detected_at: float = field(default_factory=time.time)
    times_seen: int = 1        # cuántas veces detectado (cada nuevo trigger suma)
    addressed: bool = False    # True tras adquisición exitosa
    last_attempt_at: float = 0.0
    last_attempt_outcome: str = ""


# ============================================================================
# CapabilityGapDetector
# ============================================================================
class CapabilityGapDetector:
    """Detecta + persiste + ofrece gaps al SelfImprovementLoop."""

    def __init__(self, project_root: str | Path = ".") -> None:
        self.root: Path = Path(project_root).resolve()
        self.path: Path = self.root / "data" / "capability_gaps.json"
        # RLock (re-entrante) porque `_register_internal` adquiere y dentro
        # llama a `_persist` que también adquiere. Con `Lock` plano eso era
        # un deadlock — cazado en testing.
        self._lock: threading.RLock = threading.RLock()
        self._gaps: dict[str, CapabilityGap] = {}
        self._load()

    # ==================================================================
    #                       PERSISTENCIA
    # ==================================================================
    def _load(self) -> None:
        if not self.path.is_file():
            return
        try:
            raw = json.loads(self.path.read_text(encoding="utf-8"))
            self._gaps = {
                k: CapabilityGap(**v) for k, v in (raw.get("gaps", {})).items()
            }
            log.info(
                "Gaps cargados de disco: %d (%d sin atender)",
                len(self._gaps), sum(1 for g in self._gaps.values() if not g.addressed),
            )
        except (OSError, ValueError, TypeError) as e:
            log.warning("load capability_gaps: %s — uso vacío", e)
            self._gaps = {}

    def _persist(self) -> None:
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self._lock:
                # Limitamos para no inflar el archivo.
                items = list(self._gaps.items())
                if len(items) > _MAX_GAPS_PERSISTED:
                    items = sorted(
                        items, key=lambda kv: -kv[1].detected_at,
                    )[:_MAX_GAPS_PERSISTED]
                payload = {
                    "gaps": {k: asdict(v) for k, v in items},
                }
            self.path.write_text(
                json.dumps(payload, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
        except OSError as e:
            log.debug("persist capability_gaps: %s", e)

    # ==================================================================
    #                       REGISTRO
    # ==================================================================
    def register_from_user_utterance(self, text: str) -> list[str]:
        """Escanea texto del operador y registra gaps detectados.

        Returns la lista de nombres de gaps NUEVOS detectados (no
        previamente vistos), para que el caller pueda reportar.
        """
        if not text:
            return []
        low = text.lower()
        new_gaps: list[str] = []
        for pattern, gap_name in _UTTERANCE_PATTERNS:
            if re.search(pattern, low):
                was_new = self._register_internal(
                    name=gap_name,
                    description=self._description_for(gap_name),
                    source="user_utterance",
                )
                if was_new:
                    new_gaps.append(gap_name)
        return new_gaps

    # register_from_limitation removido en Fase A (limitation_engine fuera).

    def register_manual(self, name: str, description: str = "") -> bool:
        """Para que el código pueda registrar gaps explícitos."""
        return self._register_internal(
            name=name,
            description=description or self._description_for(name),
            source="manual",
        )

    def _register_internal(
        self, name: str, description: str, source: str,
    ) -> bool:
        """Devuelve True si era nuevo, False si ya existía."""
        with self._lock:
            if name in self._gaps:
                self._gaps[name].times_seen += 1
                self._persist()
                return False
            self._gaps[name] = CapabilityGap(
                name=name, description=description, source=source,
            )
            log.info(
                "GAP NUEVO detectado: %s (source=%s) — %s",
                name, source, description,
            )
            self._persist()
            return True

    # ==================================================================
    #                       CONSULTA / DRAIN
    # ==================================================================
    def next_pending(self) -> CapabilityGap | None:
        """Devuelve el gap pendiente más PRIORITARIO (más visto, no atendido).

        No lo retira — eso lo hace `mark_addressed` o `mark_failed`.
        """
        with self._lock:
            pending = [
                g for g in self._gaps.values()
                if not g.addressed
            ]
            if not pending:
                return None
            # Prioridad: más visto primero, en empate más reciente.
            pending.sort(key=lambda g: (-g.times_seen, -g.detected_at))
            return pending[0]

    def mark_addressed(self, name: str, outcome: str) -> None:
        with self._lock:
            if name in self._gaps:
                self._gaps[name].addressed = True
                self._gaps[name].last_attempt_at = time.time()
                self._gaps[name].last_attempt_outcome = outcome
                self._persist()

    def mark_failed(self, name: str, reason: str) -> None:
        with self._lock:
            if name in self._gaps:
                self._gaps[name].last_attempt_at = time.time()
                self._gaps[name].last_attempt_outcome = f"FAIL: {reason}"
                self._persist()

    def all_gaps(self) -> list[CapabilityGap]:
        with self._lock:
            return list(self._gaps.values())

    def pending_count(self) -> int:
        with self._lock:
            return sum(1 for g in self._gaps.values() if not g.addressed)

    # ==================================================================
    #                       RECETAS
    # ==================================================================
    @staticmethod
    def recipe_for(gap_name: str) -> _Recipe | None:
        return _KNOWN_RECIPES.get(gap_name)

    @staticmethod
    def known_gap_names() -> list[str]:
        return list(_KNOWN_RECIPES.keys())

    @staticmethod
    def _description_for(name: str) -> str:
        recipe = _KNOWN_RECIPES.get(name)
        return recipe.description if recipe else name
