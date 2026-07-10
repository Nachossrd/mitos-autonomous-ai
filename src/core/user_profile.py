"""
==============================================================================
 Proyecto MITOS - UserProfile (Fase 8+ — MITOS aprende QUIÉN eres)
==============================================================================

MITOS escucha por días/semanas — toda esa información sobre tus
gustos, temas que mencionas, estado de ánimo, canciones que pides...
SE PIERDE si no la persistimos en un modelo del operador.

Este módulo construye y mantiene un perfil VIVO del operador:

  - mood_history: histórico de estados de ánimo (de voz + cara)
  - mentioned_topics: contadores de temas
  - music_likes: artistas/canciones mencionadas
  - times_of_day: cuándo interactúa normalmente
  - projects_referenced: a qué proyectos se refiere más

Se persiste en `data/user_profile.json`. Se actualiza con cada
utterance vía `update_from_utterance`. Se consulta con `recommend_*`
para que MITOS pueda PROACTIVAMENTE ofrecer cosas (música, retomar
un tema, etc.) basándose en lo aprendido.

Convenciones:
  - Logger `mitos.user_profile`.
==============================================================================
"""

from __future__ import annotations

import json
import logging
import re
import time
from collections import Counter
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

log = logging.getLogger("mitos.user_profile")


_MOOD_HISTORY_MAX: int = 100
_MUSIC_LIKES_MAX: int = 50

# Palabras clave para detectar dominio.
_MUSIC_HINTS: tuple[str, ...] = (
    "canción", "cancion", "música", "musica", "spotify", "youtube",
    "reproduce", "pon ", "escuchar", "playlist", "álbum", "album",
    "artista", "banda", "grupo",
)
_MOOD_NEGATIVE_HINTS: tuple[str, ...] = (
    "cansado", "agotado", "frustrado", "harto", "estresado",
    "triste", "mal", "horrible", "fatal", "muerto",
)
_MOOD_POSITIVE_HINTS: tuple[str, ...] = (
    "contento", "feliz", "bien", "genial", "perfecto", "increíble",
    "increible", "emocionado", "motivado",
)
_MOOD_TIRED_HINTS: tuple[str, ...] = (
    "sueño", "dormir", "descansar", "siesta", "fatigado",
)
_STOPWORDS: frozenset[str] = frozenset({
    "para", "como", "pero", "este", "esta", "esto", "esos", "esas",
    "porque", "cuando", "donde", "donde", "puedes", "puede", "hacer",
    "tengo", "tienes", "tiene", "decir", "dime", "saber", "quiero",
    "quieres", "vamos", "estar", "estoy", "estamos", "siempre",
    "nunca", "ahora", "luego", "antes", "tambien", "también",
})


@dataclass
class _MoodPoint:
    timestamp: float
    label: str           # "positive" | "negative" | "tired" | "neutral"
    source: str          # "voice_tone" | "face" | "text" | "self_report"
    intensity: float     # [0, 1]


@dataclass
class UserProfileData:
    operator_name: str = "Ignacio"
    mood_history: list[_MoodPoint] = field(default_factory=list)
    mentioned_topics: dict[str, int] = field(default_factory=dict)
    music_likes: dict[str, int] = field(default_factory=dict)
    projects_referenced: dict[str, int] = field(default_factory=dict)
    times_of_day_interaction: dict[int, int] = field(default_factory=dict)  # hour → count
    first_seen_ts: float = field(default_factory=time.time)
    last_seen_ts: float = field(default_factory=time.time)
    total_utterances: int = 0


class UserProfile:
    """Persistencia + heurísticas para aprender al operador."""

    def __init__(
        self,
        project_root: str | Path = ".",
        operator_name: str = "Ignacio",
    ) -> None:
        self.root: Path = Path(project_root).resolve()
        self.path: Path = self.root / "data" / "user_profile.json"
        self.data: UserProfileData = self._load_or_init(operator_name)
        log.info(
            "UserProfile cargado: %s (%d utterances, %d moods)",
            self.data.operator_name,
            self.data.total_utterances,
            len(self.data.mood_history),
        )

    # ==================================================================
    #                       PERSISTENCIA
    # ==================================================================
    def _load_or_init(self, name: str) -> UserProfileData:
        if not self.path.is_file():
            return UserProfileData(operator_name=name)
        try:
            raw = json.loads(self.path.read_text(encoding="utf-8"))
            data = UserProfileData(
                operator_name=raw.get("operator_name", name),
                mood_history=[
                    _MoodPoint(**m) for m in raw.get("mood_history", [])
                ],
                mentioned_topics=dict(raw.get("mentioned_topics", {})),
                music_likes=dict(raw.get("music_likes", {})),
                projects_referenced=dict(raw.get("projects_referenced", {})),
                times_of_day_interaction={
                    int(k): int(v) for k, v in
                    raw.get("times_of_day_interaction", {}).items()
                },
                first_seen_ts=float(raw.get("first_seen_ts", time.time())),
                last_seen_ts=float(raw.get("last_seen_ts", time.time())),
                total_utterances=int(raw.get("total_utterances", 0)),
            )
            return data
        except (OSError, ValueError, TypeError) as e:
            log.warning("load user_profile.json falló: %s — uso vacío", e)
            return UserProfileData(operator_name=name)

    def _persist(self) -> None:
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            payload = {
                "operator_name": self.data.operator_name,
                "mood_history": [asdict(m) for m in self.data.mood_history],
                "mentioned_topics": dict(self.data.mentioned_topics),
                "music_likes": dict(self.data.music_likes),
                "projects_referenced": dict(self.data.projects_referenced),
                "times_of_day_interaction": dict(
                    self.data.times_of_day_interaction
                ),
                "first_seen_ts": self.data.first_seen_ts,
                "last_seen_ts": self.data.last_seen_ts,
                "total_utterances": self.data.total_utterances,
            }
            self.path.write_text(
                json.dumps(payload, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
        except OSError as e:
            log.debug("persist user_profile: %s", e)

    # ==================================================================
    #                       INGESTA
    # ==================================================================
    def update_from_utterance(self, text: str) -> None:
        """Procesa el texto del operador y actualiza el perfil."""
        if not text:
            return
        self.data.total_utterances += 1
        self.data.last_seen_ts = time.time()

        # Hora del día.
        hour = int(time.localtime().tm_hour)
        self.data.times_of_day_interaction[hour] = (
            self.data.times_of_day_interaction.get(hour, 0) + 1
        )

        # Mood por palabras clave del propio operador.
        mood_label = self._infer_mood_from_text(text)
        if mood_label != "neutral":
            self.add_mood(label=mood_label, source="text", intensity=0.6)

        # Topics: palabras de >5 chars, no stopwords.
        for w in self._extract_topics(text):
            self.data.mentioned_topics[w] = (
                self.data.mentioned_topics.get(w, 0) + 1
            )

        # Música: si menciona música, extraemos lo que parezca título/artista.
        if self._mentions_music(text):
            for hint in self._extract_music_mentions(text):
                self.data.music_likes[hint] = (
                    self.data.music_likes.get(hint, 0) + 1
                )

        # Proyectos: nombre capitalizado mencionado.
        for match in re.findall(r"\b[A-Z][a-zA-Z]{3,}\b", text):
            if match.lower() not in {"mitos", "ignacio", "youtube", "google"}:
                self.data.projects_referenced[match] = (
                    self.data.projects_referenced.get(match, 0) + 1
                )

        self._persist()

    def add_mood(
        self,
        label: str,
        source: str,
        intensity: float = 0.5,
    ) -> None:
        """Registra una observación de estado de ánimo."""
        self.data.mood_history.append(_MoodPoint(
            timestamp=time.time(),
            label=label,
            source=source,
            intensity=float(max(0.0, min(1.0, intensity))),
        ))
        if len(self.data.mood_history) > _MOOD_HISTORY_MAX:
            self.data.mood_history = self.data.mood_history[-_MOOD_HISTORY_MAX:]
        self._persist()

    # ==================================================================
    #                       CONSULTAS
    # ==================================================================
    def current_mood(self, window_minutes: float = 30.0) -> str:
        """Mood dominante en los últimos `window_minutes`."""
        cutoff = time.time() - window_minutes * 60
        recent = [m for m in self.data.mood_history if m.timestamp >= cutoff]
        if not recent:
            return "neutral"
        counts = Counter(m.label for m in recent)
        label, _ = counts.most_common(1)[0]
        return label

    def top_topics(self, n: int = 5) -> list[tuple[str, int]]:
        return Counter(self.data.mentioned_topics).most_common(n)

    def top_music(self, n: int = 5) -> list[tuple[str, int]]:
        return Counter(self.data.music_likes).most_common(n)

    def recommend_music(self) -> str | None:
        """Devuelve una query de YouTube basada en perfil + mood.

        Solo devuelve algo si el favorito tiene CALIDAD (>=4 chars y al
        menos 2 menciones). Si solo hay basura, devuelve None y MITOS
        debería preguntar al operador qué quiere en lugar de inventar.
        """
        likes = self.top_music(3)
        if not likes:
            return None
        favorite, count = likes[0]
        if len(favorite) < 4 or count < 2:
            return None
        mood = self.current_mood()
        if mood == "negative":
            return f"canción relajante {favorite}"
        if mood == "tired":
            return f"{favorite} acústico relajado"
        if mood == "positive":
            return f"{favorite}"
        return favorite

    def summary(self) -> str:
        topics = ", ".join(t for t, _ in self.top_topics(3)) or "(ninguno)"
        music = ", ".join(m for m, _ in self.top_music(3)) or "(ninguno)"
        return (
            f"Operador: {self.data.operator_name} | "
            f"Utterances: {self.data.total_utterances} | "
            f"Mood reciente: {self.current_mood()} | "
            f"Top temas: {topics} | "
            f"Top música: {music}"
        )

    # ==================================================================
    #                       HEURÍSTICAS PRIVADAS
    # ==================================================================
    @staticmethod
    def _infer_mood_from_text(text: str) -> str:
        low = text.lower()
        if any(w in low for w in _MOOD_NEGATIVE_HINTS):
            return "negative"
        if any(w in low for w in _MOOD_TIRED_HINTS):
            return "tired"
        if any(w in low for w in _MOOD_POSITIVE_HINTS):
            return "positive"
        return "neutral"

    @staticmethod
    def _extract_topics(text: str) -> list[str]:
        return [
            w for w in re.findall(r"[a-záéíóúñ]{6,}", text.lower())
            if w not in _STOPWORDS
        ]

    @staticmethod
    def _mentions_music(text: str) -> bool:
        low = text.lower()
        return any(h in low for h in _MUSIC_HINTS)

    @staticmethod
    def _extract_music_mentions(text: str) -> list[str]:
        """Extrae lo que sigue a 'pon X', 'reproduce X', etc.

        Filtros de basura:
          - mínimo 3 chars (descarta 'ti', 'mi', 'yo', 'lo')
          - mínimo 1 palabra con >=4 chars (descarta 'a ti', 'a mi')
          - no es solo stopwords coloquiales
        """
        garbage = {
            "ti", "mi", "yo", "lo", "le", "se", "te", "me", "ya",
            "eso", "esto", "esa", "este", "esta", "algo", "alguno",
            "nada", "nadie", "todo", "todos",
            "a ti", "a mi", "a el", "a ella", "a nosotros",
            "algo de", "un poco", "música", "musica", "una canción",
            "la canción", "un tema", "el tema",
        }
        results: list[str] = []
        patterns = (
            r"pon(?:me)?\s+(.+?)(?:[\.\?,]|$)",
            r"reproduce\s+(.+?)(?:[\.\?,]|$)",
            r"escuchar\s+(.+?)(?:[\.\?,]|$)",
            r"playlist\s+de\s+(.+?)(?:[\.\?,]|$)",
            r"col[óo]ca(?:me|le)?\s+(.+?)(?:[\.\?,]|$)",
        )
        for pat in patterns:
            for m in re.findall(pat, text, flags=re.IGNORECASE):
                cleaned = m.strip().strip(".,;:!?\"'`").lower()
                if len(cleaned) < 3:
                    continue
                if cleaned in garbage:
                    continue
                # Que tenga al menos UNA palabra de >=4 chars (descarta
                # "no me" o "a un par" como nombres de artistas).
                words = cleaned.split()
                if not any(len(w) >= 4 for w in words):
                    continue
                if 3 <= len(cleaned) <= 80:
                    results.append(cleaned)
        return results
