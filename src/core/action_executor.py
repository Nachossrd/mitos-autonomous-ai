"""
==============================================================================
 Proyecto MITOS - ActionExecutor (Fase 8+ — acciones reales en el sistema)
==============================================================================

Cierra el gap "MITOS solo razona y habla, nunca HACE". Ahora MITOS
puede ejecutar acciones concretas que afectan el sistema del operador:

  - open_url(url)            → navegador
  - play_youtube(query)      → busca y abre el primer resultado
  - open_folder(path)        → explorador de archivos
  - take_screenshot()        → guarda a data/screenshots/
  - search_web(query)        → Google (o motor preferido)
  - run_program(name)        → lanza un .exe / app instalada

Diseño:
  - Cada acción devuelve `ActionResult` con `ok: bool` y `detail: str`.
  - NADA destructivo aquí: no se borra, no se sobreescribe, no se
    apaga el sistema. Acciones destructivas requieren otro módulo
    con confirmación explícita del operador.
  - Cross-platform: detección de OS y delegación a la llamada correcta
    (os.startfile en Windows, xdg-open en Linux, open en macOS).

Convenciones:
  - Logger `mitos.actions`.
==============================================================================
"""

from __future__ import annotations

import logging
import os
import platform
import subprocess
import time
import urllib.parse
import webbrowser
from dataclasses import dataclass
from pathlib import Path
from typing import Any

log = logging.getLogger("mitos.actions")


@dataclass(frozen=True)
class ActionResult:
    """Resultado de UNA acción ejecutada."""

    ok: bool
    action: str             # "open_url" | "play_youtube" | ...
    detail: str             # info para reportar por TTS
    payload: Any = None     # path al screenshot, url abierta, etc.


class ActionExecutor:
    """Ejecutor de acciones del sistema con whitelist segura."""

    def __init__(self, project_root: str | Path = ".") -> None:
        self.root: Path = Path(project_root).resolve()
        self.system: str = platform.system()
        log.debug("ActionExecutor listo (sistema=%s)", self.system)

    # ==================================================================
    #                       NAVEGADOR / URLs
    # ==================================================================
    def open_url(self, url: str) -> ActionResult:
        """Abre `url` en el navegador por defecto del sistema."""
        url = url.strip()
        if not url:
            return ActionResult(False, "open_url", "URL vacía")
        if not url.startswith(("http://", "https://")):
            url = "https://" + url
        try:
            opened = webbrowser.open(url, new=2)
        except Exception as e:  # noqa: BLE001
            return ActionResult(False, "open_url", f"falló: {e}")
        if not opened:
            return ActionResult(False, "open_url", "webbrowser.open devolvió False")
        return ActionResult(True, "open_url", f"abrí {url}", payload=url)

    def play_youtube(self, query: str) -> ActionResult:
        """Busca `query` en YouTube y abre el PRIMER VIDEO directamente.

        Extrae el `videoId` del HTML de resultados (regex sobre el bloque
        ytInitialData) y abre `/watch?v=...` que SÍ autoreproduce. Si no
        encuentra video_id, cae al fallback de página de búsqueda.
        """
        query = query.strip()
        if not query:
            return ActionResult(False, "play_youtube", "query vacía")
        encoded = urllib.parse.quote_plus(query)
        search_url = f"https://www.youtube.com/results?search_query={encoded}"

        video_id = self._first_video_id_for(search_url)
        if video_id:
            watch_url = f"https://www.youtube.com/watch?v={video_id}"
            result = self.open_url(watch_url)
            if result.ok:
                return ActionResult(
                    True, "play_youtube",
                    f"reproduciendo '{query}' (video {video_id})",
                    payload=watch_url,
                )

        # Fallback: si la extracción falló (cambio de HTML de YouTube,
        # sin internet, etc.), abrimos al menos los resultados.
        result = self.open_url(search_url)
        if result.ok:
            return ActionResult(
                True, "play_youtube",
                f"busqué '{query}' en YouTube (sin autoreproducir)",
                payload=search_url,
            )
        return result

    @staticmethod
    def _first_video_id_for(search_url: str) -> str | None:
        """Descarga el HTML de YouTube y extrae el primer videoId."""
        try:
            import urllib.request
            req = urllib.request.Request(
                search_url,
                headers={
                    "User-Agent": (
                        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                        "AppleWebKit/537.36 (KHTML, like Gecko) "
                        "Chrome/120.0 Safari/537.36"
                    ),
                    "Accept-Language": "es-ES,es;q=0.9,en;q=0.8",
                },
            )
            with urllib.request.urlopen(req, timeout=8) as resp:
                html = resp.read().decode("utf-8", errors="ignore")
        except Exception as e:  # noqa: BLE001
            log.debug("YouTube fetch falló: %s", e)
            return None

        # YouTube serializa videos como "videoId":"XXXXXXXXXXX" en el bloque
        # ytInitialData. Tomamos el primer match — la primera ocurrencia es
        # casi siempre el primer resultado real (los anuncios usan otro key).
        import re
        match = re.search(r'"videoId":"([A-Za-z0-9_-]{11})"', html)
        if match:
            return match.group(1)
        return None

    def search_web(self, query: str) -> ActionResult:
        """Búsqueda en Google."""
        query = query.strip()
        if not query:
            return ActionResult(False, "search_web", "query vacía")
        encoded = urllib.parse.quote_plus(query)
        url = f"https://www.google.com/search?q={encoded}"
        result = self.open_url(url)
        if result.ok:
            return ActionResult(
                True, "search_web", f"busqué '{query}' en Google",
                payload=url,
            )
        return result

    # ==================================================================
    #                       FILESYSTEM
    # ==================================================================
    def open_folder(self, path: str | Path) -> ActionResult:
        """Abre `path` en el explorador del sistema."""
        try:
            p = Path(path).expanduser().resolve()
        except (OSError, RuntimeError) as e:
            return ActionResult(False, "open_folder", f"path inválido: {e}")
        if not p.exists():
            return ActionResult(False, "open_folder", f"no existe: {p}")
        try:
            if self.system == "Windows":
                os.startfile(str(p))  # type: ignore[attr-defined]
            elif self.system == "Darwin":
                subprocess.run(["open", str(p)], check=False)
            else:
                subprocess.run(["xdg-open", str(p)], check=False)
        except Exception as e:  # noqa: BLE001
            return ActionResult(False, "open_folder", f"falló: {e}")
        return ActionResult(True, "open_folder", f"abrí {p}", payload=str(p))

    # ==================================================================
    #                       CAPTURA DE PANTALLA
    # ==================================================================
    def take_screenshot(self) -> ActionResult:
        """Captura la pantalla y guarda como PNG en data/screenshots/."""
        out_dir = self.root / "data" / "screenshots"
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"shot_{int(time.time())}.png"

        # Intento 1: PIL.ImageGrab (más portable en Win/Mac).
        try:
            from PIL import ImageGrab
            img = ImageGrab.grab()
            img.save(str(out_path), "PNG")
            return ActionResult(
                True, "take_screenshot",
                f"captura guardada en {out_path.name}",
                payload=str(out_path),
            )
        except ImportError:
            pass
        except Exception as e:  # noqa: BLE001
            log.debug("PIL.ImageGrab falló: %s", e)

        # Intento 2: pyautogui.
        try:
            import pyautogui  # type: ignore[import-not-found]
            img = pyautogui.screenshot()
            img.save(str(out_path), "PNG")
            return ActionResult(
                True, "take_screenshot",
                f"captura guardada en {out_path.name}",
                payload=str(out_path),
            )
        except ImportError:
            pass
        except Exception as e:  # noqa: BLE001
            log.debug("pyautogui falló: %s", e)

        return ActionResult(
            False, "take_screenshot",
            "ninguna librería de captura disponible (instala Pillow o pyautogui)",
        )

    # ==================================================================
    #                       PROGRAMAS
    # ==================================================================
    def run_program(self, name: str) -> ActionResult:
        """Lanza un programa por nombre. Whitelist mínima por seguridad."""
        name = name.strip().lower()
        whitelist = {
            "notepad": "notepad.exe",
            "calc": "calc.exe",
            "calculadora": "calc.exe",
            "explorer": "explorer.exe",
            "explorador": "explorer.exe",
            "chrome": "chrome.exe",
            "firefox": "firefox.exe",
            "spotify": "spotify.exe",
            "code": "code",
            "vscode": "code",
        }
        if name not in whitelist:
            return ActionResult(
                False, "run_program",
                f"'{name}' no está en la lista permitida",
            )
        executable = whitelist[name]
        try:
            subprocess.Popen([executable], shell=False)
        except FileNotFoundError:
            return ActionResult(
                False, "run_program",
                f"no encontré '{executable}' en el sistema",
            )
        except Exception as e:  # noqa: BLE001
            return ActionResult(False, "run_program", f"falló: {e}")
        return ActionResult(True, "run_program", f"lancé {executable}",
                            payload=executable)
