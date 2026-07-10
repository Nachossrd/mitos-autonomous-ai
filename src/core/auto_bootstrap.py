"""
==============================================================================
 Proyecto MITOS - AutoBootstrap (Fase 8+ — fin del "instala X y vuelve")
==============================================================================

El operador pidió: "para qué está MITOS si yo le tengo que decir
pip install esto, pip install lo otro". Razonable.

Este módulo, al arrancar el modo --voice, inspecciona qué necesita
MITOS para funcionar al 100% y se lo instala SOLO. No espera permiso
para instalar paquetes que YA están explícitos como dependencias
opcionales del proyecto — son parte del diseño, no instalación de
terceros aleatorios sugeridos por el LLM (esa decisión la sigue
moderando `_validate_pip_name` en daemon.py).

Capabilities:
  - pyttsx3                 → TTS (que MITOS hable)
  - sounddevice + numpy     → mic
  - openai-whisper          → STT local
  - opencv-contrib-python   → visión + face recognition (cv2.face)
  - resemblyzer             → speaker ID (voz)

Si alguno falla, el módulo NO se cae — el resto sigue degradando
grácilmente como antes (TTS off, mic off, etc.). Solo se cae si el
operador explícitamente quiere ese paquete imprescindible.

Convenciones:
  - Logger `mitos.auto_bootstrap`.
==============================================================================
"""

from __future__ import annotations

import logging
import subprocess
import sys
from dataclasses import dataclass

log = logging.getLogger("mitos.auto_bootstrap")


# Tiempo máximo por paquete. 14B llama.cpp = grande, opencv-contrib = grande.
_PIP_TIMEOUT_S: int = 600


@dataclass(frozen=True)
class _PackageSpec:
    """Una dependencia opcional que MITOS puede auto-instalar."""

    pip_name: str       # nombre en PyPI
    import_name: str    # nombre de import en Python
    capability: str     # qué habilita
    conflicts: tuple[str, ...] = ()  # paquetes a desinstalar primero


# El orden importa: `opencv-contrib-python` requiere desinstalar
# `opencv-python` primero (conflictan en cv2.face).
_AUTO_INSTALL_LIST: tuple[_PackageSpec, ...] = (
    # pywin32 PRIMERO: pyttsx3 + pythoncom lo necesitan en Windows.
    # Sin esto, pyttsx3.init en cualquier thread falla con
    # "No module named 'pywintypes'" y se cae al fallback nativo.
    _PackageSpec("pywin32", "pywintypes", "soporte COM Win32 (necesario para SAPI5)"),
    _PackageSpec("pyttsx3", "pyttsx3", "TTS — que MITOS hable"),
    _PackageSpec("sounddevice", "sounddevice", "captura de micrófono"),
    _PackageSpec("numpy", "numpy", "procesamiento numérico de audio"),
    _PackageSpec("openai-whisper", "whisper", "STT local"),
    _PackageSpec(
        "opencv-contrib-python", "cv2",
        "visión + cv2.face (LBPH face recognition)",
        conflicts=("opencv-python",),
    ),
    _PackageSpec("resemblyzer", "resemblyzer", "speaker ID por embedding"),
)


@dataclass
class BootstrapReport:
    installed: list[str]      # paquetes instalados ahora
    already_ok: list[str]     # ya estaban
    failed: list[str]         # falló la instalación
    needs_restart: list[str]  # instalados pero requieren reinicio del proceso


def ensure_all() -> BootstrapReport:
    """Asegura que todas las capabilities estén disponibles.

    Llamar UNA VEZ al inicio del proceso. Retorna el reporte.
    NO levanta excepciones — siempre devuelve un BootstrapReport.
    """
    installed: list[str] = []
    already_ok: list[str] = []
    failed: list[str] = []
    needs_restart: list[str] = []

    for spec in _AUTO_INSTALL_LIST:
        status = _ensure_one(spec)
        if status == "already":
            already_ok.append(spec.pip_name)
        elif status == "installed":
            installed.append(spec.pip_name)
        elif status == "needs_restart":
            needs_restart.append(spec.pip_name)
        else:
            failed.append(spec.pip_name)

    if installed:
        log.info(
            "AutoBootstrap: instalé %d paquete(s): %s",
            len(installed), ", ".join(installed),
        )
    if needs_restart:
        log.warning(
            "AutoBootstrap: %d paquete(s) requieren REINICIO del proceso "
            "para tomar efecto: %s",
            len(needs_restart), ", ".join(needs_restart),
        )
    if failed:
        log.warning(
            "AutoBootstrap: %d paquete(s) FALLARON: %s",
            len(failed), ", ".join(failed),
        )

    return BootstrapReport(
        installed=installed,
        already_ok=already_ok,
        failed=failed,
        needs_restart=needs_restart,
    )


def _ensure_one(spec: _PackageSpec) -> str:
    """Devuelve 'already' | 'installed' | 'failed' | 'needs_restart'."""
    # ¿Ya está importable?
    if _can_import(spec.import_name):
        # Caso especial: opencv. Si cv2 está pero NO tiene face, está mal.
        if spec.import_name == "cv2" and not _cv2_has_face():
            log.info(
                "cv2 importa pero falta submódulo `face` (tienes opencv-python "
                "pelado). Voy a desinstalar AMBOS y reinstalar contrib limpio."
            )
        else:
            return "already"

    # Conflictos: desinstalar TODOS primero (más agresivo).
    # Para opencv: hay que desinstalar opencv-python Y opencv-contrib-python
    # Y opencv-python-headless porque comparten el namespace cv2. Si queda
    # cualquier residuo, el nuevo install no toma efecto bien.
    if spec.import_name == "cv2":
        for opencv_variant in (
            "opencv-python", "opencv-python-headless",
            "opencv-contrib-python", "opencv-contrib-python-headless",
        ):
            _pip_uninstall(opencv_variant)
    else:
        for conflict in spec.conflicts:
            _pip_uninstall(conflict)

    # Instalar.
    if not _pip_install(spec.pip_name):
        return "failed"

    # Para cv2: el módulo ya está cargado en memoria — necesitamos RESTART
    # del proceso para que el nuevo cv2 (con .face) sea importable.
    if spec.import_name == "cv2":
        # Verificamos si el disco tiene el binario correcto SIN cargar cv2
        # nuevamente (porque Python lo cachea).
        import importlib.util
        spec_loc = importlib.util.find_spec("cv2")
        if spec_loc is None:
            return "failed"
        # No podemos verificar cv2.face sin importarlo, y ya está cargado el
        # viejo. Marcamos como needs_restart para que el caller decida.
        return "needs_restart"

    # Re-verificar normalmente.
    if _can_import(spec.import_name):
        log.info("✓ %s instalado (%s)", spec.pip_name, spec.capability)
        return "installed"
    return "failed"


def _can_import(name: str) -> bool:
    try:
        __import__(name)
        return True
    except Exception:  # noqa: BLE001
        return False


def _cv2_has_face() -> bool:
    """Verifica si la librería OpenCV está instalada y contiene el módulo 'face'.

    Este módulo suele estar disponible cuando se instala 'opencv-contrib-python'.

    Returns:
        bool: True si cv2 está disponible y tiene el atributo 'face',
              False en caso contrario.
    """
    try:
        import cv2
        return hasattr(cv2, "face")
    except Exception:  # noqa: BLE001
        return False


def _pip_install(package: str) -> bool:
    log.info("Instalando %s …", package)
    try:
        result = subprocess.run(
            [
                sys.executable, "-m", "pip", "install",
                "--quiet", "--disable-pip-version-check",
                package,
            ],
            capture_output=True, text=True, timeout=_PIP_TIMEOUT_S,
        )
    except subprocess.TimeoutExpired:
        log.warning("pip install %s: timeout tras %ds", package, _PIP_TIMEOUT_S)
        return False
    except (OSError, ValueError) as e:
        log.warning("pip install %s: error subprocess: %s", package, e)
        return False
    if result.returncode != 0:
        log.warning(
            "pip install %s falló (exit %d): %s",
            package, result.returncode,
            (result.stderr or result.stdout or "")[-300:],
        )
        return False
    return True


def _pip_uninstall(package: str) -> None:
    """Mejor-esfuerzo. No falla la app si no se puede."""
    log.info("Desinstalando %s (conflict) …", package)
    try:
        subprocess.run(
            [
                sys.executable, "-m", "pip", "uninstall", "-y",
                "--quiet", package,
            ],
            capture_output=True, text=True, timeout=120,
        )
    except (OSError, subprocess.TimeoutExpired, ValueError) as e:
        log.debug("pip uninstall %s: %s", package, e)
