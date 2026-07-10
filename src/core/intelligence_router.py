"""
==============================================================================
 Proyecto MITOS - IntelligenceRouter (Fase 8 — Brain Pool)
==============================================================================

Paradigma anterior (Fases 3-7): el LLM local (Qwen2.5-7B en CPU) era el
CEREBRO. Hacía todo: razonamiento, generación de código, planificación
estratégica. Era lento (60-300s por ciclo) y mediocre en código.

Paradigma Fase 8: el LLM local es DISPATCHER. Solo decide QUÉ hacer
(80 tokens, <2s). Las tareas complejas se delegan a APIs externas
(Gemini Flash, Groq Llama-70B, OpenRouter) que tienen 10-100× más
parámetros y son gratis para los volúmenes de un operador solo.

Decisión de routing:

  - **TRIVIAL/SIMPLE** → local. Clasificar, parsear, decisión de menú.
  - **MODERATE** → externo si hay internet, local si no.
  - **COMPLEX/CREATIVE** → externo obligatorio (generación de código,
    razonamiento estratégico, diseño arquitectónico).

Sin internet → degraded mode: todo cae al local.

Configuración: `config/brain_pool.json` con lista de modelos externos.
Si no existe, se autogenera con las plantillas vacías (el operador
añade sus API keys). El sistema funciona sin keys en modo "all-local"
pero con la calidad anterior.

Convenciones (per INFORME_FORENSE §11):
  - Logger `mitos.intelligence_router`.
  - `from __future__ import annotations`, tipado moderno.
  - `async def` para llamadas HTTP; sync wrapper para callers viejos.
  - httpx se instala bajo demanda si falta.
==============================================================================
"""

from __future__ import annotations

import asyncio
import hashlib  # noqa: F401  (usado para cache estable de prompts si se añade)
import json
import logging
import socket
import subprocess
import sys
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

log = logging.getLogger("mitos.intelligence_router")


# ============================================================================
# Constantes
# ============================================================================
_CONFIG_RELPATH: str = "config/brain_pool.json"
_INTERNET_PROBES: tuple[tuple[str, int], ...] = (
    ("1.1.1.1", 53),
    ("8.8.8.8", 53),
)
_INTERNET_TIMEOUT_S: float = 1.5
_HTTPX_INSTALL_TIMEOUT_S: float = 180.0
_HTTP_REQUEST_TIMEOUT_S: float = 60.0

# Caracteres-por-token aproximado para estimaciones.
_CHARS_PER_TOKEN: int = 4


# ============================================================================
# Enums
# ============================================================================
class TaskComplexity(str, Enum):
    """Categoría heurística de la dificultad de una tarea."""

    TRIVIAL = "trivial"    # parsing, clasificación → local
    SIMPLE = "simple"      # preguntas cortas → local
    MODERATE = "moderate"  # código simple → preferir externo
    COMPLEX = "complex"    # código complejo → externo obligatorio
    CREATIVE = "creative"  # generación creativa → externo


class ModelProvider(str, Enum):
    """Proveedores soportados por el router."""

    LOCAL = "local"
    GEMINI = "gemini"
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    GROQ = "groq"
    OPENROUTER = "openrouter"


# ============================================================================
# Dataclasses
# ============================================================================
@dataclass(frozen=True)
class RoutingDecision:
    """Decisión inmutable del router para una tarea concreta."""

    provider: ModelProvider
    model_name: str
    reason: str
    estimated_tokens: int
    fallback_provider: ModelProvider | None = None


@dataclass
class ExternalModelConfig:
    """Configuración de un endpoint de API externa."""

    provider: ModelProvider
    api_key: str
    model_name: str
    base_url: str
    max_tokens: int = 4096
    cost_per_1k_tokens: float = 0.0
    supports_code: bool = True
    supports_vision: bool = False
    priority: int = 10  # menor = más preferido


@dataclass
class _UsageEntry:
    """Entrada del log de uso (interna)."""

    provider: str
    task_preview: str
    tokens_estimated: int
    elapsed_s: float
    success: bool
    timestamp: float = field(default_factory=time.time)


# ============================================================================
# Heurísticas de complejidad
# ============================================================================
# Indicadores en español e inglés. La detección es de PRESENCIA de
# substring, así que cuidamos no incluir palabras genéricas que puedan
# matchear cualquier mensaje (e.g. "el", "que").

_COMPLEX_INDICATORS: tuple[str, ...] = (
    "implementar", "implementa", "refactorizar", "refactor",
    "diseñar", "diseña", "arquitectura", "architecture",
    "algoritmo", "algorithm", "pipeline", "sistema completo",
    "crear modulo", "crear módulo", "create module",
    "integrar", "optimizar", "optimize",
    "debuggear", "debug", "complex bug", "complejo",
)

_SIMPLE_INDICATORS: tuple[str, ...] = (
    "añadir docstring", "add docstring", "renombrar", "rename",
    "formatear", "format", "log:", "logging:", "comentario",
    "comment", "tipo:", "type hint", "import ", "mover ",
    "move ",
)

_CREATIVE_INDICATORS: tuple[str, ...] = (
    "generar idea", "explorar", "explore", "investigar",
    "investigate", "proponer", "propose", "innovar",
    "diseño nuevo", "brainstorm",
)


# ============================================================================
# IntelligenceRouter
# ============================================================================
class IntelligenceRouter:
    """Decide DÓNDE ejecutar cada tarea cognitiva."""

    def __init__(self, project_root: str | Path, local_llm: Any) -> None:
        """
        Args:
            project_root: raíz del proyecto MITOS (para resolver el
                          `config/brain_pool.json`).
            local_llm:    instancia del LLM local con `.think(prompt,
                          max_tokens=..., temperature=...) -> str`.
        """
        self.root: Path = Path(project_root).resolve()
        self.local_llm: Any = local_llm
        self.external_models: list[ExternalModelConfig] = []
        self._has_internet: bool = False
        self._usage_log: list[_UsageEntry] = []
        self._httpx: Any = None  # cache del módulo httpx tras ensure

        self._load_config()
        self._check_connectivity()

        configured = sum(1 for m in self.external_models if m.api_key)
        log.info(
            "IntelligenceRouter listo (internet=%s, modelos externos=%d, "
            "configurados=%d)",
            self._has_internet,
            len(self.external_models),
            configured,
        )

    # ==================================================================
    #                       CONFIGURACIÓN
    # ==================================================================
    def _config_path(self) -> Path:
        return self.root / _CONFIG_RELPATH

    def _load_config(self) -> None:
        """Carga `config/brain_pool.json`. Si no existe, lo genera."""
        path = self._config_path()
        if not path.is_file():
            self._create_default_config(path)
            return

        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as e:
            log.warning("Error leyendo %s: %s — usando defaults", path, e)
            return

        for entry in data.get("models", []) or []:
            if not isinstance(entry, dict):
                continue
            try:
                provider = ModelProvider(entry.get("provider", "local"))
            except ValueError:
                log.debug("provider desconocido en config: %s", entry.get("provider"))
                continue
            try:
                self.external_models.append(
                    ExternalModelConfig(
                        provider=provider,
                        api_key=str(entry.get("api_key", "")),
                        model_name=str(entry.get("model_name", "")),
                        base_url=str(entry.get("base_url", "")),
                        max_tokens=int(entry.get("max_tokens", 4096)),
                        cost_per_1k_tokens=float(
                            entry.get("cost_per_1k", 0.0)
                        ),
                        supports_code=bool(entry.get("supports_code", True)),
                        supports_vision=bool(
                            entry.get("supports_vision", False)
                        ),
                        priority=int(entry.get("priority", 10)),
                    )
                )
            except (TypeError, ValueError) as e:
                log.debug("Skip modelo con campos inválidos: %s", e)

    def _create_default_config(self, path: Path) -> None:
        """Escribe la plantilla por defecto con `api_key` vacíos."""
        default: dict[str, Any] = {
            "_instructions": (
                "Añade tus API keys en el campo 'api_key' de cada modelo. "
                "Mientras estén vacíos, MITOS usará solo el LLM local. "
                "APIs gratuitas con free tier generoso: "
                "Gemini Flash (aistudio.google.com), "
                "Groq (console.groq.com), "
                "OpenRouter (openrouter.ai)."
            ),
            "models": [
                {
                    "provider": "gemini",
                    "api_key": "",
                    "model_name": "gemini-1.5-flash",
                    "base_url": (
                        "https://generativelanguage.googleapis.com/v1beta"
                    ),
                    "max_tokens": 8192,
                    "cost_per_1k": 0.0,
                    "supports_code": True,
                    "supports_vision": True,
                    "priority": 1,
                },
                {
                    "provider": "groq",
                    "api_key": "",
                    "model_name": "llama-3.1-70b-versatile",
                    "base_url": "https://api.groq.com/openai/v1",
                    "max_tokens": 8192,
                    "cost_per_1k": 0.0,
                    "supports_code": True,
                    "supports_vision": False,
                    "priority": 2,
                },
                {
                    "provider": "openrouter",
                    "api_key": "",
                    "model_name": "meta-llama/llama-3.1-8b-instruct:free",
                    "base_url": "https://openrouter.ai/api/v1",
                    "max_tokens": 4096,
                    "cost_per_1k": 0.0,
                    "supports_code": True,
                    "supports_vision": False,
                    "priority": 3,
                },
            ],
        }
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(
                json.dumps(default, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
            log.info(
                "Config plantilla escrita en %s — añade tus API keys",
                path,
            )
        except OSError as e:
            log.warning("No pude escribir config %s: %s", path, e)

    # ==================================================================
    #                       CONECTIVIDAD
    # ==================================================================
    def _check_connectivity(self) -> None:
        """Verifica internet vía TCP a DNS público."""
        for host, port in _INTERNET_PROBES:
            try:
                with socket.create_connection(
                    (host, port), timeout=_INTERNET_TIMEOUT_S
                ):
                    self._has_internet = True
                    return
            except (OSError, socket.timeout):
                continue
        self._has_internet = False

    def refresh_connectivity(self) -> bool:
        """Re-chequea internet (útil tras `internet_recovered`)."""
        previous = self._has_internet
        self._check_connectivity()
        if previous != self._has_internet:
            log.info(
                "Cambio de conectividad: %s → %s",
                previous, self._has_internet,
            )
        return self._has_internet

    # ==================================================================
    #                       CLASIFICACIÓN DE COMPLEJIDAD
    # ==================================================================
    def classify_complexity(self, task_description: str) -> TaskComplexity:
        """Clasifica heurísticamente la complejidad sin gastar LLM."""
        if not task_description:
            return TaskComplexity.TRIVIAL

        desc_lower = task_description.lower()

        if any(ind in desc_lower for ind in _COMPLEX_INDICATORS):
            return TaskComplexity.COMPLEX
        if any(ind in desc_lower for ind in _CREATIVE_INDICATORS):
            return TaskComplexity.CREATIVE
        if any(ind in desc_lower for ind in _SIMPLE_INDICATORS):
            return TaskComplexity.SIMPLE

        # Heurística por tamaño: tareas con código embebido o muy
        # largas son al menos MODERATE.
        if "```" in task_description or len(task_description) > 500:
            return TaskComplexity.MODERATE

        return TaskComplexity.SIMPLE

    # ==================================================================
    #                       ROUTING
    # ==================================================================
    def route(
        self,
        task_description: str,
        requires_code: bool = False,
        requires_vision: bool = False,
    ) -> RoutingDecision:
        """Decide dónde se ejecuta `task_description`."""
        complexity = self.classify_complexity(task_description)
        estimated_local = min(
            2000, len(task_description) // _CHARS_PER_TOKEN + 500
        )

        # Sin internet: TODO va local.
        if not self._has_internet:
            return RoutingDecision(
                provider=ModelProvider.LOCAL,
                model_name="local",
                reason="Sin internet — modo degradado (solo local)",
                estimated_tokens=estimated_local,
                fallback_provider=None,
            )

        # Trivial o simple: local es suficiente y más barato.
        if complexity in (TaskComplexity.TRIVIAL, TaskComplexity.SIMPLE):
            return RoutingDecision(
                provider=ModelProvider.LOCAL,
                model_name="local",
                reason=f"Tarea {complexity.value} — LLM local es suficiente",
                estimated_tokens=500,
                fallback_provider=self._get_best_external_provider(
                    requires_code, requires_vision
                ),
            )

        # Moderate / Complex / Creative: probamos externo.
        best_provider = self._get_best_external_provider(
            requires_code, requires_vision
        )
        if best_provider is not None:
            config = self._get_config(best_provider)
            model_name = (
                config.model_name if config is not None else best_provider.value
            )
            estimated_external = min(
                4000, len(task_description) // _CHARS_PER_TOKEN + 1000
            )
            return RoutingDecision(
                provider=best_provider,
                model_name=model_name,
                reason=(
                    f"Tarea {complexity.value} — requiere modelo más potente"
                ),
                estimated_tokens=estimated_external,
                fallback_provider=ModelProvider.LOCAL,
            )

        # Fallback: si no hay externos configurados, local.
        return RoutingDecision(
            provider=ModelProvider.LOCAL,
            model_name="local",
            reason="Sin modelos externos con API key — fallback a local",
            estimated_tokens=estimated_local,
            fallback_provider=None,
        )

    def _get_best_external_provider(
        self, requires_code: bool, requires_vision: bool
    ) -> ModelProvider | None:
        """Selecciona el mejor externo configurado que cumpla requisitos."""
        candidates = [
            m for m in self.external_models if m.api_key
        ]
        if requires_vision:
            candidates = [m for m in candidates if m.supports_vision]
        if requires_code:
            candidates = [m for m in candidates if m.supports_code]
        if not candidates:
            return None
        candidates.sort(key=lambda m: m.priority)
        return candidates[0].provider

    def _get_config(
        self, provider: ModelProvider
    ) -> ExternalModelConfig | None:
        """Lookup del primer config que coincida con `provider`."""
        for m in self.external_models:
            if m.provider == provider:
                return m
        return None

    # ==================================================================
    #                       EJECUCIÓN — ASYNC
    # ==================================================================
    async def execute(
        self,
        task: str,
        routing: RoutingDecision,
        system_prompt: str = "",
    ) -> str:
        """Ejecuta `task` en el provider seleccionado por `routing`."""
        start = time.time()

        if routing.provider == ModelProvider.LOCAL:
            result = self._call_local(task, routing, system_prompt)
        else:
            result = await self._call_external(routing, task, system_prompt)

        elapsed = time.time() - start
        self._log_usage(
            provider=routing.provider.value,
            task=task,
            tokens_estimated=routing.estimated_tokens,
            elapsed=elapsed,
            success=bool(result),
        )

        # Fallback si la primera vía falló.
        if not result and routing.fallback_provider is not None:
            log.info(
                "execute: primary %s falló, probando fallback %s",
                routing.provider.value,
                routing.fallback_provider.value,
            )
            fallback_routing = RoutingDecision(
                provider=routing.fallback_provider,
                model_name="fallback",
                reason="Fallback tras fallo del primario",
                estimated_tokens=routing.estimated_tokens,
                fallback_provider=None,
            )
            result = await self.execute(task, fallback_routing, system_prompt)

        return result or ""

    def execute_sync(
        self,
        task: str,
        routing: RoutingDecision,
        system_prompt: str = "",
    ) -> str:
        """Wrapper síncrono para integrarse con el daemon existente."""
        try:
            loop = asyncio.new_event_loop()
            try:
                return loop.run_until_complete(
                    self.execute(task, routing, system_prompt)
                )
            finally:
                loop.close()
        except Exception as e:  # noqa: BLE001
            log.warning("execute_sync falló: %s", e)
            return ""

    # ==================================================================
    #                       VISIÓN (multimodal)
    # ==================================================================
    def describe_image(
        self,
        image_b64: str,
        question: str = "Describe lo que ves en la imagen, con detalle pero breve.",
    ) -> str:
        """Envía imagen + pregunta a un modelo con supports_vision=True.

        Args:
            image_b64: JPEG codificado en base64 (sin prefijo data:).
            question:  texto que acompaña la imagen.

        Returns:
            Descripción del modelo o cadena vacía si no hay modelo de
            visión disponible / si la API falla.
        """
        if not image_b64:
            return ""
        if not self._has_internet:
            log.info("describe_image sin internet — no puedo delegar a Gemini")
            return ""

        vision_models = [m for m in self.external_models if m.supports_vision]
        if not vision_models:
            log.warning(
                "Ningún modelo del pool tiene supports_vision=True. "
                "Edita config/brain_pool.json y marca uno (e.g. gemini)."
            )
            return ""
        vision_models.sort(key=lambda m: m.priority)
        target = vision_models[0]

        async def _run() -> str:
            return await self._call_gemini(
                config=target,
                task=question,
                system_prompt="",
                image_b64=image_b64,
            )

        try:
            loop = asyncio.new_event_loop()
            try:
                result = loop.run_until_complete(_run())
            finally:
                loop.close()
            self._log_usage(
                provider=target.provider.value,
                task=f"describe_image (vision, {len(image_b64)} b64 bytes)",
                tokens_estimated=200,
                elapsed=0.0,
                success=bool(result),
            )
            return result
        except Exception as e:  # noqa: BLE001
            log.warning("describe_image falló: %s", e)
            return ""

    # ==================================================================
    #                       LOCAL
    # ==================================================================
    def _call_local(
        self,
        task: str,
        routing: RoutingDecision,
        system_prompt: str,
    ) -> str:
        """Llamada síncrona al LLM local."""
        prompt = (
            f"{system_prompt}\n\n{task}".strip()
            if system_prompt else task
        )
        try:
            return self.local_llm.think(
                prompt,
                max_tokens=routing.estimated_tokens,
                temperature=0.4,
            )
        except Exception as e:  # noqa: BLE001
            log.warning("local_llm.think falló: %s", e)
            return ""

    # ==================================================================
    #                       HTTP HELPER
    # ==================================================================
    def _ensure_httpx(self) -> Any:
        """Importa httpx, instalándolo si falta. Cachea el módulo."""
        if self._httpx is not None:
            return self._httpx
        try:
            import httpx  # type: ignore[import-not-found]
            self._httpx = httpx
            return httpx
        except ImportError:
            log.info("Instalando httpx (necesario para APIs externas)…")
        try:
            completed = subprocess.run(
                [sys.executable, "-m", "pip", "install", "httpx"],
                capture_output=True,
                text=True,
                timeout=_HTTPX_INSTALL_TIMEOUT_S,
            )
        except (OSError, subprocess.TimeoutExpired) as e:
            log.warning("pip install httpx falló: %s", e)
            return None
        if completed.returncode != 0:
            log.warning(
                "pip install httpx exit=%d: %s",
                completed.returncode,
                (completed.stderr or "")[-200:],
            )
            return None
        try:
            import httpx  # type: ignore[import-not-found]
            self._httpx = httpx
            return httpx
        except ImportError:
            log.warning("httpx instalado pero sigue sin importar")
            return None

    # ==================================================================
    #                       EXTERNAL — DISPATCH
    # ==================================================================
    async def _call_external(
        self,
        routing: RoutingDecision,
        task: str,
        system_prompt: str,
    ) -> str:
        """Despacha al adaptador del provider concreto."""
        config = self._get_config(routing.provider)
        if config is None or not config.api_key:
            log.debug("Sin config/key para %s", routing.provider.value)
            return ""

        httpx = self._ensure_httpx()
        if httpx is None:
            return ""

        try:
            if config.provider == ModelProvider.GEMINI:
                return await self._call_gemini(config, task, system_prompt)
            # Resto (Groq, OpenRouter, OpenAI, Anthropic-compat) usan
            # el shape OpenAI-compatible.
            return await self._call_openai_compatible(
                config, task, system_prompt
            )
        except Exception as e:  # noqa: BLE001
            log.warning(
                "Llamada a %s falló: %s", config.provider.value, e
            )
            return ""

    # ==================================================================
    #                       GEMINI
    # ==================================================================
    async def _call_gemini(
        self,
        config: ExternalModelConfig,
        task: str,
        system_prompt: str,
        image_b64: str | None = None,
    ) -> str:
        """Adaptador para la REST API de Gemini (generateContent).

        Si `image_b64` está, se inyecta como `inline_data` JPEG y la
        request se vuelve multimodal — requiere modelo con
        supports_vision=True en el pool.
        """
        httpx = self._ensure_httpx()
        if httpx is None:
            return ""

        url = (
            f"{config.base_url.rstrip('/')}/models/{config.model_name}"
            f":generateContent?key={config.api_key}"
        )
        prompt_text = (
            f"{system_prompt}\n\n{task}".strip()
            if system_prompt else task
        )
        # Construimos parts con texto + imagen si la hay. El orden importa:
        # Gemini espera el texto PRIMERO y la imagen después, así no genera
        # respuestas que ignoren la imagen.
        parts: list[dict[str, Any]] = [{"text": prompt_text}]
        if image_b64:
            parts.append({
                "inline_data": {
                    "mime_type": "image/jpeg",
                    "data": image_b64,
                },
            })
        payload: dict[str, Any] = {
            "contents": [{"parts": parts}],
            "generationConfig": {
                "maxOutputTokens": config.max_tokens,
                "temperature": 0.1,
            },
        }

        async with httpx.AsyncClient(timeout=_HTTP_REQUEST_TIMEOUT_S) as client:
            resp = await client.post(url, json=payload)

        if resp.status_code != 200:
            log.warning(
                "Gemini HTTP %d: %s",
                resp.status_code,
                resp.text[:200] if resp.text else "(empty)",
            )
            return ""

        try:
            data = resp.json()
            candidates = data.get("candidates") or []
            if not candidates:
                return ""
            parts = (
                candidates[0].get("content", {}).get("parts", []) or []
            )
            return "".join(str(p.get("text", "")) for p in parts).strip()
        except (KeyError, IndexError, TypeError, ValueError) as e:
            log.debug("Gemini parse response: %s", e)
            return ""

    # ==================================================================
    #                       OPENAI-COMPATIBLE (Groq, OpenRouter, etc.)
    # ==================================================================
    async def _call_openai_compatible(
        self,
        config: ExternalModelConfig,
        task: str,
        system_prompt: str,
    ) -> str:
        """Adaptador para APIs con shape OpenAI `/chat/completions`."""
        httpx = self._ensure_httpx()
        if httpx is None:
            return ""

        url = f"{config.base_url.rstrip('/')}/chat/completions"
        headers: dict[str, str] = {
            "Authorization": f"Bearer {config.api_key}",
            "Content-Type": "application/json",
        }
        # OpenRouter requiere headers de identificación.
        if config.provider == ModelProvider.OPENROUTER:
            headers["HTTP-Referer"] = "https://mitos.local"
            headers["X-Title"] = "MITOS"

        messages: list[dict[str, str]] = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        else:
            messages.append({
                "role": "system",
                "content": (
                    "Eres un asistente experto en código y razonamiento "
                    "técnico. Responde de forma directa y concisa."
                ),
            })
        messages.append({"role": "user", "content": task})

        payload: dict[str, Any] = {
            "model": config.model_name,
            "messages": messages,
            "max_tokens": config.max_tokens,
            "temperature": 0.1,
        }

        async with httpx.AsyncClient(timeout=_HTTP_REQUEST_TIMEOUT_S) as client:
            resp = await client.post(url, json=payload, headers=headers)

        if resp.status_code != 200:
            log.warning(
                "%s HTTP %d: %s",
                config.provider.value,
                resp.status_code,
                resp.text[:200] if resp.text else "(empty)",
            )
            return ""

        try:
            data = resp.json()
            choices = data.get("choices") or []
            if not choices:
                return ""
            return str(
                choices[0].get("message", {}).get("content", "")
            ).strip()
        except (KeyError, IndexError, TypeError, ValueError) as e:
            log.debug("%s parse response: %s", config.provider.value, e)
            return ""

    # ==================================================================
    #                       LOGGING DE USO
    # ==================================================================
    def _log_usage(
        self,
        provider: str,
        task: str,
        tokens_estimated: int,
        elapsed: float,
        success: bool,
    ) -> None:
        self._usage_log.append(
            _UsageEntry(
                provider=provider,
                task_preview=task[:100],
                tokens_estimated=tokens_estimated,
                elapsed_s=elapsed,
                success=success,
            )
        )
        # Cap del log a 200 entradas — no inflar memoria en sesiones largas.
        if len(self._usage_log) > 200:
            self._usage_log = self._usage_log[-200:]

    def get_usage_summary(self) -> str:
        """Resumen para inyectar en `/status` y para el ciclo cognitivo."""
        if not self._usage_log:
            return "Sin uso de APIs externas todavía"
        recent = self._usage_log[-10:]
        providers_used = sorted({e.provider for e in recent})
        success_rate = sum(1 for e in recent if e.success) / len(recent)
        return (
            f"APIs usadas (últ. {len(recent)}): {', '.join(providers_used)} | "
            f"éxito={success_rate * 100:.0f}% | "
            f"última={recent[-1].elapsed_s:.1f}s"
        )
