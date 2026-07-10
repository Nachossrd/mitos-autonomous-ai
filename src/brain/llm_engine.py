"""
==============================================================================
 Proyecto MITOS - Motor de Razonamiento (LLM local)
==============================================================================

Núcleo de la "conciencia" del sistema. Carga un modelo LLM cuantizado en
formato GGUF y lo expone a través de una API estructurada:

    * think                -> llamada genérica con parámetros de sampling.
    * generate_code        -> fuerza salida Python (sin prosa).
    * reason_step_by_step  -> Chain-of-Thought.
    * reflect              -> metacognición: evalúa acción vs objetivo.
    * decide_action        -> "libre albedrío" computacional: elige tool.

Propiedades clave:
    - 100% local: cero llamadas a APIs externas. La inferencia ocurre en
      la CPU/GPU del propio host vía llama.cpp. Los pesos del modelo, los
      prompts y las salidas nunca abandonan la máquina.
    - Auto-descubrimiento del modelo: busca un .gguf en `models/`. Si no
      hay ninguno, levanta FileNotFoundError con una sugerencia
      explícita de descarga.
    - Imports perezosos de llama_cpp: el módulo se puede importar para
      tests/lint aunque la wheel todavía no esté instalada.

Convención de prompting:
    El motor envuelve cada llamada con CHAT TEMPLATE configurable (por
    defecto ChatML — Qwen2.5, Yi, Hermes-2-Pro). Las constantes
    `_TPL_*` y `_CHAT_STOPS` se cambian a la vez para adaptarse a otro
    modelo base. El operador no necesita formatear nada: pasa prosa
    plana y el engine añade los meta-tokens.
==============================================================================
"""

from __future__ import annotations

import json
import logging
import os
import re
import threading
from pathlib import Path
from typing import Any

from .identity import (
    PRIMING_ASSISTANT as _PRIMING_ASSISTANT,
    PRIMING_USER as _PRIMING_USER,
    SYSTEM_PROMPT as DEFAULT_SYSTEM_PROMPT,
)
from .response_filter import ResponseSanitizer

log = logging.getLogger("mitos.brain.llm")

# Sugerencia de modelo por defecto. A partir de Fase 8 usamos
# Qwen2.5-Coder-14B-Instruct: el doble de parámetros que el 7B base y
# afinado específicamente a código (Python/JS/Java/etc.), que es el
# 80% del trabajo de MITOS (`_do_self_modify`, `build_tool`,
# `generate_capability`).
#
# Sigue siendo ChatML — el `_build_prompt` no cambia.
#
# Trade-off: en CPU AVX2 sin GPU baja la velocidad de inferencia
# (~2.5× más lento que el 7B), pero el `IntelligenceRouter` de Fase 8
# delega las tareas de razonamiento complejo a Gemini/Groq vía API, así
# que el local solo se encarga del despacho rápido + código local.
_SUGGESTED_MODEL = "qwen2.5-coder-14b-instruct-q4_k_m-00001-of-00002.gguf"
_SUGGESTED_REPO = "Qwen/Qwen2.5-Coder-14B-Instruct-GGUF"

# Chat template ChatML (Qwen, Yi, Hermes-2-Pro, etc.). Formato canónico:
#     <|im_start|>system\n{system}<|im_end|>
#     <|im_start|>user\n{user}<|im_end|>
#     <|im_start|>assistant\n{assistant}<|im_end|>
# El turno final del asistente se deja ABIERTO (sin `<|im_end|>`) para
# que el modelo autocomplete a partir de ahí. Si en el futuro se cambia
# el .gguf base por uno con template distinto (Llama 3, Phi-3, etc.),
# basta editar estas 4 constantes — el resto del archivo no toca el
# template literal.
_TPL_SYSTEM_OPEN = "<|im_start|>system\n"
_TPL_USER_OPEN = "<|im_start|>user\n"
_TPL_ASSISTANT_OPEN = "<|im_start|>assistant\n"
_TPL_END = "<|im_end|>\n"
# Tokens que CORTAN la generación. ChatML emite `<|im_end|>` al cerrar
# su turno (es el stop "natural" del modelo bien entrenado). Incluimos
# `<|im_start|>` para cazar TURN-LEAKAGE: cuando el modelo se sale del
# rail y empieza a fabricar el siguiente turno por su cuenta.
# `<|endoftext|>` es la convención GPT-style; lo mantenemos por si el
# operador cambia a un modelo que lo emita.
_CHAT_STOPS = [
    "<|im_end|>",
    "<|im_start|>",
    "<|endoftext|>",
]


class LLMEngine:
    """
    Wrapper alrededor de llama_cpp.Llama con métodos cognitivos de alto
    nivel (think / generate_code / reason_step_by_step / reflect /
    decide_action).

    Atributos:
        model_path:  ruta absoluta al .gguf cargado.
        n_ctx:       tamaño de la ventana de contexto en tokens.
        n_threads:   hilos de CPU dedicados a la inferencia.
        llm:         instancia interna de llama_cpp.Llama.
    """

    # ------------------------------------------------------------------
    def __init__(
        self,
        models_dir: str | Path = "models",
        model_filename: str | None = None,
        n_ctx: int = 4096,
        n_threads: int | None = None,
        n_gpu_layers: int = 0,
        verbose: bool = False,
        system_prompt: str | None = None,
    ) -> None:
        """
        Args:
            models_dir:     carpeta donde buscar .gguf (relativa o absoluta).
            model_filename: si se da, carga ese fichero específico; si no,
                            se elige automáticamente el primer .gguf
                            encontrado, prefiriendo el sugerido.
            n_ctx:          tamaño de ventana de contexto (default 4096).
            n_threads:      hilos CPU; por defecto cpu_count - 1.
            n_gpu_layers:   capas a offload a GPU; 0 = CPU pura (MVP).
            verbose:        si True, llama.cpp imprime su propio log.
            system_prompt:  identidad/reglas inyectadas en cada llamada al
                            modelo. Si es None, se usa el de
                            `src.brain.identity.SYSTEM_PROMPT`. Pasa "" para
                            desactivar la inyección (modo raw completion).

        Raises:
            FileNotFoundError: si no se encuentra ningún .gguf adecuado.
            RuntimeError:      si llama_cpp no está instalado.
        """
        self.n_ctx = n_ctx
        self.n_threads = n_threads or max(1, (os.cpu_count() or 4) - 1)
        self.n_gpu_layers = n_gpu_layers
        self.system_prompt = (
            DEFAULT_SYSTEM_PROMPT if system_prompt is None else system_prompt
        )

        self.model_path = self._locate_model(Path(models_dir), model_filename)
        log.info(
            "[LLMEngine] cargando %s (n_ctx=%d, n_threads=%d, gpu_layers=%d)",
            self.model_path.name,
            self.n_ctx,
            self.n_threads,
            self.n_gpu_layers,
        )

        # Import perezoso: si la wheel no está, damos un error útil sin
        # que el resto del paquete falle al importarse.
        try:
            from llama_cpp import Llama  # type: ignore
        except ImportError as e:  # pragma: no cover
            raise RuntimeError(
                "llama-cpp-python no está instalado. "
                "Ejecuta: pip install -r requirements.txt"
            ) from e

        self.llm = Llama(
            model_path=str(self.model_path),
            n_ctx=self.n_ctx,
            n_threads=self.n_threads,
            n_gpu_layers=self.n_gpu_layers,
            verbose=verbose,
        )
        # Reentrant lock para serializar llamadas al modelo. llama.cpp
        # NO es thread-safe en concurrencia real; con este lock podemos
        # tener al daemon en un thread y la consola en otro sin corrupción.
        self._lock: threading.RLock = threading.RLock()
        # Purga post-inferencia: elimina residuo del fabricante (Phi/MS).
        # Se aplica a `think()` por defecto; los métodos que generan
        # código (generate_code) lo desactivan para no romper sintaxis.
        self._sanitizer: ResponseSanitizer = ResponseSanitizer()
        log.info("[LLMEngine] modelo listo")

    # ==================================================================
    #                  LOCALIZACIÓN DEL MODELO
    # ==================================================================
    @staticmethod
    def _score_gguf(path: Path) -> tuple[int, int, int, str]:
        """Scoring para priorizar `coder-14b` sobre `7b` sobre `mini`.

        Ranking (descendente, mayor score = más preferido):
          - "coder" en el nombre  → +10 (afinado a código, el dominio MITOS)
          - tamaño:
              · 70b/72b → +8
              · 14b/13b → +6
              · 8b/9b   → +4
              · 7b      → +3
              · 3b/4b   → +1
          - "instruct" → +1
          - "abliterated" / "uncensored" → +1 (sin RLHF de refusal)
          - Q4_K_M específicamente → +1 (balance memoria/calidad)

        Devuelve tupla (-1, score, neg_name) para usar con `max()`:
        max elige el de mayor score. El nombre se incluye como tiebreaker
        determinista para que `models/` con misma ranking devuelva siempre
        el mismo.
        """
        name = path.name.lower()
        score = 0
        if "coder" in name:
            score += 10
        if any(s in name for s in ("70b", "72b")):
            score += 8
        elif any(s in name for s in ("14b", "13b")):
            score += 6
        elif any(s in name for s in ("8b", "9b")):
            score += 4
        elif "7b" in name:
            score += 3
        elif any(s in name for s in ("3b", "4b")):
            score += 1
        if "instruct" in name:
            score += 1
        if "abliterated" in name or "uncensored" in name:
            score += 1
        if "q4_k_m" in name:
            score += 1
        # Tuple: (score, -mtime para más reciente desempata, name)
        try:
            mtime = int(path.stat().st_mtime)
        except OSError:
            mtime = 0
        return (score, mtime, 0, name)

    @staticmethod
    def _locate_model(models_dir: Path, preferred: str | None) -> Path:
        """
        Busca un .gguf en `models_dir`. Reglas:
          1. Si se pasó `preferred`, exige que exista exactamente ese fichero.
          2. Si no, prefiere `_SUGGESTED_MODEL` si está presente.
          3. Si tampoco, elige el .gguf con mayor SCORE (coder>base, 14b>7b,
             instruct>plain). Antes era el primer .gguf alfabético, lo que
             hacía que "qwen2.5-7b" ganara a "qwen2.5-coder-14b" en sesiones
             con ambos modelos descargados.
          4. Si no hay ninguno, levanta FileNotFoundError con instrucciones.
        """
        models_dir = models_dir.resolve() if not models_dir.is_absolute() else models_dir

        if preferred is not None:
            candidate = models_dir / preferred
            if candidate.is_file():
                return candidate
            raise FileNotFoundError(
                f"No encuentro el modelo solicitado: {candidate}\n"
                f"Verifica el nombre o copia el fichero en {models_dir}."
            )

        if not models_dir.is_dir():
            raise FileNotFoundError(
                f"No existe el directorio de modelos: {models_dir}\n"
                f"Crea la carpeta y descarga un modelo .gguf, por ejemplo:\n"
                f"    huggingface-cli download {_SUGGESTED_REPO} "
                f"{_SUGGESTED_MODEL} --local-dir {models_dir}"
            )

        suggested = models_dir / _SUGGESTED_MODEL
        if suggested.is_file():
            return suggested

        ggufs = list(models_dir.glob("*.gguf"))
        # Solo splits "00001-of-N" — los demás son chunks, no entry points.
        primary = [g for g in ggufs if "00001-of-" in g.name or "of-" not in g.name]
        if not primary:
            primary = ggufs
        if primary:
            return max(primary, key=LLMEngine._score_gguf)

        raise FileNotFoundError(
            f"No encontré ningún archivo .gguf en {models_dir}.\n"
            f"Descarga uno (recomendado {_SUGGESTED_MODEL}) con:\n"
            f"    huggingface-cli download {_SUGGESTED_REPO} "
            f"{_SUGGESTED_MODEL} --local-dir {models_dir}"
        )

    # ==================================================================
    #                       LLAMADA BASE
    # ==================================================================
    def think(
        self,
        prompt: str,
        max_tokens: int = 512,
        temperature: float = 0.7,
        top_p: float = 0.95,
        stop: list[str] | None = None,
        system: str | None = None,
        assistant_prefix: str = "",
        sanitize: bool = True,
    ) -> str:
        """
        Llamada base al LLM. Devuelve únicamente el texto generado
        (sin metadata).

        Por defecto envuelve `prompt` con el CHAT TEMPLATE ChatML
        (Qwen2.5) y le inyecta `self.system_prompt` como turno de
        sistema. Esto es lo que permite sobrescribir la identidad del
        modelo base con la identidad MITOS definida en
        `src/brain/identity.py`.

        Args:
            prompt:            mensaje del operador (turno "user").
            max_tokens:        máximo de tokens a generar.
            temperature:       0.0 = determinista, >1.0 = creativo.
            top_p:             nucleus sampling.
            stop:              stops EXTRA además de los tokens de fin de
                               turno (se añaden a los del chat template).
            system:            override puntual del system prompt. Si es
                               None usa `self.system_prompt`. Pasar "" lo
                               desactiva (modo raw completion).
            assistant_prefix:  texto que "pone en boca" del asistente al
                               inicio del turno. Útil para anclar formato
                               (p.ej. "```python\\n" o "Step 1:").

        Returns:
            Texto generado por el asistente, ya con whitespace recortado.
        """
        full_prompt = self._build_prompt(prompt, system, assistant_prefix)
        merged_stops = self._merge_stops(stop, system)
        # Bajo el lock: el modelo es secuencial. Sin él, dos threads
        # llamando a llama.cpp a la vez producen basura o crash.
        with self._lock:
            result = self.llm(
                full_prompt,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                stop=merged_stops,
            )
        # llama-cpp devuelve un dict con choices[0]['text'].
        choices = result.get("choices") if isinstance(result, dict) else None
        if not choices:
            return ""
        text = choices[0].get("text", "").strip()
        # Purga post-inferencia. Saltable cuando lo que viene es código
        # estructurado (generate_code lo desactiva) o cuando el caller
        # quiere ver la salida cruda.
        if sanitize:
            text = self._sanitizer.sanitize(text)
        return text

    # ------------------------------------------------------------------
    def _build_prompt(
        self,
        user_msg: str,
        system: str | None,
        assistant_prefix: str,
    ) -> str:
        """
        Construye el prompt final aplicando (o no) el chat template ChatML.

        Si el system prompt es no-vacío, envolvemos con:

            <|im_start|>system\\n{sys}<|im_end|>
            <|im_start|>user\\n{PRIMING_USER}<|im_end|>           <- priming
            <|im_start|>assistant\\n{PRIMING_ASSISTANT}<|im_end|> <- priming
            <|im_start|>user\\n{user_msg}<|im_end|>
            <|im_start|>assistant\\n{prefix}                       <- ABIERTO

        El turno de priming sobrescribe identidades cocidas por RLHF de
        forma mucho más efectiva que un system prompt solo. El modelo ya
        "se identificó" como MITOS antes de procesar el mensaje real,
        así que mantiene esa persona durante toda la respuesta.

        Importante: el turno final del asistente queda ABIERTO (sin
        `<|im_end|>`) para que el modelo autocomplete a partir de ahí.
        Cerrarlo es el error clásico que da respuestas vacías.

        Si el system prompt está vacío (modo raw completion), no
        envolvemos nada y devolvemos `user_msg + prefix` directo.
        """
        sys_prompt = self.system_prompt if system is None else system
        if not sys_prompt:
            return user_msg + assistant_prefix
        return (
            f"{_TPL_SYSTEM_OPEN}{sys_prompt}{_TPL_END}"
            f"{_TPL_USER_OPEN}{_PRIMING_USER}{_TPL_END}"
            f"{_TPL_ASSISTANT_OPEN}{_PRIMING_ASSISTANT}{_TPL_END}"
            f"{_TPL_USER_OPEN}{user_msg}{_TPL_END}"
            f"{_TPL_ASSISTANT_OPEN}{assistant_prefix}"
        )

    # ------------------------------------------------------------------
    @staticmethod
    def _merge_stops(
        extra: list[str] | None, system: str | None
    ) -> list[str]:
        """
        Combina los stops del operador con los del chat template, sin
        duplicados. Si estamos en modo raw (system==""), no añadimos los
        tokens especiales porque no aparecerán en la salida.
        """
        if system == "":
            return list(extra) if extra else []
        merged = list(_CHAT_STOPS)
        if extra:
            for s in extra:
                if s not in merged:
                    merged.append(s)
        return merged

    # ==================================================================
    #                   GENERACIÓN DE CÓDIGO
    # ==================================================================
    def generate_code(
        self,
        task: str,
        context: str = "",
        max_tokens: int = 768,
        temperature: float = 0.2,
    ) -> str:
        """
        Genera código Python para resolver `task`, opcionalmente con
        `context` adicional (ejemplos, restricciones, firmas a respetar).

        El prompt fuerza al LLM a devolver SOLAMENTE código dentro de un
        bloque ```python ... ``` y nosotros lo extraemos. Si el LLM no
        usa el bloque (modelos pequeños fallan a veces), devolvemos el
        texto crudo: el filtro autónomo (src.filtering) lo descartará si
        no es parseable, por lo que la calidad sigue garantizada
        aguas abajo.

        Args:
            task:        descripción de la tarea en lenguaje natural.
            context:     contexto adicional (puede ser "").
            max_tokens:  presupuesto de generación.
            temperature: baja por defecto para reducir alucinaciones.
        """
        ctx_block = f"\nContext:\n{context}\n" if context else ""
        prompt = (
            "Generate Python code for the following task. Output ONLY code "
            "inside a single fenced block; no prose, no explanations.\n"
            f"Task: {task}{ctx_block}"
        )
        # `assistant_prefix` arranca al asistente DENTRO del bloque ```python,
        # así el modelo solo necesita escribir el cuerpo y nosotros cortamos
        # al ver el cierre ```.
        # sanitize=False: el sanitizer es para prosa; aquí lo que viene
        # es Python y podríamos romper identifiers legítimos.
        raw = self.think(
            prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            assistant_prefix="```python\n",
            stop=["```"],
            sanitize=False,
        )
        cleaned = raw
        if cleaned.lower().startswith("python\n"):
            cleaned = cleaned[len("python\n") :]
        return cleaned.strip()

    # ==================================================================
    #                    CHAIN-OF-THOUGHT
    # ==================================================================
    def reason_step_by_step(
        self,
        question: str,
        max_tokens: int = 320,
        temperature: float = 0.5,
    ) -> str:
        """
        Implementa Chain-of-Thought. El prompt arranca la cadena con
        "Let me think through this step by step: Step 1:" para anclar al
        modelo a un razonamiento explícito y verificable.

        Devuelve el texto completo del razonamiento.
        """
        # Prompt deliberadamente neutro: el question dicta el idioma. Si
        # el operador pregunta en español, la pregunta lleva acentos y
        # palabras españolas y el modelo responde en español. El prompt
        # inglés viejo ("Think step by step") anclaba siempre a inglés
        # aunque la pregunta fuera en español; eso ya no.
        prompt = (
            f"{question}\n\n"
            "Razona paso a paso y termina con una respuesta clara y "
            "concisa en el MISMO idioma de la pregunta."
        )
        # Sin `assistant_prefix`: dejamos al modelo elegir formato y
        # idioma. Anclar a "Step 1:" forzaba inglés en respuestas a
        # preguntas españolas.
        body = self.think(
            prompt,
            max_tokens=max_tokens,
            temperature=temperature,
        )
        return body

    # ==================================================================
    #                       REFLEXIÓN (metacognición)
    # ==================================================================
    def reflect(
        self,
        action: str,
        result: str,
        goal: str,
        max_tokens: int = 256,
        temperature: float = 0.3,
    ) -> dict[str, str]:
        """
        Metacognición estructurada: el LLM evalúa una acción tomada hacia
        un objetivo y devuelve un diccionario con:
            - assessment: success | partial | failure
            - lesson:     una frase con lo aprendido.
            - next_action: siguiente paso recomendado.

        Forzamos un formato textual fijo (líneas KEY: value) en lugar de
        JSON, porque los modelos pequeños cumplen mejor con keys planas
        que con un JSON estricto.
        """
        prompt = (
            "You are evaluating an action taken toward a goal.\n"
            f"Goal: {goal}\n"
            f"Action: {action}\n"
            f"Result: {result}\n\n"
            "Respond using EXACTLY these three lines and nothing else:\n"
            "ASSESSMENT: <success|partial|failure>\n"
            "LESSON: <one short sentence>\n"
            "NEXT_ACTION: <one short sentence>\n"
        )
        raw = self.think(
            prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            stop=["\n\n", "Goal:", "Action:"],
        )
        return self._parse_keyed_response(
            raw,
            required=("ASSESSMENT", "LESSON", "NEXT_ACTION"),
            lowercase_keys=True,
            defaults={
                "assessment": "partial",
                "lesson": "respuesta no estructurada del modelo",
                "next_action": "reintentar con prompt más específico",
            },
        )

    # ==================================================================
    #                  DECIDIR ACCIÓN (libre albedrío)
    # ==================================================================
    def decide_action(
        self,
        goal: str,
        history: list[str] | str,
        memory: list[str] | str,
        tools: list[str],
        max_tokens: int = 256,
        temperature: float = 0.4,
    ) -> dict[str, str]:
        """
        Decide la siguiente herramienta a usar dado el objetivo, el
        historial reciente y la memoria relevante.

        Returns:
            dict con:
              - thought: razonamiento corto.
              - tool:    nombre de una herramienta de `tools` (o "none").
              - input:   input que se pasaría a esa herramienta.

        Si el LLM elige una herramienta fuera de la lista, la normalizamos
        al string "none" para no engañar al agente.
        """
        if isinstance(history, list):
            history_str = "\n".join(f"- {h}" for h in history) or "(empty)"
        else:
            history_str = history or "(empty)"
        if isinstance(memory, list):
            memory_str = "\n".join(f"- {m}" for m in memory) or "(empty)"
        else:
            memory_str = memory or "(empty)"

        tools_str = ", ".join(tools) if tools else "none"
        prompt = (
            "You are an autonomous agent. Choose ONE tool to use next.\n"
            f"Goal: {goal}\n"
            f"History:\n{history_str}\n"
            f"Relevant memory:\n{memory_str}\n"
            f"Available tools: {tools_str}\n\n"
            "Respond using EXACTLY these three lines and nothing else:\n"
            "THOUGHT: <short reasoning>\n"
            "TOOL: <one tool name from the list, or 'none'>\n"
            "INPUT: <input string for the tool>\n"
        )
        raw = self.think(
            prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            stop=["\n\n", "Goal:", "Available tools:"],
        )
        parsed = self._parse_keyed_response(
            raw,
            required=("THOUGHT", "TOOL", "INPUT"),
            lowercase_keys=True,
            defaults={
                "thought": "(modelo no produjo razonamiento)",
                "tool": "none",
                "input": "",
            },
        )

        # Saneamiento: la tool elegida debe estar en la lista.
        chosen = parsed["tool"].strip().lower()
        valid = {t.lower(): t for t in tools}
        parsed["tool"] = valid.get(chosen, "none")
        return parsed

    # ==================================================================
    #                       UTILIDADES INTERNAS
    # ==================================================================
    @staticmethod
    def _parse_keyed_response(
        raw: str,
        required: tuple[str, ...],
        lowercase_keys: bool,
        defaults: dict[str, str],
    ) -> dict[str, str]:
        """
        Parser tolerante para respuestas con formato "KEY: value\\n".

        Primero intentamos JSON (a veces el modelo lo da gratis); si falla,
        recorremos línea a línea con regex y rellenamos defaults para las
        keys que falten. Esto es deliberadamente permisivo: preferimos
        degradar a defaults antes que crashear el loop del agente.
        """
        # 1) Intento JSON.
        try:
            obj = json.loads(raw)
            if isinstance(obj, dict):
                out = {}
                for k in required:
                    key = k.lower() if lowercase_keys else k
                    out[key] = str(obj.get(k, obj.get(k.lower(), defaults.get(key, ""))))
                # Rellenar faltantes con defaults.
                for k, v in defaults.items():
                    out.setdefault(k, v)
                return out
        except (ValueError, TypeError):
            pass

        # 2) Línea a línea: KEY: valor
        out: dict[str, str] = dict(defaults)
        for line in raw.splitlines():
            m = re.match(r"\s*([A-Za-z_][A-Za-z0-9_ ]*)\s*:\s*(.*)$", line)
            if not m:
                continue
            key = m.group(1).strip().upper().replace(" ", "_")
            if key in required:
                slot = key.lower() if lowercase_keys else key
                out[slot] = m.group(2).strip()
        return out
