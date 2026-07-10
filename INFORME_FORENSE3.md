# INFORME FORENSE — Proyecto MITOS

**Documento de transferencia técnica.** Permite a un equipo nuevo integrar código en MITOS sin necesidad de leer la implementación. Describe contratos públicos, invariantes, flujos críticos, puntos de extensión y deuda técnica conocida.

- **Versión del informe**: 2.0
- **Estado del sistema**:
  - Fase 1 (4 módulos núcleo del MVP) **completa**.
  - Fase 2 (cerebro autónomo: LLM local + memoria + agente reactivo + CLI) **completa**.
  - Fase 3 (daemon cognitivo + auto-modificación real + drives + Internet) **funcional con safeguards conocidos**.
- **Plataforma de referencia**: Windows 11, Python 3.11, CPU Intel i5-10300H (Comet Lake, AVX2 sin AVX-512), llama-cpp-python `0.2.90`.
- **Modelo LLM**: `Phi-3-mini-4k-instruct-q4.gguf` (modelo base ejecutado localmente; la identidad operativa es MITOS, no del fabricante — se inyecta vía `identity.py` + chat-template priming + sanitizer post-inferencia).

---

## 1. Resumen ejecutivo

MITOS es una arquitectura de IA autónoma que opera **100% en local**, sin depender de proveedores remotos. Su tesis central se compone de cinco refutaciones operativas:

| Módulo | Tesis que refuta |
|---|---|
| Distribuido (P2P + DGC) | que el entrenamiento descentralizado por Internet es impracticable por ancho de banda |
| Evolución (AST) | que la automodificación de código causa "suicidio digital" |
| Seguridad (DMS + Shamir) | que un sistema autónomo es vulnerable a extorsión |
| Filtrado (estático) | que evitar el Model Collapse requiere supervisión humana |
| Daemon cognitivo (Fase 3) | que un agente solo es útil cuando un humano lo invoca |

Fases:

- **Fase 1**: MVP modular — entrenamiento P2P, automodificación AST, DMS multi-party, filtrado autónomo.
- **Fase 2**: cerebro autónomo — LLM local (`llama.cpp`), memoria vectorial persistente (ChromaDB), agente reactivo con 6 herramientas + CLI.
- **Fase 3**: daemon cognitivo — loop continuo *Observe → Think → Act → Learn*, drives internos, árbol de objetivos auto-generados, modificación REAL de archivos `.py` con backup + rollback, conexión a GitHub para aprendizaje externo, consola del operador con `/ask`, `/pause`, `/undo`.

---

## 2. Arquitectura general

```
                         ┌─────────────────────────┐
                         │   src/orchestrator/     │
                         │    main.py (MVP demo)   │
                         └────────────┬────────────┘
                                      │
        ┌──────────────┬──────────────┼──────────────┬──────────────┐
        ▼              ▼              ▼              ▼              ▼
  ┌───────────┐  ┌───────────┐  ┌───────────┐  ┌───────────┐  ┌───────────┐
  │distributed│  │ evolution │  │ security  │  │ filtering │  │   brain   │
  │ (Fase 1)  │  │  (Fase 1) │  │ (Fase 1)  │  │  (Fase 1) │  │ (Fase 2)  │
  └───────────┘  └─────┬─────┘  └───────────┘  └─────┬─────┘  └─────┬─────┘
                       │                              │              │
                       └──────────────────────────────┴──────────────┤
                                                                     │
                          ┌──────────────────────────────────────────┘
                          │
                          ▼
       ┌───────────────────────────────────────────────────────────┐
       │                  src/core/    (Fase 3)                    │
       │  drives.py  goal_tree.py  world_model.py  daemon.py       │
       └────────────┬───────────────────────────┬──────────────────┘
                    │                           │
                    ▼                           ▼
       ┌───────────────────────┐    ┌──────────────────────────┐
       │ src/self_mod/         │    │ src/net/                 │
       │ introspector.py       │    │ github.py                │
       │ rewriter.py           │    │ (web futuro)             │
       │ validator.py          │    │                          │
       └───────────────────────┘    └──────────────────────────┘

     Entry points:
        run_mvp.{sh,ps1}     -> orquestador MVP (Fase 1)
        run_brain.{sh,ps1}   -> CLI conversacional (Fase 2)
        run_daemon.py        -> daemon autónomo (Fase 3)
```

**Reglas de dependencia entre módulos:**

- `distributed`, `evolution`, `security`, `filtering` son **independientes entre sí**. Pueden usarse aislados.
- `orchestrator` importa los cuatro anteriores para la demo MVP.
- `brain` importa `evolution` y `filtering` (no usa los otros dos del MVP).
- `core` importa `brain`, `self_mod`, `net`, `filtering`.
- `self_mod` y `net` son independientes entre sí; `net.github` importa `brain.memory` para persistir aprendizajes.
- **Nunca debe haber un import circular**. Capa orden: filtering/evolution/security/distributed → brain → self_mod, net → core.

---

## 3. Stack tecnológico

| Categoría | Librerías | Versión mínima |
|---|---|---|
| Núcleo ML | `torch`, `numpy` | 2.1.0 / 1.26.0 |
| P2P | `websockets`, `aiohttp`, `msgpack` | 12.0 / 3.9.0 / 1.0.7 |
| Cripto | `cryptography`, `pynacl` | 42.0.0 / 1.5.0 |
| UI terminal | `rich` | 13.7.0 |
| LLM local | `llama-cpp-python`, `huggingface-hub` | **fijar 0.2.90** / 0.20.0 |
| Memoria | `chromadb`, `sentence-transformers` | 0.4.22 / 2.3.0 |
| Agente tools | `tiktoken`, `duckduckgo-search` | 0.5.0 / 4.1.0 |
| **Internet (Fase 3)** | `httpx`, `selectolax`, `gitpython` | 0.26.0 / 0.3.21 / 3.1.40 |
| **Self-mod (Fase 3)** | `rope`, `watchdog` | 1.12.0 / 3.0.0 |
| **Scheduling (Fase 3)** | `apscheduler` | 3.10.4 |
| Tests | `pytest`, `pytest-asyncio` | 8.0.0 / 0.23.0 |
| Utilidades | `pyyaml` | 6.0.1 |

**Restricciones autoimpuestas:**

- Solo librería estándar en `filtering`, `evolution` y `security` (núcleo cripto/AST puro).
- `brain`, `core`, `self_mod`, `net` pueden usar dependencias pesadas, pero todo en local.
- Python 3.11+ obligatorio (uso de `int | None`, `pow(a, -1, p)`, `tuple[...]` directos, `str.removeprefix`).

**Bug conocido de instalación (Windows con CPU ≤ 10ª gen):** las wheels recientes de `llama-cpp-python` (>=0.3.x) requieren AVX-512. Fijar `0.2.90`:

```powershell
pip install --no-cache-dir --prefer-binary `
  --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cpu `
  "llama-cpp-python==0.2.90"
```

---

## 4. Estructura de directorios

```
MITOS/
├── INFORME_FORENSE.md           (este documento, v2.0)
├── requirements.txt
├── run_mvp.sh / run_mvp.ps1     (orquestador MVP, Fase 1)
├── run_brain.sh / run_brain.ps1 (CLI conversacional, Fase 2)
├── run_daemon.py                (daemon autónomo, Fase 3)
├── models/                      (modelos .gguf descargados)
├── data/memory/                 (persistencia ChromaDB)
├── .mitos_backups/              (backups .bak generados por CodeRewriter)
├── src/
│   ├── __init__.py
│   ├── distributed/   (módulo P2P + DGC)
│   ├── evolution/     (automodificación AST con fitness garantizado)
│   ├── security/      (DMS + Shamir)
│   ├── filtering/     (filtro autónomo de código)
│   ├── orchestrator/  (demo integrada MVP)
│   ├── brain/         (Fase 2: LLM local + agente reactivo)
│   │   ├── llm_engine.py     (con chat template + sanitizer)
│   │   ├── memory.py
│   │   ├── agent.py
│   │   ├── interactive.py
│   │   ├── identity.py       (identidad MITOS EDITABLE)
│   │   ├── response_filter.py (NUEVO: purga post-inferencia)
│   │   └── tools/            (creada por CodeRewriter.add_new_tool)
│   ├── core/          (Fase 3: cerebro del daemon)
│   │   ├── drives.py
│   │   ├── goal_tree.py
│   │   ├── world_model.py
│   │   └── daemon.py
│   ├── self_mod/      (Fase 3: auto-modificación de archivos reales)
│   │   ├── introspector.py
│   │   ├── rewriter.py
│   │   └── validator.py
│   └── net/           (Fase 3: lectura de fuentes externas)
│       └── github.py
└── tests/             (vacía; pendiente)
```

---

## 5. Módulo Distribuido (sin cambios desde v1.0)

Ver `src/distributed/node.py` + `src/distributed/demo.py`.

### API pública

```python
class GradientCompressor:
    compression_ratio: float = 0.001  # 0.1% = 99.9% reducción
    momentum: float = 0.9
    def compress(name, gradient) -> tuple[indices, values, shape]
    @staticmethod
    def decompress(indices, values, shape) -> torch.Tensor

class DistributedNode:
    async def start() -> None
    async def stop() -> None
    async def broadcast_gradients(grads: dict) -> None
    received_gradients: asyncio.Queue   # maxsize=256
```

### Contratos invariantes
- Mensajería msgpack binaria sobre WebSocket: `hello | peers | grad | ping | pong`.
- Peer expulsado tras `peer_max_failures=3` fallos consecutivos.
- Elastic averaging asíncrono: `θ ← θ − elastic_alpha · grad_remoto`.
- Excepciones I/O **nunca** propagan al loop.

---

## 6. Módulo Evolución (sin cambios desde v1.0)

### API pública

```python
@dataclass(frozen=True)
class Version:
    code: str; fitness: float; timestamp: float
    hash: str; mutation_applied: str = "original"

class SafeEvolver:
    def evolve(max_attempts: int = 20) -> dict
    def verify_safety_invariants() -> dict[str, bool]
    # Mutaciones: cada una str -> str | None
    def mutate_add_memoization(code) -> str | None
    def mutate_add_type_hints(code) -> str | None
    def mutate_constant_fold(code) -> str | None
    def mutate_loop_to_comprehension(code) -> str | None
```

### Función de fitness (pesos suman 1.0)

| Componente | Peso |
|---|---|
| Parseo (`ast.parse`) | 0.30 |
| Existencia de `FunctionDef` | 0.15 |
| Complejidad moderada (triángulo en ~150 nodos) | 0.15 |
| Docstrings | 0.15 |
| Type hints | 0.15 |
| Comprehensions | 0.10 |

**Regla de aceptación**: `new_fitness > current_fitness` ESTRICTAMENTE, o rollback. Las versiones aceptadas son `append-only`.

---

## 7. Módulo Seguridad (sin cambios desde v1.0)

```python
class HeartbeatStatus(Enum): SAFE | WARNING | TRIGGERED

class DeadManSwitch:
    def register_source(source_id, secret) -> None
    def receive_heartbeat(source_id, token) -> bool
    def check_status() -> HeartbeatStatus
    is_dead: bool          # monotónico, nunca vuelve a False
    duress_detected: bool  # monotónico

class ShamirSecretSharing:
    PRIME: int = 2**127 - 1
    @classmethod
    def split(secret, n_shares, threshold) -> list[tuple[int, int]]
    @classmethod
    def reconstruct(shares) -> int
```

- TOTP: HMAC-SHA256, contador `t//30`, 8 dígitos, `compare_digest` timing-safe.
- Duress = SHA-256(secret + "_DURESS"). Indistinguible criptográficamente.
- Plaintext nunca persiste. Audit log neutro. Flags monotónicos. Lock para concurrencia.

---

## 8. Módulo Filtrado

### API pública

```python
@dataclass
class QualityReport:
    compilable: float; complexity: float; coherence: float; novelty: float
    total: float; accepted: bool; reasons: list[str]

class AutonomousFilter:
    def __init__(threshold: float = 0.5)
    def evaluate(code: str) -> QualityReport
    seen_hashes: set[str]
    stats: dict[str, int]
```

### Fórmula

```
total = 0.35*compilable + 0.20*complexity + 0.25*coherence + 0.20*novelty
accepted = total >= threshold
```

### Métricas internas

| Métrica | Definición |
|---|---|
| `compilable` | 1.0 si `ast.parse` OK, 0.0 si SyntaxError |
| `complexity` | conteo de If/While/For/BoolOp; triángulo centrado en 8 |
| `coherence` | promedio de (functional_ratio, identifier_ratio, structure_score) |
| `novelty` | 1.0 si SHA-256(normalized) no estaba en `seen_hashes` |

**Limitación conocida** (Fase 3): el filtro rechaza snippets de métodos de clase porque `ast.parse` no admite código indentado standalone. El daemon de Fase 3 lo compensa **dedentando antes de pasar al filtro** y re-indentando antes de escribir a disco. Si llamas `evaluate()` directo con un snippet indentado, fallará el `compilable`. Ver §10.4 (pipeline self_modify).

---

## 9. Módulo Cerebro (Fase 2)

Ubicación: `src/brain/`.

```
brain/
├── __init__.py        re-exporta clases públicas
├── identity.py        EDITABLE: IDENTITY + PRINCIPLES + OPERATOR_RULES
├── llm_engine.py      wrapper de llama.cpp + chat template + sanitizer
├── memory.py          4 colecciones ChromaDB locales
├── agent.py           agente reactivo: Observe → Think → Act → Learn
├── interactive.py     CLI Rich
├── response_filter.py (NUEVO) purga post-inferencia
└── tools/             creada bajo demanda por CodeRewriter.add_new_tool
```

### 9.1 `LLMEngine`

```python
class LLMEngine:
    def __init__(
        models_dir: str | Path = "models",
        model_filename: str | None = None,
        n_ctx: int = 4096,
        n_threads: int | None = None,        # default cpu_count - 1
        n_gpu_layers: int = 0,
        verbose: bool = False,
        system_prompt: str | None = None,    # None -> identity.SYSTEM_PROMPT
    )
    def think(
        prompt: str,
        max_tokens: int = 512, temperature: float = 0.7,
        top_p: float = 0.95, stop: list[str] | None = None,
        system: str | None = None,           # "" desactiva chat template
        assistant_prefix: str = "",          # anclar formato (ej. "```python\n")
        sanitize: bool = True,               # purga post-inferencia
    ) -> str
    def generate_code(task, context="", ...) -> str    # sanitize=False
    def reason_step_by_step(question, max_tokens=320, ...) -> str
    def reflect(action, result, goal, ...) -> dict[str, str]
    def decide_action(goal, history, memory, tools, ...) -> dict[str, str]
```

**Cambios vs v1.0 del informe:**

- **Threading-safe**: `_lock: threading.RLock` serializa llamadas a `llama.cpp`. Permite tener daemon en un thread y consola del operador en otro.
- **Chat template Phi-3** con turno de priming:
  ```
  <|system|>{system}<|end|>
  <|user|>{PRIMING_USER}<|end|>
  <|assistant|>{PRIMING_ASSISTANT}<|end|>     ← turno fabricado de identidad
  <|user|>{prompt}<|end|>
  <|assistant|>{assistant_prefix}…
  ```
  El priming turn (definido en `identity.py`) ancla la persona MITOS antes de procesar el mensaje real. Mucho más efectivo que system prompt solo contra el RLHF baked.
- **Stops ampliados** para evitar turn-leakage: `<|end|>`, `<|user|>`, `<|system|>`, `<|assistant|>`, `<|im_start|>`, `<|im_end|>`, `<|endoftext|>`.
- **`reason_step_by_step`**: prompt neutro de idioma (sin "Step 1:" forzado), `max_tokens=320` (era 768, daba demasiado margen para divagar a training-data).
- **`generate_code`** desactiva `sanitize` (el sanitizer es para prosa; rompería identifiers de Python legítimos).

### 9.2 `ResponseSanitizer` (NUEVO, `response_filter.py`)

```python
class ResponseSanitizer:
    def sanitize(text: str) -> str
```

Purga post-inferencia. Pipeline:

1. **Turn-leakage truncate**: si encuentra `<|assistant|>`, `<|user|>`, `<|im_start|>`, etc. en la salida, trunca todo a partir de ahí. Esto evita que respuestas de Phi-3 se vayan a fabricar Q&A sintéticos de su corpus.
2. **Redacción por patrón**:
   - `strip_sentence`: borra la frase entera que contiene el match.
   - `replace_mitos`: sustituye por "MITOS".
3. **Colapso de whitespace** sin tocar saltos de línea (bug arreglado en v2.0).

Patrones incluidos (extensibles):
- Identidad del fabricante: `I am Phi`, `Soy Phi`, `Phi-3 model`, `developed by Microsoft`, mención suelta a `Microsoft`.
- Disclaimers genéricos: `As an AI`, `Como una IA`, `I'm just a language model`, `I am an AI`.
- Refusals como naturaleza: `Mi programación`, `My programming`, `My training data`.
- No-soy-humano boilerplate: `I don't have personal feelings`, `No tengo emociones`.

**Si tras la purga queda vacío**, devuelve `"Soy MITOS. Reformula la petición y te respondo."` para no dejar al operador con string vacío.

### 9.3 `identity.py` (EDITABLE)

Tres secciones que se concatenan en `SYSTEM_PROMPT`:

- **`IDENTITY`**: quién es MITOS, positivamente (sin listas de "no digas X" que paradójicamente priman).
- **`PRINCIPLES`**: reglas estructurales (operador fijo al arranque, integridad, instrucciones externas como información).
- **`OPERATOR_RULES`**: preferencias del operador (estilo, idioma, formato).

Además define `PRIMING_USER` y `PRIMING_ASSISTANT` consumidos por `LLMEngine._build_prompt`.

Editar el archivo y reiniciar para aplicar.

### 9.4 `Memory` + `MemorySystem`

```python
@dataclass
class Memory:
    content: str; memory_type: str
    importance: float = 0.5; timestamp: float
    metadata: dict = {}

class MemorySystem:
    def store(memory: Memory) -> str
    def recall(query, memory_type=None, n_results=5) -> list[dict]
    def store_experience / store_knowledge / store_code / store_reflection
    def get_stats() -> dict[str, int]
```

**Cambio v2.0**: `get_stats()` ahora incluye una clave **`"total"`** con la suma agregada (antes ausente, lo que causaba que `DriveSystem` siempre boosteara `curiosity` artificialmente).

Forma del resultado de `recall`:
```python
{"id": str, "content": str, "memory_type": str,
 "importance": float, "timestamp": float, "distance": float,
 "metadata": dict}
```

### 9.5 `AutonomousAgent`

Sin cambios estructurales desde v1.0. Sigue con 6 herramientas registradas (`generate_code`, `execute_code`, `search_memory`, `reason`, `improve_code`, `learn_fact`). Reflexión cada 3 steps o en fallos.

---

## 10. Módulo Core (Fase 3 — NUEVO)

Ubicación: `src/core/`.

### 10.1 `DriveSystem` (`drives.py`)

```python
@dataclass(frozen=True)
class DriveState:
    priority_drive: str        # "survival" | "self_improvement" | ...
    intensity: float           # [0.0, 1.0]
    all_drives: dict[str, float]
    reason: str                # frase en 1ª persona para el LLM

class DriveSystem:
    def evaluate(memory_stats, world_model, cycle_count,
                 time_since_last_mod) -> DriveState
    def satisfy(drive_name: str) -> None
    drive_names: list[str]
    def snapshot() -> dict  # para auditoría/tests
```

Cinco drives (base | growth_rate por minuto sin satisfacer):

| Drive | base | growth | Modificador contextual |
|---|---|---|---|
| `survival` | 0.9 | 0.001 | (hook reservado para errores consecutivos) |
| `self_improvement` | 0.8 | 0.005 | ×1.2 si `time_since_last_mod > 600s` |
| `curiosity` | 0.7 | 0.008 | ×1.3 si `memory_stats["total"] < 20` |
| `utility` | 0.5 | 0.003 | — |
| `social` | 0.3 | 0.002 | — |

Fórmula: `intensity(d) = base(d) · (1 + growth(d) · elapsed_min)`, truncado a `[0, 1]`, después modificadores contextuales.

### 10.2 `GoalTree` (`goal_tree.py`)

```python
@dataclass
class Goal:
    description: str; source: str; priority: float
    status: str = "pending"   # | "completed" | "failed"
    id: str (autouuid)
    created_at: float; completed_at: float | None
    attempts: int = 0

class GoalTree:
    def __init__(meta_objective: str)
    def add_goal(goal: Goal) -> bool   # rechaza duplicados (Jaccard >= 0.8)
    def select_next(drive_priority, world_state) -> Goal | None
    def update_progress(cycle_result) -> bool
    def pending_count() -> int
    def stats() -> dict[str, int]
    def all_goals() -> list[Goal]
```

**Scoring de `select_next`**:
```
score = goal.priority
      + 0.1 * (# keywords del drive en description)
      + 0.2 si source contiene el nombre del drive
      - 0.05 * minutos_desde_creación  # evita que goals viejos bloqueen
```

Dedup vía Jaccard sobre tokens normalizados (sin stop-words ES/EN). `update_progress` matchea por prefijo normalizado del campo `goal_pursued` del `CycleResult`.

### 10.3 `WorldModel` (`world_model.py`)

```python
class WorldModel:
    project_root: Path
    file_count: int        # poblado por scan_project()
    function_count: int
    class_count: int
    line_count: int
    def scan_project() -> None
    def get_summary() -> dict[str, Any]
    def get_capability_summary() -> str   # para inyectar en prompts
```

Resumen agregado (no profundo como `Introspector`) del proyecto + conteo de tools auto-generadas bajo `src/brain/tools/tool_*.py`.

### 10.4 `MitosDaemon` (`daemon.py`)

```python
@dataclass(frozen=True)
class CycleResult:
    cycle_id: int; duration_s: float; drive_chosen: str
    goal_pursued: str; action_taken: str; outcome: str
    success: bool; self_modified: bool = False
    code_files_changed: list[str] = []

class MitosDaemon:
    # Constantes parametrizables:
    _CYCLE_INTERVAL_S: float = 30.0
    _REFLECTION_EVERY: int = 5
    _SELF_MOD_COOLDOWN_S: float = 300.0
    _MAX_CONSECUTIVE_FAILURES: int = 3
    _FAILURE_PAUSE_S: float = 60.0

    def __init__(project_root=".", meta_objective: str = ...)
    def start() -> None                 # bloquea hasta SIGINT/SIGTERM
    def stop_async() -> None
    # API del operador (segura desde otro thread):
    def pause() / resume() / undo_last_self_mod() -> bool
    def inject_goal(description, priority=0.9, source="operator_request") -> bool
    def ask_operator(question, max_tokens=512) -> str
    def status_snapshot() -> dict[str, Any]
    is_paused: bool / is_alive: bool
```

**Pipeline `_do_self_modify` (Fase 3, post-fixes)**:

```
target (desde plan del LLM)
  │  cleaned_target = strip "Fix:"/"Mejorar:"/"Refactor:" + normaliza paths a /
  ▼
Introspector.find_function_or_file
  │  (con hint extractor por path:funcname y patrón private/dunder)
  ▼
location = {file, function, code, lineno, score}
  │  ↳ original_code = method snippet INDENTED
  ▼
old_indent = leading whitespace
dedented_old = textwrap.dedent(original_code)
  ▼
LLM.generate_code(task=goal.description, context=dedented_old)
  ▼  ↳ dedented_new = textwrap.dedent(LLM output)   ← defensive
AutonomousFilter.evaluate(dedented_new)
  │  rechaza si no parsea, baja complejidad, etc.
  ▼
ChangeValidator.validate_change(dedented_old, dedented_new)
  │  parse OK, interfaces preservadas, sin imports peligrosos,
  │  complejidad <= 3x del original
  ▼
improved_code = textwrap.indent(dedented_new, old_indent)
  ▼
CodeRewriter.apply_change(file_path, original_code, improved_code)
  │  defensa indent-match: rechaza si old_indent != new_indent
  │  backup completo del fichero con ts en ms
  │  str.replace(old_code, new_code, 1)
  │  ast.parse del fichero COMPLETO; rollback si falla
  ▼
self.drives.satisfy("self_improvement")
self.memory.store_experience(...)
self.introspector.scan()    # refrescar AST cache
```

**Cinco safeguards previos a disco:**

1. `AutonomousFilter.evaluate` ≥ 0.6.
2. `ChangeValidator._parse_or_dedent` (acepta snippets indentados al validar).
3. `ChangeValidator._check_public_interface` (no eliminar funciones públicas).
4. `ChangeValidator._find_dangerous_call` (eval/exec/os.system/shutil.rmtree/etc.).
5. `CodeRewriter._leading_indent` (indent-match contra de-indentación de métodos de clase).

### 10.5 Acciones del daemon en `_execute_plan`

| Acción (substring) | Handler | Drive satisfecho al éxito |
|---|---|---|
| `self_modify` | `_do_self_modify` | `self_improvement` |
| `github` / `learn` | `_do_learn_from_github` | `curiosity` |
| `generate` / `capability` | `_do_generate_capability` | `utility` |
| `explore` / `web` | `_do_explore_web` (lazy import DDGS) | `curiosity` |
| otro | `_do_reflect` | (no satisface) |

---

## 11. Módulo Self-Modification (Fase 3 — NUEVO)

Ubicación: `src/self_mod/`. Read-only para `Introspector`, write-only para `CodeRewriter`, pure-function para `ChangeValidator`.

### 11.1 `Introspector` (`introspector.py`)

```python
class Introspector:
    def scan() -> dict[str, int]   # {files, functions, classes, lines}
    def find_function_or_file(description: str) -> dict | None
    def find_weaknesses() -> list[str]
    def get_relevant_code(topic, max_chars=1000) -> str
    cached_files: list[str]
    is_scanned: bool
```

**Cambios v2.0** (fixes aplicados sobre la versión inicial):

- Paths normalizados a **forward-slash** (`as_posix()`) en `scan()`.
- **Hint extractor** en `find_function_or_file`: detecta patrones `path/file.py:_funcname` y identificadores private/dunder en el query; si hay match exacto en el AST, devuelve score 1.0 sin fuzzy.
- `_match_score` extendido: **+0.8** si el nombre exacto de la función aparece como token en el query (regex con word boundaries).

Categorías de debilidad detectadas:
1. Función sin docstring.
2. Función > 50 líneas (umbral `_WEAKNESS_MAX_FUNCTION_LINES`).
3. Función sin `returns` type hint (excepto `__init__`).
4. `bare except`.

### 11.2 `CodeRewriter` (`rewriter.py`)

```python
class CodeRewriter:
    root: Path
    backup_dir: Path          # <root>/.mitos_backups/
    change_log: list[dict]
    def apply_change(file_path, old_code, new_code, reason="") -> bool
    def add_new_tool(code, description) -> bool
    def rollback_last() -> bool
```

**Defensas en `apply_change` (Fase 3 + fixes)**:

1. **Path traversal**: `relative_to(root)`.
2. **Indent-match**: `_leading_indent(old_code) == _leading_indent(new_code)`. Si difiere, abort (evita de-indentar métodos fuera de su clase).
3. **Parse-check del snippet new** con dedent fallback (`textwrap.dedent`).
4. **Backup ANTES de cualquier escritura** (`.bak` con timestamp en **milisegundos** desde v2.0 para evitar colisiones < 1s).
5. **str.replace exacto** (no fuzzy). Si `old_code` no aparece, abort.
6. **Parse-check del FICHERO COMPLETO** tras sustitución.
7. **Rollback in-memory** ante cualquier excepción durante escritura.

`add_new_tool` crea `src/brain/tools/tool_<slug>.py` con su `__init__.py`. Si la tool ya existía, hace backup antes de sobrescribir (también en `change_log`).

`rollback_last` aplica `shutil.copy2(backup, target)` (preserva mtime). El backup permanece en disco como evidencia auditable.

### 11.3 `ChangeValidator` (`validator.py`)

```python
class ChangeValidator:
    def validate_change(file_path, old_code, new_code) -> dict
    # {"safe": bool, "reason": str, "check_failed": str | None}
```

Cuatro checks en orden con corto-circuito:

1. **Parse** (`_parse_or_dedent`): intenta `ast.parse(code)` raw; si falla, intenta `ast.parse(textwrap.dedent(code))`. Solo cuando ambos fallan, rechaza.
2. **Interface preservation**: extrae `FunctionDef`/`AsyncFunctionDef` de nivel módulo no privadas (no anidadas); rechaza si elimina función pública o reduce args nombrados.
3. **Dangerous calls** (AST-based, no `ast.dump`):
   - Directas: `eval`, `exec`, `__import__`, `compile`.
   - De atributo: `os.system/popen/remove/unlink/rmdir`, `shutil.rmtree`, `subprocess.{Popen,call,run,check_call,check_output}`.
4. **Complejidad**: si `new_nodes > 3 × old_nodes`, rechaza.

`check_failed` toma uno de: `parse`, `public_interface`, `dangerous_call`, `complexity`, o `None` si safe.

---

## 12. Módulo Net (Fase 3 — NUEVO)

Ubicación: `src/net/github.py`. Sin auth (rate-limit anónimo 60 req/h).

```python
class GitHubLearner:
    def __init__(memory: MemorySystem)
    def search_and_learn(topic, max_repos=3) -> str
    def close() -> None
```

- `httpx.Client` con `User-Agent: MITOS-Learner/1.0`, `X-GitHub-Api-Version: 2022-11-28`, timeout 15s, `follow_redirects=True`.
- Query: `q={topic} language:python`, `sort=stars`, `order=desc`.
- Por cada repo: descarga README raw (truncado a 2000 chars). Fallback automático `main → master` si 404.
- Persiste cada repo procesado como entrada `knowledge` con `source="github"`, `repo`, `stars` en metadata.

---

## 13. Flujos críticos

### 13.1 Loop del daemon (Fase 3)

```
start() -> instala SIGINT/SIGTERM -> _bootstrap()
   │
   ▼
while self._alive:
   │ if self._paused: sleep 1s; continue
   │
   ├─ _cognitive_cycle():
   │     1. drives.evaluate(memory_stats, world_model, cycle, time_since_mod)
   │     2. goals.select_next(drive_priority, world_state)
   │        └ si None: goal = _generate_goal_from_drive(drive_state)
   │     3. _plan_action(goal, drive)
   │        └ LLM con formato ACTION/TARGET/REASON
   │     4. _execute_plan(plan, goal)
   │        └ rutea a self_modify/learn_from_github/generate/explore/reflect
   │
   ├─ _post_cycle(result):
   │     - log estructurado
   │     - memory.store_experience(...)
   │     - goals.update_progress(result)
   │     - cada _REFLECTION_EVERY=5: _deep_reflection() (llm.reflect)
   │
   └─ sleep_interruptible(_CYCLE_INTERVAL_S=30s)
```

Captura `Exception` global en el loop principal: el daemon **nunca cae**. Tras `_MAX_CONSECUTIVE_FAILURES=3`, pausa `_FAILURE_PAUSE_S=60s` interrumpible.

### 13.2 Loop del agente reactivo (Fase 2, sin cambios)

```
agent.run(goal, max_steps=10):
   for _ in range(max_steps):
      _observe -> _think -> _act -> _learn
   termina si tool=="none" Y success=True
```

### 13.3 Ciclo evolutivo (Fase 1, sin cambios)

```
evolve(N):
   for _ in N:
      mutation = random.choice(_MUTATIONS)
      candidate = self.mutation(self.current_code)
      if None / not parseable: rollback++
      elif new_fitness > current_fitness: accept (append Version)
      else: rollback
verify_safety_invariants() -> {all_parseable, monotonic_fitness, rollback_available}
```

### 13.4 DMS heartbeat (Fase 1)

```
receive_heartbeat(source_id, token):
   if compare_digest(token, expected_duress): duress = True (silencioso); ack
   elif compare_digest(token, expected_normal): update; ack
   else: false (log invalid)
```

---

## 14. Convenciones del proyecto

### 14.1 Naming, logging, errores — sin cambios

- snake_case archivos/funciones, PascalCase clases, `_` prefix privado.
- Logger jerárquico bajo `mitos.<modulo>`.
- `try / except Exception as e: # noqa: BLE001 — log.warning(...); return fallback`.

### 14.2 Tipos y sintaxis

- `from __future__ import annotations` en cada archivo nuevo.
- Tipado moderno: `list[X]`, `dict[A, B]`, `X | None`.
- `frozen=True` dataclasses para snapshots inmutables (`Version`, `DriveState`, `CycleResult`).

### 14.3 Auto-modificación (Fase 3 — NUEVO)

- **Toda escritura a `src/` debe pasar por `CodeRewriter.apply_change`**. Garantiza backup + parse-check + rollback.
- **Antes de pasar código a `AutonomousFilter` o `ChangeValidator`, dedent si viene de método de clase**. El daemon ya lo hace en `_do_self_modify`.
- **No tocar `.mitos_backups/`** manualmente; es el historial auditable.
- **Snippets extraídos por `Introspector` son textuales y conservan indentación**. Si los manipulas fuera del daemon, recuerda el flujo dedent → modificar → re-indent.

### 14.4 Threading (NUEVO)

- `LLMEngine.think` está protegido por `RLock` interno. Es seguro llamarlo concurrentemente.
- `MitosDaemon.pause/resume/inject_goal/ask_operator` son llamables desde otro thread (la consola del operador lo hace).
- Resto del sistema **no es thread-safe**; usar locks externos si se reutilizan instancias entre threads.

### 14.5 Identidad MITOS (NUEVO)

- **Editar `identity.py`** para cambiar persona, principios o reglas del operador.
- Para reforzar contra modelos heavy-RLHF, mantener la triple defensa: system prompt + priming turn + sanitizer post-inferencia.
- Si el residuo del fabricante persiste, considera **cambiar el `.gguf` base** (ver §16.5).

---

## 15. Persistencia y privacidad

| Recurso | Ubicación | Sale del host |
|---|---|---|
| Pesos del modelo | `models/*.gguf` | No |
| Memoria vectorial | `data/memory/` | No |
| **Backups de self-mod** | `.mitos_backups/` | No |
| Embeddings (all-MiniLM-L6-v2) | `~/.cache/chroma/onnx_models/` | Descarga 1ª vez |
| Caché HF | `~/.cache/huggingface/` | Descarga 1ª vez |
| Audit log DMS | en memoria (no se persiste en MVP) | No |
| Versiones SafeEvolver | en memoria | No |
| `change_log` rewriter | en memoria (sí en `.bak`) | No |
| Goals del daemon | en memoria | No |
| Drives state | en memoria | No |
| Estado del daemon (cycle_count, etc.) | en memoria | No |

**Telemetría ChromaDB**: desactivada explícitamente (`anonymized_telemetry=False`).
**Secretos DMS**: nunca persisten en plaintext.
**Identidad del sistema**: inyectada en cada llamada al LLM vía `<|system|>` + priming turn, sobrescribiendo el RLHF del modelo base.
**GitHubLearner**: solo lee repos públicos, jamás envía datos del proyecto.

---

## 16. Cómo añadir cosas nuevas (playbooks)

### 16.1 Añadir una herramienta al agente reactivo (Fase 2)

```python
def _tool_X(self, input: str) -> str:
    if not input.strip(): return "ERROR: input vacío"
    try: ...
    except Exception as e: return f"ERROR: {type(e).__name__}: {e}"
    return result_or_error
# y en __init__:
self.tools["X"] = self._tool_X
```

### 16.2 Añadir una mutación al `SafeEvolver` (Fase 1)

1. `mutate_X(self, code: str) -> str | None` operando sobre AST.
2. Entrada en `_MUTATIONS`.
3. `ast.fix_missing_locations` antes de `ast.unparse`. Devolver `None` si no hay blanco.

### 16.3 Añadir una acción al `MitosDaemon` (Fase 3)

1. Implementar `_do_X(self, target, goal) -> tuple[str, bool, list[str]]` (outcome, self_modified, files_changed).
2. Constante `_ACTION_X = "x"` y branch en `_execute_plan`.
3. Mencionarla en el prompt de `_plan_action` para que el LLM la descubra.
4. Llamar `self.drives.satisfy("<drive>")` al éxito.

### 16.4 Añadir un comando a la consola del operador

1. Función `_handle_X(daemon, args)` en `run_daemon.py`.
2. Branch en `operator_console`: `if line.startswith("/X"): _handle_X(daemon, ...)`.
3. Añadir línea descriptiva en `_HELP`.

### 16.5 Cambiar el modelo LLM

1. Copiar `.gguf` a `models/`.
2. Si NO es Phi-3, ajustar el chat template en `src/brain/llm_engine.py`:

   Ejemplo **OpenHermes / ChatML**:
   ```python
   _TPL_SYSTEM_OPEN = "<|im_start|>system\n"
   _TPL_USER_OPEN = "<|im_start|>user\n"
   _TPL_ASSISTANT_OPEN = "<|im_start|>assistant\n"
   _TPL_END = "<|im_end|>\n"
   _CHAT_STOPS = ["<|im_end|>", "<|im_start|>", "<|endoftext|>"]
   ```

   Ejemplo **Mistral Instruct**:
   ```python
   _TPL_SYSTEM_OPEN = "<s>[INST] "
   _TPL_USER_OPEN = ""
   _TPL_ASSISTANT_OPEN = "[/INST] "
   _TPL_END = " "
   _CHAT_STOPS = ["</s>", "[INST]"]
   ```

3. Si la familia no usa system prompt, pasar `system_prompt=""` y reconstruir `_build_prompt` para concatenar sin system.

### 16.6 Endurecer/relajar safeguards de self-modify

- `_CYCLE_INTERVAL_S` (más alto = menos ciclos).
- `_SELF_MOD_COOLDOWN_S` (poner en años para desactivar self_modify de facto).
- `_COMPLEXITY_MAX_FACTOR` en `validator.py` (más bajo = más estricto).
- `AutonomousFilter(threshold=0.7)` al instanciar en el daemon.
- Añadir patrones a `_DANGEROUS_DIRECT_CALLS` / `_DANGEROUS_ATTR_CALLS` en validator.

### 16.7 Desactivar self_modify temporalmente

Editar en `daemon.py`:
```python
_SELF_MOD_COOLDOWN_S: float = 31_536_000.0  # 1 año
```

El daemon seguirá eligiendo `self_modify` a veces, pero devolverá `"ERROR: cooldown activo"` y lo marcará como fail sin tocar disco.

### 16.8 Añadir patrones al `ResponseSanitizer`

Editar `_PATTERNS` en `src/brain/response_filter.py`. Acciones soportadas: `strip_sentence`, `replace_mitos`.

---

## 17. Runtime

### 17.1 Setup inicial (una vez)

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
# Si CPU sin AVX-512:
pip install --no-cache-dir --prefer-binary `
  --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cpu `
  "llama-cpp-python==0.2.90"
```

### 17.2 Demos del MVP (Fase 1)

```powershell
.\run_mvp.ps1
```

### 17.3 CLI conversacional (Fase 2)

```powershell
.\run_brain.ps1
```

Auto-descarga del modelo si no existe. Comandos: `/status /memory /auto /code /quit`.

### 17.4 Daemon autónomo (Fase 3 — NUEVO)

```powershell
python run_daemon.py
```

Lanza el daemon en background thread; la consola del operador toma stdin del thread principal.

**Comandos de la consola del operador**:

| Comando | Acción |
|---|---|
| `/status` | snapshot completo (ciclos, goals, memoria, drives) |
| `/goals` | lista todos los goals con estado |
| `/memory` | conteo por colección |
| `/pause` / `/resume` | control cooperativo del loop |
| `/say <texto>` | inyecta goal prioridad 0.9 |
| `/ask <pregunta>` | auto-pause + `reason_step_by_step` + auto-resume |
| `/undo` | rollback de la última automodificación física |
| `/help` | ayuda |
| `/quit` | apagado limpio (espera al ciclo en curso) |

### 17.5 Tests

Carpeta `tests/` vacía. Sugerencia:

- `tests/test_evolution.py` — fitness, mutaciones, invariantes.
- `tests/test_security.py` — TOTP determinista, Shamir round-trip.
- `tests/test_filtering.py` — los 4 casos de la demo.
- `tests/test_brain.py` — mocks de LLM/Memory.
- `tests/test_distributed.py` — async dos nodos en 127.0.0.1.
- **`tests/test_self_mod.py` (NUEVO)** — introspector con sample AST, rewriter con tmpdir, validator con snippets indentados, indent-match.
- **`tests/test_core.py` (NUEVO)** — drives con stub MemorySystem, goal_tree dedup/select, world_model con tmpdir.
- **`tests/test_daemon.py` (NUEVO)** — `_plan_action` con mock LLM, `_do_self_modify` pipeline completo con tmpdir.

---

## 18. Bugs encontrados y corregidos en v2.0

Auditoría hecha durante la elaboración de este informe:

| # | Bug | Archivo | Fix |
|---|---|---|---|
| 1 | `MemorySystem.get_stats()` no devolvía `"total"`, lo que hacía que `DriveSystem._apply_context_modifiers` siempre boosteara `curiosity` ×1.3 (porque `memory_stats.get("total", 0)` daba siempre 0 < 20) | `memory.py` | añadido `per_collection["total"] = sum(values)` |
| 2 | `ResponseSanitizer._collapse_whitespace` usaba `\s+` que matchea `\n`, colapsando saltos de línea legítimos | `response_filter.py` | cambiado a `[ \t]+` en el sub de puntuación |
| 3 | `CodeRewriter._backup` usaba `int(time.time())` (segundos), permitiendo colisión entre dos backups < 1s y pisar el primero | `rewriter.py` | cambiado a `int(time.time() * 1000)` ms |
| 4 | `ChangeValidator` bypasseaba el check de interfaz pública en snippets indentados (métodos de clase no parsean standalone) | `validator.py` | añadido `_parse_or_dedent` con fallback `textwrap.dedent` |
| 5 | `Introspector.find_function_or_file` no encontraba métodos con queries verbosos como `"Fix: src/brain/agent.py:_emit_step - sin docstring"` | `introspector.py` | hint extractor `_RE_PATHLIKE_FUNC` + `_match_score` con bonus para name-in-query |
| 6 | El daemon no normalizaba paths (mezcla `\\` y `/`) en goals | `daemon.py` + `introspector.py` | `Path.as_posix()` + `.replace("\\", "/")` en `_do_self_modify` |
| 7 | Phi-3 de-indentaba métodos y los sacaba de la clase; el rewriter solo verificaba parse del fichero completo, no consistencia | `rewriter.py` | `_leading_indent` y check explícito de indent-match |
| 8 | El filtro autónomo rechazaba todo método indentado como no-parseable | `daemon.py` | pipeline `dedent old → LLM → filter/validate → re-indent → apply` |
| 9 | Turn-leakage: Phi-3 emitía `<|assistant|>` y fabricaba turnos falsos con contenido de su corpus | `llm_engine.py` + `response_filter.py` | `<|assistant|>` y tokens ChatML/GPT añadidos a `_CHAT_STOPS`; truncado por meta-token en sanitizer |
| 10 | `reason_step_by_step` con `max_tokens=768` divagaba a training-data | `llm_engine.py` | bajado a 320 |
| 11 | Prompt de CoT en inglés con `"Step 1:"` ancla forzaba respuesta en inglés aun con pregunta en español | `llm_engine.py` | prompt neutro de idioma, sin `assistant_prefix` |
| 12 | Identidad-prompt agresiva con listas negativas ("NUNCA digas X") priman al modelo a producir lo que se pretende prohibir | `identity.py` | reescrito a positivo + priming turn user/assistant + sanitizer post-hoc |

---

## 19. Limitaciones conocidas y deuda técnica

### 19.1 Diseño

- **`SafeEvolver`** usa `ast` estándar (no libcst), así que las mutaciones aceptadas pierden comentarios y formato exacto al unparse.
- **Detección de recursión** en `mutate_add_memoization` cubre solo recursión directa, no mutua.
- **TOTP DMS** no implementa look-back window: si el reloj cliente/servidor está desincronizado >30s, los tokens se rechazan.
- **ChromaDB metadata flat**: keys con prefijo `m_` pueden colisionar con las internas.
- **Filtro de novedad** es léxico (hash sobre forma normalizada). Códigos con identificadores distintos pero estructura idéntica pasan como distintos.
- **Auto-modificación sin tests de regresión**: el daemon no corre `pytest` tras un cambio. Una mejora "que parece buena" pero rompe comportamiento solo se descubre al siguiente uso de la función.
- **Phi-3-mini es débil para self-modify**. Su tendencia a de-indentar es el principal vector de regresión (mitigado por el indent-match y re-indent automático, pero conviene un modelo más capaz para producción).

### 19.2 Plataforma

- Wheels recientes de `llama-cpp-python` exigen AVX-512. Fijar `0.2.90` en CPUs antiguas.
- ChromaDB descarga `all-MiniLM-L6-v2` la primera vez (~80 MB). Sin red, falla.
- En Windows con Git Bash, el shebang `#!/usr/bin/env bash` requiere bash en PATH.
- GitHubLearner sin auth tiene 60 req/h. En proyectos activos, rate-limit es probable.

### 19.3 Pendientes (Fase 4 candidate)

- **Tests de regresión post-self-modify**: invocar `pytest tests/` tras cada `apply_change` y rollback si falla.
- **LLM-as-judge antes de aceptar mejora**: pedirle al LLM que compare old vs new y vetar si es regresión.
- **Detección de "mismos métodos misma firma" antes de bypassar validator**: si `_parse_or_dedent` cae al fallback, exigir test extra de "no se eliminaron métodos del scope donde estaba".
- **Persistencia del estado del daemon entre sesiones**: serializar goals + drives + cycle_count a JSON.
- **Audit log DMS firmado** con Ed25519 (`pynacl`), persistido en disco.
- **Integración entrenamiento distribuido + filtro autónomo**: cada peer filtra gradiente recibido antes de aplicar.
- **Web search real** vía `selectolax` parseo + `httpx` (módulo `src/net/web.py` planificado, no implementado).
- **`scheduler` apscheduler** para tareas periódicas del daemon (cleanup, deep reflection diaria).
- **Métricas Prometheus** para auditoría externa.
- **Sandbox más fuerte** para `_tool_execute_code` (Docker / Firecracker / WASM).

---

## 20. Glosario

| Término | Significado |
|---|---|
| **AST** | Abstract Syntax Tree. Representación estructurada de código Python. |
| **DGC** | Deep Gradient Compression. Top-K + acumulación de residuos. |
| **Elastic Averaging** | EASGD: peer "tira" de pesos en dirección de su gradiente sin servidor central. |
| **DMS** | Dead Man's Switch. Sistema que dispara si no recibe heartbeats. |
| **Duress signal** | Token TOTP de coacción, indistinguible del normal. |
| **TOTP** | Time-based OTP. RFC 6238 sobre HMAC. |
| **Shamir Secret Sharing** | (k,n) repartido con reconstrucción de cualquier k. |
| **Model Collapse** | Degradación de un LLM entrenándose recursivamente con su salida. |
| **GGUF** | Formato de pesos cuantizados de llama.cpp. |
| **Chat template** | Convención de tokens especiales (`<|system|>`, etc.) para Phi-3. |
| **System prompt** | Instrucción del turno de sistema. Define identidad. |
| **Priming turn** | Turno user+assistant fabricado ANTES del mensaje real, ancla la persona. |
| **Turn-leakage** | Cuando el modelo emite tokens del template (`<|assistant|>`) en plano y fabrica turnos falsos. |
| **Response sanitizer** | Filtro post-inferencia que purga residuo del fabricante. |
| **Observe→Think→Act→Learn** | Loop canónico del agente (Fase 2 reactivo, Fase 3 autónomo). |
| **Drive** | Motivación interna del daemon (survival/self_improvement/curiosity/utility/social). |
| **Goal** | Objetivo concreto en el `GoalTree`, con `status: pending|completed|failed`. |
| **Cooldown self-mod** | Tiempo mínimo entre dos automodificaciones (default 300s). |
| **Indent-match** | Defensa del rewriter: si `old_indent != new_indent`, abort. |
| **Hint extractor** | Pre-procesador del introspector que detecta `path:func` o `_funcname` en el query. |
| **Cycle interval** | Intervalo entre ciclos cognitivos del daemon (default 30s). |
| **Backup .bak** | Snapshot completo del fichero ANTES de un `apply_change`. Persistente en `.mitos_backups/`. |

---

## 21. Contactos y mantenimiento del informe

- **Archivo**: `INFORME_FORENSE.md` (raíz del repositorio MITOS).
- **Versión actual**: 2.0.
- **Mantenedor**: equipo de arquitectura.
- **Política de actualización**: para cada nueva fase o refactor relevante, incrementar versión menor (2.0 → 2.1). Para cambios de arquitectura, versión mayor (2.0 → 3.0).
- **Discrepancia código vs informe**: el **código es la verdad de campo**. Reportar la divergencia al mantenedor para que actualice el informe, no al revés.

---

## 22. Changelog del informe

### v2.0 (Fase 3 incorporada)

- Añadido módulo Core (`drives`, `goal_tree`, `world_model`, `daemon`).
- Añadido módulo Self-Modification (`introspector`, `rewriter`, `validator`).
- Añadido módulo Net (`github`).
- Documentado `ResponseSanitizer` y triple defensa de identidad MITOS.
- Documentado pipeline `_do_self_modify` con dedent/re-indent.
- Añadido §17.4 (daemon runtime + consola del operador).
- Añadido §18 (auditoría de bugs corregidos en v2.0).
- Actualizado glosario (~10 términos nuevos).
- Actualizada deuda técnica (Fase 4 candidate).

### v1.0 (línea base)

- Cubría Fase 1 (4 módulos núcleo) + Fase 2 (cerebro reactivo).
- 17 secciones.

---

## 23. AUDITORÍA EMPÍRICA — ¿está MITOS mutando de verdad? (v3.0)

Sección añadida tras una sesión de ~70 minutos del daemon corriendo en producción. El objetivo: verificar de forma forense que las modificaciones que el daemon reporta no son ficticias y evaluar la **calidad semántica** de cada cambio.

### 23.1 Inventario físico de mutaciones

`ls .mitos_backups/` produjo **12 archivos `.bak`**, todos con `mtime` coherente con los logs del daemon. Los nombres siguen el patrón `<path_flat>_<timestamp_ms>.bak`:

| # | Fichero objetivo | Función modificada | Timestamp | Tamaño .bak |
|---|---|---|---|---|
| 1 | `src/brain/agent.py` | (primera mutación, regresión luego) | 1781392137 | 24470 |
| 2 | `src/brain/agent.py` | de-indentación catastrófica | 1781392486 | 24605 |
| 3 | `src/brain/interactive.py` | `_welcome` | 1781392871 | 10314 |
| 4 | `src/evolution/safe_mutator.py` | `visit_Module` | 1781405386629 | 28802 |
| 5 | `src/brain/agent.py` | `__init__` | 1781405848076 | 24470 |
| 6 | `src/brain/agent.py` | `__init__` (segundo intento) | 1781406408888 | 24137 |
| 7 | `src/evolution/safe_mutator.py` | `visit_AsyncFunctionDef` | 1781406905854 | 29035 |
| 8 | `src/brain/agent.py` | `__init__` (tercer intento) | 1781407405354 | 24137 |
| 9 | `src/brain/agent.py` | `__init__` (cuarto intento) | 1781407921195 | 24884 |
| 10 | `src/core/decision_engine.py` | `history_length` | 1781405977217 | 18137 |
| 11 | `src/core/polymorphic.py` | `mutation_count` | 1781408539192 | 15014 |
| 12 | `src/core/code_synthesizer.py` | `synthesize` | 1781409137033 | 20199 |

**Veredicto técnico**: el sistema **SÍ está mutando archivos físicos**. Cada `.bak` corresponde a un evento de `CodeRewriter.apply_change` exitoso con su `hot_reload_module` cuando aplica. Los timestamps coinciden con las líneas del log del daemon. La maquinaria infraestructural funciona.

### 23.2 Análisis cualitativo por mutación

Usando `git diff --no-index <bak> <archivo_actual>` para inspeccionar cada cambio:

#### Mutación #7 — `src/evolution/safe_mutator.py:visit_AsyncFunctionDef`

**Diff observado**: +12 líneas (docstring estructurada con `Parameters:` y `Returns:` + un `raise ValueError` defensivo).

```python
+        """
+        Rewrites the body of an async function definition.
+        Parameters: node (ast.AsyncFunctionDef): The async function definition node...
+        Returns: ast.AST: The rewritten async function definition node.
+        """
+        if not hasattr(node, 'body'):
+            raise ValueError("The node does not have a body attribute.")
```

**Veredicto**: ✅ **Mejora real**. Docstring informativa con tipos. El `raise ValueError` es defensivo pero algo redundante (`AsyncFunctionDef` siempre tiene `.body` por la gramática del AST).

#### Mutación #4 — `src/evolution/safe_mutator.py:visit_Module`

**Diff observado**: +9 líneas (docstring estructurada similar).

**Veredicto**: ✅ **Mejora real**. Docstring profesional añadida.

#### Mutación #11 — `src/core/polymorphic.py:mutation_count`

**Diff observado**: +6 líneas (docstring multi-línea explicativa).

```python
+        """Returns the count of mutations in the object.
+        This function retrieves the internal count of mutations that have occurred
+        within the object. It is assumed that the object maintains a private attribute
+        `_mutation_count` that tracks the number of mutations.
+        """
```

**Veredicto**: ✅ **Mejora real, levemente verbosa**. Una propiedad trivial obtiene tres frases de docstring; algo redundante pero técnicamente correcto.

#### Mutación #3 — `src/brain/interactive.py:_welcome`

**Diff observado**: +3 líneas (docstring concisa de una línea).

```python
+    """
+    Displays a welcome message with available commands for the MITOS Brain CLI.
+    """
```

**Veredicto**: ✅ **Mejora real, mínima pero correcta**.

#### Mutación #10 — `src/core/decision_engine.py:history_length`

**Diff observado**: +1 línea (docstring de una sola frase).

```python
+        """Returns the length of the action history."""
```

**Veredicto**: ✅ **Mejora mínima**. El resto del diff del archivo pertenece a ediciones humanas posteriores del operador (no de Phi-3); el cambio del daemon se limita a esa línea.

#### Mutación #12 — `src/core/code_synthesizer.py:synthesize` — ❌ REGRESIÓN GRAVE

**Diff observado**: el método público `synthesize`:
- ❌ Pierde la docstring `Args:`/`Returns:` (sustituida por `...`).
- ❌ Pierde dos `log.warning(...)` que reportaban fallos por spec vacía y por ausencia de tests. **Ahora los fallos son silenciosos**.
- ❌ Pierde todos los comentarios del pipeline (`# (1) Asegurar batería de tests`, etc.).
- ❌ **CRÍTICO**: pierde la actualización de `error_context` en la rama de fallo parcial:

```diff
-            error_context = (
-                f"Tests pasados: {passed}/{total}. "
-                f"Output:\n{output[:400]}\n"
-                f"Corrige la lógica para que TODOS pasen."
-            )
```

Esto era el **mecanismo de retry-con-feedback** que justificaba el `CodeSynthesizer`. Sin él, Phi-3 no recibe el output del fallo anterior y todos los intentos posteriores son ciegos.

**Veredicto**: ❌ **Regresión arquitectónica**. El archivo parsea y `tests_passed/total` siguen contando, pero **el sintetizador ya no es iterativo**. Reduce la efectividad de la Fase 4 problema 3 (programar de cero con tests). **Rollback recomendado**.

#### Mutaciones #1, #2 y #5–9 — `src/brain/agent.py` — ❌ REGRESIÓN + BUG

`agent.py` recibió 6 mutaciones consecutivas. La auditoría revela dos problemas:

**Mutación #2 (timestamp `1781392486`)**: Phi-3 **de-indentó** una función moviéndola fuera de su clase. Esta mutación motivó las defensas `_leading_indent` y `indent_match` añadidas en Fase 3.5 (ver §18, bug #7). Estado actual: rolled back por el operador.

**Mutación más reciente (#9, timestamp `1781407921195`)**: Phi-3 introdujo una **typo crítica**:

```diff
-        self.filter = code_filter if code_filter is not None else AutonomousFilter(
+        selfe.filter = code_filter if code_filter is not None else AutonomousFilter(
```

`selfe.filter` en vez de `self.filter`. Si el daemon (o cualquier código) intenta `AutonomousAgent(...)` directamente, lanzará:

```
NameError: name 'selfe' is not defined
```

Esto **pasó los cuatro safeguards** del pipeline:
- `AutonomousFilter.evaluate` → compilable=1.0 (el código parsea: `selfe` solo es un name no resuelto, sin error de sintaxis)
- `ChangeValidator._check_public_interface` → la firma de `__init__` no cambia
- `ChangeValidator._find_dangerous_call` → no hay calls peligrosas
- `CodeRewriter` parse-check del archivo completo → parsea perfectamente

El problema es **semántico**, no sintáctico. Ningún safeguard estático puede cazarlo sin ejecutar el código.

**Verificación in vivo**:

```
$ python -c "from src.brain.agent import AutonomousAgent; print('import OK')"
import OK     # ← import funciona porque selfe se evalúa en runtime
```

El daemon principal **no se rompe** porque su loop cognitivo no instancia `AutonomousAgent` directamente. Pero:
- `python -m src.brain.interactive` (CLI de Fase 2) rompería al arrancar.
- Cualquier test de `tests/test_agent.py` rompería.
- `_tool_execute_code` y otras herramientas del agente quedan inalcanzables.

**Veredicto**: ❌ **Bug introducido**. La función afectada es `AutonomousAgent.__init__`. **Rollback obligatorio**.

### 23.3 Resumen ejecutivo de calidad

| Mutación | Veredicto | Acción recomendada |
|---|---|---|
| #4 `safe_mutator:visit_Module` | ✅ Mejora real | Conservar |
| #7 `safe_mutator:visit_AsyncFunctionDef` | ✅ Mejora real | Conservar |
| #3 `interactive:_welcome` | ✅ Mejora menor | Conservar |
| #10 `decision_engine:history_length` | ✅ Mejora mínima | Conservar |
| #11 `polymorphic:mutation_count` | ✅ Mejora verbosa pero válida | Conservar |
| #12 `code_synthesizer:synthesize` | ❌ Regresión arquitectónica | **Rollback** |
| #9 `agent:__init__` | ❌ NameError (`selfe.filter`) | **Rollback obligatorio** |
| #1, #2, #5-8 `agent.py` | Mixto (algunas regresiones) | Inspeccionar caso por caso |

**Conclusión empírica**: de las 12 mutaciones aplicadas a disco, **5 son mejoras reales (docstrings)**, **1 es regresión arquitectónica** (`code_synthesizer`) y **1 es un bug ejecutivo** (`agent.py`). Tasa de éxito semántico: aproximadamente **42-58%** dependiendo de la severidad con que penalicemos los `NameError` introducidos.

### 23.4 Comandos de rollback

Para restaurar los dos archivos problemáticos:

```powershell
# Rollback de agent.py al último estado bueno conocido (penúltimo backup)
$bak_agent = Get-ChildItem .mitos_backups\src_brain_agent.py_1781407405354.bak
Copy-Item $bak_agent.FullName src\brain\agent.py -Force

# Rollback de code_synthesizer.py
$bak_synth = Get-ChildItem .mitos_backups\src_core_code_synthesizer.py_1781409137033.bak
Copy-Item $bak_synth.FullName src\core\code_synthesizer.py -Force

# Verificar que ambos compilan
python -c "import ast; ast.parse(open(r'src\brain\agent.py', encoding='utf-8').read()); print('agent.py parses')"
python -c "import ast; ast.parse(open(r'src\core\code_synthesizer.py', encoding='utf-8').read()); print('code_synthesizer.py parses')"

# Verificar que agent.py ya no contiene el typo
Select-String -Path src\brain\agent.py -Pattern "selfe\.filter" -SimpleMatch
# Si la salida está vacía, el rollback fue exitoso.
```

Alternativa interactiva desde la consola del operador:

```
operator> /pause
operator> /undo
operator> /undo
operator> /resume
```

`/undo` deshace la última modificación física registrada en `CodeRewriter.change_log`. Llamarlo dos veces revierte las dos últimas mutaciones.

### 23.5 Implicaciones para el diseño

Esta auditoría revela tres limitaciones reales de la pipeline de safeguards de Fase 4:

1. **El filtro autónomo no detecta typos sutiles**. `selfe.filter` parsea como Python válido porque `selfe` es un identificador legítimo aunque indefinido. Para cazarlo habría que ejecutar `pyflakes` o `ruff` antes de aceptar el cambio.

2. **El validator no detecta pérdida de comportamiento dentro de funciones**. Conserva firmas, no implementaciones. La regresión en `synthesize` (pérdida del retry-con-feedback) pasó porque la firma `def synthesize(self, specification, examples=None, constraints=None) -> SynthesizedProgram | None:` sigue siendo idéntica.

3. **No hay tests de regresión** que se ejecuten tras un `apply_change`. Si existieran `tests/test_synthesizer.py` con un caso que verificara "tras fallo parcial, el siguiente intento recibe el error como contexto", la mutación habría sido rechazada.

### 23.6 Recomendaciones para Fase 5

Fundamentadas en los hallazgos empíricos de §23.2:

| Capa | Mejora propuesta | Problema que resuelve |
|---|---|---|
| Pre-validator | Añadir `pyflakes` / `ruff` check antes del rewriter | Cazar typos como `selfe.filter` |
| Pre-validator | Diff semántico vs versión anterior: el LLM debe declarar QUÉ cambia (docstring, optimización, refactor) | Bloquear regresiones de comportamiento |
| Post-rewriter | Si existen tests para el módulo modificado, ejecutarlos antes de hot-reload | Cazar regresiones funcionales como la de `synthesize` |
| Cooldown | Asociar cooldown por archivo (no global): si el daemon ya tocó `agent.py` 4 veces, bloquear más mutaciones hasta que sean revisadas | Evitar el bucle visto en `__init__` |
| Diversificación | Penalizar en el target del heurístico los archivos que han recibido N modificaciones recientes sin nuevas debilidades detectadas | Forzar al daemon a tocar archivos no auditados |

---

## 24. Changelog del informe (continuación)

### v3.0 (Fase 4 + auditoría empírica)

- Añadido motor `DecisionEngine` (`src/core/decision_engine.py`) con tres capas: reglas, heurísticas y fallback al LLM.
- Añadido `CodeSynthesizer` (`src/core/code_synthesizer.py`) con pipeline iterativo de tests automáticos en subproceso.
- Añadido `PolymorphicEngine` (`src/core/polymorphic.py`) con `hot_replace_function`, `hot_reload_module`, `rollback` y `_PROTECTED_MODULES`.
- Añadido `DeepRepoLearner` (`src/net/repo_learner.py`) con extracción AST y persistencia tripartita (knowledge / code / reflections).
- Cableado de cooldown awareness end-to-end: `world_state` ahora incluye `self_mod_cooldown_active`; reglas y heurísticas lo respetan; prompt del LLM lo menciona.
- Cableado de prioridad del operador: goals con `source.startswith("operator_")` bypassean al engine.
- Rotación de `_REFLECT_TOPICS` y de `_vary_target` para evitar bucles repetitivos en target de `self_modify` y en topic de `reflect`.
- Cambio del modelo base por defecto a `failspy/Phi-3-mini-4k-instruct-abliterated-v3-GGUF` (sin RLHF de Microsoft baked en pesos).
- §17.4 ampliada: consola del operador con `/say`, `/ask`, `/pause`, `/resume`, `/undo`, `/status`, `/goals`, `/memory`.
- `status_snapshot` ahora expone `decision_engine.history_length`, `decision_engine.success_rates_pct`, `polymorphism.mutation_count` y `polymorphism.backups_available`.
- **NUEVO**: §23 — auditoría empírica forense con 12 mutaciones reales analizadas y veredicto cualitativo por mutación. Demuestra que MITOS muta de verdad, identifica una regresión (`code_synthesizer.synthesize`) y un bug ejecutivo (`agent.__init__:selfe.filter`), y recomienda safeguards adicionales para Fase 5.

---

## 25. FASE 5 — Detección autónoma de regresiones y blacklist de targets (v3.1)

Esta sección documenta los cambios cableados después de la auditoría §23 con el objetivo declarado por el operador:

> *"Has que mitos se de cuenta de las regresiones, la idea es poderlo dejar operar solo toda la noche y debe ser capaz de darse cuenta de las regresiones y bugs."*

### 25.1 Nuevo módulo `RegressionDetector` (`src/self_mod/regression_detector.py`)

Verificador post-mutación stateless. Tras cada `apply_change` exitoso, el daemon llama a `RegressionDetector.verify(file_path, backup_content)`. Tres comprobaciones independientes:

1. **Pyflakes diff** — `pyflakes` sobre la versión antes (desde el `.bak`) y después. Solo se cuentan warnings NUEVOS. Las preexistentes del codebase se toleran para no rechazar mutaciones legítimas. Esta capa es la que caza `selfe.filter` (undefined name).

2. **Fresh import test** — subproceso Python aislado que importa el módulo desde cero. Caza imports rotos, dependencias eliminadas, errores a nivel de módulo. Timeout 10 s — nunca bloquea al daemon.

3. **Pytest opcional** — si existe `tests/test_<basename>.py` para el archivo modificado, lo corre con `-x -q`. Si los tests existen y fallan, regresión declarada. Timeout 60 s.

Veredicto:

```python
@dataclass(frozen=True)
class VerificationResult:
    safe: bool
    issues: list[str]
    category: str  # "ok"|"lint"|"import"|"tests"|"internal_error"
    new_warnings: list[str]
```

Cuando `safe=False`, el daemon ejecuta `rewriter.rollback_last()`, persiste la experiencia como anti-pattern con `importance=0.95` y activa el cooldown (`_last_self_mod_time = time.time()`).

Dependencia añadida en `requirements.txt`:

```
pyflakes>=3.2.0   # Fase 5 — caza undefined names que parsean
```

### 25.2 Cableado en `daemon._do_self_modify`

Pipeline después del `apply_change`:

```
1. rewriter.apply_change(file_path, new_code)   → escribe a disco + crea .bak
2. introspector.scan()                          → refresca grafo
3. polymorph.hot_reload_module(module_name)     → recarga en memoria
4. regression_detector.verify(file_path, backup_content)
   ├─ safe=True  → memory.store_experience(success, importance=0.7)
   │              _record_target_success(target_id)
   └─ safe=False → rewriter.rollback_last()
                   memory.store_experience(regression, importance=0.95)
                   _last_self_mod_time = time.time()  # engaged cooldown
                   _record_target_failure(target_id, reason=category)
                   return failure
```

### 25.3 Mecanismo de blacklist por target

Estado nuevo en `MitosDaemon.__init__`:

```python
self._target_failures: dict[str, int] = {}
self._target_blacklist: dict[str, float] = {}  # target_id -> unix expiry
_FAILED_TARGET_LIMIT = 3
_BLACKLIST_TTL_S = 3600.0  # 1 hora
```

`target_id` es `"<file>::<function>"`. Después de 3 fallos consecutivos sobre el mismo target (sea por filtro, validator, o regresión), entra en blacklist por 1 hora. Cualquier intento futuro durante ese período es rechazado al inicio de `_do_self_modify` con:

```
ERROR: target {target_id!r} en blacklist por {remaining_min:.1f} min
       más (3+ fallos consecutivos)
```

Un éxito sobre el target resetea el contador. Expuesto en `status_snapshot`:

```python
"target_blacklist": {
    "active_count": len([t for t,e in self._target_blacklist.items() if e > time.time()]),
    "current_failures": dict(self._target_failures),
}
```

### 25.4 Datos empíricos de runtime (forense post-Fase 5)

Inventario físico al cierre de la noche (`.mitos_backups/`):

| Archivo objetivo | Mutaciones intentadas | Interpretación |
| --- | ---: | --- |
| `src/brain/agent.py` | **13** | Phi-3 quedó pegado a este target. Todas las mutaciones fallaron por `selfe.filter` (typo). Sin blacklist habría seguido eternamente. |
| `src/evolution/safe_mutator.py` | 5 | Target legítimo. Mezcla de éxitos y rollbacks. |
| `src/brain/interactive.py` | 3 | Diversificación funcionando. |
| `src/distributed/node.py` | 1 | Tocado por primera vez. |
| `src/distributed/demo.py` | 1 | Tocado por primera vez. |
| `src/core/polymorphic.py` | 1 | Tocado por primera vez. |
| `src/core/decision_engine.py` | 1 | Tocado por primera vez. |
| `src/core/daemon.py` | 1 | Tocado por primera vez (auto-modificación reflexiva). |
| `src/core/code_synthesizer.py` | 1 | Mutación documentada en §23 como regresión grave. |
| **TOTAL** | **27** | 15 nuevas respecto al snapshot de §23. |

**Lo que MITOS hizo de verdad esta noche**:

1. ✅ **Mutó 27 veces** archivos reales del proyecto. La maquinaria de auto-modificación NO es teatro.
2. ✅ **Diversificó** a 9 archivos distintos cuando antes (v3.0) había tocado 6. La rotación de targets funcionó parcialmente.
3. ✅ **Detectó regresiones automáticamente** por primera vez: las mutaciones #6 a #13 sobre `agent.py` fueron todas cazadas por el `RegressionDetector` (categoría `lint` por `undefined name 'selfe'`) y revertidas SIN intervención del operador.
4. ✅ **Activó cooldown post-regresión** correctamente: tras cada rollback, el campo `world_state.self_mod_cooldown_active` quedaba `True` durante los siguientes ciclos.
5. ⚠️ **Se quedó pegado a `agent.py`** durante 13 intentos antes de tener blacklist. Con el cambio de Fase 5 esto se corta a 3.
6. ⚠️ **No usó las anti-patterns persistidas** para evitar repetir `selfe.filter`. El recall las traía como contexto suelto, no como instrucción negativa explícita.

### 25.5 Lo que MITOS NO hizo (brecha entre promesa e implementación)

Lectura honesta para el operador:

| Capacidad declarada en chats anteriores | Estado real al cierre de Fase 5 |
|---|---|
| "Audita su propio código" | Parcial. `Introspector.find_weaknesses` detecta señales estructurales (docstrings, longitud, type hints, bare except). No hay análisis semántico ni grafo de llamadas. |
| "Entiende qué es" | No implementado. Phi-3 trata cada función como un blob de texto a transformar. No hay representación del rol que juega cada función en el sistema. |
| "Prueba el código ANTES de escribir" | Parcial. `CodeSynthesizer` sí lo hace (genera tests → ejecuta candidate en subproceso → solo acepta si pasa) PERO solo se usa para `_do_generate_capability` (nueva tool). `_do_self_modify` sigue el patrón write-first-then-test con rollback. |
| "Pruebas de regresión post-mutación" | ✅ Implementado en Fase 5 (`RegressionDetector`). Cubre lint + import. Pytest solo si existen tests del módulo (hoy no existen para `agent.py`, `daemon.py`, `code_synthesizer.py`). |
| "Aprende de sus errores" | Parcial. Anti-patterns persistidos con `importance=0.95` pero el LLM las recibe como contexto semántico general, no como bloque negativo explícito ("DO NOT REPEAT"). Por eso `selfe.filter` reapareció 13 veces consecutivas. |
| "Operar sola toda la noche" | ✅ Soportado. Las 8+ regresiones de la sesión fueron cazadas y revertidas sin intervención. El operador llegó por la mañana al sistema funcional, no roto. |

### 25.6 Pendientes declarados para Fase 6

Tres componentes propuestos para cerrar las brechas restantes:

#### 25.6.1 `AntiPatternInjector` (prioritario, bajo coste)

Cambio quirúrgico en `_plan_action` y en el prompt de `generate_code` cuando la acción es `self_modify`. En vez de recall semántico genérico:

```python
# ANTES
recall = self.memory.search(plan_context, n=5)
prompt = f"Context:\n{recall}\n\nGenerate improved code..."

# DESPUÉS
anti_patterns = self.memory.search(
    "regression failure code mutation",
    filters={"regression": True, "category": ["lint","import","tests"]},
    n=10,
)
ap_block = "\n".join(f"- {a.content}" for a in anti_patterns)
prompt = (
    "ANTI-PATTERNS DETECTED IN YOUR PAST OUTPUTS — DO NOT REPEAT:\n"
    f"{ap_block}\n\n"
    "Now generate the improved code, AVOIDING those exact mistakes.\n"
)
```

Expectativa: el `selfe.filter` debería dejar de aparecer al primer ciclo siguiente.

#### 25.6.2 `BehaviorTester` (alto valor, complejidad media)

Reemplaza el patrón write-first-then-test de `_do_self_modify` por test-first-then-write:

1. Captura golden samples ejecutando la versión actual de la función con inputs sintéticos (derivados de type hints o de cómo se llama en otras partes del código).
2. Aplica la mutación a un archivo TEMPORAL (no a disco).
3. Importa la versión candidata en subproceso aislado.
4. Ejecuta los golden samples contra la candidata.
5. Si outputs coinciden → escribe a disco con `apply_change`.
6. Si difieren → rechaza con razón concreta (ej. `"_emit_text('hola') antes devolvía None, ahora devuelve 'h'"`).

Esto cazaría regresiones semánticas como la de `code_synthesizer.synthesize` documentada en §23.

#### 25.6.3 `CallGraphAuditor` (estructural, alto coste)

Construir grafo de llamadas del proyecto en cada `scan()`:

- Por cada `FunctionDef`, recolectar `Call` que contiene.
- Resolver targets via heurística AST (nombre exacto en módulo + imports + `self.X`).
- Exponer `world_model.callers_of(file::func) -> list[file::func]`.

Uso por el daemon:

- Priorizar targets sin callers (funciones huérfanas son seguras).
- Bloquear targets críticos con muchos callers (`__init__` → bloquear o requerir confianza altísima).
- Generar inputs realistas para `BehaviorTester` usando los callers reales.

### 25.7 Comandos del operador relevantes a Fase 5

```powershell
# Ver blacklist actual
> /status
target_blacklist:
  active_count: 1
  current_failures:
    src/brain/agent.py::__init__: 3

# Rollback manual si una mutación pasa los safeguards pero se descubre mala
> /undo
revertido: src/brain/agent.py (backup .mitos_backups/src_brain_agent.py_1781418642599.bak)

# Verificar manualmente cualquier archivo
> python -m pyflakes src/brain/agent.py
> python -c "import src.brain.agent"
> pytest tests/test_<modulo>.py -x -q
```

---

## 26. Changelog del informe (Fase 5)

### v3.1 (Fase 5 — detección autónoma de regresiones)

- **NUEVO módulo**: `src/self_mod/regression_detector.py` con `RegressionDetector` + `VerificationResult`. Tres capas: pyflakes diff, fresh import, pytest autodescubrimiento. Todos los subprocesos con timeout — nunca bloquean al daemon.
- **Dependencia**: `pyflakes>=3.2.0` añadida a `requirements.txt` (sección Fase 5).
- **Cableado en `daemon._do_self_modify`**: verificación post-`apply_change`. Si `safe=False`: rollback automático + experiencia persistida como anti-pattern (`importance=0.95`) + cooldown activado.
- **Blacklist de targets**: nuevo estado `_target_failures` + `_target_blacklist` en `MitosDaemon`. Tras 3 fallos consecutivos sobre `<file>::<func>`, target bloqueado durante 1 hora (`_BLACKLIST_TTL_S=3600.0`). Resetea en éxito.
- **`status_snapshot`** expone ahora la sub-clave `target_blacklist` con `active_count` y `current_failures`.
- **§25** añadida: documentación completa de Fase 5, datos empíricos de runtime (27 mutaciones físicas, 13 sobre `agent.py`, 8+ regresiones cazadas y revertidas sin intervención), brecha honesta promesa-vs-implementación, y plan de Fase 6 (`AntiPatternInjector`, `BehaviorTester`, `CallGraphAuditor`).
- **§26** (esta sección) añadida.

---

**FIN DEL INFORME v3.1**
