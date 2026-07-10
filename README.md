# 🧠 MITOS — Autonomous Local AI

> Plataforma **experimental** para explorar hasta dónde puede llegar un agente de IA que corre **en local**: LLM propio, memoria persistente, agente reactivo, un daemon que puede actuar sin ser invocado y experimentos de auto-modificación de código con *safeguards*. Proyecto de investigación, no un producto.

**Por qué este proyecto:** nace de una pregunta — ¿cuánta autonomía y utilidad se puede conseguir sin depender de la nube, corriendo todo en una laptop? El objetivo es *explorar* el problema (privacidad, coste, límites de los modelos locales), no afirmar que está resuelto.

<p align="center">
  <img alt="Python" src="https://img.shields.io/badge/Python-3.11%2B-3776AB?logo=python&logoColor=white">
  <img alt="llama.cpp" src="https://img.shields.io/badge/llama--cpp--python-local-orange">
  <img alt="Offline" src="https://img.shields.io/badge/runs-100%25%20local-success">
  <img alt="License" src="https://img.shields.io/badge/License-MIT-green">
</p>

---

## 🎯 Qué explora

MITOS es un experimento de ingeniería. Cada módulo es una forma de **poner a prueba** una idea sobre sistemas de IA autónomos — no una solución cerrada, sino una hipótesis con código:

| Módulo | Idea que explora |
|--------|------------------|
| **Distribuido** (P2P + DGC) | ¿puede el entrenamiento descentralizado por Internet ser viable comprimiendo gradientes? |
| **Evolución** (AST) | ¿puede un programa modificar su propio código sin romperse, con validación estática? |
| **Seguridad** (Dead-Man Switch + Shamir) | ¿cómo mitigar que un sistema autónomo sea coaccionado o secuestrado? |
| **Filtrado** (estático) | ¿cuánto del *Model Collapse* se puede evitar con filtros automáticos? |
| **Daemon cognitivo** | ¿qué pasa si un agente puede tomar iniciativa en vez de esperar a ser invocado? |

## ✨ Capacidades

- **LLM local** vía `llama-cpp-python` (Phi-3 / Qwen2.5 en GGUF cuantizado) — cero llamadas remotas por defecto.
- **Router de inteligencia** — enruta entre el modelo local y un *pool* opcional de modelos externos (Gemini/Groq/OpenRouter) según capacidad y coste. Las keys viven fuera del repo.
- **Memoria persistente** + perfil de operador que sobrevive a reinicios.
- **Agente reactivo y daemon cognitivo** — bucle de pensamiento interno, *drives* e iniciativa propia.
- **Auto-mejora**: escáner de bugs, *tool builder*, *self-patcher* e introspección AST con *safeguards*.
- **Sensor hub** — visión, voz, emoción y reconocimiento de rostro/voz como entradas opcionales (los datos de registro nunca se versionan).
- **Seguridad**: Dead-Man Switch y reparto de secretos de Shamir.

## 🏗️ Estructura

```
src/
├── brain/          # agente, identidad, motor LLM, memoria, filtro de respuesta
├── core/           # daemon, motor cognitivo, router, auto-mejora, sensores, herramientas
├── distributed/    # entrenamiento P2P + compresión de gradientes
├── evolution/      # auto-modificación de código (AST)
├── filtering/      # filtro estático anti Model-Collapse
├── security/       # dead-man switch + Shamir
├── self_mod/       # introspección
└── orchestrator/   # dashboard cognitivo
config/             # brain_pool.example.json (plantilla sin claves)
tests/              # pruebas
```

## 🚀 Uso

```bash
# 1. Dependencias (Python 3.11+)
pip install -r requirements.txt

# 2. Configurar el pool de modelos externos (OPCIONAL — MITOS corre local sin esto)
cp config/brain_pool.example.json config/brain_pool.json
#   edita config/brain_pool.json y añade tus API keys (nunca se commitea)

# 3. Descargar un modelo GGUF local en ./models/ (no incluido, multi-GB)
#    p.ej. Phi-3-mini-4k-instruct-q4.gguf o qwen2.5-*-q4_k_m.gguf

# 4. Arrancar
python run_daemon.py          # daemon cognitivo autónomo
# o los runners: run_brain.sh / run_brain.ps1 / run_mvp.sh / run_mvp.ps1
```

## 🔒 Seguridad y secretos

- **`config/brain_pool.json`** (API keys) y **`.env`** están en `.gitignore` — **nunca** se versionan. Usa `config/brain_pool.example.json` como plantilla.
- Los **modelos GGUF**, la carpeta **`data/`** (memoria, rostros/voces registrados) y los **backups** están excluidos del repositorio por privacidad y tamaño.

## 📊 Benchmarks

> ⚠️ **Pendiente de medición.** Al correr con un LLM local, el rendimiento depende del modelo y del hardware, así que prefiero no dar cifras hasta medirlas de forma reproducible. Métricas objetivo:

| Métrica | Cómo se mide | Valor |
|---------|--------------|-------|
| Tokens/seg (inferencia local) | `llama-cpp` timings, por modelo (Phi-3 / Qwen2.5) | *por medir* |
| Latencia primer token | tiempo hasta el primer token en una prompt estándar | *por medir* |
| RAM pico | `psutil` durante inferencia + daemon activo | *por medir* |
| Ciclo del daemon | tiempo medio de un ciclo cognitivo completo | *por medir* |

*Entorno de referencia: Windows 11, CPU Intel i5-10300H (AVX2), sin GPU. Cifras concretas por añadir.*

## ⚠️ Limitaciones

Ser honesto sobre lo que **no** es este proyecto es parte del proyecto:

- **No sustituye** un SOC ni ninguna herramienta de seguridad profesional.
- **No garantiza** seguridad ni robustez: los *safeguards* son experimentales, no auditados.
- **Requiere sandbox**: la auto-modificación de código y el daemon pueden ejecutar acciones; córrelo aislado.
- **No usar en producción**: es una plataforma de investigación/portfolio.
- Los **modelos locales** cuantizados quedan por debajo de los modelos *frontier* en capacidad; el router externo es opcional y de mejor esfuerzo.

El detalle técnico completo (contratos, invariantes, flujos) está en [`INFORME_FORENSE3.md`](INFORME_FORENSE3.md).

## 🛠️ Stack

`Python` · `PyTorch (CPU)` · `llama-cpp-python` · `websockets` · `aiohttp` · `msgpack` · `cryptography`

---

## ✍️ Autor

**Nacho** — [@Nachossrd](https://github.com/Nachossrd)

## 📄 Licencia

MIT — ver [`LICENSE`](LICENSE).
