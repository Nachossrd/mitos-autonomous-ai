# 🧠 MITOS — Autonomous Local AI

> Arquitectura de IA autónoma que corre **100% en local**: LLM propio, memoria persistente, agente reactivo, auto-modificación de código con *safeguards* y un daemon cognitivo que actúa sin ser invocado. Sin depender de proveedores remotos.

<p align="center">
  <img alt="Python" src="https://img.shields.io/badge/Python-3.11%2B-3776AB?logo=python&logoColor=white">
  <img alt="llama.cpp" src="https://img.shields.io/badge/llama--cpp--python-local-orange">
  <img alt="Offline" src="https://img.shields.io/badge/runs-100%25%20local-success">
  <img alt="License" src="https://img.shields.io/badge/License-MIT-green">
</p>

---

## 🎯 Tesis

MITOS es un experimento de ingeniería que busca **refutar operativamente** cinco supuestos sobre los sistemas de IA autónomos:

| Módulo | Tesis que refuta |
|--------|------------------|
| **Distribuido** (P2P + DGC) | que el entrenamiento descentralizado por Internet es impracticable por ancho de banda |
| **Evolución** (AST) | que la auto-modificación de código causa "suicidio digital" |
| **Seguridad** (Dead-Man Switch + Shamir) | que un sistema autónomo es vulnerable a extorsión |
| **Filtrado** (estático) | que evitar el *Model Collapse* requiere supervisión humana |
| **Daemon cognitivo** | que un agente solo es útil cuando un humano lo invoca |

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

## 📄 Nota

Proyecto de investigación/portfolio. La auto-modificación y el daemon operan con *safeguards* conocidos; no se recomienda ejecutarlo sin sandbox. El detalle técnico completo (contratos, invariantes, flujos) está en [`INFORME_FORENSE3.md`](INFORME_FORENSE3.md).

## 🛠️ Stack

`Python` · `PyTorch (CPU)` · `llama-cpp-python` · `websockets` · `aiohttp` · `msgpack` · `cryptography`

---

## ✍️ Autor

**Nacho** — [@Nachossrd](https://github.com/Nachossrd)

## 📄 Licencia

MIT — ver [`LICENSE`](LICENSE).
