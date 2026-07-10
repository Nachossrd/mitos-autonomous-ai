"""
==============================================================================
 Proyecto MITOS - Memoria a Largo Plazo (vectorial, local, persistente)
==============================================================================

Sistema de memoria que vive 100% en disco local bajo ./data/memory, usando
ChromaDB como motor vectorial y sentence-transformers (vía el embedding
function por defecto de Chroma) para la búsqueda semántica.

Cuatro colecciones especializadas componen la "psique" del agente:

    experiences  -> acciones que ha tomado y sus resultados.
    knowledge    -> hechos del mundo aprendidos (por web search, etc.).
    code         -> snippets de código generados o validados.
    reflections  -> lecciones meta-aprendidas (output de reflect()).

Razones del diseño:
  * Persistencia local: nada va a la nube. Si el sistema se apaga, la
    psique resucita intacta en el siguiente arranque.
  * Búsqueda semántica: el agente no necesita recordar la frase exacta,
    solo el significado. Chroma + sentence-transformers cubren esto.
  * Categorización fuerte por tipo: el `recall` puede filtrar por
    memory_type para no contaminar el contexto del LLM con material
    del tipo equivocado (p.ej. "no me traigas código al razonar sobre
    ética; no me traigas reflexiones cuando pido un snippet").

Loop habilitado: Observe -> Think -> Act -> LEARN.
==============================================================================
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

log = logging.getLogger("mitos.brain.memory")


# Tipos de memoria reconocidos y sus colecciones equivalentes en Chroma.
_VALID_TYPES: tuple[str, ...] = ("experience", "knowledge", "code", "reflection")
_TYPE_TO_COLLECTION: dict[str, str] = {
    "experience": "experiences",
    "knowledge": "knowledge",
    "code": "code",
    "reflection": "reflections",
}


# ============================================================================
# 1. UNIDAD DE MEMORIA
# ============================================================================
@dataclass
class Memory:
    """
    Una unidad atómica de memoria.

    Atributos:
        content:     contenido textual (lo que se indexa y se embebe).
        memory_type: una de _VALID_TYPES (experience/knowledge/code/reflection).
        importance:  score subjetivo [0.0, 1.0]; lo usamos como tie-breaker
                     en recall (a misma distancia, gana la más importante).
        timestamp:   epoch seconds; por defecto, "ahora".
        metadata:    dict libre con info auxiliar (goal, tags, source...).
                     Se serializa a JSON al almacenar (Chroma exige
                     metadata plana con primitivos).
    """

    content: str
    memory_type: str
    importance: float = 0.5
    timestamp: float = field(default_factory=time.time)
    metadata: dict[str, Any] = field(default_factory=dict)


# ============================================================================
# 2. SISTEMA DE MEMORIA
# ============================================================================
class MemorySystem:
    """
    Fachada sobre ChromaDB con las cuatro colecciones del agente.

    Uso típico:
        mem = MemorySystem()                       # arranca en ./data/memory
        mem.store_experience("intenté X; falló")
        mem.store_code("def add(a,b): return a+b", language="python")
        hits = mem.recall("cómo sumar", memory_type="code")
    """

    # ------------------------------------------------------------------
    def __init__(
        self,
        persist_directory: str | Path = "./data/memory",
        embedding_model: str | None = None,
    ) -> None:
        """
        Args:
            persist_directory: carpeta de persistencia (se crea si no existe).
            embedding_model:   si se pasa, se usa explícitamente como
                               SentenceTransformerEmbeddingFunction.
                               Si es None, se deja el default de Chroma
                               (all-MiniLM-L6-v2 vía sentence-transformers).
        """
        self.persist_directory = Path(persist_directory).resolve()
        self.persist_directory.mkdir(parents=True, exist_ok=True)
        log.info(
            "[MemorySystem] persistencia local en %s", self.persist_directory
        )

        # Import perezoso de chromadb para no bloquear tests offline.
        try:
            import chromadb  # type: ignore
            from chromadb.config import Settings  # type: ignore
        except ImportError as e:  # pragma: no cover
            raise RuntimeError(
                "chromadb no está instalado. "
                "Ejecuta: pip install -r requirements.txt"
            ) from e

        self._client = chromadb.PersistentClient(
            path=str(self.persist_directory),
            settings=Settings(anonymized_telemetry=False),
        )

        # Embedding function: por defecto, Chroma usa all-MiniLM-L6-v2.
        # Si el operador quiere otro, se lo damos explícito.
        self._embedding_function = None
        if embedding_model is not None:
            try:
                from chromadb.utils import embedding_functions  # type: ignore

                self._embedding_function = (
                    embedding_functions.SentenceTransformerEmbeddingFunction(
                        model_name=embedding_model
                    )
                )
                log.info(
                    "[MemorySystem] usando embedding model: %s", embedding_model
                )
            except Exception as e:  # noqa: BLE001
                log.warning(
                    "[MemorySystem] no pude inicializar %s (%s); uso default",
                    embedding_model,
                    e,
                )

        # Crear/recuperar las cuatro colecciones.
        self.collections: dict[str, Any] = {}
        for col_name in _TYPE_TO_COLLECTION.values():
            kwargs: dict[str, Any] = {"name": col_name}
            if self._embedding_function is not None:
                kwargs["embedding_function"] = self._embedding_function
            self.collections[col_name] = self._client.get_or_create_collection(
                **kwargs
            )
            log.info(
                "[MemorySystem] colección lista: %s (n=%d)",
                col_name,
                self.collections[col_name].count(),
            )

    # ==================================================================
    #                          ALMACENAMIENTO
    # ==================================================================
    def store(self, memory: Memory) -> str:
        """
        Persiste una `Memory` en su colección correspondiente.

        Args:
            memory: instancia de Memory válida.

        Returns:
            El ID asignado (SHA-256 hex truncado a 32 chars, único por
            contenido + tipo + timestamp + metadata).

        Raises:
            ValueError: si memory_type no es válido.
        """
        if memory.memory_type not in _VALID_TYPES:
            raise ValueError(
                f"memory_type inválido: {memory.memory_type!r}. "
                f"Válidos: {_VALID_TYPES}"
            )
        if not isinstance(memory.content, str) or not memory.content.strip():
            raise ValueError("memory.content debe ser un string no vacío")

        col_name = _TYPE_TO_COLLECTION[memory.memory_type]
        collection = self.collections[col_name]

        # ID determinista por contenido + tipo + ts + metadata. Esto evita
        # duplicados accidentales si el llamador repite el mismo store.
        meta_json = json.dumps(memory.metadata, sort_keys=True, default=str)
        seed = f"{memory.memory_type}|{memory.timestamp}|{memory.content}|{meta_json}"
        mem_id = hashlib.sha256(seed.encode("utf-8")).hexdigest()[:32]

        # Chroma exige metadata plana con primitivos. Aplanamos lo que se
        # pueda y serializamos el resto como JSON bajo "metadata_json".
        flat_meta: dict[str, Any] = {
            "memory_type": memory.memory_type,
            "importance": float(memory.importance),
            "timestamp": float(memory.timestamp),
        }
        for k, v in memory.metadata.items():
            if isinstance(v, (str, int, float, bool)):
                # Prefijamos para no chocar con keys reservadas.
                flat_meta[f"m_{k}"] = v
        flat_meta["metadata_json"] = meta_json

        try:
            collection.add(
                ids=[mem_id],
                documents=[memory.content],
                metadatas=[flat_meta],
            )
        except Exception as e:  # noqa: BLE001
            # Probablemente ya existía (mismo ID). En ese caso, lo
            # actualizamos: la semántica de "store again" es "refrescar".
            log.debug("[MemorySystem] add lanzó %s; intentando upsert", e)
            collection.upsert(
                ids=[mem_id],
                documents=[memory.content],
                metadatas=[flat_meta],
            )

        log.info(
            "[MemorySystem] guardado %s en %s (importance=%.2f)",
            mem_id[:8],
            col_name,
            memory.importance,
        )
        return mem_id

    # ==================================================================
    #                            RECUPERACIÓN
    # ==================================================================
    def recall(
        self,
        query: str,
        memory_type: str | None = None,
        n_results: int = 5,
    ) -> list[dict[str, Any]]:
        """
        Búsqueda semántica.

        Args:
            query:       texto de consulta. Se embebe con el mismo
                         encoder de las colecciones.
            memory_type: si se da, busca sólo en esa colección. Si es None,
                         busca en LAS CUATRO y mezcla los resultados.
            n_results:   máximo de resultados a devolver. Si buscamos en
                         múltiples colecciones, traemos `n_results` de
                         cada una y nos quedamos con los `n_results`
                         globalmente mejores por distancia.

        Returns:
            Lista de dicts:
              {id, content, memory_type, importance, timestamp, distance, metadata}
            ordenada por distancia ascendente (más relevante primero).
            A misma distancia, gana la mayor `importance`.
        """
        if not query or not isinstance(query, str):
            return []
        if n_results < 1:
            return []

        if memory_type is not None:
            if memory_type not in _VALID_TYPES:
                raise ValueError(
                    f"memory_type inválido: {memory_type!r}. "
                    f"Válidos: {_VALID_TYPES}"
                )
            colnames = [_TYPE_TO_COLLECTION[memory_type]]
        else:
            colnames = list(_TYPE_TO_COLLECTION.values())

        merged: list[dict[str, Any]] = []
        for col_name in colnames:
            collection = self.collections[col_name]
            if collection.count() == 0:
                continue
            try:
                res = collection.query(
                    query_texts=[query],
                    n_results=min(n_results, collection.count()),
                )
            except Exception as e:  # noqa: BLE001
                log.warning(
                    "[MemorySystem] query falló en %s: %s", col_name, e
                )
                continue

            # Chroma devuelve listas por query. Sólo enviamos una query,
            # así que tomamos el índice 0 de cada campo.
            ids = (res.get("ids") or [[]])[0]
            docs = (res.get("documents") or [[]])[0]
            metas = (res.get("metadatas") or [[{}]])[0] or []
            dists = (res.get("distances") or [[]])[0] or []

            for i, doc_id in enumerate(ids):
                meta = metas[i] if i < len(metas) else {}
                dist = dists[i] if i < len(dists) else float("inf")
                # Restituimos el metadata original desde el JSON guardado.
                raw_json = meta.get("metadata_json", "{}")
                try:
                    original_meta = json.loads(raw_json)
                except (ValueError, TypeError):
                    original_meta = {}

                merged.append(
                    {
                        "id": doc_id,
                        "content": docs[i] if i < len(docs) else "",
                        "memory_type": meta.get("memory_type", "?"),
                        "importance": float(meta.get("importance", 0.0)),
                        "timestamp": float(meta.get("timestamp", 0.0)),
                        "distance": float(dist),
                        "metadata": original_meta,
                    }
                )

        # Orden: distancia ascendente; a empate, importance descendente.
        merged.sort(key=lambda r: (r["distance"], -r["importance"]))
        return merged[:n_results]

    # ==================================================================
    #                           HELPERS
    # ==================================================================
    def store_experience(
        self,
        content: str,
        importance: float = 0.5,
        **metadata: Any,
    ) -> str:
        """Atajo: guarda una experiencia (acción + resultado observado)."""
        return self.store(
            Memory(
                content=content,
                memory_type="experience",
                importance=importance,
                metadata=dict(metadata),
            )
        )

    def store_code(
        self,
        content: str,
        importance: float = 0.5,
        **metadata: Any,
    ) -> str:
        """Atajo: guarda un snippet de código (probado o sospechoso)."""
        return self.store(
            Memory(
                content=content,
                memory_type="code",
                importance=importance,
                metadata=dict(metadata),
            )
        )

    def store_reflection(
        self,
        content: str,
        importance: float = 0.7,
        **metadata: Any,
    ) -> str:
        """Atajo: guarda una reflexión / lección aprendida.

        Las reflexiones llevan importance por defecto un poco más alta
        que experiencias: representan aprendizaje destilado, no eventos
        crudos.
        """
        return self.store(
            Memory(
                content=content,
                memory_type="reflection",
                importance=importance,
                metadata=dict(metadata),
            )
        )

    def store_knowledge(
        self,
        content: str,
        importance: float = 0.6,
        **metadata: Any,
    ) -> str:
        """Atajo: guarda un hecho aprendido (knowledge base)."""
        return self.store(
            Memory(
                content=content,
                memory_type="knowledge",
                importance=importance,
                metadata=dict(metadata),
            )
        )

    # ==================================================================
    #                           MÉTRICAS
    # ==================================================================
    def get_stats(self) -> dict[str, int]:
        """Conteo de documentos por colección + ``total`` agregado.

        Devuelve un dict con una clave por colección y una clave especial
        ``"total"`` con la suma. La clave ``"total"`` la consume
        ``DriveSystem._apply_context_modifiers`` para decidir si la
        curiosidad necesita ser amplificada (memoria pobre = más curiosidad).
        Antes de añadir ``"total"`` aquí, ese check siempre devolvía 0 y
        el drive estaba crónicamente sobre-boosteado.
        """
        per_collection = {
            col_name: int(col.count())
            for col_name, col in self.collections.items()
        }
        per_collection["total"] = sum(per_collection.values())
        return per_collection
