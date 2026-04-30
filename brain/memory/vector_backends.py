"""
Vector Backends
---------------
Drop-in replacements for the SQLite embedding storage in Store.

When your corpus grows beyond ~50k notes, cosine search in pure Python
becomes slow. These backends offload storage and approximate-nearest-neighbour
search to dedicated vector databases.

Backends:
  ChromaBackend  — chromadb (local, no server needed)
  QdrantBackend  — qdrant-client (local file or Qdrant server)

Both implement the same four methods used by Store:
  save_embedding(note_id, vector, model)
  get_embedding(note_id) → list | None
  get_all_embeddings()   → {note_id: vector}
  search_by_embedding(query_vec, top_k) → [(note_id, score)]

Usage — patch Store at startup (main.py):
  from brain.memory.vector_backends import ChromaBackend
  store = Store("data/brain.db")
  store._vector = ChromaBackend("data/chroma")

  # All embedding calls now go through Chroma:
  store.save_embedding(note_id, vec)
  results = store._vector.search_by_embedding(q_vec, top_k=10)

Or use the VectorStore factory:
  backend = VectorBackend.from_config(cfg)
  # cfg["vector_backend"] = "chroma" | "qdrant" | "sqlite" (default)

Requirements:
  pip install chromadb          # for ChromaBackend
  pip install qdrant-client     # for QdrantBackend (local file mode)
"""

import json
import logging
import math
import os
from typing import Optional

log = logging.getLogger(__name__)


# ─── Base interface ───────────────────────────────────────────────────────────

class VectorBackend:
    """Abstract interface. Subclass and implement these four methods."""

    def save_embedding(self, note_id: str, vector: list, model: str = "unknown"):
        raise NotImplementedError

    def get_embedding(self, note_id: str) -> Optional[list]:
        raise NotImplementedError

    def get_all_embeddings(self) -> dict:
        raise NotImplementedError

    def search_by_embedding(self, query_vec: list, top_k: int = 10) -> list:
        """Returns [(note_id, score)] sorted by descending cosine similarity."""
        raise NotImplementedError

    @staticmethod
    def from_config(cfg: dict) -> "VectorBackend":
        backend = cfg.get("vector_backend", "sqlite").lower()
        if backend == "chroma":
            return ChromaBackend(
                persist_dir=cfg.get("chroma_path", "data/chroma"),
                collection=cfg.get("chroma_collection", "brain"),
            )
        elif backend == "qdrant":
            return QdrantBackend(
                path=cfg.get("qdrant_path", "data/qdrant"),
                url=cfg.get("qdrant_url"),
                api_key=cfg.get("qdrant_api_key"),
                collection=cfg.get("qdrant_collection", "brain"),
            )
        else:
            # Return None — caller falls back to SQLite Store methods
            return None


# ─── Chroma backend ───────────────────────────────────────────────────────────

class ChromaBackend(VectorBackend):
    """
    Local Chroma vector store.
    No server needed — data persists in a directory.

    pip install chromadb
    """

    def __init__(self, persist_dir: str = "data/chroma",
                 collection: str = "brain"):
        try:
            import chromadb
        except ImportError:
            raise ImportError("pip install chromadb")

        import chromadb
        self._client     = chromadb.PersistentClient(path=persist_dir)
        self._collection = self._client.get_or_create_collection(
            name=collection,
            metadata={"hnsw:space": "cosine"},
        )
        log.info(f"[chroma] Opened collection '{collection}' at {persist_dir} "
                 f"({self._collection.count()} vectors)")

    def save_embedding(self, note_id: str, vector: list, model: str = "unknown"):
        self._collection.upsert(
            ids=[note_id],
            embeddings=[vector],
            metadatas=[{"model": model}],
        )

    def get_embedding(self, note_id: str) -> Optional[list]:
        try:
            result = self._collection.get(ids=[note_id], include=["embeddings"])
            embs   = result.get("embeddings", [])
            return list(embs[0]) if embs else None
        except Exception:
            return None

    def get_all_embeddings(self) -> dict:
        result = self._collection.get(include=["embeddings"])
        ids    = result.get("ids", [])
        embs   = result.get("embeddings", [])
        return {nid: list(vec) for nid, vec in zip(ids, embs)}

    def search_by_embedding(self, query_vec: list, top_k: int = 10) -> list:
        result = self._collection.query(
            query_embeddings=[query_vec],
            n_results=min(top_k, max(self._collection.count(), 1)),
            include=["distances"],
        )
        ids       = result.get("ids", [[]])[0]
        distances = result.get("distances", [[]])[0]
        # Chroma cosine distance = 1 - cosine_similarity
        return [(nid, round(1.0 - dist, 4)) for nid, dist in zip(ids, distances)]

    def count(self) -> int:
        return self._collection.count()


# ─── Qdrant backend ───────────────────────────────────────────────────────────

class QdrantBackend(VectorBackend):
    """
    Qdrant vector store — local file or hosted server.

    Local (no server):
      QdrantBackend(path="data/qdrant")

    Remote:
      QdrantBackend(url="https://your-cluster.qdrant.io", api_key="...")

    pip install qdrant-client
    """

    def __init__(self,
                 path:       str = "data/qdrant",
                 url:        Optional[str] = None,
                 api_key:    Optional[str] = None,
                 collection: str = "brain",
                 vector_dim: int = 384):
        try:
            from qdrant_client import QdrantClient
            from qdrant_client.models import Distance, VectorParams
        except ImportError:
            raise ImportError("pip install qdrant-client")

        from qdrant_client import QdrantClient
        from qdrant_client.models import Distance, VectorParams

        if url:
            self._client = QdrantClient(url=url, api_key=api_key)
            log.info(f"[qdrant] Connected to {url}")
        else:
            os.makedirs(path, exist_ok=True)
            self._client = QdrantClient(path=path)
            log.info(f"[qdrant] Local store at {path}")

        self._collection = collection
        self._dim        = vector_dim

        # Create collection if it doesn't exist
        existing = [c.name for c in self._client.get_collections().collections]
        if collection not in existing:
            self._client.create_collection(
                collection_name=collection,
                vectors_config=VectorParams(size=vector_dim, distance=Distance.COSINE),
            )
            log.info(f"[qdrant] Created collection '{collection}' (dim={vector_dim})")

    def save_embedding(self, note_id: str, vector: list, model: str = "unknown"):
        from qdrant_client.models import PointStruct
        # Qdrant needs integer IDs — hash the note_id string to an int
        pt_id = _str_to_int_id(note_id)
        self._client.upsert(
            collection_name=self._collection,
            points=[PointStruct(
                id=pt_id,
                vector=vector,
                payload={"note_id": note_id, "model": model},
            )],
        )

    def get_embedding(self, note_id: str) -> Optional[list]:
        pt_id = _str_to_int_id(note_id)
        try:
            results = self._client.retrieve(
                collection_name=self._collection,
                ids=[pt_id],
                with_vectors=True,
            )
            return list(results[0].vector) if results else None
        except Exception:
            return None

    def get_all_embeddings(self) -> dict:
        """
        Scrolls through the entire collection.
        Use with caution on very large collections — returns everything in memory.
        """
        all_embs = {}
        offset   = None
        while True:
            result, next_offset = self._client.scroll(
                collection_name=self._collection,
                with_vectors=True,
                limit=1000,
                offset=offset,
            )
            for pt in result:
                note_id = pt.payload.get("note_id", str(pt.id))
                all_embs[note_id] = list(pt.vector)
            if next_offset is None:
                break
            offset = next_offset
        return all_embs

    def search_by_embedding(self, query_vec: list, top_k: int = 10) -> list:
        results = self._client.search(
            collection_name=self._collection,
            query_vector=query_vec,
            limit=top_k,
            with_payload=True,
        )
        return [
            (r.payload.get("note_id", str(r.id)), round(r.score, 4))
            for r in results
        ]


# ─── Store patch helper ───────────────────────────────────────────────────────

def patch_store(store, backend: VectorBackend):
    """
    Monkey-patch a Store instance to use an external vector backend.
    After calling this, store.save_embedding / get_embedding / get_all_embeddings
    / search_by_embedding all route through the backend.

    Usage:
        store   = Store("data/brain.db")
        backend = ChromaBackend("data/chroma")
        patch_store(store, backend)
    """
    if backend is None:
        return  # SQLite stays default

    import types

    def _save_embedding(self, note_id, vector, model="unknown"):
        backend.save_embedding(note_id, vector, model)

    def _get_embedding(self, note_id):
        return backend.get_embedding(note_id)

    def _get_all_embeddings(self):
        return backend.get_all_embeddings()

    store.save_embedding    = types.MethodType(_save_embedding,    store)
    store.get_embedding     = types.MethodType(_get_embedding,     store)
    store.get_all_embeddings = types.MethodType(_get_all_embeddings, store)

    log.info(f"[vector] Store patched to use {type(backend).__name__}")


# ─── Helpers ──────────────────────────────────────────────────────────────────

def _str_to_int_id(s: str) -> int:
    """Convert a string note ID to a stable positive integer for Qdrant."""
    import hashlib
    return int(hashlib.sha256(s.encode()).hexdigest()[:15], 16)


def _cosine(a: list, b: list) -> float:
    dot    = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a)) or 1.0
    norm_b = math.sqrt(sum(y * y for y in b)) or 1.0
    return dot / (norm_a * norm_b)
