"""
Embedding Provider
------------------
Priority order (auto-detected at runtime):
  1. Ollama local          — nomic-embed-text, mxbai-embed-large
  2. sentence-transformers — all-MiniLM-L6-v2, all-mpnet-base-v2
  3. OpenAI API            — text-embedding-3-small
  4. TF-IDF fallback       — always works, no dependencies

Large-scale changes vs original:
  - embed_notes() now uses save_embeddings_batch() — single transaction per chunk
    instead of one fsync per note (was the cause of the 30k freeze)
  - embed_notes() never holds all notes in RAM — iterates via iter_notes_chunked()
  - Optional: sync_to_chroma() — after embedding, push vectors to Chroma for
    fast ANN search without re-embedding
"""

from __future__ import annotations

import math
import logging
import re
from collections import Counter
from typing import Optional

log = logging.getLogger("brain.embeddings")

EMBED_BATCH = 64      # notes per embedding batch (sentence-transformers)
TFIDF_DIM   = 512
COMMIT_EVERY = 500    # write to SQLite every N embeddings (not every 1)


class EmbeddingProvider:

    def __init__(self, backend: str, model: str,
                 api_key: str = "",
                 base_url: str = "http://localhost:11434",
                 _st_model=None):
        self.backend   = backend
        self.model     = model
        self.api_key   = api_key
        self.base_url  = base_url
        self._st_model = _st_model

    # ── Factories ─────────────────────────────────────────────────────────────

    @classmethod
    def from_config(cls, cfg: dict) -> "EmbeddingProvider":
        backend = cfg.get("embedding_backend", "auto").lower()
        model   = cfg.get("local_embedding_model", "all-MiniLM-L6-v2")
        base    = cfg.get("ollama_base_url", "http://localhost:11434")
        api_key = cfg.get("openai_api_key", "")

        if backend == "ollama":
            return cls("ollama",
                       cfg.get("local_embedding_model", "nomic-embed-text"),
                       base_url=base)

        if backend == "sentence_transformers":
            st = _load_sentence_transformers(model)
            if st:
                return cls("sentence_transformers", model, _st_model=st)
            log.warning("sentence-transformers failed — falling back")

        if backend == "openai":
            return cls("openai", "text-embedding-3-small", api_key=api_key)

        if backend == "tfidf":
            return cls("tfidf", "tfidf")

        if backend in ("auto", "local"):
            log.info("[embed] Auto-detecting best embedding backend…")

            ollama_model = cfg.get("local_embedding_model", "nomic-embed-text")
            if _ollama_available(base, ollama_model):
                log.info(f"[embed] Using Ollama ({ollama_model})")
                return cls("ollama", ollama_model, base_url=base)

            st_name = cfg.get("local_embedding_model", "all-MiniLM-L6-v2")
            if "nomic" in st_name:
                st_name = "all-MiniLM-L6-v2"
            st = _load_sentence_transformers(st_name)
            if st:
                log.info(f"[embed] Using sentence-transformers ({st_name})")
                return cls("sentence_transformers", st_name, _st_model=st)

            import os
            oai_key = api_key or os.environ.get("OPENAI_API_KEY", "")
            if oai_key:
                log.info("[embed] Using OpenAI text-embedding-3-small")
                return cls("openai", "text-embedding-3-small", api_key=oai_key)

            log.warning(
                "[embed] No embedding backend found. Using TF-IDF fallback.\n"
                "        pip install sentence-transformers  (recommended)\n"
                "        ollama pull nomic-embed-text       (if Ollama running)"
            )
            return cls("tfidf", "tfidf")

        return cls("tfidf", "tfidf")

    @classmethod
    def from_registry(cls, registry) -> "EmbeddingProvider":
        try:
            profile = registry.get_for_role("embed")
            if profile.provider == "ollama":
                return cls("ollama", profile.model,
                           base_url=profile.base_url or "http://localhost:11434")
            elif profile.provider == "openai":
                return cls("openai", "text-embedding-3-small",
                           api_key=profile.api_key)
        except Exception:
            pass
        return cls.from_config({"embedding_backend": "auto"})

    # ── Embed API ─────────────────────────────────────────────────────────────

    def embed(self, text: str) -> list:
        text = text[:8000]
        if self.backend == "ollama":
            return self._ollama_embed(text)
        elif self.backend == "sentence_transformers":
            return self._st_embed(text)
        elif self.backend == "openai":
            return self._openai_embed(text)
        else:
            return self._tfidf_embed(text)

    def embed_one(self, text: str) -> list:
        return self.embed(text)

    def embed_batch(self, texts: list) -> list:
        if self.backend == "sentence_transformers" and self._st_model:
            try:
                vecs = self._st_model.encode(
                    [t[:8000] for t in texts],
                    batch_size=32,
                    show_progress_bar=False,
                    normalize_embeddings=True,
                )
                return [v.tolist() for v in vecs]
            except Exception as e:
                log.warning(f"ST batch embed failed: {e}")
        return [self.embed(t) for t in texts]

    # ── Backends ──────────────────────────────────────────────────────────────

    def _ollama_embed(self, text: str) -> list:
        import urllib.request, json
        payload = json.dumps({"model": self.model, "prompt": text}).encode()
        try:
            req = urllib.request.Request(
                f"{self.base_url}/api/embeddings", data=payload,
                headers={"Content-Type": "application/json"}, method="POST",
            )
            with urllib.request.urlopen(req, timeout=30) as r:
                return json.loads(r.read())["embedding"]
        except Exception as e:
            log.warning(f"Ollama embed failed: {e} — TF-IDF fallback")
            return self._tfidf_embed(text)

    def _st_embed(self, text: str) -> list:
        try:
            vec = self._st_model.encode(
                text, normalize_embeddings=True, show_progress_bar=False
            )
            return vec.tolist()
        except Exception as e:
            log.warning(f"ST embed failed: {e}")
            return self._tfidf_embed(text)

    def _openai_embed(self, text: str) -> list:
        try:
            import openai
            client = openai.OpenAI(api_key=self.api_key)
            resp   = client.embeddings.create(model="text-embedding-3-small", input=text)
            return resp.data[0].embedding
        except Exception as e:
            log.warning(f"OpenAI embed failed: {e} — TF-IDF fallback")
            return self._tfidf_embed(text)

    def _tfidf_embed(self, text: str) -> list:
        tokens = re.findall(r"\b[a-z]{3,}\b", text.lower())
        freq   = Counter(tokens)
        if not freq:
            return [0.0] * TFIDF_DIM
        vec   = [0.0] * TFIDF_DIM
        total = sum(freq.values())
        for token, count in freq.items():
            vec[_hash_token(token) % TFIDF_DIM] += count / total
        return _l2_norm(vec)


# ─── Bulk embedding ───────────────────────────────────────────────────────────

def embed_notes(store, provider: EmbeddingProvider,
                force: bool = False,
                chroma_sync: bool = False):
    """
    Embed all notes missing a vector (or all if force=True).

    Key improvements over original:
      1. Uses iter_notes_chunked() — only CHUNK_SIZE notes in RAM at once.
         The old version called get_all_notes() (or notes_without_embeddings())
         which loaded every full Note (including content) at once.

      2. Uses save_embeddings_batch() — single SQLite transaction per chunk.
         The old version called save_embedding() per note = one fsync per note.
         At 30k notes, that's 30,000 fsyncs. This version does ~60 fsyncs total.

      3. Optional chroma_sync: after embedding, writes vectors to Chroma
         for fast ANN semantic search during graph building.
         Enable with: python main.py build --backend chroma
    """
    CHUNK = 500  # notes loaded into RAM at a time

    if force:
        total = store.note_count()
        log.info(f"[embed] Force re-embedding all {total} notes…")

        chroma_col = _get_chroma_collection(store) if chroma_sync else None
        done = 0

        for chunk in store.iter_notes_chunked(chunk_size=CHUNK):
            done += _embed_chunk(chunk, provider, store, chroma_col)
            log.info(f"[embed] {done}/{total} done…")

    else:
        missing_ids = store.notes_without_embeddings_ids()
        if not missing_ids:
            log.info("[embed] All notes already embedded.")
            if chroma_sync:
                _sync_all_to_chroma(store)
            return

        total = len(missing_ids)
        log.info(f"[embed] Embedding {total} notes with {provider.backend}…")

        # Load missing notes in chunks using their IDs
        chroma_col = _get_chroma_collection(store) if chroma_sync else None
        done = 0

        for i in range(0, total, CHUNK):
            chunk_ids = missing_ids[i:i + CHUNK]
            # Load full notes only for this chunk
            chunk = [store.get_note(nid) for nid in chunk_ids]
            chunk = [n for n in chunk if n is not None]
            done += _embed_chunk(chunk, provider, store, chroma_col)
            if i % (CHUNK * 5) == 0 and i > 0:
                log.info(f"[embed] {done}/{total} done…")

    log.info(f"[embed] Embedding complete. {done} embedded.")


def _embed_chunk(chunk: list, provider: EmbeddingProvider, store,
                 chroma_col=None) -> int:
    """
    Embed one chunk of notes, write to SQLite in a single batch.
    Returns number of notes successfully embedded.
    """
    texts  = [f"{n.title}\n\n{n.content}" for n in chunk]
    done   = 0
    model  = provider.model

    try:
        vecs = provider.embed_batch(texts)
        rows = [(n.id, v, model) for n, v in zip(chunk, vecs) if v]
        store.save_embeddings_batch(rows)
        done = len(rows)

        # Optional Chroma sync
        if chroma_col and rows:
            try:
                chroma_col.upsert(
                    ids=[r[0] for r in rows],
                    embeddings=[r[1] for r in rows],
                    metadatas=[{"model": model}] * len(rows),
                )
            except Exception as e:
                log.debug(f"Chroma sync failed for batch: {e}")

    except Exception as e:
        log.warning(f"[embed] Batch failed ({e}), trying one-by-one…")
        rows = []
        for note, text in zip(chunk, texts):
            try:
                vec = provider.embed(text)
                rows.append((note.id, vec, model))
                done += 1
            except Exception as e2:
                log.warning(f"[embed] Failed {note.id[:8]}: {e2}")
        if rows:
            store.save_embeddings_batch(rows)
            if chroma_col:
                try:
                    chroma_col.upsert(
                        ids=[r[0] for r in rows],
                        embeddings=[r[1] for r in rows],
                        metadatas=[{"model": model}] * len(rows),
                    )
                except Exception:
                    pass

    return done


def _get_chroma_collection(store):
    """Open the Chroma collection that lives alongside this SQLite DB."""
    try:
        import chromadb
        chroma_path = str(store.db_path.parent / "chroma")
        client = chromadb.PersistentClient(path=chroma_path)
        col    = client.get_or_create_collection(
            name="brain",
            metadata={"hnsw:space": "cosine"},
        )
        log.info(f"[embed] Chroma sync enabled ({col.count()} existing vectors)")
        return col
    except Exception as e:
        log.debug(f"Chroma not available for sync: {e}")
        return None


def _sync_all_to_chroma(store):
    """Push all existing SQLite embeddings to Chroma (idempotent upsert)."""
    col = _get_chroma_collection(store)
    if col is None:
        return

    log.info("[embed] Syncing all embeddings to Chroma…")
    synced = 0
    for chunk in store.iter_embeddings_chunked(chunk_size=2000):
        ids  = list(chunk.keys())
        vecs = list(chunk.values())
        col.upsert(ids=ids, embeddings=vecs,
                   metadatas=[{"model": "synced"}] * len(ids))
        synced += len(ids)
        log.info(f"[embed] Chroma sync: {synced} done")
    log.info(f"[embed] Chroma sync complete. {synced} vectors.")


# ── Legacy search helper ──────────────────────────────────────────────────────

def search_by_embedding(store, query_embedding, top_k=5):
    """
    Semantic search over all stored embeddings.
    For small corpora: fine.
    For large corpora: use ChromaBackend.search_by_embedding() instead.
    """
    results = []
    for chunk in store.iter_embeddings_chunked(chunk_size=2000):
        for note_id, emb in chunk.items():
            if emb:
                sim = _cosine(query_embedding, emb)
                results.append((note_id, sim))

    results.sort(key=lambda x: x[1], reverse=True)
    return results[:top_k]


# ─── Helpers ──────────────────────────────────────────────────────────────────

def _load_sentence_transformers(model_name: str):
    try:
        from sentence_transformers import SentenceTransformer
        log.info(f"[embed] Loading sentence-transformers model: {model_name}")
        return SentenceTransformer(model_name)
    except ImportError:
        log.debug("sentence-transformers not installed")
        return None
    except Exception as e:
        log.warning(f"Could not load ST model {model_name}: {e}")
        return None


def _ollama_available(base_url: str, model: str) -> bool:
    import urllib.request, json
    try:
        with urllib.request.urlopen(f"{base_url}/api/tags", timeout=2) as r:
            data   = json.loads(r.read())
            models = [m.get("name", "").split(":")[0]
                      for m in data.get("models", [])]
            return model.split(":")[0] in models
    except Exception:
        return False


def _hash_token(token: str) -> int:
    h = 5381
    for ch in token:
        h = ((h << 5) + h) + ord(ch)
    return abs(h)


def _l2_norm(vec: list) -> list:
    mag = math.sqrt(sum(x * x for x in vec))
    return [x / mag for x in vec] if mag > 0 else vec


def _cosine(a: list, b: list) -> float:
    dot  = sum(x * y for x, y in zip(a, b))
    na   = math.sqrt(sum(x * x for x in a)) or 1.0
    nb   = math.sqrt(sum(y * y for y in b)) or 1.0
    return dot / (na * nb)
