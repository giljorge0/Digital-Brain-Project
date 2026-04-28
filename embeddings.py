"""
Embeddings
----------
Two backends, auto-selected by config:

  local   — sentence-transformers (runs fully offline)
  claude  — Anthropic voyage embeddings via API (higher quality)
  ollama  — Ollama embeddings endpoint (local, model-selectable)

Usage:
    provider = EmbeddingProvider.from_config(cfg)
    vectors  = provider.embed(["text one", "text two"])
"""

import logging
import os
import json
import math
import urllib.request
import urllib.error
from typing import Optional

log = logging.getLogger(__name__)


# ─── Base ─────────────────────────────────────────────────────────────────────

class EmbeddingProvider:
    name: str = "base"

    def embed(self, texts: list) -> list:
        """Embed a list of strings. Returns list of float lists."""
        raise NotImplementedError

    def embed_one(self, text: str) -> list:
        return self.embed([text])[0]

    @staticmethod
    def from_config(cfg: dict) -> "EmbeddingProvider":
        backend = cfg.get("embedding_backend", "local").lower()
        if backend == "local":
            return LocalEmbeddingProvider(
                model=cfg.get("local_embedding_model",
                              "all-MiniLM-L6-v2")
            )
        elif backend == "ollama":
            return OllamaEmbeddingProvider(
                model=cfg.get("ollama_embedding_model", "nomic-embed-text"),
                base_url=cfg.get("ollama_base_url", "http://localhost:11434"),
            )
        elif backend in ("claude", "voyage", "anthropic"):
            return VoyageEmbeddingProvider(
                api_key=cfg.get("anthropic_api_key") or os.environ.get("ANTHROPIC_API_KEY", ""),
                model=cfg.get("voyage_model", "voyage-3"),
            )
        else:
            raise ValueError(f"Unknown embedding backend: {backend}")


# ─── Local (sentence-transformers) ───────────────────────────────────────────

class LocalEmbeddingProvider(EmbeddingProvider):
    name = "local"

    def __init__(self, model: str = "all-MiniLM-L6-v2"):
        self.model_name = model
        self._model = None

    def _load(self):
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer
                log.info(f"[embed] Loading local model: {self.model_name}")
                self._model = SentenceTransformer(self.model_name)
            except ImportError:
                raise ImportError(
                    "pip install sentence-transformers"
                )

    def embed(self, texts: list) -> list:
        self._load()
        vecs = self._model.encode(texts, show_progress_bar=False)
        return [v.tolist() for v in vecs]


# ─── Ollama ───────────────────────────────────────────────────────────────────

class OllamaEmbeddingProvider(EmbeddingProvider):
    name = "ollama"

    def __init__(self, model: str = "nomic-embed-text",
                 base_url: str = "http://localhost:11434"):
        self.model = model
        self.base_url = base_url.rstrip("/")

    def embed(self, texts: list) -> list:
        results = []
        for text in texts:
            payload = json.dumps({
                "model": self.model,
                "prompt": text
            }).encode("utf-8")
            req = urllib.request.Request(
                f"{self.base_url}/api/embeddings",
                data=payload,
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            try:
                with urllib.request.urlopen(req, timeout=30) as resp:
                    data = json.loads(resp.read())
                    results.append(data["embedding"])
            except urllib.error.URLError as e:
                raise ConnectionError(
                    f"Ollama not reachable at {self.base_url}: {e}"
                )
        return results


# ─── Voyage / Anthropic ───────────────────────────────────────────────────────

class VoyageEmbeddingProvider(EmbeddingProvider):
    name = "voyage"
    _API_URL = "https://api.voyageai.com/v1/embeddings"

    def __init__(self, api_key: str, model: str = "voyage-3"):
        self.api_key = api_key
        self.model = model

    def embed(self, texts: list) -> list:
        payload = json.dumps({
            "model": self.model,
            "input": texts,
            "input_type": "document",
        }).encode("utf-8")

        req = urllib.request.Request(
            self._API_URL,
            data=payload,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=60) as resp:
                data = json.loads(resp.read())
                return [item["embedding"] for item in data["data"]]
        except urllib.error.HTTPError as e:
            raise RuntimeError(f"Voyage API error {e.code}: {e.read()}")


# ─── Batch embed + persist ────────────────────────────────────────────────────

def embed_notes(store, provider: EmbeddingProvider,
                batch_size: int = 32, force: bool = False):
    """
    Embed all notes that don't yet have embeddings (or all if force=True).
    Persists vectors to store.
    """
    from brain.ingest.note import Note

    if force:
        notes = store.get_all_notes()
    else:
        notes = store.notes_without_embeddings()

    if not notes:
        log.info("[embed] All notes already embedded")
        return

    log.info(f"[embed] Embedding {len(notes)} notes with {provider.name}...")

    for i in range(0, len(notes), batch_size):
        batch = notes[i:i + batch_size]
        texts = [f"{n.title}\n\n{n.content[:1000]}" for n in batch]
        try:
            vectors = provider.embed(texts)
            for note, vec in zip(batch, vectors):
                store.save_embedding(note.id, vec, model=provider.name)
            log.info(f"[embed] {min(i + batch_size, len(notes))}/{len(notes)}")
        except Exception as e:
            log.error(f"[embed] Batch {i} failed: {e}")

    log.info("[embed] Done")


# ─── Cosine search helper ─────────────────────────────────────────────────────

def search_by_embedding(store, query_vec: list,
                        top_k: int = 10) -> list:
    """
    Return top-k notes by cosine similarity to query_vec.
    Returns list of (note_id, score) tuples, sorted desc.
    """
    all_embeddings = store.get_all_embeddings()
    results = []
    for note_id, vec in all_embeddings.items():
        score = _cosine(query_vec, vec)
        results.append((note_id, score))
    results.sort(key=lambda x: x[1], reverse=True)
    return results[:top_k]


def _cosine(a: list, b: list) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a)) or 1.0
    norm_b = math.sqrt(sum(y * y for y in b)) or 1.0
    return dot / (norm_a * norm_b)