"""
Embedding Provider
------------------
Priority order (auto-detected at runtime):

  1. Ollama local          — nomic-embed-text, mxbai-embed-large, etc.
                             zero cost, runs on your machine
                             needs: ollama pull nomic-embed-text

  2. sentence-transformers — all-MiniLM-L6-v2, all-mpnet-base-v2, etc.
                             runs fully offline, no server needed
                             needs: pip install sentence-transformers

  3. OpenAI API            — text-embedding-3-small
                             needs: OPENAI_API_KEY

  4. TF-IDF fallback       — no dependencies, deterministic, fast
                             lower quality but always works

Config keys (config.yaml or env vars):
  embedding_backend:     ollama | sentence_transformers | openai | tfidf
  local_embedding_model: nomic-embed-text  (for ollama)
                         all-MiniLM-L6-v2  (for sentence-transformers)
  ollama_base_url:       http://localhost:11434
"""

from __future__ import annotations

import math
import logging
import re
from collections import Counter
from typing import Optional

log = logging.getLogger("brain.embeddings")

EMBED_BATCH = 64
TFIDF_DIM   = 512


class EmbeddingProvider:

    def __init__(self, backend: str, model: str,
                 api_key: str = "",
                 base_url: str = "http://localhost:11434",
                 _st_model=None):
        self.backend   = backend
        self.model     = model
        self.api_key   = api_key
        self.base_url  = base_url
        self._st_model = _st_model   # cached SentenceTransformer instance

    # ── Factories ─────────────────────────────────────────────────────────────

    @classmethod
    def from_config(cls, cfg: dict) -> "EmbeddingProvider":
        """
        Build from the config dict (from get_config() in main.py).
        Auto-detects best available backend if 'embedding_backend' is 'auto' or missing.
        """
        backend = cfg.get("embedding_backend", "auto").lower()
        model   = cfg.get("local_embedding_model", "all-MiniLM-L6-v2")
        base    = cfg.get("ollama_base_url", "http://localhost:11434")
        api_key = cfg.get("openai_api_key", "")

        if backend == "ollama":
            return cls("ollama", cfg.get("local_embedding_model", "nomic-embed-text"),
                       base_url=base)

        if backend == "sentence_transformers":
            st = _load_sentence_transformers(model)
            if st:
                return cls("sentence_transformers", model, _st_model=st)
            log.warning("sentence-transformers failed to load — falling back")

        if backend == "openai":
            return cls("openai", "text-embedding-3-small", api_key=api_key)

        if backend == "tfidf":
            return cls("tfidf", "tfidf")

        # ── Auto-detect: try in priority order ────────────────────────────────
        if backend in ("auto", "local"):
            log.info("[embed] Auto-detecting best embedding backend…")

            # 1. Ollama
            ollama_model = cfg.get("local_embedding_model", "nomic-embed-text")
            if _ollama_available(base, ollama_model):
                log.info(f"[embed] Using Ollama ({ollama_model})")
                return cls("ollama", ollama_model, base_url=base)

            # 2. sentence-transformers
            st_model_name = cfg.get("local_embedding_model", "all-MiniLM-L6-v2")
            if "nomic" in st_model_name:
                st_model_name = "all-MiniLM-L6-v2"   # nomic is Ollama-only
            st = _load_sentence_transformers(st_model_name)
            if st:
                log.info(f"[embed] Using sentence-transformers ({st_model_name})")
                return cls("sentence_transformers", st_model_name, _st_model=st)

            # 3. OpenAI
            import os
            oai_key = api_key or os.environ.get("OPENAI_API_KEY", "")
            if oai_key:
                log.info("[embed] Using OpenAI text-embedding-3-small")
                return cls("openai", "text-embedding-3-small", api_key=oai_key)

            # 4. TF-IDF fallback
            log.warning(
                "[embed] No embedding backend found. Using TF-IDF fallback.\n"
                "        For better results install one of:\n"
                "          pip install sentence-transformers   (offline, recommended)\n"
                "          ollama pull nomic-embed-text        (if Ollama is running)"
            )
            return cls("tfidf", "tfidf")

        return cls("tfidf", "tfidf")

    @classmethod
    def from_registry(cls, registry) -> "EmbeddingProvider":
        """Build from LLMRegistry (llm_profiles.yaml)."""
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
        log.warning("No embed profile in registry — falling back to auto-detect")
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
        """Helper to embed a single string instead of a list."""
        return self.embed([text])[0]

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
            log.warning(f"Ollama embed failed: {e} — falling back to TF-IDF")
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
            resp = client.embeddings.create(model="text-embedding-3-small", input=text)
            return resp.data[0].embedding
        except Exception as e:
            log.warning(f"OpenAI embed failed: {e} — falling back to TF-IDF")
            return self._tfidf_embed(text)

    def _tfidf_embed(self, text: str) -> list:
        tokens = re.findall(r"\b[a-z]{3,}\b", text.lower())
        freq = Counter(tokens)
        if not freq:
            return [0.0] * TFIDF_DIM
        vec = [0.0] * TFIDF_DIM
        total = sum(freq.values())
        for token, count in freq.items():
            h = _hash_token(token)
            vec[h % TFIDF_DIM] += count / total
        return _l2_norm(vec)


# ─── Bulk helper ──────────────────────────────────────────────────────────────

def embed_notes(store, provider: EmbeddingProvider, force: bool = False):
    """
    Embed all notes missing a vector (or all notes if force=True).
    Writes vectors to store.
    """
    if force:
        missing = store.get_all_notes()
        log.info(f"[embed] Force re-embedding all {len(missing)} notes…")
    else:
        missing = store.notes_without_embeddings()
        if not missing:
            log.info("[embed] All notes already embedded.")
            return

    log.info(f"[embed] Embedding {len(missing)} notes with {provider.backend}…")
    done = 0
    failed = 0

    for i in range(0, len(missing), EMBED_BATCH):
        batch = missing[i:i + EMBED_BATCH]
        texts = [f"{n.title}\n\n{n.content}" for n in batch]
        try:
            vecs = provider.embed_batch(texts)
            for note, vec in zip(batch, vecs):
                store.save_embedding(note.id, vec, model=provider.model)
                done += 1
        except Exception as e:
            # Individual fallback
            for note, text in zip(batch, texts):
                try:
                    vec = provider.embed(text)
                    store.save_embedding(note.id, vec, model=provider.model)
                    done += 1
                except Exception as e2:
                    log.warning(f"[embed] Failed {note.id[:8]}: {e2}")
                    failed += 1

        if (i // EMBED_BATCH) % 5 == 0:
            log.info(f"[embed] {done}/{len(missing)} done…")

    log.info(f"[embed] Done. {done} embedded, {failed} failed.")



# ─── Helpers ──────────────────────────────────────────────────────────────────

def _load_sentence_transformers(model_name: str):
    """Try to load a SentenceTransformer model. Returns None on failure."""
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
    """Check if Ollama is running and has the requested model."""
    import urllib.request, json
    try:
        with urllib.request.urlopen(f"{base_url}/api/tags", timeout=2) as r:
            data = json.loads(r.read())
            models = [m.get("name","").split(":")[0] for m in data.get("models",[])]
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


import math

def search_by_embedding(store, query_embedding, top_k=5):
    """
    Ranks notes by comparing their embeddings to a query embedding.
    """
    def calc_cosine(vec1, vec2):
        dot = sum(a * b for a, b in zip(vec1, vec2))
        mag1 = math.sqrt(sum(a * a for a in vec1))
        mag2 = math.sqrt(sum(b * b for b in vec2))
        if mag1 == 0 or mag2 == 0:
            return 0.0
        return dot / (mag1 * mag2)

    # Fetch the actual dictionary of embeddings from the store
    note_embeddings_dict = store.get_all_embeddings()

    results = []
    # (We put .items() back because it is now actually a dictionary!)
    for note_id, emb in note_embeddings_dict.items():
        if not emb: 
            continue
        sim = calc_cosine(query_embedding, emb)
        results.append((note_id, sim))
    
    # Sort by highest similarity first
    results.sort(key=lambda x: x[1], reverse=True)
    return results[:top_k]


import math

def _cosine(vec1, vec2):
    """Calculate cosine similarity between two vectors."""
    dot_product = sum(a * b for a, b in zip(vec1, vec2))
    magnitude1 = math.sqrt(sum(a * a for a in vec1))
    magnitude2 = math.sqrt(sum(b * b for b in vec2))
    if magnitude1 == 0 or magnitude2 == 0:
        return 0.0
    return dot_product / (magnitude1 * magnitude2)

