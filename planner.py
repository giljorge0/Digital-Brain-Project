"""
Query Planner
-------------
Routes a natural-language question to the right retrieval strategy:

  semantic   — embedding similarity search
  keyword    — full-text LIKE search in store
  graph      — multi-hop traversal from seed note
  temporal   — date-filtered retrieval
  hybrid     — semantic + graph expansion

Then synthesizes an answer using the configured LLM.
"""

import json
import logging
import os
import urllib.request
from datetime import datetime
from typing import Optional

from brain.memory.embeddings import EmbeddingProvider, search_by_embedding

log = logging.getLogger(__name__)


# ─── Query types ──────────────────────────────────────────────────────────────

QUERY_TYPES = {
    "semantic": "Best for conceptual or open-ended questions",
    "keyword":  "Best for specific terms, names, or titles",
    "temporal": "Best for questions about time, dates, history",
    "graph":    "Best for relationship and connection questions",
    "hybrid":   "Best for broad synthesis across topics",
}


# ─── Planner ──────────────────────────────────────────────────────────────────

class QueryPlanner:
    def __init__(self, store, embedder: EmbeddingProvider,
                 llm_cfg: dict = None):
        self.store = store
        self.embedder = embedder
        self.llm_cfg = llm_cfg or {}

    def query(self, question: str, top_k: int = 8,
              mode: str = "auto") -> dict:
        """
        Answer a question. Returns:
          {
            "answer": str,
            "mode": str,
            "sources": [{"id", "title", "score", "snippet"}],
            "confidence": float
          }
        """
        if mode == "auto":
            mode = self._detect_mode(question)

        log.info(f"[query] Mode={mode} | Q: {question[:80]}")

        if mode == "keyword":
            notes = self.store.search_notes(question, limit=top_k)
            sources = self._notes_to_sources(notes, question)

        elif mode == "temporal":
            notes = self._temporal_search(question, limit=top_k)
            sources = self._notes_to_sources(notes, question)

        elif mode == "graph":
            notes = self._graph_search(question, top_k=top_k)
            sources = self._notes_to_sources(notes, question)

        elif mode == "hybrid":
            sem_results = self._semantic_search(question, top_k=top_k // 2)
            kw_results = self.store.search_notes(question, limit=top_k // 2)
            combined = {n.id: n for n in kw_results}
            combined.update({n.id: n for n in sem_results})
            sources = self._notes_to_sources(list(combined.values()), question)

        else:  # semantic (default)
            notes = self._semantic_search(question, top_k=top_k)
            sources = self._notes_to_sources(notes, question)

        if not sources:
            return {
                "answer": "No relevant notes found for this query.",
                "mode": mode,
                "sources": [],
                "confidence": 0.0,
            }

        answer = self._synthesize(question, sources)

        return {
            "answer": answer,
            "mode": mode,
            "sources": sources[:top_k],
            "confidence": self._estimate_confidence(sources),
        }

    # ── Private retrieval ─────────────────────────────────────────────────────

    def _semantic_search(self, question: str, top_k: int = 8) -> list:
        try:
            q_vec = self.embedder.embed_one(question)
            hits = search_by_embedding(self.store, q_vec, top_k=top_k)
            notes = []
            for note_id, _ in hits:
                note = self.store.get_note(note_id)
                if note:
                    notes.append(note)
            return notes
        except Exception as e:
            log.warning(f"[query] Semantic search failed: {e}, falling back to keyword")
            return self.store.search_notes(question, limit=top_k)

    def _temporal_search(self, question: str, limit: int = 8) -> list:
        """Simple date extraction + filtered keyword search."""
        # Try to find a year in the question
        import re
        year_match = re.search(r'\b(19|20)\d{2}\b', question)
        all_notes = self.store.get_all_notes()
        if year_match:
            year = int(year_match.group())
            dated = [n for n in all_notes
                     if n.date and n.date.year == year]
            return dated[:limit] if dated else all_notes[:limit]
        # Fall back: sort by date descending
        dated = sorted([n for n in all_notes if n.date],
                       key=lambda n: n.date, reverse=True)
        return dated[:limit]

    def _graph_search(self, question: str, top_k: int = 8) -> list:
        """
        Find seed note via semantic search, then expand through graph neighbors.
        """
        seeds = self._semantic_search(question, top_k=3)
        if not seeds:
            return []

        visited = {n.id: n for n in seeds}
        for seed in seeds:
            edges = self.store.get_edges(note_id=seed.id)
            for edge in edges:
                neighbor_id = (edge["target"]
                               if edge["source"] == seed.id
                               else edge["source"])
                if neighbor_id not in visited:
                    n = self.store.get_note(neighbor_id)
                    if n:
                        visited[neighbor_id] = n
                if len(visited) >= top_k * 2:
                    break

        # Re-rank by keyword relevance
        results = list(visited.values())
        query_words = set(question.lower().split())
        def _score(note):
            text = (note.title + " " + note.content).lower()
            return sum(1 for w in query_words if w in text)
        results.sort(key=_score, reverse=True)
        return results[:top_k]

    # ── Mode detection ────────────────────────────────────────────────────────

    def _detect_mode(self, question: str) -> str:
        q = question.lower()
        temporal_signals = ["when", "date", "year", "history", "timeline",
                            "before", "after", "ago", "recently"]
        graph_signals = ["connect", "relation", "link", "between", "depends",
                         "related to", "contrast", "compare"]
        keyword_signals = ["what is", "who is", "define", "definition",
                           "explain", "meaning"]

        if any(s in q for s in temporal_signals):
            return "temporal"
        if any(s in q for s in graph_signals):
            return "graph"
        if any(s in q for s in keyword_signals):
            return "hybrid"
        return "semantic"

    # ── Synthesis ─────────────────────────────────────────────────────────────

    def _synthesize(self, question: str, sources: list) -> str:
        backend = self.llm_cfg.get("llm_backend", "claude")
        context = "\n\n---\n\n".join(
            f"[{s['title']}]\n{s['snippet']}" for s in sources[:5]
        )
        prompt = (
            f"You are a personal knowledge assistant. "
            f"Answer the question below using ONLY the provided notes.\n\n"
            f"Question: {question}\n\n"
            f"Notes:\n{context}\n\n"
            f"Answer concisely and cite the note titles used."
        )

        if backend == "ollama":
            return self._ollama_complete(prompt)
        else:
            return self._claude_complete(prompt)

    def _claude_complete(self, prompt: str) -> str:
        api_key = (self.llm_cfg.get("anthropic_api_key") or
                   os.environ.get("ANTHROPIC_API_KEY", ""))
        model = self.llm_cfg.get("claude_model", "claude-haiku-4-5-20251001")
        payload = json.dumps({
            "model": model,
            "max_tokens": 1024,
            "messages": [{"role": "user", "content": prompt}],
        }).encode("utf-8")
        req = urllib.request.Request(
            "https://api.anthropic.com/v1/messages",
            data=payload,
            headers={
                "Content-Type": "application/json",
                "x-api-key": api_key,
                "anthropic-version": "2023-06-01",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=60) as resp:
                data = json.loads(resp.read())
                return data["content"][0]["text"]
        except Exception as e:
            return f"[synthesis failed: {e}]"

    def _ollama_complete(self, prompt: str) -> str:
        base_url = self.llm_cfg.get("ollama_base_url", "http://localhost:11434")
        model = self.llm_cfg.get("ollama_model", "mistral")
        payload = json.dumps({
            "model": model,
            "prompt": prompt,
            "stream": False,
        }).encode("utf-8")
        req = urllib.request.Request(
            f"{base_url.rstrip('/')}/api/generate",
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=120) as resp:
                data = json.loads(resp.read())
                return data["response"]
        except Exception as e:
            return f"[synthesis failed: {e}]"

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _notes_to_sources(self, notes: list, query: str) -> list:
        sources = []
        q_words = set(query.lower().split())
        for note in notes:
            text = note.content.lower()
            score = sum(1 for w in q_words if w in text) / max(len(q_words), 1)
            sources.append({
                "id": note.id,
                "title": note.title,
                "score": round(score, 3),
                "snippet": note.short_content(300),
                "tags": note.tags,
                "date": note.date.isoformat() if note.date else None,
            })
        sources.sort(key=lambda x: x["score"], reverse=True)
        return sources

    def _estimate_confidence(self, sources: list) -> float:
        if not sources:
            return 0.0
        avg = sum(s["score"] for s in sources[:3]) / min(len(sources), 3)
        return round(min(avg * 2, 1.0), 2)