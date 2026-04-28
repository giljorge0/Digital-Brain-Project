"""
Relation Extractor
------------------
Uses an LLM to extract:
  1. Named relations between notes (X argues_for Y, X contradicts Y, etc.)
  2. Atomic claims from a note's content
  3. Contradiction detection between two notes

Supports two backends:
  claude  — Anthropic Messages API
  ollama  — local Ollama REST API
"""

import json
import logging
import os
import urllib.request
import urllib.error
from typing import Optional

log = logging.getLogger(__name__)


# ─── Relation types ───────────────────────────────────────────────────────────

RELATION_TYPES = [
    "supports",
    "contradicts",
    "elaborates",
    "cites",
    "depends_on",
    "is_example_of",
    "generalizes",
    "questions",
]


# ─── Prompts ──────────────────────────────────────────────────────────────────

_RELATIONS_PROMPT = """\
You are a knowledge graph extractor. Given two notes, identify the semantic
relationship between them, if any.

Note A — "{title_a}":
{content_a}

Note B — "{title_b}":
{content_b}

Choose the best relationship from: {relation_types}
Or answer "none" if no meaningful relationship exists.

Respond ONLY in JSON, no explanation:
{{
  "relation": "<type or none>",
  "confidence": <0.0-1.0>,
  "explanation": "<one sentence>"
}}"""

_CLAIMS_PROMPT = """\
You are a philosopher's assistant. Extract every distinct atomic claim from
the following note. Each claim should be a single self-contained assertion.

Note: "{title}"
{content}

Respond ONLY in JSON:
{{
  "claims": [
    {{"claim": "<atomic statement>", "confidence": <0.0-1.0>}}
  ]
}}"""

_CONTRADICTION_PROMPT = """\
Do these two claims contradict each other?

Claim A: {claim_a}
Claim B: {claim_b}

Respond ONLY in JSON:
{{
  "contradicts": <true|false>,
  "confidence": <0.0-1.0>,
  "explanation": "<one sentence>"
}}"""


# ─── Extractor base ───────────────────────────────────────────────────────────

class RelationExtractor:
    def extract_relation(self, note_a, note_b) -> Optional[dict]:
        raise NotImplementedError

    def extract_claims(self, note) -> list:
        raise NotImplementedError

    def check_contradiction(self, claim_a: str, claim_b: str) -> dict:
        raise NotImplementedError

    @staticmethod
    def from_config(cfg: dict) -> "RelationExtractor":
        backend = cfg.get("llm_backend", "claude").lower()
        if backend == "claude":
            return ClaudeExtractor(
                api_key=cfg.get("anthropic_api_key") or
                        os.environ.get("ANTHROPIC_API_KEY", ""),
                model=cfg.get("claude_model", "claude-haiku-4-5-20251001"),
            )
        elif backend == "ollama":
            return OllamaExtractor(
                model=cfg.get("ollama_model", "mistral"),
                base_url=cfg.get("ollama_base_url", "http://localhost:11434"),
            )
        else:
            raise ValueError(f"Unknown LLM backend: {backend}")


# ─── Claude backend ───────────────────────────────────────────────────────────

class ClaudeExtractor(RelationExtractor):
    _API_URL = "https://api.anthropic.com/v1/messages"

    def __init__(self, api_key: str, model: str = "claude-haiku-4-5-20251001"):
        self.api_key = api_key
        self.model = model

    def _call(self, prompt: str, max_tokens: int = 512) -> str:
        payload = json.dumps({
            "model": self.model,
            "max_tokens": max_tokens,
            "messages": [{"role": "user", "content": prompt}],
        }).encode("utf-8")

        req = urllib.request.Request(
            self._API_URL,
            data=payload,
            headers={
                "Content-Type": "application/json",
                "x-api-key": self.api_key,
                "anthropic-version": "2023-06-01",
            },
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=60) as resp:
            data = json.loads(resp.read())
            return data["content"][0]["text"]

    def _parse_json(self, text: str) -> dict:
        text = text.strip().lstrip("```json").lstrip("```").rstrip("```").strip()
        return json.loads(text)

    def extract_relation(self, note_a, note_b) -> Optional[dict]:
        prompt = _RELATIONS_PROMPT.format(
            title_a=note_a.title,
            content_a=note_a.short_content(500),
            title_b=note_b.title,
            content_b=note_b.short_content(500),
            relation_types=", ".join(RELATION_TYPES),
        )
        try:
            raw = self._call(prompt)
            result = self._parse_json(raw)
            if result.get("relation") == "none":
                return None
            return result
        except Exception as e:
            log.error(f"[relations] extract_relation failed: {e}")
            return None

    def extract_claims(self, note) -> list:
        prompt = _CLAIMS_PROMPT.format(
            title=note.title,
            content=note.short_content(1500),
        )
        try:
            raw = self._call(prompt, max_tokens=1024)
            result = self._parse_json(raw)
            return result.get("claims", [])
        except Exception as e:
            log.error(f"[relations] extract_claims failed: {e}")
            return []

    def check_contradiction(self, claim_a: str, claim_b: str) -> dict:
        prompt = _CONTRADICTION_PROMPT.format(
            claim_a=claim_a, claim_b=claim_b
        )
        try:
            raw = self._call(prompt)
            return self._parse_json(raw)
        except Exception as e:
            log.error(f"[relations] check_contradiction failed: {e}")
            return {"contradicts": False, "confidence": 0.0, "explanation": str(e)}


# ─── Ollama backend ───────────────────────────────────────────────────────────

class OllamaExtractor(RelationExtractor):
    def __init__(self, model: str = "mistral",
                 base_url: str = "http://localhost:11434"):
        self.model = model
        self.base_url = base_url.rstrip("/")

    def _call(self, prompt: str) -> str:
        payload = json.dumps({
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "format": "json",
        }).encode("utf-8")

        req = urllib.request.Request(
            f"{self.base_url}/api/generate",
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=120) as resp:
            data = json.loads(resp.read())
            return data["response"]

    def _parse_json(self, text: str) -> dict:
        return json.loads(text.strip())

    def extract_relation(self, note_a, note_b) -> Optional[dict]:
        prompt = _RELATIONS_PROMPT.format(
            title_a=note_a.title,
            content_a=note_a.short_content(500),
            title_b=note_b.title,
            content_b=note_b.short_content(500),
            relation_types=", ".join(RELATION_TYPES),
        )
        try:
            raw = self._call(prompt)
            result = self._parse_json(raw)
            return None if result.get("relation") == "none" else result
        except Exception as e:
            log.error(f"[ollama] extract_relation failed: {e}")
            return None

    def extract_claims(self, note) -> list:
        prompt = _CLAIMS_PROMPT.format(
            title=note.title,
            content=note.short_content(1500),
        )
        try:
            raw = self._call(prompt)
            return self._parse_json(raw).get("claims", [])
        except Exception as e:
            log.error(f"[ollama] extract_claims failed: {e}")
            return []

    def check_contradiction(self, claim_a: str, claim_b: str) -> dict:
        prompt = _CONTRADICTION_PROMPT.format(claim_a=claim_a, claim_b=claim_b)
        try:
            raw = self._call(prompt)
            return self._parse_json(raw)
        except Exception as e:
            return {"contradicts": False, "confidence": 0.0, "explanation": str(e)}


# ─── Batch relation extraction ────────────────────────────────────────────────

def extract_llm_edges(store, extractor: RelationExtractor,
                      max_pairs: int = 200):
    """
    Sample note pairs and extract LLM relations, persist to store.
    Uses centrality to prioritize high-value notes.
    """
    notes = store.get_all_notes()
    # Prioritize by centrality (most connected first)
    notes.sort(key=lambda n: n.centrality, reverse=True)
    top_notes = notes[:50]  # sample top 50

    pairs = []
    for i, a in enumerate(top_notes):
        for b in top_notes[i + 1:]:
            pairs.append((a, b))
            if len(pairs) >= max_pairs:
                break
        if len(pairs) >= max_pairs:
            break

    log.info(f"[relations] Extracting LLM edges for {len(pairs)} pairs...")

    for a, b in pairs:
        result = extractor.extract_relation(a, b)
        if result and result.get("relation"):
            store.upsert_edge(
                a.id, b.id,
                edge_type="llm",
                weight=result.get("confidence", 0.5),
                metadata={
                    "relation": result["relation"],
                    "explanation": result.get("explanation", ""),
                }
            )

    log.info("[relations] LLM edge extraction complete")
