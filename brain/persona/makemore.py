"""
Persona — Digital You
---------------------
Two Karpathy-inspired capabilities:

1. MAKEMORE (digital twin)
   Learns your writing style from your OUTPUT notes (your org files,
   your chat turns, your written pieces). Then generates new text in
   your voice on any topic.

2. LLM WIKI
   Given a topic, retrieves all relevant notes from your corpus and
   synthesizes a Wikipedia-style article written in *your* voice —
   as if you had written a proper essay on the topic. Connects to
   related ideas in your graph.

Both respect the input/output provenance distinction:
  - ONLY notes with provenance_role = "output" feed the style model.
  - Both input AND output notes feed the knowledge retrieval.

Usage:
    persona = Persona(store, embedder, cfg)

    # Generate text in your voice on a topic
    text = persona.makemore("consciousness and computation")

    # Generate a wiki-style article from your corpus
    article = persona.llm_wiki("epistemic limits of formal systems")

    # Get your "position" on a topic (what do YOU believe about X?)
    position = persona.my_position("free will")
"""

import json
import logging
import os
import urllib.request
from pathlib import Path
from typing import Optional

from brain.memory.embeddings import EmbeddingProvider, search_by_embedding

log = logging.getLogger(__name__)

ROLE_OUTPUT = "output"


# ─── Persona ─────────────────────────────────────────────────────────────────

class Persona:
    def __init__(self, store, embedder: EmbeddingProvider, cfg: dict):
        self.store   = store
        self.embedder = embedder
        self.cfg     = cfg

    # ── Makemore: generate text in your style ─────────────────────────────────

    def makemore(self, topic: str, length: str = "medium",
                 temperature_hint: str = "natural") -> str:
        """
        Generate new text in YOUR voice on the given topic.
        Uses your output notes as style + knowledge context.

        length: "short" | "medium" | "long"
        temperature_hint: "natural" | "exploratory" | "precise"
        """
        # 1. Get your own writing as style examples
        style_samples = self._get_style_samples(topic, n=5)

        # 2. Get relevant knowledge (all notes, including inputs)
        knowledge = self._get_knowledge(topic, n=8)

        # 3. Compose prompt
        length_guide = {
            "short":  "2-3 paragraphs",
            "medium": "4-6 paragraphs",
            "long":   "8-10 paragraphs or more",
        }.get(length, "4-6 paragraphs")

        style_block = "\n\n---\n\n".join(
            f"[from: {s['title']}]\n{s['text']}" for s in style_samples
        )
        knowledge_block = "\n\n---\n\n".join(
            f"[{k['title']}]\n{k['text']}" for k in knowledge
        )

        prompt = f"""You are a digital twin of the author whose writings appear below.
Your task is to generate NEW original writing on the given topic,
EXACTLY in the author's voice, style, vocabulary, and intellectual manner.

AUTHOR'S STYLE (their actual writings — study these carefully):
{style_block}

RELEVANT KNOWLEDGE FROM THEIR CORPUS (additional context):
{knowledge_block}

TOPIC TO WRITE ABOUT: {topic}

Instructions:
- Write {length_guide} in the author's first person voice
- Preserve their characteristic sentence structures, vocabulary, and argumentative style
- Introduce genuinely new thoughts they might have had, grounded in their existing ideas
- Do NOT be generic. This must sound unmistakably like the author.
- Do NOT use bullet points unless the author uses them
- Connect to ideas that appear in their corpus where relevant
- Tone: {temperature_hint}

Write the new passage now:"""

        return self._llm_call(prompt, max_tokens=1500)

    # ── LLM Wiki: synthesize a wiki-style article ─────────────────────────────

    def llm_wiki(self, topic: str,
                 include_related: bool = True) -> dict:
        """
        Generate a Wikipedia-style article on a topic,
        synthesized from YOUR corpus and written in YOUR voice.

        Returns:
          {
            "title": str,
            "article": str,
            "related_notes": [{"id", "title", "relevance"}],
            "open_questions": [str],   # future directions
            "word_count": int,
          }
        """
        # Retrieve all relevant notes
        knowledge = self._get_knowledge(topic, n=15)
        style_samples = self._get_style_samples(topic, n=3)

        knowledge_block = "\n\n---\n\n".join(
            f"[Note: {k['title']}]\n{k['text']}" for k in knowledge
        )
        style_block = "\n\n---\n\n".join(
            f"{s['text']}" for s in style_samples[:2]
        )

        prompt = f"""You are synthesizing a personal wiki article from the author's corpus.

AUTHOR'S WRITING STYLE (for voice reference):
{style_block}

RELEVANT CORPUS NOTES ON THIS TOPIC:
{knowledge_block}

TASK: Write a comprehensive wiki-style article on "{topic}"

Requirements:
1. Written in the FIRST PERSON voice of the author (as if they wrote it themselves)
2. Structure: Introduction → Core Arguments → Connections to Related Ideas → Open Questions
3. Draw ONLY from the knowledge in their corpus — but synthesize it into coherent prose
4. Where the corpus has gaps, explicitly note them as open questions
5. Length: 500-800 words
6. Do NOT use generic Wikipedia-style neutral voice — this is a PERSONAL wiki

After the article, list:
OPEN_QUESTIONS: [3-5 questions this topic raises that the author hasn't addressed]
RELATED_TOPICS: [3-5 related topics from their corpus worth exploring]

Respond in JSON format:
{{
  "title": "Wiki article title",
  "article": "full article text...",
  "open_questions": ["question 1", "question 2", ...],
  "related_topics": ["topic 1", "topic 2", ...]
}}"""

        raw = self._llm_call(prompt, max_tokens=2000)

        try:
            # Strip markdown fences
            clean = raw.strip().lstrip("```json").lstrip("```").rstrip("```").strip()
            result = json.loads(clean)
        except Exception:
            # Fallback: return raw as article
            result = {
                "title": topic.title(),
                "article": raw,
                "open_questions": [],
                "related_topics": [],
            }

        result["related_notes"] = [
            {"id": k["id"], "title": k["title"], "relevance": k.get("score", 0)}
            for k in knowledge[:8]
        ]
        result["word_count"] = len(result.get("article", "").split())

        return result

    # ── My position: what do YOU believe about X? ────────────────────────────

    def my_position(self, topic: str) -> dict:
        """
        Extract and articulate YOUR position on a topic from your corpus.

        Returns:
          {
            "position": str,         # your stance in 1-2 sentences
            "arguments": [str],      # your main arguments for it
            "tensions": [str],       # where you have conflicting notes
            "evolution": str,        # how your view changed over time (if dated notes)
            "source_notes": [str],   # note titles used
          }
        """
        knowledge = self._get_output_notes(topic, n=12)
        if not knowledge:
            return {
                "position": f"No writings found on '{topic}'",
                "arguments": [],
                "tensions": [],
                "evolution": "",
                "source_notes": [],
            }

        knowledge_block = "\n\n---\n\n".join(
            f"[{k['title']} | {k.get('date', 'undated')}]\n{k['text']}"
            for k in knowledge
        )

        prompt = f"""Analyze these writings by a single author to extract their
philosophical/intellectual position on: "{topic}"

WRITINGS:
{knowledge_block}

Respond in JSON:
{{
  "position": "1-2 sentence summary of their core stance",
  "arguments": ["argument 1", "argument 2", "argument 3"],
  "tensions": ["any internal contradictions or unresolved tensions"],
  "evolution": "how their view seems to have evolved over time (or 'insufficient temporal data')",
  "confidence": 0.0-1.0
}}"""

        raw = self._llm_call(prompt, max_tokens=1000)
        try:
            clean  = raw.strip().lstrip("```json").lstrip("```").rstrip("```").strip()
            result = json.loads(clean)
        except Exception:
            result = {"position": raw, "arguments": [], "tensions": [], "evolution": ""}

        result["source_notes"] = [k["title"] for k in knowledge]
        return result

    # ── Atomic idea suggester (Karpathy Zettelkasten idea) ───────────────────

    def suggest_new_notes(self, note_id: str, n: int = 5) -> list:
        """
        Given a note, suggest N new atomic notes that would naturally
        branch off from it — implementing the fractal note expansion idea.

        Returns list of {"title": str, "seed_content": str}
        """
        note = self.store.get_note(note_id)
        if not note:
            return []

        # Get neighboring notes for context
        edges = self.store.get_edges(note_id=note_id)
        neighbor_titles = []
        for edge in edges[:5]:
            other_id = edge["target"] if edge["source"] == note_id else edge["source"]
            other = self.store.get_note(other_id)
            if other:
                neighbor_titles.append(other.title)

        prompt = f"""You are helping expand a Zettelkasten knowledge graph.

Given this note, suggest {n} new ATOMIC notes that should be created
to fully develop the ideas contained within it. Each suggestion should
be a distinct idea that deserves its own standalone note.

SOURCE NOTE: "{note.title}"
CONTENT: {note.short_content(600)}

ALREADY CONNECTED NOTES: {', '.join(neighbor_titles) if neighbor_titles else 'none'}

For each suggested note:
- It should represent ONE atomic idea
- It should deepen, challenge, or extend the source note
- It should NOT duplicate existing connected notes

Respond in JSON:
{{
  "suggestions": [
    {{
      "title": "atomic note title",
      "seed_content": "1-2 sentence seed for the note",
      "relation_to_source": "how it connects (extends/challenges/examples/depends)"
    }}
  ]
}}"""

        raw = self._llm_call(prompt, max_tokens=800)
        try:
            clean = raw.strip().lstrip("```json").lstrip("```").rstrip("```").strip()
            return json.loads(clean).get("suggestions", [])
        except Exception:
            return []

    # ── Private helpers ───────────────────────────────────────────────────────

    def _get_style_samples(self, topic: str, n: int = 5) -> list:
        """Get YOUR output notes relevant to the topic for style."""
        return self._retrieve(topic, n=n, output_only=True)

    def _get_knowledge(self, topic: str, n: int = 10) -> list:
        """Get all relevant notes (input + output) for knowledge."""
        return self._retrieve(topic, n=n, output_only=False)

    def _get_output_notes(self, topic: str, n: int = 10) -> list:
        """Get only your output notes on a topic."""
        return self._retrieve(topic, n=n, output_only=True)

    def _retrieve(self, topic: str, n: int, output_only: bool) -> list:
        try:
            q_vec = self.embedder.embed_one(topic)
            hits  = search_by_embedding(self.store, q_vec, top_k=n * 3)
        except Exception:
            # Fallback to keyword
            notes = self.store.search_notes(topic, limit=n * 2)
            hits  = [(note.id, 1.0) for note in notes]

        results = []
        for note_id, score in hits:
            note = self.store.get_note(note_id)
            if not note:
                continue
            if output_only:
                role = note.metadata.get("provenance_role", ROLE_OUTPUT)
                if role != ROLE_OUTPUT:
                    continue
            results.append({
                "id":    note.id,
                "title": note.title,
                "text":  note.short_content(400),
                "date":  note.date.isoformat()[:10] if note.date else None,
                "score": round(score, 3),
            })
            if len(results) >= n:
                break

        return results

    def _llm_call(self, prompt: str, max_tokens: int = 1000) -> str:
        backend = self.cfg.get("llm_backend", "claude")
        if backend == "ollama":
            return self._ollama(prompt, max_tokens)
        return self._claude(prompt, max_tokens)

    def _claude(self, prompt: str, max_tokens: int) -> str:
        api_key = (self.cfg.get("anthropic_api_key") or
                   os.environ.get("ANTHROPIC_API_KEY", ""))
        model   = self.cfg.get("claude_model", "claude-haiku-4-5-20251001")
        payload = json.dumps({
            "model": model,
            "max_tokens": max_tokens,
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
            with urllib.request.urlopen(req, timeout=90) as resp:
                data = json.loads(resp.read())
                return data["content"][0]["text"]
        except Exception as e:
            return f"[LLM error: {e}]"

    def _ollama(self, prompt: str, max_tokens: int) -> str:
        base  = self.cfg.get("ollama_base_url", "http://localhost:11434")
        model = self.cfg.get("ollama_model", "mistral")
        payload = json.dumps({
            "model": model,
            "prompt": prompt,
            "stream": False,
            "options": {"num_predict": max_tokens},
        }).encode("utf-8")
        req = urllib.request.Request(
            f"{base.rstrip('/')}/api/generate",
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=180) as resp:
                return json.loads(resp.read())["response"]
        except Exception as e:
            return f"[Ollama error: {e}]"
