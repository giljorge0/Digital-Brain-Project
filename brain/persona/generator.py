"""
Persona Generator
-----------------
Generates text "as you" using the persona profile built by PersonaDistiller.

This is the generation half of the Makemore idea: given your corpus's
statistical and semantic DNA, produce new writing that extends your thinking.

Three generation modes:

  expand(note_id)       — Write more on a note's topic, in your voice,
                          drawing on related notes for context.

  respond(question)     — Answer a question as if you wrote it. Uses your
                          stance map and relevant notes as grounding.

  makemore(seed, n)     — Given a seed idea or title, generate n related
                          note ideas you might write next, ranked by how
                          well they fit your intellectual profile.

  synthesize(topic)     — Write a new synthesis note on a topic by combining
                          everything you've written about it. The "wiki page"
                          version of your thinking.

Usage:
    profile   = distiller.load_profile()
    generator = PersonaGenerator(store, embedder, profile, cfg)

    # Expand an existing note
    new_text = generator.expand("note-id-here")

    # Answer a question in your voice
    answer = generator.respond("What do you think about attention mechanisms?")

    # Generate 5 note ideas from a seed
    ideas = generator.makemore("phenomenology of perception", n=5)
"""

import json
import logging
import os
import urllib.request
from typing import Optional

log = logging.getLogger(__name__)


class PersonaGenerator:
    """
    Generates text conditioned on the user's persona profile.

    Parameters
    ----------
    store    : Store
    embedder : EmbeddingProvider
    profile  : dict  (from PersonaDistiller.load_profile())
    cfg      : dict
    """

    def __init__(self, store, embedder, profile: dict, cfg: dict):
        self.store    = store
        self.embedder = embedder
        self.profile  = profile
        self.cfg      = cfg

    # ── Core generation methods ──────────────────────────────────────────────

    def expand(self, note_id: str, target_words: int = 400) -> str:
        """
        Write a longer version of a note, staying in the author's voice and
        pulling in related notes as supporting material.

        Returns the generated expansion text.
        """
        note = self.store.get_note(note_id)
        if not note:
            return f"[expand] Note not found: {note_id}"

        # Find related notes via graph edges
        edges = self.store.get_edges(note_id=note_id)
        neighbor_ids = set()
        for e in edges:
            neighbor_ids.add(e["target"] if e["source"] == note_id else e["source"])

        related = []
        for nid in list(neighbor_ids)[:6]:
            n = self.store.get_note(nid)
            if n:
                related.append(n)

        context_blocks = "\n\n---\n".join(
            f"[{n.title}]\n{n.short_content(300)}" for n in related
        )

        persona_summary = self.profile.get("llm_self_description", "")
        style_note = self._style_instruction()

        stance = ""
        for tag in note.tags:
            if tag in self.profile.get("stance_map", {}):
                stance = f"Your stated position on '{tag}': {self.profile['stance_map'][tag]}\n"
                break

        prompt = f"""You are ghostwriting for a specific person. Here is their intellectual profile:

{persona_summary}

{style_note}

{stance}

Your task: Expand the following note to approximately {target_words} words.
Do NOT change the author's position or introduce new stances. Stay inside their
established thinking. Draw on the related notes below for supporting material.

NOTE TO EXPAND:
Title: {note.title}
Content:
{note.content}

RELATED NOTES FOR CONTEXT:
{context_blocks if context_blocks else "(no related notes found)"}

Write the expanded note now. Output only the note body, no preamble."""

        return self._llm_call(prompt, max_tokens=target_words * 2)

    def respond(self, question: str) -> str:
        """
        Answer a question as if the author wrote it.
        Grounds the answer in relevant notes + the persona's stance map.
        """
        # Semantic search for relevant notes
        try:
            q_vec = self.embedder.embed_one(question)
            from brain.memory.embeddings import search_by_embedding
            hits = search_by_embedding(self.store, q_vec, top_k=6)
            notes = [self.store.get_note(nid) for nid, _ in hits if self.store.get_note(nid)]
        except Exception:
            notes = self.store.search_notes(question, limit=6)

        context = "\n\n---\n".join(
            f"[{n.title}]\n{n.short_content(400)}" for n in notes
        )

        persona_summary = self.profile.get("llm_self_description", "")
        style_note      = self._style_instruction()
        stance_map      = self.profile.get("stance_map", {})
        relevant_stances = "\n".join(
            f"- On '{topic}': {stance}"
            for topic, stance in stance_map.items()
            if any(word in question.lower() for word in topic.lower().split())
        )

        prompt = f"""You are ghostwriting for a specific person. Here is their intellectual profile:

{persona_summary}

{style_note}

{f"Relevant known stances:{chr(10)}{relevant_stances}" if relevant_stances else ""}

Answer the following question AS THIS PERSON would write it — using their voice,
their concepts, their typical way of constructing an argument. Base your answer on
the notes provided below. Cite note titles where relevant.

QUESTION: {question}

RELEVANT NOTES FROM THEIR CORPUS:
{context if context else "(no directly relevant notes found — answer from general profile)"}

Write the answer now. Output only the answer, no preamble."""

        return self._llm_call(prompt, max_tokens=800)

    def makemore(self, seed: str, n: int = 5) -> list:
        """
        Given a seed topic or phrase, generate n note ideas the author
        would likely want to write next. Returns a list of dicts:
            [{"title": str, "premise": str, "fit_score": str}]
        """
        top_concepts  = list(self.profile.get("topical_fingerprint", {}).get("top_concepts", {}).keys())[:20]
        top_tags      = list(self.profile.get("topical_fingerprint", {}).get("top_tags", {}).keys())[:15]
        self_desc     = self.profile.get("llm_self_description", "")
        gap_topics    = []  # Could pull from GapAgent if available

        prompt = f"""You are helping a specific intellectual generate their next {n} note ideas.

Their profile:
{self_desc}

Their core topics: {', '.join(top_tags)}
Their core concepts: {', '.join(top_concepts[:15])}

Seed idea: "{seed}"

Generate {n} note ideas they would actually want to write, given:
1. The seed topic
2. Their established intellectual interests
3. Logical extensions of their current thinking
4. Questions they would find natural to explore next

Respond ONLY in valid JSON, no explanation or markdown fences:
{{
  "ideas": [
    {{
      "title": "note title as they would write it",
      "premise": "one-sentence description of the central claim or question",
      "why_fits": "why this fits their intellectual profile"
    }}
  ]
}}"""

        raw = self._llm_call(prompt, max_tokens=1000)
        try:
            data = json.loads(raw.strip().lstrip("```json").rstrip("```").strip())
            return data.get("ideas", [])
        except Exception as e:
            log.warning(f"[persona] makemore JSON parse failed: {e}")
            return [{"title": raw[:100], "premise": "Parse failed", "why_fits": ""}]

    def synthesize(self, topic: str) -> str:
        """
        Synthesize a new wiki-style note on a topic from everything in the corpus.
        Unlike expand(), this creates a new note, not an expansion of an existing one.
        """
        tag_notes  = self.store.get_notes_by_tag(topic)
        kw_notes   = self.store.search_notes(topic, limit=10)

        all_ids = {n.id: n for n in tag_notes}
        all_ids.update({n.id: n for n in kw_notes})
        notes   = list(all_ids.values())[:10]

        if not notes:
            return f"[synthesize] No notes found about '{topic}'."

        context = "\n\n---\n".join(
            f"[{n.title}]\n{n.short_content(500)}" for n in notes
        )
        persona_summary = self.profile.get("llm_self_description", "")
        style_note      = self._style_instruction()

        prompt = f"""You are ghostwriting for a specific person. Their intellectual profile:

{persona_summary}

{style_note}

Write a synthesis note titled: "{topic.title()}"

This note should:
- Be written entirely in the author's voice
- Synthesize all the relevant material from their corpus below
- State their actual position clearly
- Reference related ideas by name
- End with open questions they would want to pursue next

SOURCE NOTES:
{context}

Write the synthesis note now. Format it as a proper note: title, then body."""

        return self._llm_call(prompt, max_tokens=1200)

    # ── Style helper ─────────────────────────────────────────────────────────

    def _style_instruction(self) -> str:
        """Build a concise style instruction from the stylistic markers."""
        markers = self.profile.get("stylistic_markers", {})
        if not markers:
            return ""

        avg_sent = markers.get("avg_sentence_length", 0)
        voc      = markers.get("vocabulary_richness", 0)
        punct    = markers.get("punctuation_style", {})

        parts = []
        if avg_sent > 20:
            parts.append("writes in long, complex sentences")
        elif avg_sent < 12:
            parts.append("writes in short, punchy sentences")
        else:
            parts.append("writes in medium-length sentences")

        if voc > 0.6:
            parts.append("uses rich, varied vocabulary")
        elif voc < 0.4:
            parts.append("uses a consistent, focused vocabulary")

        if punct.get("uses_em_dash", 0) > 5:
            parts.append("frequently uses em-dashes")
        if punct.get("uses_parentheticals", 0) > 5:
            parts.append("frequently uses parentheticals")

        if not parts:
            return ""

        return f"Style guide for this author: they {', '.join(parts)}. Match this style."

    # ── LLM backend ──────────────────────────────────────────────────────────

    def _llm_call(self, prompt: str, max_tokens: int = 800) -> str:
        backend = self.cfg.get("llm_backend", "claude")
        if backend == "ollama":
            return self._ollama_call(prompt)
        return self._claude_call(prompt, max_tokens)

    def _claude_call(self, prompt: str, max_tokens: int = 800) -> str:
        api_key = self.cfg.get("anthropic_api_key") or os.environ.get("ANTHROPIC_API_KEY", "")
        model   = self.cfg.get("claude_model", "claude-haiku-4-5-20251001")
        payload = json.dumps({
            "model": model,
            "max_tokens": max_tokens,
            "messages": [{"role": "user", "content": prompt}],
        }).encode()
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
        with urllib.request.urlopen(req, timeout=90) as resp:
            return json.loads(resp.read())["content"][0]["text"]

    def _ollama_call(self, prompt: str) -> str:
        base  = self.cfg.get("ollama_base_url", "http://localhost:11434").rstrip("/")
        model = self.cfg.get("ollama_model", "mistral")
        payload = json.dumps({"model": model, "prompt": prompt, "stream": False}).encode()
        req = urllib.request.Request(
            f"{base}/api/generate", data=payload,
            headers={"Content-Type": "application/json"}, method="POST",
        )
        with urllib.request.urlopen(req, timeout=600) as resp:
            return json.loads(resp.read())["response"]
