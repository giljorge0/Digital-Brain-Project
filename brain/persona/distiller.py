"""
Persona Distiller
-----------------
The "Makemore" layer of the digital brain.

Karpathy's makemore learns the statistical DNA of text and generates more of it.
We can't retrain model weights on a personal corpus in a CLI tool, but we can
extract a rich, structured persona profile from your notes that:

  1. Describes your topical fingerprint (what you write about and how often)
  2. Maps your intellectual stance on recurring topics
  3. Captures stylistic markers (sentence rhythm, vocabulary richness, phrasing)
  4. Identifies your intellectual lineage (who you cite / engage with)
  5. Extracts your argument patterns (how you build and structure arguments)

This profile is then used by PersonaGenerator to answer questions, expand notes,
and generate text that sounds like you.

Usage:
    distiller = PersonaDistiller(store, cfg)
    profile   = distiller.build_profile()
    distiller.save_profile(profile)        # persists to data/persona.json

    # Later:
    profile = distiller.load_profile()
"""

import json
import logging
import math
import os
import re
import urllib.request
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

log = logging.getLogger(__name__)

PROFILE_PATH = "data/persona.json"


class PersonaDistiller:
    """
    Extracts a structured persona profile from the note corpus.

    Parameters
    ----------
    store : Store
    cfg   : dict  (llm_backend, anthropic_api_key, etc.)
    """

    def __init__(self, store, cfg: dict):
        self.store = store
        self.cfg = cfg

    # ── Main entry ───────────────────────────────────────────────────────────

    def build_profile(self) -> dict:
        """
        Build the full persona profile. Runs all sub-extractors and asks the
        LLM to synthesize a high-level self-description.
        Returns a rich dict ready to save/use.
        """
        log.info("[persona] Building persona profile from corpus...")

        notes = self.store.get_all_notes()
        if not notes:
            log.warning("[persona] No notes in store. Profile will be empty.")
            return {}

        profile = {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "corpus_size": {
                "note_count": len(notes),
                "total_words": sum(n.word_count() for n in notes),
            },
            "topical_fingerprint": self._topical_fingerprint(notes),
            "stylistic_markers":   self._stylistic_markers(notes),
            "intellectual_lineage": self._intellectual_lineage(notes),
            "argument_patterns":   self._argument_patterns(notes),
            "temporal_arc":        self._temporal_arc(notes),
            "llm_self_description": None,  # filled below
            "stance_map":          {},     # filled below
        }

        # LLM calls — enrichment pass
        profile["llm_self_description"] = self._llm_synthesize_identity(notes, profile)
        profile["stance_map"] = self._llm_extract_stances(notes)

        log.info("[persona] Profile built.")
        return profile

    # ── Sub-extractors ───────────────────────────────────────────────────────

    def _topical_fingerprint(self, notes: list) -> dict:
        """Weighted tag and concept distribution."""
        tag_counts: Counter = Counter()
        concept_counts: Counter = Counter()

        stopwords = {
            "the", "a", "an", "and", "or", "but", "in", "on", "at", "to",
            "for", "of", "with", "by", "from", "is", "was", "are", "were",
            "it", "this", "that", "i", "my", "me", "we", "you", "he", "she",
        }

        for note in notes:
            for tag in note.tags:
                tag_counts[tag] += 1
            words = re.findall(r'\b[a-z]{4,}\b', note.content.lower())
            for w in words:
                if w not in stopwords:
                    concept_counts[w] += 1

        return {
            "top_tags": dict(tag_counts.most_common(30)),
            "top_concepts": dict(concept_counts.most_common(50)),
            "tag_diversity": len(tag_counts),
        }

    def _stylistic_markers(self, notes: list) -> dict:
        """Sentence-level stylistic fingerprint."""
        all_sentences = []
        word_lengths  = []
        sentence_lengths = []
        punctuation_styles = Counter()

        for note in notes:
            # Split into sentences naively
            sentences = re.split(r'(?<=[.!?])\s+', note.content.strip())
            sentences = [s.strip() for s in sentences if len(s.split()) > 3]
            all_sentences.extend(sentences[:5])  # sample first 5 per note

            words = note.content.split()
            word_lengths.extend([len(w) for w in words if w.isalpha()])
            sentence_lengths.extend([len(s.split()) for s in sentences])

            # Punctuation style
            if re.search(r'[;:]', note.content):
                punctuation_styles["uses_semicolons"] += 1
            if re.search(r'—', note.content):
                punctuation_styles["uses_em_dash"] += 1
            if re.search(r'\(.*?\)', note.content):
                punctuation_styles["uses_parentheticals"] += 1

        avg_word_len  = (sum(word_lengths) / len(word_lengths)) if word_lengths else 0
        avg_sent_len  = (sum(sentence_lengths) / len(sentence_lengths)) if sentence_lengths else 0

        # Vocabulary richness (type/token ratio on a sample)
        sample_text = " ".join(n.content[:500] for n in notes[:100])
        tokens      = sample_text.lower().split()
        ttr         = len(set(tokens)) / max(len(tokens), 1)

        return {
            "avg_word_length":     round(avg_word_len, 2),
            "avg_sentence_length": round(avg_sent_len, 2),
            "vocabulary_richness": round(ttr, 3),
            "punctuation_style":   dict(punctuation_styles.most_common()),
            "sample_sentences":    all_sentences[:20],
        }

    def _intellectual_lineage(self, notes: list) -> dict:
        """
        Detect thinkers, authors, and intellectual traditions in the corpus
        by looking for capitalized proper-noun pairs (heuristic NER).
        """
        name_re    = re.compile(r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)\b')
        name_counts: Counter = Counter()

        for note in notes:
            for match in name_re.finditer(note.content):
                name = match.group(1)
                # Filter out common false positives
                if name not in {"The The", "New York", "United States", "In The"}:
                    name_counts[name] += 1

        return {
            "cited_figures": dict(name_counts.most_common(40)),
        }

    def _argument_patterns(self, notes: list) -> dict:
        """Detect recurring argument structures from linguistic patterns."""
        patterns = {
            "claims_therefore":     0,
            "conditional_if_then":  0,
            "contrast_however":     0,
            "evidence_because":     0,
            "question_driven":      0,
            "list_first_second":    0,
        }

        for note in notes:
            t = note.content.lower()
            if "therefore" in t or "thus," in t or "hence" in t:
                patterns["claims_therefore"] += 1
            if re.search(r'\bif\b.+\bthen\b', t):
                patterns["conditional_if_then"] += 1
            if "however," in t or "nevertheless" in t or "on the other hand" in t:
                patterns["contrast_however"] += 1
            if "because " in t or "since " in t or "given that" in t:
                patterns["evidence_because"] += 1
            if note.title.endswith("?") or t.count("?") > 2:
                patterns["question_driven"] += 1
            if re.search(r'\bfirst,?\b', t) and re.search(r'\bsecond,?\b', t):
                patterns["list_first_second"] += 1

        total = len(notes) or 1
        return {k: round(v / total, 3) for k, v in patterns.items()}

    def _temporal_arc(self, notes: list) -> dict:
        """How has writing volume and topic focus evolved over time?"""
        dated = [n for n in notes if n.date]
        if not dated:
            return {}

        dated.sort(key=lambda n: n.date)
        by_year: dict = {}
        for n in dated:
            yr = str(n.date.year)
            by_year.setdefault(yr, {"count": 0, "words": 0, "tags": Counter()})
            by_year[yr]["count"] += 1
            by_year[yr]["words"] += n.word_count()
            for tag in n.tags:
                by_year[yr]["tags"][tag] += 1

        # Simplify: top tag per year
        arc = {}
        for yr, data in by_year.items():
            top_tag = data["tags"].most_common(1)
            arc[yr] = {
                "note_count": data["count"],
                "total_words": data["words"],
                "dominant_topic": top_tag[0][0] if top_tag else "—",
            }

        return arc

    # ── LLM enrichment ───────────────────────────────────────────────────────

    def _llm_synthesize_identity(self, notes: list, profile: dict) -> str:
        """
        Ask the LLM to write a 200-word self-description based on the extracted
        profile stats. This is the "Who are you as a thinker?" summary.
        """
        top_tags = list(profile["topical_fingerprint"]["top_tags"].keys())[:15]
        top_concepts = list(profile["topical_fingerprint"]["top_concepts"].keys())[:20]
        cited = list(profile["intellectual_lineage"]["cited_figures"].keys())[:15]
        sample = [n.title for n in notes[:30]]

        prompt = f"""You are analyzing a personal knowledge corpus to describe the intellectual 
identity of the person who wrote it.

Corpus statistics:
- {profile['corpus_size']['note_count']} notes, {profile['corpus_size']['total_words']:,} words
- Most common tags: {', '.join(top_tags)}
- Most frequent concepts: {', '.join(top_concepts)}
- Figures mentioned most: {', '.join(cited) if cited else 'none detected'}
- Sample note titles: {'; '.join(sample[:20])}

Write a 200-word intellectual profile of this person in second person ("You are someone who...").
Focus on: core intellectual interests, how they think, what traditions they engage with,
what questions drive them. Be specific, not generic."""

        try:
            return self._llm_call(prompt, max_tokens=400)
        except Exception as e:
            log.warning(f"[persona] LLM identity synthesis failed: {e}")
            return f"Corpus of {profile['corpus_size']['note_count']} notes on: {', '.join(top_tags[:5])}"

    def _llm_extract_stances(self, notes: list) -> dict:
        """
        For the 10 most central topics (by tag frequency), ask the LLM to
        identify your stated or implied stance.
        Returns {"topic": "stance description"}.
        """
        all_tags: Counter = Counter()
        for n in notes:
            for t in n.tags:
                all_tags[t] += 1

        top_topics = [t for t, _ in all_tags.most_common(10)]

        stances = {}
        for topic in top_topics:
            topic_notes = [n for n in notes if topic in n.tags][:5]
            if not topic_notes:
                continue

            context = "\n\n---\n".join(
                f"[{n.title}]\n{n.short_content(400)}" for n in topic_notes
            )

            prompt = f"""Based on these notes about "{topic}", describe the author's
intellectual stance on this topic in 1-2 sentences. Be specific about their position,
not just what the topic is.

Notes:
{context}

Response format: just the stance description, no preamble."""

            try:
                stance = self._llm_call(prompt, max_tokens=150)
                stances[topic] = stance.strip()
            except Exception as e:
                log.debug(f"[persona] Stance extraction failed for '{topic}': {e}")

        return stances

    # ── LLM backend ──────────────────────────────────────────────────────────

    def _llm_call(self, prompt: str, max_tokens: int = 512) -> str:
        backend = self.cfg.get("llm_backend", "claude")
        if backend == "ollama":
            return self._ollama_call(prompt)
        return self._claude_call(prompt, max_tokens)

    def _claude_call(self, prompt: str, max_tokens: int = 512) -> str:
        api_key = self.cfg.get("anthropic_api_key") or os.environ.get("ANTHROPIC_API_KEY", "")
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
        with urllib.request.urlopen(req, timeout=60) as resp:
            data = json.loads(resp.read())
            return data["content"][0]["text"]

    def _ollama_call(self, prompt: str) -> str:
        base = self.cfg.get("ollama_base_url", "http://localhost:11434").rstrip("/")
        model = self.cfg.get("ollama_model", "mistral")
        payload = json.dumps({"model": model, "prompt": prompt, "stream": False}).encode()
        req = urllib.request.Request(
            f"{base}/api/generate", data=payload,
            headers={"Content-Type": "application/json"}, method="POST",
        )
        with urllib.request.urlopen(req, timeout=120) as resp:
            return json.loads(resp.read())["response"]

    # ── Persistence ──────────────────────────────────────────────────────────

    def save_profile(self, profile: dict, path: str = PROFILE_PATH):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(profile, f, indent=2, default=str)
        log.info(f"[persona] Profile saved to {path}")

    def load_profile(self, path: str = PROFILE_PATH) -> Optional[dict]:
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except FileNotFoundError:
            log.warning(f"[persona] No profile found at {path}. Run 'python main.py persona build' first.")
            return None
