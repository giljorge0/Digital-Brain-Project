"""
Persona Distiller
-----------------
Extracts a structured intellectual DNA profile from the note corpus.

Builds a profile covering:
  1. Topical fingerprint  — weighted tag and concept distribution
  2. Stylistic markers    — sentence rhythm, vocabulary, punctuation habits
  3. Intellectual lineage — thinkers and authors cited (heuristic NER)
  4. Argument patterns    — conditional, evidential, contrastive structures
  5. Temporal arc         — how dominant topics have shifted year by year
  6. Stance map           — LLM-extracted positions on your top 10 topics
  7. Self-description     — 200-word portrait synthesised by the LLM

Profile versioning:
  Every call to build_profile() saves the new profile and archives the old
  one to data/persona_history/persona_<timestamp>.json. This lets you track
  how your intellectual identity evolves over time.

  persona.json always contains the latest profile plus:
    - previous_version_at  : ISO timestamp of the previous build
    - version              : integer counter
    - drift                : dict showing which stances changed since last build

Usage:
    distiller = PersonaDistiller(store, cfg)
    profile   = distiller.build_profile()    # builds + saves
    profile   = distiller.load_profile()     # loads latest

    history   = distiller.load_history()     # list of all past profiles
    drift     = distiller.compute_drift()    # what changed vs. last build
    distiller.print_drift_report()           # human-readable change summary
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

PROFILE_PATH  = "data/persona.json"
HISTORY_DIR   = "data/persona_history"


class PersonaDistiller:
    """
    Extracts and versions a persona profile from the note corpus.

    Parameters
    ----------
    store : Store
    cfg   : dict  (llm_backend, anthropic_api_key, etc.)
    """

    def __init__(self, store, cfg: dict):
        self.store = store
        self.cfg   = cfg

    # ── Main entry ────────────────────────────────────────────────────────────

    def build_profile(self) -> dict:
        """
        Build the full persona profile and save it with versioning.
        Archives the previous profile to data/persona_history/ before overwriting.
        """
        log.info("[persona] Building persona profile from corpus...")

        notes = self.store.get_all_notes()
        if not notes:
            log.warning("[persona] No notes in store.")
            return {}

        profile = {
            "generated_at":    datetime.now(timezone.utc).isoformat(),
            "version":         1,
            "corpus_size": {
                "note_count":  len(notes),
                "total_words": sum(n.word_count() for n in notes),
            },
            "topical_fingerprint":  self._topical_fingerprint(notes),
            "stylistic_markers":    self._stylistic_markers(notes),
            "intellectual_lineage": self._intellectual_lineage(notes),
            "argument_patterns":    self._argument_patterns(notes),
            "temporal_arc":         self._temporal_arc(notes),
            "llm_self_description": None,
            "stance_map":           {},
            "previous_version_at":  None,
            "drift":                {},
        }

        # LLM enrichment
        profile["llm_self_description"] = self._llm_synthesize_identity(notes, profile)
        profile["stance_map"]           = self._llm_extract_stances(notes)

        # Version + archive existing profile before overwriting
        existing = self.load_profile()
        if existing:
            profile["version"]           = existing.get("version", 1) + 1
            profile["previous_version_at"] = existing.get("generated_at")
            profile["drift"]             = self._diff_stances(
                existing.get("stance_map", {}),
                profile["stance_map"],
            )
            self._archive(existing)
            log.info(f"[persona] Archived v{existing.get('version', 1)} → history/")

        self.save_profile(profile)
        log.info(f"[persona] Profile v{profile['version']} built and saved.")
        return profile

    # ── Sub-extractors ────────────────────────────────────────────────────────

    def _topical_fingerprint(self, notes: list) -> dict:
        tag_counts: Counter     = Counter()
        concept_counts: Counter = Counter()
        
        # Add this blacklist to ignore administrative metadata
        BLACKLIST = {"authored", "output", "input", "generated", "synthesis", "uncategorised", "external"}
        
        stopwords = {
            "the","a","an","and","or","but","in","on","at","to","for","of",
            "with","by","from","is","was","are","were","it","this","that",
            "i","my","me","we","you","he","she","they","have","has","not",
            "be","been","will","would","could","should","also","which","when",
        }
        for note in notes:
            for tag in note.tags:
                # Add this if-statement to filter out the metadata
                if tag.lower() not in BLACKLIST:
                    tag_counts[tag] += 1
            words = re.findall(r'\b[a-z]{4,}\b', note.content.lower())
            for w in words:
                if w not in stopwords:
                    concept_counts[w] += 1

        return {
            "top_tags":      dict(tag_counts.most_common(30)),
            "top_concepts":  dict(concept_counts.most_common(50)),
            "tag_diversity": len(tag_counts),
        }

    def _stylistic_markers(self, notes: list) -> dict:
        word_lengths, sentence_lengths = [], []
        sample_sentences = []
        punctuation_styles: Counter = Counter()

        for note in notes:
            sentences = re.split(r'(?<=[.!?])\s+', note.content.strip())
            sentences = [s.strip() for s in sentences if len(s.split()) > 3]
            sample_sentences.extend(sentences[:5])

            words = note.content.split()
            word_lengths.extend(len(w) for w in words if w.isalpha())
            sentence_lengths.extend(len(s.split()) for s in sentences)

            if re.search(r'[;:]', note.content):
                punctuation_styles["uses_semicolons"] += 1
            if re.search(r'—', note.content):
                punctuation_styles["uses_em_dash"] += 1
            if re.search(r'\(.*?\)', note.content):
                punctuation_styles["uses_parentheticals"] += 1

        avg_word = sum(word_lengths)     / max(len(word_lengths), 1)
        avg_sent = sum(sentence_lengths) / max(len(sentence_lengths), 1)

        sample = " ".join(n.content[:500] for n in notes[:100])
        tokens = sample.lower().split()
        ttr    = len(set(tokens)) / max(len(tokens), 1)

        return {
            "avg_word_length":     round(avg_word, 2),
            "avg_sentence_length": round(avg_sent, 2),
            "vocabulary_richness": round(ttr, 3),
            "punctuation_style":   dict(punctuation_styles.most_common()),
            "sample_sentences":    sample_sentences[:20],
        }

    def _intellectual_lineage(self, notes: list) -> dict:
        name_re     = re.compile(r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)\b')
        name_counts: Counter = Counter()
        noise = {"The The", "New York", "United States", "In The", "Of The"}

        for note in notes:
            for m in name_re.finditer(note.content):
                name = m.group(1)
                if name not in noise:
                    name_counts[name] += 1

        return {"cited_figures": dict(name_counts.most_common(40))}

    def _argument_patterns(self, notes: list) -> dict:
        patterns = {
            "claims_therefore":    0,
            "conditional_if_then": 0,
            "contrast_however":    0,
            "evidence_because":    0,
            "question_driven":     0,
            "list_first_second":   0,
        }
        for note in notes:
            t = note.content.lower()
            if re.search(r'\b(therefore|thus,|hence)\b', t):
                patterns["claims_therefore"] += 1
            if re.search(r'\bif\b.{1,60}\bthen\b', t):
                patterns["conditional_if_then"] += 1
            if re.search(r'\b(however,|nevertheless|on the other hand)\b', t):
                patterns["contrast_however"] += 1
            if re.search(r'\b(because |since |given that)\b', t):
                patterns["evidence_because"] += 1
            if note.title.endswith("?") or t.count("?") > 2:
                patterns["question_driven"] += 1
            if re.search(r'\bfirst,?\b', t) and re.search(r'\bsecond,?\b', t):
                patterns["list_first_second"] += 1

        total = max(len(notes), 1)
        return {k: round(v / total, 3) for k, v in patterns.items()}

    def _temporal_arc(self, notes: list) -> dict:
        dated = sorted([n for n in notes if n.date], key=lambda n: n.date)
        if not dated:
            return {}
        by_year: dict = {}
        for n in dated:
            yr = str(n.date.year)
            by_year.setdefault(yr, {"count": 0, "words": 0, "tags": Counter()})
            by_year[yr]["count"] += 1
            by_year[yr]["words"] += n.word_count()
            for tag in n.tags:
                by_year[yr]["tags"][tag] += 1

        arc = {}
        for yr, data in sorted(by_year.items()):
            top = data["tags"].most_common(1)
            arc[yr] = {
                "note_count":      data["count"],
                "total_words":     data["words"],
                "dominant_topic":  top[0][0] if top else "—",
            }
        return arc

    # ── LLM enrichment ───────────────────────────────────────────────────────

    def _llm_synthesize_identity(self, notes: list, profile: dict) -> str:
        top_tags     = list(profile["topical_fingerprint"]["top_tags"].keys())[:15]
        top_concepts = list(profile["topical_fingerprint"]["top_concepts"].keys())[:20]
        cited        = list(profile["intellectual_lineage"]["cited_figures"].keys())[:15]

        prompt = f"""You are analysing a personal knowledge corpus to describe the
intellectual identity of the person who wrote it.

Corpus statistics:
- {profile['corpus_size']['note_count']} notes, {profile['corpus_size']['total_words']:,} words
- Most common tags: {', '.join(top_tags)}
- Most frequent concepts: {', '.join(top_concepts[:15])}
- Figures mentioned most: {', '.join(cited) if cited else 'none detected'}
- Sample note titles: {'; '.join(n.title for n in notes[:20])}

Write a 200-word intellectual profile in second person ("You are someone who...").
Focus on: core interests, how they think, what traditions they engage with,
what questions drive them. Be specific, not generic."""

        try:
            return self._llm_call(prompt, max_tokens=400)
        except Exception as e:
            log.warning(f"[persona] LLM identity synthesis failed: {e}")
            return (f"Corpus of {profile['corpus_size']['note_count']} notes "
                    f"on: {', '.join(top_tags[:5])}")

    def _llm_extract_stances(self, notes: list) -> dict:
        all_tags: Counter = Counter()
        # Add the same blacklist here[cite: 5]
        BLACKLIST = {"authored", "output", "input", "generated", "synthesis", "uncategorised", "external"}

        for n in notes:
            for t in n.tags:
                # Filter the tags before counting[cite: 5]
                if t.lower() not in BLACKLIST:
                    all_tags[t] += 1

        top_topics = [t for t, _ in all_tags.most_common(10)]
        stances    = {}
        for topic in top_topics:
            topic_notes = [n for n in notes if topic in n.tags][:5]
            if not topic_notes:
                continue
            context = "\n\n---\n".join(
                f"[{n.title}]\n{n.short_content(400)}" for n in topic_notes
            )
            prompt = f"""Based on these notes about "{topic}", describe the author's
intellectual stance in 1-2 sentences. Be specific about their actual position.

Notes:
{context}

Just the stance description, no preamble."""
            try:
                stances[topic] = self._llm_call(prompt, max_tokens=150).strip()
            except Exception as e:
                log.debug(f"[persona] Stance failed for '{topic}': {e}")

        return stances

    # ── Versioning ────────────────────────────────────────────────────────────

    def _archive(self, old_profile: dict):
        """Save a copy of old_profile to data/persona_history/."""
        Path(HISTORY_DIR).mkdir(parents=True, exist_ok=True)
        ts   = (old_profile.get("generated_at", datetime.now(timezone.utc).isoformat())
                .replace(":", "-").replace("+", "Z")[:19])
        ver  = old_profile.get("version", 0)
        path = Path(HISTORY_DIR) / f"persona_v{ver}_{ts}.json"
        with open(path, "w", encoding="utf-8") as f:
            json.dump(old_profile, f, indent=2, default=str)

    def _diff_stances(self, old: dict, new: dict) -> dict:
        """
        Compute which stances changed between two profiles.
        Returns {"topic": {"old": str, "new": str}} for changed topics.
        """
        drift = {}
        all_topics = set(old) | set(new)
        for topic in all_topics:
            o = old.get(topic, "")
            n = new.get(topic, "")
            if o != n:
                drift[topic] = {"old": o, "new": n}
        return drift

    def load_history(self) -> list:
        """Return all archived profiles, sorted oldest first."""
        hist_dir = Path(HISTORY_DIR)
        if not hist_dir.exists():
            return []
        profiles = []
        for p in sorted(hist_dir.glob("persona_v*.json")):
            try:
                with open(p) as f:
                    profiles.append(json.load(f))
            except Exception:
                pass
        return profiles

    def compute_drift(self) -> dict:
        """
        Compare the current profile to the previous one.
        Returns a drift dict (or empty dict if only one version exists).
        """
        current = self.load_profile()
        if not current:
            return {}
        history = self.load_history()
        if not history:
            return {}
        previous = history[-1]
        return self._diff_stances(
            previous.get("stance_map", {}),
            current.get("stance_map",  {}),
        )

    def print_drift_report(self):
        """Print a human-readable summary of how your thinking has changed."""
        drift   = self.compute_drift()
        current = self.load_profile()
        if not current:
            print("No persona profile found. Run: python main.py persona build")
            return

        v    = current.get("version", 1)
        prev = current.get("previous_version_at", "—")
        now  = current.get("generated_at", "—")

        print(f"\n{'='*60}")
        print(f"  PERSONA DRIFT REPORT")
        print(f"  v{v-1} ({prev[:10]}) → v{v} ({now[:10]})")
        print(f"{'='*60}\n")

        if not drift:
            print("  No stance changes detected since last build.")
            return

        print(f"  {len(drift)} topic(s) show changed stances:\n")
        for topic, change in drift.items():
            print(f"  [{topic}]")
            if change["old"]:
                print(f"    Before: {change['old']}")
            else:
                print(f"    Before: (new topic)")
            if change["new"]:
                print(f"    After:  {change['new']}")
            else:
                print(f"    After:  (topic dropped)")
            print()

    # ── Persistence ───────────────────────────────────────────────────────────

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
            return None

    # ── LLM backend ──────────────────────────────────────────────────────────

    def _llm_call(self, prompt: str, max_tokens: int = 512) -> str:
        backend = self.cfg.get("llm_backend", "claude")
        if backend == "ollama":
            base  = self.cfg.get("ollama_base_url", "http://localhost:11434").rstrip("/")
            model = self.cfg.get("ollama_model", "mistral")
            payload = json.dumps({"model": model, "prompt": prompt, "stream": False}).encode()
            req = urllib.request.Request(
                f"{base}/api/generate", data=payload,
                headers={"Content-Type": "application/json"}, method="POST",
            )
            with urllib.request.urlopen(req, timeout=120) as resp:
                return json.loads(resp.read())["response"]
        else:
            api_key = self.cfg.get("anthropic_api_key") or os.environ.get("ANTHROPIC_API_KEY", "")
            model   = self.cfg.get("claude_model", "claude-haiku-4-5-20251001")
            payload = json.dumps({
                "model": model, "max_tokens": max_tokens,
                "messages": [{"role": "user", "content": prompt}],
            }).encode()
            req = urllib.request.Request(
                "https://api.anthropic.com/v1/messages", data=payload,
                headers={"Content-Type": "application/json",
                         "x-api-key": api_key,
                         "anthropic-version": "2023-06-01"},
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=60) as resp:
                return json.loads(resp.read())["content"][0]["text"]
