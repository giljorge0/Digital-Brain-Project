"""
Auto Wiki
---------
The "LLM Wiki" layer of the digital brain — inspired by Karpathy's llm.c wiki
and the broader idea of LLMs as living encyclopaedia maintainers.

For each high-centrality concept / cluster in your knowledge graph, this module
auto-generates and auto-maintains a Wikipedia-style article that synthesizes
everything you've written about that topic.

Key properties:
  • Living documents — re-generated or diff-patched on each consolidation run
  • Written in your voice — uses the persona profile for tone and stance
  • Cross-linked — every wiki page links to related wiki pages by concept name
  • Provenance-tracked — every claim is sourced to a specific note
  • Versioned — old versions stored so you can see how your thinking evolved

Storage: each wiki page is stored as a special Note in the Store with
  tag="wiki_page" and metadata={"wiki_concept": concept_name}

Usage:
    wiki = AutoWiki(store, graph_builder, persona_profile, cfg)

    # Generate/update pages for the top 20 concepts
    wiki.update_all(top_n=20)

    # Generate a single page
    page = wiki.generate_page("consciousness")

    # Export all pages to Markdown files
    wiki.export_markdown("wiki/")

    # List all existing wiki pages
    pages = wiki.list_pages()
"""

import json
import logging
import os
import re
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from brain.ingest.note import Note

log = logging.getLogger(__name__)

WIKI_TAG = "wiki_page"


class AutoWiki:
    """
    Generates and maintains living wiki pages for key concepts in the corpus.

    Parameters
    ----------
    store         : Store
    graph_builder : GraphBuilder
    persona       : dict  (from PersonaDistiller, can be None)
    cfg           : dict
    """

    def __init__(self, store, graph_builder, persona: Optional[dict], cfg: dict):
        self.store   = store
        self.builder = graph_builder
        self.persona = persona or {}
        self.cfg     = cfg

    # ── Public API ────────────────────────────────────────────────────────────

    def update_all(self, top_n: int = 20) -> list:
        """
        Generate or update wiki pages for the top_n most central concepts.
        Returns list of generated page titles.
        """
        concepts = self._identify_concepts(top_n)
        log.info(f"[wiki] Updating {len(concepts)} wiki pages...")

        pages = []
        for concept in concepts:
            try:
                page = self.generate_page(concept)
                if page:
                    pages.append(concept)
                    log.info(f"[wiki] ✓ {concept}")
            except Exception as e:
                log.warning(f"[wiki] ✗ Failed to generate page for '{concept}': {e}")

        log.info(f"[wiki] Done. {len(pages)} pages updated.")
        return pages

    def generate_page(self, concept: str) -> Optional[Note]:
        """
        Generate (or regenerate) a wiki page for a given concept.
        Stores it in the note store and returns the Note.
        """
        # Gather source notes
        source_notes = self._gather_sources(concept)
        if not source_notes:
            log.debug(f"[wiki] No source notes for '{concept}', skipping.")
            return None

        # Check for existing page (for versioning)
        existing = self._get_existing_page(concept)
        old_content = existing.content if existing else None

        # Generate new content
        content = self._generate_content(concept, source_notes, old_content)

        # Find related concepts for cross-linking
        related = self._find_related_concepts(concept, source_notes)

        # Add cross-links footer
        if related:
            link_line = "**Related concepts:** " + " · ".join(f"[[{c}]]" for c in related[:8])
            content = content + "\n\n---\n" + link_line

        # Build the wiki note
        note_id = Note.make_id(f"wiki_{concept}")
        source_ids = [n.id for n in source_notes]

        wiki_note = Note(
            id=note_id,
            title=f"Wiki: {concept.title()}",
            content=content,
            tags=[WIKI_TAG, concept.lower().replace(" ", "_")],
            source_file="auto_wiki",
            date=datetime.now(timezone.utc),
            links=source_ids,
            metadata={
                "type": "wiki_page",
                "wiki_concept": concept,
                "source_note_count": len(source_notes),
                "source_note_ids": source_ids,
                "related_concepts": related,
                "generated_at": datetime.now(timezone.utc).isoformat(),
                "version": (existing.metadata.get("version", 0) + 1) if existing else 1,
                "previous_content": old_content[:500] if old_content else None,
            },
        )

        self.store.upsert_note(wiki_note)
        return wiki_note

    def list_pages(self) -> list:
        """Return all existing wiki page notes."""
        return self.store.get_notes_by_tag(WIKI_TAG)

    def export_markdown(self, output_dir: str = "wiki/"):
        """Export all wiki pages to Markdown files."""
        pages = self.list_pages()
        out   = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)

        for page in pages:
            concept = page.metadata.get("wiki_concept", page.title)
            slug    = re.sub(r'[^\w\-]', '_', concept.lower())
            path    = out / f"{slug}.md"

            frontmatter = (
                f"---\n"
                f"title: \"{page.title}\"\n"
                f"concept: \"{concept}\"\n"
                f"generated: \"{page.date.isoformat() if page.date else ''}\"\n"
                f"version: {page.metadata.get('version', 1)}\n"
                f"sources: {page.metadata.get('source_note_count', 0)} notes\n"
                f"---\n\n"
            )

            path.write_text(frontmatter + page.content, encoding="utf-8")

        log.info(f"[wiki] Exported {len(pages)} pages to {output_dir}")
        return len(pages)

    def get_page(self, concept: str) -> Optional[Note]:
        """Retrieve the wiki page for a concept."""
        return self._get_existing_page(concept)

    # ── Concept identification ────────────────────────────────────────────────

    def _identify_concepts(self, top_n: int) -> list:
        """
        Identify the top_n most wiki-worthy concepts by combining:
        - High-centrality nodes (PageRank)
        - High-frequency tags
        - Cluster centroids
        """
        notes = self.store.get_all_notes()

        # 1. Top tags by frequency
        from collections import Counter
        tag_counts: Counter = Counter()
        for n in notes:
            for t in n.tags:
                if t not in (WIKI_TAG, "llm_chat", "web_clip", "pdf", "document"):
                    tag_counts[t] += 1

        top_tags = [t for t, _ in tag_counts.most_common(top_n * 2)]

        # 2. High-centrality note titles (not full content, just key concept words)
        high_centrality = sorted(notes, key=lambda n: n.centrality, reverse=True)[:top_n]
        central_concepts = []
        for n in high_centrality:
            # Use the first 2 content words of each title as a concept candidate
            words = re.sub(r'[^\w\s]', '', n.title).strip().split()
            if len(words) >= 2:
                central_concepts.append(" ".join(words[:2]).lower())
            elif words:
                central_concepts.append(words[0].lower())

        # Merge and deduplicate, prioritising tags
        seen = set()
        combined = []
        for c in top_tags + central_concepts:
            c_clean = c.strip().lower()
            if c_clean and c_clean not in seen and len(c_clean) > 2:
                seen.add(c_clean)
                combined.append(c_clean)
            if len(combined) >= top_n:
                break

        return combined

    def _gather_sources(self, concept: str) -> list:
        """Collect all notes relevant to a concept."""
        tag_notes  = self.store.get_notes_by_tag(concept.replace(" ", "_"))
        kw_notes   = self.store.search_notes(concept, limit=12)

        seen = {n.id: n for n in tag_notes}
        for n in kw_notes:
            if n.id not in seen:
                seen[n.id] = n

        # Exclude other wiki pages to avoid circular synthesis
        source = [n for n in seen.values() if WIKI_TAG not in n.tags]

        # Sort by centrality
        source.sort(key=lambda n: n.centrality, reverse=True)
        return source[:10]

    def _get_existing_page(self, concept: str) -> Optional[Note]:
        note_id = Note.make_id(f"wiki_{concept}")
        return self.store.get_note(note_id)

    def _find_related_concepts(self, concept: str, source_notes: list) -> list:
        """Find related concepts by looking at tags of source notes."""
        from collections import Counter
        related_tags: Counter = Counter()
        for n in source_notes:
            for t in n.tags:
                if (t != concept and t != WIKI_TAG
                        and t not in ("llm_chat", "web_clip", "pdf")):
                    related_tags[t] += 1
        return [t for t, _ in related_tags.most_common(8)]

    # ── Content generation ────────────────────────────────────────────────────

    def _generate_content(
        self,
        concept: str,
        sources: list,
        old_content: Optional[str] = None,
    ) -> str:
        """Call the LLM to generate wiki page content."""

        context = "\n\n---\n".join(
            f"[Note: {n.title}]\n{n.short_content(500)}" for n in sources
        )

        persona_desc = self.persona.get("llm_self_description", "")
        stance       = self.persona.get("stance_map", {}).get(concept, "")

        update_instruction = ""
        if old_content:
            update_instruction = f"""
An older version of this page exists (shown below). Update it to incorporate
any new information from the source notes. Preserve good existing content.
Flag any contradictions with [CONTRADICTION: ...].

PREVIOUS VERSION (excerpt):
{old_content[:600]}
"""

        prompt = f"""You are maintaining a personal wiki for a specific intellectual.

About the author:
{persona_desc if persona_desc else "(no persona profile available)"}

{f"Their stated stance on '{concept}': {stance}" if stance else ""}

{update_instruction}

Write a Wikipedia-style article about: "{concept.title()}"

This article should:
1. Open with a clear definition/framing of the concept AS THIS PERSON sees it
2. Explain the concept's significance to their work
3. Note key tensions, sub-questions, or open problems they've identified
4. Reference other concepts and thinkers that appear in their notes
5. End with a "Further questions" section listing 3-5 open questions they have

FORMAT:
# {concept.title()}

[body — 300-500 words, written in the author's voice]

## Further Questions
- ...

SOURCE NOTES TO SYNTHESIZE:
{context}

Write the wiki article now. Output only the article, no preamble."""

        try:
            return self._llm_call(prompt, max_tokens=1000)
        except Exception as e:
            log.error(f"[wiki] LLM generation failed for '{concept}': {e}")
            # Fallback: simple concatenation
            lines = [f"# {concept.title()}\n"]
            for n in sources[:5]:
                lines.append(f"## {n.title}\n{n.short_content(200)}\n")
            return "\n".join(lines)

    # ── LLM backend ──────────────────────────────────────────────────────────

    def _llm_call(self, prompt: str, max_tokens: int = 1000) -> str:
        backend = self.cfg.get("llm_backend", "claude")
        if backend == "ollama":
            return self._ollama_call(prompt)
        return self._claude_call(prompt, max_tokens)

    def _claude_call(self, prompt: str, max_tokens: int) -> str:
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
        with urllib.request.urlopen(req, timeout=120) as resp:
            return json.loads(resp.read())["response"]
