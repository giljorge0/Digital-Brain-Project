"""
Auto Wiki
---------
Generates and maintains living wiki pages for key concepts in the corpus.

For each high-centrality concept, generates a Wikipedia-style article
synthesising everything written about it. Pages are stored as Notes with
tag="wiki_page" and participate fully in the graph.

Versioning:
  Each regeneration stores the previous content in note.metadata["previous_content"]
  and increments metadata["version"]. The full version history is preserved in
  metadata["version_history"] as a list of {version, generated_at, content_excerpt}.

Diff-patch refresh mode:
  Instead of regenerating pages from scratch every run, the scheduler can
  run in diff mode: it identifies which source notes changed since the last
  wiki build and only regenerates pages whose source material has updated.
  This saves ~80% of LLM calls on a stable corpus.

Scheduled refresh:
  Use 'python main.py wiki schedule' to install a cron job.
  Or call WikiScheduler.run_if_due() in your nightly consolidation.

Usage:
    wiki = AutoWiki(store, graph_builder, persona_profile, cfg)

    wiki.update_all(top_n=20)                     # full regeneration
    wiki.update_all(top_n=20, diff_only=True)     # only stale pages
    wiki.generate_page("consciousness")            # single page
    wiki.export_markdown("wiki/")                 # export all to .md
    wiki.list_pages()                             # all wiki Note objects

    scheduler = WikiScheduler(wiki, store)
    scheduler.run_if_due(interval_hours=24)       # safe to call every run
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

WIKI_TAG      = "wiki_page"
SCHEDULE_FILE = "data/wiki_schedule.json"


class AutoWiki:
    """
    Generates and maintains living wiki pages for key concepts.

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

    def update_all(self, top_n: int = 20, diff_only: bool = False) -> list:
        """
        Generate or update wiki pages for the top_n most central concepts.

        diff_only=True  — only regenerate pages whose source notes have been
                          modified since the page was last generated.
                          Much cheaper on a stable corpus.

        Returns list of concept names that were updated.
        """
        concepts = self._identify_concepts(top_n)
        log.info(f"[wiki] {'Diff-patch' if diff_only else 'Full'} refresh "
                 f"for {len(concepts)} concepts...")

        updated = []
        for concept in concepts:
            try:
                if diff_only and not self._is_stale(concept):
                    log.debug(f"[wiki] Skipping '{concept}' — sources unchanged")
                    continue

                page = self.generate_page(concept)
                if page:
                    updated.append(concept)
                    log.info(f"[wiki] ✓ {concept} (v{page.metadata.get('version', 1)})")
            except Exception as e:
                log.warning(f"[wiki] ✗ Failed '{concept}': {e}")

        log.info(f"[wiki] Done. {len(updated)} pages updated.")
        return updated

    def generate_page(self, concept: str) -> Optional[Note]:
        """
        Generate (or update) a wiki page for a given concept.
        Stores it in the note store and returns the Note.
        """
        source_notes = self._gather_sources(concept)
        if not source_notes:
            log.debug(f"[wiki] No source notes for '{concept}'")
            return None

        existing    = self._get_existing_page(concept)
        old_content = existing.content if existing else None
        old_version = existing.metadata.get("version", 0) if existing else 0
        old_history = existing.metadata.get("version_history", []) if existing else []

        new_content = self._generate_content(concept, source_notes, old_content)
        related     = self._find_related_concepts(concept, source_notes)

        # Cross-link footer
        if related:
            link_line   = "**Related:** " + " · ".join(f"[[{c}]]" for c in related[:8])
            new_content = new_content + "\n\n---\n" + link_line

        # Build version history entry for old content
        if old_content:
            old_history.append({
                "version":      old_version,
                "generated_at": existing.metadata.get("generated_at", ""),
                "content_excerpt": old_content[:300],
            })
            # Keep only last 10 versions in metadata
            old_history = old_history[-10:]

        note_id      = Note.make_id(f"wiki_{concept}")
        source_ids   = [n.id for n in source_notes]
        latest_ts    = max(
            (n.date for n in source_notes if n.date),
            default=datetime.now(timezone.utc),
        )

        wiki_note = Note(
            id=note_id,
            title=f"Wiki: {concept.title()}",
            content=new_content,
            tags=[WIKI_TAG, concept.lower().replace(" ", "_")],
            source_file="auto_wiki",
            date=datetime.now(timezone.utc),
            links=source_ids,
            metadata={
                "type":                "wiki_page",
                "wiki_concept":        concept,
                "source_note_count":   len(source_notes),
                "source_note_ids":     source_ids,
                "source_latest_date":  latest_ts.isoformat() if latest_ts else None,
                "related_concepts":    related,
                "generated_at":        datetime.now(timezone.utc).isoformat(),
                "version":             old_version + 1,
                "version_history":     old_history,
            },
        )

        self.store.upsert_note(wiki_note)
        return wiki_note

    def list_pages(self) -> list:
        return self.store.get_notes_by_tag(WIKI_TAG)

    def get_page(self, concept: str) -> Optional[Note]:
        return self._get_existing_page(concept)

    def export_markdown(self, output_dir: str = "wiki/") -> int:
        """Export all wiki pages to .md files with YAML frontmatter."""
        pages = self.list_pages()
        out   = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)

        for page in pages:
            concept = page.metadata.get("wiki_concept", page.title)
            slug    = re.sub(r'[^\w\-]', '_', concept.lower())
            path    = out / f"{slug}.md"
            fm = (
                f"---\n"
                f"title: \"{page.title}\"\n"
                f"concept: \"{concept}\"\n"
                f"generated: \"{page.date.isoformat() if page.date else ''}\"\n"
                f"version: {page.metadata.get('version', 1)}\n"
                f"sources: {page.metadata.get('source_note_count', 0)} notes\n"
                f"related: [{', '.join(page.metadata.get('related_concepts', [])[:5])}]\n"
                f"---\n\n"
            )
            path.write_text(fm + page.content, encoding="utf-8")

        log.info(f"[wiki] Exported {len(pages)} pages to {output_dir}")
        return len(pages)

    def show_version_history(self, concept: str):
        """Print the version history of a wiki page."""
        page = self._get_existing_page(concept)
        if not page:
            print(f"No wiki page for '{concept}'.")
            return

        history = page.metadata.get("version_history", [])
        current = page.metadata.get("version", 1)

        print(f"\n{'='*56}")
        print(f"  VERSION HISTORY: {concept.title()}")
        print(f"{'='*56}")
        print(f"  Current: v{current} — {page.metadata.get('generated_at', '')[:10]}")
        if not history:
            print("  No previous versions archived.")
            return
        print(f"  {len(history)} archived version(s):\n")
        for entry in reversed(history):
            print(f"  v{entry['version']}  {entry.get('generated_at','')[:10]}")
            print(f"    {entry.get('content_excerpt','')[:120]}…")
            print()

    # ── Staleness detection ───────────────────────────────────────────────────

    def _is_stale(self, concept: str) -> bool:
        """
        Return True if the wiki page needs regeneration because source notes
        have been updated since it was last generated.
        Returns True if page doesn't exist yet.
        """
        page = self._get_existing_page(concept)
        if not page:
            return True

        page_ts_str = page.metadata.get("generated_at", "")
        try:
            page_ts = datetime.fromisoformat(page_ts_str)
            if page_ts.tzinfo is None:
                page_ts = page_ts.replace(tzinfo=timezone.utc)
        except Exception:
            return True

        source_notes = self._gather_sources(concept)
        for n in source_notes:
            if n.date and n.date > page_ts:
                return True

        # Also stale if source count changed significantly
        old_count = page.metadata.get("source_note_count", 0)
        if abs(len(source_notes) - old_count) >= 3:
            return True

        return False

    # ── Concept identification ────────────────────────────────────────────────

    def _identify_concepts(self, top_n: int) -> list:
        from collections import Counter
        notes = self.store.get_all_notes()

        tag_counts: Counter = Counter()
        for n in notes:
            for t in n.tags:
                if t not in (WIKI_TAG, "llm_chat", "web_clip", "pdf",
                             "document", "kindle", "highlights", "books",
                             "reading", "search_history", "youtube",
                             "watch_history", "synthesis", "generated"):
                    tag_counts[t] += 1

        top_tags = [t for t, _ in tag_counts.most_common(top_n * 2)]

        high_centrality = sorted(notes, key=lambda n: n.centrality, reverse=True)[:top_n]
        central = []
        for n in high_centrality:
            words = re.sub(r'[^\w\s]', '', n.title).strip().split()
            if len(words) >= 2:
                central.append(" ".join(words[:2]).lower())
            elif words:
                central.append(words[0].lower())

        seen, combined = set(), []
        for c in top_tags + central:
            c = c.strip().lower()
            if c and c not in seen and len(c) > 2:
                seen.add(c)
                combined.append(c)
            if len(combined) >= top_n:
                break

        return combined

    def _gather_sources(self, concept: str) -> list:
        tag_notes = self.store.get_notes_by_tag(concept.replace(" ", "_"))
        kw_notes  = self.store.search_notes(concept, limit=12)

        seen = {n.id: n for n in tag_notes}
        for n in kw_notes:
            if n.id not in seen:
                seen[n.id] = n

        sources = [n for n in seen.values() if WIKI_TAG not in n.tags]
        sources.sort(key=lambda n: n.centrality, reverse=True)
        return sources[:10]

    def _get_existing_page(self, concept: str) -> Optional[Note]:
        return self.store.get_note(Note.make_id(f"wiki_{concept}"))

    def _find_related_concepts(self, concept: str, sources: list) -> list:
        from collections import Counter
        related: Counter = Counter()
        for n in sources:
            for t in n.tags:
                if t != concept and t != WIKI_TAG and t not in (
                    "llm_chat", "web_clip", "pdf", "kindle", "highlights"
                ):
                    related[t] += 1
        return [t for t, _ in related.most_common(8)]

    # ── Content generation ────────────────────────────────────────────────────

    def _generate_content(self, concept: str, sources: list,
                           old_content: Optional[str] = None) -> str:
        context = "\n\n---\n".join(
            f"[{n.title}]\n{n.short_content(500)}" for n in sources
        )
        persona_desc = self.persona.get("llm_self_description", "")
        stance       = self.persona.get("stance_map", {}).get(concept, "")

        update_note = ""
        if old_content:
            update_note = f"""
An older version of this page exists. Update it to incorporate new material
from the source notes. Preserve accurate existing content. Flag any contradictions
with [CONTRADICTION: ...]. Add a brief [UPDATED: ...] note where content changed.

PREVIOUS VERSION (excerpt):
{old_content[:500]}
"""

        prompt = f"""You are maintaining a personal wiki for a specific intellectual.
{f"About the author: {persona_desc}" if persona_desc else ""}
{f"Their stance on '{concept}': {stance}" if stance else ""}
{update_note}

Write a Wikipedia-style article about: "{concept.title()}"

Structure:
# {concept.title()}

[Opening paragraph: definition as this person sees it — 2-3 sentences]

[Body: significance, sub-questions, key tensions — 200-350 words]

## Further Questions
- [3-5 open questions they would want to pursue]

SOURCE NOTES:
{context}

Write the wiki article only. No preamble."""

        try:
            return self._llm_call(prompt, max_tokens=1000)
        except Exception as e:
            log.error(f"[wiki] LLM failed for '{concept}': {e}")
            lines = [f"# {concept.title()}\n"]
            for n in sources[:5]:
                lines.append(f"## {n.title}\n{n.short_content(200)}\n")
            return "\n".join(lines)

    # ── LLM backend ──────────────────────────────────────────────────────────

    def _llm_call(self, prompt: str, max_tokens: int = 1000) -> str:
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
            with urllib.request.urlopen(req, timeout=90) as resp:
                return json.loads(resp.read())["content"][0]["text"]


# ── Scheduler ─────────────────────────────────────────────────────────────────

class WikiScheduler:
    """
    Lightweight scheduler for diff-patch wiki refresh.
    Tracks when the wiki was last updated and runs only when due.

    Usage:
        scheduler = WikiScheduler(wiki, store)
        scheduler.run_if_due(interval_hours=24)   # call from consolidation loop

    Install as a cron job:
        scheduler.install_cron(interval_hours=24)
    """

    def __init__(self, wiki: AutoWiki, store, schedule_file: str = SCHEDULE_FILE):
        self.wiki          = wiki
        self.store         = store
        self.schedule_file = schedule_file

    def run_if_due(self, interval_hours: int = 24,
                   top_n: int = 20) -> Optional[list]:
        """
        Run a diff-patch wiki refresh if the last run was more than
        interval_hours ago. Returns updated concept list or None if skipped.
        """
        if not self._is_due(interval_hours):
            log.info(f"[wiki-scheduler] Not due yet (interval={interval_hours}h). Skipping.")
            return None

        log.info("[wiki-scheduler] Running scheduled wiki refresh (diff-patch mode)...")
        updated = self.wiki.update_all(top_n=top_n, diff_only=True)
        self._record_run(len(updated))
        return updated

    def force_run(self, top_n: int = 20) -> list:
        """Force a full wiki refresh regardless of schedule."""
        log.info("[wiki-scheduler] Force-running full wiki refresh...")
        updated = self.wiki.update_all(top_n=top_n, diff_only=False)
        self._record_run(len(updated))
        return updated

    def install_cron(self, interval_hours: int = 24, python: str = "python"):
        """
        Print the crontab line to add for scheduled wiki refresh.
        (Does not modify crontab automatically — prints the line to add.)
        """
        main_py = Path("main.py").resolve()
        line = (f"0 */{interval_hours} * * *  "
                f"cd {main_py.parent} && {python} main.py wiki update --diff 2>&1 "
                f">> logs/wiki_refresh.log")
        print(f"\nAdd this line to crontab (crontab -e):\n\n  {line}\n")

    def _is_due(self, interval_hours: int) -> bool:
        record = self._load_record()
        if not record:
            return True
        last = record.get("last_run")
        if not last:
            return True
        try:
            last_dt = datetime.fromisoformat(last)
            if last_dt.tzinfo is None:
                last_dt = last_dt.replace(tzinfo=timezone.utc)
            elapsed = (datetime.now(timezone.utc) - last_dt).total_seconds() / 3600
            return elapsed >= interval_hours
        except Exception:
            return True

    def _record_run(self, n_updated: int):
        record = {
            "last_run":   datetime.now(timezone.utc).isoformat(),
            "n_updated":  n_updated,
        }
        Path(self.schedule_file).parent.mkdir(parents=True, exist_ok=True)
        with open(self.schedule_file, "w") as f:
            json.dump(record, f, indent=2)

    def _load_record(self) -> dict:
        try:
            with open(self.schedule_file) as f:
                return json.load(f)
        except FileNotFoundError:
            return {}
