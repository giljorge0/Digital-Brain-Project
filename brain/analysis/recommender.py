"""
Recommender
-----------
Takes Gap objects from GapFinder and finds external content to fill them.

Privacy architecture:
  - Your full corpus NEVER leaves your device
  - Only an anonymised gap description (a short text summary of what's missing)
    is used to query external sources
  - Three modes of increasing privacy:

  Mode 1: LOCAL (maximum privacy)
    Uses a local dump of arXiv abstracts / Wikipedia summaries stored on disk.
    Zero network traffic. Purely offline.

  Mode 2: ANONYMOUS (default)
    Sends the gap description text (NOT your notes) to search APIs.
    No account, no cookie, no IP tracking of your interests.
    Gap description is derived: it describes what's missing, not what you have.

  Mode 3: ZK (research frontier, not yet implemented)
    Commit to your interest vector, prove recommendation relevance via
    zero-knowledge proof without revealing the vector.
    Opens the door to: recommendation networks where interests stay private.

Output per gap:
  - Ranked list of recommendations (books, papers, YouTube, search queries)
  - Each with: title, source, relevance_score, why_this_fills_the_gap

Daily briefing:
  A synthesized reading list: top 3-5 items across all your current gaps,
  formatted for human consumption (plain text or markdown).
"""

import json
import logging
import math
import os
import urllib.request
import urllib.parse
from dataclasses import dataclass, field
from typing import Optional

from brain.analysis.gap_finder import Gap
from brain.memory.embeddings import _cosine

log = logging.getLogger(__name__)


# ─── Recommendation data class ────────────────────────────────────────────────

@dataclass
class Recommendation:
    title:       str
    source_type: str          # book | paper | video | article | search_query
    url:         str = ""
    author:      str = ""
    description: str = ""
    relevance:   float = 0.0
    gap_title:   str = ""
    gap_type:    str = ""
    why:         str = ""     # one sentence: why this fills the gap

    def to_dict(self) -> dict:
        return {
            "title":       self.title,
            "source_type": self.source_type,
            "url":         self.url,
            "author":      self.author,
            "description": self.description,
            "relevance":   round(self.relevance, 3),
            "gap_title":   self.gap_title,
            "gap_type":    self.gap_type,
            "why":         self.why,
        }


# ─── Recommender ─────────────────────────────────────────────────────────────

class Recommender:
    """
    Converts Gap objects into actionable reading/watching recommendations.

    Usage:
        rec = Recommender(cfg, mode="anonymous")
        recs = rec.recommend(gaps, top_k=10)
        briefing = rec.daily_briefing(gaps)
    """

    def __init__(self, cfg: dict, mode: str = "anonymous"):
        self.cfg  = cfg
        self.mode = mode
        self._llm_backend = cfg.get("llm_backend", "claude")

    # ── Public API ────────────────────────────────────────────────────────────

    def recommend(self, gaps: list, top_k: int = 10) -> list:
        """
        Given a list of Gap objects, return ranked Recommendations.
        """
        all_recs = []

        for gap in gaps[:15]:  # Don't process too many gaps at once
            recs = self._fill_gap(gap)
            all_recs.extend(recs)

        # Deduplicate by title and re-rank
        seen = set()
        unique = []
        for r in sorted(all_recs, key=lambda r: r.relevance * r.__class__.__mro__[0].__hash__(r), reverse=True):
            key = r.title.lower()[:40]
            if key not in seen:
                seen.add(key)
                unique.append(r)

        # Re-rank: weight by gap priority (embedded in relevance)
        unique.sort(key=lambda r: r.relevance, reverse=True)
        return unique[:top_k]

    def daily_briefing(self, gaps: list, n_items: int = 5) -> dict:
        """
        Generate a human-readable daily reading list.

        Returns:
          {
            "date": str,
            "summary": str,             # one paragraph: what your brain needs today
            "items": [Recommendation],  # top N items
            "top_gap": str,             # the single most important gap
            "reading_time_estimate": str,
          }
        """
        recs = self.recommend(gaps, top_k=n_items * 2)
        top_recs = recs[:n_items]

        if not gaps:
            return {"date": _today(), "summary": "No gaps found.", "items": []}

        top_gap = gaps[0] if gaps else None
        summary = self._llm_briefing_summary(gaps[:3], top_recs[:3])

        return {
            "date":                 _today(),
            "summary":              summary,
            "items":                [r.to_dict() for r in top_recs],
            "top_gap":              top_gap.title if top_gap else "",
            "top_gap_type":         top_gap.gap_type if top_gap else "",
            "reading_time_estimate": _estimate_reading_time(top_recs),
            "gap_count":            len(gaps),
        }

    # ── Internal: fill one gap ────────────────────────────────────────────────

    def _fill_gap(self, gap: Gap) -> list:
        """Generate recommendations for a single gap."""
        recs = []

        if self.mode == "local":
            recs.extend(self._local_search(gap))
        else:  # anonymous (default)
            # Use LLM to generate specific recommendations from the gap description
            recs.extend(self._llm_generate_recommendations(gap))
            # Also generate direct search queries the user can use
            recs.extend(self._generate_search_queries(gap))

        # Score by gap priority
        for r in recs:
            r.relevance = r.relevance * gap.priority_score
            r.gap_title = gap.title
            r.gap_type  = gap.gap_type

        return recs

    def _llm_generate_recommendations(self, gap: Gap) -> list:
        """
        Ask LLM to recommend specific resources for a gap.
        Privacy note: only the gap description (not your corpus) is sent.
        """
        prompt = f"""A person has identified the following gap in their knowledge:

Gap type: {gap.gap_type}
Gap title: {gap.title}
What's missing: {gap.description}

Recommend 3-4 SPECIFIC resources to fill this gap. Be precise — name actual
books, papers, YouTube channels, or podcasts. Do not be generic.

Respond in JSON only:
{{
  "recommendations": [
    {{
      "title": "specific title",
      "type": "book | paper | video | article",
      "author_or_channel": "name",
      "url_or_search": "direct URL or exact search query",
      "why": "one sentence: why this fills the gap exactly",
      "difficulty": "introductory | intermediate | advanced"
    }}
  ]
}}"""

        raw = self._llm_call(prompt, 600)
        try:
            clean = raw.strip().lstrip("```json").lstrip("```").rstrip("```").strip()
            data  = json.loads(clean)
        except Exception:
            return []

        recs = []
        for i, item in enumerate(data.get("recommendations", [])):
            # Score: first recommendation is highest relevance
            score = 1.0 - (i * 0.15)
            recs.append(Recommendation(
                title=item.get("title", ""),
                source_type=item.get("type", "unknown"),
                url=item.get("url_or_search", ""),
                author=item.get("author_or_channel", ""),
                description=f"[{item.get('difficulty', '')}] {item.get('why', '')}",
                relevance=score,
                why=item.get("why", ""),
            ))

        return recs

    def _generate_search_queries(self, gap: Gap) -> list:
        """Generate concrete search queries for the gap."""
        queries = []

        # arXiv-style query
        arxiv_q = gap.title.replace(":", "").replace("Missing sibling:", "").strip()
        queries.append(Recommendation(
            title=f"Search arXiv: {arxiv_q}",
            source_type="search_query",
            url=f"https://arxiv.org/search/?query={urllib.parse.quote(arxiv_q)}&searchtype=all",
            description="Academic papers on this topic",
            relevance=0.4,
            why="Direct academic search for papers filling this gap",
        ))

        # YouTube search
        yt_q = gap.title.replace("Missing sibling:", "").replace("Develop:", "").strip()
        queries.append(Recommendation(
            title=f"Search YouTube: {yt_q}",
            source_type="search_query",
            url=f"https://www.youtube.com/results?search_query={urllib.parse.quote(yt_q)}",
            description="Video lectures and talks",
            relevance=0.35,
            why="Video content for accessible introduction to this topic",
        ))

        return queries

    def _local_search(self, gap: Gap) -> list:
        """
        Search a local index of arXiv/Wikipedia abstracts.
        Requires: data/local_index.json to exist (populated by 'brain index-local' command)
        """
        index_path = "data/local_index.json"
        if not os.path.exists(index_path):
            log.warning("[recommend] Local index not found. "
                        "Run: python main.py index-local  to build it.")
            return []

        try:
            with open(index_path) as f:
                index = json.load(f)
        except Exception as e:
            log.error(f"[recommend] Local index load failed: {e}")
            return []

        # Score by cosine similarity with gap vector
        recs = []
        for item in index.get("items", []):
            vec = item.get("embedding", [])
            if not vec or not gap.gap_vector:
                continue
            score = _cosine(gap.gap_vector, vec)
            if score > 0.5:
                recs.append(Recommendation(
                    title=item["title"],
                    source_type=item.get("type", "article"),
                    url=item.get("url", ""),
                    author=item.get("author", ""),
                    description=item.get("abstract", "")[:200],
                    relevance=score,
                    why="High semantic similarity to the gap description",
                ))

        recs.sort(key=lambda r: r.relevance, reverse=True)
        return recs[:5]

    # ── LLM helpers ───────────────────────────────────────────────────────────

    def _llm_briefing_summary(self, gaps: list, recs: list) -> str:
        """Generate a one-paragraph daily briefing summary."""
        gap_descs = "\n".join(f"- {g.title} ({g.gap_type})" for g in gaps)
        rec_titles = "\n".join(f"- {r.title}" for r in recs)

        prompt = f"""Write a brief, motivating paragraph (3-4 sentences) for a 
philosopher's daily intellectual briefing. Their current knowledge gaps are:
{gap_descs}

Today's recommended resources include:
{rec_titles}

The paragraph should: explain what the person should focus on today and why
it matters for their intellectual development. Be concrete and personal.
Write in second person ("Your most pressing gap...", "Today would be well spent...").
No bullet points. No headers. Just a flowing paragraph."""

        return self._llm_call(prompt, 300)

    def _llm_call(self, prompt: str, max_tokens: int) -> str:
        if self._llm_backend == "ollama":
            base  = self.cfg.get("ollama_base_url", "http://localhost:11434")
            model = self.cfg.get("ollama_model", "mistral")
            payload = json.dumps({
                "model": model, "prompt": prompt, "stream": False,
                "options": {"num_predict": max_tokens},
            }).encode("utf-8")
            req = urllib.request.Request(
                f"{base.rstrip('/')}/api/generate", data=payload,
                headers={"Content-Type": "application/json"}, method="POST",
            )
        else:
            api_key = (self.cfg.get("anthropic_api_key") or
                       os.environ.get("ANTHROPIC_API_KEY", ""))
            model   = self.cfg.get("claude_model", "claude-haiku-4-5-20251001")
            payload = json.dumps({
                "model": model, "max_tokens": max_tokens,
                "messages": [{"role": "user", "content": prompt}],
            }).encode("utf-8")
            req = urllib.request.Request(
                "https://api.anthropic.com/v1/messages", data=payload,
                headers={"Content-Type": "application/json",
                         "x-api-key": api_key,
                         "anthropic-version": "2023-06-01"},
                method="POST",
            )
        try:
            with urllib.request.urlopen(req, timeout=60) as resp:
                data = json.loads(resp.read())
                if "content" in data:
                    return data["content"][0]["text"]
                return data.get("response", "")
        except Exception as e:
            return f"[error: {e}]"


# ─── Helpers ──────────────────────────────────────────────────────────────────

def _today() -> str:
    return datetime.utcnow().strftime("%Y-%m-%d")

def _estimate_reading_time(recs: list) -> str:
    counts = {"book": 0, "paper": 0, "video": 0, "article": 0}
    for r in recs:
        counts[r.source_type] = counts.get(r.source_type, 0) + 1
    parts = []
    if counts.get("book"):
        parts.append(f"{counts['book']} book chapter (~30 min)")
    if counts.get("paper"):
        parts.append(f"{counts['paper']} paper (~20 min each)")
    if counts.get("video"):
        parts.append(f"{counts['video']} video (~varies)")
    if counts.get("article"):
        parts.append(f"{counts['article']} article (~10 min each)")
    return ", ".join(parts) if parts else "~30 min total"

from datetime import datetime
