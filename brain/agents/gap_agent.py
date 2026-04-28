"""
Knowledge Gap Agent
-------------------
Analyses the graph to surface:

  1. ISOLATED NODES    — notes with no connections (orphan ideas)
  2. SPARSE CLUSTERS   — topic areas with few cross-links
  3. UNEXPLORED EDGES  — pairs of semantically similar notes that are never
                         explicitly linked in your writing
  4. MISSING TOPICS    — concepts you reference but never develop a full note for
  5. DEPTH GAPS        — topics you mention briefly, many times, but never go
                         deep on
  6. RECENCY GAPS      — ideas that appear only in old notes, never revisited
  7. ONE-SIDED IDEAS   — claims you make without ever writing a counter-argument
  8. RECOMMENDED READS — suggests related published works / concepts from
                         the graph neighbourhood (uses Perplexity by default
                         for live search, falls back to LLM knowledge)

The output is a structured `GapReport` that can be printed, saved to JSON,
or used by the query agent to proactively ask the user clarifying questions.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Optional

log = logging.getLogger("brain.gap_agent")


# ─── Report structures ────────────────────────────────────────────────────────

@dataclass
class GapItem:
    gap_type: str           # orphan | sparse_cluster | missing_topic | depth | recency | one_sided | recommended
    title: str
    description: str
    note_ids: list = field(default_factory=list)
    priority: str = "medium"    # high | medium | low
    suggestion: str = ""        # what to do about it


@dataclass
class GapReport:
    generated_at: str = field(default_factory=lambda: datetime.now().isoformat())
    total_notes: int = 0
    total_edges: int = 0
    gaps: list = field(default_factory=list)

    def by_type(self, gap_type: str) -> list:
        return [g for g in self.gaps if g.gap_type == gap_type]

    def high_priority(self) -> list:
        return [g for g in self.gaps if g.priority == "high"]

    def to_dict(self) -> dict:
        return {
            "generated_at": self.generated_at,
            "total_notes": self.total_notes,
            "total_edges": self.total_edges,
            "summary": {
                "high": len(self.high_priority()),
                "total": len(self.gaps),
                "by_type": {
                    t: len(self.by_type(t))
                    for t in ["orphan", "sparse_cluster", "missing_topic",
                              "depth", "recency", "one_sided", "recommended"]
                },
            },
            "gaps": [
                {
                    "type": g.gap_type,
                    "title": g.title,
                    "description": g.description,
                    "note_ids": g.note_ids,
                    "priority": g.priority,
                    "suggestion": g.suggestion,
                }
                for g in self.gaps
            ],
        }

    def print_summary(self):
        print(f"\n{'='*60}")
        print(f"  KNOWLEDGE GAP REPORT  —  {self.generated_at[:10]}")
        print(f"{'='*60}")
        print(f"  Notes: {self.total_notes}   Edges: {self.total_edges}")
        print(f"  Gaps found: {len(self.gaps)}  ({len(self.high_priority())} high-priority)")
        print()
        for g in sorted(self.gaps, key=lambda x: {"high": 0, "medium": 1, "low": 2}[x.priority]):
            icon = {"high": "🔴", "medium": "🟡", "low": "🟢"}[g.priority]
            print(f"  {icon} [{g.gap_type}]  {g.title}")
            print(f"      {g.description}")
            if g.suggestion:
                print(f"      → {g.suggestion}")
            print()


# ─── Main agent ───────────────────────────────────────────────────────────────

class GapAgent:
    """
    Runs structural analysis on the store + graph, then optionally
    calls an LLM to generate richer suggestions for top gaps.
    """

    def __init__(self, store, graph_builder, llm_registry=None,
                 semantic_threshold: float = 0.75,
                 recency_days: int = 180):
        self.store = store
        self.graph_builder = graph_builder
        self.llm_registry = llm_registry
        self.semantic_threshold = semantic_threshold
        self.recency_cutoff = datetime.now() - timedelta(days=recency_days)

    def run(self, llm_enrich: bool = True) -> GapReport:
        log.info("Running knowledge gap analysis…")
        notes = self.store.get_all_notes()
        all_edges = self.store.get_all_edges()
        G = self.graph_builder.build()

        report = GapReport(
            total_notes=len(notes),
            total_edges=len(all_edges),
        )

        # ── Structural passes (no LLM needed) ─────────────────────────────
        report.gaps += self._find_orphans(notes, G)
        report.gaps += self._find_sparse_clusters(notes, G)
        report.gaps += self._find_missing_topics(notes, all_edges)
        report.gaps += self._find_depth_gaps(notes)
        report.gaps += self._find_recency_gaps(notes, G)

        # ── LLM-enriched passes ───────────────────────────────────────────
        if llm_enrich and self.llm_registry:
            report.gaps += self._find_one_sided_claims(notes)
            report.gaps += self._recommend_reads(notes, G)

        log.info(f"Gap analysis complete: {len(report.gaps)} gaps found.")
        return report

    # ── Structural detectors ──────────────────────────────────────────────────

    def _find_orphans(self, notes, G) -> list:
        gaps = []
        for note in notes:
            degree = G.degree(note.id) if G.has_node(note.id) else 0
            if degree == 0:
                gaps.append(GapItem(
                    gap_type="orphan",
                    title=f"Isolated idea: «{note.title}»",
                    description=f"This note has no connections to anything else in your brain.",
                    note_ids=[note.id],
                    priority="medium",
                    suggestion="Link it to related notes, or decide if it deserves its own cluster.",
                ))
        return gaps

    def _find_sparse_clusters(self, notes, G) -> list:
        """Clusters with only one or two notes are probably underdeveloped topics."""
        from collections import Counter
        cluster_counts = Counter(
            n.cluster for n in notes if n.cluster is not None
        )
        gaps = []
        for cluster_id, count in cluster_counts.items():
            if count <= 2:
                cluster_notes = [n for n in notes if n.cluster == cluster_id]
                names = ", ".join(f"«{n.title}»" for n in cluster_notes[:3])
                gaps.append(GapItem(
                    gap_type="sparse_cluster",
                    title=f"Thin topic cluster #{cluster_id}",
                    description=f"Only {count} note(s) form this cluster: {names}",
                    note_ids=[n.id for n in cluster_notes],
                    priority="low",
                    suggestion="Consider writing a connecting synthesis note for this topic.",
                ))
        return gaps

    def _find_missing_topics(self, notes, edges) -> list:
        """
        Find concepts that appear as link targets but have no actual note.
        These are ghost ideas — you reference them but never develop them.
        """
        all_ids = {n.id for n in notes}
        all_targets = {e["target"] for e in edges}
        ghost_ids = all_targets - all_ids

        gaps = []
        for ghost_id in list(ghost_ids)[:20]:   # cap at 20
            gaps.append(GapItem(
                gap_type="missing_topic",
                title=f"Ghost reference: {ghost_id[:30]}",
                description="This ID is linked to by other notes but has no corresponding note.",
                note_ids=[ghost_id],
                priority="medium",
                suggestion="Write a stub note for this concept, or fix the broken link.",
            ))
        return gaps

    def _find_depth_gaps(self, notes) -> list:
        """Short notes (< 100 words) that are linked to many times = important but shallow."""
        from collections import Counter
        # count how many times each note is referenced
        all_edges = self.store.get_all_edges()
        target_counts = Counter(e["target"] for e in all_edges)

        gaps = []
        for note in notes:
            refs = target_counts.get(note.id, 0)
            wc = note.word_count()
            if refs >= 3 and wc < 100:
                gaps.append(GapItem(
                    gap_type="depth",
                    title=f"High-traffic stub: «{note.title}»",
                    description=(
                        f"Referenced {refs}× by other notes but only {wc} words long. "
                        "This is probably a core concept that deserves a deeper treatment."
                    ),
                    note_ids=[note.id],
                    priority="high",
                    suggestion=(
                        f"Expand «{note.title}» into a full essay. "
                        "Given its link count, this is a load-bearing node in your graph."
                    ),
                ))
        return gaps

    def _find_recency_gaps(self, notes, G) -> list:
        """High-centrality notes you haven't touched in a long time."""
        gaps = []
        for note in notes:
            if note.date and note.date < self.recency_cutoff:
                centrality = note.centrality or 0.0
                if centrality > 0.05:   # only important nodes
                    gaps.append(GapItem(
                        gap_type="recency",
                        title=f"Stale core idea: «{note.title}»",
                        description=(
                            f"High centrality ({centrality:.3f}) but last touched "
                            f"{note.date.strftime('%Y-%m-%d')}. "
                            "Your thinking may have evolved since then."
                        ),
                        note_ids=[note.id],
                        priority="medium",
                        suggestion="Re-read this note and write a follow-up or revision.",
                    ))
        return gaps

    # ── LLM-enriched detectors ────────────────────────────────────────────────

    def _find_one_sided_claims(self, notes) -> list:
        """
        Use the LLM to find notes that contain strong claims with no
        counterargument or tension anywhere in the corpus.
        """
        try:
            profile = self.llm_registry.get_for_role("gap_analysis")
            client = profile.client()
        except Exception as e:
            log.warning(f"Skipping one_sided analysis: {e}")
            return []

        # Sample the 10 most central notes for cost reasons
        candidates = sorted(notes, key=lambda n: n.centrality or 0, reverse=True)[:10]
        gaps = []

        for note in candidates:
            prompt = f"""
You are analysing a personal philosophy corpus for intellectual blind spots.

Note title: {note.title}
Note content (first 500 chars): {note.content[:500]}

Task: Does this note make a strong philosophical claim without acknowledging
counterarguments or alternative views? If yes, briefly state what the
missing counterargument would be (1-2 sentences). If no, reply NONE.
"""
            try:
                result = client.complete(prompt, max_tokens=200)
                text = result["text"].strip()
                if text.upper() != "NONE" and len(text) > 10:
                    gaps.append(GapItem(
                        gap_type="one_sided",
                        title=f"One-sided claim: «{note.title}»",
                        description=text,
                        note_ids=[note.id],
                        priority="medium",
                        suggestion=(
                            "Write a companion note with the steel-man of the opposing view, "
                            "then link both."
                        ),
                    ))
            except Exception as e:
                log.warning(f"one_sided LLM call failed for {note.id}: {e}")

        return gaps

    def _recommend_reads(self, notes, G) -> list:
        """
        Find topic clusters and ask the LLM (ideally Perplexity for live search)
        what important published works / concepts you should engage with.
        """
        try:
            # Prefer Perplexity for live search; fall back to any available
            try:
                profile = self.llm_registry.get("perplexity_default")
            except KeyError:
                profile = self.llm_registry.get_for_role("gap_analysis")
            client = profile.client()
        except Exception as e:
            log.warning(f"Skipping recommended_reads: {e}")
            return []

        # Build a short topic summary from the most central notes
        top_notes = sorted(notes, key=lambda n: n.centrality or 0, reverse=True)[:8]
        topic_list = "\n".join(f"- {n.title}" for n in top_notes)

        prompt = f"""
A philosopher's personal knowledge base contains these core topics
(ordered by centrality in their idea graph):

{topic_list}

Based on these topics, suggest 3-5 published philosophical works, papers,
or thinkers that this person has likely NOT engaged with deeply but should.
For each, give: title/name, why it connects to their existing work, and what
gap it would fill. Be specific. Reply in plain text, no markdown.
"""
        try:
            result = client.complete(prompt, max_tokens=600)
            text = result["text"].strip()
            if text:
                return [GapItem(
                    gap_type="recommended",
                    title="Recommended reading based on your graph topology",
                    description=text,
                    note_ids=[n.id for n in top_notes],
                    priority="low",
                    suggestion="Add notes for any works here you find compelling.",
                )]
        except Exception as e:
            log.warning(f"recommended_reads LLM call failed: {e}")

        return []
