"""
Gap Finder
----------
Analyzes the structure of your knowledge graph and identifies six types
of gaps — regions of your idea space that are unexplored, underdeveloped,
or need synthesis.

Gap types:
  void          — semantic regions near your clusters with no coverage
  depth         — topics you mention but never fully develop
  width         — ideas you've explored deeply but whose siblings you've ignored
  temporal      — ideas you haven't revisited in over N months
  contradiction — flagged contradictions that need a reconciling note
  orthogonal    — strongest counterarguments to your positions you've never engaged

The output is a list of Gap objects, each with:
  - type, title, description (human-readable)
  - gap_vector (float list) — the embedding representing what's missing
  - related_note_ids — notes in the vicinity
  - priority_score — how important this gap is to fill
  - suggested_actions — concrete next steps

The gap_vector is what gets passed to the Recommender.
It is already anonymised — it encodes WHAT is missing, not your raw corpus.
"""

import json
import logging
import math
import os
import urllib.request
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Optional

from brain.memory.embeddings import _cosine

log = logging.getLogger(__name__)

ROLE_OUTPUT = "output"


# ─── Gap data class ───────────────────────────────────────────────────────────

@dataclass
class Gap:
    gap_type:       str              # void | depth | width | temporal | contradiction | orthogonal
    title:          str              # short human-readable name
    description:    str              # what is missing and why it matters
    gap_vector:     list             # embedding of the gap concept
    related_ids:    list = field(default_factory=list)
    priority_score: float = 0.5
    suggested_actions: list = field(default_factory=list)
    metadata:       dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "gap_type":          self.gap_type,
            "title":             self.title,
            "description":       self.description,
            "related_ids":       self.related_ids,
            "priority_score":    round(self.priority_score, 3),
            "suggested_actions": self.suggested_actions,
            "metadata":          self.metadata,
            # gap_vector intentionally omitted from dict (passed internally)
        }


# ─── Gap Finder ───────────────────────────────────────────────────────────────

class GapFinder:
    """
    Analyzes your knowledge graph and returns prioritized Gap objects.

    Usage:
        finder = GapFinder(store, embedder, cfg)
        gaps   = finder.find_all_gaps()
        # gaps is sorted by priority, ready to pass to Recommender
    """

    def __init__(self, store, embedder, cfg: dict):
        self.store   = store
        self.embedder = embedder
        self.cfg     = cfg
        self._llm_backend = cfg.get("llm_backend", "claude")

    # ── Public API ────────────────────────────────────────────────────────────

    def find_all_gaps(self, max_per_type: int = 5) -> list:
        """Run all six gap analyses and return prioritized list."""
        all_gaps = []

        log.info("[gaps] Analyzing void gaps...")
        all_gaps.extend(self._find_void_gaps(n=max_per_type))

        log.info("[gaps] Analyzing depth gaps...")
        all_gaps.extend(self._find_depth_gaps(n=max_per_type))

        log.info("[gaps] Analyzing width gaps...")
        all_gaps.extend(self._find_width_gaps(n=max_per_type))

        log.info("[gaps] Analyzing temporal gaps...")
        all_gaps.extend(self._find_temporal_gaps(n=max_per_type))

        log.info("[gaps] Analyzing contradiction gaps...")
        all_gaps.extend(self._find_contradiction_gaps(n=max_per_type))

        log.info("[gaps] Analyzing orthogonal view gaps...")
        all_gaps.extend(self._find_orthogonal_gaps(n=max_per_type))

        # Sort by priority
        all_gaps.sort(key=lambda g: g.priority_score, reverse=True)
        log.info(f"[gaps] Found {len(all_gaps)} gaps total")
        return all_gaps

    def find_gaps_of_type(self, gap_type: str, n: int = 5) -> list:
        dispatch = {
            "void":          self._find_void_gaps,
            "depth":         self._find_depth_gaps,
            "width":         self._find_width_gaps,
            "temporal":      self._find_temporal_gaps,
            "contradiction": self._find_contradiction_gaps,
            "orthogonal":    self._find_orthogonal_gaps,
        }
        fn = dispatch.get(gap_type)
        if not fn:
            raise ValueError(f"Unknown gap type: {gap_type}. "
                             f"Valid: {list(dispatch.keys())}")
        return fn(n=n)

    # ── Gap type 1: Void — unexplored semantic regions ─────────────────────────

    def _find_void_gaps(self, n: int = 5) -> list:
        """
        Find semantic regions NEAR your cluster centroids that you haven't explored.
        Strategy: compute cluster centroids, then perturb them in random directions
        and check density. Low-density nearby directions = voids.
        """
        embeddings = self.store.get_all_embeddings()
        if len(embeddings) < 5:
            return []

        notes = self.store.get_all_notes()
        notes_by_cluster: dict = {}
        for note in notes:
            c = note.cluster
            if c is not None:
                notes_by_cluster.setdefault(c, []).append(note.id)

        if not notes_by_cluster:
            return []

        gaps = []
        all_vecs = list(embeddings.values())

        for cluster_id, note_ids in list(notes_by_cluster.items())[:10]:
            # Compute cluster centroid
            cluster_vecs = [embeddings[nid] for nid in note_ids
                            if nid in embeddings]
            if len(cluster_vecs) < 2:
                continue
            centroid = _mean_vec(cluster_vecs)

            # Find closest notes to centroid
            closest = sorted(
                [(nid, _cosine(centroid, embeddings[nid]))
                 for nid in note_ids if nid in embeddings],
                key=lambda x: x[1], reverse=True
            )[:3]
            cluster_notes = [self.store.get_note(nid) for nid, _ in closest]
            cluster_notes = [n for n in cluster_notes if n]

            if not cluster_notes:
                continue

            # Ask LLM to identify void areas around this cluster
            cluster_summary = "\n".join(
                f"- {n.title}: {n.short_content(100)}" for n in cluster_notes
            )
            void_desc = self._llm_identify_void(cluster_summary, cluster_id)
            if not void_desc:
                continue

            # Create gap vector by embedding the void description
            try:
                gap_vec = self.embedder.embed_one(void_desc["topic"])
            except Exception:
                gap_vec = centroid  # fallback

            # Score: inverse of cluster density (sparse clusters → more void)
            density = len(cluster_vecs) / max(len(notes), 1)
            priority = max(0.1, 1.0 - density * 10)

            gaps.append(Gap(
                gap_type="void",
                title=void_desc["topic"],
                description=void_desc["description"],
                gap_vector=gap_vec,
                related_ids=[n.id for n in cluster_notes],
                priority_score=priority,
                suggested_actions=[
                    f"Write a note on: {void_desc['topic']}",
                    f"Explore: {void_desc.get('search_query', void_desc['topic'])}",
                ],
                metadata={"cluster_id": cluster_id, "cluster_size": len(note_ids)}
            ))

            if len(gaps) >= n:
                break

        return gaps

    def _llm_identify_void(self, cluster_summary: str,
                            cluster_id: int) -> Optional[dict]:
        """Ask the LLM what's missing around this cluster of ideas."""
        prompt = f"""You are analyzing a philosopher's knowledge graph.
Below are some notes from one topic cluster in their personal knowledge base.
Identify ONE important topic that is ADJACENT to this cluster but NOT yet covered.
This should be something a philosopher deeply interested in these ideas would
naturally want to explore next.

CLUSTER NOTES:
{cluster_summary}

Respond in JSON only:
{{
  "topic": "short topic name (3-6 words)",
  "description": "why this gap exists and why it matters (2 sentences)",
  "search_query": "what to search for to fill this gap"
}}"""
        raw = self._llm_call(prompt, 300)
        try:
            clean = raw.strip().lstrip("```json").lstrip("```").rstrip("```").strip()
            return json.loads(clean)
        except Exception:
            return None

    # ── Gap type 2: Depth — mentioned but underdeveloped ──────────────────────

    def _find_depth_gaps(self, n: int = 5) -> list:
        """
        Find topics you reference often but never develop.
        Proxy: notes with high in-degree but low word count.
        """
        notes = self.store.get_all_notes()
        all_edges = self.store.get_all_edges()

        # Count how many times each note is referenced (in-degree)
        in_degree: dict = {}
        for edge in all_edges:
            tgt = edge["target"]
            in_degree[tgt] = in_degree.get(tgt, 0) + 1

        # Score by: high in-degree / word_count
        scored = []
        for note in notes:
            wc = note.word_count()
            deg = in_degree.get(note.id, 0)
            if wc < 20 or deg == 0:
                continue
            # Low word count + high reference = depth gap
            score = (deg / max(wc / 50, 1)) * note.centrality if note.centrality else deg / max(wc / 50, 1)
            scored.append((score, note))

        scored.sort(key=lambda x: x[0], reverse=True)
        top = scored[:n]

        gaps = []
        for score, note in top:
            try:
                gap_vec = self.embedder.embed_one(
                    f"deep analysis of {note.title}: {note.content[:200]}"
                )
            except Exception:
                gap_vec = []

            gaps.append(Gap(
                gap_type="depth",
                title=f"Develop: {note.title}",
                description=(
                    f"'{note.title}' is referenced {in_degree.get(note.id, 0)} times "
                    f"across your corpus but only has {note.word_count()} words. "
                    f"This idea deserves a full exploration."
                ),
                gap_vector=gap_vec,
                related_ids=[note.id],
                priority_score=min(score / 10, 1.0),
                suggested_actions=[
                    f"Write a full essay on: {note.title}",
                    f"Break '{note.title}' into atomic sub-claims",
                    "Use 'suggest-notes' on this note to find atomic branches",
                ],
                metadata={"word_count": note.word_count(),
                          "in_degree": in_degree.get(note.id, 0)}
            ))

        return gaps

    # ── Gap type 3: Width — unexplored siblings ───────────────────────────────

    def _find_width_gaps(self, n: int = 5) -> list:
        """
        Find canonical siblings of your most-developed topics that you haven't covered.
        E.g.: you've written about Gödel's incompleteness but not Tarski's undefinability.
        """
        # Find your most central notes (most developed ideas)
        notes = sorted(
            [n for n in self.store.get_all_notes() if n.centrality and n.centrality > 0],
            key=lambda n: n.centrality, reverse=True
        )[:15]

        if not notes:
            return []

        gaps = []
        for note in notes[:8]:
            siblings = self._llm_find_siblings(note)
            if not siblings:
                continue

            # Check which siblings are already in corpus
            all_notes = self.store.get_all_notes()
            existing_titles = {n.title.lower() for n in all_notes}

            for sibling in siblings:
                if any(sibling["name"].lower() in title or title in sibling["name"].lower()
                       for title in existing_titles):
                    continue  # Already covered

                try:
                    gap_vec = self.embedder.embed_one(
                        f"{sibling['name']}: {sibling.get('description', '')}"
                    )
                except Exception:
                    gap_vec = []

                gaps.append(Gap(
                    gap_type="width",
                    title=f"Missing sibling: {sibling['name']}",
                    description=(
                        f"You've written about '{note.title}' but not '{sibling['name']}' — "
                        f"{sibling.get('description', 'a closely related concept')}. "
                        f"Exploring this would complete your coverage of this intellectual domain."
                    ),
                    gap_vector=gap_vec,
                    related_ids=[note.id],
                    priority_score=note.centrality * 0.8,
                    suggested_actions=[
                        f"Read about: {sibling['name']}",
                        f"Write a note comparing '{note.title}' to '{sibling['name']}'",
                    ],
                    metadata={"parent_note": note.title,
                              "sibling_name": sibling["name"]}
                ))

                if len(gaps) >= n:
                    break

            if len(gaps) >= n:
                break

        return gaps

    def _llm_find_siblings(self, note) -> list:
        """Find canonical siblings of a concept."""
        prompt = f"""A philosopher has written extensively about: "{note.title}"
Content sample: {note.short_content(200)}

Name 3 CANONICAL SIBLING concepts that belong to the same intellectual domain
that a serious thinker would also need to engage with — but that are DISTINCT
from what's described above (i.e., they are NOT already covered in the sample).

Respond in JSON only:
{{
  "siblings": [
    {{"name": "concept name", "description": "one sentence on what this is"}}
  ]
}}"""
        raw = self._llm_call(prompt, 400)
        try:
            clean = raw.strip().lstrip("```json").lstrip("```").rstrip("```").strip()
            return json.loads(clean).get("siblings", [])
        except Exception:
            return []

    # ── Gap type 4: Temporal — stale ideas ───────────────────────────────────

    def _find_temporal_gaps(self, n: int = 5,
                            stale_months: int = 12) -> list:
        """
        Find important topics you haven't revisited in over `stale_months`.
        These ideas may have developed in the world since you last thought about them.
        """
        notes = self.store.get_all_notes()
        cutoff = datetime.utcnow() - timedelta(days=stale_months * 30)

        # Only your output notes, with a date, that are old
        stale = [
            n for n in notes
            if n.date and n.date < cutoff
            and n.metadata.get("provenance_role", ROLE_OUTPUT) == ROLE_OUTPUT
            and n.centrality and n.centrality > 0
        ]

        # Sort by centrality (most important stale ideas first)
        stale.sort(key=lambda n: n.centrality or 0, reverse=True)

        gaps = []
        for note in stale[:n]:
            age_months = (datetime.utcnow() - note.date).days // 30

            try:
                gap_vec = self.embedder.embed_one(
                    f"recent developments in: {note.title} {note.content[:150]}"
                )
            except Exception:
                gap_vec = []

            gaps.append(Gap(
                gap_type="temporal",
                title=f"Revisit: {note.title}",
                description=(
                    f"'{note.title}' was last written {age_months} months ago. "
                    f"This is one of your more central ideas — it may be worth "
                    f"revisiting with what you've learned since, or checking what "
                    f"has happened in this area in the world."
                ),
                gap_vector=gap_vec,
                related_ids=[note.id],
                priority_score=min((note.centrality or 0.1) * (age_months / 24), 1.0),
                suggested_actions=[
                    f"Re-read and annotate: {note.title}",
                    f"Search for recent work on: {note.title}",
                    "Write a 'revisit' note: what has changed in your thinking?",
                ],
                metadata={"age_months": age_months,
                          "last_date": note.date.isoformat()[:10]}
            ))

        return gaps

    # ── Gap type 5: Contradiction — unresolved tensions ───────────────────────

    def _find_contradiction_gaps(self, n: int = 5) -> list:
        """
        Find contradiction edges in the graph that need a reconciling note.
        """
        contradiction_edges = self.store.get_edges(edge_type="contradiction")

        gaps = []
        for edge in contradiction_edges[:n]:
            note_a = self.store.get_note(edge["source"])
            note_b = self.store.get_note(edge["target"])
            if not note_a or not note_b:
                continue

            explanation = json.loads(edge.get("metadata") or "{}").get(
                "explanation", "These notes appear to contradict each other."
            )

            try:
                gap_vec = self.embedder.embed_one(
                    f"reconciling: {note_a.title} vs {note_b.title}"
                )
            except Exception:
                gap_vec = []

            gaps.append(Gap(
                gap_type="contradiction",
                title=f"Resolve: {note_a.title} ↔ {note_b.title}",
                description=(
                    f"Your notes '{note_a.title}' and '{note_b.title}' appear to "
                    f"contradict each other. {explanation} "
                    f"Writing a synthesis or update note would strengthen your thinking."
                ),
                gap_vector=gap_vec,
                related_ids=[note_a.id, note_b.id],
                priority_score=float(edge.get("weight", 0.5)),
                suggested_actions=[
                    "Write a synthesis note that reconciles both positions",
                    "Identify which claim you now believe is wrong",
                    "Check if both claims can be true at different levels of abstraction",
                ],
                metadata={"confidence": edge.get("weight", 0.5)}
            ))

        return gaps

    # ── Gap type 6: Orthogonal — strong counterarguments ─────────────────────

    def _find_orthogonal_gaps(self, n: int = 5) -> list:
        """
        Find your strongest held positions and identify the best counterarguments
        or orthogonal views you haven't engaged with.
        """
        notes = sorted(
            [note for note in self.store.get_all_notes()
             if note.metadata.get("provenance_role", ROLE_OUTPUT) == ROLE_OUTPUT
             and note.centrality and note.centrality > 0.001],
            key=lambda n: n.centrality, reverse=True
        )[:10]

        gaps = []
        for note in notes[:6]:
            counter = self._llm_find_counterargument(note)
            if not counter:
                continue

            try:
                gap_vec = self.embedder.embed_one(counter["counterargument"])
            except Exception:
                gap_vec = []

            gaps.append(Gap(
                gap_type="orthogonal",
                title=f"Counter: {counter['stance']}",
                description=(
                    f"Against your position in '{note.title}', "
                    f"the strongest counterargument is: {counter['counterargument']} "
                    f"This view is associated with: {counter.get('associated_with', 'alternative traditions')}."
                ),
                gap_vector=gap_vec,
                related_ids=[note.id],
                priority_score=note.centrality * 0.9,
                suggested_actions=[
                    f"Read: {counter.get('recommended_reading', 'the literature on this counterargument')}",
                    f"Write a steelman of: {counter['stance']}",
                    f"Update '{note.title}' to acknowledge this objection",
                ],
                metadata={"source_note": note.title,
                          "counterargument": counter["counterargument"]}
            ))

            if len(gaps) >= n:
                break

        return gaps

    def _llm_find_counterargument(self, note) -> Optional[dict]:
        prompt = f"""A philosopher has written this position:
Title: "{note.title}"
Content: {note.short_content(300)}

Identify the STRONGEST counterargument or orthogonal philosophical position
they have not yet engaged with — something genuinely challenging to their view,
not a strawman. This should come from real philosophical or intellectual traditions.

Respond in JSON only:
{{
  "stance": "short name for the counter-position (3-6 words)",
  "counterargument": "the strongest objection in 1-2 sentences",
  "associated_with": "thinkers or traditions associated with this view",
  "recommended_reading": "one specific book, paper, or author to engage with"
}}"""
        raw = self._llm_call(prompt, 400)
        try:
            clean = raw.strip().lstrip("```json").lstrip("```").rstrip("```").strip()
            return json.loads(clean)
        except Exception:
            return None

    # ── LLM backend ───────────────────────────────────────────────────────────

    def _llm_call(self, prompt: str, max_tokens: int = 400) -> str:
        if self._llm_backend == "ollama":
            return self._ollama(prompt, max_tokens)
        return self._claude(prompt, max_tokens)

    def _claude(self, prompt: str, max_tokens: int) -> str:
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
                return json.loads(resp.read())["content"][0]["text"]
        except Exception as e:
            return f"[error: {e}]"

    def _ollama(self, prompt: str, max_tokens: int) -> str:
        base  = self.cfg.get("ollama_base_url", "http://localhost:11434")
        model = self.cfg.get("ollama_model", "mistral")
        payload = json.dumps({
            "model": model, "prompt": prompt, "stream": False,
            "format": "json", "options": {"num_predict": max_tokens},
        }).encode("utf-8")
        req = urllib.request.Request(
            f"{base.rstrip('/')}/api/generate", data=payload,
            headers={"Content-Type": "application/json"}, method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=120) as resp:
                return json.loads(resp.read())["response"]
        except Exception as e:
            return f"[error: {e}]"


# ─── Helpers ──────────────────────────────────────────────────────────────────

def _mean_vec(vecs: list) -> list:
    if not vecs:
        return []
    n = len(vecs)
    return [sum(v[i] for v in vecs) / n for i in range(len(vecs[0]))]
