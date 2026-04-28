"""
Consolidation Agent
-------------------
The "sleep cycle" of the digital brain.

Runs periodically (cron job or manual trigger) to:
  1. Re-cluster notes (Louvain community detection)
  2. Re-compute PageRank centrality
  3. Detect near-duplicate notes (cosine > 0.95) and flag/merge them
  4. Flag contradictions within clusters using LLM
  5. Apply confidence decay to old, low-centrality notes
  6. Promote recurring patterns to stable memory
  7. Audit manual notes for length (knowledge debt)

Design principle:
  The brain gets SMARTER over time, not just bigger.
  Old stale notes decay. Central ideas strengthen.
  Contradictions surface rather than silently coexisting.
"""

import json
import logging
import math
from datetime import datetime, timedelta
from typing import Optional

from brain.memory.graph import GraphBuilder
from brain.memory.embeddings import _cosine

log = logging.getLogger(__name__)


# ─── Thresholds ───────────────────────────────────────────────────────────────
DUPLICATE_THRESHOLD   = 0.95   # cosine sim above this → near-duplicate
DECAY_DAYS_THRESHOLD  = 180    # notes older than this decay faster
DECAY_BASE_RATE       = 0.05   # 5% confidence loss per consolidation run
MIN_CONFIDENCE        = 0.1    # floor — never drops to zero


# ─── Consolidation Agent ──────────────────────────────────────────────────────

class ConsolidationAgent:
    def __init__(self, store, extractor, builder: GraphBuilder):
        self.store     = store
        self.extractor = extractor
        self.builder   = builder

    def run_nightly_job(self):
        log.info("=" * 60)
        log.info("[consolidate] Starting nightly consolidation...")
        log.info("=" * 60)

        # Step 1: Rebuild graph + recompute metrics
        log.info("[consolidate] Step 1/6 — Rebuilding graph & metrics...")
        G = self.builder.build(
            use_explicit=True,
            use_tags=True,
            use_semantic=False,  # skip full semantic rebuild for speed
        )
        self.builder.compute_clusters(G)
        self.builder.compute_centrality(G)

        # Step 2: Detect near-duplicates
        log.info("[consolidate] Step 2/6 — Detecting near-duplicates...")
        duplicates = self._find_duplicates()
        self._flag_duplicates(duplicates)

        # Step 3: Flag contradictions within clusters
        log.info("[consolidate] Step 3/6 — Detecting contradictions...")
        contradictions = self._find_contradictions()
        self._flag_contradictions(contradictions)

        # Step 4: Confidence decay
        log.info("[consolidate] Step 4/6 — Applying confidence decay...")
        self._apply_decay()

        # Step 5: Surface insights (high-centrality recent notes)
        log.info("[consolidate] Step 5/6 — Surfacing emerging patterns...")
        patterns = self._find_emerging_patterns()
        self._log_patterns(patterns)

        # Step 6: Audit knowledge debt
        log.info("[consolidate] Step 6/6 — Auditing long manual notes...")
        long_notes = self.audit_long_notes(word_limit=600)

        log.info("[consolidate] Nightly job complete.")
        return {
            "duplicates_flagged": len(duplicates),
            "contradictions_flagged": len(contradictions),
            "emerging_patterns": [p["title"] for p in patterns],
            "long_notes_flagged": long_notes,
        }

    # ── Step 2: Duplicates ────────────────────────────────────────────────────

    def _find_duplicates(self) -> list:
        """Find note pairs with cosine similarity > threshold."""
        embeddings = self.store.get_all_embeddings()
        if len(embeddings) < 2:
            return []

        ids  = list(embeddings.keys())
        vecs = [embeddings[i] for i in ids]
        pairs = []

        for i in range(len(ids)):
            for j in range(i + 1, len(ids)):
                sim = _cosine(vecs[i], vecs[j])
                if sim >= DUPLICATE_THRESHOLD:
                    pairs.append({
                        "note_a": ids[i],
                        "note_b": ids[j],
                        "similarity": round(sim, 4),
                    })

        log.info(f"[consolidate] Found {len(pairs)} near-duplicate pairs")
        return pairs

    def _flag_duplicates(self, pairs: list):
        for pair in pairs:
            # Add a "duplicate" edge in the graph
            self.store.upsert_edge(
                pair["note_a"], pair["note_b"],
                edge_type="duplicate",
                weight=pair["similarity"],
                metadata={"flagged_by": "consolidation",
                          "similarity": pair["similarity"]},
            )
            log.debug(
                f"[consolidate] Duplicate: {pair['note_a'][:8]}… ↔ "
                f"{pair['note_b'][:8]}… ({pair['similarity']:.3f})"
            )

    # ── Step 3: Contradictions ────────────────────────────────────────────────

    def _find_contradictions(self, max_checks: int = 100) -> list:
        """
        Within each cluster, check semantically similar note pairs for
        contradictions using the LLM extractor.
        """
        notes = self.store.get_all_notes()
        # Group by cluster
        clusters: dict = {}
        for note in notes:
            c = note.cluster
            if c is not None:
                clusters.setdefault(c, []).append(note)

        contradictions = []
        checks_done = 0

        for cluster_id, cluster_notes in clusters.items():
            if len(cluster_notes) < 2:
                continue
            # Check pairs within cluster (up to 5 pairs per cluster)
            pairs_checked = 0
            for i, a in enumerate(cluster_notes):
                for b in cluster_notes[i + 1:]:
                    if checks_done >= max_checks or pairs_checked >= 5:
                        break
                    try:
                        result = self.extractor.check_contradiction(
                            a.short_content(300),
                            b.short_content(300),
                        )
                        if result.get("contradicts") and result.get("confidence", 0) > 0.6:
                            contradictions.append({
                                "note_a": a.id,
                                "note_b": b.id,
                                "confidence": result["confidence"],
                                "explanation": result.get("explanation", ""),
                            })
                    except Exception as e:
                        log.debug(f"[consolidate] Contradiction check failed: {e}")
                    checks_done += 1
                    pairs_checked += 1

        log.info(f"[consolidate] Found {len(contradictions)} contradictions")
        return contradictions

    def _flag_contradictions(self, contradictions: list):
        for c in contradictions:
            self.store.upsert_edge(
                c["note_a"], c["note_b"],
                edge_type="contradiction",
                weight=c["confidence"],
                metadata={
                    "flagged_by": "consolidation",
                    "explanation": c["explanation"],
                },
            )

    # ── Step 4: Confidence decay ──────────────────────────────────────────────

    def _apply_decay(self):
        """
        Reduce confidence of old, low-centrality notes.
        We use centrality as a proxy for confidence (stored in the notes table).
        """
        notes    = self.store.get_all_notes()
        now      = datetime.utcnow()
        decayed  = 0

        for note in notes:
            if note.date is None:
                continue
            age_days = (now - note.date).days if note.date else 0
            if age_days < DECAY_DAYS_THRESHOLD:
                continue

            # Decay faster for old + low-centrality notes
            decay = DECAY_BASE_RATE * (1 + (age_days - DECAY_DAYS_THRESHOLD) / 365)
            new_centrality = max(
                MIN_CONFIDENCE,
                (note.centrality or 0.01) * (1 - decay)
            )
            self.store.update_centrality(note.id, round(new_centrality, 6))
            decayed += 1

        log.info(f"[consolidate] Decayed {decayed} notes")

    # ── Step 5: Emerging patterns ─────────────────────────────────────────────

    def _find_emerging_patterns(self, top_n: int = 10) -> list:
        """
        Find notes that have gained centrality recently — these are
        'emerging ideas' worth surfacing.
        """
        notes = self.store.get_all_notes()
        now   = datetime.utcnow()
        recent_cutoff = now - timedelta(days=30)

        # Score: centrality × recency boost
        scored = []
        for note in notes:
            if not note.centrality:
                continue
            recency_boost = 2.0 if (note.date and note.date > recent_cutoff) else 1.0
            score = note.centrality * recency_boost
            scored.append((score, note))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [
            {"id": n.id, "title": n.title, "score": round(s, 6),
             "cluster": n.cluster}
            for s, n in scored[:top_n]
        ]

    def _log_patterns(self, patterns: list):
        if not patterns:
            return
        log.info("[consolidate] ── Emerging patterns ──────────────────────")
        for p in patterns:
            log.info(f"  [{p['cluster']}] {p['title']}  (score={p['score']})")

    # ── Step 6: The Length Auditor ────────────────────────────────────────────

    def audit_long_notes(self, word_limit: int = 600) -> int:
        """
        Scans for manually written notes (output) that have grown too long 
        for atomic embedding. Flags them for manual human review instead of 
        blindly chunking them.
        """
        notes = self.store.get_all_notes()
        flagged_count = 0
        
        for note in notes:
            # Check provenance_role to ensure it's YOUR writing
            role = note.metadata.get("provenance_role", "unknown")
            is_mechanical = note.metadata.get("type") in ["pdf_chunk", "web_clip", "chat_log", "youtube_watch"]
            
            # Only audit active outputs that exceed the limit
            if (role == "output" or role == "unknown") and not is_mechanical and note.word_count() > word_limit:
                meta = note.metadata or {}
                
                # If it's not already flagged, flag it
                if not meta.get("needs_refactoring"):
                    meta["needs_refactoring"] = True
                    meta["audit_reason"] = f"Note reached {note.word_count()} words. Consider splitting."
                    note.metadata = meta
                    
                    self.store.upsert_note(note)
                    flagged_count += 1
                    
        log.info(f"[consolidate] Flagged {flagged_count} long notes for manual human refactoring.")
        return flagged_count