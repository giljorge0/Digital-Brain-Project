"""
Consolidation Loop
------------------
The continual learning engine for the digital brain. 
Runs periodically to maintain the memory substrate by:
  1. Finding and linking near-duplicate claims (Semantic similarity).
  2. Flagging contradictions within topical clusters using the LLM.
  3. Promoting/demoting memory confidence based on graph centrality and recency.
"""

import logging
import json
from datetime import datetime, timezone
from typing import List

from .store import Store
from ..extract.relations import RelationExtractor
from .graph import GraphBuilder
from .embeddings import _cosine

log = logging.getLogger(__name__)

class ConsolidationAgent:
    def __init__(self, store: Store, extractor: RelationExtractor, graph_builder: GraphBuilder):
        self.store = store
        self.extractor = extractor
        self.graph_builder = graph_builder

    def run_nightly_job(self):
        """Run the full consolidation pipeline."""
        log.info("[consolidate] Waking up. Starting nightly consolidation...")
        
        # 1. Update Graph Math
        self._refresh_graph_metrics()

        # 2. Run the memory maintenance routines
        self.link_similar_notes(threshold=0.92)
        self.flag_cluster_contradictions(max_checks=50)
        self.update_confidence_scores()
        
        # 3. Audit knowledge debt
        self.audit_long_notes(word_limit=600)
        
        log.info("[consolidate] Nightly consolidation complete. Going back to sleep.")

    # ─── 1. Graph Maintenance ──────────────────────────────────────────────────

    def _refresh_graph_metrics(self):
        """Rebuilds the graph, calculates Louvain clusters, and PageRank."""
        log.info("[consolidate] Refreshing global graph metrics...")
        G = self.graph_builder.build(use_explicit=True, use_tags=True, use_semantic=True)
        self.graph_builder.compute_clusters(G)
        self.graph_builder.compute_centrality(G)

    # ─── 2. Deduplication & Merging ────────────────────────────────────────────

    def link_similar_notes(self, threshold: float = 0.92):
        """Find highly similar notes and link them to merge repeated claims."""
        log.info(f"[consolidate] Scanning for duplicate claims (threshold > {threshold})...")
        embeddings = self.store.get_all_embeddings()
        ids = list(embeddings.keys())
        vecs = [embeddings[i] for i in ids]
        
        found = 0
        for i in range(len(ids)):
            for j in range(i + 1, len(ids)):
                sim = _cosine(vecs[i], vecs[j])
                if sim >= threshold:
                    id_a, id_b = ids[i], ids[j]
                    
                    # Instead of deleting, we create a strong 'elaborates' or 'duplicate' edge
                    # This preserves provenance while grouping the memory.
                    self.store.upsert_edge(
                        source=id_a, 
                        target=id_b, 
                        edge_type="llm", 
                        weight=sim,
                        metadata={"relation": "elaborates", "automated_merge": True}
                    )
                    found += 1
        log.info(f"[consolidate] Found and linked {found} highly similar note pairs.")

    # ─── 3. Contradiction Hunting ──────────────────────────────────────────────

    def flag_cluster_contradictions(self, max_checks: int = 50):
        """
        Looks for conflicting evidence within the same topical cluster.
        We only check notes in the same cluster to save LLM tokens.
        """
        log.info("[consolidate] Hunting for contradictions in semantic clusters...")
        notes = self.store.get_all_notes()
        
        # Group by cluster
        clusters = {}
        for n in notes:
            if n.cluster is not None:
                clusters.setdefault(n.cluster, []).append(n)

        checks_done = 0
        contradictions_found = 0

        for cluster_id, cluster_notes in clusters.items():
            if len(cluster_notes) < 2:
                continue
                
            # Sort by centrality to compare the most "important" notes in the cluster
            cluster_notes.sort(key=lambda x: x.centrality, reverse=True)
            top_notes = cluster_notes[:5] 

            for i, note_a in enumerate(top_notes):
                for note_b in top_notes[i+1:]:
                    if checks_done >= max_checks:
                        break
                    
                    checks_done += 1
                    result = self.extractor.extract_relation(note_a, note_b)
                    
                    if result and result.get("relation") == "contradicts":
                        contradictions_found += 1
                        self.store.upsert_edge(
                            source=note_a.id, 
                            target=note_b.id, 
                            edge_type="llm", 
                            weight=result.get("confidence", 0.8),
                            metadata={
                                "relation": "contradicts", 
                                "explanation": result.get("explanation", "Detected during consolidation.")
                            }
                        )
                if checks_done >= max_checks:
                    break
        
        log.info(f"[consolidate] Evaluated {checks_done} pairs, found {contradictions_found} contradictions.")

    # ─── 4. Confidence / Decay ─────────────────────────────────────────────────

    def update_confidence_scores(self):
        """
        Promotes recurring patterns into stable memory and demotes stale memory.
        Calculated as a mix of Graph Centrality (how connected it is) and Recency.
        """
        log.info("[consolidate] Updating memory confidence scores (promotion/demotion)...")
        notes = self.store.get_all_notes()
        now = datetime.now(timezone.utc)
        
        for note in notes:
            # Base confidence comes from how central it is to your network
            # PageRank values are small, so we scale it up slightly for the metadata
            base_score = min(note.centrality * 100, 1.0) 
            
            # Recency multiplier (decay over time)
            recency_mult = 1.0
            if note.date:
                # Ensure note.date is offset-aware before calculating age
                note_date = note.date
                if note_date.tzinfo is None:
                    note_date = note_date.replace(tzinfo=timezone.utc)
                    
                age_days = (now - note_date).days
                if age_days > 365:
                    recency_mult = 0.8  # slightly demote old notes unless highly central
                if age_days < 30:
                    recency_mult = 1.2  # boost fresh notes
                    
            final_confidence = min(round(base_score * recency_mult, 3), 1.0)
            
            # Update the note's metadata dictionary with the new confidence score
            meta = note.metadata or {}
            meta["confidence"] = final_confidence
            note.metadata = meta
            
            # Save back to SQLite
            self.store.upsert_note(note)
            
        log.info("[consolidate] Confidence scores updated.")
# ─── 5. The Length Auditor ─────────────────────────────────────────────────

    def audit_long_notes(self, word_limit: int = 600):
        """
        Scans for manually written notes that have grown too long for atomic embedding.
        Flags them for manual human review instead of blindly chunking them.
        """
        log.info(f"[consolidate] Auditing manual notes longer than {word_limit} words...")
        notes = self.store.get_all_notes()
        
        flagged_count = 0
        for note in notes:
            # We only care about notes that aren't ALREADY mechanical chunks 
            # (like pdf_chunks) or raw dumps (like web_clips)
            is_mechanical = note.metadata.get("type") in ["pdf_chunk", "web_clip", "chat_log"]
            
            if not is_mechanical and note.word_count() > word_limit:
                meta = note.metadata or {}
                
                # If it's not already flagged, flag it
                if not meta.get("needs_refactoring"):
                    meta["needs_refactoring"] = True
                    meta["audit_reason"] = f"Note reached {note.word_count()} words. Consider splitting."
                    note.metadata = meta
                    
                    self.store.upsert_note(note)
                    flagged_count += 1
                    
        log.info(f"[consolidate] Flagged {flagged_count} long notes for manual human refactoring.")
if __name__ == "__main__":
    # Simple test execution if run directly
    import os
    logging.basicConfig(level=logging.INFO)
    
    # Mock setup assuming configs are in environment or a dict
    from .store import Store
    from ..extract.relations import ClaudeExtractor
    from .graph import GraphBuilder
    
    db = Store("data/brain.db")
    extractor = ClaudeExtractor(api_key=os.environ.get("ANTHROPIC_API_KEY", ""))
    builder = GraphBuilder(db)
    
    agent = ConsolidationAgent(db, extractor, builder)
    agent.run_nightly_job()