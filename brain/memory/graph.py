"""
Knowledge Graph
---------------
Builds a multi-edge directed graph from the Store.

Edge types:
  explicit   — [[link]] in note text (weight 1.0)
  tag        — shared tags (weight = Jaccard similarity)
  semantic   — cosine similarity of embeddings (weight = similarity score)
  llm        — relations extracted by LLM (weight from confidence)

After building, runs:
  - Louvain community detection → note.cluster
  - PageRank centrality        → note.centrality
"""

import json
import math
import logging
from typing import Optional

try:
    import networkx as nx
    HAS_NX = True
except ImportError:
    HAS_NX = False
    logging.warning("networkx not installed — graph features disabled")

from .store import Store

log = logging.getLogger(__name__)


# ─── Graph Builder ────────────────────────────────────────────────────────────

class GraphBuilder:
    def __init__(self, store: Store, similarity_threshold: float = 0.75):
        if not HAS_NX:
            raise ImportError("pip install networkx")
        self.store = store
        self.threshold = similarity_threshold

    # ── Public API ────────────────────────────────────────────────────────────

    def build(self,
              use_explicit: bool = True,
              use_tags: bool = True,
              use_semantic: bool = True) -> "nx.DiGraph":
        """Build the full graph from the store and return it."""
        G = nx.DiGraph()

        notes = self.store.get_all_notes()
        log.info(f"[graph] Loading {len(notes)} notes into graph")

        for note in notes:
            G.add_node(note.id,
                       title=note.title,
                       tags=note.tags,
                       cluster=note.cluster,
                       centrality=note.centrality,
                       word_count=note.word_count(),
                       date=note.date.isoformat() if note.date else None,
                       source_file=note.source_file)

        # Edges from store (explicit + llm + semantic already persisted)
        for edge in self.store.get_all_edges():
            G.add_edge(edge["source"], edge["target"],
                       edge_type=edge["edge_type"],
                       weight=edge["weight"])

        if use_explicit:
            self._add_explicit_edges(G, notes)

        if use_tags:
            self._add_tag_edges(G, notes)

        if use_semantic:
            self._add_semantic_edges(G)

        log.info(f"[graph] Built graph: {G.number_of_nodes()} nodes, "
                 f"{G.number_of_edges()} edges")
        return G

    def compute_clusters(self, G: "nx.DiGraph") -> dict:
        """
        Run Louvain community detection on undirected projection.
        Returns {node_id: cluster_int}.
        """
        try:
            from community import best_partition  # python-louvain
        except ImportError:
            log.warning("python-louvain not installed; using connected components")
            return self._fallback_clusters(G)

        undirected = G.to_undirected()
        partition = best_partition(undirected)
        for node_id, cluster in partition.items():
            self.store.update_cluster(node_id, cluster)
        log.info(f"[graph] Found {len(set(partition.values()))} clusters")
        return partition

    def compute_centrality(self, G: "nx.DiGraph") -> dict:
        """
        Run PageRank and persist to store.
        Returns {node_id: score}.
        """
        pr = nx.pagerank(G, weight="weight")
        for node_id, score in pr.items():
            self.store.update_centrality(node_id, round(score, 6))
        log.info(f"[graph] Centrality computed for {len(pr)} nodes")
        return pr

    def to_json(self, G: "nx.DiGraph") -> dict:
        """Serialize to {nodes, links} for D3.js force graph."""
        nodes = []
        for node_id, data in G.nodes(data=True):
            nodes.append({
                "id": node_id,
                "title": data.get("title", node_id),
                "tags": data.get("tags", []),
                "cluster": data.get("cluster"),
                "centrality": data.get("centrality", 0.0),
                "word_count": data.get("word_count", 0),
                "date": data.get("date"),
                "source_file": data.get("source_file", ""),
            })

        links = []
        for src, tgt, data in G.edges(data=True):
            links.append({
                "source": src,
                "target": tgt,
                "edge_type": data.get("edge_type", "unknown"),
                "weight": data.get("weight", 1.0),
            })

        return {"nodes": nodes, "links": links}

    # ── Private builders ──────────────────────────────────────────────────────

    def _add_explicit_edges(self, G, notes):
        """Add edges for [[links]] in note text."""
        note_ids = {n.id for n in notes}
        added = 0
        for note in notes:
            for link_target in note.links:
                if link_target in note_ids and link_target != note.id:
                    if not G.has_edge(note.id, link_target):
                        G.add_edge(note.id, link_target,
                                   edge_type="explicit", weight=1.0)
                        self.store.upsert_edge(note.id, link_target, "explicit", 1.0)
                        added += 1
        log.info(f"[graph] Added {added} explicit edges")

    def _add_tag_edges(self, G, notes):
        """Connect notes that share tags using Jaccard similarity as weight."""
        added = 0
        note_list = [(n.id, set(n.tags)) for n in notes if n.tags]
        for i, (id_a, tags_a) in enumerate(note_list):
            for id_b, tags_b in note_list[i + 1:]:
                if id_a == id_b:
                    continue
                intersection = tags_a & tags_b
                if not intersection:
                    continue
                union = tags_a | tags_b
                jaccard = len(intersection) / len(union)
                if jaccard >= 0.2:  # at least 20% overlap
                    if not G.has_edge(id_a, id_b):
                        G.add_edge(id_a, id_b,
                                   edge_type="tag", weight=round(jaccard, 3))
                        self.store.upsert_edge(id_a, id_b, "tag", jaccard,
                                               {"shared_tags": list(intersection)})
                        added += 1
        log.info(f"[graph] Added {added} tag edges")

    def _add_semantic_edges(self, G):
        """Connect notes with cosine similarity above threshold."""
        embeddings = self.store.get_all_embeddings()
        if len(embeddings) < 2:
            return

        ids = list(embeddings.keys())
        vecs = [embeddings[i] for i in ids]
        added = 0

        for i in range(len(ids)):
            for j in range(i + 1, len(ids)):
                sim = _cosine(vecs[i], vecs[j])
                if sim >= self.threshold:
                    id_a, id_b = ids[i], ids[j]
                    if not G.has_edge(id_a, id_b):
                        G.add_edge(id_a, id_b,
                                   edge_type="semantic", weight=round(sim, 4))
                        self.store.upsert_edge(id_a, id_b, "semantic", sim,
                                               {"similarity": sim})
                        added += 1

        log.info(f"[graph] Added {added} semantic edges "
                 f"(threshold={self.threshold})")

    def _fallback_clusters(self, G) -> dict:
        """Use weakly connected components if Louvain unavailable."""
        undirected = G.to_undirected()
        partition = {}
        for cluster_id, component in enumerate(
                nx.connected_components(undirected)):
            for node in component:
                partition[node] = cluster_id
        for node_id, cluster in partition.items():
            self.store.update_cluster(node_id, cluster)
        return partition


# ─── Helpers ─────────────────────────────────────────────────────────────────

def _cosine(a: list, b: list) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    mag_a = math.sqrt(sum(x * x for x in a))
    mag_b = math.sqrt(sum(y * y for y in b))
    if mag_a == 0 or mag_b == 0:
        return 0.0
    return dot / (mag_a * mag_b)
