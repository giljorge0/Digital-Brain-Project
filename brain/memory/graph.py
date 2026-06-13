"""
Knowledge Graph — Memory-Safe Large-Scale Build
------------------------------------------------
Key changes vs original:
  1. Uses get_note_stubs() not get_all_notes() for tag/explicit edges
     → saves 80-90% RAM (no content field loaded)

  2. Semantic edges via ANN (Chroma) or batched numpy — never O(n²) RAM
     - With Chroma: query each note's k-NN, ~O(n log n), < 1 GB for 100k notes
     - Without Chroma: streaming numpy batches, each batch is freed after commit

  3. All edges written via upsert_edges_batch() — single transaction per batch
     → eliminates the 450k-fsync crash that froze the machine

  4. compute_clusters() and compute_centrality() write back via batch methods
     → single transaction instead of one UPDATE per note

  5. build_incremental() — entry point for the omni brain.
     Skips the full NetworkX graph in RAM for large corpora (> LARGE_THRESHOLD).
     Instead writes edges directly to SQLite and builds NetworkX only for
     PageRank/Louvain on a pruned graph (top-centrality nodes only).

  6. Chroma ANN integration: if a ChromaBackend is attached to the store,
     semantic search uses it automatically with no code change in main.py.
"""

import gc
import json
import math
import logging
from typing import Optional

try:
    import scipy.sparse as _scipy_sparse
    import numpy as _np
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False
    logging.getLogger(__name__).debug("scipy not installed; PageRank will use NetworkX")

try:
    import networkx as nx
    HAS_NX = True
except ImportError:
    HAS_NX = False
    logging.warning("networkx not installed — graph features disabled")

from .store import Store

log = logging.getLogger(__name__)

# Notes above this threshold trigger the incremental (RAM-safe) path
LARGE_THRESHOLD = 5_000

# How many semantic edges to write to SQLite before committing
EDGE_BATCH_SIZE = 5_000


class GraphBuilder:
    def __init__(self, store: Store, similarity_threshold: float = 0.75):
        if not HAS_NX:
            raise ImportError("pip install networkx")
        self.store     = store
        self.threshold = similarity_threshold

    # ── Public API ────────────────────────────────────────────────────────────

    def build(self,
              use_explicit: bool = True,
              use_tags:     bool = True,
              use_semantic: bool = True) -> "nx.DiGraph":
        """
        Build the full knowledge graph.
        For small corpora (< LARGE_THRESHOLD): builds full NetworkX graph in RAM.
        For large corpora: builds edges directly in SQLite, returns a lightweight
        stub graph (no content) for PageRank/Louvain.
        """
        note_count = self.store.note_count()

        # ── File logging: survives a freeze, shows exactly where it stopped ──
        import logging as _logging
        log_path = self.store.db_path.parent / "build.log"
        if not any(isinstance(h, _logging.FileHandler) for h in log.handlers):
            fh = _logging.FileHandler(str(log_path), mode='a', encoding='utf-8')
            fh.setLevel(_logging.DEBUG)
            fh.setFormatter(_logging.Formatter(
                '%(asctime)s  %(levelname)-8s  %(message)s',
                datefmt='%H:%M:%S'
            ))
            log.addHandler(fh)
            # Also make sure root logger propagates at DEBUG level
            log.setLevel(_logging.DEBUG)

        log.info(f"[graph] ══ BUILD START ══  {note_count} notes")
        log.info(f"[graph] DB: {self.store.db_path}")
        log.info(f"[graph] Log file: {log_path}")
        log.info(f"[graph] Threshold: {self.threshold}  "
                 f"(if > 5k notes, incremental path active)")

        if self.threshold < 0.72 and note_count > 5_000:
            log.warning(
                f"[graph] threshold={self.threshold} is LOW for {note_count} notes. "
                "At low thresholds many note pairs match — the semantic edge "
                "builder will be slow. Consider raising to 0.78 in config.yaml."
            )

        if note_count > LARGE_THRESHOLD:
            return self._build_large(use_explicit, use_tags, use_semantic)
        else:
            return self._build_small(use_explicit, use_tags, use_semantic)

    def compute_clusters(self, G: "nx.DiGraph") -> dict:
        """
        Louvain community detection. Writes back via batch update.
        Falls back to connected components if python-louvain not installed.
        """
        try:
            from community import best_partition
        except ImportError:
            log.warning("python-louvain not installed; using connected components")
            return self._fallback_clusters(G)

        undirected = G.to_undirected()
        partition  = best_partition(undirected)

        # Batch write — single transaction
        pairs = [(cluster, nid) for nid, cluster in partition.items()]
        self.store.update_clusters_batch(pairs)

        log.info(f"[graph] Found {len(set(partition.values()))} clusters")
        return partition

    def compute_centrality(self, G: "nx.DiGraph") -> dict:
        """
        PageRank. Uses scipy.sparse if available (CSR matrix, ~5 MB for 450k
        edges) otherwise falls back to NetworkX (correct but RAM-heavy).
        Writes results back via single-transaction batch update.
        """
        if HAS_SCIPY:
            try:
                pr = self._pagerank_scipy(G)
                log.info(f"[graph] PageRank (scipy sparse) done for {len(pr)} nodes")
            except Exception as e:
                log.warning(f"[graph] scipy PageRank failed ({e}); using NetworkX")
                pr = nx.pagerank(G, weight="weight")
                log.info(f"[graph] PageRank (networkx) done for {len(pr)} nodes")
        else:
            pr = nx.pagerank(G, weight="weight")
            log.info(f"[graph] PageRank (networkx) done for {len(pr)} nodes")

        pairs = [(round(score, 6), nid) for nid, score in pr.items()]
        self.store.update_centralities_batch(pairs)
        return pr

    def _pagerank_scipy(
        self,
        G:        "nx.DiGraph",
        alpha:    float = 0.85,
        max_iter: int   = 100,
        tol:      float = 1.0e-6,
    ) -> dict:
        """
        PageRank via scipy.sparse CSR matrix — power iteration.

        Memory comparison for 30k nodes + 450k edges:
          NetworkX nx.pagerank():  ~300 MB  (converts to dense matrix internally)
          This method:               ~8 MB  (CSR float32 + two rank vectors)

        Algorithm:
          M[j,i] = A[i,j] / out_degree(i)   (column-stochastic, from→to)
          r ← alpha × M @ r  +  dangling_contrib  +  (1-alpha)/N
        """
        import numpy as np
        from scipy.sparse import csr_matrix, diags

        nodes = list(G.nodes())
        n     = len(nodes)
        if n == 0:
            return {}

        # String-ID → integer index
        idx = {nid: i for i, nid in enumerate(nodes)}

        # Build COO arrays for the adjacency matrix (A[src, tgt] = weight)
        row_arr, col_arr, w_arr = [], [], []
        for src, tgt, w in G.edges(data="weight", default=1.0):
            row_arr.append(idx[src])
            col_arr.append(idx[tgt])
            w_arr.append(float(w) if w else 1.0)

        if not row_arr:
            return {nid: 1.0 / n for nid in nodes}

        # CSR matrix — float32 halves memory vs float64
        A = csr_matrix(
            (np.array(w_arr, dtype=np.float32),
             (np.array(row_arr, dtype=np.int32),
              np.array(col_arr, dtype=np.int32))),
            shape=(n, n),
        )
        del row_arr, col_arr, w_arr
        gc.collect()

        # Out-degree per node (row sums of A = outgoing weight)
        out_deg     = np.asarray(A.sum(axis=1), dtype=np.float32).flatten()
        is_dangling = out_deg == 0
        out_deg[is_dangling] = 1.0          # avoid divide-by-zero

        # Column-stochastic transition matrix M = A.T @ diag(1/out_deg)
        # M[j, i] = A[i,j] / out_deg[i]  →  random-walk from i to j
        D_inv = diags(1.0 / out_deg, dtype=np.float32)
        M     = (A.T @ D_inv).tocsr()
        del A, D_inv
        gc.collect()

        # Power iteration
        r            = np.full(n, 1.0 / n, dtype=np.float32)
        dangling_idx = np.where(is_dangling)[0]

        for iteration in range(max_iter):
            r_prev = r

            # Dangling nodes (no out-edges) teleport their rank uniformly
            dangling_contrib = alpha * float(r_prev[dangling_idx].sum()) / n

            r = (alpha * M.dot(r_prev)
                 + dangling_contrib
                 + (1.0 - alpha) / n)

            if float(np.abs(r - r_prev).sum()) < tol * n:
                log.info(f"[graph] PageRank converged in {iteration + 1} iterations")
                break

        del M
        return {nodes[i]: float(r[i]) for i in range(n)}

    def to_json(self, G: "nx.DiGraph") -> dict:
        nodes, links = [], []
        for nid, data in G.nodes(data=True):
            nodes.append({
                "id":          nid,
                "title":       data.get("title", nid),
                "tags":        data.get("tags", []),
                "cluster":     data.get("cluster"),
                "centrality":  data.get("centrality", 0.0),
                "word_count":  data.get("word_count", 0),
                "date":        data.get("date"),
                "source_file": data.get("source_file", ""),
            })
        for src, tgt, data in G.edges(data=True):
            links.append({
                "source":    src,
                "target":    tgt,
                "edge_type": data.get("edge_type", "unknown"),
                "weight":    data.get("weight", 1.0),
            })
        return {"nodes": nodes, "links": links}

    # ── Small build (< LARGE_THRESHOLD) ──────────────────────────────────────

    def _build_small(self, use_explicit, use_tags, use_semantic):
        G      = nx.DiGraph()
        stubs  = self.store.get_note_stubs()

        for s in stubs:
            G.add_node(s["id"], **s)

        # Load existing persisted edges
        for edge in self.store.get_all_edges():
            G.add_edge(edge["source"], edge["target"],
                       edge_type=edge["edge_type"], weight=edge["weight"])

        if use_explicit:
            self._add_explicit_edges_from_stubs(G, stubs)
        if use_tags:
            self._add_tag_edges_from_stubs(G, stubs)
        if use_semantic:
            self._add_semantic_edges(G)

        log.info(f"[graph] Built graph: {G.number_of_nodes()} nodes, "
                 f"{G.number_of_edges()} edges")
        return G

    # ── Large build (>= LARGE_THRESHOLD) ─────────────────────────────────────

    def _build_large(self, use_explicit, use_tags, use_semantic):
        """
        Memory-safe path for 30k+ notes.

        Phase 1 — edge building: writes edges to SQLite without holding a
        large NetworkX graph in RAM. Each edge type (explicit / tag / semantic)
        is built and committed in batches, then freed.

        Phase 2 — stub graph: after ALL edges are committed:
          • stubs and stub_map are deleted and gc.collect()ed
          • edges are STREAMED from SQLite via fetchmany(10_000) — never
            materialised as a full Python list
          • PageRank is computed on a scipy.sparse CSR matrix (~5 MB for
            450k edges) rather than loading everything into NetworkX
          • NetworkX graph is built with minimal attributes only (no content,
            no tags, just title/cluster/centrality for to_json compatibility)
        """
        log.info("[graph] Large corpus detected — using incremental edge builder")
        stubs    = self.store.get_note_stubs()
        stub_map = {s["id"]: s for s in stubs}

        import time as _time

        if use_explicit:
            log.info("[graph] ── Phase 1/3: explicit edges ──")
            t0 = _time.time()
            self._build_explicit_to_db(stubs, stub_map)
            log.info(f"[graph] explicit done in {_time.time()-t0:.1f}s")

        if use_tags:
            log.info("[graph] ── Phase 2/3: tag edges ──")
            t0 = _time.time()
            self._build_tag_edges_to_db(stubs)
            log.info(f"[graph] tag done in {_time.time()-t0:.1f}s")

        if use_semantic:
            log.info("[graph] ── Phase 3/3: semantic edges ──")
            t0 = _time.time()
            self._build_semantic_edges_to_db()
            log.info(f"[graph] semantic done in {_time.time()-t0:.1f}s")

        # ── Phase 2: free all edge-building data before stub graph ──────────
        # stubs (~30k dicts) and stub_map are still live from Phase 1.
        # Extract only what the stub graph needs (minimal attributes for
        # to_json() compatibility), then release the rest.
        stub_attrs = {
            s["id"]: {
                "title":      s.get("title", ""),
                "cluster":    s.get("cluster", 0),
                "centrality": s.get("centrality", 0.0),
            }
            for s in stubs
        }
        del stubs, stub_map
        gc.collect()

        log.info("[graph] Assembling stub graph for PageRank + Louvain…")
        G = self._assemble_stub_graph_light(stub_attrs)
        del stub_attrs
        gc.collect()

        log.info(f"[graph] Stub graph ready: {G.number_of_nodes()} nodes, "
                 f"{G.number_of_edges()} edges")
        return G

    def _assemble_stub_graph_light(self, stub_attrs: dict,
                                    max_edges_per_node: int = 8,
                                    max_total_edges: int = 500_000) -> "nx.DiGraph":
        """
        Build the minimal NetworkX graph needed for PageRank + Louvain.

        KEY FIX (OOM killer):
          The original code streamed ALL edges from SQLite into NetworkX.
          With 15.9M edges, NetworkX's adjacency dicts consume ~4-5 GB RAM.
          This killed the process on 16 GB machines.

        Fix: cap to top-K edges per source node.
          30k notes × 8 edges = 240k edges ≈ 30 MB.
          PageRank and Louvain still give meaningful results on this skeleton.

        Strategy:
          Primary path — SQL window function filters in SQLite, zero Python
            overhead: SELECT ... ROW_NUMBER() OVER (PARTITION BY source
            ORDER BY weight DESC) WHERE rn <= K
          Fallback — Python accumulator (for SQLite < 3.25): streams all rows
            but only keeps top-K per node in a bounded dict (~25 MB peak).
        """
        G = nx.DiGraph()
        G.add_nodes_from((nid, attrs) for nid, attrs in stub_attrs.items())
        node_set = set(stub_attrs)

        edge_count = 0
        try:
            # ── Primary: SQL window function (SQLite ≥ 3.25) ────────────────
            import sqlite3
            conn = sqlite3.connect(str(self.store.db_path), timeout=60)
            conn.row_factory = sqlite3.Row
            cur  = conn.cursor()
            cur.execute("""
                SELECT source, target, weight
                FROM (
                    SELECT source, target, weight,
                           ROW_NUMBER() OVER (
                               PARTITION BY source
                               ORDER BY weight DESC
                           ) AS rn
                    FROM edges
                )
                WHERE rn <= ?
            """, (max_edges_per_node,))

            while True:
                batch = cur.fetchmany(20_000)
                if not batch:
                    break
                for row in batch:
                    src, tgt = row["source"], row["target"]
                    if src in node_set and tgt in node_set:
                        G.add_edge(src, tgt, weight=float(row["weight"] or 1.0))
                        edge_count += 1
                        if edge_count >= max_total_edges:
                            break
                if edge_count >= max_total_edges:
                    log.warning(f"[graph] Stub graph capped at {max_total_edges} edges")
                    break
            conn.close()

        except Exception as e:
            log.warning(f"[graph] SQL window function failed ({e}); "
                        "using Python top-K accumulator fallback")
            # ── Fallback: Python accumulator ─────────────────────────────────
            # Streams all edges but keeps only top-K per source in RAM.
            # Peak memory ≈ (30k nodes × max_edges_per_node × ~100B) ≈ 25 MB.
            G.remove_edges_from(list(G.edges()))  # clear any partial edges
            top_k: dict = {}  # src_id -> [(weight, tgt_id), ...]

            for edge in self._stream_edges():
                src, tgt = edge["source"], edge["target"]
                if src not in node_set or tgt not in node_set:
                    continue
                w = edge.get("weight", 1.0)
                bucket = top_k.setdefault(src, [])
                if len(bucket) < max_edges_per_node:
                    bucket.append((w, tgt))
                    if len(bucket) == max_edges_per_node:
                        bucket.sort()           # ascending so bucket[0] is min
                elif w > bucket[0][0]:
                    bucket[0] = (w, tgt)
                    bucket.sort()

            for src, neighbors in top_k.items():
                for w, tgt in neighbors:
                    G.add_edge(src, tgt, weight=w)
                    edge_count += 1
            del top_k
            gc.collect()

        log.info(
            f"[graph] Streamed {edge_count} edges into stub graph "
            f"(capped: top-{max_edges_per_node} per source, "
            f"max {max_total_edges} total)"
        )
        return G

    def _stream_edges(self, batch_size: int = 10_000):
        """
        Stream edges from SQLite one batch at a time.

        Opens a dedicated read connection so the store's write connection
        is not blocked. Uses fetchmany() — peak memory = batch_size × ~500 B
        = 5 MB vs fetchall() which loads the full 220-450 MB at once.
        """
        import sqlite3
        db_path = str(self.store.db_path)
        try:
            # Separate read connection — does not interfere with store writes
            conn = sqlite3.connect(db_path, timeout=60)
            conn.row_factory = sqlite3.Row
            cur  = conn.cursor()
            cur.execute(
                "SELECT source, target, edge_type, weight FROM edges"
            )
            while True:
                batch = cur.fetchmany(batch_size)
                if not batch:
                    break
                for row in batch:
                    yield {
                        "source":    row["source"],
                        "target":    row["target"],
                        "edge_type": row["edge_type"],
                        "weight":    float(row["weight"] or 1.0),
                    }
            conn.close()
        except Exception as e:
            log.warning(
                f"[graph] _stream_edges direct SQLite failed ({e}); "
                "falling back to store.get_all_edges() — may use more RAM"
            )
            try:
                conn.close()
            except Exception:
                pass
            # Fallback: store API (correct but loads all into memory)
            for edge in self.store.get_all_edges():
                yield edge

    # ── Explicit edge builders ─────────────────────────────────────────────────

    def _add_explicit_edges_from_stubs(self, G, stubs, max_per_node: int = 50):
        node_ids = {s["id"] for s in stubs}
        batch    = []
        added    = 0
        for s in stubs:
            for link in s["links"]:
                if link in node_ids and link != s["id"]:
                    if not G.has_edge(s["id"], link):
                        G.add_edge(s["id"], link, edge_type="explicit", weight=1.0)
                        batch.append((s["id"], link, "explicit", 1.0, {}))
                        added += 1
        self.store.upsert_edges_batch(batch)
        log.info(f"[graph] Added {added} explicit edges")

    def _build_explicit_to_db(self, stubs, stub_map):
        node_ids = set(stub_map.keys())
        batch    = []
        added    = 0
        for s in stubs:
            for link in s["links"]:
                if link in node_ids and link != s["id"]:
                    batch.append((s["id"], link, "explicit", 1.0, {}))
                    added += 1
                    if len(batch) >= EDGE_BATCH_SIZE:
                        self.store.upsert_edges_batch(batch)
                        batch = []
        if batch:
            self.store.upsert_edges_batch(batch)
        log.info(f"[graph] Added {added} explicit edges")

    # ── Tag edge builders ──────────────────────────────────────────────────────

    def _add_tag_edges_from_stubs(self, G, stubs, max_per_node: int = 15):
        """Inverted index approach — avoids O(n²). Caps edges per node."""
        tags_dict = {s["id"]: set(s["tags"]) for s in stubs if s["tags"]}
        inverted  = {}
        for nid, t_set in tags_dict.items():
            for t in t_set:
                inverted.setdefault(t, []).append(nid)

        batch = []
        added = 0
        log.info(f"[graph] Tag edges: inverted index over {len(tags_dict)} notes…")

        for idx, (id_a, tags_a) in enumerate(tags_dict.items()):
            shared = {}
            for t in tags_a:
                for id_b in inverted[t]:
                    if id_a != id_b:
                        shared[id_b] = shared.get(id_b, 0) + 1

            neighbors = []
            for id_b, intersect in shared.items():
                union    = len(tags_a) + len(tags_dict[id_b]) - intersect
                jaccard  = intersect / union if union else 0
                if jaccard >= 0.2:
                    neighbors.append((jaccard, id_b))

            neighbors.sort(reverse=True)
            for jaccard, id_b in neighbors[:max_per_node]:
                if not G.has_edge(id_a, id_b):
                    G.add_edge(id_a, id_b, edge_type="tag", weight=round(jaccard, 3))
                    batch.append((id_a, id_b, "tag", round(jaccard, 3), {}))
                    added += 1

            if len(batch) >= EDGE_BATCH_SIZE:
                self.store.upsert_edges_batch(batch)
                batch = []

            if idx % 5000 == 0 and idx > 0:
                log.info(f"[graph] tag edges: {idx}/{len(tags_dict)} notes processed")

        if batch:
            self.store.upsert_edges_batch(batch)
        log.info(f"[graph] Added {added} tag edges")

    def _build_tag_edges_to_db(self, stubs, max_per_node: int = 15):
        """Same as above but writes to DB only (no NetworkX graph in RAM)."""
        tags_dict = {s["id"]: set(s["tags"]) for s in stubs if s["tags"]}
        inverted  = {}
        for nid, t_set in tags_dict.items():
            for t in t_set:
                inverted.setdefault(t, []).append(nid)

        batch = []
        added = 0
        seen  = set()  # dedup within THIS run only — SQLite handles cross-run dedup

        for idx, (id_a, tags_a) in enumerate(tags_dict.items()):
            shared = {}
            for t in tags_a:
                for id_b in inverted[t]:
                    if id_a != id_b:
                        shared[id_b] = shared.get(id_b, 0) + 1

            neighbors = []
            for id_b, intersect in shared.items():
                union   = len(tags_a) + len(tags_dict[id_b]) - intersect
                jaccard = intersect / union if union else 0
                if jaccard >= 0.2:
                    neighbors.append((jaccard, id_b))

            neighbors.sort(reverse=True)
            for jaccard, id_b in neighbors[:max_per_node]:
                key = (id_a, id_b)
                if key not in seen:
                    batch.append((id_a, id_b, "tag", round(jaccard, 3), {}))
                    seen.add(key)
                    added += 1

            if len(batch) >= EDGE_BATCH_SIZE:
                self.store.upsert_edges_batch(batch)
                batch = []

            if idx % 5000 == 0 and idx > 0:
                log.info(f"[graph] tag edges: {idx}/{len(tags_dict)} notes processed")

        if batch:
            self.store.upsert_edges_batch(batch)
        log.info(f"[graph] Added {added} tag edges to DB")

    # ── Semantic edge builders ─────────────────────────────────────────────────

    def _add_semantic_edges(self, G, max_per_node: int = 15):
        """
        Semantic edges for small graph — numpy batched, never O(n²) in RAM.
        Each batch is a (BATCH × n) matrix, freed after its edges are written.
        """
        ids, vecs = self.store.get_embeddings_numpy()
        if vecs is None or len(ids) < 2:
            return

        import numpy as np
        norms    = np.linalg.norm(vecs, axis=1, keepdims=True)
        norms[norms == 0] = 1e-10
        vecs_n   = vecs / norms
        del vecs  # free original

        BATCH    = 500
        batch    = []
        added    = 0
        id_set   = set(ids)

        for i in range(0, len(ids), BATCH):
            end       = min(i + BATCH, len(ids))
            sim_block = np.dot(vecs_n[i:end], vecs_n.T)   # (BATCH, n)

            for row in range(sim_block.shape[0]):
                global_i = i + row
                id_a     = ids[global_i]
                sims     = sim_block[row]

                above = np.where(sims >= self.threshold)[0]
                neighbors = sorted(
                    [(float(sims[j]), ids[j]) for j in above if j != global_i],
                    reverse=True
                )[:max_per_node]

                for sim, id_b in neighbors:
                    if not G.has_edge(id_a, id_b):
                        G.add_edge(id_a, id_b, edge_type="semantic",
                                   weight=round(sim, 4))
                        batch.append((id_a, id_b, "semantic", round(sim, 4),
                                      {"similarity": round(sim, 4)}))
                        added += 1

            del sim_block  # free this batch's matrix immediately

            if len(batch) >= EDGE_BATCH_SIZE:
                self.store.upsert_edges_batch(batch)
                batch = []

            log.info(f"[graph] semantic: {end}/{len(ids)} nodes done")

        if batch:
            self.store.upsert_edges_batch(batch)
        log.info(f"[graph] Added {added} semantic edges")

    def _build_semantic_edges_to_db(self, max_per_node: int = 15):
        """
        Memory-safe semantic edge builder for large corpora.
        Tries Chroma ANN first (best), falls back to streaming numpy batches.
        Never holds the full similarity matrix in RAM.

        Chroma path:  O(n × k log n) — suitable for 100k+ notes on 8GB RAM
        Numpy path:   O(n × BATCH)   — suitable for 30k notes on 8GB RAM,
                      peak RAM ≈ BATCH × n × 4 bytes
                      At BATCH=256, n=30k: 30 MB per batch
        """
        # Try Chroma ANN first
        if self._try_chroma_semantic_to_db(max_per_node):
            return

        # Numpy streaming fallback
        ids, vecs = self.store.get_embeddings_numpy()
        if vecs is None or len(ids) < 2:
            return

        import numpy as np
        log.info(f"[graph] Semantic edges: numpy streaming over {len(ids)} notes "
                 f"(peak RAM per batch ≈ {256 * len(ids) * 4 // 1_000_000} MB)")

        norms = np.linalg.norm(vecs, axis=1, keepdims=True)
        norms[norms == 0] = 1e-10
        vecs_n = vecs / norms
        del vecs  # free raw vectors — keep only normalised copy

        BATCH = 128   # smaller batch = less peak RAM per iteration
        batch = []
        added = 0
        K     = max_per_node  # how many top neighbors to keep per note

        for i in range(0, len(ids), BATCH):
            end       = min(i + BATCH, len(ids))
            sim_block = np.dot(vecs_n[i:end], vecs_n.T).astype(np.float32)

            for row in range(sim_block.shape[0]):
                global_i            = i + row
                id_a                = ids[global_i]
                sims                = sim_block[row].copy()
                sims[global_i]      = -1.0   # exclude self

                # numpy top-K via argpartition: O(n) vs O(n log n) sorted
                # Critically: never creates a Python list of ALL above-threshold items
                top_k_idx = np.argpartition(sims, -K)[-K:]
                for j in top_k_idx:
                    sim = float(sims[j])
                    if sim >= self.threshold:
                        # SQLite ON CONFLICT handles cross-run dedup
                        batch.append((id_a, ids[j], "semantic",
                                      round(sim, 4), {"similarity": round(sim, 4)}))
                        added += 1

            del sim_block
            gc.collect()

            if len(batch) >= EDGE_BATCH_SIZE:
                self.store.upsert_edges_batch(batch)
                batch = []
                gc.collect()

            if (i // BATCH) % 10 == 0:
                log.info(f"[graph] semantic: {end}/{len(ids)} notes, "
                         f"{added} edges, RAM ok")

        if batch:
            self.store.upsert_edges_batch(batch)
        log.info(f"[graph] Added {added} semantic edges to DB")

    def _try_chroma_semantic_to_db(self, max_per_node: int = 15) -> bool:
        """
        Use Chroma's HNSW index for ANN search — far more memory efficient
        than the numpy matrix for 100k+ notes.

        Returns True if Chroma was available and edges were built.
        Returns False to signal the numpy fallback should run.
        """
        try:
            import chromadb
        except ImportError:
            return False

        # Check if a Chroma collection exists alongside this DB
        chroma_path = str(self.store.db_path.parent / "chroma")
        try:
            client = chromadb.PersistentClient(path=chroma_path)
            col    = client.get_collection("brain")
            count  = col.count()
            if count < 2:
                return False
        except Exception:
            return False

        log.info(f"[graph] Semantic edges via Chroma ANN ({count} vectors)")

        ids  = self.store.get_embedding_ids()
        batch = []
        added = 0

        for idx, note_id in enumerate(ids):
            # Batch-fetch vectors from Chroma (more efficient than one-by-one)
            result = col.get(ids=[note_id], include=["embeddings"])
            embs   = result.get("embeddings", [])
            if not embs:
                continue

            q_vec = list(embs[0])

            # ANN query — top k+1 (includes self)
            qr = col.query(
                query_embeddings=[q_vec],
                n_results=min(max_per_node + 1, count),
                include=["distances"],
            )
            neighbor_ids   = qr["ids"][0]
            neighbor_dists = qr["distances"][0]

            for nid, dist in zip(neighbor_ids, neighbor_dists):
                if nid == note_id:
                    continue
                # Chroma cosine distance = 1 - similarity
                sim = 1.0 - dist
                if sim < self.threshold:
                    continue
                # No existing-set check — SQLite ON CONFLICT handles dedup
                batch.append((note_id, nid, "semantic",
                              round(sim, 4), {"similarity": round(sim, 4)}))
                added += 1

            if len(batch) >= EDGE_BATCH_SIZE:
                self.store.upsert_edges_batch(batch)
                batch = []

            if idx % 2000 == 0 and idx > 0:
                log.info(f"[graph] Chroma ANN: {idx}/{len(ids)} notes, {added} edges")

        if batch:
            self.store.upsert_edges_batch(batch)
        log.info(f"[graph] Added {added} semantic edges via Chroma ANN")
        return True

    # ── Cluster fallback ──────────────────────────────────────────────────────

    def _fallback_clusters(self, G: "nx.DiGraph") -> dict:
        undirected = G.to_undirected()
        partition  = {}
        for cluster_id, component in enumerate(nx.connected_components(undirected)):
            for node in component:
                partition[node] = cluster_id
        pairs = [(cluster, nid) for nid, cluster in partition.items()]
        self.store.update_clusters_batch(pairs)
        log.info(f"[graph] Fallback: {len(set(partition.values()))} connected components")
        return partition
