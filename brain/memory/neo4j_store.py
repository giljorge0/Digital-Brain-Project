"""
Neo4j Store
-----------
Drop-in replacement for the SQLite Store for very large graphs (100k+ notes).

Implements the exact same public interface as brain/memory/store.py so all
other modules work without changes — just swap `Store` for `Neo4jStore` in
main.py or config.

Schema:
  Node (Note)     — all Note fields as properties + label :Note
  Relationship    — edge_type as relationship TYPE, weight + metadata as props
  Embeddings      — stored as a :Embedding node with note_id + vector property
                    (or Neo4j Vector Index if available — see _init_vector_index)

Requirements:
  pip install neo4j

Usage:
  from brain.memory.neo4j_store import Neo4jStore
  store = Neo4jStore("bolt://localhost:7687", "neo4j", "password")

  # Then use exactly like Store:
  store.upsert_note(note)
  notes = store.get_all_notes()
  store.upsert_edge(...)

Environment variables (alternative to constructor args):
  NEO4J_URI       bolt://localhost:7687
  NEO4J_USER      neo4j
  NEO4J_PASSWORD  password
"""

import json
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Optional

from brain.ingest.note import Note

log = logging.getLogger(__name__)


def _require_neo4j():
    try:
        from neo4j import GraphDatabase
        return GraphDatabase
    except ImportError:
        raise ImportError(
            "pip install neo4j\n"
            "Then start a Neo4j instance: docker run -p 7687:7687 -p 7474:7474 "
            "-e NEO4J_AUTH=neo4j/password neo4j:5"
        )


class Neo4jStore:
    """
    Neo4j-backed store. Identical public interface to SQLite Store.

    Parameters
    ----------
    uri      : bolt://localhost:7687
    user     : neo4j
    password : password
    """

    def __init__(self,
                 uri:      str = None,
                 user:     str = None,
                 password: str = None):
        GraphDatabase = _require_neo4j()

        self.uri      = uri      or os.environ.get("NEO4J_URI",      "bolt://localhost:7687")
        self.user     = user     or os.environ.get("NEO4J_USER",     "neo4j")
        self.password = password or os.environ.get("NEO4J_PASSWORD", "password")

        self._driver = GraphDatabase.driver(self.uri, auth=(self.user, self.password))
        self._init_schema()
        log.info(f"[neo4j] Connected to {self.uri}")

    # ── Schema ────────────────────────────────────────────────────────────────

    def _init_schema(self):
        with self._driver.session() as s:
            s.run("CREATE CONSTRAINT note_id IF NOT EXISTS FOR (n:Note) REQUIRE n.id IS UNIQUE")
            s.run("CREATE CONSTRAINT emb_id  IF NOT EXISTS FOR (e:Embedding) REQUIRE e.note_id IS UNIQUE")
            # Optional vector index if Neo4j 5.11+
            try:
                s.run("""
                    CREATE VECTOR INDEX note_embedding IF NOT EXISTS
                    FOR (e:Embedding) ON (e.vector)
                    OPTIONS {indexConfig: {`vector.dimensions`: 384, `vector.similarity_function`: 'cosine'}}
                """)
                log.info("[neo4j] Vector index created/verified.")
            except Exception:
                log.debug("[neo4j] Vector index not supported on this Neo4j version — using cosine in Python.")

    # ── Notes ─────────────────────────────────────────────────────────────────

    def upsert_note(self, note: Note):
        with self._driver.session() as s:
            s.run("""
                MERGE (n:Note {id: $id})
                SET n.title       = $title,
                    n.content     = $content,
                    n.tags        = $tags,
                    n.source_file = $source_file,
                    n.date        = $date,
                    n.links       = $links,
                    n.metadata    = $metadata,
                    n.cluster     = $cluster,
                    n.centrality  = $centrality,
                    n.word_count  = $word_count
            """, {
                "id":          note.id,
                "title":       note.title,
                "content":     note.content,
                "tags":        json.dumps(note.tags),
                "source_file": note.source_file,
                "date":        note.date.isoformat() if note.date else None,
                "links":       json.dumps(note.links),
                "metadata":    json.dumps(note.metadata),
                "cluster":     note.cluster,
                "centrality":  note.centrality,
                "word_count":  note.word_count(),
            })

    def upsert_notes(self, notes: list):
        for note in notes:
            self.upsert_note(note)

    def get_note(self, note_id: str) -> Optional[Note]:
        with self._driver.session() as s:
            rec = s.run("MATCH (n:Note {id: $id}) RETURN n", {"id": note_id}).single()
            return _record_to_note(rec["n"]) if rec else None

    def get_all_notes(self) -> list:
        with self._driver.session() as s:
            recs = s.run("MATCH (n:Note) RETURN n").data()
            return [_record_to_note(r["n"]) for r in recs]

    def get_notes_by_tag(self, tag: str) -> list:
        with self._driver.session() as s:
            recs = s.run(
                'MATCH (n:Note) WHERE n.tags CONTAINS $tag RETURN n',
                {"tag": f'"{tag}"'}
            ).data()
            return [_record_to_note(r["n"]) for r in recs]

    def search_notes(self, query: str, limit: int = 20) -> list:
        with self._driver.session() as s:
            recs = s.run(
                """MATCH (n:Note)
                   WHERE toLower(n.title) CONTAINS toLower($q)
                      OR toLower(n.content) CONTAINS toLower($q)
                   RETURN n LIMIT $limit""",
                {"q": query, "limit": limit}
            ).data()
            return [_record_to_note(r["n"]) for r in recs]

    def update_cluster(self, note_id: str, cluster: int):
        with self._driver.session() as s:
            s.run("MATCH (n:Note {id:$id}) SET n.cluster=$c", {"id": note_id, "c": cluster})

    def update_centrality(self, note_id: str, centrality: float):
        with self._driver.session() as s:
            s.run("MATCH (n:Note {id:$id}) SET n.centrality=$c", {"id": note_id, "c": centrality})

    def note_count(self) -> int:
        with self._driver.session() as s:
            return s.run("MATCH (n:Note) RETURN count(n) AS c").single()["c"]

    # ── Edges ─────────────────────────────────────────────────────────────────

    def upsert_edge(self, source: str, target: str, edge_type: str,
                    weight: float = 1.0, metadata: dict = None):
        rel_type = edge_type.upper().replace("-", "_").replace(" ", "_")
        with self._driver.session() as s:
            s.run(f"""
                MATCH (a:Note {{id:$src}}), (b:Note {{id:$tgt}})
                MERGE (a)-[r:{rel_type} {{edge_type:$et}}]->(b)
                SET r.weight   = $w,
                    r.metadata = $meta
            """, {
                "src":  source,
                "tgt":  target,
                "et":   edge_type,
                "w":    weight,
                "meta": json.dumps(metadata or {}),
            })

    def get_edges(self, note_id: str = None, edge_type: str = None) -> list:
        where = []
        params: dict = {}
        if note_id:
            where.append("(a.id=$nid OR b.id=$nid)")
            params["nid"] = note_id
        if edge_type:
            where.append("r.edge_type=$et")
            params["et"] = edge_type

        clause = ("WHERE " + " AND ".join(where)) if where else ""
        cypher = f"""
            MATCH (a:Note)-[r]->(b:Note)
            {clause}
            RETURN a.id AS source, b.id AS target,
                   r.edge_type AS edge_type, r.weight AS weight,
                   r.metadata AS metadata
        """
        with self._driver.session() as s:
            return [dict(rec) for rec in s.run(cypher, params).data()]

    def get_all_edges(self) -> list:
        return self.get_edges()

    def edge_count(self) -> int:
        with self._driver.session() as s:
            return s.run("MATCH ()-[r]->() RETURN count(r) AS c").single()["c"]

    # ── Embeddings ────────────────────────────────────────────────────────────

    def save_embedding(self, note_id: str, vector: list, model: str = "unknown"):
        with self._driver.session() as s:
            s.run("""
                MERGE (e:Embedding {note_id:$nid})
                SET e.vector = $vec, e.model = $model
            """, {"nid": note_id, "vec": json.dumps(vector), "model": model})

    def get_embedding(self, note_id: str) -> Optional[list]:
        with self._driver.session() as s:
            rec = s.run(
                "MATCH (e:Embedding {note_id:$nid}) RETURN e.vector AS v", {"nid": note_id}
            ).single()
            return json.loads(rec["v"]) if rec else None

    def get_all_embeddings(self) -> dict:
        with self._driver.session() as s:
            recs = s.run("MATCH (e:Embedding) RETURN e.note_id AS nid, e.vector AS v").data()
            return {r["nid"]: json.loads(r["v"]) for r in recs}

    def notes_without_embeddings(self) -> list:
        with self._driver.session() as s:
            recs = s.run("""
                MATCH (n:Note) WHERE NOT (n)<-[:Embedding]-()
                  AND NOT EXISTS { MATCH (e:Embedding {note_id:n.id}) }
                RETURN n
            """).data()
            # Fallback: find notes whose IDs are not in embeddings
            all_note_ids  = {r["n"]["id"] for r in s.run("MATCH (n:Note) RETURN n.id AS id").data()}
            embedded_ids  = {r["note_id"] for r in s.run("MATCH (e:Embedding) RETURN e.note_id").data()}
            missing_ids   = all_note_ids - embedded_ids
            if not missing_ids:
                return []
            recs = s.run(
                "MATCH (n:Note) WHERE n.id IN $ids RETURN n",
                {"ids": list(missing_ids)}
            ).data()
            return [_record_to_note(r["n"]) for r in recs]

    # ── Misc ──────────────────────────────────────────────────────────────────

    def stats(self) -> dict:
        with self._driver.session() as s:
            n_notes  = s.run("MATCH (n:Note) RETURN count(n) AS c").single()["c"]
            n_edges  = s.run("MATCH ()-[r]->() RETURN count(r) AS c").single()["c"]
            n_emb    = s.run("MATCH (e:Embedding) RETURN count(e) AS c").single()["c"]
            n_clust  = s.run("MATCH (n:Note) WHERE n.cluster IS NOT NULL "
                             "RETURN count(DISTINCT n.cluster) AS c").single()["c"]
        return {
            "notes":                 n_notes,
            "edges":                 n_edges,
            "notes_with_embeddings": n_emb,
            "clusters":              n_clust,
        }

    def close(self):
        self._driver.close()
        log.info("[neo4j] Connection closed.")


# ─── Helpers ──────────────────────────────────────────────────────────────────

def _record_to_note(node) -> Note:
    d = dict(node)
    return Note(
        id=d["id"],
        title=d.get("title", ""),
        content=d.get("content", ""),
        tags=json.loads(d.get("tags") or "[]"),
        source_file=d.get("source_file", ""),
        date=_parse_date(d.get("date")),
        links=json.loads(d.get("links") or "[]"),
        metadata=json.loads(d.get("metadata") or "{}"),
        cluster=d.get("cluster"),
        centrality=d.get("centrality", 0.0),
    )


def _parse_date(s: Optional[str]) -> Optional[datetime]:
    if not s:
        return None
    try:
        return datetime.fromisoformat(s)
    except ValueError:
        return None
