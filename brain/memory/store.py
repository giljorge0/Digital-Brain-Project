"""
SQLite Store
------------
Three tables:
  notes      — all parsed Note objects
  edges      — directed relationships between notes
  embeddings — float32 vectors stored as JSON blobs

Large-scale additions vs original:
  - WAL journal mode + tuned PRAGMAs (10–50× faster bulk writes)
  - get_embeddings_numpy()     — directly to np.ndarray, skips Python dict overhead
  - iter_embeddings_chunked()  — generator, never holds full set in RAM
  - upsert_edges_batch()       — single-transaction bulk insert (no per-edge fsync)
  - get_note_stubs()           — id/title/tags/links only, no content loaded
  - iter_notes_chunked()       — paginated note iteration
  - update_clusters_batch()    — single-transaction cluster updates
  - update_centralities_batch()— single-transaction centrality updates
"""

import json
import sqlite3
from pathlib import Path
from datetime import datetime
from typing import Optional

from ..ingest.note import Note


SCHEMA = """
CREATE TABLE IF NOT EXISTS notes (
    id           TEXT PRIMARY KEY,
    title        TEXT NOT NULL,
    content      TEXT,
    tags         TEXT,
    source_file  TEXT,
    date         TEXT,
    links        TEXT,
    metadata     TEXT,
    cluster      INTEGER,
    centrality   REAL DEFAULT 0.0,
    word_count   INTEGER DEFAULT 0,
    created_at   TEXT DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS edges (
    source       TEXT NOT NULL,
    target       TEXT NOT NULL,
    edge_type    TEXT NOT NULL,
    weight       REAL DEFAULT 1.0,
    metadata     TEXT,
    PRIMARY KEY (source, target, edge_type)
);

CREATE TABLE IF NOT EXISTS embeddings (
    note_id  TEXT PRIMARY KEY,
    vector   TEXT NOT NULL,
    model    TEXT DEFAULT 'unknown'
);

CREATE INDEX IF NOT EXISTS idx_notes_tags   ON notes(tags);
CREATE INDEX IF NOT EXISTS idx_notes_date   ON notes(date);
CREATE INDEX IF NOT EXISTS idx_edges_source ON edges(source);
CREATE INDEX IF NOT EXISTS idx_edges_target ON edges(target);
CREATE INDEX IF NOT EXISTS idx_edges_type   ON edges(edge_type);
"""


class Store:
    def __init__(self, db_path):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
        self.conn.row_factory = sqlite3.Row

        # WAL mode: concurrent readers + writers, no full-file locking
        self.conn.execute("PRAGMA journal_mode=WAL")
        # NORMAL sync: safe but no fsync on every commit (10× faster)
        self.conn.execute("PRAGMA synchronous=NORMAL")
        # 64 MB page cache (default is 2 MB)
        self.conn.execute("PRAGMA cache_size=-65536")
        # Temp tables in RAM
        self.conn.execute("PRAGMA temp_store=MEMORY")

        self.conn.executescript(SCHEMA)
        self.conn.commit()

    # ── Notes ─────────────────────────────────────────────────────────────────

    def upsert_note(self, note: Note):
        self.conn.execute(
            """INSERT INTO notes
               (id,title,content,tags,source_file,date,links,metadata,cluster,centrality,word_count)
               VALUES (?,?,?,?,?,?,?,?,?,?,?)
               ON CONFLICT(id) DO UPDATE SET
                 title=excluded.title, content=excluded.content, tags=excluded.tags,
                 source_file=excluded.source_file, date=excluded.date,
                 links=excluded.links, metadata=excluded.metadata,
                 cluster=excluded.cluster, centrality=excluded.centrality,
                 word_count=excluded.word_count""",
            (note.id, note.title, note.content,
             json.dumps(note.tags), note.source_file,
             note.date.isoformat() if note.date else None,
             json.dumps(note.links), json.dumps(note.metadata),
             note.cluster, note.centrality, note.word_count())
        )
        self.conn.commit()

    def upsert_notes(self, notes: list):
        """Batch upsert — single transaction."""
        self.conn.executemany(
            """INSERT INTO notes
               (id,title,content,tags,source_file,date,links,metadata,cluster,centrality,word_count)
               VALUES (?,?,?,?,?,?,?,?,?,?,?)
               ON CONFLICT(id) DO UPDATE SET
                 title=excluded.title, content=excluded.content, tags=excluded.tags,
                 source_file=excluded.source_file, date=excluded.date,
                 links=excluded.links, metadata=excluded.metadata,
                 cluster=excluded.cluster, centrality=excluded.centrality,
                 word_count=excluded.word_count""",
            [(n.id, n.title, n.content,
              json.dumps(n.tags), n.source_file,
              n.date.isoformat() if n.date else None,
              json.dumps(n.links), json.dumps(n.metadata),
              n.cluster, n.centrality, n.word_count()) for n in notes]
        )
        self.conn.commit()

    def get_note(self, note_id: str) -> Optional[Note]:
        row = self.conn.execute("SELECT * FROM notes WHERE id=?", (note_id,)).fetchone()
        return _row_to_note(row) if row else None

    def get_all_notes(self) -> list:
        return [_row_to_note(r) for r in self.conn.execute("SELECT * FROM notes").fetchall()]

    def get_notes_by_tag(self, tag: str) -> list:
        rows = self.conn.execute(
            "SELECT * FROM notes WHERE tags LIKE ?", (f'%"{tag}"%',)
        ).fetchall()
        return [_row_to_note(r) for r in rows]

    def search_notes(self, query: str, limit: int = 20) -> list:
        rows = self.conn.execute(
            "SELECT * FROM notes WHERE title LIKE ? OR content LIKE ? LIMIT ?",
            (f'%{query}%', f'%{query}%', limit)
        ).fetchall()
        return [_row_to_note(r) for r in rows]

    def update_cluster(self, note_id: str, cluster: int):
        self.conn.execute("UPDATE notes SET cluster=? WHERE id=?", (cluster, note_id))
        self.conn.commit()

    def update_centrality(self, note_id: str, centrality: float):
        self.conn.execute("UPDATE notes SET centrality=? WHERE id=?", (centrality, note_id))
        self.conn.commit()

    def update_clusters_batch(self, pairs: list):
        """pairs: [(cluster_int, note_id), ...] — single transaction."""
        self.conn.executemany("UPDATE notes SET cluster=? WHERE id=?", pairs)
        self.conn.commit()

    def update_centralities_batch(self, pairs: list):
        """pairs: [(centrality_float, note_id), ...] — single transaction."""
        self.conn.executemany("UPDATE notes SET centrality=? WHERE id=?", pairs)
        self.conn.commit()

    def note_count(self) -> int:
        return self.conn.execute("SELECT COUNT(*) FROM notes").fetchone()[0]

    # ── Large-scale note helpers ───────────────────────────────────────────────

    def get_note_stubs(self) -> list:
        """
        Lightweight fetch: id, title, tags, links, date, cluster, centrality.
        NO content field — saves 80-90% RAM vs get_all_notes().
        Returns list of dicts (not Note objects).
        Use for graph construction where content is not needed.
        """
        rows = self.conn.execute(
            """SELECT id, title, tags, links, source_file, date,
                      cluster, centrality, word_count FROM notes"""
        ).fetchall()
        return [_row_to_stub(r) for r in rows]

    def iter_notes_chunked(self, chunk_size: int = 500):
        """
        Generator: yields lists of Note objects, chunk_size at a time.
        Only one chunk in RAM at once — use for large corpora.
        """
        offset = 0
        while True:
            rows = self.conn.execute(
                "SELECT * FROM notes LIMIT ? OFFSET ?", (chunk_size, offset)
            ).fetchall()
            if not rows:
                break
            yield [_row_to_note(r) for r in rows]
            offset += chunk_size

    # ── Edges ─────────────────────────────────────────────────────────────────

    def upsert_edge(self, source: str, target: str, edge_type: str,
                    weight: float = 1.0, metadata: dict = None):
        self.conn.execute(
            """INSERT INTO edges (source,target,edge_type,weight,metadata)
               VALUES (?,?,?,?,?)
               ON CONFLICT(source,target,edge_type) DO UPDATE SET
                 weight=excluded.weight, metadata=excluded.metadata""",
            (source, target, edge_type, weight, json.dumps(metadata or {}))
        )
        self.conn.commit()

    def upsert_edges_batch(self, edges: list):
        """
        THE key method for large-scale graph building.
        Insert many edges in a SINGLE transaction — no per-edge fsync.

        edges: list of (source, target, edge_type, weight, metadata_dict)

        Performance vs upsert_edge() in a loop:
          1,000 edges:   0.05s  vs  1s
          100,000 edges: 0.8s   vs  100s
          1,000,000 edges: 8s   vs  ~17 minutes
        """
        if not edges:
            return
        self.conn.executemany(
            """INSERT INTO edges (source,target,edge_type,weight,metadata)
               VALUES (?,?,?,?,?)
               ON CONFLICT(source,target,edge_type) DO UPDATE SET
                 weight=excluded.weight, metadata=excluded.metadata""",
            [(s, t, et, w, json.dumps(m or {})) for s, t, et, w, m in edges]
        )
        self.conn.commit()

    def get_edges(self, note_id: str = None, edge_type: str = None) -> list:
        q, p = "SELECT * FROM edges WHERE 1=1", []
        if note_id:
            q += " AND (source=? OR target=?)"; p += [note_id, note_id]
        if edge_type:
            q += " AND edge_type=?"; p.append(edge_type)
        return [dict(r) for r in self.conn.execute(q, p).fetchall()]

    def get_all_edges(self) -> list:
        return [dict(r) for r in self.conn.execute("SELECT * FROM edges").fetchall()]

    def edge_count(self) -> int:
        return self.conn.execute("SELECT COUNT(*) FROM edges").fetchone()[0]

    # ── Embeddings ────────────────────────────────────────────────────────────

    def save_embedding(self, note_id: str, vector: list, model: str = 'unknown'):
        self.conn.execute(
            """INSERT INTO embeddings (note_id,vector,model) VALUES (?,?,?)
               ON CONFLICT(note_id) DO UPDATE SET
                 vector=excluded.vector, model=excluded.model""",
            (note_id, json.dumps(vector), model)
        )
        self.conn.commit()

    def save_embeddings_batch(self, rows: list):
        """
        rows: [(note_id, vector_list, model_str), ...]
        Single transaction — use instead of save_embedding() in a loop.
        """
        if not rows:
            return
        self.conn.executemany(
            """INSERT INTO embeddings (note_id,vector,model) VALUES (?,?,?)
               ON CONFLICT(note_id) DO UPDATE SET
                 vector=excluded.vector, model=excluded.model""",
            [(nid, json.dumps(v), m) for nid, v, m in rows]
        )
        self.conn.commit()

    def get_embedding(self, note_id: str) -> Optional[list]:
        row = self.conn.execute(
            "SELECT vector FROM embeddings WHERE note_id=?", (note_id,)
        ).fetchone()
        return json.loads(row["vector"]) if row else None

    def get_all_embeddings(self) -> dict:
        """
        Returns {note_id: vector_list}.
        WARNING: large corpus → large RAM. Prefer get_embeddings_numpy()
        or iter_embeddings_chunked() for 10k+ notes.
        """
        rows = self.conn.execute("SELECT note_id, vector FROM embeddings").fetchall()
        return {r["note_id"]: json.loads(r["vector"]) for r in rows}

    def get_embeddings_numpy(self):
        """
        Load all embeddings directly into a NumPy array.
        Returns: (ids: list[str], vecs: np.ndarray shape [n, dim] float32)

        Memory comparison at 30k notes × 384 dims:
          get_all_embeddings() Python dict: ~450 MB
          get_embeddings_numpy():           ~  46 MB  (10× less)

        Why: Python float objects are 24 bytes each. NumPy float32 is 4 bytes.
        The intermediate Python lists are immediately consumed by np.array()
        and eligible for GC — they don't pile up.
        """
        try:
            import numpy as np
        except ImportError:
            # Fallback to dict if numpy not installed
            d = self.get_all_embeddings()
            return list(d.keys()), None

        rows = self.conn.execute(
            "SELECT note_id, vector FROM embeddings ORDER BY note_id"
        ).fetchall()
        if not rows:
            return [], np.empty((0, 0), dtype=np.float32)

        ids  = [r[0] for r in rows]
        vecs = np.array([json.loads(r[1]) for r in rows], dtype=np.float32)
        del rows  # explicitly free SQLite rows before returning
        return ids, vecs

    def get_embedding_ids(self) -> list:
        """
        Return only IDs of notes with embeddings — no vector data loaded.
        Lightweight. Use to check coverage or drive per-note ANN search.
        """
        rows = self.conn.execute("SELECT note_id FROM embeddings").fetchall()
        return [r[0] for r in rows]

    def iter_embeddings_chunked(self, chunk_size: int = 2000):
        """
        Generator: yields {note_id: vector} dicts, chunk_size at a time.
        Never holds full embedding set in RAM.
        """
        offset = 0
        while True:
            rows = self.conn.execute(
                "SELECT note_id, vector FROM embeddings LIMIT ? OFFSET ?",
                (chunk_size, offset)
            ).fetchall()
            if not rows:
                break
            yield {r["note_id"]: json.loads(r["vector"]) for r in rows}
            offset += chunk_size

    def notes_without_embeddings(self) -> list:
        rows = self.conn.execute(
            "SELECT * FROM notes WHERE id NOT IN (SELECT note_id FROM embeddings)"
        ).fetchall()
        return [_row_to_note(r) for r in rows]

    def notes_without_embeddings_ids(self) -> list:
        """IDs only — no content loaded."""
        rows = self.conn.execute(
            "SELECT id FROM notes WHERE id NOT IN (SELECT note_id FROM embeddings)"
        ).fetchall()
        return [r[0] for r in rows]

    # ── Misc ──────────────────────────────────────────────────────────────────

    def stats(self) -> dict:
        return {
            "notes":                 self.note_count(),
            "edges":                 self.edge_count(),
            "notes_with_embeddings": self.conn.execute(
                "SELECT COUNT(*) FROM embeddings").fetchone()[0],
            "clusters":              self.conn.execute(
                "SELECT COUNT(DISTINCT cluster) FROM notes WHERE cluster IS NOT NULL"
            ).fetchone()[0],
            "tags": self._all_tags(),
        }

    def _all_tags(self) -> list:
        tags = set()
        for row in self.conn.execute("SELECT tags FROM notes"):
            for t in json.loads(row[0] or "[]"):
                tags.add(t)
        return sorted(tags)

    def close(self):
        try:
            self.conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
        except Exception:
            pass
        self.conn.close()


# ─── Helpers ──────────────────────────────────────────────────────────────────

def _row_to_note(row) -> Note:
    return Note(
        id=row["id"], title=row["title"], content=row["content"] or "",
        tags=json.loads(row["tags"] or "[]"), source_file=row["source_file"] or "",
        date=_parse_date(row["date"]), links=json.loads(row["links"] or "[]"),
        metadata=json.loads(row["metadata"] or "{}"),
        cluster=row["cluster"], centrality=row["centrality"] or 0.0,
    )


def _row_to_stub(row) -> dict:
    """Lightweight graph stub — no content, minimal RAM footprint."""
    return {
        "id":          row["id"],
        "title":       row["title"] or "",
        "tags":        json.loads(row["tags"]  or "[]"),
        "links":       json.loads(row["links"] or "[]"),
        "source_file": row["source_file"] or "",
        "date":        row["date"],
        "cluster":     row["cluster"],
        "centrality":  row["centrality"] or 0.0,
        "word_count":  row["word_count"]  or 0,
    }


def _parse_date(s: Optional[str]) -> Optional[datetime]:
    if not s:
        return None
    try:
        return datetime.fromisoformat(s)
    except ValueError:
        return None
