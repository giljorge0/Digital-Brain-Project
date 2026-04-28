"""
SQLite Store
------------
Three tables:
  notes      — all parsed Note objects
  edges      — directed relationships between notes
  embeddings — float32 vectors stored as JSON blobs

Edge types:
  explicit   — a [[link]] in the note text
  tag        — shares one or more tags
  semantic   — cosine similarity > threshold
  llm        — extracted by the LLM relation extractor
"""

import json
import sqlite3
from pathlib import Path
from datetime import datetime
from typing import Optional

from ..ingest.note import Note


# ─── Schema ──────────────────────────────────────────────────────────────────

SCHEMA = """
CREATE TABLE IF NOT EXISTS notes (
    id           TEXT PRIMARY KEY,
    title        TEXT NOT NULL,
    content      TEXT,
    tags         TEXT,       -- JSON list
    source_file  TEXT,
    date         TEXT,
    links        TEXT,       -- JSON list
    metadata     TEXT,       -- JSON dict
    cluster      INTEGER,
    centrality   REAL DEFAULT 0.0,
    word_count   INTEGER DEFAULT 0,
    created_at   TEXT DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS edges (
    source       TEXT NOT NULL,
    target       TEXT NOT NULL,
    edge_type    TEXT NOT NULL,    -- explicit | tag | semantic | llm
    weight       REAL DEFAULT 1.0,
    metadata     TEXT,             -- JSON
    PRIMARY KEY (source, target, edge_type)
);

CREATE TABLE IF NOT EXISTS embeddings (
    note_id  TEXT PRIMARY KEY,
    vector   TEXT NOT NULL,        -- JSON float list
    model    TEXT DEFAULT 'unknown'
);

CREATE INDEX IF NOT EXISTS idx_notes_tags      ON notes(tags);
CREATE INDEX IF NOT EXISTS idx_notes_date      ON notes(date);
CREATE INDEX IF NOT EXISTS idx_edges_source    ON edges(source);
CREATE INDEX IF NOT EXISTS idx_edges_target    ON edges(target);
CREATE INDEX IF NOT EXISTS idx_edges_type      ON edges(edge_type);
"""


# ─── Store ────────────────────────────────────────────────────────────────────

class Store:
    def __init__(self, db_path: str | Path):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self.conn.executescript(SCHEMA)
        self.conn.commit()

    # ── Notes ─────────────────────────────────────────────────────────────────

    def upsert_note(self, note: Note):
        self.conn.execute(
            """
            INSERT INTO notes (id, title, content, tags, source_file, date,
                               links, metadata, cluster, centrality, word_count)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(id) DO UPDATE SET
                title       = excluded.title,
                content     = excluded.content,
                tags        = excluded.tags,
                source_file = excluded.source_file,
                date        = excluded.date,
                links       = excluded.links,
                metadata    = excluded.metadata,
                cluster     = excluded.cluster,
                centrality  = excluded.centrality,
                word_count  = excluded.word_count
            """,
            (
                note.id, note.title, note.content,
                json.dumps(note.tags),
                note.source_file,
                note.date.isoformat() if note.date else None,
                json.dumps(note.links),
                json.dumps(note.metadata),
                note.cluster, note.centrality, note.word_count(),
            )
        )
        self.conn.commit()

    def upsert_notes(self, notes: list):
        for note in notes:
            self.upsert_note(note)

    def get_note(self, note_id: str) -> Optional[Note]:
        row = self.conn.execute(
            "SELECT * FROM notes WHERE id = ?", (note_id,)
        ).fetchone()
        return _row_to_note(row) if row else None

    def get_all_notes(self) -> list:
        rows = self.conn.execute("SELECT * FROM notes").fetchall()
        return [_row_to_note(r) for r in rows]

    def get_notes_by_tag(self, tag: str) -> list:
        rows = self.conn.execute(
            "SELECT * FROM notes WHERE tags LIKE ?", (f'%"{tag}"%',)
        ).fetchall()
        return [_row_to_note(r) for r in rows]

    def search_notes(self, query: str, limit: int = 20) -> list:
        rows = self.conn.execute(
            """SELECT * FROM notes
               WHERE title LIKE ? OR content LIKE ?
               LIMIT ?""",
            (f'%{query}%', f'%{query}%', limit)
        ).fetchall()
        return [_row_to_note(r) for r in rows]

    def update_cluster(self, note_id: str, cluster: int):
        self.conn.execute(
            "UPDATE notes SET cluster = ? WHERE id = ?", (cluster, note_id)
        )
        self.conn.commit()

    def update_centrality(self, note_id: str, centrality: float):
        self.conn.execute(
            "UPDATE notes SET centrality = ? WHERE id = ?", (centrality, note_id)
        )
        self.conn.commit()

    def note_count(self) -> int:
        return self.conn.execute("SELECT COUNT(*) FROM notes").fetchone()[0]

    # ── Edges ─────────────────────────────────────────────────────────────────

    def upsert_edge(self, source: str, target: str, edge_type: str,
                    weight: float = 1.0, metadata: dict = None):
        self.conn.execute(
            """
            INSERT INTO edges (source, target, edge_type, weight, metadata)
            VALUES (?, ?, ?, ?, ?)
            ON CONFLICT(source, target, edge_type) DO UPDATE SET
                weight   = excluded.weight,
                metadata = excluded.metadata
            """,
            (source, target, edge_type, weight, json.dumps(metadata or {}))
        )
        self.conn.commit()

    def get_edges(self, note_id: str = None, edge_type: str = None) -> list:
        query = "SELECT * FROM edges WHERE 1=1"
        params = []
        if note_id:
            query += " AND (source = ? OR target = ?)"
            params += [note_id, note_id]
        if edge_type:
            query += " AND edge_type = ?"
            params.append(edge_type)
        rows = self.conn.execute(query, params).fetchall()
        return [dict(r) for r in rows]

    def get_all_edges(self) -> list:
        return [dict(r) for r in self.conn.execute("SELECT * FROM edges").fetchall()]

    def edge_count(self) -> int:
        return self.conn.execute("SELECT COUNT(*) FROM edges").fetchone()[0]

    # ── Embeddings ────────────────────────────────────────────────────────────

    def save_embedding(self, note_id: str, vector: list, model: str = 'unknown'):
        self.conn.execute(
            """
            INSERT INTO embeddings (note_id, vector, model)
            VALUES (?, ?, ?)
            ON CONFLICT(note_id) DO UPDATE SET vector = excluded.vector, model = excluded.model
            """,
            (note_id, json.dumps(vector), model)
        )
        self.conn.commit()

    def get_embedding(self, note_id: str) -> Optional[list]:
        row = self.conn.execute(
            "SELECT vector FROM embeddings WHERE note_id = ?", (note_id,)
        ).fetchone()
        return json.loads(row["vector"]) if row else None

    def get_all_embeddings(self) -> dict:
        """Returns {note_id: vector} for all notes that have embeddings."""
        rows = self.conn.execute("SELECT note_id, vector FROM embeddings").fetchall()
        return {r["note_id"]: json.loads(r["vector"]) for r in rows}

    def notes_without_embeddings(self) -> list:
        rows = self.conn.execute(
            """SELECT * FROM notes WHERE id NOT IN
               (SELECT note_id FROM embeddings)"""
        ).fetchall()
        return [_row_to_note(r) for r in rows]

    # ── Misc ──────────────────────────────────────────────────────────────────

    def stats(self) -> dict:
        return {
            "notes": self.note_count(),
            "edges": self.edge_count(),
            "notes_with_embeddings": self.conn.execute(
                "SELECT COUNT(*) FROM embeddings"
            ).fetchone()[0],
            "clusters": self.conn.execute(
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
        self.conn.close()


# ─── Helpers ──────────────────────────────────────────────────────────────────

def _row_to_note(row) -> Note:
    return Note(
        id=row["id"],
        title=row["title"],
        content=row["content"] or "",
        tags=json.loads(row["tags"] or "[]"),
        source_file=row["source_file"] or "",
        date=_parse_date(row["date"]),
        links=json.loads(row["links"] or "[]"),
        metadata=json.loads(row["metadata"] or "{}"),
        cluster=row["cluster"],
        centrality=row["centrality"] or 0.0,
    )


def _parse_date(s: Optional[str]) -> Optional[datetime]:
    if not s:
        return None
    try:
        return datetime.fromisoformat(s)
    except ValueError:
        return None