"""
Atomic Note — the fundamental unit of the digital brain.
Every note has an ID, content, metadata, and links to other notes.
"""

from dataclasses import dataclass, field
from typing import Optional
from datetime import datetime
import hashlib


@dataclass
class Note:
    id: str                                  # Stable unique ID (from org :ID: or hash)
    title: str
    content: str                             # Raw body text (org syntax stripped or kept)
    tags: list = field(default_factory=list)
    source_file: str = ""
    date: Optional[datetime] = None
    links: list = field(default_factory=list)   # IDs / file paths this note links to
    metadata: dict = field(default_factory=dict)

    # Set after embedding/graph pass
    embedding: Optional[list] = None
    cluster: Optional[int] = None
    centrality: float = 0.0

    def short_content(self, max_chars: int = 300) -> str:
        """First N chars of content, for preview."""
        c = self.content.strip()
        return c[:max_chars] + ("…" if len(c) > max_chars else "")

    def word_count(self) -> int:
        return len(self.content.split())

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "title": self.title,
            "content": self.content,
            "tags": self.tags,
            "source_file": self.source_file,
            "date": self.date.isoformat() if self.date else None,
            "links": self.links,
            "metadata": self.metadata,
            "cluster": self.cluster,
            "centrality": self.centrality,
            "word_count": self.word_count(),
        }

    @staticmethod
    def make_id(text: str) -> str:
        return hashlib.sha256(text.encode()).hexdigest()[:16]
