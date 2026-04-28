"""
Org File Parser
---------------
Handles three kinds of .org files:
  1. org-roam style  — file has #+title: and optionally an :ID: property
  2. Regular org     — top-level headings become separate notes
  3. Dump file       — fallback: whole file = one note

Links extracted:
  [[id:xxx][title]]          → ID-based roam link
  [[file:path.org][title]]   → file-based link
  [[file:path.org::*heading]]→ heading link
"""

import re
from pathlib import Path
from datetime import datetime
from typing import Optional

from .note import Note


# ─── Compiled regexes ────────────────────────────────────────────────────────

_TITLE_RE      = re.compile(r'#\+title:\s+(.+)',         re.IGNORECASE)
_FILETAGS_RE   = re.compile(r'#\+filetags:\s+(.+)',      re.IGNORECASE)
_DATE_RE       = re.compile(r'#\+date:\s+[\[<]?(.+?)[\]>]?$', re.IGNORECASE | re.MULTILINE)
_PROPERTY_ID   = re.compile(r':ID:\s+(\S+)')
_PROPERTIES    = re.compile(r':PROPERTIES:\n(.*?):END:', re.DOTALL)
_LINK_ID       = re.compile(r'\[\[id:([^\]]+)\]')
_LINK_FILE     = re.compile(r'\[\[file:([^:\]]+?)(?:\.org)?(?:::[^\]]+?)?\]')
_HEADING_RE    = re.compile(r'^(\*+)\s+(.+)$',           re.MULTILINE)
_HEADING_TAGS  = re.compile(r'\s+:([\w:]+):\s*$')
_DATE_FMTS     = ['%Y-%m-%d %a %H:%M', '%Y-%m-%d %a', '%Y-%m-%d %H:%M', '%Y-%m-%d']


# ─── Parser ──────────────────────────────────────────────────────────────────

class OrgParser:
    """Parse .org files into lists of Note objects."""

    def parse_file(self, path: Path) -> list:
        text = path.read_text(encoding='utf-8', errors='replace')

        file_title = _extract_title(text)

        if file_title:
            # org-roam style: whole file = one note
            return [self._parse_roam_note(text, path, file_title)]
        else:
            # multi-heading file: each heading = one note
            heading_notes = self._parse_heading_notes(text, path)
            if heading_notes:
                return heading_notes
            # fallback
            return [Note(
                id=Note.make_id(str(path)),
                title=path.stem.replace('_', ' ').replace('-', ' ').title(),
                content=text.strip(),
                source_file=str(path),
                links=_extract_links(text),
                metadata={"type": "raw_file"},
            )]

    def parse_directory(self, directory: Path, glob: str = "**/*.org") -> list:
        """Recursively parse all .org files in a directory."""
        notes = []
        for path in sorted(directory.glob(glob)):
            try:
                notes.extend(self.parse_file(path))
            except Exception as e:
                print(f"[parser] Warning: could not parse {path}: {e}")
        return notes

    # ── private ──────────────────────────────────────────────────────────────

    def _parse_roam_note(self, text: str, path: Path, title: str) -> Note:
        props_block = _extract_properties_block(text)
        note_id = None
        if props_block:
            m = _PROPERTY_ID.search(props_block)
            if m:
                note_id = m.group(1).strip()
        note_id = note_id or Note.make_id(str(path))

        return Note(
            id=note_id,
            title=title,
            content=_strip_frontmatter(text),
            tags=_extract_tags(text),
            source_file=str(path),
            date=_extract_date(text),
            links=_extract_links(text),
            metadata={"type": "roam_note", "file": str(path)},
        )

    def _parse_heading_notes(self, text: str, path: Path) -> list:
        """Split file on top-level (*) headings. Returns empty list if none found."""
        # Split on lines that begin exactly one star
        chunks = re.split(r'(?m)^(?=\* )', text)
        notes = []

        for chunk in chunks:
            chunk = chunk.strip()
            if not chunk:
                continue
            m = re.match(r'^\*+\s+(.+)$', chunk, re.MULTILINE)
            if not m:
                continue

            raw_heading = m.group(1)
            tags = []
            tag_m = _HEADING_TAGS.search(raw_heading)
            if tag_m:
                tags = [t for t in tag_m.group(1).split(':') if t]
                raw_heading = raw_heading[:tag_m.start()].strip()

            # Strip TODO keywords
            raw_heading = re.sub(r'^(TODO|DONE|WAIT|CANCELLED)\s+', '', raw_heading)

            props_block = _extract_properties_block(chunk)
            note_id = None
            if props_block:
                id_m = _PROPERTY_ID.search(props_block)
                if id_m:
                    note_id = id_m.group(1).strip()
            note_id = note_id or Note.make_id(str(path) + raw_heading)

            body = _strip_heading_and_properties(chunk)

            notes.append(Note(
                id=note_id,
                title=raw_heading,
                content=body,
                tags=tags,
                source_file=str(path),
                date=_extract_date(chunk),
                links=_extract_links(chunk),
                metadata={"type": "heading", "file": str(path)},
            ))

        return notes


# ─── Helper functions ─────────────────────────────────────────────────────────

def _extract_title(text: str) -> Optional[str]:
    m = _TITLE_RE.search(text)
    return m.group(1).strip() if m else None


def _extract_tags(text: str) -> list:
    m = _FILETAGS_RE.search(text)
    if not m:
        return []
    return [t for t in m.group(1).strip().strip(':').split(':') if t]


def _extract_date(text: str) -> Optional[datetime]:
    m = _DATE_RE.search(text)
    if not m:
        return None
    raw = m.group(1).strip()
    for fmt in _DATE_FMTS:
        try:
            return datetime.strptime(raw[:len(fmt)], fmt)
        except ValueError:
            continue
    return None


def _extract_links(text: str) -> list:
    ids = _LINK_ID.findall(text)
    files = _LINK_FILE.findall(text)
    return list(dict.fromkeys(ids + files))   # dedup, preserve order


def _extract_properties_block(text: str) -> Optional[str]:
    m = _PROPERTIES.search(text)
    return m.group(1) if m else None


def _strip_frontmatter(text: str) -> str:
    """Remove #+keyword lines and :PROPERTIES: blocks at the top."""
    text = _PROPERTIES.sub('', text)
    lines = text.split('\n')
    body_lines = []
    skip = True
    for line in lines:
        if skip and (line.startswith('#+') or line.strip() in ('', ':PROPERTIES:', ':END:')):
            continue
        skip = False
        body_lines.append(line)
    return '\n'.join(body_lines).strip()


def _strip_heading_and_properties(chunk: str) -> str:
    """Remove the heading line and :PROPERTIES: block from a heading chunk."""
    chunk = _PROPERTIES.sub('', chunk)
    chunk = re.sub(r'^(\*+)\s+.+\n?', '', chunk, count=1, flags=re.MULTILINE)
    return chunk.strip()