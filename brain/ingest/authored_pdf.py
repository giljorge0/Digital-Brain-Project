"""
Authored PDF Parser
-------------------
Drop-in replacement for ImportManager.parse_pdf_text() that handles PDFs
you WROTE yourself (essays, papers, lecture notes) rather than PDFs you
merely saved.

Key differences from the generic parse_pdf_text():
  - Scans recursively (glob **/*.pdf not just *.pdf)
  - Tags notes as "authored" so persona/gap/stylometry treat them as
    primary corpus, same as org notes
  - Attempts to extract a real title from the first page (H1 heuristic)
    rather than using the filename
  - Preserves folder structure as a sub-tag (e.g. pdfs/philosophy/ → "philosophy")
  - Splits very long PDFs into chapter-sized notes (> CHAPTER_SPLIT_WORDS words)
    so the graph gets atomic nodes rather than one giant blob per file

Usage (from importers.py or main.py):
    from brain.ingest.authored_pdf import parse_authored_pdfs
    notes = parse_authored_pdfs(Path("~/brain-data/authored-pdfs/"))

Or just call it in the ingest loop — it detects PDFs anywhere recursively.
"""

from __future__ import annotations

import logging
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from brain.ingest.note import Note

log = logging.getLogger("brain.authored_pdf")

CHAPTER_SPLIT_WORDS = 3_000   # split notes longer than this at heading boundaries
MIN_CONTENT_WORDS   = 30      # skip pages/chunks with fewer words than this


def parse_authored_pdfs(directory: Path) -> list:
    """
    Recursively parse all PDFs under `directory` as authored output.
    Returns list[Note].
    """
    directory = Path(directory).expanduser()
    if not directory.exists():
        log.warning(f"Authored PDF directory not found: {directory}")
        return []

    extractor = _get_extractor()
    if extractor is None:
        log.warning(
            "No PDF library found. Install one:\n"
            "  pip install pymupdf          ← best quality\n"
            "  pip install pypdf            ← lighter\n"
            "  pip install pdfminer.six     ← fallback"
        )
        return []

    notes = []
    pdf_files = sorted(directory.rglob("*.pdf"))
    log.info(f"Found {len(pdf_files)} PDF(s) under {directory}")

    for pdf_path in pdf_files:
        try:
            file_notes = _parse_single_pdf(pdf_path, directory, extractor)
            notes.extend(file_notes)
        except Exception as e:
            log.warning(f"Skipping {pdf_path.name}: {e}")

    log.info(f"Authored PDFs: {len(pdf_files)} files → {len(notes)} notes")
    return notes


# ─── Single-file parser ───────────────────────────────────────────────────────

def _parse_single_pdf(pdf_path: Path, root: Path, extractor) -> list:
    pages = extractor(pdf_path)        # list of (page_num, text) tuples
    if not pages:
        return []

    full_text = "\n\n".join(text for _, text in pages)
    word_count = len(full_text.split())

    if word_count < MIN_CONTENT_WORDS:
        log.debug(f"Skipping {pdf_path.name} — too short ({word_count} words)")
        return []

    # Derive folder-based sub-tag (e.g. authored-pdfs/philosophy/kant.pdf → "philosophy")
    try:
        rel = pdf_path.relative_to(root)
        folder_tag = rel.parts[0] if len(rel.parts) > 1 else ""
        folder_tag = folder_tag.lower().replace(" ", "_").replace("-", "_")[:30]
    except ValueError:
        folder_tag = ""

    base_tags = ["authored", "pdf", "output"]
    if folder_tag and folder_tag not in ("authored_pdfs", "pdfs", ""):
        base_tags.append(folder_tag)

    # Try to extract a real title from the first page
    first_page_text = pages[0][1] if pages else ""
    title = _extract_title(first_page_text, pdf_path)

    # File modification date as proxy for write date
    mtime = datetime.fromtimestamp(pdf_path.stat().st_mtime, tz=timezone.utc)

    # Decide: single note or split into sections
    if word_count <= CHAPTER_SPLIT_WORDS:
        return [Note(
            id=Note.make_id(str(pdf_path)),
            title=title,
            content=full_text.strip(),
            tags=base_tags,
            source_file=str(pdf_path),
            date=mtime,
            links=[],
            metadata={
                "type":       "authored_pdf",
                "pages":      len(pages),
                "word_count": word_count,
                "file":       str(pdf_path),
            },
        )]
    else:
        return _split_into_sections(full_text, title, pdf_path, base_tags, mtime)


# ─── Section splitter ─────────────────────────────────────────────────────────

def _split_into_sections(full_text: str, base_title: str, pdf_path: Path,
                          tags: list, mtime: datetime) -> list:
    """
    Split a long PDF into atomic notes at heading boundaries.
    Headings detected: lines that are ALL CAPS, or match "Chapter N", or
    are short lines (< 8 words) followed by a blank line.
    """
    heading_re = re.compile(
        r'^(?:'
        r'(?:Chapter|Section|Part)\s+\d+[:\.]?\s*.+|'  # Chapter 1: Title
        r'[A-Z][A-Z\s]{3,50}$|'                         # ALL CAPS HEADING
        r'(?:\d+[\.\)]\s+)[A-Z].{3,60}$'               # 1. Numbered heading
        r')',
        re.MULTILINE,
    )

    # Find all heading positions
    splits = [(0, base_title)]
    for m in heading_re.finditer(full_text):
        heading = m.group(0).strip()
        if len(heading.split()) <= 10:  # sanity check: real headings are short
            splits.append((m.start(), heading))

    if len(splits) <= 1:
        # No headings found — split by word count windows
        return _split_by_word_count(full_text, base_title, pdf_path, tags, mtime)

    notes = []
    for i, (start, heading) in enumerate(splits):
        end = splits[i+1][0] if i+1 < len(splits) else len(full_text)
        chunk = full_text[start:end].strip()
        wc = len(chunk.split())
        if wc < MIN_CONTENT_WORDS:
            continue

        section_title = f"{base_title} — {heading}" if heading != base_title else base_title
        notes.append(Note(
            id=Note.make_id(str(pdf_path) + heading + str(i)),
            title=section_title,
            content=chunk,
            tags=tags,
            source_file=str(pdf_path),
            date=mtime,
            links=[],
            metadata={
                "type":       "authored_pdf_section",
                "section":    heading,
                "word_count": wc,
                "file":       str(pdf_path),
                "section_idx":i,
            },
        ))

    return notes


def _split_by_word_count(full_text: str, title: str, pdf_path: Path,
                          tags: list, mtime: datetime) -> list:
    """Fallback: split on word count windows when no headings detected."""
    words = full_text.split()
    chunks = []
    for i in range(0, len(words), CHAPTER_SPLIT_WORDS):
        chunk = " ".join(words[i:i + CHAPTER_SPLIT_WORDS])
        chunks.append(chunk)

    notes = []
    for i, chunk in enumerate(chunks):
        notes.append(Note(
            id=Note.make_id(str(pdf_path) + str(i)),
            title=f"{title} (part {i+1}/{len(chunks)})",
            content=chunk,
            tags=tags,
            source_file=str(pdf_path),
            date=mtime,
            links=[],
            metadata={
                "type":       "authored_pdf_section",
                "part":       i+1,
                "total_parts":len(chunks),
                "word_count": len(chunk.split()),
                "file":       str(pdf_path),
            },
        ))
    return notes


# ─── Title extraction ─────────────────────────────────────────────────────────

def _extract_title(first_page: str, fallback_path: Path) -> str:
    """
    Try to find the document title from the first page.
    Heuristic: first non-empty line that's < 12 words and doesn't look like
    an author name or date.
    """
    lines = [l.strip() for l in first_page.split("\n") if l.strip()]
    for line in lines[:8]:
        words = line.split()
        if (2 <= len(words) <= 12
                and not re.match(r'^\d{4}', line)       # not a date
                and not re.match(r'^by\b', line, re.I)  # not "by Author"
                and not re.match(r'^page\b', line, re.I)):
            return line
    # Fallback: filename
    return fallback_path.stem.replace("_", " ").replace("-", " ").title()


# ─── PDF library detection ────────────────────────────────────────────────────

def _get_extractor():
    """
    Returns a callable: pdf_path → list of (page_num, text).
    Tries libraries in order of quality.
    """
    # 1. PyMuPDF (best layout preservation)
    try:
        import fitz
        def _fitz_extract(path: Path):
            doc = fitz.open(str(path))
            pages = [(i, page.get_text("text")) for i, page in enumerate(doc)]
            doc.close()
            return [(i, t) for i, t in pages if t.strip()]
        log.debug("PDF extractor: PyMuPDF (fitz)")
        return _fitz_extract
    except ImportError:
        pass

    # 2. pypdf (lighter)
    try:
        from pypdf import PdfReader
        def _pypdf_extract(path: Path):
            reader = PdfReader(str(path))
            return [(i, p.extract_text() or "")
                    for i, p in enumerate(reader.pages)
                    if (p.extract_text() or "").strip()]
        log.debug("PDF extractor: pypdf")
        return _pypdf_extract
    except ImportError:
        pass

    # 3. PyPDF2 (legacy)
    try:
        from PyPDF2 import PdfReader
        def _pypdf2_extract(path: Path):
            reader = PdfReader(str(path))
            return [(i, p.extract_text() or "")
                    for i, p in enumerate(reader.pages)
                    if (p.extract_text() or "").strip()]
        log.debug("PDF extractor: PyPDF2")
        return _pypdf2_extract
    except ImportError:
        pass

    # 4. pdfminer.six (slowest but most reliable for complex layouts)
    try:
        from pdfminer.high_level import extract_text as _pdfminer_extract
        def _pdfminer(path: Path):
            text = _pdfminer_extract(str(path))
            return [(0, text)] if text.strip() else []
        log.debug("PDF extractor: pdfminer.six")
        return _pdfminer
    except ImportError:
        pass

    return None
