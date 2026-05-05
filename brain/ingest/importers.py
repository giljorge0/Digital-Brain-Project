"""
Import Manager
--------------
Handles all passive data ingestion into Note objects.

CRITICAL DESIGN PRINCIPLE:
  Every note carries a `provenance_role` in its metadata:
    "output"  — YOUR words: org notes, your turns in chats, your writing
    "input"   — EXTERNAL content: AI responses, YouTube, books, web, PDFs

This lets the persona/makemore module train only on YOUR outputs.

Supported formats:
  ChatGPT export  — conversations.json (OpenAI export format)
  Claude export   — claude_export.json (Anthropic export format)
  Markdown clips  — .md files (web clips, Readwise, etc.)
  PDF text        — via pdfminer or pymupdf
  YouTube history — watch-history.json (Google Takeout)
  Search history  — MyActivity.json (Google Takeout) or browser JSON
  Generic JSON    — fallback chat format
"""

import json
import re
import logging
from pathlib import Path
from datetime import datetime, timezone
from typing import Optional

from brain.ingest.note import Note

log = logging.getLogger(__name__)


# ─── Role constants ───────────────────────────────────────────────────────────
ROLE_OUTPUT = "output"   # Your words — feeds persona/makemore
ROLE_INPUT  = "input"    # External content — feeds knowledge graph context


# ─── Import Manager ──────────────────────────────────────────────────────────

class ImportManager:

    # ── ChatGPT / OpenAI export ───────────────────────────────────────────────

    @staticmethod
    def parse_chatgpt_export(path: Path) -> list:
        """
        Parse OpenAI's conversations.json export.
        Each conversation becomes two note streams:
          - Your messages   → provenance_role: "output"
          - GPT messages    → provenance_role: "input"
        """
        if not path.exists():
            return []

        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except Exception as e:
            log.error(f"[import] Failed to parse {path}: {e}")
            return []

        notes = []
        conversations = data if isinstance(data, list) else data.get("conversations", [])

        for conv in conversations:
            conv_id   = conv.get("id", "")
            conv_title = conv.get("title", "Untitled Chat")
            created   = conv.get("create_time") or conv.get("created_at")
            conv_date = _parse_unix_or_iso(created)

            mapping = conv.get("mapping", {})
            # Collect messages in order
            messages = _extract_chatgpt_messages(mapping)

            for i, msg in enumerate(messages):
                role    = msg.get("role", "")
                content = msg.get("content", "").strip()
                ts      = msg.get("create_time")
                if not content:
                    continue

                is_user = role == "user"
                note_id = Note.make_id(f"{conv_id}_{i}_{role}")

                notes.append(Note(
                    id=note_id,
                    title=f"{conv_title} [{role}]" if not is_user else f"{conv_title} [you]",
                    content=content,
                    source_file=str(path),
                    date=_parse_unix_or_iso(ts) or conv_date,
                    metadata={
                        "type": "llm_chat",
                        "platform": "chatgpt",
                        "conversation_id": conv_id,
                        "conversation_title": conv_title,
                        "role": role,
                        "provenance_role": ROLE_OUTPUT if is_user else ROLE_INPUT,
                        "turn_index": i,
                    }
                ))

        log.info(f"[import] ChatGPT: {len(notes)} turns from {path.name}")
        return notes



    @staticmethod
    def parse_firefox_sqlite(db_path) -> list:
        """Extract history from Firefox/LibreWolf/Iceraven places.sqlite"""
        import sqlite3
        from datetime import datetime, timezone
        from brain.ingest.note import Note
        
        notes = []
        try:
            # Connect in read-only mode to prevent locking the database
            uri = f"file:{db_path.absolute()}?mode=ro"
            conn = sqlite3.connect(uri, uri=True)
            cursor = conn.cursor()

            # Grab the last 5000 visited pages (adjust limit if you want more)
            query = """
                SELECT p.url, p.title, h.visit_date
                FROM moz_historyvisits h
                JOIN moz_places p ON h.place_id = p.id
                WHERE p.title IS NOT NULL
                ORDER BY h.visit_date DESC
                LIMIT 5000;
            """
            cursor.execute(query)
            rows = cursor.fetchall()

            for url, title, visit_date in rows:
                if not url.startswith("http"): continue

                # Firefox stores dates in microseconds since 1970
                try:
                    dt = datetime.fromtimestamp(visit_date / 1000000.0, tz=timezone.utc)
                except:
                    dt = datetime.now(timezone.utc)

                notes.append(Note(
                    id=Note.make_id(url),
                    title=f"Web: {title[:100]}",
                    content=f"Visited URL: {url}\n\nTitle: {title}",
                    tags=["browser_history"],
                    date=dt,
                    metadata={"url": url, "source": "firefox_sqlite"}
                ))

            conn.close()
            print(f"  ✓ Extracted {len(notes)} history items from {db_path.name}")
        except Exception as e:
            print(f"  ! Failed to parse SQLite {db_path.name}: {e}")

        return notes

        
    # ── Claude / Anthropic export ─────────────────────────────────────────────

    @staticmethod
    def parse_claude_export(path: Path) -> list:
        """
        Parse Anthropic's claude_export.json or conversations.json.
        Format: list of {uuid, name, created_at, updated_at, chat_messages: [...]}
        """
        if not path.exists():
            return []

        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except Exception as e:
            log.error(f"[import] Claude export parse failed: {e}")
            return []

        notes = []
        conversations = data if isinstance(data, list) else [data]

        for conv in conversations:
            conv_id    = conv.get("uuid", conv.get("id", ""))
            conv_title = conv.get("name", conv.get("title", "Claude Chat"))
            conv_date  = _parse_unix_or_iso(conv.get("created_at"))

            messages = conv.get("chat_messages", conv.get("messages", []))

            for i, msg in enumerate(messages):
                role    = msg.get("sender", msg.get("role", ""))
                # Claude exports may nest content in list of blocks
                content = _extract_claude_content(msg)
                if not content.strip():
                    continue

                is_human = role in ("human", "user")
                note_id  = Note.make_id(f"claude_{conv_id}_{i}_{role}")

                notes.append(Note(
                    id=note_id,
                    title=f"{conv_title} [{'you' if is_human else 'claude'}]",
                    content=content.strip(),
                    source_file=str(path),
                    date=_parse_unix_or_iso(msg.get("created_at")) or conv_date,
                    metadata={
                        "type": "llm_chat",
                        "platform": "claude",
                        "conversation_id": conv_id,
                        "conversation_title": conv_title,
                        "role": role,
                        "provenance_role": ROLE_OUTPUT if is_human else ROLE_INPUT,
                        "turn_index": i,
                    }
                ))

        log.info(f"[import] Claude: {len(notes)} turns from {path.name}")
        return notes

    # ── Generic LLM chat (auto-detect) ────────────────────────────────────────

    @staticmethod
    def parse_llm_chats(path: Path) -> list:
        """Auto-detect ChatGPT or Claude export format."""
        if not path.exists() or path.suffix != ".json":
            return []

        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return []

        # Sniff format
        sample = data[0] if isinstance(data, list) and data else data
        if isinstance(sample, dict):
            if "mapping" in sample or "conversation_id" in sample:
                return ImportManager.parse_chatgpt_export(path)
            if "chat_messages" in sample or "sender" in str(sample)[:300]:
                return ImportManager.parse_claude_export(path)
            # Fallback: treat as a simple [{role, content, timestamp}] list
            return ImportManager._parse_simple_chat(data, path)

        return []

    @staticmethod
    def _parse_simple_chat(data: list, path: Path) -> list:
        """Fallback: [{role, content, timestamp?}]"""
        notes = []
        for i, msg in enumerate(data):
            if not isinstance(msg, dict):
                continue
            role    = msg.get("role", "unknown")
            content = msg.get("content", msg.get("text", "")).strip()
            if not content:
                continue
            is_user = role in ("user", "human", "you", "me")
            notes.append(Note(
                id=Note.make_id(f"{path.stem}_{i}_{role}"),
                title=f"{path.stem} [{role}]",
                content=content,
                source_file=str(path),
                date=_parse_unix_or_iso(msg.get("timestamp") or msg.get("created_at")),
                metadata={
                    "type": "llm_chat",
                    "platform": "unknown",
                    "role": role,
                    "provenance_role": ROLE_OUTPUT if is_user else ROLE_INPUT,
                }
            ))
        return notes

    # ── YouTube watch history ─────────────────────────────────────────────────

    @staticmethod
    def parse_youtube_history(path: Path) -> list:
        """
        Parse Google Takeout YouTube watch-history.json.
        Format: [{titleUrl, title, time, subtitles:[{name}]}]
        All YouTube content is INPUT (not your words).
        """
        if not path.exists():
            return []

        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except Exception as e:
            log.error(f"[import] YouTube history parse failed: {e}")
            return []

        notes = []
        for item in data:
            if not isinstance(item, dict):
                continue
            title    = item.get("title", "").replace("Watched ", "")
            url      = item.get("titleUrl", "")
            ts       = item.get("time")
            channel  = ""
            subs = item.get("subtitles", [])
            if subs and isinstance(subs[0], dict):
                channel = subs[0].get("name", "")

            if not title:
                continue

            content = f"Watched: {title}"
            if channel:
                content += f"\nChannel: {channel}"
            if url:
                content += f"\nURL: {url}"

            notes.append(Note(
                id=Note.make_id(f"yt_{url}_{ts}"),
                title=title,
                content=content,
                source_file=str(path),
                date=_parse_unix_or_iso(ts),
                tags=["youtube", "watched"],
                metadata={
                    "type": "youtube_watch",
                    "url": url,
                    "channel": channel,
                    "provenance_role": ROLE_INPUT,
                }
            ))

        log.info(f"[import] YouTube: {len(notes)} items from {path.name}")
        return notes

    # ── Search history ────────────────────────────────────────────────────────

    @staticmethod
    def parse_search_history(path: Path) -> list:
        """
        Parse Google Takeout MyActivity.json (Search activity).
        All search queries are OUTPUT (your intent/questions).
        All visited pages are INPUT.
        """
        if not path.exists():
            return []

        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except Exception as e:
            log.error(f"[import] Search history parse failed: {e}")
            return []

        notes = []
        items = data if isinstance(data, list) else data.get("items", [])

        for item in items:
            if not isinstance(item, dict):
                continue
            title   = item.get("title", "")
            ts      = item.get("time")
            url     = item.get("titleUrl", "")
            product = item.get("header", "Search")

            # Google formats: "Searched for X" vs "Visited X"
            is_search = title.lower().startswith("searched for")
            query     = re.sub(r"^searched for\s+", "", title, flags=re.IGNORECASE)

            notes.append(Note(
                id=Note.make_id(f"search_{url}_{ts}"),
                title=title,
                content=f"Query: {query}\nURL: {url}\nProduct: {product}",
                source_file=str(path),
                date=_parse_unix_or_iso(ts),
                tags=["search", "activity"],
                metadata={
                    "type": "search_history",
                    "query": query,
                    "url": url,
                    "provenance_role": ROLE_OUTPUT if is_search else ROLE_INPUT,
                }
            ))

        log.info(f"[import] Search: {len(notes)} items from {path.name}")
        return notes

    # ── Markdown web clips ────────────────────────────────────────────────────

    @staticmethod
    def parse_web_clips(directory: Path) -> list:
        """
        Parse .md files as web clips / Readwise highlights / Obsidian imports.
        All web clip content is INPUT (not your words).
        Exception: if file has a 'my-note' or 'reflection' section, split it.
        """
        notes = []
        for md_file in sorted(directory.glob("**/*.md")):
            try:
                text = md_file.read_text(encoding="utf-8", errors="replace")
                # Extract YAML frontmatter
                fm, body = _parse_frontmatter(text)
                title = fm.get("title") or md_file.stem.replace("-", " ").replace("_", " ").title()
                date  = _parse_unix_or_iso(fm.get("date") or fm.get("created"))
                tags  = fm.get("tags", [])
                if isinstance(tags, str):
                    tags = [t.strip() for t in tags.split(",")]
                url   = fm.get("url", fm.get("source", ""))

                notes.append(Note(
                    id=Note.make_id(str(md_file)),
                    title=title,
                    content=body.strip(),
                    source_file=str(md_file),
                    date=date,
                    tags=tags,
                    links=[url] if url else [],
                    metadata={
                        "type": "web_clip",
                        "url": url,
                        "provenance_role": ROLE_INPUT,
                        **{k: v for k, v in fm.items()
                           if k not in ("title", "date", "tags", "url")},
                    }
                ))
            except Exception as e:
                log.warning(f"[import] Could not parse {md_file}: {e}")

        log.info(f"[import] Web clips: {len(notes)} .md files from {directory}")
        return notes

    # ── PDFs ──────────────────────────────────────────────────────────────────

    # ── PDFs ──────────────────────────────────────────────────────────────────

    @staticmethod
    def parse_pdf_text(directory: Path) -> list:
        """
        Extract text from PDF files using Hierarchical Chunking.
        Creates one Parent note for the document, and sequential Child notes for the chunks.
        All PDF content is INPUT.
        Tries pdfminer first, falls back to pymupdf.
        """
        notes = []
        for pdf_file in sorted(directory.glob("**/*.pdf")):
            text = _extract_pdf_text(pdf_file)
            if not text or len(text.strip()) < 50:
                continue

            title = pdf_file.stem.replace("_", " ").replace("-", " ").title()
            
            # Try to detect date from filename, fallback to file modification time
            date_m = re.search(r'(\d{4}[-_]\d{2}[-_]\d{2})', pdf_file.stem)
            if date_m:
                date = _parse_unix_or_iso(date_m.group(1))
            else:
                date = datetime.fromtimestamp(pdf_file.stat().st_mtime, tz=timezone.utc)

            # 1. Create the PARENT Note
            parent_id = Note.make_id(f"pdf_parent_{pdf_file.name}")
            parent_note = Note(
                id=parent_id,
                title=f"PDF: {title}",
                content=f"Full document container for: {pdf_file.name}",
                tags=["pdf_parent", "book", "document"],
                source_file=str(pdf_file),
                date=date,
                metadata={
                    "type": "pdf_parent",
                    "provenance_role": ROLE_INPUT,
                    "pages_extracted": text.count("\f") + 1,
                }
            )
            notes.append(parent_note)

            # 2. Chunk the text (Targeting ~250 words per chunk for atomic embedding)
            # Use regex to split cleanly on multiple newlines
            paragraphs = re.split(r'\n\s*\n', text)
            chunks = []
            current_chunk = ""
            
            for p in paragraphs:
                current_chunk += p.strip() + "\n\n"
                # If chunk exceeds ~250 words, save it and start a new one
                if len(current_chunk.split()) > 250:
                    chunks.append(current_chunk.strip())
                    current_chunk = ""
            
            # Catch the remainder
            if current_chunk.strip():
                chunks.append(current_chunk.strip())

            # 3. Create the CHILD Notes
            prev_child_id = None
            for i, chunk in enumerate(chunks):
                child_id = Note.make_id(f"pdf_chunk_{pdf_file.name}_{i}")
                
                # Graph Links: Link to parent, and link to the previous chunk for sequence
                links = [parent_id]
                if prev_child_id:
                    links.append(prev_child_id)
                    
                child_note = Note(
                    id=child_id,
                    title=f"{title} (Part {i+1}/{len(chunks)})",
                    content=chunk,
                    tags=["pdf_chunk", "essay_argument"],
                    source_file=str(pdf_file),
                    date=date,
                    links=links,
                    metadata={
                        "type": "pdf_chunk", 
                        "parent_id": parent_id, 
                        "sequence": i,
                        "provenance_role": ROLE_INPUT
                    }
                )
                notes.append(child_note)
                prev_child_id = child_id

            log.info(f"[import] Parsed PDF '{pdf_file.stem}' into 1 parent and {len(chunks)} chunks.")

        return notes
      
    # ── YouTube search history ────────────────────────────────────────────────

    @staticmethod
    def parse_youtube_search_history(path: Path) -> list:
        """
        Parse Google Takeout YouTube search-history.json.
        Your searches are OUTPUT (your intent), results are INPUT.
        """
        if not path.exists():
            return []
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except Exception as e:
            log.error(f"[import] YouTube search history failed: {e}")
            return []

        notes = []
        for item in data:
            if not isinstance(item, dict):
                continue
            title = item.get("title", "").replace("Searched for ", "")
            ts    = item.get("time")
            if not title:
                continue
            notes.append(Note(
                id=Note.make_id(f"ytsearch_{title}_{ts}"),
                title=f"YouTube search: {title}",
                content=f"Searched YouTube for: {title}",
                source_file=str(path),
                date=_parse_unix_or_iso(ts),
                tags=["youtube", "search"],
                metadata={"type": "youtube_search", "query": title,
                          "provenance_role": ROLE_OUTPUT},
            ))
        log.info(f"[import] YouTube search: {len(notes)} queries from {path.name}")
        return notes

    # ── Google search history ─────────────────────────────────────────────────

    @staticmethod
    def parse_google_search_history(path: Path) -> list:
        """Alias for parse_search_history — same Google Takeout format."""
        return ImportManager.parse_search_history(path)

    # ── Goodreads CSV export ──────────────────────────────────────────────────

    @staticmethod
    def parse_goodreads_csv(path: Path) -> list:
        """
        Parse Goodreads library export CSV.
        Books you've read are INPUT. Your shelf/ratings are OUTPUT signals.
        """
        if not path.exists():
            return []
        import csv
        notes = []
        try:
            with open(path, encoding="utf-8", errors="replace") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    title  = row.get("Title", "").strip()
                    author = row.get("Author", "").strip()
                    rating = row.get("My Rating", "0").strip()
                    shelf  = row.get("Exclusive Shelf", "").strip()
                    date_r = row.get("Date Read", "").strip()
                    review = row.get("My Review", "").strip()
                    if not title:
                        continue
                    content = f"Book: {title}\nAuthor: {author}\nShelf: {shelf}"
                    if rating and rating != "0":
                        content += f"\nRating: {rating}/5"
                    if review:
                        content += f"\nReview: {review}"
                    notes.append(Note(
                        id=Note.make_id(f"goodreads_{title}_{author}"),
                        title=f"{title} — {author}",
                        content=content,
                        source_file=str(path),
                        date=_parse_unix_or_iso(date_r),
                        tags=["book", "goodreads", shelf] if shelf else ["book", "goodreads"],
                        metadata={
                            "type": "book",
                            "author": author,
                            "rating": rating,
                            "shelf": shelf,
                            # The book content is INPUT; your review/rating is OUTPUT
                            "provenance_role": ROLE_OUTPUT if review else ROLE_INPUT,
                        },
                    ))
        except Exception as e:
            log.error(f"[import] Goodreads parse failed: {e}")
        log.info(f"[import] Goodreads: {len(notes)} books from {path.name}")
        return notes

    # ── Kindle clippings ──────────────────────────────────────────────────────

    @staticmethod
    def parse_kindle_clippings(path: Path) -> list:
        """
        Parse Kindle 'My Clippings.txt'.
        Highlights are INPUT (book text). Notes/annotations are OUTPUT (your words).
        Groups clippings by book title.
        """
        if not path.exists():
            return []
        try:
            raw = path.read_text(encoding="utf-8", errors="replace")
        except Exception as e:
            log.error(f"[import] Kindle clippings read failed: {e}")
            return []

        # Kindle separator
        entries = raw.split("==========")
        books: dict = {}

        for entry in entries:
            lines = [l.strip() for l in entry.strip().split("\n") if l.strip()]
            if len(lines) < 2:
                continue
            book_line = lines[0]
            # Book title is first line: "Title (Author)" or just "Title"
            book_title = book_line.split("(")[0].strip() if "(" in book_line else book_line

            # Metadata line contains type: "Your Highlight" or "Your Note"
            meta_line = lines[1] if len(lines) > 1 else ""
            content_lines = lines[2:] if len(lines) > 2 else []
            content = " ".join(content_lines).strip()
            if not content:
                continue

            is_note = "your note" in meta_line.lower()
            role = ROLE_OUTPUT if is_note else ROLE_INPUT

            if book_title not in books:
                books[book_title] = {"highlights": [], "notes": [], "role": ROLE_INPUT}
            if is_note:
                books[book_title]["notes"].append(content)
            else:
                books[book_title]["highlights"].append(content)

        notes = []
        for book_title, data in books.items():
            all_text = ""
            if data["highlights"]:
                all_text += "## Highlights\n" + "\n\n".join(
                    f"> {h}" for h in data["highlights"][:50]
                )
            if data["notes"]:
                all_text += "\n\n## Your Notes\n" + "\n\n".join(data["notes"])

            if not all_text.strip():
                continue

            # If the book has your notes, mark as output; pure highlights = input
            role = ROLE_OUTPUT if data["notes"] else ROLE_INPUT

            notes.append(Note(
                id=Note.make_id(f"kindle_{book_title}"),
                title=f"Kindle: {book_title}",
                content=all_text.strip(),
                source_file=str(path),
                tags=["kindle", "book", "highlights"],
                metadata={
                    "type": "kindle_clippings",
                    "book_title": book_title,
                    "highlight_count": len(data["highlights"]),
                    "note_count": len(data["notes"]),
                    "provenance_role": role,
                },
            ))

        log.info(f"[import] Kindle: {len(notes)} books from {path.name}")
        return notes


# ─── Helpers ──────────────────────────────────────────────────────────────────

def _extract_chatgpt_messages(mapping: dict) -> list:
    """Walk the GPT message tree in order."""
    # Find root
    root = None
    for node in mapping.values():
        if node.get("parent") is None:
            root = node
            break
    if root is None and mapping:
        root = next(iter(mapping.values()))

    messages = []
    current  = root

    while current:
        msg = current.get("message")
        if msg:
            role    = msg.get("author", {}).get("role", "")
            parts   = msg.get("content", {}).get("parts", [])
            content = " ".join(str(p) for p in parts if isinstance(p, str))
            if content.strip() and role in ("user", "assistant"):
                messages.append({
                    "role": role,
                    "content": content,
                    "create_time": msg.get("create_time"),
                })

        children = current.get("children", [])
        if children:
            current = mapping.get(children[0])
        else:
            break

    return messages


def _extract_claude_content(msg: dict) -> str:
    """Handle Claude's nested content blocks."""
    content = msg.get("content", msg.get("text", ""))
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict):
                parts.append(block.get("text", block.get("content", "")))
        return " ".join(parts)
    return str(content)


def _parse_frontmatter(text: str) -> tuple:
    """Split YAML frontmatter from markdown body."""
    fm = {}
    if text.startswith("---"):
        end = text.find("---", 3)
        if end != -1:
            try:
                import yaml
                fm = yaml.safe_load(text[3:end]) or {}
            except Exception:
                # Fallback: parse key: value manually
                for line in text[3:end].split("\n"):
                    if ":" in line:
                        k, _, v = line.partition(":")
                        fm[k.strip()] = v.strip()
            return fm, text[end + 3:]
    return fm, text


def _parse_unix_or_iso(value) -> Optional[datetime]:
    """Parse unix timestamp (int/float) or ISO date string."""
    if value is None:
        return None
    try:
        if isinstance(value, (int, float)):
            return datetime.utcfromtimestamp(float(value))
        s = str(value).strip()
        # ISO with Z
        s = s.replace("Z", "+00:00")
        # Try common formats
        for fmt in ["%Y-%m-%dT%H:%M:%S%z", "%Y-%m-%dT%H:%M:%S",
                    "%Y-%m-%d %H:%M:%S", "%Y-%m-%d"]:
            try:
                return datetime.strptime(s[:len(fmt)], fmt)
            except ValueError:
                continue
        return datetime.fromisoformat(s)
    except Exception:
        return None


def _extract_pdf_text(path: Path) -> str:
    """Try pdfminer, then pymupdf, then return empty string."""
    # Try pdfminer.six
    try:
        from pdfminer.high_level import extract_text
        return extract_text(str(path))
    except ImportError:
        pass
    except Exception as e:
        log.warning(f"[pdf] pdfminer failed on {path.name}: {e}")

    # Try pymupdf (fitz)
    try:
        import fitz
        doc  = fitz.open(str(path))
        text = "\n".join(page.get_text() for page in doc)
        doc.close()
        return text
    except ImportError:
        pass
    except Exception as e:
        log.warning(f"[pdf] pymupdf failed on {path.name}: {e}")

    log.warning(f"[pdf] No PDF parser available for {path.name}. "
                "Install: pip install pdfminer.six OR pip install pymupdf")
    return ""
