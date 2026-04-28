"""
Expanded Data Importers
-----------------------
Parses various digital artifacts into standard Note objects.
Supports:
  1. LLM Chat Logs (ChatGPT/Claude JSON exports)
  2. Web Clippings (Markdown files from browser extensions)
  3. YouTube Transcripts/Notes (JSON or text)
  4. PDFs (requires PyPDF2)
"""

import json
import logging
from pathlib import Path
from datetime import datetime, timezone
from typing import List, Optional

from .note import Note

log = logging.getLogger(__name__)

class ImportManager:
    """Orchestrates different parsers based on file extensions or directories."""
    
    @staticmethod
    def parse_llm_chats(file_path: Path) -> List[Note]:
        """
        Parses standard LLM JSON exports (e.g., ChatGPT conversations.json).
        Extracts each conversation as a single Note.
        """
        notes = []
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                
            # Assuming a generic list of conversations: [{"title": "...", "mapping": {...}}]
            for conv in data:
                title = conv.get("title", "Untitled Chat")
                create_time = conv.get("create_time")
                date = datetime.fromtimestamp(create_time, tz=timezone.utc) if create_time else datetime.now(timezone.utc)
                
                # Extract text from the conversation mapping (OpenAI format)
                content_blocks = []
                mapping = conv.get("mapping", {})
                for node_id, node in mapping.items():
                    message = node.get("message")
                    if message and message.get("content", {}).get("parts"):
                        author = message["author"]["role"]
                        text = "".join(message["content"]["parts"])
                        content_blocks.append(f"**{author.upper()}**:\n{text}\n")
                
                content = "\n".join(content_blocks)
                if not content.strip():
                    continue
                    
                note_id = Note.make_id(f"chat_{title}_{create_time}")
                
                notes.append(Note(
                    id=note_id,
                    title=f"Chat: {title}",
                    content=content,
                    tags=["llm_chat", "ai_conversation"],
                    source_file=str(file_path),
                    date=date,
                    metadata={"type": "chat_log", "platform": "openai"}
                ))
        except Exception as e:
            log.error(f"[import] Failed to parse chat log {file_path}: {e}")
            
        return notes

    @staticmethod
    def parse_web_clips(directory: Path) -> List[Note]:
        """
        Parses markdown files saved by web-clipping browser extensions (like MarkDownload).
        Expects YAML frontmatter for URLs and titles.
        """
        notes = []
        import re
        yaml_re = re.compile(r'^---\n(.*?)\n---\n', re.DOTALL)
        
        for path in directory.glob("*.md"):
            try:
                text = path.read_text(encoding='utf-8')
                match = yaml_re.search(text)
                
                metadata = {"type": "web_clip"}
                content = text
                title = path.stem.replace('-', ' ').title()
                url = ""
                
                if match:
                    frontmatter = match.group(1)
                    content = text[match.end():] # Strip frontmatter
                    
                    # Naive YAML parsing for URL and Title
                    for line in frontmatter.split('\n'):
                        if line.startswith('title:'):
                            title = line.split('title:')[1].strip().strip('"\'')
                        elif line.startswith('url:'):
                            url = line.split('url:')[1].strip().strip('"\'')
                            metadata["source_url"] = url
                
                notes.append(Note(
                    id=Note.make_id(str(path)),
                    title=f"Web: {title}",
                    content=content.strip(),
                    tags=["web_clip"],
                    source_file=str(path),
                    date=datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc),
                    links=[url] if url else [],
                    metadata=metadata
                ))
            except Exception as e:
                log.warning(f"[import] Skipping web clip {path}: {e}")
                
        return notes

    @staticmethod
    def parse_pdf_text(directory: Path) -> List[Note]:
        """
        Extracts text from PDFs using Hierarchical Chunking.
        Creates one Parent note for the document, and sequential Child notes for the chunks.
        Requires: pip install PyPDF2
        """
        notes = []
        try:
            from PyPDF2 import PdfReader
        except ImportError:
            log.warning("[import] PyPDF2 not installed. Skipping PDF ingestion.")
            return notes

        for path in directory.glob("*.pdf"):
            try:
                reader = PdfReader(path)
                full_text = ""
                for page in reader.pages:
                    # Extract text and clean up basic formatting
                    page_text = page.extract_text()
                    if page_text:
                        full_text += page_text + "\n\n"

                if not full_text.strip():
                    continue

                # 1. Create the PARENT Note
                parent_id = Note.make_id(f"pdf_parent_{path.name}")
                parent_note = Note(
                    id=parent_id,
                    title=f"Essay: {path.stem}",
                    content=f"Full document container for: {path.name}",
                    tags=["essay", "longform", "pdf_parent"],
                    source_file=str(path),
                    date=datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc),
                    metadata={"type": "pdf_parent", "pages": len(reader.pages)}
                )
                notes.append(parent_note)

                # 2. Chunk the text (Targeting ~250 words per chunk for atomic embedding)
                paragraphs = full_text.split('\n\n')
                chunks = []
                current_chunk = ""
                
                for p in paragraphs:
                    current_chunk += p.strip() + "\n\n"
                    # If chunk exceeds ~250 words, save it and start a new one
                    if len(current_chunk.split()) > 250:
                        chunks.append(current_chunk.strip())
                        current_chunk = ""
                if current_chunk.strip():
                    chunks.append(current_chunk.strip()) # catch the remainder

                # 3. Create the CHILD Notes
                prev_child_id = None
                for i, chunk in enumerate(chunks):
                    child_id = Note.make_id(f"pdf_chunk_{path.name}_{i}")
                    
                    # Graph Links: Link to parent, and link to the previous chunk
                    links = [parent_id]
                    if prev_child_id:
                        links.append(prev_child_id)
                        
                    child_note = Note(
                        id=child_id,
                        title=f"{path.stem} (Part {i+1}/{len(chunks)})",
                        content=chunk,
                        tags=["pdf_chunk", "essay_argument"],
                        source_file=str(path),
                        links=links, # These become explicit edges in the GraphBuilder
                        metadata={
                            "type": "pdf_chunk", 
                            "parent_id": parent_id, 
                            "sequence": i
                        }
                    )
                    notes.append(child_note)
                    prev_child_id = child_id

                log.info(f"[import] Parsed PDF '{path.stem}' into 1 parent and {len(chunks)} chunks.")

            except Exception as e:
                log.error(f"[import] Failed to read PDF {path}: {e}")
                
        return notes