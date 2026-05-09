"""
brain/visualize/static_export.py
----------------------------------
Exports your digital brain to a fully static GitHub Pages site.
Copies your premium SPA (index.html, style.css, app.js) and injects the JSON data.
"""

import json
import logging
import shutil
from pathlib import Path

log = logging.getLogger(__name__)

ROLE_OUTPUT = "output"

class StaticExporter:
    def __init__(self, store, cfg: dict, out_dir: str = "public_html"):
        self.store   = store
        self.cfg     = cfg
        self.out_dir = Path(out_dir)

    def export(self) -> Path:
        log.info(f"[export] Exporting Cosmic Academic Portfolio to {self.out_dir}/")

        self.out_dir.mkdir(parents=True, exist_ok=True)

        self._export_graph()
        self._export_notes()
        self._export_persona()
        self._copy_frontend()

        log.info(f"[export] ✓ Done → {self.out_dir}/")
        return self.out_dir

    def _export_graph(self):
        try:
            from brain.memory.graph import GraphBuilder
            builder = GraphBuilder(self.store)
            G = builder.build(use_explicit=True, use_tags=True, use_semantic=False)
            nodes = [{"id": nid, "title": self.store.get_note(nid).title if self.store.get_note(nid) and self.store.get_note(nid).title else "", "cluster": G.nodes[nid].get("cluster", 0), "centrality": round(G.nodes[nid].get("centrality", 0.001), 5), "role": self.store.get_note(nid).metadata.get("provenance_role","input") if self.store.get_note(nid) else "input", "snippet": self.store.get_note(nid).short_content(120) if self.store.get_note(nid) else ""} for nid in G.nodes()]
            links = [{"source": u, "target": v, "type": data.get("edge_type", "explicit"), "weight": round(float(data.get("weight", 1.0)), 3)} for u, v, data in G.edges(data=True)]
        except Exception:
            nodes, links = [], []
            
        (self.out_dir / "graph_data.json").write_text(json.dumps({"nodes": nodes, "links": links}, ensure_ascii=False), encoding="utf-8")

    def _export_notes(self):
        all_notes = self.store.get_all_notes()
        # Filter for your output only, excluding administrative wiki pages
        output_notes = [n for n in all_notes if "wiki_page" not in n.tags]
        output_notes.sort(key=lambda n: n.centrality or 0, reverse=True)

        import markdown
        # Your preferred traditional categories
        VALID_CATS = {"AI", "Business", "Philosophy", "Writing", "Life", "Math", "Physics", "Technology"}
        # Tags to strictly ignore for categorization
        BLACKLIST = {"output", "input", "llm_chat", "web_clip", "pdf", "document", 
                     "generated", "synthesis", "authored", "external", "uncategorised"}

        # ── NEW: Ensure the PDF directory exists inside public_html ──
        pdf_dir = self.out_dir / "pdfs"
        pdf_dir.mkdir(parents=True, exist_ok=True)

        categories = {}
        for note in output_notes:
            # 1. Look for a traditional category first
            cat = next((t for t in note.tags if t in VALID_CATS), None)
            
            # 2. If no traditional category, pick the first tag that isn't blacklisted
            if not cat:
                cat = next((t for t in note.tags if t.lower() not in BLACKLIST), "General")
            
            html_content = markdown.markdown(note.content)
            
            # ── NEW: Handle PDF Copying and Linking ──
            source_pdf_rel = None
            
            # Check if note has a source_file attribute and if it's a PDF
            original_pdf_path = None
            if hasattr(note, 'source_file') and note.source_file and str(note.source_file).lower().endswith('.pdf'):
                original_pdf_path = Path(note.source_file)
            elif hasattr(note, 'metadata') and note.metadata.get('source_file', '').lower().endswith('.pdf'):
                original_pdf_path = Path(note.metadata['source_file'])

            # If we found a PDF, copy it to public_html/pdfs/
            if original_pdf_path and original_pdf_path.exists():
                dest_pdf_path = pdf_dir / original_pdf_path.name
                if not dest_pdf_path.exists():
                    shutil.copy2(original_pdf_path, dest_pdf_path)
                source_pdf_rel = f"pdfs/{original_pdf_path.name}"

            note_dict = {
                "id": note.id, 
                "title": note.title, 
                "content": html_content,
                "tags": [t for t in note.tags if t.lower() not in BLACKLIST], 
                "date": note.date.strftime("%Y-%m-%d") if note.date else "",
                "word_count": note.word_count(), 
                "links": note.links[:20],
            }

            # If a PDF was copied, add the relative link to the JSON
            if source_pdf_rel:
                note_dict["source_pdf"] = source_pdf_rel

            categories.setdefault(cat, []).append(note_dict)

        sorted_cats = dict(sorted(categories.items(), key=lambda kv: len(kv[1]), reverse=True))
        (self.out_dir / "notes.json").write_text(json.dumps({"categories": sorted_cats, "total": len(output_notes)}, ensure_ascii=False), encoding="utf-8")
        
    def _export_persona(self):
        persona_path = Path("data/persona.json")
        if persona_path.exists():
            shutil.copy(persona_path, self.out_dir / "persona.json")
        else:
            (self.out_dir / "persona.json").write_text("{}", encoding="utf-8")

    def _copy_frontend(self):
        for file_name in ["index.html", "style.css", "app.js", "cv.pdf"]: # Added cv.pdf here
            src = Path("web") / file_name
            if src.exists():
                shutil.copy(src, self.out_dir / file_name)