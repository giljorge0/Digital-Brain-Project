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
            nodes = [{"id": nid, "title": G.nodes[nid].get("title", nid), "cluster": G.nodes[nid].get("cluster", 0), "centrality": round(G.nodes[nid].get("centrality", 0.001), 5), "tags": self.store.get_note(nid).tags[:6] if self.store.get_note(nid) else [], "role": self.store.get_note(nid).metadata.get("provenance_role","input") if self.store.get_note(nid) else "input", "snippet": self.store.get_note(nid).short_content(120) if self.store.get_note(nid) else ""} for nid in G.nodes()]
            links = [{"source": u, "target": v, "type": data.get("edge_type", "explicit"), "weight": round(float(data.get("weight", 1.0)), 3)} for u, v, data in G.edges(data=True)]
        except Exception:
            nodes, links = [], []
            
        (self.out_dir / "graph_data.json").write_text(json.dumps({"nodes": nodes, "links": links}, ensure_ascii=False), encoding="utf-8")

    def _export_notes(self):
        all_notes = self.store.get_all_notes()
        # Filter for your output only, excluding administrative wiki pages
        output_notes = [n for n in all_notes if n.metadata.get("provenance_role") == ROLE_OUTPUT and "wiki_page" not in n.tags]
        output_notes.sort(key=lambda n: n.centrality or 0, reverse=True)

        import markdown
        # Your preferred traditional categories
        VALID_CATS = {"AI", "Business", "Philosophy", "Writing", "Life", "Math", "Physics", "Technology"}
        # Tags to strictly ignore for categorization
        BLACKLIST = {"output", "input", "llm_chat", "web_clip", "pdf", "document", 
                     "generated", "synthesis", "authored", "external", "uncategorised"}

        categories = {}
        for note in output_notes:
            # 1. Look for a traditional category first
            cat = next((t for t in note.tags if t in VALID_CATS), None)
            
            # 2. If no traditional category, pick the first tag that isn't blacklisted
            if not cat:
                cat = next((t for t in note.tags if t.lower() not in BLACKLIST), "General")
            
            html_content = markdown.markdown(note.content)
            
            categories.setdefault(cat, []).append({
                "id": note.id, 
                "title": note.title, 
                "content": html_content,
                "tags": [t for t in note.tags if t.lower() not in BLACKLIST], 
                "date": note.date.strftime("%Y-%m-%d") if note.date else "",
                "word_count": note.word_count(), 
                "links": note.links[:20],
            })

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