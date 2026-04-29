#!/usr/bin/env python3
"""
scripts/index_all.py
--------------------
One-shot full pipeline. Run this after a big batch import.

  python scripts/index_all.py ~/Nextcloud/brain/
  python scripts/index_all.py ~/Nextcloud/brain/ --skip-embed
  python scripts/index_all.py ~/Nextcloud/brain/ --relations

Steps:
  1. Ingest all .org / .md / .pdf / .json in the given directory
  2. Generate embeddings for all new notes
  3. Build knowledge graph (explicit + tag + semantic edges)
  4. Compute PageRank centrality + Louvain community clusters
  5. [optional] Extract LLM relations (--relations flag, uses heavy LLM)
  6. Export D3 graph JSON for visualization
  7. Print summary stats
"""

import sys
import argparse
import logging
from pathlib import Path

# Make sure the repo root is on the path
ROOT = Path(__file__).parent.parent.resolve()
sys.path.insert(0, str(ROOT))

from brain.ingest.org_parser import OrgParser
from brain.ingest.importers import ImportManager
from brain.memory.store import Store
from brain.memory.embeddings import EmbeddingProvider, embed_notes
from brain.memory.graph import GraphBuilder
from brain.llm.providers import LLMRegistry
from brain.visualize.export import GraphExporter

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("index_all")


def main():
    parser = argparse.ArgumentParser(description="Full Digital Brain indexing pipeline")
    parser.add_argument("path", help="Directory to ingest (e.g. ~/Nextcloud/brain/)")
    parser.add_argument("--skip-embed", action="store_true",
                        help="Skip embedding generation (fast run, no semantic edges)")
    parser.add_argument("--relations", action="store_true",
                        help="Run LLM relation extraction (slow, costs API tokens)")
    parser.add_argument("--no-visualize", action="store_true",
                        help="Skip exporting graph JSON")
    args = parser.parse_args()

    source_dir = Path(args.path).expanduser()
    if not source_dir.exists():
        log.error(f"Directory not found: {source_dir}")
        sys.exit(1)

    db_path  = ROOT / "data" / "brain.db"
    cfg_path = ROOT / "configs" / "llm_profiles.yaml"
    web_dir  = ROOT / "web"

    store    = Store(db_path)
    registry = LLMRegistry(cfg_path)

    # ── Step 1: Ingest ────────────────────────────────────────────────────────
    log.info(f"Step 1/6 — Ingesting from {source_dir} …")
    notes = []

    org_notes = OrgParser().parse_directory(source_dir)
    log.info(f"  .org  → {len(org_notes)} notes")
    notes.extend(org_notes)

    md_notes = ImportManager.parse_web_clips(source_dir)
    if md_notes:
        log.info(f"  .md   → {len(md_notes)} web clips")
        notes.extend(md_notes)

    pdf_notes = ImportManager.parse_pdf_text(source_dir)
    if pdf_notes:
        log.info(f"  .pdf  → {len(pdf_notes)} documents")
        notes.extend(pdf_notes)

    for jf in source_dir.rglob("*.json"):
        chat_notes = ImportManager.parse_llm_chats(jf)
        if chat_notes:
            log.info(f"  chat  → {len(chat_notes)} turns ({jf.name})")
            notes.extend(chat_notes)

    store.upsert_notes(notes)
    log.info(f"  Total ingested: {len(notes)} items → DB: {db_path}")

    # ── Step 2: Embeddings ────────────────────────────────────────────────────
    if not args.skip_embed:
        log.info("Step 2/6 — Generating embeddings …")
        embedder = EmbeddingProvider.from_registry(registry)
        embed_notes(store, embedder)
    else:
        log.info("Step 2/6 — Skipping embeddings (--skip-embed)")

    # ── Step 3 + 4: Graph + Centrality + Clusters ─────────────────────────────
    log.info("Step 3/6 — Building knowledge graph …")
    builder = GraphBuilder(store)
    G = builder.build()

    log.info("Step 4/6 — PageRank + community clusters …")
    builder.compute_centrality(G)
    builder.compute_clusters(G)

    # ── Step 5: LLM relation extraction (optional) ────────────────────────────
    if args.relations:
        log.info("Step 5/6 — LLM relation extraction (this may take a few minutes) …")
        from brain.extract.relations import RelationExtractor
        extractor = RelationExtractor.from_registry(registry)
        n_edges = extractor.extract_for_store(store)
        log.info(f"  New LLM edges: {n_edges}")
    else:
        log.info("Step 5/6 — Skipping LLM relations (pass --relations to enable)")

    # ── Step 6: Visualize ──────────────────────────────────────────────────────
    if not args.no_visualize:
        log.info("Step 6/6 — Exporting D3 graph …")
        exporter = GraphExporter(store, builder)
        exporter.export_html(str(web_dir))
        log.info(f"  Graph ready at {web_dir}/index.html")
    else:
        log.info("Step 6/6 — Skipping visualization")

    # ── Summary ───────────────────────────────────────────────────────────────
    stats = store.stats()
    log.info("=" * 50)
    log.info("  DONE")
    log.info(f"  Notes      : {stats['notes']}")
    log.info(f"  Edges      : {stats['edges']}")
    log.info(f"  Embeddings : {stats['notes_with_embeddings']}")
    log.info(f"  Clusters   : {stats['clusters']}")
    log.info(f"  Tags       : {len(stats['tags'])}")
    log.info("=" * 50)
    log.info("  Next: python main.py visualize  →  http://localhost:8000")
    log.info("  Or:   python main.py query 'Your question here'")


if __name__ == "__main__":
    main()
