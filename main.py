"""
Digital Brain CLI
-----------------
python main.py ingest ~/notes/org-roam
python main.py ingest ~/Downloads/conversations.json        # ChatGPT export
python main.py ingest ~/Downloads/watch-history.json        # YouTube Takeout
python main.py build
python main.py consolidate
python main.py query "What are my main arguments about epistemic limits?"
python main.py wiki "consciousness and computation"
python main.py makemore "free will and determinism"
python main.py position "epistemic limits"
python main.py suggest-notes <note-id>
python main.py visualize
python main.py stats
"""

import os
import sys
import argparse
import logging
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s: %(message)s"
)
log = logging.getLogger("brain_cli")

# ── Lazy imports (avoid crashing if optional deps missing) ────────────────────
from brain.memory.store      import Store
from brain.memory.graph      import GraphBuilder
from brain.memory.embeddings import EmbeddingProvider, embed_notes
from brain.extract.relations import RelationExtractor
from brain.query.planner     import QueryPlanner
from brain.agents.query_agent import QueryOrchestrator
from brain.ingest.org_parser  import OrgParser
from brain.ingest.importers   import ImportManager
from brain.visualize.export   import GraphExporter
from brain.persona.makemore   import Persona
from brain.analysis.gap_finder  import GapFinder
from brain.analysis.recommender import Recommender

DB_PATH = "data/brain.db"


# ─── Config ───────────────────────────────────────────────────────────────────

def get_config() -> dict:
    """Load config from config.yaml if present, else use env vars."""
    cfg = {
        "llm_backend":           os.environ.get("BRAIN_LLM_BACKEND",   "claude"),
        "anthropic_api_key":     os.environ.get("ANTHROPIC_API_KEY",   ""),
        "ollama_base_url":       os.environ.get("BRAIN_OLLAMA_URL",    "http://localhost:11434"),
        "ollama_model":          os.environ.get("BRAIN_OLLAMA_MODEL",  "mistral"),
        "embedding_backend":     os.environ.get("BRAIN_EMBED_BACKEND", "local"),
        "local_embedding_model": os.environ.get("BRAIN_EMBED_MODEL",   "all-MiniLM-L6-v2"),
        "claude_model":          os.environ.get("BRAIN_CLAUDE_MODEL",  "claude-haiku-4-5-20251001"),
    }

    config_path = Path("config.yaml")
    if config_path.exists():
        try:
            import yaml
            with open(config_path) as f:
                file_cfg = yaml.safe_load(f) or {}
            cfg.update(file_cfg)
            log.info(f"Loaded config from {config_path}")
        except ImportError:
            log.warning("PyYAML not installed — using env vars only")
        except Exception as e:
            log.warning(f"Could not parse config.yaml: {e}")

    return cfg


# ─── Commands ─────────────────────────────────────────────────────────────────

def cli_ingest(args):
    """Ingest files or directories into the brain."""
    store = Store(DB_PATH)
    path  = Path(args.path)

    if not path.exists():
        log.error(f"Path does not exist: {path}")
        sys.exit(1)

    notes = []

    # ── Single file dispatch ──────────────────────────────────────────────────
    if path.is_file():
        name = path.name.lower()
        if name == "conversations.json" or "chatgpt" in name:
            log.info(f"Detected ChatGPT export: {path.name}")
            notes = ImportManager.parse_chatgpt_export(path)
        elif "claude" in name:
            log.info(f"Detected Claude export: {path.name}")
            notes = ImportManager.parse_claude_export(path)
        elif "watch-history" in name or "youtube" in name:
            log.info(f"Detected YouTube history: {path.name}")
            notes = ImportManager.parse_youtube_history(path)
        elif "myactivity" in name or "search" in name:
            log.info(f"Detected search history: {path.name}")
            notes = ImportManager.parse_search_history(path)
        elif name.endswith(".json"):
            log.info(f"Auto-detecting JSON chat format: {path.name}")
            notes = ImportManager.parse_llm_chats(path)
        elif name.endswith(".org"):
            notes = OrgParser().parse_file(path)
        elif name.endswith(".pdf"):
            from brain.ingest.importers import _extract_pdf_text, ROLE_INPUT
            from brain.ingest.note import Note
            text = _extract_pdf_text(path)
            if text:
                notes = [Note(
                    id=Note.make_id(str(path)),
                    title=path.stem.replace("_", " ").title(),
                    content=text[:50000],
                    source_file=str(path),
                    metadata={"type": "pdf", "provenance_role": ROLE_INPUT}
                )]
        else:
            log.error(f"Unsupported file type: {path.suffix}")
            sys.exit(1)

    # ── Directory scan ────────────────────────────────────────────────────────
    else:
        log.info(f"Scanning {path} for all supported file types...")

        # Org files
        org = OrgParser().parse_directory(path)
        if org:
            log.info(f"  .org files: {len(org)} notes")
            notes.extend(org)

        # Markdown web clips
        md = ImportManager.parse_web_clips(path)
        if md:
            log.info(f"  .md clips:  {len(md)} notes")
            notes.extend(md)

        # PDFs
        pdfs = ImportManager.parse_pdf_text(path)
        if pdfs:
            log.info(f"  .pdf files: {len(pdfs)} notes")
            notes.extend(pdfs)

        # JSON: try to detect each one
        for jf in sorted(path.glob("**/*.json")):
            jnotes = ImportManager.parse_llm_chats(jf)
            if jnotes:
                log.info(f"  {jf.name}: {len(jnotes)} chat turns")
                notes.extend(jnotes)

    if notes:
        store.upsert_notes(notes)
        # Print input/output breakdown
        n_out = sum(1 for n in notes
                    if n.metadata.get("provenance_role") == "output")
        n_in  = len(notes) - n_out
        log.info(f"Ingested {len(notes)} items "
                 f"({n_out} YOUR output, {n_in} external input)")
    else:
        log.warning("No notes found to ingest.")


def cli_build(args):
    """Build embeddings, graph, clusters, and centrality."""
    cfg   = get_config()
    store = Store(DB_PATH)

    log.info("── Step 1/3: Generating embeddings ──────────────────────")
    embedder = EmbeddingProvider.from_config(cfg)
    embed_notes(store, embedder, force=args.force)

    log.info("── Step 2/3: Building graph ──────────────────────────────")
    builder = GraphBuilder(store)
    G = builder.build(
        use_explicit=True,
        use_tags=True,
        use_semantic=not args.no_semantic,
    )

    log.info("── Step 3/3: Clusters & centrality ──────────────────────")
    builder.compute_clusters(G)
    builder.compute_centrality(G)

    stats = store.stats()
    log.info(f"Build complete: {stats['notes']} notes, "
             f"{stats['edges']} edges, "
             f"{stats['clusters']} clusters")


def cli_consolidate(args):
    """Run the nightly consolidation loop."""
    from brain.memory.consolidation import ConsolidationAgent
    cfg     = get_config()
    store   = Store(DB_PATH)
    extract = RelationExtractor.from_config(cfg)
    builder = GraphBuilder(store)
    agent   = ConsolidationAgent(store, extract, builder)
    report  = agent.run_nightly_job()
    log.info(f"Consolidation report: {report}")


def cli_query(args):
    """Query the brain with a natural language question."""
    cfg   = get_config()
    store = Store(DB_PATH)
    embedder = EmbeddingProvider.from_config(cfg)
    planner  = QueryPlanner(store, embedder, cfg)
    agent    = QueryOrchestrator(planner, hitl=not args.no_hitl)

    print(f"\n{'='*60}")
    print(f"QUESTION: {args.question}")
    print(f"{'='*60}\n")

    answer = agent.ask(args.question, mode=args.mode or "auto")

    print(f"\n{'='*60}")
    print("ANSWER:")
    print(f"{'='*60}")
    print(answer)


def cli_wiki(args):
    """Generate a personal wiki article on a topic from your corpus."""
    cfg      = get_config()
    store    = Store(DB_PATH)
    embedder = EmbeddingProvider.from_config(cfg)
    persona  = Persona(store, embedder, cfg)

    print(f"\n[brain] Generating wiki article: '{args.topic}'...\n")
    result = persona.llm_wiki(args.topic)

    print(f"\n{'='*60}")
    print(f"  {result['title'].upper()}")
    print(f"{'='*60}")
    print(result["article"])
    print(f"\n── Open Questions ──────────────────────────────────")
    for q in result.get("open_questions", []):
        print(f"  ? {q}")
    print(f"\n── Related Topics ──────────────────────────────────")
    for t in result.get("related_topics", []):
        print(f"  → {t}")
    print(f"\n── Source Notes ({len(result.get('related_notes', []))}) ──")
    for n in result.get("related_notes", [])[:5]:
        print(f"  • {n['title']}")

    # Optionally save to file
    if args.save:
        out = Path("web/wiki") / f"{args.topic.replace(' ', '_')}.md"
        out.parent.mkdir(parents=True, exist_ok=True)
        md_content = f"# {result['title']}\n\n{result['article']}\n"
        if result.get("open_questions"):
            md_content += "\n## Open Questions\n"
            for q in result["open_questions"]:
                md_content += f"- {q}\n"
        out.write_text(md_content, encoding="utf-8")
        log.info(f"Saved to {out}")


def cli_makemore(args):
    """Generate new text in your voice on a topic."""
    cfg      = get_config()
    store    = Store(DB_PATH)
    embedder = EmbeddingProvider.from_config(cfg)
    persona  = Persona(store, embedder, cfg)

    print(f"\n[brain] Writing in your voice on: '{args.topic}'...\n")
    text = persona.makemore(
        args.topic,
        length=args.length,
        temperature_hint=args.style,
    )

    print(f"\n{'='*60}")
    print(text)
    print(f"\n{'='*60}")

    if args.save:
        out = Path("web/makemore") / f"{args.topic.replace(' ', '_')}.txt"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(text, encoding="utf-8")
        log.info(f"Saved to {out}")


def cli_position(args):
    """Extract your intellectual position on a topic."""
    cfg      = get_config()
    store    = Store(DB_PATH)
    embedder = EmbeddingProvider.from_config(cfg)
    persona  = Persona(store, embedder, cfg)

    print(f"\n[brain] Extracting your position on: '{args.topic}'...\n")
    result = persona.my_position(args.topic)

    print(f"POSITION: {result['position']}")
    print(f"\nARGUMENTS:")
    for a in result.get("arguments", []):
        print(f"  + {a}")
    if result.get("tensions"):
        print(f"\nTENSIONS:")
        for t in result["tensions"]:
            print(f"  ± {t}")
    if result.get("evolution"):
        print(f"\nEVOLUTION: {result['evolution']}")
    print(f"\nSOURCE NOTES:")
    for n in result.get("source_notes", []):
        print(f"  • {n}")


def cli_suggest_notes(args):
    """Suggest new atomic notes to create from an existing note."""
    cfg      = get_config()
    store    = Store(DB_PATH)
    embedder = EmbeddingProvider.from_config(cfg)
    persona  = Persona(store, embedder, cfg)

    suggestions = persona.suggest_new_notes(args.note_id)
    if not suggestions:
        print(f"Note '{args.note_id}' not found or no suggestions generated.")
        return

    note = store.get_note(args.note_id)
    print(f"\n[brain] New note suggestions for: '{note.title if note else args.note_id}'\n")
    for i, s in enumerate(suggestions, 1):
        print(f"{i}. {s['title']}")
        print(f"   {s['seed_content']}")
        print(f"   → relation: {s.get('relation_to_source', '?')}\n")


def cli_visualize(args):
    """Export the graph and serve it locally."""
    store   = Store(DB_PATH)
    builder = GraphBuilder(store)
    G       = builder.build()
    graph_data = builder.to_json(G)

    exporter = GraphExporter("web")
    html_path = exporter.export_html(graph_data, store=store)
    json_path = exporter.export_json(graph_data)

    log.info(f"Graph exported: {html_path}")
    log.info("Starting server at http://localhost:8000")

    import http.server
    import socketserver

    os.chdir("web")
    Handler = http.server.SimpleHTTPRequestHandler
    Handler.log_message = lambda *a: None  # silence request logs
    with socketserver.TCPServer(("", 8000), Handler) as httpd:
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            log.info("Server stopped.")


def cli_stats(args):
    """Print database statistics."""
    store = Store(DB_PATH)
    s     = store.stats()
    print(f"\n── Digital Brain Stats ─────────────────────────────")
    print(f"  Notes:           {s['notes']}")
    print(f"  Edges:           {s['edges']}")
    print(f"  Embeddings:      {s['notes_with_embeddings']}")
    print(f"  Clusters:        {s['clusters']}")
    print(f"  Unique tags:     {len(s['tags'])}")
    if s['tags']:
        print(f"  Tags:            {', '.join(s['tags'][:20])}")

    # Input/output breakdown
    all_notes = store.get_all_notes()
    n_out  = sum(1 for n in all_notes
                 if n.metadata.get("provenance_role", "output") == "output")
    n_in   = len(all_notes) - n_out
    print(f"\n  YOUR output:     {n_out} notes")
    print(f"  External input:  {n_in} notes")
    print()

def cli_gap(args):
    """Analyze your idea space and surface knowledge gaps."""
    cfg      = get_config()
    store    = Store(DB_PATH)
    embedder = EmbeddingProvider.from_config(cfg)
    finder   = GapFinder(store, embedder, cfg)

    if args.type:
        gaps = finder.find_gaps_of_type(args.type, n=args.n)
    else:
        gaps = finder.find_all_gaps(max_per_type=args.n)

    if not gaps:
        print("\nNo gaps found. Try ingesting more notes and running 'build' first.")
        return

    print(f"\n{'='*65}")
    print(f"  KNOWLEDGE GAP ANALYSIS  ({len(gaps)} gaps found)")
    print(f"{'='*65}\n")

    for i, gap in enumerate(gaps, 1):
        badge = {
            "void":          "○ VOID",
            "depth":         "↓ DEPTH",
            "width":         "← WIDTH →",
            "temporal":      "⟳ STALE",
            "contradiction": "≠ CONFLICT",
            "orthogonal":    "⊥ COUNTER",
        }.get(gap.gap_type, gap.gap_type.upper())

        print(f"[{i}] {badge}  priority={gap.priority_score:.2f}")
        print(f"    {gap.title}")
        print(f"    {gap.description}")
        if gap.suggested_actions:
            print(f"    → {gap.suggested_actions[0]}")
        print()

    if args.save:
        import json as _json
        out = Path("data/gaps.json")
        out.parent.mkdir(exist_ok=True)
        out.write_text(_json.dumps(
            [g.to_dict() for g in gaps], indent=2, ensure_ascii=False
        ))
        log.info(f"Gaps saved to {out}")


def cli_recommend(args):
    """Generate a daily reading list from your knowledge gaps."""
    cfg      = get_config()
    store    = Store(DB_PATH)
    embedder = EmbeddingProvider.from_config(cfg)
    finder   = GapFinder(store, embedder, cfg)
    rec      = Recommender(cfg, mode=args.mode)

    log.info("Analyzing gaps...")
    gaps = finder.find_all_gaps(max_per_type=3)

    if not gaps:
        print("\nNo gaps found. Try 'python main.py gap' first.")
        return

    if args.briefing:
        print("\nGenerating daily briefing...")
        briefing = rec.daily_briefing(gaps, n_items=args.n)

        print(f"\n{'='*65}")
        print(f"  DAILY BRIEFING  {briefing['date']}")
        print(f"{'='*65}")
        print(f"\n{briefing['summary']}\n")
        print(f"{'─'*65}")
        print(f"TODAY'S READING LIST  ({briefing['reading_time_estimate']})")
        print(f"{'─'*65}\n")
        for i, item in enumerate(briefing["items"], 1):
            type_badge = {"book": "📚", "paper": "📄", "video": "▶",
                          "article": "🔗", "search_query": "🔍"}.get(
                item["source_type"], "·")
            print(f"{i}. {item['title']}")
            if item.get("author"):
                print(f"   by {item['author']}")
            if item.get("why"):
                print(f"   {item['why']}")
            if item.get("url"):
                print(f"   {item['url']}")
            print()

        if args.save:
            import json as _json
            out = Path(f"data/briefing_{briefing['date']}.json")
            out.write_text(_json.dumps(briefing, indent=2, ensure_ascii=False))
            log.info(f"Briefing saved to {out}")

    else:
        # Simple recommendation list
        recs = rec.recommend(gaps, top_k=args.n)
        print(f"\n{'='*65}")
        print(f"  RECOMMENDATIONS  (mode={args.mode})")
        print(f"{'='*65}\n")
        for i, r in enumerate(recs, 1):
            print(f"{i}. [{r.source_type}] {r.title}")
            if r.author:
                print(f"   by {r.author}")
            print(f"   gap: {r.gap_title}")
            print(f"   {r.why}")
            if r.url:
                print(f"   {r.url}")
            print()


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Digital Brain — Neuro-Symbolic Memory System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # ingest
    p = sub.add_parser("ingest", help="Ingest files or directory")
    p.add_argument("path", help="File or directory to ingest")

    # build
    p = sub.add_parser("build", help="Embed notes and build graph")
    p.add_argument("--force", action="store_true",
                   help="Re-embed notes that already have embeddings")
    p.add_argument("--no-semantic", action="store_true",
                   help="Skip semantic edge computation (faster)")

    # consolidate
    sub.add_parser("consolidate", help="Run nightly memory maintenance")

    # query
    p = sub.add_parser("query", help="Ask the brain a question")
    p.add_argument("question")
    p.add_argument("--mode", choices=["auto","semantic","keyword","graph","temporal","hybrid"],
                   default="auto")
    p.add_argument("--no-hitl", action="store_true",
                   help="Disable human-in-the-loop clarification")

    # wiki
    p = sub.add_parser("wiki", help="Generate wiki article from your corpus")
    p.add_argument("topic")
    p.add_argument("--save", action="store_true", help="Save to web/wiki/")

    # makemore
    p = sub.add_parser("makemore", help="Generate text in your voice")
    p.add_argument("topic")
    p.add_argument("--length", choices=["short","medium","long"], default="medium")
    p.add_argument("--style",  default="natural",
                   help="Temperature hint: natural | exploratory | precise")
    p.add_argument("--save", action="store_true")

    # position
    p = sub.add_parser("position", help="Extract your position on a topic")
    p.add_argument("topic")

    # suggest-notes
    p = sub.add_parser("suggest-notes", help="Suggest atomic child notes")
    p.add_argument("note_id", help="ID of note to expand")

    # gap
    p = sub.add_parser("gap", help="Analyze idea space and find knowledge gaps")
    p.add_argument("--type", choices=["void","depth","width","temporal","contradiction","orthogonal"],
                   help="Analyze only one gap type")
    p.add_argument("--n", type=int, default=5, help="Max gaps per type")
    p.add_argument("--save", action="store_true", help="Save gaps to data/gaps.json")

    # recommend
    p = sub.add_parser("recommend", help="Generate reading recommendations from gaps")
    p.add_argument("--n", type=int, default=8, help="Number of recommendations")
    p.add_argument("--mode", choices=["anonymous","local"], default="anonymous",
                   help="Privacy mode: anonymous (default) or local (fully offline)")
    p.add_argument("--briefing", action="store_true",
                   help="Format as a daily briefing with summary paragraph")
    p.add_argument("--save", action="store_true",
                   help="Save briefing to data/briefing_YYYY-MM-DD.json")

    # visualize
    sub.add_parser("visualize", help="Export D3 graph and serve locally")

    # stats
    sub.add_parser("stats", help="Show database statistics")

    args = parser.parse_args()
    Path("data").mkdir(exist_ok=True)

    dispatch = {
        "ingest":        cli_ingest,
        "build":         cli_build,
        "consolidate":   cli_consolidate,
        "query":         cli_query,
        "wiki":          cli_wiki,
        "makemore":      cli_makemore,
        "position":      cli_position,
        "suggest-notes": cli_suggest_notes,
        "gap":           cli_gap,
        "recommend":     cli_recommend,
        "visualize":     cli_visualize,
        "stats":         cli_stats,
    }
    dispatch[args.command](args)


if __name__ == "__main__":
    main()
