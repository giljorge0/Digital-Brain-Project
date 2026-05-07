"""
Digital Brain CLI
-----------------
Usage:
  python main.py --brain core ingest ~/notes
  python main.py --brain core build
  python main.py --brain core consolidate
  python main.py --brain core query "What are my arguments about epistemic limits?"
  python main.py --brain core visualize

  python main.py --brain core gap [--types void depth ...] [--mode anonymous|local|zk]
  python main.py --brain core recommend [--mode anonymous|local|zk]

  python main.py --brain core persona build
  python main.py --brain core persona show
  python main.py --brain core persona drift

  python main.py --brain core wiki update [--top-n 20] [--diff]
  python main.py --brain core wiki export [--output wiki/]
  python main.py --brain core wiki show [--concept "consciousness"]

  python main.py --brain core generate expand <note-id>
  python main.py --brain core export-wp [--mode graph|wiki|both]
  python main.py --brain core index-local [--sources arxiv wikipedia]
  
  python main.py --brain core export-static [--out public_html]
"""

import yaml
import os
import sys
import argparse
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger("brain_cli")

from brain.memory.store import Store
from brain.memory.graph import GraphBuilder
from brain.memory.embeddings import EmbeddingProvider, embed_notes
from brain.memory.consolidation import ConsolidationAgent
from brain.extract.relations import RelationExtractor
from brain.query.planner import QueryPlanner
from brain.agents.query_agent import QueryOrchestrator
from brain.agents.gap_agent import GapAgent
from brain.ingest.org_parser import OrgParser
from brain.ingest.importers import ImportManager
from brain.visualize.export import GraphExporter

DB_PATH = "data/brain.db"


def get_config() -> dict:
    cfg = {
        "llm_backend":           os.environ.get("LLM_BACKEND", "claude"),
        "anthropic_api_key":     os.environ.get("ANTHROPIC_API_KEY", ""),
        "embedding_backend":     os.environ.get("EMBEDDING_BACKEND", "local"),
        "local_embedding_model": "all-MiniLM-L6-v2",
        "ollama_base_url":       os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434"),
        "ollama_model":          os.environ.get("OLLAMA_MODEL", "mistral"),
        "claude_model":          os.environ.get("CLAUDE_MODEL", "claude-haiku-4-5-20251001"),
        "vector_backend":        os.environ.get("VECTOR_BACKEND", "sqlite"),
        "chroma_path":           os.environ.get("CHROMA_PATH", "data/chroma"),
        "qdrant_path":           os.environ.get("QDRANT_PATH", "data/qdrant"),
        "qdrant_url":            os.environ.get("QDRANT_URL"),
        "neo4j_uri":             os.environ.get("NEO4J_URI"),
        "neo4j_user":            os.environ.get("NEO4J_USER", "neo4j"),
        "neo4j_password":        os.environ.get("NEO4J_PASSWORD", "password"),
        "wp_url":                os.environ.get("WP_URL", ""),
        "wp_user":               os.environ.get("WP_USER", ""),
        "wp_app_password":       os.environ.get("WP_APP_PASSWORD", ""),
    }

    # 1. NEW: Load config.yaml so the brain obeys your settings
    config_path = Path("config.yaml")
    if config_path.exists():
        try:
            import yaml
            with open(config_path, "r") as f:
                file_cfg = yaml.safe_load(f) or {}
                cfg.update(file_cfg)  # Overwrite defaults with your yaml settings
        except Exception as e:
            log.warning(f"Could not parse config.yaml: {e}")

    # 2. Load llm_profiles.yaml if present
    for yaml_path in [Path("configs/llm_profiles.yaml"), Path("llm_profiles.yaml")]:
        if yaml_path.exists():
            try:
                import yaml
                with open(yaml_path) as f:
                    profiles_cfg = yaml.safe_load(f)
                defaults = profiles_cfg.get("defaults", {})
                profiles = {p["name"]: p for p in profiles_cfg.get("profiles", [])}
                daily = profiles.get(defaults.get("daily", ""), {})
                if daily.get("provider") == "claude":
                    cfg["llm_backend"]       = "claude"
                    cfg["claude_model"]      = daily.get("model", cfg["claude_model"])
                    cfg["anthropic_api_key"] = daily.get("api_key", cfg["anthropic_api_key"])
                elif daily.get("provider") == "ollama":
                    cfg["llm_backend"]  = "ollama"
                    cfg["ollama_model"] = daily.get("model", cfg["ollama_model"])
                embed = profiles.get(defaults.get("embed", ""), {})
                if embed.get("provider") == "ollama":
                    cfg["embedding_backend"]      = "ollama"
                    cfg["ollama_embedding_model"]  = embed.get("model", "nomic-embed-text")
            except Exception as e:
                log.warning(f"Could not parse llm_profiles.yaml: {e}")
            break

    return cfg

def get_store(cfg: dict) -> Store:
    """Return Store or Neo4jStore based on config."""
    if cfg.get("neo4j_uri"):
        try:
            from brain.memory.neo4j_store import Neo4jStore
            log.info(f"[store] Using Neo4j: {cfg['neo4j_uri']}")
            return Neo4jStore(cfg["neo4j_uri"], cfg["neo4j_user"], cfg["neo4j_password"])
        except ImportError:
            log.warning("[store] neo4j package not installed — falling back to SQLite")
    return Store(DB_PATH)


def patch_vector_backend(store, cfg: dict):
    """Optionally replace SQLite embedding storage with Chroma or Qdrant."""
    backend_name = cfg.get("vector_backend", "sqlite")
    if backend_name == "sqlite":
        return
    try:
        from brain.memory.vector_backends import VectorBackend, patch_store
        backend = VectorBackend.from_config(cfg)
        if backend:
            patch_store(store, backend)
            log.info(f"[vector] Using {type(backend).__name__}")
    except ImportError as e:
        log.warning(f"[vector] Could not load {backend_name} backend: {e}")


# ── ingest ────────────────────────────────────────────────────────────────────

def cli_ingest(args):
    store = get_store(get_config())
    path  = Path(args.path)
    if not path.exists():
        log.error(f"Path does not exist: {path}")
        return

    log.info(f"Scanning {path}...")
    notes = []

    notes.extend(OrgParser().parse_directory(path))
    notes.extend(ImportManager.parse_web_clips(path))

    for f in path.glob("*.json"):
        fn = f.name.lower()
        if any(k in fn for k in ("conversation", "chat", "claude", "messages")):
            notes.extend(ImportManager.parse_llm_chats(f))

    for name, parser in [
        ("watch-history.json",  ImportManager.parse_youtube_history),
        ("search-history.json", ImportManager.parse_youtube_search_history),
    ]:
        p = path / name
        if p.exists():
            notes.extend(parser(p))

    for p in path.glob("**/MyActivity.json"):
        if "search" in str(p).lower():
            notes.extend(ImportManager.parse_google_search_history(p))

    for p in path.glob("*.csv"):
        if any(k in p.name.lower() for k in ("goodreads", "library", "books")):
            notes.extend(ImportManager.parse_goodreads_csv(p))

    for p in path.glob("*Clippings*.txt"):
        notes.extend(ImportManager.parse_kindle_clippings(p))

    notes.extend(ImportManager.parse_pdf_text(path))
    for p in path.glob("*.sqlite"):
        if "places" in p.name.lower():
            notes.extend(ImportManager.parse_firefox_sqlite(p))
            
    if notes:
        store.upsert_notes(notes)
        log.info(f"Ingested {len(notes)} items.")
    else:
        log.warning("No valid files found.")


# ── build ─────────────────────────────────────────────────────────────────────

def cli_build(args):
    cfg   = get_config()
    if hasattr(args, "backend") and args.backend:
        cfg["vector_backend"] = args.backend

    store = get_store(cfg)
    patch_vector_backend(store, cfg)

    log.info("1. Generating embeddings...")
    embed_notes(store, EmbeddingProvider.from_config(cfg))

    log.info("2. Building knowledge graph...")
    builder = GraphBuilder(store)
    G = builder.build(use_explicit=True, use_tags=False, use_semantic=False)
    builder.compute_clusters(G)
    builder.compute_centrality(G)
    log.info("Build complete.")


# ── consolidate ───────────────────────────────────────────────────────────────

def cli_consolidate(args):
    cfg       = get_config()
    store     = get_store(cfg)
    extractor = RelationExtractor.from_config(cfg)
    builder   = GraphBuilder(store)
    ConsolidationAgent(store, extractor, builder).run_nightly_job()

    try:
        from brain.wiki.auto_wiki import AutoWiki, WikiScheduler
        from brain.persona.distiller import PersonaDistiller
        persona  = PersonaDistiller(store, cfg).load_profile()
        wiki     = AutoWiki(store, builder, persona, cfg)
        scheduler = WikiScheduler(wiki, store)
        scheduler.run_if_due(interval_hours=24)
    except Exception as e:
        log.debug(f"[wiki-scheduler] Skipped: {e}")


# ── query ─────────────────────────────────────────────────────────────────────

def cli_query(args):
    cfg      = get_config()
    store    = get_store(cfg)
    patch_vector_backend(store, cfg)
    embedder = EmbeddingProvider.from_config(cfg)
    planner  = QueryPlanner(store, embedder, cfg)
    agent    = QueryOrchestrator(planner)

    print(f"\n{'='*50}\nQUESTION: {args.question}\n{'='*50}\n")
    answer = agent.ask(args.question)
    print(f"\n{'='*50}\nFINAL ANSWER:\n{'='*50}\n{answer}")


# ── visualize ────────────────────────────────────────────────────────────────

def cli_visualize(args):
    store    = get_store(get_config())
    builder  = GraphBuilder(store)
    GraphExporter(store, builder).export_json("web/graph_data.json")

    log.info("Starting server at http://localhost:8000  (Ctrl+C to stop)")
    os.chdir("web")
    import http.server, socketserver
    with socketserver.TCPServer(("", 8000), http.server.SimpleHTTPRequestHandler) as httpd:
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            log.info("Server stopped.")


# ── gap ───────────────────────────────────────────────────────────────────────

def cli_gap(args):
    cfg       = get_config()
    store     = get_store(cfg)
    embedder  = EmbeddingProvider.from_config(cfg)
    agent     = GapAgent(store, embedder, cfg)
    mode      = getattr(args, "mode", "anonymous")
    types     = getattr(args, "types", None)
    print(agent.daily_briefing(gap_types=types, mode=mode))


# ── recommend ────────────────────────────────────────────────────────────────

def cli_recommend(args):
    cfg       = get_config()
    store     = get_store(cfg)
    embedder  = EmbeddingProvider.from_config(cfg)
    mode      = getattr(args, "mode", "anonymous")
    agent     = GapAgent(store, embedder, cfg)
    results   = agent.run(mode=mode, top_k=5)
    
    if not results:
        print("No gaps or recommendations generated.")
        return
    for r in results:
        gap, recs = r["gap"], r["recommendations"]
        print(f"\n── {gap.get('type','?').upper()}: {gap.get('label','?')} ──")
        if gap.get("explanation"):
            print(f"   {gap['explanation']}")
        for rec in recs:
            cert = " [ZK✓]" if rec.get("zk_certified") else ""
            print(f"   → [{rec.get('source_type','')}] {rec.get('title','')}{cert}")


# ── persona ───────────────────────────────────────────────────────────────────

def cli_persona(args):
    from brain.persona.distiller import PersonaDistiller
    cfg       = get_config()
    store     = get_store(cfg)
    distiller = PersonaDistiller(store, cfg)

    if args.persona_cmd == "build":
        profile = distiller.build_profile()
        print(f"\n✓ Persona v{profile.get('version',1)} built.")
        print(f"  Notes: {profile['corpus_size']['note_count']} | "
              f"Words: {profile['corpus_size']['total_words']:,}")
        print(f"\n{profile.get('llm_self_description','')}")

    elif args.persona_cmd == "show":
        profile = distiller.load_profile()
        if not profile:
            print("No profile. Run: python main.py persona build")
            return
        print(f"\n{'='*60}\nPERSONA v{profile.get('version',1)} "
              f"({profile.get('generated_at','')[:10]})\n{'='*60}")
        print(f"\n─ SELF DESCRIPTION ─\n{profile.get('llm_self_description','')}")
        print("\n─ TOP TOPICS ─")
        for tag, count in list(profile.get("topical_fingerprint",{})
                                        .get("top_tags",{}).items())[:15]:
            print(f"  {tag:<25} {'█' * min(count, 40)} {count}")
        print("\n─ STANCES ─")
        for topic, stance in profile.get("stance_map", {}).items():
            print(f"  [{topic}] {stance}")
        print("\n─ TEMPORAL ARC ─")
        for yr, data in profile.get("temporal_arc", {}).items():
            print(f"  {yr}: {data['note_count']} notes — {data['dominant_topic']}")

    elif args.persona_cmd == "drift":
        distiller.print_drift_report()

    else:
        print(f"Unknown persona subcommand: {args.persona_cmd}")


# ── wiki ──────────────────────────────────────────────────────────────────────

def cli_wiki(args):
    from brain.wiki.auto_wiki import AutoWiki, WikiScheduler
    from brain.persona.distiller import PersonaDistiller
    cfg      = get_config()
    store    = get_store(cfg)
    builder  = GraphBuilder(store)
    persona  = PersonaDistiller(store, cfg).load_profile()
    wiki     = AutoWiki(store, builder, persona, cfg)

    if args.wiki_cmd == "update":
        diff     = getattr(args, "diff", False)
        top_n    = getattr(args, "top_n", 20)
        pages    = wiki.update_all(top_n=top_n, diff_only=diff)
        mode_str = "diff-patch" if diff else "full"
        print(f"\n✓ {mode_str} update: {len(pages)} pages refreshed: "
              f"{', '.join(pages[:8])}{'…' if len(pages) > 8 else ''}")

    elif args.wiki_cmd == "export":
        out   = getattr(args, "output", "wiki/")
        count = wiki.export_markdown(out)
        print(f"\n✓ Exported {count} pages to {out}")

    elif args.wiki_cmd == "show":
        concept = getattr(args, "concept", None)
        if concept:
            page = wiki.get_page(concept)
            if page:
                print(f"\n{'='*60}\nWIKI: {concept.title()} "
                      f"(v{page.metadata.get('version',1)})\n{'='*60}\n")
                print(page.content)
            else:
                print(f"No page for '{concept}'. Run: python main.py wiki update")
        else:
            pages = wiki.list_pages()
            print(f"\n{len(pages)} wiki pages:")
            for p in pages:
                c = p.metadata.get("wiki_concept", p.title)
                v = p.metadata.get("version", 1)
                s = p.metadata.get("source_note_count", 0)
                print(f"  • {c:<30} v{v}  ({s} sources)")

    elif args.wiki_cmd == "history":
        concept = getattr(args, "concept", None)
        if not concept:
            print("--concept required for history. E.g.: --concept consciousness")
            return
        wiki.show_version_history(concept)

    elif args.wiki_cmd == "schedule":
        scheduler = WikiScheduler(wiki, store)
        interval  = getattr(args, "interval", 24)
        scheduler.install_cron(interval_hours=interval)

    else:
        print(f"Unknown wiki subcommand: {args.wiki_cmd}")


# ── generate ──────────────────────────────────────────────────────────────────

def cli_generate(args):
    from brain.persona.distiller import PersonaDistiller
    from brain.persona.generator import PersonaGenerator
    cfg      = get_config()
    store    = get_store(cfg)
    patch_vector_backend(store, cfg)
    embedder = EmbeddingProvider.from_config(cfg)
    profile  = PersonaDistiller(store, cfg).load_profile()
    if not profile:
        print("No persona profile. Run: python main.py persona build")
        return
    gen = PersonaGenerator(store, embedder, profile, cfg)

    if args.gen_cmd == "expand":
        print(f"\nExpanding: {args.note_id}\n{'─'*60}")
        print(gen.expand(args.note_id))
    elif args.gen_cmd == "respond":
        print(f"\nQuestion: {args.question}\n{'─'*60}")
        print(gen.respond(args.question))
    elif args.gen_cmd == "makemore":
        ideas = gen.makemore(args.seed, n=getattr(args, "n", 5))
        for i, idea in enumerate(ideas, 1):
            print(f"\n{i}. {idea.get('title','?')}")
            print(f"   {idea.get('premise','?')}")
            print(f"   Why fits: {idea.get('why_fits','?')}")
    elif args.gen_cmd == "synthesize":
        result = gen.synthesize(args.topic)
        print(result)
        if getattr(args, "save", False):
            from brain.ingest.note import Note
            from datetime import datetime, timezone
            store.upsert_note(Note(
                id=Note.make_id(f"synthesis_{args.topic}"),
                title=f"Synthesis: {args.topic.title()}",
                content=result,
                tags=["synthesis", "generated"],
                date=datetime.now(timezone.utc),
                metadata={"type": "synthesis", "topic": args.topic},
            ))
            log.info("Synthesis note saved.")


# ── export-wp ────────────────────────────────────────────────────────────────

def cli_export_wp(args):
    from brain.visualize.wp_export import WPExporter
    from brain.wiki.auto_wiki import AutoWiki
    from brain.persona.distiller import PersonaDistiller

    cfg     = get_config()
    store   = get_store(cfg)
    builder = GraphBuilder(store)
    persona = PersonaDistiller(store, cfg).load_profile()
    wiki    = AutoWiki(store, builder, persona, cfg)
    dry_run = getattr(args, "dry_run", False)
    mode    = getattr(args, "wp_mode", "graph")

    exporter = WPExporter(store, builder, wiki, cfg)

    if mode in ("graph", "both"):
        result = exporter.publish_graph(
            embed_mode=getattr(args, "embed_mode", "inline"),
            graph_url=getattr(args, "graph_url", ""),
            dry_run=dry_run,
        )
        if result:
            print(f"✓ Graph page: {result.get('link', '(dry run)')}")

    if mode in ("wiki", "both"):
        results = exporter.publish_wiki_pages(
            category=getattr(args, "category", "Digital Brain"),
            dry_run=dry_run,
        )
        print(f"✓ Wiki posts: {len(results)} published")


# ── index-local ───────────────────────────────────────────────────────────────

def cli_index_local(args):
    import json, urllib.request, urllib.parse
    from datetime import datetime, timezone
    cfg      = get_config()
    store    = get_store(cfg)
    embedder = EmbeddingProvider.from_config(cfg)
    limit    = getattr(args, "limit", 2000)
    sources  = getattr(args, "sources", ["arxiv"])

    notes    = store.get_all_notes()
    from collections import Counter
    tag_counts: Counter = Counter()
    for n in notes:
        for t in n.tags:
            if t not in ("llm_chat","web_clip","pdf","wiki_page","synthesis"):
                tag_counts[t] += 1
    top_tags = [t for t, _ in tag_counts.most_common(20)]

    items = []

    if "arxiv" in sources:
        log.info(f"[index-local] Querying arXiv for: {top_tags[:10]}")
        for tag in top_tags[:10]:
            query    = urllib.parse.quote(tag)
            api_url  = (f"https://export.arxiv.org/api/query?"
                        f"search_query=all:{query}&start=0&max_results=20")
            try:
                with urllib.request.urlopen(api_url, timeout=30) as resp:
                    import re
                    xml = resp.read().decode("utf-8")
                    titles  = re.findall(r'<title>(.*?)</title>', xml, re.DOTALL)[1:]
                    summaries = re.findall(r'<summary>(.*?)</summary>', xml, re.DOTALL)
                    ids_raw = re.findall(r'<id>http://arxiv.org/abs/([^<]+)</id>', xml)
                    for t, s, arxiv_id in zip(titles, summaries, ids_raw):
                        items.append({
                            "title":    t.strip(),
                            "abstract": s.strip()[:400],
                            "type":     "paper",
                            "author":   "",
                            "url":      f"https://arxiv.org/abs/{arxiv_id.strip()}",
                            "embedding": [],
                        })
            except Exception as e:
                log.warning(f"[index-local] arXiv query failed for '{tag}': {e}")

    items = items[:limit]
    log.info(f"[index-local] Embedding {len(items)} candidates...")

    batch_size = 32
    for i in range(0, len(items), batch_size):
        batch = items[i:i+batch_size]
        texts = [f"{it['title']} {it['abstract']}" for it in batch]
        try:
            vecs = embedder.embed(texts)
            for item, vec in zip(batch, vecs):
                item["embedding"] = vec
        except Exception as e:
            log.warning(f"[index-local] Embedding batch {i} failed: {e}")

    index = {"generated_at": datetime.now(timezone.utc).isoformat(),
             "item_count": len(items), "items": items}
    Path("data").mkdir(exist_ok=True)
    with open("data/local_index.json", "w") as f:
        json.dump(index, f)
    log.info(f"[index-local] Saved {len(items)} items to data/local_index.json")


# ── youtube ───────────────────────────────────────────────────────────────────

def cli_youtube(args):
    from brain.analysis.youtube_analyzer import YouTubeAnalyzer
    import json as _json
 
    cfg      = get_config()
    analyzer = YouTubeAnalyzer(cfg)
    root = Path(args.path).expanduser()
 
    candidates_watch = [
        root / "watch-history.json",
        root / "YouTube" / "history" / "watch-history.json",
        root / "YouTube and YouTube Music" / "history" / "watch-history.json",
    ]
    candidates_search = [
        root / "search-history.json",
        root / "YouTube" / "history" / "search-history.json",
        root / "YouTube and YouTube Music" / "history" / "search-history.json",
    ]
    candidates_playlists = [
        root / "playlists",
        root / "YouTube" / "playlists",
        root / "YouTube and YouTube Music" / "playlists",
    ]
 
    watch_path = next((p for p in candidates_watch if p.exists()), None)
    if not watch_path:
        log.error(
            f"Could not find watch-history.json under {root}\n"
            "Expected path: Takeout/YouTube/history/watch-history.json"
        )
        return
 
    search_path   = next((p for p in candidates_search   if p.exists()), None)
    playlist_dir  = next((p for p in candidates_playlists if p.exists()), None)
 
    log.info(f"Watch history:   {watch_path}")
    log.info(f"Search history:  {search_path or '(not found)'}")
    log.info(f"Playlists dir:   {playlist_dir or '(not found)'}")
 
    report = analyzer.analyze(
        watch_path   = watch_path,
        search_path  = search_path,
        playlist_dir = playlist_dir,
    )
    report.print_summary()
 
    if getattr(args, 'save', False):
        out = Path("data") / "youtube_report.json"
        report.save(out)
        log.info(f"Report saved to {out}")
 
    if getattr(args, 'integrate_persona', False):
        report.integrate_with_persona(Path("data") / "persona.json")
        log.info("Persona updated with YouTube arc.")


# ── export-static ─────────────────────────────────────────────────────────────

def cli_export_static(args):
    """Export brain to a deployable GitHub Pages static site."""
    from brain.visualize.static_export import StaticExporter
    cfg   = get_config()
    store = get_store(cfg)
    out   = getattr(args, "out", "public_html")
    exp   = StaticExporter(store, cfg, out_dir=out)
    out_path = exp.export()
    print(f"\n✓ Static site exported to {out_path}/")
    print(f"\nTo deploy to GitHub Pages:")
    print(f"  1. cd {out_path}")
    print(f"  2. git init && git add . && git commit -m 'digital brain'")
    print(f"  3. gh repo create YOUR_USERNAME/digital-brain --public --push --source=.")
    print(f"  4. Go to repo Settings → Pages → Deploy from branch (main)")


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Digital Brain CLI")
    parser.add_argument("--brain", default="core", choices=["core", "omni"], help="Target brain (core or omni)")
    sub    = parser.add_subparsers(dest="command", required=True)

    # ingest
    p = sub.add_parser("ingest")
    p.add_argument("path")

    # build
    p = sub.add_parser("build")
    p.add_argument("--backend", choices=["sqlite", "chroma", "qdrant"],
                   help="Vector backend (default: sqlite)")

    # consolidate
    sub.add_parser("consolidate")

    # query
    p = sub.add_parser("query")
    p.add_argument("question")

    # visualize
    sub.add_parser("visualize")

    # gap
    p = sub.add_parser("gap")
    p.add_argument("--types", nargs="*",
                   choices=["void","depth","width","temporal","contradiction","orthogonal"])
    p.add_argument("--mode", default="anonymous", choices=["anonymous","local","zk"])

    # recommend
    p = sub.add_parser("recommend")
    p.add_argument("--mode", default="anonymous", choices=["anonymous","local","zk"])

    # persona
    p = sub.add_parser("persona")
    p.add_argument("persona_cmd", choices=["build","show","drift"])

    # wiki
    p = sub.add_parser("wiki")
    p.add_argument("wiki_cmd", choices=["update","export","show","history","schedule"])
    p.add_argument("--top-n",    type=int, default=20, dest="top_n")
    p.add_argument("--output",   default="wiki/")
    p.add_argument("--concept")
    p.add_argument("--diff",     action="store_true",
                   help="Only regenerate stale pages (diff-patch mode)")
    p.add_argument("--interval", type=int, default=24,
                   help="Cron interval in hours for 'schedule' subcommand")

    # generate
    p    = sub.add_parser("generate")
    gsub = p.add_subparsers(dest="gen_cmd", required=True)
    gp   = gsub.add_parser("expand");    gp.add_argument("note_id")
    gp   = gsub.add_parser("respond");   gp.add_argument("question")
    gp   = gsub.add_parser("makemore");  gp.add_argument("seed"); gp.add_argument("--n", type=int, default=5)
    gp   = gsub.add_parser("synthesize");gp.add_argument("topic"); gp.add_argument("--save", action="store_true")

    # export-wp
    p = sub.add_parser("export-wp")
    p.add_argument("--mode",       default="graph",  choices=["graph","wiki","both"], dest="wp_mode")
    p.add_argument("--embed-mode", default="inline", choices=["inline","iframe"])
    p.add_argument("--graph-url",  default="")
    p.add_argument("--category",   default="Digital Brain")
    p.add_argument("--dry-run",    action="store_true")

    # youtube
    p = sub.add_parser("youtube",
        help="Deep analysis of YouTube watch/search history from Google Takeout")
    p.add_argument("path",
        help="Path to Google Takeout folder or YouTube/history/ subfolder")
    p.add_argument("--save", action="store_true",
        help="Save JSON report to data/youtube_report.json")
    p.add_argument("--integrate-persona", action="store_true", dest="integrate_persona",
        help="Merge YouTube timeline into data/persona.json")

    # index-local
    p = sub.add_parser("index-local",
                        help="Build offline recommendation index from arXiv")
    p.add_argument("--sources", nargs="*", default=["arxiv"],
                   choices=["arxiv", "wikipedia"])
    p.add_argument("--limit", type=int, default=2000)

    # export-static
    pes = sub.add_parser("export-static", help="Export to static GitHub Pages site")
    pes.add_argument("--out", default="public_html")

    args = parser.parse_args()

    global DB_PATH
    DB_PATH = f"data/{args.brain}/brain.db"
    os.environ["CHROMA_PATH"] = f"data/{args.brain}/chroma"
    os.environ["QDRANT_PATH"] = f"data/{args.brain}/qdrant"
    Path(f"data/{args.brain}").mkdir(parents=True, exist_ok=True)

    dispatch = {
        "ingest":        cli_ingest,
        "build":         cli_build,
        "consolidate":   cli_consolidate,
        "query":         cli_query,
        "visualize":     cli_visualize,
        "gap":           cli_gap,
        "recommend":     cli_recommend,
        "persona":       cli_persona,
        "wiki":          cli_wiki,
        "generate":      cli_generate,
        "export-wp":     cli_export_wp,
        "index-local":   cli_index_local,
        "youtube":       cli_youtube,
        "export-static": cli_export_static,
    }

    fn = dispatch.get(args.command)
    if fn:
        fn(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()





