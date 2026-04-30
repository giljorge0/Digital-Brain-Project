"""
Digital Brain CLI
-----------------
INGESTION
  python main.py ingest ~/notes/org-roam
  python main.py ingest ~/Downloads/conversations.json   # ChatGPT export
  python main.py ingest ~/Downloads/watch-history.json   # YouTube Takeout
  python main.py ingest ~/data/                          # Entire folder (all formats)

BUILD & MAINTAIN
  python main.py build
  python main.py build --force
  python main.py consolidate

QUERY
  python main.py query "What are my arguments about epistemic limits?"
  python main.py query "..." --mode graph

PERSONA
  python main.py persona build     # Extract intellectual profile from corpus
  python main.py persona show      # Print profile + topics + stances

GENERATE (text in your voice)
  python main.py generate expand <note-id>
  python main.py generate respond "question"
  python main.py generate makemore "seed topic"
  python main.py generate synthesize "topic" [--save]

WIKI
  python main.py wiki update
  python main.py wiki export
  python main.py wiki show
  python main.py wiki show --concept "consciousness"

GAPS & RECOMMENDATIONS
  python main.py gap
  python main.py gap --type depth
  python main.py recommend --briefing
  python main.py recommend --mode local

OTHER
  python main.py visualize
  python main.py stats
"""

import os
import sys
import argparse
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger("brain_cli")

from brain.memory.store       import Store
from brain.memory.graph       import GraphBuilder
from brain.memory.embeddings  import EmbeddingProvider, embed_notes
from brain.extract.relations  import RelationExtractor
from brain.query.planner      import QueryPlanner
from brain.agents.query_agent import QueryOrchestrator
from brain.ingest.org_parser  import OrgParser
from brain.ingest.importers   import ImportManager
from brain.visualize.export   import GraphExporter
from brain.analysis.gap_finder  import GapFinder
from brain.analysis.recommender import Recommender

DB_PATH = "data/brain.db"


def get_config() -> dict:
    cfg = {
        "llm_backend":           os.environ.get("BRAIN_LLM_BACKEND",   "claude"),
        "anthropic_api_key":     os.environ.get("ANTHROPIC_API_KEY",   ""),
        "ollama_base_url":       os.environ.get("BRAIN_OLLAMA_URL",    "http://localhost:11434"),
        "ollama_model":          os.environ.get("BRAIN_OLLAMA_MODEL",  "mistral"),
        "embedding_backend":     os.environ.get("BRAIN_EMBED_BACKEND", "local"),
        "local_embedding_model": os.environ.get("BRAIN_EMBED_MODEL",   "all-MiniLM-L6-v2"),
        "claude_model":          os.environ.get("BRAIN_CLAUDE_MODEL",  "claude-haiku-4-5-20251001"),
    }
    cfg_path = Path("config.yaml")
    if cfg_path.exists():
        try:
            import yaml
            with open(cfg_path) as f:
                cfg.update(yaml.safe_load(f) or {})
        except ImportError:
            log.warning("PyYAML not installed — using env vars only")
        except Exception as e:
            log.warning(f"config.yaml error: {e}")
    return cfg


# ─── ingest ───────────────────────────────────────────────────────────────────

def cli_ingest(args):
    store = Store(DB_PATH)
    path  = Path(args.path)
    if not path.exists():
        log.error(f"Path not found: {path}"); sys.exit(1)

    notes = []

    if path.is_file():
        name = path.name.lower()
        if "conversations" in name or "chatgpt" in name:
            notes = ImportManager.parse_chatgpt_export(path)
        elif "claude" in name:
            notes = ImportManager.parse_claude_export(path)
        elif "watch-history" in name or "youtube" in name:
            notes = ImportManager.parse_youtube_history(path)
        elif "search-history" in name:
            notes = ImportManager.parse_youtube_search_history(path)
        elif "myactivity" in name or "google" in name:
            notes = ImportManager.parse_google_search_history(path)
        elif name.endswith(".json"):
            notes = ImportManager.parse_llm_chats(path)
        elif name.endswith(".org"):
            notes = OrgParser().parse_file(path)
        elif name.endswith(".csv") and "goodreads" in name:
            notes = ImportManager.parse_goodreads_csv(path)
        elif "clippings" in name and name.endswith(".txt"):
            notes = ImportManager.parse_kindle_clippings(path)
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
            log.error(f"Unsupported: {path.suffix}"); sys.exit(1)

    else:
        log.info(f"Scanning {path}...")
        for fn, parser in [
            ("*.org dirs", lambda: OrgParser().parse_directory(path)),
            ("*.md",       lambda: ImportManager.parse_web_clips(path)),
            ("*.pdf",      lambda: ImportManager.parse_pdf_text(path)),
        ]:
            try:
                n = parser()
                if n: notes.extend(n); log.info(f"  {fn}: {len(n)}")
            except Exception as e:
                log.warning(f"  {fn} failed: {e}")

        for f in path.glob("*Clippings*.txt"):
            n = ImportManager.parse_kindle_clippings(f)
            if n: notes.extend(n); log.info(f"  Kindle: {len(n)} books")

        for f in path.glob("*.csv"):
            if "goodreads" in f.name.lower() or "library" in f.name.lower():
                n = ImportManager.parse_goodreads_csv(f)
                if n: notes.extend(n); log.info(f"  Goodreads: {len(n)} books")

        for jf in sorted(path.glob("**/*.json")):
            fname = jf.name.lower()
            if "watch-history" in fname or "youtube" in fname:
                n = ImportManager.parse_youtube_history(jf)
            elif "search-history" in fname:
                n = ImportManager.parse_youtube_search_history(jf)
            elif "myactivity" in fname:
                n = ImportManager.parse_google_search_history(jf)
            else:
                n = ImportManager.parse_llm_chats(jf)
            if n: notes.extend(n); log.info(f"  {jf.name}: {len(n)}")

    if notes:
        store.upsert_notes(notes)
        n_out = sum(1 for n in notes if n.metadata.get("provenance_role") == "output")
        log.info(f"Ingested {len(notes)} ({n_out} YOUR output, {len(notes)-n_out} external input)")
    else:
        log.warning("No notes found.")


# ─── build ────────────────────────────────────────────────────────────────────

def cli_build(args):
    cfg   = get_config()
    store = Store(DB_PATH)
    log.info("── 1/3 Embeddings ───────────────────────────────────────")
    embedder = EmbeddingProvider.from_config(cfg)
    embed_notes(store, embedder, force=getattr(args, "force", False))
    log.info("── 2/3 Graph ─────────────────────────────────────────────")
    builder = GraphBuilder(store)
    G = builder.build(use_explicit=True, use_tags=True,
                      use_semantic=not getattr(args, "no_semantic", False))
    log.info("── 3/3 Clusters & centrality ─────────────────────────────")
    builder.compute_clusters(G)
    builder.compute_centrality(G)
    s = store.stats()
    log.info(f"Done: {s['notes']} notes · {s['edges']} edges · {s['clusters']} clusters")


# ─── consolidate ──────────────────────────────────────────────────────────────

def cli_consolidate(args):
    from brain.memory.consolidation import ConsolidationAgent
    cfg   = get_config(); store = Store(DB_PATH)
    agent = ConsolidationAgent(store, RelationExtractor.from_config(cfg), GraphBuilder(store))
    log.info(f"Report: {agent.run_nightly_job()}")


# ─── query ────────────────────────────────────────────────────────────────────

def cli_query(args):
    cfg   = get_config(); store = Store(DB_PATH)
    agent = QueryOrchestrator(
        QueryPlanner(store, EmbeddingProvider.from_config(cfg), cfg),
        hitl=not getattr(args, "no_hitl", False)
    )
    print(f"\n{'='*60}\nQ: {args.question}\n{'='*60}\n")
    print(agent.ask(args.question, mode=getattr(args, "mode", "auto")))


# ─── persona ──────────────────────────────────────────────────────────────────

def cli_persona(args):
    from brain.persona.distiller import PersonaDistiller
    cfg   = get_config(); store = Store(DB_PATH)
    d     = PersonaDistiller(store, cfg)

    if args.persona_cmd == "build":
        p = d.build_profile(); d.save_profile(p)
        print(f"\n✓ Profile built: {p['corpus_size']['note_count']} notes, "
              f"{p['corpus_size']['total_words']:,} words")

    elif args.persona_cmd == "show":
        p = d.load_profile()
        if not p: print("No profile. Run: python main.py persona build"); return
        print(f"\n{'='*60}\n  YOUR INTELLECTUAL PROFILE\n{'='*60}")
        print(f"\n{p.get('llm_self_description','(none)')}")
        print("\n── Top Topics ─────────────────────────────────────")
        for tag, count in list(p.get("topical_fingerprint",{}).get("top_tags",{}).items())[:15]:
            print(f"  {tag:<25} {'█'*min(count,40)} {count}")
        print("\n── Stances ────────────────────────────────────────")
        for topic, stance in p.get("stance_map", {}).items():
            print(f"  [{topic}] {stance}")
        print("\n── Temporal Arc ───────────────────────────────────")
        for yr, data in p.get("temporal_arc", {}).items():
            print(f"  {yr}: {data['note_count']} notes — {data['dominant_topic']}")


# ─── generate ─────────────────────────────────────────────────────────────────

def cli_generate(args):
    from brain.persona.distiller import PersonaDistiller
    from brain.persona.generator import PersonaGenerator
    cfg   = get_config(); store = Store(DB_PATH)
    p     = PersonaDistiller(store, cfg).load_profile()
    if not p: print("No profile. Run: python main.py persona build"); return
    gen   = PersonaGenerator(store, EmbeddingProvider.from_config(cfg), p, cfg)

    if   args.gen_cmd == "expand":
        print(gen.expand(args.note_id))
    elif args.gen_cmd == "respond":
        print(gen.respond(args.question))
    elif args.gen_cmd == "makemore":
        ideas = gen.makemore(args.seed, n=getattr(args, "n", 5))
        for i, idea in enumerate(ideas, 1):
            print(f"\n{i}. {idea.get('title','?')}\n   {idea.get('premise','?')}\n   ↳ {idea.get('why_fits','?')}")
    elif args.gen_cmd == "synthesize":
        result = gen.synthesize(args.topic)
        print(result)
        if getattr(args, "save", False):
            from brain.ingest.note import Note
            from datetime import datetime, timezone
            store.upsert_note(Note(
                id=Note.make_id(f"synthesis_{args.topic}"),
                title=f"Synthesis: {args.topic.title()}",
                content=result, tags=["synthesis","generated"],
                date=datetime.now(timezone.utc),
                metadata={"type":"synthesis","provenance_role":"output"},
            ))


# ─── wiki ─────────────────────────────────────────────────────────────────────

def cli_wiki(args):
    from brain.wiki.auto_wiki import AutoWiki
    from brain.persona.distiller import PersonaDistiller
    cfg  = get_config(); store = Store(DB_PATH)
    wiki = AutoWiki(store, GraphBuilder(store),
                    PersonaDistiller(store, cfg).load_profile(), cfg)

    if args.wiki_cmd == "update":
        pages = wiki.update_all(top_n=getattr(args, "top_n", 20))
        print(f"✓ Updated {len(pages)} pages: {', '.join(pages[:10])}")
    elif args.wiki_cmd == "export":
        print(f"✓ Exported {wiki.export_markdown(getattr(args,'output','wiki/'))} pages")
    elif args.wiki_cmd == "show":
        concept = getattr(args, "concept", None)
        if concept:
            page = wiki.get_page(concept)
            print(page.content if page else f"No page for '{concept}'. Run: wiki update")
        else:
            for p in wiki.list_pages():
                print(f"  • {p.metadata.get('wiki_concept', p.title)}  (v{p.metadata.get('version',1)})")


# ─── gap ──────────────────────────────────────────────────────────────────────

def cli_gap(args):
    cfg   = get_config(); store = Store(DB_PATH)
    finder = GapFinder(store, EmbeddingProvider.from_config(cfg), cfg)
    gaps   = (finder.find_gaps_of_type(args.type, n=getattr(args,"n",5))
              if getattr(args,"type",None)
              else finder.find_all_gaps(max_per_type=getattr(args,"n",5)))

    if not gaps: print("\nNo gaps found. Ingest + build first."); return

    print(f"\n{'='*65}\n  KNOWLEDGE GAPS ({len(gaps)})\n{'='*65}\n")
    badges = {"void":"○ VOID","depth":"↓ DEPTH","width":"← WIDTH →",
              "temporal":"⟳ STALE","contradiction":"≠ CONFLICT","orthogonal":"⊥ COUNTER"}
    for i, g in enumerate(gaps, 1):
        print(f"[{i}] {badges.get(g.gap_type, g.gap_type.upper())}  priority={g.priority_score:.2f}")
        print(f"    {g.title}\n    {g.description}")
        if g.suggested_actions: print(f"    → {g.suggested_actions[0]}")
        print()

    if getattr(args, "save", False):
        import json as _j
        out = Path("data/gaps.json"); out.parent.mkdir(exist_ok=True)
        out.write_text(_j.dumps([g.to_dict() for g in gaps], indent=2, ensure_ascii=False))
        log.info(f"Saved to {out}")


# ─── recommend ────────────────────────────────────────────────────────────────

def cli_recommend(args):
    cfg   = get_config(); store = Store(DB_PATH)
    gaps  = GapFinder(store, EmbeddingProvider.from_config(cfg), cfg).find_all_gaps(max_per_type=3)
    if not gaps: print("No gaps found. Run 'gap' first."); return

    rec = Recommender(cfg, mode=args.mode)
    n   = getattr(args, "n", 8)

    if getattr(args, "briefing", False):
        b = rec.daily_briefing(gaps, n_items=n)
        print(f"\n{'='*65}\n  DAILY BRIEFING  {b['date']}\n{'='*65}\n{b['summary']}\n")
        print(f"{'─'*65}\nREADING LIST ({b['reading_time_estimate']})\n{'─'*65}\n")
        icons = {"book":"📚","paper":"📄","video":"▶","article":"🔗","search_query":"🔍"}
        for i, item in enumerate(b["items"], 1):
            print(f"{i}. {icons.get(item['source_type'],'·')} {item['title']}")
            if item.get("author"): print(f"   by {item['author']}")
            if item.get("why"):    print(f"   {item['why']}")
            if item.get("url"):    print(f"   {item['url']}")
            print()
        if getattr(args, "save", False):
            import json as _j
            Path(f"data/briefing_{b['date']}.json").write_text(
                _j.dumps(b, indent=2, ensure_ascii=False))
    else:
        recs = rec.recommend(gaps, top_k=n)
        print(f"\n{'='*65}\n  RECOMMENDATIONS (mode={args.mode})\n{'='*65}\n")
        for i, r in enumerate(recs, 1):
            print(f"{i}. [{r.source_type}] {r.title}")
            if r.author: print(f"   by {r.author}")
            if r.why:    print(f"   {r.why}")
            if r.url:    print(f"   {r.url}")
            print()


# ─── visualize ────────────────────────────────────────────────────────────────

def cli_visualize(args):
    store   = Store(DB_PATH)
    builder = GraphBuilder(store)
    G       = builder.build()
    html    = GraphExporter("web").export_html(builder.to_json(G), store=store)
    log.info(f"Exported: {html}")
    log.info("http://localhost:8000 — Ctrl+C to stop")
    import http.server, socketserver
    os.chdir("web")
    Handler = http.server.SimpleHTTPRequestHandler
    Handler.log_message = lambda *a: None
    with socketserver.TCPServer(("", 8000), Handler) as httpd:
        try: httpd.serve_forever()
        except KeyboardInterrupt: pass


# ─── stats ────────────────────────────────────────────────────────────────────

def cli_stats(args):
    store = Store(DB_PATH); s = store.stats()
    notes = store.get_all_notes()
    n_out = sum(1 for n in notes if n.metadata.get("provenance_role","output") == "output")
    print(f"\n── Digital Brain ───────────────────────────────────")
    print(f"  Notes:          {s['notes']}")
    print(f"  Edges:          {s['edges']}")
    print(f"  Embeddings:     {s['notes_with_embeddings']}")
    print(f"  Clusters:       {s['clusters']}")
    print(f"  YOUR output:    {n_out}")
    print(f"  External input: {len(notes) - n_out}")
    print(f"  Tags ({len(s['tags'])}):      {', '.join(s['tags'][:20])}\n")


# ─── main ─────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(description="Digital Brain — Neuro-Symbolic Memory System")
    s = p.add_subparsers(dest="command", required=True)

    pi = s.add_parser("ingest");     pi.add_argument("path")

    pb = s.add_parser("build")
    pb.add_argument("--force",       action="store_true")
    pb.add_argument("--no-semantic", action="store_true")

    s.add_parser("consolidate")

    pq = s.add_parser("query")
    pq.add_argument("question")
    pq.add_argument("--mode", choices=["auto","semantic","keyword","graph","temporal","hybrid"], default="auto")
    pq.add_argument("--no-hitl", action="store_true")

    pp = s.add_parser("persona"); pp.add_argument("persona_cmd", choices=["build","show"])

    pg  = s.add_parser("generate")
    gs  = pg.add_subparsers(dest="gen_cmd", required=True)
    ge  = gs.add_parser("expand");     ge.add_argument("note_id")
    gr  = gs.add_parser("respond");    gr.add_argument("question")
    gm  = gs.add_parser("makemore");   gm.add_argument("seed"); gm.add_argument("--n", type=int, default=5)
    gsy = gs.add_parser("synthesize"); gsy.add_argument("topic"); gsy.add_argument("--save", action="store_true")

    pw = s.add_parser("wiki")
    pw.add_argument("wiki_cmd", choices=["update","export","show"])
    pw.add_argument("--top-n",  type=int, default=20, dest="top_n")
    pw.add_argument("--output", default="wiki/")
    pw.add_argument("--concept")

    pga = s.add_parser("gap")
    pga.add_argument("--type", choices=["void","depth","width","temporal","contradiction","orthogonal"])
    pga.add_argument("--n",    type=int, default=5)
    pga.add_argument("--save", action="store_true")

    pr = s.add_parser("recommend")
    pr.add_argument("--n",        type=int, default=8)
    pr.add_argument("--mode",     choices=["anonymous","local"], default="anonymous")
    pr.add_argument("--briefing", action="store_true")
    pr.add_argument("--save",     action="store_true")

    s.add_parser("visualize")
    s.add_parser("stats")

    args = p.parse_args()
    Path("data").mkdir(exist_ok=True)

    {
        "ingest": cli_ingest, "build": cli_build, "consolidate": cli_consolidate,
        "query": cli_query, "persona": cli_persona, "generate": cli_generate,
        "wiki": cli_wiki, "gap": cli_gap, "recommend": cli_recommend,
        "visualize": cli_visualize, "stats": cli_stats,
    }[args.command](args)


if __name__ == "__main__":
    main()
