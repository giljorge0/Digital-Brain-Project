#!/usr/bin/env python3
"""
first_run.py  —  Boot the Digital Brain from your raw data
-----------------------------------------------------------
Run this ONCE after setting up the repo to go from an empty database
to a fully indexed, embedded, graph-connected, visualizable brain.

What it does (in order):
  1. Checks dependencies and prints what's missing
  2. Ingests your org files  (output: your writing)
  3. Ingests your authored PDFs  (output: your writing)
  4. Ingests LLM chat exports if found  (your conversations)
  5. Generates embeddings  (local Ollama or API)
  6. Builds the knowledge graph  (explicit + tag + semantic edges)
  7. Computes PageRank centrality + community clusters
  8. Builds your persona profile  (intellectual DNA)
  9. Runs the gap finder  (what's missing in your thinking)
 10. Exports the D3 visualization
 11. Prints a summary and next-steps guide

Usage:
  python first_run.py

  # With explicit paths (if you don't want to be prompted):
  python first_run.py \\
      --org      ~/Nextcloud/brain/ \\
      --pdfs     ~/brain-data/authored-pdfs/ \\
      --chats    ~/Downloads/conversations.json \\
      --no-gaps  \\
      --no-llm

Flags:
  --org PATH       org-mode notes directory (default: ~/Nextcloud/brain/)
  --pdfs PATH      authored PDFs directory  (default: ask)
  --chats PATH     LLM export file/dir      (default: skip)
  --no-gaps        skip gap analysis (faster first run)
  --no-llm         skip all LLM calls (fully offline run, uses TF-IDF embeddings)
  --skip-embed     skip embedding entirely (fastest, no semantic edges)
"""

import sys
import argparse
import logging
import shutil
import subprocess
from pathlib import Path
from datetime import datetime

ROOT = Path(__file__).parent.resolve()
sys.path.insert(0, str(ROOT))

DB_PATH = ROOT / "data" / "brain.db"
DB_PATH.parent.mkdir(parents=True, exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("first_run")

# ─── Colours (degrade gracefully on Windows) ─────────────────────────────────
try:
    from rich.console import Console
    from rich.panel   import Panel
    console = Console()
    def header(text):  console.print(f"\n[bold cyan]{text}[/bold cyan]")
    def ok(text):      console.print(f"  [green]✓[/green] {text}")
    def warn(text):    console.print(f"  [yellow]![/yellow] {text}")
    def err(text):     console.print(f"  [red]✗[/red] {text}")
    def info(text):    console.print(f"  {text}")
    HAS_RICH = True
except ImportError:
    def header(text):  print(f"\n=== {text} ===")
    def ok(text):      print(f"  OK  {text}")
    def warn(text):    print(f"  !   {text}")
    def err(text):     print(f"  ERR {text}")
    def info(text):    print(f"      {text}")
    HAS_RICH = False


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Boot the Digital Brain")
    parser.add_argument("--org",        default=None)
    parser.add_argument("--pdfs",       default=None)
    parser.add_argument("--chats",      default=None)
    parser.add_argument("--no-gaps",    action="store_true")
    parser.add_argument("--no-llm",     action="store_true")
    parser.add_argument("--skip-embed", action="store_true")
    args = parser.parse_args()

    if HAS_RICH:
        console.print(Panel.fit(
            "[bold]DIGITAL BRAIN — FIRST RUN[/bold]\n"
            "This will index your entire corpus and boot the knowledge graph.",
            border_style="cyan",
        ))
    else:
        print("\n" + "="*60)
        print("  DIGITAL BRAIN — FIRST RUN")
        print("="*60)

    # ── Step 0: Check dependencies ────────────────────────────────────────────
    header("Step 0/9 — Checking dependencies")
    _check_deps(args)

    # ── Gather paths ──────────────────────────────────────────────────────────
    org_path  = _resolve_path(args.org,   "Org notes directory",
                               default=Path.home() / "Nextcloud" / "brain"/ "raw-import")
    pdf_path  = _resolve_path(args.pdfs,  "Authored PDFs directory",
                               default=Path.home() / "Nextcloud" / "brain"/ "raw-import", required=False)
    chat_path = _resolve_path(args.chats, "LLM chat export (file or dir)",
                               default=None, required=False)

    # ── Step 1: Ingest org notes ──────────────────────────────────────────────
    header("Step 1/9 — Ingesting org-mode notes")
    from brain.ingest.org_parser import OrgParser
    from brain.memory.store import Store
    store = Store(DB_PATH)

    if org_path and org_path.exists():
        org_notes = OrgParser().parse_directory(org_path)
        store.upsert_notes(org_notes)
        ok(f"{len(org_notes):,} org notes ingested")
    else:
        warn("Org path not found — skipping")
        org_notes = []

    # ── Step 2: Ingest authored PDFs ─────────────────────────────────────────
    header("Step 2/9 — Ingesting authored PDFs")
    if pdf_path and pdf_path.exists():
        from brain.ingest.authored_pdf import parse_authored_pdfs
        pdf_notes = parse_authored_pdfs(pdf_path)
        store.upsert_notes(pdf_notes)
        ok(f"{len(pdf_notes):,} PDF notes ingested  (tagged as 'authored')")
    else:
        warn("No PDF path given — skipping")
        pdf_notes = []

    # ── Step 3: Ingest LLM chats ─────────────────────────────────────────────
    header("Step 3/9 — Ingesting LLM chat exports")
    if chat_path and chat_path.exists():
        from brain.ingest.importers import ImportManager
        chat_notes = ImportManager.parse_llm_chats(chat_path)
        store.upsert_notes(chat_notes)
        ok(f"{len(chat_notes):,} chat notes ingested")
    else:
        warn("No chat export given — skipping (you can add later with: python main.py ingest <path>)")
        chat_notes = []

    total_notes = store.note_count()
    info(f"Total notes in store: {total_notes:,}")

    if total_notes == 0:
        err("No notes were ingested. Check your paths and try again.")
        sys.exit(1)

    # ── Step 4: Embeddings ────────────────────────────────────────────────────
    header("Step 4/9 — Generating embeddings")
    if args.skip_embed:
        warn("Skipping embeddings (--skip-embed). No semantic edges will be built.")
    else:
        cfg = _get_config(args)
        from brain.memory.embeddings import EmbeddingProvider, embed_notes
        embedder = EmbeddingProvider.from_config(cfg)
        info(f"Embedding model: {getattr(embedder, 'model', 'local')}")
        embed_notes(store, embedder)
        ok("Embeddings generated")

    # ── Step 5: Build graph ───────────────────────────────────────────────────
    header("Step 5/9 — Building knowledge graph")
    from brain.memory.graph import GraphBuilder
    builder = GraphBuilder(store)
    G = builder.build(
        use_explicit=True,
        use_tags=True,
        use_semantic=not args.skip_embed,
    )
    ok(f"Graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")

    # ── Step 6: Centrality + clusters ────────────────────────────────────────
    header("Step 6/9 — Computing PageRank + community clusters")
    builder.compute_centrality(G)
    builder.compute_clusters(G)
    stats = store.stats()
    ok(f"{stats['clusters']} topic clusters detected")

    # ── Step 7: Persona profile ───────────────────────────────────────────────
    header("Step 7/9 — Building intellectual persona profile")
    if args.no_llm:
        warn("Skipping LLM persona (--no-llm). Run later: python main.py persona build")
    else:
        try:
            cfg = _get_config(args)
            from brain.persona.distiller import PersonaDistiller
            distiller = PersonaDistiller(store, cfg)
            profile = distiller.build_profile()
            ok(f"Persona built: {len(profile.get('stance_map', {}))} stances extracted")
            info(f"  Description: {profile.get('llm_self_description','')[:120]}…")
        except Exception as e:
            warn(f"Persona build failed: {e}  (run manually: python main.py persona build)")

    # ── Step 8: Gap analysis ──────────────────────────────────────────────────
    header("Step 8/9 — Running knowledge gap analysis")
    if args.no_gaps or args.no_llm:
        warn("Skipping gap analysis. Run later: python main.py gap")
    else:
        try:
            from brain.agents.gap_agent import GapAgent
            cfg = _get_config(args)
            from brain.extract.relations import RelationExtractor
            extractor = RelationExtractor.from_config(cfg)
            gap_agent = GapAgent(store, builder, extractor, cfg
                                )   # structural only on first run
            report = gap_agent.run(llm_enrich=False)
            high = report.high_priority()
            ok(f"{len(report.gaps)} gaps found  ({len(high)} high-priority)")
            for g in high[:3]:
                info(f"  🔴 {g.title}")
        except Exception as e:
            warn(f"Gap analysis failed: {e}")

    # ── Step 9: Visualize ─────────────────────────────────────────────────────
    header("Step 9/9 — Exporting visualization")
    try:
        from brain.visualize.export import GraphExporter
        web_dir = ROOT / "web"
        web_dir.mkdir(exist_ok=True)
        exporter = GraphExporter(store, builder)
        exporter.export_json(str(web_dir / "graph_data.json"))
        ok(f"Graph exported → {web_dir}/index.html")
    except Exception as e:
        warn(f"Export failed: {e}")

    # ── Final summary ─────────────────────────────────────────────────────────
    stats = store.stats()
    elapsed = "done"

    if HAS_RICH:
        console.print(Panel(
            f"[bold green]BRAIN IS ALIVE[/bold green]\n\n"
            f"  Notes:       [bold]{stats['notes']:,}[/bold]\n"
            f"  Edges:       [bold]{stats['edges']:,}[/bold]\n"
            f"  Embeddings:  [bold]{stats['notes_with_embeddings']:,}[/bold]\n"
            f"  Clusters:    [bold]{stats['clusters']}[/bold]\n"
            f"  Tags:        [bold]{len(stats['tags'])}[/bold]\n\n"
            f"[cyan]Next steps:[/cyan]\n"
            f"  python main.py visualize           # browse the graph\n"
            f"  python main.py query 'What do I think about X?'\n"
            f"  python main.py gap                 # knowledge gap report\n"
            f"  python main.py analyze             # writing fingerprint\n"
            f"  python main.py persona show        # intellectual DNA\n"
            f"  python main.py generate makemore 'consciousness'\n"
            f"  python main.py contradictions      # resolve conflicts\n\n"
            f"  # Add to cron for nightly maintenance:\n"
            f"  0 2 * * * python {ROOT}/scripts/consolidate.py",
            border_style="green",
            title="Summary",
        ))
    else:
        print(f"\n{'='*60}")
        print(f"  BRAIN IS ALIVE")
        print(f"  Notes:    {stats['notes']:,}")
        print(f"  Edges:    {stats['edges']:,}")
        print(f"  Clusters: {stats['clusters']}")
        print(f"\n  Next: python main.py visualize")
        print(f"        python main.py query 'Your question here'")
        print(f"{'='*60}\n")


# ─── Helpers ──────────────────────────────────────────────────────────────────

def _check_deps(args):
    required = [("networkx", "pip install networkx"),
                ("yaml",     "pip install pyyaml")]
    optional = [("fitz",         "pip install pymupdf",          "PDF extraction (best)"),
                ("pypdf",        "pip install pypdf",             "PDF extraction (alt)"),
                ("anthropic",    "pip install anthropic",         "Claude API"),
                ("openai",       "pip install openai",            "OpenAI/DeepSeek API"),
                ("rich",         "pip install rich",              "Pretty terminal output"),
                ("community",    "pip install python-louvain",    "Louvain clustering")]

    all_good = True
    for mod, install in required:
        try:
            __import__(mod)
            ok(mod)
        except ImportError:
            err(f"{mod} MISSING — run: {install}")
            all_good = False

    for mod, install, desc in optional:
        try:
            __import__(mod)
            ok(f"{mod}  ({desc})")
        except ImportError:
            warn(f"{mod} not installed — {desc}  [{install}]")

    if not all_good:
        err("Missing required dependencies. Install them and retry.")
        sys.exit(1)

    # Check Ollama if no API keys
    if not args.no_llm:
        import os
        has_api = any(os.environ.get(k) for k in [
            "ANTHROPIC_API_KEY", "OPENAI_API_KEY", "DEEPSEEK_API_KEY"
        ])
        if not has_api:
            # Try Ollama
            try:
                import urllib.request
                urllib.request.urlopen("http://localhost:11434/api/tags", timeout=2)
                ok("Ollama running locally")
            except Exception:
                warn(
                    "No API keys found and Ollama not running.\n"
                    "      LLM features will be skipped.\n"
                    "      To use Ollama: https://ollama.ai  then: ollama pull mistral\n"
                    "      To use Claude: export ANTHROPIC_API_KEY=sk-ant-..."
                )


def _resolve_path(given: str | None, label: str,
                   default: Path | None = None,
                   required: bool = True) -> Path | None:
    if given:
        p = Path(given).expanduser()
        if p.exists():
            ok(f"{label}: {p}")
            return p
        else:
            warn(f"{label} not found: {p}")
            return None

    if default and default.exists():
        ok(f"{label}: {default}  (default)")
        return default

    if required:
        warn(f"{label}: not found at default location ({default})")
        try:
            user_input = input(f"      Enter path for {label} (or press Enter to skip): ").strip()
            if user_input:
                p = Path(user_input).expanduser()
                if p.exists():
                    return p
                warn(f"Path not found: {p}")
        except (EOFError, KeyboardInterrupt):
            pass
    return None


def _get_config(args) -> dict:
    import os
    cfg = {
        "llm_backend":           "claude",
        "anthropic_api_key":     os.environ.get("ANTHROPIC_API_KEY", ""),
        "ollama_base_url":       os.environ.get("BRAIN_OLLAMA_URL", "http://localhost:11434"),
        "ollama_model":          os.environ.get("BRAIN_OLLAMA_MODEL", "mistral"),
        "embedding_backend":     "local",
        "local_embedding_model": "all-MiniLM-L6-v2",
        "claude_model":          "claude-haiku-4-5-20251001",
    }
    if args.no_llm:
        cfg["llm_backend"] = "ollama"

    config_path = ROOT / "config.yaml"
    if config_path.exists():
        try:
            import yaml
            cfg.update(yaml.safe_load(config_path.read_text()) or {})
        except Exception:
            pass
    return cfg


if __name__ == "__main__":
    main()
