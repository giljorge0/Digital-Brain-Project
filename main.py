"""
Digital Brain CLI
-----------------
The main entry point for the neuro-symbolic memory system.

Usage:
  python main.py ingest ~/my-notes
  python main.py build
  python main.py consolidate
  python main.py query "What are my main arguments about epistemic limits?"
  python main.py visualize
"""

import os
import sys
import argparse
import logging
from pathlib import Path

# Set up basic logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
log = logging.getLogger("brain_cli")

# Import our brain modules
from brain.memory.store import Store
from brain.memory.graph import GraphBuilder
from brain.memory.embeddings import EmbeddingProvider, embed_notes
from brain.memory.consolidation import ConsolidationAgent
from brain.extract.relations import RelationExtractor
from brain.query.planner import QueryPlanner
from brain.agents.query_agent import QueryOrchestrator
from brain.ingest.org_parser import OrgParser
from brain.ingest.importers import ImportManager
from brain.visualize.export import GraphExporter

DB_PATH = "data/brain.db"

def get_config():
    """Mock config. In a real app, load this from config.yaml."""
    return {
        "llm_backend": "claude",
        "anthropic_api_key": os.environ.get("ANTHROPIC_API_KEY", ""),
        "embedding_backend": "local",
        "local_embedding_model": "all-MiniLM-L6-v2"
    }

def cli_ingest(args):
    """Ingests files from a directory."""
    store = Store(DB_PATH)
    path = Path(args.path)
    
    if not path.exists():
        log.error(f"Path does not exist: {path}")
        return

    log.info(f"Scanning {path} for inputs...")
    notes = []
    
    # Org files (Emacs)
    org_parser = OrgParser()
    org_notes = org_parser.parse_directory(path)
    if org_notes:
        log.info(f"Found {len(org_notes)} Org notes.")
        notes.extend(org_notes)
        
    # Web clips
    web_notes = ImportManager.parse_web_clips(path)
    if web_notes:
        log.info(f"Found {len(web_notes)} Web clips.")
        notes.extend(web_notes)
        
    # Chat logs
    for chat_file in path.glob("*.json"):
        chat_notes = ImportManager.parse_llm_chats(chat_file)
        if chat_notes:
            log.info(f"Parsed chat file: {chat_file.name}")
            notes.extend(chat_notes)
            
    # PDFs
    pdf_notes = ImportManager.parse_pdf_text(path)
    if pdf_notes:
        log.info(f"Found {len(pdf_notes)} PDFs.")
        notes.extend(pdf_notes)

    if notes:
        store.upsert_notes(notes)
        log.info(f"Successfully ingested {len(notes)} total items into the brain.")
    else:
        log.warning("No valid files found to ingest.")

def cli_build(args):
    """Builds embeddings and network graph."""
    cfg = get_config()
    store = Store(DB_PATH)
    
    log.info("1. Generating Embeddings for new notes...")
    embedder = EmbeddingProvider.from_config(cfg)
    embed_notes(store, embedder)
    
    log.info("2. Rebuilding Knowledge Graph...")
    builder = GraphBuilder(store)
    G = builder.build()
    builder.compute_clusters(G)
    builder.compute_centrality(G)
    log.info("Build complete.")

def cli_consolidate(args):
    """Runs the nightly continual learning loop."""
    cfg = get_config()
    store = Store(DB_PATH)
    extractor = RelationExtractor.from_config(cfg)
    builder = GraphBuilder(store)
    
    agent = ConsolidationAgent(store, extractor, builder)
    agent.run_nightly_job()

def cli_query(args):
    """Asks a question using the LangGraph Orchestrator."""
    cfg = get_config()
    store = Store(DB_PATH)
    embedder = EmbeddingProvider.from_config(cfg)
    
    planner = QueryPlanner(store, embedder, cfg)
    agent = QueryOrchestrator(planner)
    
    print("\n" + "="*50)
    print(f"QUESTION: {args.question}")
    print("="*50 + "\n")
    
    answer = agent.ask(args.question)
    
    print("\n" + "="*50)
    print("FINAL ANSWER:")
    print("="*50)
    print(answer)

def cli_visualize(args):
    """Exports D3 JSON and starts a local server."""
    store = Store(DB_PATH)
    builder = GraphBuilder(store)
    exporter = GraphExporter(store, builder)
    
    exporter.export_json("web/graph_data.json")
    
    log.info("Starting local web server at http://localhost:8000")
    log.info("Press Ctrl+C to stop.")
    
    # Start a simple HTTP server in the web directory
    os.chdir("web")
    import http.server
    import socketserver
    
    Handler = http.server.SimpleHTTPRequestHandler
    with socketserver.TCPServer(("", 8000), Handler) as httpd:
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            log.info("Server stopped.")

def main():
    parser = argparse.ArgumentParser(description="Digital Brain Management CLI")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # Ingest command
    p_ingest = subparsers.add_parser("ingest", help="Ingest files from a directory")
    p_ingest.add_argument("path", help="Directory containing .org, .md, .json, or .pdf files")

    # Build command
    p_build = subparsers.add_parser("build", help="Calculate embeddings and compute graph metrics")

    # Consolidate command
    p_cons = subparsers.add_parser("consolidate", help="Run the continual learning/maintenance loop")

    # Query command
    p_query = subparsers.add_parser("query", help="Ask the brain a question")
    p_query.add_argument("question", help="The question to ask")

    # Visualize command
    p_vis = subparsers.add_parser("visualize", help="Export graph JSON and run a local UI server")

    args = parser.parse_args()

    # Make sure data directory exists
    Path("data").mkdir(exist_ok=True)

    if args.command == "ingest":
        cli_ingest(args)
    elif args.command == "build":
        cli_build(args)
    elif args.command == "consolidate":
        cli_consolidate(args)
    elif args.command == "query":
        cli_query(args)
    elif args.command == "visualize":
        cli_visualize(args)

if __name__ == "__main__":
    main()