# Digital Brain: A Provenance-Aware Neuro-Symbolic Memory System



This repository implements a digital brain designed for continual personal knowledge consolidation. It is not just another note-taking app or a flat vector database. It is a **neuro-symbolic memory system** that ingests raw artifacts (Org-mode notes, web clippings, PDFs, chat logs), converts them into a structured graph, maintains that memory over time via an automated consolidation loop, and answers queries through adaptive retrieval and agentic reasoning.

## Core Architecture

This system moves beyond standard Retrieval-Augmented Generation (RAG) by utilizing four distinct memory layers and an active maintenance loop.

* **Ingestion (The Senses):** Parses explicit top-down notes (Emacs `.org` files) and passive bottom-up data (JSON chat logs, Markdown web clips, PDFs) into standardized, atomic `Note` objects.
* **Neuro-Symbolic Graph (The Substrate):** Combines dense vector retrieval (semantic embeddings) with a symbolic knowledge graph (NetworkX/SQLite). Edges are formed explicitly via links, implicitly via tags, semantically via cosine similarity, and logically via LLM-extracted relations.
* **Continual Consolidation (The Sleep Cycle):** A periodic agent loop that runs community detection (Louvain), calculates node importance (PageRank), merges duplicate claims, flags contradictions within semantic clusters, and applies confidence decay based on recency and centrality.
* **Query Orchestrator (The Conscious Agent):** A LangGraph-powered state machine that routes questions through local note recall, vector retrieval, or graph traversal. Includes a human-in-the-loop fallback if retrieval confidence falls below a threshold.
* **Visualization (The Mind's Eye):** A D3.js force-directed physics graph that visually maps your knowledge, sizing nodes by centrality and coloring them by topic cluster.

## Repository Structure

    digital-brain/
    ├── main.py                          # The CLI control panel
    ├── requirements.txt                 # Dependencies
    ├── data/                            # Local SQLite storage (auto-generated)
    ├── web/
    │   ├── index.html                   # D3.js visualization frontend
    │   └── graph_data.json              # D3 data payload (auto-generated)
    └── brain/
        ├── agents/                      # LangGraph orchestrators
        ├── extract/                     # LLM relation and claim extractors
        ├── ingest/                      # Parsers for Org-mode, PDFs, Web clips
        ├── memory/                      # SQLite Store, NetworkX Graph, Embeddings
        ├── query/                       # Query planning and routing logic
        └── visualize/                   # JSON exporters for the frontend



## Usage

The system is controlled entirely through the `main.py` CLI. 

### Step 1: Ingest Your Life
Point the brain at your primary knowledge folder (e.g., your Emacs Org-roam directory). It will parse `.org`, `.md`, `.json` (chat logs), and `.pdf` files.

    python main.py ingest ~/notes/org-roam


### Step 2: Build the Substrate
Generate semantic embeddings for your notes, build the NetworkX graph, calculate PageRank centralities, and detect topical communities.

    python main.py build


### Step 3: Visualize the Graph
Export the graph math to JSON and spin up a local web server to explore your brain in 2D space.

    python main.py visualize

*Navigate to http://localhost:8000 in your browser.*

### Step 4: Query the Agent
Ask complex, cross-note questions. If the agent isn't confident in the retrieval, it will pause execution and ask you for clarification in the terminal.

    python main.py query "Based on my writings, how do I view the intersection of bottom-up Zettelkasten and top-down folder structures?"


### Step 5: Sleep / Consolidate
Run the nightly continual learning loop to deduplicate claims, flag contradictions, and update confidence scores. (Best run as a cron job).

    python main.py consolidate

