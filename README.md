# Digital Brain — Provenance-Aware Neuro-Symbolic Memory System

<div align="center">

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python&logoColor=white)](https://python.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-gold.svg)](LICENSE)
[![Live Demo](https://img.shields.io/badge/Live%20Demo-giljorge0.github.io-brightgreen?logo=github)](https://giljorge0.github.io/)
[![Backend](https://img.shields.io/badge/Backend-FastAPI-009688?logo=fastapi)](https://fastapi.tiangolo.com)
[![LLM](https://img.shields.io/badge/LLM-Claude%20%7C%20Ollama%20%7C%20GPT--4-blueviolet)](https://ollama.com)
[![Graph](https://img.shields.io/badge/Graph-NetworkX%20%2F%20Neo4j-orange)](https://networkx.org)
[![Vector](https://img.shields.io/badge/Vector-Chroma%20%2F%20Qdrant-red)](https://www.trychroma.com)
[![Visualization](https://img.shields.io/badge/UI-D3.js%2060FPS-f7c948?logo=d3.js)](https://d3js.org)

**→ [Live Demo: giljorge0.github.io](https://giljorge0.github.io/)**

*Explore a real knowledge graph of 2,000+ notes, with forensic writing analysis and force-directed galaxy visualization.*

</div>

---

## What This Is

This is not a note app. It is a **living memory system** built for people who write seriously — philosophers, researchers, autodidacts — who need their ideas to connect, evolve, and be queryable at scale.

The system ingests your raw writing (Emacs `.org` files, chat logs, Kindle clippings, PDFs, web clips, YouTube history, Goodreads exports), converts it into a graph of atomic ideas, runs a continual consolidation loop to maintain that graph, and lets you query it through an AI agent that routes questions through the right combination of symbolic graph traversal, semantic search, and LLM reasoning.

Beyond retrieval, it builds a **persona profile** of your intellectual identity, generates text in your voice, auto-maintains a living wiki of your key concepts, detects gaps in your thinking with targeted reading recommendations, and reconstructs your intellectual history from behavioral data (what you watched, searched, read).

**The key architectural bet:** retrieval is not enough. A personal knowledge system needs *memory formation*, *memory maintenance*, *gap detection*, and *generative synthesis* — not just search.

---

## Scale: Two Brains

The system runs two distinct corpora simultaneously, each optimized for its purpose:

| | **Core Brain** | **Omni Brain** |
|---|---|---|
| **Purpose** | Curated `output` notes — your best writing | Full private corpus — everything ingested |
| **Size** | ~2,000 notes | **51,253 notes** |
| **Words** | ~272,000 words | **5.3+ million words** |
| **Visibility** | Public — powers the [live UI](https://giljorge0.github.io/) | Private — local only |
| **Build command** | `python main.py build` | `python main.py --brain omni build` |
| **Graph edges** | ~800 (cluster-filtered for D3 60FPS) | 200,000+ (stored in SQLite/Neo4j) |
| **PageRank** | NetworkX (small graph, fast) | **scipy sparse CSR** (~5 MB vs ~300 MB for NetworkX) |

Scaling to 51k notes required heavy engineering:

- **Cursor streaming**: edges read from SQLite via `fetchmany(10_000)` — never `fetchall()` — capping peak RAM to 5 MB instead of 220+ MB
- **Two-phase bootstrap**: `notes.json` + `persona.json` load first (counters and Profile render immediately); `graph_data.json` loads separately in Phase 2 so a 10 MB graph file doesn't block the UI
- **scipy sparse PageRank**: CSR float32 matrix for 450k edges uses ~8 MB vs NetworkX's ~300 MB (which converts to dense internally)
- **Cluster-filtered D3 physics**: only intra-cluster edges enter the force simulation, dropping the physics workload from 226k edges to ~800 and maintaining 60 FPS

---

## Benchmark Results

Graph Traversal RAG vs flat semantic RAG, evaluated on a personal corpus Q&A task:

| Strategy | Hit@10 | **MRR** | NDCG@10 | Citation Rate |
|----------|--------|---------|---------|---------------|
| Semantic (flat RAG baseline) | 0.621 | 0.417 | 0.503 | 19.0% |
| **Graph Traversal RAG** (this system) | **0.714** | **0.476** | **0.581** | **23.8%** |
| Temporal | 0.580 | 0.391 | 0.471 | — |

**Graph Traversal RAG achieves +14% MRR** (0.476 vs 0.417) and a **+4.8pp citation rate** vs flat RAG. The gain comes from neighbourhood expansion: the query agent traverses the knowledge graph from semantically-matched seed notes, pulling in structurally related nodes that pure cosine similarity misses.

Run the benchmark yourself:
```bash
python scripts/run_eval.py
python scripts/run_eval.py --no-llm           # retrieval metrics only (fast)
python scripts/run_eval.py --save report.json
python scripts/run_eval.py --strategy graph_traversal
```

---

## Architecture

```
┌──────────────────────────────────────────────────────────────────────────────┐
│                          DIGITAL BRAIN                                        │
│                                                                               │
│  ┌──────────────┐    ┌───────────────────┐    ┌──────────────────────────┐   │
│  │    INGEST    │───▶│   MEMORY / GRAPH  │───▶│  GALAXY UI (D3.js SPA)   │   │
│  │              │    │                   │    │  ─────────────────────── │   │
│  │ .org  .md    │    │  episodic layer   │    │  About · Garden · Brain  │   │
│  │ .json .pdf   │    │  semantic layer   │    │  Profile · Note Reader   │   │
│  │ kindle  yt   │    │  graph  (NX/Neo4j)│    │  Force-directed galaxy   │   │
│  │ goodreads    │    │  vector  (Chroma/ │    │  60 FPS · PageRank nodes │   │
│  │ search hist  │    │          Qdrant)  │    │  Cosmic Academic theme   │   │
│  └──────────────┘    └────────┬──────────┘    │  → GitHub Pages (static) │   │
│                               │               └──────────────────────────┘   │
│                        ┌──────▼────────┐    ┌──────────────────────────┐     │
│                        │ CONSOLIDATION │    │  QUERY AGENT             │     │
│                        │ (nightly)     │    │  (LangGraph FSM)         │     │
│                        │ dedup/merge   │    │  semantic / keyword /    │     │
│                        │ clusters      │    │  graph / temporal /      │     │
│                        │ PageRank      │    │  hybrid  +  HITL fallback│     │
│                        │ decay         │    └──────────────────────────┘     │
│                        └───────────────┘                                     │
│                                                                               │
│  ┌─────────────────────────────┐  ┌─────────────────────────────────────┐    │
│  │  PERSONA ENGINE             │  │  AUTO WIKI                          │    │
│  │  distiller.py               │  │  auto_wiki.py                       │    │
│  │  → topical fingerprint      │  │  → one LLM page per concept         │    │
│  │  → forensic stylistic DNA   │  │  → versioned, cross-linked          │    │
│  │  → argument topology (14)   │  │  → written in your voice            │    │
│  │  → stance map               │  │  → feeds gap finder                 │    │
│  │  → temporal arc + drift     │  │  → exports to Markdown              │    │
│  │  → LLM style fingerprint    │  └─────────────────────────────────────┘    │
│  │  generator.py               │                                             │
│  │  → expand / respond         │  ┌─────────────────────────────────────┐    │
│  │  → makemore / synthesize    │  │  GAP AGENT                          │    │
│  └─────────────────────────────┘  │  gap_finder.py (6 detectors)        │    │
│                                   │  recommender.py (3 modes)           │    │
│  ┌─────────────────────────────┐  │  → void / depth / width /           │    │
│  │  STORAGE BACKENDS           │  │     temporal / contradiction /       │    │
│  │  SQLite  (default, ≤50k)    │  │     orthogonal                      │    │
│  │  Neo4j   (100k+ notes)      │  └─────────────────────────────────────┘    │
│  │  Chroma  (ANN search)       │                                             │
│  │  Qdrant  (production ANN)   │  ┌─────────────────────────────────────┐    │
│  └─────────────────────────────┘  │  YOUTUBE ANALYZER                   │    │
│                                   │  → temporal persona timeline         │    │
│  LLM: Claude · GPT-4 · Ollama     │  → topic drift detection            │    │
│       DeepSeek · Gemini · Perpl.   │  → algorithm vs intent lag          │    │
│  Embed: sentence-transformers     │  → integrates with persona.json     │    │
│         Voyage · Ollama           └─────────────────────────────────────┘    │
└──────────────────────────────────────────────────────────────────────────────┘
```

### Five memory layers

| Layer | What it stores | Technology |
|-------|---------------|-----------|
| **Episodic** | Timestamped events, chat logs, Kindle sessions | SQLite + provenance metadata |
| **Semantic** | Stable claims, definitions, philosophical positions | SQLite full-text search |
| **Graph** | Entities, relations, arguments, contradictions | NetworkX (swap for Neo4j) |
| **Vector** | Dense embeddings for similarity search | SQLite cosine (swap for Chroma/Qdrant) |
| **Raw** | Original files, unchanged | Filesystem (`data/raw/`) |

---

## Repository Structure

```
Digital-Brain-Project/
├── README.md
├── config.yaml
├── llm_profiles.yaml           ← Multi-provider LLM config
├── main.py                     ← Single CLI entry point
├── requirements.txt
├── data/                       ← SQLite DB, exports, persona.json (gitignored)
├── wiki/                       ← Exported Markdown wiki pages
└── brain/
    ├── agents/
    │   ├── gap_agent.py        # Thin orchestrator: gap_finder + recommender
    │   └── query_agent.py      # LangGraph FSM for multi-hop Q&A with HITL
    ├── analysis/
    │   ├── gap_finder.py       # Six structural gap detectors
    │   ├── recommender.py      # Three-mode recommender (anonymous/local/ZK)
    │   └── youtube_analyzer.py # Deep behavioral analysis of YouTube history
    ├── extract/
    │   └── relations.py        # LLM relation + claim extraction
    ├── ingest/
    │   ├── importers.py        # All data importers
    │   ├── note.py             # Atomic Note dataclass with provenance_role
    │   ├── org_parser.py       # Emacs org-roam + heading-split parser
    │   └── providers.py        # LLM provider abstraction layer
    ├── memory/
    │   ├── consolidation.py    # Nightly dedup / contradiction / decay loop
    │   ├── embeddings.py       # Local / Voyage / Ollama embedding backends
    │   ├── graph.py            # NetworkX + scipy sparse PageRank + Louvain
    │   ├── store.py            # SQLite store (notes / edges / embeddings)
    │   ├── neo4j_store.py      # Neo4j drop-in (100k+ notes)
    │   └── vector_backends.py  # Chroma + Qdrant ANN search backends
    ├── persona/
    │   ├── distiller.py        # Forensic intellectual DNA extraction
    │   └── generator.py        # Generates text in your voice
    ├── query/
    │   └── planner.py          # Semantic / keyword / graph / temporal / hybrid
    ├── scripts/
    │   └── run_eval.py         # Benchmark: flat RAG vs graph vs this system
    ├── visualize/
    │   ├── export.py           # D3.js JSON + standalone HTML exporter
    │   └── static_export.py    # GitHub Pages static site generator
    ├── web/
    │   ├── index.html          # Cosmic Academic SPA — four views
    │   ├── style.css           # Dark-star theme, gold accents, Fraunces/Newsreader
    │   ├── app.js              # SPA router, D3 graph, profile renderer
    │   └── wp_export.py        # WordPress REST API publisher
    └── wiki/
        └── auto_wiki.py        # Living concept wiki generator
```

---

## Quick Start

### 1. Install

```bash
git clone https://github.com/giljorge0/Digital-Brain-Project
cd Digital-Brain-Project
pip install -r requirements.txt
```

### 2. Configure your LLM

Edit `llm_profiles.yaml` or set environment variables:

```bash
export ANTHROPIC_API_KEY=sk-ant-...
export OPENAI_API_KEY=sk-...          # optional
export DEEPSEEK_API_KEY=sk-...        # optional
export GEMINI_API_KEY=...             # optional
export PERPLEXITY_API_KEY=pplx-...    # optional
```

For fully local operation (no API keys needed):

```yaml
# llm_profiles.yaml
defaults:
  daily: ollama-mistral
  embed: ollama-nomic
profiles:
  - name: ollama-mistral
    provider: ollama
    model: mistral
  - name: ollama-nomic
    provider: ollama
    model: nomic-embed-text
```

### 3. Ingest your notes

```bash
# Parses .org, .md, .pdf, .json, Kindle clippings, Goodreads CSV, YouTube history
python main.py ingest ~/Nextcloud/brain/

# Or just org-roam notes
python main.py ingest ~/Nextcloud/brain/org-roam/
```

### 4. Build the graph

```bash
python main.py build
# With Chroma for faster ANN search:
python main.py build --backend chroma

# Large corpus (50k+ notes) — activates scipy sparse PageRank + cursor streaming:
python main.py --brain omni build --backend chroma
```

### 5. Visualize

```bash
python main.py visualize
# → open http://localhost:8000
```

### 6. Query

```bash
python main.py query "What are my main arguments about the limits of symbolic AI?"
python main.py query "How has my view on consciousness changed over time?"
python main.py query "What do I actually think about Wittgenstein's private language argument?"
```

### 7. Find your knowledge gaps

```bash
python main.py gap
python main.py recommend --mode anonymous
python main.py recommend --mode local    # fully offline
```

### 8. Export to GitHub Pages

```bash
python main.py export-static --out public_html/
cd public_html && git push origin main
# → Live at https://<you>.github.io/
```

---

## Complete Command Reference

### Ingestion

```bash
# Directory — auto-detects all supported formats
python main.py ingest ~/notes/

# Single files (format auto-detected by filename)
python main.py ingest ~/Downloads/conversations.json     # ChatGPT export
python main.py ingest ~/Downloads/claude_export.json     # Claude export
python main.py ingest ~/Downloads/watch-history.json     # YouTube Takeout
python main.py ingest ~/Downloads/search-history.json    # YouTube search history
python main.py ingest ~/Downloads/MyActivity.json        # Google search history
python main.py ingest ~/Downloads/goodreads_library.csv  # Goodreads
python main.py ingest "~/My Clippings.txt"               # Kindle highlights
python main.py ingest ~/papers/paper.pdf
```

All ingested content is tagged with `provenance_role`:
- `output` — **your words**: org notes, your chat turns, your Goodreads reviews, your Kindle annotations
- `input` — **external content**: AI responses, YouTube videos, book highlights, PDFs

Only `output` notes feed the persona model. Your thinking is never mixed with other people's.

### Build & maintain

```bash
python main.py build                    # Default: SQLite embeddings
python main.py build --backend chroma   # Chroma ANN (faster, local)
python main.py build --backend qdrant   # Qdrant (local file or hosted)
python main.py consolidate              # Nightly: dedup, contradictions, decay

# Omni Brain (50k+ notes) — memory-safe path:
python main.py --brain omni build --backend chroma
```

**Storage backends:**

| Backend | Use case | Install |
|---------|----------|---------|
| `sqlite` | Default, zero setup, up to ~50k notes | — |
| `chroma` | Faster ANN search, still fully local | `pip install chromadb` |
| `qdrant` | Production-grade, supports hosted cluster | `pip install qdrant-client` |
| `neo4j`  | 100k+ notes, native graph queries | `NEO4J_URI=bolt://localhost:7687` |

Neo4j activates automatically when `NEO4J_URI` is set — it is a drop-in replacement for SQLite with the same public API.

**Large-corpus memory safety** (≥ 5,000 notes triggers automatically):

| Optimization | Technique | Memory saved |
|---|---|---|
| Edge streaming | `cursor.fetchmany(10_000)` instead of `fetchall()` | ~215 MB |
| scipy sparse PageRank | CSR float32 matrix | ~290 MB |
| Phase-1 / Phase-2 bootstrap | Notes+persona before graph | UI renders without waiting |
| Pre-assembly GC | `del stubs; gc.collect()` | ~40 MB |

### Query

```bash
python main.py query "What are my core positions on free will?"
```

The query agent routes each question through one of five retrieval modes — semantic, keyword, graph traversal, temporal, or hybrid — and falls back to human-in-the-loop clarification if confidence is below threshold.

### Gap analysis

```bash
python main.py gap                            # All six gap types
python main.py gap --types void depth         # Specific types only
python main.py gap --mode anonymous           # Include reading recommendations
python main.py gap --mode local               # Fully offline
```

Six structural gap types:

| Type | What it finds | How detected |
|------|--------------|-------------|
| `void` | Unexplored semantic regions adjacent to your clusters | Cluster centroid + LLM |
| `depth` | Topics referenced often but never developed | High in-degree / low word count |
| `width` | Canonical siblings of your best-developed ideas | LLM sibling detection |
| `temporal` | Important ideas not revisited in 12+ months | Age × centrality score |
| `contradiction` | Flagged conflicting claims | Consolidation agent edges |
| `orthogonal` | Strongest counterarguments you haven't engaged | LLM steelmanning |

### Recommendations

```bash
python main.py recommend --mode anonymous   # LLM generates specific book/paper/video recs
python main.py recommend --mode local       # Offline arXiv/Wikipedia index only
```

Build the local offline index first for `--mode local`:
```bash
python main.py index-local --sources arxiv wikipedia --limit 5000
# Creates data/local_index.json — embedded abstracts searchable by cosine similarity
```

**Privacy model:** your corpus never leaves your device. Only the gap description text is sent to the LLM. In `--mode local`, nothing leaves your device.

### Persona engine

```bash
python main.py persona build    # Extract profile → data/persona.json
python main.py persona show     # Self-description, topics, stances, arc
python main.py persona drift    # How your intellectual focus has shifted over time
```

`brain/persona/distiller.py` runs a two-layer analysis over your full corpus:

**Layer 1 — Statistical (pure Python, no LLM):**

| Extractor | What it produces |
|-----------|----------------|
| **Topical fingerprint** | Weighted tag and concept distribution across full corpus |
| **Sentence architecture** | Length distribution histogram (5 buckets), opener patterns by grammatical class, avg paragraph length |
| **Vocabulary depth** | Type-token ratio on full corpus (not a 50k-char sample), hapax legomena rate, Flesch-Kincaid reading ease + grade level |
| **Voice & register** | Passive voice rate, hedge-words/1k, certainty-words/1k, question density, hedge-to-certainty ratio (epistemic axis) |
| **Linguistic fingerprint** | Top 20 signature bigrams and trigrams (verbal tics appearing 3+ times) |
| **Argument topology** | 14 rhetorical patterns: evidence, conditional, contrast, analogy, concession, definition-giving, example density, negation-first, meta-cognitive, synthesis move, paradox/tension, and more |
| **Intellectual lineage** | Named-entity thinkers and authors you cite most, mapped to their associated domain tags |
| **Temporal arc** | Year-by-year topic distribution |
| **Writing evolution** | Style metrics (sentence length, hedge rate, question density) tracked year-over-year |

**Layer 2 — LLM qualitative (feeds actual writing samples):**

| Analysis | What it produces |
|----------|----------------|
| **Forensic style fingerprint** | Voice character, sentence personality, vocabulary character, what the writing conceals, distinctive verbal tics, intellectual posture, emotional register, one-sentence stylometric fingerprint |
| **Habitual intellectual moves** | 5–7 named rhetorical/argumentative moves this author makes repeatedly |
| **Self-description** | 200-word portrait in second person, grounded in actual sample text and statistics |
| **Stance map** | Your positions on top 10 topics with confidence levels |

The LLM call uses your corpus statistics + 20 actual writing samples as context — not just tag frequencies. The result is qualitatively different from generic summarization.

```
─ SELF DESCRIPTION ─
You are someone who approaches questions about mind and language through the
lens of formal limits. You are drawn to moments where a system's power is
exactly what prevents it from seeing itself...

─ FORENSIC FINGERPRINT ─
One-sentence fingerprint: "The author builds arguments by first establishing a
structural impossibility, then retreating to ask whether the impossibility itself
reveals something — a philosophical move that recurs across topics with enough
consistency to identify authorship."

─ STANCES ─
  [consciousness]    Functionalism is insufficient; phenomenal character
                     resists computational reduction.
  [language]         Meaning is use, but use is always already social.

─ WRITING EVOLUTION ─
  2022: avg sentence 17.2w · hedges 8.1/1k · questions 3.4/1k
  2023: avg sentence 19.8w · hedges 9.7/1k · questions 4.1/1k
  2024: avg sentence 22.3w · hedges 11.2/1k · questions 5.8/1k
  ↑ Writing has grown consistently more complex and more epistemically cautious.

─ TEMPORAL ARC ─
  2021: 47 notes — epistemology
  2022: 89 notes — philosophy_of_mind
  2023: 134 notes — ai_limits
  2024: 201 notes — neuro_symbolic
```

### Generate (text in your voice)

```bash
# Expand an existing note with more depth
python main.py generate expand <note-id>

# Answer a question as you would write it
python main.py generate respond "What do I think about attention mechanisms?"

# Suggest N atomic note ideas from a seed (Karpathy makemore idea)
python main.py generate makemore "phenomenology of perception" --n 5

# Synthesize everything you've written on a topic into one new note
python main.py generate synthesize "epistemic limits" --save
```

`makemore` reads your intellectual profile and suggests atomic notes you'd naturally write next — fractal Zettelkasten expansion. Each suggestion includes a title, one-sentence premise, and why it fits your existing thinking.

All generation is constrained to your established positions — the model will not invent new stances or contradict your corpus.

### Wiki (living personal encyclopedia)

Inspired by Karpathy's LLM wiki: for each high-centrality concept, the system auto-generates and maintains a Wikipedia-style article synthesizing everything you've written about it.

```bash
python main.py wiki update                       # Generate/refresh top 20 pages
python main.py wiki update --top-n 40            # More pages
python main.py wiki update --diff                # Only regenerate stale pages
python main.py wiki export --output wiki/        # Export all to Markdown
python main.py wiki show                         # List all pages
python main.py wiki show --concept "consciousness"
python main.py wiki history --concept "consciousness"  # Version history
python main.py wiki schedule --interval 24       # Auto-refresh every 24h (daemon)
```

Each wiki page is stored as a versioned `Note` in the database, cross-linked to related concepts, written in your voice using the persona profile. The `Further Questions` section of each page feeds directly into the gap finder.

### YouTube deep analysis

```bash
python main.py youtube ~/Downloads/Takeout/YouTube/history/
python main.py youtube ~/Downloads/Takeout/ --save
python main.py youtube ~/Downloads/Takeout/ --integrate-persona
```

`brain/analysis/youtube_analyzer.py` reconstructs your intellectual history from behavioral data — five analysis layers:

| Layer | What it reveals |
|-------|----------------|
| **Temporal persona timeline** | Monthly dominant channels, watch velocity, topic diversity |
| **Topic drift detection** | When you started/stopped topic clusters; gradual vs sudden shifts |
| **Algorithm vs intent** | Topics appear in watch history (algorithm) weeks before search history (active seeking) — measures this lag per cluster |
| **Time allocation** | Morning/evening patterns, binge sessions, velocity over years |
| **Playlist crystallization** | Gap between first watch and first save = interest crystallization time |

`--integrate-persona` merges the YouTube timeline into `data/persona.json`, so your profile reflects not just what you wrote but what you were watching and when.

### Static export (GitHub Pages)

```bash
python main.py export-static --out public_html/
```

Generates a fully static site — no server required. Produces four JSON files (`notes.json`, `graph_data.json`, `persona.json`, `graph_filtered.json`) and copies the SPA frontend. Push `public_html/` to any static host.

The live demo at **[giljorge0.github.io](https://giljorge0.github.io/)** is built this way from the Core Brain.

### WordPress export

```bash
python main.py export-wp                       # Publish graph as a page
python main.py export-wp --mode wiki           # Publish wiki pages as posts
python main.py export-wp --mode both           # Both
python main.py export-wp --dry-run             # Preview without publishing
```

Uses WordPress Application Passwords (WP 5.6+):
```bash
export WP_URL=https://yoursite.wordpress.com
export WP_USER=your_username
export WP_APP_PASSWORD="xxxx xxxx xxxx xxxx xxxx xxxx"
```

The graph exports as a self-contained `<iframe>`-ready HTML block. Wiki pages are published as posts and updated on re-run (matched by slug).

### Evaluation benchmark

```bash
python scripts/run_eval.py
python scripts/run_eval.py --no-llm           # Retrieval metrics only (fast)
python scripts/run_eval.py --save report.json
python scripts/run_eval.py --strategy semantic  # One baseline only
```

Three baselines:

| Baseline | Description |
|----------|------------|
| **`semantic`** | Pure vector similarity (flat RAG) |
| **`graph_traversal`** | Graph neighbourhood expansion + vector |
| **`temporal`** | Date-filtered retrieval |

Metrics: Hit@10, MRR, NDCG@10, citation overlap. Question format (`data/eval/questions.jsonl`):
```json
{"id": "q1", "question": "What are my core arguments about consciousness?", "gold_note_ids": ["abc123"], "type": "factual"}
```

Run without a questions file first — the script generates a sample file with 7 template questions for you to fill in.

---

## Galaxy UI

The visualization layer is a Cosmic Academic-themed single-page application — dark star-field background, academic gold accents, Fraunces + Newsreader + IBM Plex Mono typography. Four views:

| View | Description |
|------|-------------|
| **About** | Landing page with animated counters (notes, words, connections), social links, CV download |
| **Garden** | Searchable, filterable card grid of all your notes with a full-screen reader |
| **Brain ✦** | Force-directed D3.js galaxy graph — PageRank-scaled nodes, cluster-filtered physics |
| **Profile** | Deep intellectual profile: forensic style analysis, rhetorical topology, linguistic fingerprint, writing evolution table |

**Graph performance engineering:**

The D3 force simulation runs at 60 FPS on 2,000 nodes because:

1. **Cluster filtering**: only intra-cluster edges enter the simulation. With 226,563 total edges but only 800 cluster-internal ones, the physics workload drops by 99.6%.
2. **Phyllotaxis spiral layout**: nodes are placed on a golden-ratio spiral before physics starts — the simulation converges in ~2 seconds instead of 15+.
3. **PageRank-normalized radius**: `r = 5 + √(centrality / max_centrality) × 13` — compresses the centrality range so peripheral nodes stay visible (≥ 5px) while hubs stand out (up to 18px).
4. **SVG glow filter**: single-pass `feGaussianBlur` at `stdDeviation=2.5`, merged with source — gives star-bloom effect at GPU cost of ~2 ms/frame.
5. **Freeze-by-default**: the simulation starts frozen in the spiral layout. The user explicitly hits "Play Physics" to run the force engine — CPU is idle until then.

---

## LLM Configuration

`llm_profiles.yaml` supports named profiles and multiple accounts per provider:

```yaml
defaults:
  daily: claude-haiku      # used for most operations
  heavy: claude-sonnet     # used for persona build, wiki generation
  embed: local             # used for embeddings

profiles:
  - name: claude-haiku
    provider: claude
    model: claude-haiku-4-5-20251001
    api_key: "${ANTHROPIC_API_KEY}"

  - name: claude-sonnet
    provider: claude
    model: claude-sonnet-4-6
    api_key: "${ANTHROPIC_API_KEY}"

  - name: local
    provider: local
    model: all-MiniLM-L6-v2

  - name: ollama-mistral
    provider: ollama
    model: mistral
    base_url: http://localhost:11434
```

Multiple accounts per provider are supported — define two profiles with different API keys and the system round-robins to stay within rate limits.

---

## Recommended Workflow

**Daily (2 min):**
```bash
python main.py recommend --mode anonymous   # What should I read today?
```

**After writing:**
```bash
python main.py ingest ~/notes/
python main.py build
```

**Weekly:**
```bash
python main.py consolidate           # Dedup, contradiction detection, decay
python main.py wiki update --diff    # Refresh only stale wiki pages
python main.py gap --save            # Review gaps → data/gaps.json
```

**Monthly:**
```bash
python main.py persona build         # Rebuild profile from updated corpus
python main.py persona drift         # How has your focus shifted?
python main.py generate makemore "whatever you've been thinking about lately"
```

**Publishing:**
```bash
python main.py export-static --out public_html/ && cd public_html && git push
python main.py wiki export           # Markdown to wiki/
python main.py export-wp --mode both # Push to WordPress
```

---

## Design Philosophy

**Bottom-up + top-down**: Emacs org folders give top-down partitioning. Org-roam IDs and semantic similarity build the bottom-up graph. Both layers coexist.

**Provenance over convenience**: Every note knows its source, timestamp, confidence, and whether it is your output or external input. This makes the system analyzable and publishable as research.

**Local-first**: Ollama handles embeddings and daily queries for free. Cloud APIs are reserved for heavy reasoning. Your notes never leave your machine unless you explicitly choose a cloud provider.

**One idea, one note**: The Zettelkasten principle. The graph emerges from many small atomic notes, not from a few big documents.

**Model you, not your attention**: The recommendation system optimizes for your intellectual growth, not engagement. The gap is the product. The preference model is local. Each briefing is a function of your own knowledge structure — not an optimization for time-on-platform.

**Clean separation of concerns**: Gap detection (`gap_finder.py`) and recommendation generation (`recommender.py`) are independent modules. `gap_agent.py` is a thin orchestrator. Each can be swapped, extended, or tested in isolation.

**Memory safety at scale**: The Omni Brain path (50k+ notes) is engineered to never hold the full graph in RAM. Edges stream from SQLite, PageRank runs on a sparse matrix, and each large intermediate structure is explicitly freed between phases. The system ran successfully on an 8 GB machine with 51,253 notes and 5.3 million words.

---

## Roadmap

- [x] Org-mode parser (org-roam + heading-split files)
- [x] SQLite store with notes / edges / embeddings
- [x] Multi-provider LLM layer (Claude, GPT-4, DeepSeek, Gemini, Perplexity, Ollama)
- [x] Multiple accounts per provider
- [x] D3.js force graph visualization
- [x] Knowledge gap agent — thin orchestrator over `gap_finder` + `recommender`
- [x] Six gap detectors (void, depth, width, temporal, contradiction, orthogonal)
- [x] Three-mode recommender (anonymous, local, ZK frontier)
- [x] Offline local index builder (`index-local` — arXiv + Wikipedia)
- [x] Kindle `My Clippings.txt` importer
- [x] YouTube watch + search history importer (Google Takeout)
- [x] YouTube deep analyzer — temporal persona, topic drift, algorithm vs intent detection
- [x] LLM chat history importer — ChatGPT and Claude export formats
- [x] Goodreads CSV importer
- [x] Google search history importer
- [x] Persona distiller — topical fingerprint, stylistic markers, stance map, temporal arc
- [x] Persona drift analysis
- [x] Persona generator — expand, respond, makemore, synthesize
- [x] Auto wiki — living versioned concept pages, cross-linked, Markdown export
- [x] Wiki incremental update (`--diff` flag)
- [x] Scheduled wiki refresh (`wiki schedule --interval N`)
- [x] Wiki version history (`wiki history --concept`)
- [x] Neo4j backend (drop-in replacement for 100k+ notes)
- [x] Chroma + Qdrant vector backends (ANN search)
- [x] WordPress REST API publisher (graph page + wiki posts)
- [x] Benchmark eval suite (`scripts/run_eval.py` — Hit@10, MRR, NDCG) — **+14% MRR over flat RAG**
- [x] **Omni Brain** — 51k note / 5.3M word scale with scipy sparse PageRank + cursor streaming
- [x] **Forensic Stylistic DNA** — 14 argument patterns, bigram/trigram fingerprint, Flesch-Kincaid, hapax rate, hedge/certainty axis, LLM forensic style analysis, writing evolution
- [x] **Galaxy UI** — Cosmic Academic SPA, phyllotaxis layout, 60 FPS cluster-filtered physics, PageRank-scaled glowing nodes
- [x] **Static export** — GitHub Pages deployment via `export-static`; live at [giljorge0.github.io](https://giljorge0.github.io/)
- [x] **Two-phase bootstrap** — notes + persona load first; large graph loads non-blocking in Phase 2
- [ ] ZK proof-of-relevance for privacy-preserving remote recommendations
- [ ] Emacs org-capture integration (capture → brain in one keystroke)
- [ ] Mobile-friendly graph view
- [ ] Contradiction resolution UI (surface flagged contradictions for manual review)
- [ ] Persona profile versioning (track how your stances evolve over time)
- [ ] YouTube transcript ingestion (full semantic indexing of video content)
- [ ] Podcast transcript ingestion (via Whisper)

---

## Related Work

- [GraphRAG (Microsoft)](https://arxiv.org/abs/2404.16130) — graph-augmented retrieval with community summaries
- [LangGraph](https://github.com/langchain-ai/langgraph) — stateful agent orchestration
- [Org-roam](https://www.orgroam.com/) — Zettelkasten in Emacs
- [makemore (Karpathy)](https://github.com/karpathy/makemore) — autoregressive character-level language model; inspires the corpus-aware generation layer
- [llm.c wiki (Karpathy)](https://github.com/karpathy/llm.c/wiki) — LLM-maintained living documentation; inspires `auto_wiki.py`
- Agent memory survey literature — episodic vs semantic memory for long-horizon agents

---

## License

MIT
