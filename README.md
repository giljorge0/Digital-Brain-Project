# Digital Brain — Provenance-Aware Neuro-Symbolic Memory System

> *A personal knowledge graph that reads your writing, finds the gaps, and thinks alongside you.*

---

## What This Is

This is not a note app. It is a **living memory system** built for people who write seriously — philosophers, researchers, autodidacts — who need their ideas to connect, evolve, and be queryable at scale.

The system ingests your raw writing (Emacs `.org` files, chat logs, Kindle clippings, PDFs, web clips), converts it into a graph of atomic ideas, runs a continual consolidation loop to maintain that graph, and lets you query it through an AI agent that routes questions through the right combination of symbolic graph traversal, semantic search, and LLM reasoning.

**The key architectural bet:** retrieval is not enough. A personal knowledge system needs *memory formation*, *memory maintenance*, and *gap detection* — not just search.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                         DIGITAL BRAIN                        │
│                                                              │
│  ┌──────────┐    ┌──────────────┐    ┌───────────────────┐  │
│  │  INGEST  │───▶│   MEMORY /   │───▶│  VISUALIZATION    │  │
│  │          │    │    GRAPH     │    │  (D3.js + export) │  │
│  │ .org     │    │              │    └───────────────────┘  │
│  │ .md      │    │ episodic     │                            │
│  │ .json    │    │ semantic     │    ┌───────────────────┐  │
│  │ .pdf     │    │ graph (NX)   │───▶│  QUERY AGENT      │  │
│  │ kindle   │    │ vector index │    │  (LangGraph FSM)  │  │
│  └──────────┘    └──────┬───────┘    └───────────────────┘  │
│                         │                                    │
│                  ┌──────▼───────┐    ┌───────────────────┐  │
│                  │ CONSOLIDATION│    │   GAP AGENT        │  │
│                  │ (nightly)    │    │                    │  │
│                  │ dedup/merge  │    │ orphans            │  │
│                  │ clusters     │    │ depth gaps         │  │
│                  │ PageRank     │    │ one-sided claims   │  │
│                  │ decay        │    │ recommended reads  │  │
│                  └──────────────┘    └───────────────────┘  │
│                                                              │
│  LLM LAYER: Claude | GPT-4 | DeepSeek | Gemini | Perplexity │
│             + Local Ollama   (multiple accounts per provider)│
└─────────────────────────────────────────────────────────────┘
```

### Five memory layers

| Layer | What it stores | Technology |
|-------|---------------|-----------|
| **Episodic** | Timestamped events, chat logs, Kindle sessions | SQLite + provenance metadata |
| **Semantic** | Stable claims, definitions, philosophical positions | SQLite full-text search |
| **Graph** | Entities, relations, arguments, contradictions | NetworkX (export to Neo4j optional) |
| **Vector** | Dense embeddings for similarity search | SQLite + cosine (swap for Chroma/Qdrant) |
| **Raw** | Original files, unchanged | Filesystem (`data/raw/`) |

---

## Repository Structure

```
digital-brain/
├── main.py                     # CLI entry point
├── configs/
│   └── llm_profiles.yaml       # ALL your LLM accounts (see below)
├── brain/
│   ├── ingest/
│   │   ├── note.py             # Note dataclass (atomic unit)
│   │   ├── org_parser.py       # Emacs .org parser (roam + headings)
│   │   └── importers.py        # PDF, markdown, JSON chat logs, Kindle
│   ├── memory/
│   │   ├── store.py            # SQLite store (notes, edges, embeddings)
│   │   ├── graph.py            # NetworkX graph builder + PageRank/Louvain
│   │   ├── embeddings.py       # Embedding provider (local or API)
│   │   └── consolidation.py    # Nightly dedup / contradiction / decay loop
│   ├── extract/
│   │   └── relations.py        # LLM relation extractor (entity/edge mining)
│   ├── query/
│   │   └── planner.py          # Route: graph | vector | temporal | hybrid
│   ├── agents/
│   │   ├── query_agent.py      # LangGraph query orchestrator
│   │   └── gap_agent.py        # Knowledge gap + recommendation agent
│   ├── llm/
│   │   └── providers.py        # Multi-provider, multi-account LLM layer
│   └── visualize/
│       └── export.py           # D3.js JSON export + WordPress embed helper
├── web/
│   └── index.html              # Interactive force-directed graph (D3.js)
├── data/
│   ├── raw/                    # Your source files (never modified)
│   ├── processed/              # Parsed note JSON cache
│   └── eval/                   # Benchmark questions + scoring
└── scripts/
    ├── index_all.py            # One-shot full pipeline
    ├── consolidate.py          # Cron-friendly consolidation runner
    └── run_eval.py             # Benchmark evaluation
```

---

## Quick Start

### 1. Install

```bash
git clone https://github.com/giljorge0/Digital-Brain-Project
cd Digital-Brain-Project
pip install -r requirements.txt
```

### 2. Configure your LLM accounts

Copy the example config and fill in your API keys:

```bash
cp configs/llm_profiles.yaml configs/llm_profiles.yaml
# Edit the file — see "LLM Configuration" section below
```

Alternatively, set environment variables (the system reads these automatically):

```bash
export ANTHROPIC_API_KEY=sk-ant-...
export OPENAI_API_KEY=sk-...
export DEEPSEEK_API_KEY=sk-...
export GEMINI_API_KEY=...
export PERPLEXITY_API_KEY=pplx-...
```

### 3. Ingest your notes

```bash
# Point at your Emacs brain folder — parses .org, .md, .pdf, .json recursively
python main.py ingest ~/Nextcloud/brain/

# Or just org-roam notes
python main.py ingest ~/Nextcloud/brain/org-roam/
```

### 4. Build the graph

```bash
# Generates embeddings, builds NetworkX graph, runs PageRank + community detection
python main.py build
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
python main.py gaps
# Prints a report: orphaned ideas, shallow topics, one-sided claims, reading recommendations
```

### 8. Nightly consolidation (add to cron)

```bash
python main.py consolidate
# Deduplicates, merges near-identical claims, flags contradictions, decays stale nodes
```

---

## LLM Configuration

The system supports **multiple providers** and **multiple accounts per provider**. All configuration lives in `configs/llm_profiles.yaml`.

### Supported providers

| Provider | Models | Best for |
|----------|--------|---------|
| **Claude** (Anthropic) | `claude-opus-4-5`, `claude-sonnet-4-5` | Heavy reasoning, relation extraction |
| **OpenAI** | `gpt-4o`, `gpt-4o-mini` | Everyday queries |
| **DeepSeek** | `deepseek-chat`, `deepseek-reasoner` | Deep analysis, cost-effective |
| **Gemini** (Google) | `gemini-1.5-pro`, `gemini-1.5-flash` | Long context, multimodal (future) |
| **Perplexity** | `sonar-pro`, `sonar` | Gap agent: recommended reads with live web search |
| **Ollama** (local) | any model you've pulled | Daily driver, embeddings, zero cost |

### Multiple accounts per provider

You can add as many accounts as you need — useful if you have separate keys for personal and work projects:

```yaml
profiles:
  - name: claude_personal
    provider: claude
    model: claude-sonnet-4-5
    api_key: sk-ant-PERSONAL_KEY

  - name: claude_research
    provider: claude
    model: claude-opus-4-5
    api_key: sk-ant-RESEARCH_KEY
    role: heavy

  - name: deepseek_primary
    provider: deepseek
    model: deepseek-chat
    api_key: sk-DEEPSEEK_KEY
    role: daily

defaults:
  heavy:        claude_research
  daily:        deepseek_primary
  embed:        ollama_nomic_embed
  gap_analysis: claude_personal
```

### Roles

- `heavy` — expensive, slow, used for relation extraction and consolidation (run rarely)
- `daily` — fast, used for query answering (run often)
- `embed` — embeddings only; always use local Ollama here to keep costs zero
- `gap_analysis` — gap detection and one-sided claim analysis

---

## Gap Agent — Finding What You Don't Know You're Missing

```bash
python main.py gaps
python main.py gaps --no-llm    # structural analysis only (free, fast)
python main.py gaps --save gaps_report.json
```

The gap agent runs seven detectors:

| Detector | What it finds |
|----------|--------------|
| **Orphans** | Notes with no connections to anything else |
| **Sparse clusters** | Topic areas with only 1–2 notes |
| **Ghost references** | Concepts you link to but never wrote a note about |
| **Depth gaps** | Highly-referenced notes that are only a stub (< 100 words) |
| **Recency gaps** | Central ideas you haven't revisited in 6+ months |
| **One-sided claims** | Strong arguments with no counterargument anywhere in your corpus |
| **Recommended reads** | Published works / thinkers you should engage with, based on your graph topology (uses Perplexity for live search) |

Example output:

```
══════════════════════════════════════════════════════════════
  KNOWLEDGE GAP REPORT  —  2026-04-28
══════════════════════════════════════════════════════════════
  Notes: 847   Edges: 2,341
  Gaps found: 23  (4 high-priority)

  🔴 [depth]  High-traffic stub: «Intelligence as Compression»
      Referenced 12× by other notes but only 67 words long.
      → Expand into a full essay. This is a load-bearing node in your graph.

  🔴 [one_sided]  One-sided claim: «Consciousness is substrate-independent»
      This note argues strongly for functionalism without acknowledging
      biological naturalism (Searle) or phenomenal consciousness objections.
      → Write a companion note with the steel-man of the opposing view.

  🟡 [recency]  Stale core idea: «Language as World-Boundary»
      High centrality (0.082) but last touched 2023-11-14.
      → Re-read and write a follow-up or revision.

  🟢 [recommended]  Recommended reading based on your graph topology
      Given your focus on epistemic limits, compression, and symbolic AI,
      you should engage with: ...
```

---

## WordPress Export

```bash
python main.py export-wp
# Writes web/graph_embed.html — a self-contained iframe-ready graph
# Paste into any WordPress page/post using the HTML block
```

---

## The Karpathy Angle — Corpus Character Model

Inspired by Karpathy's makemore, the system can train a small character-level or token-level language model on your corpus, giving you:

- **Writing style analysis** — what's distinctive about your prose
- **Idea completion** — given a fragment, what would *you* likely write next
- **Note generation prompts** — seed ideas in your own voice

```bash
python scripts/train_corpus_model.py --epochs 10
python scripts/generate.py --seed "The limit of language is"
```

*(Requires PyTorch. Optional component — see `scripts/` folder.)*

---

## Evaluation Benchmark

A serious memory system needs to be measurable. The eval suite tests four question types across three baselines:

| Baseline | Description |
|----------|------------|
| **Flat RAG** | Vector similarity only |
| **GraphRAG** | Entity graph + community summaries |
| **This system** | Provenance + episodic/semantic split + consolidation + adaptive routing |

```bash
python scripts/run_eval.py
```

Question types: factual recall, cross-note synthesis, temporal reasoning ("how did my view change?"), contradiction detection.

---

## Roadmap

- [x] Org-mode parser (roam notes + heading-split files)
- [x] SQLite store with notes / edges / embeddings
- [x] Multi-provider LLM layer (Claude, GPT-4, DeepSeek, Gemini, Perplexity, Ollama)
- [x] Multiple accounts per provider
- [x] D3.js force graph visualization
- [x] Knowledge gap agent (7 detectors + LLM enrichment)
- [ ] Kindle clippings auto-importer with timestamp preservation
- [ ] YouTube transcript ingester (watch history → notes)
- [ ] LLM chat history bulk importer (ChatGPT, Claude exports)
- [ ] Neo4j backend option (for very large graphs)
- [ ] Chroma / Qdrant vector backend option
- [ ] WordPress REST API auto-publish
- [ ] Corpus character model (makemore-style)
- [ ] Emacs org-capture integration (capture → brain in one keystroke)
- [ ] Mobile-friendly graph view
- [ ] Contradiction resolution UI
- [ ] Benchmark eval suite

---

## Design Philosophy

**Bottom-up + top-down**: Emacs org folders give top-down partitioning. Org-roam IDs and semantic similarity build the bottom-up graph. Both layers coexist.

**Provenance over convenience**: Every note knows its source, timestamp, and confidence. This makes the system analyzable and publishable as research.

**Local-first**: Ollama handles embeddings and daily queries for free. Cloud APIs are reserved for heavy reasoning tasks.

**One idea, one note**: The Zettelkasten principle. The graph emerges from many small atomic notes, not from a few big documents.

---

## Related Work

- [GraphRAG (Microsoft)](https://arxiv.org/abs/2404.16130) — graph-augmented retrieval with community summaries
- [LangGraph](https://github.com/langchain-ai/langgraph) — stateful agent orchestration
- [Org-roam](https://www.orgroam.com/) — Zettelkasten in Emacs
- [makemore (Karpathy)](https://github.com/karpathy/makemore) — character-level language model
- Agent memory survey literature — episodic vs semantic memory for long-horizon agents

---

## License

MIT
