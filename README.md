# Digital Brain — Provenance-Aware Neuro-Symbolic Memory System


## What This Is

This is not a note app. It is a **living memory system** built for people who write seriously — philosophers, researchers, autodidacts — who need their ideas to connect, evolve, and be queryable at scale.

The system ingests your raw writing (Emacs `.org` files, chat logs, Kindle clippings, PDFs, web clips, YouTube history, Goodreads exports), converts it into a graph of atomic ideas, runs a continual consolidation loop to maintain that graph, and lets you query it through an AI agent that routes questions through the right combination of symbolic graph traversal, semantic search, and LLM reasoning.

Beyond retrieval, it builds a **persona profile** of your intellectual identity, generates text in your voice, auto-maintains a living wiki of your key concepts, and detects gaps in your thinking with targeted reading recommendations.

**The key architectural bet:** retrieval is not enough. A personal knowledge system needs *memory formation*, *memory maintenance*, *gap detection*, and *generative synthesis* — not just search.

---

## Architecture

```
┌──────────────────────────────────────────────────────────────────┐
│                          DIGITAL BRAIN                            │
│                                                                   │
│  ┌──────────┐    ┌───────────────┐    ┌────────────────────────┐ │
│  │  INGEST  │───▶│   MEMORY /    │───▶│  VISUALIZATION         │ │
│  │          │    │    GRAPH      │    │  (D3.js + export)      │ │
│  │ .org     │    │               │    └────────────────────────┘ │
│  │ .md      │    │ episodic      │                               │
│  │ .json    │    │ semantic      │    ┌────────────────────────┐ │
│  │ .pdf     │    │ graph (NX)    │───▶│  QUERY AGENT           │ │
│  │ kindle   │    │ vector index  │    │  (LangGraph FSM)       │ │
│  │ youtube  │    └──────┬────────┘    └────────────────────────┘ │
│  │ goodreads│           │                                        │
│  └──────────┘    ┌──────▼────────┐    ┌────────────────────────┐ │
│                  │ CONSOLIDATION │    │  GAP AGENT             │ │
│                  │ (nightly)     │    │  (thin orchestrator)   │ │
│                  │ dedup/merge   │    │                        │ │
│                  │ clusters      │    │  gap_finder.py  ──────▶│ │
│                  │ PageRank      │    │  recommender.py ──────▶│ │
│                  │ decay         │    └────────────────────────┘ │
│                  └───────────────┘                               │
│                                                                   │
│  ┌─────────────────────────────┐  ┌───────────────────────────┐  │
│  │  PERSONA ENGINE             │  │  AUTO WIKI                │  │
│  │                             │  │                           │  │
│  │  distiller.py               │  │  auto_wiki.py             │  │
│  │  → topical fingerprint      │  │  → one page per concept   │  │
│  │  → stylistic markers        │  │  → versioned, cross-linked│  │
│  │  → stance map               │  │  → written in your voice  │  │
│  │  → temporal arc             │  │  → exports to Markdown    │  │
│  │                             │  │                           │  │
│  │  generator.py               │  └───────────────────────────┘  │
│  │  → expand / respond         │                                  │
│  │  → makemore / synthesize    │                                  │
│  └─────────────────────────────┘                                  │
│                                                                   │
│  LLM LAYER: Claude | GPT-4 | DeepSeek | Gemini | Perplexity      │
│             + Local Ollama   (multiple accounts per provider)     │
└──────────────────────────────────────────────────────────────────┘
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
Digital-Brain-Project/
├── README.md
├── config.yaml
├── llm_profiles.yaml
├── main.py
├── requirements.txt
└── brain/
    ├── agents/
    │   ├── gap_agent.py          # Thin orchestrator: delegates to gap_finder + recommender
    │   └── query_agent.py        # LangGraph FSM for multi-hop Q&A
    ├── analysis/
    │   ├── gap_finder.py         # Six structural gap detectors
    │   └── recommender.py        # Three-mode recommendation engine (anonymous / local / ZK)
    ├── extract/
    │   └── relations.py          # LLM relation + claim extraction (Claude / Ollama)
    ├── ingest/
    │   ├── importers.py          # All data importers (see "Ingest" section)
    │   ├── note.py               # Atomic Note dataclass
    │   ├── org-parser.py         # Emacs org-roam + heading-split parser
    │   └── providers.py          # LLM provider abstraction layer
    ├── memory/
    │   ├── consolidation.py      # Nightly dedup / contradiction / decay loop
    │   ├── embeddings.py         # Local / Voyage / Ollama embedding backends
    │   ├── graph.py              # NetworkX graph builder + Louvain + PageRank
    │   └── store.py              # SQLite store (notes / edges / embeddings)
    ├── persona/
    │   ├── distiller.py          # Extracts intellectual DNA from corpus
    │   └── generator.py          # Generates text in your voice
    ├── query/
    │   └── planner.py            # Semantic / keyword / graph / temporal / hybrid routing
    ├── scripts/
    │   ├── consolidate.py
    │   ├── index_all.py
    │   └── run_eval.py
    ├── visualize/
    │   └── export.py             # D3.js JSON exporter
    ├── web/
    │   └── index.html            # Force-directed graph UI
    └── wiki/
        └── auto_wiki.py          # Living concept wiki generator
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
# Parses .org, .md, .pdf, .json, Kindle clippings, Goodreads CSV, YouTube history
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
python main.py gap
python main.py gap --types void depth width        # run specific detectors
python main.py recommend --mode anonymous          # gap detection + reading recommendations
```

### 8. Build your persona profile

```bash
python main.py persona build    # analyses corpus → saves data/persona.json
python main.py persona show     # prints profile, stances, temporal arc
```

### 9. Generate text in your voice

```bash
python main.py generate expand <note-id>                    # expand an existing note
python main.py generate respond "What do I think about X?"  # answer as yourself
python main.py generate makemore "phenomenology" --n 5      # 5 note ideas from a seed
python main.py generate synthesize "consciousness" --save   # wiki-style synthesis note
```

### 10. Auto-maintain your concept wiki

```bash
python main.py wiki update --top-n 20    # generate/refresh top 20 concept pages
python main.py wiki show                  # list all pages
python main.py wiki show --concept "consciousness"
python main.py wiki export --output wiki/ # export all pages to Markdown files
```

### 11. Nightly consolidation (add to cron)

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

- `heavy` — expensive, slow; used for relation extraction and consolidation
- `daily` — fast; used for query answering
- `embed` — embeddings only; always use local Ollama to keep costs zero
- `gap_analysis` — gap detection and one-sided claim analysis

---

## Ingest — What the System Can Read

`importers.py` handles format detection automatically. Drop files in a directory and run `python main.py ingest <path>`.

| Source | File / Format | What gets extracted |
|--------|--------------|-------------------|
| **Emacs org-roam** | `*.org` | One note per file (roam) or per heading; `:ID:` preserved |
| **Markdown web clips** | `*.md` with YAML frontmatter | Title, URL, domain tag |
| **ChatGPT export** | `conversations.json` | One note per conversation; key questions, insights, decisions, code extracted |
| **Claude export** | `*.json` (uuid format) | Same semantic extraction as ChatGPT |
| **PDFs** | `*.pdf` | Full text via PyMuPDF (falls back to PyPDF2) |
| **Kindle highlights** | `My Clippings.txt` | One note per book; all highlights chronologically ordered |
| **Goodreads** | CSV export | One note per book; rating, shelves, review text, read date |
| **YouTube watch history** | `watch-history.json` (Takeout) | Grouped by week; top channels as tags |
| **YouTube search history** | `search-history.json` (Takeout) | Grouped by month |
| **Google search history** | `MyActivity.json` (Takeout) | Grouped by month; navigational queries filtered |

For LLM chat logs, the importer goes beyond raw text and extracts a **semantic highlight block** per conversation: questions asked, decisions reached, insights flagged, and whether code was produced. This makes chat history actually searchable by what happened in the conversation, not just its words.

---

## Gap Agent — Finding What You Don't Know You're Missing

The gap detection pipeline is split cleanly into two modules with `gap_agent.py` as a thin orchestrator over both:

- `brain/analysis/gap_finder.py` — detects gaps (the "what is missing?" question)
- `brain/analysis/recommender.py` — generates recommendations (the "what should I read?" question)
- `brain/agents/gap_agent.py` — runs both, formats the daily briefing

```bash
python main.py gap                              # full briefing
python main.py gap --types void depth           # specific detectors only
python main.py recommend --mode anonymous       # gap + recommendations
```

### Gap detectors

| Detector | What it finds |
|----------|--------------|
| **Void** | Concepts adjacent to your clusters that you haven't entered at all |
| **Depth** | Highly-referenced notes that are stubs (high in-degree, low word count) |
| **Width** | Canonical intellectual siblings you haven't engaged with |
| **Temporal** | High-centrality ideas last touched 6+ months ago |
| **Contradiction** | Conflicting claims already flagged by the consolidation agent |
| **Orthogonal** | Steel-man counterarguments to your most central positions |

### Recommendation modes

| Mode | How it works | Privacy |
|------|-------------|---------|
| **anonymous** | Sends only the gap description to the LLM — never your notes | Your corpus stays local |
| **local** | Queries an offline index you build from arXiv abstracts | Fully air-gapped |
| **ZK** *(research frontier)* | Commits to a gap vector; proves relevance without revealing the vector | Requires homomorphic encryption or ZK proof system |

The privacy design is intentional: the service provider can host recommendations without ever learning your intellectual profile. The gap description is the smallest useful external patch.

Example output:

```
╔══════════════════════════════════════════════════╗
║        DIGITAL BRAIN — DAILY BRIEFING            ║
║        2026-04-28                                 ║
╚══════════════════════════════════════════════════╝

🕳  VOID GAPS — Adjacent territory you haven't entered

  • Tarski's undefinability theorem
    You engage deeply with Gödel but haven't written about Tarski.
    Anchor: note on formal limits of axiomatic systems
    Recommended next steps:
      → [book] Introduction to Metamathematics — Kleene
      → [paper] The Concept of Truth in Formalized Languages — Tarski

📐  DEPTH GAPS — Notes referenced often but underdeveloped

  • Intelligence as Compression
    Referenced 12× but only 67 words. Load-bearing node in your graph.
    → Expand into a full essay.
```

---

## Persona Engine — A Digital Model of You

The persona pipeline extracts your intellectual DNA from the corpus and uses it to generate text that sounds like you wrote it.

### Build your profile

```bash
python main.py persona build
```

`brain/persona/distiller.py` runs five extractors over your entire note corpus:

| Extractor | What it produces |
|-----------|----------------|
| **Topical fingerprint** | Weighted tag and concept distribution — your intellectual territory |
| **Stylistic markers** | Sentence rhythm, vocabulary richness, punctuation habits (em-dashes, parentheticals) |
| **Intellectual lineage** | Thinkers and authors you cite most, extracted by heuristic NER |
| **Argument patterns** | How you build arguments: conditional, evidential, contrastive, question-driven |
| **Temporal arc** | How your dominant topics have shifted year by year |

The LLM then synthesizes these into a **stance map** (your stated position on each of your top 10 topics) and a **self-description** (a 200-word portrait of you as a thinker, written in second person).

```bash
python main.py persona show
```

```
─ SELF DESCRIPTION ─
You are someone who approaches questions about mind and language through the
lens of formal limits. You are drawn to moments where a system's power is
exactly what prevents it from seeing itself...

─ STANCES ─
  [consciousness]    Functionalism is insufficient; phenomenal character
                     resists computational reduction.
  [language]         Meaning is use, but use is always already social —
                     private rule-following is incoherent.

─ TEMPORAL ARC ─
  2021: 47 notes, dominant topic: epistemology
  2022: 89 notes, dominant topic: philosophy_of_mind
  2023: 134 notes, dominant topic: ai_limits
  2024: 201 notes, dominant topic: neuro_symbolic
```

### Generate in your voice

`brain/persona/generator.py` uses the profile to condition generation:

```bash
# Write more of an existing note, drawing on graph neighbors for material
python main.py generate expand <note-id>

# Answer a question as you would write it, grounded in your notes and stance map
python main.py generate respond "What do I think about attention mechanisms?"

# Given a seed, generate N note ideas you'd naturally want to write next
python main.py generate makemore "phenomenology of perception" --n 5

# Synthesize everything you've written on a topic into one new note
python main.py generate synthesize "epistemic limits" --save
```

Every generation call injects the persona's style guide, self-description, and relevant stances into the prompt. The model is constrained to your established positions — it will not invent new stances or contradict your corpus.

---

## Auto Wiki — Living Pages for Every Concept

Inspired by Karpathy's LLM wiki idea: for each high-centrality concept in your graph, the system auto-generates and auto-maintains a Wikipedia-style article synthesizing everything you've written about it.

```bash
python main.py wiki update --top-n 20   # generate / refresh top 20 concept pages
python main.py wiki export --output wiki/  # export all to Markdown
python main.py wiki show --concept "consciousness"
```

`brain/wiki/auto_wiki.py` identifies concepts by combining PageRank centrality with tag frequency, then for each concept:

- Gathers all source notes (by tag + keyword search)
- Calls the LLM to write a synthesis article in your voice
- Adds cross-links to related concepts (`[[like this]]`)
- Stores the result as a first-class `Note` in the store (tagged `wiki_page`)
- Tracks **versions** — each re-generation stores the previous content in metadata, so you can see how your understanding of a concept has evolved

Wiki pages participate in the graph: they can be retrieved by queries, linked to by other notes, and refined by the consolidation loop. The `Further Questions` section of each page becomes a natural source of depth and void gaps.

---

## WordPress Export

```bash
python main.py export-wp
# Writes web/graph_embed.html — a self-contained iframe-ready graph
# Paste into any WordPress page/post using the HTML block
```

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
- [x] Knowledge gap agent — refactored as thin orchestrator over `gap_finder` + `recommender`
- [x] Six gap detectors (void, depth, width, temporal, contradiction, orthogonal)
- [x] Three-mode recommender (anonymous, local, ZK frontier)
- [x] Kindle `My Clippings.txt` importer (one note per book, highlights chronologically ordered)
- [x] YouTube watch + search history importer (Google Takeout format)
- [x] LLM chat history importer — ChatGPT and Claude export formats, with semantic highlight extraction
- [x] Goodreads CSV importer (rating, shelves, review text)
- [x] Google search history importer (Takeout format)
- [x] Persona distiller — topical fingerprint, stylistic markers, stance map, temporal arc
- [x] Persona generator — expand, respond, makemore, synthesize
- [x] Auto wiki — living versioned concept pages, cross-linked, exported to Markdown
- [ ] Neo4j backend option (for very large graphs)
- [ ] Chroma / Qdrant vector backend option
- [ ] ZK proof-of-relevance for privacy-preserving remote recommendations
- [ ] WordPress REST API auto-publish
- [ ] Emacs org-capture integration (capture → brain in one keystroke)
- [ ] Mobile-friendly graph view
- [ ] Contradiction resolution UI
- [ ] Benchmark eval suite (run_eval.py fully wired)
- [ ] Scheduled wiki refresh (cron job, diff-patch mode)
- [ ] Persona profile versioning (track how your stances evolve over time)

---

## Design Philosophy

**Bottom-up + top-down**: Emacs org folders give top-down partitioning. Org-roam IDs and semantic similarity build the bottom-up graph. Both layers coexist.

**Provenance over convenience**: Every note knows its source, timestamp, and confidence. This makes the system analyzable and publishable as research.

**Local-first**: Ollama handles embeddings and daily queries for free. Cloud APIs are reserved for heavy reasoning tasks. Your notes never leave your machine unless you explicitly choose a cloud provider.

**One idea, one note**: The Zettelkasten principle. The graph emerges from many small atomic notes, not from a few big documents.

**Model you, not your attention**: The recommendation system optimizes for your intellectual growth, not engagement. The gap is the product. The preference model is local. Each briefing is a function of your own knowledge structure — not an optimization for time-on-platform.

**Clean separation of concerns**: Gap detection (`gap_finder.py`) and recommendation generation (`recommender.py`) are independent modules. `gap_agent.py` is a thin orchestrator. This means you can run detectors without recommendations, swap recommendation backends, or add new gap types without touching the others.

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
