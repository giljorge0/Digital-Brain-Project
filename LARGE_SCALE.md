# Running the Omni Brain at Scale (30k–100k+ notes)

## What was crashing and why

| Crash | Cause | Fix |
|-------|-------|-----|
| RAM maxes out during `build` | `get_all_embeddings()` loads every vector as Python floats: 30k × 384 × 24 bytes = ~450 MB | `get_embeddings_numpy()` loads the same data as float32: ~46 MB |
| Machine freezes during semantic edges | `(30k × 30k)` sim matrix in numpy = 3.6 GB | Streaming batches of 256 rows at a time = ~30 MB peak |
| `build` takes hours, SQLite fills disk | `upsert_edge()` called 450k times = 450k `fsync` calls | `upsert_edges_batch()` = one `fsync` per 5k edges |
| Tag edge loop hangs | O(n²) all-pairs comparison | Inverted index: O(n × avg_tags) |

---

## Option A — SQLite + Numpy (works now, no setup needed)

This is the default path after the rewrite. Just run:

```bash
python main.py --brain omni build
```

**Memory profile at 30k notes:**
- Embedding pass:  ~800 MB peak (sentence-transformers model + one chunk)
- Semantic edges:  ~350 MB peak (256 × 30k float32 matrix = 30 MB × ~12 live)
- Tag edges:       ~200 MB (inverted index dict)
- NetworkX stub:   ~150 MB (no content, just IDs + edges)
- **Total peak:    ~1.3 GB**  ← well within 16 GB

At 100k notes the numpy matrix becomes `(256, 100k) × 4B = 100 MB` per batch,
still manageable, but Chroma (Option B) is faster and uses less RAM.

---

## Option B — Chroma for ANN semantic search (recommended for 30k+)

Chroma replaces the O(n²) numpy similarity matrix with an HNSW index.
Query time per note: `O(log n)` instead of `O(n)`.

### 1. Install

```bash
pip install chromadb
```

### 2. Build with Chroma sync

```bash
# First run: embed notes AND push to Chroma in one pass
python main.py --brain omni build --backend chroma
```

This sets `chroma_sync=True` in `embed_notes()`, which writes each embedding
to both SQLite and Chroma as it goes. No second pass needed.

### 3. What happens automatically

After `--backend chroma`:
- `embed_notes()` pushes each batch to `data/omni/chroma/` as it embeds
- `_build_semantic_edges_to_db()` detects Chroma and uses `col.query()` for ANN
- Each note does one Chroma query (top-16 neighbors) instead of one row of the matrix
- Memory peak drops to ~300 MB total for 100k notes

### 4. Add to config.yaml (optional, makes it permanent)

```yaml
# config.yaml
vector_backend: chroma
chroma_path: data/omni/chroma
chroma_collection: brain
```

---

## Option C — Neo4j for the graph store (100k+ notes, complex queries)

Neo4j replaces SQLite entirely for notes, edges, and graph traversal.
It's not needed for RAM — SQLite handles that fine with WAL mode.
Use Neo4j when you need: multi-hop Cypher queries, real-time graph
traversal in the web API, or native vector search (Neo4j 5.11+).

### 1. Start Neo4j

```bash
# Docker (easiest)
docker run -d \
  --name neo4j-brain \
  -p 7687:7687 \
  -p 7474:7474 \
  -e NEO4J_AUTH=neo4j/password \
  -v $(pwd)/data/neo4j:/data \
  neo4j:5

# Open browser: http://localhost:7474
# Login: neo4j / password
```

### 2. Install driver

```bash
pip install neo4j
```

### 3. Set env vars

```bash
export NEO4J_URI=bolt://localhost:7687
export NEO4J_USER=neo4j
export NEO4J_PASSWORD=password
```

### 4. Activate in main.py

In `main.py`, change the `get_store()` / `Store(DB_PATH)` call in `cli_build`:

```python
# main.py — add this function
def get_store_for_brain(brain_name: str, cfg: dict):
    use_neo4j = cfg.get("use_neo4j", False) or os.environ.get("USE_NEO4J")
    
    if use_neo4j:
        from brain.memory.neo4j_store import Neo4jStore
        return Neo4jStore()   # reads NEO4J_URI/USER/PASSWORD from env
    else:
        db_path = ROOT / "data" / brain_name / "brain.db"
        from brain.memory.store import Store
        return Store(db_path)
```

Then in `cli_build`:
```python
def cli_build(args):
    cfg   = get_config()
    store = get_store_for_brain(getattr(args, 'brain', 'core'), cfg)
    # ... rest unchanged
```

Or simply set in config.yaml:
```yaml
use_neo4j: true
```

### 5. Migrate existing SQLite data to Neo4j (one-time)

```bash
python - << 'EOF'
from brain.memory.store import Store
from brain.memory.neo4j_store import Neo4jStore

sqlite = Store("data/omni/brain.db")
neo4j  = Neo4jStore()

print("Migrating notes…")
notes = sqlite.get_all_notes()
neo4j.upsert_notes(notes)
print(f"  {len(notes)} notes done")

print("Migrating edges…")
for edge in sqlite.get_all_edges():
    neo4j.upsert_edge(
        edge["source"], edge["target"],
        edge["edge_type"], edge["weight"]
    )
print("  edges done")

print("Migrating embeddings…")
for chunk in sqlite.iter_embeddings_chunked(1000):
    for note_id, vec in chunk.items():
        neo4j.save_embedding(note_id, vec)
print("  embeddings done")
print("Migration complete.")
EOF
```

---

## Recommended setup for each corpus size

| Corpus size | Store | Embeddings | Semantic search | Peak RAM |
|-------------|-------|-----------|-----------------|----------|
| < 5k notes  | SQLite | ST/Ollama | Numpy all-pairs | < 500 MB |
| 5k–50k      | SQLite + WAL | ST/Ollama | Chroma ANN | < 2 GB |
| 50k–200k    | SQLite + WAL | ST/Ollama | Chroma ANN | < 4 GB |
| 200k+       | Neo4j | Ollama/OpenAI | Neo4j vector index | server-dependent |

For your 30k omni brain: **SQLite + Chroma is the sweet spot.**
Neo4j adds operational complexity without RAM benefit at this scale.

---

## Running the omni build end-to-end

```bash
# Option A: pure SQLite (no extra setup)
python main.py --brain omni build

# Option B: with Chroma ANN (recommended, one extra pip install)
pip install chromadb
python main.py --brain omni build --backend chroma

# After build:
python main.py --brain omni export-static

# Nightly cron:
0 2 * * * cd /path/to/Digital-Brain-Project && \
  python main.py --brain omni consolidate --no-llm
```

## Monitoring RAM during build

```bash
# In another terminal while build runs:
watch -n 2 "free -h && python -c \"
import sqlite3, os
db = 'data/omni/brain.db'
if os.path.exists(db):
    conn = sqlite3.connect(db)
    notes = conn.execute('SELECT COUNT(*) FROM notes').fetchone()[0]
    edges = conn.execute('SELECT COUNT(*) FROM edges').fetchone()[0]
    embs  = conn.execute('SELECT COUNT(*) FROM embeddings').fetchone()[0]
    print(f'Notes: {notes}  Edges: {edges}  Embedded: {embs}')
    conn.close()
\""
```
