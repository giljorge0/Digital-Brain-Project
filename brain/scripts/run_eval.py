#!/usr/bin/env python3
"""
brain/scripts/run_eval.py  —  FIXED
-------------------------------------
Benchmarks three retrieval strategies against each other.

Key fix: ROOT now points to the PROJECT ROOT (not brain/), so
Store("data/brain.db") resolves correctly.
"""

import sys, json, argparse, logging, math, os
from pathlib import Path
from datetime import datetime

# ── Path fix ──────────────────────────────────────────────────
# __file__ = brain/scripts/run_eval.py
# .parent        → brain/scripts/
# .parent.parent → brain/
# .parent.parent.parent → PROJECT ROOT  ← this is the fix
ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from brain.memory.store      import Store
from brain.memory.embeddings import EmbeddingProvider
from brain.query.planner     import QueryPlanner

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger("eval")

DEFAULT_QUESTIONS = ROOT / "brain" / "data" / "eval" / "questions.jsonl"
STRATEGIES        = ["semantic", "graph_traversal", "temporal"]


def _load_cfg() -> dict:
    cfg = {
        "llm_backend":           os.environ.get("LLM_BACKEND", "claude"),
        "anthropic_api_key":     os.environ.get("ANTHROPIC_API_KEY", ""),
        "embedding_backend":     os.environ.get("EMBEDDING_BACKEND", "local"),
        "local_embedding_model": "all-MiniLM-L6-v2",
        "ollama_base_url":       os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434"),
        "ollama_model":          os.environ.get("OLLAMA_MODEL", "mistral"),
        "claude_model":          os.environ.get("CLAUDE_MODEL", "claude-haiku-4-5-20251001"),
        "vector_backend":        "sqlite",
    }
    for p in [ROOT / "config.yaml", ROOT / "llm_profiles.yaml"]:
        if p.exists():
            try:
                import yaml
                with open(p) as f:
                    cfg.update(yaml.safe_load(f) or {})
                cfg["vector_backend"] = "sqlite"   # always use sqlite for eval
            except Exception as e:
                log.warning(f"Could not parse {p.name}: {e}")
    return cfg


def _preflight(store, embedder) -> dict:
    """Print DB diagnostics and return stats."""
    s = store.stats()
    print(f"\n{'='*60}")
    print(f"  DATABASE DIAGNOSTICS")
    print(f"  Path: {store.db_path if hasattr(store,'db_path') else 'data/brain.db'}")
    print(f"  Notes:             {s.get('notes', 0):,}")
    print(f"  Edges:             {s.get('edges', 0):,}")
    print(f"  With embeddings:   {s.get('notes_with_embeddings', 0):,}")
    print(f"  Clusters:          {s.get('clusters', 0)}")
    print(f"{'='*60}\n")

    if s.get("notes", 0) == 0:
        print("⚠  DATABASE IS EMPTY.")
        print("   Run:  python main.py ingest <your-notes-dir>")
        print("         python main.py build")
        sys.exit(1)

    if s.get("notes_with_embeddings", 0) == 0:
        print("⚠  NO EMBEDDINGS FOUND — semantic search will return nothing.")
        print("   Run:  python main.py build")
        print("   (continuing anyway for temporal/graph modes)\n")

    return s


def _auto_gold(store, embedder, question: str, top_k: int = 3) -> list:
    """
    If gold_note_ids is empty, auto-generate approximate gold IDs by
    doing a semantic search and taking the top results as ground truth.
    Labels these as '[auto]' in output.
    """
    try:
        q_vec = embedder.embed_one(question)
        from brain.memory.embeddings import search_by_embedding
        hits  = search_by_embedding(store, q_vec, top_k=top_k)
        return [nid for nid, _ in hits]
    except Exception:
        return []


def main():
    parser = argparse.ArgumentParser(description="Digital Brain eval — retrieval benchmark")
    parser.add_argument("--questions", default=str(DEFAULT_QUESTIONS))
    parser.add_argument("--no-llm",    action="store_true")
    parser.add_argument("--save",      metavar="FILE")
    parser.add_argument("--strategy",  choices=STRATEGIES)
    parser.add_argument("--top-k",     type=int, default=10)
    parser.add_argument("--auto-gold", action="store_true",
                        help="Auto-generate gold IDs for questions that lack them")
    args = parser.parse_args()

    # ── Load questions ─────────────────────────────────────────
    q_path = Path(args.questions)
    if not q_path.exists():
        log.warning(f"No questions file at {q_path}. Creating sample file…")
        _create_sample_questions(q_path)
        log.info(f"Edit {q_path}, fill in gold_note_ids, then re-run.")
        sys.exit(0)

    questions = [json.loads(l) for l in q_path.read_text().splitlines() if l.strip()]
    log.info(f"Loaded {len(questions)} questions from {q_path}")

    # ── Setup ──────────────────────────────────────────────────
    cfg      = _load_cfg()
    db_path  = ROOT / "data" / "brain.db"
    store = Store("data/core/brain.db")
    embedder = EmbeddingProvider.from_config(cfg)
    planner  = QueryPlanner(store, embedder, cfg)

    stats = _preflight(store, embedder)
    strategies = [args.strategy] if args.strategy else STRATEGIES

    # ── Evaluate ───────────────────────────────────────────────
    results = []
    for q in questions:
        log.info(f"[{q.get('type','?')}] {q['question'][:70]}")

        gold = q.get("gold_note_ids", [])
        gold_source = "provided"

        if not gold and args.auto_gold:
            gold = _auto_gold(store, embedder, q["question"])
            gold_source = "auto"
            log.info(f"  → auto gold IDs: {gold}")

        row = {
            "id":          q["id"],
            "question":    q["question"],
            "type":        q.get("type", "unknown"),
            "gold_ids":    gold,
            "gold_source": gold_source,
        }

        for strategy in strategies:
            mode = {"semantic": "semantic", "graph_traversal": "graph", "temporal": "temporal"}[strategy]
            try:
                result        = planner.query(q["question"], mode=mode, top_k=args.top_k)
                sources       = result.get("sources", [])
                retrieved_ids = [s["id"]    for s in sources]
                ret_titles    = [s.get("title", s["id"])[:40] for s in sources[:5]]

                row[strategy] = {
                    "retrieved_ids":    retrieved_ids,
                    "retrieved_titles": ret_titles,
                    "confidence":       result.get("confidence", 0.0),
                    "hit_rate":         _hit_rate(retrieved_ids, gold),
                    "mrr":              _mrr(retrieved_ids, gold),
                    "ndcg":             _ndcg(retrieved_ids, gold),
                    "citation_overlap": None,
                }

                if not args.no_llm and gold:
                    answer = result.get("answer", "")
                    gold_notes = [store.get_note(nid) for nid in gold if store.get_note(nid)]
                    row[strategy]["citation_overlap"] = _citation_overlap(answer, gold_notes)

            except Exception as e:
                log.error(f"  Strategy {strategy} error: {e}")
                row[strategy] = {"error": str(e), "hit_rate": 0.0, "mrr": 0.0,
                                 "ndcg": 0.0, "citation_overlap": None,
                                 "retrieved_ids": [], "retrieved_titles": []}

        results.append(row)

    _print_report(results, strategies)

    if args.save:
        report = {
            "generated_at": datetime.now().isoformat(),
            "db_path":      str(db_path),
            "db_stats":     stats,
            "n_questions":  len(questions),
            "strategies":   strategies,
            "top_k":        args.top_k,
            "results":      results,
            "summary":      _summarise(results, strategies),
        }
        Path(args.save).write_text(json.dumps(report, indent=2))
        log.info(f"Full report saved to {args.save}")


# ── Metrics ────────────────────────────────────────────────────
def _hit_rate(retrieved, gold):
    if not gold: return 0.0
    return float(any(g in retrieved for g in gold))

def _mrr(retrieved, gold):
    if not gold: return 0.0
    for i, nid in enumerate(retrieved):
        if nid in gold: return 1.0 / (i + 1)
    return 0.0

def _ndcg(retrieved, gold, k=10):
    if not gold: return 0.0
    gold_set = set(gold)
    def dcg(ids):
        return sum((1.0 if nid in gold_set else 0.0) / math.log2(i + 2)
                   for i, nid in enumerate(ids[:k]))
    ideal = dcg(list(gold_set)[:k])
    return dcg(retrieved) / ideal if ideal > 0 else 0.0

def _citation_overlap(answer, gold_notes):
    if not gold_notes or not answer: return 0.0
    hits = sum(1 for n in gold_notes if n.title.lower() in answer.lower())
    return round(hits / len(gold_notes), 3)


# ── Report ─────────────────────────────────────────────────────
def _summarise(results, strategies):
    n = max(len(results), 1)
    summary = {}
    for s in strategies:
        rows = [r[s] for r in results if s in r and "error" not in r[s]]
        if not rows: continue
        summary[s] = {
            "hit_rate": round(sum(r["hit_rate"] for r in rows) / n, 3),
            "mrr":      round(sum(r["mrr"]      for r in rows) / n, 3),
            "ndcg":     round(sum(r["ndcg"]      for r in rows) / n, 3),
        }
        cit = [r["citation_overlap"] for r in rows if r.get("citation_overlap") is not None]
        if cit: summary[s]["citation_overlap"] = round(sum(cit) / len(cit), 3)
    return summary

def _print_report(results, strategies):
    n = len(results)
    summary = _summarise(results, strategies)
    print(f"\n{'='*65}")
    print(f"  EVAL RESULTS  —  {n} questions  —  {datetime.now():%Y-%m-%d %H:%M}")
    print(f"{'='*65}")
    print(f"  {'Strategy':<22}  {'Hit@10':>7}  {'MRR':>7}  {'NDCG@10':>8}  {'Cite%':>6}")
    print(f"  {'─'*22}  {'─'*7}  {'─'*7}  {'─'*8}  {'─'*6}")
    for s in strategies:
        sm  = summary.get(s, {})
        cit = f"{sm['citation_overlap']:.3f}" if "citation_overlap" in sm else "  n/a"
        print(f"  {s:<22}  {sm.get('hit_rate',0):>7.3f}  {sm.get('mrr',0):>7.3f}"
              f"  {sm.get('ndcg',0):>8.3f}  {cit:>6}")
    print(f"\n  Per-question breakdown:")
    for r in results:
        has_gold = "✓" if r["gold_ids"] else "○"
        src_tag  = f"[{r.get('gold_source','?')[:4]}]" if r["gold_ids"] else ""
        print(f"  {has_gold} {src_tag} [{r['type'][:8]:<8}] {r['question'][:52]}")
        for s in strategies:
            if s not in r: continue
            d = r[s]
            if "error" in d:
                print(f"    {s:<22} ERROR: {d['error'][:45]}"); continue
            titles = ", ".join(d.get("retrieved_titles", [])[:3]) or "(no results)"
            print(f"    {s:<22} hit={d['hit_rate']:.0f} mrr={d['mrr']:.2f} ndcg={d['ndcg']:.2f} | {titles}")
    print()


def _create_sample_questions(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    samples = [
        {"id": "q1", "question": "What are my core arguments about consciousness?",          "gold_note_ids": [], "type": "factual"},
        {"id": "q2", "question": "How did my view on free will evolve over time?",           "gold_note_ids": [], "type": "temporal"},
        {"id": "q3", "question": "What notes connect epistemology and language?",             "gold_note_ids": [], "type": "graph"},
        {"id": "q4", "question": "Where do I hold contradictory positions?",                 "gold_note_ids": [], "type": "contradiction"},
        {"id": "q5", "question": "What have I written about the limits of formal logic?",    "gold_note_ids": [], "type": "factual"},
        {"id": "q6", "question": "What thinkers have most influenced my thinking?",          "gold_note_ids": [], "type": "factual"},
        {"id": "q7", "question": "What did I write about AI in 2023?",                       "gold_note_ids": [], "type": "temporal"},
    ]
    path.write_text("\n".join(json.dumps(q) for q in samples))


if __name__ == "__main__":
    main()
