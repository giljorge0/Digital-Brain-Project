#!/usr/bin/env python3
"""
scripts/run_eval.py
--------------------
Benchmark evaluation: compares three retrieval strategies on a set of
questions drawn from your own corpus.

Baselines:
  A  semantic        — pure vector similarity (top-k notes)
  B  graph_traversal — graph neighbourhood + vector (combined)
  C  temporal        — date-filtered retrieval

Evaluation metrics per question:
  - hit_rate           — did the top-10 results include the gold note(s)?
  - mrr                — mean reciprocal rank of first gold result
  - citation_overlap   — did the LLM answer cite the right notes? (needs --llm)
  - ndcg               — normalised discounted cumulative gain

Question format (data/eval/questions.jsonl):
  {"id": "q1", "question": "...", "gold_note_ids": ["abc123"], "type": "factual"}

Run:
  python scripts/run_eval.py
  python scripts/run_eval.py --questions data/eval/questions.jsonl
  python scripts/run_eval.py --no-llm          # skip LLM synthesis (just retrieval metrics)
  python scripts/run_eval.py --save report.json
  python scripts/run_eval.py --strategy semantic  # test one strategy only
"""

import sys
import json
import argparse
import logging
import math
import os
from pathlib import Path
from datetime import datetime

ROOT = Path(__file__).parent.parent.resolve()
sys.path.insert(0, str(ROOT))

from brain.memory.store import Store
from brain.memory.embeddings import EmbeddingProvider
from brain.memory.graph import GraphBuilder
from brain.query.planner import QueryPlanner

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger("eval")

DEFAULT_QUESTIONS = ROOT / "data" / "eval" / "questions.jsonl"
STRATEGIES        = ["semantic", "graph_traversal", "temporal"]


# ─── Config loader ────────────────────────────────────────────────────────────

def _load_cfg() -> dict:
    """
    Load config from configs/llm_profiles.yaml if present,
    otherwise fall back to environment variables.
    Matches the pattern used by main.py.
    """
    cfg = {
        "llm_backend":           os.environ.get("LLM_BACKEND", "claude"),
        "anthropic_api_key":     os.environ.get("ANTHROPIC_API_KEY", ""),
        "embedding_backend":     os.environ.get("EMBEDDING_BACKEND", "local"),
        "local_embedding_model": "all-MiniLM-L6-v2",
        "ollama_base_url":       os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434"),
        "ollama_model":          os.environ.get("OLLAMA_MODEL", "mistral"),
        "claude_model":          os.environ.get("CLAUDE_MODEL", "claude-haiku-4-5-20251001"),
    }

    yaml_path = ROOT / "configs" / "llm_profiles.yaml"
    if not yaml_path.exists():
        yaml_path = ROOT / "llm_profiles.yaml"

    if yaml_path.exists():
        try:
            import yaml
            with open(yaml_path) as f:
                profiles_cfg = yaml.safe_load(f)

            # Pull default daily/embed profiles
            defaults = profiles_cfg.get("defaults", {})
            daily_name = defaults.get("daily", "")
            embed_name = defaults.get("embed", "")

            profiles = {p["name"]: p for p in profiles_cfg.get("profiles", [])}

            daily = profiles.get(daily_name, {})
            if daily.get("provider") == "claude":
                cfg["llm_backend"]      = "claude"
                cfg["claude_model"]     = daily.get("model", cfg["claude_model"])
                cfg["anthropic_api_key"] = daily.get("api_key", cfg["anthropic_api_key"])
            elif daily.get("provider") == "ollama":
                cfg["llm_backend"]  = "ollama"
                cfg["ollama_model"] = daily.get("model", cfg["ollama_model"])

            embed = profiles.get(embed_name, {})
            if embed.get("provider") == "ollama":
                cfg["embedding_backend"]      = "ollama"
                cfg["ollama_embedding_model"] = embed.get("model", "nomic-embed-text")

        except Exception as e:
            log.warning(f"Could not parse llm_profiles.yaml: {e}. Using env vars.")

    return cfg


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Digital Brain benchmark evaluation")
    parser.add_argument("--questions", default=str(DEFAULT_QUESTIONS))
    parser.add_argument("--no-llm",   action="store_true",
                        help="Skip LLM synthesis calls (retrieval metrics only)")
    parser.add_argument("--save",     metavar="FILE",
                        help="Save full JSON report to FILE")
    parser.add_argument("--strategy", choices=STRATEGIES,
                        help="Evaluate a single strategy only")
    parser.add_argument("--top-k",   type=int, default=10,
                        help="Number of notes to retrieve per question (default 10)")
    args = parser.parse_args()

    # ── Load questions ─────────────────────────────────────────────────────
    q_path = Path(args.questions)
    if not q_path.exists():
        log.warning(f"No questions file at {q_path}. Creating sample file...")
        _create_sample_questions(q_path)
        log.info(f"Edit {q_path}, fill in gold_note_ids, then re-run.")
        sys.exit(0)

    questions = [json.loads(l) for l in q_path.read_text().splitlines() if l.strip()]
    log.info(f"Loaded {len(questions)} questions from {q_path}")

    # ── Setup ─────────────────────────────────────────────────────────────
    cfg      = _load_cfg()
    store    = Store(ROOT / "data" / "brain.db")
    embedder = EmbeddingProvider.from_config(cfg)
    planner  = QueryPlanner(store, embedder, cfg)

    strategies = [args.strategy] if args.strategy else STRATEGIES

    # ── Evaluate ──────────────────────────────────────────────────────────
    results = []
    for q in questions:
        log.info(f"[{q.get('type','?')}] {q['question'][:60]}")
        row = {
            "id":        q["id"],
            "question":  q["question"],
            "type":      q.get("type", "unknown"),
            "gold_ids":  q.get("gold_note_ids", []),
        }

        for strategy in strategies:
            # Map strategy name to QueryPlanner mode
            mode = _strategy_to_mode(strategy)
            try:
                result     = planner.query(q["question"], mode=mode, top_k=args.top_k)
                sources    = result.get("sources", [])
                retrieved_ids = [s["id"] for s in sources]

                row[strategy] = {
                    "retrieved_ids":    retrieved_ids,
                    "retrieved_titles": [s["title"] for s in sources[:5]],
                    "confidence":       result.get("confidence", 0.0),
                    "hit_rate":         _hit_rate(retrieved_ids, row["gold_ids"]),
                    "mrr":              _mrr(retrieved_ids, row["gold_ids"]),
                    "ndcg":             _ndcg(retrieved_ids, row["gold_ids"]),
                }

                # LLM synthesis scoring (citation overlap)
                if not args.no_llm and row["gold_ids"]:
                    answer = result.get("answer", "")
                    gold_notes = [store.get_note(nid) for nid in row["gold_ids"] if store.get_note(nid)]
                    row[strategy]["citation_overlap"] = _citation_overlap(answer, gold_notes)
                else:
                    row[strategy]["citation_overlap"] = None

            except Exception as e:
                log.error(f"Strategy {strategy} failed on q={q['id']}: {e}")
                row[strategy] = {"error": str(e), "hit_rate": 0.0, "mrr": 0.0,
                                 "ndcg": 0.0, "citation_overlap": None}

        results.append(row)

    # ── Print report ───────────────────────────────────────────────────────
    _print_report(results, strategies)

    if args.save:
        report = {
            "generated_at": datetime.now().isoformat(),
            "n_questions":  len(questions),
            "strategies":   strategies,
            "top_k":        args.top_k,
            "results":      results,
            "summary":      _summarise(results, strategies),
        }
        Path(args.save).write_text(json.dumps(report, indent=2))
        log.info(f"Full report saved to {args.save}")


# ─── Strategy mapping ─────────────────────────────────────────────────────────

def _strategy_to_mode(strategy: str) -> str:
    """Map eval strategy name to QueryPlanner mode string."""
    return {
        "semantic":         "semantic",
        "graph_traversal":  "graph",
        "temporal":         "temporal",
    }.get(strategy, "semantic")


# ─── Metrics ──────────────────────────────────────────────────────────────────

def _hit_rate(retrieved: list, gold: list) -> float:
    if not gold:
        return 0.0
    return float(any(g in retrieved for g in gold))


def _mrr(retrieved: list, gold: list) -> float:
    if not gold:
        return 0.0
    for i, nid in enumerate(retrieved):
        if nid in gold:
            return 1.0 / (i + 1)
    return 0.0


def _ndcg(retrieved: list, gold: list, k: int = 10) -> float:
    """Normalised Discounted Cumulative Gain at k."""
    if not gold:
        return 0.0
    gold_set = set(gold)

    def dcg(ids):
        score = 0.0
        for i, nid in enumerate(ids[:k]):
            rel = 1.0 if nid in gold_set else 0.0
            score += rel / math.log2(i + 2)
        return score

    actual  = dcg(retrieved)
    ideal   = dcg(list(gold_set)[:k])
    return actual / ideal if ideal > 0 else 0.0


def _citation_overlap(answer: str, gold_notes: list) -> float:
    """
    Fraction of gold notes whose title appears in the LLM's answer text.
    Rough proxy for whether the LLM cited the right sources.
    """
    if not gold_notes or not answer:
        return 0.0
    hits = sum(1 for n in gold_notes if n.title.lower() in answer.lower())
    return round(hits / len(gold_notes), 3)


# ─── Reporting ────────────────────────────────────────────────────────────────

def _summarise(results: list, strategies: list) -> dict:
    summary = {}
    n = max(len(results), 1)
    for s in strategies:
        rows = [r[s] for r in results if s in r and "error" not in r[s]]
        if not rows:
            continue
        summary[s] = {
            "hit_rate": round(sum(r["hit_rate"] for r in rows) / n, 3),
            "mrr":      round(sum(r["mrr"]      for r in rows) / n, 3),
            "ndcg":     round(sum(r["ndcg"]      for r in rows) / n, 3),
        }
        cit = [r["citation_overlap"] for r in rows if r.get("citation_overlap") is not None]
        if cit:
            summary[s]["citation_overlap"] = round(sum(cit) / len(cit), 3)
    return summary


def _print_report(results: list, strategies: list):
    n = len(results)
    summary = _summarise(results, strategies)

    print(f"\n{'='*62}")
    print(f"  EVAL RESULTS  —  {n} questions  —  {datetime.now():%Y-%m-%d %H:%M}")
    print(f"{'='*62}")
    print(f"  {'Strategy':<22}  {'Hit@10':>7}  {'MRR':>7}  {'NDCG@10':>8}  {'Cite%':>6}")
    print(f"  {'─'*22}  {'─'*7}  {'─'*7}  {'─'*8}  {'─'*6}")
    for s in strategies:
        sm  = summary.get(s, {})
        cit = f"{sm.get('citation_overlap', 0):.3f}" if "citation_overlap" in sm else "  n/a"
        print(f"  {s:<22}  {sm.get('hit_rate',0):>7.3f}  {sm.get('mrr',0):>7.3f}"
              f"  {sm.get('ndcg',0):>8.3f}  {cit:>6}")

    print(f"\n  Per-question breakdown:")
    for r in results:
        gold_flag = "✓" if r["gold_ids"] else "○"
        print(f"  {gold_flag} [{r['type'][:8]:<8}] {r['question'][:52]}")
        for s in strategies:
            if s not in r:
                continue
            d  = r[s]
            if "error" in d:
                print(f"    {s:<22} ERROR: {d['error'][:40]}")
                continue
            top3 = ", ".join(d.get("retrieved_titles", [])[:3])
            print(f"    {s:<22} hit={d['hit_rate']:.0f} mrr={d['mrr']:.2f} | {top3}")
    print()


# ─── Sample questions ─────────────────────────────────────────────────────────

def _create_sample_questions(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    samples = [
        {"id": "q1", "question": "What are my core arguments about consciousness?",
         "gold_note_ids": [], "type": "factual"},
        {"id": "q2", "question": "How did my view on free will evolve over time?",
         "gold_note_ids": [], "type": "temporal"},
        {"id": "q3", "question": "What notes connect epistemology and language?",
         "gold_note_ids": [], "type": "graph"},
        {"id": "q4", "question": "Where do I hold contradictory positions?",
         "gold_note_ids": [], "type": "contradiction"},
        {"id": "q5", "question": "What have I written about the limits of formal logic?",
         "gold_note_ids": [], "type": "factual"},
        {"id": "q6", "question": "What thinkers have most influenced my thinking?",
         "gold_note_ids": [], "type": "factual"},
        {"id": "q7", "question": "What did I write about AI in 2023?",
         "gold_note_ids": [], "type": "temporal"},
    ]
    path.write_text("\n".join(json.dumps(q) for q in samples))
    log.info(f"Sample questions written to {path}")
    log.info("Fill in gold_note_ids with real note IDs from your DB for meaningful MRR/NDCG scores.")
    log.info("Run 'python main.py query \"any question\"' to see note IDs in source output.")


if __name__ == "__main__":
    main()
