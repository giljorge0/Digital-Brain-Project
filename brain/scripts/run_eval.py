#!/usr/bin/env python3
"""
scripts/run_eval.py
--------------------
Benchmark evaluation: compares three retrieval strategies on a set of
questions drawn from your own corpus.

Baselines:
  A  flat_rag       — pure vector similarity (top-k notes)
  B  graph_rag      — graph neighbourhood + vector (combined)
  C  full_system    — temporal + graph + semantic + provenance

Evaluation metrics per question:
  - hit_rate           — did the top-10 results include the gold note(s)?
  - mrr                — mean reciprocal rank of first gold result
  - citation_overlap   — did the LLM answer cite the right notes?
  - human_score        — 0/1/2 rubric filled in by you

Question format (data/eval/questions.jsonl):
  {"id": "q1", "question": "...", "gold_note_ids": ["abc123"], "type": "factual"}

Run:
  python scripts/run_eval.py
  python scripts/run_eval.py --questions data/eval/questions.jsonl
  python scripts/run_eval.py --no-llm          # skip LLM calls (just retrieval metrics)
  python scripts/run_eval.py --save report.json
"""

import sys
import json
import argparse
import logging
import math
from pathlib import Path
from datetime import datetime

ROOT = Path(__file__).parent.parent.resolve()
sys.path.insert(0, str(ROOT))

from brain.memory.store import Store
from brain.memory.embeddings import EmbeddingProvider
from brain.memory.graph import GraphBuilder
from brain.query.planner import QueryPlanner
from brain.llm.providers import LLMRegistry

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger("eval")

DEFAULT_QUESTIONS_PATH = ROOT / "data" / "eval" / "questions.jsonl"


def main():
    parser = argparse.ArgumentParser(description="Digital Brain benchmark evaluation")
    parser.add_argument("--questions", default=str(DEFAULT_QUESTIONS_PATH))
    parser.add_argument("--no-llm", action="store_true")
    parser.add_argument("--save", metavar="FILE")
    args = parser.parse_args()

    q_path = Path(args.questions)
    if not q_path.exists():
        log.warning(f"No questions file found at {q_path}")
        log.info("Creating sample questions file…")
        _create_sample_questions(q_path)
        log.info(f"Edit {q_path} and re-run.")
        sys.exit(0)

    questions = [json.loads(l) for l in q_path.read_text().splitlines() if l.strip()]
    log.info(f"Loaded {len(questions)} questions from {q_path}")

    db_path  = ROOT / "data" / "brain.db"
    cfg_path = ROOT / "configs" / "llm_profiles.yaml"
    store    = Store(db_path)
    registry = LLMRegistry(cfg_path)
    embedder = EmbeddingProvider.from_registry(registry)
    builder  = GraphBuilder(store)
    planner  = QueryPlanner(store, embedder, registry)

    results = []
    for q in questions:
        log.info(f"Evaluating: {q['question'][:60]}")
        row = {"id": q["id"], "question": q["question"], "type": q.get("type", "?"),
               "gold_ids": q.get("gold_note_ids", [])}

        for strategy in ["semantic", "temporal", "graph_traversal"]:
            plan = {"strategy": strategy, "question": q["question"]}
            retrieved = planner.execute(plan)
            retrieved_ids = [n.id for n in retrieved]

            row[strategy] = {
                "retrieved_titles": [n.title for n in retrieved[:5]],
                "hit_rate": _hit_rate(retrieved_ids, row["gold_ids"]),
                "mrr":      _mrr(retrieved_ids, row["gold_ids"]),
            }

        results.append(row)

    # Summary
    print(f"\n{'='*60}")
    print(f"  EVAL RESULTS  —  {len(questions)} questions")
    print(f"{'='*60}")
    for strategy in ["semantic", "temporal", "graph_traversal"]:
        hr  = sum(r[strategy]["hit_rate"] for r in results) / len(results)
        mrr = sum(r[strategy]["mrr"]      for r in results) / len(results)
        print(f"  {strategy:<20}  hit_rate: {hr:.3f}   MRR: {mrr:.3f}")

    print()
    print("  Per question breakdown:")
    for r in results:
        print(f"  [{r['type'][:8]}] {r['question'][:50]}")
        for s in ["semantic", "temporal", "graph_traversal"]:
            hr = r[s]["hit_rate"]
            print(f"    {s:<20} hit={hr:.0f}  top5: {', '.join(r[s]['retrieved_titles'][:3])}")

    if args.save:
        report = {
            "generated_at": datetime.now().isoformat(),
            "n_questions": len(questions),
            "results": results,
        }
        Path(args.save).write_text(json.dumps(report, indent=2))
        log.info(f"Report saved to {args.save}")


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


# ─── Sample questions generator ───────────────────────────────────────────────

def _create_sample_questions(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    samples = [
        {
            "id": "q1",
            "question": "What are my core arguments about consciousness?",
            "gold_note_ids": [],
            "type": "factual"
        },
        {
            "id": "q2",
            "question": "How did my view on free will evolve over time?",
            "gold_note_ids": [],
            "type": "temporal"
        },
        {
            "id": "q3",
            "question": "What notes are related to epistemology and language?",
            "gold_note_ids": [],
            "type": "graph"
        },
        {
            "id": "q4",
            "question": "Where do I hold contradictory positions?",
            "gold_note_ids": [],
            "type": "contradiction"
        },
        {
            "id": "q5",
            "question": "What have I written about the limits of formal logic?",
            "gold_note_ids": [],
            "type": "factual"
        },
    ]
    path.write_text("\n".join(json.dumps(q) for q in samples))
    log.info(f"Sample questions written to {path}")
    log.info("Fill in gold_note_ids with actual note IDs from your DB,")
    log.info("then re-run for meaningful results.")


if __name__ == "__main__":
    main()
