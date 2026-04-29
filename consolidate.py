#!/usr/bin/env python3
"""
scripts/consolidate.py
-----------------------
Nightly consolidation job. Add to cron:

  # Run every night at 2am
  0 2 * * * /usr/bin/python3 /path/to/Digital-Brain-Project/scripts/consolidate.py

What it does:
  1. Deduplication    — flags near-identical notes (cosine > 0.95)
  2. Contradiction    — detects semantically similar but epistemically opposed pairs
  3. Centrality decay — penalises stale orphan nodes
  4. LLM relations    — extracts new entity/relation edges from top notes
  5. Cluster refresh  — re-runs Louvain community detection
  6. Gap report       — saves a JSON gap report to data/gap_report_<date>.json

Flags:
  --no-llm            Skip LLM relation extraction (free run)
  --gap-report        Also run the gap agent and save a report
  --quiet             Suppress info logs (errors only)
"""

import sys
import argparse
import logging
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).parent.parent.resolve()
sys.path.insert(0, str(ROOT))

from brain.memory.store import Store
from brain.memory.graph import GraphBuilder
from brain.memory.consolidation import ConsolidationAgent
from brain.llm.providers import LLMRegistry


def main():
    parser = argparse.ArgumentParser(description="Digital Brain nightly consolidation")
    parser.add_argument("--no-llm", action="store_true",
                        help="Skip LLM relation extraction")
    parser.add_argument("--gap-report", action="store_true",
                        help="Run gap agent and save JSON report")
    parser.add_argument("--quiet", action="store_true",
                        help="Only log warnings and errors")
    args = parser.parse_args()

    level = logging.WARNING if args.quiet else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s  %(levelname)-8s  %(message)s",
        datefmt="%H:%M:%S",
    )
    log = logging.getLogger("consolidate")

    db_path  = ROOT / "data" / "brain.db"
    cfg_path = ROOT / "configs" / "llm_profiles.yaml"

    store    = Store(db_path)
    registry = LLMRegistry(cfg_path)
    builder  = GraphBuilder(store)

    extractor = None
    if not args.no_llm:
        try:
            from brain.extract.relations import RelationExtractor
            extractor = RelationExtractor.from_registry(registry)
        except Exception as e:
            log.warning(f"Could not init relation extractor: {e}")

    agent = ConsolidationAgent(store, extractor, builder)
    summary = agent.run_nightly_job()

    for line in summary:
        log.info(f"  {line}")

    if args.gap_report:
        log.info("Running gap agent …")
        from brain.agents.gap_agent import GapAgent
        import json
        gap_agent = GapAgent(store, builder, llm_registry=registry)
        report = gap_agent.run(llm_enrich=not args.no_llm)
        report_path = ROOT / "data" / f"gap_report_{datetime.now().strftime('%Y%m%d')}.json"
        report_path.write_text(json.dumps(report.to_dict(), indent=2))
        log.info(f"Gap report saved → {report_path}")
        log.info(f"  Gaps found: {len(report.gaps)}  "
                 f"({len(report.high_priority())} high-priority)")

    log.info("Consolidation complete.")


if __name__ == "__main__":
    main()
