"""
Gap Agent — Thin Orchestrator
------------------------------
Delegates all detection to GapFinder and all recommendations to Recommender.
This file wires those two together and produces a formatted daily briefing.

Usage (programmatic):
    agent = GapAgent(store, embedder, cfg)
    briefing = agent.daily_briefing()
    results  = agent.run(gap_types=["void", "depth"])

Usage (CLI via main.py):
    python main.py gap
    python main.py recommend --mode anonymous --briefing
"""

import logging
from datetime import datetime, timezone
from typing import Optional

from brain.analysis.gap_finder import GapFinder
from brain.analysis.recommender import Recommender

log = logging.getLogger(__name__)


class GapAgent:
    """
    Orchestrates the full gap → recommend → report pipeline.

    Parameters
    ----------
    store    : Store
    embedder : EmbeddingProvider
    cfg      : dict  (llm_backend, api_key, etc.)
    """

    def __init__(self, store, embedder, cfg: dict):
        self.store       = store
        self.finder      = GapFinder(store, embedder, cfg)
        self.recommender = Recommender(cfg)
        self.cfg         = cfg

    # ── Public API ────────────────────────────────────────────────────────────

    def run(
        self,
        gap_types: Optional[list] = None,
        mode: str = "anonymous",
        top_k: int = 5,
    ) -> list:
        """
        Run the full pipeline.
        Returns list of {"gap": Gap, "recommendations": [Recommendation]}
        """
        log.info("[gap_agent] Detecting gaps...")
        gaps = self.finder.find_all(types=gap_types)

        if not gaps:
            log.info("[gap_agent] No gaps found.")
            return []

        log.info(f"[gap_agent] {len(gaps)} gaps. Generating recommendations...")
        results = []
        for gap in gaps:
            try:
                # Recommender.recommend() takes a list of Gap objects
                recs = self.recommender.recommend([gap], top_k=top_k)
            except Exception as e:
                log.warning(f"[gap_agent] Recommender failed for '{gap.title}': {e}")
                recs = []
            results.append({"gap": gap, "recommendations": recs})

        return results

    def daily_briefing(
        self,
        gap_types: Optional[list] = None,
        mode: str = "anonymous",
        top_k: int = 3,
    ) -> str:
        """Generate a formatted daily briefing string for the terminal."""
        results = self.run(gap_types=gap_types, mode=mode, top_k=top_k)
        now = datetime.now(timezone.utc).strftime("%Y-%m-%d")

        lines = [
            f"╔══════════════════════════════════════════════════╗",
            f"║        DIGITAL BRAIN — DAILY BRIEFING            ║",
            f"║        {now:<41}║",
            f"╚══════════════════════════════════════════════════╝",
            "",
        ]

        if not results:
            lines.append("No significant gaps detected today.")
            return "\n".join(lines)

        by_type: dict = {}
        for r in results:
            gtype = r["gap"].gap_type
            by_type.setdefault(gtype, []).append(r)

        type_labels = {
            "void":          "🕳  VOID — Adjacent territory you haven't entered",
            "depth":         "📐  DEPTH — Referenced but underdeveloped",
            "width":         "🌐  WIDTH — Canonical siblings you've ignored",
            "temporal":      "⏱  STALE — High-centrality ideas never revisited",
            "contradiction": "⚡  CONTRADICTIONS — Conflicting claims for review",
            "orthogonal":    "🔄  ORTHOGONAL — Steelman challenges to your positions",
        }

        for gtype, items in by_type.items():
            lines.append(type_labels.get(gtype, f"── {gtype.upper()} ──"))
            lines.append("")
            for item in items:
                gap  = item["gap"]
                recs = item["recommendations"]

                lines.append(f"  • {gap.title}")
                lines.append(f"    {gap.description}")
                if gap.suggested_actions:
                    lines.append(f"    → {gap.suggested_actions[0]}")

                if recs:
                    lines.append("    Recommended:")
                    for rec in recs[:top_k]:
                        by = f" — {rec.author}" if rec.author else ""
                        lines.append(f"      [{rec.source_type}] {rec.title}{by}")
                lines.append("")
            lines.append("")

        lines.append("─" * 52)
        lines.append(f"Gaps: {len(results)} | Mode: {mode} | {now}")
        return "\n".join(lines)
