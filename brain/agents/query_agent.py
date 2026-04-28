"""
Query Orchestrator
------------------
A stateful query agent that:
  1. Runs the QueryPlanner to retrieve + synthesize
  2. Checks confidence threshold
  3. If confidence too low → pauses and asks the user for clarification
     (human-in-the-loop, Karpathy "don't hallucinate" principle)
  4. Returns structured answer with provenance

Uses LangGraph if available; falls back to a pure-Python state machine.
"""

import logging
from typing import Optional

log = logging.getLogger(__name__)


# ─── Orchestrator ─────────────────────────────────────────────────────────────

class QueryOrchestrator:
    CONFIDENCE_THRESHOLD = 0.25  # below this → ask for clarification

    def __init__(self, planner, hitl: bool = True):
        """
        planner: QueryPlanner instance
        hitl: enable human-in-the-loop clarification
        """
        self.planner = planner
        self.hitl    = hitl
        self._try_langgraph()

    def _try_langgraph(self):
        """Try to wire up LangGraph; degrade gracefully if not installed."""
        try:
            from langgraph.graph import StateGraph, END
            self._has_langgraph = True
            log.info("[agent] LangGraph available")
        except ImportError:
            self._has_langgraph = False
            log.info("[agent] LangGraph not installed — using built-in state machine")

    # ── Main entry point ──────────────────────────────────────────────────────

    def ask(self, question: str, mode: str = "auto",
            clarification: str = None) -> str:
        """
        Ask a question and get an answer with provenance.
        If confidence is low and hitl=True, will ask for clarification interactively.
        """
        if self._has_langgraph:
            return self._langgraph_flow(question, mode, clarification)
        return self._simple_flow(question, mode, clarification)

    # ── Simple state machine (no LangGraph) ───────────────────────────────────

    def _simple_flow(self, question: str, mode: str,
                     clarification: Optional[str]) -> str:
        # If clarification was provided, append it to the question
        effective_q = question
        if clarification:
            effective_q = f"{question}\n\nAdditional context: {clarification}"

        result = self.planner.query(effective_q, mode=mode)
        confidence = result.get("confidence", 0)

        # Low confidence → ask for clarification if in HITL mode
        if confidence < self.CONFIDENCE_THRESHOLD and self.hitl:
            print(f"\n[brain] Low confidence ({confidence:.2f}) for: '{question}'")
            print("[brain] I found the following potentially relevant topics:")
            for s in result.get("sources", [])[:3]:
                print(f"  - {s['title']}")
            print("\n[brain] Can you clarify what you're looking for?")
            print("[brain] (Press Enter to accept the current answer, or type clarification)")
            try:
                user_input = input("> ").strip()
                if user_input:
                    return self._simple_flow(question, mode, user_input)
            except (EOFError, KeyboardInterrupt):
                pass

        return self._format_answer(result)

    # ── LangGraph flow ────────────────────────────────────────────────────────

    def _langgraph_flow(self, question: str, mode: str,
                        clarification: Optional[str]) -> str:
        try:
            from langgraph.graph import StateGraph, END
            from typing import TypedDict, Annotated

            class AgentState(TypedDict):
                question:      str
                mode:          str
                clarification: Optional[str]
                result:        Optional[dict]
                needs_hitl:    bool
                final_answer:  str

            def retrieve_node(state: AgentState) -> AgentState:
                q = state["question"]
                if state.get("clarification"):
                    q += "\n\nAdditional context: " + state["clarification"]
                result = self.planner.query(q, mode=state["mode"])
                needs_hitl = (
                    result.get("confidence", 0) < self.CONFIDENCE_THRESHOLD
                    and self.hitl
                    and not state.get("clarification")  # only ask once
                )
                return {**state, "result": result, "needs_hitl": needs_hitl}

            def hitl_node(state: AgentState) -> AgentState:
                result = state["result"]
                print(f"\n[brain] Low confidence ({result.get('confidence', 0):.2f})")
                print("[brain] Potentially relevant:")
                for s in result.get("sources", [])[:3]:
                    print(f"  - {s['title']}")
                print("[brain] Clarification? (Enter to accept)")
                try:
                    user_input = input("> ").strip()
                except (EOFError, KeyboardInterrupt):
                    user_input = ""
                return {**state, "clarification": user_input or None,
                        "needs_hitl": False}

            def format_node(state: AgentState) -> AgentState:
                return {**state,
                        "final_answer": self._format_answer(state["result"])}

            def should_hitl(state: AgentState) -> str:
                return "hitl" if state["needs_hitl"] else "format"

            graph = StateGraph(AgentState)
            graph.add_node("retrieve", retrieve_node)
            graph.add_node("hitl",    hitl_node)
            graph.add_node("format",  format_node)

            graph.set_entry_point("retrieve")
            graph.add_conditional_edges("retrieve", should_hitl,
                                        {"hitl": "hitl", "format": "format"})
            graph.add_edge("hitl", "retrieve")
            graph.add_edge("format", END)

            app = graph.compile()
            final_state = app.invoke({
                "question":      question,
                "mode":          mode,
                "clarification": clarification,
                "result":        None,
                "needs_hitl":    False,
                "final_answer":  "",
            })
            return final_state["final_answer"]

        except Exception as e:
            log.warning(f"[agent] LangGraph flow failed ({e}), using simple flow")
            return self._simple_flow(question, mode, clarification)

    # ── Formatting ────────────────────────────────────────────────────────────

    def _format_answer(self, result: dict) -> str:
        answer     = result.get("answer", "No answer generated.")
        sources    = result.get("sources", [])
        mode       = result.get("mode", "?")
        confidence = result.get("confidence", 0)

        lines = [answer, ""]

        if sources:
            lines.append(f"── Sources [{mode}, confidence={confidence:.2f}] ──")
            for s in sources[:5]:
                date = f" ({s['date'][:10]})" if s.get("date") else ""
                lines.append(f"  • {s['title']}{date}")

        return "\n".join(lines)