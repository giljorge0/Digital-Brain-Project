"""
Query Agent Orchestrator (LangGraph)
------------------------------------
Wraps the QueryPlanner in a stateful graph.
Provides a human-in-the-loop fallback: if the retrieval confidence 
is too low, the agent pauses execution and asks the user for clarification
before synthesizing a final answer.
"""

import logging
from typing import Annotated, TypedDict, Dict, Any, List
from langgraph.graph import StateGraph, START, END

from ..query.planner import QueryPlanner

log = logging.getLogger(__name__)

# ─── 1. Define the Agent State ─────────────────────────────────────────────

class AgentState(TypedDict):
    question: str
    chat_history: List[str]          # To keep track of multi-turn clarifications
    planner_result: Dict[str, Any]   # Stores mode, sources, confidence, and initial answer
    needs_clarification: bool        # Flag to trigger human-in-the-loop
    clarification_prompt: str        # The question to ask the user
    final_answer: str                # The ultimate output to display

# ─── 2. Define the Graph Nodes ─────────────────────────────────────────────

class QueryOrchestrator:
    def __init__(self, planner: QueryPlanner, confidence_threshold: float = 0.4):
        self.planner = planner
        self.threshold = confidence_threshold
        
        # Build the graph
        workflow = StateGraph(AgentState)
        
        # Add nodes
        workflow.add_node("plan_and_retrieve", self._node_plan_and_retrieve)
        workflow.add_node("evaluate_confidence", self._node_evaluate_confidence)
        workflow.add_node("ask_human", self._node_ask_human)
        workflow.add_node("format_output", self._node_format_output)
        
        # Define edges (The flow of the agent)
        workflow.add_edge(START, "plan_and_retrieve")
        workflow.add_edge("plan_and_retrieve", "evaluate_confidence")
        
        # Conditional routing based on confidence
        workflow.add_conditional_edges(
            "evaluate_confidence",
            self._route_evaluation,
            {
                "clarify": "ask_human",
                "respond": "format_output"
            }
        )
        
        # If we ask the human, the loop goes back to plan_and_retrieve with the new context
        workflow.add_edge("ask_human", "plan_and_retrieve")
        workflow.add_edge("format_output", END)
        
        # Compile the graph (We use a simple memory checkpointer in production for interrupts)
        self.app = workflow.compile()

    # ─── Node Implementations ──────────────────────────────────────────────

    def _node_plan_and_retrieve(self, state: AgentState) -> AgentState:
        """Runs the query planner to fetch sources and draft an answer."""
        log.info("[agent] Running QueryPlanner...")
        
        # If there's chat history, append it to the question for context
        full_context = "\n".join(state.get("chat_history", []) + [state["question"]])
        
        result = self.planner.query(full_context)
        return {"planner_result": result}

    def _node_evaluate_confidence(self, state: AgentState) -> AgentState:
        """Checks if the planner is confident enough to answer."""
        confidence = state["planner_result"].get("confidence", 0.0)
        sources = state["planner_result"].get("sources", [])
        
        log.info(f"[agent] Evaluating confidence: {confidence}")
        
        if confidence < self.threshold or not sources:
            return {
                "needs_clarification": True,
                "clarification_prompt": "I couldn't find strong evidence in your notes for this. Can you provide more context or a different keyword?"
            }
        
        return {"needs_clarification": False}

    def _node_ask_human(self, state: AgentState) -> AgentState:
        """
        In a real UI, this node would interrupt execution and wait for user input.
        For now, we simulate asking the user and appending it to chat history.
        """
        prompt = state["clarification_prompt"]
        print(f"\n[AGENT PAUSE]: {prompt}")
        
        # Get user input from the console (or API in the future)
        user_input = input("Your clarification: ")
        
        new_history = state.get("chat_history", [])
        new_history.append(f"System: {prompt}")
        new_history.append(f"User Clarification: {user_input}")
        
        return {
            "chat_history": new_history,
            "needs_clarification": False # Reset flag
        }

    def _node_format_output(self, state: AgentState) -> AgentState:
        """Formats the final answer with citations."""
        res = state["planner_result"]
        answer = res["answer"]
        sources = res.get("sources", [])
        
        citations = "\n".join([f"- {s['title']} (Score: {s['score']})" for s in sources])
        final_out = f"{answer}\n\nSources:\n{citations}"
        
        return {"final_answer": final_out}

    # ─── Routing Logic ─────────────────────────────────────────────────────

    def _route_evaluation(self, state: AgentState) -> str:
        """Decides which node to go to next based on the needs_clarification flag."""
        if state.get("needs_clarification"):
            return "clarify"
        return "respond"

    # ─── Public API ────────────────────────────────────────────────────────

    def ask(self, question: str) -> str:
        """Entry point to kick off the graph execution."""
        initial_state = {
            "question": question,
            "chat_history": [],
            "planner_result": {},
            "needs_clarification": False,
            "clarification_prompt": "",
            "final_answer": ""
        }
        
        # Run the graph
        final_state = self.app.invoke(initial_state)
        return final_state["final_answer"]

if __name__ == "__main__":
    # Simple test execution
    from ..store import Store
    from ..embeddings import LocalEmbeddingProvider
    
    # Mock setup
    db = Store("data/brain.db")
    embedder = LocalEmbeddingProvider()
    planner = QueryPlanner(db, embedder)
    
    agent = QueryOrchestrator(planner)
    
    # Example usage
    print(agent.ask("What are my core arguments regarding neuro-symbolic systems?"))
