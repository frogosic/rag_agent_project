import logging

from langgraph.graph import StateGraph, END

from agent.state import AgentState
from agent.nodes import (
    classify_intent,
    retrieve_qa,
    retrieve_fiction,
    retrieve_hybrid,
    evaluate_retrieval,
    generate_response,
    cite_sources,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Routing function
# ---------------------------------------------------------------------------


def _route_by_intent(state: AgentState) -> str:
    """
    Reads intent from state after classify_intent runs.
    Returns the name of the next node to execute.
    This is the conditional edge function LangGraph calls
    to decide which branch to take.
    """
    intent = state["intent"]
    logger.info("Routing by intent: %s", intent)
    return {
        "qa": "retrieve_qa",
        "fiction": "retrieve_fiction",
        "hybrid": "retrieve_hybrid",
    }.get(intent, "retrieve_hybrid")  # default to hybrid if unknown


# ---------------------------------------------------------------------------
# Graph builder
# ---------------------------------------------------------------------------


def build_graph() -> StateGraph:
    """
    Constructs and compiles the LangGraph agent graph.

    Graph structure:
                    ┌─────────────────┐
                    │ classify_intent │
                    └────────┬────────┘
                             │ conditional edge
               ┌─────────────┼─────────────┐
               ▼             ▼             ▼
        retrieve_qa   retrieve_fiction  retrieve_hybrid
               │             │             │
               └─────────────┴─────────────┘
                             │ all paths converge
                    ┌────────▼────────┐
                    │evaluate_retrieval│
                    └────────┬────────┘
                    ┌────────▼────────┐
                    │generate_response│
                    └────────┬────────┘
                    ┌────────▼────────┐
                    │  cite_sources   │
                    └────────┬────────┘
                            END
    """
    graph = StateGraph(AgentState)

    # register nodes
    graph.add_node("classify_intent", classify_intent)
    graph.add_node("retrieve_qa", retrieve_qa)
    graph.add_node("retrieve_fiction", retrieve_fiction)
    graph.add_node("retrieve_hybrid", retrieve_hybrid)
    graph.add_node("evaluate_retrieval", evaluate_retrieval)
    graph.add_node("generate_response", generate_response)
    graph.add_node("cite_sources", cite_sources)

    # entry point
    graph.set_entry_point("classify_intent")

    # conditional routing after classification
    graph.add_conditional_edges(
        "classify_intent",
        _route_by_intent,
        {
            "retrieve_qa": "retrieve_qa",
            "retrieve_fiction": "retrieve_fiction",
            "retrieve_hybrid": "retrieve_hybrid",
        },
    )

    # all retrieval branches converge at evaluation
    graph.add_edge("retrieve_qa", "evaluate_retrieval")
    graph.add_edge("retrieve_fiction", "evaluate_retrieval")
    graph.add_edge("retrieve_hybrid", "evaluate_retrieval")

    # linear from evaluation onward
    graph.add_edge("evaluate_retrieval", "generate_response")
    graph.add_edge("generate_response", "cite_sources")
    graph.add_edge("cite_sources", END)

    compiled = graph.compile()
    logger.info("Agent graph compiled successfully.")
    return compiled  # type: ignore


# ---------------------------------------------------------------------------
# Module-level compiled graph
# ---------------------------------------------------------------------------
# Built once at import time, reused across all requests.
# CollectionManager must be initialised before this is imported.

agent_graph = build_graph()
