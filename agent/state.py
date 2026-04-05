from typing import TypedDict, Annotated
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages


class AgentState(TypedDict):
    """
    The single source of truth for everything that flows through the graph.

    LangGraph passes this state object between every node. Each node receives
    the current state, does its work, and returns a dict of only the fields
    it changed — LangGraph merges that back into the state automatically.

    The Annotated[list[BaseMessage], add_messages] pattern on `messages` is
    LangGraph-specific: instead of replacing the list on each update, it
    appends to it. Every other field is replaced on update.

    Fields are grouped by which layer of the pipeline owns them:
    conversation → routing → retrieval → evaluation → response
    """

    # ---------------------------------------------------------------------------
    # Conversation
    # ---------------------------------------------------------------------------
    messages: Annotated[list[BaseMessage], add_messages]
    # full LangGraph message history — appended to on every turn

    user_query: str
    # raw input from the user, unchanged throughout the graph

    history: list[dict]
    # conversation history in {"user": ..., "assistant": ...} format
    # passed in from the API layer, used for prompt context

    # ---------------------------------------------------------------------------
    # Routing
    # ---------------------------------------------------------------------------
    intent: str
    # "qa" | "fiction" | "hybrid"
    # set by classify_intent node, drives conditional edges

    intent_confidence: float
    # 0.0 → 1.0 — how confident the classifier is
    # below INTENT_CONFIDENCE_THRESHOLD → hybrid branch

    metadata_hints: dict
    # pre-extracted signals from the query before retrieval
    # e.g. {"version": "2024-Q1"} or {"chapter": 3, "character": "Jason"}
    # passed to instructed retrievers to seed metadata filter generation

    # ---------------------------------------------------------------------------
    # Retrieval
    # ---------------------------------------------------------------------------
    qa_chunks: list[dict]
    # retrieved QA chunks — each dict has:
    # {"content", "distance", "metadata", "query_used", "source_name", "domain"}

    fiction_chunks: list[dict]
    # retrieved fiction chunks — same structure as qa_chunks

    merged_chunks: list[dict]
    # used by hybrid branch — qa_chunks + fiction_chunks merged and re-ranked
    # empty for single-domain queries

    # ---------------------------------------------------------------------------
    # Evaluation
    # ---------------------------------------------------------------------------
    retrieval_scores: dict
    # per-domain quality signals computed by evaluate_retrieval node
    # e.g. {"qa": {"chunk_count": 3, "avg_distance": 0.31, "has_results": True},
    #        "fiction": {"chunk_count": 0, "avg_distance": None, "has_results": False}}

    # ---------------------------------------------------------------------------
    # Response
    # ---------------------------------------------------------------------------
    raw_response: str
    # unformatted LLM output from generate_response node

    cited_response: str
    # final response with source attribution appended by cite_sources node

    sources: list[str]
    # source references extracted from retrieved chunks
    # e.g. ["qa_policies (version: 2024-Q1)", "dark_matter (chunk 42)"]

    domain_used: str
    # which branch was actually taken — "qa" | "fiction" | "hybrid"
    # recorded for evaluation and logging
