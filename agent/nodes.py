import logging
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import JsonOutputParser, StrOutputParser
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

from agent.state import AgentState
from config import (
    GPT_MODEL_ID,
    TEMPERATURE,
    MAX_TOKENS,
    INTENT_CONFIDENCE_THRESHOLD,
)
from retrieval.qa_retriever import QAInstructedRetriever
from retrieval.fiction_retriever import FictionInstructedRetriever
from sources.registry import get_sources_by_domain

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Shared LLM instance
# ---------------------------------------------------------------------------
# One LLM instance shared across all nodes.
# Nodes that need different params (temperature etc.) build their own chain.

_llm = ChatOpenAI(
    model=GPT_MODEL_ID,
    temperature=TEMPERATURE,
    max_tokens=MAX_TOKENS,  # type: ignore
)

# ---------------------------------------------------------------------------
# Shared collection manager — injected by app.py at startup
# ---------------------------------------------------------------------------
# Set to None here, assigned in app.py lifespan before first request.
# Retrieval nodes use this instead of instantiating their own per request.

_collection_manager = None


# ---------------------------------------------------------------------------
# Node 1: classify_intent
# ---------------------------------------------------------------------------


class IntentClassification(BaseModel):
    intent: str = Field(description="'qa' | 'fiction' | 'hybrid'")
    confidence: float = Field(
        description="0.0 to 1.0 — how confident the classification is"
    )
    metadata_hints: dict = Field(
        description=(
            "Pre-extracted signals from the query. "
            "For QA: {'version': '2024-Q1'}. "
            "For fiction: {'character': 'Jason', 'chapter': 3}. "
            "Empty dict if no hints found."
        )
    )
    reasoning: str = Field(
        description="One sentence explaining the classification decision"
    )


_intent_parser = JsonOutputParser(pydantic_object=IntentClassification)

_intent_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are an intent classifier for a dual-domain knowledge assistant.\n\n"
            "Domain definitions:\n"
            "- 'qa': questions about QA engineering policies, testing standards, "
            "  coverage requirements, bug severity, release criteria, test environments. "
            "  Keywords: policy, test, coverage, regression, severity, release, QA, quality.\n"
            "- 'fiction': questions about the novel Dark Matter by Blake Crouch — "
            "  plot, characters, themes, events, quotes. "
            "  Keywords: Jason, Daniela, the box, parallel universe, Dark Matter, Crouch.\n"
            "- 'hybrid': question genuinely spans both domains, or confidence is low.\n\n"
            "Important: both domains share vocabulary — 'variables', 'testing', 'state', "
            "'environment', 'failure', 'iteration'. Classify by INTENT not keywords alone.\n"
            "'What happens when a system fails' in a QA context → qa. "
            "In a narrative context → fiction. When genuinely ambiguous → hybrid.\n\n"
            "Also extract any metadata hints present in the query:\n"
            "- Version/date references for QA (e.g. 'Q1 2024', 'latest', 'current')\n"
            "- Character names, chapter references for fiction\n\n"
            "{format_instructions}",
        ),
        ("human", "{user_query}"),
    ]
).partial(format_instructions=_intent_parser.get_format_instructions())

_intent_chain = _intent_prompt | _llm | _intent_parser


def classify_intent(state: AgentState) -> dict:
    """
    Node 1 — classifies the user query into qa / fiction / hybrid.
    Also extracts metadata hints to seed the instructed retrievers.
    Confidence below threshold overrides intent → hybrid.
    """
    logger.info("Node: classify_intent | query=%s", state["user_query"])

    result = _intent_chain.invoke({"user_query": state["user_query"]})

    intent = result["intent"]
    confidence = result["confidence"]

    # confidence gate — uncertain classifications go to hybrid
    if confidence < INTENT_CONFIDENCE_THRESHOLD and intent != "hybrid":
        logger.info(
            "Confidence %.2f below threshold %.2f — overriding '%s' → 'hybrid'",
            confidence,
            INTENT_CONFIDENCE_THRESHOLD,
            intent,
        )
        intent = "hybrid"

    logger.info(
        "Intent: %s | confidence: %.2f | hints: %s | reasoning: %s",
        intent,
        confidence,
        result["metadata_hints"],
        result["reasoning"],
    )

    return {
        "intent": intent,
        "intent_confidence": confidence,
        "metadata_hints": result["metadata_hints"],
        "domain_used": intent,
    }


# ---------------------------------------------------------------------------
# Node 2a: retrieve_qa
# ---------------------------------------------------------------------------


def retrieve_qa(state: AgentState) -> dict:
    """
    Node 2a — runs QA instructed retrieval across all registered QA sources.
    Uses the shared collection manager injected at startup.
    """
    logger.info("Node: retrieve_qa | query=%s", state["user_query"])

    qa_sources = get_sources_by_domain("qa")
    all_chunks = []

    for source in qa_sources:
        vector_store = _collection_manager.get_collection(source.name)  # type: ignore
        retriever = QAInstructedRetriever(
            llm=_llm,
            vector_store=vector_store,
            source=source,
        )
        chunks = retriever.retrieve(
            user_query=state["user_query"],
            metadata_hints=state.get("metadata_hints", {}),
        )
        all_chunks.extend(chunks)
        logger.info("QA retrieval | source=%s chunks=%d", source.name, len(chunks))

    return {"qa_chunks": all_chunks}


# ---------------------------------------------------------------------------
# Node 2b: retrieve_fiction
# ---------------------------------------------------------------------------


def retrieve_fiction(state: AgentState) -> dict:
    """
    Node 2b — runs fiction instructed retrieval across all registered fiction sources.
    Adding a new book requires only a sources.yaml entry.
    """
    logger.info("Node: retrieve_fiction | query=%s", state["user_query"])

    fiction_sources = get_sources_by_domain("fiction")
    all_chunks = []

    for source in fiction_sources:
        vector_store = _collection_manager.get_collection(source.name)  # type: ignore
        retriever = FictionInstructedRetriever(
            llm=_llm,
            vector_store=vector_store,
            source=source,
        )
        chunks = retriever.retrieve(
            user_query=state["user_query"],
            metadata_hints=state.get("metadata_hints", {}),
        )
        all_chunks.extend(chunks)
        logger.info("Fiction retrieval | source=%s chunks=%d", source.name, len(chunks))

    return {"fiction_chunks": all_chunks}


# ---------------------------------------------------------------------------
# Node 2c: retrieve_hybrid
# ---------------------------------------------------------------------------


def retrieve_hybrid(state: AgentState) -> dict:
    """
    Node 2c — runs both QA and fiction retrieval in sequence,
    then merges and re-ranks by distance score.
    Used when intent is hybrid or confidence is below threshold.
    """
    logger.info("Node: retrieve_hybrid | query=%s", state["user_query"])

    qa_result = retrieve_qa(state)
    fiction_result = retrieve_fiction(state)

    qa_chunks = qa_result["qa_chunks"]
    fiction_chunks = fiction_result["fiction_chunks"]

    merged = sorted(
        qa_chunks + fiction_chunks,
        key=lambda x: x["distance"],
    )

    logger.info(
        "Hybrid merge | qa=%d fiction=%d merged=%d",
        len(qa_chunks),
        len(fiction_chunks),
        len(merged),
    )

    return {
        "qa_chunks": qa_chunks,
        "fiction_chunks": fiction_chunks,
        "merged_chunks": merged,
    }


# ---------------------------------------------------------------------------
# Node 3: evaluate_retrieval
# ---------------------------------------------------------------------------


def evaluate_retrieval(state: AgentState) -> dict:
    """
    Node 3 — computes quality signals on retrieved chunks.
    Pure computation — no LLM call.
    Scores feed into the evaluation layer and are logged for metrics.
    """
    logger.info("Node: evaluate_retrieval")

    def _score(chunks: list[dict]) -> dict:
        if not chunks:
            return {
                "chunk_count": 0,
                "avg_distance": None,
                "min_distance": None,
                "has_results": False,
            }
        distances = [c["distance"] for c in chunks]
        return {
            "chunk_count": len(chunks),
            "avg_distance": round(sum(distances) / len(distances), 4),
            "min_distance": round(min(distances), 4),
            "has_results": True,
        }

    scores = {
        "qa": _score(state.get("qa_chunks", [])),
        "fiction": _score(state.get("fiction_chunks", [])),
        "intent": state["intent"],
        "intent_confidence": state["intent_confidence"],
    }

    logger.info("Retrieval scores: %s", scores)
    return {"retrieval_scores": scores}


# ---------------------------------------------------------------------------
# Node 4: generate_response
# ---------------------------------------------------------------------------


def _build_context(state: AgentState) -> str:
    """
    Formats retrieved chunks into a context string for the generation prompt.
    Domain label is included so the LLM knows which source each chunk came from.
    """
    intent = state["intent"]

    if intent == "qa":
        chunks = state.get("qa_chunks", [])
    elif intent == "fiction":
        chunks = state.get("fiction_chunks", [])
    else:
        chunks = state.get("merged_chunks", [])

    if not chunks:
        return ""

    parts = []
    for chunk in chunks:
        domain = chunk.get("domain", "unknown")
        source = chunk.get("source_name", "unknown")
        parts.append(f"[{domain} | {source}]\n{chunk['content']}")

    return "\n\n---\n\n".join(parts)


def _build_system_prompt(state: AgentState) -> str:
    """
    Builds a domain-appropriate system prompt for response generation.
    QA responses: precise, cite policy sections.
    Fiction responses: contextual, narrative-aware.
    Hybrid responses: clearly separated domain perspectives.
    """
    intent = state["intent"]
    has_qa = bool(state.get("qa_chunks"))
    has_fiction = bool(state.get("fiction_chunks"))

    base = (
        "You are a knowledgeable assistant with access to two knowledge bases: "
        "QA engineering policies and the novel Dark Matter by Blake Crouch.\n\n"
        "Answer using ONLY the provided context. "
        "If the answer is not in the context, say you don't know.\n\n"
    )

    if intent == "qa":
        return base + (
            "This is a QA policy question. "
            "Be precise and definitive. "
            "Reference specific policy statements where possible. "
            "Use formal, technical language appropriate for engineering standards."
        )
    elif intent == "fiction":
        return base + (
            "This is a question about Dark Matter by Blake Crouch. "
            "Be contextual and narrative-aware. "
            "Preserve the tone of the novel in your response. "
            "Reference character names and events specifically."
        )
    else:
        parts = [
            "This question spans multiple knowledge domains. "
            "You MUST answer using the retrieved context below — do not say you don't know "
            "if relevant context exists. "
            "Clearly separate your answer into domain perspectives."
        ]
        if has_qa:
            parts.append("Address the QA policy perspective clearly.")
        if has_fiction:
            parts.append("Address the Dark Matter narrative perspective clearly.")
        parts.append(
            "Separate domain perspectives with clear labels if both are relevant."
        )
        return base + " ".join(parts)


_generation_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "{system_prompt}\n\nContext:\n{context}"),
        MessagesPlaceholder(variable_name="history"),
        ("human", "{user_query}"),
    ]
)

_generation_chain = _generation_prompt | _llm | StrOutputParser()


def generate_response(state: AgentState) -> dict:
    """
    Node 4 — generates the raw response from retrieved context.
    System prompt adapts to domain. History provides conversation context.
    """
    logger.info("Node: generate_response | intent=%s", state["intent"])

    context = _build_context(state)
    system_prompt = _build_system_prompt(state)

    history_messages = []
    for turn in state.get("history", []):
        history_messages.append(HumanMessage(content=turn["user"]))
        history_messages.append(AIMessage(content=turn["assistant"]))

    if not context:
        logger.warning(
            "No context available for generation | intent=%s",
            state["intent"],
        )
        raw_response = (
            "I don't have enough information in my knowledge base "
            "to answer that question."
        )
    else:
        raw_response = _generation_chain.invoke(
            {
                "system_prompt": system_prompt,
                "context": context,
                "history": history_messages,
                "user_query": state["user_query"],
            }
        )

    logger.info("Raw response generated (%d chars)", len(raw_response))
    return {"raw_response": raw_response}


# ---------------------------------------------------------------------------
# Node 5: cite_sources
# ---------------------------------------------------------------------------


def cite_sources(state: AgentState) -> dict:
    """
    Node 5 — appends source attribution to the raw response.
    Extracts unique sources from retrieved chunks and formats them
    as a clean citation block at the end of the response.
    Pure computation — no LLM call needed.
    """
    logger.info("Node: cite_sources")

    intent = state["intent"]
    if intent == "qa":
        chunks = state.get("qa_chunks", [])
    elif intent == "fiction":
        chunks = state.get("fiction_chunks", [])
    else:
        chunks = state.get("merged_chunks", [])

    seen = set()
    sources = []
    for chunk in chunks:
        source_name = chunk.get("source_name", "unknown")
        domain = chunk.get("domain", "unknown")
        meta = chunk.get("metadata", {})

        if domain == "qa":
            label = f"{source_name} (version: {meta.get('version', 'unknown')})"
        elif domain == "fiction":
            label = (
                f"{meta.get('book', source_name)} "
                f"by {meta.get('author', 'unknown')} "
                f"(chunk {meta.get('chunk_index', '?')})"
            )
        else:
            label = source_name

        if label not in seen:
            seen.add(label)
            sources.append(label)

    raw = state.get("raw_response", "")
    if sources:
        citation_block = "\n\n---\n**Sources:**\n" + "\n".join(
            f"- {s}" for s in sources
        )
        cited_response = raw + citation_block
    else:
        cited_response = raw

    logger.info("Sources cited: %s", sources)
    return {
        "cited_response": cited_response,
        "sources": sources,
        "messages": [AIMessage(content=cited_response)],
    }
