from dotenv import load_dotenv

load_dotenv()


import logging
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel


from agent.graph import agent_graph
from agent.state import AgentState
from ingestion.collection_manager import CollectionManager
from sources.registry import ALL_SOURCES
from langchain_core.messages import HumanMessage
from evaluation.metrics import run_evaluation_suite, EvaluationReport


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Collection manager — single shared instance
# ---------------------------------------------------------------------------
# Instantiated once at startup, injected into nodes via module-level singleton.
# This fixes the per-request CollectionManager instantiation in nodes.py.

collection_manager = CollectionManager()


# ---------------------------------------------------------------------------
# Lifespan — startup and shutdown logic
# ---------------------------------------------------------------------------


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    FastAPI lifespan context — runs startup logic before the app
    accepts requests, and shutdown logic when it stops.

    This is the right place for expensive one-time operations:
    loading ChromaDB collections, warming up embeddings, etc.
    Replaces Flask's @app.before_first_request pattern.
    """
    logger.info("Starting up — loading all document sources...")
    try:
        collection_manager.load_all(ALL_SOURCES)
        logger.info(
            "All collections loaded: %s",
            collection_manager.collection_names(),
        )
    except Exception as e:
        logger.error("Failed to load collections at startup: %s", e)
        raise

    # inject the shared collection manager into nodes
    # so they don't instantiate their own per request
    import agent.nodes as nodes_module

    nodes_module._collection_manager = collection_manager

    yield

    # shutdown
    logger.info("Shutting down.")


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

app = FastAPI(
    title="RAG Agent",
    description="Dual-domain RAG agent over QA policies and fiction.",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Request / response models
# ---------------------------------------------------------------------------


class ChatRequest(BaseModel):
    message: str
    session_id: str = "default"


class ChatResponse(BaseModel):
    response: str
    sources: list[str]
    intent: str
    intent_confidence: float
    retrieval_scores: dict
    domain_used: str
    duration_ms: float


# ---------------------------------------------------------------------------
# In-memory session store
# ---------------------------------------------------------------------------
# Keyed by session_id. Each value is a list of {"user": ..., "assistant": ...}
# dicts. In production this would be Redis or a database.

_sessions: dict[str, list[dict]] = {}


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------


@app.get("/health")
async def health():
    """Health check — confirms app is running and collections are loaded."""
    return {
        "status": "ok",
        "collections": collection_manager.collection_names(),
    }


@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """
    Main chat endpoint.
    Runs the full LangGraph agent pipeline and returns a structured response.
    Session history is maintained per session_id.
    """
    if not request.message.strip():
        raise HTTPException(status_code=400, detail="Message cannot be empty.")

    session_id = request.session_id
    history = _sessions.get(session_id, [])

    logger.info(
        "Chat request | session=%s history_turns=%d query=%s",
        session_id,
        len(history),
        request.message,
    )

    start = time.time()

    try:
        # build initial state
        initial_state: AgentState = {
            "messages": [HumanMessage(content=request.message)],
            "user_query": request.message,
            "history": history,
            "intent": "",
            "intent_confidence": 0.0,
            "metadata_hints": {},
            "qa_chunks": [],
            "fiction_chunks": [],
            "merged_chunks": [],
            "retrieval_scores": {},
            "raw_response": "",
            "cited_response": "",
            "sources": [],
            "domain_used": "",
        }

        # run the graph
        final_state = await agent_graph.ainvoke(initial_state)  # type: ignore

        duration_ms = round((time.time() - start) * 1000, 2)

        # update session history
        cited = final_state.get("cited_response", "")
        _sessions[session_id] = history + [
            {"user": request.message, "assistant": cited}
        ]

        logger.info(
            "Chat complete | session=%s intent=%s duration=%.0fms",
            session_id,
            final_state.get("intent"),
            duration_ms,
        )

        return ChatResponse(
            response=cited,
            sources=final_state.get("sources", []),
            intent=final_state.get("intent", ""),
            intent_confidence=final_state.get("intent_confidence", 0.0),
            retrieval_scores=final_state.get("retrieval_scores", {}),
            domain_used=final_state.get("domain_used", ""),
            duration_ms=duration_ms,
        )

    except Exception as e:
        logger.error("Error processing chat request: %s", e, exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/reset/{session_id}")
async def reset_session(session_id: str):
    """Clears conversation history for a session."""
    _sessions.pop(session_id, None)
    logger.info("Session reset: %s", session_id)
    return {"message": f"Session '{session_id}' cleared."}


@app.get("/sessions/{session_id}")
async def get_session(session_id: str):
    """Returns the conversation history for a session. Useful for debugging."""
    history = _sessions.get(session_id, [])
    return {"session_id": session_id, "turns": len(history), "history": history}


@app.get("/sources")
async def list_sources():
    """Lists all registered document sources and their metadata."""
    return {
        "sources": [
            {
                "name": s.name,
                "domain": s.domain,
                "source_type": s.source_type,
                "description": s.description[:100] + "...",
            }
            for s in ALL_SOURCES
        ]
    }


@app.post("/evaluate")
async def evaluate():
    """
    Runs the full evaluation suite against the agent.
    Returns retrieval relevance, answer faithfulness, and performance metrics.
    This is the before/after story for the portfolio piece.
    """
    logger.info("Starting evaluation suite...")
    try:
        report = await run_evaluation_suite(agent_graph, collection_manager)
        return report.summary()
    except Exception as e:
        logger.error("Evaluation failed: %s", e, exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/debug/chat")
async def debug_chat(request: ChatRequest):
    """Returns full state including raw chunks — for debugging only."""
    from langchain_core.messages import HumanMessage

    history = _sessions.get(request.session_id, [])
    initial_state: AgentState = {
        "messages": [HumanMessage(content=request.message)],
        "user_query": request.message,
        "history": history,
        "intent": "",
        "intent_confidence": 0.0,
        "metadata_hints": {},
        "qa_chunks": [],
        "fiction_chunks": [],
        "merged_chunks": [],
        "retrieval_scores": {},
        "raw_response": "",
        "cited_response": "",
        "sources": [],
        "domain_used": "",
    }

    final_state = await agent_graph.ainvoke(initial_state)  # type: ignore

    return {
        "intent": final_state.get("intent"),
        "qa_chunks": [
            {"content": c["content"][:200], "distance": c["distance"]}
            for c in final_state.get("qa_chunks", [])
        ],
        "fiction_chunks": [
            {"content": c["content"][:200], "distance": c["distance"]}
            for c in final_state.get("fiction_chunks", [])
        ],
        "merged_chunks": [
            {"content": c["content"][:200], "distance": c["distance"]}
            for c in final_state.get("merged_chunks", [])
        ],
        "raw_response": final_state.get("raw_response"),
    }
