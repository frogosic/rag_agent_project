# RAG Agent — Dual-Domain Instructed Retrieval System

A production-grade RAG agent built with LangChain, LangGraph, and LlamaIndex patterns over two opposing document domains — QA engineering policies and fiction novels. Demonstrates instructed retrieval, LLM-based intent routing, graph-based orchestration, and LLM-as-judge evaluation.

---

## What this project demonstrates

Most RAG systems embed a query, find similar chunks, and stuff them into a prompt. This project goes further:

- **Instructed retrieval** — the LLM generates a structured retrieval plan (semantic query + metadata filters + sub-queries) from the user's natural language input before touching the vector store. System specifications propagate through every stage of retrieval, not just the initial query.
- **Dual-domain routing** — two completely opposing document domains share vocabulary deliberately. "What happens when the environment fails?" is ambiguous between a QA policy question and a Dark Matter narrative question. The system classifies intent with a confidence score and routes accordingly, falling back to hybrid retrieval when confidence is below threshold.
- **LangGraph orchestration** — the agent pipeline is a compiled graph with explicit nodes, conditional edges, and typed state. Each stage (classify → retrieve → evaluate → generate → cite) is independently testable and observable.
- **LLM-as-judge evaluation** — retrieval relevance and answer faithfulness are scored by a separate judge LLM, producing before/after metrics that demonstrate the value of the instructed retrieval pattern over naive RAG.

---

## Architecture

```
FastAPI (/chat  /evaluate  /health)
        │
        ▼
LangGraph agent graph
  ├── classify_intent          LLM classifies qa / fiction / hybrid + confidence
  ├── retrieve_qa              QA instructed retriever
  ├── retrieve_fiction         Fiction instructed retriever  
  ├── retrieve_hybrid          Both retrievers, merged and re-ranked
  ├── evaluate_retrieval       Distance scores + chunk quality signals
  ├── generate_response        Domain-aware prompt, history-aware generation
  └── cite_sources             Source attribution appended to response
        │
        ▼
Instructed Retrievers (per domain)
  ├── Query decomposition      LLM generates structured StructuredQuery
  ├── Metadata filter          Natural language → ChromaDB where clause
  └── Sub-query expansion      Complex queries split into parallel retrievals
        │
        ▼
ChromaDB (persistent, cosine similarity)
  ├── qa_policies collection   512-token chunks, strict threshold (0.55)
  └── dark_matter collection   1024-token chunks, loose threshold (0.70)
        │
        ▼
Ingestion pipeline
  ├── loader_factory           Text, PDF, EPUB (ebooklib, chapter-level)
  ├── chunking_strategy        Domain-specific chunk size, overlap, separators
  └── collection_manager       Skip-if-exists, single shared instance
        │
        ▼
sources.yaml                   All document sources declared here
OpenAI embeddings              text-embedding-3-small (1536 dims)
OpenAI LLM                     gpt-4o-mini (intent, retrieval, generation, judge)
```

---

## Key design decisions

**Why two opposing domains?**
QA policies and sci-fi fiction share vocabulary — "environment", "variables", "state", "iteration", "failure". Naive RAG retrieves wrong-domain chunks on ambiguous queries. The instructed retriever pattern, combined with intent classification, handles this correctly. The overlap is deliberate and stress-tests the system.

**Why instructed retrieval over standard RAG?**
Standard RAG embeds the raw user query and finds similar chunks. Instructed retrieval lets the LLM read the query alongside the index schema and domain instructions, then generate a structured retrieval plan. This enables metadata filtering from natural language ("last quarter's policy" → `{"version": "2024-Q1"}`), query decomposition for multi-part questions, and domain-aware semantic query refinement.

**Why LangGraph over a linear LCEL chain?**
The pipeline has genuine branching (three retrieval paths) and convergence (all paths join at evaluation). LangGraph models this as an explicit graph with typed state — every field in `AgentState` is documented, every node is independently testable, and the execution path is observable in the response payload.

**Why YAML source registry?**
Adding a new document source (a second novel, a second policy document) requires one YAML entry. No Python changes. The `DocumentSource` dataclass acts as the schema — missing or misspelled fields raise at startup. Retrieval context (key entities, vocabulary notes, query hints) lives in the YAML alongside the source definition, keeping domain knowledge out of Python code.

---

## Project structure

```
rag_agent_project/
├── app.py                          FastAPI app, lifespan startup, session management
├── config.py                       All constants — model IDs, thresholds, chunk sizes
├── sources/
│   ├── sources.yaml                Document source registry — add sources here
│   └── registry.py                 Loads YAML → DocumentSource dataclasses
├── ingestion/
│   ├── loader_factory.py           Text / PDF / EPUB loaders (ebooklib for EPUB)
│   ├── chunking_strategy.py        Domain-specific splitters and separators
│   └── collection_manager.py       ChromaDB lifecycle, skip-if-exists, shared instance
├── retrieval/
│   ├── instructed_retriever.py     Abstract base — query gen, filter, dedup
│   ├── qa_retriever.py             QA domain config — strict threshold, policy hints
│   └── fiction_retriever.py        Fiction domain config — loose threshold, entity hints
├── agent/
│   ├── state.py                    AgentState TypedDict — full pipeline state
│   ├── nodes.py                    All node functions + shared LLM instance
│   ├── graph.py                    Graph definition, conditional edges, compile
│   └── router.py                   Intent routing helper
├── evaluation/
│   └── metrics.py                  LLM-as-judge, retrieval + faithfulness scores
├── policies/
│   └── qa_policy.txt               QA engineering policy document
├── books/
│   └── dark_matter.epub            Dark Matter by Blake Crouch
└── chroma_data/                    Persisted ChromaDB collections (git-ignored)
```

---

## Setup

**Requirements:** Python 3.12, pyenv recommended, pandoc installed at system level.

```bash
# install pandoc (required for EPUB processing)
brew install pandoc      # macOS
# apt install pandoc     # Ubuntu/Debian

# clone and set up environment
git clone https://github.com/franeqa/rag_agent_project
cd rag_agent_project
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

**Environment:**
```bash
cp .env.example .env
# add your OpenAI API key to .env
OPENAI_API_KEY=sk-...
```

**Add your documents:**
```bash
# place your QA policy in policies/
cp your_policy.txt policies/qa_policy.txt

# place your EPUB in books/
cp dark_matter.epub books/dark_matter.epub
```

**Run:**
```bash
uvicorn app:app --reload --port 8000
```

On first startup, both collections are indexed automatically. Subsequent startups skip indexing and load from disk. ChromaDB data is persisted in `chroma_data/` — add this to `.gitignore`.

---

## Adding a new document source

Edit `sources/sources.yaml` only — no Python changes needed:

```yaml
fiction:
  - name: recursion
    path: books/recursion.epub
    source_type: epub
    domain: fiction
    description: >
      Recursion by Blake Crouch — a sci-fi thriller about memory, timelines,
      and a neuroscientist whose research enables rewriting the past.
    metadata:
      domain: fiction
      author: Blake Crouch
      book: Recursion
      genre: sci-fi thriller
      type: novel
    chunk_size: 1024
    chunk_overlap: 150
    retrieval_context:
      key_entities:
        - "Barry Sutton: NYPD detective investigating False Memory Syndrome"
        - "Helena Smith: neuroscientist who built the memory chair"
      overlapping_vocabulary: >
        Uses memory, timeline, reset, restore, construct — narrative concepts here.
      query_hints: >
        Include temporal context for memory/timeline queries.
        Decompose paradox questions into cause + effect sub-queries.
```

Then wipe ChromaDB and restart to re-index:
```bash
rm -rf chroma_data/
uvicorn app:app --port 8000
```

---

## API

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/health` | Confirms app is running, lists loaded collections |
| `POST` | `/chat` | Main chat endpoint, returns structured response |
| `POST` | `/debug/chat` | Same as /chat but returns raw chunks for debugging |
| `GET` | `/sessions/{id}` | Returns conversation history for a session |
| `POST` | `/reset/{id}` | Clears session history |
| `GET` | `/sources` | Lists all registered document sources |
| `POST` | `/evaluate` | Runs LLM-as-judge evaluation suite |

**Chat request:**
```json
{
  "message": "What happens when the environment fails?",
  "session_id": "my-session"
}
```

**Chat response:**
```json
{
  "response": "**QA Policy Perspective:** ...\n\n**Dark Matter Perspective:** ...",
  "sources": ["qa_policies (version: 2024-Q1)", "Dark Matter by Blake Crouch (chunk 42)"],
  "intent": "hybrid",
  "intent_confidence": 0.5,
  "retrieval_scores": {
    "qa": {"chunk_count": 3, "avg_distance": 0.44, "has_results": true},
    "fiction": {"chunk_count": 4, "avg_distance": 0.49, "has_results": true}
  },
  "domain_used": "hybrid",
  "duration_ms": 7119.44
}
```

---

## Evaluation

The `/evaluate` endpoint runs a suite of test queries across both domains including deliberately ambiguous overlap vocabulary queries, then scores each result with a separate judge LLM:

```json
{
  "total_queries": 10,
  "retrieval": {
    "avg_relevance": 0.63,
    "qa_relevance": 0.70,
    "fiction_relevance": 0.60,
    "success_rate": 1.0
  },
  "answer": {
    "avg_faithfulness": 0.84,
    "qa_faithfulness": 1.0,
    "fiction_faithfulness": 0.75,
    "answer_rate": 0.90,
    "citation_rate": 1.0
  },
  "performance": {
    "avg_duration_ms": 5280.8
  }
}
```

To disable evaluation (it makes ~30 LLM calls), set `EVALUATION_ENABLED = False` in `config.py`.

---

## Stack

| Layer | Technology |
|-------|-----------|
| API | FastAPI + uvicorn |
| Agent orchestration | LangGraph |
| LLM + embeddings | OpenAI gpt-4o-mini + text-embedding-3-small |
| Vector store | ChromaDB (persistent, cosine similarity) |
| LangChain | langchain, langchain-openai, langchain-chroma |
| EPUB loading | ebooklib + BeautifulSoup4 |
| Config | python-dotenv, pyyaml |

---

## What's next

- LlamaIndex `SubQuestionQueryEngine` for automatic multi-part query decomposition
- Metadata-aware reranking per domain
- Additional fiction sources via `sources.yaml` (Recursion, Wool)
- Streaming responses via `/chat/stream`
- LangGraph persistence layer for cross-session memory

---

## Author

[Frane](https://github.com/franeqa) — QA Automation Engineer  
Built as part of IBM's Agentic AI Coursera certificate and personal RAG research.
