# RAG Agent Project

A RAG system with hybrid retrieval, metadata-aware filtering, and a measurable evaluation framework.

Built incrementally to understand each layer of a retrieval pipeline: extraction, chunking, embedding, hybrid retrieval, evaluation. Architectural decisions favor explicit reasoning over framework defaults.

## Architecture

### Single vector DB with metadata filtering

One ChromaDB collection holds all chunks across content types (technical, HR, support). Scoping is enforced at query time via metadata filters (`content_type`, `source`, etc.).

### Hybrid retrieval

Dense (sentence-transformers via Chroma) + sparse (BM25 via `bm25s`) results, fused via Reciprocal Rank Fusion.

BM25 indexes are partitioned by `content_type` and stored on disk as `data/bm25/{db_name}__{content_type}/`. At query time, partition selection happens before retrieval: a `where={"content_type": "technical"}` filter only queries the technical partition. This avoids the over-fetch-and-filter pattern that scales poorly on skewed corpora and gives sparse and dense paths consistent pre-filter semantics.

Inter-partition merging uses rank-based RRF rather than raw BM25 scores, since BM25 IDF statistics aren't comparable across independently-built indexes.

### Metadata schema

Every chunk carries:

| Field | Required | Source |
|---|---|---|
| `content_type` | yes | Top-level folder under `data/raw/` |
| `source` | yes | Origin filename |
| `source_path` | yes | Path from repo root |
| `chunk_index` | yes | Position in source document |
| `heading` | optional | H2 heading (technical chunks only) |
| `doc_format` | optional | File extension (`md`, `txt`) |

## Project structure

```text
rag_agent_project/
├── config/
│   ├── content_types.yaml        # per-folder ingestion rules
│   └── vector_databases.yaml     # vector DB + BM25 paths, weights
├── data/
│   ├── raw/                      # source documents organized by content type
│   ├── chroma/                   # Chroma collection (gitignored)
│   └── bm25/                     # partitioned BM25 indexes (gitignored)
├── pipeline/
│   ├── config_loader.py          # YAML config + dataclasses
│   ├── llm.py                    # LLM client wrapper (multi-provider)
│   ├── query_engine.py           # retrieve + generate orchestration
│   ├── extraction/
│   │   ├── extractors.py         # markdown + plaintext extractors
│   │   └── chunker.py            # header-based, paragraph, single strategies
│   └── retrieval/
│       └── hybrid.py             # hybrid retriever with RRF fusion
├── evaluation/
│   ├── targets.yaml              # named corpus regions for eval probes
│   ├── queries.yaml              # eval query suite
│   ├── metrics.py                # semantic hit, signal recall, anti-signals
│   └── runner.py                 # eval orchestration + aggregation
└── scripts/
    ├── ingest.py                 # extract → chunk → upsert + build BM25
    ├── query.py                  # interactive query CLI
    ├── list_chunks.py            # inspect indexed chunks
    └── evaluate.py               # run eval suite
```

## Setup

```bash
# Install dependencies
pip install -r requirements.txt

# Set environment variables
echo "ANTHROPIC_API_KEY=your_key_here" > .env
```

Optional: `LLM_PROVIDER` (anthropic | openai | ollama), `LLM_MODEL` to override the default model.

## Usage

### Ingest documents

```bash
# All content types
python scripts/ingest.py

# Single content type
python scripts/ingest.py --type technical
```

Re-running is idempotent (Chroma upserts by ID).

### Query the system

```bash
# Cross-domain query
python scripts/query.py --question "How do I refresh a JWT token?"

# Scoped to one content type
python scripts/query.py --question "parental leave eligibility" --content-type hr_docs
```

### Inspect indexed chunks

```bash
# All chunks
python scripts/list_chunks.py

# Filtered to one content type
python scripts/list_chunks.py --content-type technical
```

### Run the evaluation suite

```bash
python scripts/evaluate.py
```

Per-query verdicts plus aggregate summary. Use `--json` for machine-readable output. Exit code is non-zero if any query fails.

## Evaluation framework

The eval measures three orthogonal dimensions for each query:

1. **Semantic hit** — did the retrieved set include a chunk matching the named target (the right region of the corpus)?
2. **Signal recall** — did retrieved chunks contain the specific substrings needed to answer the question?
3. **Anti-signals (advisory)** — did retrieval drift into adjacent-but-wrong content?

### Targets vs signals

Targets (`evaluation/targets.yaml`) name regions of the corpus by content predicate (doc + heading + must-contain). They identify *where* an answer should come from. Targets use case-insensitive matching since prose varies in capitalization.

Signals (per-query, in `queries.yaml`) name specific artifacts that must surface in the answer. They identify *what content* is required. Signals use case-sensitive matching since technical artifacts (endpoints, env vars, error codes) are case-significant.

This split decouples eval from chunking strategy; if the chunker resplits content, queries don't break as long as the answer-bearing artifacts still surface somewhere in top-k.

### Reported metrics

- **Semantic hit rate**: queries whose target was found in top-k.
- **Mean union recall**: average fraction of expected signals appearing across top-k chunks.
- **Mean best-chunk recall**: average fraction of expected signals concentrated in a single best-scoring chunk.
- **Concentration gap**: union recall minus best-chunk recall. High gap means evidence is scattered; low gap means a single chunk holds most of the answer. Diagnostic for chunking quality and a leading indicator for reranker impact.

## Design decisions

A few choices worth being explicit about, since they shape future work:

**RRF weights are un-tuned defaults (0.4 sparse / 0.6 dense).** Real values come from running eval across a grid; placeholders set the goalpost until measurable.

**No LLM-as-judge for generation quality.** Phase 1 eyeballs generation manually. Faithfulness eval is deferred until eval runs are stable enough to validate against a judge prompt.

**Eval-set maintenance scales by rebuilding, not accumulating.** When the corpus grows substantively, the right move is to rewrite queries to probe the new shape, not carry forward queries that no longer test anything interesting.

