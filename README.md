# RAG Agent Project

A RAG system with hybrid retrieval, metadata-aware filtering, and a measurable evaluation framework.

Built incrementally to understand each layer of a retrieval pipeline: extraction, chunking, embedding, hybrid retrieval, reranking, evaluation. Architectural decisions favor explicit reasoning over framework defaults.

## Architecture

### Single vector DB with metadata filtering

One ChromaDB collection holds all chunks across content types (technical, HR, support). Scoping is enforced at query time via metadata filters (`content_type`, `source`, etc.).

### Hybrid retrieval

Dense (sentence-transformers via Chroma) + sparse (BM25 via `bm25s`) results, fused via Reciprocal Rank Fusion.

BM25 indexes are partitioned by `content_type` and stored on disk as `data/bm25/{db_name}__{content_type}/`. At query time, partition selection happens before retrieval: a `where={"content_type": "technical"}` filter only queries the technical partition. This avoids the over-fetch-and-filter pattern that scales poorly on skewed corpora and gives sparse and dense paths consistent pre-filter semantics.

Inter-partition merging uses rank-based RRF rather than raw BM25 scores, since BM25 IDF statistics aren't comparable across independently-built indexes.

#### BM25 heading boost

BM25 scores chunks by token frequency without distinguishing headings from body text. This causes a defining chunk to lose to a sibling chunk that merely cross-references it: if the chunk for `## POST /auth/login` is mostly JSON, while a sibling section mentions `/auth/login` in passing prose, BM25 ranks the sibling higher because it has more total token matches.

The fix is heading repetition in the BM25 corpus only. At ingest time, chunks with a heading get the heading prepended N times to the text passed to BM25. Chroma still sees the original chunk text, so dense retrieval and LLM context are unchanged. The boost lives in `scripts/ingest.py` rather than in chunking, since it's a retrieval concern, not a chunking concern.

### Cross-encoder reranker

After hybrid retrieval, an optional cross-encoder reranker rescores the candidate pool. Where bi-encoder dense retrieval embeds query and chunk separately and compares vectors, the cross-encoder reads (query, chunk) jointly — letting it bridge cases where lexical match is absent and bi-encoder semantic similarity is weak. This is the failure mode that surfaces on natural-language queries against JSON-heavy or jargon-heavy chunks: dense retrieval misses because the chunk doesn't read like prose, BM25 misses because the query lacks distinctive tokens.

Reranking is composed *over* hybrid retrieval, not inside it: hybrid pulls a wider candidate pool (`top_k * 2`), the reranker scores all candidates, and the top `top_k` reranked results are returned. The reranker can only reorder candidates that hybrid surfaced — it cannot promote chunks that hybrid missed entirely. If a chunk isn't in the candidate pool, no reranker can save it; the fix is upstream.

The reranker is toggled via `QueryEngine(rerank=True/False)` and `python scripts/evaluate.py --no-rerank`. Run both to A/B reranker contribution against the hybrid-only baseline.

Default model: `cross-encoder/ms-marco-MiniLM-L-6-v2`. Override via `Reranker(model_name=...)`.

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
│   ├── query_engine.py           # retrieve → rerank → generate orchestration
│   ├── extraction/
│   │   ├── extractors.py         # markdown + plaintext extractors
│   │   └── chunker.py            # header-based, paragraph, single strategies
│   └── retrieval/
│       ├── hybrid.py             # hybrid retriever with RRF fusion
│       └── reranker.py           # cross-encoder reranker (optional)
├── evaluation/
│   ├── targets.yaml              # named corpus regions for eval probes
│   ├── queries.yaml              # eval query suite
│   ├── metrics.py                # semantic hit, signal recall, anti-signals
│   └── runner.py                 # eval orchestration + aggregation
└── scripts/
    ├── ingest.py                 # extract → chunk → upsert + build BM25 (with heading boost)
    ├── query.py                  # interactive query CLI
    ├── list_chunks.py            # inspect indexed chunks
    └── evaluate.py               # run eval suite (--no-rerank for baseline)
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

Re-running is idempotent (Chroma upserts by ID). Heading boost is applied automatically when the BM25 corpus is built.

### Query the system

```bash
# Cross-domain query (reranking on by default)
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
# With reranker (default)
python scripts/evaluate.py

# Hybrid-only baseline (for A/B comparison)
python scripts/evaluate.py --no-rerank
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

### Pass/fail logic

A query passes when both conditions hold:

```text
passed = (semantic_hit if require_semantic_hit else True)
         AND (union_recall >= min_signal_recall)
```

Both checks use different predicates and can disagree. When they do, it points at where the problem lives:

| `semantic_hit` | `signal_recall` | Likely cause |
|---|---|---|
| miss | miss | Real retrieval failure — wrong chunks in top-k |
| miss | pass | Eval-spec drift — target predicate is stale, but signals surface |
| pass | miss | Signals are over-specific or scattered across chunks |
| pass | pass | Working as intended |

### Reported metrics

- **Semantic hit rate**: queries whose target was found in top-k.
- **Mean union recall**: average fraction of expected signals appearing across top-k chunks.
- **Mean best-chunk recall**: average fraction of expected signals concentrated in a single best-scoring chunk.
- **Concentration gap**: union recall minus best-chunk recall. High gap means evidence is scattered; low gap means a single chunk holds most of the answer. Diagnostic for chunking quality and a leading indicator for reranker impact.

### Anti-signals

Anti-signals are advisory drift indicators, not failure conditions. For queries phrased ambiguously between two adjacent docs, anti-signals list distinctive vocabulary from the *wrong* doc. If those terms appear in retrieved top-k, retrieval drifted — even if the query technically passes. Anti-signal warnings rising over time are a leading indicator that the system is becoming brittle, even when pass rate stays steady.

## Design decisions

A few choices worth being explicit about, since they shape future work:

**RRF weights are un-tuned defaults (0.4 sparse / 0.6 dense).** Real values come from running eval across a grid; placeholders set the goalpost until measurable.

**Heading boost factor is empirically chosen (3×).** Enough to put defining chunks ahead of cross-referencing siblings without making heading-only matches dominate. Tunable via `BM25_HEADING_BOOST` in `scripts/ingest.py`.

**Reranker pool size is `top_k * 2`.** Wide enough to give the reranker room to promote chunks hybrid ranked low, narrow enough to keep cross-encoder calls bounded. Tunable via `RERANK_POOL_MULTIPLIER` in `pipeline/query_engine.py`.

**No LLM-as-judge for generation quality.** Phase 1 eyeballs generation manually. Faithfulness eval is deferred until eval runs are stable enough to validate against a judge prompt.

**Eval-set maintenance scales by rebuilding, not accumulating.** When the corpus grows substantively, the right move is to rewrite queries to probe the new shape, not carry forward queries that no longer test anything interesting.
