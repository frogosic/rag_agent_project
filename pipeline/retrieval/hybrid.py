import json
from dataclasses import dataclass
from pathlib import Path

import bm25s
import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction

from pipeline.config_loader import VectorDBConfig


@dataclass
class BM25Partition:
    retriever: bm25s.BM25
    id_map: list[str]


@dataclass
class RetrievalResult:
    chunk_id: str
    text: str
    metadata: dict
    score: float
    source: str  # "sparse", "dense", "hybrid"


class HybridRetriever:
    # RRF constant. Used for both inter-partition sparse fusion
    # and for the final sparse/dense fusion in _rrf().
    RRF_K = 60

    def __init__(self, db_config: VectorDBConfig):
        self.config = db_config
        self._chroma = self._load_chroma()
        # One BM25 index per content_type, loaded from disk at init.
        self._bm25_indexes: dict[str, BM25Partition] = self._load_bm25_indexes()

    def retrieve(
        self,
        query: str,
        top_k: int = 20,
        where: dict | None = None,
    ) -> list[RetrievalResult]:
        """
        Hybrid retrieval with optional metadata filter.

        `where` is a ChromaDB-style filter dict, e.g. {"content_type": "technical"}
        or {"content_type": {"$in": ["technical", "support"]}}.

        Dense retrieval applies the full Chroma `where` filter natively.
        Sparse retrieval pre-selects BM25 partitions based on supported
        `content_type` filter shapes, then secondarily enforces the full
        filter at hydration time via Chroma.
        """
        sparse = self._sparse_search(query, top_k=top_k, where=where)
        dense = self._dense_search(query, top_k=top_k, where=where)
        return self._rrf(sparse, dense, top_k=top_k)

    def _sparse_search(
        self, query: str, top_k: int, where: dict | None
    ) -> list[RetrievalResult]:
        """
        BM25 retrieval across selected content_type partitions.

        Inter-partition merging uses rank-based RRF rather than raw BM25
        scores, because BM25 scores are not directly comparable across
        independently-built indexes (different IDF statistics).
        """
        selected = self._select_bm25_indexes(where)
        if not selected:
            return []

        tokens = bm25s.tokenize(query, stopwords="en")

        # Query each selected partition; fuse results by rank into a single
        # sparse ranking.
        fused_sparse_scores: dict[str, float] = {}

        for partition in selected:
            k = min(top_k, len(partition.id_map))
            if k == 0:
                continue

            results, _scores = partition.retriever.retrieve(tokens, k=k)
            seen_in_partition: set[str] = set()

            for rank, idx in enumerate(results[0], start=1):
                chunk_id = partition.id_map[idx]

                # Defensive dedup in case ingestion/indexing invariants drift.
                if chunk_id in seen_in_partition:
                    continue
                seen_in_partition.add(chunk_id)

                fused_sparse_scores[chunk_id] = fused_sparse_scores.get(
                    chunk_id, 0.0
                ) + 1.0 / (self.RRF_K + rank)

        if not fused_sparse_scores:
            return []

        ranked_sparse = sorted(
            fused_sparse_scores.items(),
            key=lambda x: x[1],
            reverse=True,
        )[:top_k]

        top_ids = [cid for cid, _ in ranked_sparse]
        score_map = dict(ranked_sparse)

        # Hydrate text + metadata from Chroma in one batched call.
        # Passing `where` here enforces non-content_type filter components
        # (e.g. source, doc_format) that partition selection can't express.
        fetched = self._chroma.get(
            ids=top_ids,
            where=where,
            include=["documents", "metadatas"],
        )

        out = []
        for chunk_id, text, meta in zip(
            fetched["ids"],
            fetched["documents"] or [],
            fetched["metadatas"] or [],
        ):
            out.append(
                RetrievalResult(
                    chunk_id=chunk_id,
                    text=text,
                    metadata=dict(meta) if meta is not None else {},
                    score=score_map[chunk_id],
                    source="sparse",
                )
            )

        # Preserve sparse fused ranking after hydration.
        out.sort(key=lambda r: score_map[r.chunk_id], reverse=True)
        return out

    def _dense_search(
        self, query: str, top_k: int, where: dict | None
    ) -> list[RetrievalResult]:
        """Dense vector search. Chroma pre-filters natively via `where`."""
        results = self._chroma.query(
            query_texts=[query],
            n_results=min(top_k, self._chroma.count()),
            where=where,
            include=["documents", "metadatas", "distances"],
        )

        out = []
        ids = results["ids"][0] if results.get("ids") and results["ids"] else []
        docs = (
            results["documents"][0]
            if results.get("documents") and results["documents"]
            else []
        )
        metas = (
            results["metadatas"][0]
            if results.get("metadatas") and results["metadatas"]
            else []
        )
        distances = (
            results["distances"][0]
            if results.get("distances") and results["distances"]
            else []
        )

        for chunk_id, text, meta, dist in zip(ids, docs, metas, distances):
            # Cosine distance → similarity. Keeps `score` higher-is-better
            # consistently across sparse, dense, and hybrid results.
            score = 1 - dist
            out.append(
                RetrievalResult(
                    chunk_id=chunk_id,
                    text=text,
                    metadata=dict(meta) if meta is not None else {},
                    score=score,
                    source="dense",
                )
            )
        return out

    def _select_bm25_indexes(self, where: dict | None) -> list[BM25Partition]:
        """Pick which BM25 partitions to query based on the `where` filter.

        Supported filter shapes on `content_type`:
          - None                              → all partitions
          - {"content_type": "x"}             → one partition
          - {"content_type": {"$in": [...]}}  → subset

        Unsupported filter shapes (e.g. {"$ne": ...}, compound filters) fall
        back to querying all partitions. The hydration step's `where` then
        enforces the full filter via Chroma.
        """
        if where is None:
            return list(self._bm25_indexes.values())

        ct_filter = where.get("content_type")
        if ct_filter is None:
            return list(self._bm25_indexes.values())

        if isinstance(ct_filter, str):
            idx = self._bm25_indexes.get(ct_filter)
            return [idx] if idx else []

        if isinstance(ct_filter, dict) and "$in" in ct_filter:
            return [
                self._bm25_indexes[ct]
                for ct in ct_filter["$in"]
                if ct in self._bm25_indexes
            ]

        return list(self._bm25_indexes.values())

    def _rrf(
        self,
        sparse: list[RetrievalResult],
        dense: list[RetrievalResult],
        top_k: int,
    ) -> list[RetrievalResult]:
        """Combines sparse and dense results using Reciprocal Rank Fusion (RRF)."""
        scores: dict[str, float] = {}
        texts: dict[str, str] = {}
        metas: dict[str, dict] = {}

        for ranked_list, weight in [
            (sparse, self.config.sparse_weight),
            (dense, self.config.dense_weight),
        ]:
            for rank, result in enumerate(ranked_list, start=1):
                rrf_score = weight * (1.0 / (self.RRF_K + rank))
                scores[result.chunk_id] = scores.get(result.chunk_id, 0.0) + rrf_score
                texts[result.chunk_id] = result.text
                metas[result.chunk_id] = result.metadata

        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_k]

        return [
            RetrievalResult(
                chunk_id=cid,
                text=texts[cid],
                metadata=metas[cid],
                score=score,
                source="hybrid",
            )
            for cid, score in ranked
        ]

    def _load_chroma(self):
        """Loads the Chroma collection with the specified embedding function."""
        path = Path(self.config.chroma_path)
        client = chromadb.PersistentClient(path=str(path))
        ef = SentenceTransformerEmbeddingFunction(
            model_name=self.config.embedding_model
        )
        return client.get_collection(
            name=self.config.collection_name,
            embedding_function=ef,  # type: ignore[arg-type]
        )

    def _load_bm25_indexes(self) -> dict[str, BM25Partition]:
        """Load all BM25 partitions matching {bm25_path}/{db_name}__<content_type>/."""
        base = Path(self.config.bm25_path)
        prefix = f"{self.config.name}__"
        indexes: dict[str, BM25Partition] = {}

        if not base.exists():
            return indexes

        for d in sorted(base.iterdir()):
            if not d.is_dir() or not d.name.startswith(prefix):
                continue
            content_type = d.name[len(prefix) :]
            retriever = bm25s.BM25.load(str(d / "index"), load_corpus=False)
            id_map = json.loads((d / "id_map.json").read_text())
            indexes[content_type] = BM25Partition(
                retriever=retriever,
                id_map=id_map,
            )

        return indexes
