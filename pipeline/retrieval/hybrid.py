import json
from dataclasses import dataclass
from pathlib import Path

import bm25s
import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction

from pipeline.config_loader import VectorDBConfig


@dataclass
class RetrievalResult:
    chunk_id: str
    text: str
    metadata: dict
    score: float
    source: str  # "sparse", "dense", "hybrid"


class HybridRetriever:
    def __init__(self, db_config: VectorDBConfig):
        self.config = db_config
        self._chroma = self._load_chroma()
        self._bm25, self._id_map = self._load_bm25()

    def retrieve(self, query: str, top_k: int = 20) -> list[RetrievalResult]:
        sparse = self._sparse_search(query, top_k=top_k)
        dense = self._dense_search(query, top_k=top_k)
        return self._rrf(sparse, dense, top_k=top_k)

    def _sparse_search(self, query: str, top_k: int) -> list[RetrievalResult]:
        tokens = bm25s.tokenize(query, stopwords="en")
        results, scores = self._bm25.retrieve(tokens, k=min(top_k, len(self._id_map)))

        out = []
        for idx, score in zip(results[0], scores[0]):
            chunk_id = self._id_map[idx]
            text = self._fetch_text(chunk_id)
            if text is None:
                continue
            out.append(
                RetrievalResult(
                    chunk_id=chunk_id,
                    text=text,
                    metadata={},
                    score=float(score),
                    source="sparse",
                )
            )
        return out

    def _dense_search(self, query: str, top_k: int) -> list[RetrievalResult]:
        results = self._chroma.query(
            query_texts=[query],
            n_results=min(top_k, self._chroma.count()),
            include=["documents", "metadatas", "distances"],
        )

        out = []
        ids = results["ids"][0]
        docs = results["documents"][0]
        metas = results["metadatas"][0]
        distances = results["distances"][0]

        for chunk_id, text, meta, dist in zip(ids, docs, metas, distances):
            # chroma returns cosine distance (0=identical, 2=opposite)
            # convert to similarity score
            score = 1 - dist
            out.append(
                RetrievalResult(
                    chunk_id=chunk_id,
                    text=text,
                    metadata=meta,
                    score=score,
                    source="dense",
                )
            )
        return out

    def _rrf(
        self,
        sparse: list[RetrievalResult],
        dense: list[RetrievalResult],
        top_k: int,
        k: int = 60,
    ) -> list[RetrievalResult]:
        scores: dict[str, float] = {}
        texts: dict[str, str] = {}
        metas: dict[str, dict] = {}

        for ranked_list, weight in [
            (sparse, self.config.sparse_weight),
            (dense, self.config.dense_weight),
        ]:
            for rank, result in enumerate(ranked_list, start=1):
                rrf_score = weight * (1.0 / (k + rank))
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
        path = Path(self.config.chroma_path)
        client = chromadb.PersistentClient(path=str(path))
        ef = SentenceTransformerEmbeddingFunction(
            model_name=self.config.embedding_model
        )
        return client.get_collection(
            name=self.config.collection_name,
            embedding_function=ef,
        )

    def _load_bm25(self) -> tuple:
        bm25_dir = Path("data/bm25") / self.config.name
        retriever = bm25s.BM25.load(str(bm25_dir / "index"), load_corpus=False)
        id_map = json.loads((bm25_dir / "id_map.json").read_text())
        return retriever, id_map

    def _fetch_text(self, chunk_id: str) -> str | None:
        result = self._chroma.get(ids=[chunk_id], include=["documents"])
        if result["documents"]:
            return result["documents"][0]
        return None
