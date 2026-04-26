from __future__ import annotations

import logging
from functools import lru_cache

from sentence_transformers import CrossEncoder
from pipeline.retrieval.hybrid import RetrievalResult

logger = logging.getLogger(__name__)

DEFAULT_RERANKER_MODEL = "tomaarsen/ms-marco-ettin-150m-reranker"


@lru_cache(maxsize=2)
def _load_model(model_name: str) -> CrossEncoder:
    """Lazy-load and cache the cross-encoder model.

    Cached so repeated QueryEngine instantiations within one process don't
    reload the model. maxsize=2 allows comparing two models side-by-side
    without unbounded memory growth.
    """
    logger.info(f"loading cross-encoder: {model_name}")
    return CrossEncoder(model_name)


class Reranker:
    """Cross-encoder reranker over a list of RetrievalResults.

    Stateless beyond the loaded model. The model itself is process-cached
    via _load_model so multiple Reranker instances share weights.
    """

    def __init__(self, model_name: str = DEFAULT_RERANKER_MODEL):
        self.model_name = model_name
        self._model = _load_model(model_name)

    def rerank(
        self,
        query: str,
        results: list[RetrievalResult],
        top_k: int | None = None,
    ) -> list[RetrievalResult]:
        """Rescore results by cross-encoder relevance and return reordered list.

        The returned RetrievalResults have:
          - score replaced by the cross-encoder score (raw logit, higher = better)
          - source set to "rerank"
          - chunk_id, text, metadata preserved

        If results is empty, returns []. If top_k is None, returns all
        reranked results; otherwise truncates to top_k after sorting.
        """
        if not results:
            return []

        pairs = [(query, r.text) for r in results]
        scores = self._model.predict(pairs)

        scored: list[tuple[float, RetrievalResult]] = [
            (float(s), r) for s, r in zip(scores, results)
        ]
        scored.sort(key=lambda x: x[0], reverse=True)

        if top_k is not None:
            scored = scored[:top_k]

        return [
            RetrievalResult(
                chunk_id=r.chunk_id,
                text=r.text,
                metadata=r.metadata,
                score=score,
                source="rerank",
            )
            for score, r in scored
        ]
