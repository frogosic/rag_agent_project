from dataclasses import dataclass

from pipeline.retrieval.hybrid import RetrievalResult


@dataclass
class Target:
    """Named region of the corpus, identified by content predicates."""

    name: str
    doc: str
    heading_keywords: list[str]
    must_contain: list[str]


@dataclass
class SignalScore:
    """Result of signal-recall evaluation."""

    union_recall: float  # fraction of signals present anywhere in top-k
    best_chunk_recall: float  # max signal recall achieved by any single chunk
    best_chunk_rank: (
        int  # 1-indexed rank of the best-scoring chunk; 0 if no signals matched
    )
    matched_signals: list[str]  # signals that matched somewhere
    missed_signals: list[str]  # signals not found in any chunk


@dataclass
class AntiSignalHit:
    """Record of an anti-signal appearing in retrieval results."""

    signal: str
    chunk_id: str
    rank: int


def matches_target(chunk: RetrievalResult, target: Target) -> bool:
    """A chunk matches a target if doc + heading + must_contain all hold.

    Heading match: any of target.heading_keywords appears in chunk's heading metadata.
    If target has no heading_keywords, this clause is satisfied vacuously.
    Heading match is case-insensitive.

    Must-contain: every substring in target.must_contain appears in chunk text.
    Match is case-insensitive — target predicates identify corpus regions, not
    exact content. Use expected_signals (case-sensitive) for exact content checks.
    """
    meta = chunk.metadata or {}

    if meta.get("source") != target.doc:
        return False

    if target.heading_keywords:
        chunk_heading = (meta.get("heading") or "").lower()
        if not any(kw.lower() in chunk_heading for kw in target.heading_keywords):
            return False

    text_lower = (chunk.text or "").lower()
    for substr in target.must_contain:
        if substr.lower() not in text_lower:
            return False

    return True


def semantic_hit(results: list[RetrievalResult], target: Target) -> tuple[bool, int]:
    """Did any chunk in top-k match the target?

    Returns (hit, rank). Rank is 1-indexed; 0 if no match.
    """
    for rank, chunk in enumerate(results, start=1):
        if matches_target(chunk, target):
            return True, rank
    return False, 0


def signal_recall(results: list[RetrievalResult], signals: list[str]) -> SignalScore:
    """Compute union recall across top-k and best single-chunk recall.

    Union recall: fraction of signals appearing anywhere in retrieved chunks.
    Best-chunk recall: maximum fraction achieved by any single chunk.
    """
    if not signals:
        return SignalScore(
            union_recall=1.0,
            best_chunk_recall=1.0,
            best_chunk_rank=0,
            matched_signals=[],
            missed_signals=[],
        )

    per_chunk: list[set[str]] = []
    for chunk in results:
        text = chunk.text or ""
        hit = {s for s in signals if s in text}
        per_chunk.append(hit)

    union: set[str] = set().union(*per_chunk) if per_chunk else set()
    union_recall = len(union) / len(signals)

    best_count = 0
    best_rank = 0
    for rank, hits in enumerate(per_chunk, start=1):
        if len(hits) > best_count:
            best_count = len(hits)
            best_rank = rank

    best_chunk_recall = best_count / len(signals) if signals else 0.0

    matched = sorted(union)
    missed = sorted(set(signals) - union)

    return SignalScore(
        union_recall=union_recall,
        best_chunk_recall=best_chunk_recall,
        best_chunk_rank=best_rank,
        matched_signals=matched,
        missed_signals=missed,
    )


def anti_signal_hits(
    results: list[RetrievalResult], anti_signals: list[str]
) -> list[AntiSignalHit]:
    """Find every (anti-signal, chunk, rank) where an anti-signal appears."""
    hits: list[AntiSignalHit] = []
    for rank, chunk in enumerate(results, start=1):
        text = chunk.text or ""
        for signal in anti_signals:
            if signal in text:
                hits.append(
                    AntiSignalHit(signal=signal, chunk_id=chunk.chunk_id, rank=rank)
                )
    return hits
