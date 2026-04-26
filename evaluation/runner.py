from dataclasses import dataclass, field
from pathlib import Path

import yaml

from evaluation.metrics import (
    AntiSignalHit,
    SignalScore,
    Target,
    anti_signal_hits,
    semantic_hit,
    signal_recall,
)
from pipeline.config_loader import ConfigLoader
from pipeline.retrieval.hybrid import HybridRetriever, RetrievalResult


@dataclass
class Query:
    """A single eval query loaded from queries.yaml."""

    id: str
    query: str
    expected_target: str
    expected_signals: list[str]
    anti_signals: list[str]
    min_signal_recall: float
    require_semantic_hit: bool
    notes: str


@dataclass
class QueryVerdict:
    """Full evaluation result for one query."""

    query_id: str
    query_text: str
    passed: bool
    semantic_hit: bool
    semantic_rank: int  # 1-indexed; 0 if no hit
    signal_score: SignalScore
    anti_warnings: list[AntiSignalHit]
    target_name: str
    notes: str


@dataclass
class AggregateMetrics:
    """Across-query summary."""

    total_queries: int
    passed: int
    failed: int
    semantic_hit_rate: float
    mean_union_recall: float
    mean_best_chunk_recall: float
    mean_concentration_gap: float
    queries_with_anti_warnings: int


@dataclass
class EvalReport:
    """Full eval output."""

    verdicts: list[QueryVerdict] = field(default_factory=list)
    aggregate: AggregateMetrics | None = None


def load_targets(path: Path) -> dict[str, Target]:
    """Load targets.yaml into a name → Target dict."""
    raw = yaml.safe_load(path.read_text())
    targets: dict[str, Target] = {}
    for name, data in raw["targets"].items():
        targets[name] = Target(
            name=name,
            doc=data["doc"],
            heading_keywords=data.get("heading_keywords", []) or [],
            must_contain=data.get("must_contain", []) or [],
        )
    return targets


def load_queries(path: Path) -> list[Query]:
    """Load queries.yaml into a list of Query objects."""
    raw = yaml.safe_load(path.read_text())
    queries: list[Query] = []
    for entry in raw["queries"]:
        queries.append(
            Query(
                id=entry["id"],
                query=entry["query"],
                expected_target=entry["expected_target"],
                expected_signals=entry.get("expected_signals", []) or [],
                anti_signals=entry.get("anti_signals", []) or [],
                min_signal_recall=float(entry.get("min_signal_recall", 0.5)),
                require_semantic_hit=bool(entry.get("require_semantic_hit", True)),
                notes=entry.get("notes", ""),
            )
        )
    return queries


def evaluate_query(
    query: Query,
    target: Target,
    retriever: HybridRetriever,
    top_k: int,
) -> QueryVerdict:
    """Run one query end-to-end: retrieve → measure → verdict."""
    results: list[RetrievalResult] = retriever.retrieve(query.query, top_k=top_k)

    sem_hit, sem_rank = semantic_hit(results, target)
    score = signal_recall(results, query.expected_signals)
    warnings = anti_signal_hits(results, query.anti_signals)

    passed = (
        sem_hit if query.require_semantic_hit else True
    ) and score.union_recall >= query.min_signal_recall

    return QueryVerdict(
        query_id=query.id,
        query_text=query.query,
        passed=passed,
        semantic_hit=sem_hit,
        semantic_rank=sem_rank,
        signal_score=score,
        anti_warnings=warnings,
        target_name=target.name,
        notes=query.notes,
    )


def aggregate(verdicts: list[QueryVerdict]) -> AggregateMetrics:
    """Compute summary metrics across all query verdicts."""
    n = len(verdicts)
    if n == 0:
        return AggregateMetrics(
            total_queries=0,
            passed=0,
            failed=0,
            semantic_hit_rate=0.0,
            mean_union_recall=0.0,
            mean_best_chunk_recall=0.0,
            mean_concentration_gap=0.0,
            queries_with_anti_warnings=0,
        )

    passed = sum(1 for v in verdicts if v.passed)
    sem_hits = sum(1 for v in verdicts if v.semantic_hit)
    union_recalls = [v.signal_score.union_recall for v in verdicts]
    best_recalls = [v.signal_score.best_chunk_recall for v in verdicts]
    gaps = [
        v.signal_score.union_recall - v.signal_score.best_chunk_recall for v in verdicts
    ]
    with_warnings = sum(1 for v in verdicts if v.anti_warnings)

    return AggregateMetrics(
        total_queries=n,
        passed=passed,
        failed=n - passed,
        semantic_hit_rate=sem_hits / n,
        mean_union_recall=sum(union_recalls) / n,
        mean_best_chunk_recall=sum(best_recalls) / n,
        mean_concentration_gap=sum(gaps) / n,
        queries_with_anti_warnings=with_warnings,
    )


def run_eval(
    queries_path: Path,
    targets_path: Path,
    config_path: str = "config",
    top_k: int = 10,
) -> EvalReport:
    """Run the full eval. Returns a structured report.

    Failures during query loading or target resolution raise; per-query
    retrieval/scoring errors do not — they produce a failed verdict so the
    rest of the eval still runs.
    """
    targets = load_targets(targets_path)
    queries = load_queries(queries_path)

    loader = ConfigLoader(config_path)
    retriever = HybridRetriever(loader.default_db())

    verdicts: list[QueryVerdict] = []
    for query in queries:
        target = targets.get(query.expected_target)
        if target is None:
            raise ValueError(
                f"Query '{query.id}' references unknown target "
                f"'{query.expected_target}'. Known targets: {sorted(targets.keys())}"
            )
        verdict = evaluate_query(query, target, retriever, top_k=top_k)
        verdicts.append(verdict)

    return EvalReport(verdicts=verdicts, aggregate=aggregate(verdicts))
