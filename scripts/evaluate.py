"""CLI for running the eval suite.

Usage:
  python scripts/evaluate.py
  python scripts/evaluate.py --queries evaluation/queries.yaml --targets evaluation/targets.yaml
  python scripts/evaluate.py --top-k 5
  python scripts/evaluate.py --json   # emit machine-readable output

Exits with status 1 if any query failed, 0 otherwise.
"""

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv

from evaluation.runner import EvalReport, QueryVerdict, run_eval

load_dotenv()


PASS = "PASS"
FAIL = "FAIL"


def render_verdict(v: QueryVerdict) -> str:
    """Render one query verdict as a multi-line block."""
    status = PASS if v.passed else FAIL
    lines = [f"{v.query_id}: {status}    {v.target_name}"]
    lines.append(f"  query: {v.query_text}")

    sem_str = f"hit at rank {v.semantic_rank}" if v.semantic_hit else "no hit in top-k"
    lines.append(f"  semantic_hit:        {sem_str}")

    score = v.signal_score
    matched = ", ".join(score.matched_signals) or "—"
    missed = ", ".join(score.missed_signals) or "—"
    lines.append(
        f"  signal_recall:       union={score.union_recall:.2f}  "
        f"best_chunk={score.best_chunk_recall:.2f} (rank {score.best_chunk_rank})"
    )
    lines.append(f"    matched: {matched}")
    if score.missed_signals:
        lines.append(f"    missed:  {missed}")

    if v.anti_warnings:
        lines.append(f"  anti_signals:        {len(v.anti_warnings)} warning(s)")
        for w in v.anti_warnings:
            lines.append(f"    '{w.signal}' in {w.chunk_id} at rank {w.rank}")
    else:
        lines.append("  anti_signals:        clean")

    if v.notes:
        lines.append(f"  notes: {v.notes}")

    return "\n".join(lines)


def render_aggregate(report: EvalReport) -> str:
    """Render the aggregate summary block."""
    a = report.aggregate
    if a is None or a.total_queries == 0:
        return "No queries evaluated."

    lines = [
        "=" * 60,
        "AGGREGATE",
        "=" * 60,
        f"  queries:                {a.total_queries}",
        f"  passed:                 {a.passed}",
        f"  failed:                 {a.failed}",
        f"  semantic_hit_rate:      {a.semantic_hit_rate:.2%}",
        f"  mean_union_recall:      {a.mean_union_recall:.2f}",
        f"  mean_best_chunk_recall: {a.mean_best_chunk_recall:.2f}",
        f"  concentration_gap:     {a.mean_concentration_gap:+.2f}",
        f"  queries with warnings:  {a.queries_with_anti_warnings}",
    ]
    return "\n".join(lines)


def render_text(report: EvalReport) -> str:
    """Full text report: per-query verdicts followed by aggregate."""
    blocks = [render_verdict(v) for v in report.verdicts]
    blocks.append(render_aggregate(report))
    return "\n\n".join(blocks)


def render_json(report: EvalReport) -> str:
    """Machine-readable JSON. Useful for CI or for diffing across runs."""
    return json.dumps(
        {
            "verdicts": [asdict(v) for v in report.verdicts],
            "aggregate": asdict(report.aggregate) if report.aggregate else None,
        },
        indent=2,
        default=str,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-rerank", action="store_true")
    parser.add_argument(
        "--queries",
        default="evaluation/queries.yaml",
        help="Path to queries YAML.",
    )
    parser.add_argument(
        "--targets",
        default="evaluation/targets.yaml",
        help="Path to targets YAML.",
    )
    parser.add_argument(
        "--config",
        default="config",
        help="Path to config directory.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=10,
        help="Retrieval depth for each query.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Emit machine-readable JSON instead of text.",
    )
    args = parser.parse_args()

    report = run_eval(
        queries_path=Path(args.queries),
        targets_path=Path(args.targets),
        config_path=args.config,
        top_k=args.top_k,
        rerank=not args.no_rerank,
    )

    if args.json:
        print(render_json(report))
    else:
        print(render_text(report))

    # Exit non-zero if anything failed — useful for CI later.
    failed = report.aggregate.failed if report.aggregate else 0
    sys.exit(1 if failed > 0 else 0)


if __name__ == "__main__":
    main()
