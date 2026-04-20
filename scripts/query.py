"""
Quick end-to-end test of the full pipeline.

Usage:
    python scripts/query.py
    python scripts/query.py --role engineer --question "How do I refresh a JWT token?"
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv

load_dotenv()

from pipeline.query_engine import QueryEngine


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--role", default="engineer")
    parser.add_argument("--question", default="How do I refresh a JWT token?")
    args = parser.parse_args()

    engine = QueryEngine()

    print(f"role:     {args.role}")
    print(f"question: {args.question}")
    print()

    result = engine.query(
        query=args.question,
        user_id="test_user",
        user_role=args.role,
    )

    print(f"=== routing ===")
    print(f"database:  {result['session']['database']}")
    print(f"tone:      {result['session']['tone']}")
    print(f"reasoning: {result['session']['reasoning']}")
    print()

    print(f"=== answer ===")
    print(result["answer"])
    print()

    print(f"=== sources ({len(result['sources'])}) ===")
    for s in result["sources"]:
        print(f"  [{s['score']}] {s['chunk_id']}")
        print(f"  {s['snippet'][:120]}...")
        print()


if __name__ == "__main__":
    main()
