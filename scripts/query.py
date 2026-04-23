import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv

from pipeline.query_engine import QueryEngine

load_dotenv()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--question", default="How do I refresh a JWT token?")
    parser.add_argument(
        "--content-type",
        help="Scope retrieval to a single content_type (e.g. technical, hr_docs, support).",
    )
    args = parser.parse_args()

    engine = QueryEngine()

    print(f"question: {args.question}")
    if args.content_type:
        print(f"scope:    content_type={args.content_type}")
    print()

    where = {"content_type": args.content_type} if args.content_type else None
    result = engine.query(query=args.question, where=where)

    print("=== answer ===")
    print(result["answer"])
    print()

    print(f"=== sources ({len(result['sources'])}) ===")
    for s in result["sources"]:
        print(f"  [{s['score']}] {s['chunk_id']}")
        print(f"  {s['snippet'][:120]}...")
        print()


if __name__ == "__main__":
    main()
