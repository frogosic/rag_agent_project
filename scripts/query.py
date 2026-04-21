import argparse
import sys
from pathlib import Path
from pipeline.query_engine import QueryEngine

sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv

load_dotenv()


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

    print("=== routing ===")
    print(f"database:  {result['session']['database']}")
    print(f"tone:      {result['session']['tone']}")
    print(f"reasoning: {result['session']['reasoning']}")
    print()

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
