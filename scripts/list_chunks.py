import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction

from pipeline.config_loader import ConfigLoader, VectorDBConfig


logger = logging.getLogger(__name__)


def list_chunks_for_db(db_config: VectorDBConfig, snippet_len: int) -> None:
    """Lists chunks in the specified Chroma collection, showing ID, content type, source, and a text snippet."""
    path = Path(db_config.chroma_path)
    if not path.exists():
        logger.warning("%s: no data at %s (run ingest first)", db_config.name, path)
        return

    client = chromadb.PersistentClient(path=str(path))
    ef = SentenceTransformerEmbeddingFunction(model_name=db_config.embedding_model)
    collection = client.get_or_create_collection(
        name=db_config.collection_name,
        embedding_function=ef,  # type: ignore[arg-type]
    )

    result = collection.get(include=["documents", "metadatas"])
    ids = result.get("ids", []) or []
    docs = result.get("documents", []) or []
    metas = result.get("metadatas", []) or []

    if not ids:
        logger.warning("%s: collection exists but has no chunks", db_config.name)
        return

    logger.info("%s: %d chunk(s)", db_config.name, len(ids))

    print(f"\n=== {db_config.name} ({len(ids)} chunk(s)) ===\n")
    for chunk_id, doc, meta in zip(ids, docs, metas):
        content_type = (meta or {}).get("content_type", "?")
        source = (meta or {}).get("filename", (meta or {}).get("source_path", "?"))
        snippet = (doc or "").replace("\n", " ").strip()[:snippet_len]
        print(f"  id:      {chunk_id}")
        print(f"  type:    {content_type}")
        print(f"  source:  {source}")
        print(f"  text:    {snippet}...")
        print()


def main():
    """Main entry point for the list_chunks script, which lists chunks in Chroma collections based on the config."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--db", help="list only this DB (default: all)")
    parser.add_argument("--config", default="config")
    parser.add_argument("--snippet-len", type=int, default=120)
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=args.log_level,
        format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
        datefmt="%H:%M:%S",
    )

    loader = ConfigLoader(args.config)
    all_dbs = loader.all_databases()

    if args.db:
        if args.db not in all_dbs:
            logger.error("unknown DB: %s. known: %s", args.db, sorted(all_dbs.keys()))
            sys.exit(1)
        targets = [all_dbs[args.db]]
    else:
        targets = list(all_dbs.values())

    for db_config in targets:
        list_chunks_for_db(db_config, args.snippet_len)


if __name__ == "__main__":
    main()
