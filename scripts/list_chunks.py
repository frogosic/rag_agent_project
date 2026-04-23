import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction

from pipeline.config_loader import ConfigLoader, VectorDBConfig


logger = logging.getLogger(__name__)


def list_chunks_for_db(
    db_config: VectorDBConfig,
    snippet_len: int,
    content_type: str | None = None,
) -> None:
    """Lists chunks in the Chroma collection, showing ID, content type, source, heading, and a text snippet.

    If `content_type` is provided, only chunks with that content_type metadata are shown.
    """
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

    where = {"content_type": content_type} if content_type else None
    result = collection.get(where=where, include=["documents", "metadatas"])
    ids = result.get("ids", []) or []
    docs = result.get("documents", []) or []
    metas = result.get("metadatas", []) or []

    if not ids:
        if content_type:
            logger.warning(
                "%s: no chunks with content_type=%s", db_config.name, content_type
            )
        else:
            logger.warning("%s: collection exists but has no chunks", db_config.name)
        return

    scope_label = f" [content_type={content_type}]" if content_type else ""
    logger.info("%s: %d chunk(s)%s", db_config.name, len(ids), scope_label)

    print(f"\n=== {db_config.name} ({len(ids)} chunk(s)){scope_label} ===\n")
    for chunk_id, doc, meta in zip(ids, docs, metas):
        meta = meta or {}
        ct = meta.get("content_type", "?")
        source = meta.get("source", meta.get("source_path", "?"))
        heading = meta.get("heading")
        snippet = (doc or "").replace("\n", " ").strip()[:snippet_len]
        print(f"  id:      {chunk_id}")
        print(f"  type:    {ct}")
        print(f"  source:  {source}")
        if heading:
            print(f"  heading: {heading}")
        print(f"  text:    {snippet}...")
        print()


def main():
    """Main entry point for the list_chunks script."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--content-type",
        help="List only chunks with this content_type (e.g. technical, hr_docs, support).",
    )
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
    db_config = loader.default_db()
    list_chunks_for_db(db_config, args.snippet_len, content_type=args.content_type)


if __name__ == "__main__":
    main()
