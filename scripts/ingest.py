"""
Ingest script for processing documents into the vector DB and BM25 index.
Usage:
  python scripts/ingest.py --type <content_type_name> --config <config_path>
If --type is not specified, all content types in the config will be ingested.
"""

import argparse
import json
import sys
import logging
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).parent.parent))

import bm25s
import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction

from pipeline.config_loader import ConfigLoader, ContentTypeConfig, VectorDBConfig
from pipeline.extraction.chunker import Chunk, chunk_document
from pipeline.extraction.extractors import (
    BaseExtractor,
    ExtractedDocument,
    get_extractor,
)

logger = logging.getLogger(__name__)


def get_chroma_collection(db_config: VectorDBConfig):
    """Get or create the Chroma collection for the configured DB."""
    path = Path(db_config.chroma_path)
    path.mkdir(parents=True, exist_ok=True)

    client = chromadb.PersistentClient(path=str(path))
    ef = SentenceTransformerEmbeddingFunction(model_name=db_config.embedding_model)

    return client.get_or_create_collection(
        name=db_config.collection_name,
        embedding_function=ef,  # type: ignore[arg-type]
        metadata={"hnsw:space": "cosine"},
    )


def build_and_save_bm25_index(db_name: str, chunks: list[Chunk]) -> None:
    """
    Builds a bm25s index over all chunks and saves:
      data/bm25/{db_name}/index        ← bm25s native format
      data/bm25/{db_name}/id_map.json  ← [chunk_id, ...] positional mapping
    """
    bm25_dir: Path = Path("data/bm25") / db_name
    bm25_dir.mkdir(parents=True, exist_ok=True)

    texts: list[str] = [c.text for c in chunks]
    ids: list[str] = [c.id for c in chunks]

    corpus_tokens = bm25s.tokenize(texts, stopwords="en")

    retriever = bm25s.BM25()
    retriever.index(corpus_tokens)
    retriever.save(str(bm25_dir / "index"))

    id_map_path: Path = bm25_dir / "id_map.json"
    id_map_path.write_text(json.dumps(ids, indent=2))

    logger.info(f"bm25s index: {bm25_dir}/index")
    logger.info(f"id map: {len(ids)} entries → {id_map_path}")


def ingest_content_type(
    name: str,
    loader: ConfigLoader,
    collection: chromadb.Collection,
) -> list[Chunk]:
    """Ingests all documents of a given content type and returns the chunks produced."""
    ct: ContentTypeConfig = loader.get_content_type(name)

    source_dir = Path(ct.source_dir)
    if not source_dir.exists():
        logger.info(f"[skip] source dir not found: {source_dir}")
        return []

    files = []
    for fmt in ct.formats:
        files.extend(source_dir.glob(f"*.{fmt}"))

    if not files:
        logger.info(f"[skip] no files found in {source_dir}")
        return []

    extractor: BaseExtractor = get_extractor(ct)
    all_chunks: list[Chunk] = []

    for file_path in sorted(files):
        logger.info(f"  extracting:  {file_path.name}")
        doc: ExtractedDocument = extractor.extract(file_path)
        chunks: list[Chunk] = chunk_document(doc, ct.chunking)

        if not chunks:
            logger.warning("[warn] no chunks produced")
            continue

        collection.upsert(
            ids=[c.id for c in chunks],
            documents=[c.text for c in chunks],
            metadatas=[c.metadata for c in chunks],
        )

        logger.info(f"chroma: {len(chunks)} chunk(s)")
        all_chunks.extend(chunks)

    return all_chunks


def main():
    """Main entry point for the ingest script."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--type", help="content type to ingest (default: all)")
    parser.add_argument("--config", default="config")
    args: argparse.Namespace = parser.parse_args()

    loader = ConfigLoader(args.config)
    types: list[Any] | list[str] = (
        [args.type] if args.type else list(loader.all_content_types().keys())
    )

    logger.info(f"ingesting: {types}\n")

    db = loader.default_db()
    collection = get_chroma_collection(db)

    all_chunks: list[Chunk] = []

    for name in types:
        logger.info(f"[{name}]")
        chunks = ingest_content_type(name, loader, collection)
        logger.info(f"total: {len(chunks)} chunk(s)\n")
        all_chunks.extend(chunks)

    logger.info("building bm25s index...")
    if all_chunks:
        build_and_save_bm25_index(db.name, all_chunks)

    logger.info(f"\ndone — {len(all_chunks)} chunk(s) ingested")
