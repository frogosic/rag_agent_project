import logging
import os
from pathlib import Path

from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings

from config import CHROMA_PATH, EMBEDDING_MODEL
from sources.registry import DocumentSource
from ingestion.loader_factory import load
from ingestion.chunking_strategy import split_documents

logger = logging.getLogger(__name__)


class CollectionManager:
    """
    Manages one ChromaDB collection per DocumentSource.

    Responsibilities:
    - Check if a collection already exists on disk (skip re-indexing)
    - Load and chunk documents when indexing for the first time
    - Expose a standard LangChain retriever per collection
    - Act as the single point of contact for all ChromaDB operations

    One instance of this class should exist for the lifetime of the app.
    """

    def __init__(self):
        self._embeddings = OpenAIEmbeddings(
            model=EMBEDDING_MODEL,
            api_key=os.getenv("OPENAI_API_KEY"),  # type: ignore
        )
        self._collections: dict[str, Chroma] = {}
        Path(CHROMA_PATH).mkdir(parents=True, exist_ok=True)

    # ---------------------------------------------------------------------------
    # Public interface
    # ---------------------------------------------------------------------------

    def load_source(self, source: DocumentSource) -> None:
        """
        Loads a source into ChromaDB if not already indexed.
        Safe to call multiple times — skip logic prevents re-indexing.
        """
        if source.name in self._collections:
            logger.debug("Source '%s' already loaded in memory.", source.name)
            return

        if self._collection_exists(source.name):
            self._load_existing(source)
        else:
            self._index_new(source)

    def load_all(self, sources: list[DocumentSource]) -> None:
        """Convenience method — loads all sources at startup."""
        for source in sources:
            self.load_source(source)
        logger.info(
            "All sources loaded. Collections: %s",
            list(self._collections.keys()),
        )

    def get_retriever(
        self,
        source_name: str,
        k: int = 3,
        threshold: float = 0.5,
    ):
        """
        Returns a LangChain retriever for a named collection.
        Raises clearly if the collection hasn't been loaded yet.
        """
        collection = self._get_collection(source_name)
        return collection.as_retriever(
            search_type="similarity_score_threshold",
            search_kwargs={
                "k": k,
                "score_threshold": threshold,
            },
        )

    def get_collection(self, source_name: str) -> Chroma:
        """Direct access to the Chroma vector store for a source."""
        return self._get_collection(source_name)

    def collection_names(self) -> list[str]:
        """Returns names of all currently loaded collections."""
        return list(self._collections.keys())

    # ---------------------------------------------------------------------------
    # Internal
    # ---------------------------------------------------------------------------

    def _collection_exists(self, name: str) -> bool:
        """
        Checks if a collection has been persisted to disk.
        Uses ChromaDB's internal directory structure as the signal.
        """
        chroma_dir = Path(CHROMA_PATH)
        # ChromaDB persists collections as subdirectories
        # A collection exists if the chroma.sqlite3 references it
        # Simplest reliable check: try loading and catch
        try:
            test = Chroma(
                collection_name=name,
                embedding_function=self._embeddings,
                persist_directory=CHROMA_PATH,
                collection_metadata={"hnsw:space": "cosine"},
            )
            count = test._collection.count()
            if count > 0:
                logger.info(
                    "Found existing collection '%s' with %d chunks.",
                    name,
                    count,
                )
                return True
            logger.info(
                "Collection '%s' exists but is empty — will re-index.",
                name,
            )
            return False
        except Exception:
            return False

    def _load_existing(self, source: DocumentSource) -> None:
        """Loads an already-indexed collection from disk into memory."""
        vector_store = Chroma(
            collection_name=source.name,
            embedding_function=self._embeddings,
            persist_directory=CHROMA_PATH,
            collection_metadata={"hnsw:space": "cosine"},
        )
        self._collections[source.name] = vector_store
        logger.info(
            "Loaded existing collection '%s' (%d chunks) from disk.",
            source.name,
            vector_store._collection.count(),
        )

    def _index_new(self, source: DocumentSource) -> None:
        """
        Full ingestion pipeline for a new source:
        load → chunk → embed → persist to ChromaDB.
        """
        logger.info("Indexing new source '%s'...", source.name)

        # load raw documents
        docs = load(source)

        # chunk with domain-appropriate strategy
        chunks = split_documents(
            docs=docs,
            domain=source.domain,
            source_name=source.name,
        )

        # embed and persist
        vector_store = Chroma.from_documents(
            documents=chunks,
            embedding=self._embeddings,
            collection_name=source.name,
            persist_directory=CHROMA_PATH,
            collection_metadata={"hnsw:space": "cosine"},
        )
        self._collections[source.name] = vector_store

        logger.info(
            "Indexed '%s' → %d chunks persisted to ChromaDB.",
            source.name,
            len(chunks),
        )

    def _get_collection(self, source_name: str) -> Chroma:
        """Internal getter with a clear error if collection isn't loaded."""
        collection = self._collections.get(source_name)
        if not collection:
            raise KeyError(
                f"Collection '{source_name}' not loaded. "
                f"Call load_source() first. "
                f"Loaded collections: {self.collection_names()}"
            )
        return collection
