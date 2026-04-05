import logging
from pathlib import Path

from langchain_core.documents import Document
from langchain_community.document_loaders import (
    TextLoader,
    PyPDFLoader,
    DirectoryLoader,
    UnstructuredEPubLoader,
)

from sources.registry import DocumentSource

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Loader registry
# ---------------------------------------------------------------------------
# Maps source_type → loader class.
# Adding support for a new file type = one line here, nothing else changes.

_LOADER_MAP = {
    "text": TextLoader,
    "pdf": PyPDFLoader,
    "epub": UnstructuredEPubLoader,
}


def _validate_path(source: DocumentSource) -> None:
    """Raises clearly if the source file/directory doesn't exist."""
    path = Path(source.path)
    if not path.exists():
        raise FileNotFoundError(
            f"Source '{source.name}' path not found: {source.path}\n"
            f"Create the file or update sources.yaml."
        )


def _attach_source_metadata(
    docs: list[Document],
    source: DocumentSource,
) -> list[Document]:
    """
    Attaches DocumentSource metadata to every loaded Document.
    This runs before chunking so metadata propagates to all chunks.
    """
    for doc in docs:
        doc.metadata.update(
            {
                "source_name": source.name,
                "domain": source.domain,
                **source.metadata,
            }
        )
    return docs


def load(source: DocumentSource) -> list[Document]:
    """
    Loads a DocumentSource into a list of raw Documents.
    Metadata is attached at load time so it propagates through chunking.

    Parameters
    ----------
    source : DocumentSource from the registry
    """
    _validate_path(source)

    if source.source_type == "directory":
        loader = DirectoryLoader(
            source.path,
            glob="**/*.txt",  # default — override per source if needed
            loader_cls=TextLoader,
        )
    elif source.source_type in _LOADER_MAP:
        loader_cls = _LOADER_MAP[source.source_type]
        loader = loader_cls(source.path)
    else:
        raise ValueError(
            f"Unknown source_type '{source.source_type}' for source '{source.name}'. "
            f"Supported types: {list(_LOADER_MAP.keys()) + ['directory']}"
        )

    logger.info(
        "Loading source '%s' | type=%s path=%s",
        source.name,
        source.source_type,
        source.path,
    )

    docs = loader.load()

    if not docs:
        raise ValueError(
            f"Loader returned 0 documents for source '{source.name}'. "
            f"Check the file at {source.path}."
        )

    docs = _attach_source_metadata(docs, source)

    logger.info(
        "Loaded %d raw documents from '%s'",
        len(docs),
        source.name,
    )
    return docs
