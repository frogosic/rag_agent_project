import logging
from pathlib import Path

import ebooklib
from ebooklib import epub
from bs4 import BeautifulSoup
from langchain_core.documents import Document
from langchain_community.document_loaders import (
    TextLoader,
    PyPDFLoader,
    DirectoryLoader,
)

from sources.registry import DocumentSource

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Loader registry
# ---------------------------------------------------------------------------
# Maps source_type → loader class.
# epub is handled separately via ebooklib — no pandoc required.
# Adding support for a new file type = one line here, nothing else changes.

_LOADER_MAP = {
    "text": TextLoader,
    "pdf": PyPDFLoader,
}

# front matter signals — item names containing these are skipped
_EPUB_SKIP_SIGNALS = [
    "cover",
    "toc",
    "copyright",
    "title",
    "acknowledgement",
    "dedication",
    "colophon",
    "halftitle",
    "nav",
]


# ---------------------------------------------------------------------------
# EPUB loader
# ---------------------------------------------------------------------------


def _load_epub(path: str) -> list[Document]:
    """
    Loads an EPUB chapter by chapter using ebooklib.
    Each chapter becomes a separate Document with chapter metadata.
    Front matter items (cover, toc, copyright etc.) are skipped.
    This gives the chunker proper narrative boundaries to work with
    instead of treating the entire book as one blob.
    """
    book = epub.read_epub(path)
    docs = []
    chapter_num = 0

    for item in book.get_items():
        if item.get_type() != ebooklib.ITEM_DOCUMENT:
            continue

        item_name = item.get_name().lower()

        # skip front matter
        if any(signal in item_name for signal in _EPUB_SKIP_SIGNALS):
            logger.debug("Skipping front matter item: %s", item_name)
            continue

        # extract clean text from HTML
        soup = BeautifulSoup(item.get_content(), "html.parser")
        text = soup.get_text(separator="\n", strip=True)

        # skip near-empty items — likely structural HTML with no content
        if len(text.strip()) < 100:
            logger.debug(
                "Skipping near-empty item: %s (%d chars)", item_name, len(text)
            )
            continue

        chapter_num += 1
        docs.append(
            Document(
                page_content=text,
                metadata={
                    "chapter": chapter_num,
                    "item_name": item.get_name(),
                },
            )
        )

    if not docs:
        raise ValueError(
            f"ebooklib extracted 0 chapters from {path}. "
            "Check the EPUB structure — all items may have been filtered as front matter."
        )

    logger.info(
        "ebooklib loaded %d chapters from '%s' (front matter skipped)",
        len(docs),
        path,
    )
    return docs


# ---------------------------------------------------------------------------
# Metadata attachment
# ---------------------------------------------------------------------------


def _attach_source_metadata(
    docs: list[Document],
    source: DocumentSource,
) -> list[Document]:
    """
    Attaches DocumentSource metadata to every loaded Document.
    Runs before chunking so metadata propagates to all chunks.
    Chapter metadata from ebooklib is preserved and merged.
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


# ---------------------------------------------------------------------------
# Path validation
# ---------------------------------------------------------------------------


def _validate_path(source: DocumentSource) -> None:
    """Raises clearly if the source file or directory doesn't exist."""
    path = Path(source.path)
    if not path.exists():
        raise FileNotFoundError(
            f"Source '{source.name}' path not found: {source.path}\n"
            f"Create the file or update sources.yaml."
        )


# ---------------------------------------------------------------------------
# Public interface
# ---------------------------------------------------------------------------


def load(source: DocumentSource) -> list[Document]:
    """
    Loads a DocumentSource into a list of raw Documents.
    Metadata is attached at load time so it propagates through chunking.

    EPUB sources use ebooklib directly for chapter-level structure.
    All other sources use the appropriate LangChain loader.
    """
    _validate_path(source)

    if source.source_type == "epub":
        docs = _load_epub(source.path)

    elif source.source_type == "directory":
        loader = DirectoryLoader(
            source.path,
            glob="**/*.txt",
            loader_cls=TextLoader,
        )
        docs = loader.load()

    elif source.source_type in _LOADER_MAP:
        loader_cls = _LOADER_MAP[source.source_type]
        loader = loader_cls(source.path)
        docs = loader.load()

    else:
        raise ValueError(
            f"Unknown source_type '{source.source_type}' for source '{source.name}'. "
            f"Supported types: {list(_LOADER_MAP.keys()) + ['directory', 'epub']}"
        )

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
