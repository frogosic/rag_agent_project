import logging
from dataclasses import dataclass

from langchain_core.documents import Document
from langchain_text_splitters import (
    RecursiveCharacterTextSplitter,
    CharacterTextSplitter,
)

from config import (
    QA_CHUNK_SIZE,
    QA_CHUNK_OVERLAP,
    FICTION_CHUNK_SIZE,
    FICTION_CHUNK_OVERLAP,
)

logger = logging.getLogger(__name__)


@dataclass
class ChunkingConfig:
    """
    Encapsulates chunking parameters for a specific domain.
    Separating config from execution means you can tune numbers
    in config.py without touching split logic.
    """

    chunk_size: int
    chunk_overlap: int
    splitter_type: str  # "recursive" | "character"
    separators: list[str]  # tried in order until chunk fits


# ---------------------------------------------------------------------------
# Domain-specific chunking configs
# ---------------------------------------------------------------------------

QA_CHUNKING = ChunkingConfig(
    chunk_size=QA_CHUNK_SIZE,
    chunk_overlap=QA_CHUNK_OVERLAP,
    splitter_type="recursive",
    separators=["\n\n", "\n", ". ", " ", ""],
    # QA policies are structured with clear paragraph breaks.
    # Recursive splitter respects those boundaries before falling
    # back to sentence-level splits. Keeps policy statements intact.
)

FICTION_CHUNKING = ChunkingConfig(
    chunk_size=FICTION_CHUNK_SIZE,
    chunk_overlap=FICTION_CHUNK_OVERLAP,
    splitter_type="recursive",
    separators=["\n\n\n", "\n\n", "\n", ". ", " ", ""],
    # Fiction has deeper nesting — scene breaks use triple newlines,
    # paragraph breaks use double. We try those first to preserve
    # narrative units before cutting at sentence or word level.
    # Higher overlap ensures scenes don't lose context at boundaries.
)

DOMAIN_CHUNKING: dict[str, ChunkingConfig] = {
    "qa": QA_CHUNKING,
    "fiction": FICTION_CHUNKING,
}


# ---------------------------------------------------------------------------
# Splitter factory
# ---------------------------------------------------------------------------


def _build_splitter(config: ChunkingConfig):
    """Instantiates the right splitter from a ChunkingConfig."""
    if config.splitter_type == "recursive":
        return RecursiveCharacterTextSplitter(
            chunk_size=config.chunk_size,
            chunk_overlap=config.chunk_overlap,
            separators=config.separators,
        )
    elif config.splitter_type == "character":
        return CharacterTextSplitter(
            chunk_size=config.chunk_size,
            chunk_overlap=config.chunk_overlap,
            separator=config.separators[0],
        )
    else:
        raise ValueError(f"Unknown splitter type: {config.splitter_type}")


# ---------------------------------------------------------------------------
# Public interface
# ---------------------------------------------------------------------------


def split_documents(
    docs: list[Document],
    domain: str,
    source_name: str,
) -> list[Document]:
    """
    Splits a list of Documents using the domain-appropriate strategy.
    Preserves and enriches metadata on every chunk.

    Parameters
    ----------
    docs        : raw Documents from a loader
    domain      : "qa" | "fiction" — selects chunking config
    source_name : used to tag chunks for debugging and retrieval filtering
    """
    config = DOMAIN_CHUNKING.get(domain)
    if not config:
        raise ValueError(
            f"No chunking config for domain '{domain}'. "
            f"Available: {list(DOMAIN_CHUNKING.keys())}"
        )

    splitter = _build_splitter(config)
    chunks = splitter.split_documents(docs)

    # enrich metadata on every chunk
    for i, chunk in enumerate(chunks):
        chunk.metadata.update(
            {
                "source_name": source_name,
                "domain": domain,
                "chunk_index": i,
                "chunk_total": len(chunks),
            }
        )

    logger.info(
        "Split %d docs → %d chunks | domain=%s source=%s chunk_size=%d overlap=%d",
        len(docs),
        len(chunks),
        domain,
        source_name,
        config.chunk_size,
        config.chunk_overlap,
    )
    return chunks
