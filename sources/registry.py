import logging
import yaml
from dataclasses import dataclass, field
from pathlib import Path

from config import BASE_DIR, QA_CHUNK_SIZE, QA_CHUNK_OVERLAP

logger = logging.getLogger(__name__)

SOURCES_YAML = Path(__file__).parent / "sources.yaml"


@dataclass
class DocumentSource:
    """
    Declarative definition of a single knowledge source.

    Instances are loaded from sources/sources.yaml at startup.
    The dataclass acts as the schema — if a required field is missing
    or misspelled in the YAML, DocumentSource(**entry) will raise
    a clear TypeError at startup rather than failing silently later.
    """

    name: str
    path: str
    source_type: str
    domain: str
    description: str
    metadata: dict = field(default_factory=dict)
    chunk_size: int = QA_CHUNK_SIZE
    chunk_overlap: int = QA_CHUNK_OVERLAP
    retrieval_context: dict = field(default_factory=dict)


def _resolve_path(relative_path: str) -> str:
    """Resolves a relative path from sources.yaml to an absolute project path."""
    return str(BASE_DIR / relative_path)


def _load_sources() -> list[DocumentSource]:
    """
    Loads and validates all source definitions from sources.yaml.
    Paths are resolved to absolute paths relative to the project root.
    Domain grouping in the YAML is for human readability only —
    the domain field on each entry is the actual source of truth.
    """
    if not SOURCES_YAML.exists():
        raise FileNotFoundError(
            f"sources.yaml not found at {SOURCES_YAML}. "
            "Create it before starting the application."
        )

    with open(SOURCES_YAML, "r") as f:
        raw = yaml.safe_load(f)

    if not raw:
        raise ValueError("sources.yaml is empty — no sources registered.")

    sources = []
    for domain_group, entries in raw.items():
        if not entries:
            logger.warning(
                "Domain group '%s' has no entries in sources.yaml", domain_group
            )
            continue

        for entry in entries:
            # validate domain consistency — YAML group should match domain field
            if entry.get("domain") != domain_group:
                raise ValueError(
                    f"Source '{entry.get('name')}' has domain='{entry.get('domain')}' "
                    f"but is listed under '{domain_group}' group in sources.yaml. "
                    "These must match."
                )

            entry["path"] = _resolve_path(entry["path"])
            source = DocumentSource(**entry)
            sources.append(source)
            logger.debug(
                "Registered source: %s (domain=%s)", source.name, source.domain
            )

    logger.info("Loaded %d sources from sources.yaml", len(sources))
    return sources


# ---------------------------------------------------------------------------
# Module-level singletons — loaded once at import time
# ---------------------------------------------------------------------------
ALL_SOURCES: list[DocumentSource] = _load_sources()

SOURCE_REGISTRY: dict[str, DocumentSource] = {
    source.name: source for source in ALL_SOURCES
}

QA_SOURCES: list[DocumentSource] = [s for s in ALL_SOURCES if s.domain == "qa"]

FICTION_SOURCES: list[DocumentSource] = [
    s for s in ALL_SOURCES if s.domain == "fiction"
]


def get_source(name: str) -> DocumentSource:
    """
    Retrieves a source by name. Raises clearly if not found
    rather than returning None silently.
    """
    source = SOURCE_REGISTRY.get(name)
    if not source:
        raise KeyError(
            f"Source '{name}' not found in registry. "
            f"Available sources: {list(SOURCE_REGISTRY.keys())}"
        )
    return source


def get_sources_by_domain(domain: str) -> list[DocumentSource]:
    """Returns all sources for a given domain."""
    sources = [s for s in ALL_SOURCES if s.domain == domain]
    if not sources:
        raise KeyError(
            f"No sources found for domain '{domain}'. "
            f"Available domains: {list({s.domain for s in ALL_SOURCES})}"
        )
    return sources
