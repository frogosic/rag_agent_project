import yaml
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class ExtractionConfig:
    method: str
    preserve: list = field(default_factory=list)
    strip: list = field(default_factory=list)


@dataclass
class ChunkingConfig:
    strategy: str
    max_tokens: int
    overlap: int = 0


@dataclass
class ContentTypeConfig:
    name: str
    source_dir: str
    formats: list
    extraction: ExtractionConfig
    chunking: ChunkingConfig
    metadata: dict


@dataclass
class VectorDBConfig:
    name: str
    chroma_path: str
    bm25_path: str
    collection_name: str
    embedding_model: str
    sparse_weight: float
    dense_weight: float


class ConfigLoader:
    def __init__(self, config_dir: str = "config"):
        self._dir = Path(config_dir)
        self._content_types: dict[str, ContentTypeConfig] = {}
        self._vector_dbs: dict[str, VectorDBConfig] = {}
        self._load()

    def _load(self):
        with open(self._dir / "content_types.yaml") as f:
            raw = yaml.safe_load(f)
        for name, data in raw["content_types"].items():
            self._content_types[name] = ContentTypeConfig(
                name=name,
                source_dir=data["source_dir"],
                formats=data["formats"],
                extraction=ExtractionConfig(**data["extraction"]),
                chunking=ChunkingConfig(**data["chunking"]),
                metadata=data.get("metadata", {}),
            )

        with open(self._dir / "vector_databases.yaml") as f:
            raw = yaml.safe_load(f)
        for name, data in raw["vector_databases"].items():
            self._vector_dbs[name] = VectorDBConfig(
                name=name,
                chroma_path=data["chroma_path"],
                bm25_path=data["bm25_path"],
                collection_name=data["collection_name"],
                embedding_model=data["embedding_model"],
                sparse_weight=data["sparse_weight"],
                dense_weight=data["dense_weight"],
            )

    def get_content_type(self, name: str) -> ContentTypeConfig:
        if name not in self._content_types:
            raise ValueError(f"Unknown content type: {name}")
        return self._content_types[name]

    def all_content_types(self) -> dict[str, ContentTypeConfig]:
        return self._content_types

    def get_db(self, name: str) -> VectorDBConfig:
        if name not in self._vector_dbs:
            raise ValueError(f"Unknown vector DB: {name}")
        return self._vector_dbs[name]

    def default_db(self) -> VectorDBConfig:
        """Returns the single configured DB. Convenience for the single-DB world."""
        if len(self._vector_dbs) != 1:
            raise ValueError(
                f"default_db() expects exactly one DB configured, "
                f"got {len(self._vector_dbs)}: {sorted(self._vector_dbs.keys())}"
            )
        return next(iter(self._vector_dbs.values()))
