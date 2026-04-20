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
    database: str


@dataclass
class VectorDBConfig:
    name: str
    chroma_path: str
    collection_name: str
    embedding_model: str
    description: str
    sparse_weight: float
    dense_weight: float
    access_control: str
    triggers: dict = field(default_factory=dict)


class ConfigLoader:
    def __init__(self, config_dir: str = "config"):
        self._dir = Path(config_dir)
        self._content_types: dict[str, ContentTypeConfig] = {}
        self._vector_dbs: dict[str, VectorDBConfig] = {}
        self._tones: dict[str, str] = {}
        self._load()
        self._validate()

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
                database=data["database"],
            )

        with open(self._dir / "vector_databases.yaml") as f:
            raw = yaml.safe_load(f)
        for name, data in raw["vector_databases"].items():
            self._vector_dbs[name] = VectorDBConfig(
                name=name,
                chroma_path=data["chroma_path"],
                collection_name=data["collection_name"],
                embedding_model=data["embedding_model"],
                description=data["description"].strip(),
                sparse_weight=data["sparse_weight"],
                dense_weight=data["dense_weight"],
                access_control=data["access_control"],
                triggers=data.get("triggers", {}),
            )

        with open(self._dir / "tones.yaml") as f:
            raw = yaml.safe_load(f)
        self._tones = raw["tones"]

    def _validate(self):
        """Startup validation. Fails loudly and early on bad configs."""
        db_names = set(self._vector_dbs.keys())

        for ct_name, ct in self._content_types.items():
            if ct.database not in db_names:
                raise ValueError(
                    f"content_type '{ct_name}' references unknown database "
                    f"'{ct.database}'. Known DBs: {sorted(db_names)}"
                )

        if "general" not in self._tones:
            raise ValueError(
                "tones.yaml must define a 'general' tone (used as fallback)."
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

    def all_databases(self) -> dict[str, VectorDBConfig]:
        return self._vector_dbs

    def tone_instructions(self) -> dict[str, str]:
        return self._tones

    def db_descriptions(self, access_tier: str = "internal") -> dict[str, str]:
        allowed = {"restricted": ["restricted", "internal"], "internal": ["internal"]}
        tiers = allowed.get(access_tier, ["internal"])
        return {
            name: cfg.description
            for name, cfg in self._vector_dbs.items()
            if cfg.access_control in tiers
        }
