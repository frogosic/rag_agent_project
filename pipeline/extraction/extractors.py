import hashlib
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path

from pipeline.config_loader import ContentTypeConfig


@dataclass
class ExtractedDocument:
    id: str
    content_type: str
    text: str
    metadata: dict
    database: str
    source_path: str
    flags: list = field(default_factory=list)


class BaseExtractor(ABC):
    def __init__(self, config: ContentTypeConfig):
        self.config = config

    @abstractmethod
    def extract(self, source_path: Path) -> ExtractedDocument:
        pass

    def _make_id(self, source_path: Path) -> str:
        h = hashlib.md5(str(source_path).encode()).hexdigest()[:10]
        return f"{self.config.name}_{source_path.stem}_{h}"

    def _base_metadata(self, source_path: Path) -> dict:
        meta = dict(self.config.metadata)
        meta["filename"] = source_path.name
        meta["source_path"] = str(source_path)
        return meta


class MarkdownExtractor(BaseExtractor):
    def extract(self, source_path: Path) -> ExtractedDocument:
        raw = source_path.read_text(encoding="utf-8")
        text = self._process(raw)
        return ExtractedDocument(
            id=self._make_id(source_path),
            content_type=self.config.name,
            text=text,
            metadata=self._base_metadata(source_path),
            database=self.config.database,
            source_path=str(source_path),
        )

    def _process(self, raw: str) -> str:
        preserve = self.config.extraction.preserve
        protected = {}
        counter = 0

        if "code_blocks" in preserve:

            def protect(m):
                nonlocal counter
                key = f"__CODE_{counter}__"
                protected[key] = m.group(0)
                counter += 1
                return key

            raw = re.sub(r"```[\s\S]*?```", protect, raw)
            raw = re.sub(r"`[^`\n]+`", protect, raw)

        # strip markdown syntax, preserve text
        text = re.sub(r"^#{1,6}\s+", "", raw, flags=re.MULTILINE)
        text = re.sub(r"\*{1,2}([^*\n]+)\*{1,2}", r"\1", text)
        text = re.sub(r"_{1,2}([^_\n]+)_{1,2}", r"\1", text)
        text = re.sub(r"\[([^\]]+)\]\([^\)]+\)", r"\1", text)
        text = re.sub(r"!\[[^\]]*\]\([^\)]+\)", "", text)

        for key, original in protected.items():
            text = text.replace(key, original)

        return text.strip()


class PlaintextExtractor(BaseExtractor):
    def extract(self, source_path: Path) -> ExtractedDocument:
        text = source_path.read_text(encoding="utf-8").strip()
        return ExtractedDocument(
            id=self._make_id(source_path),
            content_type=self.config.name,
            text=text,
            metadata=self._base_metadata(source_path),
            database=self.config.database,
            source_path=str(source_path),
        )


def get_extractor(config: ContentTypeConfig) -> BaseExtractor:
    method = config.extraction.method
    if method == "markdown":
        return MarkdownExtractor(config)
    elif method == "plaintext":
        return PlaintextExtractor(config)
    else:
        raise ValueError(f"Unknown extraction method: {method}")
