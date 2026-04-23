import hashlib

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from markdown_it import MarkdownIt
from markdown_it.token import Token

from pipeline.config_loader import ContentTypeConfig


@dataclass
class ExtractedDocument:
    id: str
    content_type: str
    text: str
    metadata: dict
    source_path: str
    flags: list = field(default_factory=list)


class BaseExtractor(ABC):
    def __init__(self, config: ContentTypeConfig):
        self.config: ContentTypeConfig = config

    @abstractmethod
    def extract(self, source_path: Path) -> ExtractedDocument:
        """Extract text and metadata from the given source path."""
        pass

    def _make_id(self, source_path: Path) -> str:
        """Generate a unique ID for the document based on the source path."""
        h: str = hashlib.md5(str(source_path).encode()).hexdigest()[:10]
        return f"{self.config.name}_{source_path.stem}_{h}"

    def _base_metadata(self, source_path: Path) -> dict:
        """Base metadata for all extractors. Conforms to the chunk metadata schema."""
        meta: dict = dict(self.config.metadata)
        meta["source"] = source_path.name
        meta["source_path"] = str(source_path)
        meta["doc_format"] = source_path.suffix.lstrip(".")
        return meta


class MarkdownExtractor(BaseExtractor):
    def __init__(self, config: ContentTypeConfig):
        super().__init__(config)
        self._parser = MarkdownIt("commonmark")

    def extract(self, source_path: Path) -> ExtractedDocument:
        """Read the file as markdown and convert to plaintext, with optional preservation of code blocks."""
        raw = source_path.read_text(encoding="utf-8")
        text = self._process(raw)
        return ExtractedDocument(
            id=self._make_id(source_path),
            content_type=self.config.name,
            text=text,
            metadata=self._base_metadata(source_path),
            source_path=str(source_path),
        )

    def _process(self, raw: str) -> str:
        """Convert markdown to plaintext, with optional preservation of code blocks."""
        preserve_code: bool = "code_blocks" in self.config.extraction.preserve
        tokens: list[Token] = self._parser.parse(raw)
        lines: list[str] = []

        for token in tokens:
            if token.type == "heading_open":
                level = int(token.tag[1])  # "h1" -> 1
                lines.append("#" * level + " ")
            elif token.type == "heading_close":
                lines.append("\n")
            elif token.type == "paragraph_open":
                pass
            elif token.type == "paragraph_close":
                lines.append("\n\n")
            elif token.type == "fence":
                if preserve_code:
                    fence = "```"
                    info = token.info or ""
                    lines.append(f"{fence}{info}\n{token.content}{fence}\n\n")
            elif token.type == "code_block":
                if preserve_code:
                    lines.append(token.content + "\n\n")
            elif token.type == "inline":
                lines.append(self._render_inline(token, preserve_code))
            elif token.type in (
                "bullet_list_open",
                "ordered_list_open",
                "list_item_open",
            ):
                pass
            elif token.type in ("bullet_list_close", "ordered_list_close"):
                lines.append("\n")
            elif token.type == "list_item_close":
                lines.append("\n")

        return "".join(lines).strip()

    def _render_inline(self, token: Token, preserve_code: bool) -> str:
        """Render inline tokens, with optional preservation of inline code."""
        parts: list[str] = []
        for child in token.children or []:
            if child.type == "text":
                parts.append(child.content)
            elif child.type == "code_inline":
                if preserve_code:
                    parts.append(f"`{child.content}`")
                else:
                    parts.append(child.content)
            elif child.type == "softbreak":
                parts.append(" ")
            elif child.type == "hardbreak":
                parts.append("\n")
            elif child.type == "link_open":
                pass
            elif child.type == "link_close":
                pass
            elif child.type == "image":
                pass
        return "".join(parts)


class PlaintextExtractor(BaseExtractor):
    def extract(self, source_path: Path) -> ExtractedDocument:
        """Read the entire file as plaintext."""
        text = source_path.read_text(encoding="utf-8").strip()
        return ExtractedDocument(
            id=self._make_id(source_path),
            content_type=self.config.name,
            text=text,
            metadata=self._base_metadata(source_path),
            source_path=str(source_path),
        )


def get_extractor(config: ContentTypeConfig) -> BaseExtractor:
    """Factory method to get the appropriate extractor based on config."""
    method: str = config.extraction.method
    if method == "markdown":
        return MarkdownExtractor(config)
    elif method == "plaintext":
        return PlaintextExtractor(config)
    else:
        raise ValueError(f"Unknown extraction method: {method}")
