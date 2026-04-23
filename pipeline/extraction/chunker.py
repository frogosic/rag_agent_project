import re
from dataclasses import dataclass

from pipeline.config_loader import ChunkingConfig
from pipeline.extraction.extractors import ExtractedDocument


@dataclass
class Chunk:
    id: str
    document_id: str
    content_type: str
    text: str
    metadata: dict
    chunk_index: int


def chunk_document(doc: ExtractedDocument, config: ChunkingConfig) -> list[Chunk]:
    """Split the document into chunks according to the specified strategy in config."""
    if config.strategy == "single":
        return _single(doc)
    elif config.strategy == "paragraph":
        return _paragraph(doc, config)
    elif config.strategy == "header_based":
        return _header_based(doc, config)
    else:
        raise ValueError(f"Unknown chunking strategy: {config.strategy}")


def _make_chunk(
    doc: ExtractedDocument,
    text: str,
    index: int,
    extra_metadata: dict | None = None,
) -> Chunk:
    """Create a Chunk object from the given document and text."""
    meta = dict(doc.metadata)
    meta["chunk_index"] = index
    if extra_metadata:
        meta.update(extra_metadata)
    return Chunk(
        id=f"{doc.id}_{index}",
        document_id=doc.id,
        content_type=doc.content_type,
        text=text,
        metadata=meta,
        chunk_index=index,
    )


def _single(doc: ExtractedDocument) -> list[Chunk]:
    """Return a single chunk containing the entire document."""
    return [_make_chunk(doc, doc.text.strip(), 0)]


def _paragraph(doc: ExtractedDocument, config: ChunkingConfig) -> list[Chunk]:
    """Chunk the document by paragraphs, where paragraphs are separated by blank lines."""
    return _chunk_paragraphs(doc, doc.text, config)


def _header_based(doc: ExtractedDocument, config: ChunkingConfig) -> list[Chunk]:
    """
    Split the document into sections bounded by H2 headings (## ...).

    H1 acts as the document title and is absorbed into the first section.
    H3 and deeper headings stay within their parent H2 section.
    Sections consisting only of headings with no body are dropped.
    If a section exceeds config.max_tokens, it falls back to paragraph splitting.
    """
    lines = doc.text.split("\n")
    sections: list[str] = []
    current: list[str] = []

    for line in lines:
        if _is_h2(line) and current:
            _flush_section(current, sections)
            current = [line]
        else:
            current.append(line)

    if current:
        _flush_section(current, sections)

    chunks = []
    for section in sections:
        heading = _extract_heading(section)
        extra = {"heading": heading} if heading else None

        tokens = len(section.split())
        if tokens > config.max_tokens:
            sub = _chunk_paragraphs(
                doc,
                section,
                config,
                index_offset=len(chunks),
                extra_metadata=extra,
                use_overlap=False,
            )
            chunks.extend(sub)
        else:
            chunks.append(_make_chunk(doc, section, len(chunks), extra))

    return chunks if chunks else _single(doc)


def _extract_heading(section: str) -> str | None:
    """Return the H2 heading text (without '## ' prefix) from the first line, or None."""
    first_line = section.split("\n", 1)[0].strip()
    if _is_h2(first_line):
        return first_line.lstrip("#").strip()
    return None


def _flush_section(current: list[str], sections: list[str]) -> None:
    """Append the section to sections, unless it's heading-only or empty."""
    section_text = "\n".join(current).strip()
    if not section_text:
        return
    if _is_heading_only(section_text):
        return
    sections.append(section_text)


def _is_heading_only(text: str) -> bool:
    """True if every non-empty line in text is a markdown heading."""
    for line in text.split("\n"):
        stripped = line.strip()
        if not stripped:
            continue
        if not stripped.startswith("#"):
            return False
    return True


def _is_h2(line: str) -> bool:
    """
    True if the line is an H2 heading: '## ' or '##\\t' at the start,
    and excludes lines starting with three or more '#' (i.e., H3+).
    """
    stripped = line.lstrip()
    if not stripped.startswith("##"):
        return False
    if len(stripped) < 3:
        return False
    if not stripped[2].isspace():
        return False
    return True


def _chunk_paragraphs(
    doc: ExtractedDocument,
    text: str,
    config: ChunkingConfig,
    index_offset: int = 0,
    extra_metadata: dict | None = None,
    use_overlap: bool = True,
) -> list[Chunk]:
    """Pack paragraphs into chunks bounded by config.max_tokens."""
    raw_paragraphs = re.split(r"\n\s*\n", text)
    paragraphs = [p.strip() for p in raw_paragraphs if p.strip()]

    chunks: list[Chunk] = []
    buffer: list[str] = []
    buffer_tokens = 0

    for para in paragraphs:
        tokens = len(para.split())
        if buffer_tokens + tokens > config.max_tokens and buffer:
            chunks.append(
                _make_chunk(
                    doc,
                    "\n\n".join(buffer),
                    index_offset + len(chunks),
                    extra_metadata,
                )
            )
            if use_overlap and config.overlap > 0:
                carry = buffer[-1]
                buffer = [carry]
                buffer_tokens = len(carry.split())
            else:
                buffer = []
                buffer_tokens = 0
        buffer.append(para)
        buffer_tokens += tokens

    if buffer:
        chunks.append(
            _make_chunk(
                doc,
                "\n\n".join(buffer),
                index_offset + len(chunks),
                extra_metadata,
            )
        )

    return chunks
