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
    database: str
    chunk_index: int


def chunk_document(doc: ExtractedDocument, config: ChunkingConfig) -> list[Chunk]:
    if config.strategy == "single":
        return _single(doc)
    elif config.strategy == "paragraph":
        return _paragraph(doc, config)
    elif config.strategy == "header_based":
        return _header_based(doc, config)
    else:
        raise ValueError(f"Unknown chunking strategy: {config.strategy}")


def _make_chunk(doc: ExtractedDocument, text: str, index: int) -> Chunk:
    meta = dict(doc.metadata)
    meta["chunk_index"] = index
    return Chunk(
        id=f"{doc.id}_{index}",
        document_id=doc.id,
        content_type=doc.content_type,
        text=text,
        metadata=meta,
        database=doc.database,
        chunk_index=index,
    )


def _single(doc: ExtractedDocument) -> list[Chunk]:
    return [_make_chunk(doc, doc.text.strip(), 0)]


def _paragraph(doc: ExtractedDocument, config: ChunkingConfig) -> list[Chunk]:
    raw_paragraphs = re.split(r"\n\s*\n", doc.text)
    paragraphs = [p.strip() for p in raw_paragraphs if p.strip()]

    chunks = []
    buffer: list[str] = []
    buffer_tokens = 0

    for para in paragraphs:
        tokens = len(para.split())
        if buffer_tokens + tokens > config.max_tokens and buffer:
            chunks.append(_make_chunk(doc, "\n\n".join(buffer), len(chunks)))
            if config.overlap > 0:
                carry = buffer[-1]
                buffer = [carry]
                buffer_tokens = len(carry.split())
            else:
                buffer = []
                buffer_tokens = 0
        buffer.append(para)
        buffer_tokens += tokens

    if buffer:
        chunks.append(_make_chunk(doc, "\n\n".join(buffer), len(chunks)))

    return chunks


def _header_based(doc: ExtractedDocument, config: ChunkingConfig) -> list[Chunk]:
    lines = doc.text.split("\n")
    sections: list[str] = []
    current: list[str] = []

    for line in lines:
        stripped = line.strip()
        is_header = (
            stripped
            and len(stripped) < 60
            and not stripped.endswith(".")
            and not stripped.startswith("-")
            and not stripped.startswith("*")
            and not stripped.startswith("{")
            and not stripped.startswith("`")
        )
        if is_header and current:
            section_text = "\n".join(current).strip()
            if section_text:
                sections.append(section_text)
            current = [line]
        else:
            current.append(line)

    if current:
        section_text = "\n".join(current).strip()
        if section_text:
            sections.append(section_text)

    chunks = []
    for section in sections:
        tokens = len(section.split())
        if tokens > config.max_tokens:
            # oversized section — fall back to paragraph splitting
            sub = _paragraph_from_text(doc, section, config, len(chunks))
            chunks.extend(sub)
        else:
            chunks.append(_make_chunk(doc, section, len(chunks)))

    return chunks if chunks else _single(doc)


def _paragraph_from_text(
    doc: ExtractedDocument,
    text: str,
    config: ChunkingConfig,
    index_offset: int,
) -> list[Chunk]:
    raw_paragraphs = re.split(r"\n\s*\n", text)
    paragraphs = [p.strip() for p in raw_paragraphs if p.strip()]
    chunks = []
    buffer: list[str] = []
    buffer_tokens = 0

    for para in paragraphs:
        tokens = len(para.split())
        if buffer_tokens + tokens > config.max_tokens and buffer:
            chunks.append(
                _make_chunk(doc, "\n\n".join(buffer), index_offset + len(chunks))
            )
            buffer = []
            buffer_tokens = 0
        buffer.append(para)
        buffer_tokens += tokens

    if buffer:
        chunks.append(_make_chunk(doc, "\n\n".join(buffer), index_offset + len(chunks)))

    return chunks
