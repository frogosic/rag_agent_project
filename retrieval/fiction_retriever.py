import logging

from langchain_chroma import Chroma
from langchain_openai import ChatOpenAI

from config import FICTION_SIMILARITY_THRESHOLD, FICTION_TOP_K
from sources.registry import DocumentSource
from retrieval.instructed_retriever import InstructedRetriever

logger = logging.getLogger(__name__)


class FictionInstructedRetriever(InstructedRetriever):
    """
    Instructed retriever for fiction documents.
    All book-specific knowledge is loaded from the DocumentSource
    passed at construction — this class works for any fiction source.
    """

    def __init__(
        self,
        llm: ChatOpenAI,
        vector_store: Chroma,
        source: DocumentSource,
    ):
        self._source = source
        super().__init__(
            llm=llm,
            vector_store=vector_store,
            domain="fiction",
            source_name=source.name,
        )

    @property
    def index_schema(self) -> dict:
        return {
            "domain": "string — always 'fiction'",
            "author": f"string — always '{self._source.metadata.get('author', '')}'",
            "book": f"string — always '{self._source.metadata.get('book', '')}'",
            "genre": f"string — always '{self._source.metadata.get('genre', '')}'",
            "type": "string — always 'novel'",
            "source_name": f"string — always '{self._source.name}'",
            "chunk_index": "integer — position of chunk in document",
        }

    @property
    def system_instructions(self) -> str:
        ctx = self._source.retrieval_context
        book = self._source.metadata.get("book", "this novel")
        author = self._source.metadata.get("author", "the author")

        entities = ctx.get("key_entities", [])
        entities_block = (
            "\n".join(f"- {e}" for e in entities) if entities else "Not specified."
        )

        return (
            f"You are decomposing queries for a fiction knowledge base containing "
            f"{book} by {author}.\n\n"
            f"Key entities:\n{entities_block}\n\n"
            f"Vocabulary note:\n{ctx.get('overlapping_vocabulary', '')}\n\n"
            f"Query rules:\n{ctx.get('query_hints', '')}"
        )

    @property
    def similarity_threshold(self) -> float:
        return FICTION_SIMILARITY_THRESHOLD

    @property
    def top_k(self) -> int:
        return FICTION_TOP_K
