import logging

from langchain_chroma import Chroma
from langchain_openai import ChatOpenAI

from config import QA_SIMILARITY_THRESHOLD, QA_TOP_K
from sources.registry import DocumentSource
from retrieval.instructed_retriever import InstructedRetriever

logger = logging.getLogger(__name__)


class QAInstructedRetriever(InstructedRetriever):
    """
    Instructed retriever for QA policy documents.
    All domain-specific knowledge is loaded from the DocumentSource
    passed at construction — no hardcoded policy details in this class.
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
            domain="qa",
            source_name=source.name,
        )

    @property
    def index_schema(self) -> dict:
        return {
            "domain": "string — always 'qa'",
            "version": f"string — policy version, e.g. '{self._source.metadata.get('version', '')}'",
            "type": "string — always 'policy'",
            "source_name": f"string — always '{self._source.name}'",
            "chunk_index": "integer — position of chunk in document",
        }

    @property
    def system_instructions(self) -> str:
        ctx = self._source.retrieval_context

        areas = ctx.get("policy_areas", [])
        areas_block = "\n".join(f"- {a}" for a in areas) if areas else "Not specified."

        return (
            "You are decomposing queries for a QA engineering policy knowledge base.\n\n"
            f"Policy areas covered:\n{areas_block}\n\n"
            f"Version extraction rules:\n{ctx.get('version_hint', '')}\n\n"
            f"Query rules:\n{ctx.get('query_hints', '')}"
        )

    @property
    def similarity_threshold(self) -> float:
        return QA_SIMILARITY_THRESHOLD

    @property
    def top_k(self) -> int:
        return QA_TOP_K
