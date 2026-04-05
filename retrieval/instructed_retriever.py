import logging
from abc import ABC, abstractmethod

from langchain_chroma import Chroma
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Structured query model
# ---------------------------------------------------------------------------


class StructuredQuery(BaseModel):
    """
    What the query generator produces before hitting the vector store.
    The LLM reads the user query + system instructions + index schema
    and generates this — translating natural language into a structured
    retrieval plan.

    This is the core of the Instructed Retriever pattern:
    system specifications are propagated INTO the retrieval stage,
    not dropped after the initial query.
    """

    semantic_query: str = Field(
        description="Refined semantic search query optimised for vector similarity"
    )
    metadata_filters: dict = Field(
        description=(
            "Structured ChromaDB metadata filters derived from natural language. "
            "e.g. 'last quarter' → {'version': '2024-Q1'}, "
            "'chapter 3' → {'chapter': 3}. "
            "Empty dict if no filters apply."
        )
    )
    sub_queries: list[str] = Field(
        description=(
            "Decomposed sub-questions if the query is complex or multi-part. "
            "Each sub-query will be retrieved independently and results merged. "
            "Empty list if query is simple."
        )
    )
    reasoning: str = Field(
        description="Brief explanation of why this decomposition was chosen"
    )


# ---------------------------------------------------------------------------
# Base instructed retriever
# ---------------------------------------------------------------------------


class InstructedRetriever(ABC):
    """
    Abstract base for all domain-specific instructed retrievers.

    The pattern:
    1. Query generator reads user query + system specs + index schema
    2. Produces a StructuredQuery — semantic query + filters + sub-queries
    3. Each query hits the vector store with metadata filters applied
    4. Results are deduplicated, scored, and returned as enriched dicts

    Subclasses define:
    - index_schema      : what metadata fields exist in this collection
    - system_instructions: domain-specific retrieval guidance
    - similarity_threshold: how strict similarity scoring is per domain
    """

    def __init__(
        self,
        llm: ChatOpenAI,
        vector_store: Chroma,
        domain: str,
        source_name: str,
    ):
        self.llm = llm
        self.vector_store = vector_store
        self.domain = domain
        self.source_name = source_name
        self._init_query_generator()

    # ---------------------------------------------------------------------------
    # Abstract interface — subclasses define these
    # ---------------------------------------------------------------------------

    @property
    @abstractmethod
    def index_schema(self) -> dict:
        """
        Metadata fields available in this collection.
        The query generator uses this to produce valid filters.
        """

    @property
    @abstractmethod
    def system_instructions(self) -> str:
        """
        Domain-specific retrieval guidance passed to the query generator.
        Tells the LLM how to decompose queries and what to filter on
        for this specific domain.
        """

    @property
    def similarity_threshold(self) -> float:
        """Override per domain. Default is conservative."""
        return 0.5

    @property
    def top_k(self) -> int:
        """Override per domain."""
        return 3

    # ---------------------------------------------------------------------------
    # Query generation
    # ---------------------------------------------------------------------------

    def _init_query_generator(self):
        """
        Builds the query generation chain.
        This chain translates a raw user query into a StructuredQuery
        using domain context, index schema, and system instructions.
        """
        self._query_parser = JsonOutputParser(pydantic_object=StructuredQuery)

        self._query_prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "You are a query decomposition assistant for a {domain} knowledge base.\n\n"
                    "Index schema — these are the ONLY metadata fields available for filtering:\n"
                    "{index_schema}\n\n"
                    "Domain retrieval instructions:\n"
                    "{system_instructions}\n\n"
                    "Your job:\n"
                    "1. Refine the user query into an optimal semantic search query\n"
                    "2. Extract any metadata filters the user implied in natural language\n"
                    "3. Decompose complex queries into sub-queries if needed\n"
                    "4. Only use metadata fields that exist in the index schema above\n\n"
                    "{format_instructions}",
                ),
                ("human", "{user_query}"),
            ]
        ).partial(format_instructions=self._query_parser.get_format_instructions())

        self._query_chain = self._query_prompt | self.llm | self._query_parser

    def _generate_structured_query(
        self,
        user_query: str,
        metadata_hints: dict,
    ) -> dict:
        """
        Runs the query generation chain.
        metadata_hints come from the intent classifier — pre-extracted
        signals like dates, chapter numbers, version strings.
        """
        enriched_query = user_query
        if metadata_hints:
            hints_str = ", ".join(f"{k}={v}" for k, v in metadata_hints.items())
            enriched_query = f"{user_query} [context hints: {hints_str}]"

        result = self._query_chain.invoke(
            {
                "domain": self.domain,
                "index_schema": self.index_schema,
                "system_instructions": self.system_instructions,
                "user_query": enriched_query,
            }
        )

        logger.info(
            "Structured query generated | source=%s reasoning=%s",
            self.source_name,
            result.get("reasoning", ""),
        )
        return result

    # ---------------------------------------------------------------------------
    # Retrieval
    # ---------------------------------------------------------------------------

    def _query_collection(
        self,
        query: str,
        metadata_filters: dict,
    ) -> list[dict]:
        """
        Hits the vector store with a single query and optional metadata filters.
        Embeds the query explicitly via the vector store rather than passing
        raw text to ChromaDB — avoids dimension mismatch from default embeddings.
        """
        try:
            # embed query explicitly using the same model used at index time
            query_embedding = self.vector_store._embedding_function.embed_query(query)  # type: ignore

            results = self.vector_store._collection.query(
                query_embeddings=[query_embedding],  # ← pre-embedded, not query_texts
                n_results=self.top_k,
                where=metadata_filters if metadata_filters else None,
                include=["documents", "distances", "metadatas"],
            )
        except Exception as e:
            logger.warning(
                "Filtered query failed for source '%s' (filters=%s): %s. "
                "Falling back to unfiltered.",
                self.source_name,
                metadata_filters,
                e,
            )
            query_embedding = self.vector_store._embedding_function.embed_query(query)  # type: ignore
            results = self.vector_store._collection.query(
                query_embeddings=[query_embedding],
                n_results=self.top_k,
                include=["documents", "distances", "metadatas"],
            )

        chunks = []
        for doc, dist, meta in zip(
            results["documents"][0],  # type: ignore
            results["distances"][0],  # type: ignore
            results["metadatas"][0],  # type: ignore
        ):
            if dist < self.similarity_threshold:
                chunks.append(
                    {
                        "content": doc,
                        "distance": dist,
                        "metadata": meta,
                        "query_used": query,
                        "source_name": self.source_name,
                        "domain": self.domain,
                    }
                )
        return chunks

    def _deduplicate(self, chunks: list[dict]) -> list[dict]:
        """
        Removes duplicate chunks by content.
        When multiple sub-queries retrieve the same chunk, keep
        the one with the lowest distance score (most similar).
        """
        seen: dict[str, dict] = {}
        for chunk in chunks:
            content = chunk["content"]
            if content not in seen or chunk["distance"] < seen[content]["distance"]:
                seen[content] = chunk
        return sorted(seen.values(), key=lambda x: x["distance"])

    # ---------------------------------------------------------------------------
    # Public interface
    # ---------------------------------------------------------------------------

    def retrieve(
        self,
        user_query: str,
        metadata_hints: dict | None = None,
    ) -> list[dict]:
        """
        Full instructed retrieval pipeline:
        1. Generate structured query from user query + system specs
        2. Run semantic query + all sub-queries against vector store
        3. Apply metadata filters derived from natural language
        4. Deduplicate and rank results by similarity score

        Parameters
        ----------
        user_query      : raw user question
        metadata_hints  : pre-extracted hints from intent classifier
                          e.g. {"version": "2024-Q1", "chapter": 3}
        """
        hints = metadata_hints or {}
        structured = self._generate_structured_query(user_query, hints)

        # build full query list — primary + sub-queries
        all_queries = [structured["semantic_query"]]
        all_queries.extend(structured.get("sub_queries", []))

        all_chunks = []
        for query in all_queries:
            chunks = self._query_collection(
                query=query,
                metadata_filters=structured.get("metadata_filters", {}),
            )
            all_chunks.extend(chunks)
            logger.info(
                "Query '%s' → %d chunks (source=%s)",
                query[:60],
                len(chunks),
                self.source_name,
            )

        unique_chunks = self._deduplicate(all_chunks)

        logger.info(
            "Retrieval complete | source=%s total=%d unique=%d",
            self.source_name,
            len(all_chunks),
            len(unique_chunks),
        )
        return unique_chunks
