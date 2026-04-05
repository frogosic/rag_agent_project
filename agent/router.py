import logging

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.documents import Document
from ingestion.collection_manager import CollectionManager
from sources import DocumentSource

logger = logging.getLogger(__name__)


class RouterRetriever:
    """
    Uses the LLM to pick the most relevant collection(s) for a query,
    then retrieves from those collections only.
    """

    def __init__(
        self,
        sources: list[DocumentSource],
        collection_manager: CollectionManager,
        llm: ChatOpenAI,
    ):
        self.sources = sources
        self.collection_manager = collection_manager
        self._init_router(llm)

    def _init_router(self, llm: ChatOpenAI):
        source_descriptions = "\n".join(
            f"- {s.name}: {s.description}" for s in self.sources
        )

        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "You are a routing assistant. Given a user query, return the name(s) "
                    "of the most relevant knowledge sources from this list:\n\n"
                    f"{source_descriptions}\n\n"
                    "Return ONLY the source name(s) as a comma-separated list. "
                    "No explanation, no punctuation, just the names.",
                ),
                ("human", "{query}"),
            ]
        )

        self.router_chain = prompt | llm | StrOutputParser()

    def retrieve(self, query: str) -> list[Document]:
        """Routes the query to the right collection(s) and returns documents."""
        raw = self.router_chain.invoke({"query": query})
        selected = [name.strip() for name in raw.split(",")]
        logger.info("Router selected collections: %s", selected)

        all_docs = []
        for name in selected:
            try:
                retriever = self.collection_manager.get_retriever(name)
                docs = retriever.invoke(query)
                all_docs.extend(docs)
                logger.info("Retrieved %d docs from '%s'", len(docs), name)
            except ValueError:
                logger.warning("Router selected unknown collection: %s", name)

        return all_docs
