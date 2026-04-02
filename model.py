import logging
import os

from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pydantic import BaseModel, Field
from chromadb import PersistentClient
from chromadb.utils.embedding_functions import OpenAIEmbeddingFunction
from config import GPT_MODEL_ID, EMBEDDING_MODEL, TEMPERATURE, MAX_TOKENS

logger = logging.getLogger(__name__)


class AIResponse(BaseModel):
    summary: str = Field(description="Summary of the user's message")
    response: str = Field(description="Suggested response to the user")


class AIModel:
    """Encapsulates the language model and its response parsing."""

    def __init__(
        self,
        policy_path: str,
        collection_name: str = "qa_policy",
        doc_metadata: dict | None = None,
    ):
        self.collection_name = collection_name
        self.doc_metadata = doc_metadata or {}
        self._init_llm()
        self._init_rag(policy_path)
        self._init_chain()

    def _init_llm(self):
        self.llm = ChatOpenAI(
            model=GPT_MODEL_ID,
            temperature=TEMPERATURE,
            max_completion_tokens=MAX_TOKENS,
        )

    def _init_rag(self, policy_path: str):
        loader = TextLoader(policy_path)
        docs = loader.load()
        splitter = RecursiveCharacterTextSplitter(chunk_size=512, chunk_overlap=80)
        chunks = splitter.split_documents(docs)

        embed_fn = OpenAIEmbeddingFunction(
            api_key=os.getenv("OPENAI_API_KEY"),
            model_name=EMBEDDING_MODEL,
        )
        client = PersistentClient(path="./chroma_data")
        self.collection = client.get_or_create_collection(
            name=self.collection_name,
            embedding_function=embed_fn,  # type: ignore
            metadata={"hnsw:space": "cosine"},
        )

        if self.collection.count() == 0:
            texts = [c.page_content for c in chunks]
            ids = [f"{self.collection_name}_chunk_{i}" for i in range(len(texts))]
            metadatas = [{"source": policy_path, **self.doc_metadata} for _ in texts]
            self.collection.add(documents=texts, ids=ids, metadatas=metadatas)  # type: ignore
            logger.info(
                "Indexed %d chunks from %s into ChromaDB", len(chunks), policy_path
            )
        else:
            logger.info(
                "ChromaDB collection already has %d chunks, skipping indexing",
                self.collection.count(),
            )

    def _init_chain(self):
        self.parser = JsonOutputParser(pydantic_object=AIResponse)
        self.template = PromptTemplate(
            template=(
                "{system_prompt}\n\n"
                "Use the following context to answer the user's question:\n{context}\n\n"
                "{history}"
                "You must respond ONLY with valid JSON. No explanation, no markdown, no code blocks.\n"
                "{format_prompt}\n"
                "User: {user_prompt}"
            ),
            input_variables=[
                "system_prompt",
                "context",
                "history",
                "format_prompt",
                "user_prompt",
            ],
        )
        self.chain = self.template | self.llm | self.parser

    def get_response(self, system_prompt: str, user_prompt: str, history: list) -> dict:
        """Generates a response from the AI model based on the provided prompts."""
        results = self.collection.query(
            query_texts=[user_prompt],
            n_results=2,
            include=["documents", "distances"],
        )

        docs_and_scores = zip(results["documents"][0], results["distances"][0])  # type: ignore
        relevant_docs = [doc for doc, dist in docs_and_scores if dist < 0.5]
        context = "\n\n".join(relevant_docs) if relevant_docs else ""

        history_text = ""
        if history:
            history_text = "Conversation so far:\n"
            history_text += "\n".join(
                f"User: {h['user']}\nAssistant: {h['assistant']}" for h in history
            )
            history_text += "\n\n"

        result = self.chain.invoke(
            {
                "system_prompt": system_prompt,
                "user_prompt": user_prompt,
                "context": context,
                "history": history_text,
                "format_prompt": self.parser.get_format_instructions(),
            }
        )

        logger.info("Raw LLM output: %s", result)
        return result
