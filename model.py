import logging
import os

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage
from pydantic import BaseModel, Field

from config import GPT_MODEL_ID, EMBEDDING_MODEL, TEMPERATURE, MAX_TOKENS
from sources import DocumentSource
from ingestion.collection_manager import CollectionManager
from agent.router import RouterRetriever

logger = logging.getLogger(__name__)


class AIResponse(BaseModel):
    summary: str = Field(description="Summary of the user's message")
    response: str = Field(description="Suggested response to the user")


class AIModel:
    def __init__(self, sources: list[DocumentSource]):
        self.sources = sources
        self._init_llm()
        self._init_rag()
        self._init_chain()

    def _init_llm(self):
        self.llm = ChatOpenAI(
            model=GPT_MODEL_ID,
            temperature=TEMPERATURE,
            max_completion_tokens=MAX_TOKENS,
        )

    def _init_rag(self):
        embeddings = OpenAIEmbeddings(
            model=EMBEDDING_MODEL,
            api_key=os.getenv("OPENAI_API_KEY"),  # type: ignore
        )
        self.collection_manager = CollectionManager(embeddings)

        # load all sources on startup
        for source in self.sources:
            self.collection_manager.load(source)

        self.router = RouterRetriever(
            sources=self.sources,
            collection_manager=self.collection_manager,
            llm=self.llm,
        )

    def _init_chain(self):
        self.parser = JsonOutputParser(pydantic_object=AIResponse)
        self.prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "{system_prompt}\n\n"
                    "Use the following context to answer:\n{context}\n\n"
                    "Respond ONLY with valid JSON.\n{format_instructions}",
                ),
                MessagesPlaceholder(variable_name="history"),
                ("human", "{user_prompt}"),
            ]
        ).partial(format_instructions=self.parser.get_format_instructions())

        self.chain = self.prompt | self.llm | self.parser

    def _format_context(self, user_prompt: str) -> str:
        docs = self.router.retrieve(user_prompt)
        if not docs:
            return ""
        return "\n\n".join(
            f"[{doc.metadata.get('source_name', 'unknown')}] {doc.page_content}"
            for doc in docs
        )

    @staticmethod
    def _build_message_history(history: list) -> list:
        messages = []
        for turn in history:
            messages.append(HumanMessage(content=turn["user"]))
            messages.append(AIMessage(content=turn["assistant"]))
        return messages

    def get_response(self, system_prompt: str, user_prompt: str, history: list) -> dict:
        context = self._format_context(user_prompt)
        message_history = self._build_message_history(history)

        result = self.chain.invoke(
            {
                "system_prompt": system_prompt,
                "user_prompt": user_prompt,
                "context": context,
                "history": message_history,
            }
        )

        logger.info("Raw LLM output: %s", result)
        return result
