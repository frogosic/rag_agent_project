import logging

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pydantic import BaseModel, Field
from config import GPT_MODEL_ID, TEMPERATURE, MAX_TOKENS

logger = logging.getLogger(__name__)


class AIResponse(BaseModel):
    summary: str = Field(description="Summary of the user's message")
    response: str = Field(description="Suggested response to the user")


class AIModel:
    """Encapsulates the language model and its response parsing."""

    def __init__(self, policy_path: str):
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
        splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=40)
        chunks = splitter.split_documents(docs)
        embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
        self.retriever = FAISS.from_documents(chunks, embeddings).as_retriever(
            search_kwargs={"k": 2}
        )
        logger.info(
            "Loaded and indexed policy document with %d chunks from %s",
            len(chunks),
            policy_path,
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
        context_docs = self.retriever.invoke(user_prompt)
        context = "\n\n".join(doc.page_content for doc in context_docs)

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

        history.append({"user": user_prompt, "assistant": result.get("response", "")})
        logger.info("Raw LLM output: %s", result)
        return result
