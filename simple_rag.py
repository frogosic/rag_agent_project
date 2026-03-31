import logging
import warnings

warnings.filterwarnings("ignore")

from dotenv import load_dotenv

load_dotenv()

from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnablePassthrough
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.tools import tool

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# 1. Documents
# ---------------------------------------------------------------------------
docs = [
    Document(
        page_content="The Matrix was directed by the Wachowskis and released in 1999.",
        metadata={"source": "movies.txt", "chunk_id": 1},
    ),
    Document(
        page_content="Inception was directed by Christopher Nolan and released in 2010.",
        metadata={"source": "movies.txt", "chunk_id": 2},
    ),
    Document(
        page_content="The Lord of the Rings trilogy was directed by Peter Jackson.",
        metadata={"source": "movies.txt", "chunk_id": 3},
    ),
    Document(
        page_content="Python is a high-level programming language known for readability.",
        metadata={"source": "tech.txt", "chunk_id": 4},
    ),
]

# ---------------------------------------------------------------------------
# 2. Embeddings + vector store + retriever
# ---------------------------------------------------------------------------
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
vector_store = FAISS.from_documents(docs, embeddings)
logger.info("Vector store built with %d documents", len(docs))

retriever = vector_store.as_retriever(search_kwargs={"k": 2})


# ---------------------------------------------------------------------------
# 3. Tools
# ---------------------------------------------------------------------------
# @tool turns a plain function into a LangChain tool. The docstring is not
# optional — the LLM reads it to decide when and how to use the tool.


@tool
def search_knowledge_base(query: str) -> str:
    """Search the knowledge base for information about movies and programming languages.
    Use this when the user asks about directors, release years, or technical topics."""
    retrieved_docs = retriever.invoke(query)
    if not retrieved_docs:
        return "No relevant information found in the knowledge base."
    return "\n\n".join(
        f"[source: {doc.metadata['source']}] {doc.page_content}"
        for doc in retrieved_docs
    )


@tool
def summarize_history(placeholder: str = "") -> str:
    """Summarize what has been discussed so far in the conversation.
    Use this when the user asks what was covered or wants a recap."""
    if not history:
        return "No conversation history yet."
    turns = []
    for msg in history:
        if isinstance(msg, HumanMessage):
            turns.append(f"User asked: {msg.content}")
        elif isinstance(msg, AIMessage):
            turns.append(f"Assistant answered: {msg.content}")
    return "\n".join(turns)


tools = [search_knowledge_base, summarize_history]


# ---------------------------------------------------------------------------
# 4. LLM bound to tools
# ---------------------------------------------------------------------------
# bind_tools() tells the LLM what tools exist and what their schemas are.
# The LLM can now decide to call a tool instead of answering directly.
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.3)
llm_with_tools = llm.bind_tools(tools)


# ---------------------------------------------------------------------------
# 5. Prompt with memory
# ---------------------------------------------------------------------------
prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a helpful assistant with access to a knowledge base about movies "
            "and programming. Use the search_knowledge_base tool to answer questions. "
            "If the answer is not in the knowledge base, say you don't know. "
            "Use summarize_history if the user asks what was discussed.",
        ),
        MessagesPlaceholder(variable_name="history"),
        ("human", "{input}"),
    ]
)


# ---------------------------------------------------------------------------
# 6. Tool executor
# ---------------------------------------------------------------------------
# The LLM returns a tool_call when it wants to use a tool.
# This maps tool names to their actual functions and executes them.

tool_map = {t.name: t for t in tools}


def execute_tool_calls(response) -> str:
    """If the LLM called a tool, run it and return the result as a string."""
    if not response.tool_calls:
        return response.content

    results = []
    for call in response.tool_calls:
        tool_name = call["name"]
        tool_args = call["args"]
        logger.info("Tool called: %s | args: %s", tool_name, tool_args)
        result = tool_map[tool_name].invoke(tool_args)
        results.append(result)

    # feed tool results back into the LLM for a final natural language answer
    tool_results_text = "\n\n".join(results)
    followup = llm.invoke(
        f"Based on this information, answer the user's question naturally:\n\n"
        f"{tool_results_text}"
    )
    return followup.content


# ---------------------------------------------------------------------------
# 7. Agent loop
# ---------------------------------------------------------------------------
history = []


def chat(user_input: str) -> str:
    logger.info("User: %s", user_input)

    # build the full prompt with current history
    formatted = prompt.invoke({"input": user_input, "history": history})

    # LLM decides: answer directly or call a tool
    response = llm_with_tools.invoke(formatted)

    # execute tool if needed, get final answer
    answer = execute_tool_calls(response)

    # update memory
    history.append(HumanMessage(content=user_input))
    history.append(AIMessage(content=answer))

    logger.info("Agent: %s", answer)
    return answer


# ---------------------------------------------------------------------------
# 8. Run
# ---------------------------------------------------------------------------
chat("Who directed The Matrix?")
chat("What about Inception?")
chat("What is Python?")
chat("What is the capital of France?")
chat("Can you summarize what we discussed?")
