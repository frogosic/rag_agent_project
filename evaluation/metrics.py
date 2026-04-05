# evaluation/metrics.py
import logging
import time
from dataclasses import dataclass, field

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

from config import GPT_MODEL_ID, EVALUATION_SAMPLE_SIZE

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Evaluation result models
# ---------------------------------------------------------------------------


class RetrievalEvalResult(BaseModel):
    """Scores for a single retrieval result."""

    query: str = Field(description="The original query")
    domain: str = Field(description="Domain evaluated — qa | fiction | hybrid")
    chunk_count: int = Field(description="Number of chunks retrieved")
    avg_distance: float | None = Field(description="Average similarity distance")
    min_distance: float | None = Field(description="Best similarity distance")
    has_results: bool = Field(description="Whether any chunks were retrieved")
    relevance_score: float = Field(
        description="LLM-judged relevance 0.0-1.0 of retrieved chunks to query"
    )
    relevance_reasoning: str = Field(
        description="One sentence explaining the relevance score"
    )


class AnswerEvalResult(BaseModel):
    """Scores for a single generated answer."""

    query: str = Field(description="The original query")
    domain: str = Field(description="Domain evaluated")
    faithfulness_score: float = Field(
        description=(
            "0.0-1.0 — does the answer stay grounded in the retrieved context? "
            "1.0 = every claim is supported by context, 0.0 = hallucinated"
        )
    )
    faithfulness_reasoning: str = Field(
        description="One sentence explaining the faithfulness score"
    )
    citation_present: bool = Field(
        description="Whether the response includes source citations"
    )
    answered: bool = Field(
        description="Whether the question was actually answered vs deflected"
    )


@dataclass
class EvaluationReport:
    """
    Aggregate evaluation report across a set of queries.
    Produced by running the full evaluation suite.
    """

    total_queries: int = 0
    qa_queries: int = 0
    fiction_queries: int = 0
    hybrid_queries: int = 0

    # retrieval metrics
    avg_retrieval_relevance: float = 0.0
    qa_retrieval_relevance: float = 0.0
    fiction_retrieval_relevance: float = 0.0
    retrieval_success_rate: float = 0.0

    # answer metrics
    avg_faithfulness: float = 0.0
    qa_faithfulness: float = 0.0
    fiction_faithfulness: float = 0.0
    answer_rate: float = 0.0
    citation_rate: float = 0.0

    # performance
    avg_duration_ms: float = 0.0

    # raw results for inspection
    retrieval_results: list[RetrievalEvalResult] = field(default_factory=list)
    answer_results: list[AnswerEvalResult] = field(default_factory=list)

    def summary(self) -> dict:
        """Returns a clean summary dict suitable for logging or API response."""
        return {
            "total_queries": self.total_queries,
            "retrieval": {
                "avg_relevance": round(self.avg_retrieval_relevance, 3),
                "qa_relevance": round(self.qa_retrieval_relevance, 3),
                "fiction_relevance": round(self.fiction_retrieval_relevance, 3),
                "success_rate": round(self.retrieval_success_rate, 3),
            },
            "answer": {
                "avg_faithfulness": round(self.avg_faithfulness, 3),
                "qa_faithfulness": round(self.qa_faithfulness, 3),
                "fiction_faithfulness": round(self.fiction_faithfulness, 3),
                "answer_rate": round(self.answer_rate, 3),
                "citation_rate": round(self.citation_rate, 3),
            },
            "performance": {
                "avg_duration_ms": round(self.avg_duration_ms, 1),
            },
        }


# ---------------------------------------------------------------------------
# LLM judge
# ---------------------------------------------------------------------------
# Uses a separate LLM call to evaluate retrieval relevance and answer
# faithfulness. This is the "LLM as judge" pattern — standard in RAG eval.
# Temperature 0 for deterministic, consistent scoring.

_judge_llm = ChatOpenAI(
    model=GPT_MODEL_ID,
    temperature=0.0,
    max_tokens=512,  # type: ignore
)


# ---------------------------------------------------------------------------
# Retrieval relevance judge
# ---------------------------------------------------------------------------


class _RelevanceScore(BaseModel):
    score: float = Field(description="Relevance score 0.0 to 1.0")
    reasoning: str = Field(description="One sentence explanation")


_relevance_parser = JsonOutputParser(pydantic_object=_RelevanceScore)

_relevance_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a retrieval quality judge. "
            "Given a user query and retrieved document chunks, score how relevant "
            "the chunks are to answering the query.\n\n"
            "Scoring guide:\n"
            "1.0 — chunks directly answer the query with specific relevant content\n"
            "0.7 — chunks are related and useful but not perfectly targeted\n"
            "0.4 — chunks are tangentially related, partial usefulness\n"
            "0.1 — chunks are off-topic or irrelevant\n"
            "0.0 — no chunks retrieved\n\n"
            "{format_instructions}",
        ),
        (
            "human",
            "Query: {query}\n\n"
            "Retrieved chunks:\n{chunks}\n\n"
            "Score the relevance of these chunks to the query.",
        ),
    ]
).partial(format_instructions=_relevance_parser.get_format_instructions())

_relevance_chain = _relevance_prompt | _judge_llm | _relevance_parser


# ---------------------------------------------------------------------------
# Answer faithfulness judge
# ---------------------------------------------------------------------------


class _FaithfulnessScore(BaseModel):
    score: float = Field(description="Faithfulness score 0.0 to 1.0")
    reasoning: str = Field(description="One sentence explanation")
    answered: bool = Field(description="Whether the question was actually answered")


_faithfulness_parser = JsonOutputParser(pydantic_object=_FaithfulnessScore)

_faithfulness_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are an answer faithfulness judge. "
            "Given a user query, retrieved context, and a generated answer, "
            "score how faithfully the answer is grounded in the context.\n\n"
            "Scoring guide:\n"
            "1.0 — every claim in the answer is directly supported by the context\n"
            "0.7 — most claims are supported, minor extrapolation\n"
            "0.4 — some claims unsupported or partially hallucinated\n"
            "0.1 — answer largely ignores the context\n"
            "0.0 — answer is completely hallucinated or context was empty\n\n"
            "Also set answered=true only if the question was substantively answered, "
            "not deflected with 'I don't know'.\n\n"
            "{format_instructions}",
        ),
        (
            "human",
            "Query: {query}\n\n"
            "Context:\n{context}\n\n"
            "Answer:\n{answer}\n\n"
            "Score the faithfulness of this answer to the context.",
        ),
    ]
).partial(format_instructions=_faithfulness_parser.get_format_instructions())

_faithfulness_chain = _faithfulness_prompt | _judge_llm | _faithfulness_parser


# ---------------------------------------------------------------------------
# Individual evaluators
# ---------------------------------------------------------------------------


def evaluate_retrieval(
    query: str,
    domain: str,
    chunks: list[dict],
    retrieval_scores: dict,
) -> RetrievalEvalResult:
    """
    Evaluates retrieval quality for a single query.
    Combines distance-based scores from the graph with LLM relevance judgment.
    """
    domain_scores = retrieval_scores.get(domain, {})

    if not chunks:
        return RetrievalEvalResult(
            query=query,
            domain=domain,
            chunk_count=0,
            avg_distance=None,
            min_distance=None,
            has_results=False,
            relevance_score=0.0,
            relevance_reasoning="No chunks retrieved.",
        )

    chunks_text = "\n\n---\n\n".join(
        f"[chunk {i + 1}]\n{c['content'][:300]}" for i, c in enumerate(chunks)
    )

    try:
        relevance = _relevance_chain.invoke(
            {
                "query": query,
                "chunks": chunks_text,
            }
        )
        relevance_score = relevance["score"]
        relevance_reasoning = relevance["reasoning"]
    except Exception as e:
        logger.error("Relevance scoring failed: %s", e)
        relevance_score = 0.0
        relevance_reasoning = f"Scoring failed: {e}"

    return RetrievalEvalResult(
        query=query,
        domain=domain,
        chunk_count=domain_scores.get("chunk_count", len(chunks)),
        avg_distance=domain_scores.get("avg_distance"),
        min_distance=domain_scores.get("min_distance"),
        has_results=True,
        relevance_score=relevance_score,
        relevance_reasoning=relevance_reasoning,
    )


def evaluate_answer(
    query: str,
    domain: str,
    context: str,
    answer: str,
    sources: list[str],
) -> AnswerEvalResult:
    """
    Evaluates answer quality for a single query.
    Judges faithfulness to retrieved context and whether the question was answered.
    """
    if not answer or not context:
        return AnswerEvalResult(
            query=query,
            domain=domain,
            faithfulness_score=0.0,
            faithfulness_reasoning="No answer or context available.",
            citation_present=False,
            answered=False,
        )

    try:
        faithfulness = _faithfulness_chain.invoke(
            {
                "query": query,
                "context": context[:2000],  # cap context to avoid token overflow
                "answer": answer,
            }
        )
        faithfulness_score = faithfulness["score"]
        faithfulness_reasoning = faithfulness["reasoning"]
        answered = faithfulness["answered"]
    except Exception as e:
        logger.error("Faithfulness scoring failed: %s", e)
        faithfulness_score = 0.0
        faithfulness_reasoning = f"Scoring failed: {e}"
        answered = False

    return AnswerEvalResult(
        query=query,
        domain=domain,
        faithfulness_score=faithfulness_score,
        faithfulness_reasoning=faithfulness_reasoning,
        citation_present=bool(sources),
        answered=answered,
    )


# ---------------------------------------------------------------------------
# Aggregate report builder
# ---------------------------------------------------------------------------


def build_report(
    retrieval_results: list[RetrievalEvalResult],
    answer_results: list[AnswerEvalResult],
    durations_ms: list[float],
) -> EvaluationReport:
    """
    Aggregates individual eval results into a full report.
    Called after running the evaluation suite.
    """
    report = EvaluationReport(
        total_queries=len(retrieval_results),
        retrieval_results=retrieval_results,
        answer_results=answer_results,
    )

    if not retrieval_results:
        return report

    # domain counts
    report.qa_queries = sum(1 for r in retrieval_results if r.domain == "qa")
    report.fiction_queries = sum(1 for r in retrieval_results if r.domain == "fiction")
    report.hybrid_queries = sum(1 for r in retrieval_results if r.domain == "hybrid")

    # retrieval metrics
    all_relevance = [r.relevance_score for r in retrieval_results]
    qa_relevance = [r.relevance_score for r in retrieval_results if r.domain == "qa"]
    fiction_relevance = [
        r.relevance_score for r in retrieval_results if r.domain == "fiction"
    ]

    report.avg_retrieval_relevance = _avg(all_relevance)
    report.qa_retrieval_relevance = _avg(qa_relevance)
    report.fiction_retrieval_relevance = _avg(fiction_relevance)
    report.retrieval_success_rate = sum(
        1 for r in retrieval_results if r.has_results
    ) / len(retrieval_results)

    # answer metrics
    if answer_results:
        all_faithfulness = [a.faithfulness_score for a in answer_results]
        qa_faithfulness = [
            a.faithfulness_score for a in answer_results if a.domain == "qa"
        ]
        fiction_faithfulness = [
            a.faithfulness_score for a in answer_results if a.domain == "fiction"
        ]

        report.avg_faithfulness = _avg(all_faithfulness)
        report.qa_faithfulness = _avg(qa_faithfulness)
        report.fiction_faithfulness = _avg(fiction_faithfulness)
        report.answer_rate = sum(1 for a in answer_results if a.answered) / len(
            answer_results
        )
        report.citation_rate = sum(
            1 for a in answer_results if a.citation_present
        ) / len(answer_results)

    # performance
    if durations_ms:
        report.avg_duration_ms = sum(durations_ms) / len(durations_ms)

    return report


def _avg(values: list[float]) -> float:
    return round(sum(values) / len(values), 4) if values else 0.0


# ---------------------------------------------------------------------------
# Evaluation suite
# ---------------------------------------------------------------------------
# Default test queries covering both domains and the overlap vocabulary.
# These are the queries that stress-test the router and retrieval quality.

DEFAULT_QA_QUERIES = [
    "What are the test coverage requirements?",
    "How is bug severity defined?",
    "What are the release criteria?",
    "What is the policy on regression testing?",
    "What happens when a test environment fails?",  # overlap vocab
]

DEFAULT_FICTION_QUERIES = [
    "Who is Jason Dessen?",
    "What is the box in Dark Matter?",
    "What happens when Jason first enters the box?",
    "How does the parallel universe work in the novel?",
    "What happens when the variables change?",  # overlap vocab
]

DEFAULT_HYBRID_QUERIES = [
    "How does iteration help solve problems?",  # genuine overlap
    "What do you do when the environment fails?",  # genuine overlap
    "Compare how failures are handled in QA vs Dark Matter",
]

ALL_EVAL_QUERIES = (
    [{"query": q, "domain": "qa"} for q in DEFAULT_QA_QUERIES]
    + [{"query": q, "domain": "fiction"} for q in DEFAULT_FICTION_QUERIES]
    + [{"query": q, "domain": "hybrid"} for q in DEFAULT_HYBRID_QUERIES]
)


async def run_evaluation_suite(agent_graph, collection_manager) -> EvaluationReport:
    """
    Runs the full evaluation suite against the agent graph.
    Returns an EvaluationReport with retrieval and answer quality metrics.

    Parameters
    ----------
    agent_graph         : compiled LangGraph graph from agent/graph.py
    collection_manager  : shared CollectionManager instance
    """
    from langchain_core.messages import HumanMessage
    from agent.state import AgentState

    queries = ALL_EVAL_QUERIES[:EVALUATION_SAMPLE_SIZE]
    logger.info("Running evaluation suite | %d queries", len(queries))

    retrieval_results = []
    answer_results = []
    durations_ms = []

    for item in queries:
        query = item["query"]
        expected_domain = item["domain"]

        logger.info("Evaluating query: %s (expected=%s)", query, expected_domain)

        start = time.time()
        try:
            initial_state: AgentState = {
                "messages": [HumanMessage(content=query)],
                "user_query": query,
                "history": [],
                "intent": "",
                "intent_confidence": 0.0,
                "metadata_hints": {},
                "qa_chunks": [],
                "fiction_chunks": [],
                "merged_chunks": [],
                "retrieval_scores": {},
                "raw_response": "",
                "cited_response": "",
                "sources": [],
                "domain_used": "",
            }

            final_state = await agent_graph.ainvoke(initial_state)
            duration_ms = round((time.time() - start) * 1000, 2)
            durations_ms.append(duration_ms)

            # determine which chunks were used
            actual_domain = final_state.get("intent", expected_domain)
            if actual_domain == "qa":
                chunks = final_state.get("qa_chunks", [])
            elif actual_domain == "fiction":
                chunks = final_state.get("fiction_chunks", [])
            else:
                chunks = final_state.get("merged_chunks", [])

            # build context string for faithfulness eval
            context = "\n\n".join(c["content"] for c in chunks)

            # evaluate retrieval
            ret_result = evaluate_retrieval(
                query=query,
                domain=actual_domain,
                chunks=chunks,
                retrieval_scores=final_state.get("retrieval_scores", {}),
            )
            retrieval_results.append(ret_result)

            # evaluate answer
            ans_result = evaluate_answer(
                query=query,
                domain=actual_domain,
                context=context,
                answer=final_state.get("cited_response", ""),
                sources=final_state.get("sources", []),
            )
            answer_results.append(ans_result)

            logger.info(
                "Query done | domain=%s relevance=%.2f faithfulness=%.2f duration=%.0fms",
                actual_domain,
                ret_result.relevance_score,
                ans_result.faithfulness_score,
                duration_ms,
            )

        except Exception as e:
            logger.error("Evaluation failed for query '%s': %s", query, e)
            durations_ms.append(0.0)

    report = build_report(retrieval_results, answer_results, durations_ms)
    logger.info("Evaluation complete | summary=%s", report.summary())
    return report
