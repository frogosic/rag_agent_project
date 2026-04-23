from pipeline.config_loader import ConfigLoader
from pipeline.llm import LLMClient, get_llm_client
from pipeline.retrieval.hybrid import HybridRetriever, RetrievalResult


GENERATION_PROMPT = """You are an internal knowledge assistant.

Answer the question using only the context provided below.
If the context does not contain enough information to answer, say so clearly.

Context:
{context}

Question: {query}"""


class QueryEngine:
    def __init__(
        self,
        config_path: str = "config",
        llm: LLMClient | None = None,
    ):
        self.loader = ConfigLoader(config_path)
        self.llm: LLMClient = llm or get_llm_client()
        self.retriever = HybridRetriever(self.loader.default_db())

    def query(
        self,
        query: str,
        top_k: int = 10,
        where: dict | None = None,
    ) -> dict:
        results: list[RetrievalResult] = self.retriever.retrieve(
            query, top_k=top_k, where=where
        )
        answer: str = self._generate(query, results)

        return {
            "answer": answer,
            "sources": [
                {
                    "chunk_id": r.chunk_id,
                    "score": round(r.score, 4),
                    "snippet": r.text[:200],
                }
                for r in results
            ],
        }

    def _generate(
        self,
        query: str,
        results: list[RetrievalResult],
    ) -> str:
        if not results:
            return "No relevant documents found for your query."

        context = "\n\n---\n\n".join(f"[{r.chunk_id}]\n{r.text}" for r in results)

        prompt = GENERATION_PROMPT.format(
            context=context,
            query=query,
        )

        return self.llm.complete(
            messages=[{"role": "user", "content": prompt}],
            max_tokens=1024,
        )
