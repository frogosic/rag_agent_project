from pipeline.config_loader import ConfigLoader
from pipeline.llm import LLMClient, get_llm_client
from pipeline.retrieval.hybrid import HybridRetriever, RetrievalResult
from pipeline.routing.session_resolver import SessionContext, SessionResolver


GENERATION_PROMPT = """You are an internal knowledge assistant.

Tone instruction: {tone_instruction}

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
        self.resolver = SessionResolver(config_path, llm=self.llm)
        self._retrievers: dict[str, HybridRetriever] = {}

    def query(
        self,
        query: str,
        user_id: str,
        user_role: str,
        top_k: int = 10,
    ) -> dict:
        session: SessionContext = self.resolver.resolve(user_id, user_role)
        results: list[RetrievalResult] = self._retrieve(query, session.database, top_k)

        if not results and session.fallback_databases:
            for db_name in session.fallback_databases:
                results = self._retrieve(query, db_name, top_k)
                if results:
                    break

        answer: str = self._generate(query, results, session)

        return {
            "answer": answer,
            "session": {
                "database": session.database,
                "tone": session.tone,
                "reasoning": session.reasoning,
            },
            "sources": [
                {
                    "chunk_id": r.chunk_id,
                    "score": round(r.score, 4),
                    "snippet": r.text[:200],
                }
                for r in results
            ],
        }

    def _retrieve(self, query: str, db_name: str, top_k: int) -> list[RetrievalResult]:
        if db_name not in self._retrievers:
            db_config = self.loader.get_db(db_name)
            self._retrievers[db_name] = HybridRetriever(db_config)
        return self._retrievers[db_name].retrieve(query, top_k=top_k)

    def _generate(
        self,
        query: str,
        results: list[RetrievalResult],
        session: SessionContext,
    ) -> str:
        if not results:
            return "No relevant documents found for your query."

        context = "\n\n---\n\n".join(f"[{r.chunk_id}]\n{r.text}" for r in results)

        prompt = GENERATION_PROMPT.format(
            tone_instruction=self.resolver.tone_instruction(session.tone),
            context=context,
            query=query,
        )

        return self.llm.complete(
            messages=[{"role": "user", "content": prompt}],
            max_tokens=1024,
        )
