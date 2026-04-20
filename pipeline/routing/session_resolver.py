import json
from dataclasses import dataclass

from pipeline.config_loader import ConfigLoader
from pipeline.llm import LLMClient, get_llm_client


@dataclass
class SessionContext:
    user_id: str
    user_role: str
    database: str
    fallback_databases: list[str]
    tone: str
    reasoning: str


ROUTING_PROMPT = """You are routing a user to the correct enterprise knowledge base.

User profile:
{profile}

Available databases:
{databases}

Rules:
- database is the single most relevant DB for this user's role and department
- fallback_databases are additional DBs to try if the primary returns poor results (may be empty)
- tone matches the communication style appropriate for this user
- only include databases from the list provided"""


ROUTING_TOOL = {
    "name": "route_session",
    "description": "Return the routing decision for this user session.",
    "input_schema": {
        "type": "object",
        "properties": {
            "database": {
                "type": "string",
                "description": "The single most relevant DB for this user.",
            },
            "fallback_databases": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Additional DBs to try if the primary returns poor results.",
            },
            "tone": {
                "type": "string",
                "enum": ["technical", "general", "hr", "support"],
            },
            "reasoning": {
                "type": "string",
                "description": "One sentence explaining the routing choice.",
            },
        },
        "required": ["database", "fallback_databases", "tone", "reasoning"],
    },
}


class SessionResolver:
    def __init__(
        self,
        config_path: str = "config",
        llm: LLMClient | None = None,
    ):
        self.loader = ConfigLoader(config_path)
        self.llm = llm or get_llm_client()
        self._tone_instructions = self.loader.tone_instructions()
        self._cache: dict[str, SessionContext] = {}

    def resolve(
        self, user_id: str, user_role: str, access_tier: str = "internal"
    ) -> SessionContext:
        cache_key = f"{user_id}:{user_role}:{access_tier}"
        if cache_key in self._cache:
            return self._cache[cache_key]

        context: SessionContext = self._resolve(user_id, user_role, access_tier)
        self._cache[cache_key] = context
        return context

    def _resolve(
        self, user_id: str, user_role: str, access_tier: str
    ) -> SessionContext:
        db_descriptions = self.loader.db_descriptions(access_tier)
        allowed_dbs = set(db_descriptions.keys())

        prompt = ROUTING_PROMPT.format(
            profile=json.dumps({"user_id": user_id, "user_role": user_role}, indent=2),
            databases=json.dumps(db_descriptions, indent=2),
        )

        result = self.llm.complete_with_tool(
            messages=[{"role": "user", "content": prompt}],
            tool=ROUTING_TOOL,  # type: ignore
        )

        primary = result["database"]
        if primary not in allowed_dbs:
            raise ValueError(
                f"Resolver returned unknown database '{primary}'. "
                f"Allowed: {sorted(allowed_dbs)}"
            )

        fallbacks = [db for db in result["fallback_databases"] if db in allowed_dbs]

        return SessionContext(
            user_id=user_id,
            user_role=user_role,
            database=primary,
            fallback_databases=fallbacks,
            tone=result["tone"],
            reasoning=result["reasoning"],
        )

    def tone_instruction(self, tone: str) -> str:
        return self._tone_instructions.get(tone, self._tone_instructions["general"])
