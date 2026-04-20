import os
import anthropic
from functools import lru_cache


class LLMClient:
    """
    Thin wrapper around the chat-completion API of whatever provider is active.
    """

    def __init__(self, model: str | None = None):
        self.model = model or os.getenv("LLM_MODEL", "claude-sonnet-4-20250514")
        self._client = anthropic.Anthropic()

    def complete(
        self,
        messages: list[dict],
        max_tokens: int = 1024,
        system: str | None = None,
    ) -> str:
        """Plain text completion. Returns the text of the first text block."""
        kwargs = {
            "model": self.model,
            "max_tokens": max_tokens,
            "messages": messages,
        }
        if system:
            kwargs["system"] = system
        response = self._client.messages.create(**kwargs)
        return response.content[0].text  # type: ignore

    def complete_with_tool(
        self,
        messages: list[anthropic.types.MessageParam],
        tool: anthropic.types.ToolParam,
        max_tokens: int = 300,
    ) -> dict:
        """Forces a tool call. Returns the tool's input dict (schema-validated)."""
        response = self._client.messages.create(
            model=self.model,
            max_tokens=max_tokens,
            tools=[tool],
            tool_choice={"type": "tool", "name": tool["name"]},
            messages=messages,
        )
        tool_use = next(b for b in response.content if b.type == "tool_use")
        return tool_use.input  # type: ignore


@lru_cache(maxsize=4)
def get_llm_client(model: str | None = None) -> LLMClient:
    """Cached factory so we don't create new Anthropic clients per call."""
    return LLMClient(model)
