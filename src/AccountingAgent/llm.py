from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from typing import Any

from config import (
    ANTHROPIC_API_KEY,
    ANTHROPIC_MODEL,
    LLM_PROVIDER,
    OPENAI_API_KEY,
    OPENAI_MODEL,
)

logger = logging.getLogger(__name__)


@dataclass
class ToolCall:
    id: str
    name: str
    arguments: dict[str, Any]


@dataclass
class LLMResponse:
    text: str | None = None
    tool_calls: list[ToolCall] = field(default_factory=list)


class LLMClient:
    """Unified interface for OpenAI and Anthropic tool-calling APIs."""

    def __init__(self, provider: str | None = None) -> None:
        self._provider = (provider or LLM_PROVIDER).lower()
        if self._provider == "openai":
            self._init_openai()
        elif self._provider == "anthropic":
            self._init_anthropic()
        else:
            raise ValueError(f"Unsupported LLM provider: {self._provider}")

    def _init_openai(self) -> None:
        from openai import AsyncOpenAI
        self._openai = AsyncOpenAI(api_key=OPENAI_API_KEY)
        self._model = OPENAI_MODEL

    def _init_anthropic(self) -> None:
        from anthropic import AsyncAnthropic
        self._anthropic = AsyncAnthropic(api_key=ANTHROPIC_API_KEY)
        self._model = ANTHROPIC_MODEL

    async def chat(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
    ) -> LLMResponse:
        if self._provider == "openai":
            return await self._chat_openai(messages, tools)
        return await self._chat_anthropic(messages, tools)

    # ── OpenAI ────────────────────────────────────────────────────────────

    async def _chat_openai(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None,
    ) -> LLMResponse:
        kwargs: dict[str, Any] = {
            "model": self._model,
            "messages": messages,
            "temperature": 0,
        }
        if tools:
            kwargs["tools"] = tools
            kwargs["tool_choice"] = "auto"

        resp = await self._openai.chat.completions.create(**kwargs)
        choice = resp.choices[0]
        message = choice.message

        result = LLMResponse(text=message.content)
        if message.tool_calls:
            for tc in message.tool_calls:
                result.tool_calls.append(
                    ToolCall(
                        id=tc.id,
                        name=tc.function.name,
                        arguments=json.loads(tc.function.arguments),
                    )
                )
        return result

    # ── Anthropic ─────────────────────────────────────────────────────────

    async def _chat_anthropic(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None,
    ) -> LLMResponse:
        system_text = ""
        conversation: list[dict[str, Any]] = []
        for msg in messages:
            if msg["role"] == "system":
                system_text = msg["content"] if isinstance(msg["content"], str) else ""
            else:
                conversation.append(msg)

        anthropic_tools = self._convert_tools_to_anthropic(tools) if tools else []

        kwargs: dict[str, Any] = {
            "model": self._model,
            "max_tokens": 8192,
            "temperature": 0,
            "messages": conversation,
        }
        if system_text:
            kwargs["system"] = system_text
        if anthropic_tools:
            kwargs["tools"] = anthropic_tools

        resp = await self._anthropic.messages.create(**kwargs)

        result = LLMResponse()
        text_parts: list[str] = []
        for block in resp.content:
            if block.type == "text":
                text_parts.append(block.text)
            elif block.type == "tool_use":
                result.tool_calls.append(
                    ToolCall(
                        id=block.id,
                        name=block.name,
                        arguments=block.input if isinstance(block.input, dict) else {},
                    )
                )
        if text_parts:
            result.text = "\n".join(text_parts)

        return result

    @staticmethod
    def _convert_tools_to_anthropic(
        openai_tools: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        anthropic_tools = []
        for tool in openai_tools:
            fn = tool.get("function", {})
            anthropic_tools.append({
                "name": fn["name"],
                "description": fn.get("description", ""),
                "input_schema": fn.get("parameters", {"type": "object", "properties": {}}),
            })
        return anthropic_tools

    def format_tool_result(
        self, tool_call_id: str, content: str
    ) -> dict[str, Any]:
        """Return the appropriate tool-result message for the current provider."""
        if self._provider == "openai":
            return {
                "role": "tool",
                "tool_call_id": tool_call_id,
                "content": content,
            }
        return {
            "role": "user",
            "content": [
                {
                    "type": "tool_result",
                    "tool_use_id": tool_call_id,
                    "content": content,
                }
            ],
        }

    def format_assistant_tool_calls(
        self, text: str | None, tool_calls: list[ToolCall]
    ) -> dict[str, Any]:
        """Return the assistant message that contains tool_calls so the
        conversation stays well-formed when we append tool results."""
        if self._provider == "openai":
            msg: dict[str, Any] = {
                "role": "assistant",
                "content": text or "",
                "tool_calls": [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {
                            "name": tc.name,
                            "arguments": json.dumps(tc.arguments),
                        },
                    }
                    for tc in tool_calls
                ],
            }
            return msg
        content: list[dict[str, Any]] = []
        if text:
            content.append({"type": "text", "text": text})
        for tc in tool_calls:
            content.append({
                "type": "tool_use",
                "id": tc.id,
                "name": tc.name,
                "input": tc.arguments,
            })
        return {"role": "assistant", "content": content}
