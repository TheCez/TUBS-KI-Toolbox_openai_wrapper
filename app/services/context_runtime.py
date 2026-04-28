from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any, Awaitable, Callable, Sequence

from app.models.anthropic import Message as AnthropicMessage
from app.models.openai import Message, ToolCall, ToolCallFunction
from app.services.anthropic_translation import compile_anthropic_messages_to_prompt, get_images_from_anthropic_messages
from app.services.conversation_state import build_prompt_with_compaction
from app.services.context_compaction import estimate_token_count
from app.services.context_store import context_store
from app.services.context_tools import (
    context_tool_instruction,
    execute_context_tool,
    is_context_tool,
    merge_tools,
    should_offer_context_tools,
)
from app.services.openai_bridge import (
    build_custom_instructions,
    build_tubs_payload_from_messages,
    effective_prompt_token_budget,
    parse_assistant_response,
)
from app.services.prompt import extract_reasoning
from app.services.tool_validation import validate_tool_calls
from app.models.tubs import TubsChatRequest
from app.services.model_map import resolve_model
from app.services.translation import compile_messages_to_prompt


SendRequest = Callable[[dict[str, Any], list[tuple[str, bytes, str]], str, bool], Awaitable[dict[str, Any]]]


def _context_loop_limit() -> int:
    return max(1, int(os.getenv("TUBS_CONTEXT_TOOL_LOOP_LIMIT", "4")))


def _required_context_retrievals() -> int:
    return max(1, int(os.getenv("TUBS_REQUIRED_CONTEXT_RETRIEVALS", "1")))


def _overflow_loop_message() -> str:
    return (
        "Wrapper note: this turn is using compacted bridge context. "
        "Retrieve additional wrapper context with `get_thread_state` and/or `search_context` before any final answer "
        "or external tool call. If you still lack enough context after one retrieval, call another wrapper context tool."
    )


@dataclass
class LocalContextResolution:
    thread_id: str | None
    effective_tools: list[Any]
    final_response: dict[str, Any]
    final_text: str
    reasoning: str | None
    tool_calls: list[ToolCall] | None
    finish_reason: str
    used_context_tools: bool


def _overflow_active_for_openai_messages(
    *,
    messages: Sequence[Message],
    thread_id: str | None,
    instructions: str | None,
    response_format: dict[str, Any] | None,
    tools: Sequence[Any] | None,
    reasoning: Any,
    max_output_tokens: int | None,
    tool_choice: str | dict[str, Any] | None,
) -> bool:
    custom_instructions = build_custom_instructions(
        messages=messages,
        response_format=response_format,
        tools=tools,
        instructions=instructions,
        reasoning=reasoning,
        max_output_tokens=max_output_tokens,
        tool_choice=tool_choice,
    )
    prompt_text = compile_messages_to_prompt(messages).strip()
    return estimate_token_count(prompt_text) > effective_prompt_token_budget(thread_id, custom_instructions)


def _overflow_active_for_anthropic_messages(
    *,
    messages: Sequence[AnthropicMessage],
    thread_id: str | None,
    system_instructions: str | None,
    tools: Sequence[Any] | None,
    max_output_tokens: int | None,
    tool_choice: Any,
    reasoning: Any,
) -> bool:
    custom_instructions = build_custom_instructions(
        messages=[],
        instructions=system_instructions,
        tools=tools,
        reasoning=reasoning,
        max_output_tokens=max_output_tokens,
        tool_choice=tool_choice,
    )
    prompt_text = compile_anthropic_messages_to_prompt(messages).strip()
    return estimate_token_count(prompt_text) > effective_prompt_token_budget(thread_id, custom_instructions)


def _snapshot_summary(thread_id: str) -> str | None:
    snapshot = context_store().get_hot_snapshot(thread_id)
    if snapshot is None:
        return None
    lines = []
    if snapshot.current_objective:
        lines.append(f"Current objective: {snapshot.current_objective}")
    if snapshot.current_plan:
        lines.append("Current plan: " + " | ".join(snapshot.current_plan[:4]))
    if snapshot.unresolved_blockers:
        lines.append("Unresolved blockers: " + " | ".join(snapshot.unresolved_blockers[:3]))
    if snapshot.recent_decisions:
        lines.append("Recent decisions: " + " | ".join(snapshot.recent_decisions[:3]))
    if snapshot.latest_tool_failures:
        lines.append("Latest tool failures: " + " | ".join(snapshot.latest_tool_failures[:2]))
    if snapshot.recent_messages:
        lines.append("Recent raw turns: " + " | ".join(snapshot.recent_messages[-3:]))
    if not lines:
        return None
    return "Durable thread state summary:\n" + "\n".join(f"- {line}" for line in lines)


def augment_openai_messages_with_context(messages: Sequence[Message], thread_id: str) -> list[Message]:
    summary = _snapshot_summary(thread_id)
    if not summary:
        return list(messages)
    return [Message(role="developer", content=summary), *list(messages)]


def augment_anthropic_messages_with_context(messages: Sequence[AnthropicMessage], thread_id: str) -> list[AnthropicMessage]:
    summary = _snapshot_summary(thread_id)
    if not summary:
        return list(messages)
    return [AnthropicMessage(role="user", content=summary), *list(messages)]


def _split_context_tool_calls(tool_calls: list[ToolCall] | None) -> tuple[list[ToolCall], list[ToolCall]]:
    wrapper_calls: list[ToolCall] = []
    external_calls: list[ToolCall] = []
    for tool_call in tool_calls or []:
        if is_context_tool(tool_call.function.name):
            wrapper_calls.append(tool_call)
        else:
            external_calls.append(tool_call)
    return wrapper_calls, external_calls


async def resolve_openai_context_tools(
    *,
    model: Any,
    messages: Sequence[Message],
    thread_id: str | None,
    context_thread_id: str,
    bearer_token: str,
    instructions: str | None,
    response_format: dict[str, Any] | None,
    tools: Sequence[Any] | None,
    reasoning: Any,
    max_output_tokens: int | None,
    tool_choice: str | dict[str, Any] | None,
    send_request: SendRequest,
    require_context_retrieval: bool = False,
) -> LocalContextResolution:
    enable_context_tools = should_offer_context_tools(context_thread_id)
    effective_tools = merge_tools(tools, anthropic=False) if enable_context_tools else list(tools or [])
    working_messages = list(messages)
    current_thread_id = thread_id
    used_context_tools = False
    completed_wrapper_retrievals = 0
    last_response: dict[str, Any] | None = None
    last_final_text = ""
    last_reasoning = None
    last_tool_calls: list[ToolCall] | None = None
    last_finish_reason = "stop"
    overflow_mode = require_context_retrieval and enable_context_tools

    for _ in range(_context_loop_limit()):
        payload, images, _model_str = build_tubs_payload_from_messages(
            model=model,
            messages=working_messages,
            thread_id=current_thread_id,
            instructions="\n\n".join(
                part
                for part in [
                    instructions,
                    context_tool_instruction(overflow_mode=overflow_mode) if enable_context_tools else "",
                ]
                if part
            ) or None,
            response_format=response_format,
            tools=effective_tools,
            reasoning=reasoning,
            max_output_tokens=max_output_tokens,
            tool_choice=tool_choice,
        )
        response = await send_request(payload, images, bearer_token, False)
        thread = response.get("thread")
        if isinstance(thread, dict):
            current_thread_id = thread.get("id", current_thread_id)
        elif isinstance(thread, str) and thread.strip():
            current_thread_id = thread.strip()

        final_text, reasoning, tool_calls, finish_reason = parse_assistant_response(response.get("response", ""), tools=effective_tools)
        last_response = response
        last_final_text = final_text
        last_reasoning = reasoning
        last_tool_calls = tool_calls
        last_finish_reason = finish_reason
        wrapper_calls, external_calls = _split_context_tool_calls(tool_calls)
        if wrapper_calls:
            used_context_tools = True
            completed_wrapper_retrievals += len(wrapper_calls)
            working_messages.append(
                Message(
                    role="assistant",
                    content=None,
                    tool_calls=wrapper_calls,
                )
            )
            for call in wrapper_calls:
                result = execute_context_tool(call.function.name, call.function.arguments, context_thread_id)
                working_messages.append(Message(role="tool", content=result, tool_call_id=call.id))
            if external_calls:
                working_messages.append(
                    Message(
                        role="developer",
                        content="Wrapper note: context tools have been resolved. If you still need external tools, emit them in the next response only.",
                    )
                )
            continue

        if overflow_mode and completed_wrapper_retrievals < _required_context_retrievals():
            working_messages.append(
                Message(
                    role="developer",
                    content=_overflow_loop_message(),
                )
            )
            continue

        return LocalContextResolution(
            thread_id=current_thread_id,
            effective_tools=effective_tools,
            final_response=response,
            final_text=final_text,
            reasoning=reasoning,
            tool_calls=external_calls or tool_calls,
            finish_reason=finish_reason,
            used_context_tools=used_context_tools,
        )

    return LocalContextResolution(
        thread_id=current_thread_id,
        effective_tools=effective_tools,
        final_response=last_response or {"type": "done", "response": "", "promptTokens": 0, "responseTokens": 0, "totalTokens": 0},
        final_text=last_final_text,
        reasoning=last_reasoning,
        tool_calls=last_tool_calls,
        finish_reason=last_finish_reason,
        used_context_tools=used_context_tools,
    )


async def resolve_anthropic_context_tools(
    *,
    model: str,
    messages: Sequence[AnthropicMessage],
    thread_id: str | None,
    context_thread_id: str,
    bearer_token: str,
    system_instructions: str | None,
    tools: Sequence[Any] | None,
    max_output_tokens: int | None,
    tool_choice: Any,
    reasoning: Any,
    send_request: SendRequest,
    require_context_retrieval: bool = False,
) -> LocalContextResolution:
    enable_context_tools = should_offer_context_tools(context_thread_id)
    effective_tools = merge_tools(tools, anthropic=True) if enable_context_tools else list(tools or [])
    working_messages = list(messages)
    current_thread_id = thread_id
    used_context_tools = False
    completed_wrapper_retrievals = 0
    last_response: dict[str, Any] | None = None
    last_final_text = ""
    last_reasoning = None
    last_tool_calls: list[ToolCall] | None = None
    last_finish_reason = "stop"
    overflow_mode = require_context_retrieval and enable_context_tools

    for _ in range(_context_loop_limit()):
        prompt = compile_anthropic_messages_to_prompt(working_messages)
        custom_instructions = build_custom_instructions(
            messages=[],
            instructions="\n\n".join(
                part
                for part in [
                    system_instructions,
                    context_tool_instruction(overflow_mode=overflow_mode) if enable_context_tools else "",
                ]
                if part
            ) or None,
            tools=effective_tools,
            reasoning=reasoning,
            max_output_tokens=max_output_tokens,
            tool_choice=tool_choice,
        )
        payload = TubsChatRequest(
            thread=current_thread_id,
            prompt=build_prompt_with_compaction(
                working_messages,
                compile_prompt=compile_anthropic_messages_to_prompt,
                thread_id=current_thread_id,
                prompt_token_budget=effective_prompt_token_budget(current_thread_id, custom_instructions),
            ),
            model=resolve_model(model),
            customInstructions=custom_instructions,
        ).model_dump(exclude_none=True)
        response = await send_request(
            payload,
            get_images_from_anthropic_messages(working_messages),
            bearer_token,
            False,
        )
        thread = response.get("thread")
        if isinstance(thread, dict):
            current_thread_id = thread.get("id", current_thread_id)
        elif isinstance(thread, str) and thread.strip():
            current_thread_id = thread.strip()

        final_text, reasoning_text, tool_calls, finish_reason = parse_assistant_response(response.get("response", ""), tools=effective_tools)
        last_response = response
        last_final_text = final_text
        last_reasoning = reasoning_text
        last_tool_calls = tool_calls
        last_finish_reason = finish_reason
        wrapper_calls, external_calls = _split_context_tool_calls(tool_calls)
        if wrapper_calls:
            used_context_tools = True
            completed_wrapper_retrievals += len(wrapper_calls)
            working_messages.append(
                AnthropicMessage(
                    role="assistant",
                    content=[
                        {
                            "type": "tool_use",
                            "id": call.id,
                            "name": call.function.name,
                            "input": json.loads(call.function.arguments),
                        }
                        for call in wrapper_calls
                    ],
                )
            )
            for call in wrapper_calls:
                result = execute_context_tool(call.function.name, call.function.arguments, context_thread_id)
                working_messages.append(
                    AnthropicMessage(
                        role="user",
                        content=[{"type": "tool_result", "tool_use_id": call.id, "content": result}],
                    )
                )
            if external_calls:
                working_messages.append(
                    AnthropicMessage(
                        role="user",
                        content="Wrapper note: context tools have been resolved. Emit external tools only in the next response.",
                    )
                )
            continue

        if overflow_mode and completed_wrapper_retrievals < _required_context_retrievals():
            working_messages.append(
                AnthropicMessage(
                    role="user",
                    content=_overflow_loop_message(),
                )
            )
            continue

        return LocalContextResolution(
            thread_id=current_thread_id,
            effective_tools=effective_tools,
            final_response=response,
            final_text=final_text,
            reasoning=reasoning_text,
            tool_calls=external_calls or tool_calls,
            finish_reason=finish_reason,
            used_context_tools=used_context_tools,
        )

    return LocalContextResolution(
        thread_id=current_thread_id,
        effective_tools=effective_tools,
        final_response=last_response or {"type": "done", "response": "", "promptTokens": 0, "responseTokens": 0, "totalTokens": 0},
        final_text=last_final_text,
        reasoning=last_reasoning,
        tool_calls=last_tool_calls,
        finish_reason=last_finish_reason,
        used_context_tools=used_context_tools,
    )
