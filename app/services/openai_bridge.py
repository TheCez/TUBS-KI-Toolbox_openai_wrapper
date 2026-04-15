from __future__ import annotations

import json
import uuid
from typing import Any, Iterable, List, Optional, Sequence

from app.models.openai import Message, ToolCall, ToolCallFunction
from app.models.responses import (
    ResponseCreateRequest,
    ResponseFunctionCall,
    ResponseFunctionCallOutput,
    ResponseFunctionToolChoice,
    ResponseInputImage,
    ResponseInputMessage,
    ResponseInputText,
    ResponseReasoningConfig,
    ResponseShorthandInputMessage,
)
from app.models.tubs import TubsChatRequest
from app.services.model_map import resolve_model
from app.services.prompt import (
    build_tool_instructions,
    extract_reasoning,
    has_tool_xml_start,
    parse_tool_calls_xml,
)
from app.services.translation import compile_messages_to_prompt, get_images_from_messages


def _content_item_to_message_part(item: Any) -> Optional[dict]:
    if isinstance(item, ResponseInputText):
        return {"type": "text", "text": item.text}
    if isinstance(item, ResponseInputImage):
        return {"type": "image_url", "image_url": {"url": item.image_url, "detail": item.detail}}
    return None


def _response_input_to_messages(input_value: str | List[Any]) -> List[Message]:
    if isinstance(input_value, str):
        return [Message(role="user", content=input_value)]

    normalized_messages: List[Message] = []

    for item in input_value:
        if isinstance(item, ResponseInputMessage):
            if isinstance(item.content, str) or item.content is None:
                content = item.content
            else:
                content = [part for part in (_content_item_to_message_part(part) for part in item.content) if part] or None
            normalized_messages.append(Message(role=item.role, content=content))
            continue

        if isinstance(item, ResponseShorthandInputMessage):
            if isinstance(item.content, str):
                content = item.content
            else:
                content = [part for part in (_content_item_to_message_part(part) for part in item.content) if part] or None
            normalized_messages.append(Message(role=item.role or "user", content=content))
            continue

        if isinstance(item, ResponseFunctionCall):
            normalized_messages.append(
                Message(
                    role="assistant",
                    content=None,
                    tool_calls=[
                        ToolCall(
                            id=item.call_id,
                            type="function",
                            function=ToolCallFunction(name=item.name, arguments=item.arguments),
                        )
                    ],
                )
            )
            continue

        if isinstance(item, ResponseFunctionCallOutput):
            output = item.output if isinstance(item.output, str) else json.dumps(item.output)
            normalized_messages.append(Message(role="tool", content=output, tool_call_id=item.call_id))
            continue

        if isinstance(item, ResponseInputText):
            normalized_messages.append(Message(role="user", content=item.text))
            continue

        if isinstance(item, ResponseInputImage):
            normalized_messages.append(
                Message(
                    role="user",
                    content=[{"type": "image_url", "image_url": {"url": item.image_url, "detail": item.detail}}],
                )
            )

    return normalized_messages


def build_custom_instructions(
    messages: Sequence[Message],
    response_format: Optional[dict[str, Any]] = None,
    tools: Optional[Iterable[Any]] = None,
    instructions: Optional[str] = None,
    reasoning: Optional[ResponseReasoningConfig] = None,
    max_output_tokens: Optional[int] = None,
    tool_choice: Optional[str | dict[str, Any] | ResponseFunctionToolChoice] = None,
) -> Optional[str]:
    blocks: List[str] = []

    if instructions:
        blocks.append(instructions.strip())

    system_messages = [
        msg.content.strip()
        for msg in messages
        if msg.role.lower() in ("system", "developer") and isinstance(msg.content, str) and msg.content.strip()
    ]
    if system_messages:
        blocks.append("\n".join(system_messages))

    if response_format:
        format_type = response_format.get("type")
        if format_type == "json_object":
            blocks.append("You must output your response as a valid JSON object.")
        elif format_type == "json_schema":
            schema = response_format.get("json_schema", {})
            blocks.append(
                "You must strictly adhere to the following JSON schema for your output. "
                f"Output only valid JSON.\nSchema: {json.dumps(schema)}"
            )

    if tools:
        blocks.append(build_tool_instructions(list(tools)))
        if isinstance(tool_choice, ResponseFunctionToolChoice):
            blocks.append(
                f"You must call the tool named '{tool_choice.name}' and stop immediately after emitting the tool call."
            )
        elif isinstance(tool_choice, dict) and tool_choice.get("type") == "function":
            function_name = tool_choice.get("function", {}).get("name")
            if function_name:
                blocks.append(
                    f"You must call the tool named '{function_name}' and stop immediately after emitting the tool call."
                )
        elif tool_choice == "required":
            blocks.append("You must call at least one tool and stop immediately after emitting the tool call.")
        elif tool_choice == "none":
            blocks.append("Do not call any tools in this response.")

    if reasoning and reasoning.effort and reasoning.effort != "none":
        effort_instructions = {
            "low": "Think briefly before answering, but do not over-compress the final response.",
            "medium": "Reason carefully before answering and provide enough detail to fully address the request.",
            "high": "Reason carefully and thoroughly before answering. Prefer correctness and completeness over brevity.",
            "xhigh": "Reason very carefully and comprehensively before answering. Provide a detailed and well-supported final response.",
        }
        blocks.append(effort_instructions[reasoning.effort])

    if max_output_tokens and max_output_tokens > 0:
        blocks.append(
            f"Keep the final answer within roughly {max_output_tokens} tokens while still fully answering the request."
        )

    combined = "\n\n".join(block for block in blocks if block and block.strip()).strip()
    return combined or None


def build_tubs_payload_from_messages(
    *,
    model: Any,
    messages: Sequence[Message],
    instructions: Optional[str] = None,
    response_format: Optional[dict[str, Any]] = None,
    tools: Optional[Iterable[Any]] = None,
    reasoning: Optional[ResponseReasoningConfig] = None,
    max_output_tokens: Optional[int] = None,
    tool_choice: Optional[str | dict[str, Any] | ResponseFunctionToolChoice] = None,
) -> tuple[dict[str, Any], list[tuple[str, bytes, str]], str]:
    payload = TubsChatRequest(
        thread=None,
        prompt=compile_messages_to_prompt(list(messages)),
        model=resolve_model(model),
        customInstructions=build_custom_instructions(
            messages=messages,
            response_format=response_format,
            tools=tools,
            instructions=instructions,
            reasoning=reasoning,
            max_output_tokens=max_output_tokens,
            tool_choice=tool_choice,
        ),
    ).model_dump(exclude_none=True)

    return payload, get_images_from_messages(list(messages)), str(model.value) if hasattr(model, "value") else str(model)


def build_tubs_payload_from_response_request(
    body: ResponseCreateRequest,
) -> tuple[dict[str, Any], list[tuple[str, bytes, str]], str]:
    messages = _response_input_to_messages(body.input)
    response_format = body.text.format if body.text and body.text.format else None
    return build_tubs_payload_from_messages(
        model=body.model,
        messages=messages,
        instructions=body.instructions,
        response_format=response_format,
        tools=body.tools,
        reasoning=body.reasoning,
        max_output_tokens=body.max_output_tokens,
        tool_choice=body.tool_choice,
    )


def parse_assistant_response(response_text: str) -> tuple[str, Optional[str], Optional[List[ToolCall]], str]:
    finish_reason = "stop"
    tool_calls: Optional[List[ToolCall]] = None

    response_text, reasoning = extract_reasoning(response_text)

    if has_tool_xml_start(response_text):
        parsed = parse_tool_calls_xml(response_text)
        if parsed:
            tool_calls = [
                ToolCall(
                    id=f"call_{uuid.uuid4().hex}",
                    type="function",
                    function=ToolCallFunction(name=tc["name"], arguments=tc["arguments"]),
                )
                for tc in parsed
            ]
            response_text = ""
            finish_reason = "tool_calls"

    return response_text, reasoning, tool_calls, finish_reason
