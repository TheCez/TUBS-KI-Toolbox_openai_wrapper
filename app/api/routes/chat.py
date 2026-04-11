"""
OpenAI-compatible /v1/chat/completions endpoint.
Translates OpenAI requests into TU-BS KI-Toolbox API format.
"""

import os
import time
import json
import uuid
from fastapi import APIRouter, Depends, Request
from fastapi.responses import StreamingResponse
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials

from app.models.openai import (
    ChatCompletionRequest, ChatCompletionResponse,
    ChoiceNonStreaming, Usage, Message, ToolCall, ToolCallFunction,
)
from app.models.tubs import TubsChatRequest
from app.services.translation import compile_messages_to_prompt, get_images_from_messages
from app.services.tubs_client import async_send_tubs_request
from app.services.model_map import resolve_model
from app.services.prompt import (
    build_tool_instructions, truncate_at_stop, parse_tool_calls_xml,
    is_tool_xml_complete, has_tool_xml_start, strip_tool_xml, extract_reasoning,
)

router = APIRouter()
security = HTTPBearer()

STOP_TRUNCATION_ENABLED = os.getenv("ENABLE_STOP_TRUNCATION", "false").lower() == "true"


@router.post("/chat/completions", response_model=ChatCompletionResponse)
async def chat_completions(
    request: Request,
    body: ChatCompletionRequest,
    credentials: HTTPAuthorizationCredentials = Depends(security),
):
    token = credentials.credentials

    # Compile prompt from messages array
    prompt_string = compile_messages_to_prompt(body.messages)

    # Extract images if any
    images = get_images_from_messages(body.messages)

    # Build custom instructions from system/developer messages
    system_messages = [
        msg.content for msg in body.messages
        if msg.role.lower() in ("system", "developer") and isinstance(msg.content, str)
    ]
    custom_instructions = "\n".join(system_messages).strip()

    # Inject JSON schema instructions if requested
    if body.response_format:
        format_type = body.response_format.get("type")
        if format_type == "json_object":
            custom_instructions += "\nYou must output your response as a valid JSON object.\n"
        elif format_type == "json_schema":
            schema = body.response_format.get("json_schema", {})
            custom_instructions += (
                f"\nYou must strictly adhere to the following JSON schema for your output. "
                f"Output only valid JSON.\nSchema: {json.dumps(schema)}\n"
            )

    # Inject tool-calling instructions if tools are provided
    if body.tools:
        custom_instructions += "\n" + build_tool_instructions(body.tools)

    custom_instructions = custom_instructions.strip() or None

    # Resolve model (handles Anthropic aliases transparently)
    tubs_model = resolve_model(body.model)

    # Build TU-BS request payload
    tubs_payload = TubsChatRequest(
        thread=None,
        prompt=prompt_string,
        model=tubs_model,
        customInstructions=custom_instructions,
    ).model_dump(exclude_none=True)

    # Send to TU-BS backend
    response_or_stream = await async_send_tubs_request(
        payload=tubs_payload,
        images=images,
        bearer_token=token,
        stream=bool(body.stream),
    )

    req_id = f"chatcmpl-{uuid.uuid4().hex}"
    created_time = int(time.time())
    model_str = str(body.model.value) if hasattr(body.model, "value") else str(body.model)

    # --- Streaming path ---
    if body.stream:
        stop_sequences = None
        if STOP_TRUNCATION_ENABLED and body.stop:
            stop_sequences = body.stop if isinstance(body.stop, list) else [body.stop]

        async def event_generator():
            text_buffer = ""
            is_buffering_tool = False
            has_yielded_tool = False

            def _sse(payload: dict) -> str:
                return f"data: {json.dumps(payload)}\n\n"

            def _chunk(delta: dict, finish_reason=None) -> dict:
                return {
                    "id": req_id,
                    "object": "chat.completion.chunk",
                    "created": created_time,
                    "model": model_str,
                    "choices": [{"index": 0, "delta": delta, "finish_reason": finish_reason}],
                }

            try:
                async for chunk in response_or_stream:
                    chunk_type = chunk.get("type")

                    if chunk_type == "chunk":
                        text_buffer += chunk.get("content", "")

                        if has_tool_xml_start(text_buffer):
                            is_buffering_tool = True

                        if not is_buffering_tool:
                            # Check stop sequences
                            if stop_sequences:
                                truncated, was_hit = truncate_at_stop(text_buffer, stop_sequences)
                                if was_hit:
                                    yield _sse(_chunk({"role": "assistant", "content": truncated}, "stop"))
                                    yield "data: [DONE]\n\n"
                                    break

                            # Partial match lookahead for <tool_calls>
                            last_open = text_buffer.rfind("<")
                            if last_open != -1 and "<tool_calls>".startswith(text_buffer[last_open:]):
                                continue  # Hold buffer

                            yield _sse(_chunk({"role": "assistant", "content": text_buffer}))
                            text_buffer = ""

                        elif is_tool_xml_complete(text_buffer):
                            parsed = parse_tool_calls_xml(text_buffer)
                            if parsed:
                                tool_calls_payload = [
                                    {
                                        "index": idx,
                                        "id": f"call_{uuid.uuid4().hex}",
                                        "type": "function",
                                        "function": {"name": tc["name"], "arguments": tc["arguments"]},
                                    }
                                    for idx, tc in enumerate(parsed)
                                ]
                                yield _sse(_chunk({"tool_calls": tool_calls_payload}))
                                has_yielded_tool = True
                            else:
                                # XML detected but regex failed — yield as text
                                yield _sse(_chunk({"role": "assistant", "content": text_buffer}))

                            text_buffer = ""
                            is_buffering_tool = False

                    elif chunk_type == "done":
                        end_payload = _chunk(
                            {},
                            finish_reason="tool_calls" if has_yielded_tool else "stop",
                        )
                        end_payload["usage"] = {
                            "prompt_tokens": chunk.get("promptTokens", 0),
                            "completion_tokens": chunk.get("responseTokens", 0),
                            "total_tokens": chunk.get("totalTokens", 0),
                        }
                        yield _sse(end_payload)
                        yield "data: [DONE]\n\n"

            except Exception as e:
                yield _sse({"error": {"message": str(e), "type": "server_error", "param": None, "code": None}})

        return StreamingResponse(event_generator(), media_type="text/event-stream")

    # --- Non-streaming path ---
    response_text = response_or_stream.get("response", "")
    p_tokens = response_or_stream.get("promptTokens", 0)
    c_tokens = response_or_stream.get("responseTokens", 0)
    t_tokens = response_or_stream.get("totalTokens", 0)

    finish_reason = "stop"
    tool_calls = None

    # Stop sequence truncation
    if STOP_TRUNCATION_ENABLED and body.stop:
        stop_sequences = body.stop if isinstance(body.stop, list) else [body.stop]
        response_text, _ = truncate_at_stop(response_text, stop_sequences)

    # Extract reasoning blocks
    response_text, reasoning = extract_reasoning(response_text)

    # Parse tool calls from XML
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
            response_text = strip_tool_xml(response_text)
            finish_reason = "tool_calls"

    return ChatCompletionResponse(
        id=req_id,
        created=created_time,
        model=model_str,
        choices=[
            ChoiceNonStreaming(
                index=0,
                message=Message(
                    role="assistant",
                    content=response_text or None,
                    reasoning=reasoning,
                    reasoning_content=reasoning,
                    tool_calls=tool_calls,
                ),
                finish_reason=finish_reason,
            )
        ],
        usage=Usage(prompt_tokens=p_tokens, completion_tokens=c_tokens, total_tokens=t_tokens),
    )
