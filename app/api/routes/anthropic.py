"""
Anthropic-compatible /v1/messages endpoint.
Translates Anthropic requests into TU-BS KI-Toolbox API format.
"""

import os
import json
import uuid
from fastapi import APIRouter, Depends, Request
from fastapi.responses import StreamingResponse
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials

from app.models.anthropic import (
    MessageRequest, MessageResponse, MessageResponseUsage,
    TextContentBlock, ToolUseContentBlock,
)
from app.models.tubs import TubsChatRequest
from app.services.anthropic_translation import (
    compile_anthropic_messages_to_prompt, get_images_from_anthropic_messages,
)
from app.services.tubs_client import async_send_tubs_request
from app.services.model_map import resolve_model
from app.services.prompt import (
    build_tool_instructions, truncate_at_stop, parse_tool_calls_xml,
    is_tool_xml_complete, has_tool_xml_start, strip_tool_xml,
)

router = APIRouter()
security = HTTPBearer()

STOP_TRUNCATION_ENABLED = os.getenv("ENABLE_STOP_TRUNCATION", "false").lower() == "true"


@router.post("/messages", response_model=MessageResponse)
async def anthropic_messages(
    request: Request,
    body: MessageRequest,
    credentials: HTTPAuthorizationCredentials = Depends(security),
):
    token = credentials.credentials

    # Compile prompt from Anthropic messages
    prompt_string = compile_anthropic_messages_to_prompt(body.messages)

    # Extract images if any
    images = get_images_from_anthropic_messages(body.messages)

    # Build custom instructions from system prompt
    custom_instructions = ""
    if body.system:
        if isinstance(body.system, str):
            custom_instructions += body.system + "\n\n"
        elif isinstance(body.system, list):
            custom_instructions += "\n".join(
                part.text for part in body.system if hasattr(part, "text")
            ) + "\n\n"

    # Inject tool-calling instructions if tools are provided
    if body.tools:
        custom_instructions += build_tool_instructions(body.tools)

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

    req_id = f"msg_{uuid.uuid4().hex}"

    # --- Streaming path ---
    if body.stream:
        async def event_generator():
            text_buffer = ""
            is_buffering_tool = False
            has_yielded_tool = False
            content_block_idx = 0

            def _event(event_type: str, data: dict) -> str:
                return f"event: {event_type}\ndata: {json.dumps(data)}\n\n"

            try:
                # Anthropic message_start
                yield _event("message_start", {
                    "type": "message_start",
                    "message": {
                        "id": req_id, "type": "message", "role": "assistant",
                        "model": body.model, "content": [],
                        "stop_reason": None, "stop_sequence": None,
                        "usage": {"input_tokens": 0, "output_tokens": 0},
                    },
                })

                # Initial text content block
                yield _event("content_block_start", {
                    "type": "content_block_start", "index": content_block_idx,
                    "content_block": {"type": "text", "text": ""},
                })

                async for chunk in response_or_stream:
                    chunk_type = chunk.get("type")

                    if chunk_type == "chunk":
                        text_buffer += chunk.get("content", "")

                        if has_tool_xml_start(text_buffer):
                            is_buffering_tool = True

                        if not is_buffering_tool:
                            # Stop sequence check
                            if STOP_TRUNCATION_ENABLED and body.stop_sequences:
                                truncated, was_hit = truncate_at_stop(text_buffer, body.stop_sequences)
                                if was_hit:
                                    yield _event("content_block_delta", {
                                        "type": "content_block_delta", "index": content_block_idx,
                                        "delta": {"type": "text_delta", "text": truncated},
                                    })
                                    yield _event("content_block_stop", {"type": "content_block_stop", "index": content_block_idx})
                                    yield _event("message_delta", {
                                        "type": "message_delta",
                                        "delta": {"stop_reason": "stop_sequence", "stop_sequence": None},
                                        "usage": {"output_tokens": 0},
                                    })
                                    yield _event("message_stop", {"type": "message_stop"})
                                    break

                            # Partial match lookahead
                            last_open = text_buffer.rfind("<")
                            if last_open != -1 and "<tool_calls>".startswith(text_buffer[last_open:]):
                                continue  # Hold buffer

                            yield _event("content_block_delta", {
                                "type": "content_block_delta", "index": content_block_idx,
                                "delta": {"type": "text_delta", "text": text_buffer},
                            })
                            text_buffer = ""

                        elif is_tool_xml_complete(text_buffer):
                            parsed = parse_tool_calls_xml(text_buffer)
                            if parsed:
                                # Close text block
                                yield _event("content_block_stop", {"type": "content_block_stop", "index": content_block_idx})
                                content_block_idx += 1

                                for tc in parsed:
                                    tool_id = f"toolu_{uuid.uuid4().hex}"
                                    yield _event("content_block_start", {
                                        "type": "content_block_start", "index": content_block_idx,
                                        "content_block": {"type": "tool_use", "id": tool_id, "name": tc["name"]},
                                    })
                                    yield _event("content_block_delta", {
                                        "type": "content_block_delta", "index": content_block_idx,
                                        "delta": {"type": "input_json_delta", "partial_json": tc["arguments"]},
                                    })
                                    yield _event("content_block_stop", {"type": "content_block_stop", "index": content_block_idx})
                                    content_block_idx += 1

                                has_yielded_tool = True
                            else:
                                yield _event("content_block_delta", {
                                    "type": "content_block_delta", "index": content_block_idx,
                                    "delta": {"type": "text_delta", "text": text_buffer},
                                })

                            text_buffer = ""
                            is_buffering_tool = False

                    elif chunk_type == "done":
                        if not has_yielded_tool:
                            yield _event("content_block_stop", {"type": "content_block_stop", "index": content_block_idx})

                        yield _event("message_delta", {
                            "type": "message_delta",
                            "delta": {
                                "stop_reason": "tool_use" if has_yielded_tool else "end_turn",
                                "stop_sequence": None,
                            },
                            "usage": {"output_tokens": chunk.get("responseTokens", 0)},
                        })
                        yield _event("message_stop", {"type": "message_stop"})

            except Exception as e:
                yield _event("error", {"type": "error", "error": {"type": "api_error", "message": str(e)}})

        return StreamingResponse(event_generator(), media_type="text/event-stream")

    # --- Non-streaming path ---
    response_text = response_or_stream.get("response", "")
    p_tokens = response_or_stream.get("promptTokens", 0)
    c_tokens = response_or_stream.get("responseTokens", 0)

    stop_reason = "end_turn"
    content_blocks = []

    # Stop sequence truncation
    if STOP_TRUNCATION_ENABLED and body.stop_sequences:
        response_text, was_truncated = truncate_at_stop(response_text, body.stop_sequences)
        if was_truncated:
            stop_reason = "stop_sequence"

    # Parse tool calls from XML
    if has_tool_xml_start(response_text):
        parsed = parse_tool_calls_xml(response_text)
        if parsed:
            text_content = strip_tool_xml(response_text)
            if text_content:
                content_blocks.append(TextContentBlock(type="text", text=text_content))

            for tc in parsed:
                try:
                    args_json = json.loads(tc["arguments"])
                except (json.JSONDecodeError, TypeError):
                    args_json = {}

                content_blocks.append(ToolUseContentBlock(
                    type="tool_use",
                    id=f"toolu_{uuid.uuid4().hex}",
                    name=tc["name"],
                    input=args_json,
                ))
            stop_reason = "tool_use"
        else:
            content_blocks.append(TextContentBlock(type="text", text=response_text))
    elif response_text:
        content_blocks.append(TextContentBlock(type="text", text=response_text))

    return MessageResponse(
        id=req_id,
        type="message",
        role="assistant",
        content=content_blocks,
        model=body.model,
        stop_reason=stop_reason,
        usage=MessageResponseUsage(input_tokens=p_tokens, output_tokens=c_tokens),
    )
