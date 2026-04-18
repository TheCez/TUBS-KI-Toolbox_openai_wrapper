"""
OpenAI-compatible /v1/responses endpoint.
Translates Responses API requests into TU-BS KI-Toolbox API format.
"""

from __future__ import annotations

import json
import logging
import time
import uuid

from fastapi import APIRouter, Depends, Request
from fastapi.responses import StreamingResponse
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from app.models.responses import ResponseCreateRequest
from app.services.conversation_state import build_conversation_key, get_cached_thread_id, remember_thread_id
from app.services.openai_bridge import (
    build_tubs_payload_from_messages,
    parse_assistant_response,
    response_input_to_messages,
)
from app.services.prompt import has_tool_xml_start, is_tool_xml_complete, parse_tool_calls_xml
from app.services.staged_ingestion import prepare_staged_messages
from app.services.tool_validation import validate_tool_calls
from app.services.tubs_client import async_send_tubs_request

router = APIRouter()
security = HTTPBearer()
logger = logging.getLogger("uvicorn.error")


def _response_usage(done_chunk: dict) -> dict:
    input_tokens = done_chunk.get("promptTokens", 0)
    output_tokens = done_chunk.get("responseTokens", 0)
    total_tokens = done_chunk.get("totalTokens", input_tokens + output_tokens)
    return {
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": total_tokens,
        "input_token_details": {"cached_tokens": 0},
        "output_token_details": {"reasoning_tokens": 0},
    }


def _response_output_message(text: str, item_id: str | None = None) -> dict:
    return {
        "id": item_id or f"msg_{uuid.uuid4().hex}",
        "type": "message",
        "status": "completed",
        "role": "assistant",
        "content": [{"type": "output_text", "text": text, "annotations": []}],
    }


def _response_function_call_item(name: str, arguments: str, call_id: str | None = None, item_id: str | None = None) -> dict:
    return {
        "id": item_id or f"fc_{uuid.uuid4().hex}",
        "type": "function_call",
        "status": "completed",
        "call_id": call_id or f"call_{uuid.uuid4().hex}",
        "name": name,
        "arguments": arguments,
    }


@router.post("/responses")
async def create_response(
    request: Request,
    body: ResponseCreateRequest,
    credentials: HTTPAuthorizationCredentials = Depends(security),
):
    sanitized_headers = {
        key: ("<redacted>" if key.lower() == "authorization" else value)
        for key, value in request.headers.items()
        if key.lower() in {"authorization", "content-type", "user-agent"}
    }
    try:
        logger.info(
            "Incoming request path=%s headers=%s body=%s",
            request.url.path,
            sanitized_headers,
            json.loads((await request.body()).decode("utf-8")),
        )
    except Exception:
        logger.info("Incoming request path=%s headers=%s", request.url.path, sanitized_headers)

    token = credentials.credentials
    conversation_messages = body.input if isinstance(body.input, list) else [{"role": "user", "content": body.input}]
    conversation_key = build_conversation_key(
        bearer_token=token,
        model=str(body.model),
        messages=conversation_messages,
        explicit_user=body.user,
    )
    thread_id = get_cached_thread_id(conversation_key)
    normalized_messages = response_input_to_messages(body.input)
    staged = await prepare_staged_messages(
        model=body.model,
        messages=normalized_messages,
        thread_id=thread_id,
        conversation_key=conversation_key,
        bearer_token=token,
    )
    response_format = body.text.format if body.text and body.text.format else None
    payload, images, model_str = build_tubs_payload_from_messages(
        model=body.model,
        messages=staged.messages,
        thread_id=staged.thread_id,
        instructions=body.instructions,
        response_format=response_format,
        tools=body.tools,
        reasoning=body.reasoning,
        max_output_tokens=body.max_output_tokens,
        tool_choice=body.tool_choice,
    )

    response_or_stream = await async_send_tubs_request(
        payload=payload,
        images=images,
        bearer_token=token,
        stream=body.stream,
    )

    response_id = f"resp_{uuid.uuid4().hex}"
    created_at = int(time.time())

    if body.stream:
        async def event_generator():
            output_index = 0
            text_item_id = f"msg_{uuid.uuid4().hex}"
            yielded_content = False
            text_buffer = ""
            is_buffering_tool = False
            has_yielded_tool = False
            completed_items: list[dict] = []

            def emit(event_type: str, data: dict) -> str:
                event = {"type": event_type, **data}
                logger.info("Outgoing responses event=%s payload=%s", event_type, event)
                return f"event: {event_type}\ndata: {json.dumps(event)}\n\n"

            try:
                yield emit(
                    "response.created",
                    {
                        "response": {
                            "id": response_id,
                            "object": "response",
                            "created_at": created_at,
                            "status": "in_progress",
                            "model": model_str,
                            "output": [],
                            "parallel_tool_calls": bool(body.parallel_tool_calls),
                        }
                    },
                )
                yield emit("response.in_progress", {"response": {"id": response_id, "status": "in_progress"}})
                yield emit(
                    "response.output_item.added",
                    {
                        "response_id": response_id,
                        "output_index": output_index,
                        "item": {
                            "id": text_item_id,
                            "type": "message",
                            "status": "in_progress",
                            "role": "assistant",
                            "content": [],
                        },
                    },
                )

                async for chunk in response_or_stream:
                    chunk_type = chunk.get("type")

                    if chunk_type == "chunk":
                        text_buffer += chunk.get("content", "")
                        if not text_buffer:
                            continue

                        if has_tool_xml_start(text_buffer):
                            is_buffering_tool = True

                        if not is_buffering_tool:
                            last_open = text_buffer.rfind("<")
                            if last_open != -1 and "<tool_calls>".startswith(text_buffer[last_open:]):
                                continue

                            if not yielded_content:
                                yield emit(
                                    "response.content_part.added",
                                    {
                                        "response_id": response_id,
                                        "output_index": output_index,
                                        "item_id": text_item_id,
                                        "content_index": 0,
                                        "part": {"type": "output_text", "text": "", "annotations": []},
                                    },
                                )
                                yielded_content = True

                            yield emit(
                                "response.output_text.delta",
                                {
                                    "response_id": response_id,
                                    "output_index": output_index,
                                    "item_id": text_item_id,
                                    "content_index": 0,
                                    "delta": text_buffer,
                                },
                            )
                            text_buffer = ""
                            continue

                        if is_tool_xml_complete(text_buffer):
                            parsed = parse_tool_calls_xml(text_buffer)
                            if parsed:
                                validated = validate_tool_calls(parsed, body.tools)
                                if validated.valid_calls:
                                    has_yielded_tool = True
                                    for tc in validated.valid_calls:
                                        tool_item = _response_function_call_item(tc.name, tc.arguments)
                                        completed_items.append(tool_item)
                                        yield emit(
                                            "response.output_item.added",
                                            {
                                                "response_id": response_id,
                                                "output_index": output_index,
                                                "item": tool_item,
                                            },
                                        )
                                        yield emit(
                                            "response.output_item.done",
                                            {
                                                "response_id": response_id,
                                                "output_index": output_index,
                                                "item": tool_item,
                                            },
                                        )
                                        output_index += 1
                                elif validated.fallback_text:
                                    if not yielded_content:
                                        yield emit(
                                            "response.content_part.added",
                                            {
                                                "response_id": response_id,
                                                "output_index": output_index,
                                                "item_id": text_item_id,
                                                "content_index": 0,
                                                "part": {"type": "output_text", "text": "", "annotations": []},
                                            },
                                        )
                                        yielded_content = True
                                    yield emit(
                                        "response.output_text.delta",
                                        {
                                            "response_id": response_id,
                                            "output_index": output_index,
                                            "item_id": text_item_id,
                                            "content_index": 0,
                                            "delta": validated.fallback_text,
                                        },
                                    )
                            text_buffer = ""
                            is_buffering_tool = False

                    elif chunk_type == "done":
                        remember_thread_id(conversation_key, chunk)
                        final_text, _reasoning, _tool_calls, _finish_reason = parse_assistant_response(
                            chunk.get("response", ""),
                            tools=body.tools,
                        )

                        if yielded_content or final_text:
                            if not yielded_content:
                                yield emit(
                                    "response.content_part.added",
                                    {
                                        "response_id": response_id,
                                        "output_index": output_index,
                                        "item_id": text_item_id,
                                        "content_index": 0,
                                        "part": {"type": "output_text", "text": "", "annotations": []},
                                    },
                                )

                            yield emit(
                                "response.output_text.done",
                                {
                                    "response_id": response_id,
                                    "output_index": output_index,
                                    "item_id": text_item_id,
                                    "content_index": 0,
                                    "text": final_text,
                                },
                            )
                            final_message = _response_output_message(final_text, item_id=text_item_id)
                            yield emit(
                                "response.content_part.done",
                                {
                                    "response_id": response_id,
                                    "output_index": output_index,
                                    "item_id": text_item_id,
                                    "content_index": 0,
                                    "part": {"type": "output_text", "text": final_text, "annotations": []},
                                },
                            )
                            yield emit(
                                "response.output_item.done",
                                {
                                    "response_id": response_id,
                                    "output_index": output_index,
                                    "item": final_message,
                                },
                            )
                            completed_items.insert(0, final_message)

                        yield emit(
                            "response.completed",
                            {
                                "response": {
                                    "id": response_id,
                                    "object": "response",
                                    "created_at": created_at,
                                    "status": "completed",
                                    "model": model_str,
                                    "output": completed_items,
                                    "output_text": final_text,
                                    "parallel_tool_calls": bool(body.parallel_tool_calls or has_yielded_tool),
                                    "usage": _response_usage(chunk),
                                }
                            },
                        )
            except Exception as exc:
                yield emit(
                    "error",
                    {"error": {"message": str(exc), "type": "server_error", "code": "internal_error"}},
                )

        return StreamingResponse(event_generator(), media_type="text/event-stream")

    output_text, reasoning, tool_calls, _finish_reason = parse_assistant_response(
        response_or_stream.get("response", ""),
        tools=body.tools,
    )
    remember_thread_id(conversation_key, response_or_stream)
    output_items = []
    if output_text:
        output_items.append(_response_output_message(output_text))
    if tool_calls:
        output_items.extend(
            _response_function_call_item(
                tool_call.function.name,
                tool_call.function.arguments,
                call_id=tool_call.id,
            )
            for tool_call in tool_calls
        )

    response_payload = {
        "id": response_id,
        "object": "response",
        "created_at": created_at,
        "status": "completed",
        "model": model_str,
        "output": output_items,
        "output_text": output_text,
        "parallel_tool_calls": bool(body.parallel_tool_calls or tool_calls),
        "tools": [tool.model_dump() for tool in body.tools] if body.tools else [],
        "reasoning": body.reasoning.model_dump(exclude_none=True) if body.reasoning else None,
        "usage": _response_usage(response_or_stream),
    }

    logger.info("Outgoing responses payload=%s", response_payload)

    if reasoning:
        response_payload["reasoning_summary"] = reasoning
    if tool_calls:
        response_payload["tool_calls"] = [tool.model_dump() for tool in tool_calls]

    return response_payload
