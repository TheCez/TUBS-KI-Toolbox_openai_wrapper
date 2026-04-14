"""
OpenAI-compatible /v1/responses endpoint.
Translates Responses API requests into TU-BS KI-Toolbox API format.
"""

from __future__ import annotations

import json
import time
import uuid

from fastapi import APIRouter, Depends
from fastapi.responses import StreamingResponse
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from app.models.responses import ResponseCreateRequest
from app.services.openai_bridge import build_tubs_payload_from_response_request, parse_assistant_response
from app.services.prompt import has_tool_xml_start, is_tool_xml_complete, parse_tool_calls_xml
from app.services.tubs_client import async_send_tubs_request

router = APIRouter()
security = HTTPBearer()


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
    body: ResponseCreateRequest,
    credentials: HTTPAuthorizationCredentials = Depends(security),
):
    token = credentials.credentials
    payload, images, model_str = build_tubs_payload_from_response_request(body)

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
                                has_yielded_tool = True
                                for tc in parsed:
                                    tool_item = _response_function_call_item(tc["name"], tc["arguments"])
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
                            text_buffer = ""
                            is_buffering_tool = False

                    elif chunk_type == "done":
                        final_text, _reasoning, _tool_calls, _finish_reason = parse_assistant_response(
                            chunk.get("response", "")
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
        response_or_stream.get("response", "")
    )
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

    if reasoning:
        response_payload["reasoning_summary"] = reasoning
    if tool_calls:
        response_payload["tool_calls"] = [tool.model_dump() for tool in tool_calls]

    return response_payload
