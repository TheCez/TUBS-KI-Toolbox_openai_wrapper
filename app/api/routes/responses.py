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
            item_id = f"msg_{uuid.uuid4().hex}"
            yielded_content = False

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
                            "id": item_id,
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
                        text = chunk.get("content", "")
                        if not text:
                            continue
                        if not yielded_content:
                            yield emit(
                                "response.content_part.added",
                                {
                                    "response_id": response_id,
                                    "output_index": output_index,
                                    "item_id": item_id,
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
                                "item_id": item_id,
                                "content_index": 0,
                                "delta": text,
                            },
                        )
                    elif chunk_type == "done":
                        final_text, _reasoning, _tool_calls, _finish_reason = parse_assistant_response(
                            chunk.get("response", "")
                        )
                        if not yielded_content:
                            yield emit(
                                "response.content_part.added",
                                {
                                    "response_id": response_id,
                                    "output_index": output_index,
                                    "item_id": item_id,
                                    "content_index": 0,
                                    "part": {"type": "output_text", "text": "", "annotations": []},
                                },
                            )
                        yield emit(
                            "response.output_text.done",
                            {
                                "response_id": response_id,
                                "output_index": output_index,
                                "item_id": item_id,
                                "content_index": 0,
                                "text": final_text,
                            },
                        )
                        yield emit(
                            "response.content_part.done",
                            {
                                "response_id": response_id,
                                "output_index": output_index,
                                "item_id": item_id,
                                "content_index": 0,
                                "part": {"type": "output_text", "text": final_text, "annotations": []},
                            },
                        )
                        yield emit(
                            "response.output_item.done",
                            {
                                "response_id": response_id,
                                "output_index": output_index,
                                "item": _response_output_message(final_text, item_id=item_id),
                            },
                        )
                        yield emit(
                            "response.completed",
                            {
                                "response": {
                                    "id": response_id,
                                    "object": "response",
                                    "created_at": created_at,
                                    "status": "completed",
                                    "model": model_str,
                                    "output": [_response_output_message(final_text, item_id=item_id)],
                                    "output_text": final_text,
                                    "parallel_tool_calls": bool(body.parallel_tool_calls),
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
    response_payload = {
        "id": response_id,
        "object": "response",
        "created_at": created_at,
        "status": "completed",
        "model": model_str,
        "output": [_response_output_message(output_text)],
        "output_text": output_text,
        "parallel_tool_calls": bool(body.parallel_tool_calls),
        "tools": [tool.model_dump() for tool in body.tools] if body.tools else [],
        "reasoning": body.reasoning.model_dump(exclude_none=True) if body.reasoning else None,
        "usage": _response_usage(response_or_stream),
    }

    if reasoning:
        response_payload["reasoning_summary"] = reasoning
    if tool_calls:
        response_payload["tool_calls"] = [tool.model_dump() for tool in tool_calls]

    return response_payload
