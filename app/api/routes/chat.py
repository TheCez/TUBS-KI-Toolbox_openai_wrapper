"""
OpenAI-compatible /v1/chat/completions endpoint.
Translates OpenAI requests into TU-BS KI-Toolbox API format.
"""

import json
import os
import time
import uuid

from fastapi import APIRouter, Depends, Request
from fastapi.responses import StreamingResponse
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from app.models.openai import ChatCompletionRequest, ChatCompletionResponse, ChoiceNonStreaming, Message, Usage
from app.services.conversation_state import (
    build_conversation_key,
    forget_thread_id,
    get_cached_thread_id,
    messages_for_upstream_thread,
    remember_thread_id,
)
from app.services.context_ingest import context_ingest_service
from app.services.context_runtime import (
    _is_low_information_final_text,
    augment_openai_messages_with_context,
    resolve_openai_context_tools,
    _overflow_active_for_openai_messages,
)
from app.services.openai_bridge import build_tubs_payload_from_messages, parse_assistant_response
from app.services.prompt import has_tool_xml_start, is_tool_xml_complete, parse_tool_calls_xml, truncate_at_stop
from app.services.staged_ingestion import prepare_staged_messages
from app.services.thread_recovery import retry_with_fresh_thread_on_limit
from app.services.tool_validation import validate_tool_calls
from app.services.tubs_client import async_send_tubs_request

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
    conversation_key = build_conversation_key(
        bearer_token=token,
        model=str(body.model.value) if hasattr(body.model, "value") else str(body.model),
        messages=body.messages,
        explicit_user=body.user,
    )
    context_thread_id = conversation_key
    model_str = str(body.model.value) if hasattr(body.model, "value") else str(body.model)
    low_information_retry_note = (
        "The previous assistant reply was a non-answer placeholder. "
        "Answer the user's latest request directly and concretely. "
        "Do not reply with bootstrap filler, generic closure, or 'Nothing else to say here'."
    )

    async def _build_non_stream_response(force_fresh_thread: bool, retry_note: str | None = None):
        thread_id = None if force_fresh_thread else get_cached_thread_id(conversation_key)
        staged = await prepare_staged_messages(
            model=body.model,
            messages=body.messages,
            thread_id=thread_id,
            conversation_key=conversation_key,
            bearer_token=token,
        )
        working_messages = messages_for_upstream_thread(staged.messages, staged.thread_id)
        effective_messages = (
            working_messages
            if staged.thread_id
            else augment_openai_messages_with_context(working_messages, context_thread_id)
        )
        overflow_mode = staged.applied or _overflow_active_for_openai_messages(
            messages=effective_messages,
            thread_id=staged.thread_id,
            instructions=retry_note,
            response_format=body.response_format,
            tools=body.tools,
            reasoning=None,
            max_output_tokens=body.max_completion_tokens or body.max_tokens,
            tool_choice=body.tool_choice,
        )
        if overflow_mode:
            context_ingest_service().ingest_turn(context_thread_id, body.messages)
        resolved = await resolve_openai_context_tools(
            model=body.model,
            messages=effective_messages,
            thread_id=staged.thread_id,
            context_thread_id=context_thread_id,
            bearer_token=token,
            instructions=retry_note,
            response_format=body.response_format,
            tools=body.tools,
            reasoning=None,
            max_output_tokens=body.max_completion_tokens or body.max_tokens,
            tool_choice=body.tool_choice,
            send_request=async_send_tubs_request,
            require_context_retrieval=overflow_mode,
        )
        return resolved.final_response

    if body.stream:
        thread_id = get_cached_thread_id(conversation_key)
        staged = await prepare_staged_messages(
            model=body.model,
            messages=body.messages,
            thread_id=thread_id,
            conversation_key=conversation_key,
            bearer_token=token,
        )
        working_messages = messages_for_upstream_thread(staged.messages, staged.thread_id)
        effective_messages = (
            working_messages
            if staged.thread_id
            else augment_openai_messages_with_context(working_messages, context_thread_id)
        )
        thread_id = staged.thread_id
        tubs_payload, images, model_str = build_tubs_payload_from_messages(
            model=body.model,
            messages=effective_messages,
            thread_id=thread_id,
            response_format=body.response_format,
            tools=body.tools,
            max_output_tokens=body.max_completion_tokens or body.max_tokens,
            tool_choice=body.tool_choice,
        )
        response_or_stream = await async_send_tubs_request(
            payload=tubs_payload,
            images=images,
            bearer_token=token,
            stream=True,
        )
    else:
        response_or_stream = await retry_with_fresh_thread_on_limit(
            conversation_key,
            _build_non_stream_response,
        )
        model_str = str(body.model.value) if hasattr(body.model, "value") else str(body.model)

    req_id = f"chatcmpl-{uuid.uuid4().hex}"
    created_time = int(time.time())

    if body.stream:
        stop_sequences = None
        if STOP_TRUNCATION_ENABLED and body.stop:
            stop_sequences = body.stop if isinstance(body.stop, list) else [body.stop]

        async def event_generator():
            text_buffer = ""
            is_buffering_tool = False
            has_yielded_tool = False
            yielded_role = False

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
                            if stop_sequences:
                                truncated, was_hit = truncate_at_stop(text_buffer, stop_sequences)
                                if was_hit:
                                    delta = {"content": truncated}
                                    if not yielded_role:
                                        delta["role"] = "assistant"
                                        yielded_role = True
                                    yield _sse(_chunk(delta, "stop"))
                                    yield "data: [DONE]\n\n"
                                    break

                            last_open = text_buffer.rfind("<")
                            if last_open != -1 and "<tool_calls>".startswith(text_buffer[last_open:]):
                                continue

                            delta = {"content": text_buffer}
                            if not yielded_role:
                                delta["role"] = "assistant"
                                yielded_role = True
                            yield _sse(_chunk(delta))
                            text_buffer = ""
                            continue

                        if is_tool_xml_complete(text_buffer):
                            parsed = parse_tool_calls_xml(text_buffer)
                            if parsed:
                                validated = validate_tool_calls(parsed, body.tools)
                                if validated.valid_calls:
                                    tool_calls_payload = [
                                        {
                                            "index": idx,
                                            "id": f"call_{uuid.uuid4().hex}",
                                            "type": "function",
                                            "function": {"name": tc.name, "arguments": tc.arguments},
                                        }
                                        for idx, tc in enumerate(validated.valid_calls)
                                    ]
                                    yield _sse(_chunk({"tool_calls": tool_calls_payload}))
                                    has_yielded_tool = True
                                elif validated.fallback_text:
                                    delta = {"content": validated.fallback_text}
                                    if not yielded_role:
                                        delta["role"] = "assistant"
                                        yielded_role = True
                                    yield _sse(_chunk(delta))
                            text_buffer = ""
                            is_buffering_tool = False

                    elif chunk_type == "done":
                        remember_thread_id(conversation_key, chunk)
                        final_text, _reasoning, final_tool_calls, final_finish_reason = parse_assistant_response(
                            chunk.get("response", ""),
                            tools=body.tools,
                        )

                        if final_text and not has_yielded_tool:
                            delta = {"content": final_text}
                            if not yielded_role:
                                delta["role"] = "assistant"
                                yielded_role = True
                            yield _sse(_chunk(delta))
                        elif final_tool_calls and not has_yielded_tool:
                            tool_calls_payload = [
                                {
                                    "index": idx,
                                    "id": tool_call.id,
                                    "type": "function",
                                    "function": {
                                        "name": tool_call.function.name,
                                        "arguments": tool_call.function.arguments,
                                    },
                                }
                                for idx, tool_call in enumerate(final_tool_calls)
                            ]
                            yield _sse(_chunk({"tool_calls": tool_calls_payload}))
                            has_yielded_tool = True

                        end_payload = _chunk({}, finish_reason=final_finish_reason if has_yielded_tool else "stop")
                        end_payload["usage"] = {
                            "prompt_tokens": chunk.get("promptTokens", 0),
                            "completion_tokens": chunk.get("responseTokens", 0),
                            "total_tokens": chunk.get("totalTokens", 0),
                        }
                        yield _sse(end_payload)
                        yield "data: [DONE]\n\n"
            except Exception as exc:
                yield _sse({"error": {"message": str(exc), "type": "server_error", "param": None, "code": None}})

        return StreamingResponse(event_generator(), media_type="text/event-stream")

    response_text = response_or_stream.get("response", "")
    remember_thread_id(conversation_key, response_or_stream)
    p_tokens = response_or_stream.get("promptTokens", 0)
    c_tokens = response_or_stream.get("responseTokens", 0)
    t_tokens = response_or_stream.get("totalTokens", 0)

    if STOP_TRUNCATION_ENABLED and body.stop:
        stop_sequences = body.stop if isinstance(body.stop, list) else [body.stop]
        response_text, _ = truncate_at_stop(response_text, stop_sequences)
    response_text, reasoning, tool_calls, finish_reason = parse_assistant_response(
        response_text,
        tools=body.tools,
    )
    if not tool_calls and _is_low_information_final_text(response_text):
        context_ingest_service().ingest_turn(context_thread_id, body.messages)
        forget_thread_id(conversation_key)
        response_or_stream = await _build_non_stream_response(True, low_information_retry_note)
        response_text = response_or_stream.get("response", "")
        remember_thread_id(conversation_key, response_or_stream)
        p_tokens = response_or_stream.get("promptTokens", 0)
        c_tokens = response_or_stream.get("responseTokens", 0)
        t_tokens = response_or_stream.get("totalTokens", 0)
        if STOP_TRUNCATION_ENABLED and body.stop:
            stop_sequences = body.stop if isinstance(body.stop, list) else [body.stop]
            response_text, _ = truncate_at_stop(response_text, stop_sequences)
        response_text, reasoning, tool_calls, finish_reason = parse_assistant_response(
            response_text,
            tools=body.tools,
        )
    context_ingest_service().ingest_turn(context_thread_id, body.messages, response_text=response_text or None)

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
