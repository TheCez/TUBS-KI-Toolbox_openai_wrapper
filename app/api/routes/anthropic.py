"""
Anthropic-compatible /v1/messages endpoint.
Translates Anthropic requests into TU-BS KI-Toolbox API format.
"""

import os
import json
import uuid
from fastapi import APIRouter, Depends, Header, HTTPException, Request
from fastapi.responses import StreamingResponse
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials

from app.models.anthropic import (
    MessageRequest, MessageResponse, MessageResponseUsage,
    TextContentBlock, ToolUseContentBlock, ToolChoice,
)
from app.models.responses import ResponseReasoningConfig
from app.services.conversation_state import (
    build_conversation_key,
    build_prompt_with_compaction,
    forget_thread_id,
    get_cached_thread_id,
    messages_for_upstream_thread,
    remember_thread_id,
)
from app.services.context_ingest import context_ingest_service
from app.services.context_runtime import (
    _is_low_information_final_text,
    augment_anthropic_messages_with_context,
    pinned_state_instruction,
    resolve_anthropic_context_tools,
    _overflow_active_for_anthropic_messages,
)
from app.services.anthropic_translation import (
    compile_anthropic_messages_to_prompt, get_images_from_anthropic_messages,
)
from app.services.openai_bridge import build_custom_instructions, effective_prompt_token_budget
from app.services.tubs_client import async_send_tubs_request
from app.services.staged_ingestion import prepare_staged_messages
from app.services.thread_recovery import retry_with_fresh_thread_on_limit
from app.services.model_map import resolve_model
from app.services.prompt import (
    truncate_at_stop, parse_tool_calls_xml, is_tool_xml_complete, has_tool_xml_start,
)
from app.services.tool_validation import validate_tool_calls
from app.models.tubs import TubsChatRequest

router = APIRouter()
security = HTTPBearer(auto_error=False)

STOP_TRUNCATION_ENABLED = os.getenv("ENABLE_STOP_TRUNCATION", "false").lower() == "true"


async def get_anthropic_token(
    credentials: HTTPAuthorizationCredentials | None = Depends(security),
    x_api_key: str | None = Header(default=None),
) -> str:
    if credentials:
        return credentials.credentials
    if x_api_key:
        return x_api_key
    raise HTTPException(status_code=401, detail="Not authenticated")


def _tool_choice_to_openai_style(tool_choice):
    if not tool_choice:
        return None
    if isinstance(tool_choice, ToolChoice):
        if tool_choice.type == "tool" and tool_choice.name:
            return {"type": "function", "function": {"name": tool_choice.name}}
        if tool_choice.type == "any":
            return "required"
        return tool_choice.type
    if isinstance(tool_choice, dict):
        if tool_choice.get("type") == "tool" and tool_choice.get("name"):
            return {"type": "function", "function": {"name": tool_choice["name"]}}
        if tool_choice.get("type") == "any":
            return "required"
        return tool_choice.get("type")
    return None


def _thinking_to_reasoning(thinking) -> ResponseReasoningConfig | None:
    if not thinking or getattr(thinking, "type", None) == "disabled":
        return None
    budget = getattr(thinking, "budget_tokens", None) or 0
    effort = "medium"
    if budget >= 12000:
        effort = "xhigh"
    elif budget >= 4000:
        effort = "high"
    elif budget >= 1500:
        effort = "medium"
    else:
        effort = "low"
    return ResponseReasoningConfig(effort=effort)


@router.post("/messages", response_model=MessageResponse)
async def anthropic_messages(
    request: Request,
    body: MessageRequest,
    token: str = Depends(get_anthropic_token),
):
    conversation_key = build_conversation_key(
        bearer_token=token,
        model=body.model,
        messages=body.messages,
    )
    context_thread_id = conversation_key

    system_messages = []
    if body.system:
        if isinstance(body.system, str):
            system_messages.append(body.system)
        elif isinstance(body.system, list):
            system_messages.extend(part.text for part in body.system if hasattr(part, "text"))
    low_information_retry_note = (
        "The previous assistant reply was a non-answer placeholder. "
        "Answer the user's latest request directly and concretely. "
        "Do not reply with bootstrap filler, generic closure, or 'Nothing else to say here'."
    )

    async def _build_non_stream_response(force_fresh_thread: bool, retry_note: str | None = None):
        thread_id = None if force_fresh_thread else get_cached_thread_id(conversation_key)
        pinned_instruction = pinned_state_instruction(context_thread_id)
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
            else augment_anthropic_messages_with_context(working_messages, context_thread_id)
        )
        overflow_mode = staged.applied or _overflow_active_for_anthropic_messages(
            messages=effective_messages,
            thread_id=staged.thread_id,
            system_instructions="\n\n".join(
                part for part in [pinned_instruction, "\n".join(system_messages).strip() or None, retry_note] if part
            ) or None,
            tools=body.tools,
            max_output_tokens=body.max_tokens,
            tool_choice=_tool_choice_to_openai_style(body.tool_choice),
            reasoning=_thinking_to_reasoning(body.thinking),
        )
        if overflow_mode:
            context_ingest_service().ingest_turn(context_thread_id, body.messages)
        resolved = await resolve_anthropic_context_tools(
            model=body.model,
            messages=effective_messages,
            thread_id=staged.thread_id,
            context_thread_id=context_thread_id,
            bearer_token=token,
            system_instructions="\n\n".join(
                part for part in [pinned_instruction, "\n".join(system_messages).strip() or None, retry_note] if part
            ) or None,
            tools=body.tools,
            max_output_tokens=body.max_tokens,
            tool_choice=_tool_choice_to_openai_style(body.tool_choice),
            reasoning=_thinking_to_reasoning(body.thinking),
            send_request=async_send_tubs_request,
            require_context_retrieval=overflow_mode,
        )
        return resolved.final_response

    if body.stream:
        thread_id = get_cached_thread_id(conversation_key)
        pinned_instruction = pinned_state_instruction(context_thread_id)
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
            else augment_anthropic_messages_with_context(working_messages, context_thread_id)
        )
        images = get_images_from_anthropic_messages(effective_messages)
        custom_instructions = build_custom_instructions(
            messages=[],
            instructions="\n\n".join(part for part in [pinned_instruction, "\n".join(system_messages).strip() or None] if part) or None,
            tools=body.tools,
            reasoning=_thinking_to_reasoning(body.thinking),
            max_output_tokens=body.max_tokens,
            tool_choice=_tool_choice_to_openai_style(body.tool_choice),
        )
        prompt_string = build_prompt_with_compaction(
            effective_messages,
            compile_prompt=compile_anthropic_messages_to_prompt,
            thread_id=staged.thread_id,
            prompt_token_budget=effective_prompt_token_budget(staged.thread_id, custom_instructions),
        )
        tubs_payload = TubsChatRequest(
            thread=staged.thread_id,
            prompt=prompt_string,
            model=resolve_model(body.model),
            customInstructions=custom_instructions,
        ).model_dump(exclude_none=True)
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

    req_id = f"msg_{uuid.uuid4().hex}"

    if body.stream:
        async def event_generator():
            text_buffer = ""
            is_buffering_tool = False
            has_yielded_tool = False
            content_block_idx = 0
            text_block_open = True
            output_tokens = 0

            def _event(event_type: str, data: dict) -> str:
                return f"event: {event_type}\ndata: {json.dumps(data)}\n\n"

            try:
                yield _event("message_start", {
                    "type": "message_start",
                    "message": {
                        "id": req_id, "type": "message", "role": "assistant",
                        "model": body.model, "content": [],
                        "stop_reason": None, "stop_sequence": None,
                        "usage": {"input_tokens": 0, "output_tokens": 0},
                    },
                })

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
                            if not text_block_open:
                                if not text_buffer.strip():
                                    text_buffer = ""
                                    continue
                                yield _event("content_block_start", {
                                    "type": "content_block_start", "index": content_block_idx,
                                    "content_block": {"type": "text", "text": ""},
                                })
                                text_block_open = True

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
                                        "usage": {"output_tokens": output_tokens},
                                    })
                                    yield _event("message_stop", {"type": "message_stop"})
                                    break

                            last_open = text_buffer.rfind("<")
                            if last_open != -1 and "<tool_calls>".startswith(text_buffer[last_open:]):
                                continue

                            yield _event("content_block_delta", {
                                "type": "content_block_delta", "index": content_block_idx,
                                "delta": {"type": "text_delta", "text": text_buffer},
                            })
                            text_buffer = ""

                        elif is_tool_xml_complete(text_buffer):
                            parsed = parse_tool_calls_xml(text_buffer)
                            if parsed:
                                validated = validate_tool_calls(parsed, body.tools)
                                if validated.valid_calls:
                                    if text_block_open:
                                        yield _event("content_block_stop", {"type": "content_block_stop", "index": content_block_idx})
                                        text_block_open = False
                                        content_block_idx += 1

                                    for tc in validated.valid_calls:
                                        tool_id = f"toolu_{uuid.uuid4().hex}"
                                        yield _event("content_block_start", {
                                            "type": "content_block_start", "index": content_block_idx,
                                            "content_block": {"type": "tool_use", "id": tool_id, "name": tc.name},
                                        })
                                        yield _event("content_block_delta", {
                                            "type": "content_block_delta", "index": content_block_idx,
                                            "delta": {"type": "input_json_delta", "partial_json": tc.arguments},
                                        })
                                        yield _event("content_block_stop", {"type": "content_block_stop", "index": content_block_idx})
                                        content_block_idx += 1
                                    has_yielded_tool = True
                                elif validated.fallback_text:
                                    yield _event("content_block_delta", {
                                        "type": "content_block_delta", "index": content_block_idx,
                                        "delta": {"type": "text_delta", "text": validated.fallback_text},
                                    })

                            text_buffer = ""
                            is_buffering_tool = False

                    elif chunk_type == "done":
                        remember_thread_id(conversation_key, chunk)
                        output_tokens = chunk.get("responseTokens", 0)
                        final_response = chunk.get("response", "")
                        parsed = parse_tool_calls_xml(final_response) if has_tool_xml_start(final_response) else []

                        if parsed and not has_yielded_tool:
                            validated = validate_tool_calls(parsed, body.tools)
                            if validated.valid_calls:
                                if text_block_open:
                                    yield _event("content_block_stop", {"type": "content_block_stop", "index": content_block_idx})
                                    text_block_open = False
                                    content_block_idx += 1
                                for tc in validated.valid_calls:
                                    tool_id = f"toolu_{uuid.uuid4().hex}"
                                    yield _event("content_block_start", {
                                        "type": "content_block_start", "index": content_block_idx,
                                        "content_block": {"type": "tool_use", "id": tool_id, "name": tc.name},
                                    })
                                    yield _event("content_block_delta", {
                                        "type": "content_block_delta", "index": content_block_idx,
                                        "delta": {"type": "input_json_delta", "partial_json": tc.arguments},
                                    })
                                    yield _event("content_block_stop", {"type": "content_block_stop", "index": content_block_idx})
                                    content_block_idx += 1
                                has_yielded_tool = True
                            elif validated.fallback_text:
                                yield _event("content_block_delta", {
                                    "type": "content_block_delta", "index": content_block_idx,
                                    "delta": {"type": "text_delta", "text": validated.fallback_text},
                                })
                        elif text_block_open:
                            yield _event("content_block_stop", {"type": "content_block_stop", "index": content_block_idx})

                        yield _event("message_delta", {
                            "type": "message_delta",
                            "delta": {
                                "stop_reason": "tool_use" if has_yielded_tool else "end_turn",
                                "stop_sequence": None,
                            },
                            "usage": {"output_tokens": output_tokens},
                        })
                        yield _event("message_stop", {"type": "message_stop"})

            except Exception as exc:
                yield _event("error", {"type": "error", "error": {"type": "api_error", "message": str(exc)}})

        return StreamingResponse(event_generator(), media_type="text/event-stream")

    response_text = response_or_stream.get("response", "")
    remember_thread_id(conversation_key, response_or_stream)
    p_tokens = response_or_stream.get("promptTokens", 0)
    c_tokens = response_or_stream.get("responseTokens", 0)
    if _is_low_information_final_text(response_text):
        context_ingest_service().ingest_turn(context_thread_id, body.messages)
        forget_thread_id(conversation_key)
        response_or_stream = await _build_non_stream_response(True, low_information_retry_note)
        response_text = response_or_stream.get("response", "")
        remember_thread_id(conversation_key, response_or_stream)
        p_tokens = response_or_stream.get("promptTokens", 0)
        c_tokens = response_or_stream.get("responseTokens", 0)

    stop_reason = "end_turn"
    content_blocks = []

    if STOP_TRUNCATION_ENABLED and body.stop_sequences:
        response_text, was_truncated = truncate_at_stop(response_text, body.stop_sequences)
        if was_truncated:
            stop_reason = "stop_sequence"

    if has_tool_xml_start(response_text):
        parsed = parse_tool_calls_xml(response_text)
        if parsed:
            validated = validate_tool_calls(parsed, body.tools)
            if validated.valid_calls:
                for tc in validated.valid_calls:
                    args_json = json.loads(tc.arguments)
                    content_blocks.append(ToolUseContentBlock(
                        type="tool_use",
                        id=f"toolu_{uuid.uuid4().hex}",
                        name=tc.name,
                        input=args_json,
                    ))
                stop_reason = "tool_use"
            elif validated.fallback_text:
                content_blocks.append(TextContentBlock(type="text", text=validated.fallback_text))
        elif response_text:
            content_blocks.append(TextContentBlock(type="text", text=response_text))
    elif response_text:
        content_blocks.append(TextContentBlock(type="text", text=response_text))
    context_ingest_service().ingest_turn(
        context_thread_id,
        body.messages,
        response_text=response_text or None,
    )

    return MessageResponse(
        id=req_id,
        type="message",
        role="assistant",
        content=content_blocks,
        model=body.model,
        stop_reason=stop_reason,
        usage=MessageResponseUsage(input_tokens=p_tokens, output_tokens=c_tokens),
    )
