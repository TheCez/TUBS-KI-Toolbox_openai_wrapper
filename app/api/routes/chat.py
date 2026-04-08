import os
import time
import json
import uuid
from fastapi import APIRouter, Depends, Request
from fastapi.responses import StreamingResponse
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
import re
from app.models.openai import ChatCompletionRequest, ChatCompletionResponse, ChoiceNonStreaming, Usage, Message, ToolCall, ToolCallFunction
from app.models.tubs import TubsChatRequest
from app.services.translation import compile_messages_to_prompt, get_images_from_messages
from app.services.tubs_client import async_send_tubs_request

router = APIRouter()
security = HTTPBearer()

@router.post("/chat/completions", response_model=ChatCompletionResponse)
async def chat_completions(
    request: Request,
    body: ChatCompletionRequest,
    credentials: HTTPAuthorizationCredentials = Depends(security)
):
    token = credentials.credentials

    # Compile prompt
    prompt_string = compile_messages_to_prompt(body.messages)
    
    # Extract images if any
    images = get_images_from_messages(body.messages)

    # Extract System Messages
    system_messages = [msg.content for msg in body.messages if msg.role.lower() in ["system", "developer"] and isinstance(msg.content, str)]
    system_instructions = "\n".join(system_messages).strip()

    custom_instructions = ""
    if system_instructions:
        custom_instructions += system_instructions + "\n\n"
    if body.response_format:
        format_type = body.response_format.get("type")
        if format_type == "json_object":
            custom_instructions += "You must output your response as a valid JSON object.\n"
        elif format_type == "json_schema":
            schema = body.response_format.get("json_schema", {})
            schema_str = json.dumps(schema)
            custom_instructions += f"You must strictly adhere to the following JSON schema for your output. Output only valid JSON.\nSchema: {schema_str}\n"

    if body.tools:
        tools_str = json.dumps(body.tools)
        xml_rule = (
            "You have access to tools. If you need to use a tool, DO NOT write conversational text. Instead, output an XML block. If you need to trigger multiple tools simultaneously, you MUST wrap them in a <tool_calls> root element and output all of them back-to-back before stopping. Use this EXACT format:\n"
            "<tool_calls><tool_call><name>function_name</name><arguments>{\"actual_key\": \"actual_value\"}</arguments></tool_call></tool_calls>\n"
            "The <arguments> MUST contain raw, valid JSON that perfectly matches the provided tool schema. DO NOT wrap the arguments inside a {\"json\": ...} parent object.\n"
            f"Available tools: {tools_str}\n"
        )
        custom_instructions += xml_rule

    custom_instructions = custom_instructions.strip() if custom_instructions else None

    # Build Tubs request payload
    tubs_payload = TubsChatRequest(
        thread=None,
        prompt=prompt_string,
        model=body.model,
        customInstructions=custom_instructions,
    ).model_dump(exclude_none=True)

    # Perform request
    response_or_stream = await async_send_tubs_request(
        payload=tubs_payload,
        images=images,
        bearer_token=token,
        stream=bool(body.stream)
    )

    req_id = f"chatcmpl-{uuid.uuid4().hex}"
    created_time = int(time.time())

    if body.stream:
        # returns StreamingResponse
        async def event_generator():
            text_buffer = ""
            is_buffering_tool = False
            has_yielded_tool = False
            
            try:
                # Need to iterate the generator
                async for chunk in response_or_stream:
                    chunk_type = chunk.get("type")
                    if chunk_type == "chunk":
                        content = chunk.get("content", "")
                        text_buffer += content
                        
                        if "<tool_call>" in text_buffer or "<tool_calls>" in text_buffer:
                            is_buffering_tool = True
                            
                        if not is_buffering_tool:
                            # Stop Sequence Truncator Logic
                            if os.getenv("ENABLE_STOP_TRUNCATION", "False").lower() == "true" and body.stop:
                                stop_sequences = body.stop if isinstance(body.stop, list) else [body.stop]
                                earliest_idx = -1
                                for stop_seq in stop_sequences:
                                    idx = text_buffer.find(stop_seq)
                                    if idx != -1:
                                        if earliest_idx == -1 or idx < earliest_idx:
                                            earliest_idx = idx
                                
                                if earliest_idx != -1:
                                    trunc_text = text_buffer[:earliest_idx]
                                    sse_payload = {
                                        "id": req_id,
                                        "object": "chat.completion.chunk",
                                        "created": created_time,
                                        "model": body.model,
                                        "choices": [
                                            {
                                                "index": 0,
                                                "delta": {"role": "assistant", "content": trunc_text},
                                                "finish_reason": "stop"
                                            }
                                        ]
                                    }
                                    yield f"data: {json.dumps(sse_payload)}\n\n"
                                    yield "data: [DONE]\n\n"
                                    break

                            # Partial Match Lookahead
                            last_open = text_buffer.rfind("<")
                            if last_open != -1 and "<tool_calls>".startswith(text_buffer[last_open:]):
                                # Hold the buffer, waiting for the next chunk to confirm or deny
                                pass
                            else:
                                sse_payload = {
                                    "id": req_id,
                                    "object": "chat.completion.chunk",
                                    "created": created_time,
                                    "model": body.model,
                                    "choices": [
                                        {
                                            "index": 0,
                                            "delta": {"role": "assistant", "content": text_buffer},
                                            "finish_reason": None
                                        }
                                    ]
                                }
                                yield f"data: {json.dumps(sse_payload)}\n\n"
                                text_buffer = ""
                        else:
                            is_xml_done = False
                            if "<tool_calls>" in text_buffer:
                                if "</tool_calls>" in text_buffer:
                                    is_xml_done = True
                            else:
                                if "</tool_call>" in text_buffer:
                                    is_xml_done = True
                                    
                            if is_xml_done:
                                try:
                                    matches = list(re.finditer(r'<tool_call>\s*<name>(.*?)</name>\s*<arguments>(.*?)</arguments>\s*</tool_call>', text_buffer, re.DOTALL))
                                    if matches:
                                        tool_calls_payload = []
                                        for idx, match in enumerate(matches):
                                            name = match.group(1).strip() if match.group(1) else ""
                                            arguments = match.group(2).strip() if match.group(2) else ""
                                            tool_calls_payload.append({
                                                "index": idx,
                                                "id": f"call_{uuid.uuid4().hex}",
                                                "type": "function",
                                                "function": {
                                                    "name": name,
                                                    "arguments": arguments
                                                }
                                            })
                                        
                                        sse_payload = {
                                            "id": req_id,
                                            "object": "chat.completion.chunk",
                                            "created": created_time,
                                            "model": body.model,
                                            "choices": [
                                                {
                                                    "index": 0,
                                                    "delta": {
                                                        "tool_calls": tool_calls_payload
                                                    },
                                                    "finish_reason": None
                                                }
                                            ]
                                        }
                                        yield f"data: {json.dumps(sse_payload)}\n\n"
                                        has_yielded_tool = True
                                    else:
                                        # Regex failed to match despite </tool_call> being present
                                        sse_payload = {
                                            "id": req_id,
                                            "object": "chat.completion.chunk",
                                            "created": created_time,
                                            "model": body.model,
                                            "choices": [
                                                {
                                                    "index": 0,
                                                    "delta": {"role": "assistant", "content": text_buffer},
                                                    "finish_reason": None
                                                }
                                            ]
                                        }
                                        yield f"data: {json.dumps(sse_payload)}\n\n"
                                    
                                    # Clear the buffer and reset the flag
                                    text_buffer = ""
                                    is_buffering_tool = False
                                except Exception as e:
                                    # Fallback: safely yield text_buffer as normal content
                                    sse_payload = {
                                        "id": req_id,
                                        "object": "chat.completion.chunk",
                                        "created": created_time,
                                        "model": body.model,
                                        "choices": [
                                            {
                                                "index": 0,
                                                "delta": {"role": "assistant", "content": text_buffer},
                                                "finish_reason": None
                                            }
                                        ]
                                    }
                                    yield f"data: {json.dumps(sse_payload)}\n\n"
                                    text_buffer = ""
                                    is_buffering_tool = False
                    elif chunk_type == "done":
                        # Output finish reason
                        end_payload = {
                            "id": req_id,
                            "object": "chat.completion.chunk",
                            "created": created_time,
                            "model": body.model,
                            "choices": [
                                {
                                    "index": 0,
                                    "delta": {},
                                    "finish_reason": "tool_calls" if has_yielded_tool else "stop"
                                }
                            ],
                            "usage": {
                                "prompt_tokens": chunk.get("promptTokens", 0),
                                "completion_tokens": chunk.get("responseTokens", 0),
                                "total_tokens": chunk.get("totalTokens", 0),
                            }
                        }
                        yield f"data: {json.dumps(end_payload)}\n\n"
                        yield "data: [DONE]\n\n"
            except Exception as e:
                # Log error, send error chunk
                error_payload = {
                    "error": {
                        "message": str(e),
                        "type": "server_error",
                        "param": None,
                        "code": None
                    }
                }
                yield f"data: {json.dumps(error_payload)}\n\n"
                
        return StreamingResponse(event_generator(), media_type="text/event-stream")

    else:
        # response_or_stream is a dict representing the "done" chunk
        response_text = response_or_stream.get("response", "")
        p_tokens = response_or_stream.get("promptTokens", 0)
        c_tokens = response_or_stream.get("responseTokens", 0)
        t_tokens = response_or_stream.get("totalTokens", 0)
        
        finish_reason = "stop"
        tool_calls = None
        
        # Stop Sequence Truncator Logic
        if os.getenv("ENABLE_STOP_TRUNCATION", "False").lower() == "true" and body.stop:
            stop_sequences = body.stop if isinstance(body.stop, list) else [body.stop]
            earliest_idx = -1
            for stop_seq in stop_sequences:
                idx = response_text.find(stop_seq)
                if idx != -1:
                    if earliest_idx == -1 or idx < earliest_idx:
                        earliest_idx = idx
            
            if earliest_idx != -1:
                response_text = response_text[:earliest_idx]
                
        reasoning = None
        reasoning_content = None
        if "<thought>" in response_text and "</thought>" in response_text:
            match = re.search(r'<thought>(.*?)</thought>', response_text, re.DOTALL)
            if match:
                reasoning = match.group(1).strip()
                reasoning_content = reasoning
                response_text = re.sub(r'<thought>.*?</thought>', '', response_text, flags=re.DOTALL).strip()

        # Check for tool_call XML
        if "<tool_call>" in response_text or "<tool_calls>" in response_text:
            matches = list(re.finditer(r'<tool_call>\s*<name>(.*?)</name>\s*<arguments>(.*?)</arguments>\s*</tool_call>', response_text, re.DOTALL))
            if matches:
                tool_calls = []
                for match in matches:
                    name = match.group(1).strip()
                    arguments = match.group(2).strip()
                    tool_calls.append(
                        ToolCall(
                            id=f"call_{uuid.uuid4().hex}",
                            type="function",
                            function=ToolCallFunction(name=name, arguments=arguments)
                        )
                    )
                
                # Remove the XML from response_text
                response_text = re.sub(r'<tool_call>.*?</tool_call>', '', response_text, flags=re.DOTALL)
                response_text = response_text.replace('<tool_calls>', '').replace('</tool_calls>', '').strip()
                
                finish_reason = "tool_calls"
        
        return ChatCompletionResponse(
            id=req_id,
            created=created_time,
            model=body.model,
            choices=[
                ChoiceNonStreaming(
                    index=0,
                    message=Message(role="assistant", content=response_text if response_text else None, reasoning=reasoning, reasoning_content=reasoning_content, tool_calls=tool_calls),
                    finish_reason=finish_reason
                )
            ],
            usage=Usage(
                prompt_tokens=p_tokens,
                completion_tokens=c_tokens,
                total_tokens=t_tokens
            )
        )
