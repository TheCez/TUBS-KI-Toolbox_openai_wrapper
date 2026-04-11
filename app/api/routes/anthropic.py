import os
import time
import json
import uuid
from fastapi import APIRouter, Depends, Request
from fastapi.responses import StreamingResponse
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
import re

from app.models.anthropic import MessageRequest, MessageResponse, MessageResponseUsage, TextContentBlock, ToolUseContentBlock
from app.models.tubs import TubsChatRequest
from app.services.anthropic_translation import compile_anthropic_messages_to_prompt, get_images_from_anthropic_messages
from app.services.tubs_client import async_send_tubs_request
from app.api.routes.models import get_anthropic_model_map

router = APIRouter()
security = HTTPBearer()

@router.post("/messages", response_model=MessageResponse)
async def anthropic_messages(
    request: Request,
    body: MessageRequest,
    credentials: HTTPAuthorizationCredentials = Depends(security)
):
    token = credentials.credentials

    # Compile prompt
    prompt_string = compile_anthropic_messages_to_prompt(body.messages)
    
    # Extract images if any
    images = get_images_from_anthropic_messages(body.messages)

    custom_instructions = ""
    if body.system:
        if isinstance(body.system, str):
            custom_instructions += body.system + "\n\n"
        elif isinstance(body.system, list):
            custom_instructions += "\n".join([part.text for part in body.system if hasattr(part, 'text')]) + "\n\n"

    if body.tools:
        tools_str = json.dumps([t.model_dump() for t in body.tools])
        xml_rule = (
            "You have access to the tools provided in the current request. To use any tool, you MUST follow this XML format. If you need to trigger multiple tools simultaneously, you MUST wrap them in a <tool_calls> root element and output all of them back-to-back before stopping. Use this EXACT format:\n"
            "<tool_calls><tool_call><name>tool_name</name><arguments>{\"actual_key\": \"actual_value\"}</arguments></tool_call></tool_calls>\n"
            "The <arguments> MUST contain raw, valid JSON that perfectly matches the provided tool schema. DO NOT wrap the arguments inside a {\"json\": ...} parent object.\n"
            "CRITICAL RULES:\n"
            "1. CRITICAL: When outputting XML tags like <tool_call>, <name>, or </arguments>, you MUST NOT escape the forward slash. Output raw, plain XML tags only.\n"
            "2. When creating a new file, DO NOT use apply_patch if possible. Use bash with a heredoc (e.g., cat << \"EOF\" > file.ext) or write_file if available. Only use apply_patch for modifying existing code blocks or no other valid tool is available.\n"
            "3. Ensure the JSON inside <arguments> is valid, single-line or properly escaped, and contains no trailing commas or unescaped characters that could break a json.loads() call. Be extremely precise with newlines and special characters.\n"
            "4. Avoid generating massive files in a single tool call if they contain complex nested structures. If a file is over 100 lines, consider writing it in logical stages if the client tools support it.\n"
            "5. Your output must contain ONLY the XML blocks. No conversational filler like \"Here is your code\" before or after the <tool_calls> tag, as this can confuse the client parser.\n"
            f"Available tools: {tools_str}\n"
        )
        custom_instructions += xml_rule

    custom_instructions = custom_instructions.strip() if custom_instructions else None

    # Map Anthropic request model to TU-BS model
    anthropic_map = get_anthropic_model_map()
    tubs_target_model = anthropic_map.get(body.model, body.model)

    # Build Tubs request payload
    tubs_payload = TubsChatRequest(
        thread=None,
        prompt=prompt_string,
        model=tubs_target_model,
        customInstructions=custom_instructions,
    ).model_dump(exclude_none=True)

    # Perform request
    response_or_stream = await async_send_tubs_request(
        payload=tubs_payload,
        images=images,
        bearer_token=token,
        stream=bool(body.stream)
    )

    req_id = f"msg_{uuid.uuid4().hex}"
    
    if body.stream:
        async def event_generator():
            text_buffer = ""
            is_buffering_tool = False
            has_yielded_tool = False
            
            try:
                # Anthropic message_start event
                message_start_payload = {
                    "type": "message_start",
                    "message": {
                        "id": req_id,
                        "type": "message",
                        "role": "assistant",
                        "model": body.model,
                        "content": [],
                        "stop_reason": None,
                        "stop_sequence": None,
                        "usage": {"input_tokens": 0, "output_tokens": 0}
                    }
                }
                yield f"event: message_start\ndata: {json.dumps(message_start_payload)}\n\n"
                
                content_block_idx = 0
                
                # We yield a content block start for the text
                content_block_start = {
                    "type": "content_block_start",
                    "index": content_block_idx,
                    "content_block": {"type": "text", "text": ""}
                }
                yield f"event: content_block_start\ndata: {json.dumps(content_block_start)}\n\n"

                async for chunk in response_or_stream:
                    chunk_type = chunk.get("type")
                    if chunk_type == "chunk":
                        content = chunk.get("content", "")
                        text_buffer += content
                        
                        if "<tool_call>" in text_buffer or "<tool_calls>" in text_buffer:
                            is_buffering_tool = True
                            
                        if not is_buffering_tool:
                            # Stop Sequence Truncator Logic
                            if os.getenv("ENABLE_STOP_TRUNCATION", "False").lower() == "true" and body.stop_sequences:
                                earliest_idx = -1
                                for stop_seq in body.stop_sequences:
                                    idx = text_buffer.find(stop_seq)
                                    if idx != -1:
                                        if earliest_idx == -1 or idx < earliest_idx:
                                            earliest_idx = idx
                                
                                if earliest_idx != -1:
                                    trunc_text = text_buffer[:earliest_idx]
                                    delta_payload = {
                                        "type": "content_block_delta",
                                        "index": content_block_idx,
                                        "delta": {"type": "text_delta", "text": trunc_text}
                                    }
                                    yield f"event: content_block_delta\ndata: {json.dumps(delta_payload)}\n\n"
                                    
                                    # Terminate directly
                                    yield f"event: content_block_stop\ndata: {json.dumps({'type': 'content_block_stop', 'index': content_block_idx})}\n\n"
                                    
                                    usage_payload = {
                                        "type": "message_delta",
                                        "delta": {"stop_reason": "stop_sequence", "stop_sequence": None},
                                        "usage": {"output_tokens": 0}
                                    }
                                    yield f"event: message_delta\ndata: {json.dumps(usage_payload)}\n\n"
                                    
                                    yield f"event: message_stop\ndata: {json.dumps({'type': 'message_stop'})}\n\n"
                                    break

                            # Partial Match Lookahead
                            last_open = text_buffer.rfind("<")
                            if last_open != -1 and "<tool_calls>".startswith(text_buffer[last_open:]):
                                pass
                            else:
                                delta_payload = {
                                    "type": "content_block_delta",
                                    "index": content_block_idx,
                                    "delta": {"type": "text_delta", "text": text_buffer}
                                }
                                yield f"event: content_block_delta\ndata: {json.dumps(delta_payload)}\n\n"
                                text_buffer = ""
                        else:
                            is_xml_done = False
                            if "<tool_calls>" in text_buffer:
                                if re.search(r'<\\?/tool_calls>', text_buffer):
                                    is_xml_done = True
                            else:
                                if re.search(r'<\\?/tool_call>', text_buffer):
                                    is_xml_done = True
                                    
                            if is_xml_done:
                                try:
                                    matches = list(re.finditer(r'<tool_call>\s*<name>(.*?)<\\?/name>\s*<arguments>(.*?)<\\?/arguments>\s*<\\?/tool_call>', text_buffer, re.DOTALL))
                                    if matches:
                                        # Stop text block
                                        yield f"event: content_block_stop\ndata: {json.dumps({'type': 'content_block_stop', 'index': content_block_idx})}\n\n"
                                        content_block_idx += 1
                                        
                                        for idx, match in enumerate(matches):
                                            name = match.group(1).strip() if match.group(1) else ""
                                            arguments = match.group(2).strip() if match.group(2) else ""
                                            arguments = arguments.replace('\\/', '/')
                                            
                                            tool_id = f"toolu_{uuid.uuid4().hex}"
                                            
                                            # Content block start for tool
                                            yield f"event: content_block_start\ndata: {json.dumps({'type': 'content_block_start', 'index': content_block_idx, 'content_block': {'type': 'tool_use', 'id': tool_id, 'name': name}})}\n\n"
                                            
                                            # Content block delta for tool input
                                            yield f"event: content_block_delta\ndata: {json.dumps({'type': 'content_block_delta', 'index': content_block_idx, 'delta': {'type': 'input_json_delta', 'partial_json': arguments}})}\n\n"
                                            
                                            # Content block stop
                                            yield f"event: content_block_stop\ndata: {json.dumps({'type': 'content_block_stop', 'index': content_block_idx})}\n\n"
                                            content_block_idx += 1
                                        
                                        has_yielded_tool = True
                                    else:
                                        delta_payload = {
                                            "type": "content_block_delta",
                                            "index": content_block_idx,
                                            "delta": {"type": "text_delta", "text": text_buffer}
                                        }
                                        yield f"event: content_block_delta\ndata: {json.dumps(delta_payload)}\n\n"
                                        
                                    text_buffer = ""
                                    is_buffering_tool = False
                                except Exception:
                                    delta_payload = {
                                        "type": "content_block_delta",
                                        "index": content_block_idx,
                                        "delta": {"type": "text_delta", "text": text_buffer}
                                    }
                                    yield f"event: content_block_delta\ndata: {json.dumps(delta_payload)}\n\n"
                                    text_buffer = ""
                                    is_buffering_tool = False
                                    
                    elif chunk_type == "done":
                        # End of text block if we didn't end it yet
                        if not has_yielded_tool:
                            yield f"event: content_block_stop\ndata: {json.dumps({'type': 'content_block_stop', 'index': content_block_idx})}\n\n"
                            
                        # Message delta with usage
                        usage_payload = {
                            "type": "message_delta",
                            "delta": {
                                "stop_reason": "tool_use" if has_yielded_tool else "end_turn",
                                "stop_sequence": None
                            },
                            "usage": {
                                "output_tokens": chunk.get("responseTokens", 0)
                            }
                        }
                        yield f"event: message_delta\ndata: {json.dumps(usage_payload)}\n\n"
                        yield f"event: message_stop\ndata: {json.dumps({'type': 'message_stop'})}\n\n"
                        
            except Exception as e:
                # Log error
                error_payload = {
                    "type": "error",
                    "error": {
                        "type": "api_error",
                        "message": str(e)
                    }
                }
                yield f"event: error\ndata: {json.dumps(error_payload)}\n\n"
                
        return StreamingResponse(event_generator(), media_type="text/event-stream")

    else:
        # response_or_stream is a dict representing the "done" chunk
        response_text = response_or_stream.get("response", "")
        p_tokens = response_or_stream.get("promptTokens", 0)
        c_tokens = response_or_stream.get("responseTokens", 0)
        
        stop_reason = "end_turn"
        content_blocks = []
        
        # Stop Sequence Truncator Logic
        if os.getenv("ENABLE_STOP_TRUNCATION", "False").lower() == "true" and body.stop_sequences:
            earliest_idx = -1
            for stop_seq in body.stop_sequences:
                idx = response_text.find(stop_seq)
                if idx != -1:
                    if earliest_idx == -1 or idx < earliest_idx:
                        earliest_idx = idx
            
            if earliest_idx != -1:
                response_text = response_text[:earliest_idx]
                stop_reason = "stop_sequence"
                
        # Check for tool_call XML
        if "<tool_call>" in response_text or "<tool_calls>" in response_text:
            matches = list(re.finditer(r'<tool_call>\s*<name>(.*?)<\\?/name>\s*<arguments>(.*?)<\\?/arguments>\s*<\\?/tool_call>', response_text, re.DOTALL))
            if matches:
                # Remove XML from response_text for the text block
                text_content = re.sub(r'<tool_call>.*?<\\?/tool_call>', '', response_text, flags=re.DOTALL)
                text_content = re.sub(r'<\\?/tool_calls>', '', text_content).replace('<tool_calls>', '').strip()
                
                if text_content:
                    content_blocks.append(TextContentBlock(type="text", text=text_content))
                    
                for match in matches:
                    name = match.group(1).strip()
                    arguments = match.group(2).strip()
                    arguments = arguments.replace('\\/', '/')
                    try:
                        args_json = json.loads(arguments)
                    except:
                        args_json = {}
                        
                    content_blocks.append(
                        ToolUseContentBlock(
                            type="tool_use",
                            id=f"toolu_{uuid.uuid4().hex}",
                            name=name,
                            input=args_json
                        )
                    )
                
                stop_reason = "tool_use"
            else:
                content_blocks.append(TextContentBlock(type="text", text=response_text))
        else:
            if response_text:
                content_blocks.append(TextContentBlock(type="text", text=response_text))
            
        return MessageResponse(
            id=req_id,
            type="message",
            role="assistant",
            content=content_blocks,
            model=body.model,
            stop_reason=stop_reason,
            usage=MessageResponseUsage(input_tokens=p_tokens, output_tokens=c_tokens)
        )
