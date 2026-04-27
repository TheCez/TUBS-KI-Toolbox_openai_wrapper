import base64
import uuid
from typing import Tuple, List, Any, Optional
from app.services.tool_error_guidance import guidance_for_tool_errors, guidance_for_tool_successes
from app.models.anthropic import Message
from app.services.translation import (
    extract_text_from_content,
    extract_tool_calls_from_content,
    extract_tool_results_from_content,
    has_tool_result_blocks,
)

def compile_anthropic_messages_to_prompt(messages: List[Message]) -> str:
    compiled_prompt = ""
    deferred_hints: list[str] = []
    deferred_success_hints: list[str] = []
    for msg in messages:
        role = msg.role.capitalize()

        tool_calls = extract_tool_calls_from_content(msg.content)
        if tool_calls:
            tool_lines = [
                f"{tool_call['name']}({tool_call['arguments']}) [id={tool_call['id']}]"
                for tool_call in tool_calls
            ]
            compiled_prompt += "[Tool Intention]: " + "; ".join(tool_lines) + "\n"

        content_text = extract_text_from_content(msg.content)
        tool_results = extract_tool_results_from_content(msg.content)
        if tool_results:
            role = "Tool Result"
            for tool_result in tool_results:
                marker = "ERROR" if tool_result["is_error"] else "OK"
                tool_id = f" id={tool_result['id']}" if tool_result["id"] else ""
                if tool_result["text"]:
                    compiled_prompt += f"[Tool Result {marker}{tool_id}]: {tool_result['text'].strip()}\n"
            content_text = ""
            deferred_hints.extend(guidance_for_tool_errors(tool_results))
            deferred_success_hints.extend(guidance_for_tool_successes(tool_results))
        elif has_tool_result_blocks(msg.content):
            role = "Tool Result"

        if content_text:
            compiled_prompt += f"[{role}]: {content_text.strip()}\n"

    for hint in dict.fromkeys(deferred_hints):
        compiled_prompt += f"[Wrapper Repair Hint]: {hint}\n"
    for hint in dict.fromkeys(deferred_success_hints):
        compiled_prompt += f"[Wrapper Completion Hint]: {hint}\n"

    return compiled_prompt.strip()

def extract_anthropic_base64_image(source: Any) -> Tuple[Optional[str], Optional[bytes], Optional[str]]:
    if not source: return None, None, None
    if isinstance(source, dict):
        base64_data = source.get("data")
        mime = source.get("media_type", "image/png")
    else:
        base64_data = getattr(source, "data", None)
        mime = getattr(source, "media_type", "image/png")
        
    if not base64_data or not mime:
        return None, None, None
        
    try:
        content = base64.b64decode(base64_data)
        extension = mime.split("/")[1]
        if extension == "jpeg":
            extension = "jpg"
        filename = f"{uuid.uuid4()}.{extension}"
        return filename, content, mime
    except Exception:
        return None, None, None

def get_images_from_anthropic_messages(messages: List[Message]) -> List[Tuple[str, bytes, str]]:
    images = []
    for msg in messages:
        if isinstance(msg.content, list):
            for part in msg.content:
                if getattr(part, "type", None) == "image":
                    fname, fbytes, fmime = extract_anthropic_base64_image(getattr(part, "source", None))
                    if fname and fbytes and fmime:
                        images.append((fname, fbytes, fmime))
                elif isinstance(part, dict) and part.get("type") == "image":
                    fname, fbytes, fmime = extract_anthropic_base64_image(part.get("source", None))
                    if fname and fbytes and fmime:
                        images.append((fname, fbytes, fmime))
    return images
