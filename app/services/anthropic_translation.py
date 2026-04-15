import base64
import uuid
from typing import Tuple, List, Optional, Any
from app.models.anthropic import Message

def compile_anthropic_messages_to_prompt(messages: List[Message]) -> str:
    compiled_prompt = ""
    for msg in messages:
        role = msg.role.capitalize()
        content_text = ""
        
        if isinstance(msg.content, str):
            content_text = msg.content
        elif isinstance(msg.content, list):
            for part in msg.content:
                if getattr(part, "type", None) == "text":
                    content_text += getattr(part, "text", "") + "\n"
                elif isinstance(part, dict) and part.get("type") == "text":
                    content_text += part.get("text", "") + "\n"
                elif getattr(part, "type", None) == "tool_use":
                    tool_id = getattr(part, "id", "")
                    tool_input = getattr(part, "input", {})
                    content_text += (
                        f"[Tool Intention]: {getattr(part, 'name', '')}({tool_input}) [id={tool_id}]\n"
                    )
                elif isinstance(part, dict) and part.get("type") == "tool_use":
                    content_text += (
                        f"[Tool Intention]: {part.get('name', '')}({part.get('input', {})}) [id={part.get('id', '')}]\n"
                    )
                elif getattr(part, "type", None) == "tool_result":
                    res = getattr(part, 'content', '')
                    if isinstance(res, list):
                        res = " ".join([getattr(r, 'text', '') for r in res if getattr(r, 'type', None) == 'text'])
                    role = "Tool Result"
                    content_text += str(res) + "\n"
                elif isinstance(part, dict) and part.get("type") == "tool_result":
                    res = part.get("content", "")
                    if isinstance(res, list):
                        res = " ".join([r.get("text", "") for r in res if r.get("type") == "text"])
                    role = "Tool Result"
                    content_text += str(res) + "\n"
        
        if content_text:
            compiled_prompt += f"[{role}]: {content_text.strip()}\n"
            
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
