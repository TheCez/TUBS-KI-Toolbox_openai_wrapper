import base64
import json
import uuid
from typing import Tuple, List, Optional, Any, Iterable
from app.models.openai import Message, ContentPartImage

TEXT_BLOCK_TYPES = {"text", "input_text", "output_text"}
TOOL_RESULT_BLOCK_TYPES = {"tool_result", "function_call_output"}


def _as_dict(value: Any) -> Optional[dict]:
    if isinstance(value, dict):
        return value
    if hasattr(value, "model_dump"):
        return value.model_dump(exclude_none=True)
    return None


def _stringify_tool_payload(value: Any) -> str:
    if isinstance(value, str):
        return value
    if value is None:
        return ""
    try:
        return json.dumps(value, ensure_ascii=False)
    except TypeError:
        return str(value)


def _extract_text_from_content_value(value: Any) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        parts = []
        for item in value:
            text = extract_text_from_content(item)
            if text:
                parts.append(text)
        return "\n".join(parts).strip()
    return ""


def iter_content_dicts(content: Any) -> Iterable[dict]:
    if not isinstance(content, list):
        return []
    normalized = []
    for part in content:
        part_dict = _as_dict(part)
        if part_dict:
            normalized.append(part_dict)
    return normalized


def extract_text_from_content(content: Any) -> str:
    if isinstance(content, str):
        return content

    parts: List[str] = []
    for part_dict in iter_content_dicts(content):
        part_type = part_dict.get("type")
        if part_type in TEXT_BLOCK_TYPES:
            text = part_dict.get("text")
            if isinstance(text, str) and text:
                parts.append(text)
        elif part_type in TOOL_RESULT_BLOCK_TYPES:
            nested = part_dict.get("content", part_dict.get("output"))
            nested_text = _extract_text_from_content_value(nested)
            if nested_text:
                parts.append(nested_text)

    return "\n".join(parts).strip()


def extract_tool_calls_from_content(content: Any) -> List[dict]:
    tool_calls = []
    for part_dict in iter_content_dicts(content):
        part_type = part_dict.get("type")
        if part_type == "tool_use":
            tool_calls.append(
                {
                    "id": part_dict.get("id", ""),
                    "name": part_dict.get("name", ""),
                    "arguments": _stringify_tool_payload(part_dict.get("input", {})),
                }
            )
        elif part_type == "function_call":
            tool_calls.append(
                {
                    "id": part_dict.get("call_id", part_dict.get("id", "")),
                    "name": part_dict.get("name", ""),
                    "arguments": _stringify_tool_payload(part_dict.get("arguments", "")),
                }
            )
    return tool_calls


def has_tool_result_blocks(content: Any) -> bool:
    return any(part_dict.get("type") in TOOL_RESULT_BLOCK_TYPES for part_dict in iter_content_dicts(content))


def compile_messages_to_prompt(messages: List[Message]) -> str:
    """
    Translates an OpenAI messages array into a single concatenated prompt string.
    This mimics how a stateless LLM receives conversation history.
    """
    compiled_prompt = ""
    for msg in messages:
        if msg.role.lower() in ["system", "developer"]:
            continue
            
        role = msg.role.capitalize()
        if msg.role.lower() == "tool":
            role = "Tool Result"
        content_text = ""

        if msg.role.lower() == "assistant" and msg.tool_calls:
            tool_lines = [
                f"{tool_call.function.name}({tool_call.function.arguments}) [id={tool_call.id}]"
                for tool_call in msg.tool_calls
            ]
            compiled_prompt += "[Assistant Tool Calls]: " + "; ".join(tool_lines) + "\n"

        content_tool_calls = extract_tool_calls_from_content(msg.content)
        if msg.role.lower() == "assistant" and content_tool_calls:
            tool_lines = [
                f"{tool_call['name']}({tool_call['arguments']}) [id={tool_call['id']}]"
                for tool_call in content_tool_calls
            ]
            compiled_prompt += "[Assistant Tool Calls]: " + "; ".join(tool_lines) + "\n"

        content_text = extract_text_from_content(msg.content)
        if msg.role.lower() != "tool" and has_tool_result_blocks(msg.content):
            role = "Tool Result"

        if content_text:
            compiled_prompt += f"[{role}]: {content_text.strip()}\n"
            
    # Remove the very last newline if exists
    return compiled_prompt.strip()

def extract_base64_image(image_url: str) -> Tuple[Optional[str], Optional[bytes], Optional[str]]:
    """
    Extracts binary data, filename, and mime type from a data URI base64 string.
    Format: data:image/jpeg;base64,/9j/4AAQSkZJRg...
    """
    if str(image_url).startswith("data:image/"):
        try:
            header, encoded = image_url.split(",", 1)
            mime = header.split(";")[0].split(":")[1]  # 'image/jpeg'
            content = base64.b64decode(encoded)
            extension = mime.split("/")[1]
            if extension == "jpeg":
                extension = "jpg"
            filename = f"{uuid.uuid4()}.{extension}"
            return filename, content, mime
        except Exception:
            return None, None, None
    return None, None, None

def get_images_from_messages(messages: List[Message]) -> List[Tuple[str, bytes, str]]:
    """
    Cycles through messages and extracts all images.
    Returns: List of (filename, file_bytes, mime_type)
    """
    images = []
    for msg in messages:
        if not isinstance(msg.content, list):
            continue
        for part in msg.content:
            if isinstance(part, ContentPartImage):
                url = part.image_url.url
            else:
                part_dict = _as_dict(part) or {}
                if part_dict.get("type") not in {"image_url", "input_image"}:
                    continue
                image_url = part_dict.get("image_url")
                if isinstance(image_url, dict):
                    url = image_url.get("url", "")
                else:
                    url = image_url or part_dict.get("url", "")
            fname, fbytes, fmime = extract_base64_image(url)
            if fname and fbytes and fmime:
                images.append((fname, fbytes, fmime))
    return images
