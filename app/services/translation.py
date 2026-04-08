import base64
import uuid
from typing import Tuple, List, Optional, Any
from app.models.openai import ChatCompletionRequest, Message, ContentPartText, ContentPartImage

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
        
        if isinstance(msg.content, str):
            content_text = msg.content
        elif isinstance(msg.content, list):
            # Parse array of content parts
            for part in msg.content:
                if isinstance(part, ContentPartText) or (hasattr(part, "type") and getattr(part, "type") == "text"):
                    # Check safely if standard dict/pydantic
                    content_text += part.text + "\n"
                elif isinstance(part, dict) and part.get("type") == "text":
                    content_text += part.get("text", "") + "\n"
        
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
        if isinstance(msg.content, list):
            for part in msg.content:
                if isinstance(part, ContentPartImage):
                    url = part.image_url.url
                    fname, fbytes, fmime = extract_base64_image(url)
                    if fname and fbytes and fmime:
                        images.append((fname, fbytes, fmime))
                elif isinstance(part, dict) and part.get("type") == "image_url":
                    url = part.get("image_url", {}).get("url", "")
                    fname, fbytes, fmime = extract_base64_image(url)
                    if fname and fbytes and fmime:
                        images.append((fname, fbytes, fmime))
    return images
