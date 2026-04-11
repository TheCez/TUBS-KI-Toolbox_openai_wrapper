from typing import List, Optional, Union, Literal, Dict, Any
from pydantic import BaseModel, Field

class ImageSource(BaseModel):
    type: Literal["base64"]
    media_type: str
    data: str

class TextContentBlock(BaseModel):
    type: Literal["text"]
    text: str

class ImageContentBlock(BaseModel):
    type: Literal["image"]
    source: ImageSource

class ToolUseContentBlock(BaseModel):
    type: Literal["tool_use"]
    id: str
    name: str
    input: Dict[str, Any]

class ToolResultContentBlock(BaseModel):
    type: Literal["tool_result"]
    tool_use_id: str
    content: Union[str, List[Union[TextContentBlock, ImageContentBlock]]]
    is_error: Optional[bool] = None

class Message(BaseModel):
    role: Literal["user", "assistant"]
    content: Union[str, List[Union[TextContentBlock, ImageContentBlock, ToolUseContentBlock, ToolResultContentBlock]]]

class Tool(BaseModel):
    name: str
    description: Optional[str] = None
    input_schema: Dict[str, Any]

class MessageRequest(BaseModel):
    model: str
    messages: List[Message]
    system: Optional[Union[str, List[TextContentBlock]]] = None
    max_tokens: Optional[int] = 1024
    stream: Optional[bool] = False
    tools: Optional[List[Tool]] = None
    temperature: Optional[float] = None
    stop_sequences: Optional[List[str]] = None

class MessageResponseUsage(BaseModel):
    input_tokens: int
    output_tokens: int

class MessageResponse(BaseModel):
    id: str
    type: Literal["message"]
    role: Literal["assistant"]
    content: List[Union[TextContentBlock, ToolUseContentBlock]]
    model: str
    stop_reason: Optional[str] = None
    stop_sequence: Optional[str] = None
    usage: MessageResponseUsage
