from typing import List, Optional, Union, Literal, Dict, Any
from pydantic import BaseModel, ConfigDict

class ImageSource(BaseModel):
    model_config = ConfigDict(extra="forbid")
    type: Literal["base64"]
    media_type: str
    data: str

class TextContentBlock(BaseModel):
    model_config = ConfigDict(extra="forbid")
    type: Literal["text"]
    text: str

class ImageContentBlock(BaseModel):
    model_config = ConfigDict(extra="forbid")
    type: Literal["image"]
    source: ImageSource

class ToolUseContentBlock(BaseModel):
    model_config = ConfigDict(extra="forbid")
    type: Literal["tool_use"]
    id: str
    name: str
    input: Dict[str, Any]

class ToolResultContentBlock(BaseModel):
    model_config = ConfigDict(extra="forbid")
    type: Literal["tool_result"]
    tool_use_id: str
    content: Union[str, List[Union[TextContentBlock, ImageContentBlock]]]
    is_error: Optional[bool] = None

class Message(BaseModel):
    model_config = ConfigDict(extra="forbid")
    role: Literal["user", "assistant"]
    content: Union[str, List[Union[TextContentBlock, ImageContentBlock, ToolUseContentBlock, ToolResultContentBlock]]]

class Tool(BaseModel):
    model_config = ConfigDict(extra="forbid")
    name: str
    description: Optional[str] = None
    input_schema: Dict[str, Any]

class ToolChoice(BaseModel):
    model_config = ConfigDict(extra="forbid")
    type: Literal["auto", "any", "tool", "none"]
    name: Optional[str] = None

class ThinkingConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    type: Optional[Literal["enabled", "disabled"]] = None
    budget_tokens: Optional[int] = None

class MessageRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    model: str
    messages: List[Message]
    system: Optional[Union[str, List[TextContentBlock]]] = None
    max_tokens: Optional[int] = 1024
    stream: Optional[bool] = False
    tools: Optional[List[Tool]] = None
    temperature: Optional[float] = None
    stop_sequences: Optional[List[str]] = None
    metadata: Optional[Dict[str, str]] = None
    service_tier: Optional[str] = None
    tool_choice: Optional[Union[ToolChoice, Dict[str, Any]]] = None
    thinking: Optional[ThinkingConfig] = None

class MessageResponseUsage(BaseModel):
    model_config = ConfigDict(extra="forbid")
    input_tokens: int
    output_tokens: int

class MessageResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")
    id: str
    type: Literal["message"]
    role: Literal["assistant"]
    content: List[Union[TextContentBlock, ToolUseContentBlock]]
    model: str
    stop_reason: Optional[str] = None
    stop_sequence: Optional[str] = None
    usage: MessageResponseUsage
