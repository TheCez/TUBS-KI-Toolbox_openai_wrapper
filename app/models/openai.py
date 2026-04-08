from typing import List, Optional, Union, Dict, Any, Literal
from pydantic import BaseModel, Field
from app.models.tubs import CloudModel, LocalModel

class ImageUrl(BaseModel):
    url: str
    detail: Optional[Literal["low", "high", "auto"]] = "auto"

class ContentPartText(BaseModel):
    type: Literal["text"]
    text: str

class ContentPartImage(BaseModel):
    type: Literal["image_url"]
    image_url: ImageUrl

class ToolCallFunction(BaseModel):
    name: str
    arguments: str

class ToolCall(BaseModel):
    id: str
    type: Literal["function"] = "function"
    function: ToolCallFunction

class Message(BaseModel):
    role: Literal["system", "user", "assistant", "tool", "developer"]
    content: Union[str, List[Union[ContentPartText, ContentPartImage]], None] = None
    reasoning: Optional[str] = None
    reasoning_content: Optional[str] = None
    name: Optional[str] = None
    tool_calls: Optional[List[ToolCall]] = None
    tool_call_id: Optional[str] = None

class ChatCompletionRequest(BaseModel):
    model: Union[CloudModel, LocalModel]
    messages: List[Message]
    stream: Optional[bool] = False
    max_tokens: Optional[int] = None
    temperature: Optional[float] = 1.0
    top_p: Optional[float] = 1.0
    n: Optional[int] = 1
    stop: Optional[Union[str, List[str]]] = None
    presence_penalty: Optional[float] = 0.0
    frequency_penalty: Optional[float] = 0.0
    user: Optional[str] = None
    response_format: Optional[Dict[str, Any]] = None
    tools: Optional[List[Dict[str, Any]]] = None
    tool_choice: Optional[Union[str, Dict[str, Any]]] = None

class ChoiceNonStreaming(BaseModel):
    index: int
    message: Message
    finish_reason: Optional[Literal["stop", "length", "tool_calls", "content_filter"]] = "stop"

class Usage(BaseModel):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int

class ChatCompletionResponse(BaseModel):
    id: str
    object: Literal["chat.completion"] = "chat.completion"
    created: int
    model: str
    choices: List[ChoiceNonStreaming]
    usage: Optional[Usage] = None

class Delta(BaseModel):
    role: Optional[Literal["system", "user", "assistant", "tool"]] = None
    content: Optional[str] = None
    reasoning: Optional[str] = None
    reasoning_content: Optional[str] = None

class ChoiceStreaming(BaseModel):
    index: int
    delta: Delta
    finish_reason: Optional[Literal["stop", "length", "tool_calls", "content_filter"]] = None

class ChatCompletionChunk(BaseModel):
    id: str
    object: Literal["chat.completion.chunk"] = "chat.completion.chunk"
    created: int
    model: str
    choices: List[ChoiceStreaming]
    usage: Optional[Usage] = None

class ErrorDetail(BaseModel):
    message: str
    type: str
    param: Optional[str] = None
    code: Optional[str] = None

class ErrorResponse(BaseModel):
    error: ErrorDetail
