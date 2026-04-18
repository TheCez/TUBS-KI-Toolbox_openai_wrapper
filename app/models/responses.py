from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional, Union

from pydantic import BaseModel, ConfigDict, Field, model_validator

from app.models.tubs import CloudModel, LocalModel


class StrictBaseModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class ResponseInputText(StrictBaseModel):
    model_config = ConfigDict(extra="allow")
    type: Literal["input_text", "text"]
    text: str


class ResponseInputImage(StrictBaseModel):
    model_config = ConfigDict(extra="allow")
    type: Literal["input_image", "image_url"]
    image_url: str
    detail: Optional[Literal["low", "high", "auto"]] = "auto"


class ResponseReasoningConfig(StrictBaseModel):
    effort: Optional[Literal["none", "low", "medium", "high", "xhigh"]] = None
    summary: Optional[Literal["auto", "concise", "detailed"]] = None


class ResponseTextConfig(StrictBaseModel):
    format: Optional[Dict[str, Any]] = None


class ResponseFunctionTool(StrictBaseModel):
    type: Literal["function"]
    name: str
    description: Optional[str] = None
    parameters: Dict[str, Any] = Field(default_factory=dict)
    strict: Optional[bool] = None


class ResponseFunctionToolChoice(StrictBaseModel):
    type: Literal["function"]
    name: str


class ResponseInputMessage(StrictBaseModel):
    role: Literal["system", "developer", "user", "assistant", "tool"]
    content: Union[str, List[Union[ResponseInputText, ResponseInputImage, Dict[str, Any]]], None] = None
    type: Literal["message"] = "message"


class ResponseShorthandInputMessage(StrictBaseModel):
    content: Union[str, List[Union[ResponseInputText, ResponseInputImage, Dict[str, Any]]]]
    role: Optional[Literal["system", "developer", "user", "assistant", "tool"]] = None
    type: Optional[str] = None


class ResponseFunctionCallOutput(StrictBaseModel):
    type: Literal["function_call_output"]
    call_id: str
    output: Union[str, Dict[str, Any], List[Any]]


class ResponseFunctionCall(StrictBaseModel):
    type: Literal["function_call"]
    call_id: str
    name: str
    arguments: str


ResponseInputItem = Union[
    ResponseInputMessage,
    ResponseShorthandInputMessage,
    ResponseFunctionCall,
    ResponseFunctionCallOutput,
    ResponseInputText,
    ResponseInputImage,
]


class ResponseCreateRequest(StrictBaseModel):
    model: Union[CloudModel, LocalModel, str]
    input: Union[str, List[ResponseInputItem]]
    instructions: Optional[str] = None
    stream: bool = False
    store: Optional[bool] = None
    include: Optional[List[str]] = None
    truncation: Optional[Literal["auto", "disabled"]] = None
    background: Optional[bool] = None
    service_tier: Optional[str] = None
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    max_output_tokens: Optional[int] = None
    max_tool_calls: Optional[int] = None
    tools: Optional[List[ResponseFunctionTool]] = None
    tool_choice: Optional[Union[Literal["auto", "none", "required"], ResponseFunctionToolChoice]] = None
    parallel_tool_calls: Optional[bool] = None
    reasoning: Optional[ResponseReasoningConfig] = None
    text: Optional[ResponseTextConfig] = None
    metadata: Optional[Dict[str, str]] = None
    previous_response_id: Optional[str] = None
    user: Optional[str] = None

    @model_validator(mode="after")
    def validate_tool_choice(self) -> "ResponseCreateRequest":
        if isinstance(self.tool_choice, ResponseFunctionToolChoice) and not self.tools:
            raise ValueError("tool_choice requires tools to be provided")
        return self
