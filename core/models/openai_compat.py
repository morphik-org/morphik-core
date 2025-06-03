"""
OpenAI API compatible models for Morphik.
Provides compatibility with OpenAI SDK while maintaining Morphik's functionality.
"""

from typing import Any, Dict, List, Literal, Optional, Union

from pydantic import BaseModel, Field


class OpenAIMessage(BaseModel):
    """OpenAI chat completion message format."""
    
    role: Literal["system", "user", "assistant", "function", "tool"]
    content: Optional[Union[str, List[Dict[str, Any]]]] = None
    name: Optional[str] = None
    function_call: Optional[Dict[str, Any]] = None
    tool_calls: Optional[List[Dict[str, Any]]] = None
    tool_call_id: Optional[str] = None


class OpenAIFunctionCall(BaseModel):
    """OpenAI function call format."""
    
    name: str
    arguments: str


class OpenAIFunction(BaseModel):
    """OpenAI function definition format."""
    
    name: str
    description: Optional[str] = None
    parameters: Optional[Dict[str, Any]] = None


class OpenAITool(BaseModel):
    """OpenAI tool definition format."""
    
    type: Literal["function"]
    function: OpenAIFunction


class OpenAIResponseFormat(BaseModel):
    """OpenAI response format specification."""
    
    type: Literal["text", "json_object", "json_schema"] = "text"
    json_schema: Optional[Dict[str, Any]] = None


class OpenAIStreamOptions(BaseModel):
    """OpenAI streaming options."""
    
    include_usage: Optional[bool] = False


class OpenAIChatCompletionRequest(BaseModel):
    """OpenAI chat completion request format."""
    
    model: str
    messages: List[OpenAIMessage]
    frequency_penalty: Optional[float] = Field(default=0, ge=-2.0, le=2.0)
    logit_bias: Optional[Dict[str, int]] = None
    logprobs: Optional[bool] = None
    top_logprobs: Optional[int] = Field(default=None, ge=0, le=20)
    max_tokens: Optional[int] = Field(default=None, ge=1)
    max_completion_tokens: Optional[int] = Field(default=None, ge=1)
    n: Optional[int] = Field(default=1, ge=1, le=128)
    presence_penalty: Optional[float] = Field(default=0, ge=-2.0, le=2.0)
    response_format: Optional[OpenAIResponseFormat] = None
    seed: Optional[int] = None
    service_tier: Optional[Literal["auto", "default"]] = None
    stop: Optional[Union[str, List[str]]] = None
    stream: Optional[bool] = False
    stream_options: Optional[OpenAIStreamOptions] = None
    temperature: Optional[float] = Field(default=1.0, ge=0, le=2.0)
    top_p: Optional[float] = Field(default=1.0, ge=0, le=1.0)
    tools: Optional[List[OpenAITool]] = None
    tool_choice: Optional[Union[str, Dict[str, Any]]] = None
    parallel_tool_calls: Optional[bool] = None
    user: Optional[str] = None
    
    # Morphik-specific extensions
    chat_id: Optional[str] = None
    folder_name: Optional[str] = None
    use_rag: Optional[bool] = True
    top_k: Optional[int] = 5


class OpenAIUsage(BaseModel):
    """OpenAI usage statistics."""
    
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    prompt_tokens_details: Optional[Dict[str, Any]] = None
    completion_tokens_details: Optional[Dict[str, Any]] = None


class OpenAIChoice(BaseModel):
    """OpenAI completion choice."""
    
    index: int
    message: OpenAIMessage
    logprobs: Optional[Dict[str, Any]] = None
    finish_reason: Optional[Literal["stop", "length", "function_call", "tool_calls", "content_filter"]]


class OpenAIChatCompletionResponse(BaseModel):
    """OpenAI chat completion response format."""
    
    id: str
    object: Literal["chat.completion"] = "chat.completion"
    created: int
    model: str
    system_fingerprint: Optional[str] = None
    choices: List[OpenAIChoice]
    usage: Optional[OpenAIUsage] = None
    service_tier: Optional[str] = None


class OpenAIStreamChoice(BaseModel):
    """OpenAI streaming completion choice."""
    
    index: int
    delta: OpenAIMessage
    logprobs: Optional[Dict[str, Any]] = None
    finish_reason: Optional[Literal["stop", "length", "function_call", "tool_calls", "content_filter"]]


class OpenAIChatCompletionChunk(BaseModel):
    """OpenAI chat completion streaming chunk."""
    
    id: str
    object: Literal["chat.completion.chunk"] = "chat.completion.chunk"
    created: int
    model: str
    system_fingerprint: Optional[str] = None
    choices: List[OpenAIStreamChoice]
    usage: Optional[OpenAIUsage] = None
    service_tier: Optional[str] = None


class OpenAIModel(BaseModel):
    """OpenAI model information."""
    
    id: str
    object: Literal["model"] = "model"
    created: int
    owned_by: str


class OpenAIModelList(BaseModel):
    """OpenAI model list response."""
    
    object: Literal["list"] = "list"
    data: List[OpenAIModel]


class OpenAIError(BaseModel):
    """OpenAI error response."""
    
    message: str
    type: str
    param: Optional[str] = None
    code: Optional[str] = None


class OpenAIErrorResponse(BaseModel):
    """OpenAI error response wrapper."""
    
    error: OpenAIError