"""OpenAI-compatible API models for /v1/chat/completions endpoint."""

import os
from typing import Annotated, Any, Literal, Union
from pydantic import AfterValidator, BaseModel, Field


# =============================================================================
# Model Validation
# =============================================================================


def model_exists(model: str) -> str:
    """Validate that a model exists in the models directory."""
    # Strip :latest tag if present (Ollama-style)
    model_name = model.split(":")[0] if ":" in model else model

    if model_name.startswith("models/"):
        model_path = model_name
    else:
        model_path = os.path.join("./models", model_name)

    if not os.path.exists(model_path):
        raise ValueError(f"Model does not exist: {model}")
    return model_name


# =============================================================================
# Message Content Types
# =============================================================================


class TextContent(BaseModel):
    """Text content in a message."""
    type: Literal["text"]
    text: str


class ImageURL(BaseModel):
    """Image URL reference."""
    url: str = Field(description="URL or base64 data URI of the image")
    detail: Literal["auto", "low", "high"] | None = Field(
        default="auto",
        description="Image detail level"
    )


class ImageContent(BaseModel):
    """Image content in a message."""
    type: Literal["image_url"]
    image_url: ImageURL


ContentPart = Union[TextContent, ImageContent]


# =============================================================================
# Tool/Function Definitions
# =============================================================================


class FunctionParameters(BaseModel):
    """JSON Schema for function parameters."""
    type: Literal["object"] = "object"
    properties: dict[str, Any] | None = None
    required: list[str] | None = None


class FunctionDefinition(BaseModel):
    """Function definition for tools."""
    name: str = Field(description="The name of the function")
    description: str | None = Field(default=None, description="Description of what the function does")
    parameters: FunctionParameters | None = Field(default=None, description="JSON schema for parameters")


class Tool(BaseModel):
    """Tool definition."""
    type: Literal["function"] = "function"
    function: FunctionDefinition


# =============================================================================
# Tool Calls (in responses)
# =============================================================================


class FunctionCall(BaseModel):
    """Function call details."""
    name: str = Field(description="Name of the function to call")
    arguments: str = Field(description="JSON string of arguments")


class ToolCall(BaseModel):
    """Tool call made by the model."""
    id: str = Field(description="Unique ID for this tool call")
    index: int = Field(default=0, description="Index of this tool call")
    type: Literal["function"] = "function"
    function: FunctionCall


# =============================================================================
# Messages
# =============================================================================


class SystemMessage(BaseModel):
    """System message."""
    role: Literal["system"]
    content: str


class UserMessage(BaseModel):
    """User message with text or multimodal content."""
    role: Literal["user"]
    content: str | list[ContentPart]


class AssistantMessage(BaseModel):
    """Assistant message, possibly with tool calls."""
    role: Literal["assistant"]
    content: str | None = None
    tool_calls: list[ToolCall] | None = None


class ToolMessage(BaseModel):
    """Tool result message."""
    role: Literal["tool"]
    content: str
    tool_call_id: str = Field(description="ID of the tool call this responds to")


Message = Union[SystemMessage, UserMessage, AssistantMessage, ToolMessage]


# =============================================================================
# Response Format
# =============================================================================


class ResponseFormatText(BaseModel):
    """Plain text response format."""
    type: Literal["text"] = "text"


class ResponseFormatJSON(BaseModel):
    """JSON object response format."""
    type: Literal["json_object"] = "json_object"


class JSONSchema(BaseModel):
    """JSON schema definition."""
    name: str
    description: str | None = None
    schema_: dict[str, Any] | None = Field(default=None, alias="schema")
    strict: bool | None = None


class ResponseFormatJSONSchema(BaseModel):
    """JSON schema response format."""
    type: Literal["json_schema"] = "json_schema"
    json_schema: JSONSchema


ResponseFormat = Union[ResponseFormatText, ResponseFormatJSON, ResponseFormatJSONSchema]


# =============================================================================
# Stream Options
# =============================================================================


class StreamOptions(BaseModel):
    """Options for streaming responses."""
    include_usage: bool | None = Field(
        default=False,
        description="Include usage stats in final chunk"
    )


# =============================================================================
# Chat Completion Request
# =============================================================================


class ChatCompletionRequest(BaseModel):
    """Request body for /v1/chat/completions endpoint."""

    # Required fields
    model: Annotated[str, AfterValidator(model_exists)] = Field(
        description="Model ID to use for completion"
    )
    messages: list[Message] = Field(
        description="List of messages in the conversation"
    )

    # Sampling parameters
    temperature: float | None = Field(
        default=1.0,
        ge=0.0,
        le=2.0,
        description="Sampling temperature (0-2)"
    )
    top_p: float | None = Field(
        default=1.0,
        ge=0.0,
        le=1.0,
        description="Nucleus sampling threshold"
    )

    # Token limits
    max_tokens: int | None = Field(
        default=None,
        description="Maximum tokens to generate"
    )

    # Streaming
    stream: bool | None = Field(
        default=False,
        description="Enable streaming response"
    )
    stream_options: StreamOptions | None = Field(
        default=None,
        description="Streaming options"
    )

    # Stop sequences
    stop: str | list[str] | None = Field(
        default=None,
        description="Stop sequences (up to 4)"
    )

    # Tools
    tools: list[Tool] | None = Field(
        default=None,
        description="List of tools the model can call"
    )

    # Penalties
    frequency_penalty: float | None = Field(
        default=0.0,
        ge=-2.0,
        le=2.0,
        description="Frequency penalty (-2 to 2)"
    )

    # Response format
    response_format: ResponseFormat | None = Field(
        default=None,
        description="Response format specification"
    )

    # Reproducibility
    seed: int | None = Field(
        default=None,
        description="Seed for deterministic generation"
    )


# =============================================================================
# Chat Completion Response (Non-Streaming)
# =============================================================================


class ResponseMessage(BaseModel):
    """Message in a completion response."""
    role: Literal["assistant"] = "assistant"
    content: str | None = None
    tool_calls: list[ToolCall] | None = None


class Choice(BaseModel):
    """A completion choice."""
    index: int = Field(description="Index of this choice")
    message: ResponseMessage = Field(description="The generated message")
    finish_reason: Literal["stop", "length", "tool_calls"] | None = Field(
        default=None,
        description="Why generation stopped"
    )


class Usage(BaseModel):
    """Token usage statistics."""
    prompt_tokens: int = Field(description="Tokens in the prompt")
    completion_tokens: int = Field(description="Tokens in the completion")
    total_tokens: int = Field(description="Total tokens used")


class ChatCompletionResponse(BaseModel):
    """Response from /v1/chat/completions (non-streaming)."""
    id: str = Field(description="Unique completion ID")
    object: Literal["chat.completion"] = "chat.completion"
    created: int = Field(description="Unix timestamp of creation")
    model: str = Field(description="Model used")
    choices: list[Choice] = Field(description="Completion choices")
    usage: Usage = Field(description="Token usage stats")


# =============================================================================
# Chat Completion Chunk (Streaming)
# =============================================================================


class DeltaMessage(BaseModel):
    """Delta content in a streaming chunk."""
    role: Literal["assistant"]
    content: str = ""
    tool_calls: list[ToolCall] | None = None
    reasoning: str | None = None


class ChunkChoice(BaseModel):
    """A choice in a streaming chunk."""
    index: int = Field(description="Index of this choice")
    delta: DeltaMessage = Field(description="Delta content")
    finish_reason: Literal["stop", "length", "tool_calls"] | None = Field(
        default=None,
        description="Why generation stopped (null until final)"
    )


class ChatCompletionChunk(BaseModel):
    """A streaming chunk from /v1/chat/completions."""
    id: str = Field(description="Unique completion ID")
    object: Literal["chat.completion.chunk"] = "chat.completion.chunk"
    created: int = Field(description="Unix timestamp of creation")
    model: str = Field(description="Model used")
    choices: list[ChunkChoice] = Field(description="Chunk choices")
    usage: Usage | None = Field(
        default=None,
        description="Usage stats (only in final chunk if requested)"
    )


# =============================================================================
# Models List Response
# =============================================================================


class ModelObject(BaseModel):
    """A model object for /v1/models endpoint."""
    id: str = Field(description="Model identifier")
    object: Literal["model"] = "model"
    created: int = Field(description="Unix timestamp")
    owned_by: str = Field(default="local", description="Owner")


class ModelsListResponse(BaseModel):
    """Response from /v1/models endpoint."""
    object: Literal["list"] = "list"
    data: list[ModelObject] = Field(description="List of models")
