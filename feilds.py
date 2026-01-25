import os
from typing import Annotated, Any, Literal, Union
from pydantic import AfterValidator, BaseModel, Field


# =============================================================================
# Image Source Types
# =============================================================================


class Base64ImageSource(BaseModel):
    """Base64-encoded image source."""

    data: str = Field(description="Base64-encoded image data")
    media_type: Literal["image/jpeg", "image/png", "image/gif", "image/webp"] = Field(
        description="MIME type of the image"
    )
    type: Literal["base64"]


class URLImageSource(BaseModel):
    """URL-based image source."""

    type: Literal["url"]
    url: str = Field(description="URL of the image")


ImageSource = Union[Base64ImageSource, URLImageSource]


# =============================================================================
# Document Source Types
# =============================================================================

# Work In Progress


class Base64PDFSource(BaseModel):
    """Base64-encoded PDF source."""

    data: str = Field(description="Base64-encoded PDF data")
    media_type: Literal["application/pdf"] = Field(description="MIME type")
    type: Literal["base64"]


# Work In Progress
class PlainTextSource(BaseModel):
    """Plain text document source."""

    data: str = Field(description="Plain text content")
    media_type: Literal["text/plain"] = Field(description="MIME type")
    type: Literal["text"]


# Work In Progress
class ContentBlockSource(BaseModel):
    """Content block-based document source."""

    content: str | list["TextBlockParam"] = Field(
        description="Content as string or array of text blocks"
    )
    type: Literal["content"]


# Work In Progress
class URLPDFSource(BaseModel):
    """URL-based PDF source."""

    type: Literal["url"]
    url: str = Field(description="URL of the PDF")


DocumentSource = Union[
    Base64PDFSource, PlainTextSource, ContentBlockSource, URLPDFSource
]


# =============================================================================
# Content Block Params
# =============================================================================


class TextBlockParam(BaseModel):
    """Text content block."""

    text: str = Field(description="Text content")
    type: Literal["text"]


class ImageBlockParam(BaseModel):
    """Image content block."""

    source: ImageSource = Field(description="Image source")
    type: Literal["image"]


class DocumentBlockParam(BaseModel):
    """Document content block."""

    source: DocumentSource = Field(description="Document source")
    type: Literal["document"]
    context: str | None = Field(
        default=None, description="Additional context for the document"
    )
    title: str | None = Field(default=None, description="Title of the document")


class ToolUseBlockParam(BaseModel):
    """Tool use block for requesting tool execution."""

    id: str = Field(description="Unique identifier for this tool use")
    input: dict[str, Any] = Field(description="Input parameters for the tool")
    name: str = Field(description="Name of the tool to use")
    type: Literal["tool_use"]


class ToolResultBlockParam(BaseModel):
    """Tool result block containing the result of a tool execution."""

    tool_use_id: str = Field(description="ID of the tool use this is responding to")
    type: Literal["tool_result"]
    content: (
        str
        | list[
            Union[
                TextBlockParam,
                ImageBlockParam,
                DocumentBlockParam,
            ]
        ]
        | None
    ) = Field(default=None, description="Result content from the tool execution")
    is_error: bool | None = Field(
        default=None, description="Whether the tool execution resulted in an error"
    )


# FOR TRANSLATE GEMMA MODELS
class TranslateTextBlockParam(BaseModel):
    """Translation content block."""

    type: Literal["text"] = Field(description="Text content")
    source_lang_code: str = Field(description="Source language code")
    target_lang_code: str = Field(description="Target language code")
    text: str = Field(description="Text content")


# FOR TRANSLATE GEMMA MODELS
class TranslateImageBlockParam(BaseModel):
    """Translation content block."""

    type: Literal["image"] = Field(description="Text content")
    source_lang_code: str = Field(description="Source language code")
    target_lang_code: str = Field(description="Target language code")
    image: ImageSource = Field(description="Image source")


# =============================================================================
# Content Block Union
# =============================================================================

ContentBlockParam = Union[
    TextBlockParam,
    ImageBlockParam,
    DocumentBlockParam,
    ToolUseBlockParam,
    ToolResultBlockParam,
    TranslateTextBlockParam,
    TranslateImageBlockParam,
]


# =============================================================================
# Message Param
# =============================================================================


class MessageParam(BaseModel):
    """Message in a conversation."""

    role: Literal["user", "assistant"] = Field(
        description="The role of the message's author"
    )
    content: str | list[ContentBlockParam] = Field(description="The message content")


# =============================================================================
# Tool Choice
# =============================================================================


class ToolChoiceAuto(BaseModel):
    """The model will automatically decide whether to use tools."""

    type: Literal["auto"]
    disable_parallel_tool_use: bool | None = Field(
        default=None, description="Whether to disable parallel tool use"
    )


class ToolChoiceAny(BaseModel):
    """The model will use any available tools."""

    type: Literal["any"]
    disable_parallel_tool_use: bool | None = Field(
        default=None, description="Whether to disable parallel tool use"
    )


class ToolChoiceTool(BaseModel):
    """The model will use the specified tool."""

    type: Literal["tool"]
    name: str = Field(description="The name of the tool to use")
    disable_parallel_tool_use: bool | None = Field(
        default=None, description="Whether to disable parallel tool use"
    )


class ToolChoiceNone(BaseModel):
    """The model will not be allowed to use tools."""

    type: Literal["none"]


ToolChoice = Union[ToolChoiceAuto, ToolChoiceAny, ToolChoiceTool, ToolChoiceNone]


# =============================================================================
# Tool Definitions
# =============================================================================


class ToolInputSchema(BaseModel):
    """JSON schema for tool input."""

    type: Literal["object"]
    properties: dict[str, Any] | None = Field(
        default=None, description="Properties of the input schema"
    )
    required: list[str] | None = Field(default=None, description="Required properties")


class Tool(BaseModel):
    """User-defined tool definition."""

    name: str = Field(min_length=1, max_length=128, description="Name of the tool")
    input_schema: ToolInputSchema = Field(
        description="JSON schema for this tool's input"
    )
    description: str | None = Field(
        default=None, description="Description of what this tool does"
    )
    type: Literal["custom"] | None = Field(default=None, description="Tool type")


# =============================================================================
# Model Validation
# =============================================================================


def model_exists(model: str):
    """Validate that a model exists in the models directory."""
    if model.startswith("models/"):
        model_path = model
    else:
        model_path = os.path.join("./models", model)

    if not os.path.exists(model_path):
        raise ValueError(f"Model does not exist: {model}")
    return model


# =============================================================================
# Chat Completion Request/Response
# =============================================================================


class ChatCompletionParams(BaseModel):
    max_tokens: int = Field(ge=1, description="Maximum number of tokens to generate")
    # Required Parameters
    model: Annotated[str, AfterValidator(model_exists)] = Field(
        description="The file system path to the model"
    )
    messages: list[MessageParam] = Field(
        description="A list of messages comprising the conversation so far."
    )

    # Core Sampling Parameters
    temperature: float | None = Field(
        default=0.6,
        ge=0.0,
        le=1.0,
        description="Sampling temperature (0.0-1.0)",
    )
    max_completion_tokens: int | None = Field(
        default=None, description="Maximum number of tokens to generate"
    )
    stream: bool | None = Field(
        default=False, description="Enable streaming of the response"
    )
    stop_sequences: list[str] | None = Field(
        default=None, description="Custom text sequences that will stop generation"
    )
    top_logprobs: int | None = Field(
        default=0, description="Number of top logprobs to return"
    )
    top_k: int | None = Field(
        default=None,
        ge=0,
        description="Only sample from the top K options for each token",
    )
    top_p: float | None = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Use nucleus sampling with this cumulative probability",
    )

    # System Prompt
    system: str | list[TextBlockParam] | None = Field(
        default=None, description="System prompt for context and instructions"
    )

    # Tools
    tools: list[Tool] | None = Field(
        default=None, description="Definitions of tools that the model may use"
    )
    tool_choice: ToolChoice | None = Field(
        default=None, description="How the model should use the provided tools"
    )

    # MLX-specific Parameters
    max_kv_size: int | None = Field(
        default=None, description="Max context size of the model"
    )
    kv_bits: int | None = Field(
        default=None,
        ge=3,
        le=8,
        description="Number of bits for KV cache quantization. Must be between 3 and 8",
    )
    kv_group_size: int | None = Field(
        default=None, description="Group size for KV cache quantization"
    )
    quantized_kv_start: int | None = Field(
        default=None,
        description="When --kv-bits is set, start quantizing the KV cache from this step onwards",
    )
    draft_model: str | None = Field(
        default=None,
        description="The file system path to the draft model for speculative decoding",
    )
    num_draft_tokens: int | None = Field(
        default=None,
        description="Number of tokens to draft when using speculative decoding",
    )
    print_prompt_progress: bool | None = Field(
        default=False, description="Enable printed prompt processing progress callback"
    )
    max_img_size: int | None = Field(
        default=None, description="Downscale images to this side length (px)"
    )
    json_schema: str | None = Field(
        default=None, description="JSON schema for the response"
    )


# =============================================================================
# Response Models
# =============================================================================


class Usage(BaseModel):
    """Token usage statistics."""

    completion_tokens: int = Field(
        description="Number of tokens in the generated completion"
    )
    prompt_tokens: int = Field(description="Number of tokens in the prompt")
    total_tokens: int = Field(
        description="Total number of tokens used (prompt + completion)"
    )
    tokens_per_second: float | None = Field(
        default=None, description="Token generation rate (tokens/second)"
    )


class Choice(BaseModel):
    """Choice in the chat completion response."""

    finish_reason: Literal["stop", "length", "tool_calls", "content_filter"] | None = (
        Field(default=None, description="Reason for finishing")
    )
    index: int = Field(description="Index of the choice")
    message: MessageParam = Field(description="The message generated by the model")
    logprobs: Any | None = Field(
        default=None, description="Log probabilities (optional)"
    )


class ChatCompletionResponse(BaseModel):
    """Response from chat completion endpoint."""

    id: str = Field(description="Unique identifier for the response")
    choices: list[Choice] = Field(description="Response choices")
    created: int = Field(description="Unix timestamp of creation")
    model: str = Field(description="Model used")
    object: Literal["chat.completion"] = Field(
        default="chat.completion", description="Object type"
    )
    usage: Usage = Field(description="Token usage stats")
