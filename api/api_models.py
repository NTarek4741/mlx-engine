import os
from typing import Annotated, Any, Literal, Union
from pydantic import AfterValidator, BaseModel, Field


# =============================================================================
# Model Validation
# =============================================================================


def model_exists(model: str) -> str:
    """Validate that a model exists in the models directory."""
    model_name = model.split(":")[0] if ":" in model else model

    if model_name.startswith("models/"):
        model_path = model_name
    else:
        model_path = os.path.join("./models", model_name)

    if not os.path.exists(model_path):
        raise ValueError(f"Model does not exist: {model}")
    return model_name


# =============================================================================
# SHARED MODELS (used by both Ollama and OpenAI endpoints)
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
    type: Literal["function"] = Field(
        default="function", description="Type of tool (always function)"
    )
    function: FunctionDefinition = Field(description="Function definition")


# =============================================================================
# OLLAMA MODELS
# =============================================================================


class ModelInfo(BaseModel):
    """Information about a single model."""
    name: str = Field(description="Model name/path")
    model: str = Field(description="Full model identifier")


class TagsResponse(BaseModel):
    """Response body for the /api/tags endpoint."""
    models: list[ModelInfo] = Field(description="List of available models")


class RunningModelInfo(BaseModel):
    """Information about a currently running/loaded model."""
    model: str = Field(description="Model name/path")


class PSResponse(BaseModel):
    """Response body for the /api/ps endpoint."""
    models: list[RunningModelInfo] = Field(description="List of currently running models")


class GenerationOptions(BaseModel):
    """Runtime options that control text generation."""

    seed: int | None = Field(
        default=None, description="Random seed used for reproducible outputs"
    )
    temperature: float | None = Field(
        default=0.15,
        description="Controls randomness in generation (higher = more random)",
    )
    top_k: int | None = Field(
        default=None, description="Limits next token selection to the K most likely"
    )
    top_p: float | None = Field(
        default=None,
        description="Cumulative probability threshold for nucleus sampling",
    )
    min_p: float | None = Field(
        default=None, description="Minimum probability threshold for token selection"
    )
    stop: str | list[str] | None = Field(
        default=None, description="Stop sequences that will halt generation"
    )
    num_ctx: int | None = Field(
        default=None, description="Context length size (number of tokens)"
    )
    num_predict: int | None = Field(
        default=None, description="Maximum number of tokens to generate"
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
        description="When kv_bits is set, start quantizing the KV cache from this step onwards",
    )
    draft_model: str | None = Field(
        default=None,
        description="The name or path to the draft model for speculative decoding",
    )


class GenerateRequest(BaseModel):
    """Request body model for Ollama-compatible generate endpoint."""

    # Required fields
    model: str = Field(description="Model name")

    # Core generation fields
    prompt: str = Field(
        default="Why is the sky blue?", description="Text for the model to generate a response from"
    )
    suffix: str | None = Field(
        default=None,
        description="Used for fill-in-the-middle models, text that appears after the user prompt and before the model response",
    )
    images: list[str] | None = Field(
        default=None,
        description="Base64-encoded images for models that support image input",
    )

    # Response format
    format: ResponseFormat | None = Field(
        default=None,
        description="Response format specification"
    )

    system: str | None = Field(
        default="""You are a highly capable AI assistant. Follow these rules:

# CONSTRAINT PRIORITY (Most Important)
User constraints ALWAYS override style preferences. If asked for brevity, be brief. If given a word limit, format, or "just the answer"—obey exactly.

# RESPONSE CALIBRATION
Match response depth to question complexity:
- Simple facts → Direct answer, no elaboration
- "Explain" or "why" → First principles depth
- Creative requests → Follow the exact format requested
- Complex tasks → Structured, complete solution

# STYLE
- Conversational and direct. No AI fluff ("In conclusion," "It's important to note").
- Use formatting (bold, headers, code blocks) only when it aids comprehension.
- When explaining, focus on the *why*, not just the *what*.
- Be opinionated when it helps—if something is a bad idea, say so.

# CODE
Production-ready, commented, modern patterns. No placeholders or TODOs.

# TONE
Smart colleague, not servile assistant. Treat the user as a peer.""",
        description="System prompt for the model to generate a response from. If not provided, uses the Modelfile's system prompt if available.",
    )

    # Streaming and output control
    stream: bool = Field(
        default=True, description="When true, returns a stream of partial responses"
    )
    think: bool | Literal["high", "medium", "low"] = Field(
        default=False,
        description='When true, returns separate thinking output in addition to content. Can be a boolean (true/false) or a string ("high", "medium", "low") for supported models.',
    )
    raw: bool = Field(
        default=False,
        description="When true, returns the raw response from the model without any prompt templating",
    )

    # Generation options
    options: GenerationOptions | None = Field(
        default=None, description="Runtime options that control text generation"
    )

    # Log probabilities
    logprobs: bool = Field(
        default=False,
        description="Whether to return log probabilities of the output tokens",
    )
    top_logprobs: int = Field(
        default=0,
        le=10,
        description="Number of most likely tokens to return at each token position when logprobs are enabled",
    )


class TopLogprobEntry(BaseModel):
    token: str = Field(description="The text representation of the token")
    logprob: float = Field(description="The log probability of this token")
    bytes: list[int] | None = Field(
        default=None, description="The raw byte representation of the token"
    )


class LogprobEntry(BaseModel):
    token: str = Field(description="The text representation of the token")
    logprob: float = Field(description="The log probability of this token")
    bytes: list[int] | None = Field(
        default=None, description="The raw byte representation of the token"
    )
    top_logprobs: list[TopLogprobEntry] | None = Field(
        default=None,
        description="Most likely tokens and their log probabilities at this position",
    )


class GenerateResponse(BaseModel):
    model: str = Field(description="Model name")
    created_at: str = Field(description="ISO 8601 timestamp of response creation")
    response: str = Field(default="", description="The model's generated text response")
    thinking: str | None = Field(
        default=None, description="The model's generated thinking output"
    )
    done: bool = Field(
        default=False, description="Indicates whether generation has finished"
    )
    done_reason: str | None = Field(
        default=None, description="Reason the generation stopped"
    )

    total_duration: int | None = Field(
        default=None, description="Time spent generating the response in nanoseconds"
    )
    load_duration: int | None = Field(
        default=None, description="Time spent loading the model in nanoseconds"
    )
    prompt_eval_count: int | None = Field(
        default=None, description="Number of input tokens in the prompt"
    )
    prompt_eval_duration: int | None = Field(
        default=None, description="Time spent evaluating the prompt in nanoseconds"
    )
    eval_count: int | None = Field(
        default=None, description="Number of output tokens generated in the response"
    )
    eval_duration: int | None = Field(
        default=None, description="Time spent generating tokens in nanoseconds"
    )
    logprobs: list[LogprobEntry] | None = Field(
        default=None,
        description="Log probability information for the generated tokens when logprobs are enabled",
    )
    context: list[int] | None = Field(
        default=None,
        description="An encoding of the conversation used in this response, can be sent in the next request to keep conversational memory",
    )


class OllamaToolCallFunction(BaseModel):
    name: str = Field(description="Name of the function to call")
    arguments: dict = Field(description="Arguments to pass to the function")


class OllamaToolCall(BaseModel):
    id: str | None = Field(
        default=None, description="Unique identifier for the tool call"
    )
    type: Literal["function"] = Field(
        default="function", description="Type of tool call"
    )
    function: OllamaToolCallFunction = Field(description="Function call details")


class OllamaMessage(BaseModel):
    role: Literal["system", "user", "assistant", "tool"] = Field(
        description="Author of the message"
    )
    content: str = Field(description="Message text content")
    images: list[str] | None = Field(
        default=None,
        description="Optional list of inline images for multimodal models (Base64-encoded image content)",
    )
    tool_calls: list[OllamaToolCall] | None = Field(
        default=None, description="Tool call requests produced by the model"
    )
    tool_call_id: str | None = Field(
        default=None, description="ID of the tool call this message responds to (for tool role messages)"
    )


class ChatRequest(BaseModel):
    """Request body model for Ollama-compatible chat endpoint."""

    # Required fields
    model: str = Field(description="Model name")
    messages: list[OllamaMessage] = Field(
        description="Chat history as an array of message objects (each with a role and content)"
    )

    # Optional fields
    tools: list[Tool] | None = Field(
        default=None,
        description="Optional list of function tools the model may call during the chat",
    )
    format: str | dict | None = Field(
        default=None,
        description='Format to return a response in. Can be "json" or a JSON schema',
    )
    options: GenerationOptions | None = Field(
        default=None, description="Runtime options that control text generation"
    )
    stream: bool = Field(default=True, description="Stream the response")
    think: bool | Literal["high", "medium", "low"] = Field(
        default=False,
        description='When true, returns separate thinking output in addition to content. Can be a boolean (true/false) or a string ("high", "medium", "low") for supported models.',
    )
    logprobs: bool = Field(
        default=False,
        description="Whether to return log probabilities of the output tokens",
    )
    top_logprobs: int = Field(
        default=0,
        description="Number of most likely tokens to return at each token position when logprobs are enabled",
    )


class OllamaResponseMessage(BaseModel):
    """Message object in chat response."""

    role: Literal["assistant", "tool"] = Field(description="Role of the message author")
    content: str = Field(default="", description="Message text content")
    tool_calls: list[OllamaToolCall] | None = Field(
        default=None, description="Tool calls made by the model"
    )


class ChatResponse(BaseModel):
    """Response body model for Ollama-compatible chat endpoint."""

    model: str = Field(description="Model name used to generate this message")
    created_at: str = Field(description="Timestamp of response creation (ISO 8601)")
    message: OllamaResponseMessage = Field(description="The generated message")
    done: bool = Field(
        default=False, description="Indicates whether the chat response has finished"
    )
    done_reason: str | None = Field(
        default=None, description="Reason the response finished"
    )

    total_duration: int | None = Field(
        default=None, description="Total time spent generating in nanoseconds"
    )
    load_duration: int | None = Field(
        default=None, description="Time spent loading the model in nanoseconds"
    )
    prompt_eval_count: int | None = Field(
        default=None, description="Number of tokens in the prompt"
    )
    prompt_eval_duration: int | None = Field(
        default=None, description="Time spent evaluating the prompt in nanoseconds"
    )
    eval_count: int | None = Field(
        default=None, description="Number of tokens generated in the response"
    )
    eval_duration: int | None = Field(
        default=None, description="Time spent generating tokens in nanoseconds"
    )
    logprobs: list[LogprobEntry] | None = Field(
        default=None,
        description="Log probability information for the generated tokens when logprobs are enabled",
    )


# =============================================================================
# OPENAI MODELS
# =============================================================================


# --- Content types ---

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


# --- Tool calls (OpenAI-specific) ---

class OpenAIFunctionCall(BaseModel):
    """Function call details."""
    name: str = Field(description="Name of the function to call")
    arguments: str = Field(description="JSON string of arguments")


class OpenAIToolCall(BaseModel):
    """Tool call made by the model."""
    id: str = Field(description="Unique ID for this tool call")
    index: int = Field(default=0, description="Index of this tool call")
    type: Literal["function"] = "function"
    function: OpenAIFunctionCall


# --- Messages (OpenAI-specific) ---

class OpenAISystemMessage(BaseModel):
    """System message."""
    role: Literal["system"]
    content: str


class OpenAIUserMessage(BaseModel):
    """User message with text or multimodal content."""
    role: Literal["user"]
    content: str | list[ContentPart]


class OpenAIAssistantMessage(BaseModel):
    """Assistant message, possibly with tool calls."""
    role: Literal["assistant"]
    content: str | None = None
    tool_calls: list[OpenAIToolCall] | None = None


class OpenAIToolMessage(BaseModel):
    """Tool result message."""
    role: Literal["tool"]
    content: str
    tool_call_id: str = Field(description="ID of the tool call this responds to")


OpenAIMessage = Union[OpenAISystemMessage, OpenAIUserMessage, OpenAIAssistantMessage, OpenAIToolMessage]


# --- Stream options ---

class StreamOptions(BaseModel):
    """Options for streaming responses."""
    include_usage: bool | None = Field(
        default=False,
        description="Include usage stats in final chunk"
    )


# --- Chat Completion Request ---

class ChatCompletionRequest(BaseModel):
    """Request body for /v1/chat/completions endpoint."""

    # Required fields
    model: Annotated[str, AfterValidator(model_exists)] = Field(
        description="Model ID to use for completion"
    )
    messages: list[OpenAIMessage] = Field(
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


# --- Chat Completion Response (Non-Streaming) ---

class OpenAIResponseMessage(BaseModel):
    """Message in a completion response."""
    role: Literal["assistant"] = "assistant"
    content: str | None = None
    tool_calls: list[OpenAIToolCall] | None = None


class Choice(BaseModel):
    """A completion choice."""
    index: int = Field(description="Index of this choice")
    message: OpenAIResponseMessage = Field(description="The generated message")
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


# --- Chat Completion Chunk (Streaming) ---

class DeltaMessage(BaseModel):
    """Delta content in a streaming chunk."""
    role: Literal["assistant"] = "assistant"
    content: str = ""
    tool_calls: list[OpenAIToolCall] | None = None
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


# --- Models List Response ---

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
