from typing import Literal
from pydantic import BaseModel, Field

class GenerationOptions(BaseModel):
    """Runtime options that control text generation."""

    seed: int | None = Field(
        default=None, description="Random seed used for reproducible outputs"
    )
    temperature: float | None = Field(
        default=None,
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
    stop: str | list[str]  = Field(
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

    # Output formatting
    format: dict | None = Field(
        default=None,
        description='Structured output format for the model to generate a response from. Supports either the string "json" or a JSON schema object.',
    )
    system: str = Field(
        default="You are a helpful assistant.",
        description="System prompt for the model to generate a response from",
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


# Response for Generate API Endpoint
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


# Chat Request Models


class ToolCallFunction(BaseModel):
    name: str = Field(description="Name of the function to call")
    arguments: dict = Field(description="Arguments to pass to the function")


class ToolCall(BaseModel):
    id: str | None = Field(
        default=None, description="Unique identifier for the tool call"
    )
    type: Literal["function"] = Field(
        default="function", description="Type of tool call"
    )
    function: ToolCallFunction = Field(description="Function call details")


class Message(BaseModel):
    role: Literal["system", "user", "assistant", "tool"] = Field(
        description="Author of the message"
    )
    content: str = Field(description="Message text content")
    images: list[str] | None = Field(
        default=None,
        description="Optional list of inline images for multimodal models (Base64-encoded image content)",
    )
    tool_calls: list[ToolCall] | None = Field(
        default=None, description="Tool call requests produced by the model"
    )


class FunctionDefinition(BaseModel):
    name: str = Field(description="Name of the function")
    description: str | None = Field(
        default=None, description="Description of what the function does"
    )
    parameters: dict | None = Field(
        default=None, description="JSON schema describing the function parameters"
    )


class Tool(BaseModel):
    type: Literal["function"] = Field(
        default="function", description="Type of tool (always function)"
    )
    function: FunctionDefinition = Field(description="Function definition")


class ChatRequest(BaseModel):
    """Request body model for Ollama-compatible chat endpoint."""

    # Required fields
    model: str = Field(description="Model name")
    messages: list[Message] = Field(
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
    keep_alive: int = Field(
        default=0,
        description="Model keep-alive duration (for example 60 for 1 minute or 0 to unload immediately)",
    )
    logprobs: bool = Field(
        default=False,
        description="Whether to return log probabilities of the output tokens",
    )
    top_logprobs: int = Field(
        default=0,
        description="Number of most likely tokens to return at each token position when logprobs are enabled",
    )


############################
####Chat Response Models####
############################


class ResponseMessage(BaseModel):
    """Message object in chat response."""

    role: Literal["assistant", "tool"] = Field(description="Role of the message author")
    content: str = Field(default="", description="Message text content")
    tool_calls: list[ToolCall] | None = Field(
        default=None, description="Tool calls made by the model"
    )


class ChatResponse(BaseModel):
    """Response body model for Ollama-compatible chat endpoint."""

    model: str = Field(description="Model name used to generate this message")
    created_at: str = Field(description="Timestamp of response creation (ISO 8601)")
    message: ResponseMessage = Field(description="The generated message")
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
