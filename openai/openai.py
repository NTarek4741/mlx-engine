"""OpenAI-compatible API endpoint for MLX Engine."""

from fastapi.responses import StreamingResponse

from mlx_engine.generate import create_generator, tokenize
from mlx_engine.utils.prompt_progress_reporter import LoggerReporter

from openai.openai_models import (
    ChatCompletionRequest,
    ChatCompletionResponse,
)
from api.api_utils import load_and_cache_model, GenerationStatsCollector, model_cache
from openai.openai_utils import (
    convert_openai_messages,
    build_openai_response,
    openai_stream,
)


async def chat_completions(params: ChatCompletionRequest) -> ChatCompletionResponse | StreamingResponse:
    """
    OpenAI-compatible /v1/chat/completions endpoint handler.

    This function handles chat completion requests compatible with OpenAI's API.
    It supports both text and vision inputs (if the model supports vision).

    Args:
        params (ChatCompletionRequest): Request parameters containing:
            - model (str): Model ID to use
            - messages (list): Conversation history
            - temperature (float): Sampling temperature
            - max_tokens (int): Maximum tokens to generate
            - stream (bool): Whether to stream the response
            - tools (list): Optional tool definitions
            - ... (other params like top_p, stop, etc.)

    Returns:
        ChatCompletionResponse (JSON) if stream=False:
            {
                "id": "chatcmpl-...",
                "object": "chat.completion",
                "created": 1234567890,
                "model": "model_name",
                "choices": [{
                    "index": 0,
                    "message": {"role": "assistant", "content": "..."},
                    "finish_reason": "stop"
                }],
                "usage": {...}
            }

        StreamingResponse (SSE) if stream=True:
            Yields chunks with delta content.
    """
    print()
    # Determine max tokens
    max_tokens = params.max_completion_tokens or params.max_tokens

    # 1. Load model using shared utility
    model_kit, load_duration_ns = load_and_cache_model(
        model=params.model,
        num_ctx=None,  # Could be added to request params
        kv_bits=None,
        kv_group_size=None,
        quantized_kv_start=None,
        draft_model=None,
    )

    # 2. Convert OpenAI messages to internal format
    messages_dicts, images_b64, tools_dicts = convert_openai_messages(
        params.messages,
        params.tools,
    )

    # 3. Apply chat template and tokenize
    tf_tokenizer = model_cache["model_kit"].tokenizer._tokenizer
    prompt = tf_tokenizer.apply_chat_template(
        messages_dicts,
        tools=tools_dicts,
        tokenize=False,
        add_generation_prompt=True,
    )
    prompt_tokens = tokenize(model_kit, prompt)

    # 4. Initialize stats collector
    stats_collector = GenerationStatsCollector()
    stats_collector.load_duration = load_duration_ns

    # 5. Handle response format for JSON mode
    json_schema = None
    if params.response_format:
        if params.response_format.type == "json_object":
            # Basic JSON mode - model should output valid JSON
            json_schema = '{"type": "object"}'
        elif params.response_format.type == "json_schema":
            # Structured output with schema
            if params.response_format.json_schema and params.response_format.json_schema.schema_:
                import json
                json_schema = json.dumps(params.response_format.json_schema.schema_)

    # 6. Normalize stop sequences
    stop_sequences = None
    if params.stop:
        if isinstance(params.stop, str):
            stop_sequences = [params.stop]
        else:
            stop_sequences = params.stop[:4]  # Max 4 stop sequences

    # 7. Create generator
    generator = create_generator(
        model_kit,
        prompt_tokens,
        images_b64=images_b64 if images_b64 else None,
        stop_strings=stop_sequences,
        max_tokens=max_tokens,
        top_logprobs=None,
        prompt_progress_reporter=None,
        seed=params.seed,
        temp=params.temperature,
        top_k=None,
        top_p=params.top_p,
        json_schema=json_schema,
    )

    # 8. Return streaming or non-streaming response
    if params.stream:
        include_usage = False
        if params.stream_options and params.stream_options.include_usage:
            include_usage = True

        return StreamingResponse(
            openai_stream(
                generator,
                params.model,
                stats_collector,
                prompt_tokens,
                include_usage,
            ),
            media_type="text/event-stream",
        )
    else:
        return await generate_openai_output(
            generator,
            stats_collector,
            params,
            len(prompt_tokens),
        )


async def generate_openai_output(
    generator,
    stats_collector: GenerationStatsCollector,
    params: ChatCompletionRequest,
    prompt_token_count: int,
) -> ChatCompletionResponse:
    """
    Collect full generation output and return OpenAI response.

    Args:
        generator: The MLX generation iterator
        stats_collector: Stats collector for timing
        params: Original request parameters
        prompt_token_count: Number of tokens in the prompt

    Returns:
        ChatCompletionResponse with the complete generation
    """
    result_text = ""
    generation_result = None

    for generation_result in generator:
        result_text += generation_result.text
        stats_collector.add_tokens(generation_result.tokens)

    # Determine finish reason
    finish_reason = "length"
    if generation_result and generation_result.stop_condition:
        stop_reason = generation_result.stop_condition.stop_reason
        if stop_reason in ("stop_string", "eos_token"):
            finish_reason = "stop"

    return build_openai_response(
        model=params.model,
        result_text=result_text,
        finish_reason=finish_reason,
        prompt_token_count=prompt_token_count,
        completion_token_count=stats_collector.total_tokens,
        tool_calls=None,  # TODO: Parse tool calls from output if applicable
    )
