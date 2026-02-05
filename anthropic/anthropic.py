"""Anthropic Messages API endpoint for MLX Engine."""

from fastapi.responses import StreamingResponse
from transformers import AutoTokenizer

from mlx_engine.generate import create_generator, tokenize
from mlx_engine.utils.prompt_progress_reporter import LoggerReporter

from anthropic.anthropic_models import MessagesParams, ChatCompletionResponse
from api.api_utils import load_and_cache_model, GenerationStatsCollector, model_cache
from anthropic.anthropic_utils import (
    convert_anthropic_messages,
    build_anthropic_response,
    anthropic_stream,
)


async def messages(params: MessagesParams) -> ChatCompletionResponse | StreamingResponse:
    """
    Anthropic Messages API endpoint handler.

    This function handles chat completion requests compatible with Anthropic's API.
    It supports both text and vision inputs (if the model supports vision).

    Args:
        params (MessagesParams): Request parameters containing:
            - messages (list): List of message objects (role, content)
            - model (str): Path/name of the model to use
            - temperature (float): Sampling temperature
            - max_tokens (int): Maximum tokens to generate
            - stream (bool): Whether to stream the response
            - system: Optional system prompt
            - tools: Optional tool definitions
            - ... (other params like top_p, top_k, etc.)

    Returns:
        ChatCompletionResponse (JSON) if stream=False:
            {
                "id": "...",
                "created": 1234567890,
                "model": "model_name",
                "choices": [{
                    "index": 0,
                    "message": { "role": "assistant", "content": [...] },
                    "finish_reason": "stop" | "length" | "tool_calls"
                }],
                "usage": { ... }
            }

        StreamingResponse (SSE) if stream=True:
            Yields SSE events with generated text deltas.
    """
    # 1. Load model using shared utility
    model_kit, load_duration_ns = load_and_cache_model(
        model=params.model,
        num_ctx=params.max_kv_size,
        kv_bits=params.kv_bits,
        kv_group_size=params.kv_group_size,
        quantized_kv_start=params.quantized_kv_start,
        draft_model=params.draft_model,
    )

    # 2. Convert Anthropic messages to internal format
    messages_dicts, images_b64, tools_dicts = convert_anthropic_messages(
        params.messages,
        params.system,
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

    # 5. Create generator
    generator = create_generator(
        model_kit,
        prompt_tokens,
        images_b64=images_b64 if images_b64 else None,
        stop_strings=params.stop_sequences,
        max_tokens=params.max_tokens,
        top_logprobs=params.top_logprobs if params.top_logprobs else None,
        prompt_progress_reporter=LoggerReporter() if params.print_prompt_progress else None,
        num_draft_tokens=params.num_draft_tokens,
        temp=params.temperature,
        top_k=params.top_k,
        top_p=params.top_p,
    )

    # 6. Return streaming or non-streaming response
    if params.stream:
        return StreamingResponse(
            anthropic_stream(
                generator,
                params.model,
                stats_collector,
                prompt_tokens,
                params.top_logprobs > 0 if params.top_logprobs else False,
                params.top_logprobs or 0,
            ),
            media_type="text/event-stream",
        )
    else:
        return await generate_anthropic_output(
            generator,
            stats_collector,
            params,
            len(prompt_tokens),
        )


async def generate_anthropic_output(
    generator,
    stats_collector: GenerationStatsCollector,
    params: MessagesParams,
    prompt_token_count: int,
) -> ChatCompletionResponse:
    """
    Collect full generation output and return Anthropic response.

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
    logprobs_list = []

    for generation_result in generator:
        result_text += generation_result.text
        stats_collector.add_tokens(generation_result.tokens)

        # Collect logprobs if enabled
        if params.top_logprobs and generation_result.top_logprobs:
            logprobs_list.extend(generation_result.top_logprobs)

    # Determine finish reason
    finish_reason = "length"
    if generation_result and generation_result.stop_condition:
        stop_reason = generation_result.stop_condition.stop_reason
        if stop_reason in ("stop_string", "eos_token"):
            finish_reason = "stop"
        elif stop_reason == "tool_call":
            finish_reason = "tool_calls"

    return build_anthropic_response(
        model=params.model,
        result_text=result_text,
        finish_reason=finish_reason,
        prompt_token_count=prompt_token_count,
        completion_token_count=stats_collector.total_tokens,
        tokens_per_second=stats_collector.get_tokens_per_second(),
        logprobs=logprobs_list if logprobs_list else None,
    )
