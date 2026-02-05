"""Anthropic-specific utility functions for message conversion and response building."""

import asyncio
import base64
import json
import time
import uuid
from datetime import datetime, timezone
from typing import AsyncGenerator
from urllib.request import urlopen

from api.api_utils import GenerationStatsCollector
from anthropic.anthropic_models import (
    MessageParam,
    TextBlockParam,
    ImageBlockParam,
    ToolUseBlockParam,
    ToolResultBlockParam,
    ChatCompletionResponse,
    Choice,
    Usage,
    Tool,
)


def normalize_system_prompt(system: str | list[TextBlockParam] | None) -> str | None:
    """
    Convert Anthropic system prompt format to plain string.

    Args:
        system: Can be None, str, or list of TextBlockParam

    Returns:
        Plain string system prompt or None
    """
    if system is None:
        return None
    if isinstance(system, str):
        return system
    # List of TextBlockParam - concatenate all text
    return "\n".join(block.text for block in system)


def image_url_to_base64(url: str) -> str:
    """
    Fetch image from URL and convert to base64.

    Args:
        url: Image URL

    Returns:
        Base64-encoded image data
    """
    with urlopen(url) as response:
        image_data = response.read()
    return base64.b64encode(image_data).decode("utf-8")


def convert_anthropic_messages(
    messages: list[MessageParam],
    system: str | list[TextBlockParam] | None,
    tools: list[Tool] | None = None,
) -> tuple[list[dict], list[str], list[dict] | None]:
    """
    Convert Anthropic message format to internal format for chat_template.

    Args:
        messages: List of Anthropic MessageParam objects
        system: Optional system prompt (string or list of TextBlockParam)
        tools: Optional list of Tool definitions

    Returns:
        Tuple of (messages_dicts, images_b64, tools_dicts) where:
        - messages_dicts: List of dicts ready for apply_chat_template()
        - images_b64: List of base64-encoded images extracted from content blocks
        - tools_dicts: List of tool definitions in dict format, or None
    """
    messages_dicts = []
    images_b64 = []

    # Add system message if present
    system_str = normalize_system_prompt(system)
    if system_str:
        messages_dicts.append({"role": "system", "content": system_str})

    for msg in messages:
        if isinstance(msg.content, str):
            # Simple string content
            messages_dicts.append({
                "role": msg.role,
                "content": msg.content
            })
        else:
            # List of content blocks
            content_parts = []
            tool_calls = []

            for block in msg.content:
                if isinstance(block, TextBlockParam) or (hasattr(block, 'type') and block.type == "text"):
                    content_parts.append({
                        "type": "text",
                        "text": block.text
                    })
                elif isinstance(block, ImageBlockParam) or (hasattr(block, 'type') and block.type == "image"):
                    # Handle image block
                    image_idx = len(images_b64)
                    content_parts.append({
                        "type": "image",
                        "image": f"image{image_idx}"
                    })
                    # Extract base64 data
                    if block.source.type == "base64":
                        images_b64.append(block.source.data)
                    elif block.source.type == "url":
                        images_b64.append(image_url_to_base64(block.source.url))
                elif isinstance(block, ToolUseBlockParam) or (hasattr(block, 'type') and block.type == "tool_use"):
                    # Handle tool use (for assistant messages)
                    tool_calls.append({
                        "id": block.id,
                        "type": "function",
                        "function": {
                            "name": block.name,
                            "arguments": json.dumps(block.input) if isinstance(block.input, dict) else block.input
                        }
                    })
                elif isinstance(block, ToolResultBlockParam) or (hasattr(block, 'type') and block.type == "tool_result"):
                    # Handle tool result - these go as separate tool messages
                    tool_content = block.content
                    if isinstance(tool_content, list):
                        tool_content = "\n".join(
                            b.text if hasattr(b, 'text') else str(b)
                            for b in tool_content
                        )
                    messages_dicts.append({
                        "role": "tool",
                        "tool_call_id": block.tool_use_id,
                        "content": tool_content or ""
                    })
                    continue  # Don't add to content_parts

            # Build the message dict
            if content_parts or tool_calls:
                msg_dict = {"role": msg.role}

                # Determine final content structure
                if len(content_parts) == 1 and content_parts[0].get("type") == "text" and not tool_calls:
                    # Single text block -> simplify to string
                    msg_dict["content"] = content_parts[0]["text"]
                elif content_parts:
                    msg_dict["content"] = content_parts
                else:
                    msg_dict["content"] = ""

                # Add tool calls if present
                if tool_calls:
                    msg_dict["tool_calls"] = tool_calls

                messages_dicts.append(msg_dict)

    # Convert tools to dict format
    tools_dicts = None
    if tools:
        tools_dicts = []
        for tool in tools:
            tool_dict = {
                "type": "function",
                "function": {
                    "name": tool.name,
                    "description": tool.description or "",
                    "parameters": {
                        "type": tool.input_schema.type,
                        "properties": tool.input_schema.properties or {},
                        "required": tool.input_schema.required or []
                    }
                }
            }
            tools_dicts.append(tool_dict)

    return messages_dicts, images_b64, tools_dicts


def build_anthropic_response(
    model: str,
    result_text: str,
    finish_reason: str,
    prompt_token_count: int,
    completion_token_count: int,
    tokens_per_second: float | None,
    logprobs: list | None = None
) -> ChatCompletionResponse:
    """
    Build Anthropic ChatCompletionResponse from generation results.

    Args:
        model: Model name
        result_text: Generated text
        finish_reason: Why generation stopped ("stop", "length", "tool_calls")
        prompt_token_count: Number of prompt tokens
        completion_token_count: Number of generated tokens
        tokens_per_second: Generation speed metric
        logprobs: Optional logprob information

    Returns:
        ChatCompletionResponse with proper structure
    """
    response_id = f"chatcmpl-{uuid.uuid4().hex[:24]}"

    return ChatCompletionResponse(
        id=response_id,
        created=int(time.time()),
        model=model,
        choices=[
            Choice(
                index=0,
                message=MessageParam(
                    role="assistant",
                    content=[TextBlockParam(type="text", text=result_text)]
                ),
                finish_reason=finish_reason,
                logprobs=logprobs,
            )
        ],
        usage=Usage(
            prompt_tokens=prompt_token_count,
            completion_tokens=completion_token_count,
            total_tokens=prompt_token_count + completion_token_count,
            tokens_per_second=tokens_per_second,
        ),
    )


async def anthropic_stream(
    generator,
    model_name: str,
    stats_collector: GenerationStatsCollector,
    prompt_tokens: list[int],
    include_logprobs: bool,
    top_logprobs: int
) -> AsyncGenerator[str, None]:
    """
    Stream generation results in Anthropic SSE format.

    Yields SSE events with format:
    - event: message_start
    - event: content_block_start
    - event: content_block_delta (text chunks)
    - event: content_block_stop
    - event: message_delta (usage stats)
    - event: message_stop

    Each event is formatted as: "event: <type>\\ndata: <json>\\n\\n"
    """
    message_id = f"msg_{uuid.uuid4().hex[:24]}"

    # 1. message_start event
    message_start = {
        "type": "message_start",
        "message": {
            "id": message_id,
            "type": "message",
            "role": "assistant",
            "content": [],
            "model": model_name,
            "stop_reason": None,
            "stop_sequence": None,
            "usage": {
                "input_tokens": len(prompt_tokens),
                "output_tokens": 0
            }
        }
    }
    yield f"event: message_start\ndata: {json.dumps(message_start)}\n\n"

    # 2. content_block_start event
    content_block_start = {
        "type": "content_block_start",
        "index": 0,
        "content_block": {
            "type": "text",
            "text": ""
        }
    }
    yield f"event: content_block_start\ndata: {json.dumps(content_block_start)}\n\n"

    # 3. Stream content deltas
    stop_reason = None
    try:
        for generation_result in generator:
            # Check if client disconnected
            try:
                await asyncio.sleep(0)  # Yield control to check for cancellation
            except asyncio.CancelledError:
                print("Client disconnected, stopping generation")
                return

            stats_collector.add_tokens(generation_result.tokens)

            if generation_result.text:
                delta_event = {
                    "type": "content_block_delta",
                    "index": 0,
                    "delta": {
                        "type": "text_delta",
                        "text": generation_result.text
                    }
                }
                yield f"event: content_block_delta\ndata: {json.dumps(delta_event)}\n\n"

            if generation_result.stop_condition:
                stop_reason = generation_result.stop_condition.stop_reason
                if stop_reason in ("stop_string", "eos_token"):
                    stop_reason = "end_turn"
                elif stop_reason == "max_tokens":
                    stop_reason = "max_tokens"
                break
    except asyncio.CancelledError:
        print("Client disconnected, stopping generation")
        return

    # 4. content_block_stop event
    content_block_stop = {
        "type": "content_block_stop",
        "index": 0
    }
    yield f"event: content_block_stop\ndata: {json.dumps(content_block_stop)}\n\n"

    # 5. message_delta with usage
    message_delta = {
        "type": "message_delta",
        "delta": {
            "stop_reason": stop_reason or "end_turn",
            "stop_sequence": None
        },
        "usage": {
            "output_tokens": stats_collector.total_tokens
        }
    }
    yield f"event: message_delta\ndata: {json.dumps(message_delta)}\n\n"

    # 6. message_stop event
    message_stop = {
        "type": "message_stop"
    }
    yield f"event: message_stop\ndata: {json.dumps(message_stop)}\n\n"
