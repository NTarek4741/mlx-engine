"""OpenAI-specific utility functions for message conversion and response building."""

import asyncio
import base64
import json
import re
import time
import uuid
from datetime import datetime, timezone
from typing import AsyncGenerator
from urllib.request import urlopen

from api.api_utils import GenerationStatsCollector
from openai.openai_models import (
    Message,
    SystemMessage,
    UserMessage,
    AssistantMessage,
    ToolMessage,
    TextContent,
    ImageContent,
    Tool,
    ToolCall,
    FunctionCall,
    ChatCompletionResponse,
    ChatCompletionChunk,
    Choice,
    ChunkChoice,
    ResponseMessage,
    DeltaMessage,
    Usage,
)


def image_url_to_base64(url: str) -> str:
    """
    Fetch image from URL and convert to base64.
    If already a data URI, extract the base64 part.
    """
    if url.startswith("data:"):
        # Extract base64 from data URI
        # Format: data:image/jpeg;base64,<data>
        if ";base64," in url:
            return url.split(";base64,")[1]
        return url

    # Fetch from URL
    with urlopen(url) as response:
        image_data = response.read()
    return base64.b64encode(image_data).decode("utf-8")


def parse_tool_calls(text: str) -> tuple[list[ToolCall] | None, str | None]:
    """
    Parse tool calls from model-generated text.
    Supports multiple formats:
    - <tool_call>{"name":"...","arguments":{...}}</tool_call> (Qwen/Hermes)
    - [TOOL_CALLS]name[ARGS]{"key":"value"} (Mistral)

    Returns:
        (tool_calls, remaining_content) where:
        - tool_calls: list of ToolCall objects if found, else None
        - remaining_content: text outside tool call tags, or None if empty
    """
    tool_calls = []
    remaining = text

    # Format 1: Mistral — [TOOL_CALLS]funcName[ARGS]{...}
    mistral_pattern = r'\[TOOL_CALLS\]\s*(\w+)\s*\[ARGS\]\s*(\{.*?\})'
    mistral_matches = re.findall(mistral_pattern, text, re.DOTALL)
    if mistral_matches:
        for i, (name, args_str) in enumerate(mistral_matches):
            try:
                parsed_args = json.loads(args_str)
            except json.JSONDecodeError:
                parsed_args = {}
            arguments = json.dumps(parsed_args) if isinstance(parsed_args, dict) else args_str
            call_id = f"call_{uuid.uuid4().hex[:24]}"
            tool_calls.append(ToolCall(
                id=call_id,
                index=i,
                type="function",
                function=FunctionCall(name=name, arguments=arguments),
            ))
        remaining = re.sub(mistral_pattern, '', text, flags=re.DOTALL).strip()

    # Format 2: Qwen/Hermes — <tool_call>...</tool_call>
    if not tool_calls:
        xml_pattern = r'<tool_call>\s*(.*?)\s*</tool_call>'
        xml_matches = re.findall(xml_pattern, text, re.DOTALL)
        if xml_matches:
            for i, match in enumerate(xml_matches):
                try:
                    parsed = json.loads(match)
                except json.JSONDecodeError:
                    continue
                if "function" in parsed:
                    name = parsed["function"].get("name", "")
                    arguments = parsed["function"].get("arguments", {})
                else:
                    name = parsed.get("name", "")
                    arguments = parsed.get("arguments", parsed.get("parameters", {}))
                if isinstance(arguments, dict):
                    arguments = json.dumps(arguments)
                call_id = f"call_{uuid.uuid4().hex[:24]}"
                tool_calls.append(ToolCall(
                    id=call_id,
                    index=i,
                    type="function",
                    function=FunctionCall(name=name, arguments=arguments),
                ))
            remaining = re.sub(xml_pattern, '', text, flags=re.DOTALL).strip()

    if not tool_calls:
        return None, text

    return tool_calls, remaining if remaining else None


def convert_openai_messages(
    messages: list[Message],
    tools: list[Tool] | None = None,
) -> tuple[list[dict], list[str], list[dict] | None]:
    """
    Convert OpenAI message format to internal format for chat_template.

    Args:
        messages: List of OpenAI Message objects
        tools: Optional list of Tool definitions

    Returns:
        Tuple of (messages_dicts, images_b64, tools_dicts) where:
        - messages_dicts: List of dicts ready for apply_chat_template()
        - images_b64: List of base64-encoded images extracted from content
        - tools_dicts: List of tool definitions in dict format, or None
    """
    messages_dicts = []
    images_b64 = []

    for msg in messages:
        if isinstance(msg, SystemMessage) or (hasattr(msg, 'role') and msg.role == "system"):
            messages_dicts.append({
                "role": "system",
                "content": msg.content
            })

        elif isinstance(msg, UserMessage) or (hasattr(msg, 'role') and msg.role == "user"):
            if isinstance(msg.content, str):
                messages_dicts.append({
                    "role": "user",
                    "content": msg.content
                })
            else:
                # List of content parts (multimodal)
                content_parts = []
                for part in msg.content:
                    if isinstance(part, TextContent) or (hasattr(part, 'type') and part.type == "text"):
                        content_parts.append({
                            "type": "text",
                            "text": part.text
                        })
                    elif isinstance(part, ImageContent) or (hasattr(part, 'type') and part.type == "image_url"):
                        image_idx = len(images_b64)
                        content_parts.append({
                            "type": "image",
                            "image": f"image{image_idx}"
                        })
                        # Extract base64 from URL or data URI
                        images_b64.append(image_url_to_base64(part.image_url.url))

                # Simplify if only text
                if len(content_parts) == 1 and content_parts[0].get("type") == "text":
                    messages_dicts.append({
                        "role": "user",
                        "content": content_parts[0]["text"]
                    })
                else:
                    messages_dicts.append({
                        "role": "user",
                        "content": content_parts
                    })

        elif isinstance(msg, AssistantMessage) or (hasattr(msg, 'role') and msg.role == "assistant"):
            msg_dict = {
                "role": "assistant",
                "content": msg.content or ""
            }
            # Add tool calls if present
            if msg.tool_calls:
                msg_dict["tool_calls"] = [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {
                            "name": tc.function.name,
                            "arguments": tc.function.arguments
                        }
                    }
                    for tc in msg.tool_calls
                ]
            messages_dicts.append(msg_dict)

        elif isinstance(msg, ToolMessage) or (hasattr(msg, 'role') and msg.role == "tool"):
            messages_dicts.append({
                "role": "tool",
                "tool_call_id": msg.tool_call_id,
                "content": msg.content
            })

    # Convert tools to dict format
    tools_dicts = None
    if tools:
        tools_dicts = []
        for tool in tools:
            tool_dict = {
                "type": "function",
                "function": {
                    "name": tool.function.name,
                    "description": tool.function.description or "",
                    "parameters": {
                        "type": tool.function.parameters.type if tool.function.parameters else "object",
                        "properties": tool.function.parameters.properties if tool.function.parameters else {},
                        "required": tool.function.parameters.required if tool.function.parameters else []
                    }
                }
            }
            tools_dicts.append(tool_dict)

    return messages_dicts, images_b64, tools_dicts


def build_openai_response(
    model: str,
    result_text: str,
    finish_reason: str,
    prompt_token_count: int,
    completion_token_count: int,
    tool_calls: list[ToolCall] | None = None,
) -> dict:
    """
    Build OpenAI ChatCompletionResponse from generation results.

    Args:
        model: Model name
        result_text: Generated text
        finish_reason: Why generation stopped ("stop", "length", "tool_calls")
        prompt_token_count: Number of prompt tokens
        completion_token_count: Number of generated tokens
        tool_calls: Optional list of tool calls

    Returns:
        ChatCompletionResponse with proper structure
    """
    response_id = f"chatcmpl-{uuid.uuid4().hex[:24]}"

    response = ChatCompletionResponse(
        id=response_id,
        created=int(time.time()),
        model=model,
        choices=[
            Choice(
                index=0,
                message=ResponseMessage(
                    role="assistant",
                    content=result_text if result_text else None,
                    tool_calls=tool_calls,
                ),
                finish_reason=finish_reason,
            )
        ],
        usage=Usage(
            prompt_tokens=prompt_token_count,
            completion_tokens=completion_token_count,
            total_tokens=prompt_token_count + completion_token_count,
        ),
    )
    return json.loads(response.model_dump_json(exclude_none=True))


TOOL_CALL_PREFIXES = ("[TOOL_CALLS]", "<tool_call>")

async def openai_stream(
    generator,
    model_name: str,
    stats_collector: GenerationStatsCollector,
    prompt_tokens: list[int],
    include_usage: bool = False,
) -> AsyncGenerator[str, None]:
    """
    Stream generation results in OpenAI SSE format.

    Streams tokens in real-time. If the first tokens indicate a tool call,
    switches to buffering mode and emits structured tool_calls at the end.
    """
    chunk_id = f"chatcmpl-{uuid.uuid4().hex[:24]}"
    created = int(time.time())

    # First chunk with role
    first_chunk = ChatCompletionChunk(
        id=chunk_id,
        created=created,
        model=model_name,
        choices=[
            ChunkChoice(
                index=0,
                delta=DeltaMessage(role="assistant", content=""),
                finish_reason=None,
            )
        ],
    )
    yield f"data: {first_chunk.model_dump_json(exclude_none=True)}\n\n"

    full_text = ""
    think = ""
    thinking = False
    buffering = None  # None = undecided, True = buffer for tool calls, False = stream normally
    pending_chunks = []  # tokens held while deciding
    finish_reason = None

    try:
        for generation_result in generator:
            try:
                await asyncio.sleep(0)
            except asyncio.CancelledError:
                return

            stats_collector.add_tokens(generation_result.tokens)

            if generation_result.text:
                if generation_result.text == '<think>':
                    thinking = True
                if generation_result.text == '</think>':
                    think += generation_result.text
                    thinking = False
                    continue
                if thinking == True:
                    think += generation_result.text
                    continue
                
                full_text += generation_result.text

                if buffering is None:
                    # Still deciding — check if text starts with a tool call prefix
                    pending_chunks.append(generation_result.text)
                    print(len(full_text))
                    print(full_text)
                    v = [p in full_text for p in TOOL_CALL_PREFIXES]
                    print(v)
                    if any(v):
                        buffering = True
                        print("Hello")
                    elif len(full_text) >= 15:
                        # Enough text to decide — not a tool call, flush and stream
                        print("GoodBye")
                        buffering = False
                        for chunk_text in pending_chunks:
                            content_chunk = ChatCompletionChunk(
                                id=chunk_id, created=created, model=model_name,
                                choices=[ChunkChoice(index=0, delta=DeltaMessage(content=chunk_text))],
                            )
                            yield f"data: {content_chunk.model_dump_json(exclude_none=True)}\n\n"
                        pending_chunks.clear()
                elif buffering is False:
                    # Streaming mode — emit token immediately
                    content_chunk = ChatCompletionChunk(
                        id=chunk_id, created=created, model=model_name,
                        choices=[ChunkChoice(index=0, delta=DeltaMessage(content=generation_result.text))],
                    )
                    yield f"data: {content_chunk.model_dump_json(exclude_none=True)}\n\n"
                # buffering is True — just accumulate

            if generation_result.stop_condition:
                stop_reason = generation_result.stop_condition.stop_reason
                if stop_reason in ("stop_string", "eos_token"):
                    finish_reason = "stop"
                elif stop_reason == "max_tokens":
                    finish_reason = "length"
                else:
                    finish_reason = "stop"
                break

    except asyncio.CancelledError:
        return

    # If we never decided (very short output), flush pending as content
    if buffering is None:
        buffering = any(full_text.startswith(p) for p in TOOL_CALL_PREFIXES)
        if not buffering:
            for chunk_text in pending_chunks:
                content_chunk = ChatCompletionChunk(
                    id=chunk_id, created=created, model=model_name,
                    choices=[ChunkChoice(index=0, delta=DeltaMessage(content=chunk_text))],
                )
                yield f"data: {content_chunk.model_dump_json(exclude_none=True)}\n\n"

    # If buffering, parse tool calls and emit them
    if buffering:
        tool_calls, remaining_content = parse_tool_calls(full_text)
        if tool_calls:
            if remaining_content:
                content_chunk = ChatCompletionChunk(
                    id=chunk_id, created=created, model=model_name,
                    choices=[ChunkChoice(index=0, delta=DeltaMessage(content=remaining_content))],
                )
                yield f"data: {content_chunk.model_dump_json(exclude_none=True)}\n\n"

            tool_call_chunk = ChatCompletionChunk(
                id=chunk_id, created=created, model=model_name,
                choices=[ChunkChoice(index=0, delta=DeltaMessage(tool_calls=tool_calls))],
            )
            yield f"data: {tool_call_chunk.model_dump_json(exclude_none=True)}\n\n"
            finish_reason = "tool_calls"
        else:
            # Looked like tool call but wasn't — flush as content
            if full_text:
                content_chunk = ChatCompletionChunk(
                    id=chunk_id, created=created, model=model_name,
                    choices=[ChunkChoice(index=0, delta=DeltaMessage(content=full_text))],
                )
                yield f"data: {content_chunk.model_dump_json(exclude_none=True)}\n\n"

    # Final chunk with finish_reason
    final_chunk = ChatCompletionChunk(
        id=chunk_id,
        created=created,
        model=model_name,
        choices=[
            ChunkChoice(
                index=0,
                delta=DeltaMessage(),
                finish_reason=finish_reason or "stop",
            )
        ],
        usage=Usage(
            prompt_tokens=len(prompt_tokens),
            completion_tokens=stats_collector.total_tokens,
            total_tokens=len(prompt_tokens) + stats_collector.total_tokens,
        ) if include_usage else None,
    )
    yield f"data: {final_chunk.model_dump_json(exclude_none=True)}\n\n"
    yield "data: [DONE]\n\n"
