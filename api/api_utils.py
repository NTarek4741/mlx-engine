import mlx as mx
import gc
import asyncio
import uuid
import base64
import json
import re
import time
from datetime import datetime, timezone
from typing import AsyncGenerator
from urllib.request import urlopen

from mlx_engine.generate import load_model, create_generator, tokenize, load_draft_model
from transformers import AutoTokenizer, AutoProcessor
from mlx_engine.model_kit.model_kit import ModelKit
from mlx_engine.utils.token import Token

from api.api_models import (
    GenerateResponse,
    ChatResponse,
    OllamaResponseMessage,
    LogprobEntry,
    TopLogprobEntry,
    OllamaMessage,
    OllamaToolCall,
    OllamaToolCallFunction,
    Tool,
    ChatRequest,
    GenerationOptions,
    FunctionDefinition,
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatCompletionChunk,
    Choice,
    ChunkChoice,
    OpenAIResponseMessage,
    DeltaMessage,
    Usage,
    OpenAIToolCall,
    OpenAIFunctionCall,
    TextContent,
    ImageContent,
    OpenAISystemMessage,
    OpenAIUserMessage,
    OpenAIAssistantMessage,
    OpenAIToolMessage,
    OpenAIMessage,
)



model_cache = {"load_params": None, "model_kit": None}


def load_and_cache_model(model:str, num_ctx:int, kv_bits:int, kv_group_size:int, quantized_kv_start:int, draft_model:str | None):
    """
    Load and cache a model. Returns a tuple of (model_kit, load_duration_ns).
    load_duration_ns is the time spent loading the model in nanoseconds, or 0 if cached.
    """
    # Logic for Loading and Model Caching
    global model_cache

    load_params = {
        "model_path": f"models/{model}",
        "max_kv_size": num_ctx,
        "kv_bits": kv_bits,
        "kv_group_size": kv_group_size,
        "quantized_kv_start": quantized_kv_start,
        "draft_model": f"models/{draft_model}" if draft_model else None,
    }
    #Load Model from cache or clear cache if new load params requested
    if model_cache.get("load_params") == load_params:
        print("Model already loaded ✓", end="\n", flush=True)
        return model_cache.get("model_kit"), 0  # No load time for cached model
    else:
        if model_cache.get("model_kit") is not None:
            del model_cache["model_kit"]
            del model_cache["load_params"]
            mx.core.clear_cache()
            gc.collect()
            print("New Model Requetsted, Previous Model cleared from Cache ✓", end="\n", flush=True)

    load_start_time = time.time()
    print("Loading model...", end="\n", flush=True)
    model_kit = load_model(
        load_params["model_path"],
        max_kv_size=load_params["max_kv_size"],
        trust_remote_code=False,
        kv_bits=load_params["kv_bits"],
        kv_group_size=load_params["kv_group_size"],
        quantized_kv_start=load_params["quantized_kv_start"],
    )
    print("\rModel load complete ✓", end="\n", flush=True)

    # Load draft model if specified
    if load_params["draft_model"]:
        print(f"Loading draft model: {load_params['draft_model']}...", end="\n", flush=True)
        load_draft_model(model_kit, load_params["draft_model"])
        print("Draft model loaded ✓", end="\n", flush=True)

    load_end_time = time.time()
    load_duration_ns = int((load_end_time - load_start_time) * 1e9)

    # Update cache
    model_cache["model_kit"] = model_kit
    model_cache["load_params"] = load_params

    print("✅ Model ready for inference!", end="\n", flush=True)
    return model_kit, load_duration_ns

def prompt_render(raw:bool, images: list[str], prompt: str, system: str, model: str, suffix: str):
    if not raw:
        if images:
            user_content = []
            for x in range(images):
                user_content.append({"type": "image", "image": "image"+x})
            user_content.append({"type": "text", "text": prompt})
        else:
            user_content = prompt

        conversation = []
        if system:
            conversation.append({"role": "system", "content": system})
        conversation.append({"role": "user", "content": user_content})

        tf_tokenizer = model_cache["model_kit"].tokenizer._tokenizer
        prompt = tf_tokenizer.apply_chat_template(
            conversation, tokenize=False, add_generation_prompt=True
        )
        return tokenize(model_cache["model_kit"], prompt)
    else:
        # Raw mode: Handle FIM or simple concatenation
        if suffix:
            model_lower = model.lower()

            # Check if it's a Mistral/Ministral model
            if "ministral" in model_lower or "mistral" in model_lower or "codestral" in model_lower:
                # Mistral FIM format: [SUFFIX]suffix[PREFIX]prefix[MIDDLE]
                FIM_PREFIX = "[PREFIX]"
                FIM_SUFFIX = "[SUFFIX]"
                FIM_MIDDLE = "[MIDDLE]"

                print("✓ Using FIM mode with Mistral tokens")
                raw_prompt = (
                    f"{FIM_SUFFIX}{suffix}"
                    f"{FIM_PREFIX}"
                    f"{system + chr(10) if system else ''}{prompt}"
                    f"{FIM_MIDDLE}"
                )
            else:
                # Standard FIM tokens (DeepSeek, Qwen, CodeGemma format)
                FIM_PREFIX = "<|fim_prefix|>"
                FIM_SUFFIX = "<|fim_suffix|>"
                FIM_MIDDLE = "<|fim_middle|>"

                print("✓ Using FIM mode with standard tokens")
                raw_prompt = (
                    f"{FIM_PREFIX}"
                    f"{system + chr(10) if system else ''}{prompt}"
                    f"{FIM_SUFFIX}{suffix}"
                    f"{FIM_MIDDLE}"
                )
        else:
            # No suffix, simple concatenation
            raw_prompt = f"{system + chr(10) if system else ''}{prompt}"

        return tokenize(model_kit, raw_prompt)


class GenerationStatsCollector:
    def __init__(self):
        self.start_time = time.time()
        self.first_token_time = None
        self.total_tokens = 0
        self.num_accepted_draft_tokens: int | None = None
        self.load_duration: int | None = None  # Model load time in nanoseconds

    def add_tokens(self, tokens: list[Token]):
        """Record new tokens and their timing."""
        if self.first_token_time is None:
            self.first_token_time = time.time()

        draft_tokens = sum(1 for token in tokens if token.from_draft)
        if self.num_accepted_draft_tokens is None:
            self.num_accepted_draft_tokens = 0
        self.num_accepted_draft_tokens += draft_tokens

        self.total_tokens += len(tokens)

    def print_stats(self):
        """Print generation statistics."""
        end_time = time.time()
        total_time = end_time - self.start_time

        # Check if first token was generated
        if self.first_token_time is None:
            print("\n\nNo tokens generated")
            return

        time_to_first_token = self.first_token_time - self.start_time
        effective_time = total_time - time_to_first_token
        tokens_per_second = (
            self.total_tokens / effective_time if effective_time > 0 else float("inf")
        )
        print("\n\nGeneration stats:")
        print(f" - Tokens per second: {tokens_per_second:.2f}")
        if self.num_accepted_draft_tokens is not None:
            print(
                f" - Number of accepted draft tokens: {self.num_accepted_draft_tokens}"
            )
        print(f" - Time to first token: {time_to_first_token:.2f}s")
        print(f" - Total tokens generated: {self.total_tokens}")
        print(f" - Total time: {total_time:.2f}s")

    def get_tokens_per_second(self):
        """Calculate and return tokens per second."""
        end_time = time.time()
        total_time = end_time - self.start_time

        if self.first_token_time is None or total_time == 0:
            return 0

        time_to_first_token = self.first_token_time - self.start_time
        effective_time = total_time - time_to_first_token

        return self.total_tokens / effective_time if effective_time > 0 else 0

def build_logprobs(tokens: list[Token], top_logprobs: list | None, top_logprobs_count:int) -> list[LogprobEntry]:
    """
    Build logprob entries from tokens and their top logprobs.
    Matches Ollama's format with UTF-8 bytes populated.
    """
    entries = []
    if not top_logprobs:
        return entries

    for token, candidates in zip(tokens, top_logprobs):
        top_entries = [
            TopLogprobEntry(
                token=candidates[x].text,
                logprob=candidates[x].logprob,
                bytes=list(candidates[x].text.encode('utf-8'))
            ) for x in range(top_logprobs_count)
        ]

        entries.append(LogprobEntry(
            token=token.text,
            logprob=token.logprob,
            bytes=list(token.text.encode('utf-8')),
            top_logprobs=top_entries
        ))

    return entries


async def generate_stream(generator, model_name: str, stats_collector: GenerationStatsCollector, prompt_tokens: list[int], include_logprobs: bool, top_logprobs: int):
    """
    Stream generation results as JSON chunks matching Ollama /api/generate format.
    Uses 'response' field for text content.
    """
    generated_token_ids = []

    try:
        for generation_result in generator:
            # Check if client disconnected
            try:
                await asyncio.sleep(0)  # Yield control to check for cancellation
            except asyncio.CancelledError:
                print("Client disconnected, stopping generation")
                return

            done = generation_result.stop_condition is not None
            stats_collector.add_tokens(generation_result.tokens)

            # Collect generated token IDs for context
            generated_token_ids.extend(token.id for token in generation_result.tokens)

            chunk = {
                "model": model_name,
                "created_at": datetime.now(timezone.utc).isoformat(),
                "response": generation_result.text,
                "done": done,
            }

            # Include logprobs in every chunk when enabled
            if include_logprobs and generation_result.top_logprobs:
                chunk["logprobs"] = [entry.model_dump() for entry in build_logprobs(generation_result.tokens, generation_result.top_logprobs, top_logprobs)]

            if done:
                chunk["done_reason"] = generation_result.stop_condition.stop_reason
                # Add timing stats on final chunk
                end_time = time.time()
                total_time = end_time - stats_collector.start_time
                chunk["total_duration"] = int(total_time * 1e9)
                chunk["load_duration"] = stats_collector.load_duration
                chunk["prompt_eval_count"] = len(prompt_tokens)
                chunk["eval_count"] = stats_collector.total_tokens
                if stats_collector.first_token_time:
                    prompt_time = stats_collector.first_token_time - stats_collector.start_time
                    chunk["prompt_eval_duration"] = int(prompt_time * 1e9)
                    chunk["eval_duration"] = int((total_time - prompt_time) * 1e9)
                # Add context (prompt + generated tokens) for conversation memory
                chunk["context"] = prompt_tokens + generated_token_ids

            yield json.dumps(chunk) + "\n"
    except asyncio.CancelledError:
        print("Client disconnected, stopping generation")
        return


async def chat_stream(generator, model_name: str, stats_collector: GenerationStatsCollector, prompt_tokens: list[int], include_logprobs: bool, top_logprobs: int):
    """
    Stream chat results as JSON chunks matching Ollama /api/chat format.
    Uses ChatResponse model for consistent formatting.
    """
    try:
        for generation_result in generator:
            # Check if client disconnected
            try:
                await asyncio.sleep(0)  # Yield control to check for cancellation
            except asyncio.CancelledError:
                print("Client disconnected, stopping generation")
                return

            done = generation_result.stop_condition is not None
            stats_collector.add_tokens(generation_result.tokens)

            response = ChatResponse(
                model=model_name,
                created_at=datetime.now(timezone.utc).isoformat(),
                message=OllamaResponseMessage(role="assistant", content=generation_result.text),
                done=done,
            )

            # Add logprobs if enabled
            if include_logprobs and generation_result.top_logprobs:
                response.logprobs = build_logprobs(generation_result.tokens, generation_result.top_logprobs, top_logprobs)

            # Add stats on final chunk
            if done:
                response.done_reason = generation_result.stop_condition.stop_reason
                end_time = time.time()
                total_time = end_time - stats_collector.start_time
                response.total_duration = int(total_time * 1e9)
                response.load_duration = stats_collector.load_duration
                response.prompt_eval_count = len(prompt_tokens)
                response.eval_count = stats_collector.total_tokens
                if stats_collector.first_token_time:
                    prompt_time = stats_collector.first_token_time - stats_collector.start_time
                    response.prompt_eval_duration = int(prompt_time * 1e9)
                    response.eval_duration = int((total_time - prompt_time) * 1e9)

            yield response.model_dump_json() + "\n"
    except asyncio.CancelledError:
        print("Client disconnected, stopping generation")
        return


async def generate_output(
    generator, stats_collector: GenerationStatsCollector, generate_query, prompt_tokens: list[int]
):
    """
    Collect full generation output for non-streaming response.
    """
    result_text = ""
    generation_result = None
    logprobs_list = [] if generate_query.logprobs else None
    generated_token_ids = []

    for generation_result in generator:
        result_text += generation_result.text
        stats_collector.add_tokens(generation_result.tokens)

        # Collect generated token IDs for context
        generated_token_ids.extend(token.id for token in generation_result.tokens)

        if logprobs_list is not None and generation_result.top_logprobs:
            logprobs_list.extend(build_logprobs(generation_result.tokens, generation_result.top_logprobs))

    # Thinking process
    thinking_content = None
    if generate_query.think:
        re_think = re.compile(r"<think>(.*?)</think>", re.DOTALL)
        match = re_think.search(result_text)
        if match:
            extracted_thoughts = match.group(1).strip()
            # Remove the think block from response
            result_text = re_think.sub("", result_text).strip()
            thinking_content = result_text

    # Calculate durations and stats
    total_duration = None
    prompt_eval_duration = None
    eval_duration = None
    eval_count = None

    finish_reason = "length"
    if generation_result and generation_result.stop_condition:
        if generation_result.stop_condition.stop_reason == "stop_string":
            finish_reason = "stop"
        elif generation_result.stop_condition.stop_reason == "end_token":
            finish_reason = "end"

    # Calculate timing in nanoseconds
    end_time = time.time()
    total_time_sec = end_time - stats_collector.start_time
    total_duration = int(total_time_sec * 1e9)

    eval_count = stats_collector.total_tokens

    if stats_collector.first_token_time:
        prompt_eval_time_sec = stats_collector.first_token_time - stats_collector.start_time
        prompt_eval_duration = int(prompt_eval_time_sec * 1e9)
        gen_time_sec = total_time_sec - prompt_eval_time_sec
        eval_duration = int(gen_time_sec * 1e9)

    # Construct structured response
    final_logprobs = logprobs_list if (generate_query.logprobs and logprobs_list) else None

    return GenerateResponse(
            model = generate_query.model,
            created_at = datetime.now(timezone.utc).isoformat(),
            response = result_text,
            thinking = thinking_content,
            done = True,
            done_reason = finish_reason,
            total_duration = total_duration,
            load_duration = stats_collector.load_duration,
            prompt_eval_count = len(prompt_tokens),
            prompt_eval_duration = prompt_eval_duration,
            eval_count = eval_count,
            eval_duration = eval_duration,
            logprobs = final_logprobs,
            context = prompt_tokens + generated_token_ids,
            )

async def chat_render(messages: list[OllamaMessage], tools: list[Tool] | None, images: list[str]):
    tf_tokenizer = model_cache["model_kit"].tokenizer._tokenizer

    # Convert Pydantic models to dicts and collect images
    messages_dicts = []
    for msg in messages:
        if msg.images:
            # Build content with image placeholders first, then text
            content = []
            for i in range(len(msg.images)):
                content.append({"type": "image", "image": "image" + str(len(images) + i)})
            content.append({"type": "text", "text": msg.content})
            images.extend(msg.images)
            messages_dicts.append({"role": msg.role, "content": content})
        else:
            # No images - content is just a string
            msg_dict = {"role": msg.role, "content": msg.content}
            if msg.tool_calls:
                msg_dict["tool_calls"] = [
                    {"id": tc.id, "type": tc.type, "function": {"name": tc.function.name, "arguments": tc.function.arguments}}
                    for tc in msg.tool_calls
                ]
            if msg.tool_call_id:
                msg_dict["tool_call_id"] = msg.tool_call_id
            messages_dicts.append(msg_dict)

    tools_dicts = [tool.model_dump(exclude_none=True) for tool in tools] if tools else None

    prompt = tf_tokenizer.apply_chat_template(
        messages_dicts, tools=tools_dicts, tokenize=False, add_generation_prompt=True
    )
    return tokenize(model_cache["model_kit"], prompt)

def openai_to_chat_convert(req: ChatCompletionRequest):
    # Convert OpenAI messages to Chat messages
    chat_messages = []
    for msg in req.messages:
        content = ""
        images = None

        if isinstance(msg.content, str):
            content = msg.content
        elif isinstance(msg.content, list):
            # Handle multimodal content
            image_list = []
            for part in msg.content:
                if part.type == "text":
                    content = part.text
                elif part.type == "image_url":
                    # Read file from path and convert to base64
                    file_path = part.image_url.url
                    with open(file_path, "rb") as f:
                        base64_data = base64.b64encode(f.read()).decode("utf-8")
                    image_list.append(base64_data)
            if image_list:
                images = image_list

        # Convert OpenAI tool_calls to Ollama tool_calls
        tool_calls = None
        if hasattr(msg, 'tool_calls') and msg.tool_calls:
            tool_calls = [
                OllamaToolCall(
                    id=tc.id,
                    type=tc.type,
                    function=OllamaToolCallFunction(
                        name=tc.function.name,
                        arguments=json.loads(tc.function.arguments)
                    )
                )
                for tc in msg.tool_calls
            ]

        # Preserve tool_call_id for tool role messages
        tool_call_id = getattr(msg, 'tool_call_id', None)

        chat_messages.append(OllamaMessage(
            role=msg.role,
            content=content,
            images=images,
            tool_calls=tool_calls,
            tool_call_id=tool_call_id
        ))

    # Convert OpenAI tools to Chat tools
    chat_tools = None
    if req.tools:
        chat_tools = []
        for tool in req.tools:
            parameters_dict = None
            if tool.function.parameters:
                parameters_dict = tool.function.parameters.model_dump()
            chat_tools.append(Tool(
                type=tool.type,
                function=FunctionDefinition(
                    name=tool.function.name,
                    description=tool.function.description,
                    parameters=parameters_dict
                )
            ))

    # Convert response_format to format
    format_value = None
    if req.response_format:
        if req.response_format.type == "json_object":
            format_value = "json"
        elif req.response_format.type == "json_schema":
            if req.response_format.json_schema and req.response_format.json_schema.schema_:
                format_value = req.response_format.json_schema.schema_

    # Build GenerationOptions
    options = GenerationOptions(
        temperature=req.temperature,
        top_p=req.top_p,
        num_predict=req.max_tokens,
        stop=req.stop,
        seed=req.seed
    )

    # Create and return ChatRequest
    return ChatRequest(
        model=req.model,
        messages=chat_messages,
        tools=chat_tools,
        format=format_value,
        options=options,
        stream=req.stream if req.stream is not None else True
    )


# =============================================================================
# OpenAI Utility Functions
# =============================================================================


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


def parse_tool_calls(text: str) -> tuple[list[OpenAIToolCall] | None, str | None]:
    """
    Parse tool calls from model-generated text.
    Supports multiple formats:
    - <tool_call>{"name":"...","arguments":{...}}</tool_call> (Qwen/Hermes)
    - [TOOL_CALLS]name[ARGS]{"key":"value"} (Mistral)

    Returns:
        (tool_calls, remaining_content) where:
        - tool_calls: list of OpenAIToolCall objects if found, else None
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
            tool_calls.append(OpenAIToolCall(
                id=call_id,
                index=i,
                type="function",
                function=OpenAIFunctionCall(name=name, arguments=arguments),
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
                tool_calls.append(OpenAIToolCall(
                    id=call_id,
                    index=i,
                    type="function",
                    function=OpenAIFunctionCall(name=name, arguments=arguments),
                ))
            remaining = re.sub(xml_pattern, '', text, flags=re.DOTALL).strip()

    if not tool_calls:
        return None, text

    return tool_calls, remaining if remaining else None


def convert_openai_messages(
    messages: list[OpenAIMessage],
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
        if isinstance(msg, OpenAISystemMessage) or (hasattr(msg, 'role') and msg.role == "system"):
            messages_dicts.append({
                "role": "system",
                "content": msg.content
            })

        elif isinstance(msg, OpenAIUserMessage) or (hasattr(msg, 'role') and msg.role == "user"):
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

        elif isinstance(msg, OpenAIAssistantMessage) or (hasattr(msg, 'role') and msg.role == "assistant"):
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

        elif isinstance(msg, OpenAIToolMessage) or (hasattr(msg, 'role') and msg.role == "tool"):
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
    tool_calls: list[OpenAIToolCall] | None = None,
) -> dict:
    """
    Build OpenAI ChatCompletionResponse from generation results.
    """
    response_id = f"chatcmpl-{uuid.uuid4().hex[:24]}"

    response = ChatCompletionResponse(
        id=response_id,
        created=int(time.time()),
        model=model,
        choices=[
            Choice(
                index=0,
                message=OpenAIResponseMessage(
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


async def generate_openai_output(
    generator,
    stats_collector: GenerationStatsCollector,
    params: ChatCompletionRequest,
    prompt_token_count: int,
) -> ChatCompletionResponse:
    """
    Collect full generation output and return OpenAI response.
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

    # Parse tool calls from generated text
    tool_calls, remaining_content = parse_tool_calls(result_text)
    if tool_calls:
        finish_reason = "tool_calls"

    return build_openai_response(
        model=params.model,
        result_text=remaining_content,
        finish_reason=finish_reason,
        prompt_token_count=prompt_token_count,
        completion_token_count=stats_collector.total_tokens,
        tool_calls=tool_calls,
    )
