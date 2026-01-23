from typing import Annotated
import base64
import time
import os
import gc
import mlx.core as mx

from mlx_engine.generate import load_model, create_generator, tokenize
from mlx_engine.utils.token import Token
from mlx_engine.utils.prompt_progress_reporter import LoggerReporter
from transformers import AutoTokenizer, AutoProcessor
from fastapi import FastAPI, Body
from fastapi.responses import StreamingResponse
from huggingface_hub import snapshot_download
import json
from mlx_engine.model_kit.model_kit import ModelKit
from mlx_engine.vision_model_kit.vision_model_kit import VisionModelKit

# Import schema from feilds.py
from feilds import (
    ChatCompletionParams,
    ChatCompletionResponse,
    Choice,
    Usage,
    MessageParam,
    ToolUseBlockParam,
    DEFAULT_TEMP,
    TextBlockParam,
    ContentBlockParam,
)

# Import tool formatting
from tools import format_tools_for_model, detect_tool_calls

from mlx_audio.tts.utils import load_model as load_tts_model

# remove this import after finished with audio
import sounddevice as sd


app = FastAPI()
loaded_model_cache = {"model": None, "params": {}}


@app.put("/download")
async def download(repo_id: str):
    creator, model = repo_id.split("/")
    snapshot_download(repo_id=repo_id, local_dir=f"./models/{creator}/{model}")


# Schemas now imported from feilds.py


def image_to_base64(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")


class GenerationStatsCollector:
    def __init__(self):
        self.start_time = time.time()
        self.first_token_time = None
        self.total_tokens = 0
        self.num_accepted_draft_tokens: int | None = None

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


@app.get("/v1/models")
async def models():
    models = []
    for model in os.listdir("./models"):
        for file in os.listdir(f"./models/{model}"):
            models.append(model + "/" + file)
    models.sort()
    return models


async def generate_stream(generator):
    for generation_result in generator:
        # print(generation_result.text, end="", flush=True)
        yield json.dumps({"content": generation_result.text, "ID": "12345"})


async def generate_output(generator, stats_collector, logprobs_list, generate_query):
    result_text = ""
    generation_result = None

    for generation_result in generator:
        result_text += generation_result.text
        stats_collector.add_tokens(generation_result.tokens)
        logprobs_list.extend(generation_result.top_logprobs)
    finish_reason = "length"
    if generation_result and generation_result.stop_condition:
        stats_collector.print_stats()
        print(
            f"\nStopped generation due to: {generation_result.stop_condition.stop_reason}"
        )
        if generation_result.stop_condition.stop_string:
            print(f"Stop string: {generation_result.stop_condition.stop_string}")

        if generation_result.stop_condition.stop_reason == "stop_string":
            finish_reason = "stop"
        elif generation_result.stop_condition.stop_reason == "end_token":
            finish_reason = "stop"

    if generate_query.top_logprobs:
        [print(x) for x in logprobs_list]

    # Detect tool calls
    content: list[ContentBlockParam] = []
    tool_calls = detect_tool_calls(result_text, generate_query.model)

    if tool_calls:
        finish_reason = "tool_calls"
        content.extend(tool_calls)

        # Add text if detected and not empty
        if len(result_text.strip()) > 0:
            content.insert(0, TextBlockParam(type="text", text=result_text))
    else:
        # Standard text response
        content = [TextBlockParam(type="text", text=result_text)]

    # Construct structured response
    return ChatCompletionResponse(
        id=f"chatcmpl-{int(time.time())}",
        created=int(time.time()),
        model=generate_query.model,
        choices=[
            Choice(
                index=0,
                message=MessageParam(role="assistant", content=content),
                finish_reason=finish_reason,
                logprobs=None,
            )
        ],
        usage=Usage(
            prompt_tokens=0,  # TODO: Track prompt tokens
            completion_tokens=stats_collector.total_tokens,
            total_tokens=stats_collector.total_tokens,
        ),
    )


@app.post("/v1/messages")
async def generate(generate_query: Annotated[ChatCompletionParams, Body()]):
    global loaded_model_cache

    current_params = {
        "model_path": f"models/{generate_query.model}",
        "max_kv_size": generate_query.max_kv_size,
        "kv_bits": generate_query.kv_bits,
        "kv_group_size": generate_query.kv_group_size,
        "quantized_kv_start": generate_query.quantized_kv_start,
    }

    if loaded_model_cache["params"] == current_params:
        model_kit = loaded_model_cache["model"]
        print("Model already loaded ✓", end="\n", flush=True)
    else:
        # Clear previous model implementation
        if loaded_model_cache["model"] != None:
            del loaded_model_cache["model"]
            mx.clear_cache()
            gc.collect()
            print("Model cleared ✓", end="\n", flush=True)
        print("Loading model...", end="\n", flush=True)
        print(current_params)
        model_kit = load_model(
            current_params["model_path"],
            max_kv_size=current_params["max_kv_size"],
            trust_remote_code=False,
            kv_bits=current_params["kv_bits"],
            kv_group_size=current_params["kv_group_size"],
            quantized_kv_start=current_params["quantized_kv_start"],
        )
        print("\rModel load complete ✓", end="\n", flush=True)

        loaded_model_cache["model"] = model_kit
        loaded_model_cache["params"] = current_params

    tf_tokenizer = AutoProcessor.from_pretrained(generate_query.model)
    images_base64 = []
    for message in generate_query.messages:
        if message.role == "user" and isinstance(message.content, list):
            for content in message.content:
                # Handle ImageBlockParam
                if content.type == "image":
                    # Check source type and extract appropriately
                    if content.source.type == "url":
                        images_base64.append(image_to_base64(content.source.url))  # type: ignore
                    elif content.source.type == "base64":
                        images_base64.append(content.source.data)  # type: ignore
    # Build conversation with optional system prompt and tool definitions
    conversation = generate_query.messages
    # Add tools to system message if provided
    if generate_query.tools:
        tools_text = format_tools_for_model(generate_query.tools, generate_query.model)
        # Prepend tools to system prompt or create new system message
        if generate_query.system:
            system_content = (
                generate_query.system
                if isinstance(generate_query.system, str)
                else generate_query.system[0].text
            )
            system_content = f"{tools_text}\n\n{system_content}"
        else:
            system_content = tools_text

        # Insert system message at beginning (dict format for chat template)
        conversation = [{"role": "system", "content": system_content}] + list(
            generate_query.messages
        )

    tf_tokenizer = AutoTokenizer.from_pretrained(generate_query.model)
    prompt = tf_tokenizer.apply_chat_template(
        conversation, tokenize=False, add_generation_prompt=True
    )
    prompt_tokens = tokenize(model_kit, prompt)
    # Record top logprobs
    logprobs_list = []

    # Initialize generation stats collector
    stats_collector = GenerationStatsCollector()

    # Generate the response
    generator = create_generator(
        model_kit,
        prompt_tokens,
        images_b64=images_base64,
        stop_strings=generate_query.stop_sequences,
        max_tokens=generate_query.max_tokens,
        top_logprobs=generate_query.top_logprobs,
        prompt_progress_reporter=LoggerReporter(),
        num_draft_tokens=generate_query.num_draft_tokens,
        temp=generate_query.temperature,
        top_k=generate_query.top_k,
        top_p=generate_query.top_p,
    )
    if generate_query.stream:
        return StreamingResponse(generate_stream(generator))
    else:
        return await generate_output(
            generator, stats_collector, logprobs_list, generate_query
        )


# async def stream_tts(output):
#     for result in output:
#         yield result.audio


# @app.post("/v1/audio")
# async def tts(tts_query: str, voice: str = "af_heart"):
#     model = load_tts_model(path="models/mlx-community/Kokoro-82M-bf16")
#     output = model.generate(tts_query, voice=voice)
#     for result in output:
#         sd.play(result.audio, 24000)
#         sd.wait()
#     return StreamingResponse(stream_tts(output))
