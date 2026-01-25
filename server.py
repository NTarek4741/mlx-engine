from typing import Annotated

import os
import gc
import mlx.core as mx

from mlx_engine.generate import load_model, create_generator, tokenize
from mlx_engine.utils.prompt_progress_reporter import LoggerReporter
from transformers import AutoTokenizer, AutoProcessor
from fastapi import FastAPI, Body
from fastapi.responses import StreamingResponse
from huggingface_hub import snapshot_download

from feilds import ChatCompletionParams

# Import tool formatting
from tools import format_tools_for_model

from mlx_audio.tts.utils import load_model as load_tts_model

# remove this import after finished with audio
import sounddevice as sd

# Import model conversion utilities
from mlx_lm import convert as mlx_lm_convert
from mlx_vlm import convert as mlx_vlm_convert

# Import utility functions
from server_utils import (
    image_to_base64,
    GenerationStatsCollector,
    generate_stream,
    generate_output,
)


app = FastAPI()
loaded_model_cache = {"model": None, "params": {}}


@app.put("v1/download")
async def download(repo_id: str):
    """
    Download a model from Hugging Face Hub.

    This endpoint downloads the specified model repository from Hugging Face to the local
    `./models` directory. It organizes models by creator and model name.

    Args:
        repo_id: The Hugging Face repository ID in the format "creator/model_name"
                 (e.g., "meta-llama/Llama-2-7b-hf").

    Returns:
        dict: A JSON object indicating success or failure.
            Success format:
            {
                "status": "success",
                "message": "Model {repo_id} downloaded successfully"
            }
            Error format:
            {
                "status": "error",
                "message": "{error_details}"
            }
    """
    try:
        creator, model = repo_id.split("/")
        snapshot_download(repo_id=repo_id, local_dir=f"./models/{creator}/{model}")
        return {
            "status": "success",
            "message": f"Model {repo_id} downloaded successfully",
        }
    except Exception as e:
        return {"status": "error", "message": str(e)}


@app.get("/v1/models")
async def models():
    """
    List available models in the local `models` directory.

    Scans the `./models` directory and returns a list of all downloaded models
    that are available for loading.

    Returns:
        list[str]: A list of model paths in the format "creator/model_name".
                   Example: ["mlx-community/Llama-2-7b-mlx", "google/gemma-7b"]
    """
    models = []
    for model in os.listdir("./models"):
        for file in os.listdir(f"./models/{model}"):
            models.append(model + "/" + file)
    models.sort()
    return models


@app.post("/v1/convert")
async def convert_model(repo_id: str, output_dir: str = None):
    """
    Convert a Hugging Face model to MLX format.

    This endpoint attempts to convert a model using `mlx_lm.convert` first (for text models).
    If that fails, it tries `mlx_vlm.convert` (for vision-language models).
    It applies 4-bit quantization by default.

    Args:
        repo_id: Hugging Face repository ID (e.g., "meta-llama/Llama-2-7b-hf").
                 Must be a repo that exists on HF or locally.
        output_dir: Optional output directory. If not provided, defaults to `./models/{repo_id}`.

    Returns:
        dict: A JSON object with conversion status and details.
            Success format:
            {
                "status": "success",
                "message": "...",
                "output_path": "...",
                "converter": "mlx_lm" | "mlx_vlm",
                "quantization": "4-bit"
            }
            Error format:
            {
                "status": "error",
                "message": "Model conversion failed",
                "details": { "mlx_lm_error": "...", "mlx_vlm_error": "..." }
            }
    """
    # Set default output directory if not provided
    if output_dir is None:
        output_dir = f"./models/{repo_id}"
    try:
        # Second attempt: Try mlx_vlm.convert with 4-bit quantization
        print(
            f"Attempting to convert {repo_id} using mlx_vlm.convert with 4-bit quantization..."
        )
        mlx_vlm_convert(
            hf_path=repo_id,
            mlx_path=output_dir,
            quantize=True,
            q_group_size=64,
            q_bits=4,
        )
        return {
            "status": "success",
            "message": f"Model {repo_id} successfully converted using mlx_vlm with 4-bit quantization",
            "output_path": output_dir,
            "converter": "mlx_vlm",
            "quantization": "4-bit",
        }
    except Exception as vlm_error:
        print(f"mlx_vlm.convert failed: {str(vlm_error)}")
    try:
        # First attempt: Try mlx_lm.convert with 4-bit quantization
        print(
            f"Attempting to convert {repo_id} using mlx_lm.convert with 4-bit quantization..."
        )
        mlx_lm_convert(
            hf_path=repo_id,
            mlx_path=output_dir,
            quantize=True,
            q_group_size=64,
            q_bits=4,
        )
        return {
            "status": "success",
            "message": f"Model {repo_id} successfully converted using mlx_lm with 4-bit quantization",
            "output_path": output_dir,
            "converter": "mlx_lm",
            "quantization": "4-bit",
        }
    except Exception as lm_error:
        print(f"mlx_lm.convert failed: {str(lm_error)}")

    # Both converters failed
    return {
        "status": "error",
        "message": "Model conversion failed",
        "details": {
            "mlx_lm_error": str(lm_error),
            "mlx_vlm_error": str(vlm_error),
        },
    }


@app.post("/v1/messages")
async def generate(generate_query: Annotated[ChatCompletionParams, Body()]):
    """
    Generate a chat completion response.

    This endpoint accepts a chat history and generation parameters to produce a response
    from the loaded model. It supports both text and vision inputs (if the model supports vision).

    Args:
        generate_query (ChatCompletionParams): A Pydantic model containing:
            - messages (list): List of message objects (role, content). Content can be text or list of content blocks (text/image).
            - model (str): Path/name of the model to use (must be capable of loading).
            - temperature (float): Sampling temperature.
            - max_tokens (int): Maximum tokens to generate.
            - stream (bool): Whether to stream the response.
            - ... (other params like top_p, tools, etc.)

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
            Yields chunks of JSON data containing generated text deltas.
    """
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
            generator,
            stats_collector,
            logprobs_list,
            generate_query,
            len(prompt_tokens),
        )


async def stream_tts(output):
    for result in output:
        yield result.audio


@app.post("/v1/audio")
async def tts(tts_query: str, voice: str = "af_heart"):
    model = load_tts_model(path="models/mlx-community/Kokoro-82M-bf16")
    output = model.generate(tts_query, voice=voice)
    # for testing purposes
    for result in output:
        sd.play(result.audio, 24000)
        sd.wait()
    return StreamingResponse(stream_tts(output))
