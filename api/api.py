from typing import Annotated

import os
import gc
import json
import time
import re
import hashlib
from datetime import datetime, timezone, timedelta
import mlx.core as mx

from mlx_engine.vision_model_kit.vision_model_kit import VisionModelKit
from mlx_engine.utils.prompt_progress_reporter import LoggerReporter
from fastapi import FastAPI, Body
from fastapi.responses import StreamingResponse
from huggingface_hub import snapshot_download
from mlx_engine.generate import create_generator
from huggingface_hub import snapshot_download
from mlx_lm import convert as mlx_lm_convert
from mlx_vlm import convert as mlx_vlm_convert


from api.api_models import (
    GenerateRequest,
    ChatRequest,
    GenerationOptions,
    TagsResponse,
    ModelInfo,
    PSResponse,
    RunningModelInfo,
)

# Import utility functions
from api.api_utils import (
    load_and_cache_model,
    prompt_render,
    GenerationStatsCollector,
    generate_stream,
    chat_stream,
    generate_output,
    chat_render,
    model_cache
)


app = FastAPI()

# POST - Generate a response
@app.post("/api/generate")
async def generate(request: Annotated[GenerateRequest, Body]):
    # Strip Ollama-style tags (e.g., ":latest") from model name
    model_name = request.model.split(":")[0] if ":" in request.model else request.model

    # Use default GenerationOptions if not provided
    options = request.options or GenerationOptions()

    model_kit, load_duration_ns = load_and_cache_model(
        model = model_name,
        num_ctx = None,
        kv_bits = options.kv_bits,
        kv_group_size = options.kv_group_size,
        quantized_kv_start = options.quantized_kv_start,
        draft_model = options.draft_model
    )

    prompt_tokens = prompt_render(
        request.raw,
        request.images,
        request.prompt,
        request.system,
        request.model,
        request.suffix
    )

    stats_collector = GenerationStatsCollector()
    stats_collector.load_duration = load_duration_ns

    # Handle JSON schema validation
    json_schema = None
    if request.format:
        json_schema = json.dumps(request.format)

    # Generate the response
    generator = create_generator(
        model_kit,
        prompt_tokens,
        images_b64=request.images,
        stop_strings=options.stop,
        max_tokens=options.num_predict,
        top_logprobs=request.top_logprobs if request.logprobs else None,
        prompt_progress_reporter=LoggerReporter(),
        seed=options.seed,
        temp=options.temperature,
        top_k=options.top_k,
        top_p=options.top_p,
        min_p=options.min_p,
        json_schema=json_schema,
    )

    if request.stream:
        return StreamingResponse(
            generate_stream(generator, request.model, stats_collector, prompt_tokens, request.logprobs, request.top_logprobs)
        )
    else:
        return await generate_output(generator, stats_collector, request, prompt_tokens)


# POST - Generate a chat message
@app.post("/api/chat")
async def chat(request: Annotated[ChatRequest, Body]):
    # Strip Ollama-style tags (e.g., ":latest") from model name
    model_name = request.model.split(":")[0] if ":" in request.model else request.model

    options = request.options or GenerationOptions()

    model_kit, load_duration_ns =  load_and_cache_model(
        model = model_name,
        num_ctx = options.num_ctx,
        kv_bits = options.kv_bits,
        kv_group_size = options.kv_group_size,
        quantized_kv_start = options.quantized_kv_start,
        draft_model = options.draft_model
    )
    images = []
    prompt_tokens = await chat_render(
        request.messages,
        request.tools,
        images
    )

    stats_collector = GenerationStatsCollector()
    stats_collector.load_duration = load_duration_ns

    # Handle JSON schema validation
    json_schema = None
    if request.format:
        json_schema = json.dumps(request.format)

    # Handle stop strings - can be None, str, or list[str]
    print("------------------------------------------------------")
    print(images)
    generator = create_generator(
        model_kit,
        prompt_tokens,
        images_b64=images,
        stop_strings=options.stop,
        max_tokens=options.num_predict,
        top_logprobs=request.top_logprobs if request.logprobs else None,
        prompt_progress_reporter=LoggerReporter(),
        seed=options.seed,
        temp=options.temperature,
        top_k=options.top_k,
        top_p=options.top_p,
        min_p=options.min_p,
        json_schema=json_schema,
    )

    if request.stream:
        return StreamingResponse(
            chat_stream(generator, model_name, stats_collector, prompt_tokens, request.logprobs, request.top_logprobs)
        )
    else:
        return await generate_output(generator, stats_collector, request, prompt_tokens)

# POST - Generate embeddings
@app.post("/api/embeddings")
async def embeddings():
    """Generate embeddings"""
    return {"embeddings": []}

 




#Conveniant Endpoints ____________________

# GET - List models
@app.get("/api/tags")
async def list_models() -> TagsResponse:
    models = []
    models_dir = "./models"

    for org in os.listdir(models_dir):
        org_path = os.path.join(models_dir, org)

        for model_name in os.listdir(org_path):
            model_path = os.path.join(org_path, model_name)

            # Use Ollama-style naming with :latest tag
            full_name = f"{org}/{model_name}:latest"

            model_info = ModelInfo(
                name=full_name,
                model=full_name,
            )
            models.append(model_info)

    models.sort(key=lambda m: m.name)
    return TagsResponse(models=models)


# GET - List running models
@app.get("/api/ps")
async def list_running_models() -> PSResponse:
    """List currently loaded/running models."""
    models = [] 

    # Check if a model is currently loaded
    if model_cache.get("model_kit") is not None and model_cache.get("load_params") is not None:
        load_params = model_cache["load_params"]
        model_path = load_params["model_path"]

        # Extract model name from path (e.g., "models/google/gemma-7b" -> "google/gemma-7b:latest")
        model_name = model_path.replace("models/", "", 1) if model_path.startswith("models/") else model_path
        model_name = f"{model_name}:latest"  # Add Ollama-style tag

        model_info = RunningModelInfo(model=model_name)
        models.append(model_info)
        print(models)
    return PSResponse(models=models)


# POST - Show model details
@app.post("/api/show")
async def show_model():
    """Show model details"""
    return {"details": "Endpoint available"}


# POST - Create a model
@app.post("/api/create")
async def create_model(repo_id:str, output_dir: str| None = None):
     # Set default output directory if not provided
    if output_dir is None:
        output_dir = f"./models/{repo_id}"
    
    vlm_error = None
    lm_error = None
    
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
    except Exception as e:
        vlm_error = e
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
    except Exception as e:
        lm_error = e
        print(f"mlx_lm.convert failed: {str(lm_error)}")

    # Both converters failed
    return {
        "status": "error",
        "message": "Model conversion failed",
        "details": {
            "mlx_lm_error": str(lm_error) if lm_error else "Not attempted/Unknown",
            "mlx_vlm_error": str(vlm_error) if vlm_error else "Not attempted/Unknown",
        },
    }


# POST - Copy a model
@app.post("/api/copy")
async def copy_model():
    """Copy a model"""
    return {"status": "success"}


# POST - Pull a model
@app.post("/api/pull")
async def pull_model(repo_id:str):
    try:
        creator, model = repo_id.split("/")
        snapshot_download(repo_id=repo_id, local_dir=f"./models/{creator}/{model}")
        return {
            "status": "success",
            "message": f"Model {repo_id} downloaded successfully to ./models/{creator}/download-{model}",
        }
    except Exception as e:
        return {"status": "error", "message": str(e)}


# DELETE - Delete a model
@app.delete("/api/delete") 
async def delete_model():
    """Delete a model"""
    return {"status": "success"}

# GET - Get version
@app.get("/api/version")
async def get_version():
    """Get version"""
    return {"version": "0.1.0"}
