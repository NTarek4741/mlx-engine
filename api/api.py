from typing import Annotated

import os
import gc
import json
import time
import mlx.core as mx

from mlx_engine.vision_model_kit.vision_model_kit import VisionModelKit
from mlx_engine.utils.prompt_progress_reporter import LoggerReporter
from fastapi import FastAPI, Body
from fastapi.responses import StreamingResponse
from huggingface_hub import snapshot_download
from mlx_engine.generate import create_generator

from api.api_models import GenerateRequest, ChatRequest, GenerationOptions

# Import utility functions
from api.api_utils import (
    load_and_cache_model, 
    prompt_render,
    GenerationStatsCollector,
    generate_stream,
    generate_output,
)


app = FastAPI()

# POST - Generate a response
@app.post("/api/generate")
async def generate(request: Annotated[GenerateRequest, Body]):
    # Use default GenerationOptions if not provided
    options = request.options or GenerationOptions()

    model_kit, load_duration_ns = load_and_cache_model(
        model = request.model,
        num_ctx = options.num_ctx,
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
        model_kit,
        request.suffix
    )

    stats_collector = GenerationStatsCollector()
    stats_collector.load_duration = load_duration_ns

    # Handle JSON schema validation
    json_schema = None
    if request.format:
        json_schema = json.dumps(request.format)

    # Handle stop strings - can be None, str, or list[str]

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
    """Generate a chat message"""
    return {"message": "Endpoint available"}


# POST - Generate embeddings
@app.post("/api/embeddings")
async def embeddings():
    """Generate embeddings"""
    return {"embeddings": []}


# GET - List models
@app.get("/api/tags")
async def list_models():
    """List models"""
    return {"models": []}


# GET - List running models
@app.get("/api/ps")
async def list_running_models():
    """List running models"""
    return {"models": []}


# POST - Show model details
@app.post("/api/show")
async def show_model():
    """Show model details"""
    return {"details": "Endpoint available"}


# POST - Create a model
@app.post("/api/create")
async def create_model():
    """Create a model"""
    return {"status": "success"}


# POST - Copy a model
@app.post("/api/copy")
async def copy_model():
    """Copy a model"""
    return {"status": "success"}


# POST - Pull a model
@app.post("/api/pull")
async def pull_model():
    """Pull a model"""
    return {"status": "success"}


# POST - Push a model
@app.post("/api/push")
async def push_model():
    """Push a model"""
    return {"status": "success"}


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
