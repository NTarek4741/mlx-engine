from typing import Annotated

import os
import gc
import mlx.core as mx
from mlx_engine.generate import load_model, create_generator, tokenize, load_draft_model
from mlx_engine.utils.prompt_progress_reporter import LoggerReporter
from transformers import AutoTokenizer, AutoProcessor
from fastapi import FastAPI, Body
from fastapi.responses import StreamingResponse

from anthropic.anthropic_models import MessagesParams, SpeachModel

app = FastAPI()
# loaded_model_cache = {"model": None, "params": {}}


# @app.post("/v1/messages")
# async def generate(generate_query: Annotated[MessagesParams, Body()]):
#     """
#     Generate a chat completion response.

#     This endpoint accepts a chat history and generation parameters to produce a response
#     from the loaded model. It supports both text and vision inputs (if the model supports vision).

#     Args:
#         generate_query (ChatCompletionParams): A Pydantic model containing:
#             - messages (list): List of message objects (role, content). Content can be text or list of content blocks (text/image).
#             - model (str): Path/name of the model to use (must be capable of loading).
#             - temperature (float): Sampling temperature.
#             - max_tokens (int): Maximum tokens to generate.
#             - stream (bool): Whether to stream the response.
#             - ... (other params like top_p, tools, etc.)

#     Returns:
#         ChatCompletionResponse (JSON) if stream=False:
#             {
#                 "id": "...",
#                 "created": 1234567890,
#                 "model": "model_name",
#                 "choices": [{
#                     "index": 0,
#                     "message": { "role": "assistant", "content": [...] },
#                     "finish_reason": "stop" | "length" | "tool_calls"
#                 }],
#                 "usage": { ... }
#             }

#         StreamingResponse (SSE) if stream=True:
#             Yields chunks of JSON data containing generated text deltas.
#     """
#     # Load model or get from cache
#     model_kit = load_or_get_cached_model(generate_query)

#     # Image Extraction
#     tf_tokenizer = AutoProcessor.from_pretrained(generate_query.model)
#     images_base64 = []
#     for message in generate_query.messages:
#         if message.role == "user" and isinstance(message.content, list):
#             for content in message.content:
#                 # Handle ImageBlockParam
#                 if content.type == "image":
#                     # Check source type and extract appropriately
#                     if content.source.type == "url":
#                         images_base64.append(image_to_base64(content.source.url))  # type: ignore
#                     elif content.source.type == "base64":
#                         images_base64.append(content.source.data)  # type: ignore
#     # Build conversation with optional system prompt and tool definitions
#     conversation = generate_query.messages
#     # Insert system message at beginning (dict format for chat template)
#     conversation = [{"role": "system", "content": generate_query.system}] + list(
#         generate_query.messages
#     )

#     tf_tokenizer = AutoTokenizer.from_pretrained(generate_query.model)
#     prompt = tf_tokenizer.apply_chat_template(
#         conversation, tokenize=False, add_generation_prompt=True
#     )
#     prompt_tokens = tokenize(model_kit, prompt)
#     # Initialize generation stats collector
#     stats_collector = GenerationStatsCollector()
#     logprobs_list = []

#     # Generate the response
#     generator = create_generator(
#         model_kit,
#         prompt_tokens,
#         images_b64=images_base64,
#         stop_strings=generate_query.stop_sequences,
#         max_tokens=generate_query.max_tokens,
#         top_logprobs=generate_query.top_logprobs,
#         prompt_progress_reporter=LoggerReporter(),
#         num_draft_tokens=generate_query.num_draft_tokens,
#         temp=generate_query.temperature,
#         top_k=generate_query.top_k,
#         top_p=generate_query.top_p,
#     )
#     if generate_query.stream:
#         return StreamingResponse(generate_stream(generator))
#     else:
#         return await generate_output(
#             generator,
#             stats_collector,
#             logprobs_list,
#             generate_query,
#             len(prompt_tokens),
#         )

