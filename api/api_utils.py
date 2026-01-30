import mlx as mx
import gc
from mlx_engine.generate import load_model, create_generator, tokenize, load_draft_model
from transformers import AutoTokenizer, AutoProcessor
from mlx_engine.model_kit.model_kit import ModelKit
import time
import json
from mlx_engine.utils.token import Token
import re
from datetime import datetime, timezone
from api.api_models import GenerateResponse, LogprobEntry, TopLogprobEntry



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

def prompt_render(raw:bool, images: list[str], prompt: str, system: str, model: str, model_kit: ModelKit, suffix: str):
    if not raw:
        if images:
            user_content = []
            for encoding in images:
                user_content.append({"type": "image", "image": encoding})
            user_content.append({"type": "text", "text": prompt})
        else:
            user_content = prompt

        conversation = [
            {"role": "system", "content": system}, 
            {"role": "user", "content": user_content}, 
        ]
        
        tf_tokenizer = model_cache["model_kit"].tokenizer._tokenizer
        prompt = tf_tokenizer.apply_chat_template(
            conversation, tokenize=False, add_generation_prompt=True
        )
        return tokenize(model_kit, prompt)
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
    Stream generation results as JSON chunks matching Ollama format.
    Logprobs are included in every chunk when enabled.
    """
    generated_token_ids = []

    for generation_result in generator:
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
            finish_reason = "stop_string"
        elif generation_result.stop_condition.stop_reason == "end_token":
            finish_reason = "end_token"
    
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
