import base64
import time
import json
from mlx_engine.utils.token import Token
from feilds import (
    ChatCompletionResponse,
    Choice,
    Usage,
    MessageParam,
    TextBlockParam,
    ContentBlockParam,
)
from tools import detect_tool_calls


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

    def get_tokens_per_second(self):
        """Calculate and return tokens per second."""
        end_time = time.time()
        total_time = end_time - self.start_time

        if self.first_token_time is None or total_time == 0:
            return 0

        time_to_first_token = self.first_token_time - self.start_time
        effective_time = total_time - time_to_first_token

        return self.total_tokens / effective_time if effective_time > 0 else 0


async def generate_stream(generator):
    for generation_result in generator:
        # print(generation_result.text, end="", flush=True)
        yield json.dumps({"content": generation_result.text, "ID": "12345"})


async def generate_output(
    generator, stats_collector, logprobs_list, generate_query, prompt_token_count
):
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
            prompt_tokens=prompt_token_count,
            completion_tokens=stats_collector.total_tokens,
            total_tokens=prompt_token_count + stats_collector.total_tokens,
            tokens_per_second=round(stats_collector.get_tokens_per_second(), 2),
        ),
    )
