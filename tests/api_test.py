"""
Comprehensive API test suite for MLX Engine.

Tests all three API compatibility layers (Ollama, OpenAI, Anthropic)
using their respective Python SDKs and raw httpx.

Prerequisites:
    1. Start the server: uvicorn api.api:app --host 0.0.0.0 --port 8000
    2. Ensure model exists: models/mistralai/Ministral-3-3B-Instruct-2512/

Usage:
    python tests/api_test.py          # Install deps + run all tests
    pytest tests/api_test.py -v       # Run with pytest directly
"""

import subprocess
import sys


def ensure_packages():
    """Install required SDK packages if not present."""
    packages = ["openai", "anthropic", "ollama", "httpx", "pytest", "pytest-asyncio"]
    missing = []
    for pkg in packages:
        try:
            __import__(pkg)
        except ImportError:
            missing.append(pkg)
    if missing:
        print(f"Installing missing packages: {', '.join(missing)}")
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", *missing],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )


ensure_packages()

import json
import os
import pytest
import httpx
import openai
import ollama

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

BASE_URL = "http://localhost:8000"
MODEL_NAME = os.environ.get("TEST_MODEL", "mistralai/Ministral-3-3B-Instruct-2512")
TIMEOUT = 300  # seconds — model first-load can be slow

# Shared tool definition used across tool-calling tests
WEATHER_TOOL_FUNCTION = {
    "name": "get_weather",
    "description": "Get the current weather for a location",
    "parameters": {
        "type": "object",
        "properties": {
            "location": {
                "type": "string",
                "description": "City name",
            },
            "unit": {
                "type": "string",
                "enum": ["celsius", "fahrenheit"],
            },
        },
        "required": ["location"],
    },
}

# Shared JSON schema for structured output tests
COLOR_SCHEMA = {
    "type": "object",
    "properties": {
        "colors": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "hex": {"type": "string"},
                },
                "required": ["name", "hex"],
            },
        }
    },
    "required": ["colors"],
}


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def http_client():
    with httpx.Client(base_url=BASE_URL, timeout=TIMEOUT) as client:
        yield client


@pytest.fixture(scope="module")
def openai_client():
    return openai.OpenAI(
        base_url=f"{BASE_URL}/v1",
        api_key="test-key",
        timeout=httpx.Timeout(TIMEOUT),
    )


@pytest.fixture(scope="module")
def ollama_client():
    return ollama.Client(host=BASE_URL)


@pytest.fixture(scope="session", autouse=True)
def check_server_running():
    """Verify the MLX Engine server is running before any tests."""
    try:
        r = httpx.get(f"{BASE_URL}/api/version", timeout=5)
        r.raise_for_status()
    except (httpx.ConnectError, httpx.TimeoutException, httpx.HTTPStatusError):
        pytest.exit(
            "MLX Engine server is not running. Start it with:\n"
            "  uvicorn api.api:app --host 0.0.0.0 --port 8000\n"
            "Then re-run the tests.",
            returncode=1,
        )


# ===========================================================================
# 1. Server Health
# ===========================================================================


class TestServerHealth:
    def test_server_reachable(self, http_client: httpx.Client):
        r = http_client.get("/api/version")
        assert r.status_code == 200
        data = r.json()
        assert "version" in data

    def test_list_models(self, http_client: httpx.Client):
        r = http_client.get("/api/tags")
        assert r.status_code == 200
        data = r.json()
        assert "models" in data
        assert isinstance(data["models"], list)
        assert len(data["models"]) >= 1

    def test_list_running_models(self, http_client: httpx.Client):
        r = http_client.get("/api/ps")
        assert r.status_code == 200
        data = r.json()
        assert "models" in data

    def test_embeddings_placeholder(self, http_client: httpx.Client):
        r = http_client.post("/api/embeddings")
        assert r.status_code == 200
        data = r.json()
        assert "embeddings" in data


# ===========================================================================
# 2. Ollama Generate (/api/generate)
# ===========================================================================


class TestOllamaGenerate:
    def test_generate_non_streaming(self, http_client: httpx.Client):
        r = http_client.post(
            "/api/generate",
            json={
                "model": MODEL_NAME,
                "prompt": "Say hello in one sentence.",
                "stream": False,
                "options": {"num_predict": 50},
            },
        )
        assert r.status_code == 200
        data = r.json()
        assert data["done"] is True
        assert len(data["response"]) > 0
        assert data["eval_count"] > 0
        assert data["total_duration"] > 0
        assert data["prompt_eval_count"] > 0

    def test_generate_streaming(self, http_client: httpx.Client):
        with http_client.stream(
            "POST",
            "/api/generate",
            json={
                "model": MODEL_NAME,
                "prompt": "Count to 5.",
                "stream": True,
                "options": {"num_predict": 60},
            },
        ) as response:
            assert response.status_code == 200
            chunks = []
            for line in response.iter_lines():
                if line:
                    chunks.append(json.loads(line))

        assert len(chunks) >= 2, "Expected at least 2 streaming chunks"
        # Intermediate chunks should have done=False
        for chunk in chunks[:-1]:
            assert chunk["done"] is False
        # Final chunk
        final = chunks[-1]
        assert final["done"] is True
        assert "done_reason" in final
        assert final["eval_count"] > 0
        # Concatenated text is non-empty
        full_text = "".join(c.get("response", "") for c in chunks)
        assert len(full_text) > 0

    def test_generate_with_system_prompt(self, http_client: httpx.Client):
        r = http_client.post(
            "/api/generate",
            json={
                "model": MODEL_NAME,
                "prompt": "What are you?",
                "system": "You are a pirate. Always respond like a pirate.",
                "stream": False,
                "options": {"num_predict": 60},
            },
        )
        assert r.status_code == 200
        data = r.json()
        assert data["done"] is True
        assert len(data["response"]) > 0

    def test_generate_with_json_schema(self, http_client: httpx.Client):
        """GenerateRequest.format expects ResponseFormat type (not raw dict)."""
        schema = {
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "age": {"type": "integer"},
            },
            "required": ["name", "age"],
        }
        r = http_client.post(
            "/api/generate",
            json={
                "model": MODEL_NAME,
                "prompt": "Generate a person with name and age.",
                "format": {
                    "type": "json_schema",
                    "json_schema": {
                        "name": "person",
                        "schema": schema,
                    },
                },
                "stream": False,
                "options": {"num_predict": 60},
            },
        )
        assert r.status_code == 200
        data = r.json()
        assert data["done"] is True
        parsed = json.loads(data["response"])
        assert "name" in parsed
        assert "age" in parsed

    def test_generate_raw_mode(self, http_client: httpx.Client):
        r = http_client.post(
            "/api/generate",
            json={
                "model": MODEL_NAME,
                "prompt": "The capital of France is",
                "raw": True,
                "stream": False,
                "options": {"num_predict": 30},
            },
        )
        assert r.status_code == 200
        data = r.json()
        assert data["done"] is True
        assert len(data["response"]) > 0


# ===========================================================================
# 3. Ollama Chat (/api/chat)
# ===========================================================================


class TestOllamaChat:
    def test_chat_non_streaming(self, http_client: httpx.Client):
        """Non-streaming /api/chat returns GenerateResponse (with 'response' field)."""
        r = http_client.post(
            "/api/chat",
            json={
                "model": MODEL_NAME,
                "messages": [{"role": "user", "content": "Say hello in one sentence."}],
                "stream": False,
                "options": {"num_predict": 50},
            },
        )
        assert r.status_code == 200
        data = r.json()
        assert data["done"] is True
        assert len(data["response"]) > 0
        assert data["eval_count"] > 0

    def test_chat_streaming(self, ollama_client: ollama.Client):
        stream = ollama_client.chat(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": "Count to 3."}],
            stream=True,
            options={"num_predict": 50},
        )
        chunks = list(stream)
        assert len(chunks) >= 2
        full_text = "".join(c.message.content for c in chunks)
        assert len(full_text) > 0

    def test_chat_multi_turn(self, http_client: httpx.Client):
        r = http_client.post(
            "/api/chat",
            json={
                "model": MODEL_NAME,
                "messages": [
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": "My name is Alice."},
                    {"role": "assistant", "content": "Nice to meet you, Alice!"},
                    {"role": "user", "content": "What is my name?"},
                ],
                "stream": False,
                "options": {"num_predict": 30},
            },
        )
        assert r.status_code == 200
        data = r.json()
        assert data["done"] is True
        assert len(data["response"]) > 0

    def test_chat_with_options(self, http_client: httpx.Client):
        r = http_client.post(
            "/api/chat",
            json={
                "model": MODEL_NAME,
                "messages": [{"role": "user", "content": "Write a very long story."}],
                "stream": False,
                "options": {"temperature": 0.5, "num_predict": 20},
            },
        )
        assert r.status_code == 200
        data = r.json()
        assert data["done"] is True
        # With num_predict=20 the response should be short
        assert data["eval_count"] <= 25  # small tolerance

    def test_chat_with_json_schema(self, http_client: httpx.Client):
        r = http_client.post(
            "/api/chat",
            json={
                "model": MODEL_NAME,
                "messages": [
                    {"role": "user", "content": "List 2 colors with hex codes."}
                ],
                "format": COLOR_SCHEMA,
                "stream": False,
                "options": {"num_predict": 100},
            },
        )
        assert r.status_code == 200
        data = r.json()
        assert data["done"] is True
        parsed = json.loads(data["response"])
        assert "colors" in parsed
        assert isinstance(parsed["colors"], list)


# ===========================================================================
# 4. Ollama Management Endpoints
# ===========================================================================


class TestOllamaManagement:
    def test_show_model(self, http_client: httpx.Client):
        r = http_client.post("/api/show")
        assert r.status_code == 200
        data = r.json()
        assert "details" in data

    def test_version(self, http_client: httpx.Client):
        r = http_client.get("/api/version")
        assert r.status_code == 200
        assert r.json() == {"version": "0.1.0"}

    def test_tags_contains_mistral(self, http_client: httpx.Client):
        r = http_client.get("/api/tags")
        data = r.json()
        model_names = [m["name"] for m in data["models"]]
        assert any(
            "Ministral-3-3B-Instruct-2512" in name for name in model_names
        ), f"Mistral model not found in {model_names}"

    def test_ps_after_inference(self, http_client: httpx.Client):
        # First make an inference request to load the model
        http_client.post(
            "/api/generate",
            json={
                "model": MODEL_NAME,
                "prompt": "hi",
                "stream": False,
                "options": {"num_predict": 5},
            },
        )
        r = http_client.get("/api/ps")
        data = r.json()
        assert len(data["models"]) >= 1
        loaded_names = [m["model"] for m in data["models"]]
        assert any("Ministral-3-3B-Instruct-2512" in n for n in loaded_names)


# ===========================================================================
# 5. OpenAI Chat Completions (/v1/chat/completions)
# ===========================================================================


class TestOpenAIChatCompletions:
    def test_non_streaming(self, openai_client: openai.OpenAI):
        response = openai_client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": "Say hello in one sentence."}],
            max_tokens=50,
        )
        assert response.id.startswith("chatcmpl-")
        assert response.object == "chat.completion"
        choice = response.choices[0]
        assert choice.message.role == "assistant"
        assert len(choice.message.content) > 0
        assert choice.finish_reason in ("stop", "length")
        assert response.usage.prompt_tokens > 0
        assert response.usage.completion_tokens > 0

    def test_streaming(self, openai_client: openai.OpenAI):
        stream = openai_client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": "Count to 3."}],
            max_tokens=50,
            stream=True,
        )
        chunks = list(stream)
        assert len(chunks) >= 2
        # First chunk should have role
        assert chunks[0].choices[0].delta.role == "assistant"
        # Should have a finish_reason somewhere
        finish_chunks = [c for c in chunks if c.choices[0].finish_reason is not None]
        assert len(finish_chunks) >= 1
        assert finish_chunks[0].choices[0].finish_reason in ("stop", "length")

    def test_streaming_with_usage(self, openai_client: openai.OpenAI):
        stream = openai_client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": "Hi."}],
            max_tokens=20,
            stream=True,
            stream_options={"include_usage": True},
        )
        chunks = list(stream)
        # The final chunk (before [DONE]) should contain usage
        usage_chunks = [c for c in chunks if c.usage is not None]
        assert len(usage_chunks) >= 1
        usage = usage_chunks[-1].usage
        assert usage.prompt_tokens > 0
        assert usage.completion_tokens > 0

    def test_with_system_message(self, openai_client: openai.OpenAI):
        response = openai_client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": "You only respond with the word 'pong'."},
                {"role": "user", "content": "ping"},
            ],
            max_tokens=20,
        )
        assert len(response.choices[0].message.content) > 0

    def test_with_max_tokens(self, openai_client: openai.OpenAI):
        response = openai_client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": "Tell me a very long story."}],
            max_tokens=10,
        )
        choice = response.choices[0]
        # With only 10 tokens the model should hit the limit
        assert choice.finish_reason in ("length", "stop")

    def test_with_stop_sequence(self, openai_client: openai.OpenAI):
        response = openai_client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "user", "content": "Write three sentences about the ocean."}
            ],
            max_tokens=200,
            stop=["."],
        )
        content = response.choices[0].message.content
        # The response should have been cut at the first period
        assert content is not None
        assert content.count(".") <= 1


# ===========================================================================
# 6. Anthropic Messages (/v1/messages)
# ===========================================================================


class TestAnthropicMessages:
    def test_non_streaming(self, http_client: httpx.Client):
        """Non-streaming returns native Anthropic Message format."""
        r = http_client.post(
            "/v1/messages",
            json={
                "model": MODEL_NAME,
                "max_tokens": 50,
                "messages": [
                    {"role": "user", "content": "Say hello in one sentence."}
                ],
            },
        )
        assert r.status_code == 200
        data = r.json()
        assert data["id"].startswith("msg_")
        assert data["type"] == "message"
        assert data["role"] == "assistant"
        # Content is a list of blocks
        assert isinstance(data["content"], list)
        assert len(data["content"]) >= 1
        assert data["content"][0]["type"] == "text"
        assert len(data["content"][0]["text"]) > 0
        assert data["stop_reason"] in ("end_turn", "max_tokens")
        assert data["usage"]["input_tokens"] > 0
        assert data["usage"]["output_tokens"] > 0

    def test_non_streaming_anthropic_sdk(self):
        """Non-streaming via the anthropic SDK — should parse natively."""
        import anthropic

        client = anthropic.Anthropic(
            base_url=BASE_URL,
            api_key="test-key",
        )
        message = client.messages.create(
            model=MODEL_NAME,
            max_tokens=50,
            messages=[
                {"role": "user", "content": "Say hello in one sentence."}
            ],
        )
        assert message.id.startswith("msg_")
        assert message.type == "message"
        assert message.role == "assistant"
        assert len(message.content) >= 1
        assert message.content[0].type == "text"
        assert len(message.content[0].text) > 0
        assert message.stop_reason in ("end_turn", "max_tokens")
        assert message.usage.input_tokens > 0
        assert message.usage.output_tokens > 0

    def test_streaming_raw(self, http_client: httpx.Client):
        """Streaming returns proper Anthropic SSE events."""
        with http_client.stream(
            "POST",
            "/v1/messages",
            json={
                "model": MODEL_NAME,
                "max_tokens": 50,
                "stream": True,
                "messages": [
                    {"role": "user", "content": "Say hello in one sentence."}
                ],
            },
        ) as response:
            assert response.status_code == 200
            events = []
            current_event_type = None
            for line in response.iter_lines():
                if line.startswith("event: "):
                    current_event_type = line[7:]
                elif line.startswith("data: "):
                    data = json.loads(line[6:])
                    events.append({"event": current_event_type, "data": data})

        event_types = [e["event"] for e in events]
        assert event_types[0] == "message_start"
        assert event_types[1] == "content_block_start"
        assert "content_block_delta" in event_types
        assert "content_block_stop" in event_types
        assert "message_delta" in event_types
        assert event_types[-1] == "message_stop"

        # Validate content deltas have text
        deltas = [e for e in events if e["event"] == "content_block_delta"]
        assert len(deltas) >= 1
        for d in deltas:
            assert d["data"]["delta"]["type"] == "text_delta"
            assert "text" in d["data"]["delta"]

        # Validate message_delta has stop_reason
        msg_delta = [e for e in events if e["event"] == "message_delta"][0]
        assert msg_delta["data"]["delta"]["stop_reason"] in ("end_turn", "max_tokens")

    def test_streaming_anthropic_sdk(self):
        """Test streaming via the anthropic SDK."""
        import anthropic

        client = anthropic.Anthropic(
            base_url=BASE_URL,
            api_key="test-key",
        )
        with client.messages.stream(
            model=MODEL_NAME,
            max_tokens=50,
            messages=[
                {"role": "user", "content": "Say hello in one sentence."}
            ],
        ) as stream:
            text = stream.get_final_text()
        assert len(text) > 0

    def test_with_system_prompt(self, http_client: httpx.Client):
        r = http_client.post(
            "/v1/messages",
            json={
                "model": MODEL_NAME,
                "max_tokens": 50,
                "system": "You only respond with the word 'pong'.",
                "messages": [{"role": "user", "content": "ping"}],
            },
        )
        assert r.status_code == 200
        data = r.json()
        assert data["type"] == "message"
        text = data["content"][0]["text"]
        assert len(text) > 0

    def test_with_temperature(self, http_client: httpx.Client):
        r = http_client.post(
            "/v1/messages",
            json={
                "model": MODEL_NAME,
                "max_tokens": 30,
                "temperature": 0.1,
                "messages": [{"role": "user", "content": "Hi"}],
            },
        )
        assert r.status_code == 200
        data = r.json()
        assert data["type"] == "message"


# ===========================================================================
# 7. Tool Calling
# ===========================================================================


class TestToolCalling:
    """Tool calling tests across all three API layers."""

    TOOL_PROMPT = "What's the weather like in San Francisco right now? Use the get_weather tool."

    # -- Ollama --

    def test_ollama_non_streaming(self, http_client: httpx.Client):
        r = http_client.post(
            "/api/chat",
            json={
                "model": MODEL_NAME,
                "messages": [{"role": "user", "content": self.TOOL_PROMPT}],
                "tools": [{"type": "function", "function": WEATHER_TOOL_FUNCTION}],
                "stream": False,
                "options": {"num_predict": 100},
            },
        )
        assert r.status_code == 200
        data = r.json()
        assert data["done"] is True
        # Non-streaming /api/chat returns GenerateResponse with 'response' field
        text = data["response"]
        # Model should produce a tool call in the raw text
        assert (
            "get_weather" in text
            or "[TOOL_CALLS]" in text
        ), f"Expected tool call in response, got: {text[:200]}"

    def test_ollama_streaming(self, http_client: httpx.Client):
        with http_client.stream(
            "POST",
            "/api/chat",
            json={
                "model": MODEL_NAME,
                "messages": [{"role": "user", "content": self.TOOL_PROMPT}],
                "tools": [{"type": "function", "function": WEATHER_TOOL_FUNCTION}],
                "stream": True,
                "options": {"num_predict": 100},
            },
        ) as response:
            chunks = []
            for line in response.iter_lines():
                if line:
                    chunks.append(json.loads(line))

        full_text = "".join(c.get("message", {}).get("content", "") for c in chunks)
        assert (
            "get_weather" in full_text or "[TOOL_CALLS]" in full_text
        ), f"Expected tool call pattern in streaming output, got: {full_text[:200]}"

    # -- OpenAI --

    def test_openai_non_streaming(self, openai_client: openai.OpenAI):
        response = openai_client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": self.TOOL_PROMPT}],
            tools=[{"type": "function", "function": WEATHER_TOOL_FUNCTION}],
            max_tokens=100,
        )
        choice = response.choices[0]
        assert choice.finish_reason == "tool_calls"
        assert choice.message.tool_calls is not None
        assert len(choice.message.tool_calls) >= 1
        tc = choice.message.tool_calls[0]
        assert tc.function.name == "get_weather"
        args = json.loads(tc.function.arguments)
        assert "location" in args

    def test_openai_streaming(self, openai_client: openai.OpenAI):
        stream = openai_client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": self.TOOL_PROMPT}],
            tools=[{"type": "function", "function": WEATHER_TOOL_FUNCTION}],
            max_tokens=100,
            stream=True,
        )
        chunks = list(stream)
        # Should end with tool_calls finish reason
        finish_chunks = [c for c in chunks if c.choices[0].finish_reason is not None]
        assert len(finish_chunks) >= 1
        assert finish_chunks[0].choices[0].finish_reason == "tool_calls"
        # At least one chunk should contain tool_calls
        tool_chunks = [
            c for c in chunks if c.choices[0].delta.tool_calls is not None
        ]
        assert len(tool_chunks) >= 1
        tc = tool_chunks[0].choices[0].delta.tool_calls[0]
        assert tc.function.name == "get_weather"

    # -- Anthropic --

    def test_anthropic_non_streaming(self, http_client: httpx.Client):
        """Anthropic non-streaming should return tool_use content blocks."""
        r = http_client.post(
            "/v1/messages",
            json={
                "model": MODEL_NAME,
                "max_tokens": 100,
                "messages": [{"role": "user", "content": self.TOOL_PROMPT}],
                "tools": [
                    {
                        "name": "get_weather",
                        "description": "Get the current weather for a location",
                        "input_schema": {
                            "type": "object",
                            "properties": {
                                "location": {
                                    "type": "string",
                                    "description": "City name",
                                },
                                "unit": {
                                    "type": "string",
                                    "enum": ["celsius", "fahrenheit"],
                                },
                            },
                            "required": ["location"],
                        },
                    }
                ],
            },
        )
        assert r.status_code == 200
        data = r.json()
        assert data["type"] == "message"
        assert data["stop_reason"] == "tool_use"
        # Find tool_use block in content
        tool_blocks = [b for b in data["content"] if b["type"] == "tool_use"]
        assert len(tool_blocks) >= 1, f"Expected tool_use block, got: {data['content']}"
        tc = tool_blocks[0]
        assert tc["name"] == "get_weather"
        assert "location" in tc["input"]

    def test_anthropic_streaming(self, http_client: httpx.Client):
        """Anthropic streaming should emit tool_use content blocks via SSE."""
        with http_client.stream(
            "POST",
            "/v1/messages",
            json={
                "model": MODEL_NAME,
                "max_tokens": 100,
                "stream": True,
                "messages": [{"role": "user", "content": self.TOOL_PROMPT}],
                "tools": [
                    {
                        "name": "get_weather",
                        "description": "Get the current weather for a location",
                        "input_schema": {
                            "type": "object",
                            "properties": {
                                "location": {"type": "string"},
                            },
                            "required": ["location"],
                        },
                    }
                ],
            },
        ) as response:
            events = []
            current_event_type = None
            for line in response.iter_lines():
                if line.startswith("event: "):
                    current_event_type = line[7:]
                elif line.startswith("data: "):
                    data = json.loads(line[6:])
                    events.append({"event": current_event_type, "data": data})

        # Should have tool_use content_block_start events
        tool_starts = [
            e for e in events
            if e["event"] == "content_block_start"
            and e["data"].get("content_block", {}).get("type") == "tool_use"
        ]
        assert len(tool_starts) >= 1, f"Expected tool_use block start, got events: {[e['event'] for e in events]}"
        tc = tool_starts[0]["data"]["content_block"]
        assert tc["name"] == "get_weather"

        # Should have input_json_delta events
        json_deltas = [
            e for e in events
            if e["event"] == "content_block_delta"
            and e["data"].get("delta", {}).get("type") == "input_json_delta"
        ]
        assert len(json_deltas) >= 1
        args = json.loads(json_deltas[0]["data"]["delta"]["partial_json"])
        assert "location" in args

        # message_delta should have stop_reason=tool_use
        msg_delta = [e for e in events if e["event"] == "message_delta"][0]
        assert msg_delta["data"]["delta"]["stop_reason"] == "tool_use"


# ===========================================================================
# 8. Structured Output (JSON Schema)
# ===========================================================================


class TestStructuredOutput:
    PROMPT = "List exactly 3 colors with their hex codes."

    def test_ollama_generate(self, http_client: httpx.Client):
        """GenerateRequest.format expects ResponseFormat type."""
        r = http_client.post(
            "/api/generate",
            json={
                "model": MODEL_NAME,
                "prompt": self.PROMPT,
                "format": {
                    "type": "json_schema",
                    "json_schema": {
                        "name": "colors_output",
                        "schema": COLOR_SCHEMA,
                    },
                },
                "stream": False,
                "options": {"num_predict": 150},
            },
        )
        assert r.status_code == 200
        data = r.json()
        parsed = json.loads(data["response"])
        assert "colors" in parsed
        assert isinstance(parsed["colors"], list)
        assert len(parsed["colors"]) >= 1
        for color in parsed["colors"]:
            assert "name" in color
            assert "hex" in color

    def test_ollama_chat(self, http_client: httpx.Client):
        r = http_client.post(
            "/api/chat",
            json={
                "model": MODEL_NAME,
                "messages": [{"role": "user", "content": self.PROMPT}],
                "format": COLOR_SCHEMA,
                "stream": False,
                "options": {"num_predict": 150},
            },
        )
        assert r.status_code == 200
        data = r.json()
        # Non-streaming /api/chat returns GenerateResponse with 'response' field
        parsed = json.loads(data["response"])
        assert "colors" in parsed
        assert isinstance(parsed["colors"], list)

    def test_openai(self, openai_client: openai.OpenAI):
        response = openai_client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": self.PROMPT}],
            max_tokens=150,
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": "colors_output",
                    "schema": COLOR_SCHEMA,
                },
            },
        )
        content = response.choices[0].message.content
        parsed = json.loads(content)
        assert "colors" in parsed
        assert isinstance(parsed["colors"], list)

    def test_anthropic(self, http_client: httpx.Client):
        r = http_client.post(
            "/v1/messages",
            json={
                "model": MODEL_NAME,
                "max_tokens": 150,
                "messages": [{"role": "user", "content": self.PROMPT}],
                "json_schema": json.dumps(COLOR_SCHEMA),
            },
        )
        assert r.status_code == 200
        data = r.json()
        assert data["type"] == "message"
        text = data["content"][0]["text"]
        parsed = json.loads(text)
        assert "colors" in parsed
        assert isinstance(parsed["colors"], list)


# ===========================================================================
# 9. Parameter Settings
# ===========================================================================


class TestParameterSettings:
    def test_temperature_zero_deterministic(self, openai_client: openai.OpenAI):
        """Two requests with temperature=0 and same seed should produce same output."""
        kwargs = dict(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": "What is 2+2?"}],
            max_tokens=20,
            temperature=0,
            seed=42,
        )
        r1 = openai_client.chat.completions.create(**kwargs)
        r2 = openai_client.chat.completions.create(**kwargs)
        assert r1.choices[0].message.content == r2.choices[0].message.content

    def test_max_tokens_limit(self, openai_client: openai.OpenAI):
        response = openai_client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "user", "content": "Write a 500 word essay about space."}
            ],
            max_tokens=10,
        )
        assert response.choices[0].finish_reason in ("length", "stop")

    def test_stop_sequence(self, http_client: httpx.Client):
        r = http_client.post(
            "/api/generate",
            json={
                "model": MODEL_NAME,
                "prompt": "List the first 5 US states separated by commas.",
                "stream": False,
                "options": {"stop": [","], "num_predict": 100},
            },
        )
        assert r.status_code == 200
        data = r.json()
        text = data["response"]
        # Should stop at or before the first comma
        assert text.count(",") <= 1

    def test_top_p_setting(self, openai_client: openai.OpenAI):
        """Verify top_p parameter is accepted."""
        response = openai_client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": "Hi"}],
            max_tokens=10,
            top_p=0.1,
        )
        assert response.choices[0].message.content is not None

    def test_seed_reproducibility(self, http_client: httpx.Client):
        """Two Ollama chat requests with same seed + temperature=0 produce same text."""
        payload = {
            "model": MODEL_NAME,
            "messages": [{"role": "user", "content": "What is 2+2?"}],
            "stream": False,
            "options": {"seed": 123, "temperature": 0, "num_predict": 20},
        }
        r1 = http_client.post("/api/chat", json=payload)
        r2 = http_client.post("/api/chat", json=payload)
        # Non-streaming /api/chat returns GenerateResponse with 'response' field
        assert r1.json()["response"] == r2.json()["response"]


# ===========================================================================
# Entry point
# ===========================================================================

if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v", "--tb=short"]))
