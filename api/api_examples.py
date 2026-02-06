"""
API Examples for FastAPI endpoints.

This module contains example request bodies and query parameter examples
used in the OpenAPI documentation for the MLX Engine API.
"""

# =============================================================================
# Generate Endpoint Examples
# =============================================================================

GENERATE_EXAMPLES = [
    {
        "summary": "Simple text generation",
        "description": "Basic prompt with default settings",
        "value": {
            "model": "mlx-community/Llama-3.2-3B-Instruct-4bit",
            "prompt": "Why is the sky blue?",
            "stream": False
        }
    },
    {
        "summary": "Generation with custom options",
        "description": "Prompt with temperature and max tokens configured",
        "value": {
            "model": "mlx-community/Llama-3.2-3B-Instruct-4bit",
            "prompt": "Write a haiku about programming",
            "system": "You are a creative poet.",
            "stream": True,
            "options": {
                "temperature": 0.8,
                "num_predict": 100,
                "top_p": 0.9
            }
        }
    },
    {
        "summary": "JSON structured output",
        "description": "Generate structured JSON response",
        "value": {
            "model": "mlx-community/Llama-3.2-3B-Instruct-4bit",
            "prompt": "List 3 programming languages with their main use cases",
            "stream": False,
            "format": {
                "type": "object",
                "properties": {
                    "languages": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "name": {"type": "string"},
                                "use_case": {"type": "string"}
                            }
                        }
                    }
                }
            }
        }
    }
]

# =============================================================================
# Chat Endpoint Examples
# =============================================================================

CHAT_EXAMPLES = [
    {
        "summary": "Simple chat message",
        "description": "Basic chat with a single user message",
        "value": {
            "model": "mlx-community/Llama-3.2-3B-Instruct-4bit",
            "messages": [
                {"role": "user", "content": "Hello, how are you?"}
            ],
            "stream": False
        }
    },
    {
        "summary": "Multi-turn conversation",
        "description": "Chat with conversation history",
        "value": {
            "model": "mlx-community/Llama-3.2-3B-Instruct-4bit",
            "messages": [
                {"role": "system", "content": "You are a helpful coding assistant."},
                {"role": "user", "content": "What is Python?"},
                {"role": "assistant", "content": "Python is a high-level programming language."},
                {"role": "user", "content": "Show me a simple example."}
            ],
            "stream": True,
            "options": {
                "temperature": 0.7
            }
        }
    },
    {
        "summary": "Chat with tool calling",
        "description": "Chat with function/tool definitions",
        "value": {
            "model": "mlx-community/Llama-3.2-3B-Instruct-4bit",
            "messages": [
                {"role": "user", "content": "What's the weather in San Francisco?"}
            ],
            "tools": [
                {
                    "type": "function",
                    "function": {
                        "name": "get_weather",
                        "description": "Get the current weather for a location",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "location": {"type": "string", "description": "City name"}
                            },
                            "required": ["location"]
                        }
                    }
                }
            ],
            "stream": False
        }
    }
]

# =============================================================================
# Anthropic Messages Endpoint Examples
# =============================================================================

MESSAGES_EXAMPLES = [
    {
        "summary": "Simple message",
        "description": "Basic message request with a single user message",
        "value": {
            "model": "mlx-community/Llama-3.2-3B-Instruct-4bit",
            "max_tokens": 1024,
            "messages": [
                {"role": "user", "content": "What is the capital of France?"}
            ]
        }
    },
    {
        "summary": "Message with system prompt",
        "description": "Message with system context and streaming enabled",
        "value": {
            "model": "mlx-community/Llama-3.2-3B-Instruct-4bit",
            "max_tokens": 2048,
            "system": "You are a helpful assistant that provides concise answers.",
            "messages": [
                {"role": "user", "content": "Explain quantum computing in simple terms."}
            ],
            "stream": True,
            "temperature": 0.7
        }
    },
    {
        "summary": "Multi-turn conversation",
        "description": "Conversation with multiple turns",
        "value": {
            "model": "mlx-community/Llama-3.2-3B-Instruct-4bit",
            "max_tokens": 1024,
            "messages": [
                {"role": "user", "content": "What is Python?"},
                {"role": "assistant", "content": "Python is a high-level programming language known for its readability."},
                {"role": "user", "content": "What are its main use cases?"}
            ],
            "temperature": 0.5
        }
    }
]

# =============================================================================
# OpenAI Chat Completions Endpoint Examples
# =============================================================================

CHAT_COMPLETIONS_EXAMPLES = [
    {
        "summary": "Simple chat completion",
        "description": "Basic chat completion request",
        "value": {
            "model": "mlx-community/Llama-3.2-3B-Instruct-4bit",
            "messages": [
                {"role": "user", "content": "Hello, how are you?"}
            ]
        }
    },
    {
        "summary": "Chat with system message",
        "description": "Chat completion with system prompt and streaming",
        "value": {
            "model": "mlx-community/Llama-3.2-3B-Instruct-4bit",
            "messages": [
                {"role": "system", "content": "You are a helpful coding assistant."},
                {"role": "user", "content": "Write a Python function to calculate factorial."}
            ],
            "stream": True,
            "temperature": 0.7,
            "max_tokens": 500
        }
    },
    {
        "summary": "Chat with tools",
        "description": "Chat completion with function calling",
        "value": {
            "model": "mlx-community/Llama-3.2-3B-Instruct-4bit",
            "messages": [
                {"role": "user", "content": "What's the weather like in Tokyo?"}
            ],
            "tools": [
                {
                    "type": "function",
                    "function": {
                        "name": "get_weather",
                        "description": "Get current weather for a location",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "location": {"type": "string"},
                                "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]}
                            },
                            "required": ["location"]
                        }
                    }
                }
            ],
            "temperature": 0.5
        }
    }
]

# =============================================================================
# Query Parameter Examples
# =============================================================================

CREATE_MODEL_REPO_ID_EXAMPLES = [
    "meta-llama/Llama-3.2-3B-Instruct",
    "google/gemma-2-2b-it",
    "Qwen/Qwen2.5-3B-Instruct"
]

CREATE_MODEL_OUTPUT_DIR_EXAMPLES = [
    "./models/meta-llama/Llama-3.2-3B-Instruct",
    "./models/custom/my-model"
]

PULL_MODEL_REPO_ID_EXAMPLES = [
    "mlx-community/Llama-3.2-3B-Instruct-4bit",
    "mlx-community/Qwen2.5-3B-Instruct-4bit"
]
