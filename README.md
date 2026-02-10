# MLX Engine

A unified local inference server for running LLMs, VLMs, embedding, audio, and realtime models on Apple Silicon through a single API. Built with FastAPI and optimized using Apple's [MLX framework](https://github.com/ml-explore/mlx), this server provides **OpenAI-compatible**, **Ollama-compatible**, and **Anthropic-compatible** APIs for seamless integration with existing AI applications and workflows.

## Overview

MLX Engine unifies the MLX ecosystem into a single server, enabling you to run language, vision, embedding, audio, and eventually realtime models locally on your Mac with Metal GPU acceleration. The server handles model loading, caching, and inference optimization automatically, allowing you to focus on building applications without worrying about the underlying infrastructure.

Whether you're developing with the OpenAI SDK, Ollama SDK, or Anthropic SDK, MLX Engine provides drop-in compatibility so you can switch from cloud APIs to local inference without changing your code.

## Features

### 🚀 Performance & Optimization
- **Smart Model Caching** - Models stay loaded in memory across requests for instant response times
- **Automatic KV Caching** - Intelligent prompt caching that reuses computations from previous requests
- **KV Cache Quantization** - Reduce memory usage with 3-8 bit quantization support
- **Speculative Decoding** - Accelerate generation using a smaller draft model for faster token generation
- **Apple Silicon Native** - Leverages Metal GPU acceleration via MLX for optimal performance on Mac

### 🔌 API & Compatibility
- **Anthropic-Compatible API** *(Work in Progress)* - Support for Anthropic's Messages API format
- **OpenAI-Compatible API** - Drop-in replacement for OpenAI's Chat Completions API
- **Ollama-Compatible API** - Native API endpoints for chat, generation, and model management
- **Multi-turn Conversations** - Full support for system/user/assistant message roles across both APIs
- **Tool & Function Calling** *(Work in Progress)* - Define and execute custom tools within conversations
- **Streaming Support** - Real-time Server-Sent Events (SSE) streaming for responsive user experiences
- **Token Probabilities** - Request top-k logprobs for advanced use cases (Anthropic API)
- **JSON Mode** - Structured output with schema validation (OpenAI API)

### 🤖 Model Support
- **Text Models** - Mistral, Qwen 2.5/3, Llama 3/3.1/3.2, Gemma 2, Phi, DeepSeek, and other transformer-based LLMs
- **Vision Models** - Gemma 3, Pixtral, Mistral 3, Qwen2-VL, Qwen3-VL, LFM-2 with image understanding capabilities
- **Audio Models** *(Work in Progress)* - Kokoro-82M and other audio-language models for text-to-speech
- **Embeddings** - Text embedding model support via mlx-embeddings
- **Model Formats** - Native MLX format support with automatic conversion from Hugging Face models
- **Easy Model Management** - Download models directly from Hugging Face with a simple API call

### ⚙️ Advanced Features
- **Flexible Sampling Controls** - Fine-tune generation with temperature, top-p, top-k, and custom stop sequences
- **Context Window Management** - Configurable KV cache sizes for handling long contexts
- **Draft Model Support** - Load secondary models for speculative decoding
- **Vision Processing** - Handle text and images together with automatic preprocessing
- **Content Block System** - Rich message content with text, images, and documents

## Architecture Highlights

**Model Caching System** - The server keeps models loaded in memory and automatically reuses them when parameters match, eliminating the 5-10 second model loading overhead on subsequent requests.

**Automatic KV Caching** - For text-only models, the server detects common prompt prefixes between requests and reuses the already-computed key-value pairs, dramatically reducing prompt processing time for similar requests.

**Memory Management** - Intelligent cleanup using garbage collection and Metal cache clearing ensures efficient memory usage when switching between models.

## Requirements

- **macOS** with Apple Silicon (M1, M2, M3, M4, or later)
- **Python 3.13** or later
- **8GB+ RAM** recommended (16GB+ for larger models)
- **Xcode Command Line Tools** (for MLX compilation)

## Installation

**Note:** First-time installation may take a few minutes as MLX compiles optimized kernels for your specific hardware.

## Quick Start

### Starting the Server

### API Endpoints

The server provides three API interfaces:

#### 1. Anthropic-Compatible API
**Endpoint:** `POST /v1/messages`

Use with the official Anthropic SDK or any client expecting Anthropic's API.

#### 2. OpenAI-Compatible API
**Endpoint:** `POST /v1/chat/completions`

Use with the OpenAI SDK or any OpenAI-compatible client.

#### 3. Native API
**Endpoints:**
- `POST /api/generate` - Single prompt generation
- `POST /api/chat` - Multi-turn conversations
- `POST /api/ps` - List running models
- `POST /api/pull` - Download models from Hugging Face
- `GET /api/tags` - List available local models

### Streaming Responses

All APIs support streaming for real-time token generation.

### Vision Models

For vision-language models, include images in your requests.

## Model Management

### Downloading Models

Use the `/api/pull` endpoint to download models from Hugging Face. Models are automatically downloaded to `./models/` directory and converted to MLX format if needed.

### Listing Available Models

Use the `/api/tags` endpoint to get a list of available local models.

### Model Configuration

When making requests, you can configure various model parameters including temperature, top_p, max_tokens, KV cache settings, quantization options, speculative decoding, and more.

## Performance Tips

1. **Model Selection**: Smaller models (3B-7B parameters) run fastest on consumer Apple Silicon
2. **KV Cache Quantization**: Use `kv_bits=4` to reduce memory usage for long contexts
3. **Speculative Decoding**: Load a draft model for 1.5-2x speedup on supported models
4. **Batch Similar Requests**: The KV cache automatically optimizes repeated prompt prefixes
5. **Vision Models**: Pre-resize images to reasonable dimensions (e.g., 1024x1024) for faster processing

## Troubleshooting

### Model Loading Issues

If models fail to load, ensure they're in MLX-compatible format or use the MLX conversion tools to convert Hugging Face models.

### Memory Errors

For large models on devices with limited RAM:
- Use quantized models (4-bit recommended)
- Reduce `max_kv_size` parameter
- Enable KV cache quantization with `kv_bits=4`

### Port Conflicts

If port 8000 is already in use, specify a custom port when starting the server with uvicorn.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Acknowledgements

This project is built upon the excellent open-source work from **[LM Studio](https://lmstudio.ai/)**. Special thanks to the LM Studio team for their contributions to making local LLM inference accessible and efficient on Apple Silicon.

Additional thanks to:
- **Apple ML Explore Team** for the [MLX framework](https://github.com/ml-explore/mlx)
- **Hugging Face** for model hosting and the transformers library
- The broader open-source AI community

## License

MIT License - Copyright (c) 2024 LM Studio
