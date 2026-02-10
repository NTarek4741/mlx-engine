# MLX Engine

> **This project is in early stages of development. APIs and features may change.**

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
- **Embeddings** *(Work in Progress)* - Text embedding model support via mlx-embeddings
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
