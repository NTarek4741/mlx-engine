# MLX Engine

A high-performance local inference server for running Large Language Models (LLMs) and Vision Language Models (VLMs) on Apple Silicon. Built with FastAPI and optimized using Apple's [MLX framework](https://github.com/ml-explore/mlx), this server provides an Anthropic-compatible API for seamless integration with existing AI applications and workflows.

## Overview

MLX Engine enables you to run powerful language and vision models locally on your Mac with Metal GPU acceleration. The server handles model loading, caching, and inference optimization automatically, allowing you to focus on building applications without worrying about the underlying infrastructure.

## Features

### 🚀 Performance & Optimization
- **Smart Model Caching** - Models stay loaded in memory across requests for instant response times
- **Automatic KV Caching** - Intelligent prompt caching that reuses computations from previous requests, reducing processing time by up to 30x for repeated prompts
- **KV Cache Quantization** - Reduce memory usage with 3-8 bit quantization support
- **Speculative Decoding** - Accelerate generation using a smaller draft model for faster token generation
- **Apple Silicon Native** - Leverages Metal GPU acceleration via MLX for optimal performance on Mac

### 🔌 API & Compatibility
- **Anthropic-Compatible API** - Follows Anthropic's Messages API design for structured conversations
- **Multi-turn Conversations** - Full support for system/user/assistant message roles
- **Tool & Function Calling** *(Work in Progress)* - Define and execute custom tools within conversations
- **Streaming Support** - Real-time token streaming for responsive user experiences
- **Token Probabilities** - Request top-k logprobs for advanced use cases

### 🤖 Model Support
- **Text Models** - Mistral, Qwen, Llama, Gemma, and other transformer-based LLMs
- **Vision Models** - Support for vision-language models like Gemma 3, Pixtral, Mistral 3 with image understanding capabilities
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

## Acknowledgements

This project is built upon the excellent open-source work from **[LM Studio](https://lmstudio.ai/)**. Special thanks to the LM Studio team for their contributions to making local LLM inference accessible and efficient on Apple Silicon.

Additional thanks to:
- **Apple ML Explore Team** for the [MLX framework](https://github.com/ml-explore/mlx)
- **Hugging Face** for model hosting and the transformers library
- The broader open-source AI community

## License

MIT License - Copyright (c) 2024 LM Studio
