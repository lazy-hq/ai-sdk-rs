# AISDK

[![Build Status](https://github.com/lazy-hq/ai-sdk-rs/actions/workflows/ci.yml/badge.svg)](https://github.com/lazy-hq/ai-sdk-rs/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Issues](https://img.shields.io/github/issues/lazy-hq/ai-sdk-rs)](https://github.com/lazy-hq/ai-sdk-rs/issues)
[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg)](https://github.com/lazy-hq/ai-sdk-rs/pulls)

An open-source Rust library for building AI-powered applications, inspired by the Vercel AI SDK. It provides a type-safe interface for interacting with Large Language Models (LLMs).

> **⚠️ Early Stage Warning**: This project is in very early development and not ready for production use. APIs may change significantly, and features are limited. Use at your own risk.

## Key Features

- **OpenAI Provider Support**: Initial support for OpenAI models with text generation and streaming.
- **Models.dev Integration**: Dynamic provider discovery and model selection through models.dev API.
- **Type-Safe API**: Built with Rust's type system for reliability.
- **Asynchronous**: Uses Tokio for async operations.
- **Prompt Templating**: Filesystem-based prompts using Tera templates (coming soon).

## Installation

Add `aisdk` to your `Cargo.toml`:

```toml
[dependencies]
aisdk = "0.1.0"
```

Enable the OpenAI feature:

```toml
aisdk = { version = "0.1.0", features = ["openai"] }
```

Enable the models.dev feature for dynamic provider discovery:

```toml
aisdk = { version = "0.1.0", features = ["models-dev"] }
```

Enable all features:

```toml
aisdk = { version = "0.1.0", features = ["full"] }
```

## Usage

### Basic Text Generation

```rust
use aisdk::{
    core::{GenerateTextCallOptions, generate_text},
    providers::openai::{OpenAI, OpenAIProviderSettings},
};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {

    // with default openai provider settings
    let openai = OpenAI::new("gpt-5");

    let options = GenerateTextCallOptions::builder()
        .prompt("Say hello.")
        .build()?;

    let result = generate_text(openai, options).await?;
    println!("{}", result.text);
    Ok(())
}
```

### Streaming Text Generation

```rust
use aisdk::{
    core::{GenerateTextCallOptions, generate_stream},
    providers::openai::{OpenAI, OpenAIProviderSettings},
};
use futures::StreamExt;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {

    // with custom openai provider settings
    let openai = OpenAI::builder()
        .api_key("your-api-key")
        .model_name("gpt-4o")
        .build()?;

    let options = GenerateTextCallOptions::builder()
        .prompt("Count from 1 to 10.")
        .build()?;

    let mut stream = generate_stream(openai, options).await?;
    while let Some(chunk) = stream.stream.next().await {
        print!("{}", chunk.text);
    }
    Ok(())
}
```



### Providers

- **Yes**: ✅
- **NA**: Not Applicable

| Model/Input     | Max Tokens      | Temprature      | Top P           | Top K           | Stop            |
| --------------- | --------------- | --------------- | --------------- | --------------- | --------------- |
| OpenAi          | ✅              | ✅              | ✅              | NA              | ✅              |


### Prompts
The file in `./prompts` contains various example prompt files to demonstrate the capabilities of the `aisdk` prompt templating system, powered by the `tera` engine. These examples showcase different features like variable substitution, conditionals, loops, and template inclusion, simulating common AI prompt constructions.

## Technologies Used

- **Rust**: Core language.
- **Tokio**: Async runtime.
- **Tera**: Template engine for prompts.
- **async-openai**: OpenAI API client.
- **reqwest**: HTTP client for models.dev API.
- **serde**: JSON serialization and deserialization.

## Models.dev Feature

The `models-dev` feature provides comprehensive integration with the models.dev API for dynamic provider discovery and model selection.

### Key Capabilities

- **Dynamic Provider Discovery**: Automatically discover AI model providers through the models.dev API
- **Intelligent Model Selection**: Find the best models based on capabilities, cost, and use cases
- **Comprehensive Caching**: Memory and disk caching to minimize API calls and improve performance
- **Configuration Validation**: Automatically validate provider configurations and environment variables
- **Provider Integration**: Seamless integration with existing AISDK providers

### Quick Start

```rust
use aisdk::models_dev::{ProviderRegistry, find_best_model_for_use_case};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Create a registry with default client
    let registry = ProviderRegistry::with_default_client();
    
    // Fetch the latest provider and model data
    registry.refresh().await?;
    
    // Find the best model for your use case
    let chat_model = find_best_model_for_use_case(&registry, "chat").await;
    let code_model = find_best_model_for_use_case(&registry, "code").await;
    
    println!("Best chat model: {:?}", chat_model);
    println!("Best code model: {:?}", code_model);
    
    Ok(())
}
```

### Examples

The library includes several examples demonstrating the models.dev feature:

```bash
# Basic client usage with caching
cargo run --example models_dev_client_example --features models-dev

# Registry operations and provider discovery
cargo run --example models_dev_registry_example --features models-dev

# Convenience functions and integration patterns
cargo run --example models_dev_convenience_example --features models-dev

# Complete end-to-end integration example
cargo run --example models_dev_complete_integration --features models-dev
```

### Documentation

For detailed documentation, see [MODELS_DEV_FEATURE.md](docs/MODELS_DEV_FEATURE.md).

## Contributing

We welcome contributions! Please see [CONTRIBUTING.md](./CONTRIBUTING.md) for guidelines.

## License

Licensed under the MIT License. See [LICENSE](./LICENSE) for details.
