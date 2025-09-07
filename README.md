# AISDK

[![Build Status](https://github.com/lazy-hq/ai-sdk-rs/actions/workflows/ci.yml/badge.svg)](https://github.com/lazy-hq/ai-sdk-rs/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Issues](https://img.shields.io/github/issues/lazy-hq/ai-sdk-rs)](https://github.com/lazy-hq/ai-sdk-rs/issues)
[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg)](https://github.com/lazy-hq/ai-sdk-rs/pulls)

An open-source Rust library for building AI-powered applications, inspired by the Vercel AI SDK. It provides a type-safe interface for interacting with Large Language Models (LLMs).

> **⚠️ Early Stage Warning**: This project is in very early development and not ready for production use. APIs may change significantly, and features are limited. Use at your own risk.

## Key Features

- **OpenAI Provider Support**: Initial support for OpenAI models with text generation and streaming.
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

## Usage

### Basic Text Generation

```rust
use ai_sdk_rs::{
    core::{GenerateTextCallOptions, generate_text},
    providers::openai::{OpenAI, OpenAIProviderSettings},
};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let settings = OpenAIProviderSettings::builder()
        .api_key("your-api-key".to_string())
        .model_name("gpt-4o".to_string())
        .build()?;

    let openai = OpenAI::new(settings);

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
use ai_sdk_rs::{
    core::{GenerateTextCallOptions, generate_stream},
    providers::openai::{OpenAI, OpenAIProviderSettings},
};
use futures::StreamExt;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let settings = OpenAIProviderSettings::builder()
        .api_key("your-api-key".to_string())
        .model_name("gpt-4o".to_string())
        .build()?;

    let openai = OpenAI::new(settings);

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

| Model/Input     | Max Tokens      | Temprature      | Top P           | Top K           | Stop            |
| --------------- | --------------- | --------------- | --------------- | --------------- | --------------- |
| OpenAi          | ✅              | ✅              | ✅              | NA              | ✅              |

- **Yes**: ✅ 
- **NA**: Not Applicable

### Prompts
The directory `./prompts` contains various example prompt files to demonstrate the capabilities of the `ai-sdk-rs` prompt templating system, powered by the `tera` engine. These examples showcase different features like variable substitution, conditionals, loops, and template inclusion, simulating common AI prompt constructions.

## Technologies Used

- **Rust**: Core language.
- **Tokio**: Async runtime.
- **Tera**: Template engine for prompts.
- **async-openai**: OpenAI API client.

## Contributing

We welcome contributions! Please see [CONTRIBUTING.md](./CONTRIBUTING.md) for guidelines.

## License

Licensed under the MIT License. See [LICENSE](./LICENSE) for details.
