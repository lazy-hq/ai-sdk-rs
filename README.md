# AISDK

[![Build Status](https://github.com/lazy-hq/aisdk/actions/workflows/ci.yml/badge.svg)](https://github.com/lazy-hq/aisdk/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Issues](https://img.shields.io/github/issues/lazy-hq/aisdk)](https://github.com/lazy-hq/aisdk/issues)
[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg)](https://github.com/lazy-hq/aisdk/pulls)

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
use aisdk::{
    core::{LanguageModelRequest},
    providers::openai::OpenAI,
};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {

    // with default openai provider settings
    let openai = OpenAI::new("gpt-5");

    let result = LanguageModelRequest::builder()
        .model(openai)
        .prompt("hello world")
        .build()
        .generate_text()
        .await;

    println!("{}", result.text);
    Ok(())
}
```

### Streaming Text Generation

```rust
use aisdk::{
    core::{LanguageModelRequest},
    providers::openai::OpenAI,
};
use futures::StreamExt;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {

    // with custom openai provider settings
    let openai = OpenAI::builder()
        .api_key("your-api-key")
        .model_name("gpt-4o")
        .build()?;

    let mut stream = LanguageModelRequest::builder()
        .model(openai)
        .prompt("Count from 1 to 10.")
        .build()
        .stream_text()
        .await?;

    while let Some(chunk) = stream.stream.next().await {
        print!("{}", chunk.text);
    }
    Ok(())
}
```

### Providers

#### Supported Options

| Model/Input | Max Tokens  | Temprature  | Top P   | Top K   | Stop    | Seed    | 
| ----------- | ----------- | ----------- | ------- | ------- | ------- | ------- |
| OpenAi      | ✅          | ✅          | ✅      | NA      | ✅      | NA[^1]  |

[^1]: Seed is deprecated on the newer response api so it is not supported in open ai.

### Tools

You can define a tool using the use `aisdk::core::tool`;
```rust
#[tool]
/// Returns the username
fn get_username(id: String) {
    // Your code here
}
```
A tool has a name, a description, an input and a body. all three can be infered from standard rust function. The name is the function name, `get_username` in the above example. The description is infered from the doc comments of the fucntion, `/// Returns the username` is going to be used to describe the tool. make sure to use a verbose, language model friendly description in the comments. The input is built from the function arguments and converted to a json schema using [schemars](https://docs.rs/schemars/latest/schemars/index.html) so make sure any type you add to the function arguments derive [JsonSchema](https://docs.rs/schemars/latest/schemars/trait.JsonSchema.html). Any think you implement in the function body will be executed on the language model's request and is thread safe.

The first two components can be overridden by using the macro arguments `#[tool(name, description)]` attribute.

```rust
    #[tool(
        name = "the-name-for-this-tool",
        desc = "the-description-for-this-tool"
    )]
    fn get_username(id: String) {
        // Your code here
    }

```


### Prompts
The file in `./prompts` contains various example prompt files to demonstrate the capabilities of the `aisdk` prompt templating system, powered by the `tera` engine. These examples showcase different features like variable substitution, conditionals, loops, and template inclusion, simulating common AI prompt constructions.

## Technologies Used

- **Rust**: Core language.
- **Tokio**: Async runtime.
- **Tera**: Template engine for prompts.
- **async-openai**: OpenAI API client.

## Contributing

We welcome contributions! Please see [CONTRIBUTING.md](./CONTRIBUTING.md) for guidelines.

## License

Licensed under the MIT License. See [LICENSE](./LICENSE) for details.
