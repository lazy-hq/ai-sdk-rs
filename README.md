# AI-SDK-RS

[![Build Status](https://github.com/lazyai/ai-sdk-rs/actions/workflows/ci.yml/badge.svg)](https://github.com/lazyai/ai-sdk-rs/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Issues](https://img.shields.io/github/issues/lazyai/ai-sdk-rs)](https://github.com/lazyai/ai-sdk-rs/issues)
[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg)](https://github.com/lazyai/ai-sdk-rs/pulls)

An open-source Rust library for building AI-powered applications, inspired by the Vercel AI SDK. It provides a robust, type-safe, and easy-to-use interface for interacting with various Large Language Models (LLMs).

## Key Features

- **Multi-Model Support**: Seamlessly switch between different LLM providers (e.g., OpenAI, Anthropic, Cohere) with a unified API.
- **Filesystem-based Prompt Templating**: Organize and manage your prompts efficiently using simple text files.
- **Streaming Support**: Built-in support for streaming responses for real-time applications.
- **Type-Safe**: Leverage Rust's type system to ensure correctness and reliability.
- **Ergonomic API**: Designed to be intuitive and easy to integrate into any Rust project.

## Technologies Used

- **Rust**: Built entirely in Rust for performance, safety, and concurrency.
- **Tokio**: For asynchronous operations.
- **tera**: For Jinja style template rendering.

## Installation

Add `ai-sdk-rs` to your `Cargo.toml` dependencies:

```sh
cargo add ai-sdk-rs
```

## Usage

Here is a basic example of how to use `ai-sdk-rs` to interact with an LLM:

```rust
println!("Comming Soon ...")
```

### Providers

| Model/Input     | Max Tokens      | Temprature      | Top P           | Top K           | Stop            |
| --------------- | --------------- | --------------- | --------------- | --------------- | --------------- |
| OpenAi          | ✅              | ✅              | ✅              | NA              | ✅              |

- **Yes**: ✅ 
- **NA**: Not Applicable

### Prompts
The directory `./prompts` contains various example prompt files to demonstrate the capabilities of the `ai-sdk-rs` prompt templating system, powered by the `tera` engine. These examples showcase different features like variable substitution, conditionals, loops, and template inclusion, simulating common AI prompt constructions.

**Example:**

```rust
use ai_sdk_rs::prompt::prompt::{Prompt, Promptable};

// To use prompts/system/persona/helpful_assistant.md
let prompt = Prompt::new("system/persona/helpful_assistant")
    .with_extension("md")
    .with("role", "provide technical support");
let generated_text = prompt.generate();
println!("{{}}", generated_text);

// To use prompts/user/summarize.md
let prompt = Prompt::new("user/summarize")
    .with_extension("md")
    .with("text", "This is a long text that needs summarization.")
    .with("points", "[\"Point 1\", \"Point 2\"]"); // Note: For arrays/lists, pass as a JSON string or similar
let generated_text = prompt.generate();
println!("{{}}", generated_text);
```

#### Prompt File Example Descriptions
If you are not familiar with [jinja2](https://jinja.palletsprojects.com/en/stable/), look at the prompt examples below also found in the `./prompts`

##### `prompts/system/base.prompt`

*   **Purpose:** A foundational system prompt that defines a general role for the AI. Demonstrates basic variable substitution with a default value.
*   **Features:** `{{ variable | default("value") }}`

##### `prompts/system/persona/helpful_assistant.md`

*   **Purpose:** Extends `base.prompt` to define a helpful AI assistant persona. Shows how to include other templates.
*   **Features:** `{% include "path/to/template" %}`, variable substitution.

##### `prompts/system/persona/creative_writer.txt`

*   **Purpose:** Defines a creative writer AI persona. Demonstrates conditional logic based on a variable.
*   **Features:** `{% if variable %}` `{% else %}` `{% endif %}`

##### `prompts/user/query.prompt`

*   **Purpose:** A simple user query prompt. Basic variable substitution.
*   **Features:** `{{ variable }}`

##### `prompts/user/summarize.md`

*   **Purpose:** Prompts the AI to summarize text and output key points as a bulleted list. Demonstrates looping over a list of items.
*   **Features:** `{% for item in list %}` `{% endfor %}`

##### `prompts/examples/few_shot_qa.prompt`

*   **Purpose:** Provides a few-shot question-answering example, where the AI learns from provided examples. Demonstrates looping and structured input.
*   **Features:** `{% for item in list %}` `{% endfor %}`

##### `prompts/examples/code_generation.txt`

*   **Purpose:** Prompts the AI to generate code in a specified language. Demonstrates complex conditional logic for different code structures.
*   **Features:** `{% if condition %}` `{% elif condition %}` `{% else %}` `{% endif %}`

##### `prompts/macros/common_macros.tera`

*   **Purpose:** Contains reusable macros that can be imported and used in other templates. Demonstrates macro definition and usage.
*   **Features:** `{% macro name(args) %}` `{% endmacro %}`

To use a macro from `common_macros.tera` in another template, you would typically import it:

```tera
{% import "macros/common_macros.tera" as common %}

{{ common::greeting(name="Alice") }}
```

## Contribution Guidelines

We welcome contributions from the community! Whether you're fixing a bug, adding a feature, or improving documentation, your help is appreciated.

Please read our [**Contributing Guidelines**](./.github/CONTRIBUTING.md) to get started.

### Commit Message Conventions

We use [**Conventional Commits**](https://www.conventionalcommits.org/en/v1.0.0/) for our commit messages. This helps us maintain a clear and organized commit history. Please see the [contribution guide](./.github/CONTRIBUTING.md#commit-messages) for details.

### Pull Request (PR) Standards

All PRs should be well-documented and include tests for any new code. Please fill out the [**pull request template**](./.github/pull_request_template.md) when submitting a PR.

## Code of Conduct

This project and everyone participating in it is governed by our [**Code of Conduct**](https://github.com/[your-username]/[your-repo]/blob/main/.github/CONTRIBUTING.md#code-of-conduct). By participating, you are expected to uphold this code.

## License

This project is licensed under the **MIT License**. See the [LICENSE](./LICENSE) file for details.
