//! Provides the primary user-facing function for text generation.
//!
//! This module contains the `generate_text` function, which serves as the
//! main entry point for consumers of the SDK to generate text using any
//! model that implements the `LanguageModel` trait.

use crate::{
    core::{
        language_model::LanguageModel,
        types::{GenerateTextCallOptions, GenerateTextResponse, LanguageModelCallOptions},
        utils::resolve_message,
    },
    error::Result,
};

/// Generates text using a specified language model.
///
/// Generate a text and call tools for a given prompt using a language model.
/// This function does not stream the output. If you want to stream the output, use `generate_stream` instead.
///
/// # Arguments
///
/// * `model` - A language model that implements the `LanguageModel` trait.
///
/// * `options` - A `GenerateTextCallOptions` struct containing the model, prompt,
///   and other parameters for the request.
///
/// # Errors
///
/// Returns an `Error` if the underlying model fails to generate a response.
pub async fn generate_text(
    mut model: impl LanguageModel,
    options: GenerateTextCallOptions,
) -> Result<GenerateTextResponse> {
    let (system_prompt, messages) =
        resolve_message(options.system, options.prompt, options.messages);

    let response = model
        .generate(
            LanguageModelCallOptions::builder()
                .system(system_prompt)
                .messages(messages)
                .max_tokens(options.max_tokens)
                .temprature(options.temprature)
                .top_p(options.top_p)
                .top_k(options.top_k)
                .stop(options.stop)
                .build()?,
        )
        .await?;

    let result = GenerateTextResponse::new(response.text);

    Ok(result)
}
