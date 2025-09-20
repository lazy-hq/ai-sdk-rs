//! Provides the primary user-facing function for text generation.
//!
//! This module contains the `generate_text` function, which serves as the
//! main entry point for consumers of the SDK to generate text using any
//! model that implements the `LanguageModel` trait.

use crate::{
    core::{
        language_model::{GenerateOptions, LanguageModel, LanguageModelOptions},
        types::GenerateTextResponse,
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
pub async fn generate_text<M: LanguageModel>(
    mut options: GenerateOptions<M>,
) -> Result<GenerateTextResponse> {
    let (system_prompt, messages) =
        resolve_message(&options.system, &options.prompt, &options.messages);

    let response = options
        .model
        .generate(
            LanguageModelOptions::builder()
                .system(system_prompt)
                .messages(messages)
                .max_output_tokens(options.max_output_tokens)
                .temperature(options.temperature)
                .top_p(options.top_p)
                .top_k(options.top_k)
                .stop(options.stop.clone())
                .build()?,
        )
        .await?;

    let result = GenerateTextResponse::new(response.text);

    Ok(result)
}
