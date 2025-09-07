//! Provides the primary user-facing function for text generation.
//!
//! This module contains the `generate_text` function, which serves as the
//! main entry point for consumers of the SDK to generate text using any
//! model that implements the `LanguageModel` trait.

use crate::{
    core::{
        language_model::LanguageModel,
        types::{GenerateTextCallOptions, GenerateTextResponse, LanguageModelCallOptions},
    },
    error::Result,
};

/// Generates text using a specified language model.
///
/// This function orchestrates the text generation process by taking a configured
/// set of options, invoking the `generate` method on the provided language model,
/// and returning a standardized response.
///
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
    let response = model
        .generate(
            LanguageModelCallOptions::builder()
                .prompt(options.prompt)
                .build()?,
        )
        .await?;

    let result = GenerateTextResponse::new(response.text);

    Ok(result)
}
