//! Provides the primary user-facing function for text stream generation.

use crate::{
    core::{
        language_model::LanguageModel,
        types::{GenerateStreamResponse, GenerateTextCallOptions, LanguageModelCallOptions},
        utils::resolve_message,
    },
    error::Result,
};

/// Generates Streaming text using a specified language model.
///
/// Generate a text and call tools for a given prompt using a language model.
/// This function streams the output. If you do not want to stream the output, use `generate_text` instead.
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
pub async fn generate_stream(
    mut model: impl LanguageModel,
    options: GenerateTextCallOptions,
) -> Result<GenerateStreamResponse> {
    let (system_prompt, messages) =
        resolve_message(options.system, options.prompt, options.messages);

    let response = model
        .generate_stream(
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

    Ok(GenerateStreamResponse { stream: response })
}
