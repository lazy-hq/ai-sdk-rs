//! Provides the primary user-facing function for text stream generation.

use crate::{
    core::{
        language_model::{GenerateOptions, LanguageModel, LanguageModelOptions},
        types::LanguageModelStreamResponse,
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
pub async fn generate_stream<M: LanguageModel>(
    mut options: GenerateOptions<M>,
) -> Result<LanguageModelStreamResponse> {
    let (system_prompt, messages) =
        resolve_message(&options.system, &options.prompt, &options.messages);

    let response = options
        .model
        .generate_stream(
            LanguageModelOptions::builder()
                .system(system_prompt)
                .messages(messages)
                .max_output_tokens(options.max_output_tokens)
                .temperature(options.temperature)
                .top_p(options.top_p)
                .top_k(options.top_k)
                .stop_sequences(options.stop_sequences.to_owned())
                .build()?,
        )
        .await?;

    Ok(response)
}
