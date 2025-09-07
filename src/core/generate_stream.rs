//! Provides the primary user-facing function for text stream generation. See
//! [crate::core::generate_text]
//!
//! This module contains the `generate_stream` function, which helps generate
//! text in a streaming manner using any model that implements the
//! `LanguageModel` trait.

use crate::{
    core::{
        language_model::LanguageModel,
        types::{GenerateStreamResponse, GenerateTextCallOptions, LanguageModelCallOptions},
    },
    error::Result,
};

pub async fn generate_stream(
    mut model: impl LanguageModel,
    options: GenerateTextCallOptions,
) -> Result<GenerateStreamResponse> {
    let response = model
        .generate_stream(
            LanguageModelCallOptions::builder()
                .prompt(options.prompt)
                .build()?,
        )
        .await?;

    Ok(GenerateStreamResponse { stream: response })
}
