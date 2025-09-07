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
