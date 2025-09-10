//! This module provides the OpenAI provider, which implements the `LanguageModel`
//! and `Provider` traits for interacting with the OpenAI API.

pub mod conversions;
pub mod settings;
use async_openai::types::CreateChatCompletionRequestArgs;
use async_openai::{Client, config::OpenAIConfig};
use futures::StreamExt;
pub use settings::OpenAIProviderSettings;

use crate::{
    core::{
        language_model::LanguageModel,
        provider::Provider,
        types::{LanguageModelCallOptions, LanguageModelResponse, LanguageModelStreamingResponse},
    },
    error::Result,
};
use async_trait::async_trait;
use serde::Serialize;

/// The OpenAI provider.
#[derive(Debug, Serialize)]
pub struct OpenAI {
    #[serde(skip)]
    client: Client<OpenAIConfig>,
    settings: OpenAIProviderSettings,
}

impl OpenAI {
    /// Creates a new `OpenAI` provider with the given settings.
    pub fn new(settings: OpenAIProviderSettings) -> Self {
        let client =
            Client::with_config(OpenAIConfig::new().with_api_key(settings.api_key.to_string()));

        Self { client, settings }
    }
}

impl Provider for OpenAI {}

#[async_trait]
impl LanguageModel for OpenAI {
    fn provider_name(&self) -> &str {
        &self.settings.provider_name
    }

    fn model_name(&self) -> &str {
        &self.settings.model_name
    }

    async fn generate(
        &mut self,
        options: LanguageModelCallOptions,
    ) -> Result<LanguageModelResponse> {
        let mut request_builder: CreateChatCompletionRequestArgs = From::from(options);
        request_builder.model(self.model_name());

        let response = self.client.chat().create(request_builder.build()?).await?;
        let text = match response.choices.first() {
            Some(choice) => &choice.message.content.clone().expect("no content"),
            None => "",
        };

        Ok(LanguageModelResponse {
            model: Some(response.model),
            text: text.to_string(),
        })
    }

    async fn generate_stream(
        &mut self,
        options: LanguageModelCallOptions,
    ) -> Result<LanguageModelStreamingResponse> {
        let mut request_builder: CreateChatCompletionRequestArgs = From::from(options);
        request_builder.model(self.model_name());
        request_builder.stream(true);

        let response = self
            .client
            .chat()
            .create_stream(request_builder.build()?)
            .await?;
        let r = response
            .map(|res| {
                Ok(LanguageModelResponse::new(
                    res?.choices
                        .first()
                        .ok_or::<async_openai::error::OpenAIError>(
                            async_openai::error::OpenAIError::StreamError(
                                "Stream chunk has no content".to_string(),
                            ),
                        )?
                        .delta
                        .content
                        .clone()
                        .unwrap_or("".to_string()),
                ))
            })
            .boxed();
        Ok(r)
    }
}
