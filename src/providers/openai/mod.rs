//! This module provides the OpenAI provider, which implements the `LanguageModel`
//! and `Provider` traits for interacting with the OpenAI API.

pub mod settings;

use async_openai::types::{
    ChatCompletionRequestMessage, ChatCompletionRequestUserMessage,
    ChatCompletionRequestUserMessageContent, ChatCompletionRequestUserMessageContentPart,
    CreateChatCompletionRequestArgs,
};
use async_openai::{Client, config::OpenAIConfig};
use futures::StreamExt;
pub use settings::OpenAIProviderSettings;

//use self::client::{ChatCompletionRequest, Message};
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

    fn user_message(message: &str) -> ChatCompletionRequestMessage {
        ChatCompletionRequestMessage::User(ChatCompletionRequestUserMessage::from(message))
    }
}

struct OpenAiMessage(ChatCompletionRequestMessage);

impl From<OpenAiMessage> for String {
    /// Handle the conversion from any `OpenAiMessage` to `String`. Currently it only handles
    /// user messages that are texts or part of a text. returns empty string if it is not.
    fn from(value: OpenAiMessage) -> Self {
        match value.0 {
            ChatCompletionRequestMessage::User(user_message) => match &user_message.content {
                ChatCompletionRequestUserMessageContent::Text(text) => text.to_string(),
                ChatCompletionRequestUserMessageContent::Array(arr) => match arr.first().unwrap() {
                    ChatCompletionRequestUserMessageContentPart::Text(text) => {
                        text.text.to_string()
                    }
                    _ => "".to_string(),
                },
            },
            _ => "".to_string(),
        }
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
        let openai_msg: ChatCompletionRequestMessage =
            OpenAiMessage(OpenAI::user_message(&options.prompt)).0;
        let request = CreateChatCompletionRequestArgs::default()
            .model(self.model_name().to_string())
            .messages(vec![openai_msg])
            .build()?;

        let response = self.client.chat().create(request).await?;
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
        let openai_msg: ChatCompletionRequestMessage =
            OpenAiMessage(OpenAI::user_message(&options.prompt)).0;
        let request = CreateChatCompletionRequestArgs::default()
            .model(self.model_name().to_string())
            .stream(true)
            .messages(vec![openai_msg])
            .build()?;

        let response = self.client.chat().create_stream(request).await?;

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
