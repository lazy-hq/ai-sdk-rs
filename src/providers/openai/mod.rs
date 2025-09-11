//! This module provides the OpenAI provider, which implements the `LanguageModel`
//! and `Provider` traits for interacting with the OpenAI API.

pub mod conversions;
pub mod settings;
use async_openai::types::responses::{
    Content, CreateResponse, OutputContent, Response, ResponseEvent, ResponseStream,
};
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

    async fn generate(
        &mut self,
        options: LanguageModelCallOptions,
    ) -> Result<LanguageModelResponse> {
        let mut request: CreateResponse = options.into();
        request.model = self.settings.model_name.to_string();

        let response: Response = self.client.responses().create(request).await?;
        let text = response
            .output
            .iter()
            .find_map(|out| match out {
                OutputContent::Message(msg) => msg.content.iter().find_map(|c| match c {
                    Content::OutputText(t) => Some(t.text.to_string()),
                    _ => None,
                }),
                _ => None,
            })
            .unwrap_or_default();

        Ok(LanguageModelResponse {
            model: Some(response.model.to_string()),
            text,
            stop_reason: None,
        })
    }

    async fn generate_stream(
        &mut self,
        options: LanguageModelCallOptions,
    ) -> Result<LanguageModelStreamingResponse> {
        let mut request: CreateResponse = options.into();
        request.model = self.settings.model_name.to_string();
        request.stream = Some(true);

        let openai_stream: ResponseStream = self.client.responses().create_stream(request).await?;

        #[derive(Default)]
        struct StreamState {
            stop_reason: Option<String>,
            completed: bool,
        }

        let stream = openai_stream.scan(StreamState::default(), |state, evt_res| {
            // If already completed, don't emit anything more
            if state.completed {
                return futures::future::ready(None);
            }

            futures::future::ready(match evt_res {
                Ok(ResponseEvent::ResponseOutputTextDelta(d)) => Some(Ok(LanguageModelResponse {
                    text: d.delta,
                    model: None,
                    stop_reason: None,
                })),
                Ok(ResponseEvent::ResponseCompleted(_)) => {
                    state.stop_reason = Some("completed".into());
                    state.completed = true;
                    Some(Ok(LanguageModelResponse {
                        text: String::new(),
                        model: None,
                        stop_reason: Some("completed".into()),
                    }))
                }
                Ok(ResponseEvent::ResponseFailed(f)) => {
                    let reason = f
                        .response
                        .error
                        .as_ref()
                        .map(|e| format!("{}: {}", e.code, e.message))
                        .unwrap_or_else(|| "unknown failure".to_string());

                    state.completed = true;
                    state.stop_reason = Some(reason);

                    Some(Ok(LanguageModelResponse {
                        text: String::new(),
                        model: None,
                        stop_reason: state.stop_reason.clone(),
                    }))
                }
                // TODO: handle other events
                Ok(_) => Some(Ok(LanguageModelResponse {
                    text: String::new(),
                    model: None,
                    stop_reason: None,
                })),
                Err(e) => {
                    state.completed = true;
                    Some(Err(e.into()))
                }
            })
        });

        Ok(Box::pin(stream) as LanguageModelStreamingResponse)
    }
}
