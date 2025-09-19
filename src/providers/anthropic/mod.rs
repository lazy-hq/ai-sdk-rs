//! This module provides the Anthropic provider, which implements the `LanguageModel`
//! and `Provider` traits for interacting with the Anthropic API.

pub mod conversions;
pub mod settings;

use crate::core::types::LanguageModelStreamResponse;
use crate::providers::anthropic::settings::{
    AnthropicProviderSettings, AnthropicProviderSettingsBuilder,
};
use crate::{
    core::{
        language_model::LanguageModel,
        provider::Provider,
        types::{LanguageModelCallOptions, LanguageModelResponse, StreamChunkData},
    },
    error::Result,
};
use async_trait::async_trait;
use futures::StreamExt;
use reqwest::Client;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::{Arc, Mutex};

/// The Anthropic provider.
#[derive(Debug, Serialize)]
pub struct Anthropic {
    #[serde(skip)]
    client: Client,
    settings: AnthropicProviderSettings,
}

impl Anthropic {
    /// Creates a new `Anthropic` provider with the given settings.
    pub fn new(model_name: impl Into<String>) -> Self {
        AnthropicProviderSettingsBuilder::default()
            .model_name(model_name.into())
            .build()
            .expect("Failed to build AnthropicProviderSettings")
    }

    /// Anthropic provider setting builder.
    pub fn builder() -> AnthropicProviderSettingsBuilder {
        AnthropicProviderSettings::builder()
    }
}

impl Provider for Anthropic {}

impl Anthropic {
    fn parse_sse_event(event_text: &str) -> Option<Result<StreamChunkData>> {
        let data_str = event_text
            .lines()
            .find_map(|line| line.strip_prefix("data: "))
            .map(str::trim)?;

        if data_str == "[DONE]" {
            return Some(Ok(StreamChunkData {
                text: String::new(),
                stop_reason: Some("stop".to_string()),
            }));
        }

        if let Ok(event) = serde_json::from_str::<AnthropicStreamEvent>(data_str)
            && event.event_type == "content_block_delta"
            && let Some(delta) = event.data.get("delta")
            && let Some(text) = delta.get("text")
            && let Some(text_str) = text.as_str()
        {
            return Some(Ok(StreamChunkData {
                text: text_str.to_string(),
                stop_reason: None,
            }));
        }

        if let Ok(event) = serde_json::from_str::<AnthropicStreamEvent>(data_str)
            && event.event_type == "message_stop"
        {
            return Some(Ok(StreamChunkData {
                text: String::new(),
                stop_reason: Some("stop".to_string()),
            }));
        }

        None
    }
}

#[derive(Debug, Serialize)]
struct AnthropicMessage {
    role: String,
    content: String,
}

#[derive(Debug, Serialize)]
struct AnthropicRequest {
    model: String,
    max_tokens: u32,
    messages: Vec<AnthropicMessage>,
    #[serde(skip_serializing_if = "Option::is_none")]
    system: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    stream: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    temperature: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    top_p: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    top_k: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    stop_sequences: Option<Vec<String>>,
}

#[derive(Debug, Deserialize)]
struct AnthropicContent {
    #[serde(rename = "type")]
    content_type: String,
    text: String,
}

#[derive(Debug, Deserialize)]
struct AnthropicResponse {
    model: String,
    content: Vec<AnthropicContent>,
    stop_reason: Option<String>,
}

#[derive(Debug, Deserialize)]
struct AnthropicStreamEvent {
    #[serde(rename = "type")]
    event_type: String,
    #[serde(flatten)]
    data: HashMap<String, serde_json::Value>,
}

#[async_trait]
impl LanguageModel for Anthropic {
    fn provider_name(&self) -> &str {
        &self.settings.provider_name
    }

    async fn generate(
        &mut self,
        options: LanguageModelCallOptions,
    ) -> Result<LanguageModelResponse> {
        let mut request: AnthropicRequest = options.into();
        request.model = self.settings.model_name.clone();

        let response = self
            .client
            .post("https://api.anthropic.com/v1/messages")
            .header("Content-Type", "application/json")
            .header("x-api-key", &self.settings.api_key)
            .header("anthropic-version", "2023-06-01")
            .json(&request)
            .send()
            .await?
            .json::<AnthropicResponse>()
            .await?;

        let text = response
            .content
            .iter()
            .filter(|c| c.content_type == "text")
            .fold(String::new(), |mut acc, c| {
                if !acc.is_empty() {
                    acc.push(' ');
                }
                acc.push_str(&c.text);
                acc
            });

        Ok(LanguageModelResponse {
            model: Some(response.model),
            text,
            stop_reason: response.stop_reason,
        })
    }

    async fn generate_stream(
        &mut self,
        options: LanguageModelCallOptions,
    ) -> Result<LanguageModelStreamResponse> {
        let mut request: AnthropicRequest = options.into();
        request.model = self.settings.model_name.clone();
        request.stream = Some(true);

        let response = self
            .client
            .post("https://api.anthropic.com/v1/messages")
            .header("Content-Type", "application/json")
            .header("x-api-key", &self.settings.api_key)
            .header("anthropic-version", "2023-06-01")
            .json(&request)
            .send()
            .await?;

        let buffer = Arc::new(Mutex::new(String::new()));

        let stream = response.bytes_stream().map(move |chunk_result| {
            match chunk_result {
                Ok(chunk) => {
                    let mut buffer_guard = buffer.lock().unwrap();
                    buffer_guard.push_str(&String::from_utf8_lossy(&chunk));

                    // Process complete SSE events (separated by double newlines)
                    while let Some(event_end) = buffer_guard.find("\n\n") {
                        let event_text = &buffer_guard[..event_end];
                        if let Some(chunk_data) = Self::parse_sse_event(event_text) {
                            buffer_guard.drain(..event_end + 2);
                            return chunk_data;
                        }
                        buffer_guard.drain(..event_end + 2);
                    }

                    Ok(StreamChunkData {
                        text: String::new(),
                        stop_reason: None,
                    })
                }
                Err(e) => Err(e.into()),
            }
        });

        Ok(LanguageModelStreamResponse {
            stream: Box::pin(stream),
            model: Some(self.settings.model_name.to_string()),
        })
    }
}
