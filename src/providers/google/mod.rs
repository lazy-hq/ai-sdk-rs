//! This module provides Google providers (GoogleGenerativeAI and VertexAI) which implement
//! the `LanguageModel` and `Provider` traits for interacting with Google's AI APIs.

pub mod conversions;
pub mod settings;

use reqwest::Client;
use serde::{Deserialize, Serialize};

use crate::core::types::LanguageModelStreamResponse;
use crate::providers::google::settings::{
    GoogleGenerativeAISettings, GoogleGenerativeAISettingsBuilder, VertexAISettings,
    VertexAISettingsBuilder,
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

/// The GoogleGenerativeAI provider (ai.google.dev).
#[derive(Debug, Serialize)]
pub struct GoogleGenerativeAI {
    #[serde(skip)]
    client: Client,
    settings: GoogleGenerativeAISettings,
}

/// The VertexAI provider (cloud.google.com/vertex-ai).
#[derive(Debug, Serialize)]
pub struct VertexAI {
    #[serde(skip)]
    client: Client,
    settings: VertexAISettings,
}

// Common structures for Google APIs
#[derive(Debug, Serialize)]
struct GoogleMessage {
    role: String,
    parts: Vec<GooglePart>,
}

#[derive(Debug, Serialize)]
struct GooglePart {
    text: String,
}

#[derive(Debug, Serialize)]
struct GoogleRequest {
    contents: Vec<GoogleMessage>,
    #[serde(skip_serializing_if = "Option::is_none")]
    system_instruction: Option<GoogleSystemInstruction>,
    #[serde(skip_serializing_if = "Option::is_none")]
    generation_config: Option<GoogleGenerationConfig>,
}

#[derive(Debug, Serialize)]
struct GoogleSystemInstruction {
    parts: Vec<GooglePart>,
}

#[derive(Debug, Serialize)]
struct GoogleGenerationConfig {
    #[serde(skip_serializing_if = "Option::is_none")]
    max_output_tokens: Option<u32>,
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
struct GoogleResponse {
    candidates: Vec<GoogleCandidate>,
}

#[derive(Debug, Deserialize)]
struct GoogleCandidate {
    content: GoogleContent,
    #[serde(rename = "finishReason")]
    finish_reason: Option<String>,
}

#[derive(Debug, Deserialize)]
struct GoogleContent {
    parts: Vec<GoogleResponsePart>,
}

#[derive(Debug, Deserialize)]
struct GoogleResponsePart {
    text: String,
}

#[derive(Debug, Deserialize)]
struct GoogleStreamEvent {
    candidates: Option<Vec<GoogleCandidate>>,
}

struct Google;

impl Google {
    async fn parse_stream_response(
        response: reqwest::Response,
        model_name: &str,
    ) -> Result<LanguageModelStreamResponse> {
        // Google returns a JSON array, so let's collect all bytes and parse as array
        let bytes = response.bytes().await?;
        let json_str = String::from_utf8_lossy(&bytes);

        // Parse as array of GoogleStreamEvent
        let events: Vec<GoogleStreamEvent> = serde_json::from_str(&json_str)?;

        // Convert to stream
        let stream = futures::stream::iter(events.into_iter().map(|event| {
            if let Some(candidates) = event.candidates
                && let Some(candidate) = candidates.into_iter().next()
            {
                let content_text = candidate
                    .content
                    .parts
                    .into_iter()
                    .map(|part| part.text)
                    .collect::<String>();

                Ok(StreamChunkData {
                    text: content_text,
                    stop_reason: candidate.finish_reason,
                })
            } else {
                Ok(StreamChunkData {
                    text: String::new(),
                    stop_reason: None,
                })
            }
        }));

        Ok(LanguageModelStreamResponse {
            stream: Box::pin(stream),
            model: Some(model_name.to_string()),
        })
    }
}

// ===== GoogleGenerativeAI Implementation =====

impl GoogleGenerativeAI {
    /// Creates a new `GoogleGenerativeAI` provider with the given model name.
    pub fn new(model_name: impl Into<String>) -> Self {
        GoogleGenerativeAISettingsBuilder::default()
            .model_name(model_name.into())
            .build()
            .expect("Failed to build GoogleGenerativeAISettings")
    }

    /// GoogleGenerativeAI provider setting builder.
    pub fn builder() -> GoogleGenerativeAISettingsBuilder {
        GoogleGenerativeAISettings::builder()
    }
}

impl Provider for GoogleGenerativeAI {}

#[async_trait]
impl LanguageModel for GoogleGenerativeAI {
    fn provider_name(&self) -> &str {
        &self.settings.provider_name
    }

    async fn generate(
        &mut self,
        options: LanguageModelCallOptions,
    ) -> Result<LanguageModelResponse> {
        let request: GoogleRequest = options.into();

        let url = format!(
            "https://generativelanguage.googleapis.com/v1beta/models/{}:generateContent",
            self.settings.model_name
        );

        let response = self
            .client
            .post(&url)
            .header("Content-Type", "application/json")
            .header("x-goog-api-key", &self.settings.api_key)
            .json(&request)
            .send()
            .await?
            .json::<GoogleResponse>()
            .await?;

        let (text, stop_reason) = response
            .candidates
            .into_iter()
            .next()
            .map(|candidate| {
                let text = candidate
                    .content
                    .parts
                    .into_iter()
                    .map(|part| part.text)
                    .collect::<Vec<_>>()
                    .join(" ");
                (text, candidate.finish_reason)
            })
            .unwrap_or_default();

        Ok(LanguageModelResponse {
            model: Some(self.settings.model_name.to_string()),
            text,
            stop_reason,
        })
    }

    async fn generate_stream(
        &mut self,
        options: LanguageModelCallOptions,
    ) -> Result<LanguageModelStreamResponse> {
        let request: GoogleRequest = options.into();

        let url = format!(
            "https://generativelanguage.googleapis.com/v1beta/models/{}:streamGenerateContent",
            self.settings.model_name
        );

        let response = self
            .client
            .post(&url)
            .header("Content-Type", "application/json")
            .header("x-goog-api-key", &self.settings.api_key)
            .json(&request)
            .send()
            .await?;

        Google::parse_stream_response(response, &self.settings.model_name).await
    }
}

// ===== VertexAI Implementation =====

impl VertexAI {
    /// Creates a new `VertexAI` provider with the given project, location, and model name.
    pub fn new(
        project_id: impl Into<String>,
        location: impl Into<String>,
        model_name: impl Into<String>,
    ) -> Self {
        VertexAISettingsBuilder::default()
            .project_id(project_id.into())
            .location(location.into())
            .model_name(model_name.into())
            .build()
            .expect("Failed to build VertexAISettings")
    }

    /// VertexAI provider setting builder.
    pub fn builder() -> VertexAISettingsBuilder {
        VertexAISettings::builder()
    }
}

impl Provider for VertexAI {}

#[async_trait]
impl LanguageModel for VertexAI {
    fn provider_name(&self) -> &str {
        &self.settings.provider_name
    }

    async fn generate(
        &mut self,
        options: LanguageModelCallOptions,
    ) -> Result<LanguageModelResponse> {
        let request: GoogleRequest = options.into();

        let url = format!(
            "https://{}-aiplatform.googleapis.com/v1/projects/{}/locations/{}/publishers/google/models/{}:generateContent",
            self.settings.location,
            self.settings.project_id,
            self.settings.location,
            self.settings.model_name
        );

        let response = self
            .client
            .post(&url)
            .header("Content-Type", "application/json")
            .bearer_auth(&self.settings.access_token)
            .json(&request)
            .send()
            .await?
            .json::<GoogleResponse>()
            .await?;

        let (text, stop_reason) = response
            .candidates
            .into_iter()
            .next()
            .map(|candidate| {
                let text = candidate
                    .content
                    .parts
                    .into_iter()
                    .map(|part| part.text)
                    .collect::<Vec<_>>()
                    .join(" ");
                (text, candidate.finish_reason)
            })
            .unwrap_or_default();

        Ok(LanguageModelResponse {
            model: Some(self.settings.model_name.to_string()),
            text,
            stop_reason,
        })
    }

    async fn generate_stream(
        &mut self,
        options: LanguageModelCallOptions,
    ) -> Result<LanguageModelStreamResponse> {
        let request: GoogleRequest = options.into();

        let url = format!(
            "https://{}-aiplatform.googleapis.com/v1/projects/{}/locations/{}/publishers/google/models/{}:streamGenerateContent",
            self.settings.location,
            self.settings.project_id,
            self.settings.location,
            self.settings.model_name
        );

        let response = self
            .client
            .post(&url)
            .header("Content-Type", "application/json")
            .bearer_auth(&self.settings.access_token)
            .json(&request)
            .send()
            .await?;

        Google::parse_stream_response(response, &self.settings.model_name).await
    }
}
