//! Defines the settings for Google providers.

use reqwest::Client;
use serde::{Deserialize, Serialize};

use crate::{
    error::Error,
    providers::google::{GoogleGenerativeAI, VertexAI},
};

/// Settings for the GoogleGenerativeAI provider.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GoogleGenerativeAISettings {
    /// The API key for the Google AI API.
    pub api_key: String,
    /// The name of the provider.
    pub provider_name: String,
    /// The name of the model to use.
    pub model_name: String,
}

impl GoogleGenerativeAISettings {
    /// Creates a new builder for `GoogleGenerativeAISettings`.
    pub fn builder() -> GoogleGenerativeAISettingsBuilder {
        GoogleGenerativeAISettingsBuilder::default()
    }
}

pub struct GoogleGenerativeAISettingsBuilder {
    api_key: Option<String>,
    provider_name: Option<String>,
    model_name: Option<String>,
}

impl GoogleGenerativeAISettingsBuilder {
    pub fn api_key(mut self, api_key: impl Into<String>) -> Self {
        self.api_key = Some(api_key.into());
        self
    }

    pub fn provider_name(mut self, provider_name: impl Into<String>) -> Self {
        self.provider_name = Some(provider_name.into());
        self
    }

    pub fn model_name(mut self, model_name: impl Into<String>) -> Self {
        self.model_name = Some(model_name.into());
        self
    }

    pub fn build(self) -> Result<GoogleGenerativeAI, Error> {
        let settings = GoogleGenerativeAISettings {
            api_key: self.api_key.unwrap_or_default(),
            provider_name: self
                .provider_name
                .unwrap_or_else(|| "google-generative-ai".to_string()),
            model_name: self
                .model_name
                .unwrap_or_else(|| "gemini-1.5-flash".to_string()),
        };

        let client = Client::new();

        Ok(GoogleGenerativeAI { settings, client })
    }
}

impl Default for GoogleGenerativeAISettingsBuilder {
    fn default() -> Self {
        Self {
            api_key: Some(std::env::var("GOOGLE_API_KEY").unwrap_or_default()),
            provider_name: Some("google-generative-ai".to_string()),
            model_name: Some("gemini-1.5-flash".to_string()),
        }
    }
}

/// Settings for the VertexAI provider.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VertexAISettings {
    /// The Google Cloud project ID.
    pub project_id: String,
    /// The Google Cloud region/location.
    pub location: String,
    /// The access token for authentication.
    pub access_token: String,
    /// The name of the provider.
    pub provider_name: String,
    /// The name of the model to use.
    pub model_name: String,
}

impl VertexAISettings {
    /// Creates a new builder for `VertexAISettings`.
    pub fn builder() -> VertexAISettingsBuilder {
        VertexAISettingsBuilder::default()
    }
}

pub struct VertexAISettingsBuilder {
    project_id: Option<String>,
    location: Option<String>,
    access_token: Option<String>,
    provider_name: Option<String>,
    model_name: Option<String>,
}

impl VertexAISettingsBuilder {
    pub fn project_id(mut self, project_id: impl Into<String>) -> Self {
        self.project_id = Some(project_id.into());
        self
    }

    pub fn location(mut self, location: impl Into<String>) -> Self {
        self.location = Some(location.into());
        self
    }

    pub fn access_token(mut self, access_token: impl Into<String>) -> Self {
        self.access_token = Some(access_token.into());
        self
    }

    pub fn provider_name(mut self, provider_name: impl Into<String>) -> Self {
        self.provider_name = Some(provider_name.into());
        self
    }

    pub fn model_name(mut self, model_name: impl Into<String>) -> Self {
        self.model_name = Some(model_name.into());
        self
    }

    pub fn build(self) -> Result<VertexAI, Error> {
        let settings = VertexAISettings {
            project_id: self
                .project_id
                .unwrap_or_else(|| std::env::var("GOOGLE_CLOUD_PROJECT").unwrap_or_default()),
            location: self.location.unwrap_or_else(|| "us-central1".to_string()),
            access_token: self.access_token.unwrap_or_default(),
            provider_name: self
                .provider_name
                .unwrap_or_else(|| "vertex-ai".to_string()),
            model_name: self
                .model_name
                .unwrap_or_else(|| "gemini-1.5-flash".to_string()),
        };

        let client = Client::new();

        Ok(VertexAI { settings, client })
    }
}

impl Default for VertexAISettingsBuilder {
    fn default() -> Self {
        Self {
            project_id: Some(std::env::var("GOOGLE_CLOUD_PROJECT").unwrap_or_default()),
            location: Some("us-central1".to_string()),
            access_token: Some(std::env::var("GOOGLE_ACCESS_TOKEN").unwrap_or_default()),
            provider_name: Some("vertex-ai".to_string()),
            model_name: Some("gemini-1.5-flash".to_string()),
        }
    }
}
