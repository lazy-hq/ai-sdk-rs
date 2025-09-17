//! Defines the settings for the OpenAI provider.

use async_openai::{Client, config::OpenAIConfig};
use serde::{Deserialize, Serialize};

use crate::{error::Error, providers::openai::OpenAI};

// TODO: improve the settings types to fully match the OpenAI API
/// Settings for the OpenAI provider.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OpenAIProviderSettings {
    /// The API key for the OpenAI API.
    pub api_key: String,

    /// The name of the provider.
    pub provider_name: String,

    /// The name of the model to use.
    pub model_name: String,
}

impl OpenAIProviderSettings {
    /// Creates a new builder for `OpenAISettings`.
    pub fn builder() -> OpenAIProviderSettingsBuilder {
        OpenAIProviderSettingsBuilder::default()
    }
}

pub struct OpenAIProviderSettingsBuilder {
    api_key: Option<String>,
    provider_name: Option<String>,
    model_name: Option<String>,
}

impl OpenAIProviderSettingsBuilder {
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

    pub fn build(self) -> Result<OpenAI, Error> {
        let settings = OpenAIProviderSettings {
            api_key: self.api_key.unwrap_or_default(),
            provider_name: self.provider_name.unwrap_or_else(|| "openai".to_string()),
            model_name: self.model_name.unwrap_or_else(|| "gpt-4o".to_string()),
        };

        let client =
            Client::with_config(OpenAIConfig::new().with_api_key(settings.api_key.to_string()));

        Ok(OpenAI { settings, client })
    }
}

impl Default for OpenAIProviderSettingsBuilder {
    fn default() -> Self {
        Self {
            api_key: Some(std::env::var("OPENAI_API_KEY").unwrap_or_default()),
            provider_name: Some("openai".to_string()),
            model_name: Some("gpt-4o".to_string()),
        }
    }
}
