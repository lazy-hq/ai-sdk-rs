//! Defines the settings for the Anthropic provider.

use reqwest::Client;
use serde::{Deserialize, Serialize};

use crate::{error::Error, providers::anthropic::Anthropic};

/// Settings for the Anthropic provider.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AnthropicProviderSettings {
    /// The API key for the Anthropic API.
    pub api_key: String,

    /// The name of the provider.
    pub provider_name: String,

    /// The name of the model to use.
    pub model_name: String,
}

impl AnthropicProviderSettings {
    /// Creates a new builder for `AnthropicProviderSettings`.
    pub fn builder() -> AnthropicProviderSettingsBuilder {
        AnthropicProviderSettingsBuilder::default()
    }
}

pub struct AnthropicProviderSettingsBuilder {
    api_key: Option<String>,
    provider_name: Option<String>,
    model_name: Option<String>,
}

impl AnthropicProviderSettingsBuilder {
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

    pub fn build(self) -> Result<Anthropic, Error> {
        let settings = AnthropicProviderSettings {
            api_key: self.api_key.unwrap_or_default(),
            provider_name: self
                .provider_name
                .unwrap_or_else(|| "anthropic".to_string()),
            model_name: self
                .model_name
                .unwrap_or_else(|| "claude-4-sonnet".to_string()),
        };

        let client = Client::new();

        Ok(Anthropic { settings, client })
    }
}

impl Default for AnthropicProviderSettingsBuilder {
    fn default() -> Self {
        Self {
            api_key: Some(std::env::var("ANTHROPIC_API_KEY").unwrap_or_default()),
            provider_name: Some("anthropic".to_string()),
            model_name: Some("claude-4-sonnet".to_string()),
        }
    }
}
