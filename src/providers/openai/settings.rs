//! Defines the settings for the OpenAI provider.

use derive_builder::Builder;
use serde::{Deserialize, Serialize};

use crate::error::Error;

// TODO: improve the settings types to fully match the OpenAI API
/// Settings for the OpenAI provider.
#[derive(Debug, Clone, Builder, Serialize, Deserialize)]
#[builder(pattern = "owned", setter(into), build_fn(error = "Error"))]
pub struct OpenAIProviderSettings {
    /// The API key for the OpenAI API.
    #[builder(default = "std::env::var(\"OPENAI_API_KEY\").unwrap_or_default()")]
    pub api_key: String,

    /// The name of the provider.
    #[builder(default = "\"openai\".to_string()")]
    pub provider_name: String,
}

impl OpenAIProviderSettings {
    /// Creates a new builder for `OpenAISettings`.
    pub fn builder() -> OpenAIProviderSettingsBuilder {
        OpenAIProviderSettingsBuilder::default()
    }
}
