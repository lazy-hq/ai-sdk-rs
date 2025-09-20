//! Data structures for the models.dev API schema.
//!
//! This module contains exact schema matches for the models.dev API response structures,
//! including nested objects for models, costs, limits, and modalities.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Top-level response structure from the models.dev API.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelsDevResponse {
    /// Array of provider information.
    pub providers: Vec<Provider>,
}

/// Provider information from the models.dev API.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Provider {
    /// The unique identifier for the provider.
    pub id: String,

    /// The display name of the provider.
    pub name: String,

    /// NPM package information for the provider.
    pub npm: NpmInfo,

    /// Environment variables required for the provider.
    pub env: Vec<EnvVar>,

    /// Documentation information.
    pub doc: DocInfo,

    /// API configuration information.
    pub api: ApiInfo,

    /// Available models for this provider.
    pub models: Vec<Model>,
}

/// NPM package information for a provider.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NpmInfo {
    /// The NPM package name.
    pub name: String,

    /// The NPM package version.
    pub version: String,
}

/// Environment variable configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EnvVar {
    /// The name of the environment variable.
    pub name: String,

    /// Description of what the environment variable is used for.
    pub description: String,

    /// Whether this environment variable is required.
    pub required: bool,
}

/// Documentation information for a provider.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DocInfo {
    /// URL to the provider's documentation.
    pub url: String,

    /// Additional documentation metadata.
    #[serde(default)]
    pub metadata: HashMap<String, serde_json::Value>,
}

/// API configuration information for a provider.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ApiInfo {
    /// The base URL for the provider's API.
    pub base_url: String,

    /// API version information.
    #[serde(default)]
    pub version: Option<String>,

    /// Additional API configuration.
    #[serde(default)]
    pub config: HashMap<String, serde_json::Value>,
}

/// Model information from the models.dev API.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Model {
    /// The unique identifier for the model.
    pub id: String,

    /// The display name of the model.
    pub name: String,

    /// A description of the model.
    pub description: String,

    /// Cost information for using this model.
    pub cost: ModelCost,

    /// Limits for this model.
    pub limits: ModelLimit,

    /// Supported modalities for this model.
    pub modalities: Modalities,

    /// Additional model metadata.
    #[serde(default)]
    pub metadata: HashMap<String, serde_json::Value>,
}

/// Cost information for a model.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelCost {
    /// Cost per 1M input tokens.
    pub input: f64,

    /// Cost per 1M output tokens.
    pub output: f64,

    /// Cost per 1M cache read tokens (if supported).
    #[serde(default)]
    pub cache_read: Option<f64>,

    /// Cost per 1M cache write tokens (if supported).
    #[serde(default)]
    pub cache_write: Option<f64>,

    /// Cost per 1M reasoning tokens (if supported).
    #[serde(default)]
    pub reasoning: Option<f64>,

    /// Currency for the costs (e.g., "USD").
    #[serde(default = "default_currency")]
    pub currency: String,
}

/// Limits for a model.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelLimit {
    /// Maximum context window size in tokens.
    pub context: u32,

    /// Maximum output tokens per request.
    pub output: u32,

    /// Additional limits.
    #[serde(default)]
    pub metadata: HashMap<String, serde_json::Value>,
}

/// Supported modalities for a model.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Modalities {
    /// Supported input modalities.
    pub input: Vec<String>,

    /// Supported output modalities.
    pub output: Vec<String>,
}

impl Default for ModelCost {
    fn default() -> Self {
        Self {
            input: 0.0,
            output: 0.0,
            cache_read: None,
            cache_write: None,
            reasoning: None,
            currency: default_currency(),
        }
    }
}

fn default_currency() -> String {
    "USD".to_string()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_models_dev_response_deserialization() {
        let json = r#"{
            "providers": [
                {
                    "id": "openai",
                    "name": "OpenAI",
                    "npm": {
                        "name": "@ai-sdk/openai",
                        "version": "1.0.0"
                    },
                    "env": [
                        {
                            "name": "OPENAI_API_KEY",
                            "description": "Your OpenAI API key",
                            "required": true
                        }
                    ],
                    "doc": {
                        "url": "https://platform.openai.com/docs"
                    },
                    "api": {
                        "base_url": "https://api.openai.com/v1",
                        "version": "v1"
                    },
                    "models": [
                        {
                            "id": "gpt-4",
                            "name": "GPT-4",
                            "description": "Large language model",
                            "cost": {
                                "input": 0.03,
                                "output": 0.06,
                                "currency": "USD"
                            },
                            "limits": {
                                "context": 8192,
                                "output": 4096
                            },
                            "modalities": {
                                "input": ["text"],
                                "output": ["text"]
                            }
                        }
                    ]
                }
            ]
        }"#;

        let response: ModelsDevResponse = serde_json::from_str(json).unwrap();
        assert_eq!(response.providers.len(), 1);
        assert_eq!(response.providers[0].id, "openai");
        assert_eq!(response.providers[0].models.len(), 1);
        assert_eq!(response.providers[0].models[0].id, "gpt-4");
    }

    #[test]
    fn test_model_cost_with_optional_fields() {
        let json = r#"{
            "input": 0.01,
            "output": 0.02,
            "cache_read": 0.005,
            "cache_write": 0.01,
            "reasoning": 0.015,
            "currency": "USD"
        }"#;

        let cost: ModelCost = serde_json::from_str(json).unwrap();
        assert_eq!(cost.input, 0.01);
        assert_eq!(cost.output, 0.02);
        assert_eq!(cost.cache_read, Some(0.005));
        assert_eq!(cost.cache_write, Some(0.01));
        assert_eq!(cost.reasoning, Some(0.015));
        assert_eq!(cost.currency, "USD");
    }

    #[test]
    fn test_model_cost_default_currency() {
        let json = r#"{
            "input": 0.01,
            "output": 0.02
        }"#;

        let cost: ModelCost = serde_json::from_str(json).unwrap();
        assert_eq!(cost.currency, "USD");
    }

    #[test]
    fn test_modalities() {
        let json = r#"{
            "input": ["text", "image"],
            "output": ["text"]
        }"#;

        let modalities: Modalities = serde_json::from_str(json).unwrap();
        assert_eq!(modalities.input, vec!["text", "image"]);
        assert_eq!(modalities.output, vec!["text"]);
    }
}
