//! Traits for models.dev integration.
//!
//! This module defines the ModelsDevAware trait and related types for integrating
//! with the models.dev API schema.

use crate::models_dev::types::{Model, Provider};
use std::collections::HashMap;

/// Connection information for a provider that implements ModelsDevAware.
#[derive(Debug, Clone)]
pub struct ProviderConnectionInfo {
    /// The base URL for the provider's API.
    pub base_url: String,

    /// Required environment variables for the provider.
    pub required_env_vars: Vec<String>,

    /// Optional environment variables for the provider.
    pub optional_env_vars: Vec<String>,

    /// Additional configuration for the provider.
    pub config: HashMap<String, String>,
}

impl ProviderConnectionInfo {
    /// Create a new ProviderConnectionInfo.
    pub fn new(base_url: impl Into<String>) -> Self {
        Self {
            base_url: base_url.into(),
            required_env_vars: Vec::new(),
            optional_env_vars: Vec::new(),
            config: HashMap::new(),
        }
    }

    /// Add a required environment variable.
    pub fn with_required_env(mut self, env_var: impl Into<String>) -> Self {
        self.required_env_vars.push(env_var.into());
        self
    }

    /// Add an optional environment variable.
    pub fn with_optional_env(mut self, env_var: impl Into<String>) -> Self {
        self.optional_env_vars.push(env_var.into());
        self
    }

    /// Add a configuration key-value pair.
    pub fn with_config(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.config.insert(key.into(), value.into());
        self
    }

    /// Check if all required environment variables are set.
    pub fn validate_env(&self) -> Result<(), Vec<String>> {
        let missing: Vec<String> = self
            .required_env_vars
            .iter()
            .filter(|var| std::env::var(var).is_err())
            .cloned()
            .collect();

        if missing.is_empty() {
            Ok(())
        } else {
            Err(missing)
        }
    }

    /// Get the value of an environment variable, if set.
    pub fn get_env(&self, var: &str) -> Option<String> {
        std::env::var(var).ok()
    }

    /// Get all environment variables that are set.
    pub fn get_all_env(&self) -> HashMap<String, String> {
        let mut env_vars = HashMap::new();

        for var in &self.required_env_vars {
            if let Ok(value) = std::env::var(var) {
                env_vars.insert(var.clone(), value);
            }
        }

        for var in &self.optional_env_vars {
            if let Ok(value) = std::env::var(var) {
                env_vars.insert(var.clone(), value);
            }
        }

        env_vars
    }
}

/// A trait for types that can be integrated with models.dev provider information.
///
/// This trait allows provider implementations to declare their compatibility with
/// models.dev schema and convert from models.dev provider information.
pub trait ModelsDevAware {
    /// The NPM package names that this provider supports.
    ///
    /// This should return a list of NPM package names that this provider
    /// implementation is compatible with. For example, an OpenAI provider
    /// might return `vec!["@ai-sdk/openai"]`.
    fn supported_npm_packages() -> Vec<String>;

    /// Create an instance of this type from models.dev provider information.
    ///
    /// This method should attempt to convert the models.dev provider information
    /// into an instance of the implementing type. It should return `None` if
    /// the provider information is not compatible with this implementation.
    ///
    /// # Arguments
    /// * `provider` - The provider information from models.dev API
    /// * `model` - Optional specific model information to use
    ///
    /// # Returns
    /// * `Some(Self)` if the provider information is compatible
    /// * `None` if the provider information is not compatible
    fn from_models_dev_info(provider: &Provider, model: Option<&Model>) -> Option<Self>
    where
        Self: Sized;

    /// Get connection information for this provider.
    ///
    /// This method should return connection information derived from the
    /// models.dev provider information, including base URL, required
    /// environment variables, and additional configuration.
    ///
    /// # Arguments
    /// * `provider` - The provider information from models.dev API
    ///
    /// # Returns
    /// * `ProviderConnectionInfo` with connection details
    fn connection_info(provider: &Provider) -> ProviderConnectionInfo {
        let mut info = ProviderConnectionInfo::new(&provider.api.base_url);

        // Add required environment variables
        for env_var in &provider.env {
            if env_var.required {
                info = info.with_required_env(&env_var.name);
            } else {
                info = info.with_optional_env(&env_var.name);
            }
        }

        // Add API version if available
        if let Some(version) = &provider.api.version {
            info = info.with_config("api_version", version);
        }

        // Add additional configuration
        for (key, value) in &provider.api.config {
            if let Some(value_str) = value.as_str() {
                info = info.with_config(key, value_str);
            }
        }

        info
    }

    /// Check if this provider supports a specific model.
    ///
    /// This method should check if the given model is supported by this
    /// provider implementation.
    ///
    /// # Arguments
    /// * `provider` - The provider information from models.dev API
    /// * `model_id` - The ID of the model to check
    ///
    /// # Returns
    /// * `true` if the model is supported
    /// * `false` if the model is not supported
    fn supports_model(provider: &Provider, model_id: &str) -> bool {
        provider.models.iter().any(|m| m.id == model_id)
    }

    /// Get model information for a specific model ID.
    ///
    /// This method should return the model information for the given model ID,
    /// if it exists and is supported by this provider.
    ///
    /// # Arguments
    /// * `provider` - The provider information from models.dev API
    /// * `model_id` - The ID of the model to retrieve
    ///
    /// # Returns
    /// * `Some(&Model)` if the model exists and is supported
    /// * `None` if the model does not exist or is not supported
    fn get_model<'a>(provider: &'a Provider, model_id: &str) -> Option<&'a Model> {
        provider.models.iter().find(|m| m.id == model_id)
    }
}

/// Example implementation for demonstration purposes.
#[cfg(feature = "models-dev")]
pub struct ExampleProvider {
    /// The base URL for the provider's API.
    pub base_url: String,
    /// The API key for authentication.
    pub api_key: String,
    /// The model to use.
    pub model: String,
}

#[cfg(feature = "models-dev")]
impl ModelsDevAware for ExampleProvider {
    fn supported_npm_packages() -> Vec<String> {
        vec!["@ai-sdk/example".to_string()]
    }

    fn from_models_dev_info(provider: &Provider, model: Option<&Model>) -> Option<Self> {
        // Check if this provider supports the NPM package
        if !Self::supported_npm_packages().contains(&provider.npm.name) {
            return None;
        }

        // Get the API key from environment
        let api_key = std::env::var("EXAMPLE_API_KEY").ok()?;

        // Use the provided model or the first available model
        let model_id = model.map_or_else(
            || provider.models.first().map(|m| m.id.clone()),
            |m| Some(m.id.clone()),
        )?;

        Some(Self {
            base_url: provider.api.base_url.clone(),
            api_key,
            model: model_id,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::models_dev::types::{
        ApiInfo, DocInfo, EnvVar, Modalities, ModelCost, ModelLimit, NpmInfo,
    };

    fn create_test_provider() -> Provider {
        Provider {
            id: "test-provider".to_string(),
            name: "Test Provider".to_string(),
            npm: NpmInfo {
                name: "@ai-sdk/test".to_string(),
                version: "1.0.0".to_string(),
            },
            env: vec![
                EnvVar {
                    name: "TEST_API_KEY".to_string(),
                    description: "Test API key".to_string(),
                    required: true,
                },
                EnvVar {
                    name: "TEST_OPTIONAL".to_string(),
                    description: "Optional setting".to_string(),
                    required: false,
                },
            ],
            doc: DocInfo {
                url: "https://test.example.com/docs".to_string(),
                metadata: HashMap::new(),
            },
            api: ApiInfo {
                base_url: "https://api.test.example.com".to_string(),
                version: Some("v1".to_string()),
                config: {
                    let mut config = HashMap::new();
                    config.insert("timeout".to_string(), serde_json::json!("30s"));
                    config
                },
            },
            models: vec![],
        }
    }

    #[test]
    fn test_provider_connection_info_new() {
        let info = ProviderConnectionInfo::new("https://api.example.com");
        assert_eq!(info.base_url, "https://api.example.com");
        assert!(info.required_env_vars.is_empty());
        assert!(info.optional_env_vars.is_empty());
        assert!(info.config.is_empty());
    }

    #[test]
    fn test_provider_connection_info_builder() {
        let info = ProviderConnectionInfo::new("https://api.example.com")
            .with_required_env("API_KEY")
            .with_optional_env("DEBUG")
            .with_config("timeout", "30s");

        assert_eq!(info.base_url, "https://api.example.com");
        assert_eq!(info.required_env_vars, vec!["API_KEY"]);
        assert_eq!(info.optional_env_vars, vec!["DEBUG"]);
        assert_eq!(info.config.get("timeout"), Some(&"30s".to_string()));
    }

    #[test]
    fn test_provider_connection_info_validate_env() {
        let info = ProviderConnectionInfo::new("https://api.example.com")
            .with_required_env("TEST_VAR")
            .with_optional_env("OPTIONAL_VAR");

        // Test with missing required env var
        let result = info.validate_env();
        assert!(result.is_err());
        assert_eq!(result.unwrap_err(), vec!["TEST_VAR"]);

        // Test with all required env vars set
        unsafe {
            std::env::set_var("TEST_VAR", "test_value");
        }
        let result = info.validate_env();
        assert!(result.is_ok());

        // Cleanup
        unsafe {
            std::env::remove_var("TEST_VAR");
        }
    }

    #[test]
    fn test_provider_connection_info_get_env() {
        unsafe {
            std::env::set_var("TEST_VAR", "test_value");
        }

        let info = ProviderConnectionInfo::new("https://api.example.com")
            .with_required_env("TEST_VAR")
            .with_optional_env("MISSING_VAR");

        assert_eq!(info.get_env("TEST_VAR"), Some("test_value".to_string()));
        assert_eq!(info.get_env("MISSING_VAR"), None);

        let all_env = info.get_all_env();
        assert_eq!(all_env.get("TEST_VAR"), Some(&"test_value".to_string()));
        assert!(!all_env.contains_key("MISSING_VAR"));

        // Cleanup
        unsafe {
            std::env::remove_var("TEST_VAR");
        }
    }

    #[test]
    fn test_models_dev_aware_connection_info() {
        let provider = create_test_provider();
        let info = ExampleProvider::connection_info(&provider);

        assert_eq!(info.base_url, "https://api.test.example.com");
        assert_eq!(info.required_env_vars, vec!["TEST_API_KEY"]);
        assert_eq!(info.optional_env_vars, vec!["TEST_OPTIONAL"]);
        assert_eq!(info.config.get("api_version"), Some(&"v1".to_string()));
    }

    #[test]
    fn test_models_dev_aware_supports_model() {
        let mut provider = create_test_provider();
        provider.models.push(Model {
            id: "test-model".to_string(),
            name: "Test Model".to_string(),
            description: "A test model".to_string(),
            cost: ModelCost::default(),
            limits: ModelLimit {
                context: 4096,
                output: 2048,
                metadata: HashMap::new(),
            },
            modalities: Modalities {
                input: vec!["text".to_string()],
                output: vec!["text".to_string()],
            },
            metadata: HashMap::new(),
        });

        assert!(ExampleProvider::supports_model(&provider, "test-model"));
        assert!(!ExampleProvider::supports_model(&provider, "unknown-model"));
    }

    #[test]
    fn test_models_dev_aware_get_model() {
        let mut provider = create_test_provider();
        provider.models.push(Model {
            id: "test-model".to_string(),
            name: "Test Model".to_string(),
            description: "A test model".to_string(),
            cost: ModelCost::default(),
            limits: ModelLimit {
                context: 4096,
                output: 2048,
                metadata: HashMap::new(),
            },
            modalities: Modalities {
                input: vec!["text".to_string()],
                output: vec!["text".to_string()],
            },
            metadata: HashMap::new(),
        });

        let model = ExampleProvider::get_model(&provider, "test-model");
        assert!(model.is_some());
        assert_eq!(model.unwrap().id, "test-model");

        let missing_model = ExampleProvider::get_model(&provider, "unknown-model");
        assert!(missing_model.is_none());
    }
}
