//! Provider registry for managing models.dev API data.
//!
//! This module provides a centralized registry for managing provider and model information
//! fetched from the models.dev API. It handles data transformation, caching, and provides
//! convenient lookup methods for discovering available providers and models.

use std::collections::HashMap;
use std::sync::Arc;

use tokio::sync::RwLock;

use crate::models_dev::{
    client::ModelsDevClient,
    error::ModelsDevError,
    traits::{ModelsDevAware, ProviderConnectionInfo},
    types::{Model, ModelsDevResponse, Provider as ApiProvider},
};

/// Internal provider information with transformed data.
#[derive(Debug, Clone)]
pub struct ProviderInfo {
    /// The unique identifier for the provider.
    pub id: String,
    /// The display name of the provider.
    pub name: String,
    /// The base URL for the provider's API.
    pub base_url: String,
    /// NPM package information for the provider.
    pub npm_name: String,
    pub npm_version: String,
    /// Environment variables required for the provider.
    pub env_vars: Vec<EnvVarInfo>,
    /// Documentation URL.
    pub doc_url: String,
    /// API version information.
    pub api_version: Option<String>,
    /// Whether this provider is currently available.
    pub available: bool,
    /// List of model IDs available from this provider.
    pub model_ids: Vec<String>,
}

/// Environment variable information.
#[derive(Debug, Clone)]
pub struct EnvVarInfo {
    /// The name of the environment variable.
    pub name: String,
    /// Description of what the environment variable is used for.
    pub description: String,
    /// Whether this environment variable is required.
    pub required: bool,
}

/// Internal model information with provider reference.
#[derive(Debug, Clone)]
pub struct ModelInfo {
    /// The unique identifier for the model.
    pub id: String,
    /// The display name of the model.
    pub name: String,
    /// A description of the model.
    pub description: String,
    /// The ID of the provider that offers this model.
    pub provider_id: String,
    /// Cost information for using this model.
    pub cost: ModelCostInfo,
    /// Limits for this model.
    pub limits: ModelLimitInfo,
    /// Supported input modalities.
    pub input_modalities: Vec<String>,
    /// Supported output modalities.
    pub output_modalities: Vec<String>,
    /// Additional model metadata.
    pub metadata: HashMap<String, serde_json::Value>,
}

/// Cost information for a model.
#[derive(Debug, Clone)]
pub struct ModelCostInfo {
    /// Cost per 1M input tokens.
    pub input: f64,
    /// Cost per 1M output tokens.
    pub output: f64,
    /// Cost per 1M cache read tokens (if supported).
    pub cache_read: Option<f64>,
    /// Cost per 1M cache write tokens (if supported).
    pub cache_write: Option<f64>,
    /// Cost per 1M reasoning tokens (if supported).
    pub reasoning: Option<f64>,
    /// Currency for the costs.
    pub currency: String,
}

/// Limits for a model.
#[derive(Debug, Clone)]
pub struct ModelLimitInfo {
    /// Maximum context window size in tokens.
    pub context: u32,
    /// Maximum output tokens per request.
    pub output: u32,
    /// Additional limits.
    pub metadata: HashMap<String, serde_json::Value>,
}

/// Registry for managing providers and models from the models.dev API.
#[derive(Debug, Clone)]
pub struct ProviderRegistry {
    /// The models.dev client for fetching data.
    pub client: ModelsDevClient,
    /// Cached provider information indexed by provider ID.
    pub providers: Arc<RwLock<HashMap<String, ProviderInfo>>>,
    /// Cached model information indexed by model ID.
    pub models: Arc<RwLock<HashMap<String, ModelInfo>>>,
    /// Mapping from model IDs to provider IDs for quick lookup.
    pub model_to_provider: Arc<RwLock<HashMap<String, String>>>,
}

impl ProviderRegistry {
    /// Create a new ProviderRegistry with the given models.dev client.
    pub fn new(client: ModelsDevClient) -> Self {
        Self {
            client,
            providers: Arc::new(RwLock::new(HashMap::new())),
            models: Arc::new(RwLock::new(HashMap::new())),
            model_to_provider: Arc::new(RwLock::new(HashMap::new())),
        }
    }

    /// Create a new ProviderRegistry with a default models.dev client.
    pub fn with_default_client() -> Self {
        Self::new(ModelsDevClient::new())
    }

    /// Refresh the registry data by fetching from the models.dev API.
    ///
    /// This method fetches the latest provider and model information from the API
    /// and transforms it into the internal registry format.
    ///
    /// Returns the number of providers and models loaded, or an error if the fetch fails.
    pub async fn refresh(&self) -> Result<(usize, usize), ModelsDevError> {
        // Fetch the full API response
        let api_response = self.fetch_api_response().await?;

        // Transform the data into internal format
        let (providers, models, model_to_provider) = self.transform_api_data(api_response).await?;

        // Update the registry with the new data
        {
            let mut providers_lock = self.providers.write().await;
            *providers_lock = providers;
        }

        {
            let mut models_lock = self.models.write().await;
            *models_lock = models;
        }

        {
            let mut mapping_lock = self.model_to_provider.write().await;
            *mapping_lock = model_to_provider;
        }

        let provider_count = self.providers.read().await.len();
        let model_count = self.models.read().await.len();

        Ok((provider_count, model_count))
    }

    /// Fetch the full API response from models.dev.
    async fn fetch_api_response(&self) -> Result<ModelsDevResponse, ModelsDevError> {
        // Note: The current client only fetches providers, but we need the full response
        // For now, we'll work with what we have and enhance the client later
        // This is a temporary implementation that will be updated when the client supports full API responses

        // For now, create a minimal response structure
        // This will be replaced with actual API calls when the client is enhanced
        let providers = self.client.fetch_providers().await?;

        // Convert the simplified provider structure to the full API structure
        // This is a temporary transformation until the client fetches full API data
        let api_providers = providers
            .into_iter()
            .map(|p| ApiProvider {
                id: p.id.clone(),
                name: p.name.clone(),
                npm: crate::models_dev::types::NpmInfo {
                    name: format!("@ai-sdk/{}", p.id),
                    version: "1.0.0".to_string(),
                },
                env: vec![],
                doc: crate::models_dev::types::DocInfo {
                    url: format!("{}/docs", p.base_url),
                    metadata: HashMap::new(),
                },
                api: crate::models_dev::types::ApiInfo {
                    base_url: p.base_url.clone(),
                    version: None,
                    config: HashMap::new(),
                },
                models: vec![], // Empty for now, will be populated when client supports full API
            })
            .collect();

        Ok(ModelsDevResponse {
            providers: api_providers,
        })
    }

    /// Transform API response data into internal registry format.
    pub async fn transform_api_data(
        &self,
        api_response: ModelsDevResponse,
    ) -> Result<
        (
            HashMap<String, ProviderInfo>,
            HashMap<String, ModelInfo>,
            HashMap<String, String>,
        ),
        ModelsDevError,
    > {
        let mut providers = HashMap::new();
        let mut models = HashMap::new();
        let mut model_to_provider = HashMap::new();

        for api_provider in api_response.providers {
            // Transform provider data
            let provider_info = self.transform_provider_data(&api_provider).await?;

            // Transform model data
            let mut provider_model_ids = Vec::new();
            for api_model in &api_provider.models {
                let model_info = self
                    .transform_model_data(api_model, &api_provider.id)
                    .await?;
                provider_model_ids.push(model_info.id.clone());
                models.insert(model_info.id.clone(), model_info);
                model_to_provider.insert(api_model.id.clone(), api_provider.id.clone());
            }

            // Update provider with model IDs
            let mut provider_info_with_models = provider_info;
            provider_info_with_models.model_ids = provider_model_ids;

            providers.insert(api_provider.id.clone(), provider_info_with_models);
        }

        Ok((providers, models, model_to_provider))
    }

    /// Transform API provider data into internal ProviderInfo format.
    pub async fn transform_provider_data(
        &self,
        api_provider: &ApiProvider,
    ) -> Result<ProviderInfo, ModelsDevError> {
        let env_vars = api_provider
            .env
            .iter()
            .map(|env| EnvVarInfo {
                name: env.name.clone(),
                description: env.description.clone(),
                required: env.required,
            })
            .collect();

        Ok(ProviderInfo {
            id: api_provider.id.clone(),
            name: api_provider.name.clone(),
            base_url: api_provider.api.base_url.clone(),
            npm_name: api_provider.npm.name.clone(),
            npm_version: api_provider.npm.version.clone(),
            env_vars,
            doc_url: api_provider.doc.url.clone(),
            api_version: api_provider.api.version.clone(),
            available: true,       // Assume available unless API indicates otherwise
            model_ids: Vec::new(), // Will be populated by the caller
        })
    }

    /// Transform API model data into internal ModelInfo format.
    pub async fn transform_model_data(
        &self,
        api_model: &Model,
        provider_id: &str,
    ) -> Result<ModelInfo, ModelsDevError> {
        let cost = ModelCostInfo {
            input: api_model.cost.input,
            output: api_model.cost.output,
            cache_read: api_model.cost.cache_read,
            cache_write: api_model.cost.cache_write,
            reasoning: api_model.cost.reasoning,
            currency: api_model.cost.currency.clone(),
        };

        let limits = ModelLimitInfo {
            context: api_model.limits.context,
            output: api_model.limits.output,
            metadata: api_model.limits.metadata.clone(),
        };

        Ok(ModelInfo {
            id: api_model.id.clone(),
            name: api_model.name.clone(),
            description: api_model.description.clone(),
            provider_id: provider_id.to_string(),
            cost,
            limits,
            input_modalities: api_model.modalities.input.clone(),
            output_modalities: api_model.modalities.output.clone(),
            metadata: api_model.metadata.clone(),
        })
    }

    /// Get information about a specific provider.
    ///
    /// Returns None if the provider is not found in the registry.
    pub async fn get_provider(&self, provider_id: &str) -> Option<ProviderInfo> {
        let providers = self.providers.read().await;
        providers.get(provider_id).cloned()
    }

    /// Get information about a specific model.
    ///
    /// Returns None if the model is not found in the registry.
    pub async fn get_model(&self, model_id: &str) -> Option<ModelInfo> {
        let models = self.models.read().await;
        models.get(model_id).cloned()
    }

    /// Get all available providers.
    ///
    /// Returns a vector of all provider information in the registry.
    pub async fn get_all_providers(&self) -> Vec<ProviderInfo> {
        let providers = self.providers.read().await;
        providers.values().cloned().collect()
    }

    /// Get all available models.
    ///
    /// Returns a vector of all model information in the registry.
    pub async fn get_all_models(&self) -> Vec<ModelInfo> {
        let models = self.models.read().await;
        models.values().cloned().collect()
    }

    /// Get all models for a specific provider.
    ///
    /// Returns a vector of model information for the given provider,
    /// or an empty vector if the provider is not found.
    pub async fn get_models_for_provider(&self, provider_id: &str) -> Vec<ModelInfo> {
        let models = self.models.read().await;
        models
            .values()
            .filter(|model| model.provider_id == provider_id)
            .cloned()
            .collect()
    }

    /// Find the provider that offers a specific model.
    ///
    /// This method uses heuristics to map models to providers:
    /// 1. Direct lookup in the model-to-provider mapping
    /// 2. Pattern matching for common model naming conventions
    /// 3. Fallback to provider ID extraction from model ID
    ///
    /// Returns the provider ID if found, None otherwise.
    pub async fn find_provider_for_model(&self, model_id: &str) -> Option<String> {
        // First try direct lookup
        {
            let mapping = self.model_to_provider.read().await;
            if let Some(provider_id) = mapping.get(model_id) {
                return Some(provider_id.clone());
            }
        }

        // If not found, try heuristics based on model ID patterns
        self.apply_model_heuristics(model_id).await
    }

    /// Apply heuristics to determine provider from model ID.
    pub async fn apply_model_heuristics(&self, model_id: &str) -> Option<String> {
        let model_id_lower = model_id.to_lowercase();
        let providers = self.providers.read().await;

        // Common model ID patterns
        let patterns = vec![
            // OpenAI patterns
            ("gpt-", "openai"),
            ("text-davinci-", "openai"),
            ("text-curie-", "openai"),
            ("text-babbage-", "openai"),
            ("text-ada-", "openai"),
            // Anthropic patterns
            ("claude-", "anthropic"),
            // Google patterns
            ("gemini-", "google"),
            ("text-bison-", "google"),
            ("chat-bison-", "google"),
            // Meta patterns
            ("llama-", "meta"),
            // Mistral patterns
            ("mistral-", "mistral"),
            ("mixtral-", "mistral"),
            // Cohere patterns
            ("command-", "cohere"),
            ("embed-", "cohere"),
            // Azure patterns
            ("azure-", "azure"),
        ];

        // Check for known patterns
        for (pattern, provider_id) in patterns {
            if model_id_lower.starts_with(pattern) && providers.contains_key(provider_id) {
                return Some(provider_id.to_string());
            }
        }

        // Try to extract provider from model ID (e.g., "openai/gpt-4" -> "openai")
        if let Some((provider_part, _)) = model_id.split_once('/')
            && providers.contains_key(provider_part)
        {
            return Some(provider_part.to_string());
        }

        // Try to match by checking if any provider ID is a prefix of the model ID
        for provider_id in providers.keys() {
            if model_id_lower.starts_with(&provider_id.to_lowercase()) {
                return Some(provider_id.clone());
            }
        }

        None
    }

    /// Check if a provider is available.
    ///
    /// Returns true if the provider exists and is marked as available.
    pub async fn is_provider_available(&self, provider_id: &str) -> bool {
        let providers = self.providers.read().await;
        providers
            .get(provider_id)
            .map(|provider| provider.available)
            .unwrap_or(false)
    }

    /// Check if a model is available.
    ///
    /// Returns true if the model exists in the registry.
    pub async fn is_model_available(&self, model_id: &str) -> bool {
        let models = self.models.read().await;
        models.contains_key(model_id)
    }

    /// Get the number of providers in the registry.
    pub async fn provider_count(&self) -> usize {
        let providers = self.providers.read().await;
        providers.len()
    }

    /// Get the number of models in the registry.
    pub async fn model_count(&self) -> usize {
        let models = self.models.read().await;
        models.len()
    }

    /// Clear all cached data from the registry.
    pub async fn clear(&self) {
        {
            let mut providers = self.providers.write().await;
            providers.clear();
        }
        {
            let mut models = self.models.write().await;
            models.clear();
        }
        {
            let mut mapping = self.model_to_provider.write().await;
            mapping.clear();
        }
    }

    /// Find providers that support a specific npm package.
    ///
    /// This method searches for providers that have the given npm package name
    /// in their npm information.
    ///
    /// # Arguments
    /// * `npm_package` - The npm package name to search for (e.g., "@ai-sdk/openai")
    ///
    /// # Returns
    /// * A vector of provider IDs (String) for providers that support the npm package
    pub async fn find_providers_by_npm(&self, npm_package: &str) -> Vec<String> {
        let providers = self.providers.read().await;
        providers
            .values()
            .filter(|provider| provider.npm_name == npm_package)
            .map(|provider| provider.id.clone())
            .collect()
    }

    /// Get models with specific capabilities.
    ///
    /// This method filters models based on their capabilities such as reasoning,
    /// tool calling, attachment support, or vision capabilities.
    ///
    /// # Arguments
    /// * `capability` - The capability to filter by ("reasoning", "tool_call", "attachment", "vision")
    ///
    /// # Returns
    /// * A vector of ModelInfo that have the specified capability
    pub async fn get_models_with_capability(&self, capability: &str) -> Vec<ModelInfo> {
        let models = self.models.read().await;
        models
            .values()
            .filter(|model| {
                match capability {
                    // Check for reasoning capability (cost has reasoning field)
                    "reasoning" => model.cost.reasoning.is_some(),
                    // Check for tool calling capability (metadata indicates tool support)
                    "tool_call" => model
                        .metadata
                        .get("supports_tools")
                        .and_then(|v| v.as_bool())
                        .unwrap_or(false),
                    // Check for attachment support (input modalities include file/document types)
                    "attachment" => model.input_modalities.iter().any(|modality| {
                        modality.contains("file")
                            || modality.contains("document")
                            || modality.contains("attachment")
                    }),
                    // Check for vision capability (input modalities include image)
                    "vision" => model.input_modalities.contains(&"image".to_string()),
                    // Unknown capability
                    _ => false,
                }
            })
            .cloned()
            .collect()
    }

    /// Get providers grouped with their models.
    ///
    /// This method returns a vector of tuples where each tuple contains
    /// a provider ID, a ProviderInfo, and a vector of ModelInfo for that provider.
    ///
    /// # Returns
    /// * A vector of (String, ProviderInfo, Vec<ModelInfo>) tuples
    pub async fn get_providers_with_models(&self) -> Vec<(String, ProviderInfo, Vec<ModelInfo>)> {
        let providers = self.providers.read().await;
        let models = self.models.read().await;

        providers
            .values()
            .map(|provider| {
                let provider_models: Vec<ModelInfo> = models
                    .values()
                    .filter(|model| model.provider_id == provider.id)
                    .cloned()
                    .collect();
                (provider.id.clone(), provider.clone(), provider_models)
            })
            .collect()
    }

    /// Get ProviderConnectionInfo for a provider.
    ///
    /// This method creates a ProviderConnectionInfo struct from the provider's
    /// configuration, which can be used to create provider instances.
    ///
    /// # Arguments
    /// * `provider_id` - The ID of the provider to get connection info for
    ///
    /// # Returns
    /// * Some(ProviderConnectionInfo) if the provider exists
    /// * None if the provider doesn't exist
    pub async fn get_connection_info(&self, provider_id: &str) -> Option<ProviderConnectionInfo> {
        let providers = self.providers.read().await;
        let provider = providers.get(provider_id)?;

        let mut connection_info = ProviderConnectionInfo::new(&provider.base_url);

        // Add required environment variables
        for env_var in &provider.env_vars {
            if env_var.required {
                connection_info = connection_info.with_required_env(&env_var.name);
            } else {
                connection_info = connection_info.with_optional_env(&env_var.name);
            }
        }

        // Add API version if available
        if let Some(version) = &provider.api_version {
            connection_info = connection_info.with_config("api_version", version);
        }

        Some(connection_info)
    }

    /// Create a provider instance using the ModelsDevAware trait.
    ///
    /// This method attempts to create a provider instance of type T that implements
    /// the ModelsDevAware trait. It uses the provider information from the registry
    /// to create the instance.
    ///
    /// # Type Parameters
    /// * `T` - A type that implements ModelsDevAware
    ///
    /// # Arguments
    /// * `provider_id` - The ID of the provider to create
    ///
    /// # Returns
    /// * Ok(T) if the provider was successfully created
    /// * Err(ModelsDevError) if the provider couldn't be created
    pub async fn create_provider<T: ModelsDevAware>(
        &self,
        provider_id: &str,
    ) -> Result<T, ModelsDevError> {
        let providers = self.providers.read().await;
        let models = self.models.read().await;

        // Get the provider info
        let provider_info = providers.get(provider_id).ok_or_else(|| {
            ModelsDevError::ProviderNotFound(format!("Provider '{}' not found", provider_id))
        })?;

        // Use the first available model for the provider
        let model_info = models
            .values()
            .find(|model| model.provider_id == provider_id)
            .ok_or_else(|| {
                ModelsDevError::NoModelsAvailable(format!(
                    "No models available for provider '{}'",
                    provider_id
                ))
            })?;

        // Convert the internal model info to API model format
        let api_model = Model {
            id: model_info.id.clone(),
            name: model_info.name.clone(),
            description: model_info.description.clone(),
            cost: crate::models_dev::types::ModelCost {
                input: model_info.cost.input,
                output: model_info.cost.output,
                cache_read: model_info.cost.cache_read,
                cache_write: model_info.cost.cache_write,
                reasoning: model_info.cost.reasoning,
                currency: model_info.cost.currency.clone(),
            },
            limits: crate::models_dev::types::ModelLimit {
                context: model_info.limits.context,
                output: model_info.limits.output,
                metadata: model_info.limits.metadata.clone(),
            },
            modalities: crate::models_dev::types::Modalities {
                input: model_info.input_modalities.clone(),
                output: model_info.output_modalities.clone(),
            },
            metadata: model_info.metadata.clone(),
        };

        // Clone the model for the provider models vector
        let api_model_clone = api_model.clone();

        // Convert the internal provider info to API provider format
        let api_provider = ApiProvider {
            id: provider_info.id.clone(),
            name: provider_info.name.clone(),
            npm: crate::models_dev::types::NpmInfo {
                name: provider_info.npm_name.clone(),
                version: provider_info.npm_version.clone(),
            },
            env: provider_info
                .env_vars
                .iter()
                .map(|env_var| crate::models_dev::types::EnvVar {
                    name: env_var.name.clone(),
                    description: env_var.description.clone(),
                    required: env_var.required,
                })
                .collect(),
            doc: crate::models_dev::types::DocInfo {
                url: provider_info.doc_url.clone(),
                metadata: std::collections::HashMap::new(),
            },
            api: crate::models_dev::types::ApiInfo {
                base_url: provider_info.base_url.clone(),
                version: provider_info.api_version.clone(),
                config: std::collections::HashMap::new(),
            },
            models: vec![api_model_clone], // Include the selected model
        };

        // Try to create the provider using the ModelsDevAware trait
        T::from_models_dev_info(&api_provider, Some(&api_model)).ok_or_else(|| {
            ModelsDevError::UnsupportedProvider(format!(
                "Provider '{}' is not supported by type {}",
                provider_id,
                std::any::type_name::<T>()
            ))
        })
    }
}

impl Default for ProviderRegistry {
    fn default() -> Self {
        Self::with_default_client()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_registry_creation() {
        let client = ModelsDevClient::new();
        let registry = ProviderRegistry::new(client);

        assert_eq!(registry.provider_count().await, 0);
        assert_eq!(registry.model_count().await, 0);
    }

    #[tokio::test]
    async fn test_registry_default() {
        let registry = ProviderRegistry::default();

        assert_eq!(registry.provider_count().await, 0);
        assert_eq!(registry.model_count().await, 0);
    }

    #[tokio::test]
    async fn test_clear_registry() {
        let registry = ProviderRegistry::default();

        // Add some dummy data to test clearing
        {
            let mut providers = registry.providers.write().await;
            providers.insert(
                "test".to_string(),
                ProviderInfo {
                    id: "test".to_string(),
                    name: "Test".to_string(),
                    base_url: "https://test.com".to_string(),
                    npm_name: "test".to_string(),
                    npm_version: "1.0.0".to_string(),
                    env_vars: vec![],
                    doc_url: "https://test.com/docs".to_string(),
                    api_version: None,
                    available: true,
                    model_ids: vec![],
                },
            );
        }

        assert_eq!(registry.provider_count().await, 1);

        registry.clear().await;
        assert_eq!(registry.provider_count().await, 0);
        assert_eq!(registry.model_count().await, 0);
    }

    #[tokio::test]
    async fn test_model_heuristics() {
        let registry = ProviderRegistry::default();

        // Add some test providers
        {
            let mut providers = registry.providers.write().await;
            providers.insert(
                "openai".to_string(),
                ProviderInfo {
                    id: "openai".to_string(),
                    name: "OpenAI".to_string(),
                    base_url: "https://api.openai.com".to_string(),
                    npm_name: "@ai-sdk/openai".to_string(),
                    npm_version: "1.0.0".to_string(),
                    env_vars: vec![],
                    doc_url: "https://platform.openai.com/docs".to_string(),
                    api_version: None,
                    available: true,
                    model_ids: vec![],
                },
            );

            providers.insert(
                "anthropic".to_string(),
                ProviderInfo {
                    id: "anthropic".to_string(),
                    name: "Anthropic".to_string(),
                    base_url: "https://api.anthropic.com".to_string(),
                    npm_name: "@ai-sdk/anthropic".to_string(),
                    npm_version: "1.0.0".to_string(),
                    env_vars: vec![],
                    doc_url: "https://docs.anthropic.com".to_string(),
                    api_version: None,
                    available: true,
                    model_ids: vec![],
                },
            );
        }

        // Test OpenAI patterns
        assert_eq!(
            registry.apply_model_heuristics("gpt-4").await,
            Some("openai".to_string())
        );
        assert_eq!(
            registry.apply_model_heuristics("text-davinci-003").await,
            Some("openai".to_string())
        );

        // Test Anthropic patterns
        assert_eq!(
            registry.apply_model_heuristics("claude-3-opus").await,
            Some("anthropic".to_string())
        );

        // Test slash notation
        assert_eq!(
            registry.apply_model_heuristics("openai/gpt-4").await,
            Some("openai".to_string())
        );

        // Test unknown model
        assert_eq!(registry.apply_model_heuristics("unknown-model").await, None);
    }

    #[tokio::test]
    async fn test_find_providers_by_npm() {
        let registry = ProviderRegistry::default();

        // Add test providers with different npm packages
        {
            let mut providers = registry.providers.write().await;
            providers.insert(
                "openai".to_string(),
                ProviderInfo {
                    id: "openai".to_string(),
                    name: "OpenAI".to_string(),
                    base_url: "https://api.openai.com".to_string(),
                    npm_name: "@ai-sdk/openai".to_string(),
                    npm_version: "1.0.0".to_string(),
                    env_vars: vec![],
                    doc_url: "https://platform.openai.com/docs".to_string(),
                    api_version: None,
                    available: true,
                    model_ids: vec![],
                },
            );

            providers.insert(
                "anthropic".to_string(),
                ProviderInfo {
                    id: "anthropic".to_string(),
                    name: "Anthropic".to_string(),
                    base_url: "https://api.anthropic.com".to_string(),
                    npm_name: "@ai-sdk/anthropic".to_string(),
                    npm_version: "1.0.0".to_string(),
                    env_vars: vec![],
                    doc_url: "https://docs.anthropic.com".to_string(),
                    api_version: None,
                    available: true,
                    model_ids: vec![],
                },
            );

            providers.insert(
                "another-openai".to_string(),
                ProviderInfo {
                    id: "another-openai".to_string(),
                    name: "Another OpenAI".to_string(),
                    base_url: "https://another.openai.com".to_string(),
                    npm_name: "@ai-sdk/openai".to_string(), // Same npm package
                    npm_version: "1.0.0".to_string(),
                    env_vars: vec![],
                    doc_url: "https://another.openai.com/docs".to_string(),
                    api_version: None,
                    available: true,
                    model_ids: vec![],
                },
            );
        }

        // Test finding providers by npm package
        let openai_providers = registry.find_providers_by_npm("@ai-sdk/openai").await;
        assert_eq!(openai_providers.len(), 2);
        assert!(openai_providers.contains(&"openai".to_string()));
        assert!(openai_providers.contains(&"another-openai".to_string()));

        let anthropic_providers = registry.find_providers_by_npm("@ai-sdk/anthropic").await;
        assert_eq!(anthropic_providers.len(), 1);
        assert!(anthropic_providers.contains(&"anthropic".to_string()));

        // Test non-existent npm package
        let non_existent = registry.find_providers_by_npm("@ai-sdk/non-existent").await;
        assert_eq!(non_existent.len(), 0);
    }

    #[tokio::test]
    async fn test_get_models_with_capability() {
        let registry = ProviderRegistry::default();

        // Add test models with different capabilities
        {
            let mut models = registry.models.write().await;

            // Model with reasoning capability
            models.insert(
                "reasoning-model".to_string(),
                ModelInfo {
                    id: "reasoning-model".to_string(),
                    name: "Reasoning Model".to_string(),
                    description: "A model with reasoning capability".to_string(),
                    provider_id: "test-provider".to_string(),
                    cost: ModelCostInfo {
                        input: 0.01,
                        output: 0.02,
                        cache_read: None,
                        cache_write: None,
                        reasoning: Some(0.03), // Has reasoning capability
                        currency: "USD".to_string(),
                    },
                    limits: ModelLimitInfo {
                        context: 4096,
                        output: 2048,
                        metadata: HashMap::new(),
                    },
                    input_modalities: vec!["text".to_string()],
                    output_modalities: vec!["text".to_string()],
                    metadata: HashMap::new(),
                },
            );

            // Model with tool calling capability
            let mut tool_call_metadata = HashMap::new();
            tool_call_metadata.insert("supports_tools".to_string(), serde_json::Value::Bool(true));
            models.insert(
                "tool-call-model".to_string(),
                ModelInfo {
                    id: "tool-call-model".to_string(),
                    name: "Tool Call Model".to_string(),
                    description: "A model with tool calling capability".to_string(),
                    provider_id: "test-provider".to_string(),
                    cost: ModelCostInfo {
                        input: 0.01,
                        output: 0.02,
                        cache_read: None,
                        cache_write: None,
                        reasoning: None,
                        currency: "USD".to_string(),
                    },
                    limits: ModelLimitInfo {
                        context: 4096,
                        output: 2048,
                        metadata: HashMap::new(),
                    },
                    input_modalities: vec!["text".to_string()],
                    output_modalities: vec!["text".to_string()],
                    metadata: tool_call_metadata,
                },
            );

            // Model with attachment capability
            models.insert(
                "attachment-model".to_string(),
                ModelInfo {
                    id: "attachment-model".to_string(),
                    name: "Attachment Model".to_string(),
                    description: "A model with attachment capability".to_string(),
                    provider_id: "test-provider".to_string(),
                    cost: ModelCostInfo {
                        input: 0.01,
                        output: 0.02,
                        cache_read: None,
                        cache_write: None,
                        reasoning: None,
                        currency: "USD".to_string(),
                    },
                    limits: ModelLimitInfo {
                        context: 4096,
                        output: 2048,
                        metadata: HashMap::new(),
                    },
                    input_modalities: vec![
                        "text".to_string(),
                        "file".to_string(),
                        "document".to_string(),
                    ],
                    output_modalities: vec!["text".to_string()],
                    metadata: HashMap::new(),
                },
            );

            // Model with vision capability
            models.insert(
                "vision-model".to_string(),
                ModelInfo {
                    id: "vision-model".to_string(),
                    name: "Vision Model".to_string(),
                    description: "A model with vision capability".to_string(),
                    provider_id: "test-provider".to_string(),
                    cost: ModelCostInfo {
                        input: 0.01,
                        output: 0.02,
                        cache_read: None,
                        cache_write: None,
                        reasoning: None,
                        currency: "USD".to_string(),
                    },
                    limits: ModelLimitInfo {
                        context: 4096,
                        output: 2048,
                        metadata: HashMap::new(),
                    },
                    input_modalities: vec!["text".to_string(), "image".to_string()],
                    output_modalities: vec!["text".to_string()],
                    metadata: HashMap::new(),
                },
            );

            // Model without special capabilities
            models.insert(
                "basic-model".to_string(),
                ModelInfo {
                    id: "basic-model".to_string(),
                    name: "Basic Model".to_string(),
                    description: "A basic model without special capabilities".to_string(),
                    provider_id: "test-provider".to_string(),
                    cost: ModelCostInfo {
                        input: 0.01,
                        output: 0.02,
                        cache_read: None,
                        cache_write: None,
                        reasoning: None,
                        currency: "USD".to_string(),
                    },
                    limits: ModelLimitInfo {
                        context: 4096,
                        output: 2048,
                        metadata: HashMap::new(),
                    },
                    input_modalities: vec!["text".to_string()],
                    output_modalities: vec!["text".to_string()],
                    metadata: HashMap::new(),
                },
            );
        }

        // Test reasoning capability
        let reasoning_models = registry.get_models_with_capability("reasoning").await;
        assert_eq!(reasoning_models.len(), 1);
        assert_eq!(reasoning_models[0].id, "reasoning-model");

        // Test tool calling capability
        let tool_call_models = registry.get_models_with_capability("tool_call").await;
        assert_eq!(tool_call_models.len(), 1);
        assert_eq!(tool_call_models[0].id, "tool-call-model");

        // Test attachment capability
        let attachment_models = registry.get_models_with_capability("attachment").await;
        assert_eq!(attachment_models.len(), 1);
        assert_eq!(attachment_models[0].id, "attachment-model");

        // Test vision capability
        let vision_models = registry.get_models_with_capability("vision").await;
        assert_eq!(vision_models.len(), 1);
        assert_eq!(vision_models[0].id, "vision-model");

        // Test unknown capability
        let unknown_models = registry.get_models_with_capability("unknown").await;
        assert_eq!(unknown_models.len(), 0);
    }

    #[tokio::test]
    async fn test_get_providers_with_models() {
        let registry = ProviderRegistry::default();

        // Add test providers and models
        {
            let mut providers = registry.providers.write().await;
            let mut models = registry.models.write().await;

            // Add OpenAI provider
            providers.insert(
                "openai".to_string(),
                ProviderInfo {
                    id: "openai".to_string(),
                    name: "OpenAI".to_string(),
                    base_url: "https://api.openai.com".to_string(),
                    npm_name: "@ai-sdk/openai".to_string(),
                    npm_version: "1.0.0".to_string(),
                    env_vars: vec![],
                    doc_url: "https://platform.openai.com/docs".to_string(),
                    api_version: None,
                    available: true,
                    model_ids: vec!["gpt-4".to_string(), "gpt-3.5-turbo".to_string()],
                },
            );

            // Add Anthropic provider
            providers.insert(
                "anthropic".to_string(),
                ProviderInfo {
                    id: "anthropic".to_string(),
                    name: "Anthropic".to_string(),
                    base_url: "https://api.anthropic.com".to_string(),
                    npm_name: "@ai-sdk/anthropic".to_string(),
                    npm_version: "1.0.0".to_string(),
                    env_vars: vec![],
                    doc_url: "https://docs.anthropic.com".to_string(),
                    api_version: None,
                    available: true,
                    model_ids: vec!["claude-3-opus".to_string()],
                },
            );

            // Add models for OpenAI
            models.insert(
                "gpt-4".to_string(),
                ModelInfo {
                    id: "gpt-4".to_string(),
                    name: "GPT-4".to_string(),
                    description: "GPT-4 model".to_string(),
                    provider_id: "openai".to_string(),
                    cost: ModelCostInfo {
                        input: 0.03,
                        output: 0.06,
                        cache_read: None,
                        cache_write: None,
                        reasoning: None,
                        currency: "USD".to_string(),
                    },
                    limits: ModelLimitInfo {
                        context: 8192,
                        output: 4096,
                        metadata: HashMap::new(),
                    },
                    input_modalities: vec!["text".to_string()],
                    output_modalities: vec!["text".to_string()],
                    metadata: HashMap::new(),
                },
            );

            models.insert(
                "gpt-3.5-turbo".to_string(),
                ModelInfo {
                    id: "gpt-3.5-turbo".to_string(),
                    name: "GPT-3.5 Turbo".to_string(),
                    description: "GPT-3.5 Turbo model".to_string(),
                    provider_id: "openai".to_string(),
                    cost: ModelCostInfo {
                        input: 0.0015,
                        output: 0.002,
                        cache_read: None,
                        cache_write: None,
                        reasoning: None,
                        currency: "USD".to_string(),
                    },
                    limits: ModelLimitInfo {
                        context: 4096,
                        output: 2048,
                        metadata: HashMap::new(),
                    },
                    input_modalities: vec!["text".to_string()],
                    output_modalities: vec!["text".to_string()],
                    metadata: HashMap::new(),
                },
            );

            // Add model for Anthropic
            models.insert(
                "claude-3-opus".to_string(),
                ModelInfo {
                    id: "claude-3-opus".to_string(),
                    name: "Claude 3 Opus".to_string(),
                    description: "Claude 3 Opus model".to_string(),
                    provider_id: "anthropic".to_string(),
                    cost: ModelCostInfo {
                        input: 0.015,
                        output: 0.075,
                        cache_read: None,
                        cache_write: None,
                        reasoning: None,
                        currency: "USD".to_string(),
                    },
                    limits: ModelLimitInfo {
                        context: 200000,
                        output: 4096,
                        metadata: HashMap::new(),
                    },
                    input_modalities: vec!["text".to_string()],
                    output_modalities: vec!["text".to_string()],
                    metadata: HashMap::new(),
                },
            );
        }

        // Test getting providers with models
        let providers_with_models = registry.get_providers_with_models().await;
        assert_eq!(providers_with_models.len(), 2);

        // Find OpenAI provider
        let openai_data = providers_with_models
            .iter()
            .find(|(id, _, _)| id == "openai")
            .unwrap();
        assert_eq!(openai_data.0, "openai");
        assert_eq!(openai_data.1.id, "openai");
        assert_eq!(openai_data.1.name, "OpenAI");
        assert_eq!(openai_data.2.len(), 2);
        assert!(openai_data.2.iter().any(|m| m.id == "gpt-4"));
        assert!(openai_data.2.iter().any(|m| m.id == "gpt-3.5-turbo"));

        // Find Anthropic provider
        let anthropic_data = providers_with_models
            .iter()
            .find(|(id, _, _)| id == "anthropic")
            .unwrap();
        assert_eq!(anthropic_data.0, "anthropic");
        assert_eq!(anthropic_data.1.id, "anthropic");
        assert_eq!(anthropic_data.1.name, "Anthropic");
        assert_eq!(anthropic_data.2.len(), 1);
        assert_eq!(anthropic_data.2[0].id, "claude-3-opus");
    }

    #[tokio::test]
    async fn test_get_connection_info() {
        let registry = ProviderRegistry::default();

        // Add test provider
        {
            let mut providers = registry.providers.write().await;
            providers.insert(
                "test-provider".to_string(),
                ProviderInfo {
                    id: "test-provider".to_string(),
                    name: "Test Provider".to_string(),
                    base_url: "https://api.test.com".to_string(),
                    npm_name: "@ai-sdk/test".to_string(),
                    npm_version: "1.0.0".to_string(),
                    env_vars: vec![
                        EnvVarInfo {
                            name: "TEST_API_KEY".to_string(),
                            description: "Test API key".to_string(),
                            required: true,
                        },
                        EnvVarInfo {
                            name: "TEST_OPTIONAL".to_string(),
                            description: "Optional setting".to_string(),
                            required: false,
                        },
                    ],
                    doc_url: "https://test.com/docs".to_string(),
                    api_version: Some("v1".to_string()),
                    available: true,
                    model_ids: vec![],
                },
            );
        }

        // Test getting connection info
        let connection_info = registry.get_connection_info("test-provider").await;
        assert!(connection_info.is_some());

        let info = connection_info.unwrap();
        assert_eq!(info.base_url, "https://api.test.com");
        assert_eq!(info.required_env_vars, vec!["TEST_API_KEY"]);
        assert_eq!(info.optional_env_vars, vec!["TEST_OPTIONAL"]);
        assert_eq!(info.config.get("api_version"), Some(&"v1".to_string()));

        // Test getting connection info for non-existent provider
        let non_existent = registry.get_connection_info("non-existent").await;
        assert!(non_existent.is_none());
    }

    #[tokio::test]
    async fn test_create_provider() {
        let registry = ProviderRegistry::default();

        // Add test provider and model
        {
            let mut providers = registry.providers.write().await;
            let mut models = registry.models.write().await;

            providers.insert(
                "openai".to_string(),
                ProviderInfo {
                    id: "openai".to_string(),
                    name: "OpenAI".to_string(),
                    base_url: "https://api.openai.com".to_string(),
                    npm_name: "@ai-sdk/openai".to_string(),
                    npm_version: "1.0.0".to_string(),
                    env_vars: vec![],
                    doc_url: "https://platform.openai.com/docs".to_string(),
                    api_version: None,
                    available: true,
                    model_ids: vec!["gpt-4".to_string()],
                },
            );

            models.insert(
                "gpt-4".to_string(),
                ModelInfo {
                    id: "gpt-4".to_string(),
                    name: "GPT-4".to_string(),
                    description: "GPT-4 model".to_string(),
                    provider_id: "openai".to_string(),
                    cost: ModelCostInfo {
                        input: 0.03,
                        output: 0.06,
                        cache_read: None,
                        cache_write: None,
                        reasoning: None,
                        currency: "USD".to_string(),
                    },
                    limits: ModelLimitInfo {
                        context: 8192,
                        output: 4096,
                        metadata: HashMap::new(),
                    },
                    input_modalities: vec!["text".to_string()],
                    output_modalities: vec!["text".to_string()],
                    metadata: HashMap::new(),
                },
            );
        }

        // Test creating provider (this will fail without proper environment setup, but we can test the error path)
        let result = registry
            .create_provider::<crate::models_dev::traits::OpenAIProvider>("openai")
            .await;

        // This should fail because OPENAI_API_KEY environment variable is not set
        assert!(result.is_err());

        // Test creating non-existent provider
        let non_existent_result = registry
            .create_provider::<crate::models_dev::traits::OpenAIProvider>("non-existent")
            .await;
        assert!(non_existent_result.is_err());

        // Test creating provider with no models
        {
            let mut providers = registry.providers.write().await;
            providers.insert(
                "empty-provider".to_string(),
                ProviderInfo {
                    id: "empty-provider".to_string(),
                    name: "Empty Provider".to_string(),
                    base_url: "https://api.empty.com".to_string(),
                    npm_name: "@ai-sdk/empty".to_string(),
                    npm_version: "1.0.0".to_string(),
                    env_vars: vec![],
                    doc_url: "https://empty.com/docs".to_string(),
                    api_version: None,
                    available: true,
                    model_ids: vec![],
                },
            );
        }

        let empty_result = registry
            .create_provider::<crate::models_dev::traits::OpenAIProvider>("empty-provider")
            .await;
        assert!(empty_result.is_err());
    }
}
