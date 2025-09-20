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
}
