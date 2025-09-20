//! Convenience functions for common models.dev operations.
//!
//! This module provides high-level, user-friendly wrapper functions around the
//! ProviderRegistry methods to simplify common tasks like finding providers,
//! filtering models, and getting summaries.

use std::collections::HashMap;

use crate::models_dev::registry::{ModelInfo, ProviderRegistry};

/// Find a provider for a specific cloud service.
///
/// This function provides a simple way to find a provider by cloud service name.
/// It uses common cloud service names and maps them to known provider IDs.
///
/// # Arguments
/// * `registry` - The provider registry to search in
/// * `cloud_service` - The name of the cloud service (e.g., "openai", "anthropic", "google")
///
/// # Returns
/// * Some(String) with the provider ID if found
/// * None if no matching provider is found
///
/// # Examples
///
/// ```rust
/// # use aisdk::models_dev::ProviderRegistry;
/// # use aisdk::models_dev::find_provider_for_cloud_service;
/// #[tokio::main]
/// async fn main() {
///     let registry = ProviderRegistry::with_default_client();
///     // Note: In a real application, you would call:
///     // registry.refresh().await.unwrap();
///     // For doc tests, we'll skip the refresh to avoid network calls
///     
///     let provider_id = find_provider_for_cloud_service(&registry, "openai").await;
///     // This will return None since the registry is empty without refresh
///     assert!(provider_id.is_none());
/// }
/// ```
pub async fn find_provider_for_cloud_service(
    registry: &ProviderRegistry,
    cloud_service: &str,
) -> Option<String> {
    let service_lower = cloud_service.to_lowercase();

    // Common cloud service mappings
    let service_mappings = std::collections::HashMap::from([
        ("openai", "openai"),
        ("anthropic", "anthropic"),
        ("claude", "anthropic"),
        ("google", "google"),
        ("gemini", "google"),
        ("bard", "google"),
        ("meta", "meta"),
        ("llama", "meta"),
        ("mistral", "mistral"),
        ("mixtral", "mistral"),
        ("cohere", "cohere"),
        ("command", "cohere"),
        ("azure", "azure"),
        ("microsoft", "azure"),
        ("aws", "aws"),
        ("amazon", "aws"),
        ("bedrock", "aws"),
    ]);

    // Look up the service name
    if let Some(provider_id) = service_mappings.get(&service_lower.as_str()) {
        // Check if the provider exists in the registry
        if registry.is_provider_available(provider_id).await {
            return Some(provider_id.to_string());
        }
    }

    // If not found in mappings, try direct lookup
    if registry.is_provider_available(&service_lower).await {
        return Some(service_lower);
    }

    None
}

/// List providers that support a specific npm package.
///
/// This function finds all providers in the registry that support the given
/// npm package name. It's useful for discovering which providers can be used
/// with a specific AI SDK package.
///
/// # Arguments
/// * `registry` - The provider registry to search in
/// * `npm_package` - The npm package name (e.g., "@ai-sdk/openai")
///
/// # Returns
/// * A vector of provider IDs that support the npm package
///
/// # Examples
///
/// ```rust
/// # use aisdk::models_dev::ProviderRegistry;
/// # use aisdk::models_dev::list_providers_for_npm_package;
/// #[tokio::main]
/// async fn main() {
///     let registry = ProviderRegistry::with_default_client();
///     // Note: In a real application, you would call:
///     // registry.refresh().await.unwrap();
///     // For doc tests, we'll skip the refresh to avoid network calls
///     
///     let providers = list_providers_for_npm_package(&registry, "@ai-sdk/openai").await;
///     // This will return an empty vector since the registry is empty without refresh
///     assert!(providers.is_empty());
/// }
/// ```
pub async fn list_providers_for_npm_package(
    registry: &ProviderRegistry,
    npm_package: &str,
) -> Vec<String> {
    registry.find_providers_by_npm(npm_package).await
}

/// Find models with a specific capability.
///
/// This function filters models based on their capabilities such as reasoning,
/// tool calling, attachment support, or vision capabilities.
///
/// # Arguments
/// * `registry` - The provider registry to search in
/// * `capability` - The capability to filter by ("reasoning", "tool_call", "attachment", "vision")
///
/// # Returns
/// * `Vec<ModelInfo>` - A vector of model information for models with the specified capability
///
/// # Examples
/// ```rust
/// use topaz::models_dev::{ProviderRegistry, find_models_with_capability};
///
/// # async fn example() -> Vec<topaz::models_dev::ModelInfo> {
/// let registry = ProviderRegistry::default();
/// // registry.refresh().await.unwrap(); // Load data
/// let reasoning_models = find_models_with_capability(&registry, "reasoning").await;
/// reasoning_models
/// # }
/// ```
pub async fn find_models_with_capability(
    registry: &ProviderRegistry,
    capability: &str,
) -> Vec<ModelInfo> {
    registry.get_models_with_capability(capability).await
}

/// Get all available providers with their models in a structured format.
///
/// This function provides a convenient way to get all providers along with
/// their associated models in a single call.
///
/// # Arguments
/// * `registry` - The provider registry to query
///
/// # Returns
/// * A vector of tuples containing (provider_id, provider_name, model_ids)
///
/// # Examples
///
/// ```rust
/// # use aisdk::models_dev::ProviderRegistry;
/// # use aisdk::models_dev::get_providers_summary;
/// #[tokio::main]
/// async fn main() {
///     let registry = ProviderRegistry::with_default_client();
///     // Note: In a real application, you would call:
///     // registry.refresh().await.unwrap();
///     // For doc tests, we'll skip the refresh to avoid network calls
///     
///     let providers_with_models = get_providers_summary(&registry).await;
///     // This will return an empty vector since the registry is empty without refresh
///     assert!(providers_with_models.is_empty());
/// }
/// ```
pub async fn get_providers_summary(
    registry: &ProviderRegistry,
) -> Vec<(String, String, Vec<String>)> {
    let providers_with_models = registry.get_providers_with_models().await;

    providers_with_models
        .into_iter()
        .map(|(provider_id, provider, models)| {
            let model_ids: Vec<String> = models.into_iter().map(|m| m.id).collect();
            (provider_id, provider.name, model_ids)
        })
        .collect()
}

/// Find the best model for a specific use case.
///
/// This function helps find the most suitable model based on common use cases
/// like "chat", "code", "reasoning", "vision", etc.
///
/// # Arguments
/// * `registry` - The provider registry to search in
/// * `use_case` - The use case to find a model for
///
/// # Returns
/// * Some(String) with the recommended model ID if found
/// * None if no suitable model is found
///
/// # Examples
///
/// ```rust
/// # use aisdk::models_dev::ProviderRegistry;
/// # use aisdk::models_dev::find_best_model_for_use_case;
/// #[tokio::main]
/// async fn main() {
///     let registry = ProviderRegistry::with_default_client();
///     // Note: In a real application, you would call:
///     // registry.refresh().await.unwrap();
///     // For doc tests, we'll skip the refresh to avoid network calls
///     
///     let model_id = find_best_model_for_use_case(&registry, "chat").await;
///     // This will return None since the registry is empty without refresh
///     assert!(model_id.is_none());
/// }
/// ```
pub async fn find_best_model_for_use_case(
    registry: &ProviderRegistry,
    use_case: &str,
) -> Option<String> {
    let use_case_lower = use_case.to_lowercase();

    // Define model preferences for different use cases
    let model_preferences = match use_case_lower.as_str() {
        "chat" | "conversation" => {
            // Prefer models good for conversation
            vec![
                "gpt-4",
                "claude-3-opus",
                "claude-3-sonnet",
                "gemini-pro",
                "gpt-3.5-turbo",
            ]
        }
        "code" | "programming" => {
            // Prefer models good for coding
            vec![
                "gpt-4",
                "claude-3-opus",
                "gemini-pro",
                "codellama",
                "gpt-3.5-turbo",
            ]
        }
        "reasoning" | "analysis" => {
            // Prefer models with strong reasoning capabilities
            vec!["gpt-4", "claude-3-opus", "gemini-pro", "claude-3-sonnet"]
        }
        "vision" | "image" => {
            // Prefer models with vision capabilities
            vec!["gpt-4-vision", "claude-3-opus", "gemini-pro-vision"]
        }
        "fast" | "quick" => {
            // Prefer faster, more efficient models
            vec![
                "gpt-3.5-turbo",
                "claude-3-haiku",
                "gemini-flash",
                "mistral-7b",
            ]
        }
        "cheap" | "budget" => {
            // Prefer more cost-effective models
            vec![
                "gpt-3.5-turbo",
                "claude-3-haiku",
                "gemini-flash",
                "mistral-7b",
            ]
        }
        _ => {
            // Default to general-purpose models
            vec!["gpt-4", "claude-3-opus", "gemini-pro", "gpt-3.5-turbo"]
        }
    };

    // Try to find the first available model from the preferences
    for preferred_model in model_preferences {
        if registry.is_model_available(preferred_model).await {
            return Some(preferred_model.to_string());
        }
    }

    // If no preferred model is available, get any available model
    let all_models = registry.get_all_models().await;
    if let Some(first_model) = all_models.first() {
        return Some(first_model.id.clone());
    }

    None
}

/// Check if a provider's configuration is valid.
///
/// This function validates that all required environment variables for a provider
/// are set and have valid values.
///
/// # Arguments
/// * `registry` - The provider registry to check
/// * `provider_id` - The ID of the provider to check
///
/// # Returns
/// * `Ok(())` - If the provider configuration is valid
/// * `Err(Vec<String>)` - A vector of error messages describing configuration issues
///
/// # Examples
/// ```rust
/// use topaz::models_dev::{ProviderRegistry, check_provider_configuration};
///
/// # async fn example() -> Result<(), Vec<String>> {
/// let registry = ProviderRegistry::default();
/// // registry.refresh().await.unwrap(); // Load data
/// let result = check_provider_configuration(&registry, "openai").await;
/// match result {
///     Ok(()) => println!("Configuration is valid"),
///     Err(errors) => println!("Configuration errors: {:?}", errors),
/// }
/// result
/// # }
/// ```
pub async fn check_provider_configuration(
    registry: &ProviderRegistry,
    provider_id: &str,
) -> Result<(), Vec<String>> {
    let provider = registry
        .get_provider(provider_id)
        .await
        .ok_or_else(|| vec![format!("Provider '{}' not found", provider_id)])?;

    let mut errors = Vec::new();

    // Check required environment variables
    for env_var in &provider.env_vars {
        if env_var.required {
            match std::env::var(&env_var.name) {
                Ok(value) => {
                    if value.trim().is_empty() {
                        errors.push(format!(
                            "Required environment variable '{}' is empty",
                            env_var.name
                        ));
                    }
                }
                Err(_) => {
                    errors.push(format!(
                        "Required environment variable '{}' is not set",
                        env_var.name
                    ));
                }
            }
        }
    }

    // Check if provider is available
    if !provider.available {
        errors.push(format!(
            "Provider '{}' is marked as unavailable",
            provider_id
        ));
    }

    // Check if provider has any models
    let models = registry.get_models_for_provider(provider_id).await;
    if models.is_empty() {
        errors.push(format!(
            "Provider '{}' has no available models",
            provider_id
        ));
    }

    if errors.is_empty() {
        Ok(())
    } else {
        Err(errors)
    }
}

/// Get a summary of all available capabilities and the models that support them.
///
/// This function returns a HashMap where keys are capability names and values are
/// vectors of model IDs that support each capability.
///
/// # Arguments
/// * `registry` - The provider registry to get capability summary from
///
/// # Returns
/// * `HashMap<String, Vec<String>>` - A map of capability names to lists of model IDs
///
/// # Examples
/// ```rust
/// use topaz::models_dev::{ProviderRegistry, get_capability_summary};
///
/// # async fn example() -> std::collections::HashMap<String, Vec<String>> {
/// let registry = ProviderRegistry::default();
/// // registry.refresh().await.unwrap(); // Load data
/// let capabilities = get_capability_summary(&registry).await;
/// for (capability, model_ids) in capabilities {
///     println!("{} capability: {} models", capability, model_ids.len());
/// }
/// capabilities
/// # }
/// ```
pub async fn get_capability_summary(registry: &ProviderRegistry) -> HashMap<String, Vec<String>> {
    let mut summary = HashMap::new();

    // Define all capabilities to check
    let capabilities = vec!["reasoning", "tool_call", "attachment", "vision"];

    for capability in capabilities {
        let models = find_models_with_capability(registry, capability).await;
        let model_ids: Vec<String> = models.into_iter().map(|model| model.id).collect();
        summary.insert(capability.to_string(), model_ids);
    }

    summary
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::models_dev::{ModelCostInfo, ModelInfo, ModelLimitInfo, ProviderInfo};
    use std::collections::HashMap;

    #[tokio::test]
    async fn test_find_provider_for_cloud_service() {
        let registry = ProviderRegistry::default();

        // Add test providers
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

        // Test known mappings
        assert_eq!(
            find_provider_for_cloud_service(&registry, "openai").await,
            Some("openai".to_string())
        );
        assert_eq!(
            find_provider_for_cloud_service(&registry, "claude").await,
            Some("anthropic".to_string())
        );

        // Test direct lookup
        assert_eq!(
            find_provider_for_cloud_service(&registry, "anthropic").await,
            Some("anthropic".to_string())
        );

        // Test unknown service
        assert_eq!(
            find_provider_for_cloud_service(&registry, "unknown").await,
            None
        );
    }

    #[tokio::test]
    async fn test_list_providers_for_npm_package() {
        let registry = ProviderRegistry::default();

        // Add test providers
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
                "another-openai".to_string(),
                ProviderInfo {
                    id: "another-openai".to_string(),
                    name: "Another OpenAI".to_string(),
                    base_url: "https://api.another-openai.com".to_string(),
                    npm_name: "@ai-sdk/openai".to_string(),
                    npm_version: "1.0.0".to_string(),
                    env_vars: vec![],
                    doc_url: "https://another-openai.com/docs".to_string(),
                    api_version: None,
                    available: true,
                    model_ids: vec![],
                },
            );
        }

        let providers = list_providers_for_npm_package(&registry, "@ai-sdk/openai").await;
        assert_eq!(providers.len(), 2);
        assert!(providers.contains(&"openai".to_string()));
        assert!(providers.contains(&"another-openai".to_string()));

        // Test non-existent package
        let providers = list_providers_for_npm_package(&registry, "@ai-sdk/nonexistent").await;
        assert_eq!(providers.len(), 0);
    }

    #[tokio::test]
    async fn test_find_models_with_capability() {
        let registry = ProviderRegistry::default();

        // Add test models
        {
            let mut models = registry.models.write().await;

            // Model with reasoning capability
            models.insert(
                "reasoning-model".to_string(),
                ModelInfo {
                    id: "reasoning-model".to_string(),
                    name: "Reasoning Model".to_string(),
                    description: "Model with reasoning".to_string(),
                    provider_id: "test".to_string(),
                    cost: ModelCostInfo {
                        input: 0.01,
                        output: 0.02,
                        cache_read: None,
                        cache_write: None,
                        reasoning: Some(0.03),
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

            // Model with vision capability
            models.insert(
                "vision-model".to_string(),
                ModelInfo {
                    id: "vision-model".to_string(),
                    name: "Vision Model".to_string(),
                    description: "Model with vision".to_string(),
                    provider_id: "test".to_string(),
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
        }

        let reasoning_models = find_models_with_capability(&registry, "reasoning").await;
        assert_eq!(reasoning_models.len(), 1);
        assert_eq!(reasoning_models[0].id, "reasoning-model");

        let vision_models = find_models_with_capability(&registry, "vision").await;
        assert_eq!(vision_models.len(), 1);
        assert_eq!(vision_models[0].id, "vision-model");

        // Test unknown capability
        let unknown_models = find_models_with_capability(&registry, "unknown").await;
        assert_eq!(unknown_models.len(), 0);
    }

    #[tokio::test]
    async fn test_get_providers_summary() {
        let registry = ProviderRegistry::default();

        // Add test providers and models
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
                    model_ids: vec!["gpt-4".to_string()],
                },
            );
        }

        {
            let mut models = registry.models.write().await;
            models.insert(
                "gpt-4".to_string(),
                ModelInfo {
                    id: "gpt-4".to_string(),
                    name: "GPT-4".to_string(),
                    description: "Large language model".to_string(),
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

        let summary = get_providers_summary(&registry).await;
        assert_eq!(summary.len(), 1);

        let (provider_id, provider_name, model_ids) = &summary[0];
        assert_eq!(provider_id, "openai");
        assert_eq!(provider_name, "OpenAI");
        assert_eq!(model_ids.len(), 1);
        assert!(model_ids.contains(&"gpt-4".to_string()));
    }

    #[tokio::test]
    async fn test_find_best_model_for_use_case() {
        let registry = ProviderRegistry::default();

        // Add test models
        {
            let mut models = registry.models.write().await;
            models.insert(
                "gpt-4".to_string(),
                ModelInfo {
                    id: "gpt-4".to_string(),
                    name: "GPT-4".to_string(),
                    description: "Large language model".to_string(),
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
                    description: "Fast and capable language model".to_string(),
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
        }

        // Test chat use case (should prefer GPT-4)
        assert_eq!(
            find_best_model_for_use_case(&registry, "chat").await,
            Some("gpt-4".to_string())
        );

        // Test fast use case (should prefer GPT-3.5-turbo)
        assert_eq!(
            find_best_model_for_use_case(&registry, "fast").await,
            Some("gpt-3.5-turbo".to_string())
        );

        // Test cheap use case (should prefer GPT-3.5-turbo)
        assert_eq!(
            find_best_model_for_use_case(&registry, "cheap").await,
            Some("gpt-3.5-turbo".to_string())
        );
    }

    #[tokio::test]
    async fn test_check_provider_configuration() {
        let registry = ProviderRegistry::default();

        // Add test provider with required env var and models
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
                    env_vars: vec![],
                    doc_url: "https://api.test.com/docs".to_string(),
                    api_version: None,
                    available: true,
                    model_ids: vec!["test-model".to_string()],
                },
            );
        }

        // Add a model for the test provider
        {
            let mut models = registry.models.write().await;
            models.insert(
                "test-model".to_string(),
                ModelInfo {
                    id: "test-model".to_string(),
                    name: "Test Model".to_string(),
                    description: "A test model".to_string(),
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

        // Test with non-existent provider
        let result = check_provider_configuration(&registry, "non-existent").await;
        assert!(result.is_err());

        // Test with provider that has no required env vars
        let result = check_provider_configuration(&registry, "test-provider").await;
        assert!(result.is_ok());
    }

    #[tokio::test]
    async fn test_get_capability_summary() {
        let registry = ProviderRegistry::default();

        // Add test models
        {
            let mut models = registry.models.write().await;

            // Model with reasoning capability
            models.insert(
                "reasoning-model".to_string(),
                ModelInfo {
                    id: "reasoning-model".to_string(),
                    name: "Reasoning Model".to_string(),
                    description: "Model with reasoning".to_string(),
                    provider_id: "test".to_string(),
                    cost: ModelCostInfo {
                        input: 0.01,
                        output: 0.02,
                        cache_read: None,
                        cache_write: None,
                        reasoning: Some(0.03),
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

            // Model with vision capability
            models.insert(
                "vision-model".to_string(),
                ModelInfo {
                    id: "vision-model".to_string(),
                    name: "Vision Model".to_string(),
                    description: "Model with vision".to_string(),
                    provider_id: "test".to_string(),
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
        }

        let summary = get_capability_summary(&registry).await;
        assert_eq!(summary.len(), 4); // All capabilities are checked, even if empty
        assert!(summary.contains_key("reasoning"));
        assert!(summary.contains_key("vision"));
        assert!(summary.contains_key("tool_call"));
        assert!(summary.contains_key("attachment"));

        let reasoning_models = &summary["reasoning"];
        assert_eq!(reasoning_models.len(), 1);
        assert!(reasoning_models.contains(&"reasoning-model".to_string()));

        let vision_models = &summary["vision"];
        assert_eq!(vision_models.len(), 1);
        assert!(vision_models.contains(&"vision-model".to_string()));

        // Check empty capabilities
        let tool_call_models = &summary["tool_call"];
        assert_eq!(tool_call_models.len(), 0);

        let attachment_models = &summary["attachment"];
        assert_eq!(attachment_models.len(), 0);
    }
}
