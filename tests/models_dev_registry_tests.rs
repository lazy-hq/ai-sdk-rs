//! Unit tests for the models.dev provider registry.

#[cfg(feature = "models-dev")]
#[cfg(test)]
mod tests {
    use aisdk::models_dev::{
        ModelInfo, ModelsDevClient, ProviderInfo, ProviderRegistry,
        types::{
            ApiInfo, DocInfo, Modalities, Model, ModelCost, ModelLimit, ModelsDevResponse, NpmInfo,
            Provider,
        },
    };

    use std::collections::HashMap;

    /// Create a mock API response for testing.
    fn create_mock_api_response() -> ModelsDevResponse {
        ModelsDevResponse {
            providers: vec![
                Provider {
                    id: "openai".to_string(),
                    name: "OpenAI".to_string(),
                    npm: NpmInfo {
                        name: "@ai-sdk/openai".to_string(),
                        version: "1.0.0".to_string(),
                    },
                    env: vec![],
                    doc: DocInfo {
                        url: "https://platform.openai.com/docs".to_string(),
                        metadata: HashMap::new(),
                    },
                    api: ApiInfo {
                        base_url: "https://api.openai.com/v1".to_string(),
                        version: Some("v1".to_string()),
                        config: HashMap::new(),
                    },
                    models: vec![
                        Model {
                            id: "gpt-4".to_string(),
                            name: "GPT-4".to_string(),
                            description: "Large language model".to_string(),
                            cost: ModelCost {
                                input: 0.03,
                                output: 0.06,
                                cache_read: None,
                                cache_write: None,
                                reasoning: None,
                                currency: "USD".to_string(),
                            },
                            limits: ModelLimit {
                                context: 8192,
                                output: 4096,
                                metadata: HashMap::new(),
                            },
                            modalities: Modalities {
                                input: vec!["text".to_string()],
                                output: vec!["text".to_string()],
                            },
                            metadata: HashMap::new(),
                        },
                        Model {
                            id: "gpt-3.5-turbo".to_string(),
                            name: "GPT-3.5 Turbo".to_string(),
                            description: "Fast and capable language model".to_string(),
                            cost: ModelCost {
                                input: 0.0015,
                                output: 0.002,
                                cache_read: None,
                                cache_write: None,
                                reasoning: None,
                                currency: "USD".to_string(),
                            },
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
                        },
                    ],
                },
                Provider {
                    id: "anthropic".to_string(),
                    name: "Anthropic".to_string(),
                    npm: NpmInfo {
                        name: "@ai-sdk/anthropic".to_string(),
                        version: "1.0.0".to_string(),
                    },
                    env: vec![],
                    doc: DocInfo {
                        url: "https://docs.anthropic.com".to_string(),
                        metadata: HashMap::new(),
                    },
                    api: ApiInfo {
                        base_url: "https://api.anthropic.com".to_string(),
                        version: None,
                        config: HashMap::new(),
                    },
                    models: vec![Model {
                        id: "claude-3-opus".to_string(),
                        name: "Claude 3 Opus".to_string(),
                        description: "Most capable model".to_string(),
                        cost: ModelCost {
                            input: 0.015,
                            output: 0.075,
                            cache_read: None,
                            cache_write: None,
                            reasoning: None,
                            currency: "USD".to_string(),
                        },
                        limits: ModelLimit {
                            context: 200000,
                            output: 4096,
                            metadata: HashMap::new(),
                        },
                        modalities: Modalities {
                            input: vec!["text".to_string()],
                            output: vec!["text".to_string()],
                        },
                        metadata: HashMap::new(),
                    }],
                },
            ],
        }
    }

    #[tokio::test]
    async fn test_registry_creation() {
        let client = ModelsDevClient::new();
        let registry = ProviderRegistry::new(client);

        assert_eq!(registry.provider_count().await, 0);
        assert_eq!(registry.model_count().await, 0);
    }

    #[tokio::test]
    async fn test_registry_default() {
        let registry = ProviderRegistry::with_default_client();

        assert_eq!(registry.provider_count().await, 0);
        assert_eq!(registry.model_count().await, 0);
    }

    #[tokio::test]
    async fn test_provider_lookup() {
        let registry = ProviderRegistry::default();

        // Manually add a provider for testing
        {
            let mut providers = registry.providers.write().await;
            providers.insert(
                "openai".to_string(),
                ProviderInfo {
                    id: "openai".to_string(),
                    name: "OpenAI".to_string(),
                    base_url: "https://api.openai.com/v1".to_string(),
                    npm_name: "@ai-sdk/openai".to_string(),
                    npm_version: "1.0.0".to_string(),
                    env_vars: vec![],
                    doc_url: "https://platform.openai.com/docs".to_string(),
                    api_version: Some("v1".to_string()),
                    available: true,
                    model_ids: vec!["gpt-4".to_string()],
                },
            );
        }

        let provider = registry.get_provider("openai").await;
        assert!(provider.is_some());

        let provider = provider.unwrap();
        assert_eq!(provider.id, "openai");
        assert_eq!(provider.name, "OpenAI");
        assert_eq!(provider.base_url, "https://api.openai.com/v1");
        assert!(provider.available);

        // Test non-existent provider
        let non_existent = registry.get_provider("non-existent").await;
        assert!(non_existent.is_none());
    }

    #[tokio::test]
    async fn test_model_lookup() {
        let registry = ProviderRegistry::default();

        // Manually add a model for testing
        {
            let mut models = registry.models.write().await;
            models.insert(
                "gpt-4".to_string(),
                ModelInfo {
                    id: "gpt-4".to_string(),
                    name: "GPT-4".to_string(),
                    description: "Large language model".to_string(),
                    provider_id: "openai".to_string(),
                    cost: aisdk::models_dev::ModelCostInfo {
                        input: 0.03,
                        output: 0.06,
                        cache_read: None,
                        cache_write: None,
                        reasoning: None,
                        currency: "USD".to_string(),
                    },
                    limits: aisdk::models_dev::ModelLimitInfo {
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

        let model = registry.get_model("gpt-4").await;
        assert!(model.is_some());

        let model = model.unwrap();
        assert_eq!(model.id, "gpt-4");
        assert_eq!(model.name, "GPT-4");
        assert_eq!(model.provider_id, "openai");
        assert_eq!(model.cost.input, 0.03);
        assert_eq!(model.limits.context, 8192);

        // Test non-existent model
        let non_existent = registry.get_model("non-existent").await;
        assert!(non_existent.is_none());
    }

    #[tokio::test]
    async fn test_get_all_providers() {
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

        let all_providers = registry.get_all_providers().await;
        assert_eq!(all_providers.len(), 2);

        let provider_ids: Vec<String> = all_providers.into_iter().map(|p| p.id).collect();
        assert!(provider_ids.contains(&"openai".to_string()));
        assert!(provider_ids.contains(&"anthropic".to_string()));
    }

    #[tokio::test]
    async fn test_get_all_models() {
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
                    cost: aisdk::models_dev::ModelCostInfo {
                        input: 0.03,
                        output: 0.06,
                        cache_read: None,
                        cache_write: None,
                        reasoning: None,
                        currency: "USD".to_string(),
                    },
                    limits: aisdk::models_dev::ModelLimitInfo {
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
                "claude-3-opus".to_string(),
                ModelInfo {
                    id: "claude-3-opus".to_string(),
                    name: "Claude 3 Opus".to_string(),
                    description: "Most capable model".to_string(),
                    provider_id: "anthropic".to_string(),
                    cost: aisdk::models_dev::ModelCostInfo {
                        input: 0.015,
                        output: 0.075,
                        cache_read: None,
                        cache_write: None,
                        reasoning: None,
                        currency: "USD".to_string(),
                    },
                    limits: aisdk::models_dev::ModelLimitInfo {
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

        let all_models = registry.get_all_models().await;
        assert_eq!(all_models.len(), 2);

        let model_ids: Vec<String> = all_models.into_iter().map(|m| m.id).collect();
        assert!(model_ids.contains(&"gpt-4".to_string()));
        assert!(model_ids.contains(&"claude-3-opus".to_string()));
    }

    #[tokio::test]
    async fn test_get_models_for_provider() {
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
                    cost: aisdk::models_dev::ModelCostInfo {
                        input: 0.03,
                        output: 0.06,
                        cache_read: None,
                        cache_write: None,
                        reasoning: None,
                        currency: "USD".to_string(),
                    },
                    limits: aisdk::models_dev::ModelLimitInfo {
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
                    cost: aisdk::models_dev::ModelCostInfo {
                        input: 0.0015,
                        output: 0.002,
                        cache_read: None,
                        cache_write: None,
                        reasoning: None,
                        currency: "USD".to_string(),
                    },
                    limits: aisdk::models_dev::ModelLimitInfo {
                        context: 4096,
                        output: 2048,
                        metadata: HashMap::new(),
                    },
                    input_modalities: vec!["text".to_string()],
                    output_modalities: vec!["text".to_string()],
                    metadata: HashMap::new(),
                },
            );

            models.insert(
                "claude-3-opus".to_string(),
                ModelInfo {
                    id: "claude-3-opus".to_string(),
                    name: "Claude 3 Opus".to_string(),
                    description: "Most capable model".to_string(),
                    provider_id: "anthropic".to_string(),
                    cost: aisdk::models_dev::ModelCostInfo {
                        input: 0.015,
                        output: 0.075,
                        cache_read: None,
                        cache_write: None,
                        reasoning: None,
                        currency: "USD".to_string(),
                    },
                    limits: aisdk::models_dev::ModelLimitInfo {
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

        let openai_models = registry.get_models_for_provider("openai").await;
        assert_eq!(openai_models.len(), 2);

        let openai_model_ids: Vec<String> = openai_models.into_iter().map(|m| m.id).collect();
        assert!(openai_model_ids.contains(&"gpt-4".to_string()));
        assert!(openai_model_ids.contains(&"gpt-3.5-turbo".to_string()));

        let anthropic_models = registry.get_models_for_provider("anthropic").await;
        assert_eq!(anthropic_models.len(), 1);

        let anthropic_model_ids: Vec<String> = anthropic_models.into_iter().map(|m| m.id).collect();
        assert!(anthropic_model_ids.contains(&"claude-3-opus".to_string()));

        // Test non-existent provider
        let non_existent_models = registry.get_models_for_provider("non-existent").await;
        assert_eq!(non_existent_models.len(), 0);
    }

    #[tokio::test]
    async fn test_find_provider_for_model() {
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

        // Test direct mapping (should be None since we haven't set up the mapping)
        // But heuristics will match, so we expect Some("openai")
        assert_eq!(
            registry.find_provider_for_model("gpt-4").await,
            Some("openai".to_string())
        );

        // Test heuristics
        assert_eq!(
            registry.find_provider_for_model("gpt-4").await,
            Some("openai".to_string())
        );
        assert_eq!(
            registry.find_provider_for_model("claude-3-opus").await,
            Some("anthropic".to_string())
        );
        assert_eq!(
            registry.find_provider_for_model("openai/gpt-4").await,
            Some("openai".to_string())
        );
        assert_eq!(
            registry
                .find_provider_for_model("anthropic/claude-3-opus")
                .await,
            Some("anthropic".to_string())
        );

        // Test unknown model
        assert_eq!(
            registry.find_provider_for_model("unknown-model").await,
            None
        );
    }

    #[tokio::test]
    async fn test_provider_availability() {
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
                "unavailable".to_string(),
                ProviderInfo {
                    id: "unavailable".to_string(),
                    name: "Unavailable".to_string(),
                    base_url: "https://api.unavailable.com".to_string(),
                    npm_name: "@ai-sdk/unavailable".to_string(),
                    npm_version: "1.0.0".to_string(),
                    env_vars: vec![],
                    doc_url: "https://api.unavailable.com/docs".to_string(),
                    api_version: None,
                    available: false,
                    model_ids: vec![],
                },
            );
        }

        assert!(registry.is_provider_available("openai").await);
        assert!(!registry.is_provider_available("unavailable").await);
        assert!(!registry.is_provider_available("non-existent").await);
    }

    #[tokio::test]
    async fn test_model_availability() {
        let registry = ProviderRegistry::default();

        // Add test model
        {
            let mut models = registry.models.write().await;
            models.insert(
                "gpt-4".to_string(),
                ModelInfo {
                    id: "gpt-4".to_string(),
                    name: "GPT-4".to_string(),
                    description: "Large language model".to_string(),
                    provider_id: "openai".to_string(),
                    cost: aisdk::models_dev::ModelCostInfo {
                        input: 0.03,
                        output: 0.06,
                        cache_read: None,
                        cache_write: None,
                        reasoning: None,
                        currency: "USD".to_string(),
                    },
                    limits: aisdk::models_dev::ModelLimitInfo {
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

        assert!(registry.is_model_available("gpt-4").await);
        assert!(!registry.is_model_available("non-existent").await);
    }

    #[tokio::test]
    async fn test_clear_registry() {
        let registry = ProviderRegistry::default();

        // Add test data
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
                    cost: aisdk::models_dev::ModelCostInfo {
                        input: 0.03,
                        output: 0.06,
                        cache_read: None,
                        cache_write: None,
                        reasoning: None,
                        currency: "USD".to_string(),
                    },
                    limits: aisdk::models_dev::ModelLimitInfo {
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

        assert_eq!(registry.provider_count().await, 1);
        assert_eq!(registry.model_count().await, 1);

        registry.clear().await;

        assert_eq!(registry.provider_count().await, 0);
        assert_eq!(registry.model_count().await, 0);
    }

    #[tokio::test]
    async fn test_data_transformation() {
        let registry = ProviderRegistry::default();

        let mock_response = create_mock_api_response();
        let (providers, models, model_to_provider) =
            registry.transform_api_data(mock_response).await.unwrap();

        // Check providers
        assert_eq!(providers.len(), 2);
        assert!(providers.contains_key("openai"));
        assert!(providers.contains_key("anthropic"));

        let openai_provider = &providers["openai"];
        assert_eq!(openai_provider.id, "openai");
        assert_eq!(openai_provider.name, "OpenAI");
        assert_eq!(openai_provider.base_url, "https://api.openai.com/v1");
        assert_eq!(openai_provider.npm_name, "@ai-sdk/openai");
        assert_eq!(openai_provider.api_version, Some("v1".to_string()));
        assert!(openai_provider.available);
        assert_eq!(openai_provider.model_ids.len(), 2);
        assert!(openai_provider.model_ids.contains(&"gpt-4".to_string()));
        assert!(
            openai_provider
                .model_ids
                .contains(&"gpt-3.5-turbo".to_string())
        );

        // Check models
        assert_eq!(models.len(), 3);
        assert!(models.contains_key("gpt-4"));
        assert!(models.contains_key("gpt-3.5-turbo"));
        assert!(models.contains_key("claude-3-opus"));

        let gpt4_model = &models["gpt-4"];
        assert_eq!(gpt4_model.id, "gpt-4");
        assert_eq!(gpt4_model.name, "GPT-4");
        assert_eq!(gpt4_model.provider_id, "openai");
        assert_eq!(gpt4_model.cost.input, 0.03);
        assert_eq!(gpt4_model.cost.output, 0.06);
        assert_eq!(gpt4_model.limits.context, 8192);
        assert_eq!(gpt4_model.limits.output, 4096);
        assert_eq!(gpt4_model.input_modalities, vec!["text".to_string()]);
        assert_eq!(gpt4_model.output_modalities, vec!["text".to_string()]);

        // Check model-to-provider mapping
        assert_eq!(model_to_provider.len(), 3);
        assert_eq!(model_to_provider["gpt-4"], "openai");
        assert_eq!(model_to_provider["gpt-3.5-turbo"], "openai");
        assert_eq!(model_to_provider["claude-3-opus"], "anthropic");
    }
}
