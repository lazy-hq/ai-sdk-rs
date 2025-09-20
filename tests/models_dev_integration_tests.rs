//! Integration tests for the complete models.dev workflow.
//!
//! These tests verify that all the convenience functions and registry methods
//! work together correctly in realistic scenarios.

#[cfg(feature = "models-dev")]
#[cfg(test)]
mod tests {
    use aisdk::models_dev::{
        ProviderRegistry, check_provider_configuration, find_best_model_for_use_case,
        find_models_with_capability, find_provider_for_cloud_service, get_capability_summary,
        get_providers_summary, list_providers_for_npm_package,
        traits::{AnthropicProvider, GoogleProvider, ModelsDevAware, OpenAIProvider},
        types::{
            ApiInfo, DocInfo, EnvVar, Modalities, Model, ModelCost, ModelLimit, ModelsDevResponse,
            NpmInfo, Provider,
        },
    };
    use std::collections::HashMap;

    /// Create a comprehensive mock API response for integration testing.
    fn create_comprehensive_mock_response() -> ModelsDevResponse {
        ModelsDevResponse {
            providers: vec![
                // OpenAI provider
                Provider {
                    id: "openai".to_string(),
                    name: "OpenAI".to_string(),
                    npm: NpmInfo {
                        name: "@ai-sdk/openai".to_string(),
                        version: "1.0.0".to_string(),
                    },
                    env: vec![
                        EnvVar {
                            name: "OPENAI_API_KEY".to_string(),
                            description: "Your OpenAI API key".to_string(),
                            required: true,
                        },
                        EnvVar {
                            name: "OPENAI_ORGANIZATION".to_string(),
                            description: "Your OpenAI organization ID".to_string(),
                            required: false,
                        },
                    ],
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
                            description: "Most capable GPT-4 model".to_string(),
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
                            id: "gpt-4-turbo".to_string(),
                            name: "GPT-4 Turbo".to_string(),
                            description: "Latest GPT-4 model with reasoning capabilities"
                                .to_string(),
                            cost: ModelCost {
                                input: 0.01,
                                output: 0.03,
                                cache_read: None,
                                cache_write: None,
                                reasoning: Some(0.02),
                                currency: "USD".to_string(),
                            },
                            limits: ModelLimit {
                                context: 128000,
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
                            description: "Fast and capable model for most tasks".to_string(),
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
                // Anthropic provider
                Provider {
                    id: "anthropic".to_string(),
                    name: "Anthropic".to_string(),
                    npm: NpmInfo {
                        name: "@ai-sdk/anthropic".to_string(),
                        version: "1.0.0".to_string(),
                    },
                    env: vec![EnvVar {
                        name: "ANTHROPIC_API_KEY".to_string(),
                        description: "Your Anthropic API key".to_string(),
                        required: true,
                    }],
                    doc: DocInfo {
                        url: "https://docs.anthropic.com".to_string(),
                        metadata: HashMap::new(),
                    },
                    api: ApiInfo {
                        base_url: "https://api.anthropic.com".to_string(),
                        version: Some("2023-06-01".to_string()),
                        config: HashMap::new(),
                    },
                    models: vec![
                        Model {
                            id: "claude-3-opus".to_string(),
                            name: "Claude 3 Opus".to_string(),
                            description: "Most capable Claude model".to_string(),
                            cost: ModelCost {
                                input: 0.015,
                                output: 0.075,
                                cache_read: None,
                                cache_write: None,
                                reasoning: Some(0.03),
                                currency: "USD".to_string(),
                            },
                            limits: ModelLimit {
                                context: 200000,
                                output: 4096,
                                metadata: HashMap::new(),
                            },
                            modalities: Modalities {
                                input: vec!["text".to_string(), "image".to_string()],
                                output: vec!["text".to_string()],
                            },
                            metadata: HashMap::new(),
                        },
                        Model {
                            id: "claude-3-sonnet".to_string(),
                            name: "Claude 3 Sonnet".to_string(),
                            description: "Balanced performance and speed".to_string(),
                            cost: ModelCost {
                                input: 0.003,
                                output: 0.015,
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
                        },
                    ],
                },
                // Google provider
                Provider {
                    id: "google".to_string(),
                    name: "Google".to_string(),
                    npm: NpmInfo {
                        name: "@ai-sdk/google".to_string(),
                        version: "1.0.0".to_string(),
                    },
                    env: vec![EnvVar {
                        name: "GOOGLE_API_KEY".to_string(),
                        description: "Your Google API key".to_string(),
                        required: true,
                    }],
                    doc: DocInfo {
                        url: "https://ai.google.dev/docs".to_string(),
                        metadata: HashMap::new(),
                    },
                    api: ApiInfo {
                        base_url: "https://generativelanguage.googleapis.com".to_string(),
                        version: None,
                        config: HashMap::new(),
                    },
                    models: vec![Model {
                        id: "gemini-pro".to_string(),
                        name: "Gemini Pro".to_string(),
                        description: "Most capable Gemini model".to_string(),
                        cost: ModelCost {
                            input: 0.001,
                            output: 0.002,
                            cache_read: None,
                            cache_write: None,
                            reasoning: None,
                            currency: "USD".to_string(),
                        },
                        limits: ModelLimit {
                            context: 32768,
                            output: 2048,
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
    async fn test_complete_workflow() {
        // Create a registry
        let registry = ProviderRegistry::default();

        // Manually load the mock data (since we can't call the real API in tests)
        let mock_response = create_comprehensive_mock_response();
        let (providers, models, model_to_provider) =
            registry.transform_api_data(mock_response).await.unwrap();

        {
            let mut providers_lock = registry.providers.write().await;
            *providers_lock = providers;
        }

        {
            let mut models_lock = registry.models.write().await;
            *models_lock = models;
        }

        {
            let mut mapping_lock = registry.model_to_provider.write().await;
            *mapping_lock = model_to_provider;
        }

        // Test the complete workflow

        // 1. Test convenience functions
        let openai_provider = find_provider_for_cloud_service(&registry, "openai").await;
        assert_eq!(openai_provider, Some("openai".to_string()));

        let claude_provider = find_provider_for_cloud_service(&registry, "claude").await;
        assert_eq!(claude_provider, Some("anthropic".to_string()));

        // 2. Test npm package lookup
        let openai_packages = list_providers_for_npm_package(&registry, "@ai-sdk/openai").await;
        assert_eq!(openai_packages, vec!["openai"]);

        let anthropic_packages =
            list_providers_for_npm_package(&registry, "@ai-sdk/anthropic").await;
        assert_eq!(anthropic_packages, vec!["anthropic"]);

        // 3. Test capability-based model discovery
        let reasoning_models = find_models_with_capability(&registry, "reasoning").await;
        assert_eq!(reasoning_models.len(), 2); // gpt-4-turbo and claude-3-opus
        assert!(reasoning_models.contains(&"gpt-4-turbo".to_string()));
        assert!(reasoning_models.contains(&"claude-3-opus".to_string()));

        let vision_models = find_models_with_capability(&registry, "vision").await;
        assert_eq!(vision_models.len(), 1); // claude-3-opus
        assert!(vision_models.contains(&"claude-3-opus".to_string()));

        // 4. Test provider summary
        let summary = get_providers_summary(&registry).await;
        assert_eq!(summary.len(), 3); // openai, anthropic, google

        let openai_summary = summary.iter().find(|(id, _, _)| id == "openai").unwrap();
        assert_eq!(openai_summary.1, "OpenAI");
        assert_eq!(openai_summary.2.len(), 3); // gpt-4, gpt-4-turbo, gpt-3.5-turbo

        // 5. Test use case-based model recommendation
        let chat_model = find_best_model_for_use_case(&registry, "chat").await;
        assert_eq!(chat_model, Some("gpt-4".to_string()));

        let fast_model = find_best_model_for_use_case(&registry, "fast").await;
        assert_eq!(fast_model, Some("gpt-3.5-turbo".to_string()));

        let reasoning_model = find_best_model_for_use_case(&registry, "reasoning").await;
        assert_eq!(reasoning_model, Some("gpt-4".to_string()));

        // 6. Test capability summary
        let capability_summary = get_capability_summary(&registry).await;
        assert_eq!(capability_summary.len(), 2); // reasoning and vision
        assert!(capability_summary.contains_key("reasoning"));
        assert!(capability_summary.contains_key("vision"));

        // 7. Test registry methods
        let providers_by_npm = registry.find_providers_by_npm("@ai-sdk/openai").await;
        assert_eq!(providers_by_npm.len(), 1);
        assert_eq!(providers_by_npm[0], "openai");

        let models_with_reasoning = registry.get_models_with_capability("reasoning").await;
        assert_eq!(models_with_reasoning.len(), 2);

        let providers_with_models = registry.get_providers_with_models().await;
        assert_eq!(providers_with_models.len(), 3);

        // 8. Test connection info
        let openai_connection_info = registry.get_connection_info("openai").await;
        assert!(openai_connection_info.is_some());
        let connection_info = openai_connection_info.unwrap();
        assert_eq!(connection_info.base_url, "https://api.openai.com/v1");
        assert_eq!(connection_info.required_env_vars, vec!["OPENAI_API_KEY"]);
        assert_eq!(
            connection_info.optional_env_vars,
            vec!["OPENAI_ORGANIZATION"]
        );

        // 9. Test configuration checking (should fail without env vars set)
        let config_check = check_provider_configuration(&registry, "openai").await;
        assert!(config_check.is_err());
        let missing_vars = config_check.unwrap_err();
        assert!(missing_vars.contains(&"OPENAI_API_KEY".to_string()));
    }

    #[tokio::test]
    async fn test_models_dev_aware_integration() {
        // Create a registry
        let registry = ProviderRegistry::default();

        // Manually load the mock data
        let mock_response = create_comprehensive_mock_response();
        let (providers, models, model_to_provider) =
            registry.transform_api_data(mock_response).await.unwrap();

        {
            let mut providers_lock = registry.providers.write().await;
            *providers_lock = providers;
        }

        {
            let mut models_lock = registry.models.write().await;
            *models_lock = models;
        }

        {
            let mut mapping_lock = registry.model_to_provider.write().await;
            *mapping_lock = model_to_provider;
        }

        // Test creating providers with ModelsDevAware trait (these will fail without env vars)

        // Test OpenAI provider creation - should fail because OPENAI_API_KEY is not set
        let openai_result = registry.create_provider::<OpenAIProvider>("openai").await;
        assert!(openai_result.is_err());
        let error_msg = format!("{}", openai_result.unwrap_err());
        assert!(error_msg.contains("Unsupported provider") || error_msg.contains("OPENAI_API_KEY"));

        // Test Anthropic provider creation - should fail because ANTHROPIC_API_KEY is not set
        let anthropic_result = registry
            .create_provider::<AnthropicProvider>("anthropic")
            .await;
        assert!(anthropic_result.is_err());
        let error_msg = format!("{}", anthropic_result.unwrap_err());
        assert!(
            error_msg.contains("Unsupported provider") || error_msg.contains("ANTHROPIC_API_KEY")
        );

        // Test Google provider creation - should fail because GOOGLE_API_KEY is not set
        let google_result = registry.create_provider::<GoogleProvider>("google").await;
        assert!(google_result.is_err());
        let error_msg = format!("{}", google_result.unwrap_err());
        assert!(error_msg.contains("Unsupported provider") || error_msg.contains("GOOGLE_API_KEY"));

        // Test with non-existent provider
        let nonexistent_result = registry
            .create_provider::<OpenAIProvider>("nonexistent")
            .await;
        assert!(nonexistent_result.is_err());
        let error_msg = format!("{}", nonexistent_result.unwrap_err());
        assert!(error_msg.contains("not found"));
    }

    #[tokio::test]
    async fn test_error_handling_scenarios() {
        let registry = ProviderRegistry::default();

        // Test operations on empty registry
        assert_eq!(registry.provider_count().await, 0);
        assert_eq!(registry.model_count().await, 0);

        // Test convenience functions on empty registry
        let openai_provider = find_provider_for_cloud_service(&registry, "openai").await;
        assert_eq!(openai_provider, None);

        let openai_packages = list_providers_for_npm_package(&registry, "@ai-sdk/openai").await;
        assert_eq!(openai_packages.len(), 0);

        let reasoning_models = find_models_with_capability(&registry, "reasoning").await;
        assert_eq!(reasoning_models.len(), 0);

        let summary = get_providers_summary(&registry).await;
        assert_eq!(summary.len(), 0);

        let chat_model = find_best_model_for_use_case(&registry, "chat").await;
        assert_eq!(chat_model, None);

        let capability_summary = get_capability_summary(&registry).await;
        assert_eq!(capability_summary.len(), 0);

        // Test registry methods on empty registry
        let providers_by_npm = registry.find_providers_by_npm("@ai-sdk/openai").await;
        assert_eq!(providers_by_npm.len(), 0);

        let models_with_reasoning = registry.get_models_with_capability("reasoning").await;
        assert_eq!(models_with_reasoning.len(), 0);

        let providers_with_models = registry.get_providers_with_models().await;
        assert_eq!(providers_with_models.len(), 0);

        let connection_info = registry.get_connection_info("openai").await;
        assert!(connection_info.is_none());

        // Test configuration checking on empty registry
        let config_check = check_provider_configuration(&registry, "openai").await;
        assert!(config_check.is_err());
        let missing_vars = config_check.unwrap_err();
        assert!(missing_vars.contains(&"Provider 'openai' not found".to_string()));
    }

    #[tokio::test]
    async fn test_edge_cases() {
        let registry = ProviderRegistry::default();

        // Load minimal mock data
        let minimal_response = ModelsDevResponse {
            providers: vec![Provider {
                id: "minimal".to_string(),
                name: "Minimal Provider".to_string(),
                npm: NpmInfo {
                    name: "@ai-sdk/minimal".to_string(),
                    version: "1.0.0".to_string(),
                },
                env: vec![],
                doc: DocInfo {
                    url: "https://minimal.example.com/docs".to_string(),
                    metadata: HashMap::new(),
                },
                api: ApiInfo {
                    base_url: "https://api.minimal.example.com".to_string(),
                    version: None,
                    config: HashMap::new(),
                },
                models: vec![],
            }],
        };

        let (providers, models, model_to_provider) =
            registry.transform_api_data(minimal_response).await.unwrap();

        {
            let mut providers_lock = registry.providers.write().await;
            *providers_lock = providers;
        }

        {
            let mut models_lock = registry.models.write().await;
            *models_lock = models;
        }

        {
            let mut mapping_lock = registry.model_to_provider.write().await;
            *mapping_lock = model_to_provider;
        }

        // Test provider with no models
        let minimal_provider = find_provider_for_cloud_service(&registry, "minimal").await;
        assert_eq!(minimal_provider, Some("minimal".to_string()));

        let minimal_packages = list_providers_for_npm_package(&registry, "@ai-sdk/minimal").await;
        assert_eq!(minimal_packages, vec!["minimal"]);

        // Test model recommendation when no models are available
        let chat_model = find_best_model_for_use_case(&registry, "chat").await;
        assert_eq!(chat_model, None);

        // Test capability summary with no capabilities
        let capability_summary = get_capability_summary(&registry).await;
        assert_eq!(capability_summary.len(), 0);

        // Test provider summary with no models
        let summary = get_providers_summary(&registry).await;
        assert_eq!(summary.len(), 1);
        let (provider_id, provider_name, model_ids) = &summary[0];
        assert_eq!(provider_id, "minimal");
        assert_eq!(provider_name, "Minimal Provider");
        assert_eq!(model_ids.len(), 0);
    }

    #[tokio::test]
    async fn test_real_world_scenario() {
        // Simulate a real-world scenario where a developer wants to:
        // 1. Find providers for a specific npm package
        // 2. Check which models have specific capabilities
        // 3. Get the best model for their use case
        // 4. Check if the provider is properly configured
        // 5. Get connection information

        let registry = ProviderRegistry::default();

        // Load comprehensive mock data
        let mock_response = create_comprehensive_mock_response();
        let (providers, models, model_to_provider) =
            registry.transform_api_data(mock_response).await.unwrap();

        {
            let mut providers_lock = registry.providers.write().await;
            *providers_lock = providers;
        }

        {
            let mut models_lock = registry.models.write().await;
            *models_lock = models;
        }

        {
            let mut mapping_lock = registry.model_to_provider.write().await;
            *mapping_lock = model_to_provider;
        }

        // Scenario: Developer wants to use OpenAI for a chat application
        println!("=== Real-world scenario: Chat application with OpenAI ===");

        // 1. Find OpenAI provider
        let openai_provider = find_provider_for_cloud_service(&registry, "openai").await;
        assert_eq!(openai_provider, Some("openai".to_string()));
        println!("✓ Found OpenAI provider: {}", openai_provider.unwrap());

        // 2. Check which OpenAI models are available
        let openai_models = registry.get_models_for_provider("openai").await;
        assert_eq!(openai_models.len(), 3);
        println!("✓ OpenAI has {} models available", openai_models.len());

        // 3. Get the best model for chat
        let chat_model = find_best_model_for_use_case(&registry, "chat").await;
        assert_eq!(chat_model, Some("gpt-4".to_string()));
        println!("✓ Recommended model for chat: {}", chat_model.unwrap());

        // 4. Check if OpenAI provider is configured
        let config_check = check_provider_configuration(&registry, "openai").await;
        assert!(config_check.is_err());
        println!("✓ Configuration check failed as expected (no API key set)");

        // 5. Get connection information
        let connection_info = registry.get_connection_info("openai").await;
        assert!(connection_info.is_some());
        let conn_info = connection_info.unwrap();
        println!("✓ Connection info - Base URL: {}", conn_info.base_url);
        println!("✓ Required env vars: {:?}", conn_info.required_env_vars);

        // Scenario: Developer wants to use vision capabilities
        println!("\n=== Real-world scenario: Vision capabilities ===");

        // 1. Find models with vision capability
        let vision_models = find_models_with_capability(&registry, "vision").await;
        assert_eq!(vision_models.len(), 1);
        println!(
            "✓ Found {} models with vision capability",
            vision_models.len()
        );

        // 2. Get the provider for the vision model
        if let Some(vision_model) = vision_models.first() {
            let provider_id = registry.find_provider_for_model(vision_model).await;
            assert_eq!(provider_id, Some("anthropic".to_string()));
            println!(
                "✓ Vision model {} is provided by {}",
                vision_model,
                provider_id.unwrap()
            );
        }

        // 3. Check capability summary
        let capability_summary = get_capability_summary(&registry).await;
        assert!(capability_summary.contains_key("vision"));
        println!(
            "✓ Vision capability available with {} models",
            capability_summary["vision"].len()
        );

        // Scenario: Developer wants to compare providers
        println!("\n=== Real-world scenario: Provider comparison ===");

        // 1. Get all providers with their models
        let providers_summary = get_providers_summary(&registry).await;
        assert_eq!(providers_summary.len(), 3);
        println!("✓ Found {} providers", providers_summary.len());

        for (provider_id, provider_name, model_ids) in providers_summary {
            println!(
                "  - {} ({}): {} models",
                provider_name,
                provider_id,
                model_ids.len()
            );
        }

        // 2. Find reasoning models across all providers
        let reasoning_models = find_models_with_capability(&registry, "reasoning").await;
        assert_eq!(reasoning_models.len(), 2);
        println!(
            "✓ Found {} models with reasoning capability",
            reasoning_models.len()
        );

        // 3. Get the best model for reasoning
        let reasoning_model = find_best_model_for_use_case(&registry, "reasoning").await;
        assert_eq!(reasoning_model, Some("gpt-4".to_string()));
        println!(
            "✓ Recommended model for reasoning: {}",
            reasoning_model.unwrap()
        );
    }
}
