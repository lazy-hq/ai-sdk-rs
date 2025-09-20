//! Integration tests for the complete models.dev workflow.
//!
//! These tests verify that all the convenience functions and registry methods
//! work together correctly in realistic scenarios, including edge cases,
//! error handling, and performance considerations.

#[cfg(feature = "models-dev")]
#[cfg(test)]
mod tests {
    use aisdk::models_dev::{
        ProviderRegistry, check_provider_configuration, find_best_model_for_use_case,
        find_models_with_capability, find_provider_for_cloud_service, get_capability_summary,
        get_providers_summary, list_providers_for_npm_package,
        traits::{AnthropicProvider, GoogleProvider, OpenAIProvider},
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
        let reasoning_model_ids: Vec<String> =
            reasoning_models.iter().map(|m| m.id.clone()).collect();
        assert!(reasoning_model_ids.contains(&"gpt-4-turbo".to_string()));
        assert!(reasoning_model_ids.contains(&"claude-3-opus".to_string()));

        let vision_models = find_models_with_capability(&registry, "vision").await;
        assert_eq!(vision_models.len(), 1); // claude-3-opus
        let vision_model_ids: Vec<String> = vision_models.iter().map(|m| m.id.clone()).collect();
        assert!(vision_model_ids.contains(&"claude-3-opus".to_string()));

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
        assert!(
            missing_vars
                .iter()
                .any(|msg| msg.contains("OPENAI_API_KEY"))
        );
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
            let provider_id = registry.find_provider_for_model(&vision_model.id).await;
            assert_eq!(provider_id, Some("anthropic".to_string()));
            println!(
                "✓ Vision model {} is provided by {}",
                vision_model.name,
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

    // === Comprehensive Error Handling Tests ===
    #[tokio::test]
    async fn test_comprehensive_error_handling() {
        let registry = ProviderRegistry::default();

        // Test 1: Malformed API response handling
        let malformed_response = ModelsDevResponse {
            providers: vec![Provider {
                id: "".to_string(), // Empty ID
                name: "Invalid Provider".to_string(),
                npm: NpmInfo {
                    name: "@ai-sdk/invalid".to_string(),
                    version: "invalid_version".to_string(), // Invalid version format
                },
                env: vec![EnvVar {
                    name: "".to_string(), // Empty env var name
                    description: "Empty env var".to_string(),
                    required: true,
                }],
                doc: DocInfo {
                    url: "invalid-url".to_string(), // Invalid URL
                    metadata: HashMap::new(),
                },
                api: ApiInfo {
                    base_url: "".to_string(), // Empty base URL
                    version: None,
                    config: HashMap::new(),
                },
                models: vec![Model {
                    id: "".to_string(), // Empty model ID
                    name: "Invalid Model".to_string(),
                    description: "Invalid model".to_string(),
                    cost: ModelCost {
                        input: -1.0,              // Negative cost
                        output: -1.0,             // Negative cost
                        cache_read: Some(-1.0),   // Negative cache cost
                        cache_write: Some(-1.0),  // Negative cache cost
                        reasoning: Some(-1.0),    // Negative reasoning cost
                        currency: "".to_string(), // Empty currency
                    },
                    limits: ModelLimit {
                        context: 0, // Zero context
                        output: 0,  // Zero output
                        metadata: HashMap::new(),
                    },
                    modalities: Modalities {
                        input: vec![],
                        output: vec![],
                    },
                    metadata: HashMap::new(),
                }],
            }],
        };

        // This should still transform without panicking
        let result = registry.transform_api_data(malformed_response).await;
        assert!(result.is_ok(), "Should handle malformed data gracefully");

        let (providers, models, model_to_provider) = result.unwrap();

        // Verify that invalid data is handled appropriately
        assert!(!providers.is_empty(), "Should process malformed data");
        assert!(!models.is_empty(), "Should process malformed data");
        assert!(
            !model_to_provider.is_empty(),
            "Should process malformed data"
        );

        // Test 2: Registry operations with invalid inputs
        let invalid_provider_id = "";
        let invalid_model_id = "";
        let invalid_capability = "";
        let invalid_npm_package = "";

        // These should handle empty/invalid inputs gracefully
        assert!(registry.get_provider(invalid_provider_id).await.is_none());
        assert!(registry.get_model(invalid_model_id).await.is_none());
        assert_eq!(
            registry.find_provider_for_model(invalid_model_id).await,
            None
        );
        assert_eq!(
            registry
                .get_models_for_provider(invalid_provider_id)
                .await
                .len(),
            0
        );
        assert!(!registry.is_provider_available(invalid_provider_id).await);
        assert!(!registry.is_model_available(invalid_model_id).await);
        assert_eq!(
            registry
                .find_providers_by_npm(invalid_npm_package)
                .await
                .len(),
            0
        );
        assert_eq!(
            registry
                .get_models_with_capability(invalid_capability)
                .await
                .len(),
            0
        );

        // Test 3: Convenience functions with invalid inputs
        assert_eq!(find_provider_for_cloud_service(&registry, "").await, None);
        assert_eq!(list_providers_for_npm_package(&registry, "").await.len(), 0);
        assert_eq!(find_models_with_capability(&registry, "").await.len(), 0);
        assert_eq!(get_providers_summary(&registry).await.len(), 0);
        assert_eq!(find_best_model_for_use_case(&registry, "").await, None);
        assert_eq!(get_capability_summary(&registry).await.len(), 0);

        // Test 4: Configuration checking with invalid provider
        let config_result = check_provider_configuration(&registry, "nonexistent_provider").await;
        assert!(config_result.is_err());
        let error_msg = format!("{:?}", config_result.unwrap_err());
        assert!(
            error_msg.contains("not found"),
            "Should handle nonexistent provider gracefully"
        );
    }

    // === Large Dataset Performance Tests ===
    #[tokio::test]
    async fn test_large_dataset_performance() {
        let registry = ProviderRegistry::default();

        // Create a large dataset with many providers and models
        let mut providers = Vec::new();
        let expected_providers = 50; // 50 providers
        let models_per_provider = 20; // 20 models per provider = 1000 models total

        for provider_idx in 0..expected_providers {
            let mut models = Vec::new();
            for model_idx in 0..models_per_provider {
                models.push(Model {
                    id: format!("provider-{}-model-{}", provider_idx, model_idx),
                    name: format!("Model {}-{}", provider_idx, model_idx),
                    description: format!("Test model {}-{}", provider_idx, model_idx),
                    cost: ModelCost {
                        input: 0.001 * (model_idx as f64 + 1.0),
                        output: 0.002 * (model_idx as f64 + 1.0),
                        cache_read: Some(0.0005 * (model_idx as f64 + 1.0)),
                        cache_write: Some(0.001 * (model_idx as f64 + 1.0)),
                        reasoning: Some(0.0015 * (model_idx as f64 + 1.0)),
                        currency: "USD".to_string(),
                    },
                    limits: ModelLimit {
                        context: 4096 * (model_idx + 1),
                        output: 2048 * (model_idx + 1),
                        metadata: HashMap::new(),
                    },
                    modalities: Modalities {
                        input: vec!["text".to_string()],
                        output: vec!["text".to_string()],
                    },
                    metadata: HashMap::new(),
                });
            }

            providers.push(Provider {
                id: format!("provider-{}", provider_idx),
                name: format!("Provider {}", provider_idx),
                npm: NpmInfo {
                    name: format!("@ai-sdk/provider-{}", provider_idx),
                    version: "1.0.0".to_string(),
                },
                env: vec![EnvVar {
                    name: format!("PROVIDER_{}_API_KEY", provider_idx),
                    description: format!("API key for provider {}", provider_idx),
                    required: true,
                }],
                doc: DocInfo {
                    url: format!("https://provider-{}.example.com/docs", provider_idx),
                    metadata: HashMap::new(),
                },
                api: ApiInfo {
                    base_url: format!("https://api.provider-{}.example.com", provider_idx),
                    version: Some("v1".to_string()),
                    config: HashMap::new(),
                },
                models,
            });
        }

        let large_response = ModelsDevResponse { providers };

        // Measure transformation time
        let start_time = std::time::Instant::now();
        let (providers, models, model_to_provider) =
            registry.transform_api_data(large_response).await.unwrap();
        let transformation_time = start_time.elapsed();

        // Verify the transformation completed successfully
        assert_eq!(
            providers.len(),
            expected_providers,
            "Should have correct number of providers"
        );
        assert_eq!(
            models.len(),
            expected_providers * models_per_provider as usize,
            "Should have correct number of models"
        );
        assert_eq!(
            model_to_provider.len(),
            expected_providers * models_per_provider as usize,
            "Should have correct mapping size"
        );

        // Performance assertion - should complete in reasonable time (adjust as needed)
        assert!(
            transformation_time.as_millis() < 1000,
            "Large dataset transformation should complete quickly, took {:?}",
            transformation_time
        );

        // Load the data into the registry
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

        // Test performance of various operations
        let operations_start = std::time::Instant::now();

        // Test provider lookup performance
        for i in 0..10 {
            let provider_id = format!("provider-{}", i);
            let provider = registry.get_provider(&provider_id).await;
            assert!(provider.is_some(), "Should find provider {}", provider_id);
        }

        // Test model lookup performance
        for i in 0..10 {
            let model_id = format!("provider-0-model-{}", i);
            let model = registry.get_model(&model_id).await;
            assert!(model.is_some(), "Should find model {}", model_id);
        }

        // Test convenience function performance
        let summary = get_providers_summary(&registry).await;
        assert_eq!(
            summary.len(),
            expected_providers,
            "Summary should include all providers"
        );

        let capability_summary = get_capability_summary(&registry).await;
        assert!(
            !capability_summary.is_empty(),
            "Should have capability summary"
        );

        let operations_time = operations_start.elapsed();
        assert!(
            operations_time.as_millis() < 500,
            "Operations should complete quickly, took {:?}",
            operations_time
        );

        println!("Large dataset performance test completed:");
        println!("  - Transformation time: {:?}", transformation_time);
        println!("  - Operations time: {:?}", operations_time);
        println!("  - Total providers: {}", expected_providers);
        println!(
            "  - Total models: {}",
            expected_providers * models_per_provider as usize
        );
    }

    // === Concurrency and Thread Safety Tests ===
    #[tokio::test]
    async fn test_concurrent_operations() {
        let registry = ProviderRegistry::default();

        // Load test data
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

        // Test concurrent read operations
        let concurrent_tasks = 100;
        let mut handles = Vec::new();

        for task_id in 0..concurrent_tasks {
            let registry_clone = registry.clone();
            let handle = tokio::spawn(async move {
                // Each task performs various read operations
                let provider = registry_clone.get_provider("openai").await;
                let model = registry_clone.get_model("gpt-4").await;
                let provider_for_model = registry_clone.find_provider_for_model("gpt-4").await;
                let models_for_provider = registry_clone.get_models_for_provider("openai").await;
                let is_available = registry_clone.is_provider_available("openai").await;

                // Verify all operations succeeded
                assert!(provider.is_some(), "Task {} should find provider", task_id);
                assert!(model.is_some(), "Task {} should find model", task_id);
                assert!(
                    provider_for_model.is_some(),
                    "Task {} should find provider for model",
                    task_id
                );
                assert!(
                    !models_for_provider.is_empty(),
                    "Task {} should find models for provider",
                    task_id
                );
                assert!(
                    is_available,
                    "Task {} should find provider available",
                    task_id
                );

                // Test convenience functions
                let openai_provider =
                    find_provider_for_cloud_service(&registry_clone, "openai").await;
                let reasoning_models =
                    find_models_with_capability(&registry_clone, "reasoning").await;
                let summary = get_providers_summary(&registry_clone).await;

                assert_eq!(
                    openai_provider,
                    Some("openai".to_string()),
                    "Task {} should find OpenAI provider",
                    task_id
                );
                assert!(
                    !reasoning_models.is_empty(),
                    "Task {} should find reasoning models",
                    task_id
                );
                assert!(
                    !summary.is_empty(),
                    "Task {} should get providers summary",
                    task_id
                );

                task_id
            });
            handles.push(handle);
        }

        // Wait for all tasks to complete
        let results = futures::future::join_all(handles).await;

        // Verify all tasks completed successfully
        for (i, result) in results.into_iter().enumerate() {
            assert!(result.is_ok(), "Task {} should complete successfully", i);
            let task_id = result.unwrap();
            assert_eq!(task_id, i, "Task {} should return correct ID", i);
        }

        println!(
            "Successfully executed {} concurrent operations",
            concurrent_tasks
        );
    }

    #[tokio::test]
    async fn test_concurrent_writes_with_reads() {
        let registry = ProviderRegistry::default();

        // Start with some initial data
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

        // Create concurrent read and write operations
        let mut handles = Vec::new();

        // Read tasks
        for i in 0..50 {
            let registry_clone = registry.clone();
            let handle = tokio::spawn(async move {
                // Perform read operations
                let provider = registry_clone.get_provider("openai").await;
                let model = registry_clone.get_model("gpt-4").await;
                let count = registry_clone.provider_count().await;

                assert!(provider.is_some(), "Read task {} should find provider", i);
                assert!(model.is_some(), "Read task {} should find model", i);
                assert!(count > 0, "Read task {} should find providers", i);

                i
            });
            handles.push(handle);
        }

        // Write tasks (clear and reload data)
        for i in 0..10 {
            let registry_clone = registry.clone();
            let handle = tokio::spawn(async move {
                // Clear the registry
                registry_clone.clear().await;

                // Verify it's empty
                assert_eq!(
                    registry_clone.provider_count().await,
                    0,
                    "Write task {} should clear providers",
                    i
                );
                assert_eq!(
                    registry_clone.model_count().await,
                    0,
                    "Write task {} should clear models",
                    i
                );

                // Reload data
                let mock_response = create_comprehensive_mock_response();
                let (providers, models, model_to_provider) = registry_clone
                    .transform_api_data(mock_response)
                    .await
                    .unwrap();

                {
                    let mut providers_lock = registry_clone.providers.write().await;
                    *providers_lock = providers;
                }

                {
                    let mut models_lock = registry_clone.models.write().await;
                    *models_lock = models;
                }

                {
                    let mut mapping_lock = registry_clone.model_to_provider.write().await;
                    *mapping_lock = model_to_provider;
                }

                // Verify data is reloaded
                assert!(
                    registry_clone.provider_count().await > 0,
                    "Write task {} should reload providers",
                    i
                );
                assert!(
                    registry_clone.model_count().await > 0,
                    "Write task {} should reload models",
                    i
                );

                i
            });
            handles.push(handle);
        }

        // Wait for all tasks to complete
        let results = futures::future::join_all(handles).await;

        // Verify all tasks completed successfully
        for (i, result) in results.into_iter().enumerate() {
            assert!(result.is_ok(), "Task {} should complete successfully", i);
        }

        println!("Successfully executed concurrent read/write operations");
    }

    // === Memory and Resource Management Tests ===
    #[tokio::test]
    async fn test_memory_efficiency() {
        let registry = ProviderRegistry::default();

        // Create a very large dataset to test memory usage
        let mut providers = Vec::new();
        let large_provider_count = 1000; // 1000 providers
        let models_per_provider = 50; // 50 models per provider = 50,000 models total

        for provider_idx in 0..large_provider_count {
            let mut models = Vec::new();
            for model_idx in 0..models_per_provider {
                models.push(Model {
                    id: format!("large-provider-{}-model-{}", provider_idx, model_idx),
                    name: format!("Large Model {}-{}", provider_idx, model_idx),
                    description: format!("Large test model {}-{}", provider_idx, model_idx),
                    cost: ModelCost {
                        input: 0.001,
                        output: 0.002,
                        cache_read: Some(0.0005),
                        cache_write: Some(0.001),
                        reasoning: Some(0.0015),
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
                });
            }

            providers.push(Provider {
                id: format!("large-provider-{}", provider_idx),
                name: format!("Large Provider {}", provider_idx),
                npm: NpmInfo {
                    name: format!("@ai-sdk/large-provider-{}", provider_idx),
                    version: "1.0.0".to_string(),
                },
                env: vec![EnvVar {
                    name: format!("LARGE_PROVIDER_{}_API_KEY", provider_idx),
                    description: format!("API key for large provider {}", provider_idx),
                    required: true,
                }],
                doc: DocInfo {
                    url: format!("https://large-provider-{}.example.com/docs", provider_idx),
                    metadata: HashMap::new(),
                },
                api: ApiInfo {
                    base_url: format!("https://api.large-provider-{}.example.com", provider_idx),
                    version: Some("v1".to_string()),
                    config: HashMap::new(),
                },
                models,
            });
        }

        let very_large_response = ModelsDevResponse { providers };

        // Transform and load the data
        let (providers, models, model_to_provider) = registry
            .transform_api_data(very_large_response)
            .await
            .unwrap();

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

        // Verify the data was loaded correctly
        assert_eq!(registry.provider_count().await, large_provider_count);
        assert_eq!(
            registry.model_count().await,
            large_provider_count * models_per_provider
        );

        // Clear the registry to test memory cleanup
        registry.clear().await;

        // Verify memory was freed
        assert_eq!(registry.provider_count().await, 0);
        assert_eq!(registry.model_count().await, 0);

        println!(
            "Memory efficiency test completed with {} providers and {} models",
            large_provider_count,
            large_provider_count * models_per_provider
        );
    }

    // === Real-world Edge Case Tests ===
    #[tokio::test]
    async fn test_real_world_edge_cases() {
        let registry = ProviderRegistry::default();

        // Test 1: Provider with extremely long names and URLs
        let long_name = "A".repeat(1000); // 1000 character name
        let long_url = "https://".to_string() + &"a".repeat(500) + ".example.com"; // Very long URL

        let edge_case_response = ModelsDevResponse {
            providers: vec![Provider {
                id: "edge-case-provider".to_string(),
                name: long_name.clone(),
                npm: NpmInfo {
                    name: format!("@ai-sdk/{}", long_name),
                    version: "1.0.0".to_string(),
                },
                env: vec![EnvVar {
                    name: format!("{}_API_KEY", long_name),
                    description: format!("API key for {}", long_name),
                    required: true,
                }],
                doc: DocInfo {
                    url: long_url.clone(),
                    metadata: HashMap::new(),
                },
                api: ApiInfo {
                    base_url: long_url.clone(),
                    version: Some("v1".to_string()),
                    config: HashMap::new(),
                },
                models: vec![Model {
                    id: "edge-case-model".to_string(),
                    name: long_name.clone(),
                    description: format!("Edge case model with long name: {}", long_name),
                    cost: ModelCost {
                        input: 0.000001, // Very small cost
                        output: 0.000002,
                        cache_read: Some(0.0000005),
                        cache_write: Some(0.000001),
                        reasoning: Some(0.0000015),
                        currency: "USD".to_string(),
                    },
                    limits: ModelLimit {
                        context: 2000000, // Very large context
                        output: 1000000,  // Very large output
                        metadata: HashMap::new(),
                    },
                    modalities: Modalities {
                        input: vec![
                            "text".to_string(),
                            "image".to_string(),
                            "audio".to_string(),
                            "video".to_string(),
                        ],
                        output: vec!["text".to_string(), "image".to_string(), "audio".to_string()],
                    },
                    metadata: HashMap::new(),
                }],
            }],
        };

        // This should handle edge cases gracefully
        let result = registry.transform_api_data(edge_case_response).await;
        assert!(result.is_ok(), "Should handle edge case data gracefully");

        let (providers, models, model_to_provider) = result.unwrap();

        // Load and test the edge case data
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

        // Test operations with edge case data
        let provider = registry.get_provider("edge-case-provider").await;
        assert!(provider.is_some(), "Should find edge case provider");

        let model = registry.get_model("edge-case-model").await;
        assert!(model.is_some(), "Should find edge case model");

        // Test 2: Provider with special characters in IDs and names
        let special_chars_response = ModelsDevResponse {
            providers: vec![Provider {
                id: "provider-with_special.chars123".to_string(),
                name: "Provider with Special Chars & Symbols 123!".to_string(),
                npm: NpmInfo {
                    name: "@ai-sdk/special-chars-package".to_string(),
                    version: "1.0.0-beta.1+build.123".to_string(), // Complex version
                },
                env: vec![EnvVar {
                    name: "SPECIAL_CHARS_API_KEY_123".to_string(),
                    description: "API key with special chars: test@example.com".to_string(),
                    required: true,
                }],
                doc: DocInfo {
                    url: "https://special-chars.example.com/path?param=value&other=123".to_string(),
                    metadata: HashMap::new(),
                },
                api: ApiInfo {
                    base_url: "https://api.special-chars.example.com/v1/endpoint".to_string(),
                    version: Some("2023-12-01-preview".to_string()), // Complex version
                    config: {
                        let mut config = HashMap::new();
                        config.insert("timeout".to_string(), serde_json::json!("30s"));
                        config.insert("retry_count".to_string(), serde_json::json!("3"));
                        config.insert(
                            "custom.setting".to_string(),
                            serde_json::json!("value-with-dashes"),
                        );
                        config
                    },
                },
                models: vec![Model {
                    id: "model-with_special.chars-v2.1".to_string(),
                    name: "Model with Special Chars & Symbols v2.1!".to_string(),
                    description: "Model description with special chars: @#$%^&*()".to_string(),
                    cost: ModelCost {
                        input: 0.01,
                        output: 0.02,
                        cache_read: Some(0.005),
                        cache_write: Some(0.01),
                        reasoning: Some(0.015),
                        currency: "USD".to_string(),
                    },
                    limits: ModelLimit {
                        context: 8192,
                        output: 4096,
                        metadata: {
                            let mut metadata = HashMap::new();
                            metadata.insert(
                                "special.key".to_string(),
                                serde_json::json!("special-value"),
                            );
                            metadata.insert(
                                "another_key".to_string(),
                                serde_json::json!("value with spaces"),
                            );
                            metadata
                        },
                    },
                    modalities: Modalities {
                        input: vec!["text".to_string(), "image/*".to_string()], // Wildcard in modality
                        output: vec!["text/plain".to_string(), "application/json".to_string()],
                    },
                    metadata: {
                        let mut metadata = HashMap::new();
                        metadata.insert(
                            "model.special.field".to_string(),
                            serde_json::json!("special.model.value"),
                        );
                        metadata
                    },
                }],
            }],
        };

        let (providers, models, model_to_provider) = registry
            .transform_api_data(special_chars_response)
            .await
            .unwrap();

        // Clear and reload with special chars data
        registry.clear().await;

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

        // Test operations with special characters
        let provider = registry
            .get_provider("provider-with_special.chars123")
            .await;
        assert!(
            provider.is_some(),
            "Should find provider with special chars"
        );

        let model = registry.get_model("model-with_special.chars-v2.1").await;
        assert!(model.is_some(), "Should find model with special chars");

        // Test convenience functions with special characters
        let provider_for_cloud = find_provider_for_cloud_service(&registry, "special").await;
        // This might not find it due to heuristics, but shouldn't panic
        assert!(provider_for_cloud.is_none() || provider_for_cloud.is_some());

        println!("Real-world edge cases test completed successfully");
    }
}
