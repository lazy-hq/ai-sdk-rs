//! Unit tests for models.dev data structures and traits.

#[cfg(feature = "models-dev")]
mod tests {
    use aisdk::models_dev::{
        traits::ProviderConnectionInfo,
        types::{
            ApiInfo, DocInfo, EnvVar, Modalities, Model, ModelCost, ModelLimit, ModelsDevResponse,
            NpmInfo, Provider,
        },
    };

    #[test]
    fn test_models_dev_response_creation() {
        let response = ModelsDevResponse { providers: vec![] };

        assert!(response.providers.is_empty());
    }

    #[test]
    fn test_provider_creation() {
        let provider = Provider {
            id: "test-provider".to_string(),
            name: "Test Provider".to_string(),
            npm: NpmInfo {
                name: "@ai-sdk/test".to_string(),
                version: "1.0.0".to_string(),
            },
            env: vec![],
            doc: DocInfo {
                url: "https://test.example.com/docs".to_string(),
                metadata: std::collections::HashMap::new(),
            },
            api: ApiInfo {
                base_url: "https://api.test.example.com".to_string(),
                version: None,
                config: std::collections::HashMap::new(),
            },
            models: vec![],
        };

        assert_eq!(provider.id, "test-provider");
        assert_eq!(provider.name, "Test Provider");
        assert_eq!(provider.npm.name, "@ai-sdk/test");
        assert_eq!(provider.api.base_url, "https://api.test.example.com");
    }

    #[test]
    fn test_model_cost_creation() {
        let cost = ModelCost {
            input: 0.01,
            output: 0.02,
            cache_read: Some(0.005),
            cache_write: Some(0.01),
            reasoning: Some(0.015),
            currency: "USD".to_string(),
        };

        assert_eq!(cost.input, 0.01);
        assert_eq!(cost.output, 0.02);
        assert_eq!(cost.cache_read, Some(0.005));
        assert_eq!(cost.cache_write, Some(0.01));
        assert_eq!(cost.reasoning, Some(0.015));
        assert_eq!(cost.currency, "USD");
    }

    #[test]
    fn test_model_cost_default() {
        let cost = ModelCost::default();

        assert_eq!(cost.input, 0.0);
        assert_eq!(cost.output, 0.0);
        assert_eq!(cost.cache_read, None);
        assert_eq!(cost.cache_write, None);
        assert_eq!(cost.reasoning, None);
        assert_eq!(cost.currency, "USD");
    }

    #[test]
    fn test_model_limit_creation() {
        let limit = ModelLimit {
            context: 8192,
            output: 4096,
            metadata: std::collections::HashMap::new(),
        };

        assert_eq!(limit.context, 8192);
        assert_eq!(limit.output, 4096);
    }

    #[test]
    fn test_modalities_creation() {
        let modalities = Modalities {
            input: vec!["text".to_string(), "image".to_string()],
            output: vec!["text".to_string()],
        };

        assert_eq!(modalities.input, vec!["text", "image"]);
        assert_eq!(modalities.output, vec!["text"]);
    }

    #[test]
    fn test_provider_connection_info_creation() {
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
            .with_required_env("TEST_VAR_UNIQUE")
            .with_optional_env("OPTIONAL_VAR");

        // Test with missing required env var
        let result = info.validate_env();
        assert!(result.is_err());
        assert_eq!(result.unwrap_err(), vec!["TEST_VAR_UNIQUE"]);

        // Test with all required env vars set
        unsafe {
            std::env::set_var("TEST_VAR_UNIQUE", "test_value");
        }
        let result = info.validate_env();
        assert!(result.is_ok());

        // Cleanup
        unsafe {
            std::env::remove_var("TEST_VAR_UNIQUE");
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
    fn test_full_provider_serialization() {
        let provider = Provider {
            id: "openai".to_string(),
            name: "OpenAI".to_string(),
            npm: NpmInfo {
                name: "@ai-sdk/openai".to_string(),
                version: "1.0.0".to_string(),
            },
            env: vec![EnvVar {
                name: "OPENAI_API_KEY".to_string(),
                description: "Your OpenAI API key".to_string(),
                required: true,
            }],
            doc: DocInfo {
                url: "https://platform.openai.com/docs".to_string(),
                metadata: std::collections::HashMap::new(),
            },
            api: ApiInfo {
                base_url: "https://api.openai.com/v1".to_string(),
                version: Some("v1".to_string()),
                config: std::collections::HashMap::new(),
            },
            models: vec![Model {
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
                    metadata: std::collections::HashMap::new(),
                },
                modalities: Modalities {
                    input: vec!["text".to_string()],
                    output: vec!["text".to_string()],
                },
                metadata: std::collections::HashMap::new(),
            }],
        };

        let json = serde_json::to_string(&provider).unwrap();
        let deserialized: Provider = serde_json::from_str(&json).unwrap();

        assert_eq!(deserialized.id, "openai");
        assert_eq!(deserialized.name, "OpenAI");
        assert_eq!(deserialized.npm.name, "@ai-sdk/openai");
        assert_eq!(deserialized.models.len(), 1);
        assert_eq!(deserialized.models[0].id, "gpt-4");
        assert_eq!(deserialized.models[0].cost.input, 0.03);
        assert_eq!(deserialized.models[0].limits.context, 8192);
    }

    #[test]
    fn test_models_dev_response_serialization() {
        let response = ModelsDevResponse {
            providers: vec![Provider {
                id: "test-provider".to_string(),
                name: "Test Provider".to_string(),
                npm: NpmInfo {
                    name: "@ai-sdk/test".to_string(),
                    version: "1.0.0".to_string(),
                },
                env: vec![],
                doc: DocInfo {
                    url: "https://test.example.com/docs".to_string(),
                    metadata: std::collections::HashMap::new(),
                },
                api: ApiInfo {
                    base_url: "https://api.test.example.com".to_string(),
                    version: None,
                    config: std::collections::HashMap::new(),
                },
                models: vec![],
            }],
        };

        let json = serde_json::to_string(&response).unwrap();
        let deserialized: ModelsDevResponse = serde_json::from_str(&json).unwrap();

        assert_eq!(deserialized.providers.len(), 1);
        assert_eq!(deserialized.providers[0].id, "test-provider");
    }
}
