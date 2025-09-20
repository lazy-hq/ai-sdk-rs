//! Tests for OpenAI provider ModelsDevAware integration.

use aisdk::models_dev::traits::ModelsDevAware;
use aisdk::models_dev::types::{
    ApiInfo, DocInfo, EnvVar, Modalities, Model, ModelCost, ModelLimit, NpmInfo, Provider,
};
use std::collections::HashMap;

fn create_test_openai_provider() -> Provider {
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
                description: "OpenAI API key".to_string(),
                required: true,
            },
            EnvVar {
                name: "OPENAI_ORGANIZATION".to_string(),
                description: "OpenAI organization ID".to_string(),
                required: false,
            },
        ],
        doc: DocInfo {
            url: "https://platform.openai.com/docs".to_string(),
            metadata: HashMap::new(),
        },
        api: ApiInfo {
            base_url: "https://api.openai.com/v1".to_string(),
            version: None,
            config: HashMap::new(),
        },
        models: vec![
            Model {
                id: "gpt-4o".to_string(),
                name: "GPT-4o".to_string(),
                description: "Latest GPT-4 model".to_string(),
                cost: ModelCost::default(),
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
                id: "gpt-4o-mini".to_string(),
                name: "GPT-4o Mini".to_string(),
                description: "Smaller GPT-4 model".to_string(),
                cost: ModelCost::default(),
                limits: ModelLimit {
                    context: 128000,
                    output: 16384,
                    metadata: HashMap::new(),
                },
                modalities: Modalities {
                    input: vec!["text".to_string()],
                    output: vec!["text".to_string()],
                },
                metadata: HashMap::new(),
            },
        ],
    }
}

#[test]
fn test_openai_supported_npm_packages() {
    let packages = aisdk::providers::openai::OpenAI::supported_npm_packages();
    assert_eq!(packages, vec!["@ai-sdk/openai".to_string()]);
}

#[test]
fn test_openai_from_models_dev_info_with_env() {
    // Set up environment variable for testing
    unsafe {
        std::env::set_var("OPENAI_API_KEY", "test-api-key");
    }

    let provider = create_test_openai_provider();
    let openai = aisdk::providers::openai::OpenAI::from_models_dev_info(&provider, None);

    assert!(openai.is_some());
    let openai_instance = openai.unwrap();
    assert_eq!(openai_instance.settings.model_name, "gpt-4o");
    assert_eq!(openai_instance.settings.api_key, "test-api-key");
    assert_eq!(openai_instance.settings.provider_name, "OpenAI");

    // Cleanup
    unsafe {
        std::env::remove_var("OPENAI_API_KEY");
    }
}

#[test]
fn test_openai_from_models_dev_info_with_specific_model() {
    // Set up environment variable for testing
    unsafe {
        std::env::set_var("OPENAI_API_KEY", "test-api-key");
    }

    let provider = create_test_openai_provider();
    let specific_model = provider.models.first().unwrap();
    let openai =
        aisdk::providers::openai::OpenAI::from_models_dev_info(&provider, Some(specific_model));

    assert!(openai.is_some());
    let openai_instance = openai.unwrap();
    assert_eq!(openai_instance.settings.model_name, "gpt-4o");
    assert_eq!(openai_instance.settings.api_key, "test-api-key");

    // Cleanup
    unsafe {
        std::env::remove_var("OPENAI_API_KEY");
    }
}

#[test]
fn test_openai_from_models_dev_info_wrong_npm_package() {
    // Set up environment variable for testing
    unsafe {
        std::env::set_var("OPENAI_API_KEY", "test-api-key");
    }

    let mut provider = create_test_openai_provider();
    provider.npm.name = "@ai-sdk/anthropic".to_string(); // Wrong package
    let openai = aisdk::providers::openai::OpenAI::from_models_dev_info(&provider, None);

    assert!(openai.is_none());

    // Cleanup
    unsafe {
        std::env::remove_var("OPENAI_API_KEY");
    }
}

#[test]
fn test_openai_from_models_dev_info_missing_env() {
    // Ensure OPENAI_API_KEY is not set
    unsafe {
        std::env::remove_var("OPENAI_API_KEY");
    }

    let provider = create_test_openai_provider();
    let openai = aisdk::providers::openai::OpenAI::from_models_dev_info(&provider, None);

    // The provider should NOT be created when API key is missing
    assert!(openai.is_none());
}

#[test]
fn test_openai_connection_info() {
    let provider = create_test_openai_provider();
    let info = aisdk::providers::openai::OpenAI::connection_info(&provider);

    assert_eq!(info.base_url, "https://api.openai.com/v1");
    assert_eq!(info.required_env_vars, vec!["OPENAI_API_KEY"]);
    assert_eq!(info.optional_env_vars, vec!["OPENAI_ORGANIZATION"]);
    assert!(info.config.is_empty());
}

#[test]
fn test_openai_supports_model() {
    let provider = create_test_openai_provider();

    assert!(aisdk::providers::openai::OpenAI::supports_model(
        &provider, "gpt-4o"
    ));
    assert!(aisdk::providers::openai::OpenAI::supports_model(
        &provider,
        "gpt-4o-mini"
    ));
    assert!(!aisdk::providers::openai::OpenAI::supports_model(
        &provider,
        "unknown-model"
    ));
}

#[test]
fn test_openai_get_model() {
    let provider = create_test_openai_provider();

    let model = aisdk::providers::openai::OpenAI::get_model(&provider, "gpt-4o");
    assert!(model.is_some());
    assert_eq!(model.unwrap().id, "gpt-4o");

    let missing_model = aisdk::providers::openai::OpenAI::get_model(&provider, "unknown-model");
    assert!(missing_model.is_none());
}
