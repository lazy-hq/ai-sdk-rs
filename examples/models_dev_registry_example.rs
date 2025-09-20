//! Example demonstrating how to use the models.dev provider registry.
//!
//! This example shows how to:
//! - Create a provider registry
//! - Refresh data from the models.dev API
//! - Look up providers and models
//! - Find which provider offers a specific model
//! - Use convenience functions for common operations

#[cfg(feature = "models-dev")]
#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    use aisdk::models_dev::{
        ProviderRegistry, check_provider_configuration, find_best_model_for_use_case,
        find_models_with_capability, find_provider_for_cloud_service, get_capability_summary,
        get_providers_summary, list_providers_for_npm_package,
    };

    // Create a new provider registry
    println!("Creating provider registry...");
    let registry = ProviderRegistry::with_default_client();

    // Refresh the registry data from the API
    println!("Refreshing registry data...");
    match registry.refresh().await {
        Ok((provider_count, model_count)) => {
            println!(
                "Successfully loaded {} providers and {} models",
                provider_count, model_count
            );
        }
        Err(e) => {
            eprintln!("Failed to refresh registry: {}", e);
            return Err(e.into());
        }
    }

    // === Basic Registry Operations ===
    println!("\n=== Basic Registry Operations ===");

    // List all available providers
    println!("Available providers:");
    let providers = registry.get_all_providers().await;
    for provider in &providers {
        println!("  - {} ({})", provider.name, provider.id);
        println!("    Models: {}", provider.model_ids.len());
        println!("    Base URL: {}", provider.base_url);
        println!("    Available: {}", provider.available);
    }

    // List all available models
    println!("\nAvailable models:");
    let models = registry.get_all_models().await;
    for model in &models {
        println!("  - {} ({})", model.name, model.id);
        println!("    Provider: {}", model.provider_id);
        println!(
            "    Cost: ${}/1M input tokens, ${}/1M output tokens",
            model.cost.input, model.cost.output
        );
        println!(
            "    Context: {} tokens, Output: {} tokens",
            model.limits.context, model.limits.output
        );
    }

    // === Convenience Functions ===
    println!("\n=== Convenience Functions ===");

    // Find providers by cloud service name
    println!("Finding providers by cloud service:");
    let cloud_services = vec!["openai", "anthropic", "google", "claude", "gemini"];
    for service in cloud_services {
        match find_provider_for_cloud_service(&registry, service).await {
            Some(provider_id) => {
                println!("  {} -> {}", service, provider_id);
            }
            None => {
                println!("  {} -> No provider found", service);
            }
        }
    }

    // List providers for npm packages
    println!("\nProviders for npm packages:");
    let npm_packages = vec!["@ai-sdk/openai", "@ai-sdk/anthropic", "@ai-sdk/google"];
    for package in npm_packages {
        let providers = list_providers_for_npm_package(&registry, package).await;
        println!("  {} -> {} providers", package, providers.len());
        for provider_id in providers {
            println!("    - {}", provider_id);
        }
    }

    // Find models with specific capabilities
    println!("\nModels with specific capabilities:");
    let capabilities = vec!["reasoning", "vision", "tool_call", "attachment"];
    for capability in capabilities {
        let models = find_models_with_capability(&registry, capability).await;
        println!("  {} -> {} models", capability, models.len());
        for model in models {
            println!("    - {}", model.id);
        }
    }

    // Get providers summary
    println!("\nProviders summary:");
    let summary = get_providers_summary(&registry).await;
    for (provider_id, provider_name, model_ids) in summary {
        println!(
            "  {} ({}): {} models",
            provider_name,
            provider_id,
            model_ids.len()
        );
        for model_id in model_ids {
            println!("    - {}", model_id);
        }
    }

    // Find best models for different use cases
    println!("\nBest models for use cases:");
    let use_cases = vec!["chat", "code", "reasoning", "vision", "fast", "cheap"];
    for use_case in use_cases {
        match find_best_model_for_use_case(&registry, use_case).await {
            Some(model_id) => {
                println!("  {} -> {}", use_case, model_id);
            }
            None => {
                println!("  {} -> No model found", use_case);
            }
        }
    }

    // Check provider configurations
    println!("\nProvider configuration checks:");
    let provider_ids = vec!["openai", "anthropic", "google"];
    for provider_id in provider_ids {
        match check_provider_configuration(&registry, provider_id).await {
            Ok(()) => {
                println!("  {} -> ✓ Properly configured", provider_id);
            }
            Err(missing_vars) => {
                println!(
                    "  {} -> ✗ Missing variables: {:?}",
                    provider_id, missing_vars
                );
            }
        }
    }

    // Get capability summary
    println!("\nCapability summary:");
    let capability_summary = get_capability_summary(&registry).await;
    for (capability, model_ids) in capability_summary {
        println!("  {} -> {} models", capability, model_ids.len());
        for model_id in model_ids {
            println!("    - {}", model_id);
        }
    }

    // === Advanced Registry Operations ===
    println!("\n=== Advanced Registry Operations ===");

    // Find providers by npm package
    println!("\nProviders supporting @ai-sdk/openai:");
    let openai_providers = registry.find_providers_by_npm("@ai-sdk/openai").await;
    for provider_id in openai_providers {
        if let Some(provider) = registry.get_provider(&provider_id).await {
            println!("  - {} ({})", provider.name, provider.id);
        } else {
            println!("  - {}", provider_id);
        }
    }

    // Get models with reasoning capability
    println!("\nModels with reasoning capability:");
    let reasoning_models = registry.get_models_with_capability("reasoning").await;
    for model in reasoning_models {
        println!(
            "  - {} ({}) - Provider: {}",
            model.name, model.id, model.provider_id
        );
        if let Some(reasoning_cost) = model.cost.reasoning {
            println!("    Reasoning cost: ${}/1M tokens", reasoning_cost);
        }
    }

    // Get providers with their models
    println!("\nProviders with their models:");
    let providers_with_models = registry.get_providers_with_models().await;
    for (provider_id, provider, models) in providers_with_models {
        println!("  {} ({}):", provider.name, provider_id);
        for model in models {
            println!("    - {} ({})", model.name, model.id);
        }
    }

    // Get connection information
    println!("\nConnection information:");
    if let Some(connection_info) = registry.get_connection_info("openai").await {
        println!("  OpenAI connection info:");
        println!("    Base URL: {}", connection_info.base_url);
        println!(
            "    Required env vars: {:?}",
            connection_info.required_env_vars
        );
        println!(
            "    Optional env vars: {:?}",
            connection_info.optional_env_vars
        );
        println!("    Config: {:?}", connection_info.config);
    }

    // === Detailed Provider and Model Lookups ===
    println!("\n=== Detailed Provider and Model Lookups ===");

    // Demonstrate provider lookup
    if let Some(provider) = registry.get_provider("openai").await {
        println!("\nOpenAI provider details:");
        println!("  Name: {}", provider.name);
        println!("  Base URL: {}", provider.base_url);
        println!(
            "  NPM Package: {}@{}",
            provider.npm_name, provider.npm_version
        );
        println!("  Documentation: {}", provider.doc_url);
        println!("  Models: {}", provider.model_ids.len());

        // List OpenAI models
        println!("  OpenAI models:");
        let openai_models = registry.get_models_for_provider("openai").await;
        for model in &openai_models {
            println!("    - {} ({})", model.name, model.id);
        }
    }

    // Demonstrate model lookup
    if let Some(model) = registry.get_model("gpt-4").await {
        println!("\nGPT-4 model details:");
        println!("  Name: {}", model.name);
        println!("  Provider: {}", model.provider_id);
        println!("  Description: {}", model.description);
        println!(
            "  Cost: ${}/1M input, ${}/1M output",
            model.cost.input, model.cost.output
        );
        println!("  Context: {} tokens", model.limits.context);
        println!("  Input modalities: {:?}", model.input_modalities);
        println!("  Output modalities: {:?}", model.output_modalities);
    }

    // Demonstrate provider finding for models
    println!("\nFinding providers for models:");
    let test_models = vec!["gpt-4", "claude-3-opus", "gemini-pro", "unknown-model"];
    for model_id in test_models {
        match registry.find_provider_for_model(model_id).await {
            Some(provider_id) => {
                println!("  {} -> {}", model_id, provider_id);
            }
            None => {
                println!("  {} -> No provider found", model_id);
            }
        }
    }

    // Demonstrate availability checks
    println!("\nAvailability checks:");
    println!(
        "  OpenAI available: {}",
        registry.is_provider_available("openai").await
    );
    println!(
        "  GPT-4 available: {}",
        registry.is_model_available("gpt-4").await
    );
    println!(
        "  Unknown provider available: {}",
        registry.is_provider_available("unknown").await
    );
    println!(
        "  Unknown model available: {}",
        registry.is_model_available("unknown").await
    );

    // Show registry statistics
    println!("\nRegistry statistics:");
    println!("  Total providers: {}", registry.provider_count().await);
    println!("  Total models: {}", registry.model_count().await);

    Ok(())
}

#[cfg(not(feature = "models-dev"))]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("This example requires the 'models-dev' feature to be enabled.");
    println!("Run with: cargo run --example models_dev_registry_example --features models-dev");
    Ok(())
}
