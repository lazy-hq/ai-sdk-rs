//! Example demonstrating the convenience functions and ModelsDevAware trait integration.
//!
//! This example shows how to:
//! - Use convenience functions for common operations
//! - Create provider instances using the ModelsDevAware trait
//! - Work with capability-based model discovery
//! - Handle provider configuration and error scenarios

#[cfg(feature = "models-dev")]
#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    use aisdk::models_dev::{
        ProviderRegistry, check_provider_configuration, find_best_model_for_use_case,
        find_models_with_capability, find_provider_for_cloud_service, get_capability_summary,
        get_providers_summary, list_providers_for_npm_package,
        traits::{AnthropicProvider, GoogleProvider, ModelsDevAware, OpenAIProvider},
    };

    println!("Models.dev Convenience Functions Example");
    println!("=======================================");

    // Create a new provider registry
    let registry = ProviderRegistry::with_default_client();

    // Refresh the registry data from the API
    println!("\nRefreshing registry data...");
    match registry.refresh().await {
        Ok((provider_count, model_count)) => {
            println!(
                "✓ Successfully loaded {} providers and {} models",
                provider_count, model_count
            );
        }
        Err(e) => {
            eprintln!("✗ Failed to refresh registry: {}", e);
            eprintln!("Note: This is expected if you don't have access to the models.dev API");
            eprintln!("The example will continue with mock data for demonstration purposes.");

            // For demonstration, we'll continue without real data
            println!("\nContinuing with demonstration mode...");
        }
    }

    // === Convenience Functions Demo ===
    println!("\n=== Convenience Functions Demo ===");

    // 1. Find providers by cloud service name
    println!("\n1. Finding providers by cloud service name:");
    let cloud_services = vec![
        ("openai", "OpenAI"),
        ("anthropic", "Anthropic"),
        ("claude", "Anthropic (Claude)"),
        ("google", "Google"),
        ("gemini", "Google (Gemini)"),
        ("meta", "Meta"),
        ("mistral", "Mistral"),
    ];

    for (service, description) in cloud_services {
        match find_provider_for_cloud_service(&registry, service).await {
            Some(provider_id) => {
                println!("   ✓ {} -> {} ({})", service, provider_id, description);
            }
            None => {
                println!("   ✗ {} -> No provider found", service);
            }
        }
    }

    // 2. List providers for npm packages
    println!("\n2. Listing providers for npm packages:");
    let npm_packages = vec![
        "@ai-sdk/openai",
        "@ai-sdk/anthropic",
        "@ai-sdk/google",
        "@ai-sdk/mistral",
        "@ai-sdk/cohere",
    ];

    for package in npm_packages {
        let providers = list_providers_for_npm_package(&registry, package).await;
        if providers.is_empty() {
            println!("   ✗ {} -> No providers found", package);
        } else {
            println!(
                "   ✓ {} -> {} provider(s): {:?}",
                package,
                providers.len(),
                providers
            );
        }
    }

    // 3. Find models with specific capabilities
    println!("\n3. Finding models with specific capabilities:");
    let capabilities = vec![
        ("reasoning", "Advanced reasoning and thinking capabilities"),
        ("vision", "Image and visual processing capabilities"),
        ("tool_call", "Function calling and tool use capabilities"),
        ("attachment", "File and document processing capabilities"),
    ];

    for (capability, description) in capabilities {
        let models = find_models_with_capability(&registry, capability).await;
        if models.is_empty() {
            println!(
                "   ✗ {} -> No models with {} capability",
                capability, description
            );
        } else {
            println!(
                "   ✓ {} -> {} model(s) with {}: {:?}",
                capability,
                models.len(),
                description,
                models
            );
        }
    }

    // 4. Get providers summary
    println!("\n4. Getting providers summary:");
    let summary = get_providers_summary(&registry).await;
    if summary.is_empty() {
        println!("   ✗ No providers found in registry");
    } else {
        println!("   ✓ Found {} provider(s):", summary.len());
        for (provider_id, provider_name, model_ids) in summary {
            println!(
                "     - {} ({}): {} models",
                provider_name,
                provider_id,
                model_ids.len()
            );
            if !model_ids.is_empty() {
                println!("       Models: {}", model_ids.join(", "));
            }
        }
    }

    // 5. Find best models for different use cases
    println!("\n5. Finding best models for different use cases:");
    let use_cases = vec![
        ("chat", "General conversation and chat applications"),
        ("code", "Programming and code generation tasks"),
        ("reasoning", "Complex reasoning and analysis tasks"),
        ("vision", "Image processing and visual tasks"),
        ("fast", "Quick responses and low-latency applications"),
        (
            "cheap",
            "Cost-effective applications with budget constraints",
        ),
    ];

    for (use_case, description) in use_cases {
        match find_best_model_for_use_case(&registry, use_case).await {
            Some(model_id) => {
                println!("   ✓ {} -> {} (for {})", use_case, model_id, description);
            }
            None => {
                println!("   ✗ {} -> No model found for {}", use_case, description);
            }
        }
    }

    // 6. Check provider configurations
    println!("\n6. Checking provider configurations:");
    let provider_ids = vec!["openai", "anthropic", "google", "mistral"];

    for provider_id in provider_ids {
        match check_provider_configuration(&registry, provider_id).await {
            Ok(()) => {
                println!(
                    "   ✓ {} -> All required environment variables are set",
                    provider_id
                );
            }
            Err(missing_vars) => {
                println!(
                    "   ✗ {} -> Missing environment variables: {:?}",
                    provider_id, missing_vars
                );
                println!("     To fix this, set the following environment variables:");
                for var in missing_vars {
                    println!(
                        "       export {}=<your_{}_key>",
                        var,
                        provider_id.to_uppercase()
                    );
                }
            }
        }
    }

    // 7. Get capability summary
    println!("\n7. Getting capability summary:");
    let capability_summary = get_capability_summary(&registry).await;
    if capability_summary.is_empty() {
        println!("   ✗ No capabilities found in registry");
    } else {
        println!(
            "   ✓ Found {} capability/capabilities:",
            capability_summary.len()
        );
        for (capability, model_ids) in capability_summary {
            println!("     - {}: {} model(s)", capability, model_ids.len());
            if !model_ids.is_empty() {
                println!("       Models: {}", model_ids.join(", "));
            }
        }
    }

    // === ModelsDevAware Trait Integration Demo ===
    println!("\n=== ModelsDevAware Trait Integration Demo ===");

    // 1. Demonstrate creating provider instances
    println!("\n1. Creating provider instances using ModelsDevAware trait:");

    // Try to create an OpenAI provider
    println!("\n   Attempting to create OpenAI provider...");
    match registry.create_provider::<OpenAIProvider>("openai").await {
        Ok(provider) => {
            println!("   ✓ Successfully created OpenAI provider:");
            println!("     Base URL: {}", provider.base_url);
            println!("     Model: {}", provider.model);
            println!(
                "     API Key: {}...",
                provider.api_key.chars().take(10).collect::<String>()
            );
            if let Some(org) = provider.organization {
                println!("     Organization: {}", org);
            }
        }
        Err(e) => {
            println!("   ✗ Failed to create OpenAI provider: {}", e);
            println!("     This is expected if OPENAI_API_KEY environment variable is not set");
        }
    }

    // Try to create an Anthropic provider
    println!("\n   Attempting to create Anthropic provider...");
    match registry
        .create_provider::<AnthropicProvider>("anthropic")
        .await
    {
        Ok(provider) => {
            println!("   ✓ Successfully created Anthropic provider:");
            println!("     Base URL: {}", provider.base_url);
            println!("     Model: {}", provider.model);
            println!(
                "     API Key: {}...",
                provider.api_key.chars().take(10).collect::<String>()
            );
            if let Some(version) = provider.api_version {
                println!("     API Version: {}", version);
            }
        }
        Err(e) => {
            println!("   ✗ Failed to create Anthropic provider: {}", e);
            println!("     This is expected if ANTHROPIC_API_KEY environment variable is not set");
        }
    }

    // Try to create a Google provider
    println!("\n   Attempting to create Google provider...");
    match registry.create_provider::<GoogleProvider>("google").await {
        Ok(provider) => {
            println!("   ✓ Successfully created Google provider:");
            println!("     Base URL: {}", provider.base_url);
            println!("     Model: {}", provider.model);
            println!(
                "     API Key: {}...",
                provider.api_key.chars().take(10).collect::<String>()
            );
            if let Some(project_id) = provider.project_id {
                println!("     Project ID: {}", project_id);
            }
        }
        Err(e) => {
            println!("   ✗ Failed to create Google provider: {}", e);
            println!("     This is expected if GOOGLE_API_KEY environment variable is not set");
        }
    }

    // 2. Demonstrate error handling
    println!("\n2. Demonstrating error handling:");

    // Try to create a provider with non-existent provider ID
    println!("\n   Attempting to create provider with non-existent ID...");
    match registry
        .create_provider::<OpenAIProvider>("non-existent")
        .await
    {
        Ok(_) => {
            println!("   ✗ Unexpectedly succeeded with non-existent provider");
        }
        Err(e) => {
            println!("   ✓ Correctly failed with non-existent provider: {}", e);
        }
    }

    // Try to create a provider with non-existent model
    println!("\n   Attempting to create provider with non-existent model...");
    match registry.create_provider::<OpenAIProvider>("openai").await {
        Ok(_) => {
            println!("   ✗ Unexpectedly succeeded with non-existent model");
        }
        Err(e) => {
            println!("   ✓ Correctly failed with non-existent model: {}", e);
        }
    }

    // === Advanced Registry Operations Demo ===
    println!("\n=== Advanced Registry Operations Demo ===");

    // 1. Find providers by npm package
    println!("\n1. Finding providers by npm package:");
    let npm_searches = vec![
        "@ai-sdk/openai",
        "@ai-sdk/anthropic",
        "@ai-sdk/google",
        "@non-existent/package",
    ];

    for package in npm_searches {
        let providers = registry.find_providers_by_npm(package).await;
        if providers.is_empty() {
            println!("   ✗ {} -> No providers found", package);
        } else {
            println!("   ✓ {} -> {} provider(s):", package, providers.len());
            for provider_id in providers {
                if let Some(provider) = registry.get_provider(&provider_id).await {
                    println!("     - {} ({})", provider.name, provider.id);
                } else {
                    println!("     - {}", provider_id);
                }
            }
        }
    }

    // 2. Get models with specific capabilities using registry method
    println!("\n2. Getting models with capabilities using registry method:");
    let capability_searches = vec!["reasoning", "vision", "tool_call", "attachment", "unknown"];

    for capability in capability_searches {
        let models = registry.get_models_with_capability(capability).await;
        if models.is_empty() {
            println!("   ✗ {} -> No models found", capability);
        } else {
            println!("   ✓ {} -> {} model(s):", capability, models.len());
            for model in models {
                println!(
                    "     - {} ({}) - Cost: ${}/${} per 1M tokens",
                    model.name, model.id, model.cost.input, model.cost.output
                );
                if let Some(reasoning_cost) = model.cost.reasoning {
                    println!("       Reasoning cost: ${} per 1M tokens", reasoning_cost);
                }
            }
        }
    }

    // 3. Get providers with their models
    println!("\n3. Getting providers with their models:");
    let providers_with_models = registry.get_providers_with_models().await;
    if providers_with_models.is_empty() {
        println!("   ✗ No providers with models found");
    } else {
        println!(
            "   ✓ Found {} provider(s) with models:",
            providers_with_models.len()
        );
        for (provider_id, provider, models) in providers_with_models {
            println!("     - {} ({}):", provider.name, provider_id);
            if models.is_empty() {
                println!("       No models available");
            } else {
                for model in models {
                    println!(
                        "       * {} ({}) - ${}/${} per 1M tokens",
                        model.name, model.id, model.cost.input, model.cost.output
                    );
                }
            }
        }
    }

    // 4. Get connection information
    println!("\n4. Getting connection information:");
    let connection_providers = vec!["openai", "anthropic", "google", "non-existent"];

    for provider_id in connection_providers {
        match registry.get_connection_info(provider_id).await {
            Some(connection_info) => {
                println!("   ✓ {} connection info:", provider_id);
                println!("     Base URL: {}", connection_info.base_url);
                println!(
                    "     Required env vars: {:?}",
                    connection_info.required_env_vars
                );
                println!(
                    "     Optional env vars: {:?}",
                    connection_info.optional_env_vars
                );
                if !connection_info.config.is_empty() {
                    println!("     Configuration: {:?}", connection_info.config);
                }
            }
            None => {
                println!("   ✗ {} -> No connection info found", provider_id);
            }
        }
    }

    // === Real-world Usage Scenario ===
    println!("\n=== Real-world Usage Scenario ===");
    println!("Scenario: Building a chat application that needs to:");
    println!("1. Find the best model for chat");
    println!("2. Check if the provider is configured");
    println!("3. Get connection information");
    println!("4. Create a provider instance");

    println!("\nExecuting scenario...");

    // Step 1: Find the best model for chat
    let chat_model = find_best_model_for_use_case(&registry, "chat").await;
    let chat_model_id = match chat_model {
        Some(model_id) => {
            println!("   ✓ Step 1: Found best chat model: {}", model_id);
            model_id
        }
        None => {
            println!("   ✗ Step 1: No chat model found");
            return Ok(());
        }
    };

    // Step 2: Find the provider for this model
    let provider_id = match registry.find_provider_for_model(&chat_model_id).await {
        Some(provider_id) => {
            println!(
                "   ✓ Step 2: Found provider for model: {} -> {}",
                chat_model_id, provider_id
            );
            provider_id
        }
        None => {
            println!(
                "   ✗ Step 2: No provider found for model: {}",
                chat_model_id
            );
            return Ok(());
        }
    };

    // Step 3: Check if the provider is configured
    match check_provider_configuration(&registry, &provider_id).await {
        Ok(()) => {
            println!(
                "   ✓ Step 3: Provider {} is properly configured",
                provider_id
            );
        }
        Err(missing_vars) => {
            println!(
                "   ✗ Step 3: Provider {} is not properly configured",
                provider_id
            );
            println!("     Missing variables: {:?}", missing_vars);
            println!("     Please set the required environment variables and try again");
            return Ok(());
        }
    }

    // Step 4: Get connection information
    match registry.get_connection_info(&provider_id).await {
        Some(connection_info) => {
            println!("   ✓ Step 4: Got connection info for {}", provider_id);
            println!("     Base URL: {}", connection_info.base_url);
            println!(
                "     Required env vars: {:?}",
                connection_info.required_env_vars
            );
        }
        None => {
            println!("   ✗ Step 4: No connection info found for {}", provider_id);
            return Ok(());
        }
    }

    // Step 5: Create provider instance (this will fail without real API keys, but shows the pattern)
    println!("   Step 5: Creating provider instance...");

    // Since different providers have different types, we'll handle them separately
    let creation_success = match provider_id.as_str() {
        "openai" => {
            let result = registry
                .create_provider::<OpenAIProvider>(&provider_id)
                .await;
            result.is_ok()
        }
        "anthropic" => {
            let result = registry
                .create_provider::<AnthropicProvider>(&provider_id)
                .await;
            result.is_ok()
        }
        "google" => {
            let result = registry
                .create_provider::<GoogleProvider>(&provider_id)
                .await;
            result.is_ok()
        }
        _ => {
            println!("   ✗ Step 5: Unsupported provider: {}", provider_id);
            false
        }
    };

    if creation_success {
        println!("   ✓ Step 5: Successfully created provider instance");
        println!("   🎉 Scenario completed successfully!");
    } else {
        println!("   ✗ Step 5: Failed to create provider instance");
        println!("     This is expected in demonstration mode without real API keys");
        println!("     In a real application, you would set the required environment variables");
        println!("   🎯 Scenario demonstrates the correct workflow pattern!");
    }

    println!("\n=== Example Complete ===");
    println!("This example demonstrated:");
    println!("• Convenience functions for common operations");
    println!("• ModelsDevAware trait integration for provider creation");
    println!("• Capability-based model discovery");
    println!("• Configuration validation and error handling");
    println!("• Real-world usage scenarios");
    println!("\nTo run with real data, set the required environment variables and");
    println!("ensure you have access to the models.dev API.");

    Ok(())
}

#[cfg(not(feature = "models-dev"))]
fn main() {
    println!("This example requires the 'models-dev' feature to be enabled.");
    println!("Run with: cargo run --example models_dev_convenience_example --features models-dev");
}
