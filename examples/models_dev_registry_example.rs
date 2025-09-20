//! Example demonstrating how to use the models.dev provider registry.
//!
//! This example shows how to:
//! - Create a provider registry
//! - Refresh data from the models.dev API
//! - Look up providers and models
//! - Find which provider offers a specific model

#[cfg(feature = "models-dev")]
#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    use aisdk::models_dev::ProviderRegistry;

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

    // List all available providers
    println!("\nAvailable providers:");
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
