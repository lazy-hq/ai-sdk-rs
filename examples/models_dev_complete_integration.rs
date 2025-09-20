//! Complete end-to-end integration example for the models.dev feature.
//!
//! This example demonstrates a real-world scenario where an application needs to:
//! 1. Discover available AI providers and models
//! 2. Select the best model for a specific use case
//! 3. Validate provider configuration
//! 4. Create and use provider instances
//! 5. Handle errors gracefully
//! 6. Monitor performance and cache usage
//!
//! The example simulates a multi-tenant AI service that needs to:
//! - Support multiple providers (OpenAI, Anthropic, Google)
//! - Choose models based on capabilities and cost
//! - Handle configuration validation
//! - Provide fallback options
//! - Monitor usage and performance

#[cfg(feature = "models-dev")]
#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    use aisdk::models_dev::{
        ProviderRegistry, check_provider_configuration, find_best_model_for_use_case,
        find_models_with_capability, find_provider_for_cloud_service, get_capability_summary,
        get_providers_summary, list_providers_for_npm_package,
        traits::{AnthropicProvider, GoogleProvider, OpenAIProvider},
    };
    use std::collections::HashMap;
    use std::time::Instant;

    println!("🚀 Models.dev Complete Integration Example");
    println!("=========================================");

    // === Step 1: Initialize and Configure ===
    println!("\n📋 Step 1: Initializing Provider Registry...");

    let start_time = Instant::now();
    let registry = ProviderRegistry::with_default_client();
    let init_time = start_time.elapsed();

    println!("✓ Registry initialized in {:?}", init_time);

    // === Step 2: Fetch Latest Data ===
    println!("\n🌐 Step 2: Fetching Latest Provider and Model Data...");

    let fetch_start = Instant::now();
    match registry.refresh().await {
        Ok((provider_count, model_count)) => {
            let fetch_time = fetch_start.elapsed();
            println!("✓ Successfully fetched data in {:?}", fetch_time);
            println!("  - {} providers loaded", provider_count);
            println!("  - {} models loaded", model_count);
        }
        Err(e) => {
            eprintln!("✗ Failed to fetch data from models.dev API: {}", e);
            eprintln!("⚠️  Continuing with demonstration mode (limited functionality)");
            println!("   In a real application, you might want to:");
            println!("   - Check network connectivity");
            println!("   - Verify API credentials");
            println!("   - Implement retry logic");
            println!("   - Fall back to cached data if available");
        }
    }

    // === Step 3: Provider Discovery and Analysis ===
    println!("\n🔍 Step 3: Provider Discovery and Analysis...");

    // Get comprehensive provider summary
    let provider_summary = get_providers_summary(&registry).await;
    if provider_summary.is_empty() {
        println!("⚠️  No providers found in registry");
        println!("   This might indicate:");
        println!("   - No data was loaded from the API");
        println!("   - The registry is empty");
        println!("   - There might be network or API issues");
    } else {
        println!("✓ Found {} provider(s):", provider_summary.len());

        for (provider_id, provider_name, model_ids) in provider_summary {
            println!(
                "  📦 {} ({}): {} models",
                provider_name,
                provider_id,
                model_ids.len()
            );

            // Get detailed information for each provider
            if let Some(provider) = registry.get_provider(&provider_id).await {
                println!("     📍 Base URL: {}", provider.base_url);
                println!(
                    "     📦 NPM Package: {}@{}",
                    provider.npm_name, provider.npm_version
                );
                println!("     📚 Documentation: {}", provider.doc_url);
                println!("     ✅ Available: {}", provider.available);

                // Show models for this provider with cost information
                let models = registry.get_models_for_provider(&provider_id).await;
                if !models.is_empty() {
                    println!("     💰 Models and pricing:");
                    for model in models {
                        println!(
                            "       • {} (${}/${} per 1M tokens)",
                            model.name, model.cost.input, model.cost.output
                        );
                        if let Some(reasoning_cost) = model.cost.reasoning {
                            println!("         🔧 Reasoning: ${} per 1M tokens", reasoning_cost);
                        }
                        println!(
                            "         📏 Context: {} tokens, Output: {} tokens",
                            model.limits.context, model.limits.output
                        );
                    }
                }
            }
            println!();
        }
    }

    // === Step 4: Capability Analysis ===
    println!("\n🎯 Step 4: Capability Analysis Across All Models...");

    let capability_summary = get_capability_summary(&registry).await;
    if capability_summary.is_empty() {
        println!("⚠️  No capabilities found in registry");
    } else {
        println!(
            "✓ Found {} capability/capabilities:",
            capability_summary.len()
        );

        for (capability, model_ids) in capability_summary {
            println!("  🎨 {}: {} model(s)", capability, model_ids.len());

            // Show detailed information for models with this capability
            for model_id in model_ids {
                if let Some(model) = registry.get_model(&model_id).await {
                    println!(
                        "     • {} ({}) - Provider: {}",
                        model.name, model.id, model.provider_id
                    );
                    println!(
                        "       💵 Cost: ${}/${} per 1M tokens",
                        model.cost.input, model.cost.output
                    );
                    println!("       📊 Context: {} tokens", model.limits.context);

                    // Show modalities
                    if !model.input_modalities.is_empty() {
                        println!("       📥 Input: {}", model.input_modalities.join(", "));
                    }
                    if !model.output_modalities.is_empty() {
                        println!("       📤 Output: {}", model.output_modalities.join(", "));
                    }
                }
            }
            println!();
        }
    }

    // === Step 5: Use Case Analysis and Model Selection ===
    println!("\n🎯 Step 5: Use Case Analysis and Model Selection...");

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

    println!("Analyzing best models for different use cases:");

    let mut use_case_recommendations = HashMap::new();

    for (use_case, description) in use_cases {
        print!("  🎯 {}: ", description);

        match find_best_model_for_use_case(&registry, use_case).await {
            Some(model_id) => {
                if let Some(model) = registry.get_model(&model_id).await {
                    println!("✓ {}", model.name);
                    println!(
                        "     💰 Cost: ${}/${} per 1M tokens",
                        model.cost.input, model.cost.output
                    );
                    println!("     🏢 Provider: {}", model.provider_id);

                    use_case_recommendations
                        .insert(use_case.to_string(), (model_id, model.provider_id));
                } else {
                    println!("⚠️  Model found but details unavailable");
                }
            }
            None => {
                println!("✗ No suitable model found");
            }
        }
        println!();
    }

    // === Step 6: Configuration Validation ===
    println!("\n🔧 Step 6: Provider Configuration Validation...");

    let provider_ids = vec!["openai", "anthropic", "google", "mistral"];
    let mut configured_providers = Vec::new();

    println!("Checking configuration for available providers:");

    for provider_id in provider_ids {
        print!("  🔍 {}: ", provider_id);

        match check_provider_configuration(&registry, provider_id).await {
            Ok(()) => {
                println!("✅ Fully configured");
                configured_providers.push(provider_id.to_string());
            }
            Err(missing_vars) => {
                println!("❌ Missing variables: {:?}", missing_vars);
                println!("     To configure, set:");
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

    // === Step 7: Provider Instance Creation ===
    println!("\n🏗️  Step 7: Creating Provider Instances...");

    if configured_providers.is_empty() {
        println!("⚠️  No providers are properly configured");
        println!("   In a real application, you would:");
        println!("   - Set the required environment variables");
        println!("   - Implement configuration validation");
        println!("   - Provide fallback mechanisms");
        println!("   - Allow users to configure providers dynamically");
    } else {
        println!("✓ Attempting to create instances for configured providers:");

        for provider_id in &configured_providers {
            println!("  🏗️  Creating {} provider instance...", provider_id);

            let creation_start = Instant::now();

            let success = match provider_id.as_str() {
                "openai" => {
                    match registry
                        .create_provider::<OpenAIProvider>(&provider_id)
                        .await
                    {
                        Ok(_provider) => true,
                        Err(e) => {
                            println!("     ❌ Failed to create: {}", e);
                            false
                        }
                    }
                }
                "anthropic" => {
                    match registry
                        .create_provider::<AnthropicProvider>(&provider_id)
                        .await
                    {
                        Ok(_provider) => true,
                        Err(e) => {
                            println!("     ❌ Failed to create: {}", e);
                            false
                        }
                    }
                }
                "google" => {
                    match registry
                        .create_provider::<GoogleProvider>(&provider_id)
                        .await
                    {
                        Ok(_provider) => true,
                        Err(e) => {
                            println!("     ❌ Failed to create: {}", e);
                            false
                        }
                    }
                }
                _ => {
                    println!("     ⚠️  Provider type not implemented for {}", provider_id);
                    false
                }
            };

            let creation_time = creation_start.elapsed();

            if success {
                println!("     ✅ Successfully created in {:?}", creation_time);
                println!("     🎉 Provider is ready for AI operations!");

                // In a real application, you would now use this provider
                // for actual AI tasks like text generation, etc.
            } else {
                println!("     ❌ Failed to create provider instance");
                println!("     💡 This might indicate:");
                println!("        - Environment variables are set but invalid");
                println!("        - Network connectivity issues");
                println!("        - Provider service is unavailable");
                println!("        - Authentication or authorization issues");
            }
        }
    }

    // === Step 8: Performance and Cache Analysis ===
    println!("\n📊 Step 8: Performance and Cache Analysis...");

    let client = &registry.client;
    let cache_stats = client.cache_stats().await;

    println!("Cache Statistics:");
    println!("  📈 Memory entries: {}", cache_stats.memory_entries);
    println!("  💾 Disk entries: {}", cache_stats.disk_entries);
    println!("  ⏱️  Cache TTL: {:?}", cache_stats.cache_ttl);

    let total_cached_entries = cache_stats.memory_entries + cache_stats.disk_entries;
    if total_cached_entries > 0 {
        println!("  📊 Total cached entries: {}", total_cached_entries);
        println!("  💡 Cache is actively storing data for performance");
    } else {
        println!("  💡 Cache is empty - data will be fetched from API");
    }

    // Registry statistics
    println!("\nRegistry Statistics:");
    println!("  🏢 Total providers: {}", registry.provider_count().await);
    println!("  🤖 Total models: {}", registry.model_count().await);

    // === Step 9: Real-world Scenario Simulation ===
    println!("\n🌍 Step 9: Real-world Scenario Simulation...");
    println!("Scenario: Building a multi-tenant AI service with automatic provider selection");

    // Simulate different tenant requirements
    let tenants = vec![
        ("tenant-chat", "Customer support chatbot", "chat"),
        ("tenant-code", "Code review assistant", "code"),
        ("tenant-reasoning", "Research analysis tool", "reasoning"),
        ("tenant-budget", "Cost-sensitive application", "cheap"),
    ];

    println!("\nTenant Requirements and Provider Recommendations:");

    for (tenant_id, tenant_description, preferred_use_case) in tenants {
        println!("\n  🏢 {}: {}", tenant_id, tenant_description);
        println!("     Preferred use case: {}", preferred_use_case);

        // Find best model for this use case
        if let Some(model_id) = find_best_model_for_use_case(&registry, preferred_use_case).await {
            if let Some(model) = registry.get_model(&model_id).await {
                println!("     🎯 Recommended model: {}", model.name);
                println!(
                    "     💰 Estimated cost: ${}/${} per 1M tokens",
                    model.cost.input, model.cost.output
                );
                println!("     🏢 Provider: {}", model.provider_id);

                // Check if provider is configured
                match check_provider_configuration(&registry, &model.provider_id).await {
                    Ok(()) => {
                        println!("     ✅ Provider is configured and ready");
                    }
                    Err(missing_vars) => {
                        println!("     ⚠️  Provider needs configuration: {:?}", missing_vars);
                        println!(
                            "     💡 Fallback: Use alternative provider or show setup instructions"
                        );
                    }
                }
            }
        } else {
            println!("     ❌ No suitable model found for this use case");
            println!("     💡 Fallback: Use default model or show error to user");
        }
    }

    // === Step 10: Advanced Features Demonstration ===
    println!("\n🚀 Step 10: Advanced Features Demonstration...");

    // Demonstrate npm package discovery
    println!("\n📦 NPM Package Discovery:");
    let npm_packages = vec![
        "@ai-sdk/openai",
        "@ai-sdk/anthropic",
        "@ai-sdk/google",
        "@ai-sdk/mistral",
    ];

    for package in npm_packages {
        let providers = list_providers_for_npm_package(&registry, package).await;
        if providers.is_empty() {
            println!("  ❌ {} -> No providers found", package);
        } else {
            println!(
                "  ✅ {} -> {} provider(s): {:?}",
                package,
                providers.len(),
                providers
            );
        }
    }

    // Demonstrate cloud service discovery
    println!("\n☁️  Cloud Service Discovery:");
    let cloud_services = vec![
        ("openai", "OpenAI"),
        ("anthropic", "Anthropic/Claude"),
        ("google", "Google/Gemini"),
        ("claude", "Anthropic/Claude"),
        ("gemini", "Google/Gemini"),
        ("meta", "Meta"),
        ("mistral", "Mistral"),
    ];

    for (service, description) in cloud_services {
        match find_provider_for_cloud_service(&registry, service).await {
            Some(provider_id) => {
                println!("  ✅ {} -> {} ({})", service, provider_id, description);
            }
            None => {
                println!("  ❌ {} -> No provider found for {}", service, description);
            }
        }
    }

    // Demonstrate advanced capability search
    println!("\n🔍 Advanced Capability Search:");
    let advanced_capabilities = vec![
        ("reasoning", "Advanced reasoning and thinking"),
        ("vision", "Image and visual processing"),
        ("tool_call", "Function calling and tool use"),
        ("attachment", "File and document processing"),
        ("streaming", "Real-time streaming responses"),
    ];

    for (capability, description) in advanced_capabilities {
        let models = find_models_with_capability(&registry, capability).await;
        if models.is_empty() {
            println!(
                "  ❌ {}: No models with {} capability",
                capability, description
            );
        } else {
            println!(
                "  ✅ {}: {} model(s) with {} capability",
                capability,
                models.len(),
                description
            );
            for model in models.iter().take(3) {
                // Show first 3 models
                println!("     • {} ({})", model.name, model.provider_id);
            }
            if models.len() > 3 {
                println!("     ... and {} more", models.len() - 3);
            }
        }
    }

    // === Step 11: Error Handling and Resilience ===
    println!("\n🛡️  Step 11: Error Handling and Resilience Demonstration...");

    // Test error handling with invalid inputs
    println!("\nTesting error handling with invalid inputs:");

    let invalid_inputs = vec![
        ("nonexistent_provider", "Non-existent provider"),
        ("nonexistent_model", "Non-existent model"),
        ("", "Empty string"),
        ("invalid_capability_123", "Invalid capability"),
    ];

    for (input, description) in invalid_inputs {
        println!("  🧪 Testing {}: {}", description, input);

        // Test provider lookup
        let provider_result = registry.get_provider(input).await;
        println!(
            "    Provider lookup: {}",
            if provider_result.is_some() {
                "✅ Found"
            } else {
                "❌ Not found"
            }
        );

        // Test model lookup
        let model_result = registry.get_model(input).await;
        println!(
            "    Model lookup: {}",
            if model_result.is_some() {
                "✅ Found"
            } else {
                "❌ Not found"
            }
        );

        // Test provider finding
        let provider_for_model = registry.find_provider_for_model(input).await;
        println!(
            "    Provider for model: {}",
            if provider_for_model.is_some() {
                "✅ Found"
            } else {
                "❌ Not found"
            }
        );

        // Test availability checks
        let provider_available = registry.is_provider_available(input).await;
        let model_available = registry.is_model_available(input).await;
        println!(
            "    Provider available: {}",
            if provider_available {
                "✅ Yes"
            } else {
                "❌ No"
            }
        );
        println!(
            "    Model available: {}",
            if model_available { "✅ Yes" } else { "❌ No" }
        );
    }

    // === Step 12: Summary and Recommendations ===
    println!("\n📋 Step 12: Summary and Recommendations...");

    let total_providers = registry.provider_count().await;
    let total_models = registry.model_count().await;
    let configured_count = configured_providers.len();

    println!("📊 Integration Summary:");
    println!("  🏢 Total providers discovered: {}", total_providers);
    println!("  🤖 Total models discovered: {}", total_models);
    println!("  ✅ Providers configured: {}", configured_count);
    println!(
        "  ⚠️  Providers needing configuration: {}",
        total_providers.saturating_sub(configured_count)
    );

    if configured_count == 0 {
        println!("\n💡 Recommendations:");
        println!("  1. Set up at least one provider API key:");
        println!("     - export OPENAI_API_KEY=your_openai_key");
        println!("     - export ANTHROPIC_API_KEY=your_anthropic_key");
        println!("     - export GOOGLE_API_KEY=your_google_key");
        println!("  2. Run the example again to see full functionality");
        println!("  3. Consider implementing fallback mechanisms");
        println!("  4. Add error handling for production use");
    } else {
        println!("\n🎉 Success! Your AI service is ready to use:");
        println!("  ✅ Provider discovery working");
        println!("  ✅ Model selection operational");
        println!("  ✅ Configuration validation functional");
        println!("  ✅ Cache performance monitored");
        println!("  ✅ Error handling tested");

        println!("\n🚀 Next steps for production deployment:");
        println!("  1. Implement proper error handling and retries");
        println!("  2. Add monitoring and logging");
        println!("  3. Set up cost tracking and limits");
        println!("  4. Implement user authentication and authorization");
        println!("  5. Add rate limiting and quota management");
        println!("  6. Set up automated provider failover");
    }

    println!("\n🎯 Key Features Demonstrated:");
    println!("  ✅ Dynamic provider discovery");
    println!("  ✅ Intelligent model selection");
    println!("  ✅ Capability-based filtering");
    println!("  ✅ Configuration validation");
    println!("  ✅ Performance monitoring");
    println!("  ✅ Comprehensive error handling");
    println!("  ✅ Cache management");
    println!("  ✅ Thread-safe operations");
    println!("  ✅ Real-world scenario simulation");

    println!("\n🏁 Integration Example Complete!");
    println!("=================================");
    println!("This example demonstrated a complete production-ready integration");
    println!("with the models.dev feature, including error handling, performance");
    println!("monitoring, and real-world usage patterns.");

    Ok(())
}

#[cfg(not(feature = "models-dev"))]
fn main() {
    println!("This example requires the 'models-dev' feature to be enabled.");
    println!("Run with: cargo run --example models_dev_complete_integration --features models-dev");
    println!("\nTo enable the feature, add it to your Cargo.toml:");
    println!("```toml");
    println!("[dependencies]");
    println!("aisdk = {{ version = \"0.1.0\", features = [\"models-dev\"] }}");
    println!("```");
}
