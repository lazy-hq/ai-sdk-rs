//! Example usage of the ModelsDevClient with caching.

#[cfg(feature = "models-dev")]
#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    use aisdk::models_dev::ModelsDevClient;
    use std::time::Duration;

    println!("Models.dev Client Example");
    println!("========================");

    // Create a client with default settings
    let client = ModelsDevClient::new();
    println!(
        "Created client with API base URL: {}",
        client.api_base_url()
    );
    println!("Cache TTL: {:?}", client.cache_ttl());
    println!("Disk cache path: {:?}", client.disk_cache_path());

    // Show initial cache stats
    let stats = client.cache_stats().await;
    println!("\nInitial cache stats:");
    println!("  Memory entries: {}", stats.memory_entries);
    println!("  Disk entries: {}", stats.disk_entries);

    // Try to fetch providers (this will fail since we don't have a real API)
    println!("\nAttempting to fetch providers...");
    match client.fetch_providers().await {
        Ok(providers) => {
            println!("Successfully fetched {} providers:", providers.len());
            for provider in providers {
                println!("  - {} ({})", provider.name, provider.id);
            }
        }
        Err(e) => {
            println!(
                "Failed to fetch providers (expected for this example): {}",
                e
            );
        }
    }

    // Show final cache stats
    let stats = client.cache_stats().await;
    println!("\nFinal cache stats:");
    println!("  Memory entries: {}", stats.memory_entries);
    println!("  Disk entries: {}", stats.disk_entries);

    // Example with custom builder
    println!("\nExample with custom builder:");
    let custom_client = ModelsDevClient::builder()
        .api_base_url("https://custom.api.example.com")
        .cache_ttl(Duration::from_secs(1800)) // 30 minutes
        .build()?;

    println!("Custom client API URL: {}", custom_client.api_base_url());
    println!("Custom client TTL: {:?}", custom_client.cache_ttl());

    Ok(())
}

#[cfg(not(feature = "models-dev"))]
fn main() {
    println!("This example requires the 'models-dev' feature to be enabled.");
    println!("Run with: cargo run --example models_dev_client_example --features models-dev");
}
