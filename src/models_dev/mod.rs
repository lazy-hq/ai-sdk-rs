//! Models.dev API client with caching support.
//!
//! This module provides a comprehensive client for interacting with the models.dev API,
//! featuring both memory and disk caching to minimize API calls and improve performance.
//! It includes a provider registry for managing AI model providers, convenience functions
//! for common operations, and robust error handling.
//!
//! # Overview
//!
//! The models.dev module is designed to be a complete solution for discovering and
//! working with AI model providers through the models.dev API. It provides:
//!
//! - **API Client**: A cached HTTP client for fetching provider and model information
//! - **Provider Registry**: An in-memory registry for storing and querying provider data
//! - **Convenience Functions**: High-level functions for common operations
//! - **Type Safety**: Strong typing with comprehensive error handling
//! - **Performance**: Memory and disk caching with configurable TTL
//! - **Concurrency**: Thread-safe operations suitable for async applications
//!
//! # Architecture
//!
//! The module is organized into several key components:
//!
//! ## Core Components
//!
//! - [`client`]: HTTP client with caching capabilities
//! - [`registry`]: Provider and model registry with query capabilities
//! - [`types`]: Data structures for API responses and internal representations
//! - [`error`]: Comprehensive error types for all failure scenarios
//! - [`traits`]: Traits for provider integration and type abstraction
//! - [`convenience`]: High-level utility functions
//!
//! ## Data Flow
//!
//! ```text
//! models.dev API → ModelsDevClient → ProviderRegistry → Application
//!     ↓               ↓                    ↓              ↓
//!   HTTP Request   Cache (Memory/Disk)   Data Store    Convenience Functions
//! ```
//!
//! # Quick Start
//!
//! ## Basic Usage
//!
//! ```rust
//! use aisdk::models_dev::{ProviderRegistry, find_best_model_for_use_case};
//!
//! #[tokio::main]
//! async fn main() -> Result<(), Box<dyn std::error::Error>> {
//!     // Create a registry with default client
//!     let registry = ProviderRegistry::with_default_client();
//!     
//!     // Refresh data from the API (commented out for doctest)
//!     // registry.refresh().await?;
//!     
//!     // Find the best model for a specific use case
//!     // Note: This will return None if registry hasn't been refreshed
//!     let chat_model = find_best_model_for_use_case(&registry, "chat").await;
//!     
//!     if let Some(model_id) = chat_model {
//!         println!("Best chat model: {}", model_id);
//!     } else {
//!         println!("No chat model found - try refreshing the registry first");
//!     }
//!     
//!     Ok(())
//! }
//! ```
//!
//! ## Advanced Usage with Custom Client
//!
//! ```rust
//! use aisdk::models_dev::{ModelsDevClient, ProviderRegistry};
//! use std::time::Duration;
//! use tempfile::TempDir;
//!
//! #[tokio::main]
//! async fn main() -> Result<(), Box<dyn std::error::Error>> {
//!     // Create a custom client with specific configuration
//!     let temp_dir = TempDir::new()?;
//!     let client = ModelsDevClient::builder()
//!         .cache_ttl(Duration::from_secs(3600)) // 1 hour cache
//!         .disk_cache_path(temp_dir.path())
//!         .api_base_url("https://custom.models.dev/api")
//!         .build()?;
//!     
//!     // Create registry with custom client
//!     let registry = ProviderRegistry::new(client);
//!     
//!     // Use the registry (commented out for doctest to avoid network calls)
//!     // registry.refresh().await?;
//!     
//!     // Access provider and model information
//!     let providers = registry.get_all_providers().await;
//!     println!("Found {} providers", providers.len());
//!     
//!     Ok(())
//! }
//! ```
//!
//! # Features
//!
//! ## Caching
//!
//! The client supports both memory and disk caching to minimize API calls:
//!
//! - **Memory Cache**: Fast in-memory caching for frequently accessed data
//! - **Disk Cache**: Persistent caching across application restarts
//! - **Configurable TTL**: Time-to-live settings for cache entries
//! - **Cache Statistics**: Monitor cache hit rates and performance
//!
//! ```rust
//! use aisdk::models_dev::ModelsDevClient;
//!
//! # async fn example() -> Result<(), Box<dyn std::error::Error>> {
//! let client = ModelsDevClient::new();
//!
//! // Check cache statistics
//! let stats = client.cache_stats().await;
//! println!("Memory entries: {}", stats.memory_entries);
//! println!("Disk entries: {}", stats.disk_entries);
//! println!("Cache TTL: {:?}", stats.cache_ttl);
//!
//! // Clear caches if needed
//! client.clear_caches().await?;
//! # Ok(())
//! # }
//! ```
//!
//! ## Provider Discovery
//!
//! The registry provides multiple ways to discover and query providers:
//!
//! ```rust
//! use aisdk::models_dev::{
//!     ProviderRegistry, find_provider_for_cloud_service,
//!     list_providers_for_npm_package, get_providers_summary
//! };
//!
//! # async fn example() {
//! let registry = ProviderRegistry::default();
//!
//! // Find providers by cloud service name
//! let openai_provider = find_provider_for_cloud_service(&registry, "openai").await;
//!
//! // List providers for npm packages
//! let providers = list_providers_for_npm_package(&registry, "@ai-sdk/openai").await;
//!
//! // Get comprehensive provider summary
//! let summary = get_providers_summary(&registry).await;
//! for (provider_id, provider_name, model_ids) in summary {
//!     println!("{} ({}): {} models", provider_name, provider_id, model_ids.len());
//! }
//! # }
//! ```
//!
//! ## Model Discovery
//!
//! Find models based on capabilities, use cases, or other criteria:
//!
//! ```rust
//! use aisdk::models_dev::{
//!     find_models_with_capability, find_best_model_for_use_case,
//!     get_capability_summary, ProviderRegistry
//! };
//!
//! # async fn example() {
//! let registry = ProviderRegistry::default();
//!
//! // Find models with specific capabilities
//! let reasoning_models = find_models_with_capability(&registry, "reasoning").await;
//! let vision_models = find_models_with_capability(&registry, "vision").await;
//!
//! // Get best model for specific use cases
//! let chat_model = find_best_model_for_use_case(&registry, "chat").await;
//! let code_model = find_best_model_for_use_case(&registry, "code").await;
//! let fast_model = find_best_model_for_use_case(&registry, "fast").await;
//!
//! // Get capability summary across all models
//! let capabilities = get_capability_summary(&registry).await;
//! for (capability, model_ids) in capabilities {
//!     println!("{}: {} models", capability, model_ids.len());
//! }
//! # }
//! ```
//!
//! ## Provider Integration
//!
//! The `ModelsDevAware` trait allows seamless integration with existing providers:
//!
//! ```rust
//! use aisdk::models_dev::{
//!     ProviderRegistry, traits::{ModelsDevAware}
//! };
//! # #[cfg(feature = "openai")]
//! # use aisdk::providers::openai::OpenAI;
//!
//! # async fn example() -> Result<(), Box<dyn std::error::Error>> {
//! let registry = ProviderRegistry::default();
//! registry.refresh().await?;
//!
//! // Create provider instances using registry data
//! # #[cfg(feature = "openai")]
//! match registry.create_provider::<OpenAI>("openai").await {
//!     Ok(provider) => {
//!         println!("Created OpenAI provider: {}", provider.settings.provider_name);
//!         // Use the provider for AI operations
//!     }
//!     Err(e) => {
//!         println!("Failed to create provider: {}", e);
//!     }
//! }
//! # #[cfg(not(feature = "openai"))]
//! # println!("OpenAI provider not available - enable with 'openai' feature");
//! # Ok(())
//! # }
//! ```
//!
//! # Configuration
//!
//! ## Environment Variables
//!
//! The module respects the following environment variables:
//!
//! - `MODELS_DEV_API_BASE_URL`: Custom API base URL (default: https://models.dev/api)
//! - `MODELS_DEV_CACHE_TTL`: Cache TTL in seconds (default: 3600)
//! - `MODELS_DEV_CACHE_DIR`: Custom disk cache directory
//! - Provider-specific variables (e.g., `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`)
//!
//! ## Feature Flags
//!
//! - `models-dev`: Enables the entire models.dev functionality (required)
//! - `openai`: Enables OpenAI provider integration
//! - `full`: Enables all features including models-dev and openai
//!
//! # Error Handling
//!
//! The module provides comprehensive error handling through the [`ModelsDevError`] enum:
//!
//! ```rust
//! use aisdk::models_dev::{ModelsDevError, ModelsDevClient};
//!
//! # async fn example() -> Result<(), Box<dyn std::error::Error>> {
//! let client = ModelsDevClient::new();
//!
//! match client.fetch_providers().await {
//!     Ok(providers) => {
//!         println!("Fetched {} providers", providers.len());
//!     }
//!     Err(ModelsDevError::HttpError(e)) => {
//!         eprintln!("HTTP error: {}", e);
//!     }
//!     Err(ModelsDevError::ApiError(msg)) => {
//!         eprintln!("API error: {}", msg);
//!     }
//!     Err(ModelsDevError::CacheNotFound(path)) => {
//!         eprintln!("Cache not found: {:?}", path);
//!     }
//!     Err(e) => {
//!         eprintln!("Other error: {}", e);
//!     }
//! }
//! # Ok(())
//! # }
//! ```
//!
//! # Performance Considerations
//!
//! ## Memory Usage
//!
//! - The registry stores all provider and model data in memory for fast access
//! - Large datasets (1000+ providers, 50,000+ models) are supported
//! - Memory is freed when the registry is cleared or dropped
//!
//! ## Concurrency
//!
//! - All operations are thread-safe and suitable for concurrent use
//! - Read operations can be performed concurrently without locking
//! - Write operations use appropriate locking to ensure data consistency
//!
//! ## Caching Strategy
//!
//! - Memory cache provides fast access for frequently used data
//! - Disk cache provides persistence across application restarts
//! - Cache entries expire based on configurable TTL
//! - Cache statistics help monitor performance and hit rates
//!
//! # Testing
//!
//! The module includes comprehensive tests covering:
//!
//! - Unit tests for individual components
//! - Integration tests for complete workflows
//! - Performance tests with large datasets
//! - Concurrency tests for thread safety
//! - Error handling tests for edge cases
//!
//! Run tests with:
//!
//! ```bash
//! # Run all models.dev tests
//! cargo test --features models-dev
//!
//! # Run specific test files
//! cargo test --test models_dev_integration_tests --features models-dev
//! cargo test --test models_dev_registry_tests --features models-dev
//! cargo test --test models_dev_client_tests --features models-dev
//! cargo test --test models_dev_data_structures_tests --features models-dev
//! ```
//!
//! # Examples
//!
//! The module includes several examples demonstrating different usage patterns:
//!
//! - `models_dev_client_example`: Basic client usage and caching
//! - `models_dev_registry_example`: Registry operations and queries
//! - `models_dev_convenience_example`: Convenience functions and integration
//!
//! Run examples with:
//!
//! ```bash
//! cargo run --example models_dev_client_example --features models-dev
//! cargo run --example models_dev_registry_example --features models-dev
//! cargo run --example models_dev_convenience_example --features models-dev
//! ```
//!
//! # Contributing
//!
//! When contributing to the models.dev module:
//!
//! 1. Ensure all tests pass with `cargo test --features models-dev`
//! 2. Run clippy lints with `cargo clippy --features models-dev`
//! 3. Format code with `cargo fmt`
//! 4. Add comprehensive tests for new functionality
//! 5. Update documentation for new features
//! 6. Follow the existing code style and patterns
//!
//! # License
//!
//! This module is part of the AISDK project and is licensed under the MIT License.
//! See the [LICENSE](../../LICENSE) file for details.

#[cfg(feature = "models-dev")]
pub mod client;

#[cfg(feature = "models-dev")]
pub mod error;

#[cfg(feature = "models-dev")]
pub mod types;

#[cfg(feature = "models-dev")]
pub mod traits;

#[cfg(feature = "models-dev")]
pub mod registry;

#[cfg(feature = "models-dev")]
pub mod convenience;

#[cfg(feature = "models-dev")]
pub use client::ModelsDevClient;

#[cfg(feature = "models-dev")]
pub use error::ModelsDevError;

#[cfg(feature = "models-dev")]
pub use registry::{
    EnvVarInfo, ModelCostInfo, ModelInfo, ModelLimitInfo, ProviderInfo, ProviderRegistry,
};

#[cfg(feature = "models-dev")]
pub use types::{
    ApiInfo, DocInfo, EnvVar, Modalities, Model, ModelCost, ModelLimit, ModelsDevResponse, NpmInfo,
    Provider,
};

#[cfg(feature = "models-dev")]
pub use traits::{ModelsDevAware, ProviderConnectionInfo};

#[cfg(feature = "models-dev")]
pub use convenience::{
    check_provider_configuration, find_best_model_for_use_case, find_models_with_capability,
    find_provider_for_cloud_service, get_capability_summary, get_providers_summary,
    list_providers_for_npm_package,
};
