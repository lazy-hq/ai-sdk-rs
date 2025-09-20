//! HTTP client for interacting with the models.dev API with caching support.

use std::{
    collections::HashMap,
    path::{Path, PathBuf},
    sync::Arc,
    time::{Duration, SystemTime, UNIX_EPOCH},
};

use serde::{Deserialize, Serialize};
use tokio::sync::RwLock;

use crate::models_dev::error::ModelsDevError;

/// Represents a provider from the models.dev API.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Provider {
    /// The unique identifier for the provider.
    pub id: String,
    /// The display name of the provider.
    pub name: String,
    /// The base URL for the provider's API.
    pub base_url: String,
    /// Whether the provider is currently available.
    pub available: bool,
    /// Additional metadata about the provider.
    pub metadata: HashMap<String, serde_json::Value>,
}

/// Cache entry with timestamp for expiration.
#[derive(Debug, Clone, Serialize, Deserialize)]
struct CacheEntry<T> {
    /// The cached data.
    data: T,
    /// When this cache entry expires (Unix timestamp).
    expires_at: u64,
}

impl<T> CacheEntry<T> {
    /// Create a new cache entry that expires after the given duration.
    fn new(data: T, ttl: Duration) -> Self {
        let expires_at = SystemTime::now()
            .checked_add(ttl)
            .unwrap_or_else(|| SystemTime::now() + ttl)
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();

        Self { data, expires_at }
    }

    /// Check if this cache entry is still valid.
    fn is_valid(&self) -> bool {
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();

        now < self.expires_at
    }
}

/// HTTP client for the models.dev API with caching support.
#[derive(Debug, Clone)]
pub struct ModelsDevClient {
    /// The HTTP client for making API requests.
    http_client: reqwest::Client,
    /// In-memory cache for API responses.
    memory_cache: Arc<RwLock<HashMap<String, CacheEntry<String>>>>,
    /// Path to the disk cache directory.
    disk_cache_path: PathBuf,
    /// Time-to-live for cache entries.
    cache_ttl: Duration,
    /// Base URL for the models.dev API.
    api_base_url: String,
}

impl ModelsDevClient {
    /// Create a new ModelsDevClient with default settings.
    pub fn new() -> Self {
        Self::builder()
            .build()
            .expect("Failed to build ModelsDevClient")
    }

    /// Create a builder for configuring the ModelsDevClient.
    pub fn builder() -> ModelsDevClientBuilder {
        ModelsDevClientBuilder::default()
    }

    /// Fetch the list of available providers from the models.dev API.
    ///
    /// This method implements a three-tier caching strategy:
    /// 1. Check memory cache first
    /// 2. If not in memory or expired, check disk cache
    /// 3. If not on disk or expired, fetch from API
    ///
    /// Returns a list of providers or an error if the request fails.
    pub async fn fetch_providers(&self) -> Result<Vec<Provider>, ModelsDevError> {
        let cache_key = "providers";

        // Try memory cache first
        {
            let memory_cache = self.memory_cache.read().await;
            if let Some(entry) = memory_cache.get(cache_key)
                && entry.is_valid()
            {
                let providers: Vec<Provider> = serde_json::from_str(&entry.data)
                    .map_err(|_| ModelsDevError::InvalidCacheFormat)?;
                return Ok(providers);
            }
        }

        // Try disk cache
        if let Ok(cached_data) = self.load_from_disk_cache(cache_key).await
            && let Ok(providers) = serde_json::from_str::<Vec<Provider>>(&cached_data)
        {
            // Update memory cache
            self.update_memory_cache(cache_key, &cached_data).await;
            return Ok(providers);
        }

        // Fetch from API
        let url = format!("{}/providers", self.api_base_url);
        let response = self
            .http_client
            .get(&url)
            .timeout(Duration::from_secs(30))
            .send()
            .await?;

        if !response.status().is_success() {
            let error_msg = response
                .text()
                .await
                .unwrap_or_else(|_| "Unknown error".to_string());
            return Err(ModelsDevError::ApiError(error_msg));
        }

        let providers_json = response.text().await?;
        let providers: Vec<Provider> =
            serde_json::from_str(&providers_json).map_err(ModelsDevError::JsonError)?;

        // Update both caches
        self.update_memory_cache(cache_key, &providers_json).await;
        if let Err(e) = self.save_to_disk_cache(cache_key, &providers_json).await {
            log::warn!("Failed to save to disk cache: {}", e);
        }

        Ok(providers)
    }

    /// Update the memory cache with new data.
    async fn update_memory_cache(&self, key: &str, data: &str) {
        let entry = CacheEntry::new(data.to_string(), self.cache_ttl);
        let mut memory_cache = self.memory_cache.write().await;
        memory_cache.insert(key.to_string(), entry);
    }

    /// Load data from disk cache.
    async fn load_from_disk_cache(&self, key: &str) -> Result<String, ModelsDevError> {
        let cache_file = self.disk_cache_path.join(format!("{}.json", key));

        if !cache_file.exists() {
            return Err(ModelsDevError::CacheNotFound(cache_file));
        }

        let content = tokio::fs::read_to_string(&cache_file).await?;

        // Check if the cache entry is still valid
        if let Ok(entry) = serde_json::from_str::<CacheEntry<String>>(&content)
            && entry.is_valid()
        {
            return Ok(entry.data);
        }

        Err(ModelsDevError::InvalidCacheFormat)
    }

    /// Save data to disk cache.
    async fn save_to_disk_cache(&self, key: &str, data: &str) -> Result<(), ModelsDevError> {
        // Ensure cache directory exists
        tokio::fs::create_dir_all(&self.disk_cache_path).await?;

        let cache_file = self.disk_cache_path.join(format!("{}.json", key));
        let entry = CacheEntry::new(data.to_string(), self.cache_ttl);

        let json_data = serde_json::to_string(&entry)?;
        tokio::fs::write(&cache_file, json_data).await?;

        Ok(())
    }

    /// Clear all caches (memory and disk).
    pub async fn clear_caches(&self) -> Result<(), ModelsDevError> {
        // Clear memory cache
        {
            let mut memory_cache = self.memory_cache.write().await;
            memory_cache.clear();
        }

        // Clear disk cache
        if self.disk_cache_path.exists() {
            tokio::fs::remove_dir_all(&self.disk_cache_path).await?;
            tokio::fs::create_dir_all(&self.disk_cache_path).await?;
        }

        Ok(())
    }

    /// Get the current cache statistics.
    pub async fn cache_stats(&self) -> CacheStats {
        let memory_cache = self.memory_cache.read().await;
        let memory_entry_count = memory_cache.len();

        let disk_entry_count = if self.disk_cache_path.exists() {
            match tokio::fs::read_dir(&self.disk_cache_path).await {
                Ok(mut entries) => {
                    let mut count = 0;
                    while let Ok(Some(_)) = entries.next_entry().await {
                        count += 1;
                    }
                    count
                }
                Err(_) => 0,
            }
        } else {
            0
        };

        CacheStats {
            memory_entries: memory_entry_count,
            disk_entries: disk_entry_count,
            cache_ttl: self.cache_ttl,
        }
    }

    /// Get the API base URL.
    pub fn api_base_url(&self) -> &str {
        &self.api_base_url
    }

    /// Get the cache TTL.
    pub fn cache_ttl(&self) -> Duration {
        self.cache_ttl
    }

    /// Get the disk cache path.
    pub fn disk_cache_path(&self) -> &Path {
        &self.disk_cache_path
    }
}

/// Cache statistics information.
#[derive(Debug, Clone)]
pub struct CacheStats {
    /// Number of entries in memory cache.
    pub memory_entries: usize,
    /// Number of entries in disk cache.
    pub disk_entries: usize,
    /// Time-to-live for cache entries.
    pub cache_ttl: Duration,
}

/// Builder for configuring a ModelsDevClient.
#[derive(Debug, Clone, Default)]
pub struct ModelsDevClientBuilder {
    http_client: Option<reqwest::Client>,
    disk_cache_path: Option<PathBuf>,
    cache_ttl: Option<Duration>,
    api_base_url: Option<String>,
}

impl ModelsDevClientBuilder {
    /// Set a custom HTTP client.
    pub fn http_client(mut self, client: reqwest::Client) -> Self {
        self.http_client = Some(client);
        self
    }

    /// Set the disk cache path.
    pub fn disk_cache_path<P: AsRef<Path>>(mut self, path: P) -> Self {
        self.disk_cache_path = Some(path.as_ref().to_path_buf());
        self
    }

    /// Set the cache time-to-live.
    pub fn cache_ttl(mut self, ttl: Duration) -> Self {
        self.cache_ttl = Some(ttl);
        self
    }

    /// Set the API base URL.
    pub fn api_base_url(mut self, url: impl Into<String>) -> Self {
        self.api_base_url = Some(url.into());
        self
    }

    /// Build the ModelsDevClient.
    pub fn build(self) -> Result<ModelsDevClient, ModelsDevError> {
        let http_client = self.http_client.unwrap_or_else(|| {
            reqwest::Client::builder()
                .timeout(Duration::from_secs(30))
                .build()
                .expect("Failed to build HTTP client")
        });

        let disk_cache_path = self.disk_cache_path.unwrap_or_else(|| {
            #[cfg(feature = "dirs")]
            {
                let mut path = dirs::cache_dir().unwrap_or_else(|| PathBuf::from(".cache"));
                path.push("aisdk");
                path.push("models-dev");
                path
            }
            #[cfg(not(feature = "dirs"))]
            {
                let mut path = PathBuf::from(".cache");
                path.push("aisdk");
                path.push("models-dev");
                path
            }
        });

        let cache_ttl = self.cache_ttl.unwrap_or(Duration::from_secs(3600)); // 1 hour
        let api_base_url = self
            .api_base_url
            .unwrap_or_else(|| "https://api.models.dev".to_string());

        Ok(ModelsDevClient {
            http_client,
            memory_cache: Arc::new(RwLock::new(HashMap::new())),
            disk_cache_path,
            cache_ttl,
            api_base_url,
        })
    }
}

impl Default for ModelsDevClient {
    fn default() -> Self {
        Self::new()
    }
}
