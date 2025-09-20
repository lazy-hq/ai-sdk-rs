//! Integration tests for the ModelsDevClient.

#[cfg(feature = "models-dev")]
mod tests {
    use aisdk::models_dev::ModelsDevClient;
    use tempfile::TempDir;

    #[tokio::test]
    async fn test_client_creation() {
        let client = ModelsDevClient::new();
        assert!(!client.api_base_url().is_empty());
    }

    #[tokio::test]
    async fn test_client_builder() {
        let temp_dir = TempDir::new().unwrap();
        let client = ModelsDevClient::builder()
            .disk_cache_path(temp_dir.path())
            .cache_ttl(std::time::Duration::from_secs(60))
            .api_base_url("https://api.example.com")
            .build()
            .unwrap();

        assert_eq!(client.api_base_url(), "https://api.example.com");
        assert_eq!(client.cache_ttl(), std::time::Duration::from_secs(60));
        assert_eq!(client.disk_cache_path(), temp_dir.path());
    }

    #[tokio::test]
    async fn test_cache_stats() {
        let client = ModelsDevClient::new();
        let stats = client.cache_stats().await;

        assert_eq!(stats.memory_entries, 0);
        assert!(stats.cache_ttl > std::time::Duration::from_secs(0));
    }

    #[tokio::test]
    async fn test_clear_caches() {
        let temp_dir = TempDir::new().unwrap();
        let client = ModelsDevClient::builder()
            .disk_cache_path(temp_dir.path())
            .build()
            .unwrap();

        // This should not panic
        client.clear_caches().await.unwrap();

        let stats = client.cache_stats().await;
        assert_eq!(stats.memory_entries, 0);
    }
}
