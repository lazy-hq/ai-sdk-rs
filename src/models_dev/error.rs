//! Error types for the models.dev client.

use std::path::PathBuf;

/// Errors that can occur when interacting with the models.dev API.
#[derive(Debug, thiserror::Error)]
pub enum ModelsDevError {
    /// HTTP request failed.
    #[error("HTTP request failed: {0}")]
    HttpError(#[from] reqwest::Error),

    /// JSON serialization/deserialization failed.
    #[error("JSON error: {0}")]
    JsonError(#[from] serde_json::Error),

    /// I/O operation failed.
    #[error("I/O error: {0}")]
    IoError(#[from] std::io::Error),

    /// Cache file not found.
    #[error("Cache file not found: {0}")]
    CacheNotFound(PathBuf),

    /// Invalid cache data format.
    #[error("Invalid cache data format")]
    InvalidCacheFormat,

    /// API returned an error response.
    #[error("API error: {0}")]
    ApiError(String),

    /// Network timeout.
    #[error("Network timeout")]
    Timeout,

    /// Invalid URL.
    #[error("Invalid URL: {0}")]
    InvalidUrl(String),
}
