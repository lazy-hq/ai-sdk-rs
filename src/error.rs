//! Defines the core error and result types for the SDK.
//!
//! # Examples
//!
//! ```
//! use aisdk::error::{Result, Error};
//!
//! fn might_fail(should_fail: bool) -> Result<()> {
//!     if should_fail {
//!         Err(Error::Other("Something went wrong".to_string()))
//!     } else {
//!         Ok(())
//!     }
//! }
//! ```

use std::sync::Arc;

use derive_builder::UninitializedFieldError;

/// A marker trait for provider-specific errors.
pub trait ProviderError: std::error::Error + Send + Sync {}

impl PartialEq for dyn ProviderError {
    fn eq(&self, other: &dyn ProviderError) -> bool {
        self.to_string() == other.to_string()
    }
}

/// A specialized `Result` type for SDK operations.
pub type Result<T> = std::result::Result<T, Error>;

/// The primary error enum for all SDK-related failures.
#[derive(Debug, thiserror::Error, Clone, PartialEq)]
pub enum Error {
    /// Error indicating a required field was missing.
    #[error("A required field is missing: {0}")]
    MissingField(String),

    /// An error returned from the API.
    #[error("API error: {0}")]
    ApiError(String),

    /// An error for invalid input.
    #[error("Invalid input: {0}")]
    InvalidInput(String),

    #[error("Tool error: {0}")]
    ToolCallError(String),

    /// A catch-all for other miscellaneous errors.
    #[error("AI SDK error: {0}")]
    Other(String),

    /// Provider-specific error.
    #[error("Provider error: {0}")]
    ProviderError(Arc<dyn ProviderError>),
}

/// Implements `From` for `UninitializedFieldError` to convert it to `Error`.
/// Mainly used for the `derive_builder` crate.
impl From<UninitializedFieldError> for Error {
    fn from(err: UninitializedFieldError) -> Self {
        Error::MissingField(err.field_name().to_string())
    }
}

impl From<Error> for String {
    fn from(value: Error) -> String {
        match value {
            Error::MissingField(error) => format!("Missing field: {error}"),
            Error::ApiError(error) => format!("API error: {error}"),
            Error::InvalidInput(error) => format!("Invalid input: {error}"),
            Error::ToolCallError(error) => format!("Tool error: {error}"),
            Error::Other(error) => format!("Other error: {error}"),
            Error::ProviderError(error) => format!("Provider error: {error}"),
        }
    }
}
