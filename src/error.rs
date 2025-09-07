//! Defines the core error and result types for the SDK.
//!
//! # Examples
//!
//! ```
//! use ai_sdk_rs::error::{Result, Error};
//!
//! fn might_fail(should_fail: bool) -> Result<()> {
//!     if should_fail {
//!         Err(Error::Other("Something went wrong".to_string()))
//!     } else {
//!         Ok(())
//!     }
//! }
//! ```

use derive_builder::UninitializedFieldError;

/// A specialized `Result` type for SDK operations.
pub type Result<T> = std::result::Result<T, Error>;

/// The primary error enum for all SDK-related failures.
#[derive(Debug, thiserror::Error)]
pub enum Error {
    /// Error indicating a required field was missing.
    #[error("A required field is missing: {0}")]
    MissingField(String),

    /// An error returned from the API.
    #[error("API error: {0}")]
    ApiError(String),

    /// An error from the underlying `reqwest` client.
    #[error("HTTP request error: {0}")]
    ReqwestError(#[from] reqwest::Error),

    #[error("OpenAI error: {0}")]
    OpenAIError(#[from] async_openai::error::OpenAIError),

    /// A catch-all for other miscellaneous errors.
    #[error("AI SDK error: {0}")]
    Other(String),
}

/// Implements `From` for `UninitializedFieldError` to convert it to `Error`.
/// Mainly used for the `derive_builder` crate.
impl From<UninitializedFieldError> for Error {
    fn from(err: UninitializedFieldError) -> Self {
        Error::MissingField(err.field_name().to_string())
    }
}
