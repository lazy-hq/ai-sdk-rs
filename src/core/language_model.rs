//! Defines the central `LanguageModel` trait for interacting with text-based AI models.
//!
//! This module provides the `LanguageModel` trait, which establishes the core
//! contract for all language models supported by the SDK. It abstracts the
//! underlying implementation details of different AI providers, offering a
//! unified interface for various operations like text generation or streaming.

use crate::core::types::{
    LanguageModelCallOptions, LanguageModelResponse, LanguageModelStreamingResponse,
};
use crate::error::Result;
use async_trait::async_trait;

/// The core trait abstracting the capabilities of a language model.
///
/// This trait is the foundation for all text-based AI interactions.
/// Implementors of `LanguageModel` provide the necessary logic to connect to
/// a specific model endpoint and perform operations. The trait is designed to
/// be extensible to support various functionalities, such as single-shot
/// generation and streaming responses.
#[async_trait]
pub trait LanguageModel: Send + Sync + std::fmt::Debug {
    /// Returns the identifier of the model (e.g., "gpt-4o", "claude-3-sonnet").
    ///
    /// This is used for identifying which model is being used in requests and responses.
    fn model_name(&self) -> &str;

    /// Returns the name of the provider (e.g., "openai", "anthropic").
    ///
    /// This helps differentiate between models with similar names from different services.
    fn provider_name(&self) -> &str;

    /// Performs a single, non-streaming text generation request.
    ///
    /// This method sends a prompt to the model and returns the entire response at once.
    ///
    /// # Errors
    ///
    /// Returns an `Error` if the API call fails or the request is invalid.
    async fn generate(
        &mut self,
        options: LanguageModelCallOptions,
    ) -> Result<LanguageModelResponse>;

    /// Performs a streaming text generation request.
    /// 
    /// This method sends a prompt to the model and returns a stream of responses.
    ///
    /// # Errors
    ///
    /// Returns an `Error` if the API call fails or the request is invalid.
    async fn generate_stream(
        &mut self,
        options: LanguageModelCallOptions,
    ) -> Result<LanguageModelStreamingResponse>;
}
