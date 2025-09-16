//! Response types for AI SDK functions and traits.

use futures::Stream;
use serde::{Deserialize, Serialize};
use std::pin::Pin;

use crate::error::Result;

/*
 *CORE trait responses types
 */

/* Language Model Responses*/

// TODO: constract a standard response type
/// Response from a language model.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LanguageModelResponse {
    /// The generated text.
    pub text: String,

    /// The model that generated the response.
    pub model: Option<String>,

    /// The reason the model stopped generating text.
    pub stop_reason: Option<String>,
}

impl LanguageModelResponse {
    /// Creates a new response with the generated text.
    pub fn new(text: impl Into<String>) -> Self {
        Self {
            text: text.into(),
            model: None,
            stop_reason: None,
        }
    }
}

/* Language Model Stream Responses*/

/// A response from a streaming language model.
pub struct LanguageModelStreamResponse {
    /// A stream of responses from the language model.
    pub stream: StreamChunk,

    /// The model that generated the response.
    pub model: Option<String>,
}

/// Stream of responses from mapped to a common interface.
pub type StreamChunk = Pin<Box<dyn Stream<Item = Result<StreamChunkData>> + Send>>;

/// Chunked response from a language model.
pub struct StreamChunkData {
    /// The generated text.
    pub text: String,

    /// The reason the model stopped generating text.
    pub stop_reason: Option<String>,
}

/*
 *CORE function responses types
 */

/*Generate Text Responses*/
//TODO: add standard response fields
/// Response from a `generate_text` call.
#[derive(Debug)]
pub struct GenerateTextResponse {
    /// The generated text.
    pub text: String,
}

impl GenerateTextResponse {
    /// Creates a new response with the generated text.
    pub fn new(text: impl Into<String>) -> Self {
        Self { text: text.into() }
    }
}

/*Stream Text Responses*/
//TODO: add stream text response types currently it is using the same response type of
//LanguageModelStreamResponse
