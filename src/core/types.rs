//! Core types for AI SDK functions.

use derive_builder::Builder;
use futures::Stream;
use serde::{Deserialize, Serialize};
use std::pin::Pin;

use crate::error::{Error, Result};

/// Options for a `generate_text` call.
#[derive(Debug, Clone, Serialize, Deserialize, Builder)]
#[builder(pattern = "owned", setter(into), build_fn(error = "Error"))]
pub struct GenerateTextCallOptions {
    /// The prompt to generate text from.
    pub prompt: String,
}

impl GenerateTextCallOptions {
    /// Creates a new builder for `GenerateTextCallOptions`.
    pub fn builder() -> GenerateTextCallOptionsBuilder {
        GenerateTextCallOptionsBuilder::default()
    }
}

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

pub struct GenerateStreamResponse {
    pub stream: LanguageModelStreamingResponse,
}

/// Options for a language model request.
#[derive(Debug, Clone, Serialize, Deserialize, Builder)]
#[builder(pattern = "owned", setter(into), build_fn(error = "Error"))]
pub struct LanguageModelCallOptions {
    /// The prompt to generate text from.
    pub prompt: String,
}

impl LanguageModelCallOptions {
    /// Creates a new builder for `LanguageModelCallOptions`.
    pub fn builder() -> LanguageModelCallOptionsBuilder {
        LanguageModelCallOptionsBuilder::default()
    }
}

/// Response from a language model.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LanguageModelResponse {
    /// The generated text.
    pub text: String,

    /// The model that generated the response.
    pub model: Option<String>,
}

impl LanguageModelResponse {
    /// Creates a new response with the generated text.
    pub fn new(text: impl Into<String>) -> Self {
        Self {
            text: text.into(),
            model: None,
        }
    }
}

pub type LanguageModelResponseChunk = LanguageModelResponse; // change this anytime chunk data
// deviates from text responses

pub type LanguageModelStreamingResponse =
    Pin<Box<dyn Stream<Item = Result<LanguageModelResponse>>>>;
