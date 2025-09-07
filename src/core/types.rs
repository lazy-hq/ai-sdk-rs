//! Core types for AI SDK functions.

use derive_builder::Builder;
use futures::Stream;
use serde::{Deserialize, Serialize};
use std::pin::Pin;

use crate::error::{Error, Result};

/// Shortens the definition of the `GenerateTextCallOptions` and
/// `LanguageModelCallOptions` because all the fields from the first are also
/// second.
macro_rules! define_with_lm_call_options {
        ( $( ($field:ident, $typ:ty, $default:expr, $comment:expr) ),* ) => {
            #[derive(Debug, Clone, Serialize, Deserialize, Builder)]
            #[builder(pattern = "owned", setter(into), build_fn(error = "Error"))]
            pub struct GenerateTextCallOptions  {
                $(
                    #[doc = $comment]
                    #[builder(default = $default)]
                    pub $field: $typ,
                )*
                // Define `GenerateTextCallOptions` specific entries here

                /// Maximum number of retries.
                #[builder(default = "100")]
                pub max_retries: u32,
            }

            /// Options for a language model request. The ones directly passed to the
            /// provider,`None` is used for the provider default.
            #[derive(Debug, Clone, Serialize, Deserialize, Builder)]
            #[builder(pattern = "owned", setter(into), build_fn(error = "Error"))]
            pub struct LanguageModelCallOptions {
                $(
                    #[doc = $comment]
                    #[builder(default = $default)]
                    pub $field: $typ,
                )*
            }
        };
}

define_with_lm_call_options!(
    // identifier, type, default, comment
    (
        prompt,
        String,
        "".to_string(),
        "The prompt to generate text from. uses the completion format. If both prompt and messages are set, prompt will be ignored."
    ),
    (
        system_prompt,
        Option<String>,
        None,
        "The system prompt to generate text from."
    ),
    (
        messages,
        Option<Vec<String>>,
        None,
        "The messages to generate text from. uses the chat format. If both prompt and messages are set, prompt will be ignored."
    ),
    (
        max_tokens,
        Option<u32>,
        None,
        "The maximum number of tokens to generate."
    ),
    (temprature, Option<u32>, None, "Randomness."),
    (top_p, Option<u32>, None, "Nucleus sampling."),
    (top_k, Option<u32>, None, "Top-k sampling."),
    (stop, Option<Vec<String>>, None, "Stop sequence.")
);

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

impl LanguageModelCallOptions {
    /// Creates a new builder for `LanguageModelCallOptions`.
    pub fn builder() -> LanguageModelCallOptionsBuilder {
        LanguageModelCallOptionsBuilder::default()
    }
}

/// Response from a `generate_stream` call.
pub struct GenerateStreamResponse {
    /// A stream of responses from the language model.
    pub stream: LanguageModelStreamingResponse,
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

/// Chunked response from a language model.
pub type LanguageModelResponseChunk = LanguageModelResponse; // change this anytime chunk data
// deviates from text responses

/// Stream of responses from a language model's streaming API mapped to a common
/// interface.
pub type LanguageModelStreamingResponse =
    Pin<Box<dyn Stream<Item = Result<LanguageModelResponse>>>>;
