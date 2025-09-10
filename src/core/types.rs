//! Core types for AI SDK functions.

use derive_builder::Builder;
use futures::Stream;
use serde::{Deserialize, Serialize};
use std::pin::Pin;

use crate::error::{Error, Result};

/// Role for model messages.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum Role {
    System,
    User,
    Assistant,
}

/// Model message types.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ModelMessage {
    System(SystemMessage),
    User(UserMessage),
    Assistant(AssistantMessage),
}

/// System model message.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SystemMessage {
    role: Role,
    pub content: String,
}

impl SystemMessage {
    pub fn new(content: impl Into<String>) -> Self {
        Self {
            role: Role::System,
            content: content.into(),
        }
    }
}

/// User model message.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UserMessage {
    role: Role,
    pub content: String,
}

impl UserMessage {
    pub fn new(content: impl Into<String>) -> Self {
        Self {
            role: Role::User,
            content: content.into(),
        }
    }
}

/// Assistant model message.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AssistantMessage {
    role: Role,
    pub content: String,
}

impl AssistantMessage {
    pub fn new(content: impl Into<String>) -> Self {
        Self {
            role: Role::Assistant,
            content: content.into(),
        }
    }
}

/// Shortens the definition of the `GenerateTextCallOptions` and
/// `LanguageModelCallOptions` because all the fields from the first are also
/// second.
macro_rules! define_with_lm_call_options {
        ( $( ($field:ident, $typ:ty, $default:expr, $comment:expr) ),* ) => {
            #[derive(Debug, Clone, Serialize, Deserialize, Builder)]
            #[builder(pattern = "owned", setter(into), build_fn(name = "build_inner", error = "Error"))]
            pub struct GenerateTextCallOptions  {
                $(
                    #[doc = $comment]
                    #[builder(default = $default)]
                    pub $field: $typ,
                )*
                // Define `GenerateTextCallOptions` specific entries here

                /// The prompt to generate text from. Uses the completion format.
                /// Only one of prompt or messages should be set.
                #[builder(default = "None")]
                pub prompt: Option<String>,

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

// TODO: add support for main options
define_with_lm_call_options!(
    // identifier, type, default, comment
    (
        model,
        String,
        "".to_string(),
        "The model to use for the request."
    ),
    (
        system,
        Option<String>,
        None,
        "System prompt to be used for the request."
    ),
    (
        messages,
        Option<Vec<ModelMessage>>,
        None,
        "The messages to generate text from. Uses the chat format. Only one of prompt or messages should be set."
    ),
    (
        max_tokens,
        Option<u32>,
        None,
        "The maximum number of tokens to generate."
    ),
    (temperature, Option<u32>, None, "Randomness."),
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

impl GenerateTextCallOptionsBuilder {
    pub fn build(self) -> Result<GenerateTextCallOptions> {
        let options = self.build_inner()?;

        if options.prompt.is_some() && options.messages.is_some() {
            return Err(Error::InvalidInput(
                "Cannot set both prompt and messages".to_string(),
            ));
        }

        if options.messages.is_none() && options.prompt.is_none() {
            return Err(Error::InvalidInput(
                "Messages or prompt must be set".to_string(),
            ));
        }

        Ok(options)
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

/// Stream of responses from a language model's streaming API mapped to a common
/// interface.
pub type LanguageModelStreamingResponse =
    Pin<Box<dyn Stream<Item = Result<LanguageModelResponse>>>>;
