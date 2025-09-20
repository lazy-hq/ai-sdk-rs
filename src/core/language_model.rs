//! Defines the central `LanguageModel` trait for interacting with text-based AI models.
//!
//! This module provides the `LanguageModel` trait, which establishes the core
//! contract for all language models supported by the SDK. It abstracts the
//! underlying implementation details of different AI providers, offering a
//! unified interface for various operations like text generation or streaming.

use crate::core::Message;
use crate::core::utils::resolve_message;
use crate::error::{Error, Result};
use async_trait::async_trait;
use derive_builder::Builder;
use futures::Stream;
use serde::{Deserialize, Serialize};
use std::ops::{Deref, DerefMut};
use std::pin::Pin;

// ============================================================================
// Section: traits
// ============================================================================

/// The core trait abstracting the capabilities of a language model.
///
/// This trait is the foundation for all text-based AI interactions.
/// Implementors of `LanguageModel` provide the necessary logic to connect to
/// a specific model endpoint and perform operations. The trait is designed to
/// be extensible to support various functionalities, such as single-shot
/// generation and streaming responses.
#[async_trait]
pub trait LanguageModel: Send + Sync + std::fmt::Debug {
    /// Performs a single, non-streaming text generation request.
    ///
    /// This method sends a prompt to the model and returns the entire response at once.
    ///
    /// # Errors
    ///
    /// Returns an `Error` if the API call fails or the request is invalid.
    async fn generate(&mut self, options: LanguageModelOptions) -> Result<LanguageModelResponse>;

    /// Performs a streaming text generation request.
    ///
    /// This method sends a prompt to the model and returns a stream of responses.
    ///
    /// # Errors
    ///
    /// Returns an `Error` if the API call fails or the request is invalid.
    async fn generate_stream(
        &mut self,
        options: LanguageModelOptions,
    ) -> Result<LanguageModelStreamResponse>;
}

// ============================================================================
// Section: structs and builders
// ============================================================================

/// Options for a language model request.
#[derive(Debug, Clone, Serialize, Deserialize, Default, Builder)]
#[builder(pattern = "owned", setter(into), build_fn(error = "Error"))]
pub struct LanguageModelOptions {
    /// System prompt to be used for the request.
    pub system: Option<String>,

    /// The messages to generate text from.
    /// At least User Message is required.
    pub messages: Vec<Message>,

    /// The seed (integer) to use for random sampling. If set and supported
    /// by the model, calls will generate deterministic results.
    pub seed: Option<u32>,

    /// Randomness.
    pub temperature: Option<u32>,

    /// Nucleus sampling.
    pub top_p: Option<u32>,

    /// Top-k sampling.
    pub top_k: Option<u32>,

    /// Maximum number of retries.
    pub max_retries: Option<u32>,

    /// Maxoutput tokens.
    pub max_output_tokens: Option<u32>,

    /// Stop sequences.
    /// If set, the model will stop generating text when one of the stop sequences is generated.
    pub stop_sequences: Option<Vec<String>>,

    /// Presence penalty setting. It affects the likelihood of the model to
    /// repeat information that is already in the prompt.
    pub presence_penalty: Option<f32>,

    /// Frequency penalty setting. It affects the likelihood of the model
    /// to repeatedly use the same words or phrases.
    pub frequency_penalty: Option<f32>,
    // TODO: add support for reponse format
    // pub response_format: Option<ResponseFormat>,

    // Additional provider-specific options. They are passed through
    // to the provider from the AI SDK and enable provider-specific functionality.
    //TODO: add support for provider options
    //pub provider_options: <HashMap<String, <HashMap<String, JsonValue>>>>,
}

impl LanguageModelOptions {
    pub fn builder() -> LanguageModelOptionsBuilder {
        LanguageModelOptionsBuilder::default()
    }
}

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

/// Options for text generation requests such as `generate_text` and `stream_text`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LanguageModelRequest<M: LanguageModel> {
    /// The Language Model to use.
    pub model: M,

    /// The prompt to generate text from.
    /// Only one of prompt or messages should be set.
    pub prompt: Option<String>,

    /// Language model call options for the request
    options: LanguageModelOptions,
}

impl<M: LanguageModel> LanguageModelRequest<M> {
    pub fn builder() -> LanguageModelRequestBuilder<M> {
        LanguageModelRequestBuilder::default()
    }
}

impl<M: LanguageModel> Deref for LanguageModelRequest<M> {
    type Target = LanguageModelOptions;

    fn deref(&self) -> &Self::Target {
        &self.options
    }
}

impl<M: LanguageModel> DerefMut for LanguageModelRequest<M> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.options
    }
}

// State for GenerateOptionsBuilder
// Following the type State builder pattern

/// Initial state for setting the model
/// returns SystemStage
pub struct ModelStage {}

/// Secondary state for including system prompt or not
/// returns ConversationStage
pub struct SystemStage {}

/// Third state for conversation, Message or Prompt
/// returns OptionsStage
pub struct ConversationStage {}

/// Final State for setting Options and config
/// returns builder.build
pub struct OptionsStage {}

pub struct LanguageModelRequestBuilder<M: LanguageModel, State = ModelStage> {
    model: Option<M>,
    prompt: Option<String>,
    options: LanguageModelOptions,
    state: std::marker::PhantomData<State>,
}

impl<M: LanguageModel, State> Deref for LanguageModelRequestBuilder<M, State> {
    type Target = LanguageModelOptions;

    fn deref(&self) -> &Self::Target {
        &self.options
    }
}

impl<M: LanguageModel, State> DerefMut for LanguageModelRequestBuilder<M, State> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.options
    }
}

impl<M: LanguageModel> LanguageModelRequestBuilder<M> {
    fn default() -> Self {
        LanguageModelRequestBuilder {
            model: None,
            prompt: None,
            options: LanguageModelOptions::default(),
            state: std::marker::PhantomData,
        }
    }
}

/// ModelStage Builder
impl<M: LanguageModel> LanguageModelRequestBuilder<M, ModelStage> {
    pub fn model(self, model: M) -> LanguageModelRequestBuilder<M, ConversationStage> {
        LanguageModelRequestBuilder {
            model: Some(model),
            prompt: self.prompt,
            options: self.options,
            state: std::marker::PhantomData,
        }
    }
}

/// SystemStage Builder
impl<M: LanguageModel> LanguageModelRequestBuilder<M, SystemStage> {
    pub fn system(
        self,
        system: impl Into<String>,
    ) -> LanguageModelRequestBuilder<M, ConversationStage> {
        LanguageModelRequestBuilder {
            model: self.model,
            prompt: self.prompt,
            options: LanguageModelOptions {
                system: Some(system.into()),
                ..self.options
            },
            state: std::marker::PhantomData,
        }
    }
}

/// ConversationStage Builder
impl<M: LanguageModel> LanguageModelRequestBuilder<M, ConversationStage> {
    pub fn prompt(self, prompt: impl Into<String>) -> LanguageModelRequestBuilder<M, OptionsStage> {
        LanguageModelRequestBuilder {
            model: self.model,
            prompt: Some(prompt.into()),
            options: self.options,
            state: std::marker::PhantomData,
        }
    }

    pub fn messages(self, messages: Vec<Message>) -> LanguageModelRequestBuilder<M, OptionsStage> {
        LanguageModelRequestBuilder {
            model: self.model,
            prompt: self.prompt,
            options: LanguageModelOptions {
                messages,
                ..self.options
            },
            state: std::marker::PhantomData,
        }
    }
}

/// OptionsStage Builder
impl<M: LanguageModel> LanguageModelRequestBuilder<M, OptionsStage> {
    pub fn seed(mut self, seed: impl Into<u32>) -> Self {
        self.seed = Some(seed.into());
        self
    }

    pub fn temperature(mut self, temperature: impl Into<u32>) -> Self {
        self.temperature = Some(temperature.into());
        self
    }

    pub fn top_p(mut self, top_p: impl Into<u32>) -> Self {
        self.top_p = Some(top_p.into());
        self
    }

    pub fn top_k(mut self, top_k: impl Into<u32>) -> Self {
        self.top_k = Some(top_k.into());
        self
    }

    pub fn stop_sequences(mut self, stop_sequences: impl Into<Vec<String>>) -> Self {
        self.stop_sequences = Some(stop_sequences.into());
        self
    }

    pub fn max_retries(mut self, max_retries: impl Into<u32>) -> Self {
        self.max_retries = Some(max_retries.into());
        self
    }

    pub fn frequency_penalty(mut self, frequency_penalty: impl Into<f32>) -> Self {
        self.frequency_penalty = Some(frequency_penalty.into());
        self
    }

    pub fn build(self) -> LanguageModelRequest<M> {
        let model = self
            .model
            .unwrap_or_else(|| unreachable!("Model must be set"));

        LanguageModelRequest {
            model,
            prompt: self.prompt,
            options: self.options,
        }
    }
}

// ============================================================================
// Section: core function reponse types
// ============================================================================

//TODO: add standard response fields
/// Response from a generate call on `GenerateText`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GenerateTextResponse {
    /// The generated text.
    pub text: String,
}

//TODO: add standard response fields
/// Response from a stream call on `StreamText`.
pub struct StreamTextResponse {
    /// A stream of responses from the language model.
    pub stream: StreamChunk,

    /// The model that generated the response.
    pub model: Option<String>,
}

// ============================================================================
// Section: implementations, Core functions such as generate_text and stream_text
// ============================================================================

impl<M: LanguageModel> LanguageModelRequest<M> {
    /// Generates text using a specified language model.
    ///
    /// Generate a text and call tools for a given prompt using a language model.
    /// This function does not stream the output. If you want to stream the output, use `StreamText` instead.
    ///
    /// Returns an `Error` if the underlying model fails to generate a response.
    pub async fn generate_text(&mut self) -> Result<GenerateTextResponse> {
        let (system_prompt, messages) =
            resolve_message(&self.options.system, &self.prompt, &self.options.messages);

        let response = self
            .model
            .generate(
                LanguageModelOptions::builder()
                    .system(system_prompt)
                    .messages(messages)
                    .max_output_tokens(self.options.max_output_tokens)
                    .temperature(self.options.temperature)
                    .top_p(self.options.top_p)
                    .top_k(self.options.top_k)
                    .stop_sequences(self.options.stop_sequences.clone())
                    .build()?,
            )
            .await?;

        let result = GenerateTextResponse {
            text: response.text,
        };

        Ok(result)
    }

    /// Generates Streaming text using a specified language model.
    ///
    /// Generate a text and call tools for a given prompt using a language model.
    /// This function streams the output. If you do not want to stream the output, use `GenerateText` instead.
    ///
    /// Returns an `Error` if the underlying model fails to generate a response.
    pub async fn stream_text(&mut self) -> Result<StreamTextResponse> {
        let (system_prompt, messages) =
            resolve_message(&self.options.system, &self.prompt, &self.options.messages);

        let response = self
            .model
            .generate_stream(
                LanguageModelOptions::builder()
                    .system(system_prompt)
                    .messages(messages)
                    .max_output_tokens(self.options.max_output_tokens)
                    .temperature(self.options.temperature)
                    .top_p(self.options.top_p)
                    .top_k(self.options.top_k)
                    .stop_sequences(self.options.stop_sequences.clone())
                    .build()?,
            )
            .await?;

        let result = StreamTextResponse {
            stream: Box::pin(response.stream),
            model: response.model,
        };

        Ok(result)
    }
}
