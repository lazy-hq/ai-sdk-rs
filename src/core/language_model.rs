//! Defines the central `LanguageModel` trait for interacting with text-based AI models.
//!
//! This module provides the `LanguageModel` trait, which establishes the core
//! contract for all language models supported by the SDK. It abstracts the
//! underlying implementation details of different AI providers, offering a
//! unified interface for various operations like text generation or streaming.

use std::ops::{Deref, DerefMut};

use crate::core::types::Message;
use crate::core::{LanguageModelCallOptions, LanguageModelResponse, LanguageModelStreamResponse};
use crate::error::{Error, Result};
use async_trait::async_trait;
use derive_builder::Builder;
use serde::{Deserialize, Serialize};

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
    ) -> Result<LanguageModelStreamResponse>;
}

// ============================================================================
// Section: structs and builders
// ============================================================================

/// Options for a language model request.
#[allow(dead_code)]
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

    /// Stop sequence.
    pub stop: Option<Vec<String>>,

    /// Maximum number of retries.
    pub max_retries: Option<u32>,

    /// Maxoutput tokens.
    pub max_output_tokens: Option<u32>,

    /// Stop sequences.
    /// If set, the model will stop generating text when one of the stop sequences is generated.
    /// Providers may have limits on the number of stop sequences.
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

/// Options for text generation requests such as `generate_text` and `stream_text`.
#[allow(dead_code)]
#[derive(Debug, Clone, Serialize, Deserialize)]
struct GenerateOptions<M: LanguageModel> {
    /// The Language Model to use.
    pub model: M,

    /// The prompt to generate text from.
    /// Only one of prompt or messages should be set.
    pub prompt: Option<String>,

    /// Language model call options for the request
    options: LanguageModelOptions,
}

#[allow(dead_code)]
impl<M: LanguageModel> GenerateOptions<M> {
    fn builder() -> GenerateOptionsBuilder<M> {
        GenerateOptionsBuilder::default()
    }
}

#[allow(dead_code)]
struct GenerateOptionsBuilder<M: LanguageModel> {
    model: Option<M>,
    prompt: Option<String>,
    options: LanguageModelOptions,
}

impl<M: LanguageModel> Deref for GenerateOptionsBuilder<M> {
    type Target = LanguageModelOptions;

    fn deref(&self) -> &Self::Target {
        &self.options
    }
}

impl<M: LanguageModel> DerefMut for GenerateOptionsBuilder<M> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.options
    }
}

impl<M: LanguageModel> GenerateOptionsBuilder<M> {
    fn default() -> Self {
        GenerateOptionsBuilder {
            model: None,
            prompt: None,
            options: LanguageModelOptions::default(),
        }
    }
}

#[allow(dead_code)]
impl<M: LanguageModel> GenerateOptionsBuilder<M> {
    fn model(mut self, model: M) -> Self {
        self.model = model.into();
        self
    }

    fn system(mut self, system: impl Into<String>) -> Self {
        self.system = Some(system.into());
        self
    }

    fn prompt(mut self, prompt: impl Into<String>) -> Self {
        self.prompt = Some(prompt.into());
        self
    }

    fn messages(mut self, messages: Vec<Message>) -> Self {
        self.messages = messages;
        self
    }

    fn seed(mut self, seed: impl Into<u32>) -> Self {
        self.seed = Some(seed.into());
        self
    }

    fn temperature(mut self, temperature: impl Into<u32>) -> Self {
        self.temperature = Some(temperature.into());
        self
    }

    fn top_p(mut self, top_p: impl Into<u32>) -> Self {
        self.top_p = Some(top_p.into());
        self
    }

    fn top_k(mut self, top_k: impl Into<u32>) -> Self {
        self.top_k = Some(top_k.into());
        self
    }

    fn stop(mut self, stop: impl Into<Vec<String>>) -> Self {
        self.stop = Some(stop.into());
        self
    }

    fn max_retries(mut self, max_retries: impl Into<u32>) -> Self {
        self.max_retries = Some(max_retries.into());
        self
    }

    fn max_output_tokens(mut self, max_output_tokens: impl Into<u32>) -> Self {
        self.max_output_tokens = Some(max_output_tokens.into());
        self
    }

    fn stop_sequences(mut self, stop_sequences: impl Into<Vec<String>>) -> Self {
        self.stop_sequences = Some(stop_sequences.into());
        self
    }

    fn presence_penalty(mut self, presence_penalty: impl Into<f32>) -> Self {
        self.presence_penalty = Some(presence_penalty.into());
        self
    }

    fn frequency_penalty(mut self, frequency_penalty: impl Into<f32>) -> Self {
        self.frequency_penalty = Some(frequency_penalty.into());
        self
    }

    fn build(self) -> GenerateOptions<M> {
        let model = self
            .model
            .unwrap_or_else(|| unreachable!("Model must be set"));

        GenerateOptions {
            model,
            prompt: self.prompt,
            options: self.options,
        }
    }
}

// ============================================================================
// Section: implementations
// ============================================================================
